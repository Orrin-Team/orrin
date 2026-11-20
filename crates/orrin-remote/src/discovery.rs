//! `.orrin/remote`: where a running editor is listening, and the token it
//! will accept. The editor writes it as the remote starts and removes it as
//! the remote stops; a client reads it to connect.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::token::Token;

/// Its name inside a project's editor directory (`Project::editor_dir`).
pub const FILE: &str = "remote";

/// What the file says. JSON, so a person can read it, and so it decodes
/// through the same total reader the frames do.
#[derive(Clone, Debug)]
pub struct Discovery {
    pub port: u16,
    pub token: Token,
    /// The editor's process, so a client can say which one it found.
    pub pid: u32,
}

#[derive(Serialize, Deserialize)]
struct OnDisk {
    port: u16,
    token: String,
    pid: u32,
}

/// The file for the project whose editor directory is `editor_dir`.
pub fn path(editor_dir: &Path) -> PathBuf {
    editor_dir.join(FILE)
}

impl Discovery {
    /// Write the file readable by its owner alone, by rename, so a reader
    /// never sees half of one. Any file already there — another editor's on
    /// this project, or one a crash left — is replaced: the newest editor is
    /// the one a client should find.
    pub fn write(&self, editor_dir: &Path) -> Result<PathBuf, String> {
        let file = path(editor_dir);
        let fail = |error: std::io::Error| format!("could not write {}: {error}", file.display());
        std::fs::create_dir_all(editor_dir).map_err(fail)?;
        let text = serde_json::to_string(&OnDisk {
            port: self.port,
            token: self.token.to_hex(),
            pid: self.pid,
        })
        .map_err(|error| error.to_string())?;
        let partial = editor_dir.join(format!("{FILE}.partial"));
        // The mode applies only to a file this call creates, so one left over
        // with a looser mode is not reused.
        let _ = std::fs::remove_file(&partial);
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut out = options.open(&partial).map_err(fail)?;
        out.write_all(text.as_bytes()).map_err(fail)?;
        drop(out);
        std::fs::rename(&partial, &file).map_err(fail)?;
        Ok(file)
    }

    /// The file for `editor_dir`: `Ok(None)` when there is none, and an error
    /// naming it when it does not read as one.
    pub fn read(editor_dir: &Path) -> Result<Option<Self>, String> {
        let file = path(editor_dir);
        let text = match std::fs::read(&file) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("could not read {}: {error}", file.display())),
        };
        let unreadable = || format!("{} is not an editor's remote file", file.display());
        let on_disk: OnDisk = serde_json::from_slice(&text).map_err(|_| unreadable())?;
        let token = Token::from_hex(&on_disk.token).ok_or_else(unreadable)?;
        Ok(Some(Self {
            port: on_disk.port,
            token,
            pid: on_disk.pid,
        }))
    }

    /// Remove the file, if it is still this one: a second editor opened on the
    /// project since has written its own, which is not this one's to remove.
    pub fn remove(&self, editor_dir: &Path) {
        if let Ok(Some(found)) = Self::read(editor_dir)
            && found.token.matches(&self.token)
        {
            let _ = std::fs::remove_file(path(editor_dir));
        }
    }
}
