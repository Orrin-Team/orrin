//! A copy of unsaved work, written every few minutes beside the editor's other
//! state rather than over the scene.
//!
//! **Not over the scene file.** Saving is a decision: a scene file half way
//! through an idea is a scene file someone else pulls. The copy goes to
//! `<project>/.orrin/autosave/`, named after the scene's path, and File ›
//! Recover autosave opens it as unsaved work, for Save to put where it belongs.
//! An explicit save deletes it, because it is then older than the file.
//!
//! **Only what the history has not had.** A copy is written when the scene
//! is dirty, the history has moved since the last copy, no gesture is open, and
//! the game is not playing — the fork is not the scene, and nothing done in
//! edit mode can be lost while it runs.
//!
//! **Off the frame.** The snapshot is the scene document, read on the main
//! thread because it reads the world; formatting and the write happen on a
//! thread of their own. Not on the engine's pool: a worker blocked on the disk
//! is one fewer for every parallel step of the frame (see `threads.rs`). The
//! write goes to a temporary file that is then renamed over the copy, so a
//! crash mid-write leaves the previous copy rather than half of a new one.

use std::path::{Path, PathBuf};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use orrin_registry::SceneDocument;

/// How long unsaved work goes without a copy.
pub const INTERVAL: Duration = Duration::from_secs(120);

pub struct Autosave {
    dir: PathBuf,
    /// When the last copy was started, or the session began.
    since: Instant,
    /// The history version the last copy holds.
    written: Option<u64>,
    job: Option<JoinHandle<Result<PathBuf, String>>>,
}

impl Autosave {
    pub fn new(dir: PathBuf, now: Instant) -> Self {
        Self {
            dir,
            since: now,
            written: None,
            job: None,
        }
    }

    /// Where the copy of the scene at `scene_path` goes.
    pub fn path(&self, scene_path: &str) -> PathBuf {
        let name: String = scene_path
            .trim_start_matches(['.', '/', '\\'])
            .chars()
            .map(|c| {
                if matches!(c, '/' | '\\' | ':') {
                    '_'
                } else {
                    c
                }
            })
            .collect();
        self.dir.join(name)
    }

    /// Whether a dirty scene at history `version` is owed a copy at `now`.
    pub fn due(&self, now: Instant, version: u64) -> bool {
        self.job.is_none()
            && self.written != Some(version)
            && now.saturating_duration_since(self.since) >= INTERVAL
    }

    /// Write `document` to `path` on a thread of its own.
    pub fn write(&mut self, document: SceneDocument, path: PathBuf, version: u64, now: Instant) {
        self.since = now;
        self.written = Some(version);
        let spawned = std::thread::Builder::new()
            .name("orrin-autosave".to_owned())
            .spawn(move || {
                let mut text = String::new();
                orrin_registry::write_document(&mut text, &document);
                write_atomic(&path, &text)
                    .map(|()| path.clone())
                    .map_err(|error| format!("could not autosave to {}: {error}", path.display()))
            });
        match spawned {
            Ok(job) => self.job = Some(job),
            Err(error) => tracing::warn!("could not start the autosave thread: {error}"),
        }
    }

    /// The outcome of a copy that has finished since the last call.
    pub fn finished(&mut self) -> Option<Result<PathBuf, String>> {
        if !self.job.as_ref()?.is_finished() {
            return None;
        }
        self.wait()
    }

    /// Block until the copy in flight, if any, is written.
    pub fn wait(&mut self) -> Option<Result<PathBuf, String>> {
        let outcome = self.job.take()?.join();
        Some(outcome.unwrap_or_else(|_| Err("the autosave thread panicked".to_owned())))
    }

    /// The scene was saved at `version`: the copy is older than the file, and
    /// nothing is owed until it changes again.
    pub fn saved(&mut self, scene_path: &str, version: u64) {
        // A copy still being written would land after the delete.
        self.wait();
        self.written = Some(version);
        let _ = std::fs::remove_file(self.path(scene_path));
    }
}

fn write_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let partial = path.with_extension("partial");
    std::fs::write(&partial, text)?;
    std::fs::rename(&partial, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("orrin-autosave-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn document() -> SceneDocument {
        orrin_registry::parse("orrin-scene 3 min-reader 3\n").unwrap()
    }

    #[test]
    fn nothing_is_owed_before_the_interval() {
        let start = Instant::now();
        let autosave = Autosave::new(scratch("interval"), start);
        assert!(!autosave.due(start + INTERVAL / 2, 1));
        assert!(autosave.due(start + INTERVAL, 1));
    }

    #[test]
    fn a_version_already_copied_is_not_copied_again() {
        let start = Instant::now();
        let dir = scratch("again");
        let mut autosave = Autosave::new(dir.clone(), start);
        let path = autosave.path("level.orrin");
        autosave.write(document(), path, 4, start + INTERVAL);
        assert!(autosave.wait().unwrap().is_ok());
        assert!(!autosave.due(start + INTERVAL * 3, 4));
        assert!(autosave.due(start + INTERVAL * 3, 5));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two scenes with one file name in different folders are two copies.
    #[test]
    fn copies_are_named_by_the_whole_path() {
        let autosave = Autosave::new(PathBuf::from("autosave"), Instant::now());
        let a = autosave.path("scenes/level.orrin");
        let b = autosave.path("old/level.orrin");
        assert_ne!(a, b);
        assert_eq!(a.parent(), Some(Path::new("autosave")));
        assert!(a.to_string_lossy().ends_with("level.orrin"));
    }

    #[test]
    fn a_save_deletes_the_copy() {
        let start = Instant::now();
        let dir = scratch("saved");
        let mut autosave = Autosave::new(dir.clone(), start);
        let path = autosave.path("level.orrin");
        autosave.write(document(), path.clone(), 2, start + INTERVAL);
        autosave.wait().unwrap().unwrap();
        assert!(path.is_file());

        autosave.saved("level.orrin", 2);
        assert!(!path.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
