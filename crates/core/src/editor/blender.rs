//! Settings ▸ Blender's Install add-on: the Orrin Live Link add-on from
//! `tools/blender/orrin_link`, compiled into the editor so an installed editor
//! needs no repository beside it and installs the add-on that speaks its own
//! protocol. An exported game carries none of it.
//!
//! **The route is Blender's own extension install**, the one `blender
//! --command extension install-file --enable` takes: the files are zipped by
//! Blender's Python, installed into the user's local extension repository,
//! enabled, and the preferences saved so it stays enabled. Checked against
//! Blender 5.2.3 LTS only. The manifest asks for 4.2, the first with
//! extensions; a Blender before it fails the install for want of the operator.
//!
//! **The user's preferences are what get saved.** The run starts with them —
//! never `--factory-startup`, which would save factory defaults over every
//! preference the user has — and saving afterwards writes what it loaded plus
//! the one add-on enabled. A Blender left open keeps the preferences it started
//! with and may save them over this on quit, so the page says to restart it.
//!
//! **Off the frame's thread, and bounded.** Blender runs on a thread of its
//! own, its output to a file, and is killed past [`TIMEOUT`]; a failure carries
//! the tail of what it printed. The files are written to a directory of the
//! install's own, created afresh and readable by its owner alone, because
//! what is in it is code that Blender then runs.

use std::path::{Path, PathBuf};
#[cfg(not(feature = "export"))]
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, TryRecvError};
#[cfg(not(feature = "export"))]
use std::time::{Duration, Instant};

/// How long an install may take before Blender is stopped. One took 0.6 s on
/// an M5 Pro with Blender 5.2.3, nearly all of it Blender starting.
#[cfg(not(feature = "export"))]
const TIMEOUT: Duration = Duration::from_secs(60);

#[cfg(not(feature = "export"))]
const FILES: [(&str, &str); 5] = [
    (
        "__init__.py",
        include_str!("../../../../tools/blender/orrin_link/__init__.py"),
    ),
    (
        "blender_manifest.toml",
        include_str!("../../../../tools/blender/orrin_link/blender_manifest.toml"),
    ),
    (
        "client.py",
        include_str!("../../../../tools/blender/orrin_link/client.py"),
    ),
    (
        "extract.py",
        include_str!("../../../../tools/blender/orrin_link/extract.py"),
    ),
    (
        "live.py",
        include_str!("../../../../tools/blender/orrin_link/live.py"),
    ),
];

/// Run over the directory after `--`, which holds `orrin_link/`.
#[cfg(not(feature = "export"))]
const INSTALL: &str = r#"
import bpy, os, sys, zipfile
source = sys.argv[sys.argv.index("--") + 1]
repos = [r for r in bpy.context.preferences.extensions.repos
         if r.enabled and r.source == "USER" and not r.use_remote_url]
repo = next((r for r in repos if r.module == "user_default"), repos[0] if repos else None)
if repo is None:
    raise RuntimeError("Blender has no local extension repository to install into; add one in Preferences > Get Extensions > Repositories")
archive = os.path.join(source, "orrin_link.zip")
package = os.path.join(source, "orrin_link")
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for name in sorted(os.listdir(package)):
        z.write(os.path.join(package, name), name)
bpy.ops.extensions.package_install_files(filepath=archive, repo=repo.module, enable_on_install=True)
if "bl_ext." + repo.module + ".orrin_link" not in bpy.context.preferences.addons:
    raise RuntimeError("Blender installed the add-on but did not enable it")
if "FINISHED" not in bpy.ops.wm.save_userpref():
    raise RuntimeError("Blender could not save its preferences, so the add-on would be off when it next starts")
"#;

/// An install running, or how the last one ended.
#[derive(Default)]
pub struct Install {
    running: Option<Receiver<Result<String, String>>>,
    last: Option<Result<String, String>>,
}

impl Install {
    pub fn is_running(&self) -> bool {
        self.running.is_some()
    }

    pub fn last(&self) -> Option<&Result<String, String>> {
        self.last.as_ref()
    }

    /// Install into `blender` on a thread of its own, unless one is running.
    pub fn start(&mut self, blender: PathBuf) {
        if self.running.is_some() {
            return;
        }
        let (send, done) = mpsc::channel();
        let spawned = std::thread::Builder::new()
            .name("orrin-addon".to_owned())
            .spawn(move || {
                let _ = send.send(install(&blender));
            });
        match spawned {
            Ok(_) => self.running = Some(done),
            Err(error) => {
                self.last = Some(Err(format!(
                    "could not start installing the add-on: {error}"
                )));
            }
        }
    }

    /// How the running install ended, once, when it has.
    pub fn service(&mut self) -> Option<&Result<String, String>> {
        let ended = match self.running.as_ref()?.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => {
                Err("installing the add-on stopped with no result".to_owned())
            }
        };
        self.running = None;
        self.last = Some(ended);
        self.last.as_ref()
    }
}

#[cfg(feature = "export")]
fn install(_: &Path) -> Result<String, String> {
    Err("an exported game carries no Blender add-on".to_owned())
}

#[cfg(not(feature = "export"))]
fn install(blender: &Path) -> Result<String, String> {
    install_with(Command::new(blender), blender)
}

/// `command` is Blender, with whatever environment it is to run in.
#[cfg(not(feature = "export"))]
fn install_with(mut command: Command, blender: &Path) -> Result<String, String> {
    let failed = |why: String| format!("the Blender add-on could not be installed: {why}");
    let scratch = Scratch::new().map_err(failed)?;
    let package = scratch.0.join("orrin_link");
    let written = |path: &Path, error: std::io::Error| {
        failed(format!("{} could not be written: {error}", path.display()))
    };
    std::fs::create_dir(&package).map_err(|error| written(&package, error))?;
    for (name, text) in FILES {
        let path = package.join(name);
        std::fs::write(&path, text).map_err(|error| written(&path, error))?;
    }
    let log = scratch.0.join("blender.log");
    let output = std::fs::File::create(&log).map_err(|error| written(&log, error))?;
    let errors = output.try_clone().map_err(|error| written(&log, error))?;

    let mut child = command
        .args([
            "--background",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python-expr",
            INSTALL,
            "--",
        ])
        .arg(&scratch.0)
        .current_dir(&scratch.0)
        .stdin(Stdio::null())
        .stdout(output)
        .stderr(errors)
        .spawn()
        .map_err(|error| {
            failed(format!(
                "Blender would not start from {}: {error}. Set Blender's path above",
                blender.display()
            ))
        })?;
    let deadline = Instant::now() + TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(20));
            }
            waited => {
                let _ = child.kill();
                let _ = child.wait();
                let why = match waited {
                    Err(error) => format!("Blender could not be waited for: {error}"),
                    _ => format!(
                        "Blender did not finish within {} s and was stopped",
                        TIMEOUT.as_secs()
                    ),
                };
                return Err(failed(format!(
                    "{why}.{}",
                    crate::asset_db::blend::printed(&log)
                )));
            }
        }
    };
    if !status.success() {
        return Err(failed(format!(
            "Blender exited with {status}.{}",
            crate::asset_db::blend::printed(&log)
        )));
    }
    Ok(format!(
        "the Orrin Live Link add-on is installed and enabled in {}; restart Blender if it is open",
        blender.display()
    ))
}

/// A directory of the install's own, removed when dropped.
#[cfg(not(feature = "export"))]
struct Scratch(PathBuf);

#[cfg(not(feature = "export"))]
impl Scratch {
    /// Created rather than created-if-absent, so a directory somebody else
    /// made under the name first is never written into.
    fn new() -> Result<Self, String> {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.subsec_nanos());
        let dir = std::env::temp_dir().join(format!(
            "orrin-addon-{}-{}-{nanos}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        builder
            .create(&dir)
            .map_err(|error| format!("{} could not be created: {error}", dir.display()))?;
        Ok(Self(dir))
    }
}

#[cfg(not(feature = "export"))]
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(all(test, not(feature = "export")))]
mod tests {
    use super::*;
    use crate::asset_db::tests::Dir;

    /// The files compiled in are the add-on in the repository, every one of
    /// them, so the two cannot drift.
    #[test]
    fn every_file_of_the_add_on_is_compiled_in() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/blender/orrin_link");
        let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name != "__pycache__")
            .collect();
        on_disk.sort();
        let compiled: Vec<&str> = FILES.iter().map(|(name, _)| *name).collect();
        assert_eq!(on_disk, compiled);
    }

    /// The real install, into a scratch profile: Blender is pointed at it with
    /// `BLENDER_USER_RESOURCES`, so this machine's own Blender is not touched.
    /// A second Blender on the same profile finds the add-on enabled and the
    /// preference set before the install still set, and turns the add-on off
    /// and on twice, and off with the link running, leaving no thread or
    /// handler behind.
    #[test]
    #[ignore = "needs Blender"]
    fn the_add_on_installs_into_a_scratch_profile_and_stays_enabled() {
        let blender = crate::asset_db::blend::locate(std::env::var_os("ORRIN_BLENDER"), None)
            .expect("no Blender found; set ORRIN_BLENDER");
        let profile = Dir::new("addon-profile");
        let scratch = |command: &mut Command| {
            command.env("BLENDER_USER_RESOURCES", profile.path());
        };

        // A preference of the user's own, which the install must keep.
        let mut command = Command::new(&blender);
        scratch(&mut command);
        let kept = command
            .args(["--background", "--python-exit-code", "1", "--python-expr"])
            .arg(
                "import bpy
bpy.context.preferences.view.ui_scale = 1.25
bpy.ops.wm.save_userpref()",
            )
            .output()
            .unwrap();
        assert!(kept.status.success(), "{kept:?}");

        let started = Instant::now();
        let mut command = Command::new(&blender);
        scratch(&mut command);
        let installed = install_with(command, &blender);
        assert!(installed.is_ok(), "{installed:?}");
        eprintln!("installed in {:?}", started.elapsed());
        assert!(profile.path().join("config/userpref.blend").is_file());
        assert!(
            profile
                .path()
                .join("extensions/user_default/orrin_link/blender_manifest.toml")
                .is_file()
        );

        let mut command = Command::new(&blender);
        scratch(&mut command);
        let checked = command
            .args(["--background", "--python-exit-code", "1", "--python-expr"])
            .arg(
                "import bpy, addon_utils, threading\n\
                 name = 'bl_ext.user_default.orrin_link'\n\
                 assert name in bpy.context.preferences.addons, 'enabled at start'\n\
                 assert bpy.context.preferences.view.ui_scale == 1.25, 'the user preferences kept'\n\
                 handlers = bpy.app.handlers\n\
                 counts = lambda: [len(h) for h in (handlers.load_pre, handlers.load_post, \
                 handlers.save_post, handlers.depsgraph_update_post)]\n\
                 enabled = counts()\n\
                 for _ in range(2):\n\
                 \x20   addon_utils.disable(name)\n\
                 \x20   addon_utils.enable(name)\n\
                 assert counts() == enabled, 'twice off and on'\n\
                 bpy.context.window_manager.orrin_link = True\n\
                 assert bpy.context.window_manager.orrin_link\n\
                 assert any(t.name == 'orrin-link' for t in threading.enumerate())\n\
                 assert counts() == [n + 1 for n in enabled], 'the session follows the file'\n\
                 addon_utils.disable(name)\n\
                 assert not any(t.name.startswith('orrin-link') for t in threading.enumerate())\n\
                 assert counts() == [enabled[0], enabled[1] - 1, *enabled[2:]], 'nothing left'\n\
                 addon_utils.enable(name)\n\
                 assert counts() == enabled\n\
                 print('@orrin checked')",
            )
            .output()
            .unwrap();
        let out = String::from_utf8_lossy(&checked.stdout);
        assert!(
            checked.status.success() && out.contains("@orrin checked"),
            "{out}\n{}",
            String::from_utf8_lossy(&checked.stderr)
        );
    }
}
