//! Editor preferences: what the *user* chose, as opposed to what the project
//! is. Lives at `<project>/.orrin/editor.ron`, which is why none of it belongs
//! in `orrin.toml` — a manifest is checked in and shared, a preference is not.
//!
//! RON rather than TOML because the dock layout this file will hold is a tree
//! of enums, which TOML cannot express without inventing a schema for it. User
//! *themes* stay TOML: those are hand-authored, and this file never is.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

const FILE: &str = "editor.ron";

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    pub theme: Option<String>,
    /// The whole dock tree. `None` on a first run, and after a layout written
    /// by a build whose tab set no longer parses — [`PrefsFile::load`] falls
    /// back to defaults rather than refusing to open the editor.
    pub layout: Option<egui_dock::DockState<super::dock::Tab>>,
}

/// Where preferences are read and written, or `None` when the session has no
/// project — running the built-in demo from the repo root must not scatter an
/// `.orrin/` directory into whatever the working directory happens to be.
pub struct PrefsFile {
    dir: Option<PathBuf>,
}

impl PrefsFile {
    pub fn new(editor_dir: Option<PathBuf>) -> Self {
        Self { dir: editor_dir }
    }

    fn path(&self) -> Option<PathBuf> {
        self.dir.as_ref().map(|dir| dir.join(FILE))
    }

    /// A missing file is the first run and yields defaults; a *malformed* one is
    /// named on stderr and then also yields defaults, because refusing to open
    /// the editor over a stale preference would be the worse failure.
    pub fn load(&self) -> Prefs {
        let Some(path) = self.path() else {
            return Prefs::default();
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Prefs::default();
        };
        match ron::from_str(&text) {
            Ok(prefs) => prefs,
            Err(e) => {
                eprintln!("orrin: ignoring {} — {e}", path.display());
                Prefs::default()
            }
        }
    }

    pub fn save(&self, prefs: &Prefs) {
        let (Some(dir), Some(path)) = (self.dir.as_ref(), self.path()) else {
            return;
        };
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("orrin: cannot create {}: {e}", dir.display());
            return;
        }
        let text = match ron::ser::to_string_pretty(prefs, ron::ser::PrettyConfig::default()) {
            Ok(text) => text,
            Err(e) => {
                eprintln!("orrin: cannot serialise editor preferences: {e}");
                return;
            }
        };
        if let Err(e) = std::fs::write(&path, text) {
            eprintln!("orrin: cannot write {}: {e}", path.display());
        }
    }
}
