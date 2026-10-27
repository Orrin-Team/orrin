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

/// Bumped whenever the default arrangement changes shape, so a layout saved
/// against the old one gives way to the new default instead of preserving a
/// tree of tools that no longer sit where the editor expects them.
const LAYOUT_VERSION: u32 = 2;

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    pub theme: Option<String>,
    /// The whole dock tree. `None` on a first run, and after a layout written
    /// by a build whose tab set no longer parses — [`PrefsFile::load`] falls
    /// back to defaults rather than refusing to open the editor.
    pub layout: Option<egui_dock::DockState<super::dock::Tab>>,
    /// What the gizmo snaps to; `None` until the user first changes it.
    pub snap: Option<super::state::SnapSteps>,
    /// How the asset browser lays out and orders a folder; `None` until the
    /// user first changes it.
    pub assets: Option<super::state::AssetView>,
    /// Which [`LAYOUT_VERSION`] `layout` was saved under. Absent, and so zero,
    /// in every file written before the field existed.
    layout_version: u32,
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
        match ron::from_str::<Prefs>(&text) {
            Ok(mut prefs) => {
                if prefs.layout_version != LAYOUT_VERSION {
                    prefs.layout = None;
                }
                prefs
            }
            Err(e) => {
                tracing::warn!("ignoring {} — {e}", path.display());
                Prefs::default()
            }
        }
    }

    pub fn save(&self, prefs: &Prefs) {
        let (Some(dir), Some(path)) = (self.dir.as_ref(), self.path()) else {
            return;
        };
        if let Err(e) = std::fs::create_dir_all(dir) {
            tracing::error!("cannot create {}: {e}", dir.display());
            return;
        }
        let prefs = &Prefs {
            layout_version: LAYOUT_VERSION,
            theme: prefs.theme.clone(),
            layout: prefs.layout.clone(),
            snap: prefs.snap,
            assets: prefs.assets,
        };
        let text = match ron::ser::to_string_pretty(prefs, ron::ser::PrettyConfig::default()) {
            Ok(text) => text,
            Err(e) => {
                tracing::error!("cannot serialise editor preferences: {e}");
                return;
            }
        };
        if let Err(e) = std::fs::write(&path, text) {
            tracing::error!("cannot write {}: {e}", path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("orrin-prefs-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A layout saved by an editor whose default arrangement has since changed
    /// is dropped, so the new default is what the user sees — but the rest of
    /// the file, the theme they picked, survives.
    #[test]
    fn a_layout_from_an_older_arrangement_is_dropped_and_the_theme_kept() {
        let dir = scratch("old-layout");
        let layout = ron::to_string(&super::super::dock::default_layout()).unwrap();
        std::fs::write(
            dir.join(FILE),
            format!("(theme: Some(\"Void\"), layout: Some({layout}))"),
        )
        .unwrap();

        let prefs = PrefsFile::new(Some(dir.clone())).load();
        assert_eq!(prefs.theme.as_deref(), Some("Void"));
        assert!(prefs.layout.is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn snap_steps_round_trip() {
        let dir = scratch("snap-steps");
        let file = PrefsFile::new(Some(dir.clone()));
        let steps = super::super::state::SnapSteps {
            translate: 0.5,
            rotate: 22.5,
            scale: 0.25,
        };
        file.save(&Prefs {
            snap: Some(steps),
            ..Prefs::default()
        });

        assert_eq!(file.load().snap, Some(steps));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_asset_browsers_view_round_trips_without_its_filter() {
        use super::super::state::{AssetLayout, AssetSort, AssetView};
        let dir = scratch("asset-view");
        let file = PrefsFile::new(Some(dir.clone()));
        let view = AssetView {
            layout: AssetLayout::Grid,
            sort: AssetSort::Modified,
            descending: true,
            kind: Some(crate::asset_db::AssetKind::Texture),
        };
        file.save(&Prefs {
            assets: Some(view),
            ..Prefs::default()
        });
        let back = file.load().assets.unwrap();
        assert_eq!(back, AssetView { kind: None, ..view });
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_current_layout_round_trips() {
        let dir = scratch("current-layout");
        let file = PrefsFile::new(Some(dir.clone()));
        file.save(&Prefs {
            theme: None,
            layout: Some(super::super::dock::default_layout()),
            ..Prefs::default()
        });

        assert!(file.load().layout.is_some());
        let _ = std::fs::remove_dir_all(dir);
    }
}
