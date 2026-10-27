//! Reimport-on-save: which paths under the assets directory changed.
//!
//! The same shape as the binding file's watcher: the OS thread only records a
//! path, and the database acts on it at one known point between frames, once
//! the directory has been quiet for [`DEBOUNCE`]. An editor's save is a write,
//! a rename and a sidecar touch in quick succession, and reimporting on the
//! first of them would read a half-written file.
//!
//! The whole directory is watched recursively rather than file by file, so a
//! file created after startup is seen, and a save that replaces a file by
//! renaming a temporary over it is seen as the change it is.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use notify::{RecommendedWatcher, RecursiveMode, Watcher};

/// How long the directory must stay quiet before the changes are acted on.
pub const DEBOUNCE: Duration = Duration::from_millis(200);

pub struct AssetWatcher {
    /// Held only for its `Drop`, which stops the OS watch.
    _watcher: RecommendedWatcher,
    events: Receiver<PathBuf>,
    changed: Vec<PathBuf>,
    last_event: Option<Instant>,
}

impl AssetWatcher {
    pub fn new(dir: &Path) -> Result<Self, String> {
        let (sender, events) = mpsc::channel();
        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                let Ok(event) = event else { return };
                if event.kind.is_access() {
                    return;
                }
                for path in event.paths {
                    let _ = sender.send(path);
                }
            })
            .map_err(|err| format!("could not watch {}: {err}", dir.display()))?;
        watcher
            .watch(dir, RecursiveMode::Recursive)
            .map_err(|err| format!("could not watch {}: {err}", dir.display()))?;
        Ok(Self {
            _watcher: watcher,
            events,
            changed: Vec::new(),
            last_event: None,
        })
    }

    /// The paths that changed, once they have settled; empty otherwise.
    pub fn settled(&mut self) -> Vec<PathBuf> {
        let before = self.changed.len();
        self.changed.extend(self.events.try_iter());
        if self.changed.len() > before {
            self.last_event = Some(Instant::now());
        }
        match self.last_event {
            Some(at) if at.elapsed() >= DEBOUNCE => {
                self.last_event = None;
                let mut changed = std::mem::take(&mut self.changed);
                changed.sort();
                changed.dedup();
                changed
            }
            _ => Vec::new(),
        }
    }
}
