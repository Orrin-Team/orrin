//! Rescan-on-save for the prefab index.
//!
//! The binding file's watcher, widened to a directory tree: a change raises a
//! request, and the rescan happens at one known point in the frame rather than
//! under whatever was mid-read when the OS delivered the event. Only prefabs
//! and their sidecars count, so an artist saving a texture next door costs
//! nothing.
//!
//! The scan a rescan runs can itself write a sidecar, which is heard as one
//! more change; that second scan finds nothing new and is the whole cost.
//!
//! Live rescans need `notify`, which this build only has with the `scripting`
//! feature; without it the editor's Rescan button is the way in.

use std::path::Path;
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use notify::{RecommendedWatcher, RecursiveMode, Watcher as _};

use super::index::EXTENSION;

/// How long the tree must stay quiet before it is rescanned: long enough to
/// outlast an editor's write-then-rename, short enough to feel immediate.
const DEBOUNCE: Duration = Duration::from_millis(150);

pub struct Watcher {
    /// Held only for its `Drop`, which stops the OS watch. Never read.
    _watcher: RecommendedWatcher,
    changes: Receiver<()>,
    pending_since: Option<Instant>,
}

impl Watcher {
    pub fn new(dir: &Path) -> Result<Self, String> {
        let (changes_tx, changes) = mpsc::channel();
        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                let Ok(event) = event else { return };
                if event.kind.is_access() {
                    return;
                }
                if event.paths.iter().any(|path| is_prefab_file(path)) {
                    let _ = changes_tx.send(());
                }
            })
            .map_err(|err| format!("could not watch {}: {err}", dir.display()))?;
        watcher
            .watch(dir, RecursiveMode::Recursive)
            .map_err(|err| format!("could not watch {}: {err}", dir.display()))?;

        Ok(Self {
            _watcher: watcher,
            changes,
            pending_since: None,
        })
    }

    /// Whether a rescan is due: something changed, and has since settled.
    pub fn due(&mut self) -> bool {
        if self.changes.try_iter().count() > 0 {
            self.pending_since = Some(Instant::now());
        }
        match self.pending_since {
            Some(since) if since.elapsed() >= DEBOUNCE => {
                self.pending_since = None;
                true
            }
            _ => false,
        }
    }
}

/// A prefab, or a prefab's sidecar.
fn is_prefab_file(path: &Path) -> bool {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let name = name.strip_suffix(".meta").unwrap_or(&name);
    name.strip_suffix(EXTENSION)
        .is_some_and(|stem| stem.ends_with('.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Waits on the OS, so it is bounded rather than exact: the event takes a
    /// few milliseconds, the debounce 150.
    fn due_within(watcher: &mut Watcher, limit: Duration) -> bool {
        let start = Instant::now();
        while start.elapsed() < limit {
            if watcher.due() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }

    #[test]
    fn saving_a_prefab_asks_for_a_rescan_and_other_files_do_not() {
        let dir = std::env::temp_dir().join(format!("orrin-watch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("props")).unwrap();
        let mut watcher = Watcher::new(&dir).unwrap();

        std::fs::write(dir.join("props/notes.txt"), "not a prefab").unwrap();
        assert!(!due_within(&mut watcher, Duration::from_millis(600)));

        std::fs::write(dir.join("props/door.fprefab"), "orrin-scene 1\n").unwrap();
        let due = due_within(&mut watcher, Duration::from_secs(5));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(due, "a saved prefab was not heard");
    }

    #[test]
    fn a_sidecar_counts_and_a_lookalike_does_not() {
        assert!(is_prefab_file(Path::new("a/door.fprefab")));
        assert!(is_prefab_file(Path::new("a/door.fprefab.meta")));
        assert!(!is_prefab_file(Path::new("a/door.png.meta")));
        assert!(!is_prefab_file(Path::new("a/notfprefab")));
    }
}
