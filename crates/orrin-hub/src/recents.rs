//! The projects the hub lists: every one created or opened through it, most
//! recent first, kept in one small JSON file per user.
//!
//! A project is remembered by its directory, canonicalised, so the same folder
//! reached through a symlink or a trailing slash is one entry rather than two.
//! A remembered directory that has since gone is kept and shown as missing
//! rather than silently dropped: a project on an unplugged drive comes back
//! when the drive does.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use orrin_project::export::Platform;
use serde::{Deserialize, Serialize};

/// How many projects are remembered.
pub const LIMIT: usize = 24;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Recent {
    pub path: PathBuf,
    /// Seconds since the Unix epoch.
    pub opened: u64,
    /// Starred, which lists it above every project that is not.
    #[serde(default)]
    pub favourite: bool,
    /// What Export builds for from this project's row; `None` is this machine.
    #[serde(default)]
    pub platform: Option<Platform>,
}

impl Recent {
    /// The project's name from its manifest, or its folder's when the manifest
    /// does not read.
    pub fn name(&self) -> String {
        orrin_project::Project::load(&self.path.join("orrin.toml")).map_or_else(
            |_| {
                self.path.file_name().map_or_else(
                    || self.path.display().to_string(),
                    |name| name.to_string_lossy().into_owned(),
                )
            },
            |project| project.name().to_owned(),
        )
    }

    pub fn is_missing(&self) -> bool {
        !self.path.join("orrin.toml").is_file()
    }
}

#[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Recents {
    pub projects: Vec<Recent>,
}

impl Recents {
    /// Read `file`, or start empty when it is missing or unreadable: a lost
    /// list of shortcuts is not worth refusing to open over.
    pub fn load(file: &Path) -> Self {
        std::fs::read_to_string(file)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, file: &Path) -> Result<(), String> {
        if let Some(dir) = file.parent() {
            std::fs::create_dir_all(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
        }
        let text = serde_json::to_string_pretty(self).map_err(|error| error.to_string())?;
        std::fs::write(file, text).map_err(|error| format!("{}: {error}", file.display()))
    }

    /// Put `dir` first, as opened at `now`.
    pub fn touch(&mut self, dir: &Path, now: u64) {
        let path = canonical(dir);
        let kept = self.projects.iter().position(|recent| recent.path == path);
        let mut recent = kept.map_or(
            Recent {
                path,
                opened: now,
                favourite: false,
                platform: None,
            },
            |at| self.projects.remove(at),
        );
        recent.opened = now;
        self.projects.insert(0, recent);
        self.projects.truncate(LIMIT);
    }

    pub fn toggle_favourite(&mut self, dir: &Path) {
        if let Some(recent) = self.find_mut(dir) {
            recent.favourite = !recent.favourite;
        }
    }

    pub fn set_platform(&mut self, dir: &Path, platform: Platform) {
        if let Some(recent) = self.find_mut(dir) {
            recent.platform = Some(platform);
        }
    }

    fn find_mut(&mut self, dir: &Path) -> Option<&mut Recent> {
        let path = canonical(dir);
        self.projects.iter_mut().find(|recent| recent.path == path)
    }

    pub fn forget(&mut self, dir: &Path) {
        let path = canonical(dir);
        self.projects.retain(|recent| recent.path != path);
    }
}

impl Recent {
    /// The platform this row exports for: its own pick, else this machine.
    pub fn target(&self) -> Platform {
        self.platform
            .or_else(Platform::host)
            .unwrap_or(Platform::Linux)
    }
}

/// What a row shows that has to be read off the disk. Read once in a while
/// rather than every frame: a list of projects is a handful of `stat`s each.
#[derive(Clone, Debug, PartialEq)]
pub struct Details {
    pub name: String,
    pub missing: bool,
    /// Inside a git repository.
    pub git: bool,
    /// The newest change the project's own files show, in epoch seconds.
    pub modified: Option<u64>,
}

impl Details {
    pub fn of(recent: &Recent) -> Self {
        let root = &recent.path;
        Self {
            name: recent.name(),
            missing: recent.is_missing(),
            git: root.ancestors().any(|dir| dir.join(".git").exists()),
            modified: modified(root),
        }
    }
}

/// The column a list is sorted by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sort {
    Name,
    Modified,
}

/// The rows to show, in order: favourites first, then by `sort`, keeping only
/// those whose name or path holds `search` and, when one is picked, those
/// exporting for `platform`.
pub fn order<'a>(
    rows: &'a [(Recent, Details)],
    sort: Sort,
    ascending: bool,
    search: &str,
    platform: Option<Platform>,
) -> Vec<&'a (Recent, Details)> {
    let search = search.trim().to_lowercase();
    let mut shown: Vec<_> = rows
        .iter()
        .filter(|(recent, details)| {
            search.is_empty()
                || details.name.to_lowercase().contains(&search)
                || recent
                    .path
                    .to_string_lossy()
                    .to_lowercase()
                    .contains(&search)
        })
        .filter(|(recent, _)| platform.is_none_or(|platform| recent.target() == platform))
        .collect();
    shown.sort_by(|(a, a_details), (b, b_details)| {
        let by_column = match sort {
            Sort::Name => a_details
                .name
                .to_lowercase()
                .cmp(&b_details.name.to_lowercase()),
            Sort::Modified => a_details.modified.cmp(&b_details.modified),
        };
        let by_column = if ascending {
            by_column
        } else {
            by_column.reverse()
        };
        b.favourite.cmp(&a.favourite).then(by_column)
    });
    shown
}

/// The newest write among the files a person edits: the manifest, and what
/// sits directly in the project and its assets, scripts and scenes. Shallow
/// on purpose, and blind to what tools write — `bin/`, `obj/`, `.orrin/`,
/// `exports/` — or every build and export would read as an edit.
fn modified(root: &Path) -> Option<u64> {
    const GENERATED: [&str; 4] = ["bin", "obj", ".orrin", "exports"];
    let stamp = |path: &Path| {
        path.metadata()
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(|at| at.duration_since(UNIX_EPOCH).ok())
            .map(|since| since.as_secs())
    };
    let mut newest = stamp(&root.join("orrin.toml"));
    for dir in [
        root.to_path_buf(),
        root.join("assets"),
        root.join("scripts"),
        root.join("scenes"),
    ] {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            if GENERATED.iter().any(|skip| name == *skip) {
                continue;
            }
            newest = newest.max(stamp(&entry.path()));
        }
    }
    newest
}

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// When something happened, as a person says it.
pub fn ago(then: u64, now: u64) -> String {
    let seconds = now.saturating_sub(then);
    let (count, unit) = match seconds {
        0..60 => return "just now".to_owned(),
        60..3_600 => (seconds / 60, "minute"),
        3_600..86_400 => (seconds / 3_600, "hour"),
        86_400..172_800 => return "yesterday".to_owned(),
        _ => (seconds / 86_400, "day"),
    };
    let plural = if count == 1 { "" } else { "s" };
    format!("{count} {unit}{plural} ago")
}

/// `dir` canonicalised, or — once it has gone — its parent canonicalised with
/// its name joined back, so a missing project still matches the entry it had.
fn canonical(dir: &Path) -> PathBuf {
    if let Ok(path) = dir.canonicalize() {
        return path;
    }
    match (
        dir.parent().and_then(|parent| parent.canonicalize().ok()),
        dir.file_name(),
    ) {
        (Some(parent), Some(name)) => parent.join(name),
        _ => dir.to_path_buf(),
    }
}

/// Where the list lives: the platform's per-user settings directory.
pub fn file() -> Option<PathBuf> {
    let dir = if cfg!(target_os = "macos") {
        std::env::var_os("HOME")
            .map(|home| PathBuf::from(home).join("Library/Application Support/Orrin"))
    } else if cfg!(windows) {
        std::env::var_os("APPDATA").map(|dir| PathBuf::from(dir).join("Orrin"))
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
            .map(|dir| dir.join("orrin"))
    }?;
    Some(dir.join("hub.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(parent: &Path, name: &str) -> PathBuf {
        let dir = parent.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("orrin.toml"),
            format!(
                "format_version = 1\n[project]\nname = \"{name}\"\nversion = \"1\"\n\
                 [scripts]\ndir = \"scripts\"\nentry = \"A.B, A\"\n"
            ),
        )
        .unwrap();
        dir
    }

    #[test]
    fn opening_a_project_again_moves_it_to_the_top_without_a_duplicate() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (project(dir.path(), "a"), project(dir.path(), "b"));
        let mut recents = Recents::default();

        recents.touch(&a, 10);
        recents.touch(&b, 20);
        recents.touch(&a.join("."), 30);

        let names: Vec<_> = recents.projects.iter().map(Recent::name).collect();
        assert_eq!(names, ["a", "b"]);
        assert_eq!(recents.projects[0].opened, 30);
    }

    #[test]
    fn the_list_is_capped_and_the_oldest_falls_off() {
        let dir = tempfile::tempdir().unwrap();
        let mut recents = Recents::default();
        for index in 0..LIMIT + 3 {
            recents.touch(&project(dir.path(), &format!("p{index}")), index as u64);
        }
        assert_eq!(recents.projects.len(), LIMIT);
        assert_eq!(recents.projects[0].name(), format!("p{}", LIMIT + 2));
    }

    #[test]
    fn the_list_survives_a_restart_and_a_broken_file_starts_empty() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("settings/hub.json");
        let mut recents = Recents::default();
        recents.touch(&project(dir.path(), "kept"), 5);
        recents.save(&file).unwrap();

        assert_eq!(Recents::load(&file), recents);

        std::fs::write(&file, "{ not json").unwrap();
        assert_eq!(Recents::load(&file), Recents::default());
    }

    #[test]
    fn a_project_that_has_gone_is_kept_and_shown_missing_until_forgotten() {
        let dir = tempfile::tempdir().unwrap();
        let gone = project(dir.path(), "gone");
        let mut recents = Recents::default();
        recents.touch(&gone, 1);
        std::fs::remove_dir_all(&gone).unwrap();

        assert!(recents.projects[0].is_missing());
        assert_eq!(recents.projects[0].name(), "gone", "named after its folder");

        recents.forget(&gone);
        assert!(recents.projects.is_empty());
    }

    #[test]
    fn times_read_the_way_a_person_says_them() {
        let now = 1_000_000;
        assert_eq!(ago(now - 5, now), "just now");
        assert_eq!(ago(now - 60, now), "1 minute ago");
        assert_eq!(ago(now - 60 * 25, now), "25 minutes ago");
        assert_eq!(ago(now - 3600 * 3, now), "3 hours ago");
        assert_eq!(ago(now - 86_400, now), "yesterday");
        assert_eq!(ago(now - 86_400 * 4, now), "4 days ago");
        assert_eq!(ago(now + 50, now), "just now", "a clock that moved back");
    }

    fn row(
        dir: &Path,
        name: &str,
        modified: u64,
        favourite: bool,
        platform: Option<Platform>,
    ) -> (Recent, Details) {
        (
            Recent {
                path: dir.join(name),
                opened: 0,
                favourite,
                platform,
            },
            Details {
                name: name.to_owned(),
                missing: false,
                git: false,
                modified: Some(modified),
            },
        )
    }

    fn names(rows: &[&(Recent, Details)]) -> Vec<String> {
        rows.iter()
            .map(|(_, details)| details.name.clone())
            .collect()
    }

    #[test]
    fn favourites_lead_whatever_the_sort_and_the_rest_follow_it() {
        let dir = Path::new("/projects");
        let rows = [
            row(dir, "beta", 30, false, None),
            row(dir, "Alpha", 10, false, None),
            row(dir, "gamma", 20, true, None),
        ];

        assert_eq!(
            names(&order(&rows, Sort::Name, true, "", None)),
            ["gamma", "Alpha", "beta"]
        );
        assert_eq!(
            names(&order(&rows, Sort::Modified, false, "", None)),
            ["gamma", "beta", "Alpha"]
        );
        assert_eq!(
            names(&order(&rows, Sort::Modified, true, "", None)),
            ["gamma", "Alpha", "beta"]
        );
    }

    #[test]
    fn search_and_the_platform_filter_narrow_the_list() {
        let dir = Path::new("/projects");
        let rows = [
            row(dir, "space-shooter", 1, false, Some(Platform::Windows)),
            row(dir, "farm", 2, false, Some(Platform::Mac)),
        ];

        assert_eq!(
            names(&order(&rows, Sort::Name, true, "SHOOT", None)),
            ["space-shooter"]
        );
        assert_eq!(
            names(&order(&rows, Sort::Name, true, "/projects/farm", None)),
            ["farm"]
        );
        assert_eq!(
            names(&order(&rows, Sort::Name, true, "", Some(Platform::Mac))),
            ["farm"]
        );
    }

    #[test]
    fn a_star_and_a_platform_are_kept_through_reopening_and_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let game = project(dir.path(), "game");
        let file = dir.path().join("hub.json");
        let mut recents = Recents::default();
        recents.touch(&game, 1);

        recents.toggle_favourite(&game);
        recents.set_platform(&game, Platform::Windows);
        recents.touch(&game, 2);
        recents.save(&file).unwrap();

        let back = Recents::load(&file);
        assert!(back.projects[0].favourite);
        assert_eq!(back.projects[0].target(), Platform::Windows);
        assert_eq!(back.projects[0].opened, 2);
    }

    /// A list written before stars and platforms existed still loads.
    #[test]
    fn a_list_from_an_older_hub_still_loads() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("hub.json");
        std::fs::write(&file, r#"{"projects":[{"path":"/a","opened":5}]}"#).unwrap();

        let recents = Recents::load(&file);
        assert_eq!(recents.projects.len(), 1);
        assert!(!recents.projects[0].favourite);
        assert_eq!(recents.projects[0].platform, None);
    }

    #[test]
    fn details_read_the_name_git_and_the_newest_change_off_the_disk() {
        let dir = tempfile::tempdir().unwrap();
        let game = project(dir.path(), "game");
        std::fs::create_dir_all(game.join("assets")).unwrap();
        std::fs::write(game.join("assets/rock.png"), b"").unwrap();
        let recent = Recent {
            path: canonical(&game),
            opened: 0,
            favourite: false,
            platform: None,
        };

        let details = Details::of(&recent);
        assert_eq!(details.name, "game");
        assert!(!details.missing);
        assert!(!details.git);
        assert!(
            details.modified.is_some_and(|at| at + 5 >= now()),
            "{details:?}"
        );

        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        assert!(
            Details::of(&recent).git,
            "a repository above the project counts"
        );
    }
}
