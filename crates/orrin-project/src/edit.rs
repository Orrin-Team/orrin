//! Writing `orrin.toml`: the editor's settings change a project's manifest in
//! place.
//!
//! The manifest is a file people write and commit, so an edit changes the
//! *document* — through `toml_edit` — rather than serialising [`Manifest`]
//! back out. Re-serialising would be shorter and would throw away every
//! comment, blank line and key order the author chose, and turn a one-value
//! change into a whole-file diff in their history. So a value that is changed
//! keeps its line and its trailing comment; a key the author never wrote is
//! added at the end of its table; an optional key set to nothing is removed
//! rather than written empty, along with a `[scenes]` table it leaves empty.
//! A comment above a table that goes goes with it.
//!
//! Nothing reaches disk that [`Project::load`] would refuse, because a
//! manifest that fails to load is fatal to the engine. An edit starts only
//! from a file that loads now, never from one a person half-fixed by hand,
//! and `commit` parses and validates the new text with the loader's own
//! rules before it writes. It then writes a sibling file and renames it over
//! the manifest, so a crash leaves the old file or the new one and never a
//! truncated mix. On top of the loader's rules it refuses an empty name or
//! version and an entry that names no assembly: the loader accepts those for
//! files that predate them, but nothing should write a new one.
//!
//! Paths are written with forward slashes on every platform, because the
//! manifest is shared between machines.
//!
//! It deliberately does not notice that the file changed on disk between
//! [`Project::edit`] and [`ManifestEdit::commit`] — the last commit wins —
//! and it does not edit `scripts.dir` or `assets.dir`, which move files
//! rather than settings. `toml_edit` writes `\n` throughout, so a manifest
//! saved with `\r\n` comes back with `\n`. An exported game has `orrin.json`
//! and nothing to edit.
//!
//! [`Manifest`]: crate::Manifest

use std::io::Write as _;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use toml_edit::{DocumentMut, Item, Table, TableLike, Value};

use crate::{Manifest, Project, ProjectError, check_path};

/// A pending change to a project's `orrin.toml`. Nothing is written until
/// [`commit`](Self::commit); dropping it discards the change.
pub struct ManifestEdit {
    root: PathBuf,
    path: PathBuf,
    document: DocumentMut,
    /// The first value a setter could not write, which `commit` returns.
    refused: Option<ProjectError>,
}

impl Project {
    /// `<root>/orrin.toml`.
    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("orrin.toml")
    }

    /// Begin an edit of the manifest as it is on disk now. Fails if the file
    /// cannot be read or would not load, so an edit never overwrites a file
    /// somebody is in the middle of fixing.
    pub fn edit(&self) -> Result<ManifestEdit, ProjectError> {
        if self.game_assembly.is_some() {
            return Err(ProjectError::Exported {
                root: self.root.clone(),
            });
        }
        let path = self.manifest_path();
        let text = std::fs::read_to_string(&path).map_err(|source| ProjectError::Io {
            path: path.clone(),
            source,
        })?;
        crate::parse(&path, &text)?;
        let document = text
            .parse::<DocumentMut>()
            .map_err(|err| ProjectError::Parse {
                path: path.clone(),
                message: err.to_string(),
            })?;
        Ok(ManifestEdit {
            root: self.root.clone(),
            path,
            document,
            refused: None,
        })
    }
}

impl ManifestEdit {
    pub fn set_name(&mut self, name: &str) {
        self.set("project", "name", name);
    }
    pub fn set_version(&mut self, version: &str) {
        self.set("project", "version", version);
    }
    /// The assembly-qualified entry Behaviour, `Namespace.Type, Assembly`.
    pub fn set_entry(&mut self, entry: &str) {
        self.set("scripts", "entry", entry);
    }
    /// Relative to the project root; `None` removes the key.
    pub fn set_start_scene(&mut self, scene: Option<&Path>) {
        self.set_scene("start", "scenes.start", scene);
    }
    /// Relative to the project root; `None` removes the key.
    pub fn set_persistent_scene(&mut self, scene: Option<&Path>) {
        self.set_scene("persistent", "scenes.persistent", scene);
    }

    /// Validate the edited manifest as [`Project::load`] would, then replace
    /// the file. On any error the file is untouched.
    pub fn commit(self) -> Result<Project, ProjectError> {
        if let Some(err) = self.refused {
            return Err(err);
        }
        let text = self.document.to_string();
        let manifest = crate::parse(&self.path, &text)?;
        check_written_values(&manifest)?;
        replace(&self.path, &text)?;
        Ok(Project {
            root: self.root,
            manifest,
            game_assembly: None,
        })
    }

    fn set(&mut self, table: &str, key: &str, text: &str) {
        // `edit` loaded the document, and the loader requires both tables.
        let table = self.document[table]
            .as_table_like_mut()
            .expect("a manifest that loaded has this table");
        write(table, key, text);
    }

    fn set_scene(&mut self, key: &str, field: &'static str, scene: Option<&Path>) {
        let Some(scene) = scene else {
            if let Some(scenes) = self
                .document
                .get_mut("scenes")
                .and_then(Item::as_table_like_mut)
            {
                scenes.remove(key);
                if scenes.is_empty() {
                    self.document.remove("scenes");
                }
            }
            return;
        };
        let text = match manifest_spelling(field, scene) {
            Ok(text) => text,
            Err(err) => {
                self.refused.get_or_insert(err);
                return;
            }
        };
        if self.document.get("scenes").is_none() {
            let table = self.appended_table();
            self.document.insert("scenes", Item::Table(table));
        }
        let scenes = self.document["scenes"]
            .as_table_like_mut()
            .expect("`scenes` is a table: the loader refuses anything else");
        write(scenes, key, &text);
    }

    /// A table for the end of the file. It goes after the comments that end
    /// the file as well as after the last table, since those comments are
    /// usually about what came before them, and a blank line sets it apart.
    fn appended_table(&mut self) -> Table {
        let trailing = self.document.trailing().as_str().unwrap_or("").to_owned();
        self.document.set_trailing("");
        let mut table = Table::new();
        table.decor_mut().set_prefix(format!("{trailing}\n"));
        table
    }
}

/// Replaces the value but keeps its decor, which holds the comment after it on
/// the same line.
fn write(table: &mut dyn TableLike, key: &str, text: &str) {
    match table.get_mut(key).and_then(Item::as_value_mut) {
        Some(old) => {
            let decor = old.decor().clone();
            *old = Value::from(text);
            *old.decor_mut() = decor;
        }
        None => {
            table.insert(key, toml_edit::value(text));
        }
    }
}

fn manifest_spelling(field: &'static str, path: &Path) -> Result<String, ProjectError> {
    check_path(field, path)?;
    let mut parts = Vec::new();
    for component in path.components() {
        // `check_path` has refused every other kind but `.`, which says nothing.
        if let Component::Normal(part) = component {
            parts.push(part.to_str().ok_or_else(|| ProjectError::NotUnicode {
                field,
                value: path.to_path_buf(),
            })?);
        }
    }
    if parts.is_empty() {
        return Err(ProjectError::Empty { field });
    }
    Ok(parts.join("/"))
}

fn check_written_values(manifest: &Manifest) -> Result<(), ProjectError> {
    let project = &manifest.project;
    for (field, value) in [
        ("project.name", &project.name),
        ("project.version", &project.version),
    ] {
        if value.trim().is_empty() {
            return Err(ProjectError::Empty { field });
        }
    }
    // The engine and `orrin export` both take the game DLL's name from the
    // half after the comma.
    let entry = &manifest.scripts.entry;
    let qualified = entry.split_once(',').is_some_and(|(type_name, assembly)| {
        !type_name.trim().is_empty() && !assembly.trim().is_empty()
    });
    if !qualified {
        return Err(ProjectError::UnqualifiedEntry {
            value: entry.clone(),
        });
    }
    Ok(())
}

fn replace(path: &Path, text: &str) -> Result<(), ProjectError> {
    // Unique per call, so two edits in one process never share a file.
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let temp = path.with_file_name(format!(
        ".orrin.toml.{}.{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let written = std::fs::File::create_new(&temp)
        .and_then(|mut file| {
            file.write_all(text.as_bytes())?;
            file.sync_all()
        })
        .and_then(|()| std::fs::rename(&temp, path));
    written.map_err(|source| {
        let _ = std::fs::remove_file(&temp);
        ProjectError::Write {
            path: path.to_path_buf(),
            source,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::{ExportManifest, Platform};
    use tempfile::TempDir;

    const MANIFEST: &str = r#"# The game's manifest. Edited by hand and by the editor.
format_version = 1

[project]
# Bumped on every release.
version = "0.1.0" # semver
name = "hello-orrin"

[scripts]
entry = "HelloOrrin.Main, HelloOrrin"
dir = "scripts"


[assets]
dir = "assets"
"#;

    /// `examples/hello-orrin/orrin.toml` as it stands: no `[scenes]`, but a
    /// commented-out one at the end of the file.
    const COMMENTED_SCENES: &str = r#"format_version = 1

[project]
name = "hello-orrin"
version = "0.1.0"

[scripts]
dir = "scripts"
entry = "HelloOrrin.Main, HelloOrrin"

[assets]
dir = "assets"

# Uncomment once the scene format lands (issue #6 / scene format):
# [scenes]
# start = "scenes/main.fscene"
"#;

    fn project(body: &str) -> (TempDir, Project) {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("orrin.toml"), body).expect("write manifest");
        let project = Project::load(&dir.path().join("orrin.toml")).expect("valid manifest");
        (dir, project)
    }

    fn on_disk(project: &Project) -> String {
        std::fs::read_to_string(project.manifest_path()).expect("read manifest")
    }

    #[test]
    fn an_edit_changes_the_value_and_not_one_byte_besides() {
        let (_dir, project) = project(MANIFEST);
        let mut edit = project.edit().expect("edit");
        edit.set_version("0.2.0");
        edit.set_entry("HelloOrrin.Game, HelloOrrin");
        edit.commit().expect("commit");

        let expected = MANIFEST
            .replace(r#""0.1.0""#, r#""0.2.0""#)
            .replace("HelloOrrin.Main,", "HelloOrrin.Game,");
        assert_eq!(on_disk(&project), expected);
    }

    #[test]
    fn a_scene_with_no_table_appends_one_after_the_comments() {
        let (_dir, project) = project(COMMENTED_SCENES);
        let mut edit = project.edit().expect("edit");
        edit.set_start_scene(Some(Path::new("scenes/main.orrin")));
        edit.commit().expect("commit");

        let expected = format!("{COMMENTED_SCENES}\n[scenes]\nstart = \"scenes/main.orrin\"\n");
        assert_eq!(on_disk(&project), expected);
    }

    #[test]
    fn clearing_the_last_scene_removes_its_table_and_one_of_two_keeps_it() {
        let both = format!(
            "{MANIFEST}\n[scenes]\nstart = \"scenes/main.orrin\"\npersistent = \"scenes/hud.orrin\"\n"
        );
        let (_dir, project) = project(&both);

        let mut edit = project.edit().expect("edit");
        edit.set_start_scene(None);
        edit.commit().expect("commit");
        assert_eq!(
            on_disk(&project),
            format!("{MANIFEST}\n[scenes]\npersistent = \"scenes/hud.orrin\"\n")
        );

        let mut edit = project.edit().expect("edit");
        edit.set_persistent_scene(None);
        let reloaded = edit.commit().expect("commit");
        assert_eq!(on_disk(&project), MANIFEST);
        assert_eq!(reloaded.start_scene(), None);
        assert_eq!(reloaded.persistent_scene(), None);
    }

    #[test]
    fn scene_paths_are_written_with_forward_slashes() {
        let (_dir, project) = project(MANIFEST);
        let mut edit = project.edit().expect("edit");
        edit.set_start_scene(Some(&Path::new("scenes").join("levels").join("one.orrin")));
        edit.set_persistent_scene(Some(Path::new("./scenes/hud.orrin")));
        edit.commit().expect("commit");

        let text = on_disk(&project);
        assert!(
            text.contains("start = \"scenes/levels/one.orrin\"\n"),
            "{text}"
        );
        assert!(
            text.contains("persistent = \"scenes/hud.orrin\"\n"),
            "{text}"
        );
    }

    /// Backslashes are what a Windows file dialog hands back, and a manifest
    /// written with them would not load on anyone else's machine.
    #[cfg(windows)]
    #[test]
    fn a_windows_path_is_written_with_forward_slashes() {
        let (_dir, project) = project(MANIFEST);
        let mut edit = project.edit().expect("edit");
        edit.set_start_scene(Some(Path::new(r"scenes\levels\one.orrin")));
        edit.commit().expect("commit");
        assert!(on_disk(&project).contains("start = \"scenes/levels/one.orrin\"\n"));
    }

    #[test]
    fn nothing_invalid_reaches_disk_and_the_refusal_names_the_field() {
        type Change = fn(&mut ManifestEdit);
        let cases: [(&str, Change); 8] = [
            ("scenes.start absolute", |e| {
                e.set_start_scene(Some(Path::new("/scenes/main.orrin")))
            }),
            ("scenes.start empty", |e| {
                e.set_start_scene(Some(Path::new("")))
            }),
            ("scenes.persistent with ..", |e| {
                e.set_persistent_scene(Some(Path::new("scenes/../../hud.orrin")))
            }),
            ("project.name empty", |e| e.set_name("")),
            ("project.version blank", |e| e.set_version("  ")),
            ("scripts.entry bare", |e| e.set_entry("HelloOrrin.Main")),
            ("scripts.entry no assembly", |e| {
                e.set_entry("HelloOrrin.Main, ")
            }),
            ("scripts.entry no type", |e| e.set_entry(" , HelloOrrin")),
        ];

        for (case, change) in cases {
            let (_dir, project) = project(MANIFEST);
            let mut edit = project.edit().expect("edit");
            // A valid change beside the bad one, so a refusal cannot be a
            // commit that simply had nothing to do.
            edit.set_version("9.9.9");
            change(&mut edit);
            let err = edit.commit().expect_err(case);

            let field = match &err {
                ProjectError::AbsolutePath { field, .. }
                | ProjectError::ParentDirInPath { field, .. }
                | ProjectError::Empty { field } => *field,
                ProjectError::UnqualifiedEntry { .. } => "scripts.entry",
                _ => panic!("{case}: unexpected error: {err}"),
            };
            let expected = case.split(' ').next().unwrap();
            assert_eq!(field, expected, "{case}: {err}");
            assert!(
                err.to_string().contains(&format!("`{expected}`")),
                "{case}: the message does not name the field: {err}"
            );
            assert_eq!(on_disk(&project), MANIFEST, "{case}: the file changed");
        }
    }

    #[test]
    fn a_manifest_that_no_longer_loads_is_never_edited() {
        let broken = [
            "format_version = 1\n[project\n",
            &MANIFEST.replace("dir = \"scripts\"", "dir = \"/scripts\""),
            &format!("{MANIFEST}\n[bogus]\nkey = 1\n"),
        ];
        for body in broken {
            let (_dir, project) = project(MANIFEST);
            std::fs::write(project.manifest_path(), body).expect("break the manifest");

            assert!(project.edit().is_err(), "an edit began from:\n{body}");
            assert_eq!(on_disk(&project), body);
        }
    }

    /// A rename swaps the directory entry, so a second link to the old file
    /// still reads the old bytes; a write in place would change both. That is
    /// the difference between a crash leaving the old manifest or half a new one.
    #[test]
    fn the_manifest_is_replaced_whole_rather_than_rewritten_in_place() {
        let (dir, project) = project(MANIFEST);
        let other_link = dir.path().join("orrin.toml.link");
        std::fs::hard_link(project.manifest_path(), &other_link).expect("hard link");

        let mut edit = project.edit().expect("edit");
        edit.set_name("renamed");
        edit.commit().expect("commit");

        assert_eq!(std::fs::read_to_string(&other_link).unwrap(), MANIFEST);
        assert!(on_disk(&project).contains("name = \"renamed\""));
        let mut left: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        left.sort();
        assert_eq!(
            left,
            ["orrin.toml", "orrin.toml.link"],
            "a temporary file was left behind"
        );
    }

    #[test]
    fn the_committed_project_is_what_loading_the_file_gives() {
        let (_dir, project) = project(MANIFEST);
        let mut edit = project.edit().expect("edit");
        edit.set_name("space-shooter");
        edit.set_version("1.0.0");
        edit.set_entry("SpaceShooter.Main, SpaceShooter");
        edit.set_start_scene(Some(Path::new("scenes/level.orrin")));
        edit.set_persistent_scene(Some(Path::new("scenes/hud.orrin")));
        let committed = edit.commit().expect("commit");
        let loaded = Project::load(&project.manifest_path()).expect("reload");

        for project in [&committed, &loaded] {
            assert_eq!(project.name(), "space-shooter");
            assert_eq!(project.version(), "1.0.0");
            assert_eq!(project.entry_type(), "SpaceShooter.Main, SpaceShooter");
            assert_eq!(
                project.start_scene_relative(),
                Some(Path::new("scenes/level.orrin"))
            );
            assert_eq!(
                project.persistent_scene_relative(),
                Some(Path::new("scenes/hud.orrin"))
            );
        }
        assert_eq!(committed.root(), loaded.root());
        assert_eq!(committed.start_scene(), loaded.start_scene());
        assert_eq!(committed.persistent_scene(), loaded.persistent_scene());
        assert_eq!(committed.scripts_dir(), loaded.scripts_dir());
        assert_eq!(committed.assets_dir(), loaded.assets_dir());
        assert_eq!(committed.game_assembly(), None);
    }

    #[test]
    fn an_exported_game_has_no_manifest_to_edit() {
        let (_dir, source) = project(MANIFEST);
        let out = tempfile::tempdir().expect("tempdir");
        ExportManifest::new(
            &source,
            Platform::host().unwrap_or(Platform::Linux),
            "HelloOrrin.dll",
        )
        .write(out.path())
        .expect("write orrin.json");
        let game =
            Project::from_export(&out.path().join(crate::export::FILE_NAME)).expect("export");

        let err = game.edit().err().expect("an export is not editable");
        assert!(
            matches!(err, ProjectError::Exported { .. }),
            "unexpected error: {err}"
        );
        assert!(!out.path().join("orrin.toml").exists());
    }
}
