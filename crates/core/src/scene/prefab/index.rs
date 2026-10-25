//! Which `.fprefab` files the project has, under which ids.
//!
//! The slice of the asset database (issue #22) that prefabs need, in the shape
//! that database takes: identity lives in a `<file>.meta` sidecar beside the
//! source, read through `asset_db::sidecar`; the index is derived state rebuilt
//! by scanning, and a path turns into an id here and nowhere else.
//!
//! Kept beside the database rather than inside it because it holds what the
//! database does not — each prefab parsed and checked to be one tree — and
//! because a run without a project has prefabs and no database.

use std::path::{Path, PathBuf};

use orrin_ecs::FxHashMap;
use orrin_registry::{AssetId, EntityId, SceneDocument, SceneEntity, Value};

use crate::scene::registry::PARENT;

/// The extension a prefab file is recognised by.
pub const EXTENSION: &str = "fprefab";

/// One prefab file, parsed and checked to be a single tree.
pub struct Prefab {
    pub id: AssetId,
    /// Relative to the assets directory and `/`-separated on every platform, so
    /// one prefab has one path on every machine.
    pub path: String,
    document: SceneDocument,
    root: EntityId,
    by_id: FxHashMap<EntityId, usize>,
    /// False once a rescan no longer finds the file. See [`Prefabs::rescan`].
    on_disk: bool,
}

impl Prefab {
    /// Read prefab text. Every error starts with `path`.
    pub fn parse(id: AssetId, path: impl Into<String>, text: &str) -> Result<Self, String> {
        let path = path.into();
        match orrin_registry::parse(text) {
            Ok(document) => Self::from_document(id, path, document),
            Err(error) => Err(format!("{path}: {error}")),
        }
    }

    /// Check that `document` is one tree an instance can derive its ids from:
    /// ids unique and never nil, and exactly one entity nothing in the file
    /// parents.
    pub fn from_document(
        id: AssetId,
        path: String,
        mut document: SceneDocument,
    ) -> Result<Self, String> {
        for entity in &mut document.entities {
            for (component, value) in &mut entity.components {
                crate::asset_db::upgrade_named(component, value);
                super::canonical(value);
            }
        }
        let mut by_id = FxHashMap::default();
        for (index, entity) in document.entities.iter().enumerate() {
            if entity.id.is_nil() {
                return Err(format!(
                    "{path}: an entity has the nil id, which no instance can derive an id from"
                ));
            }
            if by_id.insert(entity.id, index).is_some() {
                return Err(format!("{path}: entity {} appears twice", entity.id));
            }
        }

        let roots: Vec<EntityId> = document
            .entities
            .iter()
            .filter(|entity| parent_of(entity).is_none_or(|parent| !by_id.contains_key(&parent)))
            .map(|entity| entity.id)
            .collect();
        let root = match roots[..] {
            [root] => root,
            [] => {
                return Err(format!(
                    "{path}: a prefab is one entity tree, and this one has no root"
                ));
            }
            _ => {
                return Err(format!(
                    "{path}: a prefab is one entity tree, and this one has {} roots; \
                     parent them under one entity",
                    roots.len()
                ));
            }
        };

        Ok(Self {
            id,
            path,
            document,
            root,
            by_id,
            on_disk: true,
        })
    }

    /// The entity the tree hangs from, by its id in the file.
    pub fn root(&self) -> EntityId {
        self.root
    }

    pub fn entities(&self) -> &[SceneEntity] {
        &self.document.entities
    }

    /// One entity, by its id in the file.
    pub fn entity(&self, id: EntityId) -> Option<&SceneEntity> {
        self.by_id
            .get(&id)
            .map(|&index| &self.document.entities[index])
    }

    pub fn contains(&self, id: EntityId) -> bool {
        self.by_id.contains_key(&id)
    }

    /// The file name without its directory or extension, for display.
    pub fn name(&self) -> &str {
        let file = self.path.rsplit('/').next().unwrap_or(&self.path);
        file.strip_suffix(EXTENSION)
            .and_then(|stem| stem.strip_suffix('.'))
            .unwrap_or(file)
    }
}

fn parent_of(entity: &SceneEntity) -> Option<EntityId> {
    entity
        .components
        .iter()
        .find_map(|(id, value)| match value {
            Value::Entity(parent) if *id == PARENT => Some(*parent),
            _ => None,
        })
}

/// Every prefab the project has. A world resource.
#[derive(Default)]
pub struct Prefabs {
    /// What [`rescan`](Self::rescan) reads; `None` for an index built by hand.
    dir: Option<PathBuf>,
    /// By path, the order a browser lists them in.
    entries: Vec<Prefab>,
}

impl Prefabs {
    /// Index every `.fprefab` under `dir`, giving any without a sidecar an id.
    ///
    /// A file that cannot be indexed is left out and named in the returned
    /// issues, and the rest still load. Files are read in path order, so which
    /// of two duplicates is kept does not depend on the filesystem.
    pub fn scan(dir: impl AsRef<Path>) -> (Self, Vec<String>) {
        let dir = dir.as_ref();
        let mut index = Self {
            dir: Some(dir.to_owned()),
            entries: Vec::new(),
        };
        let mut issues = Vec::new();

        let mut files = Vec::new();
        if dir.is_dir() {
            collect(dir, dir, &mut files, &mut issues);
        }
        files.sort();

        for (rel, file) in files {
            let prefab = sidecar_id(&file, &rel).and_then(|id| {
                let text = std::fs::read_to_string(&file).map_err(|e| format!("{rel}: {e}"))?;
                Prefab::parse(id, rel.clone(), &text)
            });
            match prefab {
                Ok(prefab) => match index.get(prefab.id) {
                    Some(first) => issues.push(format!(
                        "{rel} has the same id as {}, most likely because it is a copy that kept \
                         the original's sidecar. Delete {rel}.meta to give it an id of its own",
                        first.path
                    )),
                    None => index.insert(prefab),
                },
                Err(issue) => issues.push(issue),
            }
        }
        (index, issues)
    }

    /// Read the directory again, picking up new, changed and deleted files.
    ///
    /// A prefab whose file has gone stays indexed, hidden from
    /// [`find`](Self::find) and [`iter`](Self::iter): an instance of it may
    /// still be in the open scene, and saving that instance needs the prefab
    /// it was built from.
    pub fn rescan(&mut self) -> Vec<String> {
        let Some(dir) = self.dir.clone() else {
            return Vec::new();
        };
        let (fresh, issues) = Self::scan(dir);
        let old = std::mem::replace(self, fresh);
        for mut prefab in old.entries {
            if self.get(prefab.id).is_none() {
                prefab.on_disk = false;
                self.insert(prefab);
            }
        }
        issues
    }

    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// Add `prefab`, replacing any with the same id.
    pub fn insert(&mut self, prefab: Prefab) {
        self.entries.retain(|existing| existing.id != prefab.id);
        let at = self
            .entries
            .partition_point(|existing| existing.path < prefab.path);
        self.entries.insert(at, prefab);
    }

    /// Linear: an index holds tens of prefabs, and every caller is a load, a
    /// save or a click rather than a frame.
    pub fn get(&self, id: AssetId) -> Option<&Prefab> {
        self.entries.iter().find(|prefab| prefab.id == id)
    }

    /// The prefab at `path`, relative to the assets directory with `/`
    /// separators.
    pub fn find(&self, path: &str) -> Option<AssetId> {
        self.iter()
            .find(|prefab| prefab.path == path)
            .map(|prefab| prefab.id)
    }

    /// The prefabs whose files exist, by path.
    pub fn iter(&self) -> impl Iterator<Item = &Prefab> {
        self.entries.iter().filter(|prefab| prefab.on_disk)
    }
}

/// Every `.fprefab` below `dir`, as `(relative path, path)`. Dotfiles and
/// symlinks are skipped, as the CLI's asset index skips them.
fn collect(base: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>, issues: &mut Vec<String>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) => {
            issues.push(format!("could not list {}: {error}", dir.display()));
            return;
        }
    };
    for entry in entries.flatten() {
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if kind.is_dir() {
            collect(base, &path, out, issues);
        } else if kind.is_file() && path.extension().is_some_and(|ext| ext == EXTENSION) {
            let rel: Vec<String> = path
                .strip_prefix(base)
                .unwrap_or(&path)
                .components()
                .map(|part| part.as_os_str().to_string_lossy().into_owned())
                .collect();
            out.push((rel.join("/"), path));
        }
    }
}

/// The id in `<file>.meta`, writing a fresh one when there is no sidecar.
///
/// Read by the asset database's own rules (`asset_db::sidecar`), so the two
/// indexes cannot disagree about what a sidecar says, or give one new prefab
/// two ids.
fn sidecar_id(file: &Path, rel: &str) -> Result<AssetId, String> {
    crate::asset_db::sidecar::read_or_create(file, rel, crate::asset_db::AssetKind::Prefab)
        .map(|sidecar| sidecar.id)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in crate::scene::prefab) const DOOR_ID: &str = "d0d0d0d0-0000-4000-8000-000000000001";
    pub(in crate::scene::prefab) const ROOT: &str = "00000000-0000-4000-8000-00000000000a";
    pub(in crate::scene::prefab) const PANEL: &str = "00000000-0000-4000-8000-00000000000b";
    pub(in crate::scene::prefab) const HANDLE: &str = "00000000-0000-4000-8000-00000000000c";

    /// A door: a root, a spinning panel under it, and a tagged handle under
    /// that. The panel's `game.follow` names the handle, which is what an
    /// instance has to remap.
    pub(in crate::scene::prefab) fn door_text(handle_tag: &str, panel_height: f32) -> String {
        format!(
            "\
orrin-scene 1

entity {ROOT}
  orrin.name = \"Door\"
  orrin.transform
    rotation = (0.0, 0.0, 0.0, 1.0)
    scale = (1.0, 1.0, 1.0)
    translation = (0.0, 0.0, 0.0)

entity {PANEL}
  game.follow
    target = entity({HANDLE})
  orrin.mesh = \"cube\"
  orrin.name = \"Panel\"
  orrin.parent = entity({ROOT})
  orrin.spin
    axis = (0.0, 1.0, 0.0)
    speed = 1.5
  orrin.transform
    rotation = (0.0, 0.0, 0.0, 1.0)
    scale = (1.0, 1.0, 1.0)
    translation = (0.0, {panel_height:?}, 0.0)

entity {HANDLE}
  orrin.name = \"Handle\"
  orrin.parent = entity({PANEL})
  orrin.tag = \"{handle_tag}\"
  orrin.transform
    rotation = (0.0, 0.0, 0.0, 1.0)
    scale = (1.0, 1.0, 1.0)
    translation = (0.5, 0.0, 0.0)
"
        )
    }

    pub(in crate::scene::prefab) fn door_id() -> AssetId {
        DOOR_ID.parse().unwrap()
    }

    #[test]
    fn a_prefab_is_one_tree() {
        let two_roots = format!(
            "orrin-scene 1\n\nentity {ROOT}\n  orrin.name = \"A\"\n\nentity {PANEL}\n  orrin.name = \"B\"\n"
        );
        let err = Prefab::parse(door_id(), "props/pair.fprefab", &two_roots)
            .err()
            .expect("two roots must be refused");
        assert_eq!(
            err,
            "props/pair.fprefab: a prefab is one entity tree, and this one has 2 roots; \
             parent them under one entity"
        );
    }

    #[test]
    fn a_prefab_that_does_not_parse_names_its_file_and_line() {
        let err = Prefab::parse(
            door_id(),
            "props/door.fprefab",
            "orrin-scene 1\n\nentity nope\n",
        )
        .err()
        .unwrap();
        assert!(err.starts_with("props/door.fprefab: line 3:"), "{err}");
    }

    struct Dir(PathBuf);

    impl Dir {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("orrin-prefabs-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(dir.join("props")).unwrap();
            Self(dir)
        }

        fn write(&self, rel: &str, text: &str) {
            std::fs::write(self.0.join(rel), text).unwrap();
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The id is the prefab's identity for as long as the file exists, so the
    /// first scan has to leave it where the next one finds it.
    #[test]
    fn a_prefab_without_a_sidecar_is_given_an_id_that_sticks() {
        let dir = Dir::new("sticks");
        dir.write("props/door.fprefab", &door_text("handle", 1.0));

        let (first, issues) = Prefabs::scan(dir.path());
        assert!(issues.is_empty(), "{issues:?}");
        let id = first
            .find("props/door.fprefab")
            .expect("indexed by its relative path");
        let sidecar = std::fs::read_to_string(dir.path().join("props/door.fprefab.meta")).unwrap();
        assert_eq!(sidecar, format!("id = \"{id}\"\n"));

        let (second, _) = Prefabs::scan(dir.path());
        assert_eq!(second.find("props/door.fprefab"), Some(id));
    }

    #[test]
    fn a_sidecar_id_is_used_as_written() {
        let dir = Dir::new("sidecar");
        dir.write("props/door.fprefab", &door_text("handle", 1.0));
        dir.write(
            "props/door.fprefab.meta",
            &format!("# import settings live here too\nid = \"{DOOR_ID}\"\n"),
        );

        let (prefabs, issues) = Prefabs::scan(dir.path());
        assert!(issues.is_empty(), "{issues:?}");
        assert_eq!(prefabs.find("props/door.fprefab"), Some(door_id()));
        let prefab = prefabs.get(door_id()).unwrap();
        assert_eq!(prefab.path, "props/door.fprefab");
        assert_eq!(prefab.name(), "door");
    }

    /// One sidecar format for every asset: a prefab's is read by the asset
    /// database's rules, so a setting a prefab does not have is an error by
    /// name rather than ignored, and the database and the index give a new
    /// prefab the same id whichever reads it first.
    #[test]
    fn a_prefab_sidecar_follows_the_asset_databases_rules() {
        let dir = Dir::new("shared-sidecar");
        dir.write("props/door.fprefab", &door_text("handle", 1.0));
        dir.write("props/gate.fprefab", &door_text("handle", 1.0));
        dir.write(
            "props/door.fprefab.meta",
            &format!("id = \"{DOOR_ID}\"\n\n[import]\nscale = 2.0\n"),
        );

        let (db, _) = crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let (prefabs, issues) = Prefabs::scan(dir.path());
        assert_eq!(
            issues,
            ["props/door.fprefab.meta: `scale` is not a prefab import setting; a prefab has none"]
        );
        assert_eq!(
            prefabs.find("props/gate.fprefab"),
            db.find("props/gate.fprefab").map(|entry| entry.id)
        );
    }

    /// Duplicating a file in an explorer copies its sidecar too, and two
    /// prefabs answering to one id would make every instance of either
    /// ambiguous.
    #[test]
    fn a_copy_that_kept_the_original_sidecar_is_refused_by_name() {
        let dir = Dir::new("copy");
        let sidecar = format!("id = \"{DOOR_ID}\"\n");
        for rel in ["props/door.fprefab", "props/door copy.fprefab"] {
            dir.write(rel, &door_text("handle", 1.0));
            dir.write(&format!("{rel}.meta"), &sidecar);
        }

        let (prefabs, issues) = Prefabs::scan(dir.path());
        assert_eq!(prefabs.iter().count(), 1);
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert!(
            issues[0].contains("props/door copy.fprefab")
                && issues[0].contains("props/door.fprefab"),
            "{}",
            issues[0]
        );
    }

    #[test]
    fn a_broken_prefab_is_reported_and_the_rest_still_load() {
        let dir = Dir::new("broken");
        dir.write("props/door.fprefab", &door_text("handle", 1.0));
        dir.write("props/bad.fprefab", "not a scene\n");
        dir.write("props/notes.txt", "not a prefab, and never read\n");

        let (prefabs, issues) = Prefabs::scan(dir.path());
        assert_eq!(prefabs.iter().count(), 1);
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert!(
            issues[0].starts_with("props/bad.fprefab: line 1:"),
            "{}",
            issues[0]
        );
    }

    /// An instance in the open scene must still save after its file is
    /// deleted, so the index forgets a prefab only for making new ones.
    #[test]
    fn a_rescan_keeps_a_deleted_prefab_for_the_instances_already_made() {
        let dir = Dir::new("rescan");
        dir.write("props/door.fprefab", &door_text("handle", 1.0));
        dir.write("props/door.fprefab.meta", &format!("id = \"{DOOR_ID}\"\n"));
        let (mut prefabs, _) = Prefabs::scan(dir.path());

        std::fs::remove_file(dir.path().join("props/door.fprefab")).unwrap();
        assert!(prefabs.rescan().is_empty());
        assert_eq!(prefabs.find("props/door.fprefab"), None);
        assert_eq!(prefabs.iter().count(), 0);
        assert!(prefabs.get(door_id()).is_some());
    }
}
