//! The asset database: which source files the project has, under which ids,
//! imported into what, and loaded where (architecture §5.1).
//!
//! # Shape
//!
//! - **Identity lives beside the source**, in a `<file>.meta` sidecar holding
//!   the [`AssetId`] and the import settings ([`sidecar`]). The sidecar is
//!   committed; everything else here is derived from the sources and the
//!   sidecars and rebuilt by scanning.
//! - **Artifacts live in the project's `.orrin/cache/`**, named by a hash of
//!   what they were built from ([`cache`]), so an unchanged asset is never
//!   imported twice and a changed one never reads a stale artifact.
//! - **Imports run on a thread of their own** ([`worker`]), so neither startup
//!   nor a save waits on a decode, and every handle has an explicit
//!   [`LoadState`] while it does.
//! - **A handle is a slot that never changes meaning.** A `MeshHandle` the
//!   database issues is the backend's upload index for that asset for the rest
//!   of the session: a reimport replaces what the slot holds and leaves every
//!   component naming it alone, which is what "swapped between frames" means
//!   here, and why the frame's own loops are untouched by any of this.
//!
//! # Handles before the backend
//!
//! A scene load resolves `orrin.mesh = asset(…)` through the registry, which
//! sees the world and not the backend, so it cannot upload anything. It gets a
//! *pending* handle instead — a number with [`PENDING`] set that names the
//! asset and no slot — and [`service`], which runs between frames with the
//! backend in hand, binds each pending handle to a real slot and rewrites the
//! components holding it. The rewrite is a sweep of the world, paid only on a
//! frame that follows a load; `service` runs before extraction, so no pending
//! handle ever reaches a draw.
//!
//! # What is drawn instead
//!
//! A slot always holds something drawable: an invisible degenerate triangle (or
//! a white texel) while its asset loads, the pink placeholder once it has
//! failed, and the asset once it is ready. A missing or broken asset is a named
//! error in the console and a pink object on screen, never a crash and never
//! an object that silently vanishes.
//!
//! # What is not here
//!
//! Engine-built assets — the demo library's cube and gold, a model the demo
//! scene imports — stay in [`Assets`], named rather than imported. They get
//! stable ids too, derived from their names by [`named_id`], so a scene can
//! reference one the same way it references a file.

pub mod cache;
mod handles;
pub mod import;
pub mod sidecar;
mod watch;
pub mod worker;

use std::path::{Path, PathBuf};
use std::time::Duration;

use orrin_ecs::{FxHashMap, World};
use orrin_registry::AssetId;

pub use cache::CacheKey;
pub use handles::named_id;
pub use sidecar::{AssetKind, Settings, Tint};

use crate::gfx::{RenderBackend, TextureHandle};
use crate::scene::{Assets, MaterialHandle, MeshHandle};

/// Set on a handle the database has promised and not yet bound to a slot.
pub const PENDING: u32 = 1 << 31;

/// Whether an import went through.
#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    /// Waiting for, or on, the import thread.
    Queued,
    Imported {
        key: CacheKey,
        /// Found in the cache rather than imported this session.
        cached: bool,
    },
    Failed(String),
    /// A kind read as it is, with nothing to import: a prefab.
    Indexed,
}

/// Whether what a handle draws is the asset yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadState {
    Loading,
    Ready,
    Failed,
}

/// One source asset.
#[derive(Clone, Debug)]
pub struct Entry {
    pub id: AssetId,
    /// Relative to the assets directory, `/`-separated on every platform.
    pub path: String,
    pub kind: AssetKind,
    pub settings: Settings,
    pub status: Status,
    /// The other files the last import read, relative to the assets directory.
    inputs: Vec<String>,
    /// False once a rescan no longer finds the file; see [`AssetDatabase::iter`].
    on_disk: bool,
}

impl Entry {
    /// The file name without its directory or extension, for display.
    pub fn name(&self) -> &str {
        let file = self.path.rsplit('/').next().unwrap_or(&self.path);
        file.rsplit_once('.').map_or(file, |(stem, _)| stem)
    }
}

/// One directory under the assets directory.
#[derive(Clone, Debug, PartialEq)]
pub struct Folder {
    pub path: String,
    /// Only a folder that has been given a sidecar — by being tinted — has one.
    pub id: Option<AssetId>,
    pub tint: Option<Tint>,
}

/// Every source asset in the project, and every handle issued for one. A world
/// resource.
pub struct AssetDatabase {
    dir: PathBuf,
    cache: cache::Cache,
    entries: Vec<Entry>,
    folders: Vec<Folder>,
    worker: Option<worker::Worker>,
    watcher: Option<watch::AssetWatcher>,
    noticed: Vec<PathBuf>,
    meshes: handles::Slots,
    materials: handles::Slots,
    textures: handles::Slots,
    albedo: Vec<(MaterialHandle, AssetId)>,
    reports: Vec<String>,
}

impl AssetDatabase {
    /// Index `dir` and start importing everything in it into `cache_dir`.
    ///
    /// Returns what could not be indexed — a sidecar that does not read, two
    /// files claiming one id — each naming its file. The rest still index.
    pub fn open(dir: impl Into<PathBuf>, cache_dir: impl Into<PathBuf>) -> (Self, Vec<String>) {
        todo!("{} {}", dir.into().display(), cache_dir.into().display())
    }

    /// Reimport whatever changes under the directory from now on.
    pub fn watch(&mut self) -> Result<(), String> {
        todo!()
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn cache_dir(&self) -> &Path {
        self.cache.dir()
    }

    /// The assets whose files exist, by path.
    ///
    /// An asset whose file has gone stays indexed, so the handles already
    /// issued for it keep their ids — saving a scene that uses one must not
    /// drop the reference — but it is no longer offered.
    pub fn iter(&self) -> impl Iterator<Item = &Entry> {
        self.entries.iter().filter(|entry| entry.on_disk)
    }

    /// Every directory, by path.
    pub fn folders(&self) -> &[Folder] {
        &self.folders
    }

    pub fn get(&self, id: AssetId) -> Option<&Entry> {
        todo!("{id}")
    }

    /// The asset at `path`, relative to the assets directory with `/`
    /// separators.
    pub fn find(&self, path: &str) -> Option<&Entry> {
        todo!("{path}")
    }

    /// Whether `id` can be drawn as itself yet.
    ///
    /// An asset nothing has asked to draw is [`Ready`](LoadState::Ready) once
    /// it has imported; one being drawn is ready once its slot holds it. An id
    /// the project has no file for is [`Failed`](LoadState::Failed).
    pub fn state(&self, id: AssetId) -> LoadState {
        todo!("{id}")
    }

    /// The handle for mesh `id`, pending until the next [`service`] binds it.
    ///
    /// An id with no file is still given a handle, which draws the placeholder
    /// and keeps the id, so a scene saved with the file missing still names it.
    pub fn mesh(&mut self, id: AssetId) -> Result<MeshHandle, String> {
        todo!("{id}")
    }

    /// The asset a mesh handle was issued for.
    pub fn mesh_id(&self, handle: MeshHandle) -> Option<AssetId> {
        self.meshes.id(handle.0)
    }

    /// The handle for material `id`. No material is a file yet, so every id
    /// asked for here is one that [`Assets`] does not have, and it draws as
    /// the placeholder while keeping the id.
    pub fn material(&mut self, id: AssetId) -> MaterialHandle {
        todo!("{id}")
    }

    pub fn material_id(&self, handle: MaterialHandle) -> Option<AssetId> {
        self.materials.id(handle.0)
    }

    /// The texture slot texture `id` was loaded into, if anything has asked
    /// for it.
    pub fn texture(&self, id: AssetId) -> Option<TextureHandle> {
        self.textures.handle(id).map(TextureHandle)
    }

    /// Draw `material` with texture `texture` as its albedo, from the next
    /// [`service`] on.
    pub fn assign_albedo(&mut self, material: MaterialHandle, texture: AssetId) -> Result<(), String> {
        todo!("{} {texture}", material.0)
    }

    /// Set or clear the tint a folder's icon is drawn in, writing it to the
    /// folder's sidecar.
    pub fn set_folder_tint(&mut self, path: &str, tint: Option<Tint>) -> Result<(), String> {
        todo!("{path} {tint:?}")
    }

    /// Record that `path` changed, for [`apply_changes`](Self::apply_changes).
    pub fn notice(&mut self, path: PathBuf) {
        self.noticed.push(path);
    }

    /// Scan again and reimport every asset a noticed path belongs to: its
    /// source, its sidecar, or a file its last import read.
    pub fn apply_changes(&mut self) {
        todo!()
    }

    /// Scan again, reimporting only what is new. For the browser's refresh
    /// button; a watched directory never needs it.
    pub fn rescan(&mut self) {
        todo!()
    }

    /// Block until the import thread is idle, or `timeout` passes. `false` on
    /// timeout. For tests and tools; the editor never waits.
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        self.worker.as_ref().is_none_or(|worker| worker.wait_idle(timeout))
    }

    /// Everything [`service`] does, returning the errors it would log.
    pub fn update(&mut self, world: &World, backend: &mut impl RenderBackend) -> Vec<String> {
        todo!("{} {}", world.entities().count(), backend.gpu_culling())
    }
}

/// The once-a-frame step: act on settled file changes, take finished imports,
/// bind pending handles and swap loaded artifacts into their slots.
///
/// Runs between frames, after everything that can issue a handle — the editor
/// and the script tick — and before extraction. Costs a few channel polls on a
/// frame with nothing to do.
pub fn service(world: &World, backend: &mut impl RenderBackend) {
    let Some(mut db) = world.get_resource_mut::<AssetDatabase>() else {
        return;
    };
    for error in db.update(world, backend) {
        tracing::error!("{error}");
    }
}

/// The directory a project's artifacts are cached in.
pub fn cache_dir(project: Option<&orrin_project::Project>) -> PathBuf {
    project.map_or_else(|| PathBuf::from(".orrin"), |project| project.editor_dir())
        .join("cache")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::gfx::HeadlessBackend;
    use crate::scene::{CpuMesh, MeshBounds};
    use import::tests::{png, triangle_bin, triangle_gltf};

    /// A scratch directory, removed when dropped.
    pub(crate) struct Dir(PathBuf);

    impl Dir {
        pub(crate) fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("orrin-assets-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        /// Write `rel`, creating its directories, and return its path.
        pub(crate) fn write(&self, rel: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, bytes).unwrap();
            path
        }

        pub(crate) fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const WAIT: Duration = Duration::from_secs(20);

    fn open(dir: &Dir) -> AssetDatabase {
        let (db, issues) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        assert!(issues.is_empty(), "{issues:?}");
        db
    }

    fn project(dir: &Dir) {
        dir.write("rocks/rock.png", &png(2, 2, |_, _| [10, 20, 30, 255]));
        dir.write("models/tree.gltf", triangle_gltf("").as_bytes());
        dir.write("models/tri angle.bin", &triangle_bin());
        dir.write("props/door.fprefab", b"orrin-scene 1\n");
        dir.write("notes.txt", b"not an asset");
        dir.write(".hidden/secret.png", b"never read");
    }

    #[test]
    fn a_scan_indexes_every_asset_and_folder_by_path() {
        let dir = Dir::new("db-scan");
        project(&dir);
        let db = open(&dir);

        let listed: Vec<(&str, AssetKind)> =
            db.iter().map(|entry| (entry.path.as_str(), entry.kind)).collect();
        assert_eq!(
            listed,
            [
                ("models/tree.gltf", AssetKind::Mesh),
                ("props/door.fprefab", AssetKind::Prefab),
                ("rocks/rock.png", AssetKind::Texture),
            ]
        );
        let folders: Vec<&str> = db.folders().iter().map(|f| f.path.as_str()).collect();
        assert_eq!(folders, ["models", "props", "rocks"]);

        let rock = db.find("rocks/rock.png").unwrap();
        assert_eq!(rock.name(), "rock");
        assert_eq!(db.get(rock.id).unwrap().path, "rocks/rock.png");
        assert!(db.find("notes.txt").is_none());
        assert!(dir.path().join("rocks/rock.png.meta").is_file());
        assert!(!dir.path().join("models/tri angle.bin.meta").exists());
    }

    /// Duplicating a file in an explorer copies its sidecar too, and two assets
    /// answering to one id would make every reference to either ambiguous.
    #[test]
    fn a_copy_that_kept_the_original_sidecar_is_refused_by_name() {
        let dir = Dir::new("db-copy");
        let sidecar = "id = \"5f0c3a4e-0000-4000-8000-000000000001\"\n";
        for rel in ["rock.png", "rock copy.png"] {
            dir.write(rel, &png(1, 1, |_, _| [0; 4]));
            dir.write(&format!("{rel}.meta"), sidecar.as_bytes());
        }
        let (db, issues) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        assert_eq!(db.iter().count(), 1);
        assert_eq!(
            issues,
            ["rock.png has the same id as rock copy.png, most likely because it is a copy that \
              kept the original's sidecar. Delete rock.png.meta to give it an id of its own"]
        );
    }

    #[test]
    fn a_folder_tint_is_listed_and_survives_a_fresh_scan() {
        let dir = Dir::new("db-tint");
        project(&dir);
        let mut db = open(&dir);
        db.set_folder_tint("rocks", Some(Tint([1, 2, 3]))).unwrap();
        let rocks = db.folders().iter().find(|f| f.path == "rocks").unwrap();
        assert_eq!(rocks.tint, Some(Tint([1, 2, 3])));
        assert!(rocks.id.is_some());

        let again = open(&dir);
        assert_eq!(again.folders(), db.folders());
        assert!(db.set_folder_tint("nowhere", None).is_err());
    }

    #[test]
    fn everything_imports_in_the_background_once_and_is_cached_after() {
        let dir = Dir::new("db-import");
        project(&dir);
        let mut db = open(&dir);
        assert!(db.wait_idle(WAIT));
        let mut backend = HeadlessBackend::new();
        let world = World::new();
        assert!(db.update(&world, &mut backend).is_empty());

        for path in ["rocks/rock.png", "models/tree.gltf"] {
            let entry = db.find(path).unwrap();
            let Status::Imported { key, cached } = entry.status else {
                panic!("{path}: {:?}", entry.status);
            };
            assert!(!cached, "{path}");
            assert!(dir.path().join(".cache").join(key.to_string()).is_file());
            assert_eq!(db.state(entry.id), LoadState::Ready);
        }
        assert_eq!(db.find("props/door.fprefab").unwrap().status, Status::Indexed);

        let mut again = open(&dir);
        assert!(again.wait_idle(WAIT));
        again.update(&world, &mut backend);
        assert!(matches!(
            again.find("rocks/rock.png").unwrap().status,
            Status::Imported { cached: true, .. }
        ));
        assert_eq!(backend.upload_counts(), (0, 0, 0), "nothing asked to draw anything");
    }

    #[test]
    fn a_broken_source_fails_by_name_and_the_rest_still_import() {
        let dir = Dir::new("db-broken");
        project(&dir);
        dir.write("rocks/bad.png", b"not a png");
        let mut db = open(&dir);
        assert!(db.wait_idle(WAIT));
        let errors = db.update(&World::new(), &mut HeadlessBackend::new());

        let bad = db.find("rocks/bad.png").unwrap();
        let Status::Failed(message) = &bad.status else {
            panic!("{:?}", bad.status);
        };
        assert!(message.starts_with("rocks/bad.png is not a PNG or JPEG"), "{message}");
        assert_eq!(errors, [message.clone()]);
        assert_eq!(db.state(bad.id), LoadState::Failed);
        assert!(matches!(db.find("rocks/rock.png").unwrap().status, Status::Imported { .. }));
    }

    /// A world that draws `id`'s mesh on one entity, as a loaded scene would
    /// leave it: holding the pending handle the registry was given.
    fn drawing(db: AssetDatabase, id: AssetId) -> (World, orrin_ecs::Entity) {
        let mut world = World::new();
        world.insert_resource(MeshBounds::default());
        world.insert_resource(crate::scene::AssetPreviews::default());
        world.insert_resource(crate::scene::MaterialBlends::default());
        world.insert_resource(db);
        let handle = world.resource_mut::<AssetDatabase>().mesh(id).unwrap();
        assert_ne!(handle.0 & PENDING, 0);
        let entity = world.spawn();
        world.insert(entity, handle);
        (world, entity)
    }

    fn update(world: &World, backend: &mut HeadlessBackend) -> Vec<String> {
        world.resource_mut::<AssetDatabase>().update(world, backend)
    }

    fn handle(world: &World, entity: orrin_ecs::Entity) -> MeshHandle {
        *world.get::<MeshHandle>(entity).unwrap()
    }

    #[test]
    fn a_pending_mesh_is_bound_between_frames_then_loaded_into_its_slot() {
        let dir = Dir::new("db-load");
        project(&dir);
        let db = open(&dir);
        let id = db.find("models/tree.gltf").unwrap().id;
        let (world, entity) = drawing(db, id);
        let mut backend = HeadlessBackend::new();

        assert!(update(&world, &mut backend).is_empty());
        let bound = handle(&world, entity);
        assert_eq!(bound.0 & PENDING, 0, "rewritten to a real slot");
        assert_eq!(world.resource::<AssetDatabase>().mesh_id(bound), Some(id));

        assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
        update(&world, &mut backend);
        assert_eq!(world.resource::<AssetDatabase>().state(id), LoadState::Ready);
        let triangle = crate::geom::Aabb::from_points(
            [glam::Vec3::ZERO, glam::Vec3::X, glam::Vec3::Y].into_iter(),
        );
        assert_eq!(backend.mesh_bounds(bound), Some(triangle));
        assert_eq!(world.resource::<MeshBounds>().get(bound), Some(triangle));
        assert_eq!(handle(&world, entity), bound, "the slot is the handle for good");
        assert!(world.resource::<crate::scene::AssetPreviews>().mesh(bound).is_some());
    }

    #[test]
    fn a_missing_asset_draws_the_placeholder_names_itself_and_keeps_its_id() {
        let dir = Dir::new("db-missing");
        let db = open(&dir);
        let id: AssetId = "5f0c3a4e-0000-4000-8000-0000000000ff".parse().unwrap();
        let (world, entity) = drawing(db, id);
        let mut backend = HeadlessBackend::new();

        let errors = update(&world, &mut backend);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(
            errors[0].starts_with(&format!("mesh asset {id} is not in the project")),
            "{}",
            errors[0]
        );
        let bound = handle(&world, entity);
        let db = world.resource::<AssetDatabase>();
        assert_eq!(db.state(id), LoadState::Failed);
        assert_eq!(db.mesh_id(bound), Some(id));
        assert_eq!(backend.mesh_bounds(bound), Some(CpuMesh::cube().bounds()));
        drop(db);
        assert!(update(&world, &mut backend).is_empty(), "named once, not every frame");
    }

    #[test]
    fn a_handle_of_the_wrong_kind_is_refused_by_path() {
        let dir = Dir::new("db-kind");
        project(&dir);
        let mut db = open(&dir);
        let rock = db.find("rocks/rock.png").unwrap().id;
        assert_eq!(
            db.mesh(rock).unwrap_err(),
            "`rocks/rock.png`, which is a texture"
        );
    }

    fn loaded_tree(dir: &Dir) -> (World, MeshHandle, HeadlessBackend) {
        project(dir);
        let db = open(dir);
        let id = db.find("models/tree.gltf").unwrap().id;
        let (world, entity) = drawing(db, id);
        let mut backend = HeadlessBackend::new();
        update(&world, &mut backend);
        assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
        update(&world, &mut backend);
        let slot = handle(&world, entity);
        (world, slot, backend)
    }

    fn change(world: &World, dir: &Dir, rel: &str, bytes: &[u8], backend: &mut HeadlessBackend) {
        let path = dir.write(rel, bytes);
        let mut db = world.resource_mut::<AssetDatabase>();
        db.notice(path);
        db.apply_changes();
        assert!(db.wait_idle(WAIT));
        drop(db);
        assert!(update(world, backend).is_empty());
    }

    #[test]
    fn a_saved_source_is_reimported_and_swapped_into_the_same_slot() {
        let dir = Dir::new("db-reload");
        let (world, slot, mut backend) = loaded_tree(&dir);
        let before = backend.mesh_bounds(slot).unwrap();

        let moved = triangle_gltf(r#", "translation": [0, 0, 5]"#);
        change(&world, &dir, "models/tree.gltf", moved.as_bytes(), &mut backend);
        let after = backend.mesh_bounds(slot).unwrap();
        assert_eq!(after.min.z, before.min.z + 5.0);
        assert_eq!(world.resource::<MeshBounds>().get(slot), Some(after));
        assert_eq!(backend.mesh_replacements(), 2);
    }

    #[test]
    fn a_save_that_changes_nothing_swaps_nothing() {
        let dir = Dir::new("db-touch");
        let (world, _, mut backend) = loaded_tree(&dir);
        let replaced = backend.mesh_replacements();
        change(&world, &dir, "models/tree.gltf", triangle_gltf("").as_bytes(), &mut backend);
        assert_eq!(backend.mesh_replacements(), replaced);
    }

    #[test]
    fn a_changed_setting_or_buffer_reimports_too() {
        let dir = Dir::new("db-settings");
        let (world, slot, mut backend) = loaded_tree(&dir);
        let id = world.resource::<AssetDatabase>().mesh_id(slot).unwrap();

        let sidecar = format!("id = \"{id}\"\n\n[import]\nscale = 2.0\n");
        change(&world, &dir, "models/tree.gltf.meta", sidecar.as_bytes(), &mut backend);
        assert_eq!(backend.mesh_bounds(slot).unwrap().max.x, 2.0);

        let mut bin = triangle_bin();
        bin[12..16].copy_from_slice(&3.0f32.to_le_bytes());
        change(&world, &dir, "models/tri angle.bin", &bin, &mut backend);
        assert_eq!(backend.mesh_bounds(slot).unwrap().max.x, 6.0);
    }

    #[test]
    fn an_albedo_assignment_loads_the_texture_and_edits_the_material() {
        let dir = Dir::new("db-albedo");
        project(&dir);
        let mut world = World::new();
        let mut previews = crate::scene::AssetPreviews::default();
        previews.insert_material(MaterialHandle(4), &crate::gfx::Material::default(), None);
        world.insert_resource(previews);
        let mut db = open(&dir);
        let rock = db.find("rocks/rock.png").unwrap().id;
        db.assign_albedo(MaterialHandle(4), rock).unwrap();
        world.insert_resource(db);
        let mut backend = HeadlessBackend::new();

        update(&world, &mut backend);
        let texture = world.resource::<AssetDatabase>().texture(rock).unwrap();
        let edits = world.resource_mut::<crate::scene::AssetPreviews>().take_material_edits();
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].1.albedo_texture, Some(texture));

        assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
        update(&world, &mut backend);
        assert_eq!(world.resource::<AssetDatabase>().state(rock), LoadState::Ready);
        assert_eq!(backend.texture_replacements(), 1);

        let tree = world.resource::<AssetDatabase>().find("models/tree.gltf").unwrap().id;
        assert_eq!(
            world.resource_mut::<AssetDatabase>().assign_albedo(MaterialHandle(4), tree),
            Err("`models/tree.gltf`, which is a mesh, cannot be an albedo map".to_owned())
        );
    }

    #[test]
    fn an_unknown_id_is_failed_and_a_queued_one_is_loading() {
        let dir = Dir::new("db-state");
        project(&dir);
        let db = open(&dir);
        assert_eq!(db.state(AssetId::new()), LoadState::Failed);
        let rock = db.find("rocks/rock.png").unwrap().id;
        assert!(matches!(db.state(rock), LoadState::Loading | LoadState::Ready));
    }
}
