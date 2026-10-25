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
//! scene imports — stay in [`Assets`](crate::scene::Assets), named rather than imported. They get
//! stable ids too, derived from their names by [`named_id`], so a scene can
//! reference one the same way it references a file.

pub mod cache;
mod handles;
pub mod import;
pub mod sidecar;
mod watch;
pub mod worker;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use orrin_ecs::{FxBuildHasher, FxHashMap, World};
use orrin_registry::AssetId;

pub use cache::CacheKey;
pub use handles::{named_id, upgrade_named};
pub use sidecar::{AssetKind, Settings, Tint};

use crate::gfx::{RenderBackend, TextureHandle};
use crate::scene::{
    AssetPreviews, CpuMesh, MaterialBlends, MaterialHandle, MeshBounds, MeshHandle,
};

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
    /// Issues found between frames, returned by the next [`update`](Self::update).
    reports: Vec<String>,
    /// Assets with an import on its way back for a slot.
    awaiting: HashSet<AssetId, FxBuildHasher>,
}

impl AssetDatabase {
    /// Index `dir` and start importing everything in it into `cache_dir`.
    ///
    /// Returns what could not be indexed — a sidecar that does not read, two
    /// files claiming one id — each naming its file. The rest still index.
    pub fn open(dir: impl Into<PathBuf>, cache_dir: impl Into<PathBuf>) -> (Self, Vec<String>) {
        let cache = cache::Cache::new(cache_dir);
        let mut issues = Vec::new();
        let worker = worker::Worker::spawn(cache.clone())
            .map_err(|error| issues.push(error))
            .ok();
        let mut db = Self {
            dir: dir.into(),
            cache,
            entries: Vec::new(),
            folders: Vec::new(),
            worker,
            watcher: None,
            noticed: Vec::new(),
            meshes: handles::Slots::default(),
            materials: handles::Slots::default(),
            textures: handles::Slots::default(),
            albedo: Vec::new(),
            reports: Vec::new(),
            awaiting: HashSet::default(),
        };
        issues.extend(db.scan(&[]));
        (db, issues)
    }

    /// Reimport whatever changes under the directory from now on.
    pub fn watch(&mut self) -> Result<(), String> {
        self.watcher = Some(watch::AssetWatcher::new(&self.dir)?);
        Ok(())
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
        self.entries.iter().find(|entry| entry.id == id)
    }

    /// The asset at `path`, relative to the assets directory with `/`
    /// separators.
    pub fn find(&self, path: &str) -> Option<&Entry> {
        self.iter().find(|entry| entry.path == path)
    }

    /// Whether `id` can be drawn as itself yet.
    ///
    /// An asset nothing has asked to draw is [`Ready`](LoadState::Ready) once
    /// it has imported; one being drawn is ready once its slot holds it. An id
    /// the project has no file for is [`Failed`](LoadState::Failed).
    pub fn state(&self, id: AssetId) -> LoadState {
        if let Some(resident) = self.resident(id) {
            return resident.state;
        }
        match self.get(id).map(|entry| &entry.status) {
            Some(Status::Queued) => LoadState::Loading,
            Some(Status::Imported { .. } | Status::Indexed) => LoadState::Ready,
            Some(Status::Failed(_)) | None => LoadState::Failed,
        }
    }

    /// The handle for mesh `id`, pending until the next [`service`] binds it.
    ///
    /// An id with no file is still given a handle, which draws the placeholder
    /// and keeps the id, so a scene saved with the file missing still names it.
    pub fn mesh(&mut self, id: AssetId) -> Result<MeshHandle, String> {
        self.expect_kind(id, AssetKind::Mesh)?;
        Ok(MeshHandle(self.meshes.promise(id)))
    }

    /// The asset a mesh handle was issued for.
    pub fn mesh_id(&self, handle: MeshHandle) -> Option<AssetId> {
        self.meshes.id(handle.0)
    }

    /// The handle for material `id`. No material is a file yet, so every id
    /// asked for here is one that [`Assets`](crate::scene::Assets) does not have, and it draws as
    /// the placeholder while keeping the id.
    pub fn material(&mut self, id: AssetId) -> MaterialHandle {
        MaterialHandle(self.materials.promise(id))
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
    pub fn assign_albedo(
        &mut self,
        material: MaterialHandle,
        texture: AssetId,
    ) -> Result<(), String> {
        match self.get(texture) {
            Some(entry) if entry.kind == AssetKind::Texture => {
                self.albedo.push((material, texture));
                Ok(())
            }
            Some(entry) => Err(format!(
                "`{}`, which is a {}, cannot be an albedo map",
                entry.path,
                entry.kind.label()
            )),
            None => Err(format!("texture asset {texture} is not in the project")),
        }
    }

    /// Set or clear the tint a folder's icon is drawn in, writing it to the
    /// folder's sidecar.
    pub fn set_folder_tint(&mut self, path: &str, tint: Option<Tint>) -> Result<(), String> {
        let Some(folder) = self.folders.iter_mut().find(|folder| folder.path == path) else {
            return Err(format!(
                "there is no folder `{path}` in {}",
                self.dir.display()
            ));
        };
        let id = sidecar::write_folder_tint(&self.dir.join(path), path, tint)?;
        folder.id = Some(id);
        folder.tint = tint;
        Ok(())
    }

    /// Record that `path` changed, for [`apply_changes`](Self::apply_changes).
    pub fn notice(&mut self, path: PathBuf) {
        self.noticed.push(path);
    }

    /// Scan again and reimport every asset a noticed path belongs to: its
    /// source, its sidecar, or a file its last import read.
    pub fn apply_changes(&mut self) {
        let noticed = std::mem::take(&mut self.noticed);
        let touched: Vec<String> = noticed
            .iter()
            .filter_map(|path| self.relative(path))
            .collect();
        if touched.is_empty() && !noticed.is_empty() {
            return;
        }
        let issues = self.scan(&touched);
        self.reports.extend(issues);
    }

    /// Scan again, reimporting only what is new. For the browser's refresh
    /// button; a watched directory never needs it.
    pub fn rescan(&mut self) {
        let issues = self.scan(&[]);
        self.reports.extend(issues);
    }

    /// Block until the import thread is idle, or `timeout` passes. `false` on
    /// timeout. For tests and tools; the editor never waits.
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        self.worker
            .as_ref()
            .is_none_or(|worker| worker.wait_idle(timeout))
    }

    /// Everything [`service`] does, returning the errors it would log.
    pub fn update(&mut self, world: &World, backend: &mut impl RenderBackend) -> Vec<String> {
        if let Some(watcher) = &mut self.watcher {
            self.noticed.extend(watcher.settled());
        }
        if !self.noticed.is_empty() {
            self.apply_changes();
        }
        let mut errors = std::mem::take(&mut self.reports);

        let finished: Vec<worker::Done> = match &self.worker {
            Some(worker) => worker.finished().collect(),
            None => Vec::new(),
        };
        for done in finished {
            self.finish(done, world, backend, &mut errors);
        }

        self.bind_meshes(world, backend, &mut errors);
        self.bind_materials(world, backend, &mut errors);
        for (material, texture) in std::mem::take(&mut self.albedo) {
            let handle = self.texture_slot(texture, backend, &mut errors);
            let edited = world
                .get_resource_mut::<AssetPreviews>()
                .is_some_and(|mut previews| {
                    previews.edit_material(material, |m| m.albedo_texture = Some(handle))
                });
            if !edited {
                errors.push(format!(
                    "material {} has no editable copy, so its albedo cannot be set",
                    material.0
                ));
            }
        }
        errors
    }

    /// Read the directory and every sidecar in it again.
    ///
    /// New assets, assets whose settings changed and assets `touched` names —
    /// by source, sidecar or a file their last import read, relative to the
    /// directory — are queued for import. An asset whose file has gone stays,
    /// off disk; see [`iter`](Self::iter).
    fn scan(&mut self, touched: &[String]) -> Vec<String> {
        let mut issues = Vec::new();
        let mut files = Vec::new();
        let mut dirs = Vec::new();
        if self.dir.is_dir() {
            walk(&self.dir, &self.dir, &mut files, &mut dirs, &mut issues);
        }
        files.sort();
        dirs.sort();

        self.folders = dirs
            .into_iter()
            .map(|(rel, path)| {
                let sidecar = sidecar::read_folder(&path, &rel).unwrap_or_else(|issue| {
                    issues.push(issue);
                    None
                });
                Folder {
                    path: rel,
                    id: sidecar.as_ref().map(|s| s.id),
                    tint: sidecar.and_then(|s| s.tint),
                }
            })
            .collect();

        let mut seen: FxHashMap<AssetId, String> = FxHashMap::default();
        let mut queue = Vec::new();
        for entry in &mut self.entries {
            entry.on_disk = false;
        }
        for (rel, path, kind) in files {
            let sidecar = match sidecar::read_or_create(&path, &rel, kind) {
                Ok(sidecar) => sidecar,
                Err(issue) => {
                    issues.push(issue);
                    continue;
                }
            };
            if let Some(first) = seen.get(&sidecar.id) {
                issues.push(format!(
                    "{rel} has the same id as {first}, most likely because it is a copy that kept \
                     the original's sidecar. Delete {rel}.meta to give it an id of its own"
                ));
                continue;
            }
            seen.insert(sidecar.id, rel.clone());

            let is_touched = |entry: &Entry| {
                touched.iter().any(|t| {
                    *t == entry.path
                        || t.strip_suffix(".meta") == Some(entry.path.as_str())
                        || entry.inputs.contains(t)
                })
            };
            match self.entries.iter_mut().find(|entry| entry.id == sidecar.id) {
                Some(entry) => {
                    let changed = entry.kind != kind || entry.settings != sidecar.settings;
                    entry.path = rel;
                    entry.kind = kind;
                    entry.settings = sidecar.settings;
                    entry.on_disk = true;
                    if changed || is_touched(entry) {
                        queue.push(entry.id);
                    }
                }
                None => {
                    self.entries.push(Entry {
                        id: sidecar.id,
                        path: rel,
                        kind,
                        settings: sidecar.settings,
                        status: Status::Queued,
                        inputs: Vec::new(),
                        on_disk: true,
                    });
                    queue.push(sidecar.id);
                }
            }
        }
        self.entries.sort_by(|a, b| a.path.cmp(&b.path));
        for id in queue {
            self.queue(id);
        }
        issues
    }

    /// Put `id` on the import thread: urgently, bringing the artifact back,
    /// when something draws it.
    fn queue(&mut self, id: AssetId) {
        let payload = self.resident(id).is_some();
        let known = self.resident(id).and_then(|resident| resident.key);
        let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) else {
            return;
        };
        if import::importer(entry.kind).is_none() {
            entry.status = Status::Indexed;
            return;
        }
        let Some(worker) = &self.worker else {
            entry.status = Status::Failed(format!(
                "{} cannot be imported: the import thread did not start",
                entry.path
            ));
            return;
        };
        entry.status = Status::Queued;
        if payload {
            self.awaiting.insert(id);
        }
        worker.push(worker::Job {
            id,
            kind: entry.kind,
            source: self.dir.join(&entry.path),
            rel: entry.path.clone(),
            settings: entry.settings,
            payload,
            known,
        });
    }

    /// Take one finished import: record it, and swap its artifact into the
    /// slot drawing it.
    fn finish(
        &mut self,
        done: worker::Done,
        world: &World,
        backend: &mut impl RenderBackend,
        errors: &mut Vec<String>,
    ) {
        let id = done.job.id;
        if done.job.payload {
            self.awaiting.remove(&id);
        }
        let resident = self.resident(id);
        let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) else {
            return;
        };
        match done.outcome {
            Ok(imported) => {
                let folder = entry.path.rsplit_once('/').map_or("", |(folder, _)| folder);
                entry.inputs = imported
                    .inputs
                    .iter()
                    .map(|input| match folder {
                        "" => input.clone(),
                        folder => format!("{folder}/{input}"),
                    })
                    .collect();
                entry.status = Status::Imported {
                    key: imported.key,
                    cached: imported.cached,
                };
                let srgb = matches!(entry.settings, Settings::Texture(t) if t.srgb);
                match imported.payload {
                    Some(payload) => self.install(id, imported.key, payload, srgb, world, backend),
                    None => {
                        let stale = resident.is_some_and(|r| r.key != Some(imported.key));
                        if stale && !self.awaiting.contains(&id) {
                            self.queue(id);
                        }
                    }
                }
            }
            Err(message) => {
                entry.status = Status::Failed(message.clone());
                let kind = entry.kind;
                match resident {
                    // A broken save of an asset already on screen keeps the
                    // last version that imported: the error says what to fix,
                    // and the scene stays usable while it is fixed.
                    Some(resident) if resident.key.is_some() => errors.push(format!(
                        "{message}. Still drawing the last version that imported"
                    )),
                    Some(_) => {
                        errors.push(message);
                        self.install_placeholder(id, kind, world, backend);
                    }
                    None => errors.push(message),
                }
            }
        }
    }

    fn install(
        &mut self,
        id: AssetId,
        key: CacheKey,
        payload: worker::Payload,
        srgb: bool,
        world: &World,
        backend: &mut impl RenderBackend,
    ) {
        let ready = handles::Resident {
            state: LoadState::Ready,
            key: Some(key),
        };
        match payload {
            worker::Payload::Mesh(mesh, preview) => {
                let Some(slot) = self.meshes.handle(id).filter(|h| h & PENDING == 0) else {
                    return;
                };
                if self.meshes.resident(id).and_then(|r| r.key) == Some(key) {
                    return;
                }
                let handle = MeshHandle(slot);
                backend.replace_mesh(handle, &mesh);
                if let Some(mut bounds) = world.get_resource_mut::<MeshBounds>() {
                    bounds.insert(handle, mesh.bounds());
                }
                if let Some(mut previews) = world.get_resource_mut::<AssetPreviews>() {
                    previews.insert_rendered_mesh(handle, preview);
                }
                self.meshes.set_resident(id, ready);
            }
            worker::Payload::Texture(texture) => {
                let Some(slot) = self.textures.handle(id) else {
                    return;
                };
                if self.textures.resident(id).and_then(|r| r.key) == Some(key) {
                    return;
                }
                backend.replace_texture(
                    TextureHandle(slot),
                    &texture.pixels,
                    texture.width,
                    texture.height,
                    srgb,
                );
                self.textures.set_resident(id, ready);
            }
        }
    }

    /// Draw `id` as the placeholder for its kind, marked failed.
    fn install_placeholder(
        &mut self,
        id: AssetId,
        kind: AssetKind,
        world: &World,
        backend: &mut impl RenderBackend,
    ) {
        let failed = handles::Resident {
            state: LoadState::Failed,
            key: None,
        };
        match kind {
            AssetKind::Mesh => {
                let Some(slot) = self.meshes.handle(id).filter(|h| h & PENDING == 0) else {
                    return;
                };
                let cube = placeholder_mesh();
                backend.replace_mesh(MeshHandle(slot), &cube);
                if let Some(mut bounds) = world.get_resource_mut::<MeshBounds>() {
                    bounds.insert(MeshHandle(slot), cube.bounds());
                }
                self.meshes.set_resident(id, failed);
            }
            AssetKind::Texture => {
                if let Some(slot) = self.textures.handle(id) {
                    backend.replace_texture(TextureHandle(slot), &PINK, 1, 1, true);
                    self.textures.set_resident(id, failed);
                }
            }
            AssetKind::Prefab | AssetKind::Folder => {}
        }
    }

    /// Give every pending mesh handle a slot, and the components holding it
    /// the slot's handle.
    fn bind_meshes(
        &mut self,
        world: &World,
        backend: &mut impl RenderBackend,
        errors: &mut Vec<String>,
    ) {
        if !self.meshes.has_pending() {
            return;
        }
        let mut rebound = FxHashMap::default();
        for (promised, id) in self.meshes.take_pending() {
            let (mesh, state) = match self.get(id).map(|entry| &entry.status) {
                None => {
                    errors.push(format!(
                        "mesh asset {id} is not in the project, so it draws as a pink cube. \
                         Restore its file and sidecar from version control, or assign another \
                         mesh"
                    ));
                    (placeholder_mesh(), LoadState::Failed)
                }
                Some(Status::Failed(_)) => (placeholder_mesh(), LoadState::Failed),
                Some(_) => (loading_mesh(), LoadState::Loading),
            };
            let handle = backend.load_mesh(&mesh);
            if let Some(mut bounds) = world.get_resource_mut::<MeshBounds>() {
                bounds.insert(handle, mesh.bounds());
            }
            self.meshes.bind(id, handle.0, state);
            rebound.insert(promised, handle.0);
            if state == LoadState::Loading {
                self.queue(id);
            }
        }
        world.query::<&mut MeshHandle>().for_each(|_, handle| {
            if let Some(&slot) = rebound.get(&handle.0) {
                handle.0 = slot;
            }
        });
    }

    /// [`bind_meshes`](Self::bind_meshes) for materials, every one of which is
    /// the placeholder until materials are files.
    fn bind_materials(
        &mut self,
        world: &World,
        backend: &mut impl RenderBackend,
        errors: &mut Vec<String>,
    ) {
        if !self.materials.has_pending() {
            return;
        }
        let mut rebound = FxHashMap::default();
        for (promised, id) in self.materials.take_pending() {
            errors.push(format!(
                "material asset {id} is not in the project, so it draws pink. Assign another \
                 material"
            ));
            let material = placeholder_material();
            let handle = backend.load_material(&material);
            if let Some(mut blends) = world.get_resource_mut::<MaterialBlends>() {
                blends.insert(handle, material.blend);
            }
            if let Some(mut previews) = world.get_resource_mut::<AssetPreviews>() {
                previews.insert_material(handle, &material, None);
            }
            self.materials.bind(id, handle.0, LoadState::Failed);
            rebound.insert(promised, handle.0);
        }
        world.query::<&mut MaterialHandle>().for_each(|_, handle| {
            if let Some(&slot) = rebound.get(&handle.0) {
                handle.0 = slot;
            }
        });
    }

    /// Texture `id`'s slot, loading one — white while it imports, pink if it
    /// cannot — the first time it is asked for.
    fn texture_slot(
        &mut self,
        id: AssetId,
        backend: &mut impl RenderBackend,
        errors: &mut Vec<String>,
    ) -> TextureHandle {
        if let Some(handle) = self.textures.handle(id) {
            return TextureHandle(handle);
        }
        let (pixel, state) = match self.get(id).map(|entry| &entry.status) {
            None => {
                errors.push(format!(
                    "texture asset {id} is not in the project, so it draws pink"
                ));
                (PINK, LoadState::Failed)
            }
            Some(Status::Failed(_)) => (PINK, LoadState::Failed),
            Some(_) => ([255; 4], LoadState::Loading),
        };
        let srgb = self
            .get(id)
            .is_none_or(|entry| matches!(entry.settings, Settings::Texture(t) if t.srgb));
        let handle = backend.load_texture(&pixel, 1, 1, srgb);
        self.textures.bind(id, handle.0, state);
        if state == LoadState::Loading {
            self.queue(id);
        }
        handle
    }

    /// What `id`'s slot holds, of whichever kind it is.
    fn resident(&self, id: AssetId) -> Option<handles::Resident> {
        self.meshes
            .resident(id)
            .or_else(|| self.textures.resident(id))
            .or_else(|| self.materials.resident(id))
    }

    /// Refuse `id` as `kind` when the project has it as something else.
    fn expect_kind(&self, id: AssetId, kind: AssetKind) -> Result<(), String> {
        match self.get(id) {
            Some(entry) if entry.kind != kind => Err(format!(
                "`{}`, which is a {}",
                entry.path,
                entry.kind.label()
            )),
            _ => Ok(()),
        }
    }

    /// `path` relative to the assets directory with `/` separators, or `None`
    /// for a path outside it or under a dotted directory.
    fn relative(&self, path: &Path) -> Option<String> {
        let rel = path.strip_prefix(&self.dir).ok().or_else(|| {
            let canonical = std::fs::canonicalize(&self.dir).ok()?;
            let rel = path.strip_prefix(canonical).ok()?;
            Some(rel)
        })?;
        let parts: Vec<String> = rel
            .components()
            .map(|part| part.as_os_str().to_string_lossy().into_owned())
            .collect();
        if parts.is_empty() || parts.iter().any(|part| part.starts_with('.')) {
            return None;
        }
        Some(parts.join("/"))
    }
}

/// The colour of a failed texture or material.
const PINK: [u8; 4] = [255, 0, 255, 255];

/// A failed mesh: a cube whose vertex colour tints whatever material it is
/// drawn with pink.
fn placeholder_mesh() -> CpuMesh {
    let mut cube = CpuMesh::cube();
    for vertex in &mut cube.vertices {
        vertex.color = [1.0, 0.0, 1.0];
    }
    cube
}

/// A mesh still importing: one triangle with no area, which rasterises nothing.
fn loading_mesh() -> CpuMesh {
    let vertex = crate::gfx::Vertex {
        position: [0.0; 3],
        normal: [0.0, 1.0, 0.0],
        color: [1.0; 3],
        uv: [0.0; 2],
        tangent: [1.0, 0.0, 0.0, 1.0],
    };
    CpuMesh::new(vec![vertex; 3], vec![0, 1, 2])
}

fn placeholder_material() -> crate::gfx::Material {
    crate::gfx::Material {
        base_color: glam::Vec3::new(1.0, 0.0, 1.0),
        ..Default::default()
    }
}

/// Every asset file and directory below `dir`, relative to `base`. Dotfiles
/// and symlinks are skipped, as the prefab index skips them.
fn walk(
    base: &Path,
    dir: &Path,
    files: &mut Vec<(String, PathBuf, AssetKind)>,
    dirs: &mut Vec<(String, PathBuf)>,
    issues: &mut Vec<String>,
) {
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
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        let rel: Vec<String> = path
            .strip_prefix(base)
            .unwrap_or(&path)
            .components()
            .map(|part| part.as_os_str().to_string_lossy().into_owned())
            .collect();
        let rel = rel.join("/");
        if file_type.is_dir() {
            walk(base, &path, files, dirs, issues);
            dirs.push((rel, path));
        } else if file_type.is_file()
            && let Some(kind) = AssetKind::of_path(&rel)
        {
            files.push((rel, path, kind));
        }
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
    project
        .map_or_else(|| PathBuf::from(".orrin"), |project| project.editor_dir())
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
            let dir =
                std::env::temp_dir().join(format!("orrin-assets-{name}-{}", std::process::id()));
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

        let listed: Vec<(&str, AssetKind)> = db
            .iter()
            .map(|entry| (entry.path.as_str(), entry.kind))
            .collect();
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
            [
                "rock.png has the same id as rock copy.png, most likely because it is a copy that \
              kept the original's sidecar. Delete rock.png.meta to give it an id of its own"
            ]
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
        assert_eq!(
            db.find("props/door.fprefab").unwrap().status,
            Status::Indexed
        );

        let mut again = open(&dir);
        assert!(again.wait_idle(WAIT));
        again.update(&world, &mut backend);
        assert!(matches!(
            again.find("rocks/rock.png").unwrap().status,
            Status::Imported { cached: true, .. }
        ));
        assert_eq!(
            backend.upload_counts(),
            (0, 0, 0),
            "nothing asked to draw anything"
        );
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
        assert!(
            message.starts_with("rocks/bad.png is not a PNG or JPEG"),
            "{message}"
        );
        assert_eq!(errors, std::slice::from_ref(message));
        assert_eq!(db.state(bad.id), LoadState::Failed);
        assert!(matches!(
            db.find("rocks/rock.png").unwrap().status,
            Status::Imported { .. }
        ));
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
        assert_eq!(
            world.resource::<AssetDatabase>().state(id),
            LoadState::Ready
        );
        let triangle =
            crate::geom::Aabb::from_points([glam::Vec3::ZERO, glam::Vec3::X, glam::Vec3::Y]);
        assert_eq!(backend.mesh_bounds(bound), Some(triangle));
        assert_eq!(world.resource::<MeshBounds>().get(bound), Some(triangle));
        assert_eq!(
            handle(&world, entity),
            bound,
            "the slot is the handle for good"
        );
        assert!(
            world
                .resource::<crate::scene::AssetPreviews>()
                .mesh(bound)
                .is_some()
        );
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
        assert!(
            update(&world, &mut backend).is_empty(),
            "named once, not every frame"
        );
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
        change(
            &world,
            &dir,
            "models/tree.gltf",
            moved.as_bytes(),
            &mut backend,
        );
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
        change(
            &world,
            &dir,
            "models/tree.gltf",
            triangle_gltf("").as_bytes(),
            &mut backend,
        );
        assert_eq!(backend.mesh_replacements(), replaced);
    }

    #[test]
    fn a_changed_setting_or_buffer_reimports_too() {
        let dir = Dir::new("db-settings");
        let (world, slot, mut backend) = loaded_tree(&dir);
        let id = world.resource::<AssetDatabase>().mesh_id(slot).unwrap();

        let sidecar = format!("id = \"{id}\"\n\n[import]\nscale = 2.0\n");
        change(
            &world,
            &dir,
            "models/tree.gltf.meta",
            sidecar.as_bytes(),
            &mut backend,
        );
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
        let edits = world
            .resource_mut::<crate::scene::AssetPreviews>()
            .take_material_edits();
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].1.albedo_texture, Some(texture));

        assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
        update(&world, &mut backend);
        assert_eq!(
            world.resource::<AssetDatabase>().state(rock),
            LoadState::Ready
        );
        assert_eq!(backend.texture_replacements(), 1);

        let tree = world
            .resource::<AssetDatabase>()
            .find("models/tree.gltf")
            .unwrap()
            .id;
        assert_eq!(
            world
                .resource_mut::<AssetDatabase>()
                .assign_albedo(MaterialHandle(4), tree),
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
        assert!(matches!(
            db.state(rock),
            LoadState::Loading | LoadState::Ready
        ));
    }
}
