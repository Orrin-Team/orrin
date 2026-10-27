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
//! an object that silently vanishes. A normal map is the exception: it stands
//! in as a flat normal, because white or pink read as a normal tilts the
//! surface far off true.
//!
//! # Materials
//!
//! A `.fmat` ([`material`]) has nothing to import: it is a few hundred bytes of
//! text, read on this thread when it is queued, as a prefab is. Its last good
//! read is kept, so a save that does not parse goes on drawing the version
//! before it while the error says what to fix. A reread updates the material
//! in its slot, and its blend mode with it, since that decides which queue the
//! material draws in.
//!
//! The editor writes a material through the edit pipeline
//! ([`write_asset`](AssetDatabase::write_asset)) and draws the edit at once.
//! The text last read or written is kept, and a read that finds the same text
//! is nothing new — which is what the watcher hearing the editor's own save
//! amounts to, so that save is not read back and uploaded a second time.
//!
//! # What is not here
//!
//! Engine-built assets — the demo library's cube and gold, a model the demo
//! scene imports — stay in [`Assets`](crate::scene::Assets), named rather
//! than imported. They get stable ids too, derived from their names by
//! [`AssetId::named`], so a scene can reference one the same way it
//! references a file.

pub mod cache;
mod handles;
pub mod import;
mod ktx;
pub mod material;
pub mod sidecar;
pub mod texture;
mod watch;
pub mod worker;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use orrin_ecs::{FxBuildHasher, FxHashMap, World};
use orrin_registry::AssetId;

pub use cache::CacheKey;
pub use sidecar::{AssetKind, MeshSettings, Settings, TextureSettings, Tint, Usage};
pub use texture::{Target, TexelFormat};

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
    /// What the last import made, once there is one.
    pub summary: Option<cache::Summary>,
    /// The source file's size in bytes, as of the last scan.
    pub bytes: u64,
    /// When the source file was last written, as of the last scan.
    pub modified: Option<std::time::SystemTime>,
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
    /// Each material's last `.fmat` that read, or the edit drawn in its place.
    material_files: FxHashMap<AssetId, material::MaterialFile>,
    /// Each material file's text as this database last read or wrote it.
    material_texts: FxHashMap<AssetId, String>,
    /// Materials read since the last update, whose slots are behind their files.
    reread: Vec<AssetId>,
    /// What a texture slot naming something other than a texture samples.
    wrong_kind: Option<TextureHandle>,
    /// Issues found between frames, returned by the next [`update`](Self::update).
    reports: Vec<String>,
    /// What texture artifacts are made for.
    target: Target,
    /// Each imported asset's thumbnail, with the artifact it pictures.
    thumbnails: FxHashMap<AssetId, (CacheKey, crate::gfx::thumbnail::Thumbnail)>,
    /// Assets with an import on its way back for a slot.
    awaiting: HashSet<AssetId, FxBuildHasher>,
}

impl AssetDatabase {
    /// [`open_for`](Self::open_for) a device that samples only RGBA8.
    pub fn open(dir: impl Into<PathBuf>, cache_dir: impl Into<PathBuf>) -> (Self, Vec<String>) {
        Self::open_for(dir, cache_dir, Target::Rgba8)
    }

    /// Index `dir` and start importing everything in it into `cache_dir`, each
    /// texture in the format `target` samples.
    ///
    /// The target is fixed for the database's life: it is what the device can
    /// do, and opening with it — rather than learning it later — is what keeps
    /// the first import of every texture from being made twice.
    ///
    /// Returns what could not be indexed — a sidecar that does not read, two
    /// files claiming one id — each naming its file. The rest still index.
    pub fn open_for(
        dir: impl Into<PathBuf>,
        cache_dir: impl Into<PathBuf>,
        target: Target,
    ) -> (Self, Vec<String>) {
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
            material_files: FxHashMap::default(),
            material_texts: FxHashMap::default(),
            reread: Vec::new(),
            wrong_kind: None,
            reports: Vec::new(),
            target,
            thumbnails: FxHashMap::default(),
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

    /// The handle for material `id`, pending until the next [`service`] binds
    /// it to its `.fmat`. An id with no file draws pink and keeps the id, as
    /// [`mesh`](Self::mesh) does.
    pub fn material(&mut self, id: AssetId) -> Result<MaterialHandle, String> {
        self.expect_kind(id, AssetKind::Material)?;
        Ok(MaterialHandle(self.materials.promise(id)))
    }

    pub fn material_id(&self, handle: MaterialHandle) -> Option<AssetId> {
        self.materials.id(handle.0)
    }

    /// The asset browser's picture of `id`, with the artifact it was made from,
    /// which changes when the asset is reimported. `None` for a kind with no
    /// artifact — a prefab, a material — and before the first import lands.
    pub fn thumbnail(&self, id: AssetId) -> Option<(CacheKey, &crate::gfx::thumbnail::Thumbnail)> {
        self.thumbnails
            .get(&id)
            .map(|(key, picture)| (*key, picture))
    }

    /// The texture slot texture `id` was loaded into, if anything has asked
    /// for it.
    pub fn texture(&self, id: AssetId) -> Option<TextureHandle> {
        self.textures.handle(id).map(TextureHandle)
    }

    /// Material `id` as it is drawn: its file's last good read, or an edit in
    /// progress.
    pub fn material_file(&self, id: AssetId) -> Option<&material::MaterialFile> {
        self.material_files.get(&id)
    }

    /// The text of material `id`'s file as this database last read or wrote
    /// it, which an undo writes back.
    pub fn material_text(&self, id: AssetId) -> Option<&str> {
        self.material_texts.get(&id).map(String::as_str)
    }

    /// Refuse `file` as material `id` when a texture slot it fills differently
    /// names something other than a texture. A slot the file already had is
    /// left to [`update`](Self::update) to name, as one read from disk is.
    pub fn check_material(&self, id: AssetId, file: &material::MaterialFile) -> Result<(), String> {
        let current = self.material_files.get(&id).map(|kept| kept.textures);
        let current = current.unwrap_or_default().slots();
        for ((slot, texture), (_, was)) in file.textures.slots().into_iter().zip(current) {
            let Some(texture) = texture.filter(|&texture| Some(texture) != was) else {
                continue;
            };
            match self.get(texture) {
                Some(entry) if entry.kind == AssetKind::Texture => {}
                Some(entry) => {
                    return Err(format!(
                        "{}: the `{slot}` slot takes a texture, not `{}`, which is a {}",
                        self.path_of(id),
                        entry.path,
                        entry.kind.label()
                    ));
                }
                None => {
                    return Err(format!(
                        "{}: the `{slot}` slot names texture asset {texture}, which is not in \
                         the project",
                        self.path_of(id)
                    ));
                }
            }
        }
        Ok(())
    }

    /// Draw material `id` as `file` from the next [`update`](Self::update),
    /// without writing it: an edit still in progress.
    pub fn preview_material(
        &mut self,
        id: AssetId,
        file: material::MaterialFile,
    ) -> Result<(), String> {
        let Some(kept) = self.material_files.get_mut(&id) else {
            return Err(format!(
                "{}: there is no read of it to edit",
                self.path_of(id)
            ));
        };
        *kept = file;
        if !self.reread.contains(&id) {
            self.reread.push(id);
        }
        Ok(())
    }

    /// Write `text` as asset `id`'s file, through a temporary file and a
    /// rename, and draw it from the next [`update`](Self::update).
    ///
    /// Only a material is written this way, and only a text that reads: the
    /// editor never saves a file it would then refuse to load. What is written
    /// is remembered, so the watcher hearing it is nothing new.
    pub fn write_asset(&mut self, id: AssetId, text: &str) -> Result<(), String> {
        let entry = match self.get(id) {
            Some(entry) if entry.kind == AssetKind::Material => entry,
            Some(entry) => {
                return Err(format!(
                    "`{}`, which is a {}, is not written by the editor",
                    entry.path,
                    entry.kind.label()
                ));
            }
            None => return Err(format!("asset {id} is not in the project")),
        };
        let rel = entry.path.clone();
        let file = material::parse(text, &rel)?;
        write_atomically(&self.dir.join(&rel), text)
            .map_err(|error| format!("could not write {rel}: {error}"))?;
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) {
            entry.status = Status::Indexed;
        }
        self.material_texts.insert(id, text.to_owned());
        let drawn = self
            .material_files
            .get(&id)
            .is_some_and(|kept| material::write(kept) == material::write(&file));
        self.material_files.insert(id, file);
        if !drawn && !self.reread.contains(&id) {
            self.reread.push(id);
        }
        Ok(())
    }

    /// Write a material at the default parameters into `folder`, named `New
    /// Material` and numbered past any already there, with its sidecar, and
    /// index it.
    pub fn create_material(&mut self, folder: &str) -> Result<AssetId, String> {
        let rel = (1..)
            .map(|n| {
                let name = match n {
                    1 => format!("New Material.{}", material::EXTENSION),
                    n => format!("New Material {n}.{}", material::EXTENSION),
                };
                match folder {
                    "" => name,
                    folder => format!("{folder}/{name}"),
                }
            })
            .find(|rel| !self.dir.join(rel).exists())
            .expect("an unbounded range finds one");
        let path = self.dir.join(&rel);
        write_atomically(&path, &material::write(&Default::default()))
            .map_err(|error| format!("could not write {rel}: {error}"))?;
        let id = sidecar::read_or_create(&path, &rel, AssetKind::Material)?.id;
        self.rescan();
        Ok(id)
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

    /// Rename the asset or folder at `path` to `name`, returning its new path.
    ///
    /// The sidecar goes with it, so the id does, and every scene naming the
    /// asset still finds it. An asset keeps its extension, since that is what
    /// makes it the kind it is; typing the extension too is not doubling it.
    /// A file operation rather than an edit, so it has no undo — as a rename
    /// in an explorer has none.
    pub fn rename(&mut self, path: &str, name: &str) -> Result<String, String> {
        let folder = self.folders.iter().any(|folder| folder.path == path);
        if !folder && self.find(path).is_none() {
            return Err(format!(
                "there is no asset or folder `{path}` in {}",
                self.dir.display()
            ));
        }
        let refuse = |why: String| format!("could not rename {path}: {why}");
        let name = name.trim();
        let extension = match folder {
            true => None,
            false => path.rsplit_once('.').map(|(_, extension)| extension),
        };
        let stem = extension
            .and_then(|extension| {
                let (stem, typed) = name.rsplit_once('.')?;
                typed.eq_ignore_ascii_case(extension).then_some(stem)
            })
            .unwrap_or(name);
        check_name(stem).map_err(refuse)?;
        let file = match extension {
            Some(extension) => format!("{stem}.{extension}"),
            None => stem.to_owned(),
        };
        let rel = match path.rsplit_once('/') {
            Some((parent, _)) => format!("{parent}/{file}"),
            None => file,
        };
        if rel == path {
            return Ok(rel);
        }
        let (from, to) = (self.dir.join(path), self.dir.join(&rel));
        // On a case-insensitive file system a change of case finds itself.
        if to.exists() && !rel.eq_ignore_ascii_case(path) {
            return Err(refuse(format!("{rel} already exists")));
        }
        std::fs::rename(&from, &to).map_err(|error| refuse(error.to_string()))?;
        let meta = sidecar::path_of(&from);
        if meta.exists()
            && let Err(error) = std::fs::rename(&meta, sidecar::path_of(&to))
        {
            let _ = std::fs::rename(&to, &from);
            return Err(refuse(format!("its sidecar would not move: {error}")));
        }
        self.rescan();
        Ok(rel)
    }

    /// Import asset `id` with `settings` from now on, writing them to its
    /// sidecar.
    ///
    /// The sidecar is rewritten whole, with the id it had: an asset's holds
    /// nothing else. Settings that do not fit the asset's kind, or a scale
    /// that is not a positive number, are refused before anything is written.
    pub fn set_settings(&mut self, id: AssetId, settings: Settings) -> Result<(), String> {
        let Some(entry) = self.get(id) else {
            return Err(format!("asset {id} is not in the project"));
        };
        let rel = entry.path.clone();
        let fits = matches!(
            (entry.kind, settings),
            (AssetKind::Texture, Settings::Texture(_)) | (AssetKind::Mesh, Settings::Mesh(_))
        );
        if !fits {
            return Err(format!(
                "{rel} is a {}, which does not take these import settings",
                entry.kind.label()
            ));
        }
        if let Settings::Mesh(mesh) = settings
            && !(mesh.scale.is_finite() && mesh.scale > 0.0)
        {
            return Err(format!(
                "{rel}: `scale` must be a positive number, not {}",
                mesh.scale
            ));
        }
        let meta = sidecar::path_of(&self.dir.join(&rel));
        write_atomically(&meta, &sidecar::text(id, &settings))
            .map_err(|error| format!("could not write {rel}.meta: {error}"))?;
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) {
            entry.settings = settings;
        }
        self.queue(id);
        Ok(())
    }

    /// Import asset `id` again, even when the cache holds what it would make:
    /// for an artifact suspected of being stale, or an importer being worked on.
    pub fn reimport(&mut self, id: AssetId) -> Result<(), String> {
        let Some(entry) = self.get(id) else {
            return Err(format!("asset {id} is not in the project"));
        };
        if import::importer(entry.kind).is_none() {
            return Err(format!(
                "{} is a {}, which is read as it is rather than imported",
                entry.path,
                entry.kind.label()
            ));
        }
        self.enqueue(id, true);
        Ok(())
    }

    /// Copy `files` from outside the project into `folder`, each under its own
    /// name or, where that is taken, numbered past it — `rock 2.png`. Returns
    /// the assets that came in, by path, and what could not, each by name.
    ///
    /// A `.meta` is never copied: the copy is a new asset, and its original's
    /// sidecar would give it an id another asset may answer to. A `.gltf`
    /// brings the buffers and images it names, at the paths it names them by,
    /// since those cannot be renumbered without rewriting it; one that would
    /// overwrite a different file of the same name is refused whole. A folder
    /// is refused rather than its whole tree copied.
    pub fn bring_in(&mut self, files: &[PathBuf], folder: &str) -> (Vec<String>, Vec<String>) {
        let mut brought = Vec::new();
        let mut errors = Vec::new();
        let into = self.dir.join(folder);
        let shown = if folder.is_empty() { "assets" } else { folder };
        for source in files {
            let name = source.file_name().map_or_else(
                || source.display().to_string(),
                |name| name.to_string_lossy().into_owned(),
            );
            if source.is_dir() {
                errors.push(format!(
                    "{name} is a folder; bring in the files inside it instead"
                ));
                continue;
            }
            if name.to_ascii_lowercase().ends_with(".meta") {
                continue;
            }
            let rel = self.free_name(folder, &name);
            let copy = |from: &Path, to: &Path| {
                to.parent()
                    .map_or(Ok(()), std::fs::create_dir_all)
                    .and_then(|()| std::fs::copy(from, to))
                    .map_err(|error| format!("could not copy {name} into {shown}: {error}"))
            };
            let copied = companions(source, &name, &into, folder).and_then(|companions| {
                for (from, to) in companions {
                    copy(&from, &to)?;
                }
                copy(source, &self.dir.join(&rel))
            });
            match copied {
                Ok(_) if AssetKind::of_path(&rel).is_some() => brought.push(rel),
                Ok(_) => {}
                Err(error) => errors.push(error),
            }
        }
        if !files.is_empty() {
            self.rescan();
        }
        (brought, errors)
    }

    /// `name` in `folder`, numbered past any file there — or any sidecar left
    /// without one, which would otherwise be read as the copy's.
    fn free_name(&self, folder: &str, name: &str) -> String {
        let (stem, extension) = match name.rsplit_once('.') {
            Some((stem, extension)) if !stem.is_empty() => (stem, Some(extension)),
            _ => (name, None),
        };
        (1..)
            .map(|n| {
                let file = match (n, extension) {
                    (1, _) => name.to_owned(),
                    (n, Some(extension)) => format!("{stem} {n}.{extension}"),
                    (n, None) => format!("{stem} {n}"),
                };
                match folder {
                    "" => file,
                    folder => format!("{folder}/{file}"),
                }
            })
            .find(|rel| {
                let path = self.dir.join(rel);
                !path.exists() && !sidecar::path_of(&path).exists()
            })
            .expect("an unbounded range finds one")
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
        self.refresh_materials(world, backend, &mut errors);
        self.bind_materials(world, backend, &mut errors);
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
        files.sort_by(|a, b| a.rel.cmp(&b.rel));
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
        for Found {
            rel,
            path,
            kind,
            bytes,
            modified,
        } in files
        {
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
                    // A moved asset is imported again: the files it reads are
                    // known by path. Its source is hashed under no name, so a
                    // moved texture finds the artifact it had.
                    let changed = entry.kind != kind
                        || entry.settings != sidecar.settings
                        || entry.path != rel;
                    entry.path = rel;
                    entry.kind = kind;
                    entry.settings = sidecar.settings;
                    entry.on_disk = true;
                    entry.bytes = bytes;
                    entry.modified = modified;
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
                        summary: None,
                        bytes,
                        modified,
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
        self.enqueue(id, false);
    }

    /// [`queue`](Self::queue), past the cache when `fresh`.
    fn enqueue(&mut self, id: AssetId, fresh: bool) {
        let payload = self.resident(id).is_some();
        let known = self.resident(id).and_then(|resident| resident.key);
        let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) else {
            return;
        };
        if entry.kind == AssetKind::Material {
            self.read_material(id);
            return;
        }
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
            target: self.target,
            payload,
            known,
            fresh,
        });
    }

    /// Read material `id`'s file, keeping it if it reads and naming it if not.
    /// A text already read or written is nothing new: most often it is the
    /// watcher hearing the editor's own save.
    fn read_material(&mut self, id: AssetId) {
        let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) else {
            return;
        };
        let text = std::fs::read_to_string(self.dir.join(&entry.path))
            .map_err(|error| format!("{}: {error}", entry.path));
        match &text {
            Ok(text) if self.material_texts.get(&id) == Some(text) => return,
            Ok(text) => {
                self.material_texts.insert(id, text.clone());
            }
            Err(_) => {
                self.material_texts.remove(&id);
            }
        }
        let read = text.and_then(|text| material::parse(&text, &entry.path));
        match read {
            Ok(file) => {
                entry.status = Status::Indexed;
                self.material_files.insert(id, file);
                if !self.reread.contains(&id) {
                    self.reread.push(id);
                }
            }
            Err(message) => {
                entry.status = Status::Failed(message.clone());
                let drawn = self.materials.resident(id).is_some();
                self.reports
                    .push(if drawn && self.material_files.contains_key(&id) {
                        format!("{message}. Still drawing the last good version")
                    } else {
                        message
                    });
            }
        }
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
                entry.summary = imported.summary;
                if let Some(picture) = imported.thumbnail {
                    self.thumbnails.insert(id, (imported.key, picture));
                }
                match imported.payload {
                    Some(payload) => self.install(id, imported.key, payload, world, backend),
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
                backend.replace_texture(TextureHandle(slot), &texture);
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
                    let (texel, format) = match self.usage(id) {
                        Some(Usage::Normal) => (FLAT_NORMAL, TexelFormat::Rgba8Unorm),
                        _ => (PINK, TexelFormat::Rgba8Srgb),
                    };
                    let placeholder = crate::gfx::TextureLevels {
                        format,
                        width: 1,
                        height: 1,
                        levels: vec![texel.to_vec()],
                    };
                    backend.replace_texture(TextureHandle(slot), &placeholder);
                    self.textures.set_resident(id, failed);
                }
            }
            AssetKind::Prefab | AssetKind::Material | AssetKind::Folder => {}
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

    /// [`bind_meshes`](Self::bind_meshes) for materials: each loads as its
    /// file's last good read, or as the pink placeholder without one.
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
            let (material, state) = match self.resolve_material(id, backend, errors) {
                Some(material) => (material, LoadState::Ready),
                None => {
                    if self.get(id).is_none() {
                        errors.push(format!(
                            "material asset {id} is not in the project, so it draws pink. \
                             Assign another material"
                        ));
                    }
                    (placeholder_material(), LoadState::Failed)
                }
            };
            let handle = backend.load_material(&material);
            register_material(world, handle, &material);
            self.materials.bind(id, handle.0, state);
            rebound.insert(promised, handle.0);
        }
        world.query::<&mut MaterialHandle>().for_each(|_, handle| {
            if let Some(&slot) = rebound.get(&handle.0) {
                handle.0 = slot;
            }
        });
    }

    /// Load each bound material whose file was read since the last update
    /// into the slot it already has.
    fn refresh_materials(
        &mut self,
        world: &World,
        backend: &mut impl RenderBackend,
        errors: &mut Vec<String>,
    ) {
        for id in std::mem::take(&mut self.reread) {
            let Some(slot) = self.materials.handle(id).filter(|h| h & PENDING == 0) else {
                continue;
            };
            let Some(material) = self.resolve_material(id, backend, errors) else {
                continue;
            };
            let handle = MaterialHandle(slot);
            backend.update_material(handle, &material);
            register_material(world, handle, &material);
            let ready = handles::Resident {
                state: LoadState::Ready,
                key: None,
            };
            self.materials.set_resident(id, ready);
        }
    }

    /// Material `id` as its last good file has it, each slot given its
    /// texture's handle; `None` when no read of it has succeeded.
    fn resolve_material(
        &mut self,
        id: AssetId,
        backend: &mut impl RenderBackend,
        errors: &mut Vec<String>,
    ) -> Option<crate::gfx::Material> {
        let file = *self.material_files.get(&id)?;
        let rel = self
            .get(id)
            .map_or_else(|| id.to_string(), |entry| entry.path.clone());
        Some(file.resolve(|slot, texture| match self.get(texture) {
            Some(entry) if entry.kind != AssetKind::Texture => {
                errors.push(format!(
                    "{rel}: the `{slot}` slot names `{}`, which is a {}, so it draws pink. \
                     Assign a texture",
                    entry.path,
                    entry.kind.label()
                ));
                *self
                    .wrong_kind
                    .get_or_insert_with(|| backend.load_texture(&PINK, 1, 1, true))
            }
            _ => {
                let normal = material::MaterialTextures::is_normal_slot(slot);
                self.texture_slot(texture, normal, backend, errors)
            }
        }))
    }

    /// Texture `id`'s slot, loading one the first time it is asked for: white
    /// while it imports and pink if it cannot, or a flat normal for both when
    /// it is a normal map. `normal_slot` says the asker samples it as one,
    /// which decides for an id with no sidecar to say.
    fn texture_slot(
        &mut self,
        id: AssetId,
        normal_slot: bool,
        backend: &mut impl RenderBackend,
        errors: &mut Vec<String>,
    ) -> TextureHandle {
        if let Some(handle) = self.textures.handle(id) {
            return TextureHandle(handle);
        }
        let flat = normal_slot || self.usage(id) == Some(Usage::Normal);
        let (failed, loading, drawn) = if flat {
            (FLAT_NORMAL, FLAT_NORMAL, "flat")
        } else {
            (PINK, [255; 4], "pink")
        };
        let (pixel, state) = match self.get(id).map(|entry| &entry.status) {
            None => {
                errors.push(format!(
                    "texture asset {id} is not in the project, so it draws {drawn}"
                ));
                (failed, LoadState::Failed)
            }
            Some(Status::Failed(_)) => (failed, LoadState::Failed),
            Some(_) => (loading, LoadState::Loading),
        };
        let srgb = !flat
            && self.get(id).is_none_or(
                |entry| matches!(entry.settings, Settings::Texture(t) if t.usage.is_srgb()),
            );
        let handle = backend.load_texture(&pixel, 1, 1, srgb);
        self.textures.bind(id, handle.0, state);
        if state == LoadState::Loading {
            self.queue(id);
        }
        handle
    }

    /// `id`'s path for a message, or the id itself when the project has no
    /// file for it.
    fn path_of(&self, id: AssetId) -> String {
        self.get(id)
            .map_or_else(|| id.to_string(), |entry| entry.path.clone())
    }

    /// What texture `id`'s sidecar says it holds.
    fn usage(&self, id: AssetId) -> Option<Usage> {
        match self.get(id)?.settings {
            Settings::Texture(texture) => Some(texture.usage),
            _ => None,
        }
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

/// A normal map's stand-in: a normal straight out of the surface, stored
/// linearly.
const FLAT_NORMAL: [u8; 4] = [128, 128, 255, 255];

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

/// Record what extraction and the editor read of the material in `handle`.
fn register_material(world: &World, handle: MaterialHandle, material: &crate::gfx::Material) {
    if let Some(mut blends) = world.get_resource_mut::<MaterialBlends>() {
        blends.insert(handle, material.blend);
    }
    if let Some(mut previews) = world.get_resource_mut::<AssetPreviews>() {
        previews.insert_material(handle, material, None);
    }
}

fn placeholder_material() -> crate::gfx::Material {
    crate::gfx::Material {
        base_color: glam::Vec3::new(1.0, 0.0, 1.0),
        ..Default::default()
    }
}

/// Write `text` to `path` through a dotted temporary file beside it and a
/// rename, so neither the watcher nor a crash ever sees half a file. The scan
/// and the watcher both pass over dotfiles, so the temporary is never indexed.
fn write_atomically(path: &Path, text: &str) -> std::io::Result<()> {
    let name = path
        .file_name()
        .map_or_else(Default::default, |name| name.to_string_lossy());
    let temporary = path.with_file_name(format!(".{name}.{}", std::process::id()));
    std::fs::write(&temporary, text)?;
    std::fs::rename(&temporary, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temporary);
    })
}

/// Every asset file and directory below `dir`, relative to `base`. Dotfiles
/// and symlinks are skipped, as the prefab index skips them.
/// The files a `.gltf` at `source` names, each with where it goes in `into`
/// beside the copy, leaving out any already there byte for byte. Nothing for
/// any other file.
fn companions(
    source: &Path,
    name: &str,
    into: &Path,
    folder: &str,
) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    if !name.to_ascii_lowercase().ends_with(".gltf") {
        return Ok(Vec::new());
    }
    let bytes =
        std::fs::read(source).map_err(|error| format!("{name} could not be read: {error}"))?;
    let base = source.parent().unwrap_or(Path::new("."));
    let mut out = Vec::new();
    for uri in import::referenced_files(name, &bytes)? {
        let relative = Path::new(&uri);
        let inside = relative
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)));
        if !inside {
            return Err(format!(
                "{name} reads `{uri}`, which is outside its own folder; copy the two in by hand"
            ));
        }
        let (from, to) = (base.join(relative), into.join(relative));
        let wanted = std::fs::read(&from)
            .map_err(|error| format!("{name} reads `{uri}`, which could not be read: {error}"))?;
        match std::fs::read(&to) {
            Ok(there) if there == wanted => {}
            Ok(_) => {
                let rel = match folder {
                    "" => uri.clone(),
                    folder => format!("{folder}/{uri}"),
                };
                return Err(format!(
                    "{name} reads `{uri}`, and {rel} is already there with other contents. Bring \
                     {name} into a folder of its own"
                ));
            }
            Err(_) => out.push((from, to)),
        }
    }
    Ok(out)
}

/// Why `name` cannot be a file or folder name on every system a project may
/// be opened on, or be seen by the scan at all.
fn check_name(name: &str) -> Result<(), String> {
    const RESERVED: [&str; 4] = ["CON", "PRN", "AUX", "NUL"];
    if name.is_empty() || name.chars().all(|c| c == '.') {
        return Err("it needs a name".to_owned());
    }
    let forbidden = |c: char| matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|');
    if let Some(c) = name.chars().find(|&c| forbidden(c) || c.is_control()) {
        return Err(format!("a name cannot hold `{}`", c.escape_default()));
    }
    if name.starts_with('.') {
        return Err("a name starting with `.` is hidden from the asset scan".to_owned());
    }
    if name.ends_with('.') {
        return Err("Windows drops a dot at the end of a name".to_owned());
    }
    let device = name.split('.').next().unwrap_or(name).to_ascii_uppercase();
    let numbered = |prefix: &str| {
        device
            .strip_prefix(prefix)
            .is_some_and(|n| n.len() == 1 && n.as_bytes()[0].is_ascii_digit())
    };
    if RESERVED.contains(&device.as_str()) || numbered("COM") || numbered("LPT") {
        return Err(format!("`{name}` is reserved on Windows"));
    }
    Ok(())
}

/// A source file [`walk`] found.
struct Found {
    rel: String,
    path: PathBuf,
    kind: AssetKind,
    bytes: u64,
    modified: Option<std::time::SystemTime>,
}

fn walk(
    base: &Path,
    dir: &Path,
    files: &mut Vec<Found>,
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
            // The listing's own metadata: on Windows it came with the directory
            // read, so this is not another call per file.
            let metadata = entry.metadata().ok();
            files.push(Found {
                rel,
                path,
                kind,
                bytes: metadata.as_ref().map_or(0, |m| m.len()),
                modified: metadata.and_then(|m| m.modified().ok()),
            });
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

    /// A rename takes the sidecar along, so the asset keeps its id and every
    /// scene naming it still finds it; the extension stays, since it is what
    /// makes the file the kind of asset it is.
    #[test]
    fn a_rename_moves_the_file_with_its_sidecar_and_keeps_the_id() {
        let dir = Dir::new("db-rename");
        project(&dir);
        dir.write("rocks/pebble.png", &png(1, 1, |_, _| [0; 4]));
        let mut db = open(&dir);
        let rock = db.find("rocks/rock.png").unwrap().id;

        assert_eq!(
            db.rename("rocks/rock.png", " boulder "),
            Ok("rocks/boulder.png".into())
        );
        assert_eq!(db.find("rocks/boulder.png").map(|e| e.id), Some(rock));
        assert!(db.find("rocks/rock.png").is_none());
        assert!(dir.path().join("rocks/boulder.png.meta").is_file());
        assert!(!dir.path().join("rocks/rock.png.meta").exists());
        assert_eq!(
            db.rename("rocks/boulder.png", "stone.PNG"),
            Ok("rocks/stone.png".into()),
            "its own extension typed is not doubled"
        );
        assert_eq!(
            db.rename("rocks/stone.png", "Stone"),
            Ok("rocks/Stone.png".into()),
            "a change of case is not the name being taken"
        );

        for (name, why) in [
            ("", "a name"),
            ("a/b", "/"),
            ("a\\b", "\\"),
            ("a:b", ":"),
            ("pebble", "rocks/pebble.png already exists"),
            (".", "a name"),
            ("con", "reserved"),
        ] {
            let error = db.rename("rocks/Stone.png", name).unwrap_err();
            assert!(
                error.contains("rocks/Stone.png") && error.contains(why),
                "{name}: {error}"
            );
        }
        assert!(
            db.rename("rocks/nothing.png", "x")
                .unwrap_err()
                .contains("rocks/nothing.png")
        );

        db.set_folder_tint("rocks", Some(Tint([1, 2, 3]))).unwrap();
        assert_eq!(db.rename("rocks", "stones"), Ok("stones".into()));
        assert_eq!(db.find("stones/Stone.png").map(|e| e.id), Some(rock));
        let stones = db.folders().iter().find(|f| f.path == "stones").unwrap();
        assert_eq!(stones.tint, Some(Tint([1, 2, 3])), "with its sidecar");
        assert!(!dir.path().join("rocks.meta").exists());
        assert!(db.folders().iter().all(|f| f.path != "rocks"));
        assert!(
            db.rename("stones", "props")
                .unwrap_err()
                .contains("props already exists")
        );

        // A moved glTF imports again, so the buffers it reads are known by
        // where they are now and a save to one still reimports it.
        assert!(db.wait_idle(WAIT));
        db.update(&World::new(), &mut HeadlessBackend::new());
        db.rename("models", "trees").unwrap();
        assert_eq!(db.find("trees/tree.gltf").unwrap().status, Status::Queued);
        assert!(db.wait_idle(WAIT));
        db.update(&World::new(), &mut HeadlessBackend::new());
        assert_eq!(
            db.find("trees/tree.gltf").unwrap().inputs,
            ["trees/tri angle.bin"]
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

    fn imported_key(db: &AssetDatabase, path: &str) -> (CacheKey, bool) {
        match db.find(path).unwrap().status {
            Status::Imported { key, cached } => (key, cached),
            ref other => panic!("{path} is {other:?}"),
        }
    }

    /// Files from outside the project are copied in under names of their own.
    /// A copy is a new asset, so a sidecar dropped with it is left behind; a
    /// glTF brings the buffers and images it names, where it names them.
    #[test]
    fn files_brought_in_are_copied_with_unique_names_and_what_a_gltf_reads() {
        let outside = Dir::new("bring-in-source");
        let rock = outside.write("rock.png", &png(1, 1, |_, _| [9; 4]));
        let foreign = "5f0c3a4e-0000-4000-8000-0000000000aa";
        let meta = outside.write("rock.png.meta", format!("id = \"{foreign}\"\n").as_bytes());
        let gltf = triangle_gltf("").replace(
            "\"buffers\"",
            "\"images\": [{\"uri\": \"tex/bark%201.png\"}],\n  \"buffers\"",
        );
        let tree = outside.write("tree.gltf", gltf.as_bytes());
        let bin = outside.write("tri angle.bin", &triangle_bin());
        outside.write("tex/bark 1.png", &png(1, 1, |_, _| [1; 4]));
        let folder = outside
            .write("loose/notes.txt", b"")
            .parent()
            .unwrap()
            .to_owned();

        let dir = Dir::new("bring-in");
        project(&dir);
        dir.write("props/rock.png", &png(1, 1, |_, _| [0; 4]));
        let mut db = open(&dir);
        let (brought, errors) = db.bring_in(&[rock, meta, tree.clone(), folder], "props");
        assert_eq!(brought, ["props/rock 2.png", "props/tree.gltf"]);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("loose"), "{errors:?}");
        assert!(dir.path().join("props/tri angle.bin").is_file());
        assert!(dir.path().join("props/tex/bark 1.png").is_file());
        let copy = db.find("props/rock 2.png").expect("indexed");
        assert_ne!(
            copy.id.to_string(),
            foreign,
            "a copy is an asset of its own"
        );
        assert!(db.find("props/tree.gltf").is_some());

        let (brought, errors) = db.bring_in(std::slice::from_ref(&tree), "props");
        assert_eq!(brought, ["props/tree 2.gltf"], "{errors:?}");
        assert!(errors.is_empty(), "the same buffer is shared: {errors:?}");

        let mut other = triangle_bin();
        other[0] ^= 1;
        std::fs::write(&bin, other).unwrap();
        let (brought, errors) = db.bring_in(&[tree], "props");
        assert!(brought.is_empty());
        assert!(errors[0].contains("props/tri angle.bin"), "{errors:?}");
        assert!(
            !dir.path().join("props/tree 3.gltf").exists(),
            "nothing half copied"
        );
    }

    /// The Import page's Apply: the sidecar is rewritten with the id it had and
    /// the new settings, which a fresh scan reads back, and the asset imports
    /// again under them. Settings that do not fit the asset are refused by name.
    #[test]
    fn new_settings_are_written_to_the_sidecar_with_the_id_kept_and_imported() {
        use sidecar::{MeshSettings, TextureSettings};
        let dir = Dir::new("db-set-settings");
        project(&dir);
        let mut db = open(&dir);
        assert!(db.wait_idle(WAIT));
        db.update(&World::new(), &mut HeadlessBackend::new());
        let rock = db.find("rocks/rock.png").unwrap().id;
        let (before, _) = imported_key(&db, "rocks/rock.png");

        let settings = Settings::Texture(TextureSettings {
            usage: Usage::Data,
            compress: false,
            max_size: 1,
        });
        db.set_settings(rock, settings).unwrap();
        assert_eq!(db.get(rock).unwrap().settings, settings);
        assert_eq!(db.get(rock).unwrap().status, Status::Queued);
        assert!(db.wait_idle(WAIT));
        db.update(&World::new(), &mut HeadlessBackend::new());
        assert_ne!(imported_key(&db, "rocks/rock.png"), (before, false));
        let reopened = open(&dir);
        let entry = reopened.find("rocks/rock.png").unwrap();
        assert_eq!((entry.id, entry.settings), (rock, settings));

        let tree = db.find("models/tree.gltf").unwrap().id;
        let door = db.find("props/door.fprefab").unwrap().id;
        let mesh = |scale| Settings::Mesh(MeshSettings { scale });
        for (id, settings, named) in [
            (rock, mesh(2.0), "rocks/rock.png"),
            (tree, mesh(-1.0), "models/tree.gltf"),
            (tree, mesh(f32::NAN), "models/tree.gltf"),
            (door, mesh(1.0), "props/door.fprefab"),
        ] {
            let error = db.set_settings(id, settings).unwrap_err();
            assert!(error.contains(named), "{error}");
        }
        assert_eq!(db.get(tree).unwrap().settings, mesh(1.0), "left as it was");
    }

    /// Reimport goes past the cache: the same artifact, imported rather than
    /// found.
    #[test]
    fn a_reimport_imports_again_rather_than_reading_the_cache() {
        let dir = Dir::new("db-reimport");
        project(&dir);
        let first = open(&dir);
        assert!(first.wait_idle(WAIT));
        drop(first);
        let mut db = open(&dir);
        assert!(db.wait_idle(WAIT));
        db.update(&World::new(), &mut HeadlessBackend::new());
        let (key, cached) = imported_key(&db, "rocks/rock.png");
        assert!(cached);

        let rock = db.find("rocks/rock.png").unwrap().id;
        db.reimport(rock).unwrap();
        assert_eq!(db.get(rock).unwrap().status, Status::Queued);
        assert!(db.wait_idle(WAIT));
        db.update(&World::new(), &mut HeadlessBackend::new());
        assert_eq!(imported_key(&db, "rocks/rock.png"), (key, false));
        let door = db.find("props/door.fprefab").unwrap().id;
        assert!(
            db.reimport(door)
                .unwrap_err()
                .contains("props/door.fprefab")
        );
    }

    /// A cache written before thumbnails existed holds artifacts and no
    /// pictures: such an artifact is imported once more to make one, rather
    /// than leaving the browser a glyph for as long as the cache lasts.
    #[test]
    fn an_artifact_cached_without_a_thumbnail_is_imported_again_for_one() {
        let dir = Dir::new("db-thumbless-cache");
        project(&dir);
        let first = open(&dir);
        assert!(first.wait_idle(WAIT));
        drop(first);
        for file in std::fs::read_dir(dir.path().join(".cache")).unwrap() {
            let path = file.unwrap().path();
            if path
                .extension()
                .is_some_and(|extension| extension == "thumb")
            {
                std::fs::remove_file(path).unwrap();
            }
        }

        let mut db = open(&dir);
        assert!(db.wait_idle(WAIT));
        db.update(&World::new(), &mut HeadlessBackend::new());
        for path in ["rocks/rock.png", "models/tree.gltf"] {
            let id = db.find(path).unwrap().id;
            assert!(db.thumbnail(id).is_some(), "{path}");
            assert!(!imported_key(&db, path).1, "{path} imported, not found");
        }
    }

    /// Every import leaves the browser a thumbnail and the Import page a
    /// summary, and a second run that finds the artifacts cached finds both
    /// beside them rather than importing again.
    #[test]
    fn an_import_leaves_a_thumbnail_and_a_summary_that_a_cached_run_finds() {
        let dir = Dir::new("db-thumbnails");
        project(&dir);
        for run in ["first", "cached"] {
            let mut world = World::new();
            world.insert_resource(open(&dir));
            assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
            update(&world, &mut HeadlessBackend::new());
            let db = world.resource::<AssetDatabase>();
            for (path, layout) in [("rocks/rock.png", "rgba"), ("models/tree.gltf", "grey")] {
                let entry = db.find(path).unwrap();
                assert_eq!(
                    entry.status,
                    Status::Imported {
                        key: db.thumbnail(entry.id).expect(path).0,
                        cached: run == "cached",
                    },
                    "{run}: {path}"
                );
                let (_, picture) = db.thumbnail(entry.id).unwrap();
                let rgba = matches!(picture.pixels, crate::gfx::thumbnail::Pixels::Rgba(_));
                assert_eq!(rgba, layout == "rgba", "{run}: {path}");
                assert!(entry.summary.is_some(), "{run}: {path}");
            }
            let door = db.find("props/door.fprefab").unwrap().id;
            assert!(db.thumbnail(door).is_none(), "a prefab has no artifact");
        }
    }

    /// A device that samples BC is given each texture's compressed chain; one
    /// that does not is given the same chain in RGBA8.
    #[test]
    fn a_texture_reaches_the_backend_in_the_format_the_device_samples() {
        for (backend, format) in [
            (HeadlessBackend::with_bc(), TexelFormat::Bc7Srgb),
            (HeadlessBackend::new(), TexelFormat::Rgba8Srgb),
        ] {
            let dir = Dir::new(&format!("db-target-{}", format.label()));
            material_project(&dir, &brick(0.5, crate::gfx::BlendMode::Opaque));
            let (db, issues) = AssetDatabase::open_for(
                dir.path(),
                dir.path().join(".cache"),
                Target::of(&backend),
            );
            assert!(issues.is_empty(), "{issues:?}");
            let id = db.find("brick.fmat").unwrap().id;
            let (world, _) = drawing_material(db, id);
            let mut backend = backend;

            update(&world, &mut backend);
            assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
            update(&world, &mut backend);
            assert_eq!(
                backend.last_texture(),
                Some((format, 2)),
                "a 2x2 texture's chain, in {format:?}"
            );
        }
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

    const ROCK: &str = "5f0c3a4e-0000-4000-8000-0000000000a1";
    const ROCK_NORMAL: &str = "5f0c3a4e-0000-4000-8000-0000000000a2";

    fn rock() -> AssetId {
        ROCK.parse().unwrap()
    }

    /// `brick.fmat` at `roughness` and `blend`, with the rock as its albedo.
    fn brick(roughness: f32, blend: crate::gfx::BlendMode) -> material::MaterialFile {
        material::MaterialFile {
            material: crate::gfx::Material {
                roughness,
                blend,
                ..Default::default()
            },
            textures: material::MaterialTextures {
                albedo: Some(rock()),
                ..Default::default()
            },
        }
    }

    /// [`project`] with the rock under a known id, so a material can name it,
    /// and `brick.fmat` holding `file`.
    fn material_project(dir: &Dir, file: &material::MaterialFile) {
        project(dir);
        dir.write(
            "rocks/rock.png.meta",
            format!("id = \"{ROCK}\"\n").as_bytes(),
        );
        dir.write("brick.fmat", material::write(file).as_bytes());
    }

    /// A world that draws material `id` on one entity, holding the pending
    /// handle a scene load would leave.
    fn drawing_material(db: AssetDatabase, id: AssetId) -> (World, orrin_ecs::Entity) {
        let mut world = World::new();
        world.insert_resource(MeshBounds::default());
        world.insert_resource(AssetPreviews::default());
        world.insert_resource(MaterialBlends::default());
        world.insert_resource(db);
        let handle = world.resource_mut::<AssetDatabase>().material(id).unwrap();
        assert_ne!(handle.0 & PENDING, 0);
        let entity = world.spawn();
        world.insert(entity, handle);
        (world, entity)
    }

    fn material_handle(world: &World, entity: orrin_ecs::Entity) -> MaterialHandle {
        *world.get::<MaterialHandle>(entity).unwrap()
    }

    /// The brick, bound and drawn.
    fn loaded_brick(dir: &Dir) -> (World, MaterialHandle, HeadlessBackend) {
        use crate::gfx::BlendMode;
        material_project(dir, &brick(0.25, BlendMode::Masked));
        let db = open(dir);
        let id = db.find("brick.fmat").unwrap().id;
        let (world, entity) = drawing_material(db, id);
        let mut backend = HeadlessBackend::new();
        assert_eq!(update(&world, &mut backend), Vec::<String>::new());
        let bound = material_handle(&world, entity);
        (world, bound, backend)
    }

    #[test]
    fn a_material_file_is_indexed_as_a_material_and_a_broken_one_fails_by_name() {
        let dir = Dir::new("db-fmat-index");
        dir.write(
            "brick.fmat",
            material::write(&Default::default()).as_bytes(),
        );
        dir.write(
            "bad.fmat",
            b"orrin-material 1 min-reader 1\n\nrougness = 0.5\n",
        );
        let mut db = open(&dir);

        let brick = db.find("brick.fmat").unwrap();
        assert_eq!(brick.kind, AssetKind::Material);
        assert_eq!(brick.status, Status::Indexed);
        assert_eq!(db.state(brick.id), LoadState::Ready);
        assert!(dir.path().join("brick.fmat.meta").is_file());

        let message = "bad.fmat: `rougness` is not a material parameter";
        let bad = db.find("bad.fmat").unwrap();
        assert_eq!(bad.status, Status::Failed(message.to_owned()));
        assert_eq!(db.state(bad.id), LoadState::Failed);
        assert_eq!(
            db.update(&World::new(), &mut HeadlessBackend::new()),
            [message]
        );
    }

    #[test]
    fn a_material_binds_to_its_file_with_its_textures_in_their_slots() {
        let dir = Dir::new("db-fmat-bind");
        let (world, bound, mut backend) = loaded_brick(&dir);
        assert_eq!(bound.0 & PENDING, 0, "rewritten to a real slot");

        let db = world.resource::<AssetDatabase>();
        let id = db.find("brick.fmat").unwrap().id;
        assert_eq!(db.material_id(bound), Some(id));
        assert_eq!(db.state(id), LoadState::Ready);
        let albedo = db.texture(rock()).expect("the albedo is loaded with it");
        drop(db);

        let loaded = backend.material(bound).unwrap();
        assert_eq!(loaded.roughness, 0.25);
        assert_eq!(loaded.blend, crate::gfx::BlendMode::Masked);
        assert_eq!(loaded.albedo_texture, Some(albedo));
        assert_eq!(loaded.normal_texture, None);
        assert_eq!(
            world.resource::<MaterialBlends>().get(bound),
            crate::gfx::BlendMode::Masked
        );
        let preview = world.resource::<AssetPreviews>();
        assert_eq!(preview.material(bound).unwrap().material.roughness, 0.25);
        drop(preview);

        assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
        update(&world, &mut backend);
        assert_eq!(
            world.resource::<AssetDatabase>().state(rock()),
            LoadState::Ready
        );
        assert_eq!(backend.texture_replacements(), 1);
    }

    #[test]
    fn a_saved_material_is_reread_into_the_same_slot() {
        use crate::gfx::BlendMode;
        let dir = Dir::new("db-fmat-reload");
        let (world, bound, mut backend) = loaded_brick(&dir);
        let uploads = backend.upload_counts();

        let saved = material::write(&brick(0.75, BlendMode::Blend));
        change(&world, &dir, "brick.fmat", saved.as_bytes(), &mut backend);

        let reread = backend.material(bound).unwrap();
        assert_eq!(reread.roughness, 0.75);
        assert_eq!(reread.blend, BlendMode::Blend);
        assert_eq!(backend.material_updates(), 1);
        assert_eq!(
            backend.upload_counts(),
            uploads,
            "the same slot, not a new one"
        );
        assert_eq!(
            world.resource::<MaterialBlends>().get(bound),
            BlendMode::Blend,
            "a blend change moves the material to the blended queue"
        );
        let previews = world.resource::<AssetPreviews>();
        assert_eq!(previews.material(bound).unwrap().material.roughness, 0.75);
    }

    #[test]
    fn a_broken_material_save_keeps_drawing_the_last_good_version() {
        let dir = Dir::new("db-fmat-broken");
        let (world, bound, mut backend) = loaded_brick(&dir);

        let path = dir.write(
            "brick.fmat",
            b"orrin-material 1 min-reader 1\n\nroughness = \"low\"\n",
        );
        let mut db = world.resource_mut::<AssetDatabase>();
        db.notice(path);
        db.apply_changes();
        drop(db);
        assert_eq!(
            update(&world, &mut backend),
            [
                "brick.fmat: field `roughness`: expected f32, found string. Still drawing the \
                 last good version"
            ]
        );
        assert_eq!(backend.material(bound).unwrap().roughness, 0.25);
        assert_eq!(backend.material_updates(), 0);
        assert!(
            update(&world, &mut backend).is_empty(),
            "named once, not every frame"
        );

        let fixed = material::write(&brick(0.5, crate::gfx::BlendMode::Masked));
        change(&world, &dir, "brick.fmat", fixed.as_bytes(), &mut backend);
        assert_eq!(backend.material(bound).unwrap().roughness, 0.5);
    }

    #[test]
    fn a_texture_given_as_a_material_is_refused_by_path() {
        let dir = Dir::new("db-fmat-kind");
        project(&dir);
        let mut db = open(&dir);
        let rock = db.find("rocks/rock.png").unwrap().id;
        assert_eq!(
            db.material(rock).unwrap_err(),
            "`rocks/rock.png`, which is a texture"
        );
    }

    #[test]
    fn a_slot_naming_something_other_than_a_texture_is_named() {
        let dir = Dir::new("db-fmat-slot");
        project(&dir);
        let tree: AssetId = "5f0c3a4e-0000-4000-8000-0000000000a3".parse().unwrap();
        dir.write(
            "models/tree.gltf.meta",
            format!("id = \"{tree}\"\n").as_bytes(),
        );
        let file = material::MaterialFile {
            textures: material::MaterialTextures {
                emissive: Some(tree),
                ..Default::default()
            },
            ..Default::default()
        };
        dir.write("brick.fmat", material::write(&file).as_bytes());
        let db = open(&dir);
        let id = db.find("brick.fmat").unwrap().id;
        let (world, _) = drawing_material(db, id);
        assert_eq!(
            update(&world, &mut HeadlessBackend::new()),
            [
                "brick.fmat: the `emissive` slot names `models/tree.gltf`, which is a mesh, \
                 so it draws pink. Assign a texture"
            ]
        );
    }

    /// White, or pink, read as a normal is one tilted far off the surface; a
    /// normal map that has not arrived yet must leave the surface flat.
    #[test]
    fn a_normal_map_loads_flat_while_it_imports() {
        let dir = Dir::new("db-fmat-normal");
        project(&dir);
        dir.write(
            "rocks/rock.png.meta",
            format!("id = \"{ROCK}\"\n").as_bytes(),
        );
        dir.write(
            "rocks/rock_normal.png",
            &png(2, 2, |_, _| [128, 128, 255, 255]),
        );
        dir.write(
            "rocks/rock_normal.png.meta",
            format!("id = \"{ROCK_NORMAL}\"\n\n[import]\nusage = \"normal\"\n").as_bytes(),
        );
        let normal: AssetId = ROCK_NORMAL.parse().unwrap();
        let missing: AssetId = "5f0c3a4e-0000-4000-8000-0000000000ff".parse().unwrap();
        let file = material::MaterialFile {
            textures: material::MaterialTextures {
                albedo: Some(rock()),
                normal: Some(normal),
                clearcoat_normal: Some(missing),
                ..Default::default()
            },
            ..Default::default()
        };
        dir.write("brick.fmat", material::write(&file).as_bytes());
        let db = open(&dir);
        let id = db.find("brick.fmat").unwrap().id;
        let (world, _) = drawing_material(db, id);
        let mut backend = HeadlessBackend::new();
        assert_eq!(
            update(&world, &mut backend),
            [format!(
                "texture asset {missing} is not in the project, so it draws flat"
            )]
        );

        let db = world.resource::<AssetDatabase>();
        let texel = |id| backend.texel(db.texture(id).unwrap());
        const FLAT: [u8; 4] = [128, 128, 255, 255];
        assert_eq!(texel(rock()), Some(([255; 4], true)));
        assert_eq!(texel(normal), Some((FLAT, false)), "flat, and linear");
        assert_eq!(texel(missing), Some((FLAT, false)));
    }

    #[test]
    fn a_normal_map_that_fails_to_import_stays_flat() {
        let dir = Dir::new("db-fmat-normal-bad");
        project(&dir);
        dir.write("rocks/bad_normal.png", b"not a png");
        dir.write(
            "rocks/bad_normal.png.meta",
            format!(
                "id = \"{ROCK_NORMAL}\"
"
            )
            .as_bytes(),
        );
        let normal: AssetId = ROCK_NORMAL.parse().unwrap();
        let file = material::MaterialFile {
            textures: material::MaterialTextures {
                normal: Some(normal),
                ..Default::default()
            },
            ..Default::default()
        };
        dir.write("brick.fmat", material::write(&file).as_bytes());
        let db = open(&dir);
        let id = db.find("brick.fmat").unwrap().id;
        let (world, _) = drawing_material(db, id);
        let mut backend = HeadlessBackend::new();
        update(&world, &mut backend);
        assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
        update(&world, &mut backend);

        let db = world.resource::<AssetDatabase>();
        assert_eq!(db.state(normal), LoadState::Failed);
        assert_eq!(
            backend.texel(db.texture(normal).unwrap()),
            Some(([128, 128, 255, 255], false))
        );
    }

    /// The editor's save reaches the slot once. The watcher then hears the
    /// same write, and reading it back would upload the material a second
    /// time; a change made outside the editor still does.
    #[test]
    fn the_editors_own_save_is_not_read_back_as_a_change() {
        use crate::gfx::BlendMode;
        let dir = Dir::new("db-fmat-echo");
        let (world, bound, mut backend) = loaded_brick(&dir);
        let id = world
            .resource::<AssetDatabase>()
            .find("brick.fmat")
            .unwrap()
            .id;

        let saved = material::write(&brick(0.75, BlendMode::Masked));
        world
            .resource_mut::<AssetDatabase>()
            .write_asset(id, &saved)
            .unwrap();
        assert_eq!(update(&world, &mut backend), Vec::<String>::new());
        assert_eq!(backend.material(bound).unwrap().roughness, 0.75);
        assert_eq!(backend.material_updates(), 1);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("brick.fmat")).unwrap(),
            saved
        );

        let echo = dir.path().join("brick.fmat");
        let mut db = world.resource_mut::<AssetDatabase>();
        db.notice(echo);
        db.apply_changes();
        drop(db);
        assert_eq!(update(&world, &mut backend), Vec::<String>::new());
        assert_eq!(backend.material_updates(), 1, "the echo is nothing new");

        let elsewhere = material::write(&brick(0.5, BlendMode::Masked));
        change(
            &world,
            &dir,
            "brick.fmat",
            elsewhere.as_bytes(),
            &mut backend,
        );
        assert_eq!(backend.material(bound).unwrap().roughness, 0.5);
        assert_eq!(backend.material_updates(), 2);
    }

    /// A slot takes a texture, and anything else is refused by its path
    /// before it reaches the file.
    #[test]
    fn a_material_slot_refuses_what_is_not_a_texture_by_path() {
        let dir = Dir::new("db-fmat-check");
        let (world, _, _) = loaded_brick(&dir);
        let db = world.resource::<AssetDatabase>();
        let id = db.find("brick.fmat").unwrap().id;
        let tree = db.find("models/tree.gltf").unwrap().id;
        let mut file = *db.material_file(id).unwrap();

        file.textures.normal = Some(rock());
        assert_eq!(db.check_material(id, &file), Ok(()));
        file.textures.emissive = Some(tree);
        assert_eq!(
            db.check_material(id, &file),
            Err(
                "brick.fmat: the `emissive` slot takes a texture, not `models/tree.gltf`, which \
                 is a mesh"
                    .to_owned()
            )
        );
    }

    #[test]
    fn a_text_that_does_not_read_is_never_written() {
        let dir = Dir::new("db-fmat-unwritable");
        let (world, _, _) = loaded_brick(&dir);
        let id = world
            .resource::<AssetDatabase>()
            .find("brick.fmat")
            .unwrap()
            .id;
        let before = std::fs::read_to_string(dir.path().join("brick.fmat")).unwrap();

        let refused = world
            .resource_mut::<AssetDatabase>()
            .write_asset(id, "orrin-material 1 min-reader 1\n\nrougness = 0.5\n");
        assert_eq!(
            refused,
            Err("brick.fmat: `rougness` is not a material parameter".to_owned())
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("brick.fmat")).unwrap(),
            before
        );
    }

    /// "New material" in the browser: a default material beside its
    /// sidecar, indexed at once, and drawn once something asks for it.
    #[test]
    fn a_new_material_is_written_at_the_defaults_and_binds() {
        let dir = Dir::new("db-fmat-new");
        project(&dir);
        let mut db = open(&dir);

        let id = db.create_material("rocks").unwrap();
        let entry = db.find("rocks/New Material.fmat").expect("indexed at once");
        assert_eq!((entry.id, entry.kind), (id, AssetKind::Material));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("rocks/New Material.fmat")).unwrap(),
            material::write(&Default::default())
        );
        assert!(dir.path().join("rocks/New Material.fmat.meta").is_file());
        let again = db.create_material("rocks").unwrap();
        assert_eq!(db.get(again).unwrap().path, "rocks/New Material 2.fmat");
        assert_eq!(
            db.create_material("")
                .map(|id| db.get(id).unwrap().path.clone()),
            Ok("New Material.fmat".to_owned())
        );

        let (world, entity) = drawing_material(db, id);
        assert_eq!(
            update(&world, &mut HeadlessBackend::new()),
            Vec::<String>::new()
        );
        let bound = material_handle(&world, entity);
        assert_eq!(bound.0 & PENDING, 0);
        assert_eq!(
            world.resource::<AssetDatabase>().state(id),
            LoadState::Ready
        );
    }
}
