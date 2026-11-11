//! `.oshader`: a surface shader as an asset (see `gfx::surface` for the
//! language and the artifact).
//!
//! **Import is a compile.** The worker parses and compiles the file into a
//! [`SurfaceArtifact`] and caches it like any artifact. Its key hashes the
//! source and [`key_settings`]: the engine's
//! [`SOURCES_HASH`](crate::gfx::shaders::SOURCES_HASH) and the
//! artifact's format version, so an engine whose own shaders changed
//! recompiles every surface shader rather than loading one stitched into
//! sources it no longer has. A cache hit is still checked with
//! [`SurfaceArtifact::matches_this_build`] and is a miss when it fails. A build
//! without the `shader-compiler` feature — an exported game — reads the cache
//! and the pack and never compiles: a shader with no artifact fails by name.
//!
//! **A shader's artifact always comes back from the worker**, unlike a texture
//! nothing draws, because its parameter table is what the material inspector
//! and the shader editor show whether or not anything draws it. A shader is a
//! few hundred kilobytes and a project has few of them; the artifact itself is
//! dropped once the table is read, unless something draws the shader.
//!
//! **One handle for the session.** A backend cannot unload a surface shader,
//! and holds at most `MAX_SURFACE_SHADERS`, so a shader is loaded the first
//! time a drawn material names it and every later import replaces it in that
//! handle. A shader nothing draws takes no handle.
//!
//! **The last good version keeps drawing.** A compile error or a refused
//! replace leaves the backend holding what it had, and the errors — each at
//! its line — are kept for the shader editor and logged as `path:line:
//! message`. Until a shader has compiled once, the materials naming it draw as
//! standard materials.
//!
//! **Materials attach as their inputs arrive.** [`AssetDatabase::attach_surface`]
//! gives a drawn material's shader and values to the backend whenever the
//! material is uploaded or reread and whenever its shader first loads, and
//! remembers what it gave, so nothing unchanged is sent again and a value the
//! shader ignores is warned about once per change. A texture value is resolved
//! to the texture's slot, which never changes meaning, so a texture finishing
//! its import later needs nothing attached again. A replaced shader keeps its
//! materials: the backend repacks their values by name itself.

use std::path::PathBuf;

use orrin_registry::AssetId;

use super::cache::{Artifact, Cache, CacheKey, Importer};
use super::material::ShaderValue;
use super::worker::{self, ImportError, Imported, Job, Payload};
use super::{AssetDatabase, AssetKind, PENDING, PINK, Status};
use crate::gfx::RenderBackend;
use crate::gfx::surface::{
    DefaultTexture, Ignored, ParamKind, ParamTable, ParamValue, SurfaceArtifact, SurfaceError,
    SurfaceShaderHandle,
};
use crate::scene::MaterialHandle;

pub const EXTENSION: &str = "oshader";

pub const IMPORTER: Importer = Importer {
    name: "surface-shader",
    version: 1,
};

/// What "New surface shader" writes: a working shader with one colour
/// parameter, commented for someone who has not written one before.
pub const NEW_SHADER: &str = "\
// A surface shader decides what a material's surface is, and the engine
// lights it. `o` arrives holding the material's own reading of the surface,
// so change only what this shader is for.
//
// Each `param` line is a value every material drawn through this shader can
// set in the inspector; a material that sets none gets the default.

param color tint = (1.0, 0.5, 0.2, 1.0);

void surface(in SurfaceIn i, inout SurfaceOut o) {
    o.albedo *= tint.rgb;
}
";

/// What a shader's cache key hashes beside its source: the engine shader
/// sources it is stitched into, and the layout of the artifact it becomes.
pub fn key_settings(sources_hash: u64, artifact_version: u32) -> String {
    format!("sources={sources_hash:016x};artifact={artifact_version}")
}

/// Turns a `.oshader`'s text into its artifact; the second argument names the
/// file in errors.
pub type Compiler = fn(&str, &str) -> Result<SurfaceArtifact, Vec<SurfaceError>>;

/// This build's compiler, if it has one.
#[cfg(feature = "shader-compiler")]
pub const COMPILER: Option<Compiler> = Some(compile);
#[cfg(not(feature = "shader-compiler"))]
pub const COMPILER: Option<Compiler> = None;

#[cfg(feature = "shader-compiler")]
fn compile(text: &str, rel: &str) -> Result<SurfaceArtifact, Vec<SurfaceError>> {
    let shader = crate::gfx::surface::SurfaceShader::parse(text)?;
    crate::gfx::surface::compile(&shader, rel)
}

/// The worker's run for a shader whose source is `source` and key `key`: the
/// cached artifact when there is one for this build, else a compile, else —
/// with no `compiler` — an error saying where to compile it.
pub(super) fn import(
    job: &Job,
    cache: &Cache,
    key: CacheKey,
    source: &[u8],
    compiler: Option<Compiler>,
) -> Result<Imported, ImportError> {
    let imported = |cached, payload| Imported {
        key,
        cached,
        inputs: Vec::new(),
        payload,
        thumbnail: None,
        summary: None,
    };
    if job.known == Some(key) && !job.fresh {
        return Ok(imported(true, None));
    }
    if !job.fresh
        && let Some(Ok(Artifact::Shader(artifact))) = cache.load(key)
        && artifact.matches_this_build()
    {
        return Ok(imported(true, Some(Payload::Shader(artifact))));
    }
    let Some(compile) = compiler else {
        return Err(ImportError::from(format!(
            "{} has no compiled artifact in the cache, and this build of the engine has no \
             shader compiler. Open the project in the editor, which compiles it",
            job.rel
        )));
    };
    let text = std::str::from_utf8(source)
        .map_err(|_| format!("{} is not UTF-8 text, so it is not a shader", job.rel))?;
    let artifact = compile(text, &job.rel).map_err(|errors| located(&job.rel, errors))?;
    let artifact = Artifact::Shader(artifact);
    if let Err(error) = cache.store(key, &artifact) {
        // Compiled fine and only the write failed: the shader still loads,
        // and the next run compiles it again.
        tracing::warn!("{}: {error}", job.rel);
    }
    Ok(imported(false, Some(worker::payload(artifact))))
}

fn located(rel: &str, mut errors: Vec<SurfaceError>) -> ImportError {
    if errors.is_empty() {
        errors.push(SurfaceError {
            line: None,
            message: "did not compile".to_owned(),
        });
    }
    let message = errors
        .iter()
        .map(|error| error.located(rel))
        .collect::<Vec<_>>()
        .join("\n");
    ImportError {
        message,
        lines: errors,
    }
}

/// What the database knows of one shader asset.
#[derive(Default)]
pub(super) struct Shader {
    /// The backend's, from the first load for the rest of the session.
    handle: Option<SurfaceShaderHandle>,
    /// The artifact the handle holds.
    key: Option<CacheKey>,
    /// The parameters of the newest artifact that compiled, or of the one
    /// still drawn when a newer one was refused.
    params: Option<ParamTable>,
    /// The latest import's, each at its line; empty when it went through.
    errors: Vec<SurfaceError>,
    /// A drawn material names it, so it is loaded into the backend.
    wanted: bool,
    /// The text the editor last saved and queued, until the file is heard
    /// holding something else.
    written: Option<String>,
}

impl Shader {
    pub(super) fn wanted(&self) -> bool {
        self.wanted
    }

    pub(super) fn key(&self) -> Option<CacheKey> {
        self.key
    }

    pub(super) fn is_loaded(&self) -> bool {
        self.handle.is_some()
    }
}

/// What a drawn material was last given: its shader and its values with each
/// texture resolved to a slot.
pub(super) type Attached = (SurfaceShaderHandle, Vec<(String, ParamValue)>);

impl AssetDatabase {
    /// The errors of shader `id`'s latest import, each at its line where it has
    /// one; empty once it compiles.
    pub fn shader_errors(&self, id: AssetId) -> &[SurfaceError] {
        self.shaders
            .get(&id)
            .map_or(&[], |shader| shader.errors.as_slice())
    }

    /// Shader `id`'s parameters, from the last version that compiled.
    pub fn shader_params(&self, id: AssetId) -> Option<&ParamTable> {
        self.shaders.get(&id)?.params.as_ref()
    }

    /// Where shader `id`'s source is.
    pub fn shader_file(&self, id: AssetId) -> Result<PathBuf, String> {
        let entry = self
            .get(id)
            .ok_or_else(|| format!("surface shader asset {id} is not in the project"))?;
        if entry.kind != AssetKind::Shader {
            return Err(format!(
                "`{}` is a {}, not a surface shader",
                entry.path,
                entry.kind.label()
            ));
        }
        Ok(self.dir.join(&entry.path))
    }

    /// Shader `id`'s source as it is on disk now.
    pub fn shader_source(&self, id: AssetId) -> Result<String, String> {
        let path = self.shader_file(id)?;
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", self.path_of(id)))
    }

    /// Shader `id`'s handle, loading it from the next [`update`](Self::update)
    /// if it has none yet: `None` until it has compiled and loaded.
    pub fn surface_shader(&mut self, id: AssetId) -> Result<Option<SurfaceShaderHandle>, String> {
        self.shader_file(id)?;
        Ok(self.want_shader(id))
    }

    /// Write [`NEW_SHADER`] into `folder` as `New Surface Shader`, numbered
    /// past any already there, with its sidecar, and index it.
    pub fn create_shader(&mut self, folder: &str) -> Result<AssetId, String> {
        self.create_file(
            folder,
            "New Surface Shader",
            EXTENSION,
            NEW_SHADER,
            AssetKind::Shader,
        )
    }

    /// Write `text` as shader `id`'s file, through a temporary file and a
    /// rename, and queue its import at once. Unlike a material, a text that
    /// does not compile is still written: its errors are what the shader
    /// editor shows. What is written is remembered, so the watcher hearing it
    /// does not import it a second time.
    pub fn write_shader(&mut self, id: AssetId, text: &str) -> Result<(), String> {
        let path = self.shader_file(id)?;
        let rel = self.path_of(id);
        super::write_atomically(&path, text)
            .map_err(|error| format!("could not write {rel}: {error}"))?;
        let metadata = std::fs::metadata(&path).ok();
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) {
            entry.bytes = metadata.as_ref().map_or(text.len() as u64, |m| m.len());
            entry.modified = metadata.and_then(|m| m.modified().ok());
        }
        self.queue(id);
        self.shaders.entry(id).or_default().written = Some(text.to_owned());
        Ok(())
    }

    /// Whether a change heard at shader `id`'s `path` is the editor's own last
    /// save rather than something new. Anything else forgets that save, so a
    /// later save that puts its text back is heard.
    pub(super) fn is_own_save(&mut self, id: AssetId, path: &std::path::Path) -> bool {
        let Some(shader) = self.shaders.get_mut(&id) else {
            return false;
        };
        let Some(written) = &shader.written else {
            return false;
        };
        let same = std::fs::read_to_string(path).is_ok_and(|text| text == *written);
        if !same {
            shader.written = None;
        }
        same
    }

    /// A material at the defaults drawn through shader `id` with none of its
    /// parameters set, for the shader editor to preview a shader no material
    /// uses yet. It has no file and is never listed; the handle is the same
    /// one for the session, pending until the next update binds it.
    pub fn shader_preview_material(&mut self, id: AssetId) -> Result<MaterialHandle, String> {
        self.shader_file(id)?;
        let material = AssetId::named("shader-preview", &id.to_string());
        self.material_files
            .entry(material)
            .or_insert_with(|| super::material::MaterialFile {
                shader: Some(id),
                ..Default::default()
            });
        Ok(MaterialHandle(self.materials.promise(material)))
    }

    /// Mark shader `id` as drawn and its handle, queueing its artifact when
    /// it has compiled and is not loaded yet. A shader that failed is not
    /// queued again: its next save is what imports it.
    fn want_shader(&mut self, id: AssetId) -> Option<SurfaceShaderHandle> {
        let shader = self.shaders.entry(id).or_default();
        shader.wanted = true;
        if let Some(handle) = shader.handle {
            return Some(handle);
        }
        let imported = matches!(
            self.get(id).map(|entry| &entry.status),
            Some(Status::Imported { .. })
        );
        if imported && !self.awaiting.contains(&id) {
            self.queue(id);
        }
        None
    }

    /// Take shader `id`'s finished import.
    pub(super) fn finish_shader(
        &mut self,
        done: worker::Done,
        backend: &mut impl RenderBackend,
        errors: &mut Vec<String>,
    ) {
        let id = done.job.id;
        if done.job.payload {
            self.awaiting.remove(&id);
        }
        let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) else {
            return;
        };
        let rel = entry.path.clone();
        let shader = self.shaders.entry(id).or_default();
        match done.outcome {
            Ok(imported) => {
                entry.status = Status::Imported {
                    key: imported.key,
                    cached: imported.cached,
                };
                entry.summary = imported.summary;
                shader.errors.clear();
                match imported.payload {
                    Some(Payload::Shader(artifact)) if shader.wanted => {
                        self.load_shader(id, imported.key, artifact, backend, errors);
                    }
                    Some(Payload::Shader(artifact)) => shader.params = Some(artifact.params),
                    _ => {}
                }
            }
            Err(failure) => {
                entry.status = Status::Failed(failure.message.clone());
                errors.extend(failure.message.lines().map(str::to_owned));
                if shader.handle.is_some() {
                    errors.push(format!(
                        "{rel} did not compile, so the last version that did is still drawn"
                    ));
                }
                shader.errors = match failure.lines.is_empty() {
                    true => vec![SurfaceError {
                        line: None,
                        message: failure.message,
                    }],
                    false => failure.lines,
                };
            }
        }
    }

    /// Give `artifact` to the backend: into a new handle the first time, and
    /// in place of what the handle holds after that.
    fn load_shader(
        &mut self,
        id: AssetId,
        key: CacheKey,
        artifact: SurfaceArtifact,
        backend: &mut impl RenderBackend,
        errors: &mut Vec<String>,
    ) {
        let rel = self.path_of(id);
        let shader = self.shaders.entry(id).or_default();
        let refused = match shader.handle {
            Some(_) if shader.key == Some(key) => return,
            Some(handle) => match backend.replace_surface_shader(handle, &artifact) {
                Ok(ignored) => {
                    shader.key = Some(key);
                    shader.params = Some(artifact.params);
                    for (material, ignored) in ignored {
                        self.warn_ignored(material, id, &ignored);
                    }
                    return;
                }
                Err(error) => {
                    format!("{rel}: {error}. The last version that loaded is still drawn")
                }
            },
            None => match backend.load_surface_shader(&artifact) {
                Ok(handle) => {
                    shader.handle = Some(handle);
                    shader.key = Some(key);
                    shader.params = Some(artifact.params);
                    let naming: Vec<AssetId> = self
                        .material_files
                        .iter()
                        .filter(|(_, file)| file.shader == Some(id))
                        .map(|(material, _)| *material)
                        .collect();
                    for material in naming {
                        self.attach_surface(material, backend, errors);
                    }
                    return;
                }
                Err(error) => {
                    shader.params = Some(artifact.params);
                    format!("{rel}: {error}, so the materials drawn through it draw without it")
                }
            },
        };
        shader.errors = vec![SurfaceError {
            line: None,
            message: refused.clone(),
        }];
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) {
            entry.status = Status::Failed(refused.clone());
        }
        errors.push(refused);
    }

    /// Give drawn material `id`'s shader and values to the backend, or detach
    /// it, when that differs from what it was last given.
    pub(super) fn attach_surface(
        &mut self,
        id: AssetId,
        backend: &mut impl RenderBackend,
        errors: &mut Vec<String>,
    ) {
        let Some(slot) = self.materials.handle(id).filter(|h| h & PENDING == 0) else {
            return;
        };
        let Some(file) = self.material_files.get(&id) else {
            return;
        };
        let (shader, values) = (file.shader, file.shader_values.clone());
        let wanted = shader.and_then(|shader| {
            let handle = self.shader_for(id, shader, errors)?;
            Some((shader, handle))
        });
        let desired: Option<Attached> = wanted.map(|(shader, handle)| {
            let values = self.surface_values(id, shader, &values, backend, errors);
            (handle, values)
        });
        if self.attached_surfaces.get(&id) == desired.as_ref() {
            return;
        }
        let given = desired
            .as_ref()
            .map(|(handle, values)| (*handle, values.as_slice()));
        match backend.set_material_surface(MaterialHandle(slot), given) {
            Ok(ignored) => {
                if let (Some(shader), Some(_)) = (shader, &desired) {
                    for ignored in &ignored {
                        self.warn_ignored(MaterialHandle(slot), shader, ignored);
                    }
                }
                match desired {
                    Some(desired) => self.attached_surfaces.insert(id, desired),
                    None => self.attached_surfaces.remove(&id),
                };
            }
            Err(error) => errors.push(format!(
                "{}: {error}, so it draws as a standard material",
                self.path_of(id)
            )),
        }
    }

    /// The handle material `id`'s `shader` is loaded in, wanting it; `None`
    /// while it loads, and — said by name — when it names no shader.
    fn shader_for(
        &mut self,
        id: AssetId,
        shader: AssetId,
        errors: &mut Vec<String>,
    ) -> Option<SurfaceShaderHandle> {
        match self.get(shader) {
            Some(entry) if entry.kind == AssetKind::Shader => self.want_shader(shader),
            Some(entry) => {
                errors.push(format!(
                    "{}: `shader` names `{}`, which is a {}, so it draws as a standard \
                     material. Assign a surface shader",
                    self.path_of(id),
                    entry.path,
                    entry.kind.label()
                ));
                None
            }
            None => {
                errors.push(format!(
                    "{}: its shader {shader} is not in the project, so it draws as a standard \
                     material. Restore the shader, or assign another",
                    self.path_of(id)
                ));
                None
            }
        }
    }

    /// Material `id`'s `values` as the backend takes them, each texture given
    /// its slot: loaded as a texture slot of the material is, flat while it
    /// imports when `shader` declares the parameter a normal map.
    fn surface_values(
        &mut self,
        id: AssetId,
        shader: AssetId,
        values: &[(String, ShaderValue)],
        backend: &mut impl RenderBackend,
        errors: &mut Vec<String>,
    ) -> Vec<(String, ParamValue)> {
        values
            .iter()
            .map(|(name, value)| {
                let value = match *value {
                    ShaderValue::Float(v) => ParamValue::Float(v),
                    ShaderValue::Vec2(v) => ParamValue::Vec2(v),
                    ShaderValue::Vec3(v) => ParamValue::Vec3(v),
                    ShaderValue::Vec4(v) => ParamValue::Vec4(v),
                    ShaderValue::Texture(texture) => ParamValue::Texture(
                        self.value_texture(id, shader, name, texture, backend, errors)
                            .0,
                    ),
                };
                (name.clone(), value)
            })
            .collect()
    }

    fn value_texture(
        &mut self,
        id: AssetId,
        shader: AssetId,
        name: &str,
        texture: AssetId,
        backend: &mut impl RenderBackend,
        errors: &mut Vec<String>,
    ) -> crate::gfx::TextureHandle {
        if let Some(entry) = self.get(texture)
            && entry.kind != AssetKind::Texture
        {
            errors.push(format!(
                "{}: shader parameter `{name}` names `{}`, which is a {}, so it draws pink. \
                 Assign a texture",
                self.path_of(id),
                entry.path,
                entry.kind.label()
            ));
            return *self
                .wrong_kind
                .get_or_insert_with(|| backend.load_texture(&PINK, 1, 1, true));
        }
        let normal = self
            .shader_params(shader)
            .and_then(|table| table.get(name))
            .is_some_and(|param| {
                matches!(
                    param.kind,
                    ParamKind::Texture {
                        default: DefaultTexture::Normal,
                        ..
                    }
                )
            });
        self.texture_slot(texture, normal, backend, errors)
    }

    fn warn_ignored(&mut self, material: MaterialHandle, shader: AssetId, ignored: &Ignored) {
        let material = self
            .materials
            .id(material.0)
            .map_or_else(|| format!("material {}", material.0), |id| self.path_of(id));
        let warning = format!(
            "{material}, drawn through {}: {ignored}",
            self.path_of(shader)
        );
        self.warnings.push(warning);
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::Dir;
    use super::super::{Settings, Target};
    use super::*;
    use crate::gfx::shaders::SOURCES_HASH;

    #[test]
    fn the_key_moves_with_the_engines_shader_sources_and_the_artifact_format() {
        let source: &[(&str, &[u8])] =
            &[("", b"void surface(in SurfaceIn i, inout SurfaceOut o) {}")];
        let key = |sources: u64, version: u32| {
            super::super::cache::key(IMPORTER, &key_settings(sources, version), source)
        };
        let here = key(SOURCES_HASH, SurfaceArtifact::FORMAT_VERSION);
        assert_ne!(here, key(SOURCES_HASH ^ 1, SurfaceArtifact::FORMAT_VERSION));
        assert_ne!(here, key(SOURCES_HASH, SurfaceArtifact::FORMAT_VERSION + 1));
        assert_eq!(
            super::super::import::key_settings(AssetKind::Shader, &Settings::None, Target::Bc),
            key_settings(SOURCES_HASH, SurfaceArtifact::FORMAT_VERSION),
            "what the worker hashes, on every target"
        );
        assert_eq!(
            super::super::import::key_settings(AssetKind::Shader, &Settings::None, Target::Rgba8),
            key_settings(SOURCES_HASH, SurfaceArtifact::FORMAT_VERSION),
        );
    }

    fn job(rel: &str) -> Job {
        Job {
            id: AssetId::new(),
            kind: AssetKind::Shader,
            source: PathBuf::from(rel),
            rel: rel.to_owned(),
            settings: Settings::None,
            target: Target::Rgba8,
            payload: false,
            known: None,
            fresh: false,
            packed: None,
        }
    }

    /// An exported game has no compiler: a shader the cache has an artifact
    /// for loads, and one it has none for fails saying where to compile it.
    #[test]
    fn without_a_compiler_a_cached_artifact_loads_and_a_missing_one_fails_by_name() {
        let dir = Dir::new("shader-no-compiler");
        let cache = Cache::new(dir.path().join("cache"));
        let job = job("fx/glow.oshader");

        let Err(missing) = import(&job, &cache, CacheKey(7), b"", None) else {
            panic!("nothing to load and nothing to compile with");
        };
        assert_eq!(
            missing.message,
            "fx/glow.oshader has no compiled artifact in the cache, and this build of the \
             engine has no shader compiler. Open the project in the editor, which compiles it"
        );

        let artifact = stand_in(SOURCES_HASH);
        cache
            .store(CacheKey(7), &Artifact::Shader(artifact.clone()))
            .unwrap();
        let loaded = import(&job, &cache, CacheKey(7), b"", None).unwrap();
        assert!(loaded.cached);
        let Some(Payload::Shader(back)) = loaded.payload else {
            panic!("the artifact comes back");
        };
        assert_eq!(back, artifact);

        // One from another build under the same key is a miss, never a load.
        cache
            .store(CacheKey(8), &Artifact::Shader(stand_in(SOURCES_HASH ^ 1)))
            .unwrap();
        assert!(import(&job, &cache, CacheKey(8), b"", None).is_err());
    }

    /// An artifact with `tint` and the built-in shaders standing in for its
    /// variants, which is all a backend that builds nothing reads.
    fn stand_in(sources_hash: u64) -> SurfaceArtifact {
        let text = "param color tint = (1, 1, 1, 1);\nvoid surface(in SurfaceIn i, inout SurfaceOut o) {}\n";
        let shader = crate::gfx::surface::SurfaceShader::parse(text).unwrap();
        SurfaceArtifact {
            sources_hash,
            params: shader.params().clone(),
            variants: crate::gfx::surface::DRAWN_VARIANTS
                .iter()
                .map(|id| crate::gfx::surface::CompiledVariant {
                    key: id.blob().name.to_string(),
                    shader: crate::gfx::shaders::CompiledShader::from(id.blob()),
                })
                .collect(),
        }
    }

    #[cfg(feature = "shader-compiler")]
    mod compiled {
        use std::time::Duration;

        use orrin_ecs::World;

        use super::super::super::LoadState;
        use super::super::super::material::{self, MaterialFile};
        use super::super::super::tests::{drawing_material, material_handle, open, update};
        use super::*;
        use crate::gfx::HeadlessBackend;

        const WAIT: Duration = Duration::from_secs(60);

        /// Run the import thread and the between-frames step until neither has
        /// anything left, returning every error logged on the way.
        fn settle(world: &World, backend: &mut HeadlessBackend) -> Vec<String> {
            let mut errors = Vec::new();
            for _ in 0..4 {
                assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
                errors.extend(update(world, backend));
            }
            errors
        }

        fn warnings(world: &World) -> Vec<String> {
            world.resource_mut::<AssetDatabase>().take_warnings()
        }

        /// Write `rel` and let the database hear it, as the watcher would.
        fn save(world: &World, dir: &Dir, rel: &str, text: &str) {
            let path = dir.write(rel, text.as_bytes());
            let mut db = world.resource_mut::<AssetDatabase>();
            db.notice(path);
            db.apply_changes();
        }

        const GLOW: &str = "5f0c3a4e-0000-4000-8000-0000000000c1";
        const ROCK: &str = "5f0c3a4e-0000-4000-8000-0000000000c2";

        const TINT: &str = "\
param color tint = (1, 1, 1, 1);
void surface(in SurfaceIn i, inout SurfaceOut o) {
    o.albedo = tint.rgb;
}
";

        /// [`TINT`] with a parameter in front of it, which moves where `tint`
        /// packs.
        const WEAR_THEN_TINT: &str = "\
param float wear = 0.5;
param color tint = (1, 1, 1, 1);
void surface(in SurfaceIn i, inout SurfaceOut o) {
    o.albedo = tint.rgb * wear;
}
";

        /// An undeclared name on line 3.
        const BROKEN: &str = "\
param color tint = (1, 1, 1, 1);
void surface(in SurfaceIn i, inout SurfaceOut o) {
    o.albedo = tint.rgb * nothing_by_this_name;
}
";

        const MASKED: &str = "\
param texture mask = white;
param float wear = 0.5;
void surface(in SurfaceIn i, inout SurfaceOut o) {
    o.albedo = sample_param(mask, i.uv).rgb * wear;
}
";

        fn glow() -> AssetId {
            GLOW.parse().unwrap()
        }

        fn rock() -> AssetId {
            ROCK.parse().unwrap()
        }

        fn glowing(values: &[(&str, ShaderValue)]) -> MaterialFile {
            let mut file = MaterialFile {
                shader: Some(glow()),
                ..MaterialFile::default()
            };
            for (name, value) in values {
                file.set_shader_value(name, Some(*value));
            }
            file
        }

        /// `fx/glow.oshader` holding `shader`, under a known id, and
        /// `brick.omat` holding `brick`.
        fn shader_project(dir: &Dir, shader: &str, brick: &MaterialFile) {
            dir.write("fx/glow.oshader", shader.as_bytes());
            dir.write(
                "fx/glow.oshader.meta",
                format!("id = \"{GLOW}\"\n").as_bytes(),
            );
            dir.write("brick.omat", material::write(brick).as_bytes());
        }

        /// The project opened and `brick.omat` drawn on one entity, after
        /// one update: bound, its shader not loaded.
        fn drawn(dir: &Dir) -> (World, MaterialHandle, HeadlessBackend) {
            let db = open(dir);
            let id = db.find("brick.omat").unwrap().id;
            let (world, entity) = drawing_material(db, id);
            let mut backend = HeadlessBackend::new();
            update(&world, &mut backend);
            let bound = material_handle(&world, entity);
            (world, bound, backend)
        }

        fn handle(world: &World) -> Option<SurfaceShaderHandle> {
            world
                .resource_mut::<AssetDatabase>()
                .surface_shader(glow())
                .unwrap()
        }

        /// `material`'s block of words, as the backend packed it.
        fn block(backend: &HeadlessBackend, material: MaterialHandle) -> Vec<[u32; 4]> {
            let buffer = backend.surface_buffer();
            let base = buffer.base_of(material).expect("attached") as usize;
            buffer.words()[base..].to_vec()
        }

        const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

        #[test]
        fn a_shader_imports_to_a_cached_artifact_that_a_reopened_project_reads_without_compiling() {
            let dir = Dir::new("shader-import");
            shader_project(&dir, TINT, &MaterialFile::default());
            let mut db = open(&dir);
            assert!(db.wait_idle(WAIT));
            assert_eq!(
                db.update(&World::new(), &mut HeadlessBackend::new()),
                Vec::<String>::new()
            );

            let entry = db.get(glow()).unwrap();
            assert_eq!(entry.kind, AssetKind::Shader);
            let Status::Imported { key, cached: false } = entry.status else {
                panic!("compiled: {:?}", entry.status);
            };
            let Some(Ok(Artifact::Shader(artifact))) = db.cache.load(key) else {
                panic!("a shader artifact under its key");
            };
            assert!(artifact.matches_this_build());
            assert!(db.shader_errors(glow()).is_empty());
            let params = db.shader_params(glow()).expect("its table");
            assert_eq!(params.params.len(), 1);
            assert_eq!(params.params[0].name, "tint");
            assert_eq!(db.shader_source(glow()).as_deref(), Ok(TINT));
            assert_eq!(
                db.shader_file(glow()),
                Ok(dir.path().join("fx/glow.oshader"))
            );
            assert_eq!(db.state(glow()), LoadState::Ready);

            let mut again = open(&dir);
            assert!(again.wait_idle(WAIT));
            again.update(&World::new(), &mut HeadlessBackend::new());
            assert_eq!(
                again.get(glow()).unwrap().status,
                Status::Imported { key, cached: true },
                "read from the cache, never compiled"
            );
            assert_eq!(again.shader_params(glow()), Some(&artifact.params));
        }

        /// Nothing draws it, so it takes no handle: there are only so many.
        #[test]
        fn a_shader_nothing_draws_takes_no_handle_and_one_a_material_draws_loads_once_named() {
            let dir = Dir::new("shader-before-material");
            shader_project(&dir, TINT, &glowing(&[("tint", ShaderValue::Vec4(RED))]));
            let db = open(&dir);
            let mut world = {
                let mut world = World::new();
                world.insert_resource(crate::scene::MeshBounds::default());
                world.insert_resource(crate::scene::AssetPreviews::default());
                world.insert_resource(crate::scene::MaterialBlends::default());
                world.insert_resource(db);
                world
            };
            let mut backend = HeadlessBackend::new();
            assert_eq!(settle(&world, &mut backend), Vec::<String>::new());
            assert_eq!(backend.surface_buffer().loaded(), 0);
            assert!(
                world
                    .resource::<AssetDatabase>()
                    .shader_params(glow())
                    .is_some()
            );

            let id = world
                .resource::<AssetDatabase>()
                .find("brick.omat")
                .unwrap()
                .id;
            let pending = world.resource_mut::<AssetDatabase>().material(id).unwrap();
            let entity = world.spawn();
            world.insert(entity, pending);
            assert_eq!(update(&world, &mut backend), Vec::<String>::new());
            let bound = material_handle(&world, entity);
            assert_eq!(
                backend.surface_buffer().shader_of(bound),
                None,
                "standard until the shader loads"
            );

            assert_eq!(settle(&world, &mut backend), Vec::<String>::new());
            let shader = handle(&world).expect("loaded");
            assert_eq!(backend.surface_buffer().shader_of(bound), Some(shader));
            assert_eq!(block(&backend, bound)[0], RED.map(f32::to_bits));
            assert_eq!(backend.surface_buffer().loaded(), 1);
        }

        #[test]
        fn a_material_bound_before_its_shader_compiles_attaches_when_it_loads() {
            let dir = Dir::new("shader-material-first");
            shader_project(&dir, TINT, &glowing(&[("tint", ShaderValue::Vec4(RED))]));
            let (world, bound, mut backend) = drawn(&dir);
            assert_eq!(backend.surface_buffer().shader_of(bound), None);

            assert_eq!(settle(&world, &mut backend), Vec::<String>::new());
            assert_eq!(backend.surface_buffer().shader_of(bound), handle(&world));
            assert_eq!(block(&backend, bound)[0], RED.map(f32::to_bits));
        }

        /// A save recompiles into the handle the shader already has, and the
        /// backend repacks the material's values by name into the new layout.
        #[test]
        fn a_saved_shader_is_replaced_in_its_handle_and_its_materials_keep_their_values() {
            let dir = Dir::new("shader-replace");
            shader_project(&dir, TINT, &glowing(&[("tint", ShaderValue::Vec4(RED))]));
            let (world, bound, mut backend) = drawn(&dir);
            settle(&world, &mut backend);
            let shader = handle(&world).unwrap();

            save(&world, &dir, "fx/glow.oshader", WEAR_THEN_TINT);
            assert_eq!(settle(&world, &mut backend), Vec::<String>::new());

            assert_eq!(handle(&world), Some(shader), "one handle for the session");
            assert_eq!(backend.surface_buffer().loaded(), 1);
            let table = backend.surface_params(shader).unwrap();
            assert_eq!(table.params[0].name, "wear");
            assert_eq!(
                world.resource::<AssetDatabase>().shader_params(glow()),
                Some(table)
            );
            assert_eq!(backend.surface_buffer().shader_of(bound), Some(shader));
            let words = block(&backend, bound);
            assert_eq!(words[0][0], 0.5f32.to_bits(), "wear at its default");
            assert_eq!(
                words[1],
                RED.map(f32::to_bits),
                "tint moved to the next word"
            );
        }

        #[test]
        fn a_compile_error_names_its_line_keeps_the_last_good_version_and_clears_when_fixed() {
            let dir = Dir::new("shader-error");
            shader_project(&dir, TINT, &glowing(&[("tint", ShaderValue::Vec4(RED))]));
            let (world, bound, mut backend) = drawn(&dir);
            settle(&world, &mut backend);
            let shader = handle(&world).unwrap();
            let before = block(&backend, bound);

            save(&world, &dir, "fx/glow.oshader", BROKEN);
            let errors = settle(&world, &mut backend);
            println!("{errors:#?}");
            assert!(
                errors[0].starts_with("fx/glow.oshader:3: "),
                "located at the line written: {errors:?}"
            );
            assert_eq!(
                errors.last().unwrap(),
                "fx/glow.oshader did not compile, so the last version that did is still drawn"
            );
            let db = world.resource::<AssetDatabase>();
            let lines = db.shader_errors(glow());
            assert!(!lines.is_empty());
            assert!(lines.iter().all(|error| error.line == Some(3)), "{lines:?}");
            assert_eq!(db.state(glow()), LoadState::Ready, "still drawn");
            assert_eq!(db.shader_params(glow()).unwrap().params[0].name, "tint");
            drop(db);
            assert_eq!(backend.surface_buffer().shader_of(bound), Some(shader));
            assert_eq!(block(&backend, bound), before);

            save(&world, &dir, "fx/glow.oshader", WEAR_THEN_TINT);
            assert_eq!(settle(&world, &mut backend), Vec::<String>::new());
            assert!(
                world
                    .resource::<AssetDatabase>()
                    .shader_errors(glow())
                    .is_empty()
            );
            assert_eq!(handle(&world), Some(shader));
            assert_eq!(
                backend.surface_params(shader).unwrap().params[0].name,
                "wear"
            );
        }

        #[test]
        fn a_shader_that_never_compiled_leaves_its_materials_standard() {
            let dir = Dir::new("shader-never");
            shader_project(&dir, BROKEN, &glowing(&[]));
            let (world, bound, mut backend) = drawn(&dir);
            let errors = settle(&world, &mut backend);
            assert!(errors[0].starts_with("fx/glow.oshader:3: "), "{errors:?}");

            assert_eq!(backend.surface_buffer().shader_of(bound), None);
            assert_eq!(backend.surface_buffer().loaded(), 0);
            let db = world.resource::<AssetDatabase>();
            assert_eq!(db.state(glow()), LoadState::Failed);
            assert_eq!(db.shader_params(glow()), None);
            assert_eq!(db.shader_errors(glow())[0].line, Some(3));
        }

        /// A texture value takes the texture's slot at once — white while it
        /// imports — and the slot is filled in place when it lands, so the
        /// material is not attached a second time.
        #[test]
        fn a_texture_value_draws_its_slot_while_the_texture_imports_and_then_the_texture() {
            let dir = Dir::new("shader-texture");
            let rock_png = super::super::super::import::tests::png(1, 1, |_, _| [10, 20, 30, 255]);
            dir.write("rock.png", &rock_png);
            dir.write("rock.png.meta", format!("id = \"{ROCK}\"\n").as_bytes());
            shader_project(
                &dir,
                MASKED,
                &glowing(&[("mask", ShaderValue::Texture(rock()))]),
            );
            let (world, bound, mut backend) = drawn(&dir);
            for _ in 0..4 {
                if handle(&world).is_some() {
                    break;
                }
                assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
                assert_eq!(update(&world, &mut backend), Vec::<String>::new());
            }
            let slot = world.resource::<AssetDatabase>().texture(rock()).unwrap();
            assert_eq!(block(&backend, bound)[0][0], slot.0);
            assert_eq!(
                backend.texel(slot),
                Some(([255; 4], true)),
                "white while it imports"
            );

            settle(&world, &mut backend);
            assert_eq!(
                backend.texel(slot).map(|(texel, _)| texel),
                Some([10, 20, 30, 255])
            );
            assert_eq!(block(&backend, bound)[0][0], slot.0, "the same slot");
            assert_eq!(
                world.resource::<AssetDatabase>().state(rock()),
                LoadState::Ready
            );
        }

        #[test]
        fn a_material_that_drops_its_shader_detaches() {
            let dir = Dir::new("shader-drop");
            shader_project(&dir, TINT, &glowing(&[("tint", ShaderValue::Vec4(RED))]));
            let (world, bound, mut backend) = drawn(&dir);
            settle(&world, &mut backend);
            assert!(backend.surface_buffer().shader_of(bound).is_some());

            save(
                &world,
                &dir,
                "brick.omat",
                &material::write(&MaterialFile::default()),
            );
            assert_eq!(settle(&world, &mut backend), Vec::<String>::new());
            assert_eq!(backend.surface_buffer().shader_of(bound), None);
        }

        /// A value the shader does not declare stays in the file and is
        /// warned about once each time something changes — never per frame.
        #[test]
        fn an_ignored_value_is_kept_and_warned_about_once_per_change() {
            let dir = Dir::new("shader-warn");
            let brick = glowing(&[
                ("renamed", ShaderValue::Float(1.0)),
                ("tint", ShaderValue::Vec4(RED)),
            ]);
            shader_project(&dir, TINT, &brick);
            let (world, bound, mut backend) = drawn(&dir);
            settle(&world, &mut backend);
            assert!(backend.surface_buffer().shader_of(bound).is_some());
            let once = [
                "brick.omat, drawn through fx/glow.oshader: `renamed` is not a parameter of this \
                 shader; the value is ignored",
            ];
            assert_eq!(warnings(&world), once);
            settle(&world, &mut backend);
            assert_eq!(warnings(&world), Vec::<String>::new(), "nothing changed");

            let db = world.resource::<AssetDatabase>();
            let id = db.find("brick.omat").unwrap().id;
            let kept = material::write(db.material_file(id).unwrap());
            assert!(kept.contains("  renamed = 1.0\n"), "{kept}");
            drop(db);

            let mut resaved = brick.clone();
            resaved.set_shader_value("tint", Some(ShaderValue::Vec4([0.0, 1.0, 0.0, 1.0])));
            save(&world, &dir, "brick.omat", &material::write(&resaved));
            settle(&world, &mut backend);
            assert_eq!(warnings(&world), once, "the material changed");

            save(&world, &dir, "fx/glow.oshader", WEAR_THEN_TINT);
            settle(&world, &mut backend);
            assert_eq!(warnings(&world), once, "the shader changed");
        }

        /// The shader editor's save: written, queued at once — so the editor can
        /// say it is compiling from the moment of the save — and imported once,
        /// the watcher hearing that save being nothing new.
        #[test]
        fn a_shader_the_editor_saves_is_written_queued_and_imported_once() {
            let dir = Dir::new("shader-write");
            shader_project(&dir, TINT, &MaterialFile::default());
            let mut db = open(&dir);
            assert!(db.wait_idle(WAIT));
            db.update(&World::new(), &mut HeadlessBackend::new());

            db.write_shader(glow(), WEAR_THEN_TINT).unwrap();
            let file = dir.path().join("fx/glow.oshader");
            assert_eq!(std::fs::read_to_string(&file).unwrap(), WEAR_THEN_TINT);
            let entry = db.get(glow()).unwrap();
            assert_eq!(entry.status, Status::Queued, "compiling from the save on");
            assert_eq!(entry.bytes, WEAR_THEN_TINT.len() as u64);

            db.notice(file.clone());
            db.apply_changes();
            let mut backend = HeadlessBackend::new();
            for _ in 0..3 {
                assert!(db.wait_idle(WAIT));
                assert_eq!(db.update(&World::new(), &mut backend), Vec::<String>::new());
            }
            let Status::Imported { cached: false, .. } = db.get(glow()).unwrap().status else {
                panic!(
                    "an import of the echo would have found the save's artifact cached: {:?}",
                    db.get(glow()).unwrap().status
                );
            };
            assert_eq!(db.shader_params(glow()).unwrap().params[0].name, "wear");

            // Another program's save is heard, and so is one that puts back
            // the text the editor last wrote.
            for (text, broken) in [(BROKEN, true), (WEAR_THEN_TINT, false)] {
                dir.write("fx/glow.oshader", text.as_bytes());
                db.notice(file.clone());
                db.apply_changes();
                assert_eq!(db.get(glow()).unwrap().status, Status::Queued);
                assert!(db.wait_idle(WAIT));
                db.update(&World::new(), &mut backend);
                assert_eq!(!db.shader_errors(glow()).is_empty(), broken, "{text}");
            }

            let brick = db.find("brick.omat").unwrap().id;
            assert_eq!(
                db.write_shader(brick, TINT),
                Err("`brick.omat` is a material, not a surface shader".to_owned())
            );
        }

        /// What the shader editor previews a shader on when no material in
        /// the project uses it: a material of its own at the defaults, drawn
        /// through the shader with every parameter at the shader's default.
        #[test]
        fn a_shader_previews_on_a_material_of_its_own_at_its_defaults() {
            let dir = Dir::new("shader-preview");
            shader_project(&dir, TINT, &MaterialFile::default());
            let mut world = World::new();
            world.insert_resource(crate::scene::MeshBounds::default());
            world.insert_resource(crate::scene::AssetPreviews::default());
            world.insert_resource(crate::scene::MaterialBlends::default());
            world.insert_resource(open(&dir));
            let mut backend = HeadlessBackend::new();

            let preview = |world: &World| {
                world
                    .resource_mut::<AssetDatabase>()
                    .shader_preview_material(glow())
                    .unwrap()
            };
            let pending = preview(&world);
            assert_eq!(preview(&world), pending, "one for the session");
            assert_eq!(settle(&world, &mut backend), Vec::<String>::new());

            let bound = preview(&world);
            assert_eq!(bound.0 & PENDING, 0, "bound between frames");
            assert_eq!(backend.surface_buffer().shader_of(bound), handle(&world));
            assert_eq!(
                block(&backend, bound)[0],
                [1.0f32; 4].map(f32::to_bits),
                "tint at its default"
            );
            let db = world.resource::<AssetDatabase>();
            assert_eq!(db.iter().count(), 2, "not an asset: nothing new is offered");
            let brick = db.find("brick.omat").unwrap().id;
            drop(db);
            assert!(
                world
                    .resource_mut::<AssetDatabase>()
                    .shader_preview_material(brick)
                    .is_err()
            );
        }

        #[test]
        fn a_new_shader_is_the_template_and_compiles_with_one_colour_parameter() {
            let dir = Dir::new("shader-new");
            let mut db = open(&dir);
            let id = db.create_shader("fx").unwrap();
            let entry = db
                .find("fx/New Surface Shader.oshader")
                .expect("indexed at once");
            assert_eq!((entry.id, entry.kind), (id, AssetKind::Shader));
            assert_eq!(db.shader_source(id).as_deref(), Ok(NEW_SHADER));
            assert!(
                dir.path()
                    .join("fx/New Surface Shader.oshader.meta")
                    .is_file()
            );
            let again = db.create_shader("fx").unwrap();
            assert_eq!(
                db.get(again).unwrap().path,
                "fx/New Surface Shader 2.oshader"
            );

            assert!(db.wait_idle(WAIT));
            assert_eq!(
                db.update(&World::new(), &mut HeadlessBackend::new()),
                Vec::<String>::new()
            );
            let params = db.shader_params(id).expect("it compiles");
            assert_eq!(params.params.len(), 1);
            assert!(matches!(params.params[0].kind, ParamKind::Color { .. }));
        }
    }
}
