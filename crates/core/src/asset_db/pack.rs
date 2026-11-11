//! An exported game's assets: the artifacts, never the sources.
//!
//! A pack is a directory laid out like the assets directory it came from, with
//! every importable source replaced by the artifact the cache already holds for
//! it. Sidecars are kept, because they are where an asset's id and its usage
//! live; text assets — materials, prefabs, scenes — are kept as they are,
//! because reading one *is* loading it. `index.toml` names the artifact behind
//! each imported asset and the texture target they were made for.
//!
//! Nothing is imported while packing and nothing is hashed while loading. The
//! keys are the ones the database recorded when it imported, and an artifact is
//! copied byte for byte, so a pack made after a clean import on a CI machine is
//! the same pack a developer's editor would have made. The engine opening one
//! ([`AssetDatabase::open_packed`]) never reads a source, never writes a
//! sidecar, never watches, and never reimports: a missing artifact is a named
//! error and a pink object, as a broken import is in the editor.
//!
//! A surface shader is packed as its compiled artifact, like a texture, and
//! loaded with no compiler, which a game is built without. An artifact compiled
//! by another build of the engine, or one that fails its integrity check, is a
//! named error, and the materials drawn through it draw as standard ones. The
//! packing engine is the editor's build, so a shader the cache has no artifact
//! for is compiled while packing; one that does not compile stops the pack by
//! name, as a texture that does not import does.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use orrin_ecs::FxHashMap;
use serde::{Deserialize, Serialize};

use super::{AssetDatabase, AssetKind, CacheKey, Entry, Status, Target, import, sidecar};

pub const INDEX: &str = "index.toml";
/// Where the artifacts sit inside a pack.
pub const ARTIFACTS: &str = "artifacts";

const FORMAT: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Index {
    format_version: u32,
    /// [`Target::name`] of every texture artifact in the pack.
    target: String,
    /// Every packed asset, by path.
    assets: Vec<String>,
    /// The artifact behind each imported asset, by path.
    artifacts: BTreeMap<String, String>,
}

/// Write `db`'s assets into `out` as a pack. Every importable asset must have
/// imported: a pack with a hole in it would ship a pink object, so the first
/// asset that did not is the error, by path.
///
/// Returns how many assets were packed.
pub fn write(db: &AssetDatabase, out: &Path) -> Result<usize, String> {
    let mut index = Index {
        format_version: FORMAT,
        target: db.target.name().to_owned(),
        assets: Vec::new(),
        artifacts: BTreeMap::new(),
    };
    let artifacts = out.join(ARTIFACTS);
    std::fs::create_dir_all(&artifacts)
        .map_err(|error| format!("could not create {}: {error}", artifacts.display()))?;

    for entry in db.iter() {
        let source = db.dir.join(&entry.path);
        let meta = sidecar::path_of(&source);
        if meta.is_file() {
            copy(&meta, &sidecar::path_of(&out.join(&entry.path)))?;
        }
        if import::importer(entry.kind).is_none() {
            copy(&source, &out.join(&entry.path))?;
        } else {
            let key = match &entry.status {
                Status::Imported { key, .. } => *key,
                Status::Failed(message) => {
                    return Err(format!("{message}. Fix it, or remove it, before exporting"));
                }
                Status::Queued | Status::Indexed => {
                    return Err(format!("{} has not finished importing", entry.path));
                }
            };
            copy(&db.cache.path(key), &artifacts.join(key.to_string()))?;
            index.artifacts.insert(entry.path.clone(), key.to_string());
        }
        index.assets.push(entry.path.clone());
    }

    for sky in skies(&db.dir) {
        let rel = sky.strip_prefix(&db.dir).unwrap_or(&sky);
        copy(&sky, &out.join(rel))?;
    }

    let text = toml::to_string_pretty(&index)
        .map_err(|error| format!("could not encode the pack index: {error}"))?;
    let path = out.join(INDEX);
    std::fs::write(&path, text)
        .map_err(|error| format!("could not write {}: {error}", path.display()))?;
    Ok(index.assets.len())
}

/// Import whatever `project` has not imported yet, for a device sampling
/// `target`, and pack all of it into `out`. What `orrin export` runs, on the
/// machine doing the export: the game's own executable may be for another
/// platform and cannot be run here.
pub fn pack_project(
    project: &orrin_project::Project,
    out: &Path,
    target: Target,
) -> Result<usize, String> {
    let (mut db, issues) = AssetDatabase::open_for(
        project.assets_dir(),
        super::cache_dir(Some(project)),
        target,
    );
    if let Some(issue) = issues.first() {
        return Err(issue.clone());
    }
    while !db.wait_idle(std::time::Duration::from_secs(1)) {}
    db.update(
        &orrin_ecs::World::new(),
        &mut crate::gfx::HeadlessBackend::new(),
    );
    write(&db, out)
}

/// [`pack_project`] as a command: the project at `project` into `out`, for
/// block-compressed textures, which every desktop GPU a game exports for
/// samples. Returns the exit status.
pub fn run_tool(project: &Path, out: &Path) -> i32 {
    let project = match orrin_project::Project::locate(project) {
        Ok(Some(project)) => project,
        Ok(None) => {
            eprintln!("no orrin.toml in {} or above it", project.display());
            return 1;
        }
        Err(error) => {
            eprintln!("{error}");
            return 1;
        }
    };
    match pack_project(&project, out, Target::Bc) {
        Ok(count) => {
            println!("packed {count} asset(s) into {}", out.display());
            0
        }
        Err(error) => {
            eprintln!("{error}");
            1
        }
    }
}

/// Every `.hdr` under `dir`: an environment is baked from its file as it is,
/// named by path in a scene's render settings, and is no asset the database
/// indexes. Dotfiles and links are passed over, as the scan does.
fn skies(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let hidden = entry.file_name().to_string_lossy().starts_with('.');
            match entry.file_type() {
                Ok(kind) if hidden || kind.is_symlink() => {}
                Ok(kind) if kind.is_dir() => pending.push(path),
                Ok(_)
                    if path
                        .extension()
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("hdr")) =>
                {
                    found.push(path)
                }
                _ => {}
            }
        }
    }
    found.sort();
    found
}

fn copy(from: &Path, to: &Path) -> Result<(), String> {
    if let Some(dir) = to.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|error| format!("could not create {}: {error}", dir.display()))?;
    }
    std::fs::copy(from, to)
        .map(|_| ())
        .map_err(|error| format!("could not copy {} into the pack: {error}", from.display()))
}

impl AssetDatabase {
    /// The pack in `dir`, for a device sampling `target`. Returns what could
    /// not be read, each naming its file; the rest still load.
    pub fn open_packed(dir: impl Into<PathBuf>, target: Target) -> (Self, Vec<String>) {
        let dir = dir.into();
        let (mut db, mut issues) = Self::empty(dir.clone(), dir.join(ARTIFACTS), target);
        let path = dir.join(INDEX);
        let index = std::fs::read_to_string(&path)
            .map_err(|error| format!("{}: {error}", path.display()))
            .and_then(|text| {
                toml::from_str::<Index>(&text)
                    .map_err(|error| format!("{}: {}", path.display(), error.message()))
            });
        let index = match index {
            Ok(index) if index.format_version == FORMAT => index,
            Ok(index) => {
                issues.push(format!(
                    "{} is pack format {}, and this engine reads {FORMAT}; export the game again",
                    path.display(),
                    index.format_version
                ));
                db.pack = Some(FxHashMap::default());
                return (db, issues);
            }
            Err(error) => {
                issues.push(format!("{error}; the game's assets cannot load"));
                db.pack = Some(FxHashMap::default());
                return (db, issues);
            }
        };
        if index.target != target.name() {
            issues.push(format!(
                "the assets were packed for `{}` textures and this device samples `{}`;                  textures will not load. Export the game again for this hardware",
                index.target,
                target.name()
            ));
        }

        let mut pack = FxHashMap::default();
        for rel in index.assets {
            let Some(kind) = AssetKind::of_path(&rel) else {
                issues.push(format!(
                    "{rel} in the pack is not a kind of asset this engine reads"
                ));
                continue;
            };
            let source = dir.join(&rel);
            let sidecar = match sidecar::read_existing(&source, &rel, kind) {
                Ok(sidecar) => sidecar,
                Err(issue) => {
                    issues.push(issue);
                    continue;
                }
            };
            let status = match index.artifacts.get(&rel) {
                Some(key) => match u128::from_str_radix(key, 16) {
                    Ok(key) => {
                        pack.insert(sidecar.id, CacheKey(key));
                        Status::Imported {
                            key: CacheKey(key),
                            cached: true,
                        }
                    }
                    Err(_) => Status::Failed(format!("{rel}: `{key}` in the pack is not a key")),
                },
                None if import::importer(kind).is_some() => {
                    Status::Failed(format!("{rel} has no artifact in the pack"))
                }
                None => Status::Indexed,
            };
            db.entries.push(Entry {
                id: sidecar.id,
                path: rel,
                kind,
                settings: sidecar.settings,
                status,
                inputs: Vec::new(),
                on_disk: true,
                summary: None,
                bytes: 0,
                modified: None,
            });
        }
        db.entries.sort_by(|a, b| a.path.cmp(&b.path));
        db.pack = Some(pack);
        let materials: Vec<_> = db
            .entries
            .iter()
            .filter(|entry| entry.kind == AssetKind::Material)
            .map(|entry| entry.id)
            .collect();
        for id in materials {
            db.read_material(id);
        }
        issues.extend(std::mem::take(&mut db.reports));
        (db, issues)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use orrin_ecs::World;
    #[cfg(feature = "shader-compiler")]
    use orrin_registry::AssetId;

    use super::super::import::tests::{png, triangle_bin, triangle_gltf};
    use super::super::tests::Dir;
    use super::super::{LoadState, PENDING};
    use super::*;
    use crate::gfx::{HeadlessBackend, RenderBackend};
    use crate::scene::MeshHandle;

    const WAIT: Duration = Duration::from_secs(20);

    fn project(dir: &AssetsDir) {
        dir.write("rocks/rock.png", &png(2, 2, |_, _| [10, 20, 30, 255]));
        dir.write("models/tree.gltf", triangle_gltf("").as_bytes());
        dir.write("models/tri angle.bin", &triangle_bin());
        dir.write("props/door.oprefab", b"orrin-scene 1\n");
        dir.write("scenes/level.orrin", b"orrin-scene 1\n");
    }

    /// `dir` opened and imported in full, as the editor leaves a project.
    fn imported(dir: &Dir) -> AssetDatabase {
        let (mut db, issues) =
            AssetDatabase::open(dir.path().join("assets"), dir.path().join("cache"));
        assert!(issues.is_empty(), "{issues:?}");
        assert!(db.wait_idle(WAIT));
        db.update(&World::new(), &mut HeadlessBackend::new());
        db
    }

    /// Every file under `dir`, relative and `/`-separated, with its bytes.
    fn files(dir: &Path) -> BTreeMap<String, Vec<u8>> {
        fn walk(dir: &Path, base: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
            for entry in std::fs::read_dir(dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, base, out);
                } else {
                    let rel = path.strip_prefix(base).unwrap();
                    let rel: Vec<_> = rel
                        .iter()
                        .map(|p| p.to_string_lossy().into_owned())
                        .collect();
                    out.insert(rel.join("/"), std::fs::read(&path).unwrap());
                }
            }
        }
        let mut out = BTreeMap::new();
        walk(dir, dir, &mut out);
        out
    }

    fn key_of(db: &AssetDatabase, path: &str) -> CacheKey {
        match db.find(path).unwrap().status {
            Status::Imported { key, .. } => key,
            ref other => panic!("{path}: {other:?}"),
        }
    }

    fn source_project(name: &str) -> (Dir, PathBuf) {
        let dir = Dir::new(name);
        let assets = dir.path().join("assets");
        project(&AssetsDir(&assets));
        (dir, assets)
    }

    /// [`Dir::write`] rooted somewhere other than a `Dir`'s own path.
    struct AssetsDir<'a>(&'a Path);

    impl AssetsDir<'_> {
        fn write(&self, rel: &str, bytes: &[u8]) {
            let path = self.0.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, bytes).unwrap();
        }
    }

    #[test]
    fn a_pack_holds_artifacts_sidecars_and_text_assets_and_no_importable_source() {
        let (dir, _) = source_project("pack-contents");
        let db = imported(&dir);
        let out = dir.path().join("pack");

        assert_eq!(write(&db, &out), Ok(4));

        let packed = files(&out);
        for source in ["rocks/rock.png", "models/tree.gltf", "models/tri angle.bin"] {
            assert!(!packed.contains_key(source), "{source} is a source");
        }
        for kept in ["props/door.oprefab", "scenes/level.orrin"] {
            assert!(packed.contains_key(kept), "{kept} is read as it is");
        }
        for asset in ["rocks/rock.png", "models/tree.gltf", "props/door.oprefab"] {
            assert!(
                packed.contains_key(&format!("{asset}.meta")),
                "{asset}'s id"
            );
        }
        for asset in ["rocks/rock.png", "models/tree.gltf"] {
            let artifact = format!("{ARTIFACTS}/{}", key_of(&db, asset));
            assert_eq!(
                packed.get(&artifact),
                Some(&std::fs::read(db.cache_dir().join(key_of(&db, asset).to_string())).unwrap()),
                "{asset}'s artifact, byte for byte"
            );
        }
        assert!(packed.contains_key(INDEX));
    }

    /// A sky is read by the environment bake as it is, not imported, so the
    /// pack carries the file itself — or a shipped game's sky goes black.
    #[test]
    fn a_sky_is_packed_as_it_is() {
        let (dir, assets) = source_project("pack-sky");
        AssetsDir(&assets).write("environment/meadow.hdr", b"#?RADIANCE sky");
        let db = imported(&dir);
        let out = dir.path().join("pack");

        write(&db, &out).unwrap();

        assert_eq!(
            std::fs::read(out.join("environment/meadow.hdr")).unwrap(),
            b"#?RADIANCE sky"
        );
    }

    /// A font is read by the UI as it is, so a shipped game finds the file in
    /// the pack rather than an artifact.
    #[test]
    fn a_font_is_packed_as_it_is_and_reads_from_the_pack() {
        let (dir, assets) = source_project("pack-font");
        AssetsDir(&assets).write("fonts/title.ttf", b"glyphs");
        let db = imported(&dir);
        let id = db.find("fonts/title.ttf").unwrap().id;
        let out = dir.path().join("pack");

        write(&db, &out).unwrap();
        let (packed, issues) = AssetDatabase::open_packed(&out, Target::Rgba8);
        assert!(issues.is_empty(), "{issues:?}");
        assert_eq!(packed.font(id), Ok(b"glyphs".to_vec()));
    }

    /// Audio is decoded from the file itself, so a shipped game finds it in
    /// the pack exactly as it was authored.
    #[test]
    fn an_audio_clip_is_packed_as_it_is_and_reads_from_the_pack() {
        let (dir, assets) = source_project("pack-audio");
        AssetsDir(&assets).write("sfx/door.ogg", b"vorbis");
        let db = imported(&dir);
        let id = db.find("sfx/door.ogg").unwrap().id;
        let out = dir.path().join("pack");

        write(&db, &out).unwrap();
        let (packed, issues) = AssetDatabase::open_packed(&out, Target::Rgba8);
        assert!(issues.is_empty(), "{issues:?}");
        let file = packed.audio_file(id).unwrap();
        assert!(file.starts_with(&out), "{}", file.display());
        assert_eq!(std::fs::read(file).unwrap(), b"vorbis");
    }

    #[test]
    fn the_same_imported_project_packs_to_the_same_bytes() {
        let (dir, _) = source_project("pack-reproducible");
        let db = imported(&dir);

        write(&db, &dir.path().join("one")).unwrap();
        write(&db, &dir.path().join("two")).unwrap();

        assert_eq!(
            files(&dir.path().join("one")),
            files(&dir.path().join("two"))
        );
    }

    #[test]
    fn an_asset_that_did_not_import_stops_the_pack_by_name() {
        let (dir, assets) = source_project("pack-broken");
        AssetsDir(&assets).write("rocks/bad.png", b"not a png");
        let db = imported(&dir);

        let error = write(&db, &dir.path().join("pack")).unwrap_err();

        assert!(error.contains("rocks/bad.png"), "{error}");
    }

    #[test]
    fn a_packed_mesh_draws_without_its_source_and_the_pack_is_left_untouched() {
        let (dir, _) = source_project("pack-draw");
        let db = imported(&dir);
        let tree = db.find("models/tree.gltf").unwrap().id;
        let out = dir.path().join("pack");
        write(&db, &out).unwrap();
        drop(db);
        std::fs::remove_dir_all(dir.path().join("assets")).unwrap();
        std::fs::remove_dir_all(dir.path().join("cache")).unwrap();
        let before = files(&out);

        let (packed, issues) = AssetDatabase::open_packed(&out, Target::Rgba8);
        assert!(issues.is_empty(), "{issues:?}");
        assert_eq!(packed.find("models/tree.gltf").map(|e| e.id), Some(tree));
        assert!(packed.find("scenes/level.orrin").is_some());

        let mut world = World::new();
        world.insert_resource(crate::scene::MeshBounds::default());
        world.insert_resource(crate::scene::AssetPreviews::default());
        world.insert_resource(crate::scene::MaterialBlends::default());
        world.insert_resource(packed);
        let pending = world.resource_mut::<AssetDatabase>().mesh(tree).unwrap();
        assert_ne!(pending.0 & PENDING, 0);
        let entity = world.spawn();
        world.insert(entity, pending);
        let mut backend = HeadlessBackend::new();
        for _ in 0..2 {
            let errors = world
                .resource_mut::<AssetDatabase>()
                .update(&world, &mut backend);
            assert!(errors.is_empty(), "{errors:?}");
            assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
        }

        assert_eq!(
            world.resource::<AssetDatabase>().state(tree),
            LoadState::Ready
        );
        let handle = *world.get::<MeshHandle>(entity).unwrap();
        let bounds = backend.mesh_bounds(handle).unwrap();
        assert!(
            bounds.max.x > bounds.min.x,
            "the triangle, not the loading stand-in"
        );
        assert_eq!(files(&out), before, "nothing written into the pack");
    }

    /// A project whose `brick.omat` is drawn through `fx/glow.oshader`, packed
    /// and then stripped of its sources and cache, as a shipped game is.
    #[cfg(feature = "shader-compiler")]
    fn packed_shader(name: &str) -> (Dir, PathBuf, AssetId, CacheKey) {
        use super::super::material::{self, MaterialFile, ShaderValue};
        let (dir, assets) = source_project(name);
        let glow = "param color tint = (1, 1, 1, 1);\n\
                    void surface(in SurfaceIn i, inout SurfaceOut o) {\n    o.albedo = tint.rgb;\n}\n";
        AssetsDir(&assets).write("fx/glow.oshader", glow.as_bytes());
        let id: AssetId = "5f0c3a4e-0000-4000-8000-0000000000d1".parse().unwrap();
        AssetsDir(&assets).write(
            "fx/glow.oshader.meta",
            format!("id = \"{id}\"\n").as_bytes(),
        );
        let mut brick = MaterialFile {
            shader: Some(id),
            ..MaterialFile::default()
        };
        brick.set_shader_value("tint", Some(ShaderValue::Vec4([1.0, 0.0, 0.0, 1.0])));
        AssetsDir(&assets).write("brick.omat", material::write(&brick).as_bytes());
        let db = imported(&dir);
        let key = key_of(&db, "fx/glow.oshader");
        let out = dir.path().join("pack");
        write(&db, &out).unwrap();
        assert_eq!(
            std::fs::read(out.join(ARTIFACTS).join(key.to_string())).unwrap(),
            std::fs::read(db.cache_dir().join(key.to_string())).unwrap(),
            "the artifact, byte for byte"
        );
        assert!(!out.join("fx/glow.oshader").exists(), "never the source");
        assert!(out.join("fx/glow.oshader.meta").is_file());
        drop(db);
        std::fs::remove_dir_all(dir.path().join("assets")).unwrap();
        std::fs::remove_dir_all(dir.path().join("cache")).unwrap();
        (dir, out, id, key)
    }

    /// The pack opened and `brick.omat` drawn until nothing is left to do:
    /// what was logged, and whether the material reads through a shader.
    #[cfg(feature = "shader-compiler")]
    fn draw_packed_brick(
        out: &Path,
    ) -> (
        Vec<String>,
        Option<crate::gfx::surface::SurfaceShaderHandle>,
    ) {
        let (packed, issues) = AssetDatabase::open_packed(out, Target::Rgba8);
        assert!(issues.is_empty(), "{issues:?}");
        let brick = packed.find("brick.omat").unwrap().id;
        let (world, entity) = super::super::tests::drawing_material(packed, brick);
        let mut backend = HeadlessBackend::new();
        let mut errors = Vec::new();
        for _ in 0..4 {
            errors.extend(super::super::tests::update(&world, &mut backend));
            assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
        }
        let bound = super::super::tests::material_handle(&world, entity);
        (errors, backend.surface_buffer().shader_of(bound))
    }

    /// The source and the cache are gone, so nothing could have been compiled:
    /// the shader drawn is the packed artifact.
    #[cfg(feature = "shader-compiler")]
    #[test]
    fn a_packed_shader_is_its_artifact_and_draws_with_no_source_to_compile() {
        let (_dir, out, _, _) = packed_shader("pack-shader");
        let (errors, shader) = draw_packed_brick(&out);
        assert_eq!(errors, Vec::<String>::new());
        assert!(shader.is_some(), "the material reads through the shader");
    }

    /// An artifact from another build would bind against layouts this one may
    /// not have, and a damaged one could crash a driver: each is refused by
    /// name, and the material draws as a standard one.
    #[cfg(feature = "shader-compiler")]
    #[test]
    fn a_packed_shader_from_another_build_or_damaged_is_named_and_draws_standard() {
        use crate::gfx::surface::SurfaceArtifact;
        let foreign = |bytes: Vec<u8>| {
            let mut artifact = SurfaceArtifact::decode(&bytes).unwrap();
            artifact.sources_hash ^= 1;
            artifact.encode()
        };
        let damaged = |mut bytes: Vec<u8>| {
            let last = bytes.len() - 1;
            bytes[last] ^= 0xff;
            bytes
        };
        type Spoil<'a> = &'a dyn Fn(Vec<u8>) -> Vec<u8>;
        let cases: [(&str, Spoil, &str); 2] = [
            (
                "pack-shader-foreign",
                &foreign,
                "fx/glow.oshader: its packed artifact was compiled by another build of the \
                 engine than this one, so its materials draw without it; export the game again",
            ),
            (
                "pack-shader-damaged",
                &damaged,
                "fx/glow.oshader: its packed artifact cannot be read, so its materials draw \
                 without it: the surface shader artifact is corrupt: its contents do not match \
                 its integrity hash. Export the game again",
            ),
        ];
        for (name, spoil, expected) in cases {
            let (_dir, out, _, key) = packed_shader(name);
            let artifact = out.join(ARTIFACTS).join(key.to_string());
            std::fs::write(&artifact, spoil(std::fs::read(&artifact).unwrap())).unwrap();

            let (errors, shader) = draw_packed_brick(&out);
            assert_eq!(errors, [expected], "{name}");
            assert_eq!(shader, None, "{name}: standard");
        }
    }

    #[test]
    fn a_pack_made_for_another_texture_target_is_refused_by_name() {
        let (dir, _) = source_project("pack-target");
        let db = imported(&dir);
        let out = dir.path().join("pack");
        write(&db, &out).unwrap();

        let (_, issues) = AssetDatabase::open_packed(&out, Target::Bc);

        assert_eq!(issues.len(), 1, "{issues:?}");
        assert!(
            issues[0].contains("rgba8") && issues[0].contains("bc"),
            "{}",
            issues[0]
        );
    }
}
