//! An asset edited on disk reaches the screen through the real renderer.
//!
//! The asset database's own tests swap artifacts into a `HeadlessBackend`,
//! which counts replacements and keeps bounds but draws nothing. This is the
//! half they cannot see: that `VulkanRenderer::replace_texture` and
//! `replace_mesh` change what the next frame samples and rasterises, through
//! the arena, the bindless texture array and GPU culling, without the handle
//! an entity holds ever changing — and, on a device that samples BC, that the
//! compressed chain is what is uploaded and what the shaders read right.
//!
//! `#[ignore]`d for the reason `offscreen.rs` is: it needs a GPU. Run it with:
//!
//! ```text
//! cargo test -p orrin-core --test asset_reload -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::time::Duration;

use glam::Vec3;
use orrin_core::asset_db::cache::Artifact;
use orrin_core::asset_db::sidecar::{AssetKind, Settings, TextureSettings};
use orrin_core::asset_db::{self, AssetDatabase, Status, Target, Usage, material};
use orrin_core::capture::{self, CaptureSettings};
use orrin_core::gfx::vulkan::VulkanRenderer;
use orrin_core::gfx::{Material, RenderBackend, TexelFormat, TextureLevels};
use orrin_core::scene::edit::{History, SceneEditor};
use orrin_core::scene::{
    AssetPreviews, Camera, HdrSettings, LocalTransform, MaterialBlends, MaterialHandle, MeshHandle,
    Transform, apply_material_edits, register_components,
};
use orrin_ecs::World;
use orrin_registry::Registry;

const WAIT: Duration = Duration::from_secs(60);

/// A scratch directory, removed when dropped.
struct Dir(PathBuf);

impl Dir {
    /// One per test: the tests run at once, and a shared directory would be
    /// removed from under one by the other's drop.
    fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("orrin-asset-reload-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn write(&self, rel: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(rel);
        std::fs::write(&path, bytes).unwrap();
        path
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

fn png(rgba: [u8; 4]) -> Vec<u8> {
    let image = image::RgbaImage::from_pixel(2, 2, image::Rgba(rgba));
    let mut bytes = std::io::Cursor::new(Vec::new());
    image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
    bytes.into_inner()
}

/// A two-unit square in the XY plane facing +Z: positions, `u16` indices, UVs.
fn quad_bin() -> Vec<u8> {
    let mut bin = Vec::new();
    for p in [
        [-1.0f32, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ] {
        bin.extend(p.iter().flat_map(|c| c.to_le_bytes()));
    }
    bin.extend([0u16, 1, 2, 0, 2, 3].iter().flat_map(|i| i.to_le_bytes()));
    for uv in [[0.0f32, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]] {
        bin.extend(uv.iter().flat_map(|c| c.to_le_bytes()));
    }
    bin
}

/// The document for [`quad_bin`], drawn at `x`.
fn quad_gltf(x: f32) -> String {
    format!(
        r#"{{
  "asset": {{"version": "2.0"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [{{"mesh": 0, "translation": [{x}, 0, 0]}}],
  "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0, "TEXCOORD_0": 2}}, "indices": 1}}]}}],
  "buffers": [{{"uri": "quad.bin", "byteLength": 92}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": 48}},
    {{"buffer": 0, "byteOffset": 48, "byteLength": 12}},
    {{"buffer": 0, "byteOffset": 60, "byteLength": 32}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-1, -1, 0], "max": [1, 1, 0]}},
    {{"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"}},
    {{"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"}}
  ]
}}"#
    )
}

/// What the app does between frames: take finished imports, swap them in, and
/// hand material edits to the renderer.
fn between_frames(world: &mut World, renderer: &mut VulkanRenderer) {
    asset_db::service(world, renderer);
    apply_material_edits(world, renderer);
}

/// Write `rel`, let the database notice it, and wait for the reimport to land.
fn edit(world: &mut World, renderer: &mut VulkanRenderer, dir: &Dir, rel: &str, bytes: &[u8]) {
    let path = dir.write(rel, bytes);
    {
        let mut db = world.resource_mut::<AssetDatabase>();
        db.notice(path);
        db.apply_changes();
        assert!(db.wait_idle(WAIT));
    }
    between_frames(world, renderer);
}

/// The artifact the database last imported for `id`, read back from its cache.
fn cached(world: &World, id: orrin_registry::AssetId) -> TextureLevels {
    let db = world.resource::<AssetDatabase>();
    let Some(Status::Imported { key, .. }) = db.get(id).map(|entry| entry.status.clone()) else {
        panic!("imported");
    };
    let bytes = std::fs::read(db.cache_dir().join(key.to_string())).unwrap();
    match Artifact::decode(&bytes).unwrap() {
        Artifact::Texture(texture) => texture,
        Artifact::Mesh(_) => panic!("a texture"),
    }
}

/// The pixel at the middle of the frame, where the quad is drawn.
fn centre(frame: &(Vec<u8>, [u32; 2])) -> [u8; 3] {
    let (pixels, [width, height]) = frame;
    let at = ((height / 2 * width + width / 2) * 4) as usize;
    [pixels[at], pixels[at + 1], pixels[at + 2]]
}

#[test]
#[ignore = "needs a GPU"]
fn an_edited_texture_and_mesh_reach_the_next_frame() {
    orrin_core::logging::init(orrin_core::logging::LogConfig::for_this_build());
    let settings = CaptureSettings {
        extent: [128, 128],
        camera: Some(Camera {
            position: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            ..Camera::default()
        }),
        ..CaptureSettings::default()
    };
    let mut renderer = capture::offscreen_renderer(settings.extent);
    let mut world = World::new();
    capture::prepare(&mut world, &mut renderer, &settings);
    // The rig's lights and environment stay; its geometry goes, so the middle
    // of the frame is the quad or nothing.
    let drawn: Vec<_> = world
        .entities()
        .filter(|&entity| world.has::<MeshHandle>(entity))
        .collect();
    for entity in drawn {
        world.despawn(entity);
    }

    let dir = Dir::new("edit");
    dir.write("quad.gltf", quad_gltf(0.0).as_bytes());
    dir.write("quad.bin", &quad_bin());
    dir.write("paint.png", &png([255, 0, 0, 255]));
    let paint: orrin_registry::AssetId = "5f0c3a4e-0000-4000-8000-0000000000b1".parse().unwrap();
    dir.write("paint.png.meta", format!("id = \"{paint}\"\n").as_bytes());
    let painted = material::MaterialFile {
        material: Material {
            base_color: Vec3::ONE,
            metallic: 0.0,
            roughness: 1.0,
            ..Material::default()
        },
        textures: material::MaterialTextures {
            albedo: Some(paint),
            ..Default::default()
        },
    };
    dir.write("painted.fmat", material::write(&painted).as_bytes());
    let target = Target::of(&renderer);
    println!("the device samples BC: {}", renderer.samples_bc());
    let (db, issues) = AssetDatabase::open_for(dir.path(), dir.path().join(".cache"), target);
    assert!(issues.is_empty(), "{issues:?}");
    let quad = db.find("quad.gltf").unwrap().id;
    let painted = db.find("painted.fmat").unwrap().id;
    world.insert_resource(db);

    let mesh = world.resource_mut::<AssetDatabase>().mesh(quad).unwrap();
    let material = world
        .resource_mut::<AssetDatabase>()
        .material(painted)
        .unwrap();
    let entity = world.spawn();
    world.insert(entity, LocalTransform::new(Transform::default()));
    world.insert(entity, mesh);
    world.insert(entity, material);

    between_frames(&mut world, &mut renderer);
    assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
    between_frames(&mut world, &mut renderer);
    let slot = *world.get::<MeshHandle>(entity).unwrap();

    let [r, g, b] = centre(&capture::render(&mut world, &mut renderer, &settings));
    assert!(
        r > g + 40 && r > b + 40,
        "a red texture draws red: {:?}",
        [r, g, b]
    );
    assert_eq!(
        cached(&world, paint).format,
        TexelFormat::choose(Usage::Color, true, target),
        "what the device was given"
    );

    edit(
        &mut world,
        &mut renderer,
        &dir,
        "paint.png",
        &png([0, 255, 0, 255]),
    );
    let [r, g, b] = centre(&capture::render(&mut world, &mut renderer, &settings));
    assert!(g > r + 40 && g > b + 40, "repainted green: {:?}", [r, g, b]);

    edit(
        &mut world,
        &mut renderer,
        &dir,
        "quad.gltf",
        quad_gltf(50.0).as_bytes(),
    );
    let [r, g, b] = centre(&capture::render(&mut world, &mut renderer, &settings));
    assert!(
        !(g > r + 40 && g > b + 40),
        "the quad moved out of view: {:?}",
        [r, g, b]
    );
    assert_eq!(
        *world.get::<MeshHandle>(entity).unwrap(),
        slot,
        "one handle throughout"
    );
}

/// A normal map leaning 60° along its -Y axis, imported for `target`: on this
/// quad that turns it toward the demo's high sun, which moves the shading well
/// clear of the flat surface's, where a lean along X barely moves it.
fn leaning_normal_map(target: Target) -> TextureLevels {
    let lean = 60f32.to_radians();
    let code = |v: f32| ((v * 0.5 + 0.5) * 255.0).round() as u8;
    let pixel = [code(0.0), code(-lean.sin()), code(lean.cos()), 255];
    let settings = Settings::Texture(TextureSettings {
        usage: Usage::Normal,
        compress: true,
        max_size: 0,
    });
    let inputs = [("lean_normal.png".to_owned(), png(pixel))];
    match asset_db::import::import(
        AssetKind::Texture,
        "lean_normal.png",
        &settings,
        target,
        &inputs,
    )
    .unwrap()
    {
        Artifact::Texture(texture) => texture,
        Artifact::Mesh(_) => panic!("a texture"),
    }
}

/// BC5 keeps a normal map's X and Y and nothing else; the shaders rebuild Z.
/// The same normals drawn from a BC5 chain and from an RGBA8 one have to shade
/// alike, and unlike no normal map at all — the second half is what shows the
/// map is being read.
#[test]
#[ignore = "needs a GPU"]
fn a_compressed_normal_map_shades_as_its_normals() {
    orrin_core::logging::init(orrin_core::logging::LogConfig::for_this_build());
    let settings = CaptureSettings {
        extent: [128, 128],
        camera: Some(Camera {
            position: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            ..Camera::default()
        }),
        ..CaptureSettings::default()
    };
    let mut renderer = capture::offscreen_renderer(settings.extent);
    if !renderer.samples_bc() {
        println!("the device does not sample BC; nothing to check");
        return;
    }
    let mut world = World::new();
    capture::prepare(&mut world, &mut renderer, &settings);
    // Exposure held still: metered, it adapts from one draw to the next, and
    // the draws below would differ by that alone.
    let hdr = HdrSettings {
        auto_exposure: false,
        exposure_compensation: 3.0,
        ..*world.resource::<HdrSettings>()
    };
    world.insert_resource(hdr);
    let drawn: Vec<_> = world
        .entities()
        .filter(|&entity| world.has::<MeshHandle>(entity))
        .collect();
    for entity in drawn {
        world.despawn(entity);
    }

    let dir = Dir::new("normal");
    dir.write("quad.gltf", quad_gltf(0.0).as_bytes());
    dir.write("quad.bin", &quad_bin());
    let (db, issues) = AssetDatabase::open_for(dir.path(), dir.path().join(".cache"), Target::Bc);
    assert!(issues.is_empty(), "{issues:?}");
    let quad = db.find("quad.gltf").unwrap().id;
    world.insert_resource(db);
    let mesh = world.resource_mut::<AssetDatabase>().mesh(quad).unwrap();

    let mut material_with = |normal: Option<&TextureLevels>| {
        let normal_texture = normal.map(|levels| {
            let handle = renderer.load_texture(&[128, 128, 255, 255], 1, 1, false);
            renderer.replace_texture(handle, levels);
            handle
        });
        let material = Material {
            base_color: Vec3::ONE,
            metallic: 0.0,
            roughness: 0.5,
            normal_texture,
            ..Material::default()
        };
        let handle: MaterialHandle = renderer.load_material(&material);
        world
            .resource_mut::<MaterialBlends>()
            .insert(handle, material.blend);
        world
            .resource_mut::<AssetPreviews>()
            .insert_material(handle, &material, None);
        handle
    };
    let bc5 = leaning_normal_map(Target::Bc);
    assert_eq!(bc5.format, TexelFormat::Bc5Unorm);
    let rgba8 = leaning_normal_map(Target::Rgba8);
    let flat = material_with(None);
    let compressed = material_with(Some(&bc5));
    let plain = material_with(Some(&rgba8));

    let entity = world.spawn();
    world.insert(entity, LocalTransform::new(Transform::default()));
    world.insert(entity, mesh);
    world.insert(entity, flat);
    between_frames(&mut world, &mut renderer);
    assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
    between_frames(&mut world, &mut renderer);

    let mut draw = |material: MaterialHandle| {
        world.insert(entity, material);
        centre(&capture::render(&mut world, &mut renderer, &settings))
    };
    let flat = draw(flat);
    let compressed = draw(compressed);
    let plain = draw(plain);
    println!("flat {flat:?}, BC5 {compressed:?}, RGBA8 {plain:?}");

    let differs = |a: [u8; 3], b: [u8; 3]| (0..3).map(|i| a[i].abs_diff(b[i])).max().unwrap();
    assert!(
        differs(plain, flat) >= 8,
        "the leaning normals change the shading: {plain:?} against flat {flat:?}"
    );
    assert!(
        differs(compressed, plain) <= 3,
        "BC5 shades as RGBA8 does: {compressed:?} against {plain:?}"
    );
}

/// `brick.fmat`: a rough dielectric of `base_color`.
fn brick(base_color: Vec3) -> String {
    material::write(&material::MaterialFile {
        material: Material {
            base_color,
            metallic: 0.0,
            roughness: 1.0,
            ..Material::default()
        },
        textures: material::MaterialTextures::default(),
    })
}

/// The quad drawn with `brick.fmat` at `base_color`, bound, under exposure
/// held still for the reason it is in the normal-map test above.
struct MaterialRig {
    renderer: VulkanRenderer,
    world: World,
    settings: CaptureSettings,
    dir: Dir,
    entity: orrin_ecs::Entity,
    brick: orrin_registry::AssetId,
}

fn material_rig(name: &str, base_color: Vec3) -> MaterialRig {
    orrin_core::logging::init(orrin_core::logging::LogConfig::for_this_build());
    let settings = CaptureSettings {
        extent: [128, 128],
        camera: Some(Camera {
            position: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            ..Camera::default()
        }),
        ..CaptureSettings::default()
    };
    let mut renderer = capture::offscreen_renderer(settings.extent);
    let mut world = World::new();
    capture::prepare(&mut world, &mut renderer, &settings);
    let hdr = HdrSettings {
        auto_exposure: false,
        exposure_compensation: 3.0,
        ..*world.resource::<HdrSettings>()
    };
    world.insert_resource(hdr);
    let drawn: Vec<_> = world
        .entities()
        .filter(|&entity| world.has::<MeshHandle>(entity))
        .collect();
    for entity in drawn {
        world.despawn(entity);
    }

    let dir = Dir::new(name);
    dir.write("quad.gltf", quad_gltf(0.0).as_bytes());
    dir.write("quad.bin", &quad_bin());
    dir.write("brick.fmat", brick(base_color).as_bytes());
    let target = Target::of(&renderer);
    let (db, issues) = AssetDatabase::open_for(dir.path(), dir.path().join(".cache"), target);
    assert!(issues.is_empty(), "{issues:?}");
    let quad = db.find("quad.gltf").unwrap().id;
    let brick = db.find("brick.fmat").unwrap().id;
    world.insert_resource(db);
    let mesh = world.resource_mut::<AssetDatabase>().mesh(quad).unwrap();
    let pending = world
        .resource_mut::<AssetDatabase>()
        .material(brick)
        .unwrap();
    let entity = world.spawn();
    world.insert(entity, LocalTransform::new(Transform::default()));
    world.insert(entity, mesh);
    world.insert(entity, pending);

    between_frames(&mut world, &mut renderer);
    assert!(world.resource::<AssetDatabase>().wait_idle(WAIT));
    between_frames(&mut world, &mut renderer);
    MaterialRig {
        renderer,
        world,
        settings,
        dir,
        entity,
        brick,
    }
}

fn is_red([r, g, b]: [u8; 3]) -> bool {
    r > g + 40 && r > b + 40
}

fn is_green([r, g, b]: [u8; 3]) -> bool {
    g > r + 40 && g > b + 40
}

/// A saved `.fmat` reaches the material's slot in the renderer, so the next
/// frame shades with it, and the handle the entity holds stays the one it was
/// given.
#[test]
#[ignore = "needs a GPU"]
fn an_edited_material_file_reaches_the_next_frame() {
    let MaterialRig {
        mut renderer,
        mut world,
        settings,
        dir,
        entity,
        ..
    } = material_rig("material", Vec3::new(1.0, 0.0, 0.0));
    let slot = *world.get::<MaterialHandle>(entity).unwrap();

    let drawn = centre(&capture::render(&mut world, &mut renderer, &settings));
    assert!(is_red(drawn), "a red material draws red: {drawn:?}");

    edit(
        &mut world,
        &mut renderer,
        &dir,
        "brick.fmat",
        brick(Vec3::new(0.0, 1.0, 0.0)).as_bytes(),
    );
    let drawn = centre(&capture::render(&mut world, &mut renderer, &settings));
    assert!(is_green(drawn), "resaved green: {drawn:?}");
    assert_eq!(
        *world.get::<MaterialHandle>(entity).unwrap(),
        slot,
        "one handle throughout"
    );
}

/// The inspector's path: an edit through the undo pipeline is drawn by the
/// next frame and saved to the file, and its undo is drawn and saved too.
#[test]
#[ignore = "needs a GPU"]
fn an_inspector_edit_of_a_material_reaches_the_next_frame_and_its_file() {
    let MaterialRig {
        mut renderer,
        mut world,
        settings,
        dir,
        brick: id,
        ..
    } = material_rig("inspector", Vec3::new(1.0, 0.0, 0.0));
    let registry = {
        let mut registry = Registry::new();
        register_components(&mut registry);
        registry
    };
    let mut history = History::default();
    let drawn = centre(&capture::render(&mut world, &mut renderer, &settings));
    assert!(is_red(drawn), "a red material draws red: {drawn:?}");

    let mut green = *world.resource::<AssetDatabase>().material_file(id).unwrap();
    green.material.base_color = Vec3::new(0.0, 1.0, 0.0);
    let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
    editor.set_material(id, green).unwrap();
    editor.try_commit().unwrap();
    between_frames(&mut world, &mut renderer);
    let drawn = centre(&capture::render(&mut world, &mut renderer, &settings));
    assert!(is_green(drawn), "edited green: {drawn:?}");
    let on_disk = || std::fs::read_to_string(dir.path().join("brick.fmat")).unwrap();
    assert_eq!(on_disk(), brick(Vec3::new(0.0, 1.0, 0.0)));

    SceneEditor::new(&mut world, &registry, &mut history)
        .undo()
        .unwrap();
    between_frames(&mut world, &mut renderer);
    let drawn = centre(&capture::render(&mut world, &mut renderer, &settings));
    assert!(is_red(drawn), "undone to red: {drawn:?}");
    assert_eq!(on_disk(), brick(Vec3::new(1.0, 0.0, 0.0)));
}
