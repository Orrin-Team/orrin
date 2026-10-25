//! An asset edited on disk reaches the screen through the real renderer.
//!
//! The asset database's own tests swap artifacts into a `HeadlessBackend`,
//! which counts replacements and keeps bounds but draws nothing. This is the
//! half they cannot see: that `VulkanRenderer::replace_texture` and
//! `replace_mesh` change what the next frame samples and rasterises, through
//! the arena, the bindless texture array and GPU culling, without the handle
//! an entity holds ever changing.
//!
//! `#[ignore]`d for the reason `offscreen.rs` is: it needs a GPU. Run it with:
//!
//! ```text
//! cargo test -p orrin-core --test asset_reload -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::time::Duration;

use glam::Vec3;
use orrin_core::asset_db::{self, AssetDatabase};
use orrin_core::capture::{self, CaptureSettings};
use orrin_core::gfx::vulkan::VulkanRenderer;
use orrin_core::gfx::{Material, RenderBackend};
use orrin_core::scene::{
    AssetPreviews, Camera, LocalTransform, MaterialBlends, MaterialHandle, MeshHandle, Transform,
    apply_material_edits,
};
use orrin_ecs::World;

const WAIT: Duration = Duration::from_secs(60);

/// A scratch directory, removed when dropped.
struct Dir(PathBuf);

impl Dir {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("orrin-asset-reload-{}", std::process::id()));
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

    let dir = Dir::new();
    dir.write("quad.gltf", quad_gltf(0.0).as_bytes());
    dir.write("quad.bin", &quad_bin());
    dir.write("paint.png", &png([255, 0, 0, 255]));
    let (db, issues) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
    assert!(issues.is_empty(), "{issues:?}");
    let quad = db.find("quad.gltf").unwrap().id;
    let paint = db.find("paint.png").unwrap().id;
    world.insert_resource(db);

    let white = Material {
        base_color: Vec3::ONE,
        metallic: 0.0,
        roughness: 1.0,
        ..Material::default()
    };
    let material: MaterialHandle = renderer.load_material(&white);
    world
        .resource_mut::<MaterialBlends>()
        .insert(material, white.blend);
    world
        .resource_mut::<AssetPreviews>()
        .insert_material(material, &white, None);
    let mesh = world.resource_mut::<AssetDatabase>().mesh(quad).unwrap();
    world
        .resource_mut::<AssetDatabase>()
        .assign_albedo(material, paint)
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
