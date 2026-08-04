use glam::Vec3;

use orrin_ecs::World;

use super::textures::{bump_normals, checkerboard, load_rgba, metallic_roughness, sky_equirect};
use super::{spawn_directional_light, spawn_mesh, spawn_point_light};
use crate::gfx::{Material, RenderBackend};
use crate::scene::{Assets, Camera, CpuMesh, MeshBounds, Spin, Transform};

const GRID: i32 = 10;
const SPACING: f32 = 2.0;

/// Spheres per roughness row. Five puts a sample at 0, 0.25, 0.5, 0.75 and 1,
/// which covers the prefiltered specular chain's six levels closely enough that
/// a bad level shows up as one sphere that does not belong in the row.
const SWEEP: usize = 5;

pub fn build_default_scene(world: &mut World, backend: &mut impl RenderBackend) {
    let (assets, mesh_bounds) = load_assets(backend);

    let cube = assets.mesh("cube").unwrap();
    let plane = assets.mesh("plane").unwrap();
    let textured = assets.material("textured").unwrap();
    let rock = assets.material("rock").unwrap();
    let ground_material = assets.material("ground").unwrap();
    let palette: Vec<_> = ["gold", "copper", "glossy", "clay", "neon"]
        .into_iter()
        .map(|name| assets.material(name).unwrap())
        .collect();
    let sphere = assets.mesh("sphere").unwrap();
    let sweep: Vec<_> = (0..SWEEP)
        .map(|step| {
            let percent = step * 100 / (SWEEP - 1);
            (
                assets.material(&format!("metal_{percent:03}")).unwrap(),
                assets
                    .material(&format!("dielectric_{percent:03}"))
                    .unwrap(),
            )
        })
        .collect();

    world.insert_resource(assets);
    world.insert_resource(mesh_bounds);

    let half = (GRID - 1) as f32 * SPACING * 0.5;
    let mut index = 0;
    for x in 0..GRID {
        for z in 0..GRID {
            let pos = Vec3::new(x as f32 * SPACING - half, 0.0, z as f32 * SPACING - half);

            let material = match (x + z) % 3 {
                0 => rock,
                1 => textured,
                _ => palette[(x + z) as usize % palette.len()],
            };

            let entity = spawn_mesh(
                world,
                format!("Cube {index}"),
                Transform::from_translation(pos),
                cube,
                material,
            );
            let speed = 0.5 + ((x + z) % 5) as f32 * 0.4;
            world.insert(entity, Spin::new(Vec3::Y, speed));
            index += 1;
        }
    }

    // A flattened cube gives SSAO real contact surfaces to darken; the floating
    // grid alone barely shows it. The real `plane` mesh is registered for the
    // editor, but the demo floor stays a cube so output is unchanged.
    let _ = plane;
    spawn_mesh(
        world,
        "Ground",
        Transform {
            translation: Vec3::new(0.0, -0.75, 0.0),
            scale: Vec3::new(
                GRID as f32 * SPACING * 1.5,
                0.5,
                GRID as f32 * SPACING * 1.5,
            ),
            ..Default::default()
        },
        cube,
        ground_material,
    );

    // In front of the grid and clear of the ground, so each sphere sees sky,
    // ground and cubes at once — the three things a reflection has to get
    // right, and the fastest way to spot one that does not.
    let stride = 3.0;
    let offset = (SWEEP - 1) as f32 * stride * 0.5;
    for (step, (metal, dielectric)) in sweep.iter().enumerate() {
        let x = step as f32 * stride - offset;
        let percent = step * 100 / (SWEEP - 1);
        for (name, material, z) in [
            (format!("Metal {percent}%"), *metal, 11.0),
            (format!("Dielectric {percent}%"), *dielectric, 14.0),
        ] {
            spawn_mesh(
                world,
                name,
                Transform {
                    translation: Vec3::new(x, 2.5, z),
                    scale: Vec3::splat(2.0),
                    ..Default::default()
                },
                sphere,
                material,
            );
        }
    }

    let sun_dir = Vec3::new(-0.4, -1.0, -0.6).normalize();
    spawn_directional_light(world, "Sun", sun_dir, Vec3::new(1.0, 0.97, 0.92), 1.0);

    // Fed the direction *toward* the sun, so the disc in the sky and the
    // directional light that casts the shadows agree.
    const SKY: [u32; 2] = [1024, 512];
    backend.load_environment(&sky_equirect(SKY[0], SKY[1], -sun_dir), SKY[0], SKY[1]);

    for (i, (pos, color)) in [
        (Vec3::new(-4.0, 3.0, -4.0), Vec3::new(1.0, 0.35, 0.1)),
        (Vec3::new(4.0, 3.0, 4.0), Vec3::new(0.2, 0.5, 1.0)),
        (Vec3::new(4.0, 3.0, -4.0), Vec3::new(0.2, 1.0, 0.4)),
    ]
    .into_iter()
    .enumerate()
    {
        // Fill lights from when ambient was a flat 0.15 and everything not
        // facing the sun was nearly black. With an environment supplying real
        // irradiance they are decoration rather than fill, and at their old
        // strength they washed the nearby cubes toward their own hues.
        spawn_point_light(world, format!("Point Light {i}"), pos, color, 3.0, 10.0);
    }

    let span = GRID as f32 * SPACING;
    world.insert_resource(Camera {
        position: Vec3::new(0.0, span * 0.6, span * 1.1),
        target: Vec3::ZERO,
        ..Camera::default()
    });
}

fn load_assets(backend: &mut impl RenderBackend) -> (Assets, MeshBounds) {
    let mut assets = Assets::new();
    let mut bounds = MeshBounds::default();

    load_mesh(backend, &mut assets, &mut bounds, "cube", &CpuMesh::cube());
    load_mesh(
        backend,
        &mut assets,
        &mut bounds,
        "plane",
        &CpuMesh::plane(),
    );
    load_mesh(
        backend,
        &mut assets,
        &mut bounds,
        "sphere",
        &CpuMesh::sphere(32, 16),
    );

    // A spread across the metallic-roughness range so the PBR BRDF is visible.
    let palette = [
        (
            "gold",
            Material {
                base_color: Vec3::new(1.0, 0.84, 0.40),
                metallic: 1.0,
                roughness: 0.18,
                ..Material::default()
            },
        ),
        (
            "copper",
            Material {
                base_color: Vec3::new(0.95, 0.64, 0.54),
                metallic: 1.0,
                roughness: 0.45,
                ..Material::default()
            },
        ),
        (
            "glossy",
            Material {
                base_color: Vec3::new(0.9, 0.9, 0.95),
                metallic: 0.0,
                roughness: 0.12,
                reflectance: 0.7,
                ..Material::default()
            },
        ),
        (
            "clay",
            Material {
                base_color: Vec3::splat(0.8),
                metallic: 0.0,
                roughness: 0.85,
                ..Material::default()
            },
        ),
        (
            "neon",
            Material {
                base_color: Vec3::splat(0.4),
                metallic: 0.0,
                roughness: 0.0,
                reflectance: 0.0,
                emissive: Vec3::new(8.0, 8.0, 8.0),
                ..Material::default()
            },
        ),
    ];
    for (name, material) in palette {
        assets.insert_material(name, backend.load_material(&material));
    }

    // A roughness sweep at both ends of the metallic range: the readout for
    // image-based lighting. A cube face samples essentially one direction and
    // says almost nothing about a reflection; a sphere shows the whole
    // environment at once, and a row of them shows every level of the
    // prefiltered chain side by side. Neutral base colours on purpose — a
    // tinted metal hides in its own colour what the environment is doing.
    for step in 0..SWEEP {
        let roughness = step as f32 / (SWEEP - 1) as f32;
        let percent = step * 100 / (SWEEP - 1);

        assets.insert_material(
            format!("metal_{percent:03}"),
            backend.load_material(&Material {
                base_color: Vec3::new(0.95, 0.93, 0.88),
                metallic: 1.0,
                roughness,
                ..Material::default()
            }),
        );
        assets.insert_material(
            format!("dielectric_{percent:03}"),
            backend.load_material(&Material {
                base_color: Vec3::splat(0.5),
                metallic: 0.0,
                roughness,
                reflectance: 0.5,
                ..Material::default()
            }),
        );
    }

    let tex = 256;
    let albedo = backend.load_texture(
        &checkerboard(tex, 8, [220, 60, 50], [240, 240, 245]),
        tex,
        tex,
        true, // color map: sRGB
    );
    let normal = backend.load_texture(&bump_normals(tex, 6.0, 1.5), tex, tex, false);
    let metal_rough = backend.load_texture(&metallic_roughness(tex), tex, tex, false);
    assets.insert_texture("proc_albedo", albedo);
    assets.insert_texture("proc_normal", normal);
    assets.insert_texture("proc_metal_rough", metal_rough);
    assets.insert_material(
        "textured",
        backend.load_material(&Material {
            base_color: Vec3::ONE,
            metallic: 1.0,
            roughness: 1.0, // both scaled by the metallic-roughness map
            albedo_texture: Some(albedo),
            normal_texture: Some(normal),
            metallic_roughness_texture: Some(metal_rough),
            ..Material::default()
        }),
    );

    let (px, w, h) = load_rgba(include_bytes!("../../assets/Rocks016_1K-JPG_Color.jpg"));
    let rock_albedo = backend.load_texture(&px, w, h, true);
    let (px, w, h) = load_rgba(include_bytes!("../../assets/Rocks016_1K-JPG_NormalDX.jpg"));
    let rock_normal = backend.load_texture(&px, w, h, false);
    let (px, w, h) = load_rgba(include_bytes!("../../assets/Rocks016_1K-JPG_Roughness.jpg"));
    let rock_rough = backend.load_texture(&px, w, h, false);
    assets.insert_texture("rock_albedo", rock_albedo);
    assets.insert_texture("rock_normal", rock_normal);
    assets.insert_texture("rock_rough", rock_rough);
    assets.insert_material(
        "rock",
        backend.load_material(&Material {
            base_color: Vec3::ONE,
            metallic: 0.0,  // no metallic map; rock is a dielectric
            roughness: 1.0, // driven by the roughness map (green channel)
            albedo_texture: Some(rock_albedo),
            normal_texture: Some(rock_normal),
            metallic_roughness_texture: Some(rock_rough),
            ..Material::default()
        }),
    );

    assets.insert_material(
        "ground",
        backend.load_material(&Material {
            base_color: Vec3::splat(0.7),
            metallic: 0.0,
            roughness: 0.9,
            ..Material::default()
        }),
    );

    (assets, bounds)
}

/// Upload a mesh and register it in both world-side tables at once, so a mesh
/// can never reach a draw list without the bounds the culler needs.
fn load_mesh(
    backend: &mut impl RenderBackend,
    assets: &mut Assets,
    bounds: &mut MeshBounds,
    name: &str,
    mesh: &CpuMesh,
) {
    let handle = backend.load_mesh(mesh);
    assets.insert_mesh(name, handle);
    // Read back from the backend's own upload, so the box culling tests is the
    // box the drawn geometry occupies.
    if let Some(aabb) = backend.mesh_bounds(handle) {
        bounds.insert(handle, aabb);
    }
}
