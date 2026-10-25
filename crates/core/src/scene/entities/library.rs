//! The demo asset library: every mesh, texture and material either built-in
//! scene draws, loaded once from one place.
//!
//! Both scenes read this same table on purpose. The rig
//! ([`build_default_scene`](super::build_default_scene)) and the courtyard
//! ([`build_showcase_scene`](super::build_showcase_scene)) are two arrangements
//! of one material set, so a material authored wrong is wrong in both — and a
//! change made to please a beauty shot cannot quietly disagree with the A/B
//! capture that measures the same lobe. Each scene draws a subset; a handful of
//! material-table entries nothing in this run draws costs a struct in a buffer
//! and no bandwidth at all.

use glam::Vec3;

use super::textures::{
    brick, bump_normals, checkerboard, foliage, load_rgba, metallic_roughness, scorch,
};
use crate::gfx::{BlendMode, Material, RenderBackend};
use crate::scene::{AssetPreviews, Assets, CpuMesh, MaterialBlends, MeshBounds};

/// The masonry panel the parallax A/B is drawn on: how many texels its maps get,
/// the metres of wall they are stretched over, and how deep the mortar sits
/// behind the blocks.
///
/// Five centimetres is a deeply raked joint — the deep end of what real masonry
/// does, and chosen for that: at a joint's usual centimetre the effect is
/// correct and almost invisible at any angle a capture can be taken from, and a
/// readout nobody can see is not a readout.
///
/// Module-level because the generator and every mesh drawn with the result have
/// to be told the same thing. `brick` derives the normals from the depth over the
/// tile, the material hands the same depth to the march, and each panel's
/// transform scales to the same tile — three readers of one set of numbers, which
/// is the only way the blocks that are lit and the blocks that are marched are
/// the same blocks.
pub const WALL: u32 = 512;
pub const WALL_TILE: [f32; 2] = [4.0, 2.0];
pub const WALL_DEPTH: f32 = 0.05;

/// Spheres per roughness row. Five puts a sample at 0, 0.25, 0.5, 0.75 and 1,
/// which covers the prefiltered specular chain's six levels closely enough that
/// a bad level shows up as one sphere that does not belong in the row.
pub const SWEEP: usize = 5;

pub fn load_library(
    backend: &mut impl RenderBackend,
) -> (Assets, MeshBounds, MaterialBlends, AssetPreviews) {
    let mut assets = Assets::new();
    let mut bounds = MeshBounds::default();
    let mut blends = MaterialBlends::default();
    let mut previews = AssetPreviews::default();

    load_mesh(
        backend,
        &mut assets,
        &mut bounds,
        &mut previews,
        "cube",
        &CpuMesh::cube(),
    );
    load_mesh(
        backend,
        &mut assets,
        &mut bounds,
        &mut previews,
        "plane",
        &CpuMesh::plane(),
    );
    load_mesh(
        backend,
        &mut assets,
        &mut bounds,
        &mut previews,
        "sphere",
        &CpuMesh::sphere(32, 16),
    );

    // The same three with white vertex colours — see [`CpuMesh::uncolored`]. The
    // rig wants the tint and the courtyard cannot have it, so the choice is a
    // mesh rather than a material: a cube's six faces carry six tints and no
    // single `base_color` cancels them.
    for (name, mesh) in [
        ("cube_plain", CpuMesh::cube()),
        ("plane_plain", CpuMesh::plane()),
        ("sphere_plain", CpuMesh::sphere(32, 16)),
    ] {
        load_mesh(
            backend,
            &mut assets,
            &mut bounds,
            &mut previews,
            name,
            &mesh.uncolored(),
        );
    }

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
                // Nits, like the frame, because emission is a luminance and the
                // frame is measured in luminance. The lit floor sits near 80, so
                // this is three stops above white: bright enough to feed the bloom
                // chain, not so bright that it is the only thing metering sees.
                emissive: Vec3::splat(600.0),
                ..Material::default()
            },
        ),
    ];
    for (name, material) in palette {
        load_material(
            backend,
            &mut assets,
            &mut blends,
            &mut previews,
            name,
            &material,
        );
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

        load_material(
            backend,
            &mut assets,
            &mut blends,
            &mut previews,
            format!("metal_{percent:03}"),
            &Material {
                base_color: Vec3::new(0.95, 0.93, 0.88),
                metallic: 1.0,
                roughness,
                ..Material::default()
            },
        );
        load_material(
            backend,
            &mut assets,
            &mut blends,
            &mut previews,
            format!("dielectric_{percent:03}"),
            &Material {
                base_color: Vec3::splat(0.5),
                metallic: 0.0,
                roughness,
                reflectance: 0.5,
                ..Material::default()
            },
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
    load_material(
        backend,
        &mut assets,
        &mut blends,
        &mut previews,
        "textured",
        &Material {
            base_color: Vec3::ONE,
            metallic: 1.0,
            roughness: 1.0, // both scaled by the metallic-roughness map
            albedo_texture: Some(albedo),
            normal_texture: Some(normal),
            metallic_roughness_texture: Some(metal_rough),
            ..Material::default()
        },
    );

    // The cutout, and the one material in the demo whose alpha means shape
    // rather than opacity. Two-sided and alpha-tested rather than blended, so it
    // is in the prepass, in the caster lists, and lit like any other surface —
    // which is exactly what separates `Masked` from `Blend`.
    // Decal maps. No material and no `MaterialBlends` entry, because a decal is
    // not drawn: these are three texture indices a `Decal` component names, read
    // by the two passes that were rasterising the receiver anyway.
    const SCORCH: u32 = 256;
    let scorch_maps = scorch(SCORCH);
    let scorch_albedo = backend.load_texture(&scorch_maps.albedo, SCORCH, SCORCH, true);
    let scorch_normal = backend.load_texture(&scorch_maps.normal, SCORCH, SCORCH, false);
    let scorch_rough = backend.load_texture(&scorch_maps.metallic_roughness, SCORCH, SCORCH, false);
    assets.insert_texture("scorch_albedo", scorch_albedo);
    assets.insert_texture("scorch_normal", scorch_normal);
    assets.insert_texture("scorch_rough", scorch_rough);

    const LEAF: u32 = 256;
    let leaf_albedo = backend.load_texture(&foliage(LEAF), LEAF, LEAF, true);
    assets.insert_texture("leaf_albedo", leaf_albedo);
    load_material(
        backend,
        &mut assets,
        &mut blends,
        &mut previews,
        "foliage",
        &Material {
            base_color: Vec3::ONE,
            metallic: 0.0,
            roughness: 0.6,
            albedo_texture: Some(leaf_albedo),
            blend: BlendMode::Masked,
            alpha_cutoff: 0.5,
            // A leaf is the textbook case for the transmitted lobe, and it is
            // here for a second reason: scattering is the term that makes a
            // cutout's *interior* look like a leaf once the cutout has made its
            // outline one. Together they are what a card has instead of geometry.
            subsurface_color: Vec3::new(0.28, 0.55, 0.12),
            subsurface_radius: Vec3::new(0.010, 0.014, 0.004),
            thickness: 0.0004,
            ..Material::default()
        },
    );

    let (px, w, h) = load_rgba(include_bytes!("../../assets/Rocks016_1K-JPG_Color.jpg"));
    let rock_albedo = backend.load_texture(&px, w, h, true);
    let (px, w, h) = load_rgba(include_bytes!("../../assets/Rocks016_1K-JPG_NormalDX.jpg"));
    let rock_normal = backend.load_texture(&px, w, h, false);
    let (px, w, h) = load_rgba(include_bytes!("../../assets/Rocks016_1K-JPG_Roughness.jpg"));
    let rock_rough = backend.load_texture(&px, w, h, false);
    let (px, w, h) = load_rgba(include_bytes!(
        "../../assets/Rocks016_1K-JPG_Displacement.jpg"
    ));
    let rock_height = backend.load_texture(&px, w, h, false);
    assets.insert_texture("rock_albedo", rock_albedo);
    assets.insert_texture("rock_normal", rock_normal);
    assets.insert_texture("rock_rough", rock_rough);
    assets.insert_texture("rock_height", rock_height);
    load_material(
        backend,
        &mut assets,
        &mut blends,
        &mut previews,
        "rock",
        &Material {
            base_color: Vec3::ONE,
            metallic: 0.0,  // no metallic map; rock is a dielectric
            roughness: 1.0, // driven by the roughness map (green channel)
            albedo_texture: Some(rock_albedo),
            normal_texture: Some(rock_normal),
            metallic_roughness_texture: Some(rock_rough),
            // The displacement map that shipped beside the other three, finally
            // read. A cube face here is one metre and carries the whole tile, so
            // the map's relief is centimetres of stone — and the readout is the
            // cube's own edges: the rock now slides against them as the grid
            // spins, which is the one thing a normal map can never do.
            height_texture: Some(rock_height),
            parallax_depth: 0.03,
            ..Material::default()
        },
    );

    // The same three maps with the fourth left off, for the courtyard's walls,
    // lintels and plinths — which are cubes, and a metre or more on a side.
    //
    // Dropping the height map is the whole difference, and it is a geometry
    // argument rather than a cost one. A slab's *narrow* faces carry the entire
    // tile squeezed into their thickness, so on a half-metre edge the march is
    // reading a field genuinely stretched eight times, and walks most of the map
    // in one step: the smear that is the documented reason the rig's walls are
    // quads. A cut face of masonry has no relief to march anyway — that is what
    // makes it a cut face.
    load_material(
        backend,
        &mut assets,
        &mut blends,
        &mut previews,
        "stone",
        &Material {
            base_color: Vec3::new(0.86, 0.84, 0.80),
            metallic: 0.0,
            roughness: 1.0,
            albedo_texture: Some(rock_albedo),
            normal_texture: Some(rock_normal),
            metallic_roughness_texture: Some(rock_rough),
            ..Material::default()
        },
    );

    // What the courtyard's fire is made of. Emissive in cd/m², like `neon` and
    // like the frame: a few stops over the sunlit stone around it, which is what
    // feeds the bloom chain without becoming the only thing the histogram sees.
    //
    // It carries no light of its own — the point light beside it does that. An
    // emissive surface is a surface that is bright, not a fixture, and the two
    // being separate entities is what lets the fire cast a shadow.
    load_material(
        backend,
        &mut assets,
        &mut blends,
        &mut previews,
        "ember",
        &Material {
            base_color: Vec3::new(0.25, 0.10, 0.05),
            metallic: 0.0,
            roughness: 0.9,
            emissive: Vec3::new(2600.0, 900.0, 260.0),
            ..Material::default()
        },
    );

    // A masonry wall, and the A/B that says what the parallax march is doing.
    // Both materials below are the same three maps at the same depth of relief;
    // one of them marches and one does not, so any difference between the two
    // walls in the scene is this feature and nothing else.
    //
    // Blocks of 0.5 x 0.5 m over the panel they are drawn on, and a joint two
    // centimetres deep — which is a real masonry joint, and deep enough to see
    // into at the angle the wall is turned to.
    let bricks = brick(WALL, 8, 4, WALL_TILE, WALL_DEPTH);
    let brick_albedo = backend.load_texture(&bricks.albedo, WALL, WALL, true);
    let brick_normal = backend.load_texture(&bricks.normal, WALL, WALL, false);
    let brick_height = backend.load_texture(&bricks.height, WALL, WALL, false);
    assets.insert_texture("brick_albedo", brick_albedo);
    assets.insert_texture("brick_normal", brick_normal);
    assets.insert_texture("brick_height", brick_height);
    let brick_material = Material {
        // Neutral, and drawn on `plane_plain` in both scenes. This used to be
        // (2.0, 1.0, 2.0) to cancel the quad's own vertex colour — its normal, so
        // (0.5, 1.0, 0.5) — which is the same product by a longer route: a wall of
        // violet blocks is a poor photograph of a height field. The mesh carrying
        // no tint says that once instead of every material having to know about it.
        base_color: Vec3::ONE,
        metallic: 0.0,
        roughness: 0.85,
        albedo_texture: Some(brick_albedo),
        normal_texture: Some(brick_normal),
        ..Material::default()
    };
    load_material(
        backend,
        &mut assets,
        &mut blends,
        &mut previews,
        "brick",
        &Material {
            height_texture: Some(brick_height),
            // The same depth the maps were generated at, and the same constant
            // rather than a number that matches it today: a normal map whose
            // slopes belong to a deeper field than the one being marched lights
            // a groove that is not where it is drawn.
            parallax_depth: WALL_DEPTH,
            ..brick_material
        },
    );
    load_material(
        backend,
        &mut assets,
        &mut blends,
        &mut previews,
        "brick_flat",
        &brick_material,
    );

    load_material(
        backend,
        &mut assets,
        &mut blends,
        &mut previews,
        "ground",
        &Material {
            base_color: Vec3::splat(0.7),
            metallic: 0.0,
            roughness: 0.9,
            ..Material::default()
        },
    );

    // Three panes of the same glass, stacked in front of the camera below, so
    // the overlap is the readout: weighted-blended transparency is
    // order-independent, and where they cross there must be no seam and no
    // flicker as the camera moves around them.
    load_material(
        backend,
        &mut assets,
        &mut blends,
        &mut previews,
        "glass",
        &Material {
            base_color: Vec3::new(0.6, 0.85, 0.9),
            alpha: 0.35,
            blend: BlendMode::Blend,
            metallic: 0.0,
            roughness: 0.05,
            reflectance: 0.9,
            ..Material::default()
        },
    );

    // One material per extra lobe, each authored so the lobe is the only thing
    // it shows. A material that switched on two of them at once would be a
    // prettier sphere and a worse test — when it looked wrong there would be
    // nothing to say which lobe was wrong.
    for (name, material) in [
        // Car paint: a rough, dark base under a mirror film. The readout is that
        // the highlight is *two* highlights of different widths — a coat over a
        // base of the same roughness is just a brighter base.
        (
            "clearcoat",
            Material {
                base_color: Vec3::new(0.55, 0.05, 0.08),
                metallic: 0.0,
                roughness: 0.6,
                clearcoat: 1.0,
                clearcoat_roughness: 0.03,
                ..Material::default()
            },
        ),
        // Velvet. The readout is the rim: a sheen lobe peaks away from the
        // normal, so the silhouette is brighter than the centre — which no
        // amount of roughness on the base lobe can produce.
        (
            "velvet",
            Material {
                base_color: Vec3::new(0.14, 0.03, 0.10),
                metallic: 0.0,
                roughness: 0.95,
                sheen_color: Vec3::new(0.7, 0.25, 0.45),
                sheen_roughness: 0.25,
                ..Material::default()
            },
        ),
        // Brushed metal. The readout is that the highlight is a *streak* along
        // the tangent rather than a disc, and that it stays a streak in the
        // environment reflection — the bent normal in `ibl_reflection` is what
        // that second half tests.
        (
            "brushed",
            Material {
                base_color: Vec3::new(0.91, 0.92, 0.94),
                metallic: 1.0,
                roughness: 0.35,
                anisotropy: 0.85,
                ..Material::default()
            },
        ),
        // Solid glass, with a volume rather than a pane: thickness is what bends
        // the lookup at all, so a zero-thickness version of this refracts
        // nothing and only proves the queue runs. The green tint arrives through
        // Beer-Lambert over that thickness rather than through the base colour,
        // which is what makes the edges of the sphere darker than its middle.
        (
            "crystal",
            Material {
                base_color: Vec3::ONE,
                blend: BlendMode::Transmissive,
                metallic: 0.0,
                roughness: 0.05,
                transmission: 1.0,
                ior: 1.52,
                thickness: 0.9,
                attenuation_color: Vec3::new(0.72, 0.94, 0.82),
                attenuation_distance: 1.5,
                ..Material::default()
            },
        ),
        // The same glass roughened. Its background comes from a coarser level of
        // the scene pyramid, so this is the material that says whether that
        // chain was built at all — a frosted sphere with a sharp world behind it
        // means the roughness never reached the lookup.
        (
            "frosted",
            Material {
                base_color: Vec3::ONE,
                blend: BlendMode::Transmissive,
                metallic: 0.0,
                roughness: 0.45,
                transmission: 1.0,
                ior: 1.45,
                thickness: 0.6,
                ..Material::default()
            },
        ),
        // Marble: a solid body whose mean free path is millimetres against a
        // sphere that is metres. So the readout here is *not* light coming
        // through — `exp(-0.6 / 0.012)` is zero and nothing should glow. It is
        // the terminator: the line between lit and unlit goes soft and turns
        // faintly red on the dark side, because the light that crossed it
        // travelled through a little stone to get there. A hard grey terminator
        // means the scattering never reached the diffuse lobe.
        //
        // The thickness is honest rather than tuned. Authoring a thin marble
        // sphere to make it glow is exactly the mistake this material exists to
        // rule out.
        (
            "marble",
            Material {
                base_color: Vec3::new(0.86, 0.83, 0.78),
                metallic: 0.0,
                roughness: 0.22,
                subsurface_color: Vec3::new(0.92, 0.80, 0.72),
                subsurface_radius: Vec3::new(0.012, 0.006, 0.004),
                thickness: 0.6,
                ..Material::default()
            },
        ),
        // Wax, authored as a shell a few millimetres thick — a candle rather
        // than a block. Now the mean free path and the thickness are the same
        // order, so this is the material that shows the other half: the
        // transmitted lobe, warm and brightest where the surface turns away from
        // the sun and toward the camera. Beside the marble it is the A/B for
        // which half of the effect is working, since the two differ in almost
        // nothing but that one number.
        (
            "wax",
            Material {
                base_color: Vec3::new(0.93, 0.86, 0.72),
                metallic: 0.0,
                roughness: 0.35,
                subsurface_color: Vec3::new(1.0, 0.72, 0.42),
                subsurface_radius: Vec3::new(0.010, 0.006, 0.004),
                subsurface_forward_scatter: 8.0,
                thickness: 0.003,
                ..Material::default()
            },
        ),
    ] {
        load_material(
            backend,
            &mut assets,
            &mut blends,
            &mut previews,
            name,
            &material,
        );
    }

    (assets, bounds, blends, previews)
}

/// Upload a material and register it in every world-side table at once, so a
/// material can never reach a draw list without the blend mode extraction sorts
/// it by — a blended one registered nowhere would draw as an opaque wall.
fn load_material(
    backend: &mut impl RenderBackend,
    assets: &mut Assets,
    blends: &mut MaterialBlends,
    previews: &mut AssetPreviews,
    name: impl Into<String>,
    material: &Material,
) {
    let handle = backend.load_material(material);
    assets.insert_material(name, handle);
    blends.insert(handle, material.blend);
    previews.insert_material(handle, material, None);
}

/// Upload a mesh and register it in both world-side tables at once, so a mesh
/// can never reach a draw list without the bounds the culler needs.
fn load_mesh(
    backend: &mut impl RenderBackend,
    assets: &mut Assets,
    bounds: &mut MeshBounds,
    previews: &mut AssetPreviews,
    name: &str,
    mesh: &CpuMesh,
) {
    let handle = backend.load_mesh(mesh);
    assets.insert_mesh(name, handle);
    previews.insert_mesh(handle, mesh);
    // Read back from the backend's own upload, so the box culling tests is the
    // box the drawn geometry occupies.
    if let Some(aabb) = backend.mesh_bounds(handle) {
        bounds.insert(handle, aabb);
    }
}
