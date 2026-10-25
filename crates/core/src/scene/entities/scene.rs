//! The feature rig: one readout per feature, laid out to be measured.
//!
//! Nothing here is composed to look like a place, and that is the point — every
//! object in this scene exists to say whether one thing is working, and it is
//! placed where that answer is legible rather than where it would be pretty.
//! Hence the grid of spinning cubes, the two roughness rows, the two masonry
//! panels that differ in a height map and nothing else, and the twilight that
//! keeps a sun, a lamp and a neon tube within a few stops of each other. The
//! offscreen captures frame specific coordinates in here, so moving an object is
//! moving a measurement.
//!
//! [`build_showcase_scene`](super::build_showcase_scene) is the other half: the
//! same materials arranged as somewhere.

use glam::Vec3;

use orrin_ecs::World;

use super::library::{SWEEP, WALL_TILE, load_library};
use super::textures::sky_equirect;
use super::{
    spawn_decal, spawn_directional_light, spawn_mesh, spawn_point_light, spawn_spot_light,
};
use crate::gfx::RenderBackend;
use crate::scene::{Assets, Camera, Decal, Spin, Transform};

const GRID: i32 = 10;
const SPACING: f32 = 2.0;

pub fn build_default_scene(world: &mut World, backend: &mut impl RenderBackend) {
    let (assets, mesh_bounds, material_blends, asset_previews) = load_library(backend);

    let cube = assets.mesh("cube").unwrap();
    let plane = assets.mesh("plane").unwrap();
    // The masonry panels only. Everything else here keeps the vertex-colour tint
    // it has always had, so every capture but the walls' is unchanged — and the
    // walls' is unchanged too, because the tint the mesh dropped is exactly the
    // factor `brick`'s `base_color` used to cancel.
    let plane_plain = assets.mesh("plane_plain").unwrap();
    let textured = assets.material("textured").unwrap();
    let rock = assets.material("rock").unwrap();
    let ground_material = assets.material("ground").unwrap();
    let glass = assets.material("glass").unwrap();
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
    world.insert_resource(material_blends);
    world.insert_resource(asset_previews);

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
    // grid alone barely shows it. The floor stays a cube even though `plane`
    // exists, so output is unchanged; the masonry panels below are what uses it.
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

    // Three panes at different depths, deliberately intersecting. Two things to
    // look for: the crossings have no seam, because nothing sorted them; and
    // nothing pops as the camera orbits, because there is no order to flip.
    for (index, (x, z, yaw)) in [
        (-3.0f32, 6.0f32, 0.0f32),
        (0.0, 7.5, 25.0),
        (3.0, 6.5, -20.0),
    ]
    .into_iter()
    .enumerate()
    {
        spawn_mesh(
            world,
            format!("Glass {index}"),
            Transform {
                translation: Vec3::new(x, 2.5, z),
                rotation: glam::Quat::from_rotation_y(yaw.to_radians()),
                scale: Vec3::new(4.0, 4.0, 0.1),
            },
            cube,
            glass,
        );
    }

    // A row of the extra lobes. Spheres, for the reason the roughness sweep
    // uses them: a lobe is a shape in angle, and a flat face samples one angle.
    //
    // Floating above the sweep rather than beside them, and that is not
    // decoration. At ground level this row sat behind two rows of
    // radius-2 spheres and was almost entirely hidden — a readout nobody can see
    // is not a readout. Up here each one is against the cube grid, which is what
    // the two refractive ones need behind them to refract at all, and clear of
    // the sweep's spheres by more than the two radii so nothing intersects.
    let row = [
        "clearcoat",
        "velvet",
        "brushed",
        "crystal",
        "frosted",
        "marble",
        "wax",
    ];
    let stride = 2.9;
    // Centred on the row's own length rather than on a constant, so adding a lobe
    // is one more name above instead of two numbers that have to agree.
    let offset = (row.len() - 1) as f32 * 0.5 * stride;
    for (index, name) in row.into_iter().enumerate() {
        let material = world.resource::<Assets>().material(name).unwrap();
        spawn_mesh(
            world,
            name,
            Transform {
                translation: Vec3::new(index as f32 * stride - offset, 6.0, 13.0),
                rotation: glam::Quat::IDENTITY,
                scale: Vec3::splat(1.2),
            },
            sphere,
            material,
        );
    }

    // The parallax A/B, one wall each, at the same angle rather than mirrored:
    // the two differ in the height map and nothing else, so anything that
    // separates them is this feature. Turned well away from the camera on
    // purpose, because a height field seen head-on is a normal map — the whole
    // effect is what the ray does on its way *across* the surface. Look for the
    // mortar sliding behind the blocks, and for a block's own edge cutting into
    // the course below it.
    //
    // Behind the grid, where they also do a second job: a wall gives the
    // reflections and the contact shadows something the sky is not.
    // Stacked rather than side by side, which is what makes the two comparable:
    // a view grazing enough to show the effect is a view along the wall, and two
    // panels beside each other would then sit at two distances and two angles.
    // One above the other, both are the same wall seen the same way, and the
    // horizontal seam between them is the only place the frame changes.
    //
    // A quad rather than a slab, and the reason is the parallax rather than the
    // polygon count. A cube's edge faces carry the whole brick tile squeezed into
    // their thickness, so the height field there is genuinely thirty centimetres
    // wide and five deep — the march reads that correctly and walks a quarter of
    // the map, which looks like the smear it is. A wall is a surface; giving it
    // one removes the faces that were never masonry.
    for (name, material, y) in [
        ("Wall (parallax)", "brick", 0.5f32),
        ("Wall (flat)", "brick_flat", 2.5),
    ] {
        let material = world.resource::<Assets>().material(material).unwrap();
        spawn_mesh(
            world,
            name,
            Transform {
                // The lower panel stands on the ground, whose top face is at -0.5.
                translation: Vec3::new(0.0, y, -13.0),
                // The quad lies in XZ facing up, so it is stood on edge first and
                // turned second. That lands `u` along +x and `v` up, which is the
                // orientation the blocks were generated for.
                rotation: glam::Quat::from_rotation_y(25.0f32.to_radians())
                    * glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                // In the quad's own plane, so the metres the maps were generated
                // for: a block comes out half a metre square rather than whatever
                // the mesh's aspect makes of it.
                scale: Vec3::new(WALL_TILE[0], 1.0, WALL_TILE[1]),
            },
            plane_plain,
            material,
        );
    }

    // A stand of leaf cards, and the readout for alpha to coverage.
    //
    // Crossed pairs rather than single quads, which is what foliage actually is
    // and is also the thing that would go wrong first: two cutouts intersecting
    // at ninety degrees write depth against each other, so any ordering mistake
    // shows as one card erasing the other. Nothing here is sorted and nothing
    // needs to be — this is the opaque queue.
    //
    // Look at the leaf edges against the sky. With the cutout drawn through the
    // alpha-to-coverage pipeline they resolve to the sixteen levels four samples
    // and a one-pixel ramp can express; a hard alpha test would give the same
    // silhouette in four hard steps. Look at the ground for the second half:
    // the shadows are leaf-shaped, which is the caster pipeline's alpha test and
    // not this one.
    // Outboard of the cube grid rather than standing in it, for two reasons that
    // are really one. The cards intersected the cubes, which is the sort of thing
    // a demo scene should not be showing anybody; and the sun throws their
    // shadows a metre or so back and to the left, which inside the grid means
    // onto other cubes. Out here the shadow lands on flat open ground, where the
    // caster pipeline's alpha test is the only thing deciding its shape.
    let foliage_material = world.resource::<Assets>().material("foliage").unwrap();
    for (index, (x, z, yaw)) in [
        (-11.5f32, 3.5f32, 10.0f32),
        (-10.4, 6.5, -35.0),
        (11.0, 4.0, -15.0),
        (12.1, 7.0, 40.0),
    ]
    .into_iter()
    .enumerate()
    {
        for (half, turn) in [(0usize, 0.0f32), (1, 90.0)] {
            spawn_mesh(
                world,
                format!("Foliage {index}.{half}"),
                Transform {
                    // The quad lies in XZ facing up, so it is stood on edge and
                    // then turned, exactly as the masonry panels are.
                    translation: Vec3::new(x, 1.6, z),
                    rotation: glam::Quat::from_rotation_y((yaw + turn).to_radians())
                        * glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                    scale: Vec3::splat(3.2),
                },
                plane,
                foliage_material,
            );
        }
    }

    // Three decals, and each is a different half of the claim.
    //
    // The two on the ground are aimed straight down and are deep enough to reach
    // it from above; the third is aimed at the masonry wall along its own normal.
    // What to look for, in order: the crater's rim catches the sun and the point
    // lights, because a decal lands *before* shading rather than over it; the
    // ground decals stop at the cubes standing in them instead of running up
    // their sides, which is the angle fade; and the wall decal sits in the
    // parallax-mapped brick without sliding against it as the camera moves,
    // because the height march has already chosen where this pixel's surface is
    // by the time the decal is projected onto it.
    let scorch_albedo = world.resource::<Assets>().texture("scorch_albedo").unwrap();
    let scorch_normal = world.resource::<Assets>().texture("scorch_normal").unwrap();
    let scorch_rough = world.resource::<Assets>().texture("scorch_rough").unwrap();
    let scorch_decal = Decal {
        albedo: Some(scorch_albedo),
        normal: Some(scorch_normal),
        metallic_roughness: Some(scorch_rough),
        affects_surface: true,
        // Well under a right angle, which is what keeps these off the sides of
        // the cubes their boxes swallow. Turn it up to 90 and the streaks it
        // exists to prevent come straight back.
        angle: 55.0,
        ..Decal::default()
    };
    // On the open ground outboard of the sphere rows rather than inside the cube
    // grid, and that is a lesson rather than a preference: put a five-metre decal
    // among cubes on two-metre centres and what reaches the camera is a dozen
    // slivers of ground between them. Enough to prove the projection works and
    // useless as a photograph of it.
    for (index, (x, z, size)) in [(-10.0f32, 10.0f32, 5.0f32), (9.5, 8.0, 3.5)]
        .into_iter()
        .enumerate()
    {
        spawn_decal(
            world,
            format!("Scorch {index}"),
            Transform {
                // The ground's top face is at -0.5; the box is `size * 0.4` deep
                // along its projection axis, so it reaches from above it.
                translation: Vec3::new(x, -0.5, z),
                // Forward is -Z, so a quarter turn back about X aims it at the
                // floor.
                rotation: glam::Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                scale: Vec3::new(size, size, size * 0.4),
            },
            scorch_decal,
        );
    }
    spawn_decal(
        world,
        "Scorch (wall)",
        Transform {
            translation: Vec3::new(-0.8, 1.1, -12.7),
            // The lower masonry panel's own normal: the same yaw the wall was
            // turned by, which points this decal's +Z back out of it and so
            // projects along -Z into it.
            rotation: glam::Quat::from_rotation_y(25.0f32.to_radians()),
            scale: Vec3::new(1.6, 1.6, 0.8),
        },
        Decal {
            // Half strength, so the brick under it is plainly still there — a
            // decal that replaced the surface would prove nothing about landing
            // under the lighting.
            opacity: 0.75,
            ..scorch_decal
        },
    );

    let sun_dir = Vec3::new(-0.4, -1.0, -0.6).normalize();
    // Deep twilight, and chosen for that rather than inherited. Physical units put
    // the sun, the sky and every fixture on one scale, and under a 100 000 lux
    // noon sun a real lamp lands a fraction of a percent above the ambient — which
    // is correct, and is also a demo where the punctual lights, their shadows and
    // the emissive material are all invisible. Lighting the scene at the hour when
    // a sun, a lamp and a neon tube are within a few stops of each other is what
    // keeps every feature on screen without any of them lying about its units.
    spawn_directional_light(world, "Sun", sun_dir, Vec3::new(1.0, 0.97, 0.92), 250.0);

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
        // 8 000 lumens is a large architectural fixture rather than a bulb,
        // because these sit three metres from what they light: through 4pi
        // steradian and an inverse square, that is about 70 lux where it lands,
        // a quarter of the sun. Inverse-square is unforgiving about distance in
        // a way the old unitless 3.0 hid.
        spawn_point_light(world, format!("Point Light {i}"), pos, color, 8000.0, 10.0);
    }

    // Aimed down at the grid from one corner, which is where a cone's shadow
    // reads: the cubes are far enough apart that each throws its own onto the
    // floor rather than into its neighbour.
    spawn_spot_light(
        world,
        "Spot Light",
        Vec3::new(-6.0, 9.0, 6.0),
        Vec3::new(0.55, -1.0, -0.55).normalize(),
        Vec3::new(1.0, 0.9, 0.75),
        // Nine metres up, so the beam arrives at roughly the sun's own
        // illuminance and reads as a lit cone rather than a clipped highlight — a
        // shadow inside a blown-out region is a shadow nobody can see. The
        // reflector is what makes 10 000 lumens reach that far: the 32 degree cone
        // below concentrates them into about 10 000 candela, where the same power
        // from a bare bulb would be 800.
        10_000.0,
        30.0,
        22.0,
        32.0,
    );

    let span = GRID as f32 * SPACING;
    world.insert_resource(Camera {
        position: Vec3::new(0.0, span * 0.6, span * 1.1),
        target: Vec3::ZERO,
        ..Camera::default()
    });
}
