//! Two rooms for the global illumination: what bounced light is for, and
//! where it goes wrong. Built for the captures, like the rig: each room is
//! one question, and is placed where the question can be asked.
//!
//! - A courtyard open to the sky and the sun, and behind a wall ten
//!   centimetres thick, a closed room lit by one lamp. A probe grid a metre
//!   apart straddles that wall, so this is where light leaks through it, if
//!   it does.
//! - A closed room with no light in it at all, only a glowing panel in its
//!   ceiling and a red wall and a green one. Everything in it is lit by the
//!   panel, through what the probes' rays find.
//!
//! The rooms are closed boxes of six slabs, so with the volumes off the sky's
//! diffuse light reaches into both as if they had no roof — which is the
//! frame the pair is taken against.

use glam::Vec3;

use orrin_ecs::World;

use super::library::{load_library, load_material};
use super::textures::sky_equirect;
use super::{spawn_directional_light, spawn_mesh, spawn_point_light};
use crate::gfx::{Material, RenderBackend};
use crate::scene::{Camera, EnvironmentSettings, MaterialHandle, MeshHandle, Transform};

/// Where the sun is, toward it: high and from the courtyard's open side.
const TO_SUN: Vec3 = Vec3::new(0.45, 0.85, 0.30);

pub fn build_rooms_scene(world: &mut World, backend: &mut impl RenderBackend) {
    let (mut assets, mesh_bounds, mut blends, mut previews) = load_library(backend);
    let cube = assets.mesh("cube_plain").unwrap();
    let mut material = |name: &str, material: Material| {
        load_material(
            backend,
            &mut assets,
            &mut blends,
            &mut previews,
            name,
            &material,
        );
        assets.material(name).unwrap()
    };
    let matte = |albedo: Vec3| Material {
        base_color: albedo,
        metallic: 0.0,
        roughness: 0.9,
        ..Material::default()
    };
    let plaster = material("plaster", matte(Vec3::splat(0.75)));
    let red = material("plaster_red", matte(Vec3::new(0.7, 0.08, 0.06)));
    let green = material("plaster_green", matte(Vec3::new(0.08, 0.6, 0.1)));
    let ground = material("paving", matte(Vec3::new(0.55, 0.5, 0.45)));
    // A softbox: about what a frosted ceiling panel shows lit from behind.
    let panel = material(
        "panel",
        Material {
            emissive: Vec3::new(2400.0, 2200.0, 1900.0),
            ..matte(Vec3::splat(0.9))
        },
    );
    world.insert_resource(assets);
    world.insert_resource(mesh_bounds);
    world.insert_resource(blends);
    world.insert_resource(previews);

    let slab = |world: &mut World, name: &str, min: Vec3, max: Vec3, material: MaterialHandle| {
        spawn_box(world, name, cube, min, max, material);
    };

    // The ground under everything, as deep as earth is: a slab with open air
    // under it would be one more thin wall. Then the courtyard, three walls
    // and no roof, on the far side of the thin wall.
    slab(
        world,
        "Ground",
        Vec3::new(-8.0, -4.0, -5.0),
        Vec3::new(18.0, 0.0, 5.0),
        ground,
    );
    slab(
        world,
        "Courtyard back",
        Vec3::new(-6.2, 0.0, -3.2),
        Vec3::new(0.0, 3.0, -3.0),
        plaster,
    );
    slab(
        world,
        "Courtyard side",
        Vec3::new(-6.2, 0.0, -3.0),
        Vec3::new(-6.0, 3.0, 3.0),
        plaster,
    );

    // The thin wall, and the closed room behind it: its own walls and a roof
    // over the thin wall too, so the wall's only way to the sky is through
    // itself. A domestic lamp, so there is something to see by.
    slab(
        world,
        "Thin wall",
        Vec3::new(-0.05, 0.0, -3.2),
        Vec3::new(0.05, 3.0, 3.2),
        plaster,
    );
    slab(
        world,
        "Room back",
        Vec3::new(0.05, 0.0, -3.2),
        Vec3::new(6.2, 3.0, -3.0),
        plaster,
    );
    slab(
        world,
        "Room front",
        Vec3::new(-6.2, 0.0, 3.0),
        Vec3::new(6.2, 3.0, 3.2),
        plaster,
    );
    slab(
        world,
        "Room end",
        Vec3::new(6.0, 0.0, -3.0),
        Vec3::new(6.2, 3.0, 3.0),
        plaster,
    );
    slab(
        world,
        "Room roof",
        Vec3::new(-0.05, 3.0, -3.2),
        Vec3::new(6.2, 3.2, 3.2),
        plaster,
    );
    spawn_point_light(
        world,
        "Lamp",
        Vec3::new(5.0, 2.4, -2.2),
        Vec3::new(1.0, 0.85, 0.65),
        1500.0,
        8.0,
    );

    // The room lit by its panel: no window, no lamp, and the ground for a floor.
    let (lo, hi) = (Vec3::new(8.0, 0.0, -3.0), Vec3::new(15.0, 3.0, 3.0));
    let wall = 0.2;
    slab(
        world,
        "Panel room roof",
        Vec3::new(lo.x - wall, hi.y, lo.z - wall),
        Vec3::new(hi.x + wall, hi.y + wall, hi.z + wall),
        plaster,
    );
    slab(
        world,
        "Panel room red",
        Vec3::new(lo.x - wall, 0.0, lo.z - wall),
        Vec3::new(lo.x, hi.y, hi.z + wall),
        red,
    );
    slab(
        world,
        "Panel room green",
        Vec3::new(hi.x, 0.0, lo.z - wall),
        Vec3::new(hi.x + wall, hi.y, hi.z + wall),
        green,
    );
    slab(
        world,
        "Panel room back",
        Vec3::new(lo.x, 0.0, lo.z - wall),
        Vec3::new(hi.x, hi.y, lo.z),
        plaster,
    );
    slab(
        world,
        "Panel room front",
        Vec3::new(lo.x, 0.0, hi.z),
        Vec3::new(hi.x, hi.y, hi.z + wall),
        plaster,
    );
    slab(
        world,
        "Panel",
        Vec3::new(10.75, 2.95, -0.75),
        Vec3::new(12.25, 3.0, 0.75),
        panel,
    );

    let sun = -TO_SUN.normalize();
    spawn_directional_light(world, "Sun", sun, Vec3::new(1.0, 0.95, 0.88), 60_000.0);
    const SKY: [u32; 2] = [1024, 512];
    backend.load_environment(&sky_equirect(SKY[0], SKY[1], -sun), SKY[0], SKY[1]);
    world.insert_resource(EnvironmentSettings {
        // A clear afternoon's sky, in cd/m2, beside a sun of sixty thousand lux.
        sky_luminance: 4000.0,
        ..EnvironmentSettings::default()
    });
    // Into the closed room, toward the thin wall.
    world.insert_resource(Camera {
        position: Vec3::new(5.2, 1.5, 2.2),
        target: Vec3::new(0.0, 1.3, -0.6),
        ..Camera::default()
    });
}

/// A slab of `cube` filling the box from `min` to `max`.
fn spawn_box(
    world: &mut World,
    name: &str,
    cube: MeshHandle,
    min: Vec3,
    max: Vec3,
    material: MaterialHandle,
) {
    spawn_mesh(
        world,
        name,
        Transform {
            translation: (min + max) * 0.5,
            scale: max - min,
            ..Default::default()
        },
        cube,
        material,
    );
}
