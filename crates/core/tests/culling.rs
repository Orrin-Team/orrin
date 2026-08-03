//! Frustum culling over a scene laid out like the demo's, which is where the
//! numbers in the editor's Performance panel come from.
//!
//! The demo's opening camera frames the whole grid, so it culls nothing — an
//! easy result to mistake for culling being broken. These lock in both halves:
//! nothing dropped when everything is on screen, everything dropped when it
//! isn't.

use glam::Vec3;
use orrin_core::gfx::RenderItem;
use orrin_core::scene::{
    Camera, CpuMesh, Culling, LocalTransform, MeshBounds, MeshHandle, Transform,
};
use orrin_core::systems::{extract_renderables, propagate_transforms};
use orrin_ecs::World;

const GRID: i32 = 10;
const SPACING: f32 = 2.0;
const ASPECT: f32 = 16.0 / 9.0;
const CUBE: MeshHandle = MeshHandle(0);
/// The grid plus its ground slab.
const RENDERABLES: usize = (GRID * GRID) as usize + 1;

fn demo_world() -> World {
    let mut world = World::new();
    let mut bounds = MeshBounds::default();
    bounds.insert(CUBE, CpuMesh::cube().bounds());
    world.insert_resource(bounds);
    world.insert_resource(Culling::default());

    let half = (GRID - 1) as f32 * SPACING * 0.5;
    for x in 0..GRID {
        for z in 0..GRID {
            let pos = Vec3::new(x as f32 * SPACING - half, 0.0, z as f32 * SPACING - half);
            world
                .spawn_entity()
                .with(LocalTransform::from(Transform::from_translation(pos)))
                .with(CUBE);
        }
    }
    world
        .spawn_entity()
        .with(LocalTransform::from(Transform {
            translation: Vec3::new(0.0, -0.75, 0.0),
            scale: Vec3::new(
                GRID as f32 * SPACING * 1.5,
                0.5,
                GRID as f32 * SPACING * 1.5,
            ),
            ..Default::default()
        }))
        .with(CUBE);
    world
}

/// The camera `build_default_scene` opens with.
fn demo_camera() -> Camera {
    let span = GRID as f32 * SPACING;
    Camera {
        position: Vec3::new(0.0, span * 0.6, span * 1.1),
        target: Vec3::ZERO,
        ..Camera::default()
    }
}

/// Propagates before extracting, exactly as the frame does — extraction reads
/// world transforms and only propagation produces them.
fn visible(world: &mut World, camera: Camera) -> usize {
    *world.resource_mut::<Camera>() = camera;
    propagate_transforms(world);
    let mut items: Vec<RenderItem> = Vec::new();
    extract_renderables(world, ASPECT, &mut items);
    items.len()
}

#[test]
fn the_demo_camera_frames_the_whole_scene() {
    let mut world = demo_world();
    world.insert_resource(demo_camera());

    assert_eq!(visible(&mut world, demo_camera()), RENDERABLES);
}

#[test]
fn turning_the_camera_around_culls_everything() {
    let mut world = demo_world();
    let camera = demo_camera();
    world.insert_resource(camera);

    // Look the other way: the grid is now entirely behind the near plane.
    let behind = Camera {
        target: camera.position * 2.0,
        ..camera
    };
    assert_eq!(visible(&mut world, behind), 0);
}

#[test]
fn standing_inside_the_grid_culls_what_is_behind_you() {
    let mut world = demo_world();
    world.insert_resource(demo_camera());

    let inside = Camera {
        position: Vec3::new(0.0, 1.0, 0.0),
        target: Vec3::new(100.0, 1.0, 0.0),
        ..Camera::default()
    };
    let count = visible(&mut world, inside);
    assert!(
        count > 0 && count < RENDERABLES,
        "facing one way inside the grid should drop roughly the half behind the camera, got {count}"
    );
}

/// Distance alone must not cull: the scene stays whole until it passes the far
/// plane, however small it gets on screen.
#[test]
fn a_distant_scene_is_still_drawn() {
    let mut world = demo_world();
    world.insert_resource(demo_camera());

    let far_back = Camera {
        position: Vec3::new(0.0, 5.0, 400.0),
        target: Vec3::ZERO,
        ..Camera::default()
    };
    assert_eq!(visible(&mut world, far_back), RENDERABLES);
}
