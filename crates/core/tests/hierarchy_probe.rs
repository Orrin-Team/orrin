//! The two systems the frame actually composes: `spin` writes local transforms,
//! `propagate_transforms` turns them into world ones. A child of a spinning
//! parent has to orbit it — which no unit test of either system alone can show.

use glam::{Vec3, Vec4Swizzles};
use orrin_core::scene::{LocalTransform, Spin, Transform, propagate_transforms, reparent};
use orrin_core::systems::spin;
use orrin_ecs::World;

fn world_position(world: &World, entity: orrin_ecs::Entity) -> Vec3 {
    world
        .get::<orrin_core::scene::WorldTransform>(entity)
        .unwrap()
        .0
        .w_axis
        .xyz()
}

#[test]
fn a_child_orbits_its_spinning_parent() {
    const RADIUS: f32 = 2.0;

    let mut world = World::new();
    let parent = world
        .spawn_entity()
        .with(LocalTransform::from(Transform::default()))
        .with(Spin::new(Vec3::Y, std::f32::consts::FRAC_PI_2))
        .id();
    let child = world
        .spawn_entity()
        .with(LocalTransform::from(Transform::from_translation(
            Vec3::X * RADIUS,
        )))
        .id();
    reparent(&mut world, child, Some(parent), false).unwrap();

    propagate_transforms(&mut world);
    let start = world_position(&world, child);
    assert!((start - Vec3::X * RADIUS).length() < 1e-5);

    // A quarter turn, in the order the frame runs them.
    let mut travelled = Vec3::ZERO;
    for _ in 0..60 {
        spin(&world, 1.0 / 60.0);
        propagate_transforms(&mut world);
        travelled = world_position(&world, child);
        assert!(
            (travelled.length() - RADIUS).abs() < 1e-3,
            "the child left its orbit: {travelled:?}"
        );
    }

    assert!(
        (travelled - start).length() > 1.0,
        "the child did not follow its parent's rotation: {start:?} -> {travelled:?}"
    );
    assert!(
        world_position(&world, parent).length() < 1e-5,
        "the parent moved when only its rotation should have changed"
    );
}
