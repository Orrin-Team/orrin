use glam::{Quat, Vec3};
use orrin_ecs::Entity;

use super::*;
use crate::scene::{
    Collider, ColliderShape, CpuMesh, LocalTransform, LogBuffer, MeshBounds, MeshHandle,
    PhysicsSettings, RigidBody, Transform, propagate_transforms, reparent,
};

fn test_world() -> World {
    let mut world = World::new();
    world.insert_resource(CollisionState::default());
    world.insert_resource(LogBuffer::default());
    world.insert_resource(PhysicsState::default());
    world
}

fn spawn(
    world: &mut World,
    position: Vec3,
    shape: ColliderShape,
    body: Option<RigidBody>,
) -> Entity {
    let entity = world
        .spawn_entity()
        .with(LocalTransform::from(Transform::from_translation(position)))
        .with(Collider {
            shape,
            ..Collider::default()
        })
        .id();
    if let Some(body) = body {
        world.insert(entity, body);
    }
    entity
}

/// How far from the exact contact a resting body may settle: Jolt lets one
/// sink by its penetration slop, 0.02 m, rather than fight a contact to zero.
const RESTING: f32 = 0.03;

fn weightless() -> RigidBody {
    RigidBody {
        gravity_scale: 0.0,
        ..RigidBody::default()
    }
}

/// A floor whose top face is at y = 0.
fn floor(world: &mut World) -> Entity {
    spawn(
        world,
        Vec3::new(0.0, -0.5, 0.0),
        ColliderShape::Box {
            half_extents: Vec3::new(20.0, 0.5, 20.0),
        },
        None,
    )
}

fn cube(half: f32) -> ColliderShape {
    ColliderShape::Box {
        half_extents: Vec3::splat(half),
    }
}

/// What the app does each frame: propagate, then step.
fn frame(world: &mut World, delta: f32) {
    propagate_transforms(world);
    run(world, delta);
}

fn simulate(world: &mut World, seconds: f32, fps: f32) {
    let frames = (seconds * fps).round() as u32;
    for _ in 0..frames {
        frame(world, 1.0 / fps);
    }
}

fn position(world: &World, entity: Entity) -> Vec3 {
    world.get::<LocalTransform>(entity).unwrap().translation
}

fn events(world: &World) -> Vec<(CollisionEventKind, Entity, Entity)> {
    world
        .resource::<CollisionState>()
        .events
        .iter()
        .map(|event| (event.kind, event.a, event.b))
        .collect()
}

/// One second of free fall from rest is g/2 ≈ 4.9 m, give or take what
/// the integrator and Jolt's default damping take off it.
#[test]
fn a_rigid_body_falls_under_gravity() {
    let mut world = test_world();
    let ball = spawn(
        &mut world,
        Vec3::new(0.0, 10.0, 0.0),
        ColliderShape::Sphere { radius: 0.5 },
        Some(RigidBody::default()),
    );

    simulate(&mut world, 1.0, 60.0);

    let fallen = 10.0 - position(&world, ball).y;
    assert!((fallen - 4.9).abs() < 0.2, "fell {fallen} m in one second");
}

/// The plain collider is the floor: Jolt has to stop the cube on it, and
/// nothing may push the floor, since it has no rigid body.
#[test]
fn a_rigid_body_comes_to_rest_on_a_plain_collider() {
    let mut world = test_world();
    let floor = spawn(
        &mut world,
        Vec3::ZERO,
        ColliderShape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        },
        None,
    );
    let crate_ = spawn(
        &mut world,
        Vec3::new(0.0, 3.0, 0.0),
        cube(0.5),
        Some(RigidBody::default()),
    );

    simulate(&mut world, 3.0, 60.0);

    let resting = position(&world, crate_);
    assert!(
        (resting.y - 1.0).abs() < 0.05,
        "the cube settled at {resting}, not on the floor's top face"
    );
    assert_eq!(position(&world, floor), Vec3::ZERO, "the floor was pushed");
}

/// Jolt puts a resting body to sleep and keeps its contacts. Sleep is not
/// leaving, so a script must not be told it was.
#[test]
fn a_body_falling_asleep_on_the_floor_is_still_touching_it() {
    let mut world = test_world();
    let floor = spawn(
        &mut world,
        Vec3::ZERO,
        ColliderShape::Box {
            half_extents: Vec3::new(10.0, 0.5, 10.0),
        },
        None,
    );
    let resting = spawn(
        &mut world,
        Vec3::new(0.0, 1.0, 0.0),
        cube(0.5),
        Some(RigidBody::default()),
    );

    let mut seen = Vec::new();
    for _ in 0..180 {
        frame(&mut world, STEP);
        seen.extend(events(&world));
    }

    assert_eq!(seen, [(CollisionEventKind::Enter, floor, resting)]);
}

/// The step is fixed and the transform interpolated between steps, so a
/// faster display shows the same fall, not a different one.
#[test]
fn the_trajectory_does_not_depend_on_the_frame_rate() {
    let fall = |fps: f32| {
        let mut world = test_world();
        let ball = spawn(
            &mut world,
            Vec3::new(0.0, 10.0, 0.0),
            ColliderShape::Sphere { radius: 0.5 },
            Some(RigidBody::default()),
        );
        simulate(&mut world, 1.0, fps);
        position(&world, ball).y
    };

    let (sixty, fast) = (fall(60.0), fall(144.0));
    assert!(sixty < 6.0, "the ball did not fall: {sixty}");
    assert!(
        (sixty - fast).abs() < 0.01,
        "60 fps ended at {sixty}, 144 fps at {fast}"
    );
}

/// The contract `collision` gave scripts, kept: two colliders moved by
/// their transforms alone still report entering and leaving each other,
/// with the normal pointing from `a` to `b`.
///
/// One step later than `collision` reported it, and that is Jolt's order
/// rather than slack in the test: a step finds contacts *before* it moves
/// anything, so the step that carries `b` into `a` cannot see the overlap
/// it ends in.
#[test]
fn plain_colliders_still_raise_enter_and_exit() {
    let mut world = test_world();
    let a = spawn(&mut world, Vec3::ZERO, cube(0.5), None);
    let b = spawn(&mut world, Vec3::new(3.0, 0.0, 0.0), cube(0.5), None);
    frame(&mut world, STEP);
    assert!(events(&world).is_empty());

    world.get_mut::<LocalTransform>(b).unwrap().translation = Vec3::new(0.9, 0.0, 0.0);
    frame(&mut world, STEP);
    assert!(
        events(&world).is_empty(),
        "reported in the step that moved b"
    );
    frame(&mut world, STEP);
    assert_eq!(events(&world), [(CollisionEventKind::Enter, a, b)]);
    let normal = world.resource::<CollisionState>().events[0].normal;
    assert!(normal.x > 0.9, "normal {normal} does not point from a to b");

    world.get_mut::<LocalTransform>(b).unwrap().translation = Vec3::new(3.0, 0.0, 0.0);
    frame(&mut world, STEP);
    frame(&mut world, STEP);
    assert_eq!(events(&world), [(CollisionEventKind::Exit, a, b)]);
}

/// A collider with no rigid body is not pushed, but it does push: moving
/// one into a simulated body shoves the body out of its way.
#[test]
fn a_moved_collider_pushes_a_rigid_body() {
    let mut world = test_world();
    let pusher = spawn(&mut world, Vec3::ZERO, cube(0.5), None);
    let pushed = spawn(
        &mut world,
        Vec3::new(1.5, 0.0, 0.0),
        cube(0.5),
        Some(RigidBody {
            gravity_scale: 0.0,
            ..RigidBody::default()
        }),
    );

    for _ in 0..60 {
        world
            .get_mut::<LocalTransform>(pusher)
            .unwrap()
            .translation
            .x += 2.0 * STEP;
        frame(&mut world, STEP);
    }

    let (pusher, pushed) = (position(&world, pusher), position(&world, pushed));
    assert!(
        (pusher.x - 2.0).abs() < 1e-4,
        "the pusher was pushed back to {pusher}"
    );
    assert!(
        pushed.x - pusher.x > 0.95,
        "the body is at {pushed}, inside the pusher at {pusher}"
    );
}

/// A trigger is reported against and never stands in the way.
#[test]
fn a_trigger_reports_without_pushing() {
    let mut world = test_world();
    let trigger = spawn(&mut world, Vec3::ZERO, cube(1.0), None);
    world.get_mut::<Collider>(trigger).unwrap().is_trigger = true;
    let ball = spawn(
        &mut world,
        Vec3::new(0.0, 2.0, 0.0),
        ColliderShape::Sphere { radius: 0.25 },
        Some(RigidBody::default()),
    );

    let mut seen = Vec::new();
    for _ in 0..120 {
        frame(&mut world, STEP);
        seen.extend(events(&world));
    }

    assert!(
        position(&world, ball).y < -2.0,
        "the trigger held the ball up"
    );
    assert_eq!(
        seen,
        [
            (CollisionEventKind::Enter, trigger, ball),
            (CollisionEventKind::Exit, trigger, ball),
        ]
    );
}

/// Writing a simulated body's transform moves the body, rather than being
/// overwritten by where the simulation thought it was — and it is still
/// simulated from there.
#[test]
fn writing_a_rigid_bodys_transform_teleports_it() {
    let mut world = test_world();
    let falling = spawn(
        &mut world,
        Vec3::ZERO,
        cube(0.5),
        Some(RigidBody::default()),
    );
    simulate(&mut world, 0.1, 60.0);

    world
        .get_mut::<LocalTransform>(falling)
        .unwrap()
        .translation = Vec3::new(5.0, 0.0, 0.0);
    simulate(&mut world, 0.1, 60.0);

    let at = position(&world, falling);
    assert!((at.x - 5.0).abs() < 1e-3, "body is at {at}");
    assert!(at.y < -0.01, "body stopped falling after the move: {at}");
}

/// A parent/child link is a rigid attachment, as `collision` already
/// treats it: the simulation's world-space pose cannot be written into a
/// parent-relative transform, so a parented body is kinematic, not dropped —
/// and says so once.
#[test]
fn a_parented_rigid_body_follows_its_parent() {
    let mut world = test_world();
    let parent = world
        .spawn_entity()
        .with(LocalTransform::from(Transform::from_translation(
            Vec3::new(0.0, 5.0, 0.0),
        )))
        .id();
    let child = spawn(
        &mut world,
        Vec3::ZERO,
        cube(0.5),
        Some(RigidBody::default()),
    );
    reparent(&mut world, child, Some(parent), false).unwrap();

    simulate(&mut world, 1.0, 60.0);

    assert_eq!(position(&world, child), Vec3::ZERO);
    assert_eq!(world.resource::<LogBuffer>().len(), 1);
}

/// An entity that goes away mid-contact leaves the contact, so a script on
/// the survivor sees an exit.
#[test]
fn despawning_a_touching_entity_raises_exit() {
    let mut world = test_world();
    let a = spawn(&mut world, Vec3::ZERO, cube(0.5), None);
    let b = spawn(
        &mut world,
        Vec3::new(0.9, 0.0, 0.0),
        cube(0.5),
        Some(weightless()),
    );
    frame(&mut world, STEP);
    assert_eq!(events(&world), [(CollisionEventKind::Enter, a, b)]);

    world.despawn(b);
    frame(&mut world, STEP);
    assert_eq!(events(&world), [(CollisionEventKind::Exit, a, b)]);
}

/// The testbed scene, run headless: whatever the balls do to the stack,
/// nothing may tunnel through the floor or leave the room.
#[test]
fn the_physics_scene_stays_on_its_floor() {
    let mut world = test_world();
    crate::scene::entities::build_physics_scene(
        &mut world,
        &mut crate::gfx::HeadlessBackend::new(),
    );

    // With the paddle turning, as the app turns it, so the kinematic body
    // sweeping through the skittles is part of what has to stay sane.
    for _ in 0..360 {
        crate::systems::spin(&world, STEP);
        frame(&mut world, STEP);
    }

    let mut bodies = 0;
    world
        .query::<(&LocalTransform, &RigidBody)>()
        .for_each(|entity, (transform, _)| {
            bodies += 1;
            let at = transform.translation;
            let name = world.get::<crate::scene::Name>(entity);
            assert!(
                at.y > 0.0 && at.x.abs() < 14.0 && at.z.abs() < 14.0,
                "{name:?} ended at {at}"
            );
        });
    assert!(bodies > 20, "only {bodies} rigid bodies in the scene");
}

/// Stop throws the play fork away, so the next Play must start from the
/// scene's transforms at rest — not carry last session's velocity.
#[test]
fn reset_forgets_velocity() {
    let mut world = test_world();
    let ball = spawn(
        &mut world,
        Vec3::new(0.0, 10.0, 0.0),
        ColliderShape::Sphere { radius: 0.5 },
        Some(RigidBody::default()),
    );
    simulate(&mut world, 1.0, 60.0);
    assert!(position(&world, ball).y < 6.0, "the ball did not fall");

    reset(&world);
    world.get_mut::<LocalTransform>(ball).unwrap().0 = Transform {
        translation: Vec3::new(0.0, 10.0, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };
    frame(&mut world, STEP);

    let fallen = 10.0 - position(&world, ball).y;
    assert!(fallen < 0.01, "fell {fallen} m in one step after a reset");
}

/// The unit cube's triangles under `MeshHandle(0)`, for the mesh shapes.
fn with_cube_mesh(world: &mut World) -> MeshHandle {
    let mut bounds = MeshBounds::default();
    bounds.insert_geometry(MeshHandle(0), &CpuMesh::cube());
    world.insert_resource(bounds);
    MeshHandle(0)
}

fn settings(world: &mut World, settings: PhysicsSettings) -> Entity {
    world.spawn_entity().with(settings).id()
}

fn edit_collider(world: &World, entity: Entity, edit: impl FnOnce(&mut Collider)) {
    edit(&mut world.get_mut::<Collider>(entity).unwrap());
}

/// A collider nothing moves is static: Jolt keeps it in a broadphase tree of
/// its own and never steps it. Moving it makes it kinematic, from then on.
#[test]
fn a_plain_collider_is_static_until_something_moves_it() {
    let mut world = test_world();
    let wall = spawn(&mut world, Vec3::ZERO, cube(0.5), None);
    simulate(&mut world, 0.5, 60.0);
    assert_eq!(motion(&world, wall), Some(Motion::Static));

    world.get_mut::<LocalTransform>(wall).unwrap().translation.x = 1.0;
    frame(&mut world, STEP);
    assert_eq!(motion(&world, wall), Some(Motion::Kinematic));
    simulate(&mut world, 1.0, 60.0);
    assert_eq!(motion(&world, wall), Some(Motion::Kinematic));
}

/// Unity's rule: two colliders that nothing moves never report each other,
/// however they overlap — neither can have arrived.
#[test]
fn two_static_colliders_never_report_each_other() {
    let mut world = test_world();
    spawn(&mut world, Vec3::ZERO, cube(0.5), None);
    spawn(&mut world, Vec3::new(0.5, 0.0, 0.0), cube(0.5), None);

    let mut seen = Vec::new();
    for _ in 0..30 {
        frame(&mut world, STEP);
        seen.extend(events(&world));
    }

    assert!(seen.is_empty(), "{seen:?}");
}

/// A static trigger hears a collider that is moved into it, which is how
/// a scripted mover — the collision demo's bumper — reaches a pickup.
#[test]
fn a_moved_plain_collider_reports_against_a_static_trigger() {
    let mut world = test_world();
    let trigger = spawn(&mut world, Vec3::ZERO, cube(1.0), None);
    edit_collider(&world, trigger, |c| c.is_trigger = true);
    let mover = spawn(&mut world, Vec3::new(5.0, 0.0, 0.0), cube(0.25), None);

    // From x = 5 to x = -3, through a trigger whose edge is at ±1.25 for it.
    let mut seen = Vec::new();
    for _ in 0..80 {
        world
            .get_mut::<LocalTransform>(mover)
            .unwrap()
            .translation
            .x -= 0.1;
        frame(&mut world, STEP);
        seen.extend(events(&world));
    }

    assert_eq!(
        seen,
        [
            (CollisionEventKind::Enter, trigger, mover),
            (CollisionEventKind::Exit, trigger, mover),
        ]
    );
}

/// A capsule's length is along local Y and excludes its caps, so standing on
/// end its centre is `half_height + radius` above the floor.
#[test]
fn a_capsule_stands_on_its_end() {
    let mut world = test_world();
    floor(&mut world);
    let capsule = spawn(
        &mut world,
        Vec3::new(0.0, 2.0, 0.0),
        ColliderShape::Capsule {
            radius: 0.25,
            half_height: 0.5,
        },
        Some(RigidBody {
            lock_rotation: true,
            ..RigidBody::default()
        }),
    );

    simulate(&mut world, 2.0, 60.0);

    let y = position(&world, capsule).y;
    assert!(
        (y - 0.75).abs() < RESTING,
        "the capsule's centre settled at {y}"
    );
}

#[test]
fn a_cylinder_rests_on_its_flat_face() {
    let mut world = test_world();
    floor(&mut world);
    let cylinder = spawn(
        &mut world,
        Vec3::new(0.0, 2.0, 0.0),
        ColliderShape::Cylinder {
            radius: 0.5,
            half_height: 0.25,
        },
        Some(RigidBody::default()),
    );

    simulate(&mut world, 2.0, 60.0);

    let y = position(&world, cylinder).y;
    assert!((y - 0.25).abs() < RESTING, "the cylinder settled at {y}");
}

/// A mesh collider is the entity's own triangles, scaled by its transform:
/// the unit cube stretched into a slab is a floor with its top at 0.5.
#[test]
fn a_rigid_body_comes_to_rest_on_a_mesh_collider() {
    let mut world = test_world();
    let mesh = with_cube_mesh(&mut world);
    let slab = spawn(&mut world, Vec3::ZERO, ColliderShape::Mesh, None);
    world.insert(slab, mesh);
    world.get_mut::<LocalTransform>(slab).unwrap().scale = Vec3::new(10.0, 1.0, 10.0);
    let ball = spawn(
        &mut world,
        Vec3::new(0.0, 3.0, 0.0),
        ColliderShape::Sphere { radius: 0.5 },
        Some(RigidBody::default()),
    );

    simulate(&mut world, 2.0, 60.0);

    let y = position(&world, ball).y;
    assert!((y - 1.0).abs() < RESTING, "the ball settled at {y}");
}

#[test]
fn a_convex_hull_falls_and_rests_like_its_mesh() {
    let mut world = test_world();
    floor(&mut world);
    let mesh = with_cube_mesh(&mut world);
    let block = spawn(
        &mut world,
        Vec3::new(0.0, 3.0, 0.0),
        ColliderShape::ConvexHull,
        Some(RigidBody::default()),
    );
    world.insert(block, mesh);

    simulate(&mut world, 3.0, 60.0);

    let y = position(&world, block).y;
    assert!((y - 0.5).abs() < RESTING, "the hull settled at {y}");
}

/// A triangle mesh has no inside, so Jolt cannot push a moving one out of
/// anything. A rigid body asking for one gets its hull, and the log says so —
/// once, naming the entity.
#[test]
fn a_rigid_body_asking_for_a_mesh_collider_gets_its_hull_and_a_warning() {
    let mut world = test_world();
    floor(&mut world);
    let mesh = with_cube_mesh(&mut world);
    let block = spawn(
        &mut world,
        Vec3::new(0.0, 3.0, 0.0),
        ColliderShape::Mesh,
        Some(RigidBody::default()),
    );
    world.insert(block, mesh);

    simulate(&mut world, 3.0, 60.0);

    let y = position(&world, block).y;
    assert!((y - 0.5).abs() < RESTING, "the block settled at {y}");
    let log = world.resource::<LogBuffer>();
    assert_eq!(log.len(), 1);
    let message = &log.iter().next().unwrap().message;
    assert!(message.contains(&format!("{block:?}")), "{message}");
}

#[test]
fn a_mesh_collider_with_no_mesh_is_reported_and_left_out() {
    let mut world = test_world();
    with_cube_mesh(&mut world);
    let orphan = spawn(&mut world, Vec3::ZERO, ColliderShape::ConvexHull, None);

    simulate(&mut world, 0.5, 60.0);

    assert_eq!(motion(&world, orphan), None);
    assert_eq!(world.resource::<LogBuffer>().len(), 1);
}

/// A crate on a 25° ramp slides when friction is below tan 25° ≈ 0.47 and
/// stays where it was put when it is well above.
#[test]
fn friction_decides_whether_a_crate_slides_down_a_ramp() {
    let slid = |friction: f32| {
        let mut world = test_world();
        let tilt = Quat::from_rotation_z(25f32.to_radians());
        let ramp = spawn(
            &mut world,
            Vec3::ZERO,
            ColliderShape::Box {
                half_extents: Vec3::new(10.0, 0.5, 2.0),
            },
            None,
        );
        world.get_mut::<LocalTransform>(ramp).unwrap().rotation = tilt;
        let start = tilt * Vec3::new(0.0, 1.0, 0.0);
        let crate_ = spawn(&mut world, start, cube(0.5), Some(RigidBody::default()));
        world.get_mut::<LocalTransform>(crate_).unwrap().rotation = tilt;
        for entity in [ramp, crate_] {
            edit_collider(&world, entity, |c| c.friction = friction);
        }
        simulate(&mut world, 1.5, 60.0);
        position(&world, crate_).distance(start)
    };

    let (slippery, grippy) = (slid(0.0), slid(1.0));
    assert!(
        slippery > 1.0,
        "a frictionless crate moved only {slippery} m"
    );
    assert!(grippy < 0.05, "a grippy crate slid {grippy} m");
}

#[test]
fn restitution_decides_how_high_a_ball_bounces() {
    let bounce = |restitution: f32| {
        let mut world = test_world();
        let floor = floor(&mut world);
        let ball = spawn(
            &mut world,
            Vec3::new(0.0, 3.0, 0.0),
            ColliderShape::Sphere { radius: 0.5 },
            Some(RigidBody::default()),
        );
        for entity in [floor, ball] {
            edit_collider(&world, entity, |c| c.restitution = restitution);
        }
        // Past the first impact, which lands about 0.7 s in.
        simulate(&mut world, 0.9, 60.0);
        let mut highest: f32 = 0.0;
        for _ in 0..90 {
            frame(&mut world, STEP);
            highest = highest.max(position(&world, ball).y);
        }
        highest
    };

    let (dead, lively) = (bounce(0.0), bounce(0.8));
    assert!(dead < 0.6, "a dead ball rose to {dead}");
    assert!(lively > 1.5, "a lively ball rose only to {lively}");
}

/// Layers the scene says do not meet pass through each other; by default
/// every layer meets every other.
#[test]
fn layers_the_matrix_separates_pass_through_each_other() {
    let fell = |separate: bool| {
        let mut world = test_world();
        let mut physics = PhysicsSettings::default();
        physics.set_collides(0, 1, !separate);
        settings(&mut world, physics);
        floor(&mut world);
        let ball = spawn(
            &mut world,
            Vec3::new(0.0, 1.0, 0.0),
            ColliderShape::Sphere { radius: 0.5 },
            Some(RigidBody::default()),
        );
        edit_collider(&world, ball, |c| c.layer = 1);
        simulate(&mut world, 1.5, 60.0);
        position(&world, ball).y
    };

    assert!(
        fell(true) < -2.0,
        "the ball stood on a floor it should not meet"
    );
    assert!(
        fell(false) > 0.4,
        "the ball fell through a floor it should meet"
    );
}

/// Gravity is the scene's, read every frame, so changing it while playing
/// takes effect at once.
#[test]
fn gravity_comes_from_the_scene_settings() {
    let mut world = test_world();
    let physics = settings(
        &mut world,
        PhysicsSettings {
            gravity: Vec3::ZERO,
            ..PhysicsSettings::default()
        },
    );
    let ball = spawn(
        &mut world,
        Vec3::new(0.0, 10.0, 0.0),
        ColliderShape::Sphere { radius: 0.5 },
        Some(RigidBody::default()),
    );
    simulate(&mut world, 0.5, 60.0);
    assert_eq!(position(&world, ball).y, 10.0);

    world.get_mut::<PhysicsSettings>(physics).unwrap().gravity = Vec3::new(0.0, 9.81, 0.0);
    simulate(&mut world, 0.5, 60.0);
    assert!(position(&world, ball).y > 10.5, "the ball did not rise");
}

/// A faster step changes how finely the fall is integrated, not how far it
/// goes.
#[test]
fn the_step_rate_is_the_scenes() {
    let fall = |steps_per_second: u32| {
        let mut world = test_world();
        settings(
            &mut world,
            PhysicsSettings {
                steps_per_second,
                ..PhysicsSettings::default()
            },
        );
        let ball = spawn(
            &mut world,
            Vec3::new(0.0, 10.0, 0.0),
            ColliderShape::Sphere { radius: 0.5 },
            Some(RigidBody::default()),
        );
        simulate(&mut world, 1.0, 60.0);
        10.0 - position(&world, ball).y
    };

    let (sixty, double) = (fall(60), fall(120));
    assert!(
        (sixty - double).abs() < 0.1,
        "60 Hz fell {sixty}, 120 Hz {double}"
    );
}

/// Fast enough to cross a thin floor in one step: a discrete body passes
/// through it, a continuous one is stopped.
#[test]
fn a_fast_body_tunnels_through_a_thin_floor_unless_continuous() {
    let ended = |continuous: bool| {
        let mut world = test_world();
        spawn(
            &mut world,
            Vec3::ZERO,
            ColliderShape::Box {
                half_extents: Vec3::new(5.0, 0.02, 5.0),
            },
            None,
        );
        let pellet = spawn(
            &mut world,
            Vec3::new(0.0, 40.0, 0.0),
            ColliderShape::Sphere { radius: 0.05 },
            Some(RigidBody {
                gravity_scale: 10.0,
                linear_damping: 0.0,
                continuous,
                ..RigidBody::default()
            }),
        );
        simulate(&mut world, 1.5, 60.0);
        position(&world, pellet).y
    };

    assert!(ended(false) < -1.0, "the discrete pellet was stopped");
    assert!(ended(true) > 0.0, "the continuous pellet tunnelled");
}

#[test]
fn a_body_with_its_rotation_locked_keeps_it() {
    let tumble = |lock_rotation: bool| {
        let mut world = test_world();
        floor(&mut world);
        let tilt = Quat::from_rotation_z(0.6) * Quat::from_rotation_x(0.3);
        let crate_ = spawn(
            &mut world,
            Vec3::new(0.0, 3.0, 0.0),
            cube(0.5),
            Some(RigidBody {
                lock_rotation,
                ..RigidBody::default()
            }),
        );
        world.get_mut::<LocalTransform>(crate_).unwrap().rotation = tilt;
        simulate(&mut world, 2.0, 60.0);
        world
            .get::<LocalTransform>(crate_)
            .unwrap()
            .rotation
            .angle_between(tilt)
    };

    assert!(tumble(true) < 1e-3, "a locked crate turned");
    assert!(
        tumble(false) > 0.1,
        "a free crate landed on its corner and stayed"
    );
}

fn velocity_of(world: &World, entity: Entity) -> Vec3 {
    velocity(world, entity)
        .expect("a simulated body has a velocity")
        .0
}

fn frictionless(mass: f32) -> RigidBody {
    RigidBody {
        mass,
        gravity_scale: 0.0,
        linear_damping: 0.0,
        angular_damping: 0.0,
        ..RigidBody::default()
    }
}

#[test]
fn an_impulse_changes_velocity_by_impulse_over_mass() {
    let mut world = test_world();
    let body = spawn(&mut world, Vec3::ZERO, cube(0.5), Some(frictionless(2.0)));
    frame(&mut world, STEP);

    assert!(add_force(
        &world,
        body,
        Vec3::new(4.0, 0.0, 0.0),
        ForceMode::Impulse
    ));
    frame(&mut world, STEP);

    let v = velocity_of(&world, body);
    assert!((v.x - 2.0).abs() < 1e-3, "velocity {v}");
}

#[test]
fn a_velocity_change_ignores_mass() {
    let mut world = test_world();
    let body = spawn(&mut world, Vec3::ZERO, cube(0.5), Some(frictionless(50.0)));
    frame(&mut world, STEP);

    add_force(
        &world,
        body,
        Vec3::new(0.0, 0.0, 3.0),
        ForceMode::VelocityChange,
    );
    frame(&mut world, STEP);

    let v = velocity_of(&world, body);
    assert!((v.z - 3.0).abs() < 1e-3, "velocity {v}");
}

/// A force added every frame is a force held for that frame, so a second of
/// it is the same push at 60 fps as at 144 — scripts call it from OnUpdate,
/// which runs at the display's rate rather than the step's.
#[test]
fn a_held_force_pushes_the_same_whatever_the_frame_rate() {
    let pushed = |fps: f32, mode: ForceMode| {
        let mut world = test_world();
        let body = spawn(&mut world, Vec3::ZERO, cube(0.5), Some(frictionless(2.0)));
        frame(&mut world, 1.0 / fps);
        for _ in 0..fps.round() as u32 {
            add_force(&world, body, Vec3::new(10.0, 0.0, 0.0), mode);
            frame(&mut world, 1.0 / fps);
        }
        velocity_of(&world, body).x
    };

    for (mode, expected) in [(ForceMode::Force, 5.0), (ForceMode::Acceleration, 10.0)] {
        let (sixty, fast) = (pushed(60.0, mode), pushed(144.0, mode));
        assert!(
            (sixty - expected).abs() < 0.2,
            "{mode:?} at 60 fps reached {sixty}"
        );
        assert!(
            (fast - expected).abs() < 0.2,
            "{mode:?} at 144 fps reached {fast}"
        );
    }
}

#[test]
fn an_angular_impulse_spins_a_body() {
    let mut world = test_world();
    let body = spawn(&mut world, Vec3::ZERO, cube(0.5), Some(frictionless(1.0)));
    frame(&mut world, STEP);

    assert!(add_torque(
        &world,
        body,
        Vec3::new(0.0, 1.0, 0.0),
        ForceMode::Impulse
    ));
    frame(&mut world, STEP);

    let (_, angular) = velocity(&world, body).unwrap();
    assert!(
        angular.y > 1.0 && angular.x.abs() < 1e-3,
        "angular velocity {angular}"
    );
}

#[test]
fn setting_a_velocity_moves_the_body_at_it() {
    let mut world = test_world();
    let body = spawn(&mut world, Vec3::ZERO, cube(0.5), Some(frictionless(1.0)));
    frame(&mut world, STEP);

    assert!(set_velocity(
        &world,
        body,
        Some(Vec3::new(0.0, 5.0, 0.0)),
        None
    ));
    simulate(&mut world, 1.0, 60.0);

    let y = position(&world, body).y;
    assert!((y - 5.0).abs() < 0.1, "moved {y} m in a second at 5 m/s");
}

/// A script that spawns a body and pushes it in the same OnStart pushes a body
/// that does not exist yet; the push waits for it.
#[test]
fn a_push_on_a_body_spawned_this_frame_waits_for_it() {
    let mut world = test_world();
    frame(&mut world, STEP);
    let body = spawn(&mut world, Vec3::ZERO, cube(0.5), Some(frictionless(1.0)));

    assert!(add_force(
        &world,
        body,
        Vec3::new(2.0, 0.0, 0.0),
        ForceMode::VelocityChange
    ));
    frame(&mut world, STEP);

    let v = velocity_of(&world, body);
    assert!((v.x - 2.0).abs() < 1e-3, "velocity {v}");
}

/// Only a dynamic body has a velocity of its own to change.
#[test]
fn a_push_on_anything_but_a_dynamic_body_is_refused() {
    let mut world = test_world();
    let wall = spawn(&mut world, Vec3::ZERO, cube(0.5), None);
    let lift = spawn(
        &mut world,
        Vec3::new(3.0, 0.0, 0.0),
        cube(0.5),
        Some(RigidBody {
            kinematic: true,
            ..RigidBody::default()
        }),
    );
    frame(&mut world, STEP);

    for entity in [wall, lift] {
        assert!(!add_force(&world, entity, Vec3::X, ForceMode::Impulse));
        assert!(!set_velocity(&world, entity, Some(Vec3::X), None));
    }
}

fn ray_world() -> (World, Entity, Entity) {
    let mut world = test_world();
    let near = spawn(&mut world, Vec3::new(0.0, 0.0, -5.0), cube(0.5), None);
    let far = spawn(&mut world, Vec3::new(0.0, 0.0, -10.0), cube(0.5), None);
    frame(&mut world, STEP);
    (world, near, far)
}

#[test]
fn a_ray_hits_the_nearest_collider_and_names_it() {
    let (world, near, _) = ray_world();

    let hit = raycast(&world, Vec3::ZERO, -Vec3::Z, 100.0, u32::MAX, false).unwrap();

    assert_eq!(hit.entity, near);
    assert!((hit.distance - 4.5).abs() < 1e-3, "{hit:?}");
    assert!(hit.normal.abs_diff_eq(Vec3::Z, 1e-3), "{hit:?}");
    assert!(
        hit.point.abs_diff_eq(Vec3::new(0.0, 0.0, -4.5), 1e-3),
        "{hit:?}"
    );
}

#[test]
fn a_ray_stops_at_its_length() {
    let (world, _, _) = ray_world();

    assert!(raycast(&world, Vec3::ZERO, -Vec3::Z, 4.0, u32::MAX, false).is_none());
}

#[test]
fn a_ray_passes_layers_its_mask_leaves_out() {
    let (world, near, far) = ray_world();
    edit_collider(&world, near, |c| c.layer = 2);
    let mut world = world;
    frame(&mut world, STEP);

    let hit = raycast(&world, Vec3::ZERO, -Vec3::Z, 100.0, !(1 << 2), false).unwrap();

    assert_eq!(hit.entity, far);
}

/// A trigger is a volume to be in, not a surface to see, so a ray passes it
/// unless it asks.
#[test]
fn a_ray_passes_triggers_unless_asked_to_hit_them() {
    let (world, near, far) = ray_world();
    edit_collider(&world, near, |c| c.is_trigger = true);
    let mut world = world;
    frame(&mut world, STEP);

    let through = raycast(&world, Vec3::ZERO, -Vec3::Z, 100.0, u32::MAX, false).unwrap();
    let onto = raycast(&world, Vec3::ZERO, -Vec3::Z, 100.0, u32::MAX, true).unwrap();

    assert_eq!(through.entity, far);
    assert_eq!(onto.entity, near);
}

/// Scripts hear a trigger through its own callbacks, so an event says
/// whether it was one.
#[test]
fn an_event_says_whether_a_trigger_was_in_it() {
    let mut world = test_world();
    let trigger = spawn(&mut world, Vec3::ZERO, cube(1.0), None);
    edit_collider(&world, trigger, |c| c.is_trigger = true);
    let solid = spawn(&mut world, Vec3::new(5.0, 0.0, 0.0), cube(1.0), None);
    spawn(
        &mut world,
        Vec3::new(5.0, 3.0, 0.0),
        cube(0.5),
        Some(RigidBody::default()),
    );
    spawn(
        &mut world,
        Vec3::new(0.0, 3.0, 0.0),
        cube(0.25),
        Some(RigidBody::default()),
    );

    let mut flags = Vec::new();
    for _ in 0..90 {
        frame(&mut world, STEP);
        for event in &world.resource::<CollisionState>().events {
            if event.kind == CollisionEventKind::Enter {
                flags.push((event.a == trigger || event.b == trigger, event.trigger));
            }
        }
    }

    let _ = solid;
    assert!(flags.contains(&(true, true)), "{flags:?}");
    assert!(flags.contains(&(false, false)), "{flags:?}");
    assert!(
        flags
            .iter()
            .all(|(with_trigger, flag)| with_trigger == flag),
        "{flags:?}"
    );
}

/// A body is rebuilt when what it was built from changes — a crouch that
/// halves its height, a script that makes it heavier. It is the same body
/// moving, so it keeps moving: a slide that crouched must not stop dead.
#[test]
fn a_rebuilt_body_keeps_its_velocity() {
    let mut world = test_world();
    let body = spawn(&mut world, Vec3::ZERO, cube(0.5), Some(frictionless(1.0)));
    frame(&mut world, STEP);
    set_velocity(
        &world,
        body,
        Some(Vec3::new(6.0, 0.0, 0.0)),
        Some(Vec3::new(0.0, 2.0, 0.0)),
    );
    frame(&mut world, STEP);

    world.get_mut::<LocalTransform>(body).unwrap().scale = Vec3::new(1.0, 0.5, 1.0);
    frame(&mut world, STEP);

    let (linear, angular) = velocity(&world, body).unwrap();
    assert!((linear.x - 6.0).abs() < 1e-3, "lost its speed: {linear}");
    assert!((angular.y - 2.0).abs() < 1e-3, "lost its spin: {angular}");
}

/// A script's components are added after its tick, so "spawn, add a rigid
/// body, throw" pushes an entity that has no rigid body yet when it asks. The
/// push is kept for the next step, which is after the body has arrived — and
/// dropped there, harmlessly, if it never did.
#[test]
fn a_push_waits_for_a_rigid_body_added_after_it() {
    let mut world = test_world();
    frame(&mut world, STEP);
    let ball = spawn(&mut world, Vec3::ZERO, cube(0.5), None);

    assert!(set_velocity(
        &world,
        ball,
        Some(Vec3::new(0.0, 0.0, -8.0)),
        None
    ));
    world.insert(ball, frictionless(1.0));
    frame(&mut world, STEP);

    let v = velocity_of(&world, ball);
    assert!((v.z + 8.0).abs() < 1e-3, "velocity {v}");

    let never = spawn(&mut world, Vec3::new(5.0, 0.0, 0.0), cube(0.5), None);
    world.despawn(never);
    let stays = spawn(&mut world, Vec3::new(9.0, 0.0, 0.0), cube(0.5), None);
    assert!(add_force(&world, stays, Vec3::X, ForceMode::Impulse));
    frame(&mut world, STEP);
    assert_eq!(
        motion(&world, stays),
        Some(Motion::Static),
        "the dropped push moved a wall"
    );
}
