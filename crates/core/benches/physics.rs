//! The physics phase as `app.rs` runs it: one `physics::run` at one step a
//! frame, as at 60 fps.
//!
//! Two worlds, because they cost for different reasons:
//!
//! - `plain`: colliders with no `RigidBody`, laid out by the same fixed-seed
//!   splitmix64 as the stress scene's colliders and at the same density, so a
//!   number here and an `ORRIN_STRESS` profiling run describe the same scene.
//!   Nothing moves them, so every one is static: this is what a level's walls
//!   and triggers cost.
//! - `moving`: the same colliders, every one nudged every frame — scripted
//!   movers, each now kinematic and testing every neighbour. Includes the
//!   transform propagation the nudge needs, which the frame pays too.
//! - `awake_pile`: columns of crates standing on a floor, every one awake. A
//!   settled pile goes to sleep within half a second and costs almost nothing,
//!   so each iteration is the first step of a freshly built pile — the frame a
//!   pile is knocked over, not the frame after it settles.

use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};

use glam::Vec3;

use orrin_core::physics::{self, CollisionState, Motion, PhysicsState, STEP};
use orrin_core::scene::{Collider, ColliderShape, LocalTransform, RigidBody, Transform};
use orrin_ecs::World;

const COUNTS: [usize; 3] = [100, 1_000, 5_000];

/// splitmix64, matching `scene/entities/stress.rs`.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u32 << 24) as f32
    }

    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + self.unit() * (high - low)
    }
}

fn world() -> World {
    let mut world = World::new();
    world.insert_resource(CollisionState::default());
    world.insert_resource(PhysicsState::default());
    world
}

fn collider(shape: ColliderShape, is_trigger: bool) -> Collider {
    Collider {
        shape,
        is_trigger,
        ..Collider::default()
    }
}

/// The stress scene's collider layout: spread by the cube root so density stays
/// constant, and dense enough that the broadphase reports real pairs.
fn plain_world(count: usize) -> World {
    let mut world = world();
    let mut rng = Rng(0x0DDB_A110_C0FF_EE00);
    let spread = (count.max(1) as f32).cbrt() * 1.2;
    for index in 0..count {
        let position = Vec3::new(
            rng.range(-spread, spread),
            rng.range(-spread, spread),
            rng.range(-spread, spread),
        );
        let shape = if index % 2 == 0 {
            ColliderShape::Box {
                half_extents: Vec3::splat(0.5),
            }
        } else {
            ColliderShape::Sphere { radius: 0.5 }
        };
        world
            .spawn_entity()
            .with(LocalTransform::from(Transform::from_translation(position)))
            // A third are triggers, as in the stress scene.
            .with(collider(shape, index % 3 == 0));
    }
    orrin_core::scene::propagate_transforms(&mut world);
    world
}

fn pile_world(count: usize) -> World {
    const HEIGHT: usize = 5;
    let mut world = world();
    let columns = count.div_ceil(HEIGHT);
    let side = (columns as f32).sqrt().ceil() as usize;
    let floor = side as f32 * 1.5;
    world
        .spawn_entity()
        .with(LocalTransform::from(Transform::from_translation(
            Vec3::new(0.0, -0.5, 0.0),
        )))
        .with(collider(
            ColliderShape::Box {
                half_extents: Vec3::new(floor, 0.5, floor),
            },
            false,
        ));
    let unit = ColliderShape::Box {
        half_extents: Vec3::splat(0.5),
    };
    for index in 0..count {
        let column = index / HEIGHT;
        let position = Vec3::new(
            (column % side) as f32 * 1.5 - floor * 0.5,
            0.5 + (index % HEIGHT) as f32,
            (column / side) as f32 * 1.5 - floor * 0.5,
        );
        world
            .spawn_entity()
            .with(LocalTransform::from(Transform::from_translation(position)))
            .with(collider(unit, false))
            .with(RigidBody::default());
    }
    orrin_core::scene::propagate_transforms(&mut world);
    world
}

fn plain(c: &mut Criterion) {
    let mut group = c.benchmark_group("physics_run/plain");
    for count in COUNTS {
        let mut world = plain_world(count);
        physics::run(&mut world, STEP);
        let entities: Vec<_> = world.entities().collect();
        assert!(
            entities
                .iter()
                .all(|&e| physics::motion(&world, e) == Some(Motion::Static)),
            "a plain collider was not static — the bench would measure the wrong path"
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| physics::run(black_box(&mut world), STEP))
        });
    }
    group.finish();
}

fn moving(c: &mut Criterion) {
    let mut group = c.benchmark_group("physics_run/moving");
    for count in COUNTS {
        let mut world = plain_world(count);
        let entities: Vec<_> = world.entities().collect();
        let mut nudge = 0.001;
        let mut frame = |world: &mut World| {
            nudge = -nudge;
            for &entity in &entities {
                world
                    .get_mut::<LocalTransform>(entity)
                    .unwrap()
                    .translation
                    .x += nudge;
            }
            orrin_core::scene::propagate_transforms(world);
            physics::run(world, STEP);
        };
        // Built, then promoted, then pairs established.
        for _ in 0..3 {
            frame(&mut world);
        }
        assert!(
            physics::motion(&world, entities[0]) == Some(Motion::Kinematic),
            "a moved collider was not kinematic"
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| frame(black_box(&mut world)))
        });
    }
    group.finish();
}

fn awake_pile(c: &mut Criterion) {
    let mut group = c.benchmark_group("physics_run/awake_pile");
    group.sample_size(20);
    for count in COUNTS {
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter_batched(
                || {
                    let mut world = pile_world(count);
                    // Bodies are created on the first run, and that is
                    // set-up rather than a step.
                    physics::run(&mut world, 0.0);
                    world
                },
                |mut world| physics::run(black_box(&mut world), STEP),
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

criterion_group!(benches, plain, moving, awake_pile);
criterion_main!(benches);
