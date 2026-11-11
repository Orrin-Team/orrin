//! The script phase as `app.rs` runs it: one `Scripting::tick` over a world of
//! scripted entities, every one already enabled and started, so each iteration
//! is the steady-state frame rather than the first.
//!
//! The Behaviour is `Orrin.Tests.IdleProbe`, which does nothing, so the number
//! is the engine's side of the tick — collection, dispatch, write-back — and
//! the managed transition per call, not a script's own work.
//!
//! - `tick`: N entities, one Behaviour each, no events. The common frame.
//! - `tick_with_collisions`: the same with a collision event on every tenth
//!   entity, which is what builds the routing index.
//! - `physics_then_tick`: the physics phase and the tick together, three steps
//!   a frame, over N crates resting on a floor whose Behaviour overrides
//!   neither OnFixedUpdate nor a Stay callback. `plain` steps through
//!   `physics::run`, which knows nothing of scripts; `scripted` through
//!   `Scripting::run_physics`, as `app.rs` does. The two should not differ:
//!   that is the claim that a hook nobody overrides costs nothing per step.
//!   `listening` adds one Behaviour that overrides both, which turns on the
//!   sweep for OnFixedUpdate and the stays — what a world pays once anything
//!   uses them, with every crate still opting out.
//!
//! Needs both C# assemblies built (`dotnet build scripting/Orrin` and
//! `dotnet build scripting/DemoGame`) and measures nothing without them.

use std::path::Path;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use glam::Vec3;

use orrin_core::physics::{self, CollisionEvent, CollisionEventKind, CollisionState, PhysicsState};
use orrin_core::scene::{
    Collider, ColliderShape, LocalTransform, RigidBody, Transform, propagate_transforms,
    register_components,
};
use orrin_core::scripting::Scripting;
use orrin_ecs::{Entity, World};
use orrin_registry::Registry;

const COUNTS: [usize; 3] = [100, 1_000, 10_000];
/// The physics bench's sizes, so a body count means the same in both.
const BODY_COUNTS: [usize; 3] = [100, 1_000, 5_000];
const IDLE: &str = "Orrin.Tests.IdleProbe, Orrin";
const STEPPED_IDLE: &str = "Orrin.Tests.SteppedIdleProbe, Orrin";
const FRAME: f32 = 1.0 / 60.0;
const THREE_STEPS: f32 = 3.0 * physics::STEP;

fn boot() -> Option<Scripting> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bindings = Scripting::find_bindings_dir(&root.join("scripting/Orrin"))?;
    let game = Scripting::find_game_assembly(&root.join("scripting/DemoGame"), "DemoGame")?;
    Scripting::boot(&bindings, &game)
}

/// `count` entities with one started Behaviour each.
fn scripted(scripting: &Scripting, registry: &Registry, count: usize) -> (World, Vec<Entity>) {
    let mut world = World::new();
    world.insert_resource(CollisionState::default());
    let entities: Vec<Entity> = (0..count)
        .map(|_| {
            let entity = world.spawn();
            scripting.attach(&mut world, entity, IDLE);
            entity
        })
        .collect();
    scripting.tick(&mut world, registry, FRAME);
    (world, entities)
}

/// `count` crates in a grid on a floor, each with one Behaviour that overrides
/// no per-step hook, plus one that overrides both when `listening`. Stepped
/// until the pile has settled, as a level's props spend most of their time.
fn resting(scripting: &Scripting, registry: &Registry, count: usize, listening: bool) -> World {
    let mut world = World::new();
    world.insert_resource(CollisionState::default());
    world.insert_resource(PhysicsState::default());
    let side = (count as f32).sqrt().ceil() as usize;
    let half = side as f32 * 0.75 + 2.0;
    world
        .spawn_entity()
        .with(LocalTransform::from(Transform::from_translation(
            Vec3::new(0.0, -0.5, 0.0),
        )))
        .with(Collider {
            shape: ColliderShape::Box {
                half_extents: Vec3::new(half, 0.5, half),
            },
            ..Collider::default()
        });
    for index in 0..count {
        let at = Vec3::new(
            (index % side) as f32 * 1.5 - half + 2.0,
            0.5,
            (index / side) as f32 * 1.5 - half + 2.0,
        );
        let entity = world
            .spawn_entity()
            .with(LocalTransform::from(Transform::from_translation(at)))
            .with(Collider {
                shape: ColliderShape::Box {
                    half_extents: Vec3::splat(0.5),
                },
                ..Collider::default()
            })
            .with(RigidBody::default())
            .id();
        scripting.attach(&mut world, entity, IDLE);
    }
    if listening {
        let entity = world.spawn();
        scripting.attach(&mut world, entity, STEPPED_IDLE);
    }
    for _ in 0..90 {
        propagate_transforms(&mut world);
        scripting.run_physics(&mut world, registry, THREE_STEPS);
        scripting.tick(&mut world, registry, THREE_STEPS);
    }
    world
}

fn bench(c: &mut Criterion) {
    let Some(scripting) = boot() else {
        eprintln!("skipping: build scripting/Orrin and scripting/DemoGame first");
        return;
    };
    let mut registry = Registry::new();
    register_components(&mut registry);

    let mut group = c.benchmark_group("tick");
    for count in COUNTS {
        let (mut world, _) = scripted(&scripting, &registry, count);
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| scripting.tick(&mut world, &registry, FRAME))
        });
    }
    group.finish();

    let mut group = c.benchmark_group("tick_with_collisions");
    for count in COUNTS {
        let (mut world, entities) = scripted(&scripting, &registry, count);
        let events: Vec<CollisionEvent> = entities
            .chunks(10)
            .map(|chunk| CollisionEvent {
                kind: CollisionEventKind::Enter,
                a: chunk[0],
                b: *chunk.last().unwrap(),
                point: Vec3::ZERO,
                normal: Vec3::Y,
                relative_velocity: Vec3::ZERO,
                trigger: false,
            })
            .collect();
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| {
                world
                    .resource_mut::<CollisionState>()
                    .events
                    .clone_from(&events);
                scripting.tick(&mut world, &registry, FRAME)
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group("physics_then_tick");
    for count in BODY_COUNTS {
        group.throughput(Throughput::Elements(count as u64));
        let mut world = resting(&scripting, &registry, count, false);
        group.bench_function(format!("plain/{count}"), |b| {
            b.iter(|| {
                propagate_transforms(&mut world);
                physics::run(&mut world, THREE_STEPS);
                scripting.tick(&mut world, &registry, THREE_STEPS);
            })
        });
        let mut world = resting(&scripting, &registry, count, false);
        group.bench_function(format!("scripted/{count}"), |b| {
            b.iter(|| {
                propagate_transforms(&mut world);
                scripting.run_physics(&mut world, &registry, THREE_STEPS);
                scripting.tick(&mut world, &registry, THREE_STEPS);
            })
        });
        let mut world = resting(&scripting, &registry, count, true);
        group.bench_function(format!("listening/{count}"), |b| {
            b.iter(|| {
                propagate_transforms(&mut world);
                scripting.run_physics(&mut world, &registry, THREE_STEPS);
                scripting.tick(&mut world, &registry, THREE_STEPS);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
