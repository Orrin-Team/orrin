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
//!
//! Needs both C# assemblies built (`dotnet build scripting/Orrin` and
//! `dotnet build scripting/DemoGame`) and measures nothing without them.

use std::path::Path;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use glam::Vec3;

use orrin_core::physics::{CollisionEvent, CollisionEventKind, CollisionState};
use orrin_core::scene::register_components;
use orrin_core::scripting::Scripting;
use orrin_ecs::{Entity, World};
use orrin_registry::Registry;

const COUNTS: [usize; 3] = [100, 1_000, 10_000];
const IDLE: &str = "Orrin.Tests.IdleProbe, Orrin";
const FRAME: f32 = 1.0 / 60.0;

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
}

criterion_group!(benches, bench);
criterion_main!(benches);
