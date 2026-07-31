//! Baselines for the ECS hot paths, so an optimization can be defended with a
//! number instead of an argument (#8).
//!
//! `query_two_components` is the one to watch: it is what every engine system
//! runs, and it currently re-resolves the driving parameter's entity through the
//! sparse array even though iteration already knows its dense index.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};

use orrin_ecs::{Entity, World};

#[derive(Clone, Copy)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy)]
struct Velocity {
    x: f32,
    y: f32,
    z: f32,
}

struct Tag;

const COUNTS: [usize; 3] = [1_000, 10_000, 100_000];

/// A world of `count` entities, every one carrying Position and Velocity, and
/// every fourth also carrying Tag so queries have something to filter on.
fn populated(count: usize) -> (World, Vec<Entity>) {
    let mut world = World::new();
    let mut entities = Vec::with_capacity(count);
    for i in 0..count {
        let entity = world.spawn();
        world.insert(entity, Position { x: i as f32, y: 0.0, z: 0.0 });
        world.insert(entity, Velocity { x: 1.0, y: 0.5, z: 0.0 });
        if i % 4 == 0 {
            world.insert(entity, Tag);
        }
        entities.push(entity);
    }
    (world, entities)
}

fn spawn_and_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("spawn_and_insert");
    for count in COUNTS {
        group.throughput(criterion::Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| {
                let (world, entities) = populated(black_box(count));
                (world, entities)
            })
        });
    }
    group.finish();
}

fn query_one_component(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_one_component");
    for count in COUNTS {
        let (world, _) = populated(count);
        group.throughput(criterion::Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                world.query::<&Position>().for_each(|_, position| sum += position.x);
                black_box(sum)
            })
        });
    }
    group.finish();
}

fn query_two_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_two_components");
    for count in COUNTS {
        let (world, _) = populated(count);
        group.throughput(criterion::Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| {
                world
                    .query::<(&mut Position, &Velocity)>()
                    .for_each(|_, (position, velocity)| {
                        position.x += velocity.x;
                        position.y += velocity.y;
                        position.z += velocity.z;
                    });
            })
        });
    }
    group.finish();
}

/// The sparser the match, the more the driver pays for candidates it discards.
fn query_sparse_match(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_sparse_match");
    for count in COUNTS {
        let (world, _) = populated(count);
        group.throughput(criterion::Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| {
                let mut matched = 0usize;
                world.query::<(&Position, &Tag)>().for_each(|_, _| matched += 1);
                black_box(matched)
            })
        });
    }
    group.finish();
}

/// Random-access component lookup — what the scripting FFI does per call, and
/// where the `TypeId` hash shows up.
fn get_by_handle(c: &mut Criterion) {
    let mut group = c.benchmark_group("get_by_handle");
    for count in COUNTS {
        let (world, entities) = populated(count);
        // Strided so the access pattern isn't a linear walk of the dense array.
        let probes: Vec<Entity> = entities.iter().copied().step_by(7).collect();
        group.throughput(criterion::Throughput::Elements(probes.len() as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| {
                let mut sum = 0.0f32;
                for &entity in &probes {
                    if let Some(position) = world.get::<Position>(entity) {
                        sum += position.x;
                    }
                }
                black_box(sum)
            })
        });
    }
    group.finish();
}

/// Despawn walks every registered storage, so its cost tracks component-type
/// count as much as entity count.
fn despawn(c: &mut Criterion) {
    let mut group = c.benchmark_group("despawn");
    for count in COUNTS {
        group.throughput(criterion::Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter_batched(
                || populated(count),
                |(mut world, entities)| {
                    for entity in entities {
                        world.despawn(entity);
                    }
                    world
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    spawn_and_insert,
    query_one_component,
    query_two_components,
    query_sparse_match,
    get_by_handle,
    despawn
);
criterion_main!(benches);
