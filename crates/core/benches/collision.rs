//! The whole collision phase as `app.rs` runs it, once per frame (#8).
//!
//! `bvh.rs` measures the broadphase in isolation; this measures what the frame
//! actually pays, which is the only place two of the #8 items show up: the
//! per-frame allocations behind `Bvh`, and the `touching` contact map, whose
//! key is hashed at least twice per overlapping pair per frame.
//!
//! Bodies are laid out by the same fixed-seed splitmix64 as the stress scene's
//! colliders, at the same density, so a number here and a `ORRIN_STRESS`
//! profiling run describe the same scene.
//!
//! `run` is stateful by design — it diffs this frame's contacts against last
//! frame's — so each iteration is a *second* frame over a world that already
//! has a populated `touching` map. Benchmarking the first frame instead would
//! measure only the Enter path and never the diff.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use glam::Vec3;

use orrin_core::collision::{self, CollisionState};
use orrin_core::scene::{Collider, ColliderShape, LocalTransform, Transform};
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

/// The stress scene's collider layout: spread by the cube root so density stays
/// constant, and dense enough that the broadphase reports real pairs.
fn world_with_colliders(count: usize) -> World {
    let mut world = World::new();
    world.insert_resource(CollisionState::default());

    let mut rng = Rng(0x0DDB_A11_0_C0FF_EE00);
    let spread = (count.max(1) as f32).cbrt() * 1.2;
    for index in 0..count {
        let position = Vec3::new(
            rng.range(-spread, spread),
            rng.range(-spread, spread),
            rng.range(-spread, spread),
        );
        world
            .spawn_entity()
            .with(LocalTransform::from(Transform::from_translation(position)))
            .with(Collider {
                shape: if index % 2 == 0 {
                    ColliderShape::Box { half_extents: Vec3::splat(0.5) }
                } else {
                    ColliderShape::Sphere { radius: 0.5 }
                },
                // A third are triggers, so the resolver runs without every pair
                // shoving the scene apart and emptying it of contacts.
                is_trigger: index % 3 == 0,
            })
            .id();
    }
    world
}

fn run(c: &mut Criterion) {
    let mut group = c.benchmark_group("collision_run");
    for count in COUNTS {
        let mut world = world_with_colliders(count);
        // Prime `touching`, so every measured iteration exercises the diff
        // against a populated map rather than an empty one.
        collision::run(&mut world);
        assert!(
            !world.resource::<CollisionState>().events.is_empty(),
            "{count} bodies produced no contacts — the bench would measure an empty scene"
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| {
                collision::run(black_box(&mut world));
            })
        });
    }
    group.finish();
}

criterion_group!(benches, run);
criterion_main!(benches);
