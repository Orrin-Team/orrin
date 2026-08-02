//! Broadphase baselines: the tree is rebuilt from scratch every frame, so
//! `build` is on the critical path as much as `query_pairs` is (#8).
//!
//! Both densities matter. Sparse bodies exercise the build and little else;
//! dense ones make `query_pairs` do real traversal work, which is the case that
//! degrades first as scenes grow.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

use glam::Vec3;
use orrin_core::collision::{Aabb, Bvh};

const COUNTS: [usize; 3] = [100, 1_000, 10_000];

/// splitmix64, matching the stress scene's generator so bench and profile runs
/// describe the same kind of scene.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// `spread` sets density: the same count in a smaller volume overlaps more, so
/// the broadphase reports more candidate pairs.
fn bounds(count: usize, spread: f32) -> Vec<Aabb> {
    let mut rng = Rng(0x0DDB_A11_0_C0FF_EE00);
    (0..count)
        .map(|_| {
            let center = Vec3::new(
                (rng.unit() - 0.5) * spread,
                (rng.unit() - 0.5) * spread,
                (rng.unit() - 0.5) * spread,
            );
            let half = Vec3::splat(0.5);
            Aabb {
                min: center - half,
                max: center + half,
            }
        })
        .collect()
}

/// Volume scaled by the cube root keeps bodies-per-unit-volume constant, so a
/// larger count measures scale rather than a denser scene.
fn sparse_spread(count: usize) -> f32 {
    (count as f32).cbrt() * 8.0
}

fn dense_spread(count: usize) -> f32 {
    (count as f32).cbrt() * 1.5
}

fn build(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_build");
    for count in COUNTS {
        let boxes = bounds(count, sparse_spread(count));
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| black_box(Bvh::build(black_box(&boxes))))
        });
    }
    group.finish();
}

fn query_pairs_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_query_pairs_sparse");
    for count in COUNTS {
        let boxes = bounds(count, sparse_spread(count));
        let bvh = Bvh::build(&boxes);
        let mut pairs = Vec::new();
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| {
                pairs.clear();
                bvh.query_pairs(&mut pairs);
                black_box(pairs.len())
            })
        });
    }
    group.finish();
}

fn query_pairs_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_query_pairs_dense");
    for count in COUNTS {
        let boxes = bounds(count, dense_spread(count));
        let bvh = Bvh::build(&boxes);
        let mut pairs = Vec::new();
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| {
                pairs.clear();
                bvh.query_pairs(&mut pairs);
                black_box(pairs.len())
            })
        });
    }
    group.finish();
}

/// Build plus query together — what `collision::run` actually pays per frame.
fn build_and_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_build_and_query");
    for count in COUNTS {
        let boxes = bounds(count, dense_spread(count));
        let mut pairs = Vec::new();
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| {
                pairs.clear();
                Bvh::build(black_box(&boxes)).query_pairs(&mut pairs);
                black_box(pairs.len())
            })
        });
    }
    group.finish();
}

/// The same per-frame work as `build_and_query`, but against one tree that is
/// rebuilt in place. The gap between the two groups is what `collision::run`
/// stopped paying the allocator every frame.
fn rebuild_and_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_rebuild_and_query");
    for count in COUNTS {
        let boxes = bounds(count, dense_spread(count));
        let mut bvh = Bvh::default();
        let mut pairs = Vec::new();
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(format!("{count}"), |b| {
            b.iter(|| {
                pairs.clear();
                bvh.rebuild(black_box(&boxes));
                bvh.query_pairs(&mut pairs);
                black_box(pairs.len())
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    build,
    query_pairs_sparse,
    query_pairs_dense,
    build_and_query,
    rebuild_and_query
);
criterion_main!(benches);
