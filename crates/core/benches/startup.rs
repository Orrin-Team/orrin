//! Cold-start baseline (#8).
//!
//! Architecture §6 makes "editor cold start stays under a few seconds" an
//! invariant and asks for a CI benchmark, on the grounds that startup decays one
//! dependency at a time and no single addition is ever the one that looks
//! expensive. This is that measurement.
//!
//! It covers the CPU half of startup: component registration, the default
//! resource set, mesh generation, texture decode and synthesis, and the scene
//! graph — everything up to the point a device would be needed, driven through
//! [`HeadlessBackend`] so it runs on a machine with no GPU. Vulkan instance and
//! device creation are *not* here and cannot be: CI has no adapter. That is a
//! real limit on what this guards, and it is the right trade, because device
//! init is a fixed cost of the driver while the CPU half is the part that grows
//! with every feature added to the engine.
//!
//! The number this prints is the baseline; `tests/cold_start.rs` is what
//! actually fails a build.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use orrin_core::App;
use orrin_core::gfx::HeadlessBackend;
use orrin_core::scene::entities::build_default_scene;
use orrin_core::scene::register_components;
use orrin_ecs::World;
use orrin_registry::Registry;

/// Everything the engine does before it could present a frame, minus the device.
fn boot() -> (World, HeadlessBackend) {
    let mut world = World::new();
    let mut registry = Registry::new();
    register_components(&mut registry);
    App::install_default_resources(&mut world);

    let mut backend = HeadlessBackend::new();
    build_default_scene(&mut world, &mut backend);
    (world, backend)
}

fn cold_start(c: &mut Criterion) {
    // Guard against the benchmark quietly measuring an empty scene, which is
    // the one failure mode that would make every future number meaningless.
    let (world, backend) = boot();
    let (meshes, materials, textures) = backend.upload_counts();
    assert!(meshes >= 3 && materials > 0 && textures > 0, "headless boot uploaded nothing");
    assert!(world.entities().count() > 1, "headless boot built no scene");

    c.bench_function("cold_start", |b| b.iter(|| black_box(boot())));
}

criterion_group!(benches, cold_start);
criterion_main!(benches);
