//! The cold-start tripwire architecture §6 asks for (#8).
//!
//! §6: *"Editor cold start stays under a few seconds, guarded by a CI
//! benchmark, because startup time decays one dependency at a time."* The decay
//! is the point — no single addition ever looks expensive, so the only thing
//! that catches it is a number checked on every push.
//!
//! **What this measures**, driven through [`HeadlessBackend`]: component
//! registration, the default resource set, mesh generation, texture decode and
//! synthesis, and the scene graph. **What it cannot**: Vulkan instance and
//! device creation, because CI has no adapter. Device init is a fixed driver
//! cost; the CPU half is the part that grows with every feature, so this is
//! where the decay actually shows up.
//!
//! **Why an absolute cap rather than a delta against a stored baseline.** A
//! shared CI runner's wall clock varies by more than any regression worth
//! catching, so a tight percentage gate would fail on neighbours rather than on
//! commits. The cap is set an order of magnitude above the observed cost: it
//! does not notice a 20% slip, and it does catch the thing §6 names — a
//! dependency that turns startup from milliseconds into seconds. For the
//! fine-grained number, run `cargo bench -p orrin-core --bench startup`.

use std::time::{Duration, Instant};

use orrin_core::App;
use orrin_core::gfx::HeadlessBackend;
use orrin_core::scene::entities::build_default_scene;
use orrin_core::scene::register_components;
use orrin_ecs::World;
use orrin_registry::Registry;

/// Deliberately loose; see the module docs. Override with
/// `ORRIN_COLD_START_BUDGET_MS` for a tighter local check.
const DEFAULT_BUDGET_MS: u64 = 1_500;

/// Enough runs that one descheduled sample cannot decide the result, few enough
/// that the test stays fast in a debug build.
const SAMPLES: usize = 5;

fn boot() -> (World, HeadlessBackend) {
    let mut world = World::new();
    let mut registry = Registry::new();
    register_components(&mut registry);
    App::install_default_resources(&mut world);

    let mut backend = HeadlessBackend::new();
    build_default_scene(&mut world, &mut backend);
    (world, backend)
}

#[test]
fn cold_start_stays_within_budget() {
    let budget = Duration::from_millis(
        std::env::var("ORRIN_COLD_START_BUDGET_MS")
            .ok()
            .and_then(|raw| raw.trim().parse().ok())
            .unwrap_or(DEFAULT_BUDGET_MS),
    );

    // One untimed run first: the first boot pays for lazily-initialized statics
    // and a cold allocator, neither of which is startup decay.
    let (world, backend) = boot();
    let (meshes, materials, textures) = backend.upload_counts();
    assert!(
        meshes >= 3 && materials > 0 && textures > 0,
        "the headless boot uploaded nothing ({meshes} meshes, {materials} materials, \
         {textures} textures) — this test would pass no matter how slow startup got"
    );
    assert!(world.entities().count() > 1, "the headless boot built no scene");

    let mut timings: Vec<Duration> = (0..SAMPLES)
        .map(|_| {
            let start = Instant::now();
            let booted = boot();
            let elapsed = start.elapsed();
            drop(booted);
            elapsed
        })
        .collect();
    timings.sort_unstable();
    let median = timings[SAMPLES / 2];

    println!("cold start (CPU half, headless): {:.1} ms", median.as_secs_f64() * 1e3);

    assert!(
        median <= budget,
        "cold start took {:.1} ms, over the {:.0} ms budget.\n\
         This is a tripwire for startup decay, not a tight regression gate — being over it \
         means something added to the boot path costs orders of magnitude more than the rest \
         of it, not that a recent change was a few percent slower.\n\
         Run `cargo bench -p orrin-core --bench startup` to see where it went.",
        median.as_secs_f64() * 1e3,
        budget.as_secs_f64() * 1e3,
    );
}
