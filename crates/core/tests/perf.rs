//! Where a frame's time goes, per phase and per pass, with no window — and, in
//! [`model_load_cost`], where an import's time goes across the thread axis.
//!
//! Both are `#[ignore]`d, for the reason `offscreen.rs` is: the frame harness
//! needs a GPU and CI has none, and the load harness needs a model on disk that
//! the repository does not carry. `frame_cost` is the measuring half of the
//! `offscreen` pair — `offscreen` says the frame still looks right, this says
//! what it costs — and it exists so an optimisation is chosen against a table
//! rather than against an intuition about which pass is expensive.
//!
//! Run it with:
//!
//! ```text
//! cargo test --release -p orrin-core --test perf -- --ignored --nocapture
//! ```
//!
//! and read the two tables it prints: the CPU lane is the phases this harness
//! opens (extract, record, submit), the GPU lane is every node the compiled
//! graph ran. Environment:
//!
//! - `ORRIN_SCENE` picks the scene, exactly as the engine does.
//! - `ORRIN_STRESS` adds load, exactly as the engine does — `ORRIN_STRESS=4000`
//!   is what turns the CPU lane into a readable signal, since the built-in
//!   scenes have too few entities for extraction to show up at all.
//! - `ORRIN_PERF_FRAMES` is how many frames to average over (default 200).
//! - `ORRIN_PERF_EXTENT` is `WIDTHxHEIGHT` (default 1920x1080).
//! - `ORRIN_PERF_GPU_PASSES=0` drops the per-pass GPU rows, leaving a
//!   whole-frame time the timestamps are not themselves lengthening.
//! - `ORRIN_PERF_EMPTY=1` despawns everything with a mesh after the scene is
//!   built, leaving the camera, the lights and the environment: what the frame
//!   costs with nothing in it.
//! - `ORRIN_PERF_SSAO_HALF=0` resolves the occlusion at the full extent, which
//!   is the pair of numbers the `SsaoSettings::half_resolution` default is
//!   judged against.
//! - `ORRIN_THREADS` sizes the worker pool, exactly as the engine does, and the
//!   header names what it got. `ORRIN_THREADS=1` is a genuinely serial build
//!   (see [`orrin_core::threads`]), which is the baseline every parallel change
//!   is judged against — so a threaded change is measured by running this twice,
//!   back to back in one session, not against a number from last week.
//! - `ORRIN_ASYNC_COMPUTE=0` keeps the frame on one queue, which is the control
//!   every measurement of the split is read against. Like `ORRIN_THREADS=1`, it
//!   is one binary and one variable, so the two sides of an A/B need no rebuild
//!   between them.
//! - `ORRIN_PERF_POST=1` switches on every optical stage that ships off by
//!   default — screen-space reflections, depth of field, motion blur,
//!   subsurface scattering and volumetric fog. The frame the async-compute
//!   question is actually about: with them off, the whole compute half of the
//!   graph is a fifth of a millisecond.
//! - `ORRIN_PERF_MSAA=1` rasterises the forward pass at four samples. The other
//!   half of the comparison `TaaSettings::msaa` exists for: multisampled, the
//!   pass writes and resolves its own targets and cannot depth-test `EQUAL`
//!   against the prepass, so it rasterises every triangle a second time.
//!
//! Per-pass GPU timing serialises adjacent nodes that would otherwise overlap
//! (see `profile::gpu_passes_enabled`), so the GPU rows are a breakdown, not a
//! frame time. The whole-frame row is the number to compare across runs.

use std::sync::Arc;

use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};

use orrin_core::gfx::punctual::{MAX_SHADOW_LIGHTS, ShadowAtlas};
use orrin_core::gfx::shadows::{CascadeSet, MAX_CASCADES, cascades};
use orrin_core::gfx::vulkan::{ShadowFrame, VulkanRenderer};
use orrin_core::gfx::{DecalInstance, DrawList, RenderBackend, SceneLighting};
use orrin_core::profile::{self, Lane, Profiler};
use orrin_core::scene::entities::{SceneChoice, StressSpec, spawn_stress_scene};
use orrin_core::scene::model::{self, ImportSettings};
use orrin_core::scene::{
    BloomSettings, Camera, ContactShadowSettings, Culling, DecalSettings, DofSettings,
    EnvironmentSettings, FogSettings, HdrSettings, MeshHandle, MotionBlurSettings,
    RefractionSettings, ShadowSettings, SsaoSettings, SsrSettings, SubsurfaceSettings, TaaSettings,
    TransparencySettings,
};
use orrin_core::systems::{self, FrameGeometry};
use orrin_ecs::World;

/// Frames whose numbers are thrown away: the temporal resolve has no history on
/// its first, auto-exposure adapts over several, and the graph compiles on the
/// first one it sees.
const WARMUP: u64 = 30;

fn env_usize(key: &str, fallback: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .unwrap_or(fallback)
}

fn extent() -> [u32; 2] {
    match std::env::var("ORRIN_PERF_EXTENT") {
        Ok(raw) => {
            let (w, h) = raw
                .split_once(['x', 'X'])
                .expect("ORRIN_PERF_EXTENT is WIDTHxHEIGHT");
            [
                w.trim().parse().expect("width"),
                h.trim().parse().expect("height"),
            ]
        }
        Err(_) => [1920, 1080],
    }
}

fn headless_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("failed to load vulkan library");
    Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .expect("failed to create a headless instance")
}

#[test]
#[ignore = "needs a GPU"]
fn frame_cost() {
    // The device and scene lines above the table are the engine's own, and
    // they are only printed once something is listening for them.
    orrin_core::logging::init(orrin_core::logging::LogConfig::for_this_build());
    // Before the scene is built: its texture decode is the first thing that
    // dispatches, and a pool built after that would not have been used for it.
    orrin_core::threads::init();
    let extent = extent();
    let frames = env_usize("ORRIN_PERF_FRAMES", 200) as u64;
    let instance = headless_instance();
    let mut renderer = VulkanRenderer::offscreen(&instance, extent);

    let mut world = World::new();
    world.insert_resource(MotionBlurSettings::default());
    world.insert_resource(DofSettings::default());
    world.insert_resource(BloomSettings::default());
    world.insert_resource(HdrSettings::default());
    world.insert_resource(EnvironmentSettings::default());
    world.insert_resource(FogSettings::default());

    let scene = SceneChoice::from_env();
    scene.build(&mut world, &mut renderer);
    if let Some(spec) = StressSpec::from_env() {
        spawn_stress_scene(&mut world, &spec);
    }
    // The floor: every pass still runs, with nothing to draw into them. Built
    // from a real scene and then emptied rather than from an empty world,
    // because the camera, the sun and the environment are what the screen-space
    // passes cost anything at all against — a world with no sun is a frame with
    // no shadow passes, which is a different graph rather than a lighter one.
    if env_usize("ORRIN_PERF_EMPTY", 0) != 0 {
        let drawn: Vec<_> = world
            .entities()
            .filter(|&entity| world.has::<MeshHandle>(entity))
            .collect();
        for entity in drawn {
            world.despawn(entity);
        }
    }

    world.insert_resource(SsaoSettings {
        half_resolution: env_usize("ORRIN_PERF_SSAO_HALF", 1) != 0,
        ..SsaoSettings::default()
    });
    world.insert_resource(ContactShadowSettings::default());
    // Every stage that ships disabled, switched on together: each one is a
    // handful of compute nodes, and the frame they make between them is the one
    // a scheduling change has anything to work with.
    let post = env_usize("ORRIN_PERF_POST", 0) != 0;
    world.insert_resource(SsrSettings {
        enabled: post,
        ..SsrSettings::default()
    });
    world.insert_resource(SubsurfaceSettings {
        enabled: post,
        ..SubsurfaceSettings::default()
    });
    world.insert_resource(DofSettings {
        enabled: post,
        ..DofSettings::default()
    });
    world.insert_resource(MotionBlurSettings {
        enabled: post,
        ..MotionBlurSettings::default()
    });
    // The one that is not a boolean: fog is volumetric already and costs
    // nothing at zero density, so this is what puts the froxel grid in the
    // frame.
    world.insert_resource(FogSettings {
        density: if post { 0.02 } else { 0.0 },
        ..FogSettings::default()
    });
    world.insert_resource(DecalSettings::default());
    world.insert_resource(TransparencySettings::default());
    world.insert_resource(RefractionSettings::default());
    world.insert_resource(TaaSettings {
        msaa: env_usize("ORRIN_PERF_MSAA", 0) != 0,
        ..TaaSettings::default()
    });
    world.insert_resource(Culling::default());
    world.insert_resource(ShadowSettings {
        enabled: env_usize("ORRIN_PERF_SHADOWS", 1) != 0,
        ..ShadowSettings::default()
    });

    let mut lighting = SceneLighting::default();
    let mut geometry = FrameGeometry::default();
    let mut decals: Vec<DecalInstance> = Vec::new();
    let mut cascade_set;
    let mut atlas;
    let aspect = extent[0] as f32 / extent[1] as f32;

    // Big enough that a GPU readback landing several frames late still finds the
    // frame it belongs to; the aggregate below reads the whole ring.
    let mut profiler = Profiler::new((frames + WARMUP) as usize + 8);
    profile::set_enabled(true);
    // The per-pass breakdown costs the overlap it measures: a timestamp around
    // every node drains passes the GPU would otherwise run together. Off, the
    // whole-frame row is what the hardware actually does, and the gap between
    // the two runs is the overlap the schedule is getting.
    let gpu_passes = env_usize("ORRIN_PERF_GPU_PASSES", 1) != 0;
    profile::set_gpu_passes_enabled(gpu_passes);

    let mut entities = 0usize;
    let mut draws = 0usize;
    let mut entries = 0usize;
    for frame in 0..frames + WARMUP {
        // The engine throws its numbers away for the first few frames; so does
        // this, by restarting the profiler once the transient is over.
        if frame == WARMUP {
            profiler = Profiler::new(frames as usize + 8);
        }
        {
            let _phase = profile::scope("extract");
            {
                let _s = profile::scope("propagate");
                orrin_core::scene::propagate_transforms(&mut world);
            }
            {
                let _s = profile::scope("lighting");
                systems::extract_lighting(&world, &mut lighting);
            }
            {
                let _s = profile::scope("decals");
                systems::extract_decals(&world, aspect, &mut decals);
            }
            let shadow_settings = *world.resource::<ShadowSettings>();
            let camera = *world.resource::<Camera>();
            cascade_set = if shadow_settings.enabled {
                cascades(
                    &camera,
                    aspect,
                    lighting.sun.direction,
                    &shadow_settings.cascade_config(),
                )
            } else {
                CascadeSet::default()
            };
            atlas = match shadow_settings.atlas_config() {
                Some(config) => orrin_core::gfx::punctual::fit(&lighting, camera.position, &config),
                None => ShadowAtlas::default(),
            };
            {
                let _s = profile::scope("geometry");
                systems::extract_geometry(
                    &world,
                    aspect,
                    &cascade_set,
                    &atlas,
                    renderer.gpu_culling(),
                    &mut geometry,
                );
            }
            entities = geometry.visible().len();
            // What the CPU actually records: one `draw_indexed` per maximal
            // (mesh, material) run, per list. The entity count says how much
            // there is to draw; this says how many commands saying so the
            // recording costs, which is the number the per-draw work scales
            // with.
            // Every entry of every list, which is what the frame used to
            // upload a 192-byte object row for and now uploads four bytes of
            // row number for. Larger than the entity count because an object
            // the camera sees and four cascades also draw appears in five.
            entries = geometry.visible().len()
                + geometry.transparent().len()
                + geometry.refractive().len()
                + (0..MAX_CASCADES)
                    .map(|i| geometry.cascade(i).len())
                    .sum::<usize>()
                + (0..MAX_SHADOW_LIGHTS)
                    .map(|i| geometry.punctual(i).len())
                    .sum::<usize>();
            draws = geometry.visible().runs().count()
                + geometry.transparent().runs().count()
                + geometry.refractive().runs().count()
                + (0..MAX_CASCADES)
                    .map(|i| geometry.cascade(i).runs().count())
                    .sum::<usize>()
                + (0..MAX_SHADOW_LIGHTS)
                    .map(|i| geometry.punctual(i).runs().count())
                    .sum::<usize>();
        }

        let camera = *world.resource::<Camera>();
        let shadow_settings = *world.resource::<ShadowSettings>();
        {
            let _phase = profile::scope("render submit");
            let caster_lists: [DrawList<'_>; MAX_CASCADES] =
                std::array::from_fn(|i| geometry.cascade(i));
            let punctual_lists: [DrawList<'_>; MAX_SHADOW_LIGHTS] =
                std::array::from_fn(|i| geometry.punctual(i));
            renderer.render_with_overlay(
                geometry.visible(),
                geometry.transparent(),
                geometry.refractive(),
                &decals,
                &lighting,
                &camera,
                &world.resource::<SsaoSettings>().clone(),
                &world.resource::<ContactShadowSettings>().clone(),
                &world.resource::<SsrSettings>().clone(),
                &world.resource::<SubsurfaceSettings>().clone(),
                &world.resource::<TransparencySettings>().clone(),
                &world.resource::<RefractionSettings>().clone(),
                &world.resource::<TaaSettings>().clone(),
                &world.resource::<MotionBlurSettings>().clone(),
                &world.resource::<DofSettings>().clone(),
                &world.resource::<BloomSettings>().clone(),
                &world.resource::<HdrSettings>().clone(),
                &world.resource::<EnvironmentSettings>().clone(),
                &world.resource::<FogSettings>().clone(),
                1.0 / 60.0,
                &[],
                profiler.frame_index(),
                (cascade_set.count > 0).then_some(ShadowFrame {
                    cascades: &cascade_set,
                    casters: &caster_lists,
                    atlas: &atlas,
                    punctual_casters: &punctual_lists,
                    settings: &shadow_settings,
                }),
                None,
            );
        }
        renderer.drain_gpu_spans(&mut profiler);
        profiler.end_frame();
    }

    println!(
        "\nscene {scene:?} at {}x{}, {frames} frames, {} worker threads, {entities} visible \
         items, {draws} recorded draws, {entries} list entries",
        extent[0],
        extent[1],
        orrin_core::threads::count(),
    );
    // The row for an optimisation that costs no frame time. Transient aliasing
    // moves this number and nothing else in this file, so without it there is no
    // table to choose it against — `ORRIN_ALIAS=0` is the other side.
    let (used, total) = renderer.gpu_memory();
    match used {
        Some(used) => println!(
            "     {:<28} {:>9.1} MB of {:.0} MB",
            "device-local memory",
            used as f64 / 1e6,
            total as f64 / 1e6,
        ),
        None => println!(
            "     {:<28} not reported by this driver",
            "device-local memory"
        ),
    }
    for lane in [Lane::Cpu, Lane::Gpu] {
        let mut rows = profiler.aggregate(lane);
        rows.sort_by(|a, b| b.avg_ms.total_cmp(&a.avg_ms));
        println!(
            "\n{lane:?}: {:<28} {:>9} {:>9} {:>7}",
            "pass", "avg ms", "max ms", "calls"
        );
        for row in rows {
            println!(
                "     {:<28} {:>9.3} {:>9.3} {:>7}",
                row.name, row.avg_ms, row.max_ms, row.calls
            );
        }
    }
}

/// What importing a model costs, at each point on the thread axis, back to back
/// in one process.
///
/// Back to back is the whole point. A load measured today against a number from
/// last week compares two machines' page caches as much as two builds, and the
/// first load of a `.gltf` pays for pulling seventy JPEGs off disk that every
/// later one finds in cache. So this loads once to warm, then loads again at
/// each setting, and prints the settings beside each other.
///
/// The phase split it reports is the one the axis can actually separate: decode
/// is the only part of an import that runs on the pool, so the row at one thread
/// is `serial + decode` and the difference from the widest row is what
/// parallelism took off the decode. Whatever does not shrink is the serial
/// remainder — glTF parse, buffer reads, and the mesh import.
///
/// ```text
/// cargo test --release -p orrin-core --test perf -- --ignored model_load --nocapture
/// ```
///
/// - `ORRIN_SPONZA` names the model, exactly as the `sponza` scene does; the
///   default is the path `scripts/fetch-sponza.sh` writes.
/// - `ORRIN_PERF_THREADS` is the axis, comma separated — the default is `1` and
///   whatever `ORRIN_THREADS` resolved to.
/// - `ORRIN_PERF_LOADS` averages each point over that many loads (default 3).
#[test]
#[ignore = "needs a model on disk"]
fn model_load_cost() {
    orrin_core::logging::init(orrin_core::logging::LogConfig::for_this_build());
    orrin_core::threads::init();

    let path = model_path();
    let settings = ImportSettings::default();
    let loads = env_usize("ORRIN_PERF_LOADS", 3).max(1);

    let axis: Vec<usize> = match std::env::var("ORRIN_PERF_THREADS") {
        Ok(raw) => raw
            .split(',')
            .filter_map(|field| field.trim().parse::<usize>().ok())
            .filter(|&threads| threads > 0)
            .collect(),
        Err(_) => {
            let widest = orrin_core::threads::count();
            if widest > 1 { vec![1, widest] } else { vec![1] }
        }
    };
    assert!(
        !axis.is_empty(),
        "ORRIN_PERF_THREADS named no valid setting"
    );

    // Warm the page cache, and fail here rather than inside the timed loop if
    // the model is missing — a first row of "file not found" would otherwise be
    // reported as a very fast import.
    let warm =
        model::load(&path, &settings).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let images = warm.images.len();
    let megapixels: f64 = warm
        .images
        .iter()
        .map(|image| f64::from(image.width) * f64::from(image.height) / 1.0e6)
        .sum();
    println!(
        "\n{}: {} primitives, {} materials, {images} images, {megapixels:.1} MP decoded, \
         best of {loads} per setting",
        path.display(),
        warm.primitives.len(),
        warm.materials.len(),
    );
    drop(warm);

    println!(
        "\n{:>7} {:>10} {:>10} {:>9} {:>8}",
        "threads", "best ms", "mean ms", "MP/s", "speedup"
    );

    let mut baseline: Option<f64> = None;
    for &threads in &axis {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a measurement pool");
        // `threads::map` reads the pool it is *in*, so `install` is what moves
        // the axis without a fresh process per point — and at one thread it
        // takes the serial path rather than paying a pool of one.
        let samples: Vec<f64> = pool.install(|| {
            (0..loads)
                .map(|_| {
                    let start = std::time::Instant::now();
                    let model = model::load(&path, &settings).expect("the model loaded once");
                    let elapsed = start.elapsed().as_secs_f64() * 1.0e3;
                    drop(model);
                    elapsed
                })
                .collect()
        });

        let best = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let baseline = *baseline.get_or_insert(best);
        println!(
            "{threads:>7} {best:>10.1} {mean:>10.1} {:>9.1} {:>7.2}x",
            megapixels / (best / 1.0e3),
            baseline / best,
        );
    }
}

/// The model both harnesses below measure: `ORRIN_SPONZA`, or the path
/// `scripts/fetch-sponza.sh` writes — anchored at the workspace root rather than
/// the working directory, because the engine runs from the root and a test runs
/// from its package.
///
/// The default is repeated from `sponza::DEFAULT_PATH` rather than exported:
/// measuring an import is the wrong reason for one scene's private default to
/// become public API.
fn model_path() -> std::path::PathBuf {
    const DEFAULT_MODEL: &str = "assets/sponza/Sponza.gltf";
    match std::env::var_os("ORRIN_SPONZA") {
        Some(named) => std::path::PathBuf::from(named),
        None => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(DEFAULT_MODEL),
    }
}

/// The thread count must not be able to change a pixel.
///
/// Asserted where the change actually is rather than through a rendered frame:
/// decode is the only thing that runs on the pool, so a texture that came back
/// different — reordered, truncated, or a decoder with shared state — is the
/// only way threading could reach the screen at all. Comparing the imported
/// images directly says which image and how, where a golden PNG would only say
/// that something moved.
#[test]
#[ignore = "needs a model on disk"]
fn a_parallel_import_decodes_what_a_serial_one_does() {
    orrin_core::threads::init();
    let path = model_path();
    let settings = ImportSettings::default();

    let load = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a measurement pool")
            .install(|| {
                model::load(&path, &settings)
                    .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
            })
    };

    let serial = load(1);
    let parallel = load(orrin_core::threads::count().max(2));

    assert_eq!(
        serial.images.len(),
        parallel.images.len(),
        "the two imports disagree about how many images the model has",
    );
    for (index, (a, b)) in serial.images.iter().zip(&parallel.images).enumerate() {
        assert_eq!(
            (a.width, a.height),
            (b.width, b.height),
            "image {index} came back at a different size",
        );
        assert!(
            a.pixels == b.pixels,
            "image {index} decoded to different pixels",
        );
    }
}
