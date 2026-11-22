//! What the shadow atlas holds across frames, on a GPU.
//!
//! The atlas is not redrawn whole every frame: a view keeps its tile until
//! something says the tile is out of date, and a view with something moving in
//! it puts back only the texels that thing touched. So the failure this guards
//! is a *stale* shadow — one that stays where its caster was, or where its
//! light was, or where the camera used to be standing — and a still capture
//! cannot show it, because depth left behind by the same geometry in the same
//! place is the right depth.
//!
//! Each case therefore renders a scene, changes it over some frames, and holds
//! the result against a renderer that has only ever seen the scene as it ended
//! up. `#[ignore]`d for the reason `offscreen.rs` is: it needs a GPU.
//!
//! ```text
//! cargo test -p orrin-core --test shadow_cache -- --ignored --nocapture
//! ```

use glam::Vec3;
use orrin_core::capture::{self, CaptureSettings};
use orrin_core::gfx::vulkan::VulkanRenderer;
use orrin_core::scene::{Camera, HdrSettings, Light, LocalTransform, Spin};
use orrin_ecs::{Entity, World};

/// Frames before anything changes: long enough for every caster to have been
/// baked into the cached layers, so that what the change has to undo is there.
const WARM: u32 = 24;
/// Frames after the last change, so that whatever the change started — a
/// caster settling, a cached layer waiting its turn — has finished.
const SETTLE: u32 = 16;

/// Close above a gap between cubes, with every kind of light casting into it.
fn settings() -> CaptureSettings {
    CaptureSettings {
        shadows: true,
        punctual_shadows: true,
        // No temporal resolve: its history is a function of every frame a
        // renderer has drawn, and a renderer that watched the scene change has
        // drawn different ones.
        taa: false,
        extent: [640, 360],
        camera: Some(Camera {
            position: Vec3::new(-4.0, 5.0, 8.0),
            target: Vec3::new(-4.0, -0.5, 3.0),
            ..Camera::default()
        }),
        ..CaptureSettings::default()
    }
}

fn renderer(metal: bool) -> VulkanRenderer {
    if metal {
        #[cfg(target_os = "macos")]
        return VulkanRenderer::offscreen_metal(settings().extent)
            .expect("this machine runs Metal");
        #[cfg(not(target_os = "macos"))]
        unreachable!("Metal is asked for on a Mac alone");
    }
    capture::offscreen_renderer(settings().extent)
}

/// Build the rig with its exposure held where the scene's own metering puts
/// it. Adaptation is a function of everything a renderer has looked at, so a
/// renderer that watched the camera walk and one that did not would otherwise
/// differ at every pixel, by an amount that says nothing about a shadow.
fn prepare(world: &mut World, renderer: &mut VulkanRenderer) {
    capture::prepare(world, renderer, &settings());
    let mut hdr = world.resource_mut::<HdrSettings>();
    hdr.auto_exposure = false;
    hdr.manual_ev100 = 6.5;
}

/// The cube of the rig's grid standing nearest `at`.
fn cube_at(world: &World, at: Vec3) -> Entity {
    let mut nearest = None;
    world
        .query::<(&LocalTransform, &Spin)>()
        .for_each(|entity, (transform, _)| {
            let distance = transform.0.translation.distance(at);
            if nearest.is_none_or(|(best, _)| distance < best) {
                nearest = Some((distance, entity));
            }
        });
    nearest.expect("the rig has cubes").1
}

fn the_spot(world: &World) -> Entity {
    let mut spot = None;
    world.query::<&Light>().for_each(|entity, light| {
        if matches!(light, Light::Spot { .. }) {
            spot = Some(entity);
        }
    });
    spot.expect("the rig has a spot light")
}

fn translate(world: &mut World, entity: Entity, by: Vec3) {
    world
        .get_mut::<LocalTransform>(entity)
        .expect("it has a transform")
        .0
        .translation += by;
}

/// Both halves of a comparison land in `target/capture`, because a failure
/// here is a picture of where the shadow stayed.
fn save(name: &str, pixels: &[u8], extent: [u32; 2]) {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/capture");
    std::fs::create_dir_all(&dir).expect("failed to create the capture directory");
    image::save_buffer(
        dir.join(format!("{name}.png")),
        pixels,
        extent[0],
        extent[1],
        image::ColorType::Rgba8,
    )
    .expect("failed to write a capture");
}

/// The share of pixels two frames differ at by more than a few levels.
fn differing(a: &[u8], b: &[u8]) -> f64 {
    let differ = a
        .chunks(4)
        .zip(b.chunks(4))
        .filter(|(a, b)| (0..3).any(|c| a[c].abs_diff(b[c]) > 12))
        .count();
    differ as f64 / (a.len() / 4) as f64
}

/// Render the rig, apply `step` once a frame for `steps` frames, and compare
/// what is on screen afterwards with a renderer that was shown the result from
/// its first frame.
///
/// Both run the same number of frames, so that the two differ in what they
/// were shown and in nothing else.
fn holds(metal: bool, name: &str, steps: u32, step: impl Fn(&mut World, u32)) {
    let settings = settings();
    let frames = |frames| CaptureSettings {
        frames,
        ..self::settings()
    };
    let backend = if metal { "metal" } else { "vulkan" };

    // A renderer that watches it happen.
    let mut watched = renderer(metal);
    let mut world = World::new();
    prepare(&mut world, &mut watched);
    let (before, _) = capture::render(&mut world, &mut watched, &frames(WARM));
    for index in 0..steps {
        step(&mut world, index);
        capture::render(&mut world, &mut watched, &frames(1));
    }
    let (after, _) = capture::render(&mut world, &mut watched, &frames(SETTLE));

    // And one that has only ever seen how it ended.
    let mut fresh = renderer(metal);
    let mut world = World::new();
    prepare(&mut world, &mut fresh);
    for index in 0..steps {
        step(&mut world, index);
    }
    let (expected, _) = capture::render(&mut world, &mut fresh, &frames(WARM + steps + SETTLE));

    for (half, pixels) in [("watched", &after), ("fresh", &expected)] {
        save(
            &format!("shadow-cache-{name}-{backend}-{half}"),
            pixels,
            settings.extent,
        );
    }
    let changed = differing(&before, &after);
    let stale = differing(&after, &expected);
    println!(
        "{backend:>6} {name:<8} changed {:5.2}% of the frame; {:.3}% differs from a fresh render",
        changed * 100.0,
        stale * 100.0
    );
    assert!(
        changed > 0.01,
        "{name}: only {changed} of the frame changed: the test is looking at nothing"
    );
    assert!(
        stale < 0.001,
        "{name}: {stale} of the frame differs from a renderer that never saw the scene \
         as it was: a shadow stayed behind"
    );
}

const CUBE: Vec3 = Vec3::new(-3.0, 0.0, 3.0);

/// Every way a cached shadow can go stale, one after another on one backend.
fn nothing_stays_behind(metal: bool) {
    // A caster that jumps: baked into the cached layers where it stood, which
    // have to be redrawn without it.
    holds(metal, "lift", 1, |world, _| {
        let cube = cube_at(world, CUBE);
        translate(world, cube, Vec3::Y * 2.0);
    });
    // One that slides: every frame it has to be drawn where it is and the
    // texels it touched the frame before put back.
    holds(metal, "slide", 30, |world, index| {
        // Found again each frame by where it has got to.
        let cube = cube_at(world, CUBE + Vec3::new(0.04, 0.05, 0.0) * index as f32);
        translate(world, cube, Vec3::new(0.04, 0.05, 0.0));
    });
    // One that is gone.
    holds(metal, "despawn", 1, |world, _| {
        let cube = cube_at(world, CUBE);
        world.despawn(cube);
    });
    // A light that moves: no cached layer survives a new matrix.
    holds(metal, "light", 20, |world, _| {
        let spot = the_spot(world);
        translate(world, spot, Vec3::new(0.08, 0.0, -0.05));
    });
    // And a camera that walks, far enough for the sun's nearest views to
    // follow it several times: what they hold is slid, not redrawn.
    holds(metal, "walk", 60, |world, _| {
        let mut camera = world.resource_mut::<Camera>();
        let by = Vec3::new(0.09, 0.0, -0.05);
        camera.position += by;
        camera.target += by;
    });
}

#[test]
#[ignore = "needs a GPU"]
fn nothing_stays_behind_on_vulkan() {
    nothing_stays_behind(false);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs a Metal device"]
fn nothing_stays_behind_on_metal() {
    nothing_stays_behind(true);
}
