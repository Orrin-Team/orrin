//! What the shadow atlas holds across frames, on a GPU.
//!
//! The atlas is not redrawn whole every frame: a view keeps its tile until
//! something says the tile is out of date. So the failure this guards is a
//! *stale* shadow — one that stays where its caster was, or where its light
//! was — and a still capture cannot show it, because depth left behind by the
//! same geometry in the same place is the right depth.
//!
//! Each case therefore renders a scene, changes it, renders it again, and holds
//! the result against a renderer that has only ever seen the scene as changed.
//! `#[ignore]`d for the reason `offscreen.rs` is: it needs a GPU.
//!
//! ```text
//! cargo test -p orrin-core --test shadow_cache -- --ignored --nocapture
//! ```

use glam::Vec3;
use orrin_core::capture::{self, CaptureSettings};
use orrin_core::gfx::vulkan::VulkanRenderer;
use orrin_core::scene::{Camera, LocalTransform};
use orrin_ecs::World;

/// Close above a gap between cubes, with every kind of light casting into it.
fn settings() -> CaptureSettings {
    CaptureSettings {
        shadows: true,
        punctual_shadows: true,
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

/// Lift the cube standing nearest `at` two metres off the floor, so its shadow
/// from every light lands somewhere else.
fn lift_the_cube_at(world: &mut World, at: Vec3) {
    let mut nearest = None;
    world
        .query::<(&LocalTransform, &orrin_core::scene::Spin)>()
        .for_each(|entity, (transform, _)| {
            let distance = transform.0.translation.distance(at);
            if nearest.is_none_or(|(best, _)| distance < best) {
                nearest = Some((distance, entity));
            }
        });
    let (_, entity) = nearest.expect("the rig has cubes");
    world
        .get_mut::<LocalTransform>(entity)
        .expect("the cube has a transform")
        .0
        .translation
        .y += 2.0;
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

fn a_moved_caster_takes_its_shadow_with_it(metal: bool) {
    let settings = settings();
    let cube = Vec3::new(-3.0, 0.0, 3.0);

    // A renderer that watches the cube move.
    let mut watched = renderer(metal);
    let mut world = World::new();
    capture::prepare(&mut world, &mut watched, &settings);
    let (before, _) = capture::render(&mut world, &mut watched, &settings);
    lift_the_cube_at(&mut world, cube);
    let (after, _) = capture::render(&mut world, &mut watched, &settings);

    // And one that has only ever seen it where it ended up — for as many
    // frames, because the temporal resolve's jitter and the adapted exposure
    // are both functions of how long a renderer has been running.
    let mut fresh = renderer(metal);
    let mut world = World::new();
    capture::prepare(&mut world, &mut fresh, &settings);
    lift_the_cube_at(&mut world, cube);
    capture::render(&mut world, &mut fresh, &settings);
    let (expected, _) = capture::render(&mut world, &mut fresh, &settings);

    let backend = if metal { "metal" } else { "vulkan" };
    for (name, pixels) in [("watched", &after), ("fresh", &expected)] {
        save(
            &format!("shadow-cache-{backend}-{name}"),
            pixels,
            settings.extent,
        );
    }

    let moved = differing(&before, &after);
    let stale = differing(&after, &expected);
    println!(
        "{backend}: the move changed {:.2}% of the frame; {:.3}% differs from a fresh render",
        moved * 100.0,
        stale * 100.0
    );
    assert!(
        moved > 0.01,
        "lifting the cube changed {moved} of the frame: the test is looking at nothing"
    );
    assert!(
        stale < 0.001,
        "{stale} of the frame differs from a renderer that never saw the cube \
         where it was: a shadow stayed behind"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn a_moved_caster_takes_its_shadow_with_it_on_vulkan() {
    a_moved_caster_takes_its_shadow_with_it(false);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs a Metal device"]
fn a_moved_caster_takes_its_shadow_with_it_on_metal() {
    a_moved_caster_takes_its_shadow_with_it(true);
}
