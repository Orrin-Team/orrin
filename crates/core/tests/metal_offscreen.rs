//! The built-in scenes rendered through native Metal, held against the same
//! scenes rendered through Vulkan (MoltenVK on a Mac).
//!
//! They are not bit-identical: the two paths compile the shaders differently and
//! round differently. What this asserts is that the frame is a frame (not black,
//! not uniform, not NaN) and that it is close to the reference, and it prints
//! PSNR and the largest channel difference per capture so the threshold is a
//! number someone chose rather than one the code assumed.
//!
//! ```text
//! cargo test -p orrin-core --test metal_offscreen -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use orrin_core::capture::{scene_pixels, scene_pixels_metal};

mod common;

fn output_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/capture-metal");
    std::fs::create_dir_all(&dir).expect("failed to create the capture directory");
    dir
}

/// The least PSNR a capture may have against Vulkan. The plan proposes 40 dB as
/// a start, and the first full table clears it with room: the lowest is 44 dB.
const THRESHOLD_DB: f64 = 40.0;

struct Difference {
    psnr: f64,
    max: u8,
}

fn compare(reference: &[u8], metal: &[u8]) -> Difference {
    assert_eq!(
        reference.len(),
        metal.len(),
        "the two frames differ in size"
    );
    let (mut squared, mut max) = (0.0f64, 0u8);
    let mut counted = 0u64;
    for (a, b) in reference.chunks(4).zip(metal.chunks(4)) {
        for channel in 0..3 {
            let d = a[channel].abs_diff(b[channel]);
            max = max.max(d);
            squared += f64::from(d) * f64::from(d);
            counted += 1;
        }
    }
    let mse = squared / counted as f64;
    Difference {
        psnr: if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        },
        max,
    }
}

/// A frame that is one colour everywhere, or black, is a failed render and not
/// a close one.
fn assert_a_picture(name: &str, pixels: &[u8]) {
    let first = &pixels[..3];
    assert!(
        pixels.chunks(4).any(|p| &p[..3] != first),
        "{name}: the Metal frame is a single colour {first:?}"
    );
    assert!(
        pixels.chunks(4).any(|p| p[..3].iter().any(|&c| c > 8)),
        "{name}: the Metal frame is black"
    );
}

fn save(path: PathBuf, pixels: &[u8], extent: [u32; 2]) {
    image::save_buffer(path, pixels, extent[0], extent[1], image::ColorType::Rgba8)
        .expect("failed to write a capture");
}

#[test]
#[ignore = "needs a Metal device"]
fn metal_renders_the_built_in_scenes_like_vulkan_does() {
    orrin_core::logging::init(orrin_core::logging::LogConfig::for_this_build());
    let dir = output_dir();
    let mut failures = Vec::new();

    println!("{:<22} {:>8} {:>6}", "capture", "PSNR dB", "max");
    for (name, settings) in common::captures() {
        let (reference, extent) = scene_pixels(&settings);
        let (metal, metal_extent) = scene_pixels_metal(&settings);
        assert_eq!(extent, metal_extent);
        save(dir.join(format!("{name}-vulkan.png")), &reference, extent);
        save(dir.join(format!("{name}-metal.png")), &metal, extent);
        assert_a_picture(name, &metal);
        let difference = compare(&reference, &metal);
        println!("{name:<22} {:>8.2} {:>6}", difference.psnr, difference.max);
        if difference.psnr < THRESHOLD_DB {
            failures.push(format!("{name}: {:.2} dB", difference.psnr));
        }
    }
    assert!(failures.is_empty(), "too far from Vulkan: {failures:?}");
}
