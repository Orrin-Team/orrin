//! Renders the built-in scenes to PNGs with no window.
//!
//! `#[ignore]`d because it needs a GPU and CI has none — the render graph's
//! golden is the part of the renderer CI can assert, and it asserts
//! synchronisation rather than pixels. This is the other half: a lighting term
//! that silently goes dark, a material that stops binding, a pass that
//! composites the wrong way round, none of which move the barrier plan at all.
//!
//! Run it with:
//!
//! ```text
//! cargo test -p orrin-core --test offscreen -- --ignored --nocapture
//! ```
//!
//! and look at what lands in `target/capture/`. Most of these are an A/B set on
//! purpose: the same rig scene with each non-opaque queue switched off, and with
//! the subsurface diffusion switched off, and the same air marched as froxels and
//! integrated analytically — which is what turns "this looks wrong" into "this
//! looks wrong because of that pass".
//!
//! The last two are the exception and are a different kind of file. `showcase`
//! and `showcase-materials` photograph the courtyard, where every feature is
//! contributing to one image at once, so neither is evidence about any single
//! pass. They are the check the A/B set cannot make: that the features compose.

use std::path::PathBuf;

use orrin_core::capture::capture_scene;

mod common;

fn output_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/capture");
    std::fs::create_dir_all(&dir).expect("failed to create the capture directory");
    dir
}

#[test]
#[ignore = "needs a GPU"]
fn captures_the_built_in_scenes() {
    orrin_core::logging::init(orrin_core::logging::LogConfig::for_this_build());
    let dir = output_dir();

    for (name, settings) in common::captures() {
        let path = dir.join(format!("{name}.png"));
        capture_scene(&path, &settings);

        let written = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("{} was not written: {e}", path.display()))
            .len();
        // A PNG of a frame that rendered nothing still has a header, so the
        // assertion is on something a blank image could not produce.
        assert!(
            written > 4096,
            "{} is {written} bytes, which is too small to be a rendered frame",
            path.display(),
        );
        println!("wrote {} ({written} bytes)", path.display());
    }
}
