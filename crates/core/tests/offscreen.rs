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

use glam::Vec3;
use orrin_core::capture::{CaptureSettings, capture_scene};
use orrin_core::scene::entities::SceneChoice;
use orrin_core::scene::{Camera, FogSettings};

/// Where both fog captures stand, shared so the pair differs in one field.
///
/// Low, and looking *along* the sun rather than across it: the demo's sun points
/// down and away from `+x`/`+z`, so a ray from here has `dot(L, dir)` near 0.75
/// and the Henyey-Greenstein term at `g = 0.7` is several times its isotropic
/// value. A camera facing the other way would see the same medium at a fraction
/// of the brightness — which is the anisotropy working, and the reason a fog
/// capture cannot be taken from wherever the scene shot happens to stand.
fn fog_camera() -> Camera {
    Camera {
        position: Vec3::new(-20.0, 1.0, -4.0),
        target: Vec3::new(-11.0, 4.5, 4.0),
        ..Camera::default()
    }
}

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

    for (name, settings) in [
        ("scene", CaptureSettings::default()),
        (
            "no-refraction",
            CaptureSettings {
                refraction: false,
                ..CaptureSettings::default()
            },
        ),
        (
            "opaque-only",
            CaptureSettings {
                transparency: false,
                refraction: false,
                ..CaptureSettings::default()
            },
        ),
        // The diffusion off, everything else as it ships. Scattering is still on
        // in both — `subsurface_wrap` widens to cover the missing passes — so the
        // difference between this and `scene.png` is what the two blurs and the
        // composite actually buy, on the marble and wax spheres in the lobe row.
        // It is also the shape that exercises the *other* forward render pass, so
        // a break in either variant shows up as one of the two files being wrong.
        (
            "no-subsurface-diffusion",
            CaptureSettings {
                subsurface: false,
                ..CaptureSettings::default()
            },
        ),
        // The same frame with nothing stamped on it. A decal is applied before
        // shading, so in `scene.png` it is indistinguishable from a surface that
        // was always that colour — which is the point of the technique and the
        // reason it cannot be checked by looking at one image. The difference
        // between this file and that one is every decal in the scene, and
        // nothing else: no pass is added or removed, no target is reallocated,
        // and the barrier plan is byte for byte the same.
        (
            "no-decals",
            CaptureSettings {
                decals: false,
                ..CaptureSettings::default()
            },
        ),
        // Close on the ground decals and the foliage behind them, because both
        // features live in the pixels and neither survives the demo camera's
        // distance. Three things are legible only from here. The crater's rim is
        // shaded by the sun and the point lights rather than painted, so it has
        // a lit side and a dark one. The decal stops where the ground turns up
        // into the cubes standing in it, which is the angle fade. And the leaf
        // cards' edges are resolved across the four MSAA samples the frame was
        // already paying for — against the sky they are a gradient, where a hard
        // alpha test would give the same silhouette in four steps.
        (
            "zoom-decals-foliage",
            CaptureSettings {
                camera: Some(Camera {
                    position: Vec3::new(-16.5, 3.2, 15.5),
                    target: Vec3::new(-10.5, 0.4, 5.5),
                    ..Camera::default()
                }),
                ..CaptureSettings::default()
            },
        ),
        // The only place the cutout's *caster* pipeline is visible. A leaf card
        // is a quad, so with the plain depth-only pipeline it casts the shadow of
        // a rectangle — the single most obvious way foliage goes wrong, and one
        // that no other capture here can show, because every other capture has
        // the cascades switched off. Look at the ground beneath the stand: leaves
        // and stems, with sky between them.
        (
            "zoom-foliage-shadow",
            CaptureSettings {
                shadows: true,
                camera: Some(Camera {
                    // Steeply down onto the open ground the stand throws its
                    // shadow across, with the cards themselves still in frame:
                    // the point is the pair, not either alone.
                    position: Vec3::new(-13.5, 7.5, 11.0),
                    target: Vec3::new(-12.0, 0.0, 1.5),
                    ..Camera::default()
                }),
                ..CaptureSettings::default()
            },
        ),
        // The two masonry walls, close and at a glancing angle, which is the one
        // vantage point where parallax occlusion mapping is legible: the left
        // wall marches the height field and the right one has the same albedo,
        // the same normals and no field to march. So this is an A/B inside a
        // single frame — same light, same exposure, same mip level — and the
        // difference between the two halves is the whole feature. Mortar that
        // sits behind its blocks on the left and level with them on the right,
        // and courses that stay parallel on the right while the left's shift
        // against each other with depth.
        (
            "zoom-parallax",
            CaptureSettings {
                // The scene's third decal is on this wall, deliberately — a
                // decal projected onto a marched height field is the sharpest
                // check that the prepass and the forward pass agree about where
                // a surface *is*. It is taken back out here, because this
                // capture's whole claim is that the only difference between the
                // upper panel and the lower one is the height map, and a mark
                // straddling the seam between them is a second variable. Each
                // capture isolates one feature; that is what the switches are
                // for.
                decals: false,
                camera: Some(Camera {
                    position: Vec3::new(3.76, 1.5, -12.99),
                    target: Vec3::new(0.0, 1.5, -13.0),
                    ..Camera::default()
                }),
                ..CaptureSettings::default()
            },
        ),
        // The froxel fog, and the same medium integrated analytically — the pair
        // is the capture, not either file. Both carry identical density,
        // falloff and albedo, so everything that differs between them is what
        // marching the volume buys: shafts where the cutout stand and the cubes
        // block the sun, and air that darkens inside their shadows. The analytic
        // one has no shadow term at all and cannot — past the froxels there is
        // no map deep enough to ask — so it is the same haze at a uniform
        // brightness.
        //
        // Shadows on in both, and that is not optional here: with no cascades
        // fitted the scatter pass's visibility is a constant 1.0 and the two
        // files would be near enough identical, which would make the A/B look
        // like the feature does nothing.
        //
        // The sun is low and the camera looks across it rather than along it, so
        // the anisotropy is doing visible work: at `g = 0.7` the air on the sun
        // side of the frame is several times brighter than the air away from it,
        // and that gradient is the phase function rather than the density.
        (
            "fog-volumetric",
            CaptureSettings {
                shadows: true,
                fog: Some(FogSettings {
                    density: 0.04,
                    height_falloff: 0.08,
                    volumetric: true,
                    ..FogSettings::default()
                }),
                camera: Some(fog_camera()),
                ..CaptureSettings::default()
            },
        ),
        (
            "fog-analytic",
            CaptureSettings {
                shadows: true,
                fog: Some(FogSettings {
                    density: 0.04,
                    height_falloff: 0.08,
                    volumetric: false,
                    ..FogSettings::default()
                }),
                camera: Some(fog_camera()),
                ..CaptureSettings::default()
            },
        ),
        // The courtyard, from the camera it is composed for: low sun through the
        // gateway, everything in frame backlit, and the air marched so the beam
        // has an edge. Shadows are not optional in either showcase capture — the
        // scatter pass's visibility is a constant 1.0 with no cascades fitted,
        // and a shaft is the shape of an occluder.
        //
        // `fog` is left `None`, unlike the pair above: this scene describes its
        // own medium and the capture is of the scene, not of a comparison.
        (
            "showcase",
            CaptureSettings {
                scene: SceneChoice::Showcase,
                shadows: true,
                ..CaptureSettings::default()
            },
        ),
        // The same courtyard from the one place the sun is behind the camera,
        // looking along the brick pier's lit face. Contre-jour is what makes the
        // shafts and the transmitted lobe legible and it is exactly wrong for
        // surface detail, so the two cameras split that: this is where the
        // parallax march, the decal sitting in it, and the flagstones' relief are
        // readable, and it is deliberately not a prettier version of the first.
        (
            "showcase-materials",
            CaptureSettings {
                scene: SceneChoice::Showcase,
                shadows: true,
                camera: Some(Camera {
                    // Two metres off the pier's face and four along it, which is
                    // 62° from its normal — the same obliquity the rig's
                    // `zoom-parallax` stands at, and for the same reason. It is
                    // not a matter of "as grazing as possible": past about 73°
                    // the march deliberately fades the field back to flat,
                    // because displacement without a silhouette shears the
                    // courses into diagonal smears. Standing closer to the wall
                    // photographs that fade instead of the relief.
                    position: Vec3::new(-0.8, 1.9, -7.0),
                    target: Vec3::new(1.28, 1.7, -3.0),
                    fov_y: 45f32.to_radians(),
                    ..Camera::default()
                }),
                ..CaptureSettings::default()
            },
        ),
    ] {
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
