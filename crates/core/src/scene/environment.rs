use std::fmt;
use std::path::{Component, Path, PathBuf};

use glam::Vec3;

/// The environment the scene is drawn against and lit by.
///
/// The cubemap and its irradiance live in the renderer, baked from an
/// equirectangular source. These are the parameters that can change without a
/// rebake, plus the name of the source that would cause one.
#[derive(Clone, Debug)]
pub struct EnvironmentSettings {
    /// The HDRI to bake from, relative to the assets directory. Empty means
    /// whatever the scene loaded programmatically — for the built-in demo, its
    /// generated placeholder sky.
    ///
    /// Relative on purpose, and validated as such: this is the kind of field
    /// that ends up in a scene file, and an absolute path there is a machine
    /// that only builds on one desk.
    pub hdri: String,
    /// Set to ask the app to (re)bake from [`hdri`](Self::hdri); the app clears
    /// it once it has.
    ///
    /// A request rather than a load for the reason the build watcher is: the
    /// bake blocks on the GPU and replaces resources the frame in flight may
    /// still be reading, so it happens at the one point in the loop that is
    /// safe rather than wherever a button was clicked.
    pub reload_requested: bool,
    /// How bright this environment's sky is, in **cd/m²** — a clear zenith
    /// around 8 000, an overcast one 1 000–2 000, dusk in the tens.
    ///
    /// A stated luminance rather than a multiplier, and that is what makes an
    /// HDRI usable at all now that the lights around it are photometric. An
    /// `.hdr` carries *relative* radiance: nothing in the file says whether a
    /// pixel of 1.0 is a dim wall or a bright sky, and the ones people download
    /// land two to four orders of magnitude below real luminance. A raw
    /// multiplier makes that the author's problem — find 4 500 by dragging, and
    /// write a number into the scene that means nothing without that exact file
    /// beside it.
    ///
    /// So the bake measures the source's own sky luminance
    /// ([`sh::sky_luminance`](crate::gfx::sh::sky_luminance)) and the renderer
    /// scales by the ratio. The number here is then a fact about the sky being
    /// depicted rather than about the file depicting it: swap in a different
    /// capture of the same weather and the lighting does not move.
    ///
    /// Separate from [`HdrSettings`](super::HdrSettings) because that is the
    /// camera and this is the scene — exposure decides what a luminance looks
    /// like, not what it is.
    pub sky_luminance: f32,
    /// Extra stops on the environment, on top of the calibration above.
    ///
    /// The one control here that is deliberately *not* physical, because two
    /// things it cannot express are worth having anyway: an HDRI whose ground is
    /// wrong for the scene standing on it, and a look. Zero is the calibrated
    /// answer, and a scene that never touches this is lit by what its sky says
    /// it is.
    pub exposure_offset: f32,
    /// Rotation of the environment about world Y, in degrees. Applied to the
    /// sampling direction, so it costs nothing and needs no rebake.
    pub yaw: f32,
    /// Whether the environment is drawn as the background. Turning it off
    /// leaves [`background`](Self::background) showing.
    pub show_skybox: bool,
    /// What the forward pass clears its colour target to, in **cd/m²** — the
    /// same luminance the rest of the frame is in, since this is one more thing
    /// the exposure model has to expose correctly.
    ///
    /// Only ever visible where nothing was drawn: with a baked environment and
    /// [`show_skybox`](Self::show_skybox) on, the sky covers every such pixel and
    /// this is never seen. It is the background of a scene that has *no*
    /// environment, which is why the default is zero — an absence of sky is
    /// black, and any other value is a claim about a light source the scene does
    /// not have. (It also means the frame is not lit by whatever the background
    /// suggests: this is a clear value, not an ambient term.)
    ///
    /// Zero is also the value AMD documents as clearable on the compressed
    /// fast-clear path — RDNA takes it for the all-zeros and all-ones colours and
    /// writes the target in full for anything else — which is the second reason
    /// the default moved off the near-black navy this was previously hardcoded
    /// to.
    ///
    /// **Measured, so it is not overclaimed:** on an RX 9070 XT (RADV, GFX1201)
    /// at 1920x1080 the change is *not* visible in a frame time, at one sample or
    /// at four — 1.441 ms against 1.448 ms across five runs each, which is inside
    /// the run-to-run spread. Either RADV was already taking the same path for
    /// both colours or it takes it for neither. So the value of black here is a
    /// design one — an absence of sky is black, and the old value was a look
    /// nothing in the engine had chosen — with the fast-clear path as a reason it
    /// cannot cost anything on hardware that does distinguish them.
    pub background: Vec3,
}

impl Default for EnvironmentSettings {
    fn default() -> Self {
        Self {
            hdri: String::new(),
            reload_requested: false,
            // Dusk, matching the placeholder sky the demo scene is lit at. A
            // default has to be *some* weather; this is the one the engine ships
            // content for.
            sky_luminance: 30.0,
            exposure_offset: 0.0,
            yaw: 0.0,
            show_skybox: true,
            background: Vec3::ZERO,
        }
    }
}

impl EnvironmentSettings {
    /// The factor to scale sampled environment radiance by, given what the
    /// source's own sky measured.
    ///
    /// `measured` is in the file's units and [`sky_luminance`](Self::sky_luminance)
    /// in cd/m², so the ratio between them is the calibration; the offset is then
    /// stops on top of it. A source that measured nothing — a black image, or no
    /// environment at all — has no ratio to take, and 1.0 leaves whatever it holds
    /// alone rather than dividing by zero.
    ///
    /// One definition because three consumers scale by it: the irradiance probe,
    /// the specular tint, and the skybox draw. The first two are what lights the
    /// scene and the third is what stands behind it, so a copy that drifted would
    /// be geometry lit by one sky against a picture of a different one.
    pub fn calibration(&self, measured: f32) -> f32 {
        let scale = if measured > 1e-6 {
            self.sky_luminance.max(0.0) / measured
        } else {
            1.0
        };
        scale * self.exposure_offset.exp2()
    }
}

/// A decoded equirectangular radiance map: tightly packed RGBA f32, row-major
/// from the top-left, exactly what the renderer's bake consumes.
pub struct Hdri {
    pub pixels: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
pub enum HdriError {
    AbsolutePath(PathBuf),
    ParentDirInPath(PathBuf),
    Read {
        path: PathBuf,
        error: std::io::Error,
    },
    Decode {
        path: PathBuf,
        error: image::ImageError,
    },
}

impl fmt::Display for HdriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AbsolutePath(path) => write!(
                f,
                "environment `{}` must be relative to the assets directory; \
                 absolute paths are machine-local and must not be committed",
                path.display()
            ),
            Self::ParentDirInPath(path) => write!(
                f,
                "environment `{}` must stay inside the assets directory; \
                 remove the `..` components",
                path.display()
            ),
            Self::Read { path, error } => {
                write!(
                    f,
                    "could not read environment `{}`: {error}",
                    path.display()
                )
            }
            Self::Decode { path, error } => write!(
                f,
                "could not decode environment `{}`: {error}; \
                 expected a Radiance .hdr equirectangular image",
                path.display()
            ),
        }
    }
}

/// Resolve `relative` against `assets_dir` and decode it.
///
/// Radiance files carry RGBE, which decodes to unbounded linear float — the sun
/// in a real capture runs four or five orders of magnitude above the sky, and
/// that range is the point. Nothing here clamps it.
pub fn load_hdri(assets_dir: &Path, relative: &str) -> Result<Hdri, HdriError> {
    let relative = Path::new(relative);
    // Not `is_absolute`: on Windows that needs a drive *and* a root, yet `\sky`
    // and `C:sky` both make `join` discard the assets directory.
    if matches!(
        relative.components().next(),
        Some(Component::Prefix(_) | Component::RootDir)
    ) {
        return Err(HdriError::AbsolutePath(relative.to_path_buf()));
    }
    if relative
        .components()
        .any(|part| matches!(part, Component::ParentDir))
    {
        return Err(HdriError::ParentDirInPath(relative.to_path_buf()));
    }

    let path = assets_dir.join(relative);
    let bytes = std::fs::read(&path).map_err(|error| HdriError::Read {
        path: path.clone(),
        error,
    })?;

    let decoded = image::load_from_memory(&bytes)
        .map_err(|error| HdriError::Decode {
            path: path.clone(),
            error,
        })?
        .to_rgba32f();

    let (width, height) = (decoded.width(), decoded.height());
    Ok(Hdri {
        pixels: decoded.into_raw(),
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    /// The contract the control rests on: a source measuring anything at all,
    /// calibrated to N cd/m², comes out at N. Without this the number on the
    /// slider is decoration.
    #[test]
    fn a_calibrated_sky_is_the_luminance_it_claims() {
        let settings = EnvironmentSettings {
            sky_luminance: 2000.0,
            ..Default::default()
        };
        for measured in [0.01f32, 0.33, 1.0, 7.5] {
            let lit = measured * settings.calibration(measured);
            assert!(
                (lit / 2000.0 - 1.0).abs() < 1e-3,
                "a sky measuring {measured} calibrated to {lit}, not 2000",
            );
        }
    }

    /// A stop is a doubling here too, and it composes with the calibration rather
    /// than replacing it — the offset is stops *on top of* a physical answer.
    #[test]
    fn an_offset_stop_doubles_a_calibrated_sky() {
        let base = EnvironmentSettings {
            sky_luminance: 2000.0,
            ..Default::default()
        };
        let brighter = EnvironmentSettings {
            exposure_offset: 1.0,
            ..base.clone()
        };
        assert!((brighter.calibration(0.33) / base.calibration(0.33) - 2.0).abs() < 1e-4);
    }

    /// Nothing to measure is not a reason to divide by it. A black source, or no
    /// environment at all, has no ratio to take and must leave what it holds
    /// alone — this is the guard between an unloaded environment and an infinity
    /// in the lighting uniform.
    #[test]
    fn an_unmeasurable_source_is_left_alone() {
        let settings = EnvironmentSettings::default();
        assert_eq!(settings.calibration(0.0), 1.0);
        assert!(settings.calibration(f32::MIN_POSITIVE).is_finite());
    }

    use super::*;

    /// The same rule `orrin.toml` paths follow, and for the same reason: this
    /// field is destined for a scene file, and a scene file that names one
    /// machine's disk is a scene file nobody else can open.
    #[test]
    fn a_path_that_escapes_the_assets_directory_is_rejected() {
        let assets = Path::new("/project/assets");

        assert!(matches!(
            load_hdri(assets, "../../etc/passwd"),
            Err(HdriError::ParentDirInPath(_))
        ));
        assert!(matches!(
            load_hdri(assets, "sky/../../outside.hdr"),
            Err(HdriError::ParentDirInPath(_))
        ));
    }

    #[test]
    fn an_absolute_path_is_rejected() {
        let error = load_hdri(Path::new("/project/assets"), "/Users/someone/sky.hdr");
        assert!(matches!(error, Err(HdriError::AbsolutePath(_))));
    }

    /// On Windows `C:sky.hdr` is relative to drive C's current directory, not a
    /// file name, and joining it discards the assets directory entirely.
    #[cfg(windows)]
    #[test]
    fn a_drive_relative_path_is_rejected() {
        let error = load_hdri(Path::new(r"C:\project\assets"), "C:sky.hdr");
        assert!(matches!(error, Err(HdriError::AbsolutePath(_))));
    }

    /// Radiance shares one exponent across a pixel's three channels, so a
    /// value is only ever approximately recovered — 1% is the format, not the
    /// decoder.
    fn rgbe(value: [f32; 3]) -> [u8; 4] {
        let peak = value[0].max(value[1]).max(value[2]);
        if peak < 1e-32 {
            return [0, 0, 0, 0];
        }
        let exponent = peak.log2().floor() as i32 + 1;
        let scale = 256.0 / 2f32.powi(exponent);
        [
            (value[0] * scale) as u8,
            (value[1] * scale) as u8,
            (value[2] * scale) as u8,
            (exponent + 128) as u8,
        ]
    }

    /// A 2x1 Radiance image. Two pixels wide on purpose: the format only uses
    /// its run-length encoding from eight up, so this is unambiguously flat
    /// scanlines and the test is about decoding, not about RLE.
    fn write_hdr(path: &Path, pixels: [[f32; 3]; 2]) {
        let mut bytes = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n".to_vec();
        for pixel in pixels {
            bytes.extend_from_slice(&rgbe(pixel));
        }
        std::fs::write(path, bytes).expect("could not write the test image");
    }

    /// The reason this path exists at all. Every LDR format clips at 1.0, which
    /// would throw away the sun — the one part of an environment that drives a
    /// specular reflection and the one part the prefilter has to survive.
    #[test]
    fn decoding_preserves_radiance_far_above_one() {
        let dir = std::env::temp_dir().join(format!("orrin-hdri-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("could not create the test directory");
        write_hdr(
            &dir.join("sun.hdr"),
            [[4000.0, 4000.0, 4000.0], [0.2, 0.3, 0.5]],
        );

        let image = load_hdri(&dir, "sun.hdr").expect("a written file must load");
        assert_eq!((image.width, image.height), (2, 1));
        assert_eq!(image.pixels.len(), 2 * 4);

        let sun = image.pixels[0];
        assert!(
            (sun - 4000.0).abs() / 4000.0 < 0.02,
            "sun decoded to {sun}, expected about 4000 — anything near 1.0 means \
             the value was clipped rather than decoded",
        );

        let sky = image.pixels[4];
        assert!((sky - 0.2).abs() < 0.01, "sky decoded to {sky}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A missing file has to say which file, since the whole point of the
    /// relative path is that where it resolved to is not obvious.
    #[test]
    fn a_missing_file_names_the_path_it_looked_for() {
        let error = load_hdri(Path::new("/project/assets"), "sky/kloppenheim.hdr")
            .err()
            .expect("a nonexistent file cannot load");
        let message = error.to_string();
        assert!(
            message.contains("kloppenheim.hdr") && message.contains("assets"),
            "unhelpful message: {message}",
        );
    }
}
