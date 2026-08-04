use std::fmt;
use std::path::{Component, Path, PathBuf};

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
    /// Multiplier on environment radiance. Separate from
    /// [`HdrSettings::exposure`](super::HdrSettings) because that scales the
    /// whole frame, while this balances the environment against the analytic
    /// lights.
    pub intensity: f32,
    /// Rotation of the environment about world Y, in degrees. Applied to the
    /// sampling direction, so it costs nothing and needs no rebake.
    pub yaw: f32,
    /// Whether the environment is drawn as the background. Turning it off
    /// leaves the forward pass's clear colour showing.
    pub show_skybox: bool,
}

impl Default for EnvironmentSettings {
    fn default() -> Self {
        Self {
            hdri: String::new(),
            reload_requested: false,
            intensity: 1.0,
            yaw: 0.0,
            show_skybox: true,
        }
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
    if relative.is_absolute() {
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
