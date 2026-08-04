//! Spherical-harmonic irradiance, bands 0 through 2.
//!
//! Nine coefficients reproduce essentially all of the diffuse irradiance from
//! any environment (Ramamoorthi & Hanrahan 2001): irradiance is radiance
//! convolved with a clamped cosine, and that kernel's spectrum falls off fast
//! enough that band 2 is already within about 1% of the exact answer. Which is
//! why the diffuse half of image-based lighting needs no image — nine `vec3`s
//! in the lighting uniform do what a 32x32 irradiance cubemap would, with no
//! descriptor, no sampler, and nothing to filter.
//!
//! Everything here is device-free and runs on the equirectangular source before
//! it is ever uploaded, so there is no GPU readback anywhere in the path.

use std::f32::consts::{PI, TAU};

use glam::Vec3;

/// Coefficient count for bands 0..=2.
pub const SH9: usize = 9;

/// Which band each coefficient belongs to.
const BAND: [usize; SH9] = [0, 1, 1, 1, 2, 2, 2, 2, 2];

/// The clamped-cosine convolution per band (`pi`, `2pi/3`, `pi/4`), divided by
/// pi. The division is folded in here because the diffuse term is
/// `albedo * E / pi`, so carrying the pi through to the shader only to divide it
/// out again would cost an operation per fragment and an explanation per reader.
const COSINE_LOBE: [f32; 3] = [1.0, 2.0 / 3.0, 0.25];

/// Per-band attenuation that suppresses ringing (Sloan, *Stupid Spherical
/// Harmonics Tricks*). A truncated series overshoots at sharp features — a sun
/// disc two orders of magnitude above the sky is exactly that — and the
/// overshoot goes *negative* on the far side, which reads as black patches on
/// surfaces facing away from the sun.
///
/// These are the Lanczos sigma factors `sinc(pi * l / 4)`. The window width
/// trades ringing against directionality: narrower is smoother and flatter,
/// wider keeps more of the environment's shape and more of the overshoot. Four
/// is gentle enough to keep a sky reading as a sky.
const WINDOW: [f32; 3] = [1.0, 0.900_316_3, 0.636_619_8];

/// The real spherical-harmonic basis for bands 0..=2.
///
/// Mirrored by `sh_irradiance` in `forward.frag`, which must evaluate these same
/// nine terms in this same order against the same components. The two are a
/// lock-step pair: a mismatch is not a compile error, it is lighting that is
/// subtly rotated or mirrored.
fn basis(d: Vec3) -> [f32; SH9] {
    [
        0.282095,
        0.488603 * d.y,
        0.488603 * d.z,
        0.488603 * d.x,
        1.092548 * d.x * d.y,
        1.092548 * d.y * d.z,
        0.315392 * (3.0 * d.z * d.z - 1.0),
        1.092548 * d.x * d.z,
        0.546274 * (d.x * d.x - d.y * d.y),
    ]
}

/// Fold the cosine convolution and the ringing window into raw projection
/// coefficients, so what comes out is what the shader dots against the basis.
fn convolve(mut coefficients: [Vec3; SH9]) -> [Vec3; SH9] {
    for (coefficient, band) in coefficients.iter_mut().zip(BAND) {
        *coefficient *= COSINE_LOBE[band] * WINDOW[band];
    }
    coefficients
}

/// Project an equirectangular radiance map onto the basis.
///
/// `pixels` is tightly packed RGBA f32 and must use the same mapping the bake
/// does — `u` around from -X, `v` from +Y down — since these coefficients light
/// the same scene the cubemap is drawn behind.
pub fn project_equirect(pixels: &[f32], extent: [u32; 2]) -> [Vec3; SH9] {
    let (width, height) = (extent[0] as usize, extent[1] as usize);
    let mut coefficients = [Vec3::ZERO; SH9];

    // A texel's solid angle varies only with latitude: sin(theta) d(theta)
    // d(phi). Without it the poles — where texels are narrowest — would count
    // as much as the equator, and every environment would be lit from above.
    let d_theta = PI / height as f32;
    let d_phi = TAU / width as f32;

    // Midpoint quadrature over sin(theta) does not sum to exactly 4*pi, and the
    // shortfall grows as the source gets shorter — about 0.04% at 32 rows. Left
    // alone that is a systematic darkening that depends on the resolution of
    // the file someone happened to supply, so the total is measured and divided
    // back out below rather than assumed.
    let mut measure = 0.0;

    for y in 0..height {
        let theta = (y as f32 + 0.5) * d_theta;
        let (sin_theta, cos_theta) = theta.sin_cos();
        let solid_angle = sin_theta * d_theta * d_phi;
        measure += solid_angle * width as f32;

        for x in 0..width {
            let phi = ((x as f32 + 0.5) / width as f32 - 0.5) * TAU;
            let (sin_phi, cos_phi) = phi.sin_cos();
            let direction = Vec3::new(cos_phi * sin_theta, cos_theta, sin_phi * sin_theta);

            let texel = (y * width + x) * 4;
            let radiance = Vec3::new(pixels[texel], pixels[texel + 1], pixels[texel + 2]);

            for (coefficient, harmonic) in coefficients.iter_mut().zip(basis(direction)) {
                *coefficient += radiance * (harmonic * solid_angle);
            }
        }
    }

    let correction = 2.0 * TAU / measure;
    for coefficient in &mut coefficients {
        *coefficient *= correction;
    }

    convolve(coefficients)
}

/// The coefficients for radiance that is the same in every direction.
///
/// This is what makes a scene with no environment take the same path as one
/// with an environment rather than a second branch in the shader: flat ambient
/// is just an environment with nothing above band 0. Evaluating these returns
/// `radiance` for every normal, which is exactly what the flat ambient term it
/// replaces computed.
pub fn from_constant(radiance: Vec3) -> [Vec3; SH9] {
    let mut coefficients = [Vec3::ZERO; SH9];
    // Integrating a constant against y00 over the sphere: 4*pi * 0.282095.
    coefficients[0] = radiance * (2.0 * PI.sqrt());
    convolve(coefficients)
}

/// Irradiance divided by pi in `direction` — what the shader computes, here so
/// tests can assert against it without a GPU.
#[cfg(test)]
fn evaluate(coefficients: &[Vec3; SH9], direction: Vec3) -> Vec3 {
    let basis = basis(direction.normalize());
    let mut total = Vec3::ZERO;
    for (coefficient, harmonic) in coefficients.iter().zip(basis) {
        total += *coefficient * harmonic;
    }
    total.max(Vec3::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NORMALS: [Vec3; 6] = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
    ];

    fn equirect(width: u32, height: u32, mut radiance: impl FnMut(Vec3) -> Vec3) -> Vec<f32> {
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            let theta = (y as f32 + 0.5) / height as f32 * PI;
            let (sin_theta, cos_theta) = theta.sin_cos();
            for x in 0..width {
                let phi = ((x as f32 + 0.5) / width as f32 - 0.5) * TAU;
                let (sin_phi, cos_phi) = phi.sin_cos();
                let value = radiance(Vec3::new(
                    cos_phi * sin_theta,
                    cos_theta,
                    sin_phi * sin_theta,
                ));
                pixels.extend_from_slice(&[value.x, value.y, value.z, 1.0]);
            }
        }
        pixels
    }

    /// The whole normalization in one assertion: solid-angle weighting summing
    /// to 4*pi, the basis constants, and the cosine convolution. A uniform
    /// environment of radiance L lights every surface to exactly L no matter
    /// which way it faces, and any error in any of those three shows up here as
    /// a scale factor.
    #[test]
    fn a_uniform_environment_lights_every_normal_to_its_radiance() {
        let pixels = equirect(64, 32, |_| Vec3::new(0.25, 0.5, 0.75));
        let sh = project_equirect(&pixels, [64, 32]);

        for normal in NORMALS {
            let lit = evaluate(&sh, normal);
            assert!(
                (lit - Vec3::new(0.25, 0.5, 0.75)).abs().max_element() < 1e-3,
                "normal {normal} lit to {lit}, expected the source radiance",
            );
        }
    }

    /// The fallback has to be indistinguishable from projecting a uniform
    /// source, or a scene without an environment would change brightness the
    /// moment a featureless one was loaded.
    #[test]
    fn the_constant_fallback_matches_a_projected_uniform_environment() {
        let radiance = Vec3::new(0.4, 0.6, 0.9);
        let projected = project_equirect(&equirect(64, 32, |_| radiance), [64, 32]);
        let constant = from_constant(radiance);

        for (a, b) in projected.iter().zip(constant.iter()) {
            assert!((*a - *b).abs().max_element() < 1e-3, "{a} vs {b}");
        }
    }

    /// Directionality survives the truncation and the window: a lit upper
    /// hemisphere over a dark lower one has to leave up brighter than down, and
    /// the horizon between them.
    #[test]
    fn a_lit_hemisphere_is_brightest_facing_into_it() {
        let pixels = equirect(64, 32, |d| if d.y > 0.0 { Vec3::ONE } else { Vec3::ZERO });
        let sh = project_equirect(&pixels, [64, 32]);

        let up = evaluate(&sh, Vec3::Y).x;
        let horizon = evaluate(&sh, Vec3::X).x;
        let down = evaluate(&sh, -Vec3::Y).x;

        assert!(
            up > horizon && horizon > down,
            "expected up > horizon > down, got {up} / {horizon} / {down}",
        );
        // Half the sphere at radiance 1 delivers half the irradiance a full one
        // would, so the horizon sits near the midpoint whatever the window does
        // to the bands above it.
        assert!(
            (horizon - 0.5).abs() < 0.1,
            "horizon lit to {horizon}, expected about half",
        );
    }

    /// Ringing is what the window exists to contain, and the shader's clamp is
    /// the backstop. A sun disc far brighter than its surroundings is the case
    /// that produces it.
    #[test]
    fn a_bright_sun_does_not_drive_any_normal_negative() {
        let to_sun = Vec3::new(0.4, 1.0, 0.6).normalize();
        let pixels = equirect(128, 64, |d| {
            if d.dot(to_sun) > 0.999 {
                Vec3::splat(200.0)
            } else {
                Vec3::splat(0.05)
            }
        });
        let sh = project_equirect(&pixels, [128, 64]);

        for normal in NORMALS {
            let lit = evaluate(&sh, normal);
            assert!(lit.min_element() >= 0.0, "normal {normal} lit to {lit}");
        }
    }
}
