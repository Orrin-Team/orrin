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
//!
//! Every scan over the source splits by row across the pool. A row is
//! independent given its latitude, and its partial sum comes back to be added
//! in row order on the calling thread — so the result is bit-identical whatever
//! the pool size. Not a rayon `fold`/`reduce`, whose association depends on how
//! the work happened to be stolen and would make the lighting differ, in the
//! last bits, between two runs of the same scene.

use std::f32::consts::{FRAC_2_PI, PI, TAU};

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
/// `sinc(pi/2)` is `2/pi` exactly, so the last factor is the named constant
/// rather than a rounded literal — it is the closer of the two in `f32`.
const WINDOW: [f32; 3] = [1.0, 0.900_316_3, FRAC_2_PI];

/// Radiance above this multiple of the environment's mean is left out of the
/// *diffuse* projection.
///
/// An outdoor capture is mostly sun: measured on a 2K meadow HDRI, the sun disc
/// is 0.02% of the pixels and 81% of the irradiance. A scene lit by that
/// environment almost always also has a directional light standing in for the
/// same sun — that is what casts the shadows, since an environment cannot — so
/// leaving the disc in the diffuse series delivers the sun twice, and the
/// second copy is *unshadowed*. It lights the insides of shadows the cascade
/// maps correctly darkened, which does not read as "too bright" so much as
/// "shadows do not work".
///
/// So the analytic light owns the sun's diffuse and the environment owns the
/// sky's. The specular chain is deliberately not clamped: nothing else puts a
/// sun in a metal's reflection.
///
/// The exact value barely matters, which is what makes it defensible rather
/// than magic — the energy sits in the extreme tail, so on that same file any
/// ceiling from 70x to 1400x the mean lands within 15% of the same answer.
const SUN_CLAMP: f32 = 100.0;

const LUMA: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// `row(y)` for every row in `rows`, across the pool, in order.
fn per_row<T: Send>(
    rows: std::ops::Range<usize>,
    row: impl Fn(usize) -> T + Send + Sync,
) -> Vec<T> {
    let rows: Vec<usize> = rows.collect();
    crate::threads::map(&rows, |&y| row(y))
}

/// The luminance of every texel in row `y`.
fn row_luminance(pixels: &[f32], width: usize, y: usize) -> impl Iterator<Item = f32> + '_ {
    pixels[y * width * 4..(y + 1) * width * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&[r, g, b, _]| Vec3::new(r, g, b).dot(LUMA))
}

/// Solid-angle-weighted mean luminance, counting nothing above `ceiling`.
fn mean_luminance(pixels: &[f32], extent: [u32; 2], ceiling: f32) -> f32 {
    let (width, height) = (extent[0] as usize, extent[1] as usize);

    let rows = per_row(0..height, |y| {
        let theta = (y as f32 + 0.5) / height as f32 * PI;
        let weight = theta.sin() as f64;
        let sum: f64 = row_luminance(pixels, width, y)
            .map(|luminance| luminance.min(ceiling) as f64)
            .sum();
        (sum * weight, weight * width as f64)
    });
    let (mut total, mut measure) = (0.0f64, 0.0f64);
    for (row_total, row_measure) in rows {
        total += row_total;
        measure += row_measure;
    }

    if measure > 0.0 {
        (total / measure) as f32
    } else {
        0.0
    }
}

/// The luminance the diffuse projection clips at: [`SUN_CLAMP`] times the mean,
/// which is the scale the threshold is relative to so it means the same thing
/// whatever units a capture was authored in.
///
/// Iterated twice because the first mean is itself sun-dominated — on an
/// outdoor capture the disc supplies most of the average it is being compared
/// against, which drags the threshold up toward the very value it exists to
/// exclude. The second pass measures a mean the sun has already been taken out
/// of. It converges immediately after that and a third pass moves nothing.
fn diffuse_ceiling(pixels: &[f32], extent: [u32; 2]) -> f32 {
    let mut ceiling = f32::INFINITY;
    for _ in 0..2 {
        ceiling = mean_luminance(pixels, extent, ceiling) * SUN_CLAMP;
    }
    ceiling
}

/// The luminance of this environment's sky, in whatever units the source
/// carries — the measurement a physical calibration is stated relative to.
///
/// Two choices in here, and both are about what "the sky" means to someone
/// reading a number off a reference table:
///
/// - The **upper hemisphere only**. A capture's lower half is ground, and ground
///   is dark; including it would answer a question about the *environment* while
///   the control above it asks about the sky, and the two differ by a factor that
///   depends on how much dirt the photographer stood on.
/// - A **geometric mean**, which is what makes this robust to a sun. A clear-sky
///   capture puts most of its arithmetic mean into a disc a fraction of a degree
///   across, so an ordinary average would report the sun and the calibration
///   derived from it would light the scene orders of magnitude too dark. In log
///   space that disc is a thousandth of the frame carrying a bounded number of
///   stops, and it moves the answer by well under a percent.
///
/// Deliberately *not* the clamped mean [`diffuse_ceiling`] builds, even though it
/// sits right there and also excludes the sun. That one exists to preserve
/// energy: irradiance is a linear integral, so an outlier has to be clamped and
/// still counted. This one summarises a level, where the outlier should not count
/// at all. Two different questions, and answering the second with the first
/// leaves about a quarter of a bright sun in the result.
pub fn sky_luminance(pixels: &[f32], extent: [u32; 2]) -> f32 {
    let (width, height) = (extent[0] as usize, extent[1] as usize);
    // Row 0 is the zenith — `basis`'s `cos_theta` is +Y at theta 0 — so the sky
    // is the first half of the rows.
    let rows = per_row(0..height / 2, |y| {
        let theta = (y as f32 + 0.5) / height as f32 * PI;
        let weight = theta.sin() as f64;
        // Floored rather than skipped: a black pixel is information about how
        // dark this sky is, and dropping it would report a night sky as
        // whatever few stars it has.
        let sum: f64 = row_luminance(pixels, width, y)
            .map(|luminance| (luminance.max(1e-6) as f64).ln())
            .sum();
        (sum * weight, weight * width as f64)
    });
    let (mut total, mut measure) = (0.0f64, 0.0f64);
    for (row_total, row_measure) in rows {
        total += row_total;
        measure += row_measure;
    }

    if measure > 0.0 {
        (total / measure).exp() as f32
    } else {
        0.0
    }
}

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
    let ceiling = diffuse_ceiling(pixels, extent);

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

    // Longitude depends only on the column, so its sine and cosine are one row's
    // worth of transcendentals rather than one per texel.
    let longitudes: Vec<(f32, f32)> = (0..width)
        .map(|x| (((x as f32 + 0.5) / width as f32 - 0.5) * TAU).sin_cos())
        .collect();

    // Each row sums unweighted, and takes its solid angle once at the end: the
    // weight is constant along a row.
    let rows = per_row(0..height, |y| {
        let theta = (y as f32 + 0.5) * d_theta;
        let (sin_theta, cos_theta) = theta.sin_cos();
        let (row, _) = pixels[y * width * 4..(y + 1) * width * 4].as_chunks::<4>();

        let mut sums = [Vec3::ZERO; SH9];
        for (&[r, g, b, _], &(sin_phi, cos_phi)) in row.iter().zip(&longitudes) {
            let direction = Vec3::new(cos_phi * sin_theta, cos_theta, sin_phi * sin_theta);
            let mut radiance = Vec3::new(r, g, b);

            // Scaled rather than clamped per channel, so a clipped sun keeps
            // its colour instead of drifting toward whichever channel was
            // furthest over.
            let luminance = radiance.dot(LUMA);
            if ceiling > 0.0 && luminance > ceiling {
                radiance *= ceiling / luminance;
            }

            for (sum, harmonic) in sums.iter_mut().zip(basis(direction)) {
                *sum += radiance * harmonic;
            }
        }
        (sums, sin_theta * d_theta * d_phi)
    });

    let mut coefficients = [Vec3::ZERO; SH9];
    for (sums, solid_angle) in rows {
        for (coefficient, sum) in coefficients.iter_mut().zip(sums) {
            *coefficient += sum * solid_angle;
        }
        measure += solid_angle * width as f32;
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

    /// The measurement the whole calibration is stated against: a uniform sky of
    /// luminance L has to measure L, or "this sky is 2 000 cd/m²" scales to
    /// something else entirely.
    #[test]
    fn a_uniform_sky_measures_its_own_luminance() {
        let pixels = equirect(64, 32, |_| Vec3::splat(0.5));
        let expected = Vec3::splat(0.5).dot(LUMA);
        let measured = sky_luminance(&pixels, [64, 32]);
        assert!(
            (measured / expected - 1.0).abs() < 0.02,
            "measured {measured}, expected {expected}",
        );
    }

    /// A sun disc is a fraction of a degree across and four or five orders of
    /// magnitude above the sky around it, so a mean that counted it would answer
    /// with the sun. The calibration derived from that answer would light the
    /// scene orders of magnitude too dark, which is the failure this measurement
    /// exists to avoid — and it is invisible, because the sky still looks like a
    /// sky, just wrong.
    #[test]
    fn a_sun_does_not_drag_the_measurement_up() {
        let to_sun = Vec3::new(0.3, 0.8, 0.5).normalize();
        let clear = equirect(256, 128, |_| Vec3::splat(0.4));
        let sunny = equirect(256, 128, |d| {
            if d.dot(to_sun) > 0.999 {
                Vec3::splat(40_000.0)
            } else {
                Vec3::splat(0.4)
            }
        });
        let without = sky_luminance(&clear, [256, 128]);
        let with = sky_luminance(&sunny, [256, 128]);
        assert!(
            (with / without - 1.0).abs() < 0.05,
            "a sun moved the sky from {without} to {with}",
        );
    }

    /// The upper hemisphere only. A capture's ground is dark and its share of the
    /// frame depends on where the photographer stood, so including it would make
    /// the number answer a question about the environment while the control asks
    /// about the sky.
    #[test]
    fn the_ground_is_not_the_sky() {
        let pixels = equirect(64, 32, |d| {
            if d.y >= 0.0 {
                Vec3::splat(1.0)
            } else {
                Vec3::splat(0.01)
            }
        });
        let measured = sky_luminance(&pixels, [64, 32]);
        let sky = Vec3::splat(1.0).dot(LUMA);
        assert!(
            (measured / sky - 1.0).abs() < 0.02,
            "measured {measured}, expected the sky's {sky}",
        );
    }

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

    /// The sun clamp's whole job. An environment whose sun carries most of its
    /// energy has to project to roughly what the *sky alone* would, because the
    /// scene's directional light is about to deliver that sun again — and
    /// unlike this series, that copy is shadowed.
    ///
    /// A large disc on purpose: a pinprick sun is clamped by any threshold at
    /// all, while one big enough to dominate the mean is what defeats a
    /// single-pass estimate of it.
    #[test]
    fn a_sun_does_not_dominate_the_diffuse_irradiance() {
        let to_sun = Vec3::new(0.3, 1.0, 0.2).normalize();
        let sky = 0.05;
        let lit = equirect(256, 128, |d| {
            if d.dot(to_sun) > 0.9995 {
                Vec3::splat(50_000.0)
            } else {
                Vec3::splat(sky)
            }
        });
        let sunless = equirect(256, 128, |_| Vec3::splat(sky));

        let with_sun = evaluate(&project_equirect(&lit, [256, 128]), Vec3::Y).x;
        let without = evaluate(&project_equirect(&sunless, [256, 128]), Vec3::Y).x;

        assert!(
            (with_sun - without).abs() < sky,
            "a 50000:0.05 sun left {with_sun} against a sunless {without}; \
             the disc is still dominating the irradiance",
        );
    }

    /// A source with every feature the projection treats specially: a gradient
    /// for the solid-angle weighting, a sun over the clamp, and a dark ground.
    fn awkward_sky() -> Vec<f32> {
        let to_sun = Vec3::new(0.3, 0.9, -0.4).normalize();
        equirect(256, 128, |d| {
            if d.dot(to_sun) > 0.999 {
                Vec3::new(60_000.0, 55_000.0, 50_000.0)
            } else if d.y < 0.0 {
                Vec3::new(0.02, 0.015, 0.01)
            } else {
                Vec3::new(0.2, 0.3, 0.6) * (0.5 + d.y) + Vec3::splat(d.x * 0.1)
            }
        })
    }

    /// The bake sums per-row partials in row order whichever path runs, so the
    /// pool changes how fast it finishes and nothing else. Bit equality is the
    /// claim, not a tolerance: an environment that lit a scene differently
    /// under `ORRIN_THREADS=1` would make every A/B against it meaningless.
    #[test]
    fn the_bake_does_not_depend_on_the_thread_count() {
        let pixels = awkward_sky();
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("a pool")
                .install(|| {
                    (
                        project_equirect(&pixels, [256, 128]),
                        sky_luminance(&pixels, [256, 128]),
                    )
                })
        };
        assert_eq!(run(1), run(4));
    }

    /// The projection as one running sum over every texel, with `phi` evaluated
    /// per texel — the definition the row-parallel version has to agree with.
    fn reference_projection(pixels: &[f32], extent: [u32; 2]) -> [Vec3; SH9] {
        let (width, height) = (extent[0] as usize, extent[1] as usize);
        let ceiling = diffuse_ceiling(pixels, extent);
        let (d_theta, d_phi) = (PI / height as f32, TAU / width as f32);
        let mut coefficients = [Vec3::ZERO; SH9];
        let mut measure = 0.0;
        for y in 0..height {
            let (sin_theta, cos_theta) = ((y as f32 + 0.5) * d_theta).sin_cos();
            let solid_angle = sin_theta * d_theta * d_phi;
            measure += solid_angle * width as f32;
            for x in 0..width {
                let phi = ((x as f32 + 0.5) / width as f32 - 0.5) * TAU;
                let (sin_phi, cos_phi) = phi.sin_cos();
                let direction = Vec3::new(cos_phi * sin_theta, cos_theta, sin_phi * sin_theta);
                let texel = (y * width + x) * 4;
                let mut radiance = Vec3::new(pixels[texel], pixels[texel + 1], pixels[texel + 2]);
                let luminance = radiance.dot(LUMA);
                if ceiling > 0.0 && luminance > ceiling {
                    radiance *= ceiling / luminance;
                }
                for (c, h) in coefficients.iter_mut().zip(basis(direction)) {
                    *c += radiance * (h * solid_angle);
                }
            }
        }
        convolve(coefficients.map(|c| c * (2.0 * TAU / measure)))
    }

    /// Summing per row then across rows reorders the floating-point additions,
    /// so agreement is to a tolerance: one part in 10^4 of the largest
    /// coefficient, far below anything a lit surface could show.
    #[test]
    fn the_bake_matches_a_texel_by_texel_reference() {
        let pixels = awkward_sky();
        let baked = project_equirect(&pixels, [256, 128]);
        let reference = reference_projection(&pixels, [256, 128]);
        let scale = reference
            .iter()
            .map(|c| c.abs().max_element())
            .fold(0.0, f32::max);
        for (index, (a, b)) in baked.iter().zip(reference.iter()).enumerate() {
            assert!(
                (*a - *b).abs().max_element() <= scale * 1e-4,
                "coefficient {index}: baked {a}, reference {b}",
            );
        }
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
