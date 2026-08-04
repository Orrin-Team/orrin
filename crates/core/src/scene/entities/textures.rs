use glam::Vec3;

pub fn load_rgba(bytes: &[u8]) -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(bytes)
        .expect("failed to decode texture")
        .to_rgba8();
    let (width, height) = img.dimensions();
    (img.into_raw(), width, height)
}

pub fn checkerboard(size: u32, checks: u32, a: [u8; 3], b: [u8; 3]) -> Vec<u8> {
    let cell = (size / checks).max(1);
    let mut data = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let c = if ((x / cell) + (y / cell)).is_multiple_of(2) {
                a
            } else {
                b
            };
            data.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    data
}

pub fn bump_normals(size: u32, freq: f32, strength: f32) -> Vec<u8> {
    let mut data = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32 * std::f32::consts::TAU * freq;
            let v = y as f32 / size as f32 * std::f32::consts::TAU * freq;
            // Slope is the gradient of the height field h = sin(u) * sin(v).
            let dx = strength * u.cos() * v.sin();
            let dy = strength * u.sin() * v.cos();
            let n = Vec3::new(-dx, -dy, 1.0).normalize();
            let e = (n * 0.5 + 0.5) * 255.0;
            data.extend_from_slice(&[e.x as u8, e.y as u8, e.z as u8, 255]);
        }
    }
    data
}

// glTF convention: G = roughness, B = metallic.
/// A stand-in equirectangular sky: a zenith-to-horizon gradient over dull
/// ground, with a sun disc along `to_sun`. RGBA f32, linear, unbounded — the
/// disc is two orders of magnitude above the sky, which is the dynamic range a
/// real HDRI has and the thing a prefilter has to survive.
///
/// Generated rather than shipped so the environment path has content without a
/// multi-megabyte file in the repo. The bake consumes an equirect whatever
/// produced it, so a decoded `.hdr` is a different source for these pixels, not
/// a different path.
pub fn sky_equirect(width: u32, height: u32, to_sun: Vec3) -> Vec<f32> {
    use std::f32::consts::{PI, TAU};

    let to_sun = to_sun.normalize();
    let zenith = Vec3::new(0.10, 0.20, 0.42);
    let horizon = Vec3::new(0.55, 0.62, 0.72);
    let ground = Vec3::new(0.06, 0.055, 0.05);

    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        // The inverse of the mapping in equirect_to_cube.frag: v = acos(y)/PI
        // and u = atan2(z, x)/TAU + 0.5. Sampling at texel centres keeps the
        // poles off the exact singularity.
        let theta = (y as f32 + 0.5) / height as f32 * PI;
        let (sin_theta, cos_theta) = theta.sin_cos();
        for x in 0..width {
            let phi = ((x as f32 + 0.5) / width as f32 - 0.5) * TAU;
            let (sin_phi, cos_phi) = phi.sin_cos();
            let dir = Vec3::new(cos_phi * sin_theta, cos_theta, sin_phi * sin_theta);

            let mut color = if dir.y >= 0.0 {
                horizon.lerp(zenith, dir.y.powf(0.45))
            } else {
                ground.lerp(horizon, (1.0 + dir.y * 6.0).clamp(0.0, 1.0) * 0.35)
            };

            // ~1.5 degrees across, with a wide soft halo so the disc does not
            // alias into a ring of fireflies at the cube's resolution.
            let cosine = dir.dot(to_sun);
            color += Vec3::splat(60.0) * ((cosine - 0.9996) / 0.0004).clamp(0.0, 1.0);
            color += Vec3::new(1.0, 0.85, 0.6) * cosine.max(0.0).powf(64.0) * 2.0;

            data.extend_from_slice(&[color.x, color.y, color.z, 1.0]);
        }
    }
    data
}

pub fn metallic_roughness(size: u32) -> Vec<u8> {
    let band = (size / 8).max(1);
    let mut data = Vec::with_capacity((size * size * 4) as usize);
    for _ in 0..size {
        for x in 0..size {
            let roughness = (x * 255 / size.max(1)) as u8;
            let metallic = if (x / band).is_multiple_of(2) { 255 } else { 0 };
            data.extend_from_slice(&[0, roughness, metallic, 255]);
        }
    }
    data
}
