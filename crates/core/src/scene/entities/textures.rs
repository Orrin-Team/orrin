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
            let c = if ((x / cell) + (y / cell)).is_multiple_of(2) { a } else { b };
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
