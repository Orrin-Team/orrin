use glam::Vec3;

#[derive(Clone, Copy, Debug)]
pub enum Light {
    Directional {
        color: Vec3,
        intensity: f32,
    },
    Point {
        color: Vec3,
        intensity: f32,
        range: f32,
    },
}

impl Light {
    #[inline]
    pub fn directional(color: Vec3, intensity: f32) -> Self {
        Self::Directional { color, intensity }
    }

    #[inline]
    pub fn point(color: Vec3, intensity: f32, range: f32) -> Self {
        Self::Point {
            color,
            intensity,
            range,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AmbientLight {
    pub color: Vec3,
    pub intensity: f32,
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self {
            color: Vec3::new(0.6, 0.7, 1.0),
            intensity: 0.15,
        }
    }
}
