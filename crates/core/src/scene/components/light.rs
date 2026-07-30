use glam::Vec3;
use orrin_registry::Reflect;

/// Variant names are part of the on-disk format, exactly like a component's id:
/// renaming `Directional` orphans every saved light that used it, and nothing
/// catches that at compile time. Add variants freely; rename them never.
#[derive(Clone, Copy, Debug, Reflect)]
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

impl Default for Light {
    fn default() -> Self {
        Self::directional(Vec3::ONE, 1.0)
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
