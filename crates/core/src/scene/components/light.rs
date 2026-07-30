use glam::Vec3;
use orrin_registry::{Reflect, Value, ValueError, take};

/// Variant names are part of the on-disk format, exactly like a component's id:
/// renaming `Directional` orphans every saved light that used it, and nothing
/// catches that at compile time. Add variants freely; rename them never.
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

impl Default for Light {
    fn default() -> Self {
        Self::directional(Vec3::ONE, 1.0)
    }
}

impl Reflect for Light {
    fn to_value(&self) -> Value {
        match self {
            Light::Directional { color, intensity } => Value::enumeration(
                "Directional",
                [
                    ("color", color.to_value()),
                    ("intensity", intensity.to_value()),
                ],
            ),
            Light::Point {
                color,
                intensity,
                range,
            } => Value::enumeration(
                "Point",
                [
                    ("color", color.to_value()),
                    ("intensity", intensity.to_value()),
                    ("range", range.to_value()),
                ],
            ),
        }
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        let Some(variant) = value.variant() else {
            return Err(ValueError::mismatch("enum", value));
        };
        match variant {
            "Directional" => Ok(Light::Directional {
                color: take(value, "color")?,
                intensity: take(value, "intensity")?,
            }),
            "Point" => Ok(Light::Point {
                color: take(value, "color")?,
                intensity: take(value, "intensity")?,
                range: take(value, "range")?,
            }),
            other => Err(ValueError::unknown_variant("Directional or Point", other)),
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
