use glam::{Mat4, Quat, Vec3};
use orrin_registry::{Reflect, Value, ValueError, take};

#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    #[inline]
    pub fn from_translation(translation: Vec3) -> Self {
        Self { translation, ..Default::default() }
    }

    #[inline]
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

impl Reflect for Transform {
    fn to_value(&self) -> Value {
        Value::strukt([
            ("translation", self.translation.to_value()),
            ("rotation", self.rotation.to_value()),
            ("scale", self.scale.to_value()),
        ])
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        Ok(Self {
            translation: take(value, "translation")?,
            rotation: take(value, "rotation")?,
            scale: take(value, "scale")?,
        })
    }
}
