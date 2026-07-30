use glam::Vec3;
use orrin_registry::{Reflect, Value, ValueError, take};

/// Dimensions are local-space and get scaled by the entity's transform when the
/// collision system computes world bounds.
///
/// Variant names are part of the on-disk format — see [`Light`](super::Light).
#[derive(Clone, Copy, Debug)]
pub enum ColliderShape {
    /// Narrowphase treats boxes as world-space AABBs, so a rotated entity gets a
    /// conservatively enlarged volume rather than a true OBB test.
    Box { half_extents: Vec3 },
    Sphere { radius: f32 },
}

/// `is_trigger` colliders fire the same enter/exit events as solid ones but are
/// exempt from overlap resolution — nothing is pushed out of a trigger, and a
/// trigger is never pushed.
#[derive(Clone, Copy, Debug)]
pub struct Collider {
    pub shape: ColliderShape,
    pub is_trigger: bool,
}

impl Default for Collider {
    fn default() -> Self {
        Self {
            shape: ColliderShape::Box {
                half_extents: Vec3::splat(0.5),
            },
            is_trigger: false,
        }
    }
}

impl Reflect for ColliderShape {
    fn to_value(&self) -> Value {
        match self {
            ColliderShape::Box { half_extents } => {
                Value::enumeration("Box", [("half_extents", half_extents.to_value())])
            }
            ColliderShape::Sphere { radius } => {
                Value::enumeration("Sphere", [("radius", radius.to_value())])
            }
        }
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        let Some(variant) = value.variant() else {
            return Err(ValueError::mismatch("enum", value));
        };
        match variant {
            "Box" => Ok(ColliderShape::Box {
                half_extents: take(value, "half_extents")?,
            }),
            "Sphere" => Ok(ColliderShape::Sphere {
                radius: take(value, "radius")?,
            }),
            other => Err(ValueError::unknown_variant("Box or Sphere", other)),
        }
    }
}

impl Reflect for Collider {
    fn to_value(&self) -> Value {
        Value::strukt([
            ("shape", self.shape.to_value()),
            ("is_trigger", self.is_trigger.to_value()),
        ])
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        Ok(Self {
            shape: take(value, "shape")?,
            is_trigger: take(value, "is_trigger")?,
        })
    }
}
