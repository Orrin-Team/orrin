use glam::{Quat, Vec3};
use orrin_registry::{Reflect, Value, ValueError, take};

use crate::scene::LocalTransform;

/// Spins an entity about `axis` at `speed` radians per second.
#[derive(Clone, Copy, Debug)]
pub struct Spin {
    axis: Vec3,
    speed: f32,
}

impl Spin {
    /// Normalization is skipped for an axis that is already unit-length, which
    /// makes this idempotent. That matters beyond saving a division:
    /// re-normalizing a normalized `Vec3` is not a no-op in `f32`
    /// (`0.70710677` becomes `0.7071068`), so a `Spin` that went through a
    /// save/load cycle would serialize differently the second time and every
    /// reload would show up as a scene diff.
    #[inline]
    pub fn new(axis: Vec3, speed: f32) -> Self {
        Self {
            axis: if axis.is_normalized() {
                axis
            } else {
                axis.normalize()
            },
            speed,
        }
    }

    /// Re-normalizes the result so the rotation doesn't drift as error
    /// accumulates over a long-running spin.
    #[inline]
    pub fn apply(&self, transform: &mut LocalTransform, dt: f32) {
        let delta = Quat::from_axis_angle(self.axis, self.speed * dt);
        transform.rotation = (delta * transform.rotation).normalize();
    }
}

impl Default for Spin {
    fn default() -> Self {
        Self::new(Vec3::Y, 0.0)
    }
}

/// Reconstructed through [`Spin::new`], never by assigning the fields.
///
/// `axis` carries an invariant — `apply` feeds it to `Quat::from_axis_angle`,
/// which is only a rotation for a unit vector — and a scene file is untrusted
/// input. Writing the field directly would let a hand-edited or stale scene
/// install a `Spin` that silently scales the rotation it produces.
///
/// This is why a derive macro cannot be applied blindly: any type whose
/// constructor establishes something its fields don't must opt out.
impl Reflect for Spin {
    fn to_value(&self) -> Value {
        Value::strukt([
            ("axis", self.axis.to_value()),
            ("speed", self.speed.to_value()),
        ])
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        let axis: Vec3 = take(value, "axis")?;
        // `normalize` on a zero or non-finite vector yields NaN, and a NaN axis
        // turns every later rotation into NaN — a corruption that spreads
        // through the transform hierarchy and is unrecoverable by the time it's
        // visible. Rejected here, at the boundary where untrusted input enters.
        if axis.try_normalize().is_none() {
            return Err(ValueError::invalid("a non-zero axis", format!("{axis:?}")).at_field("axis"));
        }
        Ok(Self::new(axis, take(value, "speed")?))
    }
}
