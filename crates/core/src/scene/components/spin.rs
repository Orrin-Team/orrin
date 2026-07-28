use glam::{Quat, Vec3};

use crate::scene::LocalTransform;

/// Spins an entity about `axis` at `speed` radians per second.
#[derive(Clone, Copy, Debug)]
pub struct Spin {
    axis: Vec3,
    speed: f32,
}

impl Spin {
    #[inline]
    pub fn new(axis: Vec3, speed: f32) -> Self {
        Self {
            axis: axis.normalize(),
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
