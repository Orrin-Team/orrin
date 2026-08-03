use std::ops::{Deref, DerefMut};

use glam::Mat4;

/// An entity's transform in world space, derived from its
/// [`LocalTransform`](super::LocalTransform) by
/// [`propagate_transforms`](crate::systems::propagate_transforms).
///
/// Derived, never authored: a value written here survives until the next
/// propagation and no longer. It is deliberately absent from the component
/// registry for that reason — see `scene::registry`.
///
/// A matrix rather than a [`Transform`](crate::scene::Transform) because
/// composing a non-uniformly scaled parent with a rotated child produces shear,
/// and no translation/rotation/scale triple can represent that. Decomposing one
/// back into a `Transform` would silently lose it, which is the same admission
/// Unity's `lossyScale` makes.
#[derive(Clone, Copy, Debug)]
pub struct WorldTransform(pub Mat4);

impl Default for WorldTransform {
    fn default() -> Self {
        Self(Mat4::IDENTITY)
    }
}

impl WorldTransform {
    #[inline]
    pub fn new(matrix: Mat4) -> Self {
        Self(matrix)
    }

    /// The entity's world-space position — the matrix's translation column.
    ///
    /// Exact under any hierarchy, unlike a world scale or rotation, which only
    /// exist as a lossy fit once shear is in play.
    #[inline]
    pub fn translation(&self) -> glam::Vec3 {
        self.0.w_axis.truncate()
    }
}

impl Deref for WorldTransform {
    type Target = Mat4;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for WorldTransform {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
