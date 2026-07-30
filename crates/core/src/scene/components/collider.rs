use glam::Vec3;
use orrin_registry::Reflect;

/// Dimensions are local-space and get scaled by the entity's transform when the
/// collision system computes world bounds.
///
/// Variant names are part of the on-disk format — see [`Light`](super::Light).
#[derive(Clone, Copy, Debug, Reflect)]
pub enum ColliderShape {
    /// Narrowphase treats boxes as world-space AABBs, so a rotated entity gets a
    /// conservatively enlarged volume rather than a true OBB test.
    Box { half_extents: Vec3 },
    Sphere { radius: f32 },
}

/// `is_trigger` colliders fire the same enter/exit events as solid ones but are
/// exempt from overlap resolution — nothing is pushed out of a trigger, and a
/// trigger is never pushed.
#[derive(Clone, Copy, Debug, Reflect)]
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

