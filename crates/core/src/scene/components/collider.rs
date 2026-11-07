use glam::Vec3;
use orrin_registry::Reflect;

/// Dimensions are local-space and get scaled by the entity's transform when the
/// physics step builds the body.
///
/// Variant names are part of the on-disk format — see [`Light`](super::Light).
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub enum ColliderShape {
    Box {
        half_extents: Vec3,
    },
    Sphere {
        radius: f32,
    },
    /// Along the local Y axis: two hemispheres `half_height` either side of the
    /// centre, so the whole is `2 * (half_height + radius)` tall.
    Capsule {
        radius: f32,
        half_height: f32,
    },
    /// Along the local Y axis.
    Cylinder {
        radius: f32,
        half_height: f32,
    },
    /// The triangles of the entity's own mesh. Exact, and hollow: it has no
    /// inside to push anything out of, so it is for what does not move — a
    /// rigid body asking for one is given [`ConvexHull`](Self::ConvexHull)
    /// instead, and told so.
    Mesh,
    /// The tightest convex shape around the entity's own mesh: what a moving
    /// body built from a model should use.
    ConvexHull,
}

impl ColliderShape {
    /// Whether the shape is read from the entity's mesh rather than authored.
    pub fn uses_mesh(&self) -> bool {
        matches!(self, Self::Mesh | Self::ConvexHull)
    }
}

/// `is_trigger` colliders report entering and leaving like solid ones but push
/// nothing — nothing is pushed out of a trigger, and a trigger is never pushed.
///
/// The surface is the collider's rather than the rigid body's, as Unity has it,
/// because it belongs to what is touched: a static floor has a friction and no
/// rigid body. Where two meet, friction is their geometric mean and restitution
/// the larger, which is Jolt's rule. `layer` indexes the collision matrix in
/// [`PhysicsSettings`](super::PhysicsSettings).
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct Collider {
    pub shape: ColliderShape,
    pub is_trigger: bool,
    #[reflect(default = DEFAULT_FRICTION)]
    pub friction: f32,
    #[reflect(default)]
    pub restitution: f32,
    #[reflect(default)]
    pub layer: u32,
}

/// Between Jolt's 0.2, on which a stack of crates slides apart, and Unity's 0.6.
const DEFAULT_FRICTION: f32 = 0.5;

impl Default for Collider {
    fn default() -> Self {
        Self {
            shape: ColliderShape::Box {
                half_extents: Vec3::splat(0.5),
            },
            is_trigger: false,
            friction: DEFAULT_FRICTION,
            restitution: 0.0,
            layer: 0,
        }
    }
}
