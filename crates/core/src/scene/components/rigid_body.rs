use orrin_registry::Reflect;

/// Makes an entity's [`Collider`](super::Collider) a simulated body: the
/// physics step owns its pose and writes it back into the transform. Without
/// one a collider is moved only by whatever moves its transform, and nothing
/// pushes it.
///
/// SI throughout, as every other length in the engine: kilograms, and a
/// `gravity_scale` of 1 is [`PhysicsSettings::gravity`](super::PhysicsSettings).
/// The surface it slides and bounces on is its collider's.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct RigidBody {
    /// Moved by its transform like a plain collider, but still pushes what it
    /// moves into — a lift, a door.
    pub kinematic: bool,
    pub mass: f32,
    pub gravity_scale: f32,
    /// The fraction of velocity lost per second, Jolt's model of drag.
    #[reflect(default = 0.05)]
    pub linear_damping: f32,
    #[reflect(default = 0.05)]
    pub angular_damping: f32,
    /// Sweeps the body along its motion each step, so a fast one cannot pass
    /// through something thinner than a step's travel. Costs a cast per step,
    /// which is why it is per body.
    #[reflect(default)]
    pub continuous: bool,
    /// Never turns: a character's capsule, a crate that should only slide.
    #[reflect(default)]
    pub lock_rotation: bool,
}

impl Default for RigidBody {
    /// Unity's kilogram and Jolt's damping.
    fn default() -> Self {
        Self {
            kinematic: false,
            mass: 1.0,
            gravity_scale: 1.0,
            linear_damping: 0.05,
            angular_damping: 0.05,
            continuous: false,
            lock_rotation: false,
        }
    }
}
