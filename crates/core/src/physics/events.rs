//! What the simulation tells scripts: enter and exit per pair of entities.
//!
//! Contact normals are unit length and point from `a` toward `b`, where
//! `(a, b)` is the canonical order from [`pair_key`]. The physics step only
//! writes events into [`CollisionState`]; delivering them to C# is the script
//! tick's job.

use glam::Vec3;

use orrin_ecs::Entity;

#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub point: Vec3,
    pub normal: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollisionEventKind {
    Enter,
    Exit,
}

#[derive(Clone, Copy, Debug)]
pub struct CollisionEvent {
    pub kind: CollisionEventKind,
    pub a: Entity,
    pub b: Entity,
    pub point: Vec3,
    pub normal: Vec3,
    /// Either side is a trigger, which scripts hear through their own pair
    /// of callbacks.
    pub trigger: bool,
}

impl CollisionEvent {
    pub(crate) fn new(
        kind: CollisionEventKind,
        (a, b): (Entity, Entity),
        contact: Contact,
        trigger: bool,
    ) -> Self {
        Self {
            kind,
            a,
            b,
            point: contact.point,
            normal: contact.normal,
            trigger,
        }
    }
}

/// This frame's events, a world resource. Cleared by each physics run and
/// drained by the script tick.
#[derive(Default)]
pub struct CollisionState {
    pub events: Vec<CollisionEvent>,
}

/// Canonical ordering for an unordered entity pair, so `(a, b)` and `(b, a)`
/// hash to the same key. Stored contacts are oriented a → b in this order.
pub fn pair_key(a: Entity, b: Entity) -> (Entity, Entity) {
    if (a.index, a.generation) <= (b.index, b.generation) {
        (a, b)
    } else {
        (b, a)
    }
}
