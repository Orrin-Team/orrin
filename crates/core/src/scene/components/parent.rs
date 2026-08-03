use orrin_ecs::Entity;

/// A child's link to its parent — the only authored piece of hierarchy state.
///
/// Everything else about the hierarchy (which entities are children of what, the
/// order transforms propagate in, which entities are roots) is derived from
/// these links by [`Hierarchy`](crate::scene::Hierarchy) and cached. That is
/// deliberate: a stored `Children` list is a second thing that can disagree with
/// this one, and `World::despawn` drops components without knowing anything
/// about relationships, so a surviving parent's list would keep a dangling entry
/// that nothing ever cleans up.
///
/// The field is private and the constructor is scoped to `scene`, so the only
/// way to attach one is [`reparent`](crate::scene::reparent) — which is what
/// makes cycle rejection unavoidable rather than merely available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parent(Entity);

impl Parent {
    #[inline]
    pub(in crate::scene) fn new(entity: Entity) -> Self {
        Self(entity)
    }

    /// The parent entity. May be stale if the parent was despawned; the
    /// hierarchy rebuild treats a child of a dead parent as a root.
    #[inline]
    pub fn get(self) -> Entity {
        self.0
    }
}
