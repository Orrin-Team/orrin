//! What the editor has hidden from its own view of the scene: the eye on a
//! hierarchy row.
//!
//! A view setting, not scene data, so it is not a component. It is never
//! saved, never undone and never marks the scene dirty, and the game does not
//! see it: while the game plays nothing is hidden, and an exported game never
//! creates this resource at all. A hidden entity is left out of the frame —
//! its mesh, decal and light — and out of the viewport's reach, and so is
//! everything under it in the hierarchy.
//!
//! What was hidden is held by [`EntityId`], not by `Entity`, because Stop puts
//! every entity back in a new slot (`EditorState::stop`): a set of handles
//! would forget everything the first time the game ran. Ids also survive
//! reopening the scene, which is why nothing clears the set when one loads.
//!
//! The renderer asks per entity per frame, so it reads a per-slot table
//! derived from the ids rather than the ids themselves: one indexed load
//! instead of a component lookup, a hash and a walk to the root. [`refresh`]
//! rebuilds that table only when the set or the world's shape has moved, and
//! with nothing hidden it is empty and extraction skips the test entirely.

use std::collections::HashSet;

use orrin_ecs::{Entity, World};
use orrin_registry::EntityId;

use super::{Hierarchy, Play};

#[derive(Default)]
pub struct EditorHidden {
    ids: HashSet<EntityId>,
    version: u64,
    /// Per entity slot, the generation hidden there. Empty while nothing is.
    slots: Vec<Option<u32>>,
    /// The world's structural version and the set's, when `slots` was derived.
    built_at: Option<(u64, u64)>,
}

impl EditorHidden {
    /// Whether `id` was hidden itself, rather than through an ancestor.
    pub fn hides_id(&self, id: EntityId) -> bool {
        self.ids.contains(&id)
    }

    pub fn toggle(&mut self, id: EntityId) {
        if !self.ids.remove(&id) {
            self.ids.insert(id);
        }
        self.version += 1;
    }

    /// Whether `entity` is left out of the view, itself or through an
    /// ancestor, as of the last [`refresh`].
    pub fn contains(&self, entity: Entity) -> bool {
        self.slots.get(entity.index() as usize).copied().flatten() == Some(entity.generation())
    }

    /// Nothing is left out, so a caller may skip asking.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

/// Whether `entity` is left out of the editor's view. False wherever the
/// editor has hidden nothing, including every exported game.
pub fn is_hidden(world: &World, entity: Entity) -> bool {
    world
        .get_resource::<EditorHidden>()
        .is_some_and(|hidden| hidden.contains(entity))
}

/// Bring the per-slot table up to date with the hidden ids, the hierarchy and
/// whether the game is playing.
pub fn refresh(world: &mut World) {
    let Some(hidden) = world.get_resource::<EditorHidden>() else {
        return;
    };
    let live = world
        .get_resource::<Play>()
        .is_some_and(|play| play.is_live());
    if live || hidden.ids.is_empty() {
        drop(hidden);
        let mut hidden = world.resource_mut::<EditorHidden>();
        hidden.slots.clear();
        hidden.built_at = None;
        return;
    }
    let key = (world.structural_version(), hidden.version);
    if hidden.built_at == Some(key) {
        return;
    }
    drop(hidden);

    super::ensure_current(world);
    let hierarchy = world.resource::<Hierarchy>();
    let mut hidden = world.resource_mut::<EditorHidden>();
    let order = hierarchy.order();
    // Parents come strictly before children in `order`, so one forward pass
    // carries a hide down every subtree.
    let mut under = vec![false; order.len()];
    let mut slots: Vec<Option<u32>> = Vec::new();
    for (index, &entity) in order.iter().enumerate() {
        let own = world
            .get::<EntityId>(entity)
            .is_some_and(|id| hidden.ids.contains(&id));
        if !(own || under[index]) {
            continue;
        }
        for child in hierarchy.children_of(entity) {
            if let Some(position) = hierarchy.position(*child) {
                under[position] = true;
            }
        }
        let slot = entity.index() as usize;
        if slot >= slots.len() {
            slots.resize(slot + 1, None);
        }
        slots[slot] = Some(entity.generation());
    }
    hidden.slots = slots;
    hidden.built_at = Some(key);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{PlayState, reparent};

    fn world() -> World {
        let mut world = World::new();
        world.insert_resource(EditorHidden::default());
        world.insert_resource(Play::default());
        world
    }

    fn spawn(world: &mut World) -> (Entity, EntityId) {
        let entity = world.spawn();
        let id = EntityId::new();
        world.insert(entity, id);
        (entity, id)
    }

    fn hide(world: &mut World, id: EntityId) {
        world.resource_mut::<EditorHidden>().toggle(id);
        refresh(world);
    }

    #[test]
    fn hiding_an_entity_hides_everything_under_it_and_nothing_else() {
        let mut world = world();
        let (folder, folder_id) = spawn(&mut world);
        let (child, _) = spawn(&mut world);
        let (grandchild, _) = spawn(&mut world);
        let (sibling, _) = spawn(&mut world);
        reparent(&mut world, child, Some(folder), false).unwrap();
        reparent(&mut world, grandchild, Some(child), false).unwrap();

        hide(&mut world, folder_id);
        for entity in [folder, child, grandchild] {
            assert!(is_hidden(&world, entity), "{entity:?} is still shown");
        }
        assert!(!is_hidden(&world, sibling));

        hide(&mut world, folder_id);
        assert!(!is_hidden(&world, grandchild), "shown again");
    }

    /// The table follows the hierarchy: a child moved under a hidden parent
    /// after the hide is hidden too, with no second toggle.
    #[test]
    fn a_child_moved_under_a_hidden_parent_is_hidden() {
        let mut world = world();
        let (folder, folder_id) = spawn(&mut world);
        let (crate_, _) = spawn(&mut world);
        hide(&mut world, folder_id);
        assert!(!is_hidden(&world, crate_));

        reparent(&mut world, crate_, Some(folder), false).unwrap();
        refresh(&mut world);
        assert!(is_hidden(&world, crate_));
    }

    #[test]
    fn nothing_is_hidden_while_the_game_plays() {
        let mut world = world();
        let (crate_, id) = spawn(&mut world);
        hide(&mut world, id);
        assert!(is_hidden(&world, crate_));

        world.resource_mut::<Play>().set(PlayState::Playing);
        refresh(&mut world);
        assert!(!is_hidden(&world, crate_));
        assert!(world.resource::<EditorHidden>().is_empty());

        world.resource_mut::<Play>().set(PlayState::Editing);
        refresh(&mut world);
        assert!(is_hidden(&world, crate_));
    }

    /// Stop despawns the game's world and loads the scene back into new slots;
    /// what was hidden before Play is hidden after it.
    #[test]
    fn a_hidden_entity_brought_back_in_a_new_slot_is_still_hidden() {
        let mut world = world();
        let (crate_, id) = spawn(&mut world);
        hide(&mut world, id);

        world.despawn(crate_);
        let _filler = world.spawn();
        let back = world.spawn();
        world.insert(back, id);
        refresh(&mut world);
        assert_ne!(back, crate_);
        assert!(is_hidden(&world, back));
    }

    /// A slot outlives the entity in it, and whatever is spawned there next is
    /// another entity.
    #[test]
    fn a_reused_slot_is_not_hidden() {
        let mut world = world();
        let (crate_, id) = spawn(&mut world);
        hide(&mut world, id);
        world.despawn(crate_);
        let next = world.spawn();
        refresh(&mut world);
        assert_eq!(next.index(), crate_.index());
        assert!(!is_hidden(&world, next));
    }

    #[test]
    fn with_nothing_hidden_the_table_is_empty() {
        let mut world = world();
        let (_, id) = spawn(&mut world);
        hide(&mut world, id);
        assert!(!world.resource::<EditorHidden>().is_empty());
        hide(&mut world, id);
        assert!(world.resource::<EditorHidden>().is_empty());
    }
}
