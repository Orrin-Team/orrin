//! Which entities the editor is acting on.
//!
//! An ordered set with a *primary*: the entity clicked last, which the
//! inspector edits. Ordered by when each was added rather than by the tree, so
//! the primary is simply the last, and a Ctrl-click that removes the primary
//! hands the role back to whichever was added before it.
//!
//! A Shift-click selects the run of visible rows between the clicked row and
//! the *anchor* — the last row clicked without Shift — and keeps that anchor,
//! so a second Shift-click re-pivots on the same row rather than on the end of
//! the previous run, as a file browser does.

use orrin_ecs::{Entity, World};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Selection {
    entities: Vec<Entity>,
    anchor: Option<Entity>,
}

impl Selection {
    /// The entity the inspector edits.
    pub fn primary(&self) -> Option<Entity> {
        self.entities.last().copied()
    }

    pub fn contains(&self, entity: Entity) -> bool {
        self.entities.contains(&entity)
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// In the order they were selected; the primary last.
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Select `entity` alone: a plain click.
    pub fn set(&mut self, entity: Entity) {
        self.entities.clear();
        self.entities.push(entity);
        self.anchor = Some(entity);
    }

    /// Add `entity` as the primary, or take it away if it is already selected:
    /// a Ctrl-click.
    pub fn toggle(&mut self, entity: Entity) {
        match self.entities.iter().position(|&e| e == entity) {
            Some(at) => {
                self.entities.remove(at);
            }
            None => self.entities.push(entity),
        }
        self.anchor = Some(entity);
    }

    /// Select the rows of `order` from the anchor to `entity`, with `entity`
    /// the primary: a Shift-click. Without an anchor in `order` — nothing
    /// clicked yet, or the anchor's row filtered away — it is a plain click.
    pub fn extend_to(&mut self, entity: Entity, order: &[Entity]) {
        let anchor = self.anchor;
        let (Some(from), Some(to)) = (
            anchor.and_then(|anchor| order.iter().position(|&e| e == anchor)),
            order.iter().position(|&e| e == entity),
        ) else {
            self.set(entity);
            return;
        };
        let run = if from <= to {
            &order[from..=to]
        } else {
            &order[to..=from]
        };
        self.entities.clear();
        self.entities
            .extend(run.iter().copied().filter(|&e| e != entity));
        self.entities.push(entity);
        self.anchor = anchor;
    }

    pub fn clear(&mut self) {
        self.entities.clear();
        self.anchor = None;
    }

    /// Drop anything despawned since it was selected.
    pub fn retain_alive(&mut self, world: &World) {
        self.entities.retain(|&e| world.is_alive(e));
        if self.anchor.is_some_and(|anchor| !world.is_alive(anchor)) {
            self.anchor = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entities(n: usize) -> (World, Vec<Entity>) {
        let mut world = World::new();
        let all = (0..n).map(|_| world.spawn()).collect();
        (world, all)
    }

    #[test]
    fn a_plain_click_selects_one() {
        let (_, e) = entities(3);
        let mut selection = Selection::default();
        selection.set(e[0]);
        selection.set(e[2]);
        assert_eq!(selection.entities(), [e[2]]);
        assert_eq!(selection.primary(), Some(e[2]));
    }

    #[test]
    fn ctrl_click_adds_and_removes_and_the_primary_follows() {
        let (_, e) = entities(3);
        let mut selection = Selection::default();
        selection.set(e[0]);
        selection.toggle(e[1]);
        assert_eq!(selection.entities(), [e[0], e[1]]);
        assert_eq!(selection.primary(), Some(e[1]));

        selection.toggle(e[1]);
        assert_eq!(selection.primary(), Some(e[0]));
    }

    #[test]
    fn shift_click_takes_the_run_from_the_anchor_either_way() {
        let (_, e) = entities(5);
        let mut selection = Selection::default();
        selection.set(e[1]);
        selection.extend_to(e[3], &e);
        assert_eq!(selection.entities(), [e[1], e[2], e[3]]);
        assert_eq!(selection.primary(), Some(e[3]));

        // Pivots on the same anchor, not on the end of the last run.
        selection.extend_to(e[0], &e);
        assert_eq!(selection.entities(), [e[1], e[0]]);
    }

    #[test]
    fn shift_click_with_nothing_to_pivot_on_is_a_plain_click() {
        let (_, e) = entities(3);
        let mut selection = Selection::default();
        selection.extend_to(e[2], &e);
        assert_eq!(selection.entities(), [e[2]]);
    }

    #[test]
    fn a_despawned_entity_leaves_the_selection() {
        let (mut world, e) = entities(2);
        let mut selection = Selection::default();
        selection.set(e[0]);
        selection.toggle(e[1]);
        world.despawn(e[1]);
        selection.retain_alive(&world);
        assert_eq!(selection.entities(), [e[0]]);
    }
}
