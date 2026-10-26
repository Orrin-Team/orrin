//! The undo and redo stacks, the open gesture, and whether the scene has
//! changed since it was saved.

use std::collections::VecDeque;

use orrin_ecs::Entity;
use orrin_registry::{ComponentId, EntityId, Value};

use super::op::Op;

/// How many steps undo reaches back. A step holding a despawned subtree holds
/// all of it, so the stack is bounded rather than left to grow for a session.
pub const LIMIT: usize = 256;

/// A step: always an [`Op::Batch`], so every step has a label to show.
#[derive(Debug)]
pub(super) struct Step {
    /// Never reused, which is what lets "is this the state that was saved"
    /// survive undoing back to it.
    pub id: u64,
    pub op: Op,
}

/// A component an unfinished gesture has changed, and its value before.
#[derive(Debug)]
pub(super) struct Pending {
    pub id: EntityId,
    pub entity: Entity,
    pub component: ComponentId,
    pub before: Value,
}

/// Everything the edit pipeline keeps between frames. Owned by whoever runs
/// edit mode; a [`SceneEditor`](super::SceneEditor) borrows it.
#[derive(Debug, Default)]
pub struct History {
    pub(super) undo: VecDeque<Step>,
    pub(super) redo: Vec<Step>,
    pub(super) gesture: Vec<Pending>,
    /// The batch a [`SceneEditor::batch`](super::SceneEditor::batch) call is
    /// collecting, if one is open.
    pub(super) batch: Option<(String, Vec<Op>)>,
    pub(super) next: u64,
    /// The state below the bottom of the undo stack: the start, or the last
    /// step [`LIMIT`] pushed off it.
    floor: u64,
    saved: u64,
}

impl History {
    /// What undo would undo.
    pub fn undo_label(&self) -> Option<&str> {
        self.undo.back().and_then(|step| step.op.label())
    }

    /// What redo would redo.
    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().and_then(|step| step.op.label())
    }

    /// Whether the scene differs from the last one [`mark_saved`] was told
    /// about, counting a gesture still in progress.
    ///
    /// [`mark_saved`]: Self::mark_saved
    pub fn is_dirty(&self) -> bool {
        !self.gesture.is_empty() || self.current() != self.saved
    }

    pub fn mark_saved(&mut self) {
        self.saved = self.current();
    }

    /// Forget every step, and count the scene as saved: the one they
    /// described has just been replaced by one read from disk.
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.gesture.clear();
        self.batch = None;
        self.floor = self.next;
        self.saved = self.next;
    }

    /// A new step: what could be redone no longer follows from here.
    pub(super) fn push(&mut self, op: Op) {
        self.redo.clear();
        self.next += 1;
        self.undo.push_back(Step { id: self.next, op });
        if self.undo.len() > LIMIT
            && let Some(forgotten) = self.undo.pop_front()
        {
            self.floor = forgotten.id;
        }
    }

    fn current(&self) -> u64 {
        self.undo.back().map_or(self.floor, |step| step.id)
    }
}
