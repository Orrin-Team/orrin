//! The one path every edit-mode change to the scene takes (architecture §4.4).
//!
//! An edit is described as an [`Op`] before it happens, and applying it is
//! what changes the world. Undo and redo are the first consumer of that
//! stream and dirty tracking the second; prefab override capture and
//! collaboration sync are meant to be the next, without the editor changing.
//! Play mode does not come through here, and neither does editor state that
//! dies with the session — the selection, the camera, a panel's filter. The
//! rule is the scene file's: if it would be saved, it is an op.
//!
//! **Gestures.** A drag changes a value every frame, but is one edit. [`set`]
//! applies each frame's value at once, so the world shows the drag, and
//! remembers each component's value from before the gesture began; [`commit`]
//! turns the lot into one step, from the first value to the last. The editor
//! commits when the pointer is released and nothing has keyboard focus.
//!
//! **Recorded state.** A despawn records the subtree it removes, through the
//! same path a save takes — prefab instances folded to their records — and an
//! undo respawns it through the path a load takes. An undone delete is
//! therefore exactly the entities a save and reload would have given back,
//! with the same [`EntityId`]s, and nothing a save would not keep.
//!
//! **Identity.** Ops name entities by [`EntityId`], so an entity an edit
//! touches is given one then, rather than at its first save.
//!
//! **Bulk changes** whose logic lives elsewhere — instantiating a prefab,
//! reverting an instance, saving a selection as a prefab — run as they always
//! have and are recorded around ([`spawned`], [`replace`]): the subtree before
//! is despawned and the one after spawned. Not an op of its own, because a
//! despawn and a spawn of the same ids is exactly what it is.
//!
//! [`set`]: SceneEditor::set
//! [`commit`]: SceneEditor::commit
//! [`spawned`]: SceneEditor::spawned
//! [`replace`]: SceneEditor::replace

mod history;
mod op;

pub use history::{History, LIMIT};
pub use op::{EditError, Op};

use orrin_ecs::{Entity, FxBuildHasher, World};
use orrin_registry::{ComponentId, EntityId, Registry, SceneEntity, Value};

use super::registry::{MATERIAL, MESH, PARENT, TRANSFORM};
use super::{Name, Parent, persist};
use history::Pending;

type FxHashSet<T> = std::collections::HashSet<T, FxBuildHasher>;

/// Edit-mode access to the scene: read the world, change it only by op.
pub struct SceneEditor<'a> {
    world: &'a mut World,
    registry: &'a Registry,
    history: &'a mut History,
}

impl<'a> SceneEditor<'a> {
    pub fn new(world: &'a mut World, registry: &'a Registry, history: &'a mut History) -> Self {
        Self {
            world,
            registry,
            history,
        }
    }

    pub fn world(&self) -> &World {
        self.world
    }

    /// Give `component` on `entity` the value `value`, as part of the open
    /// gesture. A value equal to the current one opens nothing.
    pub fn set(
        &mut self,
        entity: Entity,
        component: &ComponentId,
        value: Value,
    ) -> Result<(), EditError> {
        let fail = |world: &World, reason: &str| {
            EditError(format!(
                "{component} on {}: {reason}",
                describe(world, entity)
            ))
        };
        if !self.world.is_alive(entity) {
            return Err(fail(self.world, "the entity is gone"));
        }
        let Some(current) = persist::read_component(self.world, self.registry, entity, component)
        else {
            return Err(fail(self.world, "the entity does not have it"));
        };
        let changes = orrin_registry::diff(&current, &value);
        if changes.is_empty() {
            return Ok(());
        }
        let id = ensure_id(self.world, entity);
        let open = self
            .history
            .gesture
            .iter()
            .any(|pending| pending.entity == entity && pending.component == *component);
        if !open {
            self.history.gesture.push(Pending {
                id,
                entity,
                component: component.clone(),
                before: current,
            });
        }
        op::set_fields(self.world, self.registry, entity, component, &changes)
            .map_err(|reason| fail(self.world, &reason))
    }

    /// Close the open gesture into one step, from each component's value
    /// before the gesture to its value now.
    pub fn commit(&mut self) {
        let mut ops = Vec::new();
        for pending in std::mem::take(&mut self.history.gesture) {
            let same = self
                .world
                .get::<EntityId>(pending.entity)
                .is_some_and(|id| *id == pending.id);
            let Some(after) = same
                .then(|| {
                    persist::read_component(
                        self.world,
                        self.registry,
                        pending.entity,
                        &pending.component,
                    )
                })
                .flatten()
            else {
                continue;
            };
            let new = orrin_registry::diff(&pending.before, &after);
            if new.is_empty() {
                continue;
            }
            ops.push(Op::SetField {
                entity: pending.id,
                component: pending.component,
                old: orrin_registry::diff(&after, &pending.before),
                new,
            });
        }
        let Some(Op::SetField { component, .. }) = ops.first() else {
            return;
        };
        let alike = ops
            .iter()
            .all(|op| matches!(op, Op::SetField { component: c, .. } if c == component));
        let label = if alike {
            format!("Edit {}", self.component_name(component))
        } else {
            "Edit".to_owned()
        };
        self.record(label, ops);
    }

    pub fn add_component(
        &mut self,
        entity: Entity,
        component: &ComponentId,
        value: Value,
    ) -> Result<(), EditError> {
        self.commit();
        if !self.world.is_alive(entity) {
            return Err(EditError(format!("adding {component}: the entity is gone")));
        }
        let op = Op::AddComponent {
            entity: ensure_id(self.world, entity),
            component: component.clone(),
            value,
        };
        op.apply(self.world, self.registry)?;
        let label = format!("Add {}", self.component_name(component));
        self.record(label, vec![op]);
        Ok(())
    }

    pub fn remove_component(
        &mut self,
        entity: Entity,
        component: &ComponentId,
    ) -> Result<(), EditError> {
        self.commit();
        let Some(value) = self
            .world
            .is_alive(entity)
            .then(|| persist::read_component(self.world, self.registry, entity, component))
            .flatten()
        else {
            return Err(EditError(format!(
                "removing {component} from {}: it has none",
                describe(self.world, entity)
            )));
        };
        let op = Op::RemoveComponent {
            entity: ensure_id(self.world, entity),
            component: component.clone(),
            value,
        };
        op.apply(self.world, self.registry)?;
        let label = format!("Remove {}", self.component_name(component));
        self.record(label, vec![op]);
        Ok(())
    }

    /// Despawn each of `roots` and everything under it, as one step.
    pub fn despawn(&mut self, roots: &[Entity]) -> Result<(), EditError> {
        self.commit();
        let roots: Vec<Entity> = roots
            .iter()
            .copied()
            .filter(|&root| self.world.is_alive(root))
            .collect();
        let label = match roots[..] {
            [] => return Ok(()),
            [one] => format!("Delete {}", describe(self.world, one)),
            _ => format!("Delete {} entities", roots.len()),
        };
        let (entities, slots) = self.snapshot(&roots);
        let op = Op::Despawn { entities, slots };
        op.apply(self.world, self.registry)?;
        self.record(label, vec![op]);
        Ok(())
    }

    /// Record subtrees something else has just spawned.
    pub fn spawned(&mut self, label: impl Into<String>, roots: &[Entity]) {
        self.commit();
        let (entities, slots) = self.snapshot(roots);
        if !entities.is_empty() {
            self.record(label.into(), vec![Op::Spawn { entities, slots }]);
        }
    }

    /// Run `change`, which reshapes the subtrees under `before` and returns
    /// the roots of what it left, and record the whole as one step.
    pub fn replace<R>(
        &mut self,
        label: impl Into<String>,
        before: &[Entity],
        change: impl FnOnce(&mut World, &Registry) -> (R, Vec<Entity>),
    ) -> R {
        self.commit();
        let old = self.snapshot(before);
        let (result, after) = change(self.world, self.registry);
        let new = self.snapshot(&after);
        if old != new {
            let mut ops = Vec::with_capacity(2);
            let ((old, old_slots), (new, new_slots)) = (old, new);
            if !old.is_empty() {
                ops.push(Op::Despawn {
                    entities: old,
                    slots: old_slots,
                });
            }
            if !new.is_empty() {
                ops.push(Op::Spawn {
                    entities: new,
                    slots: new_slots,
                });
            }
            self.record(label.into(), ops);
        }
        result
    }

    /// Run `change`, which alters one component of `entity`, and record it as
    /// the op it amounted to — added, removed, or set.
    pub fn changed<R>(
        &mut self,
        label: impl Into<String>,
        entity: Entity,
        component: &ComponentId,
        change: impl FnOnce(&mut World, &Registry) -> R,
    ) -> R {
        self.commit();
        let read = |world: &World| persist::read_component(world, self.registry, entity, component);
        let before = self
            .world
            .is_alive(entity)
            .then(|| read(self.world))
            .flatten();
        let result = change(self.world, self.registry);
        if !self.world.is_alive(entity) {
            return result;
        }
        let after = read(self.world);
        let id = ensure_id(self.world, entity);
        let component = component.clone();
        let op = match (before, after) {
            (None, Some(value)) => Op::AddComponent {
                entity: id,
                component,
                value,
            },
            (Some(value), None) => Op::RemoveComponent {
                entity: id,
                component,
                value,
            },
            (Some(before), Some(after)) => {
                let new = orrin_registry::diff(&before, &after);
                if new.is_empty() {
                    return result;
                }
                Op::SetField {
                    entity: id,
                    component,
                    old: orrin_registry::diff(&after, &before),
                    new,
                }
            }
            (None, None) => return result,
        };
        self.record(label.into(), vec![op]);
        result
    }

    /// Attach `child` to `parent`, or make it a root. With `keep_world`, its
    /// transform is rewritten so it does not move, and that is part of the
    /// same step.
    pub fn reparent(
        &mut self,
        child: Entity,
        parent: Option<Entity>,
        keep_world: bool,
    ) -> Result<(), EditError> {
        self.commit();
        let name = describe(self.world, child);
        if !self.world.is_alive(child) || parent.is_some_and(|p| !self.world.is_alive(p)) {
            return Err(EditError(format!(
                "reparenting {name}: an entity in the move is gone"
            )));
        }
        let from = self
            .world
            .get::<Parent>(child)
            .map(|parent| parent.get())
            .filter(|&parent| self.world.is_alive(parent));
        let before = keep_world
            .then(|| persist::read_component(self.world, self.registry, child, &TRANSFORM))
            .flatten();
        super::reparent(self.world, child, parent, keep_world)
            .map_err(|error| EditError(format!("reparenting {name}: {error}")))?;

        let entity = ensure_id(self.world, child);
        let mut ops = vec![Op::Reparent {
            entity,
            from: from.map(|from| ensure_id(self.world, from)),
            to: parent.map(|to| ensure_id(self.world, to)),
        }];
        if let Some(before) = before
            && let Some(after) =
                persist::read_component(self.world, self.registry, child, &TRANSFORM)
        {
            let new = orrin_registry::diff(&before, &after);
            if !new.is_empty() {
                ops.push(Op::SetField {
                    entity,
                    component: TRANSFORM,
                    old: orrin_registry::diff(&after, &before),
                    new,
                });
            }
        }
        self.record(format!("Reparent {name}"), ops);
        Ok(())
    }

    /// Everything `edits` does, as one step under `label`. Nested batches
    /// join the outermost.
    pub fn batch<R>(&mut self, label: impl Into<String>, edits: impl FnOnce(&mut Self) -> R) -> R {
        self.commit();
        if self.history.batch.is_some() {
            return edits(self);
        }
        self.history.batch = Some((label.into(), Vec::new()));
        let result = edits(self);
        self.commit();
        let (label, ops) = self.history.batch.take().expect("opened above");
        if !ops.is_empty() {
            self.history.push(Op::Batch { label, ops });
        }
        result
    }

    /// Undo the last step and return its label, or `None` when there is
    /// nothing to undo. A step that cannot be undone stays where it was.
    pub fn undo(&mut self) -> Result<Option<String>, EditError> {
        self.commit();
        let Some(step) = self.history.undo.pop_back() else {
            return Ok(None);
        };
        match step.op.clone().invert().apply(self.world, self.registry) {
            Ok(()) => {
                let label = step.op.label().map(str::to_owned);
                self.history.redo.push(step);
                Ok(label)
            }
            Err(error) => {
                self.history.undo.push_back(step);
                Err(error)
            }
        }
    }

    pub fn redo(&mut self) -> Result<Option<String>, EditError> {
        self.commit();
        let Some(step) = self.history.redo.pop() else {
            return Ok(None);
        };
        match step.op.apply(self.world, self.registry) {
            Ok(()) => {
                let label = step.op.label().map(str::to_owned);
                self.history.undo.push_back(step);
                Ok(label)
            }
            Err(error) => {
                self.history.redo.push(step);
                Err(error)
            }
        }
    }

    /// Already-applied ops, as a step of their own or into the open batch.
    fn record(&mut self, label: String, ops: Vec<Op>) {
        match &mut self.history.batch {
            Some((_, batch)) => batch.extend(ops),
            None => self.history.push(Op::Batch { label, ops }),
        }
    }

    /// The subtrees under `roots` as a save would write them, and the slot
    /// each entity in them sits in.
    fn snapshot(&mut self, roots: &[Entity]) -> (Vec<SceneEntity>, Vec<(EntityId, u32)>) {
        super::ensure_current(self.world);
        let mut order = Vec::new();
        let mut seen = FxHashSet::default();
        {
            let hierarchy = self.world.resource::<super::Hierarchy>();
            let mut stack: Vec<Entity> = roots.iter().rev().copied().collect();
            while let Some(entity) = stack.pop() {
                if self.world.is_alive(entity) && seen.insert(entity) {
                    order.push(entity);
                    stack.extend(hierarchy.children_of(entity).iter().rev());
                }
            }
        }
        let slots = order
            .iter()
            .map(|&entity| (ensure_id(self.world, entity), entity.index()))
            .collect();
        let mut entities: Vec<SceneEntity> = order
            .into_iter()
            .filter_map(|entity| persist::scene_entity(self.world, self.registry, entity))
            .collect();
        super::prefab::fold(self.world, &mut entities);
        (entities, slots)
    }

    fn component_name(&self, id: &ComponentId) -> String {
        match id {
            id if *id == MESH => "Mesh".to_owned(),
            id if *id == MATERIAL => "Material".to_owned(),
            id if *id == PARENT => "Parent".to_owned(),
            id => self
                .registry
                .get(id)
                .map_or_else(|| id.to_string(), |vtable| vtable.name.to_string()),
        }
    }
}

/// The id `entity` is addressed by in ops, given one if it has none yet.
fn ensure_id(world: &mut World, entity: Entity) -> EntityId {
    if let Some(id) = world.get::<EntityId>(entity) {
        return *id;
    }
    let id = EntityId::new();
    world.insert(entity, id);
    id
}

fn describe(world: &World, entity: Entity) -> String {
    world.get::<Name>(entity).map_or_else(
        || format!("entity {}", entity.index()),
        |name| name.0.clone(),
    )
}

#[cfg(test)]
mod tests {
    use glam::Vec3;
    use orrin_registry::{EntityId, Reflect};

    use super::*;
    use crate::scene::registry::{MESH, TAG, TRANSFORM};
    use crate::scene::{
        Assets, LocalTransform, MaterialHandle, MeshHandle, Name, Parent, Prefabs, Tag, Transform,
        register_components,
    };

    fn registry() -> Registry {
        let mut registry = Registry::new();
        register_components(&mut registry);
        registry
    }

    fn world() -> World {
        let mut world = World::new();
        let mut assets = Assets::new();
        assets.insert_mesh("cube", MeshHandle(3));
        assets.insert_material("gold", MaterialHandle(7));
        world.insert_resource(assets);
        world
    }

    fn cube(world: &mut World, name: &str) -> Entity {
        let entity = world.spawn();
        world.insert(entity, Name::new(name));
        world.insert(entity, LocalTransform::default());
        world.insert(entity, MeshHandle(3));
        world.insert(entity, MaterialHandle(7));
        entity
    }

    fn at(x: f32) -> Value {
        LocalTransform::new(Transform::from_translation(Vec3::new(x, 0.0, 0.0))).to_value()
    }

    fn x_of(world: &World, entity: Entity) -> f32 {
        world.get::<LocalTransform>(entity).unwrap().translation.x
    }

    fn file(world: &mut World, registry: &Registry) -> String {
        crate::scene::save(world, registry)
    }

    fn find(world: &World, name: &str) -> Entity {
        world
            .entities()
            .find(|&e| world.get::<Name>(e).is_some_and(|n| n.0 == name))
            .unwrap_or_else(|| panic!("no entity is named {name}"))
    }

    #[test]
    fn an_undone_edit_leaves_the_scene_file_as_it_was() {
        let registry = registry();
        let mut world = world();
        let mut history = History::default();
        let entity = cube(&mut world, "Cube");
        let before = file(&mut world, &registry);

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.set(entity, &TRANSFORM, at(2.0)).unwrap();
        editor.commit();
        let edited = file(&mut world, &registry);
        assert_ne!(edited, before);

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        assert_eq!(editor.undo().unwrap().as_deref(), Some("Edit Transform"));
        assert_eq!(file(&mut world, &registry), before);

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.redo().unwrap();
        assert_eq!(file(&mut world, &registry), edited);
    }

    /// The case gizmos and inspector drags exist for: every frame shows the
    /// new value, and the whole drag is one step.
    #[test]
    fn a_drag_is_one_step_from_its_first_value_to_its_last() {
        let registry = registry();
        let mut world = world();
        let mut history = History::default();
        let entity = cube(&mut world, "Cube");

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        for x in [1.0, 2.0, 3.0] {
            editor.set(entity, &TRANSFORM, at(x)).unwrap();
            assert_eq!(x_of(editor.world(), entity), x, "a frame of the drag shows");
        }
        editor.commit();

        assert_eq!(editor.undo().unwrap().as_deref(), Some("Edit Transform"));
        assert_eq!(x_of(editor.world(), entity), 0.0);
        assert_eq!(editor.undo().unwrap(), None, "the drag was one step");
        editor.redo().unwrap();
        assert_eq!(x_of(editor.world(), entity), 3.0);
    }

    #[test]
    fn a_new_edit_forgets_what_could_be_redone() {
        let registry = registry();
        let mut world = world();
        let mut history = History::default();
        let entity = cube(&mut world, "Cube");

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.set(entity, &TRANSFORM, at(1.0)).unwrap();
        editor.commit();
        editor.undo().unwrap();
        editor.set(entity, &TRANSFORM, at(5.0)).unwrap();
        editor.commit();

        assert_eq!(editor.redo().unwrap(), None);
        assert_eq!(x_of(editor.world(), entity), 5.0);
    }

    /// Undo of a delete gives back the same identities, so anything that
    /// refers to them — a child elsewhere, a script field, a later step on the
    /// stack — still finds them.
    #[test]
    fn an_undone_delete_restores_the_subtree_with_its_ids() {
        let registry = registry();
        let mut world = world();
        let mut history = History::default();
        let parent = cube(&mut world, "Parent");
        let child = cube(&mut world, "Child");
        crate::scene::reparent(&mut world, child, Some(parent), false).unwrap();
        let before = file(&mut world, &registry);

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.despawn(&[parent]).unwrap();
        assert!(!world.is_alive(parent) && !world.is_alive(child));

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        assert_eq!(editor.undo().unwrap().as_deref(), Some("Delete Parent"));
        assert_eq!(file(&mut world, &registry), before);
        let child = find(&world, "Child");
        let parent = world.get::<Parent>(child).unwrap().get();
        assert_eq!(world.get::<Name>(parent).unwrap().0, "Parent");

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.redo().unwrap();
        assert_eq!(world.entities().count(), 0);
    }

    /// Names in the order the hierarchy panel draws them.
    fn tree(world: &mut World) -> Vec<String> {
        crate::scene::ensure_current(world);
        let hierarchy = world.resource::<crate::scene::Hierarchy>();
        hierarchy
            .order()
            .iter()
            .map(|&entity| world.get::<Name>(entity).unwrap().0.clone())
            .collect()
    }

    /// Nothing but where each entity sits in the ECS orders siblings, so a
    /// respawn into whatever slot is free would reshuffle the tree.
    #[test]
    fn an_undone_delete_keeps_the_order_of_the_tree() {
        let registry = registry();
        let mut world = world();
        let mut history = History::default();
        let parent = cube(&mut world, "Parent");
        let a = cube(&mut world, "A");
        let b = cube(&mut world, "B");
        let c = cube(&mut world, "C");
        let d = cube(&mut world, "D");
        for child in [b, c] {
            crate::scene::reparent(&mut world, child, Some(parent), false).unwrap();
        }
        let before = tree(&mut world);

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.despawn(&[a, parent, d]).unwrap();
        editor.undo().unwrap();
        assert_eq!(tree(&mut world), before);

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.redo().unwrap();
        editor.undo().unwrap();
        assert_eq!(tree(&mut world), before, "and again after a redo");
    }

    /// The step after a delete names the entity by id, and finds it again
    /// once the delete has been undone into a different slot.
    #[test]
    fn a_step_finds_its_entity_after_an_undo_respawned_it() {
        let registry = registry();
        let mut world = world();
        let mut history = History::default();
        let entity = cube(&mut world, "Cube");

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.set(entity, &TRANSFORM, at(4.0)).unwrap();
        editor.commit();
        editor.despawn(&[entity]).unwrap();
        editor.undo().unwrap();
        editor.undo().unwrap();

        let entity = find(&world, "Cube");
        assert_eq!(x_of(&world, entity), 0.0);
    }

    #[test]
    fn a_batch_undoes_as_one_step() {
        let registry = registry();
        let mut world = world();
        let mut history = History::default();
        let a = cube(&mut world, "A");
        let b = cube(&mut world, "B");
        let before = file(&mut world, &registry);

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.batch("Tidy", |editor| {
            editor.reparent(b, Some(a), false).unwrap();
            editor
                .add_component(a, &TAG, Tag::new("crate").to_value())
                .unwrap();
            editor.set(b, &TRANSFORM, at(9.0)).unwrap();
        });

        assert_eq!(editor.undo().unwrap().as_deref(), Some("Tidy"));
        assert_eq!(editor.undo().unwrap(), None);
        assert_eq!(file(&mut world, &registry), before);
    }

    /// Half an undo is worse than a refused one: a step that fails part-way
    /// puts back what it had already done, and stays on the stack.
    #[test]
    fn a_step_that_cannot_be_undone_changes_nothing() {
        let registry = registry();
        let mut world = world();
        let mut history = History::default();
        let a = cube(&mut world, "A");
        let b = cube(&mut world, "B");

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.batch("Both", |editor| {
            editor.set(b, &TRANSFORM, at(1.0)).unwrap();
            editor
                .add_component(a, &TAG, Tag::new("crate").to_value())
                .unwrap();
        });
        // Gone behind the pipeline's back, as a script in play mode might.
        world.despawn(b);
        let before = file(&mut world, &registry);

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        let error = editor.undo().unwrap_err();
        assert!(error.0.contains("orrin.transform"), "{error}");
        assert_eq!(file(&mut world, &registry), before);
        assert!(world.has::<Tag>(a), "the half already undone was put back");
        assert_eq!(history.undo_label(), Some("Both"));
    }

    #[test]
    fn a_reparent_that_keeps_the_world_position_undoes_both_halves() {
        let registry = registry();
        let mut world = world();
        let mut history = History::default();
        let parent = cube(&mut world, "Parent");
        world.insert(
            parent,
            LocalTransform::new(Transform::from_translation(Vec3::new(10.0, 0.0, 0.0))),
        );
        let child = cube(&mut world, "Child");
        crate::scene::propagate_transforms(&mut world);
        let before = file(&mut world, &registry);

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.reparent(child, Some(parent), true).unwrap();
        assert_eq!(x_of(editor.world(), child), -10.0, "it stayed where it was");

        assert_eq!(editor.undo().unwrap().as_deref(), Some("Reparent Child"));
        assert_eq!(file(&mut world, &registry), before);
        assert!(!world.has::<Parent>(child));
    }

    /// A component held by name rather than through the registry — the mesh —
    /// comes back the way a load would bring it back.
    #[test]
    fn a_removed_mesh_comes_back_on_undo() {
        let registry = registry();
        let mut world = world();
        let mut history = History::default();
        let entity = cube(&mut world, "Cube");

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.remove_component(entity, &MESH).unwrap();
        assert!(!editor.world().has::<MeshHandle>(entity));
        assert_eq!(editor.undo().unwrap().as_deref(), Some("Remove Mesh"));
        assert_eq!(
            world.get::<MeshHandle>(entity).map(|h| *h),
            Some(MeshHandle(3))
        );
    }

    /// A change whose logic lives elsewhere — an override given back to its
    /// prefab — is recorded as whichever op it amounted to.
    #[test]
    fn a_component_changed_elsewhere_is_recorded_as_the_op_it_amounts_to() {
        let registry = registry();
        let mut world = world();
        let mut history = History::default();
        let entity = cube(&mut world, "Cube");
        let before = file(&mut world, &registry);

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.changed("Revert Tag", entity, &TAG, |world, _| {
            world.insert(entity, Tag::new("crate"));
        });
        editor.changed("Revert Transform", entity, &TRANSFORM, |world, _| {
            world.insert(
                entity,
                LocalTransform::new(Transform::from_translation(Vec3::X)),
            );
        });
        editor.changed("Revert Mesh", entity, &MESH, |world, _| {
            world.remove::<MeshHandle>(entity);
        });
        editor.changed("Nothing", entity, &MESH, |_, _| {});

        assert_eq!(editor.undo().unwrap().as_deref(), Some("Revert Mesh"));
        assert!(editor.world().has::<MeshHandle>(entity));
        assert_eq!(editor.undo().unwrap().as_deref(), Some("Revert Transform"));
        assert_eq!(x_of(editor.world(), entity), 0.0);
        assert_eq!(editor.undo().unwrap().as_deref(), Some("Revert Tag"));
        assert_eq!(file(&mut world, &registry), before);
    }

    #[test]
    fn a_recorded_spawn_undoes_and_redoes_with_its_id() {
        let registry = registry();
        let mut world = world();
        let mut history = History::default();
        let entity = cube(&mut world, "Cube");

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.spawned("Create Cube", &[entity]);
        let id = *editor.world().get::<EntityId>(entity).unwrap();
        editor.undo().unwrap();
        assert_eq!(world.entities().count(), 0);

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        assert_eq!(editor.redo().unwrap().as_deref(), Some("Create Cube"));
        let entity = find(&world, "Cube");
        assert_eq!(world.get::<EntityId>(entity).map(|id| *id), Some(id));
    }

    /// An instance is recorded as a save writes it — its record, not its
    /// members — and comes back expanded from its prefab.
    #[test]
    fn an_undone_delete_of_a_prefab_instance_brings_the_instance_back() {
        let registry = registry();
        let mut world = world();
        let mut history = History::default();
        let id = orrin_registry::AssetId::new();
        let text = "orrin-scene 1\n\nentity 00000000-0000-4000-8000-000000000001\n  \
                    orrin.name = \"Door\"\n\nentity 00000000-0000-4000-8000-000000000002\n  \
                    orrin.name = \"Handle\"\n  \
                    orrin.parent = entity(00000000-0000-4000-8000-000000000001)\n";
        let mut prefabs = Prefabs::default();
        prefabs.insert(crate::scene::prefab::Prefab::parse(id, "door.fprefab", text).unwrap());
        world.insert_resource(prefabs);
        let (root, issues) = crate::scene::prefab::instantiate(&mut world, &registry, id).unwrap();
        assert!(issues.is_empty(), "{issues:?}");
        for name in ["Knob A", "Knob B"] {
            let knob = cube(&mut world, name);
            crate::scene::reparent(&mut world, knob, Some(root), false).unwrap();
        }
        let order = tree(&mut world);
        let before = file(&mut world, &registry);

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.despawn(&[root]).unwrap();
        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.undo().unwrap();

        assert_eq!(file(&mut world, &registry), before);
        assert_eq!(tree(&mut world), order);
        assert!(world.has::<crate::scene::PrefabInstance>(find(&world, "Door")));
        find(&world, "Handle");
    }

    #[test]
    fn undoing_back_to_the_saved_scene_makes_it_clean_again() {
        let registry = registry();
        let mut world = world();
        let mut history = History::default();
        let entity = cube(&mut world, "Cube");
        history.mark_saved();
        assert!(!history.is_dirty());

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.set(entity, &TRANSFORM, at(1.0)).unwrap();
        assert!(editor.history.is_dirty(), "an unfinished drag is a change");
        editor.commit();
        editor.undo().unwrap();
        assert!(!history.is_dirty());

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        editor.redo().unwrap();
        assert!(history.is_dirty());
    }

    #[test]
    fn undo_reaches_back_a_bounded_number_of_steps() {
        let registry = registry();
        let mut world = world();
        let mut history = History::default();
        let entity = cube(&mut world, "Cube");

        let mut editor = SceneEditor::new(&mut world, &registry, &mut history);
        for step in 0..=LIMIT {
            editor
                .set(entity, &TRANSFORM, at(step as f32 + 1.0))
                .unwrap();
            editor.commit();
        }
        let mut undone = 0;
        while editor.undo().unwrap().is_some() {
            undone += 1;
        }
        assert_eq!(undone, LIMIT);
        assert_eq!(x_of(&world, entity), 1.0, "the oldest step was forgotten");
    }
}
