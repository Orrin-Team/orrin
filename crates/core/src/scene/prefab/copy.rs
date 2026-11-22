//! Copying subtrees of a scene and putting them back as new entities: the
//! editor's duplicate, copy and paste.
//!
//! A copy is taken the way a save takes the scene — each entity's components
//! as a file holds them, every prefab instance folded to its record — so a
//! paste is a load of that, under fresh ids, and a copied instance comes back
//! an instance of the same prefab rather than as entities that happen to look
//! like one. It lives with prefabs because the re-iding has to know their one
//! rule: an instance's member ids are derived from its root's, so a member is
//! never given an id of its own, and anything that named one is pointed at the
//! member the new root derives.
//!
//! References are remapped only within the copy. One to an entity outside it
//! is kept, which is what lets a duplicate go on following what the original
//! followed; pasted into another scene, the same reference names nothing.

use orrin_ecs::{Entity, FxHashMap, World};
use orrin_registry::{EntityId, Registry, SceneEntity, Value};

use super::{PrefabInstance, Prefabs, is_member, remap, xor};
use crate::scene::persist::{self, LoadIssue};
use crate::scene::registry::PARENT;

/// Subtrees of a scene, held as a save would write them.
#[derive(Clone, Debug)]
pub struct Copied {
    entities: Vec<SceneEntity>,
    /// The ids of the copied trees' roots, in the order they were selected.
    roots: Vec<EntityId>,
    /// Each id folded into an instance's record, with the root of the
    /// instance it is a member of.
    members: Vec<(EntityId, EntityId)>,
    /// What a step that pastes it is called after: the one tree's name, or
    /// how many trees.
    label: String,
}

impl Copied {
    /// How many trees were copied.
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Where pasted trees go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    /// Under the parent each was copied from: a duplicate.
    Beside,
    /// Under this entity, or at the top level for `None`.
    Under(Option<Entity>),
}

/// The trees rooted at `selection`, each whole; an entity selected under
/// another selected one comes with its ancestor. `None` when nothing in the
/// selection is alive.
pub fn copy(world: &mut World, registry: &Registry, selection: &[Entity]) -> Option<Copied> {
    let tops = super::create::tops(world, selection);
    if tops.is_empty() {
        return None;
    }
    let children = super::create::children_of(world);
    let mut order = Vec::new();
    for &top in &tops {
        for entity in super::create::subtree(&children, top) {
            if !order.contains(&entity) {
                order.push(entity);
            }
        }
    }
    for &entity in &order {
        crate::scene::edit::ensure_id(world, entity);
    }
    let mut entities: Vec<SceneEntity> = order
        .iter()
        .filter_map(|&entity| persist::scene_entity(world, registry, entity))
        .collect();
    let before: Vec<(Entity, EntityId)> = order
        .iter()
        .map(|&entity| (entity, *world.get::<EntityId>(entity).expect("given above")))
        .collect();
    super::fold(world, &mut entities);
    let kept: std::collections::HashSet<EntityId> = entities.iter().map(|e| e.id).collect();
    let members = before
        .into_iter()
        .filter(|(_, id)| !kept.contains(id))
        .filter_map(|(entity, id)| Some((id, enclosing_root(world, entity, id)?)))
        .collect();
    let roots = tops
        .iter()
        .map(|&top| *world.get::<EntityId>(top).expect("given above"))
        .collect();
    let label = match tops[..] {
        [one] => super::create::label(world, one),
        _ => format!("{} entities", tops.len()),
    };
    Some(Copied {
        entities,
        roots,
        members,
        label,
    })
}

/// Spawn `copied` again under fresh ids, placed as `place` says. Returns the
/// new roots, in the order they were copied, with whatever the spawn could not
/// apply.
pub fn paste(
    world: &mut World,
    registry: &Registry,
    copied: &Copied,
    place: Place,
) -> (Vec<Entity>, Vec<LoadIssue>) {
    let mut ids: FxHashMap<EntityId, EntityId> = copied
        .entities
        .iter()
        .map(|entity| (entity.id, EntityId::new()))
        .collect();
    // A member's id is its prefab id XOR its root's, so a new root moves it by
    // the XOR of the old root and the new. A root that is itself a member,
    // nested in another instance, has moved by its own root's first.
    let mut pending: Vec<(EntityId, EntityId)> = copied.members.clone();
    while !pending.is_empty() {
        let before = pending.len();
        pending.retain(|&(member, root)| match ids.get(&root) {
            Some(&new_root) => {
                ids.insert(member, xor(xor(member, root), new_root));
                false
            }
            None => true,
        });
        // A root outside the copy cannot be: `fold` only folds an instance
        // whose root it was given.
        if pending.len() == before {
            break;
        }
    }

    let parent = match place {
        Place::Beside => None,
        Place::Under(parent) => Some(parent.map(|p| crate::scene::edit::ensure_id(world, p))),
    };
    let mut entities = copied.entities.clone();
    for entity in &mut entities {
        let root = copied.roots.contains(&entity.id);
        entity.id = ids[&entity.id];
        for (_, value) in &mut entity.components {
            remap(value, &mut |id| ids.get(&id).copied());
        }
        if root && let Some(parent) = parent {
            entity
                .components
                .retain(|(component, _)| *component != PARENT);
            if let Some(parent) = parent {
                entity.components.push((PARENT, Value::Entity(parent)));
            }
        }
    }
    let issues = persist::spawn(&entities, world, registry, &FxHashMap::default());
    let live = persist::ids_of(world);
    let roots = copied
        .roots
        .iter()
        .filter_map(|root| live.get(&ids[root]).copied())
        .collect();
    (roots, issues)
}

/// The root of the instance `entity`, whose id is `id`, is a member of: the
/// nearest one above it that counts it among its members. An instance root
/// is looked for above itself, since it is the root of its own.
fn enclosing_root(world: &World, entity: Entity, id: EntityId) -> Option<EntityId> {
    let prefabs = world.get_resource::<Prefabs>()?;
    let parent = |entity: Entity| persist::live_parent(world, entity);
    let mut cursor = if world.has::<PrefabInstance>(entity) {
        parent(entity)?
    } else {
        entity
    };
    // Bounded: a cycle the hierarchy has not broken yet would otherwise spin.
    for _ in 0..4096 {
        if let Some(instance) = world.get::<PrefabInstance>(cursor)
            && let Some(root) = world.get::<EntityId>(cursor).map(|id| *id)
            && prefabs
                .get(instance.prefab)
                .is_some_and(|prefab| is_member(prefab, root, id))
        {
            return Some(root);
        }
        cursor = parent(cursor)?;
    }
    None
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::super::index::tests::{DOOR_ID, HANDLE, door_id, door_text, registry};
    use super::super::{Prefab, instantiate};
    use super::*;
    use crate::scene::{LocalTransform, Name, Parent, Transform, reparent};

    #[derive(Default, orrin_registry::Reflect)]
    struct Note {
        about: EntityId,
    }

    const HOUSE_ID: &str = "40000000-0000-4000-8000-000000000001";
    const HOUSE_ROOT: &str = "40000000-0000-4000-8000-00000000000a";
    const HOUSE_DOOR: &str = "40000000-0000-4000-8000-00000000000b";

    /// The door, and a house holding one as a record.
    fn world() -> (World, Registry) {
        let mut registry = registry();
        registry.register::<Note>(orrin_registry::ComponentId::new("game.note"), "Note");
        let house = format!(
            "orrin-scene 1\n\nentity {HOUSE_ROOT}\n  orrin.name = \"House\"\n\n\
             entity {HOUSE_DOOR}\n  orrin.parent = entity({HOUSE_ROOT})\n  orrin.prefab\n    \
             source = asset({DOOR_ID})\n"
        );
        let mut prefabs = Prefabs::default();
        prefabs
            .insert(Prefab::parse(door_id(), "door.oprefab", &door_text("handle", 1.0)).unwrap());
        prefabs.insert(Prefab::parse(HOUSE_ID.parse().unwrap(), "house.oprefab", &house).unwrap());
        let mut world = World::new();
        let mut assets = crate::scene::Assets::new();
        assets.insert_mesh("cube", crate::scene::MeshHandle(3));
        world.insert_resource(assets);
        world.insert_resource(prefabs);
        (world, registry)
    }

    fn named(world: &World, name: &str) -> Vec<Entity> {
        world
            .entities()
            .filter(|&e| world.get::<Name>(e).is_some_and(|n| n.0 == name))
            .collect()
    }

    fn id(world: &World, entity: Entity) -> EntityId {
        *world.get::<EntityId>(entity).unwrap()
    }

    fn parent(world: &World, entity: Entity) -> Option<Entity> {
        world.get::<Parent>(entity).map(|p| p.get())
    }

    /// An override, and a scene entity under a member that names it, both
    /// come with a copied instance, onto the copy's own members.
    #[test]
    fn a_pasted_instance_keeps_its_overrides_and_what_hangs_off_its_members() {
        let (mut world, registry) = world();
        let (door, _) = instantiate(&mut world, &registry, door_id()).unwrap();
        let panel = named(&world, "Panel")[0];
        world.insert(
            panel,
            LocalTransform::new(Transform::from_translation(Vec3::new(0.0, 3.0, 0.0))),
        );
        let latch = world.spawn();
        world.insert(latch, Name::new("Latch"));
        reparent(&mut world, latch, Some(panel), false).unwrap();
        let panel_id = id(&world, panel);
        world.insert(latch, Note { about: panel_id });

        let copied = copy(&mut world, &registry, &[door]).unwrap();
        let (roots, issues) = paste(&mut world, &registry, &copied, Place::Under(None));
        assert!(issues.is_empty(), "{issues:?}");

        let [copy] = roots[..] else {
            panic!("{roots:?}")
        };
        assert_ne!(copy, door);
        assert_eq!(world.get::<PrefabInstance>(copy).unwrap().prefab, door_id());
        let panels = named(&world, "Panel");
        let new_panel = *panels.iter().find(|&&p| p != panel).unwrap();
        assert_eq!(parent(&world, new_panel), Some(copy));
        assert_eq!(
            world
                .get::<LocalTransform>(new_panel)
                .unwrap()
                .translation
                .y,
            3.0,
            "the override came too"
        );
        let new_latch = *named(&world, "Latch")
            .iter()
            .find(|&&l| l != latch)
            .unwrap();
        assert_eq!(parent(&world, new_latch), Some(new_panel));
        assert_eq!(
            world.get::<Note>(new_latch).unwrap().about,
            id(&world, new_panel)
        );
        assert_eq!(world.get::<Note>(latch).unwrap().about, panel_id);
    }

    /// A member of an instance nested in another moves with the outer root:
    /// what named it finds the nested copy.
    #[test]
    fn a_member_of_a_nested_instance_is_found_in_the_copy() {
        let (mut world, registry) = world();
        let (house, issues) =
            instantiate(&mut world, &registry, HOUSE_ID.parse().unwrap()).unwrap();
        assert!(issues.is_empty(), "{issues:?}");
        let handle = named(&world, "Handle")[0];
        let note = world.spawn();
        world.insert(note, Name::new("Note"));
        reparent(&mut world, note, Some(handle), false).unwrap();
        let handle_id = id(&world, handle);
        world.insert(note, Note { about: handle_id });

        let copied = copy(&mut world, &registry, &[house]).unwrap();
        let (roots, issues) = paste(&mut world, &registry, &copied, Place::Beside);
        assert!(issues.is_empty(), "{issues:?}");

        let [copy] = roots[..] else {
            panic!("{roots:?}")
        };
        let new_handle = *named(&world, "Handle")
            .iter()
            .find(|&&h| h != handle)
            .unwrap();
        let new_note = *named(&world, "Note").iter().find(|&&n| n != note).unwrap();
        assert_eq!(parent(&world, new_note), Some(new_handle));
        assert_eq!(
            world.get::<Note>(new_note).unwrap().about,
            id(&world, new_handle)
        );
        let new_door = parent(&world, parent(&world, new_handle).unwrap()).unwrap();
        assert_eq!(parent(&world, new_door), Some(copy));
        let prefabs = world.resource::<Prefabs>();
        assert!(is_member(
            prefabs.get(door_id()).unwrap(),
            id(&world, new_door),
            id(&world, new_handle)
        ));
        assert_ne!(id(&world, new_handle), HANDLE.parse().unwrap());
    }

    /// Of a tree and something under it, the tree is copied once.
    #[test]
    fn an_entity_selected_under_another_comes_with_it_once() {
        let (mut world, registry) = world();
        let lamp = world.spawn();
        world.insert(lamp, Name::new("Lamp"));
        let bulb = world.spawn();
        world.insert(bulb, Name::new("Bulb"));
        reparent(&mut world, bulb, Some(lamp), false).unwrap();

        let copied = copy(&mut world, &registry, &[bulb, lamp]).unwrap();
        assert_eq!(copied.len(), 1);
        let (roots, _) = paste(&mut world, &registry, &copied, Place::Beside);
        assert_eq!(roots.len(), 1);
        assert_eq!(named(&world, "Bulb").len(), 2);
    }
}
