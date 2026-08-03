//! Turning a world into a scene file and back.
//!
//! The registry supplies every component's read and write; this module supplies
//! only the rules that need a world — assigning identities, spawning, and what
//! to do with data nothing claims.

use orrin_ecs::{Entity, FxHashMap, World};
use orrin_registry::{
    ComponentId, EntityId, ParseError, SceneDocument, SceneEntity, Value, write_document,
};

use super::registry::{MATERIAL, MESH, PARENT};
use super::{Assets, MaterialHandle, MeshHandle, Parent, UnknownComponents};

/// A component a load could not apply. The data is preserved on the entity (see
/// [`UnknownComponents`]); this is what the editor console should say about it.
#[derive(Clone, Debug)]
pub struct LoadIssue {
    pub entity: EntityId,
    pub component: ComponentId,
    pub message: String,
}

impl std::fmt::Display for LoadIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "entity {}: `{}` was kept but not applied — {}",
            self.entity, self.component, self.message
        )
    }
}

/// Serialize every live entity.
///
/// Takes `&mut World` because saving assigns an [`EntityId`] to any entity that
/// lacks one. That makes identity a consequence of being saved: entities a
/// script spawns and despawns within a session never acquire one, and an entity
/// that has been saved keeps the same id forever after — so a second save of an
/// unchanged scene is byte-identical.
pub fn save(world: &mut World, registry: &orrin_registry::Registry) -> String {
    let document = to_document(world, registry);
    let mut text = String::new();
    write_document(&mut text, &document);
    text
}

pub fn to_document(world: &mut World, registry: &orrin_registry::Registry) -> SceneDocument {
    assign_missing_ids(world);

    let entities = save_order(world)
        .into_iter()
        .map(|entity| {
            let id = *world.get::<EntityId>(entity).expect("just assigned");

            let mut components: Vec<(ComponentId, orrin_registry::Value)> = registry
                .components()
                .filter_map(|c| (c.read)(world, entity).map(|value| (c.id.clone(), value)))
                .collect();

            components.extend(asset_refs(world, entity));
            components.extend(parent_ref(world, entity));

            if let Some(unknown) = world.get::<UnknownComponents>(entity) {
                components.extend(unknown.0.iter().cloned());
            }

            SceneEntity { id, components }
        })
        .collect();

    SceneDocument { entities }
}

/// Depth-first pre-order, siblings by [`EntityId`] — parents before children,
/// and each subtree contiguous.
///
/// Two payoffs. A subtree reads as a block rather than being scattered through
/// the file, and the order is a function of identities rather than of slots, so
/// a save after a load is byte-identical even though the second session
/// allocated its slots in a different order.
fn save_order(world: &World) -> Vec<Entity> {
    let mut children: FxHashMap<Entity, Vec<Entity>> = FxHashMap::default();
    let mut roots: Vec<Entity> = Vec::new();
    for entity in world.entities() {
        match live_parent(world, entity) {
            Some(parent) => children.entry(parent).or_default().push(entity),
            None => roots.push(entity),
        }
    }

    let by_id = |list: &mut Vec<Entity>| {
        list.sort_by_key(|&entity| world.get::<EntityId>(entity).map(|id| *id));
    };
    by_id(&mut roots);

    let mut order = Vec::with_capacity(roots.len());
    // Reversed on the way in so `pop` yields siblings in ascending id order.
    let mut stack: Vec<Entity> = roots.into_iter().rev().collect();
    while let Some(entity) = stack.pop() {
        order.push(entity);
        if let Some(kids) = children.get(&entity) {
            let mut kids = kids.clone();
            by_id(&mut kids);
            stack.extend(kids.into_iter().rev());
        }
    }

    // A cycle has no root, so the walk above would never reach it — and an
    // entity missing from the save is data destroyed rather than a link lost.
    // The hierarchy breaks cycles on the next propagation; until then they still
    // get written, as roots.
    if order.len() != world.entities().count() {
        let written: std::collections::HashSet<Entity> = order.iter().copied().collect();
        order.extend(world.entities().filter(|e| !written.contains(e)));
    }

    order
}

/// The parent link, as the parent's persistent identity.
///
/// `None` for an entity with no parent, and for one whose parent has been
/// despawned — an orphan is written as a root, which is exactly what it is.
fn parent_ref(world: &World, entity: Entity) -> Option<(ComponentId, Value)> {
    let parent = live_parent(world, entity)?;
    let id = *world.get::<EntityId>(parent)?;
    Some((PARENT, Value::Entity(id)))
}

fn live_parent(world: &World, entity: Entity) -> Option<Entity> {
    let parent = world.get::<Parent>(entity)?.get();
    world.is_alive(parent).then_some(parent)
}

/// Read a scene and add its entities to `world`.
///
/// Additive by design — this is the primitive both "open a scene" and "spawn a
/// prefab" are built from, and only the caller knows whether the existing
/// contents should have been despawned first.
pub fn load(
    text: &str,
    world: &mut World,
    registry: &orrin_registry::Registry,
) -> Result<Vec<LoadIssue>, ParseError> {
    let document = orrin_registry::parse(text)?;
    Ok(instantiate(&document, world, registry))
}

/// Spawn `document`'s entities. Returns whatever could not be applied; the
/// scene still loads, because refusing to open a scene over one stale field
/// makes a recoverable problem unrecoverable.
///
/// Components holding [`EntityId`] rather than raw entity handles means a
/// reference in the file is already the value the component should hold, so
/// almost everything applies in one pass. `orrin.parent` is the exception: it
/// resolves to a live handle, and the entity it names may not have been spawned
/// yet. Saves are written parents-first, but a file can be hand-edited into any
/// order, so the links are deferred to a second pass over a map built during
/// the first.
pub fn instantiate(
    document: &SceneDocument,
    world: &mut World,
    registry: &orrin_registry::Registry,
) -> Vec<LoadIssue> {
    let mut issues = Vec::new();
    let mut spawned: FxHashMap<EntityId, Entity> = FxHashMap::default();
    let mut deferred_parents: Vec<(Entity, EntityId, Value)> = Vec::new();

    for scene_entity in &document.entities {
        let entity = world.spawn();
        world.insert(entity, scene_entity.id);
        spawned.insert(scene_entity.id, entity);

        let mut unknown = Vec::new();
        for (id, value) in &scene_entity.components {
            if *id == PARENT {
                match value {
                    Value::Entity(parent_id) => {
                        deferred_parents.push((entity, *parent_id, value.clone()))
                    }
                    _ => {
                        issues.push(LoadIssue {
                            entity: scene_entity.id,
                            component: id.clone(),
                            message: format!(
                                "expected an entity reference, found {}",
                                value.type_name()
                            ),
                        });
                        unknown.push((id.clone(), value.clone()));
                    }
                }
                continue;
            }
            if *id == MESH || *id == MATERIAL {
                if let Err(message) = apply_asset_ref(world, entity, id, value) {
                    issues.push(LoadIssue {
                        entity: scene_entity.id,
                        component: id.clone(),
                        message,
                    });
                    unknown.push((id.clone(), value.clone()));
                }
                continue;
            }
            match registry.get(id) {
                Some(vtable) => {
                    if let Err(error) = (vtable.write)(world, entity, value) {
                        // Kept as well as reported: a field the current build
                        // rejects is exactly the data a save must not silently
                        // drop.
                        issues.push(LoadIssue {
                            entity: scene_entity.id,
                            component: id.clone(),
                            message: error.to_string(),
                        });
                        unknown.push((id.clone(), value.clone()));
                    }
                }
                None => {
                    issues.push(LoadIssue {
                        entity: scene_entity.id,
                        component: id.clone(),
                        message: "no component type is registered under this id".to_owned(),
                    });
                    unknown.push((id.clone(), value.clone()));
                }
            }
        }

        if !unknown.is_empty() {
            world.insert(entity, UnknownComponents(unknown));
        }
    }

    // Pass two: the parent links, now that every id in the document names a
    // spawned entity. Routed through `reparent` rather than inserted directly,
    // so a file describing a cycle is refused link by link with a message
    // naming the entities, instead of loading and being silently dismantled by
    // the next hierarchy rebuild.
    for (child, parent_id, value) in deferred_parents {
        let child_id = *world.get::<EntityId>(child).expect("just inserted");
        let message = match spawned.get(&parent_id) {
            Some(&parent) => match super::reparent(world, child, Some(parent), false) {
                Ok(()) => continue,
                Err(error) => error.to_string(),
            },
            None => format!("no entity in this scene has the id {parent_id}"),
        };

        issues.push(LoadIssue {
            entity: child_id,
            component: PARENT,
            message,
        });
        keep_unapplied(world, child, (PARENT, value));
    }

    issues
}

/// Preserve a component the load could not apply, so a later save does not drop
/// it. Appends when the entity already collected others during the first pass.
fn keep_unapplied(world: &mut World, entity: Entity, component: (ComponentId, Value)) {
    if let Some(mut existing) = world.get_mut::<UnknownComponents>(entity) {
        existing.0.push(component);
        return;
    }
    world.insert(entity, UnknownComponents(vec![component]));
}

/// Mesh and material references, stored by asset name rather than by handle.
///
/// A `MeshHandle` is an index into the backend's upload table: correct for the
/// session that produced it and meaningless in any other. Writing one to disk
/// would attach a scene to whatever happens to occupy that slot next time the
/// assets are registered in a different order — wrong silently, with no error
/// anywhere. The name is the only part that means the same thing twice.
///
/// This lives outside the registry because resolving a handle to a name needs
/// the `Assets` resource, and `Reflect::to_value` sees only `&self`. That
/// restriction is deliberate — it keeps conversions pure — so the translation
/// happens here, where the world is in scope. It goes away when assets gain
/// stable ids and these components can hold one directly.
fn asset_refs(world: &World, entity: Entity) -> Vec<(ComponentId, Value)> {
    let mesh = world.get::<MeshHandle>(entity).map(|h| *h);
    let material = world.get::<MaterialHandle>(entity).map(|h| *h);
    if mesh.is_none() && material.is_none() {
        return Vec::new();
    }
    let Some(assets) = world.get_resource::<Assets>() else {
        return Vec::new();
    };

    // A handle `Assets` cannot name was fabricated outside it, which nothing
    // does; there is no id to write for it, so it is left out rather than
    // written as a number that will mean something else later.
    let mut refs = Vec::new();
    if let Some(name) = mesh.and_then(|h| assets.mesh_name(h)) {
        refs.push((MESH, Value::String(name.to_owned())));
    }
    if let Some(name) = material.and_then(|h| assets.material_name(h)) {
        refs.push((MATERIAL, Value::String(name.to_owned())));
    }
    refs
}

fn apply_asset_ref(
    world: &mut World,
    entity: Entity,
    id: &ComponentId,
    value: &Value,
) -> Result<(), String> {
    let Value::String(name) = value else {
        return Err(format!(
            "expected an asset name, found {}",
            value.type_name()
        ));
    };

    // Resolved before inserting: the lookup borrows the `Assets` resource and
    // the insert needs the world mutably.
    let handle = {
        let Some(assets) = world.get_resource::<Assets>() else {
            return Err("no asset registry is loaded".to_owned());
        };
        if *id == MESH {
            assets.mesh(name).map(Asset::Mesh)
        } else {
            assets.material(name).map(Asset::Material)
        }
    };

    match handle {
        Some(Asset::Mesh(handle)) => {
            world.insert(entity, handle);
            Ok(())
        }
        Some(Asset::Material(handle)) => {
            world.insert(entity, handle);
            Ok(())
        }
        None => Err(format!("no asset named `{name}`")),
    }
}

enum Asset {
    Mesh(MeshHandle),
    Material(MaterialHandle),
}

fn assign_missing_ids(world: &mut World) {
    // Collected before inserting: `entities()` borrows the world, and `insert`
    // needs it mutably.
    let missing: Vec<Entity> = world
        .entities()
        .filter(|&entity| !world.has::<EntityId>(entity))
        .collect();
    for entity in missing {
        world.insert(entity, EntityId::new());
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;
    use orrin_registry::{ComponentId, Registry, Value};

    use super::*;
    use crate::scene::{Light, LocalTransform, Name, Tag, Transform, register_components};

    fn registry() -> Registry {
        let mut registry = Registry::new();
        register_components(&mut registry);
        registry
    }

    fn assets() -> Assets {
        let mut assets = Assets::new();
        assets.insert_mesh("cube", MeshHandle(3));
        assets.insert_material("gold", MaterialHandle(7));
        assets
    }

    fn populated() -> World {
        let mut world = World::new();
        world.insert_resource(assets());
        let cube = world.spawn();
        world.insert(cube, MeshHandle(3));
        world.insert(cube, MaterialHandle(7));
        world.insert(
            cube,
            LocalTransform::new(Transform::from_translation(Vec3::new(0.0, 1.5, 0.0))),
        );
        world.insert(cube, Name::new("Cube"));
        world.insert(cube, Tag::new("player"));

        let light = world.spawn();
        world.insert(light, Name::new("Sun"));
        world.insert(light, Light::directional(Vec3::ONE, 3.0));
        world
    }

    /// The whole point of the round trip: an entity comes back renderable.
    /// `systems::extract` requires a `MeshHandle`, so a scene that loses one
    /// loads an entity that exists and draws nothing.
    #[test]
    fn a_renderable_comes_back_renderable() {
        let registry = registry();
        let mut world = populated();
        let text = save(&mut world, &registry);
        assert!(text.contains("orrin.mesh = \"cube\""), "{text}");
        assert!(text.contains("orrin.material = \"gold\""), "{text}");

        // A different session, where the same names uploaded to different slots.
        let mut loaded = World::new();
        let mut assets = Assets::new();
        assets.insert_mesh("cube", MeshHandle(11));
        assets.insert_material("gold", MaterialHandle(12));
        loaded.insert_resource(assets);

        let issues = load(&text, &mut loaded, &registry).unwrap();
        assert!(issues.is_empty(), "{issues:?}");

        let cube = loaded
            .entities()
            .find(|&e| loaded.get::<Name>(e).is_some_and(|n| n.0 == "Cube"))
            .unwrap();
        // Resolved through the names, not carried across as raw indices.
        assert_eq!(*loaded.get::<MeshHandle>(cube).unwrap(), MeshHandle(11));
        assert_eq!(
            *loaded.get::<MaterialHandle>(cube).unwrap(),
            MaterialHandle(12)
        );
    }

    #[test]
    fn an_asset_name_the_registry_lacks_is_reported_and_kept() {
        let registry = registry();
        let id = EntityId::new();
        let text = format!("orrin-scene 1\n\nentity {id}\n  orrin.mesh = \"teapot\"\n");

        let mut world = World::new();
        world.insert_resource(assets());
        let issues = load(&text, &mut world, &registry).unwrap();

        assert_eq!(issues.len(), 1);
        assert!(issues[0].message.contains("no asset named `teapot`"));

        let entity = world.entities().next().unwrap();
        assert!(world.get::<MeshHandle>(entity).is_none());
        assert_eq!(save(&mut world, &registry), text);
    }

    #[test]
    fn a_world_survives_a_save_load_save_cycle_byte_for_byte() {
        let registry = registry();
        let mut world = populated();
        let first = save(&mut world, &registry);

        let mut loaded = World::new();
        loaded.insert_resource(assets());
        let issues = load(&first, &mut loaded, &registry).unwrap();
        assert!(issues.is_empty(), "{issues:?}");

        let second = save(&mut loaded, &registry);
        assert_eq!(first, second);
    }

    #[test]
    fn saving_twice_is_stable_because_ids_are_assigned_once() {
        let registry = registry();
        let mut world = populated();
        assert_eq!(save(&mut world, &registry), save(&mut world, &registry));
    }

    #[test]
    fn the_components_actually_land_on_the_entities() {
        let registry = registry();
        let mut world = populated();
        let text = save(&mut world, &registry);

        let mut loaded = World::new();
        load(&text, &mut loaded, &registry).unwrap();

        let names: Vec<String> = loaded
            .entities()
            .filter_map(|e| loaded.get::<Name>(e).map(|n| n.0.clone()))
            .collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"Cube".to_owned()));

        let cube = loaded
            .entities()
            .find(|&e| loaded.get::<Name>(e).is_some_and(|n| n.0 == "Cube"))
            .unwrap();
        assert_eq!(
            loaded.get::<LocalTransform>(cube).unwrap().translation,
            Vec3::new(0.0, 1.5, 0.0)
        );
        assert_eq!(loaded.get::<Tag>(cube).unwrap().0, "player");
    }

    #[test]
    fn an_unregistered_component_is_reported_and_written_back_unchanged() {
        let registry = registry();
        let id = EntityId::new();
        let text = format!(
            "orrin-scene 1\n\nentity {id}\n  game.enemy\n    aggression = 0.75\n    name = \"Boss\"\n"
        );

        let mut world = World::new();
        let issues = load(&text, &mut world, &registry).unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].component, ComponentId::owned("game.enemy"));

        // The point of the exercise: saving without the game assembly loaded
        // must not destroy its data.
        assert_eq!(save(&mut world, &registry), text);
    }

    #[test]
    fn a_value_the_type_rejects_is_kept_rather_than_dropped() {
        let registry = registry();
        let id = EntityId::new();
        let text =
            format!("orrin-scene 1\n\nentity {id}\n  orrin.name = \"Cube\"\n  orrin.tag = 7\n");

        let mut world = World::new();
        let issues = load(&text, &mut world, &registry).unwrap();
        assert_eq!(issues.len(), 1);
        assert!(
            issues[0].message.contains("expected string"),
            "{}",
            issues[0]
        );

        let entity = world.entities().next().unwrap();
        assert!(world.get::<Tag>(entity).is_none());
        assert_eq!(save(&mut world, &registry), text);
    }

    #[test]
    fn instantiating_twice_adds_entities_rather_than_replacing_them() {
        let registry = registry();
        let mut source = populated();
        let document = to_document(&mut source, &registry);

        let mut world = World::new();
        instantiate(&document, &mut world, &registry);
        instantiate(&document, &mut world, &registry);
        assert_eq!(world.entities().count(), 4);
    }

    #[test]
    fn a_parse_error_leaves_the_world_untouched() {
        let registry = registry();
        let mut world = World::new();
        let err = load("not a scene\n", &mut world, &registry).unwrap_err();
        assert_eq!(err.line, 1);
        assert_eq!(world.entities().count(), 0);
    }

    #[test]
    fn an_empty_world_saves_and_reloads() {
        let registry = registry();
        let mut world = World::new();
        let text = save(&mut world, &registry);
        assert_eq!(text, "orrin-scene 1\n");

        let mut loaded = World::new();
        assert!(load(&text, &mut loaded, &registry).unwrap().is_empty());
        assert_eq!(loaded.entities().count(), 0);
    }

    #[test]
    fn an_entity_reference_field_survives_because_it_is_an_identity() {
        // No engine component holds one yet, so this exercises the leaf
        // directly: the id written is the id read, with no remapping step.
        let target = EntityId::new();
        let value = Value::Entity(target);
        let mut text = String::new();
        write_document(
            &mut text,
            &SceneDocument {
                entities: vec![SceneEntity {
                    id: EntityId::new(),
                    components: vec![(ComponentId::owned("game.follow"), value.clone())],
                }],
            },
        );

        let parsed = orrin_registry::parse(&text).unwrap();
        assert_eq!(parsed.entities[0].components[0].1, value);
    }

    fn hierarchy_world() -> (World, Entity, Entity, Entity) {
        let mut world = World::new();
        world.insert_resource(assets());
        let root = world.spawn();
        world.insert(root, Name::new("Root"));
        world.insert(
            root,
            LocalTransform::new(Transform::from_translation(Vec3::new(5.0, 0.0, 0.0))),
        );
        let child = world.spawn();
        world.insert(child, Name::new("Child"));
        world.insert(child, LocalTransform::new(Transform::default()));
        let grandchild = world.spawn();
        world.insert(grandchild, Name::new("Grandchild"));
        world.insert(grandchild, LocalTransform::new(Transform::default()));

        crate::scene::reparent(&mut world, child, Some(root), false).unwrap();
        crate::scene::reparent(&mut world, grandchild, Some(child), false).unwrap();
        (world, root, child, grandchild)
    }

    fn name_of(world: &World, entity: Entity) -> String {
        world.get::<Name>(entity).unwrap().0.clone()
    }

    fn find_by_name(world: &World, name: &str) -> Entity {
        world
            .entities()
            .find(|&e| world.get::<Name>(e).map(|n| n.0 == name).unwrap_or(false))
            .expect("no entity by that name")
    }

    /// The hierarchy has to survive the round trip, or reparenting in the editor
    /// is work the next save silently discards.
    #[test]
    fn a_parent_link_survives_a_round_trip() {
        let registry = registry();
        let (mut world, _, _, _) = hierarchy_world();
        let text = save(&mut world, &registry);
        assert!(text.contains("orrin.parent"), "{text}");

        let mut loaded = World::new();
        loaded.insert_resource(assets());
        let issues = load(&text, &mut loaded, &registry).unwrap();
        assert!(issues.is_empty(), "{issues:?}");

        let root = find_by_name(&loaded, "Root");
        let child = find_by_name(&loaded, "Child");
        let grandchild = find_by_name(&loaded, "Grandchild");
        assert_eq!(loaded.get::<Parent>(child).unwrap().get(), root);
        assert_eq!(loaded.get::<Parent>(grandchild).unwrap().get(), child);
        assert!(loaded.get::<Parent>(root).is_none());
    }

    /// The composed transform has to come back too, not just the link.
    #[test]
    fn a_loaded_child_propagates_from_its_parent() {
        let registry = registry();
        let (mut world, _, _, _) = hierarchy_world();
        let text = save(&mut world, &registry);

        let mut loaded = World::new();
        loaded.insert_resource(assets());
        load(&text, &mut loaded, &registry).unwrap();
        crate::scene::propagate_transforms(&mut loaded);

        let grandchild = find_by_name(&loaded, "Grandchild");
        let world_pos = loaded
            .get::<crate::scene::WorldTransform>(grandchild)
            .unwrap()
            .translation();
        assert!((world_pos - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5);
    }

    /// Parents before children, and each subtree contiguous — what makes a
    /// hierarchy readable in the file and keeps a diff local to what moved.
    #[test]
    fn entities_are_written_parents_first() {
        let registry = registry();
        let (mut world, _, _, _) = hierarchy_world();
        let document = to_document(&mut world, &registry);

        let names: Vec<String> = document
            .entities
            .iter()
            .map(|e| {
                let entity = world
                    .entities()
                    .find(|&x| world.get::<EntityId>(x).map(|id| *id) == Some(e.id))
                    .unwrap();
                name_of(&world, entity)
            })
            .collect();
        assert_eq!(names, vec!["Root", "Child", "Grandchild"]);
    }

    /// The reason the load is two-pass at all. Saves are written parents-first,
    /// but a file is a text file and someone will reorder it.
    #[test]
    fn a_child_listed_before_its_parent_still_resolves() {
        let registry = registry();
        let (mut world, _, _, _) = hierarchy_world();
        let mut document = to_document(&mut world, &registry);
        document.entities.reverse();

        let mut loaded = World::new();
        loaded.insert_resource(assets());
        let issues = instantiate(&document, &mut loaded, &registry);
        assert!(issues.is_empty(), "{issues:?}");

        let root = find_by_name(&loaded, "Root");
        let child = find_by_name(&loaded, "Child");
        assert_eq!(loaded.get::<Parent>(child).unwrap().get(), root);
    }

    /// A dangling reference is reported and kept, never silently dropped — the
    /// same rule the loader already applies to a field a component rejects.
    #[test]
    fn a_parent_id_no_entity_has_is_reported_and_kept() {
        let registry = registry();
        let (mut world, _, _, _) = hierarchy_world();
        let mut document = to_document(&mut world, &registry);
        // Point the child at an id nothing in the document carries.
        let missing = EntityId::new();
        for entity in &mut document.entities {
            for (id, value) in &mut entity.components {
                if *id == PARENT {
                    *value = Value::Entity(missing);
                }
            }
        }

        let mut loaded = World::new();
        loaded.insert_resource(assets());
        let issues = instantiate(&document, &mut loaded, &registry);

        assert_eq!(issues.len(), 2, "{issues:?}");
        assert!(issues[0].message.contains(&missing.to_string()));
        let child = find_by_name(&loaded, "Child");
        assert!(loaded.get::<Parent>(child).is_none(), "left as a root");
        assert!(
            loaded.get::<UnknownComponents>(child).is_some(),
            "the unresolved link was dropped instead of kept"
        );
    }

    /// A file describing a cycle is refused link by link, with a message naming
    /// the entities — better than loading it and having the next hierarchy
    /// rebuild silently dismantle it.
    #[test]
    fn a_cycle_in_the_file_is_reported_rather_than_loaded() {
        let registry = registry();
        let (mut world, root, _, grandchild) = hierarchy_world();
        let mut document = to_document(&mut world, &registry);
        let grandchild_id = *world.get::<EntityId>(grandchild).unwrap();
        let root_id = *world.get::<EntityId>(root).unwrap();
        // Close the loop: root's parent becomes its own grandchild.
        for entity in &mut document.entities {
            if entity.id == root_id {
                entity
                    .components
                    .push((PARENT, Value::Entity(grandchild_id)));
            }
        }

        let mut loaded = World::new();
        loaded.insert_resource(assets());
        let issues = instantiate(&document, &mut loaded, &registry);

        assert_eq!(issues.len(), 1, "{issues:?}");
        assert!(issues[0].message.contains("cycle"), "{:?}", issues[0]);
        // And the scene is still usable rather than half-built.
        crate::scene::propagate_transforms(&mut loaded);
        assert_eq!(loaded.entities().count(), 3);
    }

    /// Byte-for-byte stability has to hold across a session boundary, not just
    /// within one — which is why the save order is keyed on identities rather
    /// than on slots.
    #[test]
    fn a_hierarchy_survives_a_save_load_save_cycle_byte_for_byte() {
        let registry = registry();
        let (mut world, _, _, _) = hierarchy_world();
        let first = save(&mut world, &registry);

        let mut loaded = World::new();
        loaded.insert_resource(assets());
        load(&first, &mut loaded, &registry).unwrap();
        let second = save(&mut loaded, &registry);

        assert_eq!(first, second);
    }
}
