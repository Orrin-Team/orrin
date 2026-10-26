//! Turning a world into a scene file and back.
//!
//! The registry supplies every component's read and write; this module supplies
//! only the rules that need a world — assigning identities, spawning, and what
//! to do with data nothing claims.

use orrin_ecs::{Entity, FxHashMap, World};
use orrin_registry::{
    ComponentId, EntityId, ParseError, SceneDocument, SceneEntity, Value, write_document,
};

use super::prefab::{PrefabInstance, Prefabs};
use super::registry::{MATERIAL, MESH, PARENT, PREFAB};
use super::{Assets, MaterialHandle, MeshHandle, Parent, PendingScripts, UnknownComponents};

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

    let mut entities: Vec<SceneEntity> = save_order(world)
        .into_iter()
        .map(|entity| scene_entity(world, registry, entity).expect("just assigned"))
        .collect();
    super::prefab::fold(world, &mut entities);

    SceneDocument { entities }
}

/// One entity as a scene file writes it, or `None` if it has no [`EntityId`].
pub(super) fn scene_entity(
    world: &World,
    registry: &orrin_registry::Registry,
    entity: Entity,
) -> Option<SceneEntity> {
    let id = *world.get::<EntityId>(entity)?;

    let mut components: Vec<(ComponentId, orrin_registry::Value)> = registry
        .components()
        .filter_map(|c| c.read(world, entity).map(|value| (c.id.clone(), value)))
        .collect();

    components.extend(asset_refs(world, entity));
    components.extend(parent_ref(world, entity));

    if let Some(unknown) = world.get::<UnknownComponents>(entity) {
        components.extend(unknown.0.iter().cloned());
    }
    if let Some(pending) = world.get::<PendingScripts>(entity) {
        components.extend(pending.0.iter().cloned());
    }

    Some(SceneEntity { id, components })
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
    spawn(&document.entities, world, registry, &FxHashMap::default())
}

/// [`instantiate`] over a slice, spawning an id found in `reserved` into the
/// entity it names rather than a new one — which is how a prefab instance a
/// script asked for lands on the handle the script was already given.
///
/// A prefab instance's record is expanded into its members first, so from the
/// spawning loop on they are ordinary entities.
pub(super) fn spawn(
    entities: &[SceneEntity],
    world: &mut World,
    registry: &orrin_registry::Registry,
    reserved: &FxHashMap<EntityId, Entity>,
) -> Vec<LoadIssue> {
    let (holders, plain): (Vec<&SceneEntity>, Vec<&SceneEntity>) = entities
        .iter()
        .partition(|entity| super::prefab::record_of(entity).is_some());
    let expanded = {
        let prefabs = world.get_resource::<Prefabs>();
        let prefabs = prefabs.as_deref();
        super::prefab::expand_all(&|id| prefabs.and_then(|p| p.get(id)), holders)
    };
    let mut issues = expanded.issues;
    issues.extend(spawn_plain(
        plain.into_iter().chain(&expanded.entities),
        &expanded.instances,
        world,
        registry,
        reserved,
    ));
    issues
}

/// Every instance of one component type in a load, in file order.
type Section<'a> = (&'a ComponentId, Vec<(Entity, EntityId, &'a Value)>);

/// Spawn entities that carry no prefab record — the second half of [`spawn`],
/// and how a refreshed instance gains the members its prefab now has.
///
/// A parent link resolves against the entities spawned here first and then
/// against the world, which is what lets a new member find a parent that was
/// already live.
///
/// Components are applied a type at a time rather than an entity at a time:
/// every entity is spawned first, then each component type's vtable is looked
/// up once and every instance of it in the file streamed through. Per-instance
/// dispatch is the slow path the format's one-section-per-type layout exists
/// to avoid (#41).
pub(super) fn spawn_plain<'a>(
    entities: impl Iterator<Item = &'a SceneEntity>,
    instances: &[(EntityId, PrefabInstance)],
    world: &mut World,
    registry: &orrin_registry::Registry,
    reserved: &FxHashMap<EntityId, Entity>,
) -> Vec<LoadIssue> {
    let mut issues = Vec::new();
    let mut spawned: FxHashMap<EntityId, Entity> = FxHashMap::default();
    let mut deferred_parents: Vec<(Entity, EntityId, Value)> = Vec::new();

    // In the order each type first appears, so a load's issues come out in
    // the same order every time.
    let mut sections: Vec<Section> = Vec::new();
    let mut section_of: FxHashMap<&ComponentId, usize> = FxHashMap::default();
    for scene_entity in entities {
        let entity = reserved
            .get(&scene_entity.id)
            .copied()
            .unwrap_or_else(|| world.spawn());
        world.insert(entity, scene_entity.id);
        spawned.insert(scene_entity.id, entity);

        for (id, value) in &scene_entity.components {
            let section = *section_of.entry(id).or_insert_with(|| {
                sections.push((id, Vec::new()));
                sections.len() - 1
            });
            sections[section].1.push((entity, scene_entity.id, value));
        }
    }

    for (id, members) in sections {
        // A record that could not be expanded, reported above, or one inside
        // a prefab, which is kept but not expanded.
        if *id == PREFAB {
            for (entity, _, value) in members {
                keep_unapplied(world, entity, (id.clone(), value.clone()));
            }
            continue;
        }

        let vtable = registry.get(id);
        for (entity, entity_id, value) in members {
            let result = if *id == PARENT {
                match value {
                    Value::Entity(parent_id) => {
                        deferred_parents.push((entity, *parent_id, value.clone()));
                        continue;
                    }
                    _ => Err(format!(
                        "expected an entity reference, found {}",
                        value.type_name()
                    )),
                }
            } else if *id == MESH || *id == MATERIAL {
                apply_asset_ref(world, entity, id, value)
            } else {
                apply_registered(vtable, world, entity, id, value)
            };

            if let Err(message) = result {
                // Kept as well as reported: a field the current build rejects
                // is exactly the data a save must not silently drop.
                issues.push(LoadIssue {
                    entity: entity_id,
                    component: id.clone(),
                    message,
                });
                keep_unapplied(world, entity, (id.clone(), value.clone()));
            }
        }
    }

    for (root, instance) in instances {
        if let Some(&entity) = spawned.get(root) {
            world.insert(entity, instance.clone());
        }
    }

    // Pass two: the parent links, now that every id in the document names a
    // spawned entity. Routed through `reparent` rather than inserted directly,
    // so a file describing a cycle is refused link by link with a message
    // naming the entities, instead of loading and being silently dismantled by
    // the next hierarchy rebuild.
    let mut live: Option<FxHashMap<EntityId, Entity>> = None;
    for (child, parent_id, value) in deferred_parents {
        let child_id = *world.get::<EntityId>(child).expect("just inserted");
        let found = spawned.get(&parent_id).copied().or_else(|| {
            live.get_or_insert_with(|| ids_of(world))
                .get(&parent_id)
                .copied()
        });
        let message = match found {
            Some(parent) => match super::reparent(world, child, Some(parent), false) {
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

/// Every entity that has a persistent identity, by it.
pub(super) fn ids_of(world: &World) -> FxHashMap<EntityId, Entity> {
    world
        .entities()
        .filter_map(|entity| world.get::<EntityId>(entity).map(|id| (*id, entity)))
        .collect()
}

/// Give `entity` one component from its file form: an asset reference by name,
/// or a registered component through its vtable. Not the parent link, which
/// needs every entity of a load spawned first.
fn apply_component(
    world: &mut World,
    registry: &orrin_registry::Registry,
    entity: Entity,
    id: &ComponentId,
    value: &Value,
) -> Result<(), String> {
    if *id == MESH || *id == MATERIAL {
        return apply_asset_ref(world, entity, id, value);
    }
    apply_registered(registry.get(id), world, entity, id, value)
}

/// [`apply_component`] for a registered type, with its vtable already looked up.
fn apply_registered(
    vtable: Option<&orrin_registry::ComponentVtable>,
    world: &mut World,
    entity: Entity,
    id: &ComponentId,
    value: &Value,
) -> Result<(), String> {
    match vtable {
        Some(vtable) if vtable.script.is_some() && !vtable.has(world, entity) => {
            wait_for_host(world, entity, id, value);
            Ok(())
        }
        Some(vtable) => vtable
            .write(world, entity, value)
            .map_err(|error| error.to_string()),
        None => Err("no component type is registered under this id".to_owned()),
    }
}

/// Park a Behaviour's value until the script host can create it. See
/// [`PendingScripts`].
fn wait_for_host(world: &mut World, entity: Entity, id: &ComponentId, value: &Value) {
    if let Some(mut pending) = world.get_mut::<PendingScripts>(entity) {
        match pending.0.iter_mut().find(|(waiting, _)| waiting == id) {
            Some((_, slot)) => *slot = value.clone(),
            None => pending.0.push((id.clone(), value.clone())),
        }
        return;
    }
    world.insert(entity, PendingScripts(vec![(id.clone(), value.clone())]));
}

/// [`apply_component`] for an entity already live, where a parent link resolves
/// against the whole world and an unregistered component replaces its entry in
/// the [`UnknownComponents`].
pub(super) fn write_component(
    world: &mut World,
    registry: &orrin_registry::Registry,
    entity: Entity,
    id: &ComponentId,
    value: &Value,
) -> Result<(), String> {
    if *id == PARENT {
        let Value::Entity(parent_id) = value else {
            return Err(format!(
                "expected an entity reference, found {}",
                value.type_name()
            ));
        };
        let parent = world
            .entities()
            .find(|&e| world.get::<EntityId>(e).is_some_and(|id| *id == *parent_id))
            .ok_or_else(|| format!("no entity has the id {parent_id}"))?;
        return super::reparent(world, entity, Some(parent), false).map_err(|e| e.to_string());
    }
    if *id != MESH && *id != MATERIAL && registry.get(id).is_none() {
        remove_component(world, registry, entity, id);
        keep_unapplied(world, entity, (id.clone(), value.clone()));
        return Ok(());
    }
    apply_component(world, registry, entity, id, value)
}

/// One component of a live entity in its file form: what [`write_component`]
/// takes and [`scene_entity`] would write for it.
pub(super) fn read_component(
    world: &World,
    registry: &orrin_registry::Registry,
    entity: Entity,
    id: &ComponentId,
) -> Option<Value> {
    if *id == PARENT {
        return parent_ref(world, entity).map(|(_, value)| value);
    }
    if *id == MESH || *id == MATERIAL {
        return asset_refs(world, entity)
            .into_iter()
            .find_map(|(component, value)| (component == *id).then_some(value));
    }
    if let Some(value) = registry
        .get(id)
        .and_then(|vtable| vtable.read(world, entity))
    {
        return Some(value);
    }
    let parked = |list: &[(ComponentId, Value)]| {
        list.iter()
            .find_map(|(component, value)| (component == id).then(|| value.clone()))
    };
    world
        .get::<PendingScripts>(entity)
        .and_then(|pending| parked(&pending.0))
        .or_else(|| {
            world
                .get::<UnknownComponents>(entity)
                .and_then(|unknown| parked(&unknown.0))
        })
}

/// Take one component, in its file form, off a live entity.
pub(super) fn remove_component(
    world: &mut World,
    registry: &orrin_registry::Registry,
    entity: Entity,
    id: &ComponentId,
) {
    if let Some(mut pending) = world.get_mut::<PendingScripts>(entity) {
        pending.0.retain(|(waiting, _)| waiting != id);
    }
    if *id == PARENT {
        let _ = super::reparent(world, entity, None, false);
    } else if *id == MESH {
        world.remove::<MeshHandle>(entity);
    } else if *id == MATERIAL {
        world.remove::<MaterialHandle>(entity);
    } else if let Some(vtable) = registry.get(id) {
        vtable.remove(world, entity);
    } else if let Some(mut unknown) = world.get_mut::<UnknownComponents>(entity) {
        unknown.0.retain(|(kept, _)| kept != id);
    }
}

/// Preserve a component the load could not apply, so a later save does not drop
/// it. Appends when the entity already collected others during the first pass.
pub(crate) fn keep_unapplied(world: &mut World, entity: Entity, component: (ComponentId, Value)) {
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
        let text =
            format!("orrin-scene 2 min-reader 2\n\nentity {id}\n  orrin.mesh = \"teapot\"\n");

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
            "orrin-scene 2 min-reader 2\n\nentity {id}\n  game.enemy\n    aggression = 0.75\n    name = \"Boss\"\n"
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
        let text = format!("orrin-scene 2 min-reader 2\n\nentity {id} \"Cube\"\n  orrin.tag = 7\n");

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

    /// Instances of one type are applied together, so a rejection must still
    /// land on the entity it belongs to and leave the type's other instances
    /// applied.
    #[test]
    fn a_type_shared_by_several_entities_reports_and_keeps_per_entity() {
        let registry = registry();
        let [a, b, c] = [1, 2, 3].map(|n| {
            format!("00000000-0000-4000-8000-{n:012}")
                .parse::<EntityId>()
                .unwrap()
        });
        let text = format!(
            "orrin-scene 2 min-reader 2\n\n\
             entity {a} \"A\"\n  orrin.tag = 7\n\n\
             entity {b} \"B\"\n  orrin.tag = \"ok\"\n\n\
             entity {c} \"C\"\n  game.enemy = 1\n  orrin.tag = 8\n"
        );

        let mut world = World::new();
        let issues = load(&text, &mut world, &registry).unwrap();
        let mut reported: Vec<(EntityId, &str)> = issues
            .iter()
            .map(|issue| (issue.entity, issue.component.as_str()))
            .collect();
        reported.sort();
        assert_eq!(
            reported,
            [(a, "orrin.tag"), (c, "game.enemy"), (c, "orrin.tag")]
        );

        let tagged: Vec<String> = world
            .entities()
            .filter_map(|entity| world.get::<Tag>(entity).map(|tag| tag.0.clone()))
            .collect();
        assert_eq!(tagged, ["ok"]);
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
        assert_eq!(text, "orrin-scene 2 min-reader 2\n");

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

    /// A script component's bridge with no managed side: it has nothing live
    /// to write into, which is the state every Behaviour is in while a scene
    /// loads.
    struct NoHost;

    impl orrin_registry::ScriptBridge for NoHost {
        fn has(&self, _: &World, _: Entity, _: &str) -> bool {
            false
        }
        fn read(&self, _: &World, _: Entity, _: &str) -> Option<Value> {
            None
        }
        fn write(
            &self,
            _: &mut World,
            _: Entity,
            _: &str,
            _: &Value,
        ) -> Result<(), orrin_registry::ValueError> {
            Err(orrin_registry::ValueError::invalid(
                "a live script component",
                "nothing",
            ))
        }
        fn remove(&self, _: &mut World, _: Entity, _: &str) {}
        fn default(&self, _: &str) -> Value {
            Value::Struct(Vec::new())
        }
    }

    /// A Behaviour cannot be written until the script host has created it, and
    /// a load has no host. It waits on the entity instead of being reported
    /// as data nothing could apply — and is saved back while it waits.
    #[test]
    fn a_behaviour_in_a_scene_waits_for_the_script_host() {
        let mut registry = registry();
        registry.register_script(
            ComponentId::new("demo.spinner"),
            "Spinner",
            "DemoGame.Spinner, DemoGame",
            std::sync::Arc::new(NoHost),
        );
        let id = EntityId::new();
        let text = format!(
            "orrin-scene 2 min-reader 2\n\nentity {id}\n  demo.spinner\n    DegreesPerSecond = 45.0\n"
        );

        let mut world = World::new();
        let issues = load(&text, &mut world, &registry).unwrap();
        assert!(issues.is_empty(), "{issues:?}");
        let entity = world.entities().next().unwrap();
        let pending = world
            .get::<PendingScripts>(entity)
            .expect("waiting to attach");
        assert_eq!(pending.0.len(), 1);
        assert!(world.get::<UnknownComponents>(entity).is_none());
        drop(pending);
        assert_eq!(save(&mut world, &registry), text);
    }
}
