//! Turning a world into a scene file and back.
//!
//! The registry supplies every component's read and write; this module supplies
//! only the rules that need a world — assigning identities, spawning, and what
//! to do with data nothing claims.

use orrin_ecs::{Entity, World};
use orrin_registry::{
    ComponentId, EntityId, ParseError, SceneDocument, SceneEntity, Value, write_document,
};

use super::registry::{MATERIAL, MESH};
use super::{Assets, MaterialHandle, MeshHandle, UnknownComponents};

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

    let entities = world
        .entities()
        .map(|entity| {
            let id = *world.get::<EntityId>(entity).expect("just assigned");

            let mut components: Vec<(ComponentId, orrin_registry::Value)> = registry
                .components()
                .filter_map(|c| (c.read)(world, entity).map(|value| (c.id.clone(), value)))
                .collect();

            components.extend(asset_refs(world, entity));

            if let Some(unknown) = world.get::<UnknownComponents>(entity) {
                components.extend(unknown.0.iter().cloned());
            }

            SceneEntity { id, components }
        })
        .collect();

    SceneDocument { entities }
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
/// Single-pass, and that is a consequence of components holding [`EntityId`]
/// rather than raw entity handles: a reference in the file is already the value
/// the component should hold, so there is nothing to patch up afterwards. Slot
/// indices would have forced a spawn-everything-then-fix-references pass.
pub fn instantiate(
    document: &SceneDocument,
    world: &mut World,
    registry: &orrin_registry::Registry,
) -> Vec<LoadIssue> {
    let mut issues = Vec::new();

    for scene_entity in &document.entities {
        let entity = world.spawn();
        world.insert(entity, scene_entity.id);

        let mut unknown = Vec::new();
        for (id, value) in &scene_entity.components {
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

    issues
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
}
