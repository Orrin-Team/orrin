//! A Behaviour saved in a scene or a prefab comes back attached, with its
//! fields, once the script host picks it up.
//!
//! Needs both C# assemblies built (`dotnet build scripting/Orrin` and
//! `dotnet build scripting/DemoGame`) and skips without them, as
//! orrin-script's lifecycle test does. One test, because CoreCLR boots once
//! per process.

#![cfg(feature = "scripting")]

use std::path::{Path, PathBuf};

use orrin_core::scene::prefab::{self, Prefab};
use orrin_core::scene::{Name, Prefabs, Scripts, load, register_components, save};
use orrin_core::scripting::{ReloadOutcome, Scripting};
use orrin_ecs::{Entity, World};
use orrin_registry::{AssetId, ComponentId, EntityId, Registry, Value};

fn assemblies() -> Option<(PathBuf, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bindings = Scripting::find_bindings_dir(&root.join("scripting/Orrin"))?;
    let game = Scripting::find_game_assembly(&root.join("scripting/DemoGame"), "DemoGame")?;
    Some((bindings, game))
}

fn spin(world: &World, registry: &Registry, entity: Entity) -> Option<Value> {
    registry
        .get(&ComponentId::new("demo.spinner"))?
        .read(world, entity)?
        .field("DegreesPerSecond")
        .cloned()
}

fn named(world: &World, name: &str) -> Entity {
    world
        .entities()
        .find(|&e| world.get::<Name>(e).is_some_and(|n| n.0 == name))
        .unwrap_or_else(|| panic!("no entity named {name}"))
}

#[test]
fn a_saved_behaviour_comes_back_attached_with_its_fields() {
    let Some((bindings, game)) = assemblies() else {
        eprintln!("skipping: build scripting/Orrin and scripting/DemoGame first");
        return;
    };
    let scripting = Scripting::boot(&bindings, &game).expect("CoreCLR failed to boot");
    let mut registry = Registry::new();
    register_components(&mut registry);
    assert!(scripting.register_components(&mut registry) > 0);

    // From a scene.
    let id = EntityId::new();
    let text = format!(
        "orrin-scene 3 min-reader 3\n\nentity {id} \"Fan\"\n  demo.spinner\n    \
         DegreesPerSecond = 45.0\n"
    );
    let mut world = World::new();
    let issues = load(&text, &mut world, &registry).unwrap();
    assert!(issues.is_empty(), "{issues:?}");
    scripting.attach_pending(&mut world, &registry);

    let fan = named(&world, "Fan");
    assert_eq!(
        world.get::<Scripts>(fan).map(|scripts| scripts.len()),
        Some(1)
    );
    assert_eq!(spin(&world, &registry, fan), Some(Value::F32(45.0)));
    assert_eq!(save(&mut world, &registry), text);

    // From a prefab.
    let mut prefabs = Prefabs::default();
    let windmill = AssetId::new();
    let source = format!(
        "orrin-scene 1\n\nentity {}\n  demo.spinner\n    DegreesPerSecond = 12.0\n  \
         orrin.name = \"Windmill\"\n",
        EntityId::new()
    );
    prefabs.insert(Prefab::parse(windmill, "windmill.oprefab", &source).unwrap());
    world.insert_resource(prefabs);
    let (root, issues) = prefab::instantiate(&mut world, &registry, windmill).unwrap();
    assert!(issues.is_empty(), "{issues:?}");
    scripting.attach_pending(&mut world, &registry);
    assert_eq!(spin(&world, &registry, root), Some(Value::F32(12.0)));

    // Every handle into the game assembly gone before the reload below, so the
    // retired context can unload.
    drop(world);
    two_on_one_entity_save_load_and_reload(&scripting, &mut registry);
}

fn field(world: &World, registry: &Registry, entity: Entity, id: &str, name: &str) -> Value {
    registry
        .get(&ComponentId::owned(id.to_owned()))
        .and_then(|vtable| vtable.read(world, entity))
        .and_then(|bag| bag.field(name).cloned())
        .unwrap_or_else(|| panic!("{id}.{name} is not readable on entity {}", entity.index()))
}

/// The demo's moving target is a `Target` and a `Shuttle` on one entity: both
/// come back from a scene with their fields, save back as they loaded, and
/// survive a hot reload with the state they had.
fn two_on_one_entity_save_load_and_reload(scripting: &Scripting, registry: &mut Registry) {
    let id = EntityId::new();
    let text = format!(
        "orrin-scene 3 min-reader 3\n\nentity {id} \"Shuttle\"\n  demo.target\n    \
         Hits = 3\n  demo.shuttle\n    Distance = 9.0\n"
    );
    let mut world = World::new();
    let issues = load(&text, &mut world, registry).unwrap();
    assert!(issues.is_empty(), "{issues:?}");
    scripting.attach_pending(&mut world, registry);

    let board = named(&world, "Shuttle");
    assert_eq!(
        field(&world, registry, board, "demo.target", "Hits"),
        Value::I32(3)
    );
    assert_eq!(
        field(&world, registry, board, "demo.shuttle", "Distance"),
        Value::F32(9.0)
    );
    let saved = save(&mut world, registry);
    let mut again = World::new();
    assert!(load(&saved, &mut again, registry).unwrap().is_empty());
    scripting.attach_pending(&mut again, registry);
    assert_eq!(save(&mut again, registry), saved);
    drop(again);

    // The reload carries both instances' fields across the swap.
    let outcome = scripting.reload(&mut world);
    assert!(
        matches!(
            outcome,
            ReloadOutcome::Swapped {
                restored: 2,
                lost: 0,
                ..
            }
        ),
        "{outcome}"
    );
    registry.clear_game();
    scripting.register_components(registry);
    assert_eq!(
        field(&world, registry, board, "demo.target", "Hits"),
        Value::I32(3)
    );
    assert_eq!(
        field(&world, registry, board, "demo.shuttle", "Distance"),
        Value::F32(9.0)
    );
    assert_eq!(save(&mut world, registry), saved);
}
