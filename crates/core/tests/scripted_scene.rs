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
use orrin_core::scene::{Name, Prefabs, ScriptComponent, load, register_components, save};
use orrin_core::scripting::Scripting;
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
        "orrin-scene 2 min-reader 2\n\nentity {id} \"Fan\"\n  demo.spinner\n    \
         DegreesPerSecond = 45.0\n"
    );
    let mut world = World::new();
    let issues = load(&text, &mut world, &registry).unwrap();
    assert!(issues.is_empty(), "{issues:?}");
    scripting.attach_pending(&mut world, &registry);

    let fan = named(&world, "Fan");
    assert!(world.get::<ScriptComponent>(fan).is_some());
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
    prefabs.insert(Prefab::parse(windmill, "windmill.fprefab", &source).unwrap());
    world.insert_resource(prefabs);
    let (root, issues) = prefab::instantiate(&mut world, &registry, windmill).unwrap();
    assert!(issues.is_empty(), "{issues:?}");
    scripting.attach_pending(&mut world, &registry);
    assert_eq!(spin(&world, &registry, root), Some(Value::F32(12.0)));
}
