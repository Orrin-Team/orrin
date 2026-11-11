//! A Behaviour's entity references, `Vector2`s, `Color`s and lists survive a
//! save and load, a prefab instantiation and a hot reload, through the real
//! script host.
//!
//! The probe is `Orrin.Tests.FieldsProbe`. No game assembly lists it, so it is
//! registered by hand under `test.fields` — after the boot, and again after the
//! reload, as `register_components` would be.
//!
//! Needs both C# assemblies built (`dotnet build scripting/Orrin` and
//! `dotnet build scripting/DemoGame`) and skips without them, as
//! `scripted_scene` does. One test, because CoreCLR boots once per process.

#![cfg(feature = "scripting")]

use std::path::{Path, PathBuf};

use orrin_core::logging::{ConsoleQueue, LogConfig, Stream};
use orrin_core::scene::prefab::{self, Prefab};
use orrin_core::scene::{
    LogBuffer, LogLevel, Name, Parent, Prefabs, canonical_type_name, load, register_components,
    save,
};
use orrin_core::scripting::{ReloadOutcome, Scripting};
use orrin_ecs::{Entity, World};
use orrin_registry::{AssetId, ComponentId, EntityId, Registry, Value};

const PROBE: &str = "Orrin.Tests.FieldsProbe, Orrin";
const FIELDS: ComponentId = ComponentId::new("test.fields");

fn assemblies() -> Option<(PathBuf, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bindings = Scripting::find_bindings_dir(&root.join("scripting/Orrin"))?;
    let game = Scripting::find_game_assembly(&root.join("scripting/DemoGame"), "DemoGame")?;
    Some((bindings, game))
}

fn register_probe(registry: &mut Registry) {
    registry.register_script(
        FIELDS,
        "Fields",
        canonical_type_name(PROBE),
        Vec::new(),
        orrin_core::scripting::bridge(),
    );
}

fn id(n: u32) -> EntityId {
    format!("00000000-0000-4000-8000-{n:012}").parse().unwrap()
}

fn named(world: &World, name: &str) -> Entity {
    world
        .entities()
        .find(|&e| world.get::<Name>(e).is_some_and(|n| n.0 == name))
        .unwrap_or_else(|| panic!("no entity named {name}"))
}

fn field(world: &World, registry: &Registry, entity: Entity, name: &str) -> Value {
    registry
        .get(&FIELDS)
        .and_then(|vtable| vtable.read(world, entity))
        .and_then(|bag| bag.field(name).cloned())
        .unwrap_or_else(|| panic!("test.fields.{name} is not readable"))
}

struct Log(ConsoleQueue);

impl Log {
    fn warnings(&self) -> Vec<String> {
        let mut buffer = LogBuffer::default();
        self.0.drain_into(&mut buffer, 0);
        buffer
            .iter()
            .filter(|entry| entry.level == LogLevel::Warning)
            .map(|entry| entry.message.clone())
            .collect()
    }
}

#[test]
fn a_behaviours_new_field_types_survive_save_load_prefabs_and_reload() {
    let Some((bindings, game)) = assemblies() else {
        eprintln!("skipping: build scripting/Orrin and scripting/DemoGame first");
        return;
    };
    let (stream, console, _) = Stream::new(LogConfig {
        stderr: false,
        ..LogConfig::editor()
    });
    let _subscriber = tracing::subscriber::set_default(stream);
    let log = Log(console);

    let scripting = Scripting::boot(&bindings, &game).expect("CoreCLR failed to boot");
    let mut registry = Registry::new();
    register_components(&mut registry);
    register_probe(&mut registry);

    a_scene_round_trips_its_references_and_values(&scripting, &registry, &log);
    a_prefab_instance_points_at_its_own_members(&scripting, &registry);
    an_unresolvable_reference_loads_as_null_and_says_so_once(&scripting, &registry, &log);
    a_reload_carries_every_new_field_type(&scripting, &mut registry);
}

fn a_scene_round_trips_its_references_and_values(
    scripting: &Scripting,
    registry: &Registry,
    log: &Log,
) {
    let (door, lever) = (id(1), id(2));
    let text = format!(
        "orrin-scene 4 min-reader 4\n\n\
         entity {door} \"Door\"\n\n\
         entity {lever} \"Lever\"\n  test.fields\n    Offset = (3.0, 4.0)\n    Others\n      \
         [0] = entity({door})\n      [1] = entity({lever})\n    Steps\n      [0] = 5\n      \
         [1] = 6\n    Target = entity({door})\n    Tint = rgba(1.0, 0.5, 0.0, 0.25)\n"
    );
    let mut world = World::new();
    assert!(load(&text, &mut world, registry).unwrap().is_empty());
    scripting.attach_pending(&mut world, registry);
    let warnings = log.warnings();
    assert!(warnings.is_empty(), "{warnings:?}");

    let at = named(&world, "Lever");
    assert_eq!(field(&world, registry, at, "Target"), Value::Entity(door));
    assert_eq!(
        field(&world, registry, at, "Others"),
        Value::List(vec![Value::Entity(door), Value::Entity(lever)])
    );
    assert_eq!(
        field(&world, registry, at, "Offset"),
        Value::Vec2(glam::Vec2::new(3.0, 4.0))
    );
    assert_eq!(
        field(&world, registry, at, "Tint"),
        Value::Color([1.0, 0.5, 0.0, 0.25])
    );
    assert_eq!(
        field(&world, registry, at, "Steps"),
        Value::List(vec![Value::I32(5), Value::I32(6)])
    );
    assert_eq!(save(&mut world, registry), text);

    // A second load into a world whose slots carry other generations: the
    // reference follows the id, not the handle it had last time.
    let mut again = World::new();
    let padding: Vec<Entity> = (0..3).map(|_| again.spawn()).collect();
    for entity in padding {
        again.despawn(entity);
    }
    assert!(load(&text, &mut again, registry).unwrap().is_empty());
    scripting.attach_pending(&mut again, registry);
    let lever_again = named(&again, "Lever");
    assert_eq!(
        field(&again, registry, lever_again, "Target"),
        Value::Entity(door)
    );
    assert_eq!(save(&mut again, registry), text);
}

/// The prefab's lever points at the prefab's door. Each instance's lever must
/// point at *its own* door, which is the remap parent links already get.
fn a_prefab_instance_points_at_its_own_members(scripting: &Scripting, registry: &Registry) {
    let (lever, door) = (id(10), id(11));
    let source = format!(
        "orrin-scene 4 min-reader 4\n\n\
         entity {lever} \"Lever\"\n  test.fields\n    Target = entity({door})\n\n\
         entity {door} \"Door\"\n  orrin.parent = entity({lever})\n"
    );
    let gate = AssetId::new();
    let mut prefabs = Prefabs::default();
    prefabs.insert(Prefab::parse(gate, "gate.oprefab", &source).unwrap());
    let mut world = World::new();
    world.insert_resource(prefabs);

    let mut doors = Vec::new();
    for _ in 0..2 {
        let (root, issues) = prefab::instantiate(&mut world, registry, gate).unwrap();
        assert!(issues.is_empty(), "{issues:?}");
        scripting.attach_pending(&mut world, registry);
        let own_door = world
            .entities()
            .find(|&e| world.get::<Parent>(e).is_some_and(|p| p.get() == root))
            .expect("the instance has its door");
        let own_id = *world.get::<EntityId>(own_door).unwrap();
        assert_eq!(
            field(&world, registry, root, "Target"),
            Value::Entity(own_id)
        );
        doors.push(own_id);
    }
    assert_ne!(doors[0], doors[1]);
    assert!(
        !doors.contains(&door),
        "an instance pointed at the prefab's own entity"
    );
}

/// A reference nothing in the scene answers to loads as null — never as
/// whichever entity happens to hold the slot — and the log names the field.
/// The same for a handle that was stale when the scene was saved.
fn an_unresolvable_reference_loads_as_null_and_says_so_once(
    scripting: &Scripting,
    registry: &Registry,
    log: &Log,
) {
    let elsewhere = EntityId::new();
    let text = format!(
        "orrin-scene 4 min-reader 4\n\n\
         entity {} \"Lever\"\n  test.fields\n    Target = entity({elsewhere})\n",
        id(20)
    );
    let mut world = World::new();
    assert!(load(&text, &mut world, registry).unwrap().is_empty());
    scripting.attach_pending(&mut world, registry);
    let at = named(&world, "Lever");
    assert_eq!(
        field(&world, registry, at, "Target"),
        Value::Entity(EntityId::NIL)
    );
    let warnings = log.warnings();
    let about: Vec<&String> = warnings
        .iter()
        .filter(|line| line.contains("Target") && line.contains(&elsewhere.to_string()))
        .collect();
    assert_eq!(about.len(), 1, "{warnings:?}");
    assert!(about[0].contains("FieldsProbe"), "{}", about[0]);

    // Stale at save: the field points at an entity that is then despawned.
    let gone = world.spawn();
    let vtable = registry.get(&FIELDS).unwrap();
    let mut bag = vtable.read(&world, at).unwrap();
    *bag.field_mut("Target").unwrap() =
        Value::Entity(orrin_core::scene::entity_ref::id_of(&world, gone));
    vtable.write(&mut world, at, &bag).unwrap();
    world.despawn(gone);
    let saved = save(&mut world, registry);

    let mut again = World::new();
    assert!(load(&saved, &mut again, registry).unwrap().is_empty());
    log.warnings();
    scripting.attach_pending(&mut again, registry);
    let lever = named(&again, "Lever");
    assert_eq!(
        field(&again, registry, lever, "Target"),
        Value::Entity(EntityId::NIL)
    );
    let warnings = log.warnings();
    assert_eq!(
        warnings
            .iter()
            .filter(|line| line.contains("Target"))
            .count(),
        1,
        "{warnings:?}"
    );
}

/// Values a hot reload has to carry: deviations from the constructor's, a live
/// handle as it is, and a list.
fn a_reload_carries_every_new_field_type(scripting: &Scripting, registry: &mut Registry) {
    let mut world = World::new();
    let target = world.spawn();
    let at = world.spawn();
    scripting.attach(&mut world, at, PROBE);
    let vtable = registry.get(&FIELDS).unwrap();
    let mut bag = vtable.read(&world, at).unwrap();
    let target_id = orrin_core::scene::entity_ref::id_of(&world, target);
    *bag.field_mut("Target").unwrap() = Value::Entity(target_id);
    *bag.field_mut("Offset").unwrap() = Value::Vec2(glam::Vec2::new(-1.0, 0.5));
    *bag.field_mut("Tint").unwrap() = Value::Color([0.25, 0.5, 0.75, 1.0]);
    *bag.field_mut("Steps").unwrap() = Value::List(vec![Value::I32(9)]);
    *bag.field_mut("Others").unwrap() = Value::List(vec![Value::Entity(target_id)]);
    vtable.write(&mut world, at, &bag).unwrap();

    let outcome = scripting.reload(&mut world);
    assert!(
        matches!(
            outcome,
            ReloadOutcome::Swapped {
                restored: 1,
                lost: 0,
                ..
            }
        ),
        "{outcome}"
    );
    registry.clear_game();
    scripting.register_components(registry);
    register_probe(registry);

    let after = registry.get(&FIELDS).unwrap().read(&world, at).unwrap();
    assert_eq!(after, bag);
}
