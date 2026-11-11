//! Several Behaviours on one entity, through the engine's real script tick: each
//! gets its own lifecycle in attach order, an entity's events reach all of
//! them, a fault stays with the one that threw, a second of one type is
//! refused, and a despawn or a reload treats each exactly once.
//!
//! The probes are `Orrin.Tests.*Probe`, which log every hook as
//! `{type}:{hook}`; the log is read back through the engine's own stream.
//!
//! Needs both C# assemblies built (`dotnet build scripting/Orrin` and
//! `dotnet build scripting/DemoGame`) and skips without them, as
//! `scripted_scene` does. One test, because CoreCLR boots once per process.

#![cfg(feature = "scripting")]

use std::path::{Path, PathBuf};

use glam::Vec3;
use orrin_core::logging::{ConsoleQueue, LogConfig, Stream};
use orrin_core::physics::{CollisionEvent, CollisionEventKind, CollisionState};
use orrin_core::scene::{LogBuffer, LogLevel, despawn_recursive, register_components};
use orrin_core::scripting::Scripting;
use orrin_ecs::{Entity, World};
use orrin_registry::Registry;

const FIRST: &str = "Orrin.Tests.FirstProbe, Orrin";
const SECOND: &str = "Orrin.Tests.SecondProbe, Orrin";
const FAULTING: &str = "Orrin.Tests.FaultingProbe, Orrin";
/// `FIRST` as `typeof(T).AssemblyQualifiedName` spells it, which is what
/// `World.AddScript<T>` sends.
const FIRST_QUALIFIED: &str =
    "Orrin.Tests.FirstProbe, Orrin, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null";

const FRAME: f32 = 1.0 / 60.0;

fn assemblies() -> Option<(PathBuf, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bindings = Scripting::find_bindings_dir(&root.join("scripting/Orrin"))?;
    let game = Scripting::find_game_assembly(&root.join("scripting/DemoGame"), "DemoGame")?;
    Some((bindings, game))
}

struct Log(ConsoleQueue);

impl Log {
    /// Everything logged since the last call, as `(level, message)`.
    fn take(&self) -> Vec<(LogLevel, String)> {
        let mut buffer = LogBuffer::default();
        self.0.drain_into(&mut buffer, 0);
        buffer
            .iter()
            .map(|entry| (entry.level, entry.message.clone()))
            .collect()
    }

    /// The probes' own lines since the last call, in order.
    fn probes(&self) -> Vec<String> {
        self.take()
            .into_iter()
            .map(|(_, message)| message)
            .filter(|message| message.contains("Probe:"))
            .collect()
    }
}

struct Game<'a> {
    world: World,
    registry: Registry,
    scripting: &'a Scripting,
}

impl Game<'_> {
    fn tick(&mut self) {
        self.scripting.tick(&mut self.world, &self.registry, FRAME);
    }

    fn collide(&mut self, a: Entity, b: Entity, trigger: bool) {
        self.world
            .resource_mut::<CollisionState>()
            .events
            .push(CollisionEvent {
                kind: CollisionEventKind::Enter,
                a,
                b,
                point: Vec3::ZERO,
                normal: Vec3::Y,
                relative_velocity: Vec3::ZERO,
                trigger,
            });
    }
}

#[test]
fn an_entity_carries_several_behaviours() {
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
    let mut world = World::new();
    world.insert_resource(CollisionState::default());
    let mut game = Game {
        world,
        registry,
        scripting: &scripting,
    };
    log.take();

    both_run_in_attach_order(&mut game, &log);
    a_fault_stays_with_the_behaviour_that_threw(&mut game, &log);
}

fn both_run_in_attach_order(game: &mut Game, log: &Log) {
    let entity = game.world.spawn();
    game.scripting.attach(&mut game.world, entity, FIRST);
    game.scripting.attach(&mut game.world, entity, SECOND);
    assert_eq!(log.probes(), ["FirstProbe:ctor", "SecondProbe:ctor"]);

    game.tick();
    assert_eq!(
        log.probes(),
        [
            "FirstProbe:OnEnable",
            "FirstProbe:OnStart",
            "SecondProbe:OnEnable",
            "SecondProbe:OnStart",
            "FirstProbe:OnUpdate",
            "SecondProbe:OnUpdate",
        ]
    );

    // A collision and a trigger on the entity each reach both, in attach
    // order, before either updates.
    let other = game.world.spawn();
    game.collide(entity, other, false);
    game.collide(other, entity, true);
    game.tick();
    assert_eq!(
        log.probes(),
        [
            "FirstProbe:OnCollisionEnter",
            "SecondProbe:OnCollisionEnter",
            "FirstProbe:OnTriggerEnter",
            "SecondProbe:OnTriggerEnter",
            "FirstProbe:OnUpdate",
            "SecondProbe:OnUpdate",
        ]
    );

    // A second of one type is refused before it is created, whichever way its
    // name is spelt, and the one already there carries on alone.
    game.scripting.attach(&mut game.world, entity, FIRST);
    game.scripting
        .attach(&mut game.world, entity, FIRST_QUALIFIED);
    let lines = log.take();
    let refusals: Vec<&String> = lines
        .iter()
        .filter(|(level, message)| {
            *level == LogLevel::Warning
                && message.contains("Orrin.Tests.FirstProbe, Orrin")
                && message.contains(&format!("entity {}", entity.index()))
        })
        .map(|(_, message)| message)
        .collect();
    assert_eq!(refusals.len(), 2, "{lines:?}");
    assert!(
        !lines.iter().any(|(_, message)| message.contains("Probe:")),
        "a refused attach still ran managed code: {lines:?}"
    );
    game.tick();
    assert_eq!(
        log.probes(),
        ["FirstProbe:OnUpdate", "SecondProbe:OnUpdate"]
    );

    // A reload captures both before destroying either, then re-creates both,
    // in order, where their lifecycles left off: enabled again, never started
    // twice. The first two constructors are the capture's throwaway instances,
    // which it reads each type's defaults from (see `BehaviourState`).
    game.scripting.reload(&mut game.world);
    assert_eq!(
        log.probes(),
        [
            "FirstProbe:ctor",
            "SecondProbe:ctor",
            "FirstProbe:OnDisable",
            "FirstProbe:OnDestroy",
            "SecondProbe:OnDisable",
            "SecondProbe:OnDestroy",
            "FirstProbe:ctor",
            "SecondProbe:ctor",
        ]
    );
    game.tick();
    assert_eq!(
        log.probes(),
        [
            "FirstProbe:OnEnable",
            "SecondProbe:OnEnable",
            "FirstProbe:OnUpdate",
            "SecondProbe:OnUpdate",
        ]
    );

    // A despawn destroys each exactly once.
    despawn_recursive(&mut game.world, entity);
    assert_eq!(
        log.probes(),
        [
            "FirstProbe:OnDisable",
            "FirstProbe:OnDestroy",
            "SecondProbe:OnDisable",
            "SecondProbe:OnDestroy",
        ]
    );
    game.tick();
    assert!(log.probes().is_empty());
}

fn a_fault_stays_with_the_behaviour_that_threw(game: &mut Game, log: &Log) {
    let entity = game.world.spawn();
    game.scripting.attach(&mut game.world, entity, FAULTING);
    game.scripting.attach(&mut game.world, entity, SECOND);
    log.take();

    game.tick();
    assert_eq!(
        log.probes(),
        [
            "FaultingProbe:OnEnable",
            "FaultingProbe:OnStart",
            "SecondProbe:OnEnable",
            "SecondProbe:OnStart",
            "FaultingProbe:OnUpdate",
            "SecondProbe:OnUpdate",
        ]
    );
    game.tick();
    assert_eq!(log.probes(), ["SecondProbe:OnUpdate"]);

    despawn_recursive(&mut game.world, entity);
    assert_eq!(
        log.probes(),
        [
            "FaultingProbe:OnDisable",
            "FaultingProbe:OnDestroy",
            "SecondProbe:OnDisable",
            "SecondProbe:OnDestroy",
        ]
    );
}
