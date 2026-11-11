//! What a Behaviour hears from the physics step, through the engine's real
//! physics phase and script tick: `OnFixedUpdate` once before every step, the
//! Stay callbacks once a step while a pair touches, the closing velocity at a
//! contact, and no call at all for a hook its type does not override. Then the
//! per-type timing the Scripts panel shows.
//!
//! The probes are `Orrin.Tests.*Probe` in `StepProbes.cs`, which log each hook
//! they override; the log is read back through the engine's own stream.
//!
//! Needs both C# assemblies built (`dotnet build scripting/Orrin` and
//! `dotnet build scripting/DemoGame`) and skips without them, as
//! `scripted_scene` does. One test, because CoreCLR boots once per process.

#![cfg(feature = "scripting")]

use std::path::{Path, PathBuf};

use glam::Vec3;
use orrin_core::logging::{ConsoleQueue, LogConfig, Stream};
use orrin_core::physics::{self, CollisionEventKind, CollisionState, PhysicsState, STEP};
use orrin_core::scene::{
    Collider, ColliderShape, LocalTransform, LogBuffer, LogLevel, RigidBody, Scripts, Transform,
    propagate_transforms, register_components,
};
use orrin_core::scripting::{ScriptTimings, Scripting};
use orrin_ecs::{Entity, World};
use orrin_registry::Registry;

const STEPPED: &str = "Orrin.Tests.SteppedProbe, Orrin";
const SECOND_STEPPED: &str = "Orrin.Tests.SecondSteppedProbe, Orrin";
const THROWING_STEP: &str = "Orrin.Tests.ThrowingStepProbe, Orrin";
const PUSHING: &str = "Orrin.Tests.PushingProbe, Orrin";
const CONTACT: &str = "Orrin.Tests.ContactProbe, Orrin";
const THROWING_STAY: &str = "Orrin.Tests.ThrowingStayProbe, Orrin";
const IDLE: &str = "Orrin.Tests.IdleProbe, Orrin";
const SLOW: &str = "Orrin.Tests.SlowProbe, Orrin";

fn assemblies() -> Option<(PathBuf, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bindings = Scripting::find_bindings_dir(&root.join("scripting/Orrin"))?;
    let game = Scripting::find_game_assembly(&root.join("scripting/DemoGame"), "DemoGame")?;
    Some((bindings, game))
}

struct Log(ConsoleQueue);

impl Log {
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

/// `{type}:{hook}`, without whatever the probe logged after it.
fn hook(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}

fn hooks(lines: &[String]) -> Vec<&str> {
    lines.iter().map(|line| hook(line)).collect()
}

/// The numbers a probe logged after its hook.
fn numbers(line: &str) -> Vec<f32> {
    line.split_whitespace()
        .skip(1)
        .map(|word| word.parse().expect("a probe logs numbers"))
        .collect()
}

fn cube(half: f32) -> ColliderShape {
    ColliderShape::Box {
        half_extents: Vec3::splat(half),
    }
}

fn frictionless() -> RigidBody {
    RigidBody {
        mass: 1.0,
        gravity_scale: 0.0,
        linear_damping: 0.0,
        angular_damping: 0.0,
        ..RigidBody::default()
    }
}

struct Game<'a> {
    world: World,
    registry: &'a Registry,
    scripting: &'a Scripting,
}

impl<'a> Game<'a> {
    fn new(scripting: &'a Scripting, registry: &'a Registry) -> Self {
        let mut world = World::new();
        world.insert_resource(CollisionState::default());
        world.insert_resource(PhysicsState::default());
        world.insert_resource(LogBuffer::default());
        world.insert_resource(ScriptTimings::default());
        Self {
            world,
            registry,
            scripting,
        }
    }

    fn body(&mut self, at: Vec3, shape: ColliderShape, body: Option<RigidBody>) -> Entity {
        let entity = self
            .world
            .spawn_entity()
            .with(LocalTransform::from(Transform::from_translation(at)))
            .with(Collider {
                shape,
                ..Collider::default()
            })
            .id();
        if let Some(body) = body {
            self.world.insert(entity, body);
        }
        entity
    }

    /// A floor whose top face is at y = 0.
    fn floor(&mut self) -> Entity {
        self.body(
            Vec3::new(0.0, -0.5, 0.0),
            ColliderShape::Box {
                half_extents: Vec3::new(50.0, 0.5, 50.0),
            },
            None,
        )
    }

    fn attach(&mut self, entity: Entity, type_name: &str) {
        self.scripting.attach(&mut self.world, entity, type_name);
    }

    /// The physics phase alone, as `app.rs` runs it.
    fn physics(&mut self, delta: f32) {
        propagate_transforms(&mut self.world);
        self.scripting
            .run_physics(&mut self.world, self.registry, delta);
    }

    /// The script phase alone, and the propagation after it.
    fn tick(&mut self, delta: f32) {
        self.scripting.tick(&mut self.world, self.registry, delta);
        propagate_transforms(&mut self.world);
    }

    fn frame(&mut self, delta: f32) {
        self.physics(delta);
        self.tick(delta);
    }

    fn faulted(&self, entity: Entity, type_name: &str) -> bool {
        self.world
            .get::<Scripts>(entity)
            .and_then(|scripts| scripts.get(type_name).map(|script| script.faulted))
            .expect("the behaviour is attached")
    }
}

#[test]
fn a_behaviour_hears_the_physics_step() {
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
    log.take();

    fixed_update_runs_once_before_each_step_in_attach_order(&scripting, &registry, &log);
    a_force_from_fixed_update_is_the_same_push_at_any_frame_rate(&scripting, &registry, &log);
    stays_arrive_once_a_step_and_never_for_a_dormant_pair(&scripting, &registry, &log);
    a_trigger_pair_stays_as_a_trigger(&scripting, &registry, &log);
    an_impact_carries_its_closing_velocity_to_both_sides(&scripting, &registry, &log);
    a_hook_not_overridden_costs_no_dispatch(&scripting, &registry, &log);
    a_throwing_fixed_update_or_stay_faults_only_its_behaviour(&scripting, &registry, &log);
    timing_off_records_nothing_and_on_attributes_by_type(&scripting, &registry, &log);
}

fn fixed_update_runs_once_before_each_step_in_attach_order(
    scripting: &Scripting,
    registry: &Registry,
    log: &Log,
) {
    let mut game = Game::new(scripting, registry);
    let body = game.body(Vec3::ZERO, cube(0.5), Some(frictionless()));
    game.attach(body, STEPPED);
    game.attach(body, SECOND_STEPPED);
    log.take();

    // Attached this frame: the step ran before the behaviours had started, so
    // the first they hear of it is the next frame's.
    game.frame(STEP);
    assert_eq!(
        log.probes(),
        [
            "SteppedProbe:OnStart",
            "SecondSteppedProbe:OnStart",
            "SteppedProbe:OnUpdate",
            "SecondSteppedProbe:OnUpdate",
        ]
    );

    physics::set_velocity(&game.world, body, Some(Vec3::X), None);
    for (delta, steps) in [(0.5 * STEP, 0), (STEP, 1), (3.0 * STEP, 3)] {
        game.frame(delta);
        let lines = log.probes();
        let mut expected = Vec::new();
        for _ in 0..steps {
            expected.extend([
                "SteppedProbe:OnFixedUpdate",
                "SecondSteppedProbe:OnFixedUpdate",
            ]);
        }
        expected.extend(["SteppedProbe:OnUpdate", "SecondSteppedProbe:OnUpdate"]);
        assert_eq!(hooks(&lines), expected, "a frame of {steps} step(s)");

        // Told the step, and reading where the last step left the body: both
        // behaviours see one x per step, a step's travel apart.
        let calls: Vec<Vec<f32>> = lines
            .iter()
            .filter(|line| hook(line).ends_with("OnFixedUpdate"))
            .map(|line| numbers(line))
            .collect();
        for pair in calls.chunks(2) {
            assert_eq!(pair[0], pair[1], "{lines:?}");
            assert!((pair[0][0] - STEP).abs() < 1e-7, "{lines:?}");
        }
        for (earlier, later) in calls.iter().step_by(2).zip(calls.iter().step_by(2).skip(1)) {
            assert!(
                (later[1] - earlier[1] - STEP).abs() < 1e-5,
                "not a step apart: {lines:?}"
            );
        }
    }
}

fn a_force_from_fixed_update_is_the_same_push_at_any_frame_rate(
    scripting: &Scripting,
    registry: &Registry,
    log: &Log,
) {
    let pushed = |fps: f32| {
        let mut game = Game::new(scripting, registry);
        let body = game.body(Vec3::ZERO, cube(0.5), Some(frictionless()));
        game.attach(body, PUSHING);
        game.frame(1.0 / fps);
        let start = physics::velocity(&game.world, body).expect("a body").0.x;
        log.take();
        for _ in 0..fps.round() as u32 {
            game.frame(1.0 / fps);
        }
        let steps = log.probes().len();
        let end = physics::velocity(&game.world, body).expect("a body").0.x;
        (end - start, steps)
    };

    let mut changes = Vec::new();
    for fps in [30.0, 144.0] {
        let (change, steps) = pushed(fps);
        let expected = 2.0 * steps as f32 * STEP;
        assert!(
            (change - expected).abs() < 1e-3,
            "{fps} fps: {change} m/s over {steps} steps, not {expected}"
        );
        assert!(steps.abs_diff(60) <= 1, "{fps} fps took {steps} steps");
        changes.push(change);
    }
    assert!(
        (changes[0] - changes[1]).abs() <= 2.0 * STEP + 1e-3,
        "{changes:?}"
    );
}

fn stays_arrive_once_a_step_and_never_for_a_dormant_pair(
    scripting: &Scripting,
    registry: &Registry,
    log: &Log,
) {
    let mut game = Game::new(scripting, registry);
    game.floor();
    let resting = game.body(
        Vec3::new(0.0, 0.5, 0.0),
        cube(0.5),
        Some(RigidBody::default()),
    );
    game.attach(resting, CONTACT);
    game.frame(STEP);
    log.take();

    let mut frames = Vec::new();
    for _ in 0..60 {
        game.frame(3.0 * STEP);
        let lines = log.probes();
        let contacts: Vec<&str> = hooks(&lines)
            .into_iter()
            .filter(|hook| !hook.ends_with("OnUpdate"))
            .collect();
        assert!(
            contacts
                .iter()
                .all(|hook| *hook == "ContactProbe:OnCollisionStay"),
            "{lines:?}"
        );
        frames.push(contacts.len());
    }

    // Awake and settling, it stays three times a frame; asleep, it is dormant
    // and stays not at all — and never exits.
    assert!(frames[..3].iter().all(|&n| n == 3), "{frames:?}");
    assert!(frames[40..].iter().all(|&n| n == 0), "{frames:?}");
}

fn a_trigger_pair_stays_as_a_trigger(scripting: &Scripting, registry: &Registry, log: &Log) {
    let mut game = Game::new(scripting, registry);
    let trigger = game.body(Vec3::ZERO, cube(1.0), None);
    game.world.get_mut::<Collider>(trigger).unwrap().is_trigger = true;
    let ball = game.body(
        Vec3::new(-2.0, 0.0, 0.0),
        ColliderShape::Sphere { radius: 0.25 },
        Some(frictionless()),
    );
    game.attach(ball, CONTACT);
    game.frame(STEP);
    physics::set_velocity(&game.world, ball, Some(Vec3::new(2.0, 0.0, 0.0)), None);
    log.take();

    let mut frames = Vec::new();
    for _ in 0..120 {
        game.frame(STEP);
        let lines = log.probes();
        frames.push(
            hooks(&lines)
                .into_iter()
                .filter(|hook| !hook.ends_with("OnUpdate"))
                .map(str::to_owned)
                .collect::<Vec<_>>(),
        );
    }

    let all = frames.concat();
    assert!(all.len() > 3, "{all:?}");
    assert_eq!(all.first().unwrap(), "ContactProbe:OnTriggerEnter");
    assert_eq!(all.last().unwrap(), "ContactProbe:OnTriggerExit");
    assert!(
        all[1..all.len() - 1]
            .iter()
            .all(|hook| hook == "ContactProbe:OnTriggerStay"),
        "{all:?}"
    );
    // One a step, from the step after the enter to the one before the exit.
    let entered = frames.iter().position(|f| !f.is_empty()).unwrap();
    let exited = frames.iter().rposition(|f| !f.is_empty()).unwrap();
    for frame in &frames[entered + 1..exited] {
        assert_eq!(frame, &["ContactProbe:OnTriggerStay"], "{frames:?}");
    }
}

fn an_impact_carries_its_closing_velocity_to_both_sides(
    scripting: &Scripting,
    registry: &Registry,
    log: &Log,
) {
    let mut game = Game::new(scripting, registry);
    let wall = game.body(Vec3::new(3.0, 0.0, 0.0), cube(0.5), None);
    let ball = game.body(Vec3::ZERO, cube(0.5), Some(frictionless()));
    game.attach(wall, CONTACT);
    game.attach(ball, CONTACT);
    game.frame(STEP);
    physics::set_velocity(&game.world, ball, Some(Vec3::new(4.0, 0.0, 0.0)), None);
    log.take();

    let mut enters = Vec::new();
    for _ in 0..60 {
        game.frame(STEP);
        enters.extend(
            log.probes()
                .into_iter()
                .filter(|line| hook(line) == "ContactProbe:OnCollisionEnter"),
        );
        if !enters.is_empty() {
            break;
        }
    }
    assert_eq!(enters.len(), 2, "{enters:?}");

    let side = |entity: Entity| {
        let line = enters
            .iter()
            .find(|line| numbers(line)[0] == entity.index() as f32)
            .unwrap_or_else(|| panic!("{entity:?} heard no enter: {enters:?}"));
        let n = numbers(line);
        (Vec3::new(n[2], n[3], n[4]), Vec3::new(n[5], n[6], n[7]))
    };
    let (ball_velocity, ball_normal) = side(ball);
    let (wall_velocity, wall_normal) = side(wall);
    assert!(
        (ball_velocity - Vec3::new(4.0, 0.0, 0.0)).length() < 0.05,
        "the ball closed at {ball_velocity}"
    );
    assert!((ball_velocity.dot(ball_normal) - 4.0).abs() < 0.05);
    assert_eq!(wall_velocity, -ball_velocity);
    assert_eq!(wall_normal, -ball_normal);
}

fn a_hook_not_overridden_costs_no_dispatch(scripting: &Scripting, registry: &Registry, log: &Log) {
    let mut game = Game::new(scripting, registry);
    game.floor();
    // Crates standing on the floor, awake and touching it, whose behaviour
    // overrides neither OnFixedUpdate nor a Stay callback...
    let idle: Vec<Entity> = (0..20)
        .map(|i| {
            let at = Vec3::new(i as f32 * 2.0, 0.5, 0.0);
            let entity = game.body(at, cube(0.5), Some(RigidBody::default()));
            game.attach(entity, IDLE);
            entity
        })
        .collect();
    // ...beside one that overrides both, so the steps are dispatched to and
    // the stays reported.
    let listening = game.body(
        Vec3::new(-4.0, 0.5, 0.0),
        cube(0.5),
        Some(RigidBody::default()),
    );
    game.attach(listening, CONTACT);
    game.attach(listening, STEPPED);
    game.frame(STEP);
    log.take();

    let mut idle_stays = 0;
    for _ in 0..10 {
        let before = scripting.host().dispatched();
        game.physics(3.0 * STEP);
        idle_stays += game
            .world
            .resource::<CollisionState>()
            .events
            .iter()
            .filter(|event| {
                event.kind == CollisionEventKind::Stay
                    && (idle.contains(&event.a) || idle.contains(&event.b))
            })
            .count();
        game.tick(3.0 * STEP);
        let dispatched = scripting.host().dispatched() - before;
        let probes = log.probes();
        assert_eq!(
            dispatched,
            (idle.len() + probes.len()) as u64,
            "an OnUpdate for each idle crate and one call per probe line: {probes:?}"
        );
    }
    assert!(idle_stays > 0, "the idle crates never stayed on the floor");
}

fn a_throwing_fixed_update_or_stay_faults_only_its_behaviour(
    scripting: &Scripting,
    registry: &Registry,
    log: &Log,
) {
    let mut game = Game::new(scripting, registry);
    game.floor();
    let stepping = game.body(Vec3::new(0.0, 5.0, 0.0), cube(0.5), Some(frictionless()));
    game.attach(stepping, THROWING_STEP);
    game.attach(stepping, SECOND_STEPPED);
    let resting = game.body(
        Vec3::new(4.0, 0.5, 0.0),
        cube(0.5),
        Some(RigidBody::default()),
    );
    game.attach(resting, THROWING_STAY);
    game.attach(resting, CONTACT);
    game.frame(STEP);
    log.take();

    game.frame(2.0 * STEP);
    let lines = log.take();
    let probes: Vec<String> = lines
        .iter()
        .map(|(_, message)| message.clone())
        .filter(|message| message.contains("Probe:"))
        .collect();
    assert_eq!(
        hooks(&probes),
        [
            "ThrowingStepProbe:OnFixedUpdate",
            "SecondSteppedProbe:OnFixedUpdate",
            "SecondSteppedProbe:OnFixedUpdate",
            "ThrowingStayProbe:OnCollisionStay",
            "ContactProbe:OnCollisionStay",
            "ContactProbe:OnCollisionStay",
            "SecondSteppedProbe:OnUpdate",
            "ContactProbe:OnUpdate",
        ]
    );
    for hook in ["OnFixedUpdate", "OnCollisionStay"] {
        assert!(
            lines
                .iter()
                .any(|(level, message)| *level == LogLevel::Error && message.contains(hook)),
            "no error named {hook}: {lines:?}"
        );
    }
    assert!(game.faulted(stepping, THROWING_STEP));
    assert!(!game.faulted(stepping, SECOND_STEPPED));
    assert!(game.faulted(resting, THROWING_STAY));
    assert!(!game.faulted(resting, CONTACT));

    game.frame(STEP);
    let lines = log.probes();
    assert!(
        !lines.iter().any(|line| line.starts_with("Throwing")),
        "{lines:?}"
    );
}

fn timing_off_records_nothing_and_on_attributes_by_type(
    scripting: &Scripting,
    registry: &Registry,
    log: &Log,
) {
    let mut game = Game::new(scripting, registry);
    let slow = game.world.spawn();
    game.attach(slow, SLOW);
    for _ in 0..10 {
        let entity = game.world.spawn();
        game.attach(entity, IDLE);
    }
    for _ in 0..3 {
        game.frame(STEP);
    }
    assert_eq!(game.world.resource::<ScriptTimings>().rows().count(), 0);

    game.world.resource_mut::<ScriptTimings>().enabled = true;
    for _ in 0..3 {
        game.frame(STEP);
    }
    let timings = game.world.resource::<ScriptTimings>();
    let row = |name: &str| {
        timings
            .rows()
            .find(|row| row.type_name == name)
            .unwrap_or_else(|| panic!("no row for {name}"))
    };
    let (slow, idle) = (row(SLOW), row(IDLE));
    assert!(
        slow.last_frame.as_secs_f64() >= 2.0e-3,
        "{:?}",
        slow.last_frame
    );
    assert!(
        slow.average().as_secs_f64() >= 2.0e-3,
        "{:?}",
        slow.average()
    );
    assert!(
        idle.last_frame < slow.last_frame / 4,
        "ten idle behaviours took {:?}",
        idle.last_frame
    );
    log.take();
}
