//! What a Behaviour reads of the clock, the wheel and the keys, and what the
//! shape queries answer it, through real CoreCLR and the engine's own physics
//! phase and script tick, each frame's delta coming through `advance_time` as
//! `app.rs`'s does.
//!
//! The probes are `Orrin.Tests.ClockProbe` and `QueryProbe` in
//! `FrameProbes.cs`, which log what they read; the log is read back through the
//! engine's own stream.
//!
//! Needs both C# assemblies built (`dotnet build scripting/Orrin` and
//! `dotnet build scripting/DemoGame`) and skips without them, as
//! `fixed_steps` does. One test, because CoreCLR boots once per process.

#![cfg(feature = "scripting")]

use std::path::{Path, PathBuf};

use glam::Vec3;
use orrin_core::logging::{ConsoleQueue, LogConfig, Stream};
use orrin_core::physics::{CollisionState, PhysicsState, STEP};
use orrin_core::scene::input::keys::key_by_name;
use orrin_core::scene::{
    Collider, ColliderShape, InputState, LocalTransform, LogBuffer, Play, PlayState, Time,
    Transform, advance_time, propagate_transforms, register_components,
};
use orrin_core::scripting::Scripting;
use orrin_ecs::{Entity, World};
use orrin_registry::Registry;

const CLOCK: &str = "Orrin.Tests.ClockProbe, Orrin";
const QUERY: &str = "Orrin.Tests.QueryProbe, Orrin";

fn assemblies() -> Option<(PathBuf, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bindings = Scripting::find_bindings_dir(&root.join("scripting/Orrin"))?;
    let game = Scripting::find_game_assembly(&root.join("scripting/DemoGame"), "DemoGame")?;
    Some((bindings, game))
}

struct Log(ConsoleQueue);

impl Log {
    /// The probes' own lines since the last call, in order.
    fn probes(&self) -> Vec<String> {
        let mut buffer = LogBuffer::default();
        self.0.drain_into(&mut buffer, 0);
        buffer
            .iter()
            .map(|entry| entry.message.clone())
            .filter(|message| message.contains("Probe:"))
            .collect()
    }
}

/// `{type}:{hook}`, without whatever the probe logged after it.
fn hook(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}

/// The numbers a probe logged after its hook.
fn numbers(line: &str) -> Vec<f32> {
    line.split_whitespace()
        .skip(1)
        .map(|word| word.parse().expect("a probe logs numbers"))
        .collect()
}

/// The one line `hook` logged, as its numbers.
fn logged(lines: &[String], name: &str) -> Vec<f32> {
    let found: Vec<&String> = lines.iter().filter(|line| hook(line) == name).collect();
    assert_eq!(found.len(), 1, "{name} in {lines:#?}");
    numbers(found[0])
}

fn indices(entities: &[Entity]) -> Vec<f32> {
    entities.iter().map(|entity| entity.index as f32).collect()
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
        world.insert_resource(Time::new());
        world.insert_resource(InputState::new());
        let mut play = Play::default();
        play.set(PlayState::Playing);
        world.insert_resource(play);
        Self {
            world,
            registry,
            scripting,
        }
    }

    fn collider(&mut self, at: Vec3, collider: Collider) -> Entity {
        self.world
            .spawn_entity()
            .with(LocalTransform::from(Transform::from_translation(at)))
            .with(collider)
            .id()
    }

    /// A frame `delta` real seconds long, as `app.rs` runs its game half.
    fn frame(&mut self, delta: f32) {
        let game = advance_time(&self.world, delta);
        propagate_transforms(&mut self.world);
        self.scripting
            .run_physics(&mut self.world, self.registry, game);
        self.scripting.tick(&mut self.world, self.registry, game);
        self.world.resource_mut::<InputState>().end_frame();
    }
}

#[test]
fn a_behaviour_reads_the_clock_the_wheel_and_the_shape_queries() {
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
    log.probes();

    the_time_scale_reaches_every_hook_and_stop_undoes_it(&scripting, &registry, &log);
    every_shape_query_answers_from_the_last_step(&scripting, &registry, &log);
}

fn the_time_scale_reaches_every_hook_and_stop_undoes_it(
    scripting: &Scripting,
    registry: &Registry,
    log: &Log,
) {
    const REAL: f32 = STEP;
    let mut game = Game::new(scripting, registry);
    let entity = game.world.spawn();
    game.scripting.attach(&mut game.world, entity, CLOCK);

    // OnStart halves the scale; this frame's delta was taken before it did.
    game.frame(REAL);
    let lines = log.probes();
    let first = logged(&lines, "ClockProbe:OnUpdate");
    assert_eq!(&first[..5], [REAL, REAL, REAL, STEP, 0.5], "{lines:?}");

    // A real second at half speed: OnUpdate gets half of each frame, physics
    // takes half the steps at full length, and the half-second timer started
    // in OnStart fires after a real second rather than half of one.
    let mut fixed = Vec::new();
    let mut timer_frame = None;
    for frame in 1..=60 {
        game.frame(REAL);
        for line in log.probes() {
            match hook(&line) {
                "ClockProbe:OnUpdate" => {
                    let read = numbers(&line);
                    assert_eq!(
                        &read[..5],
                        [REAL / 2.0, REAL / 2.0, REAL, STEP, 0.5],
                        "{line}"
                    );
                    assert_eq!(&read[5..], [0.0, 0.0, 0.0], "{line}");
                }
                "ClockProbe:OnFixedUpdate" => fixed.push(numbers(&line)),
                "ClockProbe:Timer" => timer_frame = timer_frame.or(Some(frame)),
                other => panic!("unexpected {other}"),
            }
        }
    }
    assert!(
        fixed.len().abs_diff(30) <= 1,
        "{} steps in a second at 0.5",
        fixed.len()
    );
    for read in &fixed {
        assert_eq!(
            read,
            &[STEP, STEP, REAL],
            "Time.DeltaTime in OnFixedUpdate is the step"
        );
    }
    let timer_frame = timer_frame.expect("the timer never fired");
    assert!(
        (57..=61).contains(&timer_frame),
        "fired on frame {timer_frame}"
    );

    // F5 and a turn of the wheel arrive together; F5 pauses. The wheel reads
    // for its frame alone, and a paused game still updates, with no time and
    // no steps.
    let f5 = key_by_name("F5").expect("F5 is a key");
    {
        let mut input = game.world.resource_mut::<InputState>();
        input.press(f5);
        input.scroll_wheel((0.0, 2.0));
        input.scroll_wheel((-0.5, 1.0));
    }
    game.frame(REAL);
    let pressed = logged(&log.probes(), "ClockProbe:OnUpdate");
    assert_eq!(pressed[4..], [0.0, -0.5, 3.0, 1.0], "{pressed:?}");

    for _ in 0..30 {
        game.frame(REAL);
        let lines = log.probes();
        assert_eq!(lines.len(), 1, "{lines:?}");
        let read = logged(&lines, "ClockProbe:OnUpdate");
        assert_eq!(
            &read[..7],
            [0.0, 0.0, REAL, STEP, 0.0, 0.0, 0.0],
            "{lines:?}"
        );
    }

    // Stop, then Play: the next session runs at full speed.
    let mut play = game.world.resource_mut::<Play>();
    play.set(PlayState::Editing);
    play.set(PlayState::Playing);
    assert_eq!(play.time_scale(), 1.0);
}

fn every_shape_query_answers_from_the_last_step(
    scripting: &Scripting,
    registry: &Registry,
    log: &Log,
) {
    let cube = ColliderShape::Box {
        half_extents: Vec3::splat(0.5),
    };
    let mut game = Game::new(scripting, registry);
    let wall = game.collider(
        Vec3::new(0.0, 0.0, -5.0),
        Collider {
            shape: cube,
            layer: 2,
            ..Collider::default()
        },
    );
    let trigger = game.collider(
        Vec3::new(5.0, 0.0, 0.0),
        Collider {
            shape: cube,
            is_trigger: true,
            ..Collider::default()
        },
    );
    let crate_ = game.collider(
        Vec3::new(-5.0, 0.0, 0.0),
        Collider {
            shape: cube,
            ..Collider::default()
        },
    );
    let asker = game.world.spawn();
    game.scripting.attach(&mut game.world, asker, QUERY);

    // The physics phase builds the bodies before the tick asks about them.
    game.frame(STEP);
    let lines = log.probes();

    let everything = indices(&[wall, trigger, crate_]);
    assert_eq!(
        logged(&lines, "QueryProbe:OverlapInto"),
        [1.0, everything[0], 0.0],
        "the first in entity order fills a one-slot buffer, and asking again allocates nothing"
    );
    assert_eq!(logged(&lines, "QueryProbe:OverlapSphere"), everything);
    assert_eq!(
        logged(&lines, "QueryProbe:OverlapSphereSolid"),
        indices(&[wall, crate_])
    );
    assert_eq!(
        logged(&lines, "QueryProbe:OverlapBoxTurned"),
        indices(&[wall])
    );
    assert!(logged(&lines, "QueryProbe:OverlapBoxUnturned").is_empty());

    for cast in ["QueryProbe:SphereCast", "QueryProbe:BoxCast"] {
        let hit = logged(&lines, cast);
        assert_eq!(hit[..2], [1.0, wall.index as f32], "{cast}: {hit:?}");
        assert!((hit[2] - 4.0).abs() < 1e-3, "{cast}: {hit:?}");
        assert!(
            Vec3::new(hit[3], hit[4], hit[5]).abs_diff_eq(Vec3::Z, 1e-3),
            "{cast}: {hit:?}"
        );
    }
    assert_eq!(logged(&lines, "QueryProbe:SphereCastShort")[0], 0.0);
    assert_eq!(logged(&lines, "QueryProbe:CastPastTrigger")[0], 0.0);
    let at = logged(&lines, "QueryProbe:CastAtTrigger");
    assert_eq!(at[..2], [1.0, trigger.index as f32], "{at:?}");
    assert!((at[2] - 4.0).abs() < 1e-3, "{at:?}");
    assert_eq!(logged(&lines, "QueryProbe:CastMasked")[0], 0.0);
}
