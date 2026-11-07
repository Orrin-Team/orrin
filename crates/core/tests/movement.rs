//! The demo's first-person controller and ball launcher, driven by simulated
//! keys and mouse through the real C# scripts and the real physics step.
//!
//! What the controller is tuned to feel like is asserted as numbers: it reaches
//! its top speed and holds it, it stops within a fraction of a second, a jump
//! leaves the ground once, a slide carries more than walking can, and it
//! stands still on a slope. A regression in any of those is a game that feels
//! wrong, which nothing else here would catch.
//!
//! Needs both C# assemblies built (`dotnet build scripting/Orrin` and
//! `dotnet build scripting/DemoGame`) and skips without them, as
//! `scripted_scene` does. One test, because CoreCLR boots once per process.

#![cfg(feature = "scripting")]

use std::path::{Path, PathBuf};

use glam::{Quat, Vec3};
use orrin_core::physics::{self, CollisionState, PhysicsState};
use orrin_core::scene::{
    Collider, ColliderShape, InputState, LocalTransform, LogBuffer, Name, Tag, Transform,
    propagate_transforms, register_components,
};
use orrin_core::scripting::Scripting;
use orrin_ecs::{Entity, World};
use orrin_registry::{ComponentId, Registry, Value};
use winit::event::DeviceEvent;

const W: u32 = 23;
const SPACE: u32 = 44;
const LEFT_CONTROL: u32 = 51;
const LEFT_MOUSE: u32 = 0;

const FRAME: f32 = 1.0 / 60.0;
const PLAYER: &str = "DemoGame.FpsPlayer, DemoGame";
const TARGET: &str = "DemoGame.Target, DemoGame";

fn assemblies() -> Option<(PathBuf, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bindings = Scripting::find_bindings_dir(&root.join("scripting/Orrin"))?;
    let game = Scripting::find_game_assembly(&root.join("scripting/DemoGame"), "DemoGame")?;
    Some((bindings, game))
}

struct Game<'a> {
    world: World,
    registry: &'a Registry,
    scripting: &'a Scripting,
}

impl<'a> Game<'a> {
    /// A flat floor with its top at y = 0, the engine library loaded so
    /// scripts can spawn meshes, and nothing else.
    fn new(scripting: &'a Scripting, registry: &'a Registry) -> Self {
        let mut world = World::new();
        world.insert_resource(CollisionState::default());
        world.insert_resource(PhysicsState::default());
        world.insert_resource(LogBuffer::default());
        world.insert_resource(InputState::new());
        world.insert_resource(orrin_core::scene::CursorLock::default());
        orrin_core::scene::entities::install_library(
            &mut world,
            &mut orrin_core::gfx::HeadlessBackend::new(),
        );
        let mut game = Self {
            world,
            registry,
            scripting,
        };
        game.solid(
            Vec3::new(0.0, -0.5, 0.0),
            Quat::IDENTITY,
            Vec3::new(100.0, 0.5, 100.0),
        );
        game
    }

    fn solid(&mut self, at: Vec3, rotation: Quat, half_extents: Vec3) -> Entity {
        self.world
            .spawn_entity()
            .with(LocalTransform::from(Transform {
                translation: at,
                rotation,
                scale: Vec3::ONE,
            }))
            .with(Collider {
                shape: ColliderShape::Box { half_extents },
                ..Collider::default()
            })
            .id()
    }

    /// The player standing at `at`, facing -Z, as the demo's entry spawns it.
    fn player(&mut self, at: Vec3) -> Entity {
        let player = self
            .world
            .spawn_entity()
            .with(LocalTransform::from(Transform::from_translation(at)))
            .id();
        // It attaches its own launcher, to its camera: an entity holds one
        // behaviour. Its first tick builds its body and its camera, a second
        // lets the physics run create the body and the launcher start.
        self.scripting.attach(&mut self.world, player, PLAYER);
        self.frames(3);
        player
    }

    /// What the app does with a frame, in its order.
    fn frame(&mut self) {
        propagate_transforms(&mut self.world);
        physics::run(&mut self.world, FRAME);
        self.scripting.tick(&mut self.world, self.registry, FRAME);
        propagate_transforms(&mut self.world);
        self.world.resource_mut::<InputState>().end_frame();
    }

    fn frames(&mut self, count: u32) {
        for _ in 0..count {
            self.frame();
        }
    }

    fn seconds(&mut self, seconds: f32) {
        self.frames((seconds / FRAME).round() as u32);
    }

    fn hold(&mut self, key: u32) {
        self.world.resource_mut::<InputState>().press(key);
    }

    fn release(&mut self, key: u32) {
        self.world.resource_mut::<InputState>().release(key);
    }

    fn click(&mut self) {
        let mut input = self.world.resource_mut::<InputState>();
        input.press_mouse_button(LEFT_MOUSE);
        drop(input);
        self.frame();
        self.world
            .resource_mut::<InputState>()
            .release_mouse_button(LEFT_MOUSE);
        self.frame();
    }

    fn mouse(&mut self, dx: f64, dy: f64) {
        self.world
            .resource_mut::<InputState>()
            .on_device_event(&DeviceEvent::MouseMotion { delta: (dx, dy) });
    }

    fn position(&self, entity: Entity) -> Vec3 {
        self.world
            .get::<LocalTransform>(entity)
            .unwrap()
            .translation
    }

    fn velocity(&self, entity: Entity) -> Vec3 {
        physics::velocity(&self.world, entity).expect("a body").0
    }

    fn horizontal_speed(&self, entity: Entity) -> f32 {
        let v = self.velocity(entity);
        Vec3::new(v.x, 0.0, v.z).length()
    }

    fn field(&self, entity: Entity, component: &'static str, field: &str) -> f32 {
        let value = self
            .registry
            .get(&ComponentId::new(component))
            .and_then(|vtable| vtable.read(&self.world, entity))
            .and_then(|bag| bag.field(field).cloned());
        match value {
            Some(Value::F32(v)) => v,
            Some(Value::I32(v)) => v as f32,
            other => panic!("{component}.{field} is {other:?}"),
        }
    }

    fn tagged(&self, tag: &str) -> Vec<Entity> {
        self.world
            .entities()
            .filter(|&e| self.world.get::<Tag>(e).is_some_and(|t| t.as_str() == tag))
            .collect()
    }
}

#[test]
fn the_demo_controller_moves_like_it_is_tuned_to() {
    let Some((bindings, game)) = assemblies() else {
        eprintln!("skipping: build scripting/Orrin and scripting/DemoGame first");
        return;
    };
    let scripting = Scripting::boot(&bindings, &game).expect("CoreCLR failed to boot");
    let mut registry = Registry::new();
    register_components(&mut registry);
    assert!(scripting.register_components(&mut registry) > 0);

    walks_to_its_top_speed_and_stops_when_let_go(&scripting, &registry);
    jumps_once_per_landing(&scripting, &registry);
    a_slide_carries_more_than_walking_can(&scripting, &registry);
    the_mouse_turns_where_it_walks(&scripting, &registry);
    stands_still_on_a_ramp(&scripting, &registry);
    a_ball_fired_at_a_target_scores(&scripting, &registry);
    the_course_builds_and_nothing_falls_through_it(&scripting, &registry);
}

fn walks_to_its_top_speed_and_stops_when_let_go(scripting: &Scripting, registry: &Registry) {
    let mut game = Game::new(scripting, registry);
    let player = game.player(Vec3::new(0.0, 1.0, 0.0));
    let top = game.field(player, "demo.fps_player", "MaxSpeed");

    game.hold(W);
    let mut fastest: f32 = 0.0;
    for _ in 0..180 {
        game.frame();
        fastest = fastest.max(game.horizontal_speed(player));
    }
    let cruising = game.horizontal_speed(player);
    assert!(
        (cruising - top).abs() < top * 0.1,
        "cruised at {cruising} m/s against a top speed of {top}"
    );
    assert!(fastest < top * 1.1, "overshot to {fastest} m/s");
    assert!(
        game.position(player).z < -10.0,
        "W did not walk forward, toward -Z"
    );

    game.release(W);
    game.seconds(0.5);
    let coasting = game.horizontal_speed(player);
    eprintln!(
        "walk: top {top} m/s, cruised {cruising:.2}, fastest {fastest:.2}, 0.5 s after letting go {coasting:.3}"
    );
    assert!(
        coasting < 1.0,
        "still moving at {coasting} m/s half a second after letting go"
    );
}

fn jumps_once_per_landing(scripting: &Scripting, registry: &Registry) {
    let mut game = Game::new(scripting, registry);
    let player = game.player(Vec3::new(0.0, 1.0, 0.0));
    game.seconds(0.5);
    let standing = game.position(player).y;

    game.hold(SPACE);
    let mut highest = standing;
    let mut left_ground = None;
    for frame in 0..40 {
        game.frame();
        let y = game.position(player).y;
        highest = highest.max(y);
        if left_ground.is_none() && y > standing + 0.05 {
            left_ground = Some(frame);
        }
    }
    let rise = highest - standing;
    eprintln!("jump: rose {rise:.2} m, left the ground on frame {left_ground:?}");
    assert!(
        left_ground.is_some_and(|f| f < 5),
        "the jump did not leave the ground at once"
    );
    assert!(rise > 0.8, "jumped only {rise} m");
    assert!(rise < 3.0, "holding Space flew {rise} m: a jump in mid-air");
}

fn a_slide_carries_more_than_walking_can(scripting: &Scripting, registry: &Registry) {
    let mut game = Game::new(scripting, registry);
    let player = game.player(Vec3::new(0.0, 1.0, 0.0));
    let top = game.field(player, "demo.fps_player", "MaxSpeed");
    game.hold(W);
    game.seconds(2.0);

    game.hold(LEFT_CONTROL);
    game.frames(3);
    let sliding = game.horizontal_speed(player);
    let height = game.world.get::<LocalTransform>(player).unwrap().scale.y;
    assert!(
        sliding > top * 1.2,
        "slid at {sliding} m/s, no faster than walking's {top}"
    );
    assert!(
        height < 0.75,
        "crouching left the capsule at scale {height}"
    );

    game.seconds(1.5);
    let later = game.horizontal_speed(player);
    eprintln!(
        "slide: {sliding:.2} m/s against a walk of {top}, {later:.2} after 1.5 s, crouched to scale {height}"
    );
    assert!(
        later < sliding,
        "a slide on the flat kept its speed: {sliding} then {later}"
    );

    game.release(LEFT_CONTROL);
    game.frames(3);
    let height = game.world.get::<LocalTransform>(player).unwrap().scale.y;
    assert!(
        (height - 1.0).abs() < 1e-3,
        "stood back up to scale {height}"
    );
}

fn the_mouse_turns_where_it_walks(scripting: &Scripting, registry: &Registry) {
    let mut game = Game::new(scripting, registry);
    let player = game.player(Vec3::new(0.0, 1.0, 0.0));
    // Mouse look waits for the cursor lock, which a click asks for.
    game.click();
    let sensitivity = game.field(player, "demo.fps_player", "Sensitivity");
    // A quarter turn to the right.
    game.mouse(f64::from(90.0 / sensitivity), 0.0);
    game.frame();

    game.hold(W);
    game.seconds(1.0);
    let moved = game.position(player);
    eprintln!("look: a quarter turn right then a second of W ended at {moved}");
    assert!(
        moved.x > 3.0 && moved.z.abs() < 1.0,
        "a quarter turn right then W walked to {moved}, not toward +X"
    );
}

fn stands_still_on_a_ramp(scripting: &Scripting, registry: &Registry) {
    let mut game = Game::new(scripting, registry);
    game.solid(
        Vec3::new(0.0, 0.0, 0.0),
        Quat::from_rotation_x(20f32.to_radians()),
        Vec3::new(5.0, 0.5, 10.0),
    );
    let player = game.player(Vec3::new(0.0, 2.5, 0.0));
    game.seconds(1.0);
    let settled = game.position(player);

    game.seconds(2.0);
    let crept = game.position(player).distance(settled);
    eprintln!("ramp: crept {crept:.3} m in 2 s on 20°");
    assert!(
        crept < 0.2,
        "slid {crept} m down a 20° ramp while standing on it"
    );
}

fn a_ball_fired_at_a_target_scores(scripting: &Scripting, registry: &Registry) {
    let mut game = Game::new(scripting, registry);
    let target = game.solid(
        Vec3::new(0.0, 1.5, -12.0),
        Quat::IDENTITY,
        Vec3::new(2.0, 1.5, 0.25),
    );
    game.world.insert(target, Name::new("Target"));
    scripting.attach(&mut game.world, target, TARGET);
    let player = game.player(Vec3::new(0.0, 1.0, 0.0));
    game.seconds(0.3);

    // The first click locks the cursor, the second fires.
    game.click();
    assert!(game.tagged("ball").is_empty(), "the locking click fired");
    game.click();
    assert_eq!(game.tagged("ball").len(), 1, "the click fired no ball");
    game.seconds(1.0);

    let hits = game.field(target, "demo.target", "Hits");
    eprintln!("shoot: {hits} hit on a target 12 m ahead");
    assert_eq!(hits, 1.0, "the ball did not score on the target ahead");
    let _ = player;
}

fn the_course_builds_and_nothing_falls_through_it(scripting: &Scripting, registry: &Registry) {
    let mut world = World::new();
    world.insert_resource(CollisionState::default());
    world.insert_resource(PhysicsState::default());
    world.insert_resource(LogBuffer::default());
    world.insert_resource(InputState::new());
    world.insert_resource(orrin_core::scene::CursorLock::default());
    orrin_core::scene::entities::install_library(
        &mut world,
        &mut orrin_core::gfx::HeadlessBackend::new(),
    );
    let mut game = Game {
        world,
        registry,
        scripting,
    };
    let entry = game.world.spawn();
    scripting.attach(&mut game.world, entry, "DemoGame.MovementDemo, DemoGame");
    game.seconds(3.0);

    let players = game.tagged("player");
    assert_eq!(players.len(), 1, "the course has {} players", players.len());
    let y = game.position(players[0]).y;
    assert!(
        y > 0.0 && y < 3.0,
        "the player stood at {y} m after three seconds"
    );
    assert!(
        !game.tagged("target").is_empty(),
        "the course has no targets"
    );
    for target in game.tagged("target") {
        assert!(
            game.position(target).y > -1.0,
            "a target fell through the course"
        );
    }
}
