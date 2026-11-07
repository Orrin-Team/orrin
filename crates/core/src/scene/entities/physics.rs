//! The physics testbed: a stack to knock over, balls to knock
//! it over with, and a paddle with no rigid body sweeping through the lot.
//!
//! The paddle is the part worth watching. It is turned by `Spin`, which writes
//! its transform and nothing else, so it starts static and becomes kinematic the
//! first frame it turns — and it still shoves what it meets, because kinematic
//! bodies are driven by velocity rather than teleported.
//!
//! # The drop panel
//!
//! A corner of retained UI drops more into the scene while it plays. The
//! buttons are known by their [`Tag`] rather than by a component of their own:
//! Stop rebuilds the scene from its saved document, which keeps only registered
//! components, and a marker registered for one testbed would sit in every
//! project's Add Component menu. [`spawn_from_clicks`] reads the tick's clicks
//! without draining them, so scripts still hear about the same ones.
//!
//! What it drops is tagged [`SPAWNED`], which is all Clear removes, and the
//! count is capped: past a few hundred bodies a single-threaded Jolt step
//! starts to cost real frame time, and the panel says why the buttons stopped.

use glam::{EulerRot, Quat, Vec2, Vec3};

use orrin_ecs::{Entity, World};

use super::library::load_library;
use super::spawn_mesh;
use crate::gfx::RenderBackend;
use crate::scene::{
    Assets, Camera, Collider, ColliderShape, Name, RigidBody, Spin, Tag, TextAnchor, Transform,
    despawn_recursive, reparent,
};
use crate::ui::events::{PointerKind, UiEvents};
use crate::ui::{Button, Canvas, Corner, Image, RectTransform, ScaleMode, Shadow, Text};

const PYRAMID_BASE: u32 = 6;

/// Everything the drop panel made.
pub const SPAWNED: &str = "physics:spawned";

/// Beyond this the drop buttons switch off until something is cleared.
pub const MAX_SPAWNED: usize = 300;

const TOWER: usize = 5;

/// The panel's controls, by the tag each carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Control {
    Crate,
    Ball,
    Tower,
    Clear,
}

impl Control {
    const ALL: [Control; 4] = [Self::Crate, Self::Ball, Self::Tower, Self::Clear];

    fn tag(self) -> &'static str {
        match self {
            Self::Crate => "physics:drop_crate",
            Self::Ball => "physics:drop_ball",
            Self::Tower => "physics:drop_tower",
            Self::Clear => "physics:clear",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Crate => "Drop a crate",
            Self::Ball => "Drop a ball",
            Self::Tower => "Drop a tower",
            Self::Clear => "Clear",
        }
    }

    fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|control| control.tag() == tag)
    }
}

const COUNT_LABEL: &str = "physics:count";

/// The library's meshes are unit-sized, so a half-extent of 0.5 scaled by the
/// transform is the mesh's own outline.
fn unit_box() -> Collider {
    Collider {
        shape: ColliderShape::Box {
            half_extents: Vec3::splat(0.5),
        },
        ..Collider::default()
    }
}

fn unit_ball() -> Collider {
    Collider {
        shape: ColliderShape::Sphere { radius: 0.5 },
        restitution: 0.5,
        ..Collider::default()
    }
}

fn ball_body() -> RigidBody {
    RigidBody {
        mass: 4.0,
        ..RigidBody::default()
    }
}

pub fn build_physics_scene(world: &mut World, backend: &mut impl RenderBackend) {
    let (assets, mesh_bounds, material_blends, asset_previews) = load_library(backend);
    let cube = assets.mesh("cube").unwrap();
    let sphere = assets.mesh("sphere").unwrap();
    let ground = assets.material("ground").unwrap();
    let palette: Vec<_> = ["gold", "copper", "glossy", "clay"]
        .into_iter()
        .map(|name| assets.material(name).unwrap())
        .collect();
    let neon = assets.material("neon").unwrap();
    world.insert_resource(assets);
    world.insert_resource(mesh_bounds);
    world.insert_resource(material_blends);
    world.insert_resource(asset_previews);

    let (unit_box, unit_ball) = (unit_box(), unit_ball());

    let floor = spawn_mesh(
        world,
        "Floor",
        Transform {
            translation: Vec3::new(0.0, -0.25, 0.0),
            scale: Vec3::new(40.0, 0.5, 40.0),
            ..Default::default()
        },
        cube,
        ground,
    );
    world.insert(floor, unit_box);

    // A sphere has no rolling resistance, so a ball knocked off the stack rolls
    // until something stops it — a floor's edge, without these.
    for (index, (centre, scale)) in [
        (Vec3::new(0.0, 0.5, -14.0), Vec3::new(28.5, 1.0, 0.5)),
        (Vec3::new(0.0, 0.5, 14.0), Vec3::new(28.5, 1.0, 0.5)),
        (Vec3::new(-14.0, 0.5, 0.0), Vec3::new(0.5, 1.0, 28.5)),
        (Vec3::new(14.0, 0.5, 0.0), Vec3::new(0.5, 1.0, 28.5)),
    ]
    .into_iter()
    .enumerate()
    {
        let wall = spawn_mesh(
            world,
            format!("Wall {index}"),
            Transform {
                translation: centre,
                scale,
                ..Default::default()
            },
            cube,
            ground,
        );
        world.insert(wall, unit_box);
    }

    // A hair of air between neighbours, so the stack starts at rest rather
    // than resolving a row of exact contacts on its first step.
    let pitch = 1.02;
    let mut index = 0;
    for row in 0..PYRAMID_BASE {
        let count = PYRAMID_BASE - row;
        let start = -((count - 1) as f32) * pitch * 0.5;
        for column in 0..count {
            let entity = spawn_mesh(
                world,
                format!("Block {index}"),
                Transform::from_translation(Vec3::new(
                    start + column as f32 * pitch,
                    0.5 + row as f32,
                    0.0,
                )),
                cube,
                palette[index % palette.len()],
            );
            world.insert(entity, unit_box);
            world.insert(entity, RigidBody::default());
            index += 1;
        }
    }

    for (index, (x, height)) in [(-1.5f32, 9.0f32), (0.3, 12.0), (1.2, 15.0), (-0.4, 18.0)]
        .into_iter()
        .enumerate()
    {
        let entity = spawn_mesh(
            world,
            format!("Ball {index}"),
            Transform::from_translation(Vec3::new(x, height, 0.2)),
            sphere,
            palette[(index + 1) % palette.len()],
        );
        world.insert(entity, unit_ball);
        world.insert(entity, ball_body());
    }

    let paddle = spawn_mesh(
        world,
        "Paddle",
        Transform {
            translation: Vec3::new(0.0, 0.5, 7.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(9.0, 1.0, 0.4),
        },
        cube,
        neon,
    );
    world.insert(paddle, unit_box);
    world.insert(paddle, Spin::new(Vec3::Y, 0.8));

    for (index, x) in [-3.0f32, -1.5, 1.5, 3.0].into_iter().enumerate() {
        let entity = spawn_mesh(
            world,
            format!("Skittle {index}"),
            Transform {
                translation: Vec3::new(x, 0.75, 9.5),
                rotation: Quat::IDENTITY,
                scale: Vec3::new(0.5, 1.5, 0.5),
            },
            cube,
            palette[index % palette.len()],
        );
        world.insert(entity, unit_box);
        world.insert(entity, RigidBody::default());
    }

    spawn_controls(world);

    super::spawn_built_in_sun(world);
    let (sky, width, height) = super::built_in_sky();
    backend.load_environment(&sky, width, height);

    world.insert_resource(Camera {
        position: Vec3::new(9.0, 7.0, 20.0),
        target: Vec3::new(0.0, 2.0, 3.0),
        ..Camera::default()
    });
}

/// The panel's palette: one raised surface, two text colours, and the button
/// fill's three states. sRGB, straight alpha.
const SURFACE: [f32; 4] = [0.11, 0.12, 0.14, 0.92];
const TEXT: [f32; 4] = [0.93, 0.94, 0.96, 1.0];
const TEXT_SECONDARY: [f32; 4] = [0.64, 0.67, 0.72, 1.0];
const FILL: Button = Button {
    normal: [0.22, 0.24, 0.28, 1.0],
    hover: [0.28, 0.31, 0.36, 1.0],
    pressed: [0.17, 0.19, 0.22, 1.0],
    disabled: [0.16, 0.17, 0.20, 1.0],
    interactable: true,
};
/// Clear is the one destructive control, so it carries no fill until hovered
/// and sits apart from the three that add.
const QUIET: Button = Button {
    normal: SURFACE,
    hover: [0.22, 0.24, 0.28, 1.0],
    pressed: [0.17, 0.19, 0.22, 1.0],
    disabled: SURFACE,
    interactable: false,
};

const PANEL_WIDTH: f32 = 232.0;
const PADDING: f32 = 16.0;
const BUTTON_HEIGHT: f32 = 44.0;
const GAP: f32 = 8.0;

/// A row across the panel's inner width, its top `y` units below the panel's.
fn row(y: f32, height: f32) -> RectTransform {
    RectTransform {
        anchor_min: Vec2::new(0.0, 0.0),
        anchor_max: Vec2::new(1.0, 0.0),
        pivot: Vec2::new(0.5, 0.0),
        anchored_position: Vec2::new(0.0, y),
        size_delta: Vec2::new(-2.0 * PADDING, height),
        scale: Vec2::ONE,
    }
}

fn spawn_controls(world: &mut World) {
    let canvas = world.spawn();
    world.insert(canvas, Name::new("Drop panel canvas"));
    world.insert(
        canvas,
        Canvas {
            scale_mode: ScaleMode::ScaleWithScreen,
            match_width_or_height: 1.0,
            ..Canvas::default()
        },
    );

    let drops = [Control::Crate, Control::Ball, Control::Tower];
    let clear_top = PADDING + drops.len() as f32 * (BUTTON_HEIGHT + GAP) - GAP + 2.0 * GAP;
    let count_top = clear_top + 36.0 + GAP;
    let height = count_top + 20.0 + PADDING;

    let panel = world.spawn();
    world.insert(panel, Name::new("Drop panel"));
    world.insert(
        panel,
        RectTransform {
            anchor_min: Vec2::ZERO,
            anchor_max: Vec2::ZERO,
            pivot: Vec2::ZERO,
            anchored_position: Vec2::splat(24.0),
            size_delta: Vec2::new(PANEL_WIDTH, height),
            scale: Vec2::ONE,
        },
    );
    world.insert(
        panel,
        Image {
            color: SURFACE,
            ..Image::default()
        },
    );
    world.insert(
        panel,
        Corner {
            radius: 12.0,
            ..Corner::default()
        },
    );
    world.insert(
        panel,
        Shadow {
            color: [0.0, 0.0, 0.0, 0.35],
            offset: Vec2::new(0.0, 6.0),
            blur: 18.0,
            spread: 0.0,
        },
    );
    attach(world, panel, canvas);

    for (index, control) in drops.into_iter().enumerate() {
        let y = PADDING + index as f32 * (BUTTON_HEIGHT + GAP);
        button(world, panel, control, row(y, BUTTON_HEIGHT), FILL, TEXT);
    }
    button(
        world,
        panel,
        Control::Clear,
        row(clear_top, 36.0),
        QUIET,
        TEXT_SECONDARY,
    );

    let count = world.spawn();
    world.insert(count, Name::new("Dropped count"));
    world.insert(count, Tag::new(COUNT_LABEL));
    world.insert(count, row(count_top, 20.0));
    world.insert(
        count,
        Text {
            text: count_text(0),
            size: 14.0,
            color: TEXT_SECONDARY,
            anchor: TextAnchor::MiddleCenter,
            ..Text::default()
        },
    );
    attach(world, count, panel);
}

fn button(
    world: &mut World,
    panel: Entity,
    control: Control,
    rect: RectTransform,
    states: Button,
    text_color: [f32; 4],
) {
    let entity = world.spawn();
    world.insert(entity, Name::new(control.label()));
    world.insert(entity, Tag::new(control.tag()));
    world.insert(entity, rect);
    world.insert(entity, Image::default());
    world.insert(entity, states);
    world.insert(
        entity,
        Corner {
            radius: 8.0,
            ..Corner::default()
        },
    );
    attach(world, entity, panel);

    let label = world.spawn();
    world.insert(label, Name::new("Label"));
    world.insert(label, RectTransform::stretched());
    world.insert(
        label,
        Text {
            text: control.label().into(),
            size: 17.0,
            color: text_color,
            anchor: TextAnchor::MiddleCenter,
            ..Text::default()
        },
    );
    attach(world, label, entity);
}

fn attach(world: &mut World, child: Entity, parent: Entity) {
    reparent(world, child, Some(parent), false)
        .expect("a new entity has no children to cycle through");
}

fn count_text(count: usize) -> String {
    match count {
        0 => "Nothing dropped yet".into(),
        MAX_SPAWNED.. => format!("{MAX_SPAWNED} dropped, clear to drop more"),
        count => format!("{count} dropped"),
    }
}

/// Act on this tick's clicks on the drop panel. Reads [`UiEvents`] without
/// draining it, so it runs between the hit test and the script tick.
pub fn spawn_from_clicks(world: &mut World) {
    let mut clicked = Vec::new();
    if let Some(events) = world.get_resource::<UiEvents>() {
        for event in events.events() {
            if event.kind != PointerKind::Click {
                continue;
            }
            let control = events.chain(event).iter().find_map(|&entity| {
                world
                    .get::<Tag>(entity)
                    .and_then(|tag| Control::from_tag(tag.as_str()))
            });
            clicked.extend(control);
        }
    }
    if clicked.is_empty() {
        return;
    }

    for control in clicked {
        let dropped = tagged(world, SPAWNED);
        let room = MAX_SPAWNED.saturating_sub(dropped.len());
        match control {
            Control::Clear => {
                for entity in dropped {
                    despawn_recursive(world, entity);
                }
            }
            Control::Crate if room > 0 => drop_crates(world, dropped.len(), 1, true),
            Control::Tower => drop_crates(world, dropped.len(), TOWER.min(room), false),
            Control::Ball if room > 0 => drop_ball(world, dropped.len()),
            Control::Crate | Control::Ball => {}
        }
    }
    refresh_controls(world);
}

fn tagged(world: &World, tag: &str) -> Vec<Entity> {
    let mut found = Vec::new();
    world.query::<&Tag>().for_each(|entity, t| {
        if t.as_str() == tag {
            found.push(entity);
        }
    });
    found
}

/// The `index`th drop's spot on a sunflower spiral over the stack, so a run
/// of drops spreads out instead of landing on itself.
fn drop_point(index: usize) -> (Vec3, f32) {
    const GOLDEN_ANGLE: f32 = 2.399_963;
    let angle = index as f32 * GOLDEN_ANGLE;
    let radius = 3.0 * ((index % 24) as f32 / 24.0).sqrt();
    (
        Vec3::new(radius * angle.cos(), 0.0, radius * angle.sin()),
        angle,
    )
}

/// `count` crates, one above the other. A lone crate starts tilted so it
/// lands on an edge and tumbles; a tower's stay square, or it is a pile.
fn drop_crates(world: &mut World, index: usize, count: usize, tilted: bool) {
    let (spot, angle) = drop_point(index);
    let (cube, material) = {
        let assets = world.resource::<Assets>();
        (assets.mesh("cube").unwrap(), palette(&assets, index))
    };
    let rotation = if tilted {
        Quat::from_euler(EulerRot::YXZ, angle, 0.5, 0.3)
    } else {
        Quat::from_rotation_y(angle)
    };
    for level in 0..count {
        let entity = spawn_mesh(
            world,
            "Dropped crate",
            Transform {
                translation: spot + Vec3::Y * (if tilted { 10.0 } else { 6.0 } + level as f32),
                rotation,
                scale: Vec3::ONE,
            },
            cube,
            material,
        );
        world.insert(entity, unit_box());
        world.insert(entity, RigidBody::default());
        world.insert(entity, Tag::new(SPAWNED));
    }
}

fn drop_ball(world: &mut World, index: usize) {
    let (spot, _) = drop_point(index);
    let (sphere, material) = {
        let assets = world.resource::<Assets>();
        (assets.mesh("sphere").unwrap(), palette(&assets, index + 1))
    };
    let entity = spawn_mesh(
        world,
        "Dropped ball",
        Transform::from_translation(spot + Vec3::Y * 12.0),
        sphere,
        material,
    );
    world.insert(entity, unit_ball());
    world.insert(entity, ball_body());
    world.insert(entity, Tag::new(SPAWNED));
}

fn palette(assets: &Assets, index: usize) -> crate::scene::MaterialHandle {
    let names = ["gold", "copper", "glossy", "clay"];
    assets.material(names[index % names.len()]).unwrap()
}

/// Bring the buttons and the count in line with what is dropped.
fn refresh_controls(world: &mut World) {
    let count = tagged(world, SPAWNED).len();
    for control in Control::ALL {
        let interactable = match control {
            Control::Clear => count > 0,
            _ => count < MAX_SPAWNED,
        };
        for entity in tagged(world, control.tag()) {
            if let Some(mut button) = world.get_mut::<Button>(entity) {
                button.interactable = interactable;
            }
        }
    }
    for entity in tagged(world, COUNT_LABEL) {
        if let Some(mut text) = world.get_mut::<Text>(entity) {
            text.text = count_text(count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{GameScreen, Hierarchy, ensure_current};
    use crate::ui::events::{PointerInput, UiPointer, step};
    use crate::ui::{Layout, layout};

    fn scene() -> World {
        let mut world = World::new();
        build_physics_scene(&mut world, &mut crate::gfx::HeadlessBackend::new());
        world
    }

    /// Press and release on the control, through the same layout and hit
    /// test the game's pointer goes through, then run the handler.
    fn click(world: &mut World, control: Control) {
        ensure_current(world);
        let mut layout = Layout::default();
        layout::resolve(
            world,
            &world.resource::<Hierarchy>(),
            &GameScreen::window([1600, 1200]),
            &mut layout,
        );
        let button = tagged(world, control.tag())[0];
        let rect = layout.elements()[layout.index_of(button).unwrap()].rect;
        let position = [rect.x + rect.width * 0.5, rect.y + rect.height * 0.5];

        let mut pointer = UiPointer::default();
        let mut events = UiEvents::default();
        for (pressed, released) in [(true, false), (false, true)] {
            events.clear();
            step(
                world,
                &layout,
                &mut pointer,
                PointerInput {
                    position,
                    pressed,
                    released,
                },
                &mut events,
            );
        }
        world.insert_resource(events);
        spawn_from_clicks(world);
    }

    fn label(world: &World) -> String {
        let entity = tagged(world, COUNT_LABEL)[0];
        world.get::<Text>(entity).unwrap().text.clone()
    }

    fn interactable(world: &World, control: Control) -> bool {
        let entity = tagged(world, control.tag())[0];
        world.get::<Button>(entity).unwrap().interactable
    }

    #[test]
    fn dropping_a_ball_adds_one_simulated_ball() {
        let mut world = scene();

        click(&mut world, Control::Ball);

        let dropped = tagged(&world, SPAWNED);
        assert_eq!(dropped.len(), 1);
        let ball = dropped[0];
        assert_eq!(world.get::<Collider>(ball).as_deref(), Some(&unit_ball()));
        assert!(world.get::<RigidBody>(ball).is_some_and(|b| !b.kinematic));
        assert_eq!(label(&world), "1 dropped");
        assert!(interactable(&world, Control::Clear));
    }

    #[test]
    fn dropping_a_tower_stacks_five_crates_in_one_column() {
        let mut world = scene();

        click(&mut world, Control::Tower);

        let dropped = tagged(&world, SPAWNED);
        assert_eq!(dropped.len(), TOWER);
        let mut heights: Vec<Vec3> = dropped
            .iter()
            .map(|&e| {
                world
                    .get::<crate::scene::LocalTransform>(e)
                    .unwrap()
                    .translation
            })
            .collect();
        heights.sort_by(|a, b| a.y.total_cmp(&b.y));
        for pair in heights.windows(2) {
            assert!((pair[1].y - pair[0].y - 1.0).abs() < 1e-3, "{heights:?}");
            assert_eq!(pair[0].x, pair[1].x);
        }
    }

    /// Clear takes back what the panel dropped and nothing the scene authored.
    #[test]
    fn clear_removes_only_what_was_dropped() {
        let mut world = scene();
        let authored = tagged(&world, SPAWNED).len() + world.query::<&RigidBody>().count();

        click(&mut world, Control::Crate);
        click(&mut world, Control::Crate);
        assert_eq!(tagged(&world, SPAWNED).len(), 2);
        click(&mut world, Control::Clear);

        assert!(tagged(&world, SPAWNED).is_empty());
        assert_eq!(world.query::<&RigidBody>().count(), authored);
        assert_eq!(label(&world), "Nothing dropped yet");
        assert!(!interactable(&world, Control::Clear));
    }

    /// Stop rebuilds the scene from its saved document, so the panel has to
    /// survive one: everything it is found by has to be saved with it.
    #[test]
    fn the_panel_still_drops_after_a_save_and_a_load() {
        let mut registry = orrin_registry::Registry::new();
        crate::scene::registry::register_components(&mut registry);
        let mut world = scene();
        let text = crate::scene::save(&mut world, &registry);
        let everything: Vec<Entity> = world.entities().collect();
        for entity in everything {
            world.despawn(entity);
        }
        crate::scene::load(&text, &mut world, &registry).unwrap();

        click(&mut world, Control::Ball);

        assert_eq!(tagged(&world, SPAWNED).len(), 1);
    }

    /// At the cap the drop buttons switch off, a tower that would cross it
    /// stops at it, and the label says what to do about it.
    #[test]
    fn dropping_stops_at_the_cap_until_something_is_cleared() {
        let mut world = scene();
        for _ in 0..MAX_SPAWNED.div_ceil(TOWER) + 2 {
            click(&mut world, Control::Tower);
        }

        assert_eq!(tagged(&world, SPAWNED).len(), MAX_SPAWNED);
        assert!(!interactable(&world, Control::Crate));
        assert!(!interactable(&world, Control::Tower));
        assert_eq!(
            label(&world),
            format!("{MAX_SPAWNED} dropped, clear to drop more")
        );

        click(&mut world, Control::Clear);
        assert!(interactable(&world, Control::Ball));
    }
}
