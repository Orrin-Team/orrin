//! The in-game UI's CPU cost per frame: laying a HUD's canvas out into quads,
//! once its glyphs are in the atlas — the steady state, which is every frame
//! after the first few a HUD is on screen.
//!
//! The canvas is a busy HUD rather than a minimal one: forty labels of mixed
//! sizes and anchors, wrapped and clipped, on twenty panels.
//!
//! The same HUD built as a retained tree measures what the scene costs before
//! the painter sees it: resolving every element's rectangle and recording its
//! commands, which `ui::update` does every frame.
//!
//! The styled HUD is that tree with every modifier on every panel — a corner, a
//! shadow, a stroke and a gradient — an outline on every heading, and the
//! panels in a scroll frame, so the cost of the styling is read against the
//! plain HUD's in the same run.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

use orrin_core::gfx::TextureHandle;
use orrin_core::gfx::ui::{Painter, UiAssets};
use orrin_core::scene::{Rect, TextAnchor, TextStyle, UiCanvas, reparent};
use orrin_core::ui::{
    Canvas, Corner, Gradient, Image, RectTransform, ScrollFrame, Shadow, Stroke, StrokePlacement,
    Text,
};
use orrin_ecs::World;
use orrin_registry::AssetId;

struct NoAssets;

impl UiAssets for NoAssets {
    fn texture(&mut self, _: AssetId) -> Option<TextureHandle> {
        None
    }

    fn font(&mut self, _: AssetId) -> Result<Vec<u8>, String> {
        Err(String::new())
    }
}

fn hud() -> UiCanvas {
    let mut canvas = UiCanvas::default();
    canvas.begin([1920, 1080]);
    for i in 0..20 {
        let row = (i / 4) as f32;
        let column = (i % 4) as f32;
        let panel = Rect::new(20.0 + column * 470.0, 20.0 + row * 210.0, 450.0, 190.0);
        canvas.panel(panel, [12, 14, 20, 200]);
        let heading = TextStyle {
            font: None,
            size: 28.0,
            anchor: TextAnchor::UpperLeft,
            color: [255; 4],
            wrap: true,
        };
        canvas.text(
            Rect::new(panel.x + 12.0, panel.y + 8.0, 426.0, 40.0),
            heading,
            &format!("Objective {i}: hold the line"),
        );
        let body = TextStyle {
            size: 18.0,
            anchor: TextAnchor::MiddleCenter,
            color: [230, 210, 160, 255],
            ..heading
        };
        canvas.text(
            Rect::new(panel.x + 12.0, panel.y + 52.0, 426.0, 126.0),
            body,
            "Defend the relay until the convoy arrives. Ammunition is limited, \
             so pick your shots; reinforcements reach the east gate in 3:20.",
        );
    }
    canvas
}

fn paint(c: &mut Criterion) {
    let canvas = hud();
    let mut painter = Painter::new();
    painter.paint(&canvas, &mut NoAssets);
    let quads = painter.frame().quads.len() as u64;

    let mut group = c.benchmark_group("ui");
    group.throughput(Throughput::Elements(quads));
    group.bench_function("paint a cached hud", |b| {
        b.iter(|| painter.paint(black_box(&canvas), &mut NoAssets));
    });
    group.finish();
}

/// [`hud`] as entities: a canvas, twenty panels, two labels on each.
fn retained_hud() -> World {
    let mut world = World::new();
    let canvas = world.spawn();
    world.insert(canvas, Canvas::default());
    let child = |world: &mut World, parent, rect: RectTransform| {
        let entity = world.spawn();
        world.insert(entity, rect);
        reparent(world, entity, Some(parent), false).unwrap();
        entity
    };
    for i in 0..20 {
        let row = (i / 4) as f32;
        let column = (i % 4) as f32;
        let panel = child(
            &mut world,
            canvas,
            RectTransform {
                anchor_min: glam::Vec2::ZERO,
                anchor_max: glam::Vec2::ZERO,
                pivot: glam::Vec2::ZERO,
                anchored_position: glam::Vec2::new(20.0 + column * 470.0, 20.0 + row * 210.0),
                size_delta: glam::Vec2::new(450.0, 190.0),
                scale: glam::Vec2::ONE,
            },
        );
        world.insert(
            panel,
            Image {
                color: [0.05, 0.05, 0.08, 0.8],
                ..Image::default()
            },
        );
        let heading = child(
            &mut world,
            panel,
            RectTransform {
                anchor_max: glam::Vec2::new(1.0, 0.0),
                pivot: glam::Vec2::ZERO,
                size_delta: glam::Vec2::new(-24.0, 40.0),
                anchored_position: glam::Vec2::new(12.0, 8.0),
                ..RectTransform::stretched()
            },
        );
        world.insert(
            heading,
            Text {
                text: format!("Objective {i}: hold the line"),
                size: 28.0,
                color: [1.0; 4],
                ..Text::default()
            },
        );
        let body = child(
            &mut world,
            panel,
            RectTransform {
                size_delta: glam::Vec2::new(-24.0, -64.0),
                anchored_position: glam::Vec2::new(0.0, 20.0),
                ..RectTransform::stretched()
            },
        );
        world.insert(
            body,
            Text {
                text: "Defend the relay until the convoy arrives. Ammunition is limited, \
                       so pick your shots; reinforcements reach the east gate in 3:20."
                    .into(),
                size: 18.0,
                anchor: TextAnchor::MiddleCenter,
                ..Text::default()
            },
        );
    }
    world
}

fn update(c: &mut Criterion) {
    let mut world = retained_hud();
    orrin_core::ui::update(&mut world, [1920, 1080]);
    let elements = world
        .resource::<orrin_core::ui::UiTree>()
        .layout
        .elements()
        .len() as u64;
    let commands = world
        .resource::<orrin_core::ui::UiTree>()
        .canvas
        .commands()
        .len();
    assert_eq!(
        (elements, commands),
        (60, 60),
        "every panel and label is drawn"
    );

    let mut group = c.benchmark_group("ui");
    group.throughput(Throughput::Elements(elements));
    group.bench_function("resolve and record a retained hud", |b| {
        b.iter(|| orrin_core::ui::update(black_box(&mut world), [1920, 1080]));
    });
    group.finish();
}

/// [`retained_hud`] with every modifier: see the module docs.
fn styled_hud() -> World {
    let mut world = retained_hud();
    let (mut canvas, mut panels, mut headings) = (None, Vec::new(), Vec::new());
    world
        .query::<&Canvas>()
        .for_each(|entity, _| canvas = Some(entity));
    world
        .query::<&Image>()
        .for_each(|entity, _| panels.push(entity));
    world.query::<&Text>().for_each(|entity, text| {
        if text.size == 28.0 {
            headings.push(entity);
        }
    });
    let canvas = canvas.unwrap();
    let list = world.spawn();
    world.insert(list, RectTransform::stretched());
    world.insert(
        list,
        ScrollFrame {
            content_size: glam::Vec2::new(0.0, 1400.0),
            scroll_position: glam::Vec2::new(0.0, 120.0),
            ..ScrollFrame::default()
        },
    );
    reparent(&mut world, list, Some(canvas), false).unwrap();
    for panel in panels {
        reparent(&mut world, panel, Some(list), false).unwrap();
        world.insert(
            panel,
            Corner {
                radius: 16.0,
                ..Corner::default()
            },
        );
        world.insert(panel, Shadow::default());
        world.insert(
            panel,
            Stroke {
                placement: StrokePlacement::Outside,
                ..Stroke::default()
            },
        );
        world.insert(panel, Gradient::default());
    }
    for heading in headings {
        world.insert(heading, Stroke::default());
    }
    world
}

fn styled(c: &mut Criterion) {
    let mut world = styled_hud();
    orrin_core::ui::update(&mut world, [1920, 1080]);
    let tree = world.resource::<orrin_core::ui::UiTree>();
    let mut painter = Painter::new();
    painter.paint(&tree.canvas, &mut NoAssets);
    let drawn = (painter.frame().quads.len() + painter.frame().shapes.len()) as u64;
    assert!(!painter.frame().shapes.is_empty(), "the panels are shapes");

    let mut group = c.benchmark_group("ui");
    group.throughput(Throughput::Elements(drawn));
    group.bench_function("paint a cached styled hud", |b| {
        b.iter(|| painter.paint(black_box(&tree.canvas), &mut NoAssets));
    });
    group.finish();
    drop(tree);

    let elements = world
        .resource::<orrin_core::ui::UiTree>()
        .layout
        .elements()
        .len() as u64;
    let mut group = c.benchmark_group("ui");
    group.throughput(Throughput::Elements(elements));
    group.bench_function("resolve and record a styled retained hud", |b| {
        b.iter(|| orrin_core::ui::update(black_box(&mut world), [1920, 1080]));
    });
    group.finish();
}

/// A thousand elements each sliding and fading forever: two tweens apiece,
/// so the cost is read per tween. The empty case is what every frame with no
/// UI animation pays.
fn tweens(c: &mut Criterion) {
    use orrin_core::ui::tween::{Tween, TweenProperty, UiTweens, advance};

    let mut world = World::new();
    world.insert_resource(UiTweens::default());
    let mut group = c.benchmark_group("ui");
    group.bench_function("advance no tweens", |b| {
        b.iter(|| advance(black_box(&world), 1.0 / 60.0));
    });
    for _ in 0..1000 {
        let element = world.spawn();
        world.insert(element, RectTransform::default());
        world.insert(element, Image::default());
        let mut tweens = world.resource_mut::<UiTweens>();
        for (property, to) in [
            (TweenProperty::Position, [100.0, 50.0, 0.0, 0.0]),
            (TweenProperty::ImageColor, [1.0, 0.5, 0.0, 0.5]),
        ] {
            tweens.play(Tween {
                repeat: -1,
                reverses: true,
                ..Tween::new(element, property, to, 0.5)
            });
        }
    }
    group.throughput(Throughput::Elements(2000));
    group.bench_function("advance 2000 tweens", |b| {
        b.iter(|| advance(black_box(&world), 1.0 / 60.0));
    });
    group.finish();
}

criterion_group!(benches, paint, update, styled, tweens);
criterion_main!(benches);
