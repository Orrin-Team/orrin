//! The retained in-game UI: trees of entities under a [`Canvas`], laid out by
//! each element's [`RectTransform`] — uGUI's model, in this engine's top-left,
//! y-down screen frame.
//!
//! # Entities, not a document
//!
//! A UI element is an entity with a `RectTransform` and whatever draws on it —
//! an [`Image`], a [`Text`], a [`Button`] tinting the image. So a menu is saved
//! in the scene, made a prefab, edited with undo and driven from C# exactly as
//! a crate is, with no second format beside the scene's. The hierarchy *is* the
//! UI tree, and its child order is the draw order.
//!
//! An element carries a `RectTransform` and never a `LocalTransform`: the
//! transform propagation treats it as an organisational node, which is what
//! keeps the 3D systems — culling, picking, the gizmo — from mistaking a
//! button for something standing at the world's origin.
//!
//! # Layout
//!
//! [`layout::resolve`] walks each root canvas depth-first and turns every
//! element into a rectangle in physical pixels, which is what the painter
//! draws, the pointer hits and the editor's rect tool grabs. A root canvas is
//! the game's screen, [`GameScreen`]: the whole window — the space the 3D
//! camera draws into and the cursor is reported in — so a HUD sits where it
//! will in the shipped game, unless the editor is previewing a resolution, when
//! the tree is laid out at that size and placed inside the viewport. The canvas scaler maps canvas units to pixels once per canvas;
//! below it everything is units, and an element's `scale` multiplies its own
//! frame and its subtree's about its pivot. There is no rotation: the painter
//! draws axis-aligned quads, and the rect tool and the hit test lean on that.
//!
//! # One command stream, two layers
//!
//! [`update`] runs every frame, in edit mode too, and records the trees into a
//! [`UiCanvas`] of their own — the command record `UI.*` calls write — which
//! the painter draws first, with the immediate canvas over it. Keeping the two
//! apart is what leaves the immediate canvas's rules alone: it is still emptied
//! only by a tick that runs game code, so a paused game keeps its HUD, while
//! the retained layer is rebuilt from the scene whenever it draws. Canvases
//! draw by `sort_order`, ties in hierarchy order, and within one an element's
//! image before its text before its children.
//!
//! A sliced image is nine image commands over parts of one texture, measured
//! in the texture's uploaded pixels, so the painter needs nothing new to draw
//! one and a frame of them still batches by texture.
//!
//! # Modifiers
//!
//! [`Gradient`], [`Corner`], [`Stroke`] and [`Shadow`] change how an element
//! draws without being anything it draws: they sit on the element's entity
//! beside its `Image` or `Text`, so restyling a button is an inspector edit and
//! a prefab override like any other. [`extract`] resolves them into pixels and
//! records each piece as its own command — the shadow under the image, the
//! stroke over it — with a style beside it; the painter's module says how a
//! style is drawn. A corner rounds the box and everything that follows it; a
//! gradient colours the image and the text; a stroke rings the box, or each
//! glyph of an element that is only text. A shadow is the box's alone, because
//! a glyph's would need a blurred copy of every glyph in the atlas.
//!
//! An element with no modifier, in a world that has none, is recorded exactly
//! as before there were any: the walk checks once a frame which of them the
//! world holds at all, and it and the recording skip every per-element lookup
//! for those it does not.
//!
//! # Scrolling
//!
//! A [`ScrollFrame`]'s children are anchored to a content area at least as
//! large as the frame and moved by its scroll, clamped as it is laid out, so a
//! script may write any position and see the nearest one there is. The content
//! is cut to the frame — and to every frame around it, the clips intersecting
//! — by the painter's own clipping, and the hit test cuts by the same
//! rectangle, so a child scrolled out of view is not under the pointer. The
//! clip is the frame's rectangle, not its `Corner`. Its bars draw over the
//! content once the frame's subtree is done, as pills.
//!
//! # Pointer events
//!
//! [`events::update`] hit-tests the pointer before the script tick against the
//! layout the last frame drew — what the player is looking at — and queues
//! uGUI's events: enter and exit to each element the pointer starts or stops
//! being inside, deepest first and not bubbling, and down, up and click from the
//! topmost raycast target, each offered up the hierarchy until a behaviour
//! handles it. Up goes to what was pressed, wherever the pointer is; a click is
//! a press and a release on the same element. A button that is not interactable
//! stops the pointer and lets nothing at or under it take part. Only a ticking
//! game gets events: edit mode and a paused game get none, and edit mode keeps
//! no pointer state, so a button tints only while the game runs.
//!
//! The wheel goes the same way, to the nearest scroll frame at or above the
//! topmost target that can still move — one at its end hands it to the frame
//! around it — and a frame stops the pointer anywhere in its viewport, so the
//! gaps between its rows still scroll it. A press on a bar is the bar's: it
//! drags the thumb until release, and nothing under it hears of the press.
//!
//! # Tweens and stories
//!
//! [`tween`] eases an element's fields over time as engine data, advanced
//! between the script tick and [`update`], so an animated element is laid out
//! where its tween put it that same frame. [`story`] is the storybook's model:
//! a piece of UI built by game code into a world of its own and drawn in a
//! window of its own, which is how a menu is looked at without the game around
//! it.

pub mod components;
pub mod create;
pub mod events;
pub mod extract;
pub mod layout;
pub mod story;
pub mod tween;

pub use components::{
    Button, Canvas, Corner, CornerStyle, Gradient, Image, RectTransform, ScaleMode, ScrollFrame,
    ScrollbarVisibility, Shadow, Stroke, StrokePlacement, Text,
};
pub use layout::{Element, Layout};

use orrin_ecs::World;

use crate::asset_db::AssetDatabase;
use crate::scene::{GameScreen, UiCanvas};

/// The retained UI as last resolved: a world resource, rebuilt by [`update`].
#[derive(Default)]
pub struct UiTree {
    pub layout: Layout,
    /// What the trees draw, under the immediate canvas.
    pub canvas: UiCanvas,
}

/// Resolve every canvas onto the game's screen — the [`GameScreen`] resource,
/// or a window of `extent` without one — and record what it draws. Runs every
/// frame, after the last thing that can edit the scene.
pub fn update(world: &mut World, extent: [u32; 2]) {
    crate::scene::ensure_current(world);
    if world.get_resource::<UiTree>().is_none() {
        world.insert_resource(UiTree::default());
    }
    let mut tree = world.resource_mut::<UiTree>();
    let UiTree { layout, canvas } = &mut *tree;
    let screen = world
        .get_resource::<GameScreen>()
        .map_or_else(|| GameScreen::window(extent), |screen| *screen);
    layout::resolve(
        world,
        &world.resource::<crate::scene::Hierarchy>(),
        &screen,
        layout,
    );
    canvas.begin(screen.size.map(|side| side.round() as u32));
    if layout.is_empty() {
        return;
    }
    let db = world.get_resource::<AssetDatabase>();
    let extent = |id| db.as_ref().and_then(|db| db.texture_extent(id));
    extract::extract(world, layout, extent, canvas);
}

#[cfg(test)]
mod tests {
    use glam::Vec2;
    use orrin_ecs::Entity;
    use orrin_registry::{AssetId, Registry};

    use super::*;
    use crate::scene::{
        Name, Parent, Prefabs, TextAnchor, load, register_components, reparent, save,
    };

    fn registry() -> Registry {
        let mut registry = Registry::new();
        register_components(&mut registry);
        registry
    }

    fn named(world: &World, name: &str) -> Entity {
        world
            .entities()
            .find(|&e| world.get::<Name>(e).is_some_and(|n| n.0 == name))
            .unwrap_or_else(|| panic!("no entity named {name}"))
    }

    fn parent_of(world: &World, entity: Entity) -> Option<Entity> {
        world.get::<Parent>(entity).map(|p| p.get())
    }

    fn label() -> Text {
        Text {
            text: "Play".into(),
            font: Some(AssetId::new()),
            size: 32.0,
            anchor: TextAnchor::MiddleCenter,
            ..Text::default()
        }
    }

    fn backdrop() -> Image {
        Image {
            texture: Some(AssetId::new()),
            slice: [8.0, 8.0, 8.0, 8.0],
            ..Image::default()
        }
    }

    /// A menu: a canvas, a panel on it, a button on the panel, its label.
    fn menu(world: &mut World) -> Entity {
        let spawn = |world: &mut World, name: &str, parent: Option<Entity>| {
            let entity = world.spawn();
            world.insert(entity, Name::new(name));
            if let Some(parent) = parent {
                reparent(world, entity, Some(parent), false).unwrap();
            }
            entity
        };
        let canvas = spawn(world, "Canvas", None);
        world.insert(
            canvas,
            Canvas {
                scale_mode: ScaleMode::ScaleWithScreen,
                sort_order: 2,
                ..Canvas::default()
            },
        );
        let panel = spawn(world, "Panel", Some(canvas));
        world.insert(panel, RectTransform::stretched());
        world.insert(panel, backdrop());
        let button = spawn(world, "Button", Some(panel));
        world.insert(
            button,
            RectTransform {
                anchored_position: Vec2::new(0.0, 40.0),
                size_delta: Vec2::new(160.0, 30.0),
                ..RectTransform::default()
            },
        );
        world.insert(button, Image::default());
        world.insert(
            button,
            Button {
                interactable: false,
                ..Button::default()
            },
        );
        let text = spawn(world, "Label", Some(button));
        world.insert(text, RectTransform::stretched());
        world.insert(text, label());
        canvas
    }

    fn assert_menu(world: &World) {
        let (canvas, panel) = (named(world, "Canvas"), named(world, "Panel"));
        let (button, text) = (named(world, "Button"), named(world, "Label"));
        assert_eq!(parent_of(world, panel), Some(canvas));
        assert_eq!(parent_of(world, button), Some(panel));
        assert_eq!(parent_of(world, text), Some(button));
        assert_eq!(world.get::<Canvas>(canvas).unwrap().sort_order, 2);
        assert_eq!(
            *world.get::<RectTransform>(panel).unwrap(),
            RectTransform::stretched()
        );
        assert_eq!(
            world.get::<Image>(panel).unwrap().slice,
            [8.0, 8.0, 8.0, 8.0]
        );
        assert!(!world.get::<Button>(button).unwrap().interactable);
        assert_eq!(
            world.get::<RectTransform>(button).unwrap().size_delta,
            Vec2::new(160.0, 30.0)
        );
        let text = world.get::<Text>(text).unwrap();
        assert_eq!(text.text, "Play");
        assert_eq!(text.anchor, TextAnchor::MiddleCenter);
    }

    /// The retained layer is the scene's, rebuilt whatever the game is doing;
    /// the immediate one is the tick's, and `update` never touches it.
    #[test]
    fn an_update_rebuilds_the_retained_layer_and_leaves_the_immediate_one() {
        let mut world = World::new();
        menu(&mut world);
        let mut immediate = UiCanvas::default();
        immediate.begin([800, 600]);
        immediate.panel(crate::scene::Rect::new(0.0, 0.0, 5.0, 5.0), [255; 4]);
        world.insert_resource(immediate);

        update(&mut world, [1920, 1080]);
        update(&mut world, [1920, 1080]);
        let tree = world.resource::<UiTree>();
        assert_eq!(tree.layout.elements().len(), 3);
        // The panel's image, the button's, and the label.
        assert_eq!(
            tree.canvas.commands().len(),
            3,
            "{:?}",
            tree.canvas.commands()
        );
        assert_eq!(tree.canvas.screen(), [1920.0, 1080.0]);
        assert_eq!(world.resource::<UiCanvas>().commands().len(), 1);
    }

    /// What the world has is remembered between frames, and forgotten the
    /// moment its structure changes: a modifier or a scroll frame added to a
    /// tree that had none is drawn by the very next update.
    #[test]
    fn a_modifier_added_between_frames_is_drawn_the_next() {
        let mut world = World::new();
        menu(&mut world);
        update(&mut world, [1920, 1080]);
        assert!(world.resource::<UiTree>().canvas.styles().is_empty());

        let button = named(&world, "Button");
        world.insert(button, Corner::default());
        update(&mut world, [1920, 1080]);
        assert_eq!(world.resource::<UiTree>().canvas.styles().len(), 1);

        let panel = named(&world, "Panel");
        world.insert(panel, ScrollFrame::default());
        update(&mut world, [1920, 1080]);
        assert_eq!(world.resource::<UiTree>().layout.scrolls().len(), 1);
        world.remove::<Corner>(button);
        update(&mut world, [1920, 1080]);
        assert!(world.resource::<UiTree>().canvas.styles().is_empty());
    }

    #[test]
    fn a_ui_tree_survives_a_save_and_a_load() {
        let registry = registry();
        let mut world = World::new();
        menu(&mut world);
        let text = save(&mut world, &registry);
        for id in [
            "orrin.ui.canvas",
            "orrin.ui.rect_transform",
            "orrin.ui.image",
            "orrin.ui.text",
            "orrin.ui.button",
        ] {
            assert!(text.contains(id), "{id} missing from:\n{text}");
        }

        let mut loaded = World::new();
        let issues = load(&text, &mut loaded, &registry).unwrap();
        assert!(issues.is_empty(), "{issues:?}");
        assert_menu(&loaded);
    }

    /// Every modifier is saved under its own id and read back as written.
    #[test]
    fn a_styled_tree_survives_a_save_and_a_load() {
        let registry = registry();
        let mut world = World::new();
        menu(&mut world);
        let button = named(&world, "Button");
        let panel = named(&world, "Panel");
        let gradient = Gradient {
            rotation: 90.0,
            ..Gradient::default()
        };
        let corner = Corner {
            style: CornerStyle::Squircle,
            radius_fraction: 0.5,
            ..Corner::default()
        };
        let stroke = Stroke {
            placement: StrokePlacement::Outside,
            ..Stroke::default()
        };
        let shadow = Shadow {
            spread: 3.0,
            ..Shadow::default()
        };
        let scroll = ScrollFrame {
            content_size: Vec2::new(0.0, 2000.0),
            scroll_position: Vec2::new(0.0, 50.0),
            ..ScrollFrame::default()
        };
        world.insert(button, gradient);
        world.insert(button, corner);
        world.insert(button, stroke);
        world.insert(button, shadow);
        world.insert(panel, scroll);
        let text = save(&mut world, &registry);
        for id in [
            "orrin.ui.gradient",
            "orrin.ui.corner",
            "orrin.ui.stroke",
            "orrin.ui.shadow",
            "orrin.ui.scroll_frame",
        ] {
            assert!(text.contains(id), "{id} missing from:\n{text}");
        }

        let mut loaded = World::new();
        let issues = load(&text, &mut loaded, &registry).unwrap();
        assert!(issues.is_empty(), "{issues:?}");
        assert_menu(&loaded);
        let (button, panel) = (named(&loaded, "Button"), named(&loaded, "Panel"));
        assert_eq!(*loaded.get::<Gradient>(button).unwrap(), gradient);
        assert_eq!(*loaded.get::<Corner>(button).unwrap(), corner);
        assert_eq!(*loaded.get::<Stroke>(button).unwrap(), stroke);
        assert_eq!(*loaded.get::<Shadow>(button).unwrap(), shadow);
        assert_eq!(*loaded.get::<ScrollFrame>(panel).unwrap(), scroll);
    }

    /// A scene saved before the modifiers existed names none of them, and
    /// loads as the tree it was, with nothing added.
    #[test]
    fn a_scene_saved_without_the_modifiers_still_loads() {
        let registry = registry();
        let mut world = World::new();
        menu(&mut world);
        let text = save(&mut world, &registry);
        assert!(!text.contains("orrin.ui.corner"), "{text}");

        let mut loaded = World::new();
        let issues = load(&text, &mut loaded, &registry).unwrap();
        assert!(issues.is_empty(), "{issues:?}");
        assert_menu(&loaded);
        let styled = loaded.entities().any(|e| {
            loaded.has::<Gradient>(e)
                || loaded.has::<Corner>(e)
                || loaded.has::<Stroke>(e)
                || loaded.has::<Shadow>(e)
                || loaded.has::<ScrollFrame>(e)
        });
        assert!(!styled);
    }

    #[test]
    fn a_ui_tree_round_trips_through_a_prefab() {
        let dir = crate::asset_db::tests::Dir::new("ui-prefab");
        let registry = registry();
        let mut world = World::new();
        let (prefabs, issues) = Prefabs::scan(dir.path());
        assert!(issues.is_empty(), "{issues:?}");
        world.insert_resource(prefabs);
        let canvas = menu(&mut world);

        let (_, id) =
            crate::scene::prefab::create(&mut world, &registry, &[canvas], "menu.oprefab").unwrap();
        let mut placed = World::new();
        let (prefabs, issues) = Prefabs::scan(dir.path());
        assert!(issues.is_empty(), "{issues:?}");
        placed.insert_resource(prefabs);
        let (_, issues) = crate::scene::prefab::instantiate(&mut placed, &registry, id).unwrap();
        assert!(issues.is_empty(), "{issues:?}");
        assert_menu(&placed);
    }
}
