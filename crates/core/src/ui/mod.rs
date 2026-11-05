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
//! the whole window — the space the 3D camera draws into and the cursor is
//! reported in — not the editor's viewport, so a HUD sits where it will in the
//! shipped game. The canvas scaler maps canvas units to pixels once per canvas;
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

pub mod components;
pub mod extract;
pub mod layout;

pub use components::{Button, Canvas, Image, RectTransform, ScaleMode, Text};
pub use layout::{Element, Layout};

use orrin_ecs::World;

use crate::asset_db::AssetDatabase;
use crate::scene::UiCanvas;

/// The retained UI as last resolved: a world resource, rebuilt by [`update`].
#[derive(Default)]
pub struct UiTree {
    pub layout: Layout,
    /// What the trees draw, under the immediate canvas.
    pub canvas: UiCanvas,
}

/// Resolve every canvas onto a `screen`-sized window and record what it draws.
/// Runs every frame, after the last thing that can edit the scene.
pub fn update(world: &mut World, screen: [u32; 2]) {
    crate::scene::ensure_current(world);
    if world.get_resource::<UiTree>().is_none() {
        world.insert_resource(UiTree::default());
    }
    let mut tree = world.resource_mut::<UiTree>();
    let UiTree { layout, canvas } = &mut *tree;
    let size = screen.map(|side| side as f32);
    layout::resolve(
        world,
        &world.resource::<crate::scene::Hierarchy>(),
        size,
        layout,
    );
    canvas.begin(screen);
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
            crate::scene::prefab::create(&mut world, &registry, &[canvas], "menu.fprefab").unwrap();
        let mut placed = World::new();
        let (prefabs, issues) = Prefabs::scan(dir.path());
        assert!(issues.is_empty(), "{issues:?}");
        placed.insert_resource(prefabs);
        let (_, issues) = crate::scene::prefab::instantiate(&mut placed, &registry, id).unwrap();
        assert!(issues.is_empty(), "{issues:?}");
        assert_menu(&placed);
    }
}
