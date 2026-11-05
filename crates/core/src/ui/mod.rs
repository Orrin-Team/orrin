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

pub mod components;
pub mod layout;

pub use components::{Button, Canvas, Image, RectTransform, ScaleMode, Text};
pub use layout::{Element, Layout};

#[cfg(test)]
mod tests {
    use glam::Vec2;
    use orrin_ecs::{Entity, World};
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
