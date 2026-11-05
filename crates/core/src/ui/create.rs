use glam::Vec2;
use orrin_ecs::{Entity, World};

use super::{Button, Canvas, Image, RectTransform, Text};
use crate::scene::{Name, Parent, TextAnchor, reparent};

/// What the editor's Create ▸ UI menu makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiKind {
    Canvas,
    Panel,
    Image,
    Text,
    Button,
}

impl UiKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Canvas => "Canvas",
            Self::Panel => "Panel",
            Self::Image => "Image",
            Self::Text => "Text",
            Self::Button => "Button",
        }
    }
}

/// What [`create`] made: the element to select, and the root of everything it
/// spawned — the element, or a canvas made to hold it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Created {
    pub element: Entity,
    pub root: Entity,
}

/// Make a `kind` of element, as uGUI's menu does: under `under` when that is a
/// canvas or an element on one, else under the scene's first canvas, else under
/// a canvas made for it. A canvas is always a new root.
pub fn create(world: &mut World, kind: UiKind, under: Option<Entity>) -> Created {
    if kind == UiKind::Canvas {
        let canvas = canvas(world);
        return Created {
            element: canvas,
            root: canvas,
        };
    }
    let (parent, root) = match under
        .filter(|&e| on_a_canvas(world, e))
        .or_else(|| first_canvas(world))
    {
        Some(parent) => (parent, None),
        None => {
            let made = canvas(world);
            (made, Some(made))
        }
    };
    let element = match kind {
        UiKind::Canvas => unreachable!("made above"),
        UiKind::Panel => element(
            world,
            parent,
            kind,
            RectTransform::stretched(),
            |world, entity| {
                world.insert(
                    entity,
                    Image {
                        color: [1.0, 1.0, 1.0, 0.39],
                        ..Image::default()
                    },
                );
            },
        ),
        UiKind::Image => element(
            world,
            parent,
            kind,
            RectTransform::default(),
            |world, entity| {
                world.insert(entity, Image::default());
            },
        ),
        UiKind::Text => element(world, parent, kind, sized(160.0, 30.0), |world, entity| {
            world.insert(
                entity,
                Text {
                    text: "New Text".into(),
                    ..Text::default()
                },
            );
        }),
        UiKind::Button => {
            let button = element(world, parent, kind, sized(160.0, 30.0), |world, entity| {
                world.insert(entity, Image::default());
                world.insert(entity, Button::default());
            });
            element(
                world,
                button,
                UiKind::Text,
                RectTransform::stretched(),
                |world, entity| {
                    world.insert(
                        entity,
                        Text {
                            text: "Button".into(),
                            size: 20.0,
                            anchor: TextAnchor::MiddleCenter,
                            ..Text::default()
                        },
                    );
                },
            );
            button
        }
    };
    Created {
        element,
        root: root.unwrap_or(element),
    }
}

fn sized(width: f32, height: f32) -> RectTransform {
    RectTransform {
        size_delta: Vec2::new(width, height),
        ..RectTransform::default()
    }
}

fn canvas(world: &mut World) -> Entity {
    let entity = world.spawn();
    world.insert(entity, Name::new(UiKind::Canvas.name()));
    world.insert(entity, Canvas::default());
    entity
}

fn element(
    world: &mut World,
    parent: Entity,
    kind: UiKind,
    rect: RectTransform,
    draws: impl FnOnce(&mut World, Entity),
) -> Entity {
    let entity = world.spawn();
    world.insert(entity, Name::new(kind.name()));
    world.insert(entity, rect);
    draws(world, entity);
    reparent(world, entity, Some(parent), false)
        .expect("a new entity has no children to cycle through");
    entity
}

/// Whether `entity` is a canvas or has one above it.
fn on_a_canvas(world: &World, entity: Entity) -> bool {
    let mut cursor = Some(entity);
    // Bounded: a cycle the hierarchy has not broken yet would otherwise spin.
    for _ in 0..4096 {
        let Some(at) = cursor else {
            return false;
        };
        if world.has::<Canvas>(at) {
            return true;
        }
        cursor = world.get::<Parent>(at).map(|parent| parent.get());
    }
    false
}

/// The scene's first root canvas, in hierarchy order.
fn first_canvas(world: &mut World) -> Option<Entity> {
    crate::scene::ensure_current(world);
    let hierarchy = world.resource::<crate::scene::Hierarchy>();
    let mut found: Vec<Entity> = Vec::new();
    world.query::<&Canvas>().for_each(|entity, _| {
        if world.get::<Parent>(entity).is_none() {
            found.push(entity);
        }
    });
    found
        .into_iter()
        .min_by_key(|&entity| hierarchy.position(entity).unwrap_or(usize::MAX))
}
