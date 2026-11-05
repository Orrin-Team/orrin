use glam::Vec2;
use orrin_ecs::{Entity, World};

use super::{Canvas, Image, RectTransform, ScaleMode, Text};
use crate::scene::{Hierarchy, Parent, Rect};

/// One UI element, resolved onto the screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Element {
    pub entity: Entity,
    /// On screen, in physical pixels: what is drawn and what is hit.
    pub rect: Rect,
    /// The element above it in [`Layout::elements`], or `None` for a child of
    /// the canvas itself.
    pub parent: Option<u32>,
    /// Into [`Layout::canvases`].
    pub canvas: u32,
    /// Screen pixels per unit of the parent's frame: the canvas's scale times
    /// every ancestor's own. What turns a pointer's travel into the units
    /// `anchored_position` is in.
    pub units: Vec2,
    /// The element's own `scale`, which also scales its size on screen.
    pub scale: Vec2,
    /// The parent rectangle's size in its own frame, which the anchors are
    /// fractions of.
    pub parent_size: Vec2,
    /// Whether an `Image` or `Text` here stops the pointer.
    pub raycast: bool,
}

/// A root canvas, resolved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanvasRoot {
    pub entity: Entity,
    /// Screen pixels per canvas unit.
    pub scale: f32,
}

/// Every canvas's tree, resolved for one window size, in draw order.
#[derive(Default)]
pub struct Layout {
    screen: [f32; 2],
    canvases: Vec<CanvasRoot>,
    elements: Vec<Element>,
    /// The depth-first walk's stack, kept for its allocation.
    stack: Vec<Visit>,
}

#[derive(Clone, Copy)]
struct Visit {
    entity: Entity,
    parent: Option<u32>,
    canvas: u32,
    /// The parent's rectangle in its own frame.
    min: Vec2,
    size: Vec2,
    /// The parent frame's mapping to the screen: `offset + point * units`.
    offset: Vec2,
    units: Vec2,
}

impl Layout {
    pub fn screen(&self) -> [f32; 2] {
        self.screen
    }

    pub fn canvases(&self) -> &[CanvasRoot] {
        &self.canvases
    }

    /// Back to front.
    pub fn elements(&self) -> &[Element] {
        &self.elements
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn index_of(&self, entity: Entity) -> Option<usize> {
        self.elements
            .iter()
            .position(|element| element.entity == entity)
    }

    /// The rectangle element `index`'s anchors are fractions of, on screen:
    /// its parent's, or the window for a child of the canvas.
    pub fn parent_rect(&self, index: usize) -> Rect {
        match self.elements[index].parent {
            Some(parent) => self.elements[parent as usize].rect,
            None => Rect::new(0.0, 0.0, self.screen[0], self.screen[1]),
        }
    }

    /// The topmost element under `point`: the last one drawn there. With
    /// `raycast_only`, only an element that stops the pointer counts, and the
    /// rest let it through to what is behind.
    pub fn hit(&self, point: [f32; 2], raycast_only: bool) -> Option<usize> {
        self.elements.iter().rposition(|element| {
            (element.raycast || !raycast_only) && contains(element.rect, point)
        })
    }

    /// Whether `entity` is element `index` or one of its ancestors.
    pub fn is_at_or_above(&self, entity: Entity, index: usize) -> bool {
        let mut cursor = Some(index as u32);
        while let Some(at) = cursor {
            let element = &self.elements[at as usize];
            if element.entity == entity {
                return true;
            }
            cursor = element.parent;
        }
        false
    }
}

/// Half-open, so two elements that share an edge never both own the pixel on
/// it, and an element with no area owns none.
pub fn contains(rect: Rect, [x, y]: [f32; 2]) -> bool {
    x >= rect.x && y >= rect.y && x < rect.x + rect.width && y < rect.y + rect.height
}

/// Screen pixels per canvas unit for `canvas` on a `screen`-sized window.
///
/// `ScaleWithScreen` blends the width and height ratios in log space, as uGUI's
/// scaler does: a window twice as wide and the same height, matched halfway,
/// scales by the square root of two rather than by one and a half, so the
/// blend treats doubling and halving alike.
pub fn scale_factor(canvas: &Canvas, screen: [f32; 2]) -> f32 {
    match canvas.scale_mode {
        ScaleMode::ConstantPixel => 1.0,
        ScaleMode::ScaleWithScreen => {
            let reference = canvas.reference_resolution.max(Vec2::ONE);
            let ratio = Vec2::from(screen).max(Vec2::ONE) / reference;
            let blend = canvas.match_width_or_height.clamp(0.0, 1.0);
            let log = ratio.x.log2() + (ratio.y.log2() - ratio.x.log2()) * blend;
            log.exp2()
        }
    }
}

/// Resolve every root canvas's tree onto a `screen`-sized window. `hierarchy`
/// must be current.
pub fn resolve(world: &World, hierarchy: &Hierarchy, screen: [f32; 2], out: &mut Layout) {
    out.screen = screen;
    out.canvases.clear();
    out.elements.clear();
    let mut stack = std::mem::take(&mut out.stack);
    stack.clear();

    world.query::<&Canvas>().for_each(|entity, canvas| {
        if !under_a_canvas(world, entity) {
            out.canvases.push(CanvasRoot {
                entity,
                scale: scale_factor(canvas, screen),
            });
        }
    });
    let sort_key = |root: &CanvasRoot| {
        let order = world.get::<Canvas>(root.entity).map_or(0, |c| c.sort_order);
        (order, hierarchy.position(root.entity).unwrap_or(usize::MAX))
    };
    out.canvases.sort_by_cached_key(sort_key);

    for (index, root) in out.canvases.iter().enumerate() {
        let units = Vec2::splat(root.scale);
        let visit = |entity| Visit {
            entity,
            parent: None,
            canvas: index as u32,
            min: Vec2::ZERO,
            size: Vec2::from(screen) / units,
            offset: Vec2::ZERO,
            units,
        };
        // Reversed, so popping visits them in hierarchy order.
        stack.extend(
            hierarchy
                .children_of(root.entity)
                .iter()
                .rev()
                .map(|&e| visit(e)),
        );
        while let Some(at) = stack.pop() {
            let Some(rect) = world.get::<RectTransform>(at.entity).map(|rect| *rect) else {
                continue;
            };
            let (min, size) = rect.rect_in(at.min, at.size);
            let pivot = min + rect.pivot * size;
            // Scaled about the pivot in the parent's frame, then onto the screen.
            let scaled_min = pivot + (min - pivot) * rect.scale;
            let screen_min = at.offset + scaled_min * at.units;
            let screen_size = size * rect.scale * at.units;
            let raycast = world
                .get::<Image>(at.entity)
                .is_some_and(|i| i.raycast_target)
                || world
                    .get::<Text>(at.entity)
                    .is_some_and(|t| t.raycast_target);
            let this = out.elements.len() as u32;
            out.elements.push(Element {
                entity: at.entity,
                rect: Rect::new(screen_min.x, screen_min.y, screen_size.x, screen_size.y),
                parent: at.parent,
                canvas: at.canvas,
                units: at.units,
                scale: rect.scale,
                parent_size: at.size,
                raycast,
            });
            let child = |entity| Visit {
                entity,
                parent: Some(this),
                canvas: at.canvas,
                min,
                size,
                offset: at.offset + pivot * (Vec2::ONE - rect.scale) * at.units,
                units: at.units * rect.scale,
            };
            stack.extend(
                hierarchy
                    .children_of(at.entity)
                    .iter()
                    .rev()
                    .map(|&e| child(e)),
            );
        }
    }
    out.stack = stack;
}

/// Whether a canvas sits under another, which makes it an ordinary element.
fn under_a_canvas(world: &World, entity: Entity) -> bool {
    let mut cursor = entity;
    // Bounded: a cycle the hierarchy has not broken yet would otherwise spin.
    for _ in 0..4096 {
        match world.get::<Parent>(cursor).map(|parent| parent.get()) {
            Some(parent) if world.has::<Canvas>(parent) => return true,
            Some(parent) => cursor = parent,
            None => return false,
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Name, ensure_current, reparent};

    const SCREEN: [f32; 2] = [1920.0, 1080.0];

    struct Tree {
        world: World,
    }

    impl Tree {
        fn new() -> Self {
            Self {
                world: World::new(),
            }
        }

        fn canvas(&mut self, canvas: Canvas) -> Entity {
            let entity = self.world.spawn();
            self.world.insert(entity, canvas);
            self.world.insert(entity, Name::new("Canvas"));
            entity
        }

        fn element(&mut self, parent: Entity, rect: RectTransform) -> Entity {
            let entity = self.world.spawn();
            self.world.insert(entity, rect);
            reparent(&mut self.world, entity, Some(parent), false).unwrap();
            entity
        }

        fn layout(&mut self, screen: [f32; 2]) -> Layout {
            ensure_current(&mut self.world);
            let mut layout = Layout::default();
            resolve(
                &self.world,
                &self.world.resource::<Hierarchy>(),
                screen,
                &mut layout,
            );
            layout
        }

        fn rect_of(&mut self, entity: Entity, screen: [f32; 2]) -> Rect {
            let layout = self.layout(screen);
            let index = layout.index_of(entity).expect("laid out");
            layout.elements()[index].rect
        }
    }

    fn fixed(anchor: Vec2, pivot: Vec2, position: Vec2, size: Vec2) -> RectTransform {
        RectTransform {
            anchor_min: anchor,
            anchor_max: anchor,
            pivot,
            anchored_position: position,
            size_delta: size,
            scale: Vec2::ONE,
        }
    }

    fn close(a: Rect, b: Rect) -> bool {
        [a.x - b.x, a.y - b.y, a.width - b.width, a.height - b.height]
            .iter()
            .all(|d| d.abs() < 1e-3)
    }

    #[test]
    fn a_stretched_element_fills_its_parent_less_its_size_delta() {
        let mut tree = Tree::new();
        let canvas = tree.canvas(Canvas::default());
        let panel = tree.element(
            canvas,
            RectTransform {
                size_delta: Vec2::new(-40.0, -20.0),
                ..RectTransform::stretched()
            },
        );
        let rect = tree.rect_of(panel, SCREEN);
        assert!(
            close(rect, Rect::new(20.0, 10.0, 1880.0, 1060.0)),
            "{rect:?}"
        );
    }

    #[test]
    fn a_fixed_anchor_places_the_pivot_at_the_anchor_plus_the_position() {
        let mut tree = Tree::new();
        let canvas = tree.canvas(Canvas::default());
        // Bottom right, pivot at its own bottom right, 10 in from each edge.
        let corner = tree.element(
            canvas,
            fixed(
                Vec2::ONE,
                Vec2::ONE,
                Vec2::new(-10.0, -10.0),
                Vec2::new(200.0, 50.0),
            ),
        );
        // Centred, pivot at its top left.
        let centred = tree.element(
            canvas,
            fixed(
                Vec2::splat(0.5),
                Vec2::ZERO,
                Vec2::ZERO,
                Vec2::new(100.0, 40.0),
            ),
        );
        let layout = tree.layout(SCREEN);
        let rect = |entity| layout.elements()[layout.index_of(entity).unwrap()].rect;
        assert!(
            close(rect(corner), Rect::new(1710.0, 1020.0, 200.0, 50.0)),
            "{:?}",
            rect(corner)
        );
        assert!(
            close(rect(centred), Rect::new(960.0, 540.0, 100.0, 40.0)),
            "{:?}",
            rect(centred)
        );
    }

    #[test]
    fn a_child_is_anchored_to_its_parent_and_scaled_about_its_parents_pivot() {
        let mut tree = Tree::new();
        let canvas = tree.canvas(Canvas::default());
        let parent = tree.element(
            canvas,
            fixed(
                Vec2::ZERO,
                Vec2::ZERO,
                Vec2::new(100.0, 100.0),
                Vec2::new(400.0, 200.0),
            ),
        );
        // The right half of its parent.
        let half = tree.element(
            parent,
            RectTransform {
                anchor_min: Vec2::new(0.5, 0.0),
                anchor_max: Vec2::ONE,
                size_delta: Vec2::ZERO,
                ..RectTransform::default()
            },
        );
        assert!(
            close(
                tree.rect_of(half, SCREEN),
                Rect::new(300.0, 100.0, 200.0, 200.0)
            ),
            "{:?}",
            tree.rect_of(half, SCREEN)
        );

        // Doubled about its top-left pivot, which stays put; the child scales
        // with it.
        tree.world.get_mut::<RectTransform>(parent).unwrap().scale = Vec2::splat(2.0);
        let layout = tree.layout(SCREEN);
        let rect = |entity| layout.elements()[layout.index_of(entity).unwrap()].rect;
        assert!(
            close(rect(parent), Rect::new(100.0, 100.0, 800.0, 400.0)),
            "{:?}",
            rect(parent)
        );
        assert!(
            close(rect(half), Rect::new(500.0, 100.0, 400.0, 400.0)),
            "{:?}",
            rect(half)
        );
        let child = layout.elements()[layout.index_of(half).unwrap()];
        assert_eq!(
            child.units,
            Vec2::splat(2.0),
            "the parent's frame is doubled"
        );
        assert_eq!(child.parent_size, Vec2::new(400.0, 200.0));
    }

    #[test]
    fn constant_pixel_is_one_unit_to_a_pixel_at_any_size() {
        let canvas = Canvas::default();
        assert_eq!(scale_factor(&canvas, [3840.0, 2160.0]), 1.0);
        assert_eq!(scale_factor(&canvas, [800.0, 600.0]), 1.0);
    }

    #[test]
    fn scale_with_screen_blends_width_and_height_logarithmically() {
        let by = |amount: f32| Canvas {
            scale_mode: ScaleMode::ScaleWithScreen,
            reference_resolution: Vec2::new(1920.0, 1080.0),
            match_width_or_height: amount,
            sort_order: 0,
        };
        // Twice as wide, the same height.
        let screen = [3840.0, 1080.0];
        assert!((scale_factor(&by(0.0), screen) - 2.0).abs() < 1e-5);
        assert!((scale_factor(&by(1.0), screen) - 1.0).abs() < 1e-5);
        assert!(
            (scale_factor(&by(0.5), screen) - 2f32.sqrt()).abs() < 1e-5,
            "the geometric mean, not the arithmetic"
        );

        // A full-screen panel still fills the window, in canvas units.
        let mut tree = Tree::new();
        let canvas = tree.canvas(by(0.0));
        let half_size = tree.element(
            canvas,
            fixed(Vec2::ZERO, Vec2::ZERO, Vec2::ZERO, Vec2::new(960.0, 540.0)),
        );
        let rect = tree.rect_of(half_size, [3840.0, 2160.0]);
        assert!(
            close(rect, Rect::new(0.0, 0.0, 1920.0, 1080.0)),
            "authored at half the reference, drawn at half the window: {rect:?}"
        );
    }

    #[test]
    fn a_zero_size_parent_still_places_its_children() {
        let mut tree = Tree::new();
        let canvas = tree.canvas(Canvas::default());
        let point = tree.element(
            canvas,
            fixed(Vec2::ZERO, Vec2::ZERO, Vec2::new(50.0, 60.0), Vec2::ZERO),
        );
        let stretched = tree.element(point, RectTransform::stretched());
        let sized = tree.element(
            point,
            fixed(
                Vec2::splat(0.5),
                Vec2::splat(0.5),
                Vec2::ZERO,
                Vec2::new(10.0, 10.0),
            ),
        );
        let layout = tree.layout(SCREEN);
        let rect = |entity| layout.elements()[layout.index_of(entity).unwrap()].rect;
        assert!(close(rect(stretched), Rect::new(50.0, 60.0, 0.0, 0.0)));
        assert!(close(rect(sized), Rect::new(45.0, 55.0, 10.0, 10.0)));
        assert!(rect(sized).x.is_finite() && rect(stretched).width.is_finite());
    }

    /// Only a `RectTransform` makes an element: anything else under a canvas
    /// — and its subtree — is not UI, and a canvas that is itself under one is
    /// laid out by its own rect rather than taking the window.
    #[test]
    fn only_rect_transforms_under_a_root_canvas_are_elements() {
        let mut tree = Tree::new();
        let canvas = tree.canvas(Canvas::default());
        let plain = tree.world.spawn();
        reparent(&mut tree.world, plain, Some(canvas), false).unwrap();
        let hidden = tree.element(plain, RectTransform::stretched());
        let nested = tree.element(
            canvas,
            fixed(
                Vec2::ZERO,
                Vec2::ZERO,
                Vec2::new(10.0, 10.0),
                Vec2::new(20.0, 20.0),
            ),
        );
        tree.world.insert(nested, Canvas::default());
        let orphan = tree.world.spawn();
        tree.world.insert(orphan, RectTransform::stretched());

        let layout = tree.layout(SCREEN);
        assert!(layout.index_of(hidden).is_none());
        assert!(layout.index_of(orphan).is_none(), "no canvas above it");
        assert_eq!(layout.canvases().len(), 1);
        let rect = layout.elements()[layout.index_of(nested).unwrap()].rect;
        assert!(close(rect, Rect::new(10.0, 10.0, 20.0, 20.0)));
    }

    #[test]
    fn elements_are_in_draw_order_depth_first_and_canvases_by_sort_order() {
        let mut tree = Tree::new();
        let front = tree.canvas(Canvas {
            sort_order: 5,
            ..Canvas::default()
        });
        let back = tree.canvas(Canvas::default());
        let a = tree.element(back, RectTransform::stretched());
        let a_child = tree.element(a, RectTransform::stretched());
        let b = tree.element(back, RectTransform::stretched());
        let over = tree.element(front, RectTransform::stretched());

        let layout = tree.layout(SCREEN);
        let order: Vec<Entity> = layout.elements().iter().map(|e| e.entity).collect();
        assert_eq!(order, [a, a_child, b, over]);
        assert_eq!(layout.elements()[1].parent, Some(0));
        assert_eq!(layout.elements()[2].parent, None);
        assert_eq!(layout.canvases()[0].entity, back);
    }

    #[test]
    fn the_topmost_hit_is_the_last_drawn_and_non_targets_let_the_pointer_through() {
        let mut tree = Tree::new();
        let canvas = tree.canvas(Canvas::default());
        let back = tree.element(canvas, RectTransform::stretched());
        tree.world.insert(back, Image::default());
        let front = tree.element(
            canvas,
            fixed(Vec2::ZERO, Vec2::ZERO, Vec2::ZERO, Vec2::new(100.0, 100.0)),
        );
        tree.world.insert(
            front,
            Image {
                raycast_target: false,
                ..Image::default()
            },
        );
        let label = tree.element(front, RectTransform::stretched());
        tree.world.insert(label, Text::default());

        let layout = tree.layout(SCREEN);
        let entity = |hit: Option<usize>| hit.map(|i| layout.elements()[i].entity);
        assert_eq!(entity(layout.hit([50.0, 50.0], false)), Some(label));
        assert_eq!(
            entity(layout.hit([50.0, 50.0], true)),
            Some(back),
            "neither the image nor the text stops the pointer"
        );
        assert_eq!(entity(layout.hit([500.0, 500.0], true)), Some(back));
        assert_eq!(layout.hit([-1.0, 5.0], false), None);
        let label_index = layout.index_of(label).unwrap();
        assert!(layout.is_at_or_above(front, label_index));
        assert!(layout.is_at_or_above(label, label_index));
        assert!(!layout.is_at_or_above(back, label_index));
    }
}
