use glam::Vec2;
use orrin_ecs::{Entity, World};

use super::{
    Canvas, Corner, Gradient, Image, RectTransform, ScaleMode, ScrollFrame, ScrollbarVisibility,
    Shadow, Stroke, Text,
};
use crate::scene::{GameScreen, Hierarchy, Parent, Rect};

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
    /// Whether the pointer stops here: at a raycast-target `Image` or `Text`,
    /// or anywhere in a `ScrollFrame`, which the wheel scrolls.
    pub raycast: bool,
}

/// A root canvas, resolved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanvasRoot {
    pub entity: Entity,
    /// Game pixels per canvas unit: the scaler's answer, before the screen's
    /// own scale into the window.
    pub scale: f32,
}

/// A [`ScrollFrame`], resolved: what it shows of its content and where.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScrollView {
    /// The frame, in [`Layout::elements`].
    pub element: u32,
    /// One past its last descendant, so its content is `element + 1..end`.
    pub end: u32,
    /// The frame's own rectangle on screen.
    pub viewport: Rect,
    /// What the content may cover: the viewport, cut to every frame around it.
    pub clip: Rect,
    /// The content area on screen, at the scroll it was laid out with.
    pub content: Rect,
    /// That scroll, in the frame's units, clamped to `0..=max_scroll`.
    pub scroll: Vec2,
    /// How far each axis can scroll, in units; zero on an axis that does not.
    pub max_scroll: Vec2,
    /// Screen pixels per unit of the scroll.
    pub pixels: Vec2,
    /// The horizontal bar and the vertical one, where they are drawn.
    pub bars: [Option<Scrollbar>; 2],
}

/// One axis's bar: the strip it runs in, and the thumb that shows the view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scrollbar {
    pub track: Rect,
    pub thumb: Rect,
}

/// Which of the components that cost the walk or the recording more work
/// anything in the world has: a lookup per element is most of what an element
/// costs, so a world with none of one pays five storage checks a frame for
/// them rather than one lookup per element for each.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Presence {
    pub scroll_frame: bool,
    pub corner: bool,
    pub gradient: bool,
    pub stroke: bool,
    pub shadow: bool,
}

impl Presence {
    fn of(world: &World) -> Self {
        Self {
            scroll_frame: world.query::<&ScrollFrame>().candidates() > 0,
            corner: world.query::<&Corner>().candidates() > 0,
            gradient: world.query::<&Gradient>().candidates() > 0,
            stroke: world.query::<&Stroke>().candidates() > 0,
            shadow: world.query::<&Shadow>().candidates() > 0,
        }
    }

    /// Whether any element can have a modifier.
    pub fn modifiers(self) -> bool {
        self.corner || self.gradient || self.stroke || self.shadow
    }
}

/// No enclosing [`ScrollView`], in [`Layout::clip_of`]'s table.
const UNCLIPPED: u32 = u32::MAX;

/// [`Layout::draws`]'s bits.
const HAS_IMAGE: u8 = 1;
const HAS_TEXT: u8 = 2;

/// Every canvas's tree, resolved for one window size, in draw order.
#[derive(Default)]
pub struct Layout {
    screen: GameScreen,
    canvases: Vec<CanvasRoot>,
    elements: Vec<Element>,
    /// Beside `elements`, the innermost scroll view each is content of, or
    /// [`UNCLIPPED`]: four bytes an element, which is all a hit test needs.
    /// Empty when there are no scroll views.
    clipped_by: Vec<u32>,
    /// Beside `elements`, which of an `Image` and a `Text` each has: the walk
    /// looks both up for the hit test anyway, and recording what an element
    /// draws then skips the lookups that would find nothing.
    drawn: Vec<u8>,
    scrolls: Vec<ScrollView>,
    /// The depth-first walk's stack, kept for its allocation.
    stack: Vec<Visit>,
    /// Beside `stack` while there are scroll views, the one each visit is
    /// content of: apart, so a walk with none moves no more than it did.
    clips: Vec<u32>,
    presence: Presence,
    /// Scroll views whose subtree the walk is still inside, with the stack's
    /// depth beneath their children: when the stack is back to it, the
    /// subtree has ended.
    open: Vec<(u32, usize)>,
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

impl Default for GameScreen {
    fn default() -> Self {
        Self::window([0, 0])
    }
}

impl Layout {
    pub fn screen(&self) -> &GameScreen {
        &self.screen
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

    /// What the world had when this was resolved.
    pub fn presence(&self) -> Presence {
        self.presence
    }

    /// In draw order: a frame before any frame inside it.
    pub fn scrolls(&self) -> &[ScrollView] {
        &self.scrolls
    }

    /// Whether element `index` has an `Image`, and whether it has a `Text`.
    pub fn draws(&self, index: usize) -> (bool, bool) {
        let drawn = self.drawn[index];
        (drawn & HAS_IMAGE != 0, drawn & HAS_TEXT != 0)
    }

    /// What element `index` may be drawn on and hit in, if a scroll frame
    /// above it cuts it.
    pub fn clip(&self, index: usize) -> Option<Rect> {
        self.clip_of(index).map(|view| view.clip)
    }

    /// The scroll view element `index` is content of, if any.
    pub fn clip_of(&self, index: usize) -> Option<&ScrollView> {
        self.scrolls.get(*self.clipped_by.get(index)? as usize)
    }

    /// The scroll view element `index` is the frame of, if it is one.
    pub fn scroll_of(&self, index: usize) -> Option<&ScrollView> {
        let at = self
            .scrolls
            .binary_search_by_key(&(index as u32), |view| view.element)
            .ok()?;
        Some(&self.scrolls[at])
    }

    pub fn index_of(&self, entity: Entity) -> Option<usize> {
        self.elements
            .iter()
            .position(|element| element.entity == entity)
    }

    /// The rectangle element `index`'s anchors are fractions of, on screen:
    /// its parent's, or the game's screen for a child of the canvas.
    pub fn parent_rect(&self, index: usize) -> Rect {
        match self.elements[index].parent {
            Some(parent) => match self.scroll_of(parent as usize) {
                Some(view) => view.content,
                None => self.elements[parent as usize].rect,
            },
            None => self.screen.rect(),
        }
    }

    /// The topmost element under `point`: the last one drawn there. With
    /// `raycast_only`, only an element that stops the pointer counts, and the
    /// rest let it through to what is behind.
    pub fn hit(&self, point: [f32; 2], raycast_only: bool) -> Option<usize> {
        let clipped = |index: usize| self.clip(index).is_some_and(|clip| !contains(clip, point));
        (0..self.elements.len()).rev().find(|&index| {
            let element = &self.elements[index];
            (element.raycast || !raycast_only) && contains(element.rect, point) && !clipped(index)
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

/// Resolve every root canvas's tree onto the game's `screen`. `hierarchy` must
/// be current.
pub fn resolve(world: &World, hierarchy: &Hierarchy, game: &GameScreen, out: &mut Layout) {
    out.presence = Presence::of(world);
    if out.presence.scroll_frame {
        resolve_with::<true>(world, hierarchy, game, out);
    } else {
        resolve_with::<false>(world, hierarchy, game, out);
    }
}

/// [`resolve`], compiled twice: a world with no scroll frame walks its elements
/// with none of the bookkeeping one needs — the lookup, the clip table, the
/// subtree ends — which measured as a quarter of the walk's time over a HUD
/// that has none.
fn resolve_with<const SCROLLING: bool>(
    world: &World,
    hierarchy: &Hierarchy,
    game: &GameScreen,
    out: &mut Layout,
) {
    out.screen = *game;
    let screen = game.size;
    out.canvases.clear();
    out.elements.clear();
    out.clipped_by.clear();
    out.drawn.clear();
    out.scrolls.clear();
    let mut stack = std::mem::take(&mut out.stack);
    stack.clear();
    let mut open = std::mem::take(&mut out.open);
    open.clear();
    let mut clips = std::mem::take(&mut out.clips);
    clips.clear();

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
        // The scaler sees the game's size; the screen's own scale and place
        // then carry the result into the window.
        let size = Vec2::from(screen) / root.scale;
        let units = Vec2::splat(root.scale * game.scale);
        let visit = |entity| Visit {
            entity,
            parent: None,
            canvas: index as u32,
            min: Vec2::ZERO,
            size,
            offset: Vec2::from(game.origin),
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
        if SCROLLING {
            clips.resize(stack.len(), UNCLIPPED);
        }
        loop {
            if SCROLLING {
                while let Some(&(view, depth)) = open.last()
                    && depth == stack.len()
                {
                    out.scrolls[view as usize].end = out.elements.len() as u32;
                    open.pop();
                }
            }
            let Some(at) = stack.pop() else {
                break;
            };
            let at_clip = if SCROLLING {
                clips.pop().unwrap_or(UNCLIPPED)
            } else {
                UNCLIPPED
            };
            let Some(rect) = world.get::<RectTransform>(at.entity).map(|rect| *rect) else {
                continue;
            };
            let (min, size) = rect.rect_in(at.min, at.size);
            let pivot = min + rect.pivot * size;
            // Scaled about the pivot in the parent's frame, then onto the screen.
            let scaled_min = pivot + (min - pivot) * rect.scale;
            let screen_min = at.offset + scaled_min * at.units;
            let screen_size = size * rect.scale * at.units;
            let scroll = if SCROLLING {
                scroll_frame(world, at.entity)
            } else {
                None
            };
            let image = world.get::<Image>(at.entity).map(|i| i.raycast_target);
            let text = world.get::<Text>(at.entity).map(|t| t.raycast_target);
            let raycast = scroll.is_some() || image == Some(true) || text == Some(true);
            out.drawn.push(
                (u8::from(image.is_some()) * HAS_IMAGE) | (u8::from(text.is_some()) * HAS_TEXT),
            );
            let this = out.elements.len() as u32;
            let screen_rect = Rect::new(screen_min.x, screen_min.y, screen_size.x, screen_size.y);
            out.elements.push(Element {
                entity: at.entity,
                rect: screen_rect,
                parent: at.parent,
                canvas: at.canvas,
                units: at.units,
                scale: rect.scale,
                parent_size: at.size,
                raycast,
            });
            let offset = at.offset + pivot * (Vec2::ONE - rect.scale) * at.units;
            let units = at.units * rect.scale;
            let (mut content_min, mut content_size, mut clip) = (min, size, at_clip);
            if SCROLLING {
                out.clipped_by.push(at_clip);
                if let Some(frame) = scroll {
                    let outer = out.scrolls.get(at_clip as usize).map(|view| view.clip);
                    let view =
                        scroll_view(&frame, this, screen_rect, outer, min, size, offset, units);
                    content_min = min - view.scroll;
                    content_size = frame.content_size.max(size);
                    clip = out.scrolls.len() as u32;
                    open.push((clip, stack.len()));
                    out.scrolls.push(view);
                }
            }
            let child = |entity| Visit {
                entity,
                parent: Some(this),
                canvas: at.canvas,
                min: content_min,
                size: content_size,
                offset,
                units,
            };
            stack.extend(
                hierarchy
                    .children_of(at.entity)
                    .iter()
                    .rev()
                    .map(|&e| child(e)),
            );
            if SCROLLING {
                clips.resize(stack.len(), clip);
            }
        }
    }
    out.stack = stack;
    out.clips = clips;
    out.open = open;
}

/// Out of line with [`scroll_view`], so the walk over elements that are not
/// frames stays the size it is.
#[inline(never)]
fn scroll_frame(world: &World, entity: Entity) -> Option<ScrollFrame> {
    world.get::<ScrollFrame>(entity).map(|frame| *frame)
}

/// Frame `element`, at `viewport` on screen and `min`/`size` in its parent's
/// frame — which `offset + point * units` maps to the screen — resolved.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn scroll_view(
    frame: &ScrollFrame,
    element: u32,
    viewport: Rect,
    outer: Option<Rect>,
    min: Vec2,
    size: Vec2,
    offset: Vec2,
    units: Vec2,
) -> ScrollView {
    let content = frame.content_size.max(size);
    let axes = Vec2::new(
        f32::from(u8::from(frame.horizontal)),
        f32::from(u8::from(frame.vertical)),
    );
    let max_scroll = (content - size).max(Vec2::ZERO) * axes;
    let scroll = frame.scroll_position.clamp(Vec2::ZERO, max_scroll);
    let corner = offset + (min - scroll) * units;
    let extent = content * units;
    let shown = [0, 1].map(|axis| {
        axes[axis] > 0.0
            && match frame.scrollbar {
                ScrollbarVisibility::Auto => max_scroll[axis] > 0.0,
                ScrollbarVisibility::Always => true,
                ScrollbarVisibility::Never => false,
            }
    });
    let thickness = Vec2::splat(frame.scrollbar_thickness.max(0.0)) * units;
    let [right, bottom] = viewport.max();
    let bar = |axis: usize| {
        if !shown[axis] {
            return None;
        }
        // Each bar gives up the corner the other one occupies.
        let other = if shown[1 - axis] {
            thickness[1 - axis]
        } else {
            0.0
        };
        let track = if axis == 0 {
            Rect::new(
                viewport.x,
                bottom - thickness.y,
                (viewport.width - other).max(0.0),
                thickness.y,
            )
        } else {
            Rect::new(
                right - thickness.x,
                viewport.y,
                thickness.x,
                (viewport.height - other).max(0.0),
            )
        };
        let length = if axis == 0 { track.width } else { track.height };
        let shortest = length.min(2.0 * thickness[axis]);
        let thumb_length =
            (length * size[axis] / content[axis].max(f32::EPSILON)).clamp(shortest, length);
        let along = if max_scroll[axis] > 0.0 {
            (length - thumb_length) * scroll[axis] / max_scroll[axis]
        } else {
            0.0
        };
        let thumb = if axis == 0 {
            Rect::new(track.x + along, track.y, thumb_length, track.height)
        } else {
            Rect::new(track.x, track.y + along, track.width, thumb_length)
        };
        Some(Scrollbar { track, thumb })
    };
    ScrollView {
        element,
        end: element + 1,
        viewport,
        clip: outer.map_or(viewport, |outer| intersect(viewport, outer)),
        content: Rect::new(corner.x, corner.y, extent.x, extent.y),
        scroll,
        max_scroll,
        pixels: units,
        bars: [bar(0), bar(1)],
    }
}

/// The overlap of `a` and `b`, with no area when there is none.
pub fn intersect(a: Rect, b: Rect) -> Rect {
    let (x, y) = (a.x.max(b.x), a.y.max(b.y));
    let [a_right, a_bottom] = a.max();
    let [b_right, b_bottom] = b.max();
    Rect::new(
        x,
        y,
        (a_right.min(b_right) - x).max(0.0),
        (a_bottom.min(b_bottom) - y).max(0.0),
    )
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
            self.layout_on(&GameScreen::window(screen.map(|side| side as u32)))
        }

        fn layout_on(&mut self, screen: &GameScreen) -> Layout {
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

    /// A previewed resolution lays the tree out at its own size and draws it,
    /// scaled, into its place in the window: a canvas scaled with the screen
    /// keeps the proportions it will ship with.
    #[test]
    fn a_previewed_screen_lays_out_at_its_size_inside_its_rectangle() {
        let mut tree = Tree::new();
        let canvas = tree.canvas(Canvas {
            scale_mode: ScaleMode::ScaleWithScreen,
            reference_resolution: Vec2::new(1920.0, 1080.0),
            match_width_or_height: 0.0,
            sort_order: 0,
        });
        let full = tree.element(canvas, RectTransform::stretched());
        let corner = tree.element(
            canvas,
            fixed(
                Vec2::ZERO,
                Vec2::ZERO,
                Vec2::new(10.0, 10.0),
                Vec2::new(100.0, 50.0),
            ),
        );
        let screen = GameScreen {
            size: [1920.0, 1080.0],
            origin: [100.0, 50.0],
            scale: 0.5,
            clip: Rect::new(100.0, 50.0, 960.0, 540.0),
        };
        let layout = tree.layout_on(&screen);
        let rect = |entity| layout.elements()[layout.index_of(entity).unwrap()].rect;
        assert!(close(rect(full), screen.rect()), "{:?}", rect(full));
        assert!(
            close(rect(corner), Rect::new(105.0, 55.0, 50.0, 25.0)),
            "{:?}",
            rect(corner)
        );
        let index = layout.index_of(corner).unwrap();
        assert_eq!(layout.parent_rect(index), screen.rect());
        assert_eq!(layout.elements()[index].units, Vec2::splat(0.5));
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

    fn frame(content: Vec2, scroll: Vec2) -> ScrollFrame {
        ScrollFrame {
            content_size: content,
            scroll_position: scroll,
            scrollbar: ScrollbarVisibility::Never,
            ..ScrollFrame::default()
        }
    }

    /// A frame's children are anchored to its content area, which the scroll
    /// moves, and are cut to the frame; the frame itself is not.
    #[test]
    fn a_scroll_frames_children_sit_in_its_content_moved_by_the_scroll() {
        let mut tree = Tree::new();
        let canvas = tree.canvas(Canvas::default());
        let list = tree.element(
            canvas,
            fixed(
                Vec2::ZERO,
                Vec2::ZERO,
                Vec2::new(100.0, 100.0),
                Vec2::new(200.0, 100.0),
            ),
        );
        tree.world
            .insert(list, frame(Vec2::new(0.0, 400.0), Vec2::new(0.0, 50.0)));
        let fill = tree.element(list, RectTransform::stretched());
        let row = tree.element(
            list,
            fixed(
                Vec2::ZERO,
                Vec2::ZERO,
                Vec2::new(0.0, 150.0),
                Vec2::new(200.0, 30.0),
            ),
        );
        let after = tree.element(canvas, RectTransform::stretched());

        let layout = tree.layout(SCREEN);
        let index = |entity| layout.index_of(entity).unwrap();
        let rect = |entity| layout.elements()[index(entity)].rect;
        assert!(
            close(rect(fill), Rect::new(100.0, 50.0, 200.0, 400.0)),
            "as wide as the frame, as tall as the content: {:?}",
            rect(fill)
        );
        assert!(
            close(rect(row), Rect::new(100.0, 200.0, 200.0, 30.0)),
            "{:?}",
            rect(row)
        );
        let viewport = Rect::new(100.0, 100.0, 200.0, 100.0);
        assert_eq!(layout.clip(index(fill)), Some(viewport));
        assert_eq!(layout.clip(index(row)), Some(viewport));
        assert_eq!(
            layout.clip(index(list)),
            None,
            "a frame is not cut by itself"
        );
        assert_eq!(layout.clip(index(after)), None, "nor is what follows it");

        let [view] = layout.scrolls() else {
            panic!("{:?}", layout.scrolls());
        };
        assert_eq!(view.element as usize, index(list));
        assert_eq!(view.end as usize, index(after), "its subtree ends here");
        assert_eq!(view.viewport, viewport);
        assert_eq!(view.scroll, Vec2::new(0.0, 50.0));
        assert_eq!(view.max_scroll, Vec2::new(0.0, 300.0));
        assert!(std::ptr::eq(layout.scroll_of(index(list)).unwrap(), view));
        assert!(std::ptr::eq(layout.clip_of(index(row)).unwrap(), view));
    }

    /// A scroll past either end lays out at the end, and an axis that does not
    /// scroll lays out unscrolled whatever the component says.
    #[test]
    fn a_scroll_is_laid_out_clamped_to_what_the_content_allows() {
        let mut tree = Tree::new();
        let canvas = tree.canvas(Canvas::default());
        let list = tree.element(
            canvas,
            fixed(Vec2::ZERO, Vec2::ZERO, Vec2::ZERO, Vec2::new(200.0, 100.0)),
        );
        let row = tree.element(list, RectTransform::stretched());
        let mut scrolled = |scroll: Vec2, content: Vec2| {
            tree.world.insert(list, frame(content, scroll));
            let layout = tree.layout(SCREEN);
            let view = layout.scrolls()[0];
            let rect = layout.elements()[layout.index_of(row).unwrap()].rect;
            (view.scroll, view.max_scroll, rect.x, rect.y)
        };
        let content = Vec2::new(600.0, 400.0);
        assert_eq!(
            scrolled(Vec2::new(0.0, 1000.0), content),
            (Vec2::new(0.0, 300.0), Vec2::new(0.0, 300.0), 0.0, -300.0)
        );
        assert_eq!(
            scrolled(Vec2::new(-20.0, -5.0), content),
            (Vec2::ZERO, Vec2::new(0.0, 300.0), 0.0, 0.0)
        );
        assert_eq!(
            scrolled(Vec2::new(250.0, 0.0), content),
            (Vec2::ZERO, Vec2::new(0.0, 300.0), 0.0, 0.0),
            "horizontal scrolling is off"
        );
        assert_eq!(
            scrolled(Vec2::new(0.0, 40.0), Vec2::new(0.0, 50.0)),
            (Vec2::ZERO, Vec2::ZERO, 0.0, 0.0),
            "content smaller than the frame is the frame"
        );
    }

    /// Units, not pixels: a frame scaled up scrolls its content further on
    /// screen for the same scroll.
    #[test]
    fn a_scaled_frame_scrolls_in_its_own_units() {
        let mut tree = Tree::new();
        let canvas = tree.canvas(Canvas::default());
        let list = tree.element(
            canvas,
            RectTransform {
                scale: Vec2::splat(2.0),
                ..fixed(Vec2::ZERO, Vec2::ZERO, Vec2::ZERO, Vec2::new(100.0, 100.0))
            },
        );
        tree.world
            .insert(list, frame(Vec2::new(0.0, 300.0), Vec2::new(0.0, 25.0)));
        let row = tree.element(list, RectTransform::stretched());
        let layout = tree.layout(SCREEN);
        let rect = layout.elements()[layout.index_of(row).unwrap()].rect;
        assert!(close(rect, Rect::new(0.0, -50.0, 200.0, 600.0)), "{rect:?}");
        assert_eq!(layout.scrolls()[0].pixels, Vec2::splat(2.0));
    }

    /// A frame inside a frame cuts its content to both.
    #[test]
    fn nested_scroll_frames_intersect_their_clips() {
        let mut tree = Tree::new();
        let canvas = tree.canvas(Canvas::default());
        let outer = tree.element(
            canvas,
            fixed(Vec2::ZERO, Vec2::ZERO, Vec2::ZERO, Vec2::new(300.0, 300.0)),
        );
        tree.world
            .insert(outer, frame(Vec2::new(0.0, 1000.0), Vec2::ZERO));
        let inner = tree.element(
            outer,
            fixed(
                Vec2::ZERO,
                Vec2::ZERO,
                Vec2::new(100.0, 250.0),
                Vec2::new(150.0, 200.0),
            ),
        );
        tree.world
            .insert(inner, frame(Vec2::new(0.0, 1000.0), Vec2::ZERO));
        let item = tree.element(inner, RectTransform::stretched());
        let sibling = tree.element(outer, RectTransform::stretched());

        let layout = tree.layout(SCREEN);
        let index = |entity| layout.index_of(entity).unwrap();
        assert_eq!(
            layout.clip(index(inner)),
            Some(Rect::new(0.0, 0.0, 300.0, 300.0))
        );
        assert_eq!(
            layout.clip(index(item)),
            Some(Rect::new(100.0, 250.0, 150.0, 50.0)),
            "the inner viewport, less what of it the outer one hides"
        );
        assert_eq!(
            layout.clip(index(sibling)),
            Some(Rect::new(0.0, 0.0, 300.0, 300.0)),
            "back to the outer frame's once the inner one's subtree ends"
        );
        let ends: Vec<u32> = layout.scrolls().iter().map(|view| view.end).collect();
        assert_eq!(ends, [index(sibling) as u32 + 1, index(sibling) as u32]);
    }

    /// What a frame hides cannot be hit, and the frame itself stops the
    /// pointer over all of its viewport.
    #[test]
    fn a_hit_test_respects_the_clip() {
        let mut tree = Tree::new();
        let canvas = tree.canvas(Canvas::default());
        let list = tree.element(
            canvas,
            fixed(
                Vec2::ZERO,
                Vec2::ZERO,
                Vec2::new(0.0, 100.0),
                Vec2::new(200.0, 100.0),
            ),
        );
        tree.world
            .insert(list, frame(Vec2::new(0.0, 400.0), Vec2::new(0.0, 80.0)));
        // Content y 0..40 is scrolled out above; 60..120 straddles the bottom.
        let hidden = tree.element(
            list,
            fixed(Vec2::ZERO, Vec2::ZERO, Vec2::ZERO, Vec2::new(200.0, 40.0)),
        );
        tree.world.insert(hidden, Image::default());
        let straddling = tree.element(
            list,
            fixed(
                Vec2::ZERO,
                Vec2::ZERO,
                Vec2::new(0.0, 140.0),
                Vec2::new(200.0, 60.0),
            ),
        );
        tree.world.insert(straddling, Image::default());

        let layout = tree.layout(SCREEN);
        let entity = |hit: Option<usize>| hit.map(|i| layout.elements()[i].entity);
        // `hidden` is drawn at y 20..60, above the frame's 100..200.
        assert_eq!(entity(layout.hit([50.0, 40.0], true)), None);
        assert_eq!(entity(layout.hit([50.0, 40.0], false)), None);
        assert_eq!(entity(layout.hit([50.0, 120.0], true)), Some(list));
        // `straddling` is drawn at 160..220.
        assert_eq!(entity(layout.hit([50.0, 170.0], true)), Some(straddling));
        assert_eq!(entity(layout.hit([50.0, 210.0], true)), None);
    }

    /// A bar runs along the frame's far edge, its thumb the share of the
    /// content in view, at the share of the scroll done.
    #[test]
    fn a_scrollbars_thumb_shows_the_view_of_the_content() {
        let mut tree = Tree::new();
        let canvas = tree.canvas(Canvas::default());
        let list = tree.element(
            canvas,
            fixed(Vec2::ZERO, Vec2::ZERO, Vec2::ZERO, Vec2::new(200.0, 100.0)),
        );
        let bars = |tree: &mut Tree, frame: ScrollFrame| {
            tree.world.insert(list, frame);
            tree.layout(SCREEN).scrolls()[0].bars
        };
        let vertical = ScrollFrame {
            content_size: Vec2::new(0.0, 400.0),
            scroll_position: Vec2::new(0.0, 150.0),
            scrollbar_thickness: 10.0,
            ..ScrollFrame::default()
        };
        let [none, Some(bar)] = bars(&mut tree, vertical) else {
            panic!("a vertical bar only");
        };
        assert_eq!(none, None);
        assert_eq!(bar.track, Rect::new(190.0, 0.0, 10.0, 100.0));
        // A quarter in view, half of the way down.
        assert_eq!(bar.thumb, Rect::new(190.0, 37.5, 10.0, 25.0));

        let fits = ScrollFrame {
            content_size: Vec2::new(0.0, 80.0),
            ..vertical
        };
        assert_eq!(bars(&mut tree, fits), [None, None], "nothing to scroll");
        let always = ScrollFrame {
            scrollbar: ScrollbarVisibility::Always,
            ..fits
        };
        let [_, Some(bar)] = bars(&mut tree, always) else {
            panic!("always drawn");
        };
        assert_eq!(bar.thumb, bar.track, "the whole content is in view");

        let both = ScrollFrame {
            content_size: Vec2::new(800.0, 400.0),
            horizontal: true,
            ..vertical
        };
        let [Some(across), Some(down)] = bars(&mut tree, both) else {
            panic!("both bars");
        };
        assert_eq!(across.track, Rect::new(0.0, 90.0, 190.0, 10.0));
        assert_eq!(down.track, Rect::new(190.0, 0.0, 10.0, 90.0));
    }
}
