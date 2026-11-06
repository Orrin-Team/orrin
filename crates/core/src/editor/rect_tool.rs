//! The rect tool: the selected UI element's rectangle in UI mode, with a handle
//! on each corner and edge, one on its pivot, and its body to drag.
//!
//! **Screen space, like the gizmo.** The element's rectangle is the one the
//! layout resolved for the window, so the outline, the handles and what the
//! pointer grabs are where the painter drew it. Handles are built in points and
//! hit-tested against the same shapes the strokes are made from, then written
//! out in pixels for the gizmo pass, which draws them without knowing what they
//! are.
//!
//! **A drag writes the transform, never the rectangle.** Each frame of a drag
//! computes a new `RectTransform` from the one the handle was grabbed on and the
//! pointer's whole travel since — not from last frame's result, which would
//! accumulate rounding — and posts it as an edit, so the gesture is one undo
//! step when the button comes up. Moving changes `anchored_position`; resizing
//! changes `size_delta` and moves the pivot point by its share of the change,
//! so the opposite edge stays put whatever the pivot; the pivot handle moves
//! the pivot and keeps the rectangle where it is. All of it is in the parent's
//! units, which the layout reports per element, so a scaled canvas or a scaled
//! ancestor is followed exactly.
//!
//! **Snapping** is magnetic, to the parent's edges and centre: an edge or the
//! centre of what is being dragged that comes within a few points of one lands
//! on it, and the line it landed on is drawn across the parent. Hold Ctrl to
//! place freely. Rotation does not exist in Phase 1's UI, which is what lets
//! every one of these be an axis-aligned comparison.

use glam::Vec2;
use orrin_ecs::Entity;

use crate::editor::gizmo::{fill_quad, stroke};
use crate::gfx::vulkan::GizmoVertex;
use crate::scene::Rect;
use crate::ui::{Element, RectTransform};

/// Half a handle's side, in points.
const HANDLE: f32 = 4.0;

/// How far from a handle the pointer may be and still grab it, in points.
const REACH: f32 = 7.0;

/// How near an edge comes to one of the parent's before it lands on it, in
/// points.
pub const SNAP: f32 = 6.0;

const OUTLINE: [u8; 4] = [64, 156, 255, 255];
const HOVERED: [u8; 4] = [64, 156, 255, 140];
const HOT: [u8; 4] = [255, 214, 64, 255];
const GUIDE: [u8; 4] = [255, 64, 160, 220];
const FILL: [u8; 4] = [240, 242, 246, 255];

/// The rect tool on one element, in points.
pub struct Frame {
    rect: egui::Rect,
    pivot: egui::Pos2,
}

impl Frame {
    pub fn of(element: &Element, transform: &RectTransform, pixels_per_point: f32) -> Self {
        let rect = points(element.rect, pixels_per_point);
        let pivot = rect.min + rect.size() * egui::vec2(transform.pivot.x, transform.pivot.y);
        Self { rect, pivot }
    }

    /// Each corner and edge handle and where it stands.
    fn handles(&self) -> impl Iterator<Item = (Grip, egui::Pos2)> + '_ {
        (-1..=1).flat_map(move |y: i8| {
            (-1..=1)
                .filter(move |&x| (x, y) != (0, 0))
                .map(move |x: i8| {
                    let along = |side: i8| (f32::from(side) + 1.0) / 2.0;
                    let at = self.rect.min + self.rect.size() * egui::vec2(along(x), along(y));
                    (Grip::Edge { x, y }, at)
                })
        })
    }

    /// What the pointer at `pointer` would grab: the pivot over a handle over
    /// the body.
    pub fn hit(&self, pointer: egui::Pos2) -> Option<Grip> {
        if pointer.distance(self.pivot) <= REACH {
            return Some(Grip::Pivot);
        }
        let near = |at: egui::Pos2| (pointer.x - at.x).abs().max((pointer.y - at.y).abs()) <= REACH;
        if let Some((grip, _)) = self.handles().find(|&(_, at)| near(at)) {
            return Some(grip);
        }
        self.rect.contains(pointer).then_some(Grip::Body)
    }

    /// The outline, the handles and the pivot, the `hot` one lit.
    pub fn strokes(&self, hot: Option<Grip>, out: &mut Vec<GizmoVertex>, pixels_per_point: f32) {
        let colour = if hot == Some(Grip::Body) {
            HOT
        } else {
            OUTLINE
        };
        outline(out, self.rect, colour, pixels_per_point);
        for (grip, at) in self.handles() {
            let d = egui::vec2(HANDLE, HANDLE);
            let e = egui::vec2(HANDLE, -HANDLE);
            let corners = [at - d, at + e, at + d, at - e];
            let fill = if hot == Some(grip) { HOT } else { FILL };
            fill_quad(out, &corners, fill, pixels_per_point);
            for i in 0..4 {
                stroke(
                    out,
                    corners[i],
                    corners[(i + 1) % 4],
                    0.75,
                    0.75,
                    OUTLINE,
                    pixels_per_point,
                );
            }
        }
        let ring = if hot == Some(Grip::Pivot) {
            HOT
        } else {
            OUTLINE
        };
        let points: Vec<egui::Pos2> = (0..=16)
            .map(|step| {
                let angle = step as f32 / 16.0 * std::f32::consts::TAU;
                self.pivot + egui::vec2(angle.cos(), angle.sin()) * HANDLE
            })
            .collect();
        for pair in points.windows(2) {
            stroke(out, pair[0], pair[1], 1.25, 1.25, ring, pixels_per_point);
        }
    }
}

/// A thin outline round an element the pointer is over.
pub fn hovered(out: &mut Vec<GizmoVertex>, rect: Rect, pixels_per_point: f32) {
    outline(
        out,
        points(rect, pixels_per_point),
        HOVERED,
        pixels_per_point,
    );
}

/// The lines a drag snapped to, across `parent`.
pub fn guides(out: &mut Vec<GizmoVertex>, guides: &[Guide], parent: Rect, pixels_per_point: f32) {
    let parent = points(parent, pixels_per_point);
    for guide in guides {
        let at = guide.at / pixels_per_point;
        let (a, b) = if guide.vertical {
            (
                egui::pos2(at, parent.top()),
                egui::pos2(at, parent.bottom()),
            )
        } else {
            (
                egui::pos2(parent.left(), at),
                egui::pos2(parent.right(), at),
            )
        };
        stroke(out, a, b, 0.75, 0.75, GUIDE, pixels_per_point);
    }
}

fn outline(out: &mut Vec<GizmoVertex>, rect: egui::Rect, colour: [u8; 4], pixels_per_point: f32) {
    let corners = [
        rect.left_top(),
        rect.right_top(),
        rect.right_bottom(),
        rect.left_bottom(),
    ];
    for i in 0..4 {
        stroke(
            out,
            corners[i],
            corners[(i + 1) % 4],
            1.0,
            1.0,
            colour,
            pixels_per_point,
        );
    }
}

/// `rect`, in physical pixels, as an egui rectangle in points.
pub fn points(rect: Rect, pixels_per_point: f32) -> egui::Rect {
    egui::Rect::from_min_size(
        egui::pos2(rect.x, rect.y) / pixels_per_point,
        egui::vec2(rect.width, rect.height) / pixels_per_point,
    )
}

/// What part of the rect tool the pointer is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grip {
    Body,
    Pivot,
    /// A corner or an edge: `-1` the left or top side, `1` the right or
    /// bottom, `0` neither.
    Edge {
        x: i8,
        y: i8,
    },
}

/// A line across the parent that a drag snapped to, in physical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Guide {
    /// A vertical line at `x = at`, else a horizontal one at `y = at`.
    pub vertical: bool,
    pub at: f32,
}

/// A drag of the rect tool in progress, as everything was when it began.
#[derive(Clone, Copy, Debug)]
pub struct RectDrag {
    pub entity: Entity,
    pub grip: Grip,
    start: RectTransform,
    /// Where the pointer was, in physical pixels.
    from: Vec2,
    /// Physical pixels per unit of the parent's frame.
    units: Vec2,
    /// The element's rectangle and its parent's, on screen.
    rect: Rect,
    parent: Rect,
}

impl RectDrag {
    pub fn begin(
        grip: Grip,
        start: RectTransform,
        element: &Element,
        parent: Rect,
        from: Vec2,
    ) -> Self {
        Self {
            entity: element.entity,
            grip,
            start,
            from,
            units: element.units,
            rect: element.rect,
            parent,
        }
    }

    /// The transform the drag has reached with the pointer at `to`, and the
    /// guides it snapped to. `snap` is how near, in pixels, an edge has to come
    /// to one of the parent's to land on it; `None` places freely.
    pub fn update(&self, to: Vec2, snap: Option<f32>) -> (RectTransform, Vec<Guide>) {
        let mut delta = to - self.from;
        let mut guides = Vec::new();
        let mut out = self.start;
        let (rect, parent) = (spans(self.rect), spans(self.parent));
        // The parent's two edges and its centre, on each axis.
        let lines = parent.map(|[min, max]| [min, (min + max) / 2.0, max]);
        let mut land = |axis: usize, candidates: &[f32], lines: [f32; 3], delta: &mut Vec2| {
            let Some(reach) = snap else {
                return;
            };
            let nearest = candidates
                .iter()
                .flat_map(|&c| lines.map(|line| (line, line - (c + delta[axis]))))
                .filter(|(_, gap)| gap.abs() <= reach)
                .min_by(|a, b| a.1.abs().total_cmp(&b.1.abs()));
            if let Some((line, gap)) = nearest {
                delta[axis] += gap;
                guides.push(Guide {
                    vertical: axis == 0,
                    at: line,
                });
            }
        };
        match self.grip {
            Grip::Body => {
                for (axis, [min, max]) in rect.into_iter().enumerate() {
                    land(
                        axis,
                        &[min, (min + max) / 2.0, max],
                        lines[axis],
                        &mut delta,
                    );
                }
                out.anchored_position += delta / self.units;
            }
            Grip::Edge { x, y } => {
                for (axis, side) in [x, y].into_iter().enumerate() {
                    if side == 0 || self.start.scale[axis].abs() < f32::EPSILON {
                        continue;
                    }
                    let edge = if side > 0 {
                        rect[axis][1]
                    } else {
                        rect[axis][0]
                    };
                    land(axis, &[edge], lines[axis], &mut delta);
                    let moved = delta[axis] / self.units[axis];
                    let grown = moved * f32::from(side) / self.start.scale[axis];
                    out.size_delta[axis] += grown;
                    let share = if side > 0 {
                        self.start.pivot[axis]
                    } else {
                        1.0 - self.start.pivot[axis]
                    };
                    out.anchored_position[axis] += share * moved;
                }
            }
            Grip::Pivot => {
                for (axis, [min, max]) in rect.into_iter().enumerate() {
                    let size = max - min;
                    if size.abs() < f32::EPSILON {
                        continue;
                    }
                    let point = min + self.start.pivot[axis] * size;
                    // The pivot snaps to its own rectangle's edges and centre.
                    land(axis, &[point], [min, (min + max) / 2.0, max], &mut delta);
                    out.pivot[axis] = (point + delta[axis] - min) / size;
                    out.anchored_position[axis] += delta[axis] / self.units[axis];
                }
            }
        }
        (out, guides)
    }
}

/// `rect`'s `[min, max]` along x and along y.
fn spans(rect: Rect) -> [[f32; 2]; 2] {
    let [right, bottom] = rect.max();
    [[rect.x, right], [rect.y, bottom]]
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARENT: Rect = Rect::new(0.0, 0.0, 1000.0, 800.0);

    /// An element with its rectangle as the layout would have resolved it from
    /// `start`, under a parent `PARENT` filling a frame of `units`.
    fn drag(grip: Grip, start: RectTransform, units: f32) -> RectDrag {
        let parent_size = Vec2::new(PARENT.width, PARENT.height) / units;
        let (min, size) = start.rect_in(Vec2::ZERO, parent_size);
        let pivot = min + start.pivot * size;
        let scaled = pivot + (min - pivot) * start.scale;
        let rect = Rect::new(
            scaled.x * units,
            scaled.y * units,
            size.x * start.scale.x * units,
            size.y * start.scale.y * units,
        );
        let element = Element {
            entity: Entity::default(),
            rect,
            parent: None,
            canvas: 0,
            units: Vec2::splat(units),
            scale: start.scale,
            parent_size,
            raycast: true,
        };
        RectDrag::begin(grip, start, &element, PARENT, Vec2::new(rect.x, rect.y))
    }

    /// The rectangle `rect` resolves to in `PARENT`, in pixels.
    fn drawn(rect: &RectTransform, units: f32) -> [f32; 4] {
        let parent_size = Vec2::new(PARENT.width, PARENT.height) / units;
        let (min, size) = rect.rect_in(Vec2::ZERO, parent_size);
        let pivot = min + rect.pivot * size;
        let a = (pivot + (min - pivot) * rect.scale) * units;
        let b = a + size * rect.scale * units;
        [a.x, a.y, b.x, b.y]
    }

    fn box_of(x: f32, y: f32, w: f32, h: f32) -> RectTransform {
        RectTransform {
            anchor_min: Vec2::ZERO,
            anchor_max: Vec2::ZERO,
            pivot: Vec2::splat(0.5),
            anchored_position: Vec2::new(x + w / 2.0, y + h / 2.0),
            size_delta: Vec2::new(w, h),
            scale: Vec2::ONE,
        }
    }

    fn close(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-3)
    }

    #[test]
    fn dragging_the_body_moves_the_position_in_the_parents_units() {
        let start = box_of(100.0, 100.0, 50.0, 50.0);
        let held = drag(Grip::Body, start, 2.0);
        let from = held.from;
        let (moved, guides) = held.update(from + Vec2::new(40.0, -20.0), None);
        assert_eq!(
            moved.anchored_position,
            start.anchored_position + Vec2::new(20.0, -10.0),
            "forty pixels at two pixels a unit"
        );
        assert_eq!(moved.size_delta, start.size_delta);
        assert!(guides.is_empty());
    }

    #[test]
    fn a_resize_keeps_the_opposite_edge_whatever_the_pivot() {
        for pivot in [Vec2::ZERO, Vec2::splat(0.5), Vec2::new(1.0, 0.25)] {
            let start = RectTransform {
                pivot,
                scale: Vec2::new(2.0, 1.0),
                ..box_of(100.0, 100.0, 50.0, 50.0)
            };
            let before = drawn(&start, 1.0);
            let right = drag(Grip::Edge { x: 1, y: 0 }, start, 1.0);
            let (wider, _) = right.update(right.from + Vec2::new(30.0, 99.0), None);
            let after = drawn(&wider, 1.0);
            assert!(
                close(after, [before[0], before[1], before[2] + 30.0, before[3]]),
                "pivot {pivot}: {before:?} -> {after:?}"
            );

            let corner = drag(Grip::Edge { x: -1, y: -1 }, start, 1.0);
            let (grown, _) = corner.update(corner.from + Vec2::new(-10.0, -20.0), None);
            let after = drawn(&grown, 1.0);
            assert!(
                close(
                    after,
                    [before[0] - 10.0, before[1] - 20.0, before[2], before[3]]
                ),
                "pivot {pivot}: {before:?} -> {after:?}"
            );
        }
    }

    #[test]
    fn the_pivot_handle_moves_the_pivot_and_not_the_rectangle() {
        let start = box_of(100.0, 100.0, 200.0, 100.0);
        let before = drawn(&start, 1.0);
        let held = drag(Grip::Pivot, start, 1.0);
        // From the centre to a quarter of the way across and the top edge.
        let centre = Vec2::new(200.0, 150.0);
        let held = RectDrag {
            from: centre,
            ..held
        };
        let (moved, _) = held.update(centre + Vec2::new(-50.0, -50.0), None);
        assert!(
            moved.pivot.abs_diff_eq(Vec2::new(0.25, 0.0), 1e-5),
            "{moved:?}"
        );
        assert!(close(drawn(&moved, 1.0), before));
    }

    #[test]
    fn a_drag_near_the_parents_centre_or_edge_lands_on_it() {
        let start = box_of(100.0, 100.0, 100.0, 100.0);
        let held = drag(Grip::Body, start, 1.0);
        // The box's centre to within three pixels of the parent's centre line.
        let near_centre = held.from + Vec2::new(500.0 - 150.0 + 3.0, 0.0);
        let (snapped, guides) = held.update(near_centre, Some(6.0));
        assert_eq!(drawn(&snapped, 1.0)[0] + 50.0, 500.0);
        assert_eq!(
            guides,
            [Guide {
                vertical: true,
                at: 500.0
            }]
        );

        let (free, guides) = held.update(near_centre, None);
        assert_eq!(drawn(&free, 1.0)[0] + 50.0, 503.0);
        assert!(guides.is_empty());

        // The left edge, four pixels from the parent's.
        let (edge, guides) = held.update(held.from + Vec2::new(-96.0, 0.0), Some(6.0));
        assert_eq!(drawn(&edge, 1.0)[0], 0.0);
        assert_eq!(
            guides,
            [Guide {
                vertical: true,
                at: 0.0
            }]
        );

        // A resize snaps the edge it moves.
        let right = drag(Grip::Edge { x: 1, y: 0 }, start, 1.0);
        let (wider, _) = right.update(right.from + Vec2::new(500.0 - 200.0 - 2.0, 0.0), Some(6.0));
        assert_eq!(drawn(&wider, 1.0)[2], 500.0);
    }
}
