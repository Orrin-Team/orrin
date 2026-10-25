//! The editor's frame cut into the phases the renderer draws it in, so a
//! translucent surface can blur what is already beneath it.
//!
//! egui hands back one list of shapes, ordered by layer: all of
//! `Order::Background`, then `Middle`, `Foreground`, `Tooltip` and `Debug`. A menu's glass has to blur the panels under it, and a panel can
//! only be sampled once it has been drawn — to the image, not merely recorded.
//! So the list is cut where one order ends and the next begins, and the
//! renderer blurs between the pieces: the panels, then the viewport's floating
//! controls (`Middle`), then menus and popups (`Foreground`), then tooltips.
//!
//! The cuts are marker shapes painted last. The first goes on the background
//! layer itself, after every panel. The others go on a layer of their own in
//! `Middle` and `Foreground`: egui draws an order's areas first and its other
//! layers after, so such a marker lands after everything its order holds as
//! long as nothing else paints a non-area layer in that order. Nothing in the
//! editor does, and the one egui_dock paints in `Foreground`, a tab drag's drop
//! preview, is a tint nobody reads through glass.
//!
//! One blur per order rather than per surface: a submenu over its menu blurs
//! the panels under both, not the menu. A capture per surface would be exact
//! and would cost a downsample per open popup for a difference of a few pixels
//! of overlap.

use std::sync::Arc;

use egui::epaint::ClippedShape;
use egui::{Order, Shape};

/// The panels; the viewport's controls; menus and popups; tooltips and egui's
/// debug layer.
pub const COUNT: usize = 4;

/// The phase a layer of `order` is drawn in.
pub fn of(order: Order) -> usize {
    match order {
        Order::Background => 0,
        Order::Middle => 1,
        Order::Foreground => 2,
        Order::Tooltip | Order::Debug => 3,
    }
}

/// A translucent surface: a rect the renderer blurs what is beneath into,
/// just before drawing `phase`. In points; the editor scales it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glass {
    pub phase: usize,
    pub rect: egui::Rect,
    pub radius: f32,
}

/// Where a phase begins: the marker carries the phase that follows it.
struct Cut(usize);

/// Paint the three cuts. Run after everything else in the frame has painted,
/// so each marker is the last shape of its layer.
pub fn mark(ctx: &egui::Context) {
    let cut = |next| {
        Shape::Callback(egui::PaintCallback {
            rect: egui::Rect::NOTHING,
            callback: Arc::new(Cut(next)),
        })
    };
    ctx.layer_painter(egui::LayerId::background()).add(cut(1));
    for (order, next) in [(Order::Middle, 2), (Order::Foreground, 3)] {
        let layer = egui::LayerId::new(order, egui::Id::new(("orrin_phase_cut", next)));
        ctx.layer_painter(layer).add(cut(next));
    }
}

/// Split one frame's shapes at the cuts, dropping the markers.
///
/// A missing cut leaves its phase empty and its shapes in the phase before,
/// which draws them in the right order and merely skips one blur.
pub fn split(shapes: Vec<ClippedShape>) -> [Vec<ClippedShape>; COUNT] {
    let mut phases: [Vec<ClippedShape>; COUNT] = Default::default();
    let mut current = 0;
    for shape in shapes {
        if let Shape::Callback(callback) = &shape.shape
            && let Some(Cut(next)) = callback.callback.downcast_ref::<Cut>()
        {
            current = (*next).max(current);
            continue;
        }
        phases[current].push(shape);
    }
    phases
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORDERS: [Order; 5] = [
        Order::Background,
        Order::Middle,
        Order::Foreground,
        Order::Tooltip,
        Order::Debug,
    ];

    /// One frame with a rect painted in each order, so every phase has exactly
    /// one shape and its width says which order it came from — not its colour,
    /// which an area's fade-in scales. Painted the way
    /// the editor paints each order: panels on the background layer, floating
    /// surfaces as areas, and egui's own debug painter.
    fn frame() -> Vec<ClippedShape> {
        let ctx = egui::Context::default();
        // Twice, because an area is invisible on the frame it first appears.
        let mut output = None;
        for _ in 0..2 {
            output = Some(ctx.run(Default::default(), |ctx| {
                let probe = |index: usize| {
                    egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(index as f32 + 1.0, 4.0))
                };
                for (index, order) in ORDERS.iter().enumerate() {
                    let (probe, fill) = (probe(index), egui::Color32::WHITE);
                    match order {
                        Order::Background => {
                            ctx.layer_painter(egui::LayerId::background())
                                .rect_filled(probe, 0.0, fill);
                        }
                        Order::Debug => {
                            ctx.debug_painter().rect_filled(probe, 0.0, fill);
                        }
                        _ => {
                            egui::Area::new(egui::Id::new(("probe", index)))
                                .order(*order)
                                .show(ctx, |ui| ui.painter().rect_filled(probe, 0.0, fill));
                        }
                    }
                }
                mark(ctx);
            }));
        }
        output.expect("ran").shapes
    }

    fn orders(phase: &[ClippedShape]) -> Vec<Order> {
        phase
            .iter()
            .filter_map(|shape| match &shape.shape {
                Shape::Rect(rect) => Some(ORDERS[rect.rect.width() as usize - 1]),
                Shape::Noop => None,
                other => panic!("an unexpected shape {other:?}"),
            })
            .collect()
    }

    #[test]
    fn each_order_lands_in_its_phase_and_the_cuts_are_dropped() {
        let phases = split(frame());
        assert_eq!(
            phases.each_ref().map(|phase| orders(phase)),
            [
                vec![Order::Background],
                vec![Order::Middle],
                vec![Order::Foreground],
                vec![Order::Tooltip, Order::Debug],
            ]
        );
        for order in ORDERS {
            assert!(
                phases[of(order)]
                    .iter()
                    .any(|shape| matches!(&shape.shape, Shape::Rect(_))),
                "{order:?}"
            );
        }
    }

    /// A popup is an area in `Foreground`, and areas are drawn before the
    /// order's other layers — so the cut after it must still follow it, or the
    /// popup would draw before its own blur.
    #[test]
    fn a_popup_area_draws_after_its_blur() {
        let ctx = egui::Context::default();
        let frame = || {
            ctx.run(Default::default(), |ctx| {
                egui::Area::new(egui::Id::new("popup"))
                    .order(Order::Foreground)
                    .show(ctx, |ui| ui.label("menu"));
                egui::CentralPanel::default().show(ctx, |ui| ui.label("panel"));
                mark(ctx);
            })
        };
        frame();
        let phases = split(frame().shapes);
        let text = |phase: &[ClippedShape]| -> Vec<String> {
            phase
                .iter()
                .filter_map(|shape| match &shape.shape {
                    Shape::Text(text) => Some(text.galley.text().to_owned()),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(text(&phases[0]), ["panel"]);
        assert_eq!(text(&phases[2]), ["menu"]);
    }

    #[test]
    fn without_cuts_everything_is_the_first_phase() {
        let shapes: Vec<_> = frame()
            .into_iter()
            .filter(|shape| !matches!(shape.shape, Shape::Callback(_)))
            .collect();
        let count = shapes.len();
        let phases = split(shapes);
        assert_eq!(phases[0].len(), count);
        assert!(phases[1..].iter().all(Vec::is_empty));
    }
}
