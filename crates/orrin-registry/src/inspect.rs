//! Drawing a [`Value`] as editable widgets, with no knowledge of the type it
//! came from.
//!
//! The default every registration gets. A component whose numbers mean
//! something — a light's lumens, an angle in degrees — replaces it through
//! [`Registry::set_inspector`](crate::Registry::set_inspector); everything else,
//! including every component a game assembly registers, is inspectable the
//! moment it is registered rather than when someone writes a panel for it.
//!
//! What this deliberately cannot do is change a value's *shape*: no switching
//! enum variant, no adding or removing list elements. Both need the set of
//! variants and the element's default, which the v1 vtable does not carry —
//! and a caller that could only offer half the variants would be worse than one
//! that offers none.

use crate::value::Value;

/// Draw `value`'s widgets, reporting whether the user moved any of them.
pub type InspectFn = fn(&mut egui::Ui, &mut Value) -> bool;

/// Speed of every unlabelled numeric drag. Slow enough that a drag across the
/// panel is a readable adjustment rather than a jump to a random magnitude.
const DRAG_SPEED: f64 = 0.05;

/// X, Y and Z. Defined here because this module draws vectors for components
/// the engine has never heard of; the editor's theme re-exports the same three
/// so its gizmo and its hand-drawn sections cannot drift from them.
pub const AXIS: [egui::Color32; 3] = [
    egui::Color32::from_rgb(0xE5, 0x53, 0x4B),
    egui::Color32::from_rgb(0x5F, 0xB5, 0x5A),
    egui::Color32::from_rgb(0x4C, 0x86, 0xF0),
];

/// The chip's letter, dark on every axis colour, which are all mid-value.
const CHIP_TEXT: egui::Color32 = egui::Color32::from_rgb(11, 12, 14);
const CHIP_WIDTH: f32 = 16.0;
/// How far a label sits in from the edge of its row, so it does not butt
/// against the panel's border.
const LABEL_INSET: f32 = 8.0;
const AXIS_GAP: f32 = 4.0;

/// The label column: a share of the row, so a narrow dock still leaves the
/// value most of the width, bounded so a wide one does not strand the label far
/// from what it names.
pub fn label_width(ui: &egui::Ui) -> f32 {
    // Whole pixels, here and in the field widths: a fractional column rounds
    // differently from one row to the next and the edges stop lining up.
    (ui.available_width() * 0.34).clamp(56.0, 116.0).round()
}

/// The slot every row keeps at its right end: a reset button where a row has
/// one, empty where it does not. Kept on every row so the value columns end
/// on one edge as well as starting on one.
pub const TRAIL: f32 = 22.0;

/// One property: a weak, truncating label in a fixed column, then the value
/// filling the rest. Every row in every section goes through this, which is
/// what lines up the values of a component down one edge.
pub fn property<R>(ui: &mut egui::Ui, label: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    property_with(ui, label, add, |_| {})
}

/// [`property`] with something in the row's trailing slot.
///
/// The value and the trailing slot are drawn into child `Ui`s over rects
/// reserved up front, rather than added in sequence: an egui widget moves the
/// cursor to wherever it actually ended, so anything wider than its share —
/// a long number, a long name — would push the rest of the row out of line.
pub fn property_with<R>(
    ui: &mut egui::Ui,
    label: &str,
    add: impl FnOnce(&mut egui::Ui) -> R,
    trail: impl FnOnce(&mut egui::Ui),
) -> R {
    ui.horizontal(|ui| {
        let height = ui.spacing().interact_size.y;
        let size = egui::vec2(label_width(ui), height);
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::hover());
        // Painted rather than added as a `Label`, for the same reason the
        // value is drawn into a reserved rect.
        let galley = egui::WidgetText::from(label).into_galley(
            ui,
            Some(egui::TextWrapMode::Truncate),
            (rect.width() - LABEL_INSET).max(0.0),
            egui::TextStyle::Body,
        );
        let at = egui::pos2(
            rect.left() + LABEL_INSET,
            rect.center().y - galley.size().y / 2.0,
        );
        if galley.elided {
            response.on_hover_text(label);
        }
        ui.painter()
            .galley(at, galley, ui.visuals().weak_text_color());

        let gap = ui.spacing().item_spacing.x;
        let width = (ui.available_width() - TRAIL - gap).max(0.0).floor();
        let (value_rect, _) =
            ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
        let (trail_rect, _) =
            ui.allocate_exact_size(egui::vec2(TRAIL, height), egui::Sense::hover());

        let mut value = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(value_rect)
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
        );
        // Sliders and single-line edits take their width from these, not from
        // the space they are given; without this they stop at egui's defaults
        // and leave the column ragged.
        value.spacing_mut().slider_width = (width - 64.0).max(32.0);
        value.spacing_mut().text_edit_width = width;
        let inner = add(&mut value);
        // A child does not grow its parent, so a value taller than one control
        // (an asset slot) claims its height here.
        ui.set_min_height(value.min_rect().height());

        let mut slot = ui.new_child(egui::UiBuilder::new().max_rect(trail_rect).layout(
            egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
        ));
        trail(&mut slot);
        inner
    })
    .inner
}

/// A figure at two decimals with the zeros that say nothing dropped: `0`,
/// `1.5`, `-15.27`. A field narrow enough to sit three to a row cannot spare
/// width for `0.00`.
pub fn compact(value: f32) -> String {
    let text = format!("{value:.2}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text == "-0" {
        "0".to_owned()
    } else {
        text.to_owned()
    }
}

/// Three axis fields side by side, each a coloured chip and a number. The
/// number drags and types as any `DragValue` does; the chip scrubs too, so a
/// field whose number is being edited still has a handle.
pub fn axes(ui: &mut egui::Ui, values: [&mut f32; 3], speed: f64, suffix: &str) -> bool {
    let height = ui.spacing().interact_size.y;
    let width = ((ui.available_width() - 2.0 * AXIS_GAP) / 3.0)
        .floor()
        .max(CHIP_WIDTH + 16.0);
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = AXIS_GAP;
        let [x, y, z] = values;
        for (axis, value) in [x, y, z].into_iter().enumerate() {
            changed |= axis_field(ui, axis, value, speed, suffix, egui::vec2(width, height));
        }
    });
    changed
}

/// Move `value` by a chip drag of `dx` points, at the field's own drag speed so
/// the chip and the number feel like one control.
fn scrub(value: &mut f32, dx: f32, speed: f32) -> bool {
    if dx == 0.0 {
        return false;
    }
    *value += dx * speed;
    true
}

fn axis_field(
    ui: &mut egui::Ui,
    axis: usize,
    value: &mut f32,
    speed: f64,
    suffix: &str,
    size: egui::Vec2,
) -> bool {
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let visuals = ui.visuals();
    let radius = visuals.widgets.inactive.corner_radius;
    let outline = visuals.widgets.noninteractive.bg_stroke.color;
    let well = visuals.extreme_bg_color;
    let hover = visuals.widgets.hovered.bg_stroke;
    let painter = ui.painter().clone();
    painter.rect(
        rect,
        radius,
        well,
        egui::Stroke::new(1.0_f32, outline),
        egui::StrokeKind::Inside,
    );

    let chip = egui::Rect::from_min_size(rect.min, egui::vec2(CHIP_WIDTH, size.y));
    // From the layout position, not the parent: every row of a component
    // draws an X chip, and they must not share one drag state.
    let id = ui.auto_id_with(("axis_chip", axis));
    let handle = ui
        .interact(chip, id, egui::Sense::drag())
        .on_hover_cursor(egui::CursorIcon::ResizeHorizontal);
    let lift = if handle.hovered() || handle.dragged() {
        1.15
    } else {
        1.0
    };
    painter.rect_filled(
        chip,
        egui::CornerRadius {
            nw: radius.nw,
            sw: radius.sw,
            ne: 0,
            se: 0,
        },
        AXIS[axis].gamma_multiply(lift),
    );
    painter.text(
        chip.center(),
        egui::Align2::CENTER_CENTER,
        ["X", "Y", "Z"][axis],
        egui::FontId::proportional(10.0),
        CHIP_TEXT,
    );

    let changed = handle.dragged() && scrub(value, handle.drag_delta().x, speed as f32);

    let field = egui::Rect::from_min_max(egui::pos2(chip.right(), rect.top()), rect.max);
    // A child over the rect already reserved, clipped to it: a value too long
    // for the field is cut off rather than allowed to widen it and push the
    // next field along.
    let mut number = ui.new_child(egui::UiBuilder::new().max_rect(field).layout(
        egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
    ));
    number.set_clip_rect(field.intersect(ui.clip_rect()));
    // The well painted above is the field's surface; the button egui would
    // draw for a `DragValue` would be a second one on top of it.
    let widgets = &mut number.visuals_mut().widgets;
    for state in [
        &mut widgets.inactive,
        &mut widgets.hovered,
        &mut widgets.active,
    ] {
        state.weak_bg_fill = egui::Color32::TRANSPARENT;
        state.bg_fill = egui::Color32::TRANSPARENT;
        state.bg_stroke = egui::Stroke::NONE;
    }
    let response = number.add(
        egui::DragValue::new(value)
            .speed(speed)
            .suffix(suffix)
            .custom_formatter(|n, _| compact(n as f32)),
    );
    if response.hovered() || response.has_focus() || handle.hovered() {
        painter.rect_stroke(rect, radius, hover, egui::StrokeKind::Inside);
    }
    changed | response.changed()
}

/// The default [`InspectFn`]: one row per field, nested under a collapsing
/// header wherever a field is itself composite.
pub fn value(ui: &mut egui::Ui, value: &mut Value) -> bool {
    match value {
        Value::Struct(fields) => fields_rows(ui, fields),
        Value::Enum { variant, fields } => {
            // The variant is shown, not chosen — see the module docs.
            property(ui, "variant", |ui| ui.label(variant.as_str()));
            fields_rows(ui, fields)
        }
        Value::List(items) => {
            let mut changed = false;
            for (index, item) in items.iter_mut().enumerate() {
                changed |= row(ui, &format!("[{index}]"), item);
            }
            changed
        }
        leaf => ui.horizontal(|ui| self::leaf(ui, leaf)).inner,
    }
}

/// A snake_case field name as a label: `is_trigger` reads "Is trigger".
fn label(name: &str) -> String {
    let mut out = name.replace('_', " ");
    if let Some(first) = out.get(0..1) {
        let upper = first.to_uppercase();
        out.replace_range(0..1, &upper);
    }
    out
}

fn fields_rows(ui: &mut egui::Ui, fields: &mut [(String, Value)]) -> bool {
    let mut changed = false;
    for (name, field) in fields {
        changed |= row(ui, name, field);
    }
    changed
}

/// A leaf gets a label beside its widget; anything composite gets a header it
/// can be folded away under, because a component of components otherwise fills
/// the panel with rows whose nesting is invisible.
fn row(ui: &mut egui::Ui, name: &str, field: &mut Value) -> bool {
    match field {
        Value::Struct(_) | Value::Enum { .. } | Value::List(_) => {
            egui::CollapsingHeader::new(label(name))
                .default_open(true)
                .show(ui, |ui| value(ui, field))
                .body_returned
                .unwrap_or(false)
        }
        leaf => property(ui, &label(name), |ui| self::leaf(ui, leaf)),
    }
}

fn leaf(ui: &mut egui::Ui, value: &mut Value) -> bool {
    match value {
        Value::Bool(v) => ui.checkbox(v, "").changed(),
        Value::I32(v) => ui.add(egui::DragValue::new(v)).changed(),
        Value::U32(v) => ui.add(egui::DragValue::new(v)).changed(),
        Value::F32(v) => ui.add(egui::DragValue::new(v).speed(DRAG_SPEED)).changed(),
        Value::String(v) => ui.text_edit_singleline(v).changed(),
        Value::Vec3(v) => axes(ui, [&mut v.x, &mut v.y, &mut v.z], DRAG_SPEED, ""),
        Value::Quat(v) => quat(ui, v),
        // Retargeting a reference needs a picker over the live world, which is
        // one level above a function that has only the value. Shown so the link
        // is at least visible.
        Value::Entity(id) => {
            ui.weak(id.to_string());
            false
        }
        // Routed to `value` by `row` and never reached here.
        Value::Struct(_) | Value::Enum { .. } | Value::List(_) => false,
    }
}

/// Edited as Euler degrees in the same convention as the C# scripting API
/// (`Quaternion.Euler`): Z-X-Y application, i.e. glam's intrinsic
/// `EulerRot::YXZ`. The three fields read X, Y, Z (pitch, yaw, roll) while the
/// conversion takes and yields (yaw, pitch, roll), hence the reordering.
///
/// Written back only when a drag actually moved, because euler is a lossy
/// coordinate for a quaternion: recomposing every frame would let a rotation
/// creep while nobody was touching it.
fn quat(ui: &mut egui::Ui, value: &mut glam::Quat) -> bool {
    let (yaw, pitch, roll) = value.to_euler(glam::EulerRot::YXZ);
    let mut euler = [pitch.to_degrees(), yaw.to_degrees(), roll.to_degrees()];
    let [x, y, z] = &mut euler;
    let changed = axes(ui, [x, y, z], 0.5, "°");
    if changed {
        *value = glam::Quat::from_euler(
            glam::EulerRot::YXZ,
            euler[1].to_radians(),
            euler[0].to_radians(),
            euler[2].to_radians(),
        );
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    use crate::EntityId;
    use glam::{EulerRot, Quat, Vec3};

    /// One of every variant, nested, so that adding a `Value` variant without
    /// teaching the inspector about it is caught here rather than by a component
    /// that silently draws nothing.
    fn every_shape() -> Value {
        Value::strukt([
            ("flag", Value::Bool(true)),
            ("count", Value::I32(-3)),
            ("index", Value::U32(7)),
            ("speed", Value::F32(1.5)),
            ("label", Value::String("cube".to_owned())),
            ("position", Value::Vec3(Vec3::new(1.0, 2.0, 3.0))),
            // A compound rotation, not a single-axis one: the euler round trip
            // is exact for the latter, so a fixture built from one would let the
            // creep this test exists for pass unnoticed.
            (
                "rotation",
                Value::Quat(Quat::from_euler(EulerRot::YXZ, 0.3, 0.4, 0.5)),
            ),
            ("target", Value::Entity(EntityId::NIL)),
            ("points", Value::List(vec![Value::F32(0.5)])),
            (
                "light",
                Value::enumeration("Spot", [("angle", Value::F32(30.0))]),
            ),
        ])
    }

    /// Drawing is not editing. Every widget here reports "unchanged" when nobody
    /// touches it, and the value must come back byte-identical — which is the
    /// claim `quat`'s write-back guard exists to keep, since euler is a lossy
    /// coordinate and recomposing unconditionally would let an untouched
    /// rotation creep every frame the panel was open.
    #[test]
    fn drawing_without_interacting_reports_nothing_and_changes_nothing() {
        let before = every_shape();
        // `__run_test_ui` takes an `Fn`, so the value under edit lives behind a
        // cell rather than being captured mutably.
        let after = RefCell::new(before.clone());
        egui::__run_test_ui(|ui| {
            assert!(!value(ui, &mut after.borrow_mut()));
        });
        assert_eq!(crate::diff(&before, &after.into_inner()), Vec::new());
    }

    /// Every fill the frame painted. A field's axis chip is a filled rect, so
    /// this is how its colour is asserted.
    fn painted_fills(draw: impl FnMut(&mut egui::Ui)) -> Vec<egui::Color32> {
        fn collect(shape: &egui::Shape, out: &mut Vec<egui::Color32>) {
            match shape {
                egui::Shape::Rect(rect) => out.push(rect.fill),
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, out)),
                _ => {}
            }
        }
        let ctx = egui::Context::default();
        let mut draw = draw;
        let mut out = Vec::new();
        for _ in 0..2 {
            let output = ctx.run(Default::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| draw(ui));
            });
            out.clear();
            for clipped in &output.shapes {
                collect(&clipped.shape, &mut out);
            }
        }
        out
    }

    fn painted_fills_with_rects(
        draw: impl FnMut(&mut egui::Ui),
    ) -> Vec<(egui::Rect, egui::Color32)> {
        fn collect(shape: &egui::Shape, out: &mut Vec<(egui::Rect, egui::Color32)>) {
            match shape {
                egui::Shape::Rect(rect) => out.push((rect.rect, rect.fill)),
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, out)),
                _ => {}
            }
        }
        let ctx = egui::Context::default();
        let mut draw = draw;
        let mut out = Vec::new();
        for _ in 0..2 {
            let output = ctx.run(Default::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| draw(ui));
            });
            out.clear();
            for clipped in &output.shapes {
                collect(&clipped.shape, &mut out);
            }
        }
        out
    }

    /// A vector reads as X, Y and Z by colour as well as by position, in the
    /// generic inspector as much as in a hand-drawn section — a game
    /// assembly's component is not a second-class citizen of the panel.
    #[test]
    fn a_vector_is_drawn_with_its_three_axis_colours() {
        let mut v = Value::Vec3(Vec3::new(1.0, 2.0, 3.0));
        let fills = painted_fills(|ui| {
            value(ui, &mut v);
        });
        for (axis, colour) in ["X", "Y", "Z"].into_iter().zip(AXIS) {
            assert!(fills.contains(&colour), "no {axis} chip was painted");
        }
    }

    /// Rotations are vectors to the reader too, whatever they are underneath.
    #[test]
    fn a_rotation_is_drawn_with_the_same_axis_colours() {
        let mut v = Value::Quat(Quat::from_euler(EulerRot::YXZ, 0.3, 0.4, 0.5));
        let fills = painted_fills(|ui| {
            value(ui, &mut v);
        });
        assert!(AXIS.iter().all(|colour| fills.contains(colour)));
    }

    /// A number too long for its field is clipped to it, never allowed to
    /// widen it: a field that grows as its value changes shoves its neighbours
    /// along every frame a spinning object's rotation ticks over.
    #[test]
    fn a_long_value_does_not_push_its_neighbours() {
        let chip_lefts = |x: f32| {
            let mut v = Value::Vec3(Vec3::new(x, 0.0, 0.0));
            let fills = painted_fills_with_rects(|ui| {
                ui.set_width(240.0);
                value(ui, &mut v);
            });
            let mut lefts: Vec<f32> = fills
                .into_iter()
                .filter(|(_, fill)| AXIS.contains(fill))
                .map(|(rect, _)| rect.left())
                .collect();
            lefts.sort_by(f32::total_cmp);
            lefts
        };
        assert_eq!(chip_lefts(1.0), chip_lefts(-123_456.8));
    }

    /// Numbers drop the zeros that say nothing, so a short field still shows
    /// the digits that do.
    #[test]
    fn figures_drop_their_trailing_zeros() {
        assert_eq!(compact(0.0), "0");
        assert_eq!(compact(1.0), "1");
        assert_eq!(compact(-15.273), "-15.27");
        assert_eq!(compact(0.5), "0.5");
        assert_eq!(compact(-0.0001), "0");
    }

    /// A chip drag moves its value at the field's own speed, and a press that
    /// does not move reports nothing — otherwise every click on a chip would
    /// post an empty edit into the change stream.
    #[test]
    fn scrubbing_a_chip_moves_by_distance_times_speed() {
        let mut value = 2.0_f32;
        assert!(scrub(&mut value, 10.0, 0.5));
        assert_eq!(value, 7.0);
        assert!(!scrub(&mut value, 0.0, 0.5));
        assert_eq!(value, 7.0);
    }

    #[test]
    fn a_bare_leaf_draws_without_a_surrounding_struct() {
        let leaf = RefCell::new(Value::F32(2.0));
        egui::__run_test_ui(|ui| {
            assert!(!value(ui, &mut leaf.borrow_mut()));
        });
        assert_eq!(leaf.into_inner(), Value::F32(2.0));
    }
}
