//! The editor's own controls, built once so every tool looks the same.
//!
//! egui's widgets are the parts; these are the assemblies the design repeats —
//! a section bar, an icon toggle, a search well, an asset slot. A panel that
//! hand-rolls one of them is how the editor drifted into four slightly
//! different headers before, so a new tool reaches for these first.
//!
//! The property row and the axis field are not here: they live with the
//! registry's inspector, which draws components the engine has never heard of
//! and so has to own the shape every row takes.

use egui::Sense;

use super::icons::Icon;

use super::icons;

pub use orrin_registry::inspect::{axes, property, property_with};

/// A section bar is a step taller than a control, so the structure of a
/// component list reads before its contents do.
const SECTION_HEIGHT: f32 = 26.0;

/// The side margin of a panel whose content pads itself — the inspector, whose
/// section bars run edge to edge and whose other content does not.
pub const PAD: f32 = 8.0;

/// `add` with the panel's side margin.
pub fn padded<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::new()
        .inner_margin(egui::Margin::symmetric(PAD as i8, 0))
        .show(ui, add)
        .inner
}

/// An asset slot's thumbnail. The rendered previews are larger and are scaled
/// down to this, so a high-DPI display still gets a sharp one.
pub const THUMB: f32 = 36.0;

/// What a toolbar button is showing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tool {
    /// Bare until hovered.
    Off,
    /// A filter or toggle that is on: a neutral raise. Several can be on at
    /// once, so it must not look like a selection.
    On,
    /// The one active mode of a group: the accent tint.
    Selected,
    /// Not built yet. Legible, never lit, and says so on hover.
    Disabled,
}

/// The button every toolbar is made of: an icon, optionally a short word or
/// figure beside it, in one of [`Tool`]'s four states.
///
/// Painted by hand rather than as an `egui::Button`, because egui draws a
/// frameless button the same hovered as idle, and a framed one with a resting
/// fill — a toolbar wants neither. `tint` colours the icon and text while the
/// button is on; off, everything falls back to the weak text colour.
pub fn tool_button(
    ui: &mut egui::Ui,
    icon: Icon,
    tint: Option<egui::Color32>,
    text: Option<egui::RichText>,
    state: Tool,
    hint: &str,
) -> egui::Response {
    const PAD: f32 = 5.0;
    let height = ui.spacing().interact_size.y + 2.0;
    let glyph = icons::ROW - 2.0;
    let galley = text.map(|text| {
        egui::WidgetText::from(text).into_galley(
            ui,
            Some(egui::TextWrapMode::Extend),
            f32::INFINITY,
            egui::TextStyle::Button,
        )
    });
    let width = galley.as_ref().map_or(height, |galley| {
        PAD + glyph + PAD + galley.size().x + PAD + 1.0
    });
    let sense = if state == Tool::Disabled {
        Sense::hover()
    } else {
        Sense::click()
    };
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, height), sense);

    let visuals = ui.visuals();
    let live = state != Tool::Disabled;
    let fill = if live && response.is_pointer_button_down_on() {
        Some(visuals.widgets.active.weak_bg_fill)
    } else if state == Tool::Selected {
        Some(visuals.selection.bg_fill)
    } else if live && response.hovered() {
        Some(visuals.widgets.hovered.weak_bg_fill)
    } else if state == Tool::On {
        Some(visuals.widgets.inactive.weak_bg_fill)
    } else {
        None
    };
    if let Some(fill) = fill {
        ui.painter()
            .rect_filled(rect, visuals.widgets.inactive.corner_radius, fill);
    }

    let colour = match state {
        Tool::On | Tool::Selected => tint.unwrap_or(visuals.strong_text_color()),
        Tool::Off if response.hovered() => visuals.text_color(),
        Tool::Off => visuals.weak_text_color(),
        // Opaque, between the panel and the text: dimmer than a live control,
        // and never egui's half-alpha fade, which reads as missing.
        Tool::Disabled => visuals.panel_fill.lerp_to_gamma(visuals.text_color(), 0.62),
    };
    let left = if galley.is_some() {
        rect.left() + PAD + glyph / 2.0
    } else {
        rect.center().x
    };
    let at =
        egui::Rect::from_center_size(egui::pos2(left, rect.center().y), egui::Vec2::splat(glyph));
    icons::paint(ui, icon, at, colour);
    if let Some(galley) = galley {
        let at = egui::pos2(at.right() + PAD, rect.center().y - galley.size().y / 2.0);
        ui.painter().galley(at, galley, colour);
    }

    response.on_hover_text(hint)
}

/// An icon-only toolbar toggle. `on` is a held mode (a gizmo, a toggle) and
/// draws as a selection.
pub fn toggle_icon(ui: &mut egui::Ui, icon: Icon, on: bool, hint: &str) -> egui::Response {
    let state = if on { Tool::Selected } else { Tool::Off };
    tool_button(ui, icon, None, None, state, hint)
}

/// An icon and a short word, for a toolbar toggle whose state is a value
/// rather than on/off — "World", "0.25".
pub fn icon_label_button(
    ui: &mut egui::Ui,
    icon: Icon,
    text: &str,
    on: bool,
    hint: &str,
) -> egui::Response {
    let state = if on { Tool::Selected } else { Tool::Off };
    tool_button(ui, icon, None, Some(egui::RichText::new(text)), state, hint)
}

/// A squircle: the square, filled button for the actions that sit beside a
/// field or a title — add, and the `⋯` menus. The height of a control, so it
/// lines up with the field beside it.
pub fn squircle(ui: &mut egui::Ui, icon: Icon, hint: &str) -> egui::Response {
    let side = ui.spacing().interact_size.y;
    let (rect, response) = ui.allocate_exact_size(egui::Vec2::splat(side), Sense::click());
    let visuals = ui.visuals();
    let fill = if response.is_pointer_button_down_on() {
        visuals.widgets.active.weak_bg_fill
    } else if response.hovered() {
        visuals.widgets.hovered.weak_bg_fill
    } else {
        visuals.widgets.inactive.weak_bg_fill
    };
    let colour = if response.hovered() {
        visuals.strong_text_color()
    } else {
        visuals.text_color()
    };
    ui.painter()
        .rect_filled(rect, egui::CornerRadius::same((side * 0.3) as u8), fill);
    let glyph = icons::ROW - 2.0;
    icons::paint(
        ui,
        icon,
        egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(glyph)),
        colour,
    );
    response.on_hover_text(hint)
}

/// A [`squircle`] that opens a menu below itself.
pub fn squircle_menu(
    ui: &mut egui::Ui,
    icon: Icon,
    hint: &str,
    add: impl FnOnce(&mut egui::Ui),
) -> egui::Response {
    let response = squircle(ui, icon, hint);
    let popup = ui.make_persistent_id(("squircle_menu", hint));
    if response.clicked() {
        ui.memory_mut(|memory| memory.toggle_popup(popup));
    }
    egui::popup::popup_below_widget(
        ui,
        popup,
        &response,
        egui::PopupCloseBehavior::CloseOnClick,
        |ui| {
            ui.set_min_width(180.0);
            add(ui);
        },
    );
    response
}

/// The corner radius of [`toolbar_frame`], which the renderer's blur behind it
/// has to match.
pub const TOOLBAR_RADIUS: u8 = 6;

/// The floating frame a toolbar group sits in over the scene: the bar colour,
/// translucent over the frosted backdrop the renderer draws behind it.
pub fn toolbar_frame(visuals: &egui::Visuals, bar: egui::Color32) -> egui::Frame {
    egui::Frame::new()
        .fill(bar.gamma_multiply(super::theme::FLOATING_OPACITY))
        .stroke(visuals.window_stroke)
        .corner_radius(TOOLBAR_RADIUS)
        .inner_margin(3)
}

/// A search field: a sunken well with a magnifier, and a clear button once
/// there is something to clear. Fills the width it is given.
pub fn search(ui: &mut egui::Ui, text: &mut String, hint: &str) -> egui::Response {
    let height = ui.spacing().interact_size.y;
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), Sense::hover());

    let visuals = ui.visuals();
    let weak = visuals.weak_text_color();
    let stroke = egui::Stroke::new(1.0_f32, visuals.widgets.noninteractive.bg_stroke.color);
    ui.painter().rect(
        rect,
        visuals.widgets.inactive.corner_radius,
        visuals.extreme_bg_color,
        stroke,
        egui::StrokeKind::Inside,
    );
    let glyph = egui::Rect::from_center_size(
        egui::pos2(rect.left() + 12.0, rect.center().y),
        egui::Vec2::splat(icons::ROW - 3.0),
    );
    icons::paint(ui, icons::search(), glyph, weak);

    let clear = egui::Rect::from_center_size(
        egui::pos2(rect.right() - 11.0, rect.center().y),
        egui::Vec2::splat(16.0),
    );
    let field_right = if text.is_empty() {
        rect.right() - 4.0
    } else {
        clear.left() - 2.0
    };
    let field = egui::Rect::from_min_max(
        egui::pos2(glyph.right() + 4.0, rect.top()),
        egui::pos2(field_right, rect.bottom()),
    );
    let response = ui.put(
        field,
        egui::TextEdit::singleline(text)
            .hint_text(hint)
            .frame(false)
            .vertical_align(egui::Align::Center)
            .desired_width(field.width()),
    );
    if !text.is_empty() {
        let x = ui.put(
            clear,
            egui::Button::new(icons::text(icons::x(), 12.0, weak)).frame(false),
        );
        if x.on_hover_text("Clear").clicked() {
            text.clear();
        }
    }
    if response.has_focus() {
        ui.painter().rect_stroke(
            rect,
            ui.visuals().widgets.inactive.corner_radius,
            ui.visuals().selection.stroke,
            egui::StrokeKind::Inside,
        );
    }
    response
}

/// A collapsible section: a full-width bar with a disclosure chevron, the
/// section's icon in its tint, its title, and optionally a `⋯` menu at the
/// right. Open by default, and remembered per `id`.
///
/// Drawn by hand rather than with `CollapsingHeader` so the whole bar is the
/// target, the icon has a place, and the menu sits inside the bar rather than
/// beside it.
pub fn section<R>(
    ui: &mut egui::Ui,
    id: egui::Id,
    icon: Icon,
    tint: egui::Color32,
    title: &str,
    menu: Option<&mut dyn FnMut(&mut egui::Ui)>,
    body: impl FnOnce(&mut egui::Ui) -> R,
) -> Option<R> {
    section_titled(ui, id, icon, tint, title, None, menu, body)
}

/// [`section`] with its title in `title_colour` rather than the strong text
/// colour — how the inspector marks a component a prefab instance overrides.
#[allow(clippy::too_many_arguments)]
pub fn section_titled<R>(
    ui: &mut egui::Ui,
    id: egui::Id,
    icon: Icon,
    tint: egui::Color32,
    title: &str,
    title_colour: Option<egui::Color32>,
    menu: Option<&mut dyn FnMut(&mut egui::Ui)>,
    body: impl FnOnce(&mut egui::Ui) -> R,
) -> Option<R> {
    let mut state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true);

    let width = ui.available_width();
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(width, SECTION_HEIGHT), Sense::click());
    let visuals = ui.visuals();
    let outline = visuals.widgets.noninteractive.bg_stroke.color;
    let fill = if response.hovered() {
        visuals.widgets.hovered.weak_bg_fill
    } else {
        visuals.window_fill.to_opaque()
    };
    let weak = visuals.weak_text_color();
    let strong = title_colour.unwrap_or_else(|| visuals.strong_text_color());
    let painter = ui.painter();
    // Square: a band across the panel, not a button laid over it.
    painter.rect_filled(rect, 0.0, fill);
    painter.hline(
        rect.x_range(),
        rect.bottom(),
        egui::Stroke::new(1.0_f32, outline),
    );

    let chevron = if state.is_open() {
        icons::chevron_down()
    } else {
        icons::chevron_right()
    };
    let mut x = rect.left() + PAD;
    for (glyph, color, size) in [(chevron, weak, 12.0), (icon, tint, icons::ROW - 2.0)] {
        let at = egui::Rect::from_center_size(
            egui::pos2(x + size / 2.0, rect.center().y),
            egui::Vec2::splat(size),
        );
        icons::paint(ui, glyph, at, color);
        x = at.right() + 6.0;
    }
    ui.painter().text(
        egui::pos2(x, rect.center().y),
        egui::Align2::LEFT_CENTER,
        title,
        egui::TextStyle::Button.resolve(ui.style()),
        strong,
    );

    if let Some(menu) = menu {
        let at = egui::Rect::from_center_size(
            egui::pos2(rect.right() - PAD - 10.0, rect.center().y),
            egui::Vec2::splat(SECTION_HEIGHT - 4.0),
        );
        let hint = format!("{title} actions");
        ui.put(at, |ui: &mut egui::Ui| {
            squircle_menu(ui, icons::ellipsis(), &hint, |ui| menu(ui))
        });
    }

    if response.clicked() {
        state.toggle(ui);
    }
    let body = state.show_body_unindented(ui, |ui| {
        ui.add_space(4.0);
        // No left margin: a property label insets itself, and lines up with
        // the header's padded content that way.
        let inner = egui::Frame::new()
            .inner_margin(egui::Margin {
                left: 0,
                right: PAD as i8,
                top: 0,
                bottom: 0,
            })
            .show(ui, body)
            .inner;
        ui.add_space(6.0);
        inner
    });
    body.map(|inner| inner.inner)
}

/// A slot holding one asset: its thumbnail, its name and one line about it,
/// and a chevron saying the slot opens a picker. Clicking it is the caller's
/// to answer; this only draws and reports.
pub fn asset_slot(
    ui: &mut egui::Ui,
    thumbnail: Option<&egui::TextureHandle>,
    fallback: Icon,
    title: &str,
    subtitle: &str,
) -> egui::Response {
    let height = THUMB + 8.0;
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), height), Sense::click());
    let visuals = ui.visuals();
    let stroke = if response.hovered() {
        visuals.widgets.hovered.bg_stroke
    } else {
        egui::Stroke::new(1.0_f32, visuals.widgets.noninteractive.bg_stroke.color)
    };
    let radius = visuals.widgets.inactive.corner_radius;
    let weak = visuals.weak_text_color();
    let strong = visuals.strong_text_color();
    ui.painter().rect(
        rect,
        radius,
        visuals.extreme_bg_color,
        stroke,
        egui::StrokeKind::Inside,
    );

    let thumb =
        egui::Rect::from_min_size(rect.min + egui::vec2(4.0, 4.0), egui::Vec2::splat(THUMB));
    ui.painter()
        .rect_filled(thumb, radius, visuals.widgets.inactive.bg_fill);
    match thumbnail {
        Some(texture) => {
            egui::Image::new((texture.id(), egui::Vec2::splat(THUMB))).paint_at(ui, thumb);
        }
        None => {
            let glyph = egui::Rect::from_center_size(thumb.center(), egui::Vec2::splat(18.0));
            icons::paint(ui, fallback, glyph, weak);
        }
    }

    let text_left = thumb.right() + 8.0;
    let text_right = rect.right() - 24.0;
    let clip = egui::Rect::from_min_max(
        egui::pos2(text_left, rect.top()),
        egui::pos2(text_right, rect.bottom()),
    );
    let painter = ui.painter().with_clip_rect(clip);
    painter.text(
        egui::pos2(text_left, rect.center().y - 1.0),
        egui::Align2::LEFT_BOTTOM,
        title,
        egui::TextStyle::Body.resolve(ui.style()),
        strong,
    );
    painter.text(
        egui::pos2(text_left, rect.center().y + 1.0),
        egui::Align2::LEFT_TOP,
        subtitle,
        egui::TextStyle::Monospace.resolve(ui.style()),
        weak,
    );
    let chevron = egui::Rect::from_center_size(
        egui::pos2(rect.right() - 12.0, rect.center().y),
        egui::Vec2::splat(12.0),
    );
    icons::paint(ui, icons::chevron_down(), chevron, weak);

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fill a tool button painted behind itself, if any: a plain rect
    /// the size of the button, as opposed to the icon's textured one.
    fn fill_of(state: Tool, hovered: bool) -> Option<egui::Color32> {
        let ctx = egui::Context::default();
        super::super::theme::install_fonts(&ctx);
        super::super::theme::apply(&ctx, &super::super::theme::Theme::default());
        let mut fill = None;
        for _ in 0..3 {
            let mut input = egui::RawInput::default();
            if hovered {
                input
                    .events
                    .push(egui::Event::PointerMoved(egui::pos2(20.0, 20.0)));
            }
            let output = ctx.run(input, |ctx| {
                // A new area fades in over its first frames, which would scale
                // every fill's alpha.
                egui::Area::new("tool".into())
                    .fade_in(false)
                    .fixed_pos(egui::pos2(10.0, 10.0))
                    .show(ctx, |ui| {
                        tool_button(ui, icons::play(), None, None, state, "Play")
                    });
            });
            fill = output
                .shapes
                .iter()
                .find_map(|clipped| match &clipped.shape {
                    egui::Shape::Rect(rect)
                        if rect.brush.is_none() && rect.rect.width() >= 20.0 =>
                    {
                        Some(rect.fill)
                    }
                    _ => None,
                });
        }
        fill
    }

    /// Off is bare until the pointer reaches it — every toolbar button has to
    /// answer the pointer, or it reads as a label.
    #[test]
    fn an_idle_tool_shows_nothing_until_hovered() {
        assert_eq!(fill_of(Tool::Off, false), None);
        assert!(fill_of(Tool::Off, true).is_some());
    }

    /// On (a filter that is showing) is a neutral raise; Selected (the one
    /// active mode) is the accent tint. The two must not be confused, or five
    /// filters that are all on look like five selected tools.
    #[test]
    fn on_is_neutral_and_selected_is_the_accent() {
        let theme = super::super::theme::Theme::default();
        assert_eq!(fill_of(Tool::Selected, false), Some(theme.select_fill()));
        let on = fill_of(Tool::On, false).expect("on is filled");
        assert_ne!(on, theme.select_fill());
    }

    /// The add and actions buttons are squircles: a square with a generous
    /// corner, filled at rest so it reads as a button beside a field.
    #[test]
    fn a_squircle_is_a_filled_rounded_square() {
        let ctx = egui::Context::default();
        super::super::theme::install_fonts(&ctx);
        super::super::theme::apply(&ctx, &super::super::theme::Theme::default());
        let mut found = None;
        for _ in 0..2 {
            let output = ctx.run(Default::default(), |ctx| {
                egui::Area::new("squircle".into())
                    .fade_in(false)
                    .fixed_pos(egui::pos2(10.0, 10.0))
                    .show(ctx, |ui| squircle(ui, icons::plus(), "Add"));
            });
            found = output
                .shapes
                .iter()
                .find_map(|clipped| match &clipped.shape {
                    egui::Shape::Rect(rect) if rect.brush.is_none() && rect.fill.a() > 0 => {
                        Some((rect.rect, rect.corner_radius))
                    }
                    _ => None,
                });
        }
        let (rect, radius) = found.expect("a filled shape");
        assert_eq!(rect.width(), rect.height());
        assert!(
            radius.nw as f32 >= rect.width() * 0.25,
            "{radius:?} on {rect:?}"
        );
    }

    /// Disabled answers the pointer with a tooltip and nothing else.
    #[test]
    fn a_disabled_tool_never_lights_up() {
        assert_eq!(fill_of(Tool::Disabled, true), None);
    }
}
