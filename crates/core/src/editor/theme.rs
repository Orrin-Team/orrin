use egui::{Color32, CornerRadius, Stroke, Visuals};

const TEXT: Color32 = Color32::from_rgb(214, 218, 224);
const PANEL: Color32 = Color32::from_rgb(24, 25, 28);
const WINDOW: Color32 = Color32::from_rgb(30, 31, 35);
const EXTREME: Color32 = Color32::from_rgb(18, 19, 21);
const FAINT: Color32 = Color32::from_rgb(36, 38, 42);
const WIDGET: Color32 = Color32::from_rgb(45, 47, 52);
const WIDGET_HOVER: Color32 = Color32::from_rgb(56, 59, 66);
const OUTLINE: Color32 = Color32::from_rgb(48, 50, 56);
const ACCENT: Color32 = Color32::from_rgb(80, 140, 255);

// Set on every theme variant so it sticks regardless of the OS preference.
pub fn apply(ctx: &egui::Context) {
    ctx.all_styles_mut(|style| {
        style.visuals = visuals();
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(8.0, 4.0);
        style.spacing.window_margin = egui::Margin::same(8);
    });
}

fn visuals() -> Visuals {
    let mut v = Visuals::dark();

    v.override_text_color = Some(TEXT);
    v.panel_fill = PANEL;
    v.window_fill = WINDOW;
    v.extreme_bg_color = EXTREME;
    v.faint_bg_color = FAINT;
    v.hyperlink_color = ACCENT;

    v.window_corner_radius = CornerRadius::same(8);
    v.menu_corner_radius = CornerRadius::same(6);

    v.selection.bg_fill = ACCENT.linear_multiply(0.45);
    v.selection.stroke = Stroke::new(1.0, ACCENT);

    let radius = CornerRadius::same(5);

    // egui's per-state widget styling. Note `noninteractive` = panels/labels,
    // `inactive` = idle interactive controls (the two are easy to confuse).
    v.widgets.noninteractive.bg_fill = PANEL;
    v.widgets.noninteractive.weak_bg_fill = PANEL;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, OUTLINE);
    v.widgets.noninteractive.corner_radius = radius;

    v.widgets.inactive.bg_fill = WIDGET;
    v.widgets.inactive.weak_bg_fill = WIDGET;
    v.widgets.inactive.bg_stroke = Stroke::NONE;
    v.widgets.inactive.corner_radius = radius;

    v.widgets.hovered.bg_fill = WIDGET_HOVER;
    v.widgets.hovered.weak_bg_fill = WIDGET_HOVER;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT.linear_multiply(0.6));
    v.widgets.hovered.corner_radius = radius;

    v.widgets.active.bg_fill = ACCENT.linear_multiply(0.7);
    v.widgets.active.weak_bg_fill = ACCENT.linear_multiply(0.7);
    v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    v.widgets.active.corner_radius = radius;

    v.widgets.open.bg_fill = WIDGET;
    v.widgets.open.weak_bg_fill = WIDGET;
    v.widgets.open.corner_radius = radius;

    v
}
