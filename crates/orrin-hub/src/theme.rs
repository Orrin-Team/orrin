//! The hub's look: a neutral dark launcher in the shape people know from other
//! engines' hubs — a near-black frame, two raised panels, a table of projects,
//! and one blue button for the thing to do.
//!
//! Neutral greys rather than a tinted palette, because a launcher is looked
//! past on the way to a project; colour is kept for state — the accent for the
//! primary action and selection, green and red for an outcome, amber for a
//! star.

use egui::{Color32, CornerRadius, FontFamily, FontId, Stroke, TextStyle, Visuals};

/// Behind everything: the strip the title sits on and the gaps between panels.
pub const CHROME: Color32 = Color32::from_rgb(0x13, 0x13, 0x13);
/// The sidebar and the projects panel.
pub const PANEL: Color32 = Color32::from_rgb(0x18, 0x18, 0x18);
/// The table's header strip.
pub const HEADER: Color32 = Color32::from_rgb(0x22, 0x22, 0x22);
/// A row under the pointer.
pub const HOVER: Color32 = Color32::from_rgb(0x1f, 0x1f, 0x1f);
/// Text fields, and the selected sidebar entry.
pub const FIELD: Color32 = Color32::from_rgb(0x29, 0x29, 0x29);
pub const SELECTED: Color32 = Color32::from_rgb(0x31, 0x31, 0x31);
pub const OUTLINE: Color32 = Color32::from_rgb(0x2e, 0x2e, 0x2e);
/// The hairline between rows.
pub const RULE: Color32 = Color32::from_rgb(0x24, 0x24, 0x24);
pub const INK: Color32 = Color32::from_rgb(0xe6, 0xe6, 0xe6);
pub const INK_2: Color32 = Color32::from_rgb(0xc4, 0xc4, 0xc4);
pub const INK_3: Color32 = Color32::from_rgb(0x9b, 0x9b, 0x9b);
pub const INK_4: Color32 = Color32::from_rgb(0x74, 0x74, 0x74);
pub const ACCENT: Color32 = Color32::from_rgb(0x0d, 0x71, 0xc6);
pub const ACCENT_HOVER: Color32 = Color32::from_rgb(0x24, 0x84, 0xd8);
pub const ACCENT_INK: Color32 = Color32::WHITE;
pub const STAR: Color32 = Color32::from_rgb(0xf2, 0xb1, 0x34);
pub const OK: Color32 = Color32::from_rgb(0x5f, 0xb5, 0x5a);
pub const ERROR: Color32 = Color32::from_rgb(0xe5, 0x5b, 0x52);

pub const RADIUS_SM: u8 = 4;
pub const RADIUS_MD: u8 = 8;

pub const MONO: &str = "JetBrainsMono";
pub const ICONS: &str = "icons";

pub fn install(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        MONO.to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(include_bytes!(
            "../../core/assets/fonts/JetBrainsMono-Regular.ttf"
        ))),
    );
    fonts
        .families
        .entry(FontFamily::Monospace)
        .or_default()
        .insert(0, MONO.to_owned());
    fonts.font_data.insert(
        ICONS.to_owned(),
        std::sync::Arc::new(
            egui::FontData::from_static(include_bytes!("../../core/assets/fonts/Phosphor.ttf"))
                .tweak(egui::FontTweak {
                    y_offset_factor: 0.08,
                    ..Default::default()
                }),
        ),
    );
    fonts
        .families
        .insert(FontFamily::Name(ICONS.into()), vec![ICONS.to_owned()]);
    // Last in the body family, so a glyph written inline with a label — "＋ New
    // project" — is found there instead of drawing as a missing-glyph box.
    fonts
        .families
        .entry(FontFamily::Proportional)
        .or_default()
        .push(ICONS.to_owned());
    ctx.set_fonts(fonts);
    egui_extras::install_image_loaders(ctx);

    ctx.style_mut(|style| {
        style.visuals = visuals();
        style.spacing.item_spacing = egui::vec2(8.0, 8.0);
        style.spacing.button_padding = egui::vec2(12.0, 6.0);
        style.spacing.interact_size.y = 30.0;
        // A launcher has no text worth selecting, and a selectable label
        // senses the clicks and drags meant for the row it sits on.
        style.interaction.selectable_labels = false;
        style.text_styles = [
            (TextStyle::Small, FontId::proportional(12.0)),
            (TextStyle::Body, FontId::proportional(14.0)),
            (TextStyle::Button, FontId::proportional(14.0)),
            (TextStyle::Heading, FontId::proportional(24.0)),
            (TextStyle::Monospace, FontId::monospace(12.5)),
        ]
        .into();
    });
}

fn visuals() -> Visuals {
    let mut v = Visuals::dark();
    v.override_text_color = Some(INK);
    v.panel_fill = CHROME;
    v.window_fill = PANEL;
    v.window_stroke = Stroke::new(1.0_f32, OUTLINE);
    v.window_corner_radius = CornerRadius::same(RADIUS_MD);
    v.menu_corner_radius = CornerRadius::same(RADIUS_SM + 2);
    v.extreme_bg_color = FIELD;
    v.faint_bg_color = HOVER;
    v.hyperlink_color = ACCENT_HOVER;
    v.selection.bg_fill = ACCENT.gamma_multiply(0.5);
    v.selection.stroke = Stroke::new(1.0_f32, ACCENT_HOVER);

    let radius = CornerRadius::same(RADIUS_SM);
    let w = &mut v.widgets;
    w.noninteractive.bg_fill = PANEL;
    w.noninteractive.weak_bg_fill = PANEL;
    w.noninteractive.bg_stroke = Stroke::new(1.0_f32, OUTLINE);
    w.noninteractive.fg_stroke = Stroke::new(1.0_f32, INK);
    w.noninteractive.corner_radius = radius;
    for (state, fill, stroke) in [
        (&mut w.inactive, FIELD, OUTLINE),
        (&mut w.hovered, SELECTED, Color32::from_gray(0x44)),
        (&mut w.active, SELECTED, Color32::from_gray(0x55)),
        (&mut w.open, SELECTED, OUTLINE),
    ] {
        state.bg_fill = fill;
        state.weak_bg_fill = fill;
        state.bg_stroke = Stroke::new(1.0_f32, stroke);
        state.fg_stroke = Stroke::new(1.0_f32, INK);
        state.corner_radius = radius;
        state.expansion = 0.0;
    }
    v
}

/// A glyph from the Phosphor set the editor also uses.
pub fn icon(glyph: char, size: f32) -> egui::RichText {
    egui::RichText::new(glyph.to_string()).font(FontId::new(size, FontFamily::Name(ICONS.into())))
}

/// Phosphor 2.1.1 codepoints, from its stylesheet.
pub mod glyph {
    pub const FOLDER: char = '\u{e25a}';
    pub const PLUS: char = '\u{e3d4}';
    pub const PLAY: char = '\u{e3d0}';
    pub const ELLIPSIS: char = '\u{e1fe}';
    pub const BOX: char = '\u{e1da}';
    pub const CHECK_CIRCLE: char = '\u{e184}';
    pub const X_CIRCLE: char = '\u{e4f8}';
    pub const WARNING: char = '\u{e4e0}';
    pub const SEARCH: char = '\u{e30c}';
    pub const GLOBE: char = '\u{e288}';
    pub const TERMINAL: char = '\u{eae8}';
    pub const LIGHTBULB: char = '\u{e2dc}';
    pub const X: char = '\u{e4f6}';
    pub const ARROW_OUT: char = '\u{e5de}';
    pub const STAR: char = '\u{e46a}';
    pub const GIT_BRANCH: char = '\u{e278}';
    pub const GITHUB: char = '\u{e576}';
    pub const ARROW_UP: char = '\u{e08e}';
    pub const ARROW_DOWN: char = '\u{e03e}';
    pub const CARET_DOWN: char = '\u{e136}';
    pub const GRADUATION_CAP: char = '\u{e62c}';
}

/// The one button on a screen that is the thing to do.
pub fn primary(text: impl Into<String>) -> egui::Button<'static> {
    egui::Button::new(egui::RichText::new(text.into()).color(ACCENT_INK).strong())
        .fill(ACCENT)
        .stroke(Stroke::NONE)
        .corner_radius(CornerRadius::same(RADIUS_SM))
}

/// Everything else: an outline until hovered.
pub fn secondary(text: impl Into<egui::WidgetText>) -> egui::Button<'static> {
    egui::Button::new(text)
        .fill(Color32::TRANSPARENT)
        .stroke(Stroke::new(1.0_f32, OUTLINE))
        .corner_radius(CornerRadius::same(RADIUS_SM))
}

/// A raised panel on the chrome: the sidebar, the projects, a Learn card.
pub fn panel() -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL)
        .stroke(Stroke::new(1.0_f32, OUTLINE))
        .corner_radius(CornerRadius::same(RADIUS_MD))
}
