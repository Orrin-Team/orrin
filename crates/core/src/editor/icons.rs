//! The editor's icon set: Phosphor (regular weight), MIT licensed, vendored as
//! a font under `assets/fonts/`.
//!
//! A font rather than a folder of SVGs, for sharpness. An SVG rasterised to a
//! 14px texture is only crisp if it lands exactly on the pixel grid at exactly
//! the size it was rasterised for, and a 1.5px stroke at that size is an
//! antialiased smear either way. A glyph goes through egui's font rasteriser,
//! which renders at the display's real pixel density and positions on whole
//! pixels — the same path that keeps the text beside it sharp. It also costs
//! one font atlas instead of a texture per icon per size.
//!
//! The font is embedded whole: 1 530 glyphs, 0.5 MB, of which the editor uses
//! a few dozen. Subsetting it would need a font tool in the build; the cost is
//! binary size only, since egui parses a font's tables lazily.
//!
//! The one image that is not a glyph is the Orrin mark, which keeps its own
//! colours and so stays an SVG.

use egui::{Color32, FontFamily, FontId, ImageSource, RichText};

/// In bars and rows.
pub const ROW: f32 = 16.0;

/// The font family the glyphs are set in.
pub fn family() -> FontFamily {
    FontFamily::Name(FAMILY.into())
}

const FAMILY: &str = "icons";
const FONT: &str = "Phosphor";

/// One glyph of the set.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Icon(char);

impl Icon {
    #[cfg(test)]
    pub fn glyph(self) -> char {
        self.0
    }
}

/// Add the icon font and its family to `fonts`. Called by the theme when it
/// installs the editor's fonts, so nothing can draw an icon without it.
pub fn add_font(fonts: &mut egui::FontDefinitions) {
    fonts.font_data.insert(
        FONT.to_owned(),
        std::sync::Arc::new(
            egui::FontData::from_static(include_bytes!("../../assets/fonts/Phosphor.ttf")).tweak(
                egui::FontTweak {
                    // The glyphs sit on the em square's baseline; nudged down so
                    // an icon centres on a line of body text rather than riding
                    // above it.
                    y_offset_factor: 0.08,
                    ..Default::default()
                },
            ),
        ),
    );
    fonts.families.insert(family(), vec![FONT.to_owned()]);
}

/// The image loaders, for the one SVG the editor still draws: the logo.
pub fn install(ctx: &egui::Context) {
    egui_extras::install_image_loaders(ctx);
}

macro_rules! icons {
    ($($name:ident => $code:literal,)*) => {
        $(
            #[allow(dead_code, reason = "the icon set is a palette, not a use list")]
            pub fn $name() -> Icon {
                Icon($code)
            }
        )*

        /// Every icon the editor names, for the tests that check each is in
        /// the font.
        #[cfg(test)]
        const ALL: &[(&str, char)] = &[$((stringify!($name), $code),)*];
    };
}

// Phosphor's codepoints, from its `style.css` at 2.1.1.
icons! {
    activity => '\u{e000}',
    arrow_up => '\u{e08e}',
    box_ => '\u{e1da}',
    check => '\u{e182}',
    chevron_down => '\u{e136}',
    chevron_right => '\u{e13a}',
    circle => '\u{e18a}',
    circle_dot => '\u{ece0}',
    circle_x => '\u{e4f8}',
    ellipsis => '\u{e1fe}',
    file_code => '\u{e914}',
    folder => '\u{e25a}',
    globe => '\u{e288}',
    grid => '\u{e296}',
    hourglass => '\u{e2b2}',
    image => '\u{e2ca}',
    info => '\u{e2ce}',
    layers => '\u{e466}',
    lightbulb => '\u{e2dc}',
    list_tree => '\u{ee48}',
    mountain_snow => '\u{e7ae}',
    move_ => '\u{e0a4}',
    palette => '\u{e6c8}',
    pause => '\u{e39e}',
    play => '\u{e3d0}',
    plus => '\u{e3d4}',
    pointer => '\u{e1dc}',
    refresh => '\u{e094}',
    rotate => '\u{e036}',
    rotate_ccw => '\u{e038}',
    scaling => '\u{e0a2}',
    search => '\u{e30c}',
    sliders => '\u{e434}',
    step_forward => '\u{e5a6}',
    sun => '\u{e472}',
    terminal => '\u{eae8}',
    trash => '\u{e4a6}',
    warning => '\u{e4e0}',
    x => '\u{e4f6}',
}

/// The Orrin mark. Drawn in its own colours and never tinted — it is the one
/// image here that is not a glyph.
pub fn logo() -> ImageSource<'static> {
    egui::include_image!("../../assets/logo.svg")
}

/// An icon as text, for a label or a button.
pub fn text(icon: Icon, size: f32, color: Color32) -> RichText {
    RichText::new(icon.0.to_string())
        .family(family())
        .size(size)
        .color(color)
}

/// An icon and a word in one piece of text — a menu entry, a labelled button.
pub fn with_label(icon: Icon, color: Option<Color32>, label: &str) -> egui::WidgetText {
    let mut job = egui::text::LayoutJob::default();
    job.append(
        &icon.0.to_string(),
        0.0,
        egui::TextFormat::simple(
            FontId::new(ROW - 2.0, family()),
            color.unwrap_or(Color32::PLACEHOLDER),
        ),
    );
    job.append(
        label,
        7.0,
        egui::TextFormat::simple(FontId::proportional(12.5), Color32::PLACEHOLDER),
    );
    job.into()
}

/// Paint an icon centred in `rect`, as tall as the rect.
pub fn paint(ui: &egui::Ui, icon: Icon, rect: egui::Rect, color: Color32) {
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        icon.0,
        FontId::new(rect.height(), family()),
        color,
    );
}

/// An icon-only button.
pub fn button(
    ui: &mut egui::Ui,
    icon: Icon,
    size: f32,
    color: Color32,
    frame: bool,
) -> egui::Response {
    ui.add(egui::Button::new(text(icon, size, color)).frame(frame))
}

/// A row-height icon followed by its label, in one colour. Used where a glyph
/// used to lead a string — `"⚠ faulted"` and the like.
pub fn labelled(ui: &mut egui::Ui, icon: Icon, color: Color32, label: &str) {
    ui.horizontal(|ui| {
        ui.label(text(icon, ROW - 2.0, color));
        ui.colored_label(color, label);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> egui::Context {
        let ctx = egui::Context::default();
        let mut fonts = egui::FontDefinitions::default();
        add_font(&mut fonts);
        ctx.set_fonts(fonts);
        let _ = ctx.run(Default::default(), |_| {});
        ctx
    }

    /// Every icon the editor names has to be a glyph the font actually has. A
    /// codepoint it lacks falls back to egui's replacement box, which looks
    /// like a rendering bug rather than a typo in a table.
    #[test]
    fn every_icon_is_in_the_font() {
        let ctx = context();
        let font = FontId::new(ROW, family());
        for (name, code) in ALL {
            assert!(
                ctx.fonts(|fonts| fonts.has_glyph(&font, *code)),
                "{name} (U+{:04X}) is not in the icon font",
                *code as u32
            );
        }
    }

    /// No two names for one glyph: a table edited by hand drifts that way, and
    /// two commands drawn alike read as one.
    #[test]
    fn no_two_icons_share_a_glyph() {
        for (index, (name, code)) in ALL.iter().enumerate() {
            for (other, other_code) in &ALL[index + 1..] {
                assert_ne!(code, other_code, "{name} and {other}");
            }
        }
    }

    /// The handoff's warning about icons: they are exactly the kind of addition
    /// that quietly costs a fifth of a second at startup. Laying out every
    /// glyph, which is what rasterises them into the atlas, stays far inside a
    /// frame.
    #[test]
    fn the_whole_set_lays_out_in_well_under_a_frame_budget() {
        let ctx = context();
        let start = std::time::Instant::now();
        let text: String = ALL.iter().map(|(_, code)| *code).collect();
        ctx.fonts(|fonts| fonts.layout_no_wrap(text, FontId::new(ROW, family()), Color32::WHITE));
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(250),
            "{} icons took {elapsed:?}",
            ALL.len()
        );
    }
}
