//! The editor's whole look, in one place: nine base greys plus an accent, the
//! diagnostic hues, and the values derived from them.
//!
//! A *theme* is exactly those nine colours — everything else in [`Visuals`] is
//! derived from them, which is what keeps a user theme to ten lines of TOML
//! instead of a copy of egui's style tree.
//!
//! The token set is complete by design rather than by use — a `scripting`-less
//! build consumes fewer of them, and a token that exists only so panels stop
//! hand-rolling it is doing its job even when one build configuration doesn't
//! reach it.
#![allow(dead_code)]

use std::path::Path;

use egui::{Color32, CornerRadius, Stroke, Visuals};
use serde::{Deserialize, Serialize};

/// Saturated hue carries meaning here and nowhere else: a build state, a log
/// severity, a profiler lane. These are deliberately *not* part of a [`Theme`] —
/// green means a good build in every theme, and a palette that could recolour
/// them could break what they say.
pub const OK: Color32 = Color32::from_rgb(0x7A, 0xC0, 0x8A);
pub const PENDING: Color32 = Color32::from_rgb(0xE8, 0xB3, 0x4A);
pub const ERROR: Color32 = Color32::from_rgb(0xE0, 0x5A, 0x4A);
pub const LOG_INFO: Color32 = Color32::LIGHT_GRAY;
pub const LOG_WARN: Color32 = Color32::from_rgb(255, 200, 80);
pub const LOG_ERROR: Color32 = Color32::from_rgb(255, 110, 110);
pub const CPU: Color32 = Color32::from_rgb(120, 200, 255);
pub const GPU: Color32 = Color32::from_rgb(255, 170, 80);
pub const GUIDE_60: Color32 = Color32::from_rgb(60, 120, 60);
pub const GUIDE_30: Color32 = Color32::from_rgb(130, 95, 40);

/// Zebra wash for list rows. Far weaker than the faint fill, which at this
/// density reads as two separate lists rather than one striped one. Premultiplied
/// so it darkens whatever surface it lands on, in any theme.
pub const ROW_BAND: Color32 = Color32::from_rgba_premultiplied(5, 5, 5, 5);

pub const BUILT_IN: &str = "orrin-dark";

/// One editor palette. Serialised as the `*.toml` a user drops in
/// `<project>/.orrin/themes/`, so every field is a plain `[r, g, b]` triple.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Theme {
    pub name: String,
    pub text: [u8; 3],
    pub panel: [u8; 3],
    pub window: [u8; 3],
    pub extreme: [u8; 3],
    pub faint: [u8; 3],
    pub widget: [u8; 3],
    pub widget_hover: [u8; 3],
    pub outline: [u8; 3],
    pub accent: [u8; 3],
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            name: BUILT_IN.to_owned(),
            text: [214, 218, 224],
            panel: [24, 25, 28],
            window: [30, 31, 35],
            extreme: [18, 19, 21],
            faint: [36, 38, 42],
            widget: [45, 47, 52],
            widget_hover: [56, 59, 66],
            outline: [48, 50, 56],
            accent: [80, 140, 255],
        }
    }
}

fn rgb([r, g, b]: [u8; 3]) -> Color32 {
    Color32::from_rgb(r, g, b)
}

impl Theme {
    pub fn accent(&self) -> Color32 {
        rgb(self.accent)
    }

    /// Selected row / active tab.
    pub fn select_fill(&self) -> Color32 {
        self.accent().linear_multiply(0.45)
    }

    /// Pressed widget / held mode.
    pub fn widget_active(&self) -> Color32 {
        self.accent().linear_multiply(0.70)
    }

    pub fn hover_stroke(&self) -> Stroke {
        Stroke::new(1.0, self.accent().linear_multiply(0.60))
    }

    pub fn visuals(&self) -> Visuals {
        // Dark as the base even for a light theme: every value this system cares
        // about is overridden below, and the leftovers (shadows, code hues) read
        // better dark than egui's light defaults do against these fills.
        let mut v = Visuals::dark();

        v.override_text_color = Some(rgb(self.text));
        v.panel_fill = rgb(self.panel);
        v.window_fill = rgb(self.window);
        v.extreme_bg_color = rgb(self.extreme);
        v.faint_bg_color = rgb(self.faint);
        v.hyperlink_color = self.accent();

        v.window_corner_radius = CornerRadius::same(8);
        v.menu_corner_radius = CornerRadius::same(6);

        v.selection.bg_fill = self.select_fill();
        v.selection.stroke = Stroke::new(1.0, self.accent());

        let radius = CornerRadius::same(5);

        // egui's per-state widget styling. Note `noninteractive` = panels/labels,
        // `inactive` = idle interactive controls (the two are easy to confuse).
        v.widgets.noninteractive.bg_fill = rgb(self.panel);
        v.widgets.noninteractive.weak_bg_fill = rgb(self.panel);
        v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, rgb(self.outline));
        v.widgets.noninteractive.corner_radius = radius;

        v.widgets.inactive.bg_fill = rgb(self.widget);
        v.widgets.inactive.weak_bg_fill = rgb(self.widget);
        v.widgets.inactive.bg_stroke = Stroke::NONE;
        v.widgets.inactive.corner_radius = radius;

        v.widgets.hovered.bg_fill = rgb(self.widget_hover);
        v.widgets.hovered.weak_bg_fill = rgb(self.widget_hover);
        v.widgets.hovered.bg_stroke = self.hover_stroke();
        v.widgets.hovered.corner_radius = radius;

        v.widgets.active.bg_fill = self.widget_active();
        v.widgets.active.weak_bg_fill = self.widget_active();
        v.widgets.active.bg_stroke = Stroke::new(1.0, self.accent());
        v.widgets.active.corner_radius = radius;

        v.widgets.open.bg_fill = rgb(self.widget);
        v.widgets.open.weak_bg_fill = rgb(self.widget);
        v.widgets.open.corner_radius = radius;

        v
    }
}

/// Every theme this session can switch between: the built-in first, then
/// whatever `<project>/.orrin/themes/` holds.
pub struct ThemeSet {
    themes: Vec<Theme>,
    active: usize,
}

impl Default for ThemeSet {
    fn default() -> Self {
        Self {
            themes: vec![Theme::default()],
            active: 0,
        }
    }
}

impl ThemeSet {
    /// Read `*.toml` themes from `dir`, in filename order so two machines list
    /// them the same way. A theme that fails to parse is named on stderr and
    /// skipped — one bad file must not cost the user the rest of their set.
    pub fn load(dir: &Path) -> Self {
        let mut set = Self::default();

        let Ok(entries) = std::fs::read_dir(dir) else {
            return set;
        };
        let mut paths: Vec<_> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "toml"))
            .collect();
        paths.sort();

        for path in paths {
            match std::fs::read_to_string(&path).map(|text| toml::from_str::<Theme>(&text)) {
                Ok(Ok(theme)) => set.themes.push(theme),
                Ok(Err(e)) => eprintln!("orrin: {} is not a theme: {e}", path.display()),
                Err(e) => eprintln!("orrin: cannot read {}: {e}", path.display()),
            }
        }
        set
    }

    pub fn active(&self) -> &Theme {
        &self.themes[self.active]
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.themes.iter().map(|theme| theme.name.as_str())
    }

    /// Selects by name, returning whether anything changed. A name that no
    /// longer exists (a theme file deleted since it was last chosen) leaves the
    /// current selection alone rather than resetting it.
    pub fn select(&mut self, name: &str) -> bool {
        match self.themes.iter().position(|theme| theme.name == name) {
            Some(index) if index != self.active => {
                self.active = index;
                true
            }
            _ => false,
        }
    }
}

/// Applies `theme` to every style variant, so it sticks regardless of the OS
/// preference.
pub fn apply(ctx: &egui::Context, theme: &Theme) {
    ctx.all_styles_mut(|style| {
        style.visuals = theme.visuals();
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(8.0, 4.0);
        style.spacing.window_margin = egui::Margin::same(8);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOID: &str = r#"
name = "Void"
text = [220, 224, 230]
panel = [16, 16, 18]
window = [22, 22, 25]
extreme = [10, 10, 12]
faint = [28, 28, 32]
widget = [38, 38, 44]
widget_hover = [50, 50, 58]
outline = [44, 44, 50]
accent = [140, 120, 255]
"#;

    #[test]
    fn a_theme_is_ten_lines_of_toml() {
        let theme: Theme = toml::from_str(VOID).expect("parses");
        assert_eq!(theme.name, "Void");
        assert_eq!(theme.accent(), Color32::from_rgb(140, 120, 255));
    }

    /// The accent alphas are the whole state model: a theme that only swaps the
    /// accent must move selection, hover and press with it.
    #[test]
    fn every_accent_alpha_follows_the_theme() {
        let theme: Theme = toml::from_str(VOID).unwrap();
        let visuals = theme.visuals();
        assert_eq!(visuals.selection.bg_fill, theme.select_fill());
        assert_eq!(visuals.widgets.active.bg_fill, theme.widget_active());
        assert_eq!(visuals.widgets.hovered.bg_stroke, theme.hover_stroke());
        assert_ne!(theme.select_fill(), Theme::default().select_fill());
    }

    #[test]
    fn selecting_an_absent_theme_keeps_the_current_one() {
        let mut set = ThemeSet::default();
        assert!(!set.select("Nothing By That Name"));
        assert_eq!(set.active().name, BUILT_IN);
    }

    #[test]
    fn a_missing_themes_directory_still_gives_the_built_in() {
        let set = ThemeSet::load(std::path::Path::new("this/does/not/exist"));
        assert_eq!(set.names().collect::<Vec<_>>(), [BUILT_IN]);
    }
}
