//! The in-game UI's per-frame record: what game code asked to be drawn on
//! screen this tick, in the order it asked.
//!
//! Immediate mode. A script calls `UI.Label`, `UI.Button`, `UI.Panel` and
//! `UI.DrawTexture` from `OnUpdate` every frame it wants them drawn, each call
//! lands here as one [`UiCommand`], and the canvas is emptied at the start of
//! every tick that runs game code. A paused game ticks nothing, so it keeps the
//! last frame's UI on screen rather than losing it; edit mode runs no game code
//! and draws none.
//!
//! The record is deliberately unresolved: a font or a texture is an
//! [`AssetId`], text is a byte range into one string, and nothing is measured.
//! A call happens inside the script dispatch window, where there is no backend
//! to upload a texture to and no reason to lay out text that a later call may
//! cover. `gfx::ui` turns the record into quads between the tick and the frame.
//!
//! Coordinates are physical pixels from the window's top left, the space the
//! cursor is reported in, so a button can be hit-tested against the cursor with
//! no conversion on either side. Colours are sRGB-encoded with straight alpha,
//! the numbers a colour picker shows.

use std::ops::Range;

use orrin_registry::AssetId;

/// An axis-aligned rectangle in physical pixels, top-left origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn max(&self) -> [f32; 2] {
        [self.x + self.width, self.y + self.height]
    }

    /// False for a rectangle with no area, which nothing can be drawn in.
    pub fn is_visible(&self) -> bool {
        self.width > 0.0 && self.height > 0.0
    }
}

/// Where text sits inside its rectangle. Lock-step with C# `Orrin.TextAnchor`:
/// row-major from the upper left, so `self as u32 % 3` is the column and
/// `/ 3` the row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, orrin_registry::Reflect)]
#[repr(u32)]
pub enum TextAnchor {
    #[default]
    UpperLeft,
    UpperCenter,
    UpperRight,
    MiddleLeft,
    MiddleCenter,
    MiddleRight,
    LowerLeft,
    LowerCenter,
    LowerRight,
}

impl TextAnchor {
    const ALL: [Self; 9] = [
        Self::UpperLeft,
        Self::UpperCenter,
        Self::UpperRight,
        Self::MiddleLeft,
        Self::MiddleCenter,
        Self::MiddleRight,
        Self::LowerLeft,
        Self::LowerCenter,
        Self::LowerRight,
    ];

    /// The anchor C# numbered `value`, or `None` past the last.
    pub fn from_u32(value: u32) -> Option<Self> {
        Self::ALL.get(value as usize).copied()
    }

    /// 0 for left, 0.5 for centre, 1 for right: how much of the slack across
    /// the rectangle goes before the line.
    pub fn horizontal(self) -> f32 {
        (self as u32 % 3) as f32 * 0.5
    }

    /// The same down the rectangle, for the block of lines.
    pub fn vertical(self) -> f32 {
        (self as u32 / 3) as f32 * 0.5
    }
}

/// How a run of text is set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextStyle {
    /// `None` is the engine's built-in font.
    pub font: Option<AssetId>,
    /// The font's ascent-to-descent height, in physical pixels.
    pub size: f32,
    pub anchor: TextAnchor,
    pub color: [u8; 4],
    /// Off, each paragraph sets on one line, running past the rectangle's
    /// sides rather than breaking; it is still cut at the top and bottom.
    pub wrap: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UiCommand {
    Panel {
        rect: Rect,
        color: [u8; 4],
    },
    /// Wrapped to the rectangle's width and clipped to the rectangle.
    Text {
        rect: Rect,
        style: TextStyle,
        /// Into [`UiCanvas::text`].
        text: Range<u32>,
    },
    /// `uv`'s part of the texture stretched over the rectangle and multiplied
    /// by `tint`.
    Image {
        rect: Rect,
        texture: AssetId,
        /// Left, top, right, bottom, normalised.
        uv: [f32; 4],
        tint: [u8; 4],
    },
}

/// This frame's UI, in draw order. A world resource.
#[derive(Default)]
pub struct UiCanvas {
    screen: [f32; 2],
    commands: Vec<UiCommand>,
    /// Every text command's bytes, end to end, so a frame of labels is one
    /// allocation that is kept, not one per label.
    text: String,
}

impl UiCanvas {
    /// Empty the canvas for a tick about to run game code, which will see
    /// `screen` as the size of the window.
    pub fn begin(&mut self, screen: [u32; 2]) {
        self.screen = screen.map(|side| side as f32);
        self.commands.clear();
        self.text.clear();
    }

    /// Drop whatever was drawn, keeping the screen size.
    pub fn clear(&mut self) {
        self.commands.clear();
        self.text.clear();
    }

    pub fn screen(&self) -> [f32; 2] {
        self.screen
    }

    pub fn panel(&mut self, rect: Rect, color: [u8; 4]) {
        if rect.is_visible() && color[3] > 0 {
            self.commands.push(UiCommand::Panel { rect, color });
        }
    }

    pub fn text(&mut self, rect: Rect, style: TextStyle, text: &str) {
        if !rect.is_visible() || style.color[3] == 0 || text.is_empty() || style.size <= 0.0 {
            return;
        }
        let start = self.text.len() as u32;
        self.text.push_str(text);
        self.commands.push(UiCommand::Text {
            rect,
            style,
            text: start..self.text.len() as u32,
        });
    }

    pub fn image(&mut self, rect: Rect, texture: AssetId, tint: [u8; 4]) {
        self.image_region(rect, texture, [0.0, 0.0, 1.0, 1.0], tint);
    }

    /// The `uv` part of `texture` over `rect`: one piece of a sliced image.
    pub fn image_region(&mut self, rect: Rect, texture: AssetId, uv: [f32; 4], tint: [u8; 4]) {
        if rect.is_visible() && tint[3] > 0 {
            self.commands.push(UiCommand::Image {
                rect,
                texture,
                uv,
                tint,
            });
        }
    }

    pub fn commands(&self) -> &[UiCommand] {
        &self.commands
    }

    /// The text a [`UiCommand::Text`] names.
    pub fn str(&self, range: &Range<u32>) -> &str {
        &self.text[range.start as usize..range.end as usize]
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

/// Where the game's screen is in the window, and the size the game is told it
/// has. The whole window, except while the editor previews a resolution (see
/// `editor::preview`): then the game is laid out at that size and drawn,
/// scaled, inside the viewport. A world resource; absent means the window.
///
/// The UI is the only thing placed by it. The 3D scene still draws across the
/// window, because its camera's aspect is the swapchain's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GameScreen {
    /// The screen as the game sees it, in its own pixels.
    pub size: [f32; 2],
    /// Its top left, in window pixels.
    pub origin: [f32; 2],
    /// Window pixels per game pixel.
    pub scale: f32,
    /// What may be drawn on, in window pixels: the screen, less whatever of it
    /// is out of view.
    pub clip: Rect,
}

impl GameScreen {
    /// The whole window, one to one.
    pub fn window(extent: [u32; 2]) -> Self {
        let size = extent.map(|side| side as f32);
        Self {
            size,
            origin: [0.0; 2],
            scale: 1.0,
            clip: Rect::new(0.0, 0.0, size[0], size[1]),
        }
    }

    /// The screen's rectangle in the window.
    pub fn rect(&self) -> Rect {
        let [x, y] = self.origin;
        Rect::new(x, y, self.size[0] * self.scale, self.size[1] * self.scale)
    }

    /// A point in the window, in the game's pixels.
    pub fn to_game(&self, [x, y]: [f32; 2]) -> [f32; 2] {
        [
            (x - self.origin[0]) / self.scale,
            (y - self.origin[1]) / self.scale,
        ]
    }

    /// A rectangle in the game's pixels, in the window's.
    pub fn to_window(&self, rect: Rect) -> Rect {
        Rect::new(
            self.origin[0] + rect.x * self.scale,
            self.origin[1] + rect.y * self.scale,
            rect.width * self.scale,
            rect.height * self.scale,
        )
    }

    /// Whether this is the window itself, so nothing needs mapping.
    pub fn is_window(&self, extent: [u32; 2]) -> bool {
        *self == Self::window(extent)
    }
}

/// A C# `Color` as the canvas stores it: each channel clamped and rounded to a
/// byte. A channel above one means something to a light, not to a pixel.
pub fn color_bytes(rgba: [f32; 4]) -> [u8; 4] {
    rgba.map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn style() -> TextStyle {
        TextStyle {
            font: None,
            size: 16.0,
            anchor: TextAnchor::UpperLeft,
            color: [255; 4],
            wrap: true,
        }
    }

    #[test]
    fn commands_keep_the_order_they_were_issued_in() {
        let mut canvas = UiCanvas::default();
        canvas.begin([800, 600]);
        let rect = Rect::new(10.0, 10.0, 100.0, 20.0);
        canvas.panel(rect, [0, 0, 0, 128]);
        canvas.text(rect, style(), "score");
        canvas.text(rect, style(), "lives");
        let texts: Vec<_> = canvas
            .commands()
            .iter()
            .filter_map(|command| match command {
                UiCommand::Text { text, .. } => Some(canvas.str(text)),
                _ => None,
            })
            .collect();
        assert!(matches!(canvas.commands()[0], UiCommand::Panel { .. }));
        assert_eq!(texts, ["score", "lives"]);
        assert_eq!(canvas.screen(), [800.0, 600.0]);
    }

    #[test]
    fn beginning_a_tick_forgets_the_last_one_and_keeps_its_buffers() {
        let mut canvas = UiCanvas::default();
        canvas.begin([800, 600]);
        canvas.text(Rect::new(0.0, 0.0, 50.0, 50.0), style(), "a long label");
        let capacity = canvas.text.capacity();
        canvas.begin([1024, 768]);
        assert!(canvas.is_empty());
        assert_eq!(canvas.text.capacity(), capacity);
        assert_eq!(canvas.screen(), [1024.0, 768.0]);
    }

    /// Nothing that would draw no pixels reaches the painter.
    #[test]
    fn invisible_calls_record_nothing() {
        let mut canvas = UiCanvas::default();
        let rect = Rect::new(0.0, 0.0, 10.0, 10.0);
        canvas.panel(Rect::new(0.0, 0.0, 0.0, 10.0), [255; 4]);
        canvas.panel(rect, [255, 255, 255, 0]);
        canvas.text(rect, style(), "");
        canvas.text(
            rect,
            TextStyle {
                size: 0.0,
                ..style()
            },
            "x",
        );
        canvas.image(Rect::new(0.0, 0.0, 10.0, -1.0), AssetId::new(), [255; 4]);
        assert!(canvas.is_empty(), "{:?}", canvas.commands());
    }

    #[test]
    fn anchors_split_into_a_column_and_a_row() {
        assert_eq!(TextAnchor::from_u32(4), Some(TextAnchor::MiddleCenter));
        assert_eq!(TextAnchor::from_u32(9), None);
        assert_eq!(TextAnchor::UpperRight.horizontal(), 1.0);
        assert_eq!(TextAnchor::UpperRight.vertical(), 0.0);
        assert_eq!(TextAnchor::LowerCenter.horizontal(), 0.5);
        assert_eq!(TextAnchor::LowerCenter.vertical(), 1.0);
    }

    #[test]
    fn the_window_screen_maps_nothing() {
        let screen = GameScreen::window([1280, 720]);
        assert_eq!(screen.size, [1280.0, 720.0]);
        assert_eq!(screen.rect(), Rect::new(0.0, 0.0, 1280.0, 720.0));
        assert_eq!(screen.clip, screen.rect());
        assert_eq!(screen.to_game([10.0, 20.0]), [10.0, 20.0]);
        assert!(screen.is_window([1280, 720]));
        assert!(!screen.is_window([1920, 1080]));
    }

    #[test]
    fn a_placed_screen_maps_between_its_pixels_and_the_windows() {
        let screen = GameScreen {
            size: [1920.0, 1080.0],
            origin: [100.0, 50.0],
            scale: 0.5,
            clip: Rect::new(100.0, 50.0, 960.0, 540.0),
        };
        assert_eq!(screen.rect(), Rect::new(100.0, 50.0, 960.0, 540.0));
        assert_eq!(screen.to_game([100.0, 50.0]), [0.0, 0.0]);
        assert_eq!(screen.to_game([1060.0, 590.0]), [1920.0, 1080.0]);
        assert_eq!(
            screen.to_window(Rect::new(20.0, 40.0, 200.0, 100.0)),
            Rect::new(110.0, 70.0, 100.0, 50.0)
        );
    }

    #[test]
    fn colours_clamp_to_a_byte_per_channel() {
        assert_eq!(color_bytes([1.0, 0.5, -1.0, 4.0]), [255, 128, 0, 255]);
    }
}
