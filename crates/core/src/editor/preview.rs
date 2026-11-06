//! A resolution to see the game's UI at while editing it: a template or a
//! size typed in, shown inside the viewport either scaled to fit it or at one
//! game pixel to one screen pixel.
//!
//! What is previewed is the game's *screen*, not a second render. The editor
//! sets the [`GameScreen`] resource from it every frame it draws, and
//! everything that asks the game how big its screen is reads that: the canvas
//! scaler, the layout, `UI.ScreenSize`, and the cursor a script sees. So a
//! canvas that scales with the screen, a HUD anchored to a corner and an
//! immediate-mode button all behave at 1080×1920 exactly as they will on a
//! phone, and the rect tool and the pointer still hit what is drawn, because
//! the layout lands in the window already placed.
//!
//! The 3D scene is not reframed: its camera's aspect is the swapchain's, and
//! drawing it into a rectangle of another aspect is a different renderer
//! feature. Outside the previewed screen the viewport is dimmed, which is where
//! the game would have no pixels.

use serde::{Deserialize, Serialize};

use crate::scene::{GameScreen, Rect};

/// Whether the previewed screen is scaled to the viewport or shown at its
/// size in pixels, cut where the viewport ends.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreviewSize {
    #[default]
    Fit,
    TrueSize,
}

/// What the viewport previews. Remembered with the editor's preferences.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Preview {
    /// `None` is the window as it is, which is what a shipped game gets.
    pub resolution: Option<[u32; 2]>,
    pub size: PreviewSize,
}

/// The resolutions offered without typing one, most common first in each
/// group.
pub const TEMPLATES: &[(&str, [u32; 2])] = &[
    ("HD", [1280, 720]),
    ("Full HD", [1920, 1080]),
    ("QHD", [2560, 1440]),
    ("4K UHD", [3840, 2160]),
    ("Ultrawide 21:9", [2560, 1080]),
    ("Handheld 16:10", [1280, 800]),
    ("4:3", [1024, 768]),
    ("Tablet 4:3", [2048, 1536]),
    ("Phone landscape", [2340, 1080]),
    ("Phone portrait", [1080, 2340]),
];

/// The largest side a typed resolution may have: an 8K screen.
pub const MAX_SIDE: u32 = 7680;

/// The game's screen for `preview`, inside `viewport` in a window of
/// `extent`, both in physical pixels.
pub fn screen(preview: &Preview, viewport: Rect, extent: [u32; 2]) -> GameScreen {
    let Some(resolution) = preview.resolution.filter(|r| r[0] > 0 && r[1] > 0) else {
        return GameScreen::window(extent);
    };
    let size = resolution.map(|side| side as f32);
    let scale = match preview.size {
        PreviewSize::Fit => (viewport.width / size[0]).min(viewport.height / size[1]),
        PreviewSize::TrueSize => 1.0,
    };
    let drawn = [size[0] * scale, size[1] * scale];
    let origin = [
        (viewport.x + (viewport.width - drawn[0]) / 2.0).round(),
        (viewport.y + (viewport.height - drawn[1]) / 2.0).round(),
    ];
    let screen = GameScreen {
        size,
        origin,
        scale,
        clip: Rect::new(origin[0], origin[1], drawn[0], drawn[1]),
    };
    GameScreen {
        clip: intersect(screen.clip, viewport),
        ..screen
    }
}

fn intersect(a: Rect, b: Rect) -> Rect {
    let [ax, ay] = a.max();
    let [bx, by] = b.max();
    let (x, y) = (a.x.max(b.x), a.y.max(b.y));
    Rect::new(x, y, (ax.min(bx) - x).max(0.0), (ay.min(by) - y).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: [u32; 2] = [2000, 1200];

    fn viewport() -> Rect {
        Rect::new(400.0, 100.0, 1200.0, 900.0)
    }

    fn preview(resolution: [u32; 2], size: PreviewSize) -> Preview {
        Preview {
            resolution: Some(resolution),
            size,
        }
    }

    #[test]
    fn no_resolution_is_the_window_itself() {
        let screen = screen(&Preview::default(), viewport(), WINDOW);
        assert!(screen.is_window(WINDOW));
    }

    /// A wide screen in a squarer viewport is as wide as the viewport and
    /// centred down it; a tall one is as tall and centred across.
    #[test]
    fn fit_scales_to_the_viewport_and_centres_in_it() {
        let wide = screen(&preview([1920, 1080], PreviewSize::Fit), viewport(), WINDOW);
        assert_eq!(wide.size, [1920.0, 1080.0]);
        assert!((wide.scale - 1200.0 / 1920.0).abs() < 1e-6);
        assert_eq!(wide.rect().width, 1200.0);
        let gap = 900.0 - wide.rect().height;
        assert!((wide.rect().y - (100.0 + gap / 2.0)).abs() <= 0.5);
        assert_eq!(wide.clip, wide.rect());

        let tall = screen(&preview([1080, 2340], PreviewSize::Fit), viewport(), WINDOW);
        assert!((tall.rect().height - 900.0).abs() < 1e-3);
        let gap = 1200.0 - tall.rect().width;
        assert!((tall.rect().x - (400.0 + gap / 2.0)).abs() <= 0.5);
    }

    /// One game pixel to one screen pixel, centred; what does not fit in the
    /// viewport is cut there rather than drawn under the panels.
    #[test]
    fn true_size_is_one_to_one_and_cut_to_the_viewport() {
        let small = screen(
            &preview([1024, 768], PreviewSize::TrueSize),
            viewport(),
            WINDOW,
        );
        assert_eq!(small.scale, 1.0);
        assert_eq!(small.rect(), Rect::new(488.0, 166.0, 1024.0, 768.0));
        assert_eq!(small.clip, small.rect());

        let large = screen(
            &preview([3840, 2160], PreviewSize::TrueSize),
            viewport(),
            WINDOW,
        );
        assert_eq!(large.scale, 1.0);
        assert_eq!(large.size, [3840.0, 2160.0]);
        assert_eq!(large.clip, viewport());
    }

    /// The corner lands on a whole pixel, so text drawn one to one stays sharp.
    #[test]
    fn the_screen_starts_on_a_whole_pixel() {
        let odd = Rect::new(400.5, 100.25, 1201.0, 901.0);
        for size in [PreviewSize::Fit, PreviewSize::TrueSize] {
            let screen = screen(&preview([1366, 768], size), odd, WINDOW);
            assert!(screen.origin.iter().all(|v| v.fract() == 0.0), "{screen:?}");
        }
    }
}
