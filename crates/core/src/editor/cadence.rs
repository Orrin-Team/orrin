//! How often the editor's UI is rebuilt, which is not how often the scene is
//! drawn.
//!
//! The renderer runs as fast as the window lets it — over a thousand frames a
//! second on a desktop GPU with vsync off — and laying the dock out is a
//! quarter of a millisecond of CPU before tessellation (`editor_frame_cost`),
//! so rebuilding it every frame spends a large share of the frame redrawing
//! text that has not changed. A frame that is not due hands the renderer the
//! last one's tessellated output again, which it can draw twice because it
//! drained the texture deltas the first time.
//!
//! A rebuild is due whenever something could have changed what the UI shows or
//! how it responds: input reached the editor, egui asked for a repaint (an
//! animation, a tooltip's delay) and its time has come, or the caller says the
//! scene camera moved — the gizmo and the selection outline are projected
//! through it. Otherwise it is due at most [`INTERVAL`] after the last, so a
//! panel showing live values — the profiler, an inspector on a moving entity,
//! the console — still updates at a display's rate rather than freezing until
//! the mouse next moves. What that costs is a selected object moving under its
//! outline lagging it by up to one interval.

use std::time::{Duration, Instant};

/// The longest a live panel goes without being redrawn: one frame of a 60 Hz
/// display.
pub const INTERVAL: Duration = Duration::from_micros(16_667);

/// How early a frame may land and still count as the interval having passed.
/// Two frames of a 120 Hz display span one interval to within timing jitter,
/// so without this the second of them is a few microseconds short half the
/// time and the rebuild slips to every third frame — 40 Hz, not 60.
const SLACK: Duration = Duration::from_millis(1);

#[derive(Debug, Default)]
pub struct Cadence {
    last: Option<Instant>,
    repaint_at: Option<Instant>,
    touched: bool,
}

impl Cadence {
    /// Something happened that the UI has to see: input, or a camera move.
    pub fn touch(&mut self) {
        self.touched = true;
    }

    pub fn due(&self, now: Instant) -> bool {
        let Some(last) = self.last else {
            return true;
        };
        self.touched
            || now.saturating_duration_since(last) + SLACK >= INTERVAL
            || self.repaint_at.is_some_and(|at| now >= at)
    }

    /// The UI was rebuilt at `now`, and egui asked to be repainted again after
    /// `repaint_delay` — `Duration::MAX` when it has nothing pending.
    pub fn ran(&mut self, now: Instant, repaint_delay: Duration) {
        self.touched = false;
        self.last = Some(now);
        self.repaint_at = now.checked_add(repaint_delay);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn after(start: Instant, cadence: &mut Cadence, repaint: Duration) -> Instant {
        cadence.ran(start, repaint);
        start
    }

    #[test]
    fn the_first_frame_is_due() {
        assert!(Cadence::default().due(Instant::now()));
    }

    /// The whole point: a thousand frames a second, and the UI rebuilt for
    /// only one of every sixteen or so.
    #[test]
    fn an_idle_ui_is_not_rebuilt_before_the_interval() {
        let mut cadence = Cadence::default();
        let start = after(Instant::now(), &mut cadence, Duration::MAX);
        assert!(!cadence.due(start + Duration::from_millis(1)));
        assert!(!cadence.due(start + INTERVAL - SLACK - Duration::from_micros(1)));
        assert!(cadence.due(start + INTERVAL));
    }

    /// A 120 Hz display presents a frame every 8.33 ms, so two frames span one
    /// interval *almost* exactly — and with real timing jitter the second of
    /// them lands a hair early half the time. The UI must still rebuild every
    /// other frame, at the display's 60 Hz, and not slip to every third.
    #[test]
    fn a_high_frame_rate_rebuilds_at_the_interval_not_below_it() {
        let mut cadence = Cadence::default();
        let start = Instant::now();
        let mut runs = 0;
        for frame in 0..240u32 {
            let jitter = if frame % 2 == 0 { 0 } else { 60 };
            let now = start + Duration::from_micros(u64::from(frame) * 8_333 - jitter);
            if cadence.due(now) {
                cadence.ran(now, Duration::MAX);
                runs += 1;
            }
        }
        assert!(
            (118..=122).contains(&runs),
            "{runs} rebuilds in two seconds at 120 Hz"
        );
    }

    #[test]
    fn input_makes_the_next_frame_due() {
        let mut cadence = Cadence::default();
        let start = after(Instant::now(), &mut cadence, Duration::MAX);
        cadence.touch();
        assert!(cadence.due(start + Duration::from_micros(100)));
        cadence.ran(start + Duration::from_micros(100), Duration::MAX);
        assert!(
            !cadence.due(start + Duration::from_micros(200)),
            "a rebuild consumes the touch"
        );
    }

    /// An animation asks for the very next frame and must get it, or a hover
    /// transition steps at the interval instead of the frame rate.
    #[test]
    fn a_repaint_egui_asked_for_is_honoured_when_it_falls_due() {
        let mut cadence = Cadence::default();
        let start = after(Instant::now(), &mut cadence, Duration::ZERO);
        assert!(cadence.due(start));

        let start = after(start, &mut cadence, Duration::from_millis(5));
        assert!(!cadence.due(start + Duration::from_millis(4)));
        assert!(cadence.due(start + Duration::from_millis(5)));
    }
}
