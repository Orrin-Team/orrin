//! Whether the game is running, as the frame driver reads it.
//!
//! Edit mode runs no game code: scripts, `spin` and collision write the world
//! directly, and in edit mode every change goes through the edit pipeline
//! (`scene::edit`), so a frame that ticked them would be a stream of edits
//! nobody could undo or save deliberately. Play runs them at full speed on a
//! fork of the scene that Stop throws away. The fork itself is the editor's
//! (`EditorState`); this resource is only what the app needs to know about it.
//!
//! It also says how fast: the time scale a game sets (`Time.timeScale`) lives
//! here rather than on `Time`, because a session's start and end are the
//! transitions `set` already sees. A game that paused itself is therefore
//! unpaused by Stop and by the next Play, whoever drives them, and pausing or
//! stepping inside a session leaves it alone.

/// Where the transport stands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlayState {
    #[default]
    Editing,
    Playing,
    Paused,
}

/// The transport, as a world resource. The editor sets it; the app asks it
/// once a frame whether the simulation advances.
#[derive(Clone, Copy, Debug)]
pub struct Play {
    state: PlayState,
    /// One frame asked for while paused.
    step: bool,
    time_scale: f32,
}

impl Default for Play {
    fn default() -> Self {
        Self {
            state: PlayState::Editing,
            step: false,
            time_scale: 1.0,
        }
    }
}

impl Play {
    pub fn state(&self) -> PlayState {
        self.state
    }

    /// Playing or paused: the world is the fork, not the scene.
    pub fn is_live(&self) -> bool {
        self.state != PlayState::Editing
    }

    pub fn set(&mut self, state: PlayState) {
        if (state == PlayState::Editing) != (self.state == PlayState::Editing) {
            self.time_scale = 1.0;
        }
        self.state = state;
        self.step = false;
    }

    /// How much game time a second of real time is: what spin, physics,
    /// scripts and tweens advance by. Zero is paused with scripts still
    /// updating, which the transport's own pause is not.
    pub fn time_scale(&self) -> f32 {
        self.time_scale
    }

    /// Clamped at zero. A scale that is not finite is ignored rather than
    /// clamped: infinity would hand physics an unbounded step debt.
    pub fn set_time_scale(&mut self, scale: f32) {
        if scale.is_finite() {
            self.time_scale = scale.max(0.0);
        }
    }

    /// Advance one frame and stay paused. Pauses first if playing.
    pub fn step(&mut self) {
        if self.is_live() {
            self.state = PlayState::Paused;
            self.step = true;
        }
    }

    /// Whether the simulation runs this frame. Consumes a pending step, so it
    /// is asked exactly once a frame.
    pub fn advance(&mut self) -> bool {
        self.state == PlayState::Playing || std::mem::take(&mut self.step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_mode_never_advances() {
        let mut play = Play::default();
        assert!(!play.advance());
        play.step();
        assert!(
            !play.advance(),
            "a step is a paused game's, not the editor's"
        );
    }

    #[test]
    fn a_step_advances_exactly_one_frame_and_stays_paused() {
        let mut play = Play::default();
        play.set(PlayState::Playing);
        play.step();
        assert_eq!(play.state(), PlayState::Paused);
        assert!(play.advance());
        assert!(!play.advance());
        assert_eq!(play.state(), PlayState::Paused);
    }

    #[test]
    fn the_time_scale_starts_at_one_and_never_goes_below_zero() {
        let mut play = Play::default();
        assert_eq!(play.time_scale(), 1.0);
        play.set_time_scale(0.25);
        assert_eq!(play.time_scale(), 0.25);
        play.set_time_scale(-2.0);
        assert_eq!(play.time_scale(), 0.0);
        play.set_time_scale(f32::NAN);
        play.set_time_scale(f32::INFINITY);
        assert_eq!(play.time_scale(), 0.0, "a non-finite scale is refused");
    }

    /// A game that paused itself must not leave the next session frozen, and
    /// pausing or stepping inside one session must not undo what it asked for.
    #[test]
    fn the_time_scale_lasts_one_session() {
        let mut play = Play::default();
        play.set(PlayState::Playing);
        play.set_time_scale(0.0);
        play.set(PlayState::Paused);
        play.step();
        play.set(PlayState::Playing);
        assert_eq!(play.time_scale(), 0.0);

        play.set(PlayState::Editing);
        assert_eq!(play.time_scale(), 1.0, "Stop resets it");

        play.set_time_scale(3.0);
        play.set(PlayState::Playing);
        assert_eq!(play.time_scale(), 1.0, "Play starts at one");
    }

    #[test]
    fn stopping_drops_a_step_not_yet_taken() {
        let mut play = Play::default();
        play.set(PlayState::Paused);
        play.step();
        play.set(PlayState::Editing);
        assert!(!play.advance());
    }
}
