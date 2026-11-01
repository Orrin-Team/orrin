//! Whether the game is running, as the frame driver reads it.
//!
//! Edit mode runs no game code: scripts, `spin` and collision write the world
//! directly, and in edit mode every change goes through the edit pipeline
//! (`scene::edit`), so a frame that ticked them would be a stream of edits
//! nobody could undo or save deliberately. Play runs them at full speed on a
//! fork of the scene that Stop throws away. The fork itself is the editor's
//! (`EditorState`); this resource is only what the app needs to know about it.

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
#[derive(Clone, Copy, Debug, Default)]
pub struct Play {
    state: PlayState,
    /// One frame asked for while paused.
    step: bool,
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
        self.state = state;
        self.step = false;
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
    fn stopping_drops_a_step_not_yet_taken() {
        let mut play = Play::default();
        play.set(PlayState::Paused);
        play.step();
        play.set(PlayState::Editing);
        assert!(!play.advance());
    }
}
