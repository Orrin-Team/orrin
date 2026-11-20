//! Whether the game wants the cursor locked: hidden, held in the window, and
//! reported only as motion — what a first-person game looks with.
//!
//! A request, not the window's state. The window is grabbed only while the
//! game is running and the window has focus, and the editor's fly camera,
//! which grabs it while the right button is held, is the other reason it may
//! be. Both go through [`wants_grab`], so neither releases the other's grab.
//! In the editor, Escape drops the request, as Unity's does, so a locked game
//! can always be got out of; a shipped game decides about Escape itself.
//!
//! **An agent's click never takes the developer's pointer.** A request raised
//! on a frame whose only presses were the virtual device's — an agent's
//! playtest clicking to lock, as a first-person game asks — is the game's to
//! see (`Input.cursorLocked` reads it, so the game looks around with the
//! agent's mouse motion) and never the window's to grab, focused or not, for
//! as long as it stands. The developer takes the pointer back as they would
//! from any locked game: Escape drops the request, and their own click raises
//! one that grabs.

/// The game's request: a world resource, written by `Input.cursorLocked`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CursorLock {
    pub requested: bool,
    /// The request was raised by the virtual device's input alone.
    pub by_agent: bool,
}

impl CursorLock {
    /// What `Input.cursorLocked = locked` does. `by_agent`: whether this
    /// frame's presses were all the virtual device's, which marks a request it
    /// raises; one already standing keeps whose it was.
    pub fn request(&mut self, locked: bool, by_agent: bool) {
        if locked && !self.requested {
            self.by_agent = by_agent;
        }
        self.requested = locked;
        self.by_agent &= locked;
    }
}

/// Whether the window should hold the cursor this frame.
pub fn wants_grab(
    fly_camera_looking: bool,
    playing: bool,
    focused: bool,
    lock: CursorLock,
) -> bool {
    fly_camera_looking || (playing && focused && lock.requested && !lock.by_agent)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCKED: CursorLock = CursorLock {
        requested: true,
        by_agent: false,
    };

    #[test]
    fn the_cursor_is_held_only_for_a_running_focused_game_or_the_fly_camera() {
        assert!(wants_grab(false, true, true, LOCKED));
        assert!(!wants_grab(false, false, true, LOCKED), "held in edit mode");
        assert!(
            !wants_grab(false, true, false, LOCKED),
            "held without focus"
        );
        assert!(!wants_grab(false, true, true, CursorLock::default()));
        assert!(
            wants_grab(true, false, true, CursorLock::default()),
            "the fly camera lost it"
        );
        let mut agents = CursorLock::default();
        agents.request(true, true);
        assert!(
            !wants_grab(false, true, true, agents),
            "an agent's click grabbed"
        );
        agents.request(true, false);
        assert!(
            !wants_grab(false, true, true, agents),
            "a standing request changed hands"
        );
        agents.request(false, false);
        agents.request(true, false);
        assert!(wants_grab(false, true, true, agents));
    }
}
