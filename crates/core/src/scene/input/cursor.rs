//! Whether the game wants the cursor locked: hidden, held in the window, and
//! reported only as motion — what a first-person game looks with.
//!
//! A request, not the window's state. The window is grabbed only while the
//! game is running and the window has focus, and the editor's fly camera,
//! which grabs it while the right button is held, is the other reason it may
//! be. Both go through [`wants_grab`], so neither releases the other's grab.
//! In the editor, Escape drops the request, as Unity's does, so a locked game
//! can always be got out of; a shipped game decides about Escape itself.

/// The game's request: a world resource, written by `Input.cursorLocked`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CursorLock {
    pub requested: bool,
}

/// Whether the window should hold the cursor this frame.
pub fn wants_grab(
    fly_camera_looking: bool,
    playing: bool,
    focused: bool,
    lock: CursorLock,
) -> bool {
    fly_camera_looking || (playing && focused && lock.requested)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCKED: CursorLock = CursorLock { requested: true };

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
    }
}
