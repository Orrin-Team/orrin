//! Device state — what the keyboard, mouse and pads are doing right now —
//! stored as a world resource.
//!
//! The key codes are the engine's own stable numbering, not winit's enum, so
//! scripts can read them through the scripting ABI; [`super::keys`] holds that
//! numbering and the names either side of the boundary spell it with.
//!
//! Events reach this module through [`InputState::press`] and its siblings, and
//! the backends do nothing but decode an event into one of those calls. That is
//! the seam: a new device is a new adapter, and nothing above here — bindings,
//! actions, axes — learns which backend filled the state it reads.
//!
//! `pressed`/`released` are edge-triggered and valid for exactly one frame —
//! [`end_frame`](InputState::end_frame) clears them after scripts have observed
//! them. Keys and mouse buttons both carry them, which is what lets a tap that
//! begins and ends inside one frame still reach a script; a pad does not,
//! because it is sampled rather than evented (see [`super::gamepad`]).

use std::collections::HashSet;

use winit::event::{
    DeviceEvent, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::keyboard::PhysicalKey;

use super::MAX_PLAYERS;
use super::keys::{self, PadAxis, PadButton};

#[derive(Default)]
pub struct InputState {
    held: HashSet<u32>,
    pressed: HashSet<u32>,
    released: HashSet<u32>,
    mouse_held: u8, // bitmask: bit 0 = left, 1 = right, 2 = middle
    mouse_pressed: u8,
    mouse_released: u8,
    cursor: (f32, f32),
    mouse_delta: (f32, f32),
    /// Notches of the wheel this frame, x right and y away from the user.
    wheel: (f32, f32),
    pads: [PadState; MAX_PLAYERS],
    focused: bool,
}

/// One player's pad. Buttons are level state rather than edges: an action
/// derives its own edges from its aggregate, so no source needs to.
#[derive(Clone, Copy, Default)]
struct PadState {
    connected: bool,
    buttons: u16,
    axes: [f32; PadAxis::COUNT],
}

impl InputState {
    pub fn new() -> Self {
        Self {
            focused: true,
            ..Self::default()
        }
    }

    /// `egui_wants` is the editor's claim on the event: presses it wants are
    /// ignored (typing in a panel isn't game input), but releases are always
    /// honored so keys never stick.
    pub fn on_window_event(&mut self, event: &WindowEvent, egui_wants: bool) {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => {
                let Some(key) = keys::key_from_winit(*code) else {
                    return;
                };
                match state {
                    ElementState::Pressed => {
                        if !egui_wants {
                            self.press(key);
                        }
                    }
                    ElementState::Released => self.release(key),
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                let Some(index) = mouse_index(*button) else {
                    return;
                };
                if state.is_pressed() {
                    if !egui_wants {
                        self.press_mouse_button(index);
                    }
                } else {
                    self.release_mouse_button(index);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !egui_wants {
                    self.scroll_wheel(match delta {
                        MouseScrollDelta::LineDelta(x, y) => (*x, *y),
                        // A trackpad reports pixels; the editor camera reads
                        // them at the same rate.
                        MouseScrollDelta::PixelDelta(p) => (p.x as f32 / 50.0, p.y as f32 / 50.0),
                    });
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x as f32, position.y as f32);
            }
            WindowEvent::Focused(focused) => {
                self.focused = *focused;
                if !focused {
                    self.clear_held();
                }
            }
            _ => {}
        }
    }

    pub fn on_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.mouse_delta.0 += *dx as f32;
            self.mouse_delta.1 += *dy as f32;
        }
    }

    pub fn end_frame(&mut self) {
        self.pressed.clear();
        self.released.clear();
        self.mouse_pressed = 0;
        self.mouse_released = 0;
        self.mouse_delta = (0.0, 0.0);
        self.wheel = (0.0, 0.0);
    }

    pub fn key_down(&self, code: u32) -> bool {
        self.held.contains(&code)
    }

    pub fn key_pressed(&self, code: u32) -> bool {
        self.pressed.contains(&code)
    }

    pub fn key_released(&self, code: u32) -> bool {
        self.released.contains(&code)
    }

    /// `button`: 0 = left, 1 = right, 2 = middle.
    pub fn mouse_button_down(&self, button: u32) -> bool {
        button < 3 && self.mouse_held & (1 << button) != 0
    }

    pub fn mouse_button_pressed(&self, button: u32) -> bool {
        button < 3 && self.mouse_pressed & (1 << button) != 0
    }

    pub fn mouse_button_released(&self, button: u32) -> bool {
        button < 3 && self.mouse_released & (1 << button) != 0
    }

    /// Cursor position in window coordinates (physical pixels).
    pub fn cursor(&self) -> (f32, f32) {
        self.cursor
    }

    #[cfg(test)]
    pub(crate) fn place_cursor(&mut self, at: (f32, f32)) {
        self.cursor = at;
    }

    pub fn mouse_delta(&self) -> (f32, f32) {
        self.mouse_delta
    }

    /// Notches the wheel turned this frame: x right, y away from the user.
    pub fn wheel(&self) -> (f32, f32) {
        self.wheel
    }

    pub fn scroll_wheel(&mut self, (x, y): (f32, f32)) {
        self.wheel.0 += x;
        self.wheel.1 += y;
    }

    /// Whether the window has focus. Keyboard events stop arriving without it,
    /// but a gamepad's do not — see [`super::Gamepads::pump`].
    pub fn focused(&self) -> bool {
        self.focused
    }

    /// Release everything currently down.
    ///
    /// Called when the window loses focus, where the releases are delivered to
    /// whatever took it: a key held across an alt-tab otherwise stays held for
    /// the rest of the session. The releases are recorded rather than dropped,
    /// so a script sees the key let go exactly once.
    fn clear_held(&mut self) {
        for code in std::mem::take(&mut self.held) {
            self.released.insert(code);
        }
        self.mouse_released |= self.mouse_held;
        self.mouse_held = 0;
        for slot in 0..MAX_PLAYERS {
            self.clear_pad(slot);
        }
    }

    /// Zero a pad's buttons and axes, leaving it connected.
    pub fn clear_pad(&mut self, player: usize) {
        if let Some(pad) = self.pads.get_mut(player) {
            pad.buttons = 0;
            pad.axes = [0.0; PadAxis::COUNT];
        }
    }

    /// Record a key going down. The `insert` guard also filters OS key-repeat,
    /// which arrives as a second press while the key is already held.
    pub fn press(&mut self, code: u32) {
        if self.held.insert(code) {
            self.pressed.insert(code);
        }
    }

    pub fn release(&mut self, code: u32) {
        if self.held.remove(&code) {
            self.released.insert(code);
        }
    }

    /// Record a mouse button going down. `button`: 0 = left, 1 = right,
    /// 2 = middle, matching [`mouse_button_down`](Self::mouse_button_down).
    pub fn press_mouse_button(&mut self, button: u32) {
        if button < 3 {
            let bit = 1 << button;
            self.mouse_pressed |= bit & !self.mouse_held;
            self.mouse_held |= bit;
        }
    }

    pub fn release_mouse_button(&mut self, button: u32) {
        if button < 3 {
            let bit = 1 << button;
            self.mouse_released |= bit & self.mouse_held;
            self.mouse_held &= !bit;
        }
    }

    /// Whether a player has a pad. Player 0 always has the keyboard, so a
    /// keyboard-only session still plays; a pad binding for a slot nobody has
    /// plugged in simply reads as unpressed.
    pub fn pad_connected(&self, player: usize) -> bool {
        self.pads.get(player).is_some_and(|pad| pad.connected)
    }

    pub fn pad_button_down(&self, player: usize, button: PadButton) -> bool {
        self.pads
            .get(player)
            .is_some_and(|pad| pad.buttons & (1 << button.index()) != 0)
    }

    pub fn pad_axis(&self, player: usize, axis: PadAxis) -> f32 {
        self.pads
            .get(player)
            .map_or(0.0, |pad| pad.axes[axis.index()])
    }

    pub fn set_pad_button(&mut self, player: usize, button: PadButton, down: bool) {
        if let Some(pad) = self.pads.get_mut(player) {
            let bit = 1 << button.index();
            if down {
                pad.buttons |= bit;
            } else {
                pad.buttons &= !bit;
            }
        }
    }

    pub fn set_pad_axis(&mut self, player: usize, axis: PadAxis, value: f32) {
        if let Some(pad) = self.pads.get_mut(player) {
            pad.axes[axis.index()] = value;
        }
    }

    /// Attach or detach a player's pad. Detaching clears its state, for the
    /// same reason releases survive the editor's claim on a press: a button
    /// held at the moment the cable leaves sends no release of its own, and a
    /// source that stays down forever is worse than one that misses an event.
    pub fn set_pad_connected(&mut self, player: usize, connected: bool) {
        if let Some(pad) = self.pads.get_mut(player) {
            *pad = PadState {
                connected,
                ..PadState::default()
            };
        }
    }
}

/// Our button numbering for a winit button, or `None` for one we have no name
/// for. See [`keys::MOUSE_BUTTONS`](super::keys::MOUSE_BUTTONS).
fn mouse_index(button: MouseButton) -> Option<u32> {
    Some(match button {
        MouseButton::Left => 0,
        MouseButton::Right => 1,
        MouseButton::Middle => 2,
        _ => return None,
    })
}
