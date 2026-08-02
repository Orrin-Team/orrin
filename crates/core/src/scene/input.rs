//! Keyboard/mouse state collected from winit events, stored as a world resource.
//!
//! The key codes are the engine's own stable numbering, not winit's enum, so
//! scripts can read them through the scripting ABI: `map_key` must stay in
//! lock-step with the `KeyCode` enum in `scripting/Orrin/Input.cs`.
//!
//! `pressed`/`released` are edge-triggered and valid for exactly one frame —
//! [`end_frame`](InputState::end_frame) clears them after scripts have observed
//! them.

use std::collections::HashSet;

use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

#[derive(Default)]
pub struct InputState {
    held: HashSet<u32>,
    pressed: HashSet<u32>,
    released: HashSet<u32>,
    mouse_held: u8, // bitmask: bit 0 = left, 1 = right, 2 = middle
    mouse_pressed: u8,
    cursor: (f32, f32),
    mouse_delta: (f32, f32),
}

impl InputState {
    pub fn new() -> Self {
        Self::default()
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
                let Some(key) = map_key(*code) else { return };
                match state {
                    ElementState::Pressed => {
                        // The `insert` guard also filters OS key-repeat, which
                        // arrives as extra Pressed events while the key is held.
                        if !egui_wants && self.held.insert(key) {
                            self.pressed.insert(key);
                        }
                    }
                    ElementState::Released => {
                        if self.held.remove(&key) {
                            self.released.insert(key);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { button, state, .. } => {
                let Some(bit) = mouse_bit(*button) else {
                    return;
                };
                if state.is_pressed() {
                    if !egui_wants {
                        self.mouse_pressed |= bit & !self.mouse_held;
                        self.mouse_held |= bit;
                    }
                } else {
                    self.mouse_held &= !bit;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x as f32, position.y as f32);
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
        self.mouse_delta = (0.0, 0.0);
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

    /// Cursor position in window coordinates (physical pixels).
    pub fn cursor(&self) -> (f32, f32) {
        self.cursor
    }

    pub fn mouse_delta(&self) -> (f32, f32) {
        self.mouse_delta
    }
}

fn mouse_bit(button: MouseButton) -> Option<u8> {
    Some(match button {
        MouseButton::Left => 1 << 0,
        MouseButton::Right => 1 << 1,
        MouseButton::Middle => 1 << 2,
        _ => return None,
    })
}

/// Map a winit key to the engine's stable code. winit's discriminants aren't a
/// stable ABI, so the numbering is defined here explicitly and must match the
/// C# `Orrin.KeyCode` enum — extend both together.
fn map_key(code: KeyCode) -> Option<u32> {
    use KeyCode::*;
    Some(match code {
        KeyA => 1,
        KeyB => 2,
        KeyC => 3,
        KeyD => 4,
        KeyE => 5,
        KeyF => 6,
        KeyG => 7,
        KeyH => 8,
        KeyI => 9,
        KeyJ => 10,
        KeyK => 11,
        KeyL => 12,
        KeyM => 13,
        KeyN => 14,
        KeyO => 15,
        KeyP => 16,
        KeyQ => 17,
        KeyR => 18,
        KeyS => 19,
        KeyT => 20,
        KeyU => 21,
        KeyV => 22,
        KeyW => 23,
        KeyX => 24,
        KeyY => 25,
        KeyZ => 26,
        Digit0 => 30,
        Digit1 => 31,
        Digit2 => 32,
        Digit3 => 33,
        Digit4 => 34,
        Digit5 => 35,
        Digit6 => 36,
        Digit7 => 37,
        Digit8 => 38,
        Digit9 => 39,
        ArrowLeft => 40,
        ArrowRight => 41,
        ArrowUp => 42,
        ArrowDown => 43,
        Space => 44,
        Enter => 45,
        Escape => 46,
        Tab => 47,
        Backspace => 48,
        ShiftLeft => 49,
        ShiftRight => 50,
        ControlLeft => 51,
        ControlRight => 52,
        AltLeft => 53,
        AltRight => 54,
        _ => return None,
    })
}
