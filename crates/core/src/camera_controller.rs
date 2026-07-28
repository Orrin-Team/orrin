//! Fly-through editor camera: hold right mouse to look (WASD move, Q/E down/up,
//! Shift faster), scroll to dolly. Drives the [`Camera`] resource.

use glam::Vec3;
use winit::event::{DeviceEvent, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::scene::Camera;

// Just under 90°, so the camera can't flip over at the poles.
const PITCH_LIMIT: f32 = 1.55;

#[derive(Default)]
struct Keys {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    fast: bool,
}

impl Keys {
    fn moving(&self) -> bool {
        self.forward || self.back || self.left || self.right || self.up || self.down
    }
}

pub struct CameraController {
    yaw: f32,
    pitch: f32,
    move_speed: f32,
    boost: f32,
    sensitivity: f32,
    scroll_speed: f32,

    looking: bool,
    keys: Keys,
    look_delta: (f32, f32),
    scroll: f32,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            move_speed: 6.0,
            boost: 4.0,
            sensitivity: 0.0025,
            scroll_speed: 1.5,
            looking: false,
            keys: Keys::default(),
            look_delta: (0.0, 0.0),
            scroll: 0.0,
        }
    }
}

impl CameraController {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn sync_from(&mut self, camera: &Camera) {
        let dir = (camera.target - camera.position).normalize_or_zero();
        if dir.length_squared() > 1e-6 {
            self.yaw = dir.x.atan2(-dir.z);
            self.pitch = dir.y.clamp(-1.0, 1.0).asin();
        }
    }

    pub fn looking(&self) -> bool {
        self.looking
    }

    /// `egui_wants` presses are ignored, but releases are always honored so keys
    /// and look mode never get stuck.
    pub fn process_window_event(&mut self, event: &WindowEvent, egui_wants: bool) {
        match event {
            WindowEvent::MouseInput {
                button: MouseButton::Right,
                state,
                ..
            } => {
                if state.is_pressed() {
                    self.looking = !egui_wants;
                } else {
                    self.looking = false;
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state.is_pressed();
                if pressed && egui_wants {
                    return;
                }
                match code {
                    KeyCode::KeyW => self.keys.forward = pressed,
                    KeyCode::KeyS => self.keys.back = pressed,
                    KeyCode::KeyA => self.keys.left = pressed,
                    KeyCode::KeyD => self.keys.right = pressed,
                    KeyCode::KeyE => self.keys.up = pressed,
                    KeyCode::KeyQ => self.keys.down = pressed,
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => self.keys.fast = pressed,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if egui_wants {
                    return;
                }
                self.scroll += match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                };
            }
            _ => {}
        }
    }

    pub fn process_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if self.looking {
                self.look_delta.0 += *dx as f32;
                self.look_delta.1 += *dy as f32;
            }
        }
    }

    /// Does nothing when idle, so the editor's manual camera edits are left
    /// untouched.
    pub fn update(&mut self, camera: &mut Camera, dt: f32) {
        let active = self.looking || self.keys.moving() || self.scroll != 0.0;
        if !active {
            self.look_delta = (0.0, 0.0);
            self.scroll = 0.0;
            return;
        }

        // Re-derive from the camera each frame so edits made elsewhere (e.g. the
        // environment panel) are respected, then fold in this frame's look delta.
        self.sync_from(camera);
        if self.looking {
            self.yaw += self.look_delta.0 * self.sensitivity;
            self.pitch =
                (self.pitch - self.look_delta.1 * self.sensitivity).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        }

        let forward = forward_dir(self.yaw, self.pitch);
        let right = forward.cross(Vec3::Y).normalize_or_zero();

        let speed = self.move_speed * if self.keys.fast { self.boost } else { 1.0 } * dt;
        let mut pos = camera.position;
        if self.keys.forward {
            pos += forward * speed;
        }
        if self.keys.back {
            pos -= forward * speed;
        }
        if self.keys.right {
            pos += right * speed;
        }
        if self.keys.left {
            pos -= right * speed;
        }
        if self.keys.up {
            pos += Vec3::Y * speed;
        }
        if self.keys.down {
            pos -= Vec3::Y * speed;
        }
        pos += forward * self.scroll * self.scroll_speed;

        camera.position = pos;
        camera.target = pos + forward;

        self.look_delta = (0.0, 0.0);
        self.scroll = 0.0;
    }
}

// Unit forward vector for a yaw/pitch pair (yaw 0 faces -Z).
fn forward_dir(yaw: f32, pitch: f32) -> Vec3 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    Vec3::new(sy * cp, sp, -cy * cp)
}
