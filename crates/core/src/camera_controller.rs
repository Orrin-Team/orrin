//! The editor's camera: hold right mouse to look, and while it is held WASD
//! move, Q/E go down/up and Shift goes faster; Alt and the left button orbit
//! the pivot, the middle button pans, and scroll dollies toward the pivot.
//! Drives the [`Camera`] resource.
//!
//! The keys fly only while the look is held. Without it W is a letter and S is
//! half of Ctrl+S, and a camera that moved on either would drift on every
//! save. A press with Ctrl, Alt or Super held is a shortcut for the same
//! reason, and never starts a move. Releases are always taken, so no key can
//! stick down, and letting go of the look stops the camera whatever key is
//! still held.
//!
//! The pivot is the selection's centre, which the app hands over each frame,
//! else where the view meets the ground, else a point [`REACH`] ahead. A
//! notch moves a share of the distance to it rather than a fixed step, so the
//! wheel neither crawls across a large scene nor leaps past a small one, and
//! it stops [`MIN_DISTANCE`] short of a pivot in the scene; the point ahead
//! moves with the camera, so there it steps as it always has. A pan moves as far as keeps
//! the pivot's depth under the pointer, which needs the window's height in the
//! pixels the cursor is reported in. Orbit, pan and the wheel are refused
//! while a running game holds the cursor, as they are over a tool; the pointer
//! is the game's then.

use glam::{Quat, Vec3};
use winit::event::{DeviceEvent, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};

use crate::scene::Camera;

// Just under 90°, so the camera can't flip over at the poles.
const PITCH_LIMIT: f32 = 1.55;

/// What a [`FlyCamera::sensitivity`] of 1 turns the view by, in radians per
/// pixel of mouse motion.
const LOOK_RADIANS_PER_PIXEL: f32 = 0.0025;

/// What a sensitivity of 1 orbits by, per pixel of cursor travel: a drag
/// across half a 1280-pixel window goes about half way round the pivot.
const ORBIT_RADIANS_PER_PIXEL: f32 = 0.005;

/// Metres. How far ahead the pivot is when nothing is selected and the view
/// meets no ground.
const REACH: f32 = 4.0;

/// Metres. The distance from the pivot at which a notch moves
/// [`FlyCamera::scroll_speed`] metres: the default 1.5 is 15% of the way a
/// notch. Much more and three notches are most of the way there.
const DOLLY_REFERENCE: f32 = 10.0;

/// Metres. The ground the view meets further off than this is the horizon,
/// and a pivot there would make every notch a leap.
const GROUND_REACH: f32 = 250.0;

/// Metres. The closest the wheel brings the camera to its pivot.
const MIN_DISTANCE: f32 = 0.1;

/// How the fly camera moves: the user's to set, and kept with their other
/// settings rather than with any project.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FlyCamera {
    /// Metres a second.
    pub fly_speed: f32,
    /// What Shift multiplies `fly_speed` by.
    pub boost: f32,
    /// A multiple of the base look rate, about 0.14 degrees per pixel of mouse
    /// motion; a raw radians-per-pixel figure is too small a number to set.
    pub sensitivity: f32,
    /// Metres per notch of the scroll wheel 10 m from the pivot; nearer it is
    /// less and further more, in proportion.
    pub scroll_speed: f32,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self {
            fly_speed: 6.0,
            boost: 4.0,
            sensitivity: 1.0,
            scroll_speed: 1.5,
        }
    }
}

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

pub struct CameraController {
    yaw: f32,
    pitch: f32,
    fly: FlyCamera,

    looking: bool,
    orbiting: bool,
    panning: bool,
    keys: Keys,
    modifiers: ModifiersState,
    look_delta: (f32, f32),
    /// Cursor travel since the last update while orbiting or panning, in
    /// window pixels.
    drag: (f32, f32),
    /// Where the window last reported the cursor; `None` once it has left.
    cursor: Option<(f32, f32)>,
    scroll: f32,
    /// What this orbit turns about, fixed when it starts.
    orbit_pivot: Option<Vec3>,
    selection: Option<Vec3>,
    view_height: u32,
    /// A running game holds the cursor.
    game_cursor: bool,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            fly: FlyCamera::default(),
            looking: false,
            orbiting: false,
            panning: false,
            keys: Keys::default(),
            modifiers: ModifiersState::empty(),
            look_delta: (0.0, 0.0),
            drag: (0.0, 0.0),
            cursor: None,
            scroll: 0.0,
            orbit_pivot: None,
            selection: None,
            view_height: 0,
            game_cursor: false,
        }
    }
}

impl CameraController {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn configure(&mut self, fly: FlyCamera) {
        self.fly = fly;
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
            WindowEvent::MouseInput { button, state, .. } => {
                self.button(*button, state.is_pressed(), egui_wants);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_moved(position.x as f32, position.y as f32);
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => self.key(*code, state.is_pressed(), egui_wants),
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::MouseWheel { delta, .. } => {
                let notches = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                };
                self.wheel(notches, egui_wants);
            }
            _ => {}
        }
    }

    fn button(&mut self, button: MouseButton, pressed: bool, egui_wants: bool) {
        let free = pressed && !egui_wants && !self.game_cursor;
        match button {
            MouseButton::Right => self.looking = pressed && !egui_wants,
            MouseButton::Left => {
                self.orbiting = free && self.modifiers.alt_key();
                self.orbit_pivot = None;
            }
            MouseButton::Middle => self.panning = free,
            _ => {}
        }
    }

    fn cursor_moved(&mut self, x: f32, y: f32) {
        if let Some((from_x, from_y)) = self.cursor
            && (self.orbiting || self.panning)
        {
            self.drag.0 += x - from_x;
            self.drag.1 += y - from_y;
        }
        self.cursor = Some((x, y));
    }

    fn wheel(&mut self, notches: f32, egui_wants: bool) {
        if !egui_wants && !self.game_cursor {
            self.scroll += notches;
        }
    }

    /// Whether this frame's update reads the pivot, so the selection's centre
    /// is measured only on a frame that uses it.
    pub fn wants_pivot(&self) -> bool {
        let dragged = self.drag != (0.0, 0.0);
        self.scroll != 0.0 || (dragged && (self.panning || self.orbit_pivot.is_none()))
    }

    /// The centre of what the editor has selected, which the camera orbits
    /// and dollies toward; `None` with nothing selected.
    pub fn set_selection_centre(&mut self, centre: Option<Vec3>) {
        self.selection = centre;
    }

    /// The window's height in the pixels the cursor is reported in.
    pub fn set_view_height(&mut self, pixels: u32) {
        self.view_height = pixels;
    }

    /// Whether a running game holds the cursor, as `scene::input::cursor`
    /// decides it.
    pub fn set_game_cursor(&mut self, held: bool) {
        self.game_cursor = held;
    }

    /// What the camera at `position` looking along `forward` turns about.
    fn pivot(&self, position: Vec3, forward: Vec3) -> Vec3 {
        self.selection
            .or_else(|| ground(position, forward))
            .unwrap_or(position + forward * REACH)
    }

    /// How far in front of the camera a pivot in the scene is: the
    /// selection's when it is in front, since a dolly runs along the view,
    /// else the ground's. `None` when the pivot is only the point ahead, which
    /// moves with the camera and so has nothing to stop short of.
    fn pivot_depth(&self, position: Vec3, forward: Vec3) -> Option<f32> {
        let depth = |point: Vec3| (point - position).dot(forward);
        self.selection
            .map(depth)
            .filter(|&depth| depth > MIN_DISTANCE)
            .or_else(|| ground(position, forward).map(depth))
    }

    fn key(&mut self, code: KeyCode, pressed: bool, egui_wants: bool) {
        let shortcut =
            self.modifiers.control_key() || self.modifiers.alt_key() || self.modifiers.super_key();
        if pressed && (egui_wants || shortcut) {
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

    pub fn process_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event
            && self.looking
        {
            self.look_delta.0 += *dx as f32;
            self.look_delta.1 += *dy as f32;
        }
    }

    /// Does nothing when idle, so the editor's manual camera edits are left
    /// untouched.
    pub fn update(&mut self, camera: &mut Camera, dt: f32) {
        let dragged = (self.orbiting || self.panning) && self.drag != (0.0, 0.0);
        if dragged && self.orbiting {
            self.orbit(camera);
        } else if dragged {
            self.pan(camera);
        }
        self.drag = (0.0, 0.0);
        let active = self.looking || self.scroll != 0.0;
        if !active {
            self.look_delta = (0.0, 0.0);
            self.scroll = 0.0;
            return;
        }

        // Re-derive from the camera each frame so edits made elsewhere (e.g. the
        // environment panel) are respected, then fold in this frame's look delta.
        self.sync_from(camera);
        if self.looking {
            let rate = self.fly.sensitivity * LOOK_RADIANS_PER_PIXEL;
            self.yaw += self.look_delta.0 * rate;
            self.pitch = (self.pitch - self.look_delta.1 * rate).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        }

        let forward = forward_dir(self.yaw, self.pitch);
        let right = forward.cross(Vec3::Y).normalize_or_zero();

        let speed = self.fly.fly_speed * if self.keys.fast { self.fly.boost } else { 1.0 } * dt;
        let mut pos = camera.position;
        let keys = &self.keys;
        if self.looking {
            for (held, step) in [
                (keys.forward, forward),
                (keys.back, -forward),
                (keys.right, right),
                (keys.left, -right),
                (keys.up, Vec3::Y),
                (keys.down, -Vec3::Y),
            ] {
                if held {
                    pos += step * speed;
                }
            }
        }
        if self.scroll != 0.0 {
            let (depth, limit) = match self.pivot_depth(pos, forward) {
                Some(depth) => (depth, depth - MIN_DISTANCE),
                None => (DOLLY_REFERENCE, f32::INFINITY),
            };
            let step = self.scroll * self.fly.scroll_speed * depth / DOLLY_REFERENCE;
            pos += forward * step.min(limit);
        }

        camera.position = pos;
        camera.target = pos + forward;

        self.look_delta = (0.0, 0.0);
        self.scroll = 0.0;
    }

    /// Turn the camera and its target together about the pivot, so whatever
    /// the view held stays where it was relative to it.
    fn orbit(&mut self, camera: &mut Camera) {
        let forward = (camera.target - camera.position).normalize_or(Vec3::NEG_Z);
        let pivot = match self.orbit_pivot {
            Some(pivot) => pivot,
            None => *self
                .orbit_pivot
                .insert(self.pivot(camera.position, forward)),
        };
        let rate = self.fly.sensitivity * ORBIT_RADIANS_PER_PIXEL;
        let pitch = forward.y.clamp(-1.0, 1.0).asin();
        let tilt = (pitch - self.drag.1 * rate).clamp(-PITCH_LIMIT, PITCH_LIMIT) - pitch;
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);
        // About Y the other way to the drag: a turn to the right is a
        // negative angle about +Y in a right-handed frame.
        let rotation =
            Quat::from_rotation_y(-self.drag.0 * rate) * Quat::from_axis_angle(right, tilt);
        camera.position = pivot + rotation * (camera.position - pivot);
        camera.target = pivot + rotation * (camera.target - pivot);
    }

    /// Slide the camera and its target across the view, by as much as the
    /// pivot's depth moves under the pointer.
    fn pan(&mut self, camera: &mut Camera) {
        let forward = (camera.target - camera.position).normalize_or(Vec3::NEG_Z);
        let depth = self
            .pivot_depth(camera.position, forward)
            .unwrap_or(REACH)
            .max(MIN_DISTANCE);
        let per_pixel = 2.0 * depth * (camera.fov_y * 0.5).tan() / self.view_height.max(1) as f32;
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(forward);
        let offset = (up * self.drag.1 - right * self.drag.0) * per_pixel;
        camera.position += offset;
        camera.target += offset;
    }
}

/// Where the view from `position` along `forward` meets the ground `y = 0`,
/// when it does in front and nearer than the horizon.
fn ground(position: Vec3, forward: Vec3) -> Option<Vec3> {
    if forward.y.abs() <= 1e-4 {
        return None;
    }
    let t = -position.y / forward.y;
    (t > MIN_DISTANCE && t <= GROUND_REACH).then(|| position + forward * t)
}

// Unit forward vector for a yaw/pitch pair (yaw 0 faces -Z).
fn forward_dir(yaw: f32, pitch: f32) -> Vec3 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    Vec3::new(sy * cp, sp, -cy * cp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> Camera {
        Camera {
            position: Vec3::ZERO,
            target: Vec3::NEG_Z,
            ..Camera::default()
        }
    }

    /// W typed with the scene merely in view is a letter, not a command.
    #[test]
    fn w_without_the_look_held_does_not_move() {
        let mut controller = CameraController::new();
        let mut camera = camera();
        controller.key(KeyCode::KeyW, true, false);
        controller.update(&mut camera, 0.5);
        assert_eq!(camera.position, Vec3::ZERO);
    }

    /// Letting go of the look stops the camera, whatever key is still down,
    /// and a key let go of while a field had the keyboard is still let go of.
    #[test]
    fn the_camera_stops_with_the_look_and_keys_never_stick() {
        let mut controller = CameraController::new();
        let mut camera = camera();
        controller.looking = true;
        controller.key(KeyCode::KeyW, true, false);
        controller.looking = false;
        controller.update(&mut camera, 0.5);
        assert_eq!(camera.position, Vec3::ZERO, "drifted after the look");

        controller.key(KeyCode::KeyW, false, true);
        controller.looking = true;
        controller.update(&mut camera, 0.5);
        assert_eq!(camera.position, Vec3::ZERO, "W stuck down");
    }

    #[test]
    fn the_look_and_the_wheel_follow_the_configuration() {
        let mut controller = CameraController::new();
        controller.configure(FlyCamera {
            sensitivity: 2.0,
            scroll_speed: 4.0,
            ..FlyCamera::default()
        });
        let mut camera = camera();
        controller.looking = true;
        controller.look_delta = (100.0, 0.0);
        controller.update(&mut camera, 0.0);
        let turned = 100.0 * LOOK_RADIANS_PER_PIXEL * 2.0;
        assert!((controller.yaw - turned).abs() < 1e-5, "{}", controller.yaw);

        controller.looking = false;
        let before = camera.position;
        controller.scroll = 1.0;
        controller.update(&mut camera, 0.0);
        assert!(((camera.position - before).length() - 4.0).abs() < 1e-4);
    }

    fn alt(controller: &mut CameraController) {
        controller.process_window_event(
            &WindowEvent::ModifiersChanged(ModifiersState::ALT.into()),
            false,
        );
    }

    /// Drag the pointer `by` pixels from the middle of an 800-pixel-high
    /// window, in steps, as the window reports it.
    fn drag(controller: &mut CameraController, by: (f32, f32)) {
        controller.cursor = Some((640.0, 400.0));
        for step in 1..=4 {
            let t = step as f32 / 4.0;
            controller.cursor_moved(640.0 + by.0 * t, 400.0 + by.1 * t);
        }
    }

    fn looking_at(camera: &Camera, point: Vec3) -> bool {
        let forward = (camera.target - camera.position).normalize();
        forward.dot((point - camera.position).normalize()) > 0.9999
    }

    /// Alt and the left button turn the camera about the selection's centre:
    /// it keeps its distance and keeps looking at it, and the turn stops with
    /// the button.
    #[test]
    fn alt_and_the_left_button_orbit_the_selection() {
        let mut controller = CameraController::new();
        let pivot = Vec3::new(1.0, 0.5, -2.0);
        let mut camera = Camera {
            position: pivot + Vec3::new(0.0, 0.0, 5.0),
            target: pivot,
            ..Camera::default()
        };
        controller.set_selection_centre(Some(pivot));
        alt(&mut controller);
        controller.button(MouseButton::Left, true, false);
        drag(&mut controller, (120.0, -60.0));
        controller.update(&mut camera, 0.016);

        assert!(((camera.position - pivot).length() - 5.0).abs() < 1e-3);
        assert!(looking_at(&camera, pivot), "{camera:?}");
        assert!(camera.position.x < pivot.x - 0.5, "turned: {camera:?}");
        assert!(camera.position.y < pivot.y - 0.2, "and tilted: {camera:?}");

        controller.button(MouseButton::Left, false, true);
        let stopped = camera.position;
        drag(&mut controller, (120.0, 0.0));
        controller.update(&mut camera, 0.016);
        assert_eq!(camera.position, stopped, "a release is always taken");
    }

    /// A middle drag slides the camera and its target together across the
    /// view, as far as keeps the pivot under the pointer.
    #[test]
    fn a_middle_drag_pans_with_the_pointer() {
        let mut controller = CameraController::new();
        controller.set_view_height(800);
        let mut camera = Camera {
            position: Vec3::new(0.0, 0.0, 6.0),
            target: Vec3::ZERO,
            ..Camera::default()
        };
        controller.set_selection_centre(Some(Vec3::ZERO));
        let aspect = 1.6;
        let on_screen = |camera: &Camera| {
            let ndc = camera.view_projection(aspect).project_point3(Vec3::ZERO);
            glam::Vec2::new(ndc.x * 0.5 * 1280.0, ndc.y * 0.5 * 800.0)
        };
        let before = on_screen(&camera);
        let forward = (camera.target - camera.position).normalize();

        controller.button(MouseButton::Middle, true, false);
        drag(&mut controller, (80.0, 40.0));
        controller.update(&mut camera, 0.016);
        let moved = on_screen(&camera) - before;
        assert!(
            (moved - glam::Vec2::new(80.0, 40.0)).length() < 1.0,
            "{moved}"
        );
        let turned = (camera.target - camera.position).normalize();
        assert!(turned.dot(forward) > 0.99999, "a pan does not turn");
        assert!((camera.position.z - 6.0).abs() < 1e-4, "nor dolly");

        controller.button(MouseButton::Middle, false, true);
        let stopped = camera.position;
        drag(&mut controller, (80.0, 0.0));
        controller.update(&mut camera, 0.016);
        assert_eq!(camera.position, stopped);
    }

    /// A notch moves a share of the distance to the pivot, so it is a step at
    /// any scale; the default is about today's 1.5 m in the default scene,
    /// and no amount of scrolling goes through the pivot.
    #[test]
    fn a_notch_dollies_a_share_of_the_distance_and_never_through_the_pivot() {
        let mut controller = CameraController::new();
        let step = |controller: &mut CameraController, distance: f32| {
            let mut camera = Camera {
                position: Vec3::new(0.0, 0.0, distance),
                target: Vec3::ZERO,
                ..Camera::default()
            };
            controller.wheel(1.0, false);
            controller.update(&mut camera, 0.0);
            distance - camera.position.z
        };
        controller.set_selection_centre(Some(Vec3::ZERO));
        let (near, far) = (step(&mut controller, 2.0), step(&mut controller, 40.0));
        assert!(
            near > 0.0 && (far / near - 20.0).abs() < 1e-3,
            "{near} {far}"
        );

        controller.set_selection_centre(None);
        let mut camera = Camera::default();
        controller.wheel(1.0, false);
        controller.update(&mut camera, 0.0);
        let moved = (camera.position - Camera::default().position).length();
        let away = (Camera::default().target - Camera::default().position).length();
        let share = moved / away;
        assert!((0.1..=0.2).contains(&share), "{share} of the way a notch");

        controller.set_selection_centre(Some(Vec3::ZERO));
        let mut camera = Camera {
            position: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            ..Camera::default()
        };
        controller.wheel(500.0, false);
        controller.update(&mut camera, 0.0);
        assert!(camera.position.z > 0.0, "through the pivot: {camera:?}");
        assert!(looking_at(&camera, Vec3::ZERO));

        controller.wheel(1.0, true);
        let before = camera.position;
        controller.update(&mut camera, 0.0);
        assert_eq!(camera.position, before, "over a tool");
    }

    /// Over a tool, or while a running game holds the cursor, a press is not
    /// the camera's; a release always is.
    #[test]
    fn a_press_over_a_tool_or_a_held_cursor_is_not_the_cameras() {
        let mut controller = CameraController::new();
        let mut camera = Camera::default();
        alt(&mut controller);
        controller.button(MouseButton::Left, true, true);
        controller.button(MouseButton::Middle, true, true);
        drag(&mut controller, (120.0, 40.0));
        controller.update(&mut camera, 0.016);
        assert_eq!(camera.position, Camera::default().position, "over a tool");

        controller.set_game_cursor(true);
        controller.button(MouseButton::Left, true, false);
        controller.button(MouseButton::Middle, true, false);
        controller.wheel(1.0, false);
        drag(&mut controller, (120.0, 40.0));
        controller.update(&mut camera, 0.016);
        assert_eq!(camera.position, Camera::default().position, "the game's");
    }
}
