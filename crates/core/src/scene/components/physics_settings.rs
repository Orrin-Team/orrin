use glam::Vec3;
use orrin_registry::Reflect;

/// How many collision layers there are. A collider's `layer` past the last
/// is read as the last.
pub const PHYSICS_LAYERS: u32 = 8;

/// The scene's physics, on one entity of it — the first one found, as with
/// [`RenderSettings`](super::RenderSettings). A scene without one simulates with
/// the defaults.
#[derive(Clone, Debug, PartialEq, Reflect)]
pub struct PhysicsSettings {
    /// m/s².
    pub gravity: Vec3,
    /// The fixed step's rate. Higher is stiffer stacking and costs proportionally
    /// more; the frame interpolates between steps either way.
    pub steps_per_second: u32,
    /// A frame slower than this many steps drops the rest of its time rather
    /// than spending more steps catching up, which would make the next frame
    /// slower still.
    pub max_steps_per_frame: u32,
    /// Bit `j` of row `i`: layer `i` collides with layer `j`. A pair collides
    /// only when both of its rows say so, and a missing row says yes to
    /// everything — so the default, empty, is that every layer meets every
    /// other.
    #[reflect(default)]
    pub layer_collisions: Vec<u32>,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            steps_per_second: 60,
            max_steps_per_frame: 4,
            layer_collisions: Vec::new(),
        }
    }
}

impl PhysicsSettings {
    pub fn collides(&self, a: u32, b: u32) -> bool {
        let says = |row: u32, column: u32| {
            self.layer_collisions
                .get(row as usize)
                .is_none_or(|bits| bits & (1 << column) != 0)
        };
        says(a, b) && says(b, a)
    }

    /// Set whether `a` and `b` collide, in both of their rows.
    pub fn set_collides(&mut self, a: u32, b: u32, collide: bool) {
        let rows = (a.max(b) + 1) as usize;
        if self.layer_collisions.len() < rows {
            self.layer_collisions.resize(rows, u32::MAX);
        }
        for (row, column) in [(a, b), (b, a)] {
            let bits = &mut self.layer_collisions[row as usize];
            if collide {
                *bits |= 1 << column;
            } else {
                *bits &= !(1 << column);
            }
        }
    }

    /// The step, clamped to a rate Jolt is stable at and a frame can afford.
    pub fn step(&self) -> f32 {
        1.0 / self.steps_per_second.clamp(10, 1000) as f32
    }
}
