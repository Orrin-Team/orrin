use glam::Vec3;

/// Exponential height fog. Density decays with altitude, and the amount for a
/// given view ray is the analytic integral of that density along it, so the
/// effect is correct for rays that climb or descend rather than only for
/// horizontal ones.
#[derive(Clone, Copy, Debug)]
pub struct FogSettings {
    pub color: Vec3,
    /// Extinction at `height`. Zero disables the effect entirely.
    pub density: f32,
    /// How fast density decays with altitude. Larger values make a thinner,
    /// more ground-hugging layer; zero makes the fog uniform at every height.
    pub height_falloff: f32,
    /// World-space altitude that `density` is measured at.
    pub height: f32,
}

impl Default for FogSettings {
    fn default() -> Self {
        Self {
            color: Vec3::new(0.55, 0.62, 0.72),
            density: 0.0,
            height_falloff: 0.1,
            height: 0.0,
        }
    }
}
