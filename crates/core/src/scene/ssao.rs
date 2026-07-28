#[derive(Clone, Copy, Debug)]
pub struct SsaoSettings {
    /// When false the SSAO passes are skipped and occlusion is 1.0 everywhere.
    pub enabled: bool,
    /// Hemisphere sample radius in world units.
    pub radius: f32,
    /// Depth bias that fights self-occlusion acne on flat surfaces.
    pub bias: f32,
    /// Contrast applied to the result (`ao = pow(ao, power)`).
    pub power: f32,
}

impl Default for SsaoSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            radius: 1.0,
            bias: 0.025,
            power: 1.0,
        }
    }
}
