#[derive(Clone, Copy, Debug)]
pub struct HdrSettings {
    /// Linear exposure multiplier applied to scene radiance before the ACES
    /// filmic curve.
    pub exposure: f32,
}

impl Default for HdrSettings {
    fn default() -> Self {
        Self { exposure: 1.0 }
    }
}
