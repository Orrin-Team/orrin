/// The environment the scene is drawn against, and — once irradiance and the
/// prefiltered specular chain land — lit by.
///
/// The cubemap itself lives in the renderer, baked from an equirectangular
/// source when one is loaded. These are the parameters that can change without
/// a rebake.
#[derive(Clone, Copy, Debug)]
pub struct EnvironmentSettings {
    /// Multiplier on environment radiance. Separate from
    /// [`HdrSettings::exposure`](super::HdrSettings) because that scales the
    /// whole frame, while this balances the environment against the analytic
    /// lights.
    pub intensity: f32,
    /// Rotation of the environment about world Y, in degrees. Applied to the
    /// sampling direction, so it costs nothing and needs no rebake.
    pub yaw: f32,
    /// Whether the environment is drawn as the background. Turning it off
    /// leaves the forward pass's clear colour showing.
    pub show_skybox: bool,
}

impl Default for EnvironmentSettings {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            yaw: 0.0,
            show_skybox: true,
        }
    }
}
