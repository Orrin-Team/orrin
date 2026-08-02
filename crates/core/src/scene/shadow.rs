use crate::gfx::shadows::CascadeConfig;

#[derive(Clone, Copy, Debug)]
pub struct ShadowSettings {
    /// When false the cascade passes are never declared and the forward shader
    /// samples a 1x1 "fully lit" depth texture instead.
    pub enabled: bool,
    /// Clamped to `MAX_CASCADES` when the cascades are built.
    pub cascade_count: usize,
    /// Edge length of one cascade's depth map, in texels.
    pub resolution: u32,
    /// How far from the camera shadows are cast. Deliberately independent of
    /// the camera's far plane: splitting across a 1000 m view distance would
    /// spend every cascade on geometry nobody can see the shadows of.
    pub max_distance: f32,
    /// Blend between the logarithmic (1.0) and uniform (0.0) split schemes.
    pub lambda: f32,
    /// How far each cascade's near plane is pulled back toward the light, so
    /// casters outside the cascade still write depth into it.
    pub pullback: f32,
    pub constant_bias: f32,
    pub slope_bias: f32,
    /// How dark a fully shadowed fragment gets. 1.0 is physically what the
    /// shadow map says; less is an art dial.
    pub strength: f32,
    /// Tint each fragment by which cascade it sampled.
    pub debug_cascades: bool,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            cascade_count: 4,
            resolution: 2048,
            max_distance: 100.0,
            lambda: 0.75,
            pullback: 50.0,
            constant_bias: 1.25,
            slope_bias: 2.5,
            strength: 1.0,
            debug_cascades: false,
        }
    }
}

impl ShadowSettings {
    pub fn cascade_config(&self) -> CascadeConfig {
        CascadeConfig {
            count: self.cascade_count,
            max_distance: self.max_distance,
            lambda: self.lambda,
            resolution: self.resolution,
            pullback: self.pullback,
        }
    }
}
