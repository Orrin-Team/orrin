use orrin_registry::Reflect;

use crate::gfx::punctual::AtlasConfig;
use crate::gfx::shadows::CascadeConfig;

#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct ShadowSettings {
    /// When false the cascade passes are never declared and the forward shader
    /// samples a 1x1 "fully lit" depth texture instead.
    pub enabled: bool,
    /// Clamped to `MAX_CASCADES` when the cascades are built.
    pub cascade_count: u32,
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
    /// Whether point and spot lights write into the punctual atlas at all. The
    /// per-light `casts_shadows` says which of them ask; this is the master
    /// switch, and turning it off drops the atlas pass and its image.
    pub punctual_enabled: bool,
    /// Edge of the punctual atlas in texels, and of one face's tile within it.
    /// Both are frame *structure* in the same way the cascade resolution is —
    /// the first reallocates the image, and the second changes how many lights
    /// fit in it.
    pub atlas_resolution: u32,
    pub atlas_tile_size: u32,
    /// Near plane of every punctual frustum. It cannot come from the light's
    /// range the way the far plane does: depth precision near the light is what
    /// a contact shadow looks like, and a lamp with a long reach still stands
    /// inches from the table it lights.
    pub punctual_near: f32,
    /// The punctual maps' own bias pair. Separate from the cascades' because a
    /// perspective face and an orthographic slice do not mean the same thing by
    /// a depth unit — see `ShadowPass::punctual_constant_bias`.
    pub punctual_constant_bias: f32,
    pub punctual_slope_bias: f32,
    /// How dark a fully shadowed fragment gets. 1.0 is physically what the
    /// shadow map says; less is an art dial.
    pub strength: f32,
    /// Tint each fragment by which cascade it sampled. A view of the scene
    /// rather than part of it, so never saved with one.
    #[reflect(skip)]
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
            punctual_enabled: true,
            // 4096 with 512-texel tiles is 64 faces: ten point lights, or a mix
            // of points and spots up to the eight-caster budget. One D16 image
            // of 32 MB, the same as four 2048 cascades.
            atlas_resolution: 4096,
            atlas_tile_size: 512,
            punctual_near: 0.05,
            punctual_constant_bias: 2.0,
            punctual_slope_bias: 3.0,
            strength: 1.0,
            debug_cascades: false,
        }
    }
}

impl ShadowSettings {
    pub fn cascade_config(&self) -> CascadeConfig {
        CascadeConfig {
            count: self.cascade_count as usize,
            max_distance: self.max_distance,
            lambda: self.lambda,
            resolution: self.resolution,
            pullback: self.pullback,
        }
    }

    /// How the punctual atlas is cut up, or `None` when nothing punctual casts —
    /// which is what makes the graph declare no atlas at all.
    pub fn atlas_config(&self) -> Option<AtlasConfig> {
        self.punctual_enabled.then(|| AtlasConfig {
            resolution: self.atlas_resolution,
            // A tile larger than the atlas would divide to zero columns, and a
            // slider pair can express that.
            tile_size: self.atlas_tile_size.clamp(1, self.atlas_resolution.max(1)),
            near: self.punctual_near,
        })
    }
}
