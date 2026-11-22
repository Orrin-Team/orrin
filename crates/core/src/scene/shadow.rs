use orrin_registry::Reflect;

use crate::gfx::shadow_views::{AtlasConfig, floor_power_of_two};
use crate::gfx::shadows::CascadeConfig;

/// Three centimetres: about what the 3x3 kernel this replaced spread a near
/// cascade's edge over, so a scene authored before the width was a setting
/// keeps the contact shadows it had.
const DEFAULT_SOFTNESS: f32 = 0.03;
const DEFAULT_SHARPNESS: f32 = 0.0;
const DEFAULT_FAR_RESOLUTION: u32 = 1024;
/// Past this the remap is a threshold and the edge shows the map's texels.
pub const MAX_SHADOW_SHARPNESS: f32 = 0.9;

#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct ShadowSettings {
    /// When false the shadow pass is never declared and the forward shader
    /// samples a 1x1 "fully lit" depth texture instead.
    pub enabled: bool,
    /// Clamped to `MAX_CASCADES` when the cascades are built.
    pub cascade_count: u32,
    /// Edge of the tile each of the sun's first two views gets, in texels.
    /// Held to half the atlas.
    pub resolution: u32,
    /// Edge of the tile each of the sun's remaining views gets. Those answer
    /// for a distance where a texel is fewer pixels, so they take a smaller
    /// tile by default — which is what leaves the lamps their room in an atlas
    /// the sun shares. Held to a quarter of the atlas.
    #[reflect(default = DEFAULT_FAR_RESOLUTION)]
    pub far_resolution: u32,
    /// How many of the sun's views, nearest first, are fitted to the slice of
    /// the view they answer for rather than placed around the camera.
    ///
    /// A fitted view spends its texels on what is in front of the camera, which
    /// a narrow lens makes a much smaller box than the disc around it. It is
    /// also a different box every time the camera moves or turns, so what it
    /// holds cannot be kept from one frame to the next: it is cleared and
    /// redrawn whole while the camera does either, where a view placed around
    /// the camera keeps what it has. None by default.
    #[reflect(default)]
    pub fitted_cascades: u32,
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
    /// How wide the sun's penumbra is, in metres. A width rather than a kernel
    /// size because a kernel counted in texels is a different width in every
    /// cascade, which shows as a step in softness at each seam. The filter
    /// holds it wherever a cascade's texels allow: no wider than its kernel
    /// reaches and no narrower than antialiases, so a far cascade is softer
    /// than this by as much as its texels are coarse.
    #[reflect(default = DEFAULT_SOFTNESS)]
    pub softness: f32,
    /// How much of the filtered penumbra is visible, 0 to 1. The kernel stays
    /// as wide as `softness` asks, which is what antialiases the edge, and this
    /// narrows the gradient around the edge's midpoint — so a soft kernel does
    /// not read as light leaking under its caster.
    #[reflect(default = DEFAULT_SHARPNESS)]
    pub sharpness: f32,
    /// Whether point and spot lights get tiles of the atlas at all. The
    /// per-light `casts_shadows` says which of them ask; this is the master
    /// switch.
    pub punctual_enabled: bool,
    /// Edge of the atlas every shadow view is a tile of, the sun's and the
    /// lamps' alike, and the largest tile one punctual face takes within it: a
    /// lamp's tile follows how much of the screen its reach covers, up to this
    /// or to the light's own `LightShadows::resolution`. The first is frame
    /// *structure* — it reallocates the image — and the second changes how
    /// many lights fit beside the sun.
    pub atlas_resolution: u32,
    pub atlas_tile_size: u32,
    /// Near plane of every punctual frustum. It cannot come from the light's
    /// range the way the far plane does: depth precision near the light is what
    /// a contact shadow looks like, and a lamp with a long reach still stands
    /// inches from the table it lights.
    pub punctual_near: f32,
    /// The punctual maps' own bias pair, for a light that sets neither itself.
    /// Separate from the sun's because a perspective face and an orthographic
    /// slice do not mean the same thing by a depth unit.
    pub punctual_constant_bias: f32,
    pub punctual_slope_bias: f32,
    /// `softness` and `sharpness` for a punctual light that sets neither
    /// itself. Their own pair for the reason the biases are: a lamp is a
    /// smaller source than the sky's share of the sun, and its texels grow
    /// with distance from it.
    #[reflect(default = DEFAULT_SOFTNESS)]
    pub punctual_softness: f32,
    #[reflect(default = DEFAULT_SHARPNESS)]
    pub punctual_sharpness: f32,
    /// How dark a fully shadowed fragment gets. 1.0 is physically what the
    /// shadow map says; less is an art dial.
    pub strength: f32,
    /// Tint each fragment by which cascade it sampled. A view of the scene
    /// rather than part of it, so never saved with one.
    #[reflect(skip)]
    pub debug_cascades: bool,
    /// Another way of looking at the shadows, for the same reason never saved.
    #[reflect(skip)]
    pub debug_view: ShadowDebugView,
}

/// A way of looking at the shadows rather than at the scene.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShadowDebugView {
    #[default]
    None,
    /// The atlas itself, as a picture in the corner of the frame: which tiles
    /// hold what. Drawn by the surfaces under it, so it shows where there is
    /// one to draw it.
    Atlas,
    /// Each surface by how large a texel of the sun's shadow is against the
    /// pixel it shades: green at one for one, toward red where the shadow is
    /// coarser than the frame, toward blue where it is finer.
    TexelDensity,
    /// Each surface by what this frame did to the sun view it reads: grey for
    /// nothing, green for its moving casters redrawn, blue for a strip scrolled
    /// in, yellow for its cached layer redrawn, red for drawn whole.
    CacheState,
}

impl ShadowSettings {
    /// Which debug view the shaders draw, as the number they switch on. Zero
    /// is none; the cascade tint is one of them.
    pub fn debug_mode(&self) -> f32 {
        match (self.debug_cascades, self.debug_view) {
            (true, _) => 1.0,
            (_, ShadowDebugView::None) => 0.0,
            (_, ShadowDebugView::Atlas) => 2.0,
            (_, ShadowDebugView::TexelDensity) => 3.0,
            (_, ShadowDebugView::CacheState) => 4.0,
        }
    }
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            cascade_count: 4,
            resolution: 2048,
            far_resolution: DEFAULT_FAR_RESOLUTION,
            fitted_cascades: 0,
            max_distance: 100.0,
            lambda: 0.75,
            pullback: 50.0,
            constant_bias: 1.25,
            slope_bias: 2.5,
            softness: DEFAULT_SOFTNESS,
            sharpness: DEFAULT_SHARPNESS,
            punctual_enabled: true,
            // One D16 image of 32 MB, and a second for the cached layers. The
            // sun's two 2048 views and two 1024 ones take five eighths of it,
            // and what is left is 24 tiles of 512: four point lights, or a mix
            // of points and spots.
            atlas_resolution: 4096,
            atlas_tile_size: 512,
            punctual_near: 0.05,
            punctual_constant_bias: 2.0,
            punctual_slope_bias: 3.0,
            punctual_softness: DEFAULT_SOFTNESS,
            punctual_sharpness: DEFAULT_SHARPNESS,
            strength: 1.0,
            debug_cascades: false,
            debug_view: ShadowDebugView::None,
        }
    }
}

impl ShadowSettings {
    /// The filter block as the shaders read it: the sun's width and sharpness,
    /// then the punctual pair. Clamped here because every one of them reaches
    /// a slider, and a sharpness of one divides by zero.
    pub fn filtering(&self) -> [f32; 4] {
        [
            self.softness.max(0.0),
            self.sharpness.clamp(0.0, MAX_SHADOW_SHARPNESS),
            self.punctual_softness.max(0.0),
            self.punctual_sharpness.clamp(0.0, MAX_SHADOW_SHARPNESS),
        ]
    }

    pub fn cascade_config(&self) -> CascadeConfig {
        let atlas = floor_power_of_two(self.atlas_resolution);
        CascadeConfig {
            count: self.cascade_count as usize,
            max_distance: self.max_distance,
            lambda: self.lambda,
            // Powers of two, because a tile is one and a cascade is fitted to
            // its tile's texel count; and bounded so that four views always
            // fit, whatever a slider pair says.
            resolution: floor_power_of_two(self.resolution).min(atlas / 2),
            far_resolution: floor_power_of_two(self.far_resolution).min(atlas / 4),
            pullback: self.pullback,
            fitted: self.fitted_cascades as usize,
        }
    }

    /// How the atlas is cut up for the punctual lights.
    pub fn atlas_config(&self) -> AtlasConfig {
        AtlasConfig {
            resolution: self.atlas_resolution,
            punctual: self.punctual_enabled,
            tile_size: self.atlas_tile_size.clamp(1, self.atlas_resolution.max(1)),
            near: self.punctual_near,
        }
    }

    /// How many punctual faces the atlas has room for beside the sun's views.
    pub fn punctual_tiles(&self) -> u32 {
        let mut tiles = crate::gfx::shadow_views::TileAllocator::new(self.atlas_resolution);
        if self.enabled {
            let config = self.cascade_config();
            for index in 0..config.count.clamp(1, crate::gfx::shadows::MAX_CASCADES) {
                tiles.allocate(config.resolution_of(index));
            }
        }
        let size = self.atlas_config().tile_size;
        std::iter::from_fn(|| tiles.allocate(size)).count() as u32
    }
}
