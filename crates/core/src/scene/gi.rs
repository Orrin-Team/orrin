use orrin_registry::Reflect;

use crate::gfx::probes::MAX_RAYS;

/// Global illumination: the light that bounced, from irradiance volumes
/// traced round the camera and a cache of the radiance their rays find.
/// `gfx/probes.rs` and `gfx/vulkan/gi.rs` are the design.
///
/// The volumes stand in for the environment's diffuse light wherever they
/// reach: a room is lit by what comes through its windows and off its walls,
/// not by the sky as if it had no roof. Nothing here traces a ray per pixel,
/// so nothing on screen is noisy; what changes over time is a probe's
/// content, and a change of light settles over a few frames.
///
/// Needs ray tracing, on the device and in [`RayTracingSettings`]. Without it
/// every field here is inert and the frame is lit by the environment alone.
///
/// [`RayTracingSettings`]: super::RayTracingSettings
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct GiSettings {
    pub enabled: bool,
    /// Probes along each side of a cascade. Sixteen, or twelve to spend less:
    /// each of the six cascades is this many cubed.
    pub probes: u32,
    /// Rays each probe traces when its cascade is updated: 64, 32 or 16. One
    /// cascade is updated a frame.
    pub rays: u32,
    /// The distance between the finest cascade's probes, in metres. Each
    /// cascade's is twice the last. Light leaks through anything thinner, and
    /// what is smaller casts no indirect shadow.
    pub spacing: f32,
    /// How much of what a probe held it keeps at each update, once it has
    /// averaged its first few. Higher is steadier and slower to follow a
    /// change of light the probe does not see as sudden.
    pub hysteresis: f32,
    /// Whether the environment's reflections are dimmed where less of the
    /// sky's light reaches than the sky alone would give, so a closed room's
    /// speculars do not glow with sky.
    pub reflections: bool,
    /// Whether the volumetric fog takes the light it scatters from the
    /// volumes rather than from the scene's flat ambient.
    pub fog: bool,
    /// A way of looking at the volumes rather than at the frame, so never
    /// saved with one.
    #[reflect(skip)]
    pub debug_view: GiDebugView,
}

/// The volumes shown in place of the frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GiDebugView {
    #[default]
    None,
    /// The light every surface receives from the volumes, and the sky where
    /// they run out, on white: what a matte white world would look like
    /// lit by them alone.
    Irradiance,
}

impl GiDebugView {
    /// The number `shading.glsl` switches on.
    pub fn mode(self) -> u32 {
        match self {
            GiDebugView::None => 0,
            GiDebugView::Irradiance => 1,
        }
    }
}

impl Default for GiSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            probes: 16,
            rays: 64,
            spacing: 1.0,
            hysteresis: 0.95,
            reflections: true,
            fog: true,
            debug_view: GiDebugView::None,
        }
    }
}

impl GiSettings {
    /// What the volumes are built with: a side the atlases can hold, at least
    /// eight rays and no more than an update can read, and a spacing a probe
    /// can be placed at.
    pub fn held(&self) -> Self {
        Self {
            probes: self.probes.clamp(4, 16),
            rays: self.rays.clamp(8, MAX_RAYS),
            spacing: self.spacing.clamp(0.1, 64.0),
            hysteresis: self.hysteresis.clamp(0.0, 0.995),
            ..*self
        }
    }
}
