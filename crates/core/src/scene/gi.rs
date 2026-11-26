use orrin_registry::Reflect;

use crate::gfx::probes::MAX_RAYS;

/// Global illumination: the light that bounced, from irradiance volumes
/// traced round the camera and a cache of the radiance their rays find.
/// `gfx/probes.rs` and `gfx/vulkan/gi.rs` are the design.
///
/// The volumes stand in for the environment's diffuse light wherever they
/// reach: a room is lit by what comes through its windows and off its walls,
/// not by the sky as if it had no roof. What changes over time is a probe's
/// content, and a change of light settles over a few frames.
///
/// The final gather adds what the probes, a metre apart, cannot see: one ray
/// per pixel at a fraction of the resolution, averaged over space and time,
/// so a small object casts its indirect shadow and colours what is beside it.
/// It is the one part of this that is noisy and keeps a history on screen.
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
    /// averaged its first few, where its updates disagree by about a tenth: a
    /// probe whose updates agree more keeps less, and one whose updates
    /// disagree more keeps more, up to all but a 256th. Higher is steadier and
    /// slower to follow a change of light the probe does not see as sudden.
    pub hysteresis: f32,
    /// Whether the environment's reflections are dimmed where less of the
    /// sky's light reaches than the sky alone would give, so a closed room's
    /// speculars do not glow with sky.
    pub reflections: bool,
    /// Whether the volumetric fog takes the light it scatters from the
    /// volumes rather than from the scene's flat ambient.
    pub fog: bool,
    /// Whether the final gather lights the opaque surfaces in place of the
    /// volumes. Blended and refractive surfaces and the fog keep the volumes.
    #[reflect(default = true)]
    pub gather: bool,
    /// The share of the frame's pixels the gather traces a ray for.
    #[reflect(default)]
    pub gather_resolution: GiGatherResolution,
    /// The frames the gather averages a pixel over where its rays agree: the
    /// lag a change of light shows against the noise it leaves. Where they
    /// disagree it averages more, up to 1024, and a light switched on starts
    /// it again. 1 to 64.
    #[reflect(default = DEFAULT_GATHER_HISTORY)]
    pub gather_history: u32,
    /// A way of looking at the light that bounced rather than at the frame,
    /// so never saved with one.
    #[reflect(skip)]
    pub debug_view: GiDebugView,
}

/// The light that bounced, shown in place of the frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GiDebugView {
    #[default]
    None,
    /// The light every surface receives from the volumes, or from the final
    /// gather where it runs, and the sky where they run out, on white: what
    /// a matte white world would look like lit by them alone.
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

/// How many pixels the final gather traces a ray for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
#[repr(u32)]
pub enum GiGatherResolution {
    /// One in four: half each way.
    #[default]
    Half,
    /// One in sixteen: a quarter each way.
    Quarter,
}

impl GiGatherResolution {
    /// How many times each axis of the frame is halved.
    pub fn shift(self) -> u32 {
        match self {
            GiGatherResolution::Half => 1,
            GiGatherResolution::Quarter => 2,
        }
    }
}

const DEFAULT_GATHER_HISTORY: u32 = 32;

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
            gather: true,
            gather_resolution: GiGatherResolution::Half,
            gather_history: DEFAULT_GATHER_HISTORY,
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
            gather_history: self.gather_history.clamp(1, 64),
            ..*self
        }
    }
}
