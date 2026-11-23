use orrin_registry::Reflect;

/// Ray tracing: whether the frame keeps a copy of the scene a ray can be
/// traced against, and what traces it.
///
/// Rays are a visibility tool beside the rasterised frame, not a second
/// renderer: the frame is drawn as it always was, and a pass that wants to know
/// what lies in a direction the screen does not show asks the scene's
/// acceleration structure. `gfx/vulkan/accel.rs` is how that structure is
/// built.
///
/// On a device that cannot trace, every field here is inert and the frame is
/// the one it would have been with [`enabled`](Self::enabled) off.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct RayTracingSettings {
    /// Off builds nothing and traces nothing. On, the structures are still
    /// only built in a frame something traces, which is why this alone costs
    /// nothing.
    pub enabled: bool,
    /// How far from the camera an object still stands in the traced scene, in
    /// metres, measured to the nearest point of its bounds. By distance and
    /// not by the view: a ray is traced toward what the camera does not see.
    pub distance: f32,
    /// A way of looking at the traced scene rather than at the frame, so never
    /// saved with one.
    #[reflect(skip)]
    pub debug_view: RayDebugView,
}

/// The traced scene drawn in place of the frame, from one ray per pixel
/// through the camera.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RayDebugView {
    #[default]
    None,
    /// How far each ray went before it hit, near bright and far dark.
    HitDistance,
    /// A colour per instance hit: what the scene holds, and as what.
    Instance,
    /// The hit triangle's own normal, from its three corners: what the
    /// structures were built from, with no interpolation over it.
    GeometricNormal,
}

impl RayDebugView {
    /// The number `ray_debug.frag` switches on.
    pub fn mode(self) -> u32 {
        match self {
            RayDebugView::None => 0,
            RayDebugView::HitDistance => 1,
            RayDebugView::Instance => 2,
            RayDebugView::GeometricNormal => 3,
        }
    }
}

impl Default for RayTracingSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            distance: 200.0,
            debug_view: RayDebugView::None,
        }
    }
}

impl RayTracingSettings {
    /// Whether this frame traces anything, and so builds anything.
    pub fn traces(&self) -> bool {
        self.enabled && self.debug_view != RayDebugView::None
    }
}
