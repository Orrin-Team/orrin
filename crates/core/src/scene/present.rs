/// How a finished frame is handed to the display.
///
/// Named behaviour rather than a `PresentMode`, because nothing in `scene` may
/// name a graphics API — `gfx::vulkan::swapchain` does the mapping, the same
/// rule that keeps [`RenderBackend`](crate::gfx::RenderBackend) implementable
/// twice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VsyncMode {
    /// Capped to the display's refresh, no tearing. The only mode a driver has
    /// to support, so it is also what the other three fall back to.
    #[default]
    Fifo,
    /// [`Fifo`](Self::Fifo), except that a frame which arrives *late* is
    /// presented immediately rather than held for the next refresh.
    ///
    /// The mode AMD's RDNA guide names first for V-Sync on, and it is worth
    /// having as a named choice because of what it does to the failure mode
    /// rather than to the average: under plain `Fifo` a frame that misses its
    /// vblank by a millisecond waits a whole refresh, so one slow frame costs
    /// two frames' worth of latency and the rate halves until the renderer gets
    /// far enough ahead again. Relaxed tears on exactly those frames and keeps
    /// the rate. On a renderer comfortably inside its budget the two are the
    /// same mode, since neither ever misses.
    FifoRelaxed,
    /// Uncapped and tear-free: the presentation engine keeps the newest queued
    /// frame and discards the rest.
    Mailbox,
    /// Uncapped, may tear — and the mode to measure in. It never withholds an
    /// image, so a frame time taken under it is the renderer's own cost rather
    /// than a queue depth.
    Immediate,
}

/// What the swapchain is built with.
///
/// A setting rather than a constant because it is unmeasurable otherwise: every
/// frame-time figure this engine produces is only meaningful next to the present
/// mode it was taken under, and recompiling to change one is how a baseline goes
/// unrecorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PresentSettings {
    pub vsync: VsyncMode,
    /// How many images the swapchain holds.
    ///
    /// Two is double buffering. Three is what an uncapped mode wants: with two,
    /// the CPU blocks in the acquire as soon as one image is queued for
    /// presentation and the other is being drawn, so the measurement becomes the
    /// presentation engine's turnaround rather than the frame's.
    ///
    /// Clamped to what the surface advertises, so a value it cannot honour falls
    /// back instead of failing swapchain creation.
    pub images: u32,
}

impl Default for PresentSettings {
    fn default() -> Self {
        // Exactly what the engine did when these were constants, so a baseline
        // taken before this became a setting and one taken after are the same
        // measurement. Changing either default is a change to every number
        // anybody has already written down.
        //
        // Two suits the `Fifo` beside it and suits nothing else, which is why
        // the floor an uncapped mode needs is applied where the mode is
        // resolved rather than raised here — see `swapchain::image_count`.
        Self {
            vsync: VsyncMode::Fifo,
            images: 2,
        }
    }
}
