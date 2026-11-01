use orrin_registry::Reflect;

/// Temporal antialiasing: the frame is jittered by a subpixel offset each frame
/// and the result accumulated against a reprojected history.
///
/// It is the only pass whose *input* it also produces — the camera jitter and
/// the motion vectors exist for it — so turning it off changes the projection
/// matrix the whole frame is drawn with, not just which nodes the graph
/// registers.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct TaaSettings {
    /// Off drops the resolve node *and* stops jittering the projection. A frame
    /// that jittered without resolving would simply shake.
    pub enabled: bool,
    /// Weight the reprojected history keeps in the steady state. Higher is
    /// smoother and slower to respond; the neighbourhood clip is what keeps a
    /// high value from smearing rather than this dial.
    pub feedback: f32,
    /// Multiplier on the Halton offset, in pixels. One covers the whole pixel,
    /// which is what actually antialiases; lower trades edge quality for less
    /// texture softening.
    pub jitter_scale: f32,
    /// Rasterise the forward pass at four samples rather than one.
    ///
    /// The alternative to `enabled` above rather than a companion to it, and off
    /// by default. TAA already resolves geometric edges, so multisampling on top
    /// of it buys the absence of one frame of temporal lag on silhouettes for
    /// roughly a quarter of the frame time — and it costs more than the samples:
    /// at one sample the forward pass can attach the geometry prepass's depth
    /// read-only and test `EQUAL` against it, which gives it perfect early-Z and
    /// no shading overdraw. Multisampled, the same geometry is rasterised twice
    /// and the second rasterisation gets nothing from the first.
    ///
    /// The multisampled targets ask for lazily-allocated memory, which makes
    /// them tile-only and free on MoltenVK. No desktop driver exposes that
    /// memory type, so on AMD and NVIDIA they are ordinary VRAM paying full
    /// write-and-resolve traffic — the most expensive pass in the frame.
    ///
    /// Kept switchable rather than deleted so that comparison stays reproducible
    /// and the tiler path stays reachable. Structural: it recompiles the graph.
    pub msaa: bool,
}

impl Default for TaaSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            feedback: 0.92,
            jitter_scale: 1.0,
            msaa: false,
        }
    }
}
