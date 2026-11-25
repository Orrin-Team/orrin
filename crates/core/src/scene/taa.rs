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
    ///
    /// With upscaling on, the four samples are taken at the render extent and
    /// the temporal resolve upscales what they resolve to.
    pub msaa: bool,
    /// How far below the output's extent the scene is rendered, for the
    /// temporal resolve to bring back up. Anything but
    /// [`Native`](Upscaling::Native) implies the resolve, and with it the
    /// jitter, whatever `enabled` says: a frame rendered small and never
    /// resolved would simply be small. Structural, like `enabled`.
    #[reflect(default)]
    pub upscaling: Upscaling,
}

impl Default for TaaSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            feedback: 0.92,
            jitter_scale: 1.0,
            msaa: false,
            upscaling: Upscaling::Native,
        }
    }
}

impl TaaSettings {
    /// Whether the frame jitters and resolves: asked for, or implied by an
    /// upscaling mode.
    pub fn resolves(&self) -> bool {
        self.enabled || self.upscaling != Upscaling::Native
    }
}

/// How far below the output's extent the scene is rendered, per axis.
///
/// The ratios are AMD's FSR quality modes, and the render extent is rounded
/// the way AMD rounds it, so a vendor upscaler bound in place of the engine's
/// resolve is handed the extent it would have chosen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Reflect)]
pub enum Upscaling {
    /// Rendered at the output's extent.
    #[default]
    Native,
    /// 1.5 times each way.
    Quality,
    /// 1.7 times each way.
    Balanced,
    /// 2 times each way.
    Performance,
    /// 3 times each way.
    UltraPerformance,
}

impl Upscaling {
    pub const ALL: [Upscaling; 5] = [
        Upscaling::Native,
        Upscaling::Quality,
        Upscaling::Balanced,
        Upscaling::Performance,
        Upscaling::UltraPerformance,
    ];

    /// The output's extent over the render's, on each axis.
    pub fn ratio(self) -> f32 {
        match self {
            Upscaling::Native => 1.0,
            Upscaling::Quality => 1.5,
            Upscaling::Balanced => 1.7,
            Upscaling::Performance => 2.0,
            Upscaling::UltraPerformance => 3.0,
        }
    }

    /// The extent the scene is rendered at for an `output` this size:
    /// divided by the ratio in single precision and truncated, as AMD's
    /// documented quality modes compute it, and never below one.
    pub fn render_extent(self, output: [u32; 2]) -> [u32; 2] {
        if self == Upscaling::Native {
            return output;
        }
        let ratio = self.ratio();
        output.map(|side| ((side as f32 / ratio) as u32).max(1))
    }

    /// How many frames the jitter sequence takes before it repeats: eight
    /// times the ratio squared, truncated as AMD's documented phase count is,
    /// so the samples still cover each render pixel about eight times per
    /// output pixel. Eight at native, which is the sequence the frame had
    /// before it could upscale.
    pub fn jitter_phases(self) -> u32 {
        let ratio = self.ratio();
        (8.0 * ratio * ratio) as u32
    }

    /// What every material's level of detail is offset by, so a texture
    /// sampled at the render extent is as sharp as at the output's: AMD's
    /// `log2(render / output) - 1`, the extra level for the samples the
    /// resolve gathers over time. Zero at native.
    pub fn texture_lod_bias(self) -> f32 {
        match self {
            Upscaling::Native => 0.0,
            _ => -self.ratio().log2() - 1.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Upscaling::Native => "Native",
            Upscaling::Quality => "Quality (1.5x)",
            Upscaling::Balanced => "Balanced (1.7x)",
            Upscaling::Performance => "Performance (2x)",
            Upscaling::UltraPerformance => "Ultra performance (3x)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The numbers a vendor upscaler will compute for itself, which this has
    /// to agree with for the history and the jitter it is handed to be the
    /// ones it expects: the render extent truncated, and the phase counts the
    /// issue lists for the four modes.
    #[test]
    fn the_modes_agree_with_the_vendor_arithmetic() {
        let phases = Upscaling::ALL.map(Upscaling::jitter_phases);
        assert_eq!(phases, [8, 18, 23, 32, 72]);
        assert_eq!(Upscaling::Native.render_extent([1921, 1081]), [1921, 1081]);
        assert_eq!(Upscaling::Quality.render_extent([1920, 1080]), [1280, 720]);
        assert_eq!(Upscaling::Balanced.render_extent([1920, 1080]), [1129, 635]);
        assert_eq!(
            Upscaling::Performance.render_extent([1921, 1081]),
            [960, 540]
        );
        assert_eq!(Upscaling::UltraPerformance.render_extent([2, 2]), [1, 1]);
    }
}
