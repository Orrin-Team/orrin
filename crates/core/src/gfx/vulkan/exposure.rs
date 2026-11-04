//! Auto-exposure: a luminance histogram over the HDR target, reduced to one
//! adapted value that the tonemap pass reads.
//!
//! Two buffers here are **imported** into the graph rather than created by it,
//! and that is the whole design. A transient starts every frame `Undefined` —
//! that contract is what makes the barrier plan a pure function of the graph's
//! structure — while temporal adaptation is precisely a value that has to
//! survive a frame boundary. So this module owns both allocations and the graph
//! only orders access to them.

use crate::gfx::graph::BufferUsage;
use crate::gfx::rhi::{
    Binding, BindingResource, BufferDesc, BufferId, ComputePipelineDesc, Device, Encoder,
    MemoryLocation, PipelineId, SamplerDesc, SamplerId, TextureViewId,
};
use crate::gfx::shaders::ShaderId;
use crate::scene::HdrSettings;

/// Bins in the luminance histogram. Also both shaders' workgroup size: the
/// histogram pass is 16x16 so one invocation owns one bin when it flushes shared
/// memory, and the averaging pass is one workgroup of this many so the tree
/// reduction needs no second dispatch. Changing it means changing both.
const BINS: u32 = 256;

/// Side of the histogram pass's workgroup, so that `TILE * TILE == BINS`.
const TILE: u32 = 16;

/// The histogram meters one pixel per `STRIDE x STRIDE` block. A geometric mean
/// over a quarter of the frame is the same exposure to well under a bin, and
/// it is a quarter of the invocations and shared atomics. Point-sampled rather
/// than a downsampled image, which would average luminance before the log and
/// meter a different statistic. Pushed to the shader, so the two cannot drift.
const STRIDE: u32 = 2;

/// The grid of metered pixels, which is what the histogram pass is dispatched
/// over and bounds-checks against.
fn metered_extent(extent: [u32; 2]) -> [u32; 2] {
    [extent[0].div_ceil(STRIDE), extent[1].div_ceil(STRIDE)]
}

/// How many pixels the histogram bins, which the averaging pass divides by.
fn metered_pixel_count(extent: [u32; 2]) -> u32 {
    let [width, height] = metered_extent(extent);
    width * height
}

pub struct ExposurePass {
    histogram_pipeline: PipelineId,
    average_pipeline: PipelineId,
    /// Nearest and clamped: the histogram reads by `texelFetch`, which ignores
    /// filtering, but a sampler is still what a `sampler2D` binding wants.
    sampler: SamplerId,
    histogram: BufferId,
    /// What the averaging pass writes and the tonemap pass reads: three
    /// tightly packed floats, `exposure`, `average_luminance` and `ev100`,
    /// matching the `Exposure` blocks in `luminance_average.comp` and
    /// `tonemap.frag` field for field.
    exposure: BufferId,
    /// Mirrored from the world's [`HdrSettings`] each frame, like the SSAO and
    /// shadow tunables next door.
    settings: HdrSettings,
    /// Seconds since the last frame, for the adaptation rates.
    dt: f32,
}

impl ExposurePass {
    pub fn new(device: &Device) -> Self {
        let histogram_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            shader: ShaderId::LuminanceHistogramComp,
        });
        let average_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            shader: ShaderId::LuminanceAverageComp,
        });

        let sampler = device.create_sampler(SamplerDesc {
            linear: false,
            mipmap_linear: false,
            repeat: false,
            compare: false,
            anisotropy: None,
            max_lod: Some(0.0),
        });

        // Zeroed at creation and thereafter reset by the averaging pass itself,
        // which is why the frame needs no clearing pass in front of the
        // histogram.
        let histogram = device.create_buffer_init(
            BufferDesc {
                size: BINS as u64 * 4,
                usage: BufferUsage::STORAGE_BUFFER,
                location: MemoryLocation::GpuOnly,
            },
            &[0u32; BINS as usize],
        );

        // No adapted luminance yet, which the averaging pass reads as "adopt this
        // frame's measurement outright" — so the first metered frame is exposed
        // for the scene rather than for a constant somebody picked.
        //
        // It used to be seeded at 0.18, middle grey, which was defensible while
        // scene radiance was uncalibrated and averaged near one. Physical light
        // units put an overcast afternoon nine stops above that, so the seed
        // became a frame that opens blown out and spends a second recovering. A
        // seed cannot be right for every scene; not needing one can.
        //
        // The exposure and EV100 beside it are still filled from the default
        // manual exposure, for the frames where metering is off and nothing
        // writes this buffer at all.
        let ev100 = HdrSettings::default().manual_ev100;
        let exposure = device.create_buffer_init(
            BufferDesc {
                size: 12,
                usage: BufferUsage::STORAGE_BUFFER,
                location: MemoryLocation::GpuOnly,
            },
            &[
                HdrSettings::exposure_from_ev100(ev100).to_bits(),
                0.0f32.to_bits(),
                ev100.to_bits(),
            ],
        );

        Self {
            histogram_pipeline,
            average_pipeline,
            sampler,
            histogram,
            exposure,
            settings: HdrSettings::default(),
            dt: 0.0,
        }
    }

    pub fn begin_frame(&mut self, settings: &HdrSettings, dt: f32) {
        self.settings = *settings;
        self.dt = dt;
    }

    /// The buffer the tonemap pass reads its exposure out of. Bound every frame,
    /// metering on or off — with it off nothing writes the buffer and the
    /// tonemap shader ignores what it holds.
    pub fn exposure_buffer(&self) -> BufferId {
        self.exposure
    }

    /// Log2-luminance span of the histogram. Never zero: it is a divisor in the
    /// histogram shader, and settings arrive from a UI that can put the two ends
    /// in either order.
    fn log_luminance_range(&self) -> f32 {
        (self.settings.max_log_luminance - self.settings.min_log_luminance).max(1e-3)
    }

    pub fn record_histogram(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        extent: [u32; 2],
        hdr_view: TextureViewId,
    ) {
        let set = device.create_bind_group(
            self.histogram_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::SampledTexture {
                        view: hdr_view,
                        sampler: self.sampler,
                    },
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::StorageBuffer(self.histogram),
                },
            ],
        );

        encoder.bind_pipeline(self.histogram_pipeline);
        encoder.bind_group(0, set);
        let metered = metered_extent(extent);
        encoder.push_constants(&[
            self.settings.min_log_luminance.to_bits(),
            (1.0 / self.log_luminance_range()).to_bits(),
            metered[0],
            metered[1],
            STRIDE,
        ]);

        // The dispatch covers the metered grid and the shader bounds-checks the
        // tail invocations of a partial tile against the same grid, so no
        // invocation reads outside the image.
        encoder.dispatch([metered[0].div_ceil(TILE), metered[1].div_ceil(TILE), 1]);
    }

    pub fn record_average(&self, device: &Device, encoder: &mut Encoder, extent: [u32; 2]) {
        let set = device.create_bind_group(
            self.average_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::StorageBuffer(self.histogram),
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::StorageBuffer(self.exposure),
                },
            ],
        );

        encoder.bind_pipeline(self.average_pipeline);
        encoder.bind_group(0, set);
        encoder.push_constants(&[
            self.settings.min_log_luminance.to_bits(),
            self.log_luminance_range().to_bits(),
            HdrSettings::adaptation_rate(self.settings.adaptation_brighten, self.dt).to_bits(),
            HdrSettings::adaptation_rate(self.settings.adaptation_darken, self.dt).to_bits(),
            self.settings.exposure_compensation.to_bits(),
            // Exactly what the histogram pass counted: it bins one pixel
            // per invocation inside the metered grid and skips the rest.
            metered_pixel_count(extent),
        ]);

        // One workgroup, sized in the shader to the bin count, so every
        // invocation indexes a bin that exists.
        encoder.dispatch([1, 1, 1]);
    }
}

/// The bin mapping, mirrored from the two shaders so it can be asserted without
/// a GPU.
///
/// It is a pair of inverses split across two files — `luminance_histogram.comp`
/// maps a luminance to a bin, `luminance_average.comp` maps an average bin back
/// to a luminance — and the `* 254 + 1` there against the `- 1` here is exactly
/// the kind of thing that survives review while metering half a stop off. That
/// failure is invisible: the image looks exposed, just not correctly, and no
/// validation layer has an opinion about it.
#[cfg(test)]
mod tests {
    use super::*;

    fn to_bin(luminance: f32, min_log: f32, range: f32) -> u32 {
        if luminance < 0.005 {
            return 0;
        }
        let normalized = ((luminance.log2() - min_log) / range).clamp(0.0, 1.0);
        (normalized * 254.0 + 1.0) as u32
    }

    fn from_bin(average_bin: f32, min_log: f32, range: f32) -> f32 {
        (((average_bin - 1.0) / 254.0) * range + min_log).exp2()
    }

    /// A frame of one uniform luminance must meter to that luminance. This is
    /// the round trip that pins the two offsets against each other.
    #[test]
    fn a_uniform_frame_meters_to_its_own_luminance() {
        let settings = HdrSettings::default();
        let min_log = settings.min_log_luminance;
        let range = settings.max_log_luminance - min_log;

        for luminance in [0.02f32, 0.18, 1.0, 4.0, 40.0, 400.0] {
            let bin = to_bin(luminance, min_log, range);
            let recovered = from_bin(bin as f32, min_log, range);
            // One bin spans range/254 stops, so a round trip is exact only to
            // within the quantisation the histogram is built on.
            let stops_off = (recovered / luminance).log2().abs();
            assert!(
                stops_off < range / 254.0,
                "{luminance} cd/m² binned to {bin} and came back {recovered} \
                 ({stops_off} stops off)",
            );
        }
    }

    /// The black bucket has to stay reserved. If a real luminance ever lands in
    /// bin 0 it is both excluded from the average and subtracted from the pixel
    /// count, so it goes missing twice.
    #[test]
    fn only_black_lands_in_the_reserved_bin() {
        let settings = HdrSettings::default();
        let min_log = settings.min_log_luminance;
        let range = settings.max_log_luminance - min_log;

        assert_eq!(to_bin(0.0, min_log, range), 0);
        assert_eq!(to_bin(0.004, min_log, range), 0);
        assert!(to_bin(0.005, min_log, range) >= 1);

        // The black threshold sits *above* the meter's floor, and deliberately:
        // it asks whether a pixel carries information about the exposure, not
        // whether the histogram can represent it. Pulling the floor down widens
        // what can be measured and does not widen what counts as black.
        assert!(0.005f32.log2() > min_log);
    }

    /// The histogram meters one pixel in every `STRIDE x STRIDE` block. Every
    /// pixel it fetches must be inside the frame, the blocks must reach the
    /// frame's far edges (an odd width included), and the averaging pass must
    /// divide by exactly the number of pixels binned, or every frame meters
    /// dark by the ratio between the two.
    #[test]
    fn the_metered_grid_covers_the_frame_and_counts_what_it_bins() {
        for extent in [[1, 1], [2, 2], [3, 5], [1920, 1080], [2561, 1441]] {
            let metered = metered_extent(extent);
            for axis in 0..2 {
                let last_fetch = (metered[axis] - 1) * STRIDE;
                assert!(last_fetch < extent[axis], "{extent:?} fetches outside");
                assert!(
                    last_fetch + STRIDE >= extent[axis],
                    "{extent:?} leaves its far edge unmetered"
                );
            }
            assert_eq!(metered_pixel_count(extent), metered[0] * metered[1]);
        }
        assert_eq!(metered_pixel_count([1920, 1080]), 1920 * 1080 / 4);
    }

    /// The top bin has to be reachable, or the brightest stop of the metering
    /// range is one nothing can ever be measured at.
    #[test]
    fn the_meter_ceiling_reaches_the_last_bin() {
        let settings = HdrSettings::default();
        let min_log = settings.min_log_luminance;
        let range = settings.max_log_luminance - min_log;

        assert_eq!(
            to_bin(settings.max_log_luminance.exp2(), min_log, range),
            255
        );
        assert_eq!(to_bin(1.0e9, min_log, range), 255);
    }
}
