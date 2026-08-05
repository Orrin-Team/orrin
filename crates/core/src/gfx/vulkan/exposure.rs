//! Auto-exposure: a luminance histogram over the HDR target, reduced to one
//! adapted value that the tonemap pass reads.
//!
//! Two buffers here are **imported** into the graph rather than created by it,
//! and that is the whole design. A transient starts every frame `Undefined` —
//! that contract is what makes the barrier plan a pure function of the graph's
//! structure — while temporal adaptation is precisely a value that has to
//! survive a frame boundary. So this module owns both allocations and the graph
//! only orders access to them.

use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};

use crate::scene::HdrSettings;

use super::context::VkContext;

/// Bins in the luminance histogram. Also both shaders' workgroup size: the
/// histogram pass is 16x16 so one invocation owns one bin when it flushes shared
/// memory, and the averaging pass is one workgroup of this many so the tree
/// reduction needs no second dispatch. Changing it means changing both.
const BINS: u32 = 256;

/// Side of the histogram pass's workgroup, so that `TILE * TILE == BINS`.
const TILE: u32 = 16;

/// What the averaging pass writes and the tonemap pass reads.
///
/// Layout-frozen against the `Exposure` blocks in `luminance_average.comp` and
/// `tonemap.frag`: std430 over three floats is three tightly packed floats, and
/// the Rust side must match field for field.
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
pub(super) struct GpuExposure {
    /// The linear multiplier applied to scene radiance before the ACES curve.
    exposure: f32,
    /// Adapted average scene luminance. The one value carried across frames, and
    /// therefore the reason this buffer is imported rather than transient.
    average_luminance: f32,
    ev100: f32,
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct HistogramPush {
    min_log_luminance: f32,
    inverse_log_luminance_range: f32,
    extent: [u32; 2],
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct AveragePush {
    min_log_luminance: f32,
    log_luminance_range: f32,
    brighten: f32,
    darken: f32,
    exposure_compensation: f32,
    pixel_count: u32,
}

pub struct ExposurePass {
    histogram_pipeline: Arc<ComputePipeline>,
    average_pipeline: Arc<ComputePipeline>,
    /// Nearest and clamped: the histogram reads by `texelFetch`, which ignores
    /// filtering, but a sampler is still what a `sampler2D` binding wants.
    sampler: Arc<Sampler>,
    histogram: Subbuffer<[u32]>,
    exposure: Subbuffer<GpuExposure>,
    /// Mirrored from the world's [`HdrSettings`] each frame, like the SSAO and
    /// shadow tunables next door.
    settings: HdrSettings,
    /// Seconds since the last frame, for the adaptation rates.
    dt: f32,
}

impl ExposurePass {
    pub fn new(ctx: &VkContext) -> Self {
        let device = &ctx.device;
        let histogram_pipeline = build_pipeline(
            device,
            histogram_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );
        let average_pipeline = build_pipeline(
            device,
            average_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::default()
            },
        )
        .unwrap();

        // Zeroed at creation and thereafter reset by the averaging pass itself,
        // which is why the frame needs no clearing pass in front of the
        // histogram.
        let histogram = Buffer::from_iter(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (0..BINS).map(|_| 0u32),
        )
        .expect("failed to allocate the luminance histogram");

        // Seeded at middle grey so the first frame is exposed sanely rather than
        // adapting up out of black while the user watches.
        let seed = 0.18f32;
        let ev100 = (seed * 100.0 / 12.5).log2();
        let exposure = Buffer::from_data(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            GpuExposure {
                exposure: HdrSettings::exposure_from_ev100(ev100),
                average_luminance: seed,
                ev100,
            },
        )
        .expect("failed to allocate the exposure buffer");

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
    pub fn exposure_buffer(&self) -> Subbuffer<GpuExposure> {
        self.exposure.clone()
    }

    /// Log2-luminance span of the histogram. Never zero: it is a divisor in the
    /// histogram shader, and settings arrive from a UI that can put the two ends
    /// in either order.
    fn log_luminance_range(&self) -> f32 {
        (self.settings.max_log_luminance - self.settings.min_log_luminance).max(1e-3)
    }

    pub fn record_histogram(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        ctx: &VkContext,
        extent: [u32; 2],
        hdr_view: Arc<ImageView>,
    ) {
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.histogram_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, hdr_view, self.sampler.clone()),
                WriteDescriptorSet::buffer(1, self.histogram.clone()),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.histogram_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.histogram_pipeline.layout().clone(),
                0,
                vec![set],
            )
            .unwrap()
            .push_constants(
                self.histogram_pipeline.layout().clone(),
                0,
                HistogramPush {
                    min_log_luminance: self.settings.min_log_luminance,
                    inverse_log_luminance_range: 1.0 / self.log_luminance_range(),
                    extent,
                },
            )
            .unwrap();

        // SAFETY: the dispatch covers every pixel of `extent` and the shader
        // bounds-checks the tail invocations of a partial tile against the same
        // extent, so no invocation reads outside the image. The descriptors bound
        // above match the shader's layout, and the graph declared both resources
        // for this pass, so their barriers precede it.
        unsafe {
            builder
                .dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1])
                .unwrap()
        };
    }

    pub fn record_average(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        ctx: &VkContext,
        extent: [u32; 2],
    ) {
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.average_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, self.histogram.clone()),
                WriteDescriptorSet::buffer(1, self.exposure.clone()),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.average_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.average_pipeline.layout().clone(),
                0,
                vec![set],
            )
            .unwrap()
            .push_constants(
                self.average_pipeline.layout().clone(),
                0,
                AveragePush {
                    min_log_luminance: self.settings.min_log_luminance,
                    log_luminance_range: self.log_luminance_range(),
                    brighten: HdrSettings::adaptation_rate(
                        self.settings.adaptation_brighten,
                        self.dt,
                    ),
                    darken: HdrSettings::adaptation_rate(self.settings.adaptation_darken, self.dt),
                    exposure_compensation: self.settings.exposure_compensation,
                    // Exactly what the histogram pass counted: it bins one pixel
                    // per invocation inside the extent and skips the rest.
                    pixel_count: extent[0] * extent[1],
                },
            )
            .unwrap();

        // SAFETY: one workgroup, sized in the shader to the bin count, so every
        // invocation indexes a bin that exists. The descriptors match the
        // shader's layout, and the graph ordered this pass behind the histogram
        // that fills what it reads.
        unsafe { builder.dispatch([1, 1, 1]).unwrap() };
    }
}

fn build_pipeline(
    device: &Arc<Device>,
    entry_point: vulkano::shader::EntryPoint,
) -> Arc<ComputePipeline> {
    let stage = PipelineShaderStageCreateInfo::new(entry_point);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

mod histogram_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/luminance_histogram.comp" }
}

mod average_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/luminance_average.comp" }
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
