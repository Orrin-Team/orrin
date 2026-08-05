//! Bloom: a downsample chain over the exposed frame, spread back up and blended
//! into the tonemap.
//!
//! Two chains of images, not one. The upsample adds the coarser level's spread
//! to the down-chain level of the same size, and it cannot write that sum back
//! into the level it read: resources here are unversioned, so a level that is
//! both an input and an output of the chain closes a dependency cycle the
//! compiler rejects. The second chain is what a version per write would
//! otherwise buy, and `an_upsample_cannot_accumulate_into_the_level_it_read`
//! is the test that pins the reason.

use std::sync::Arc;

use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};

use crate::scene::{BloomSettings, HdrSettings};

use super::context::VkContext;
use super::exposure::GpuExposure;

/// Most levels the chain will ever have. Six covers a 4K frame down to about
/// 60x34, past which a level contributes nothing a wider tent filter would not.
pub const MAX_BLOOM_MIPS: usize = 6;

/// Below this the next level would be too small for the 13-tap filter to mean
/// anything, so the chain stops however many levels short of the cap that puts
/// it.
const MIN_BLOOM_EXTENT: u32 = 8;

/// Side of the compute workgroup for every pass in the chain.
const TILE: u32 = 8;

/// How many levels a frame of this size supports.
///
/// Structural: the count decides how many passes the graph registers, so it
/// lives in `FrameConfig` and a resize that changes it recompiles — exactly how
/// the shadow cascade count behaves.
pub fn mip_count(extent: [u32; 2]) -> u8 {
    let mut mips = 0;
    for level in 1..=MAX_BLOOM_MIPS as u32 {
        let width = extent[0] >> level;
        let height = extent[1] >> level;
        if width.min(height) < MIN_BLOOM_EXTENT {
            break;
        }
        mips = level as u8;
    }
    mips
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct PrefilterPush {
    manual_exposure: f32,
    use_auto: u32,
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct UpsamplePush {
    radius: f32,
    scatter: f32,
}

pub struct BloomPass {
    prefilter_pipeline: Arc<ComputePipeline>,
    downsample_pipeline: Arc<ComputePipeline>,
    upsample_pipeline: Arc<ComputePipeline>,
    /// Linear and clamped. Linear because every filter in the chain leans on
    /// bilinear taps to halve its sample count; clamped so the taps that fall
    /// outside a level repeat its edge instead of wrapping the glow around to
    /// the opposite side of the screen.
    sampler: Arc<Sampler>,
    /// Bound to the tonemap pass when bloom is off, so one pipeline serves both
    /// cases. Black rather than white: at a zero strength the blend ignores it,
    /// and if a strength ever leaks through, black is the harmless answer.
    black: Arc<ImageView>,
    settings: BloomSettings,
    hdr: HdrSettings,
}

impl BloomPass {
    pub fn new(ctx: &VkContext) -> Self {
        let device = &ctx.device;
        let prefilter_pipeline = build_pipeline(
            device,
            prefilter_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );
        let downsample_pipeline = build_pipeline(
            device,
            downsample_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );
        let upsample_pipeline = build_pipeline(
            device,
            upsample_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        )
        .unwrap();

        let black = super::texture::upload_texture(
            ctx,
            &[0, 0, 0, 255],
            [1, 1],
            Format::R8G8B8A8_UNORM,
            super::texture::MipPolicy::None,
        );

        Self {
            prefilter_pipeline,
            downsample_pipeline,
            upsample_pipeline,
            sampler,
            black,
            settings: BloomSettings::default(),
            hdr: HdrSettings::default(),
        }
    }

    pub fn begin_frame(&mut self, settings: &BloomSettings, hdr: &HdrSettings) {
        self.settings = *settings;
        self.hdr = *hdr;
    }

    pub fn black_view(&self) -> Arc<ImageView> {
        self.black.clone()
    }

    /// Zero when bloom is off, which is what makes the tonemap pass's blend a
    /// no-op without a second pipeline.
    pub fn strength(&self) -> f32 {
        if self.settings.enabled {
            self.settings.strength
        } else {
            0.0
        }
    }

    /// Half the frame, exposure applied, fireflies weighted down.
    pub fn record_prefilter(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        ctx: &VkContext,
        source: Arc<ImageView>,
        target: Arc<ImageView>,
        exposure: Subbuffer<GpuExposure>,
    ) {
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.prefilter_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, source, self.sampler.clone()),
                WriteDescriptorSet::image_view(1, target.clone()),
                WriteDescriptorSet::buffer(2, exposure),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.prefilter_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.prefilter_pipeline.layout().clone(),
                0,
                vec![set],
            )
            .unwrap()
            .push_constants(
                self.prefilter_pipeline.layout().clone(),
                0,
                PrefilterPush {
                    manual_exposure: self.hdr.manual_exposure(),
                    use_auto: self.hdr.auto_exposure as u32,
                },
            )
            .unwrap();
        dispatch_over(builder, &target);
    }

    pub fn record_downsample(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        ctx: &VkContext,
        source: Arc<ImageView>,
        target: Arc<ImageView>,
    ) {
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.downsample_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, source, self.sampler.clone()),
                WriteDescriptorSet::image_view(1, target.clone()),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.downsample_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.downsample_pipeline.layout().clone(),
                0,
                vec![set],
            )
            .unwrap();
        dispatch_over(builder, &target);
    }

    /// `coarse` is the level above being spread; `same_level` is the down-chain
    /// level of the target's size, which the spread is added to.
    pub fn record_upsample(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        ctx: &VkContext,
        coarse: Arc<ImageView>,
        same_level: Arc<ImageView>,
        target: Arc<ImageView>,
    ) {
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.upsample_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, coarse, self.sampler.clone()),
                WriteDescriptorSet::image_view_sampler(1, same_level, self.sampler.clone()),
                WriteDescriptorSet::image_view(2, target.clone()),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.upsample_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.upsample_pipeline.layout().clone(),
                0,
                vec![set],
            )
            .unwrap()
            .push_constants(
                self.upsample_pipeline.layout().clone(),
                0,
                UpsamplePush {
                    radius: self.settings.radius,
                    scatter: self.settings.scatter,
                },
            )
            .unwrap();
        dispatch_over(builder, &target);
    }
}

/// Cover every texel of `target` with `TILE`-sized groups. Every shader in the
/// chain bounds-checks against `imageSize`, so a partial group at the edge
/// writes nothing extra.
fn dispatch_over(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    target: &Arc<ImageView>,
) {
    let extent = target.image().extent();

    // SAFETY: the dispatch covers exactly `extent`, and each shader discards
    // invocations past `imageSize(u_target)`, so nothing writes outside the
    // image. The descriptors bound above match the shader's layout, and the
    // graph declared every resource this pass touches, so its barriers precede
    // it.
    unsafe {
        builder
            .dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1])
            .unwrap()
    };
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

mod prefilter_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/bloom_prefilter.comp" }
}

mod downsample_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/bloom_downsample.comp" }
}

mod upsample_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/bloom_upsample.comp" }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chain must stop before a level is too small for a 13-tap filter to
    /// mean anything, and must never exceed the array the ids are carried in.
    #[test]
    fn the_chain_length_suits_the_frame() {
        assert_eq!(mip_count([1920, 1080]), MAX_BLOOM_MIPS as u8);
        assert_eq!(mip_count([3840, 2160]), MAX_BLOOM_MIPS as u8);
        // 1080 >> 6 == 16, still above the floor; 256 >> 5 == 8 is the last that
        // clears it.
        assert_eq!(mip_count([256, 256]), 5);
        assert_eq!(mip_count([64, 64]), 3);
        assert_eq!(mip_count([16, 16]), 1);
    }

    /// A window dragged to nothing must not declare a chain at all, rather than
    /// one whose levels resolve to a single texel.
    #[test]
    fn a_tiny_frame_gets_no_chain() {
        assert_eq!(mip_count([8, 8]), 0);
        assert_eq!(mip_count([1, 1]), 0);
        // Narrow but tall: the smaller side is what limits the chain.
        assert_eq!(mip_count([4096, 12]), 0);
    }
}
