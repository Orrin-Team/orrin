use std::sync::Arc;

use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};

use super::context::VkContext;
use super::exposure::GpuExposure;
use super::record::Recorder;
use super::rendering;

/// Offscreen colour format the frame's radiance is carried in. Float, so values
/// can exceed 1.0 before tonemapping clamps them back to displayable range.
///
/// Packed into 32 bits rather than 64. Every full-resolution pass in the optical
/// chain reads one image of this and writes another, so the width of this
/// constant is most of the frame's bandwidth — halving it is worth more than the
/// arithmetic in any of those passes. At 1080p it also brings a read-one-write-one
/// pass to a 16 MB working set, which fits inside a 9070 XT's 64 MB Infinity
/// Cache; that is worth more again than the raw halving.
///
/// Three things are given up, and none of them is used by anything this carries:
/// there is no alpha channel, no negative values, and about five bits of
/// mantissa. Radiance is non-negative by definition, and in physical units with
/// EV100 metering five bits of mantissa is far below what the tonemap resolves —
/// it is what Unreal has used for scene colour for a decade. The targets that
/// *do* need one of the three use [`HDR_WIDE_FORMAT`].
///
/// Not in the Vulkan mandatory `STORAGE_IMAGE` list, though every desktop driver
/// supports it; [`VkContext`](super::context::VkContext) checks at startup rather
/// than letting a compute pass fail to bind much later.
pub const HDR_FORMAT: Format = Format::B10G11R11_UFLOAT_PACK32;

/// The wider colour format, for the targets [`HDR_FORMAT`] cannot carry.
///
/// Three kinds of target need it, for three different reasons:
///
/// - `msaa_hdr`, because alpha to coverage reads the first attachment's alpha,
///   and an attachment with no alpha component behaves as though it were 1.0 —
///   which would silently stop every cutout being cut. A render-pass resolve
///   also requires both images to have the same format, so `hdr_color` follows
///   it whenever the frame is multisampled.
/// - `dof_prefiltered`, `dof_near` and `dof_far`, because the depth-of-field
///   chain packs a *signed* circle of confusion and then a coverage weight into
///   alpha, and the composite divides by it.
/// - The depth-of-field and motion-blur tile images, because they carry two
///   channels of signed maxima — velocities, and near/far CoC radii.
pub const HDR_WIDE_FORMAT: Format = Format::R16G16B16A16_SFLOAT;

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct TonemapPush {
    manual_exposure: f32,
    use_auto: u32,
    bloom_strength: f32,
}

pub struct HdrPass {
    tonemap_pipeline: Arc<GraphicsPipeline>,
    sampler: Arc<Sampler>,
    /// The exposure applied when metering is off, already carrying the
    /// compensation dial. With metering on the shader ignores it and reads the
    /// value the averaging pass wrote instead.
    pub manual_exposure: f32,
    pub auto_exposure: bool,
    /// Zero when bloom is off, which turns the shader's blend into a no-op
    /// without a second pipeline or a branch that costs anything.
    pub bloom_strength: f32,
}

impl HdrPass {
    pub fn new(ctx: &VkContext, swapchain_format: Format) -> Self {
        let device = &ctx.device;
        let tonemap_pipeline = build_tonemap_pipeline(ctx, swapchain_format);

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        )
        .unwrap();

        Self {
            tonemap_pipeline,
            sampler,
            manual_exposure: 1.0,
            auto_exposure: true,
            bloom_strength: 0.0,
        }
    }

    /// `hdr_view` is the graph's resolved HDR color target, declared as this
    /// pass's `Sampled` input; `exposure` is the buffer the metering passes
    /// wrote, declared as its `StorageRead`.
    pub fn record_tonemap(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        extent: [u32; 2],
        hdr_view: Arc<ImageView>,
        exposure: Subbuffer<GpuExposure>,
        bloom_view: Arc<ImageView>,
    ) {
        let set = ctx.descriptor_sets.get(
            &ctx.descriptor_set_allocator,
            &self.tonemap_pipeline.layout().set_layouts()[0],
            [
                WriteDescriptorSet::image_view_sampler(0, hdr_view, self.sampler.clone()),
                WriteDescriptorSet::buffer(1, exposure),
                WriteDescriptorSet::image_view_sampler(2, bloom_view, self.sampler.clone()),
            ],
        );

        builder
            .set_viewport(
                0,
                &[Viewport {
                    offset: [0.0, 0.0],
                    extent: [extent[0] as f32, extent[1] as f32],
                    depth_range: 0.0..=1.0,
                }],
            )
            .bind_pipeline_graphics(&self.tonemap_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.tonemap_pipeline.layout(),
                0,
                &[set],
            )
            .push_constants(
                self.tonemap_pipeline.layout(),
                0,
                &TonemapPush {
                    manual_exposure: self.manual_exposure,
                    use_auto: self.auto_exposure as u32,
                    bloom_strength: self.bloom_strength,
                },
            );
        builder.draw(3, 1, 0, 0);
    }
}

fn build_tonemap_pipeline(ctx: &VkContext, color_format: Format) -> Arc<GraphicsPipeline> {
    let device = &ctx.device;
    let vs = fullscreen_vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = tonemap_fs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    GraphicsPipeline::new(
        device.clone(),
        ctx.pipeline_cache(),
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(VertexInputState::default()),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: None,
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                1,
                ColorBlendAttachmentState::default(),
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(rendering::pipeline_info(&[color_format], None).into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

mod fullscreen_vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/fullscreen.vert" }
}

mod tonemap_fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/tonemap.frag" }
}
