use std::sync::Arc;

use vulkano::buffer::BufferContents;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
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
use vulkano::render_pass::{RenderPass, Subpass};

use super::context::VkContext;

/// Offscreen color format the forward pass renders into. Float, so values can
/// exceed 1.0 before tonemapping clamps them back to displayable range.
pub const HDR_FORMAT: Format = Format::R16G16B16A16_SFLOAT;

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct TonemapPush {
    exposure: f32,
}

pub struct HdrPass {
    pub tonemap_rp: Arc<RenderPass>,
    tonemap_pipeline: Arc<GraphicsPipeline>,
    sampler: Arc<Sampler>,
    pub exposure: f32,
}

impl HdrPass {
    pub fn new(ctx: &VkContext, swapchain_format: Format) -> Self {
        let device = &ctx.device;
        let tonemap_rp = tonemap_render_pass(device, swapchain_format);
        let tonemap_pipeline = build_tonemap_pipeline(device, &tonemap_rp);

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        )
        .unwrap();

        Self {
            tonemap_rp,
            tonemap_pipeline,
            sampler,
            exposure: 1.0,
        }
    }

    /// `hdr_view` is the graph's resolved HDR color target, declared as this
    /// pass's `Sampled` input.
    pub fn record_tonemap(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        ctx: &VkContext,
        extent: [u32; 2],
        hdr_view: Arc<ImageView>,
    ) {
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.tonemap_pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                hdr_view,
                self.sampler.clone(),
            )],
            [],
        )
        .unwrap();

        builder
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [extent[0] as f32, extent[1] as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_pipeline_graphics(self.tonemap_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.tonemap_pipeline.layout().clone(),
                0,
                vec![set],
            )
            .unwrap()
            .push_constants(
                self.tonemap_pipeline.layout().clone(),
                0,
                TonemapPush {
                    exposure: self.exposure,
                },
            )
            .unwrap();
        unsafe { builder.draw(3, 1, 0, 0).unwrap() };
    }
}

fn tonemap_render_pass(device: &Arc<Device>, format: Format) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: { format: format, samples: 1, load_op: DontCare, store_op: Store },
        },
        pass: { color: [color], depth_stencil: {} },
    )
    .unwrap()
}

fn build_tonemap_pipeline(
    device: &Arc<Device>,
    render_pass: &Arc<RenderPass>,
) -> Arc<GraphicsPipeline> {
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
    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(VertexInputState::default()),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: None,
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(subpass.into()),
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
