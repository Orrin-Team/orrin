use std::sync::Arc;

use vulkano::buffer::BufferContents;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage, SampleCount};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::memory::MemoryPropertyFlags;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};

use super::context::VkContext;
use super::swapchain::DEPTH_FORMAT;

/// Offscreen color format the forward pass renders into. Float, so values can
/// exceed 1.0 before tonemapping clamps them back to displayable range.
pub const HDR_FORMAT: Format = Format::R16G16B16A16_SFLOAT;

/// Must match the forward pass's MSAA sample count.
const MSAA: SampleCount = SampleCount::Sample4;

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct TonemapPush {
    exposure: f32,
}

struct HdrTargets {
    /// Resolved (1-sample) HDR color — what the tonemap pass samples.
    hdr_view: Arc<ImageView>,
    /// [msaa_hdr, depth, hdr_view], matching the forward render pass attachments.
    forward_fb: Arc<Framebuffer>,
}

impl HdrTargets {
    fn new(
        mem: &Arc<StandardMemoryAllocator>,
        forward_rp: &Arc<RenderPass>,
        extent: [u32; 2],
    ) -> Self {
        let make = |format: Format, usage: ImageUsage, samples: SampleCount, alloc: AllocationCreateInfo| {
            ImageView::new_default(
                Image::new(
                    mem.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format,
                        extent: [extent[0], extent[1], 1],
                        usage,
                        samples,
                        ..Default::default()
                    },
                    alloc,
                )
                .unwrap(),
            )
            .unwrap()
        };

        // MSAA color + depth are transient and never read back, so prefer
        // lazily-allocated memory: on Apple/MoltenVK these become memoryless
        // (tile-only), so the heavy 4x HDR + depth targets cost ~no DRAM. On
        // backends without a lazy memory type the allocator just falls back.
        let lazy = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter {
                preferred_flags: MemoryPropertyFlags::DEVICE_LOCAL
                    | MemoryPropertyFlags::LAZILY_ALLOCATED,
                ..MemoryTypeFilter::PREFER_DEVICE
            },
            ..Default::default()
        };

        let msaa_hdr = make(
            HDR_FORMAT,
            ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
            MSAA,
            lazy.clone(),
        );
        let depth = make(
            DEPTH_FORMAT,
            ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
            MSAA,
            lazy,
        );

        let hdr_view = make(
            HDR_FORMAT,
            ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            SampleCount::Sample1,
            AllocationCreateInfo::default(),
        );

        let forward_fb = Framebuffer::new(
            forward_rp.clone(),
            FramebufferCreateInfo {
                attachments: vec![msaa_hdr, depth, hdr_view.clone()],
                ..Default::default()
            },
        )
        .unwrap();

        Self { hdr_view, forward_fb }
    }
}

pub struct HdrPass {
    pub tonemap_rp: Arc<RenderPass>,
    tonemap_pipeline: Arc<GraphicsPipeline>,
    sampler: Arc<Sampler>,
    targets: HdrTargets,
    pub exposure: f32,
}

impl HdrPass {
    pub fn new(
        ctx: &VkContext,
        forward_rp: &Arc<RenderPass>,
        swapchain_format: Format,
        extent: [u32; 2],
    ) -> Self {
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

        let targets = HdrTargets::new(&ctx.memory_allocator, forward_rp, extent);

        Self {
            tonemap_rp,
            tonemap_pipeline,
            sampler,
            targets,
            exposure: 1.0,
        }
    }

    pub fn resize(
        &mut self,
        mem: &Arc<StandardMemoryAllocator>,
        forward_rp: &Arc<RenderPass>,
        extent: [u32; 2],
    ) {
        self.targets = HdrTargets::new(mem, forward_rp, extent);
    }
    
    pub fn forward_framebuffer(&self) -> Arc<Framebuffer> {
        self.targets.forward_fb.clone()
    }
    
    pub fn record_tonemap(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        ctx: &VkContext,
        extent: [u32; 2],
    ) {
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.tonemap_pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                self.targets.hdr_view.clone(),
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
                TonemapPush { exposure: self.exposure },
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
    let vs = fullscreen_vs::load(device.clone()).unwrap().entry_point("main").unwrap();
    let fs = tonemap_fs::load(device.clone()).unwrap().entry_point("main").unwrap();
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
