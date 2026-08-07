use std::sync::Arc;

use super::VulkanRenderer;
use super::context::VkContext;
use super::prepass::{FrameUbo, NORMAL_FORMAT};
use super::texture::MipPolicy;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{BufferContents, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::MemoryTypeFilter;
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
use vulkano::shader::EntryPoint;

pub(super) const AO_FORMAT: Format = Format::R8_UNORM;
const NOISE_SIZE: u32 = 4;
const KERNEL_SIZE: usize = 32;
const KERNEL_MAX: usize = 64;

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct SsaoParamsUbo {
    kernel: [[f32; 4]; KERNEL_MAX],
    noise_scale: [f32; 2],
    radius: f32,
    bias: f32,
    power: f32,
    kernel_size: i32,
    _pad: [f32; 2], // round 1048 -> 1056 (multiple of 16) for std140
}

fn build_kernel() -> [[f32; 4]; KERNEL_MAX] {
    let mut seed: u64 = 0x2545F491_4F6CDD1D;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / (1u64 << 31) as f32 // [0,1)
    };
    let mut kernel = [[0.0f32; 4]; KERNEL_MAX];
    for (i, k) in kernel.iter_mut().enumerate().take(KERNEL_SIZE) {
        let v = glam::Vec3::new(next() * 2.0 - 1.0, next() * 2.0 - 1.0, next()).normalize_or_zero()
            * next();
        let t = i as f32 / KERNEL_SIZE as f32;
        let v = v * (0.1 + 0.9 * t * t); // cluster samples near the origin
        *k = [v.x, v.y, v.z, 0.0];
    }
    kernel
}

fn build_noise(ctx: &VkContext) -> Arc<ImageView> {
    let mut seed: u64 = 0x9E3779B9_7F4A7C15;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as f32 / (1u64 << 31) as f32
    };
    let mut pixels = Vec::with_capacity((NOISE_SIZE * NOISE_SIZE * 4) as usize);
    for _ in 0..(NOISE_SIZE * NOISE_SIZE) {
        let x = next() * 2.0 - 1.0;
        let y = next() * 2.0 - 1.0;
        pixels.extend_from_slice(&[
            ((x * 0.5 + 0.5) * 255.0) as u8,
            ((y * 0.5 + 0.5) * 255.0) as u8,
            0,
            255,
        ]);
    }
    super::texture::upload_texture(
        ctx,
        &pixels,
        [NOISE_SIZE, NOISE_SIZE],
        NORMAL_FORMAT,
        MipPolicy::None,
    )
}

fn ao_render_pass(device: &Arc<Device>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            ao: { format: AO_FORMAT, samples: 1, load_op: Clear, store_op: Store },
        },
        pass: { color: [ao], depth_stencil: {} },
    )
    .unwrap()
}

fn build_fullscreen_pipeline(
    device: &Arc<Device>,
    render_pass: &Arc<RenderPass>,
    vs: EntryPoint,
    fs: EntryPoint,
) -> Arc<GraphicsPipeline> {
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

/// What the resolve binds: the tunables, plus the prepass's camera block. The
/// camera half is uploaded by the prepass rather than here, because every other
/// consumer of the prepass reads the same one — the resolve reconstructs view
/// positions from the depth that pass wrote, so it has to use the projection
/// that wrote it, jitter and all.
pub(super) struct SsaoUniforms {
    frame: Subbuffer<FrameUbo>,
    params: Subbuffer<SsaoParamsUbo>,
}

pub struct SsaoPass {
    pub(super) ssao_rp: Arc<RenderPass>,
    pub(super) blur_rp: Arc<RenderPass>,
    ssao_pipeline: Arc<GraphicsPipeline>,
    blur_pipeline: Arc<GraphicsPipeline>,
    uniform_allocator: SubbufferAllocator,
    nearest_clamp: Arc<Sampler>,
    nearest_repeat: Arc<Sampler>,
    noise_view: Arc<ImageView>,
    /// 1x1 white (=1.0) AO view bound when SSAO is disabled, so the forward
    /// shader samples "no occlusion" without a separate code path. Not a graph
    /// resource: with SSAO off the graph has no AO node at all, so there is
    /// nothing for the forward pass to declare a read of.
    white_view: Arc<ImageView>,
    kernel: [[f32; 4]; KERNEL_MAX],
    pub radius: f32,
    pub bias: f32,
    pub power: f32,
}

impl SsaoPass {
    pub fn new(ctx: &VkContext) -> Self {
        let device = &ctx.device;
        let ssao_rp = ao_render_pass(device);
        let blur_rp = ao_render_pass(device);

        let full_vs = fullscreen_vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let ssao_fs = ssao_fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let blur_fs = blur_fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let ssao_pipeline = build_fullscreen_pipeline(device, &ssao_rp, full_vs.clone(), ssao_fs);
        let blur_pipeline = build_fullscreen_pipeline(device, &blur_rp, full_vs, blur_fs);

        let uniform_allocator = SubbufferAllocator::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let nearest_clamp = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap();
        let nearest_repeat = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        Self {
            ssao_rp,
            blur_rp,
            ssao_pipeline,
            blur_pipeline,
            uniform_allocator,
            nearest_clamp,
            nearest_repeat,
            noise_view: build_noise(ctx),
            white_view: super::texture::upload_texture(
                ctx,
                &[255u8],
                [1, 1],
                AO_FORMAT,
                MipPolicy::None,
            ),
            kernel: build_kernel(),
            radius: 1.0,
            bias: 0.025,
            power: 1.0,
        }
    }

    /// A 1x1 white (=1.0) view, bound when SSAO is disabled.
    pub fn white_view(&self) -> Arc<ImageView> {
        self.white_view.clone()
    }

    pub(super) fn begin_frame(&self, extent: [u32; 2], frame: Subbuffer<FrameUbo>) -> SsaoUniforms {
        let params = self
            .uniform_allocator
            .allocate_sized::<SsaoParamsUbo>()
            .unwrap();
        *params.write().unwrap() = SsaoParamsUbo {
            kernel: self.kernel,
            noise_scale: [
                extent[0] as f32 / NOISE_SIZE as f32,
                extent[1] as f32 / NOISE_SIZE as f32,
            ],
            radius: self.radius,
            bias: self.bias,
            power: self.power,
            kernel_size: KERNEL_SIZE as i32,
            _pad: [0.0; 2],
        };

        SsaoUniforms { frame, params }
    }

    /// `depth_view` and `normal_view` are the prepass's graph outputs, declared
    /// as this pass's `Sampled` inputs.
    pub(super) fn record_ao(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        renderer: &VulkanRenderer,
        extent: [u32; 2],
        uniforms: &SsaoUniforms,
        depth_view: Arc<ImageView>,
        normal_view: Arc<ImageView>,
    ) {
        let uniform_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.ssao_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, uniforms.frame.clone()),
                WriteDescriptorSet::buffer(1, uniforms.params.clone()),
            ],
            [],
        )
        .unwrap();
        let texture_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.ssao_pipeline.layout().set_layouts()[1].clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, depth_view, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(1, normal_view, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(
                    2,
                    self.noise_view.clone(),
                    self.nearest_repeat.clone(),
                ),
            ],
            [],
        )
        .unwrap();

        set_viewport(builder, extent);
        builder
            .bind_pipeline_graphics(self.ssao_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.ssao_pipeline.layout().clone(),
                0,
                vec![uniform_set, texture_set],
            )
            .unwrap();
        unsafe { builder.draw(3, 1, 0, 0).unwrap() };
    }

    pub(super) fn record_blur(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        renderer: &VulkanRenderer,
        extent: [u32; 2],
        raw_ao_view: Arc<ImageView>,
    ) {
        let input = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.blur_pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                raw_ao_view,
                self.nearest_clamp.clone(),
            )],
            [],
        )
        .unwrap();

        set_viewport(builder, extent);
        builder
            .bind_pipeline_graphics(self.blur_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.blur_pipeline.layout().clone(),
                0,
                vec![input],
            )
            .unwrap();
        unsafe { builder.draw(3, 1, 0, 0).unwrap() };
    }
}

fn set_viewport(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    extent: [u32; 2],
) {
    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [extent[0] as f32, extent[1] as f32],
        depth_range: 0.0..=1.0,
    };
    builder
        .set_viewport(0, [viewport].into_iter().collect())
        .unwrap();
}

mod fullscreen_vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/fullscreen.vert" }
}
mod ssao_fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/ssao.frag" }
}
mod blur_fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/ssao_blur.frag" }
}
