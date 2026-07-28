use std::sync::Arc;

use glam::{Mat3, Mat4};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{BufferContents, BufferUsage};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassContents,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex as _, VertexDefinition, VertexInputState};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::EntryPoint;
use crate::gfx::{RenderItem, Vertex};
use crate::scene::Camera;
use super::context::VkContext;
use super::swapchain::DEPTH_FORMAT;
use super::VulkanRenderer;

const NORMAL_FORMAT: Format = Format::R8G8B8A8_UNORM;
const AO_FORMAT: Format = Format::R8_UNORM;
const NOISE_SIZE: u32 = 4;
const KERNEL_SIZE: usize = 32;
const KERNEL_MAX: usize = 64;

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct FrameUbo {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
}

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

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct PrepassPush {
    mvp: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
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
        let v = glam::Vec3::new(next() * 2.0 - 1.0, next() * 2.0 - 1.0, next())
            .normalize_or_zero()
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
    super::texture::upload_texture(ctx, &pixels, [NOISE_SIZE, NOISE_SIZE], NORMAL_FORMAT)
}

fn prepass_render_pass(device: &Arc<Device>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            normal: { format: NORMAL_FORMAT, samples: 1, load_op: Clear, store_op: Store },
            depth:  { format: DEPTH_FORMAT,  samples: 1, load_op: Clear, store_op: Store },
        },
        pass: { color: [normal], depth_stencil: {depth}}
    ).unwrap()
}

fn ao_render_pass(device: &Arc<Device>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            ao: { format: AO_FORMAT, samples: 1, load_op: Clear, store_op: Store },
        },
        pass: { color: [ao], depth_stencil: {} },
    ).unwrap()
}

fn build_prepass_pipeline(
    device: &Arc<Device>,
    render_pass: &Arc<RenderPass>,
) -> Arc<GraphicsPipeline> {
    let vs = prepass_vs::load(device.clone()).unwrap().entry_point("main").unwrap();
    let fs = prepass_fs::load(device.clone()).unwrap().entry_point("main").unwrap();
    let vertex_input_state = Vertex::per_vertex().definition(&vs).unwrap();
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone()).unwrap(),
    ).unwrap();
    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    ).unwrap()
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
    ).unwrap();
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
    ).unwrap()
}

struct SsaoTargets {
    _extent: [u32; 2],
    depth_view: Arc<ImageView>,
    normal_view: Arc<ImageView>,
    raw_ao_view: Arc<ImageView>,
    pub blur_ao_view: Arc<ImageView>,
    prepass_fb: Arc<Framebuffer>,
    ssao_fb: Arc<Framebuffer>,
    blur_fb: Arc<Framebuffer>,
}

impl SsaoTargets {
    fn new(
        mem: &Arc<StandardMemoryAllocator>,
        prepass_rp: &Arc<RenderPass>,
        ssao_rp: &Arc<RenderPass>,
        blur_rp: &Arc<RenderPass>,
        extent: [u32; 2],
    ) -> Self {
        let make = |format: Format, usage: ImageUsage| {
            ImageView::new_default(
                Image::new(
                    mem.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format,
                        extent: [extent[0], extent[1], 1],
                        usage,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                ).unwrap()
            ).unwrap()
        };

        let color = ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED;
        let normal_view = make(NORMAL_FORMAT, color);
        let depth_view = make(DEPTH_FORMAT, ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED);
        let raw_ao_view = make(AO_FORMAT, color);
        let blur_ao_view = make(AO_FORMAT, color);

        let prepass_fb = Framebuffer::new(
            prepass_rp.clone(),
            FramebufferCreateInfo {
                attachments: vec![normal_view.clone(), depth_view.clone()],
                ..Default::default()
            }
        ).unwrap();

        let ssao_fb = Framebuffer::new(
            ssao_rp.clone(),
            FramebufferCreateInfo { attachments: vec!{raw_ao_view.clone()}, ..Default::default() }
        ).unwrap();
        let blur_fb = Framebuffer::new(
            blur_rp.clone(),
            FramebufferCreateInfo {attachments: vec!{blur_ao_view.clone()}, ..Default::default() }
        ).unwrap();

        Self { _extent: extent, depth_view, normal_view, raw_ao_view, blur_ao_view, prepass_fb, ssao_fb, blur_fb }
    }
}

pub struct SsaoPass {
    prepass_rp: Arc<RenderPass>,
    ssao_rp: Arc<RenderPass>,
    blur_rp: Arc<RenderPass>,
    prepass_pipeline: Arc<GraphicsPipeline>,
    ssao_pipeline: Arc<GraphicsPipeline>,
    blur_pipeline: Arc<GraphicsPipeline>,
    uniform_allocator: SubbufferAllocator,
    nearest_clamp: Arc<Sampler>,
    nearest_repeat: Arc<Sampler>,
    noise_view: Arc<ImageView>,
    /// 1x1 white (=1.0) AO view bound when SSAO is disabled, so the forward
    /// shader samples "no occlusion" without a separate code path.
    white_view: Arc<ImageView>,
    kernel: [[f32; 4]; KERNEL_MAX],
    targets: SsaoTargets,
    pub radius: f32,
    pub bias: f32,
    pub power: f32,
}

impl SsaoPass {
    pub fn new(ctx: &VkContext, extent: [u32; 2]) -> Self {
        let device = &ctx.device;
        let prepass_rp = prepass_render_pass(device);
        let ssao_rp = ao_render_pass(device);
        let blur_rp = ao_render_pass(device);

        let prepass_pipeline = build_prepass_pipeline(device, &prepass_rp);

        let full_vs = fullscreen_vs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let ssao_fs = ssao_fs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let blur_fs = blur_fs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let ssao_pipeline =
            build_fullscreen_pipeline(device, &ssao_rp, full_vs.clone(), ssao_fs);
        let blur_pipeline = build_fullscreen_pipeline(device, &blur_rp, full_vs, blur_fs);

        let uniform_allocator = SubbufferAllocator::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            }
        );

        let nearest_clamp = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            }
        ).unwrap();
        let nearest_repeat = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            }
        ).unwrap();

        let targets = SsaoTargets::new(
            &ctx.memory_allocator, &prepass_rp, &ssao_rp, &blur_rp, extent,
        );

        Self {
            prepass_rp,
            ssao_rp,
            blur_rp,
            prepass_pipeline,
            ssao_pipeline,
            blur_pipeline,
            uniform_allocator,
            nearest_clamp,
            nearest_repeat,
            noise_view: build_noise(ctx),
            white_view: super::texture::upload_texture(ctx, &[255u8], [1, 1], AO_FORMAT),
            kernel: build_kernel(),
            targets,
            radius: 1.0,
            bias: 0.025,
            power: 1.0,
        }
    }

    pub fn resize(&mut self, mem: &Arc<StandardMemoryAllocator>, extent: [u32; 2]) {
        self.targets =
            SsaoTargets::new(mem, &self.prepass_rp, &self.ssao_rp, &self.blur_rp, extent);
    }

    /// The blurred-AO view to bind into the forward pass this frame.
    pub fn ao_view(&self) -> Arc<ImageView> {
        self.targets.blur_ao_view.clone()
    }

    /// A 1x1 white (=1.0) view, bound when SSAO is disabled.
    pub fn white_view(&self) -> Arc<ImageView> {
        self.white_view.clone()
    }

    pub fn record(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        renderer: &VulkanRenderer,
        items: &[RenderItem],
        camera: &Camera,
        extent: [u32; 2],
    ) {
        let aspect = extent[0] as f32 / extent[1] as f32;
        let view = camera.view();
        let proj = camera.projection(aspect);
        let view_proj = proj * view;

        let frame_buf = self.uniform_allocator.allocate_sized::<FrameUbo>().unwrap();
        *frame_buf.write().unwrap() = FrameUbo {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            inv_proj: proj.inverse().to_cols_array_2d(),
        };

        let params_buf = self.uniform_allocator.allocate_sized::<SsaoParamsUbo>().unwrap();
        *params_buf.write().unwrap() = SsaoParamsUbo {
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

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: 0.0..=1.0,
        };
        let set_vp = |b: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>| {
            b.set_viewport(0, [viewport.clone()].into_iter().collect()).unwrap();
        };

        let frame_set_prepass = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.prepass_pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer(0, frame_buf.clone())],
            [],
        )
            .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.5, 0.5, 1.0, 0.0].into()), Some(1.0.into())],
                    ..RenderPassBeginInfo::framebuffer(self.targets.prepass_fb.clone())
                },
                SubpassBeginInfo { contents: SubpassContents::Inline, ..Default::default() },
            )
            .unwrap();
        set_vp(builder);
        builder
            .bind_pipeline_graphics(self.prepass_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.prepass_pipeline.layout().clone(),
                0,
                vec![frame_set_prepass],
            )
            .unwrap();

        for item in items {
            let Some(mesh) = renderer.meshes.get(item.mesh.0 as usize) else { continue };
            let model = item.model;
            let normal_matrix =
                Mat4::from_mat3(Mat3::from_mat4(model).inverse().transpose());
            let push = PrepassPush {
                mvp: (view_proj * model).to_cols_array_2d(),
                normal_matrix: normal_matrix.to_cols_array_2d(),
            };
            builder
                .push_constants(self.prepass_pipeline.layout().clone(), 0, push)
                .unwrap()
                .bind_vertex_buffers(0, mesh.vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(mesh.index_buffer.clone())
                .unwrap();
            unsafe { builder.draw_indexed(mesh.index_count, 1, 0, 0, 0).unwrap() };
        }
        builder.end_render_pass(Default::default()).unwrap();

        let ssao_uniforms = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.ssao_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, frame_buf),
                WriteDescriptorSet::buffer(1, params_buf),
            ],
            [],
        )
            .unwrap();
        let ssao_textures = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.ssao_pipeline.layout().set_layouts()[1].clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, self.targets.depth_view.clone(), self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(1, self.targets.normal_view.clone(), self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(2, self.noise_view.clone(), self.nearest_repeat.clone()),
            ],
            [],
        )
            .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([1.0, 0.0, 0.0, 0.0].into())],
                    ..RenderPassBeginInfo::framebuffer(self.targets.ssao_fb.clone())
                },
                SubpassBeginInfo { contents: SubpassContents::Inline, ..Default::default() },
            )
            .unwrap();
        set_vp(builder);
        builder
            .bind_pipeline_graphics(self.ssao_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.ssao_pipeline.layout().clone(),
                0,
                vec![ssao_uniforms, ssao_textures],
            )
            .unwrap();
        unsafe { builder.draw(3, 1, 0, 0).unwrap() };
        builder.end_render_pass(Default::default()).unwrap();

        let blur_input = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.blur_pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::image_view_sampler(0, self.targets.raw_ao_view.clone(), self.nearest_clamp.clone())],
            [],
        )
            .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([1.0, 0.0, 0.0, 0.0].into())],
                    ..RenderPassBeginInfo::framebuffer(self.targets.blur_fb.clone())
                },
                SubpassBeginInfo { contents: SubpassContents::Inline, ..Default::default() },
            )
            .unwrap();
        set_vp(builder);
        builder
            .bind_pipeline_graphics(self.blur_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.blur_pipeline.layout().clone(),
                0,
                vec![blur_input],
            )
            .unwrap();
        unsafe { builder.draw(3, 1, 0, 0).unwrap() };
        builder.end_render_pass(Default::default()).unwrap();
    }
}

mod fullscreen_vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/fullscreen.vert" }
}
mod prepass_vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/ssao_prepass.vert" }
}
mod prepass_fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/ssao_prepass.frag" }
}
mod ssao_fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/ssao.frag" }
}
mod blur_fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/ssao_blur.frag" }
}