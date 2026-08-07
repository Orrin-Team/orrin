//! The one geometry pass in front of shading: view-space normals, screen-space
//! motion, and the depth both are read against.
//!
//! It started as SSAO's private prepass and is shared now because TAA needs the
//! same rasterisation for a different attachment. Everything downstream of it —
//! SSAO today, screen-space reflections and motion blur when they arrive — reads
//! the targets it leaves rather than rasterising the scene again.

use std::sync::Arc;

use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{BufferContents, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex as _, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{RenderPass, Subpass};

use crate::gfx::{DrawList, Vertex};

use super::VulkanRenderer;
use super::context::VkContext;
use super::swapchain::DEPTH_FORMAT;
use super::taa::FrameView;

pub(super) const NORMAL_FORMAT: Format = Format::R8G8B8A8_UNORM;

/// Signed and float, unlike the normal target: a motion vector is a UV *delta*,
/// so it is negative half the time, and a UNORM encoding would need a bias that
/// costs precision exactly where the vectors are smallest and matter most.
pub(super) const VELOCITY_FORMAT: Format = Format::R16G16_SFLOAT;

/// The per-frame camera block. Shared with the SSAO passes, which reconstruct
/// view-space positions from the same depth this pass wrote and so must use the
/// same — jittered — projection to do it.
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
pub(super) struct FrameUbo {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    jitter: [f32; 4],
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct PrepassPush {
    /// First object row of this instanced run; the shader adds `gl_InstanceIndex`.
    object_base: u32,
}

pub struct GeometryPrepass {
    pub(super) render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    uniform_allocator: SubbufferAllocator,
}

impl GeometryPrepass {
    pub fn new(ctx: &VkContext) -> Self {
        let device = &ctx.device;
        let render_pass = build_render_pass(device);
        let pipeline = build_pipeline(device, &render_pass);
        let uniform_allocator = SubbufferAllocator::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        Self {
            render_pass,
            pipeline,
            uniform_allocator,
        }
    }

    /// Upload the camera block once for every pass in the frame that reads it.
    pub(super) fn begin_frame(&self, view: &FrameView) -> Subbuffer<FrameUbo> {
        let frame = self.uniform_allocator.allocate_sized::<FrameUbo>().unwrap();
        *frame.write().unwrap() = FrameUbo {
            view: view.view.to_cols_array_2d(),
            proj: view.proj.to_cols_array_2d(),
            inv_proj: view.proj.inverse().to_cols_array_2d(),
            prev_view_proj: view.prev_view_proj.to_cols_array_2d(),
            jitter: [view.jitter.x, view.jitter.y, 0.0, 0.0],
        };
        frame
    }

    /// The set-1 per-object descriptor set this pass binds.
    pub(super) fn build_object_set(
        &self,
        ctx: &VkContext,
        objects: &Subbuffer<[super::forward::GpuObject]>,
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[1].clone(),
            [WriteDescriptorSet::buffer(0, objects.clone())],
            [],
        )
        .unwrap()
    }

    pub(super) fn record(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        renderer: &VulkanRenderer,
        draws: DrawList<'_>,
        extent: [u32; 2],
        frame: Subbuffer<FrameUbo>,
        object_set: Arc<DescriptorSet>,
    ) {
        let frame_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer(0, frame)],
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
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                vec![frame_set, object_set],
            )
            .unwrap();

        // One instanced draw per (mesh, material) run, matching the forward pass.
        // The model and normal matrices come from the shared object buffer, so
        // this pass no longer recomputes an inverse-transpose per item.
        for run in draws.runs() {
            let item = draws.item(run.start);
            let Some(mesh) = renderer.meshes.get(item.mesh.0 as usize) else {
                continue;
            };
            let push = PrepassPush {
                object_base: run.start as u32,
            };
            builder
                .push_constants(self.pipeline.layout().clone(), 0, push)
                .unwrap()
                .bind_vertex_buffers(0, mesh.vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(mesh.index_buffer.clone())
                .unwrap();
            unsafe {
                builder
                    .draw_indexed(mesh.index_count, run.len() as u32, 0, 0, 0)
                    .unwrap()
            };
        }
    }
}

fn build_render_pass(device: &Arc<Device>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            normal:   { format: NORMAL_FORMAT,   samples: 1, load_op: Clear, store_op: Store },
            velocity: { format: VELOCITY_FORMAT, samples: 1, load_op: Clear, store_op: Store },
            depth:    { format: DEPTH_FORMAT,    samples: 1, load_op: Clear, store_op: Store },
        },
        pass: { color: [normal, velocity], depth_stencil: {depth}}
    )
    .unwrap()
}

fn build_pipeline(device: &Arc<Device>, render_pass: &Arc<RenderPass>) -> Arc<GraphicsPipeline> {
    let vs = prepass_vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = prepass_fs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let vertex_input_state = Vertex::per_vertex().definition(&vs).unwrap();
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
    )
    .unwrap()
}

mod prepass_vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/prepass.vert" }
}
mod prepass_fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/prepass.frag" }
}
