//! Debug-line overlay pass.
//!
//! Draws the engine's per-frame [`DebugLine`] buffer as GPU line primitives into
//! the forward (HDR) render pass, right after the scene geometry. Sharing that
//! subpass is deliberate: the pass reuses the existing depth buffer, so debug
//! lines are correctly *occluded* by scene geometry, and it reuses the camera
//! view-projection. The one accepted cost is colour: the forward target is HDR
//! and tonemapped downstream, so a line's on-screen colour drifts from the exact
//! RGBA the script requested. Drawing post-tonemap would fix the colour but lose
//! depth occlusion — see the design notes on the issue.

use std::sync::Arc;

use vulkano::buffer::{BufferContents, BufferUsage};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::device::Device;
use vulkano::memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::vertex_input::{Vertex as VertexTrait, VertexDefinition};
use vulkano::pipeline::{DynamicState, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::render_pass::{RenderPass, Subpass};
use crate::scene::{Camera, DebugLine};

/// One endpoint of a debug line: world-space position + RGBA colour. Two of
/// these make a segment, drawn with `PrimitiveTopology::LineList`.
#[derive(BufferContents, VertexTrait, Clone, Copy)]
#[repr(C)]
pub struct LineVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}

pub struct LinePass {
    pipeline: Arc<GraphicsPipeline>,
    subbuffer_allocator: SubbufferAllocator,
}

impl LinePass {
    /// Build the line pipeline against the forward render pass's subpass 0.
    pub fn new(
        device: &Arc<Device>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        render_pass: &Arc<RenderPass>,
    ) -> Self {
        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let vertex_input_state = LineVertex::per_vertex().definition(&vs).unwrap();

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

        let subbuffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::VERTEX_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState  {
                    topology: PrimitiveTopology::LineList,
                    ..Default::default()
                }),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState {
                    rasterization_samples: vulkano::image::SampleCount::Sample4,
                    ..Default::default()
                }),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState {
                        write_enable: false,
                        compare_op: CompareOp::Less,
                        ..Default::default()
                    }),
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
            .unwrap();

        Self {
            pipeline,
            subbuffer_allocator,
        }
    }

    /// Record this frame's lines into `builder`. Must be called *inside* the
    /// forward render pass, after the scene geometry.
    pub fn record(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        lines: &[DebugLine],
        camera: &Camera,
        extent: [u32; 2],
    ) {
        if lines.is_empty() {
            return;
        }

        // Flatten each segment into its two endpoints for `LineList` topology.
        let mut vertices: Vec<LineVertex> = Vec::with_capacity(lines.len() * 2);
        for line in lines {
            vertices.push(LineVertex {
                position: line.from.to_array(),
                color: line.color,
            });
            vertices.push(LineVertex {
                position: line.to.to_array(),
                color: line.color,
            });
        }

        let buffer = self
            .subbuffer_allocator
            .allocate_slice::<LineVertex>(vertices.len() as u64)
            .unwrap();
        buffer.write().unwrap().copy_from_slice(&vertices);

        let aspect = extent[0] as f32 / extent[1] as f32;
        let view_proj = camera.view_projection(aspect).to_cols_array_2d();

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
            .push_constants(self.pipeline.layout().clone(), 0, view_proj)
            .unwrap()
            .bind_vertex_buffers(0, buffer)
            .unwrap();

        let vertex_count = vertices.len() as u32;

        // SAFETY: the bound pipeline and vertex buffer cover [0, vertex_count); no
        // index buffer or instancing is used.
        unsafe {
            builder.draw(vertex_count, 1, 0, 0).unwrap();
        }
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/line.vert",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/line.frag",
    }
}