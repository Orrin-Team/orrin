//! The transform gizmo's pass: the editor's strokes, over the finished frame.
//!
//! A node of its own after the tonemap, writing the swapchain image, rather
//! than lines in the forward pass the way debug lines are. Drawn there, a
//! handle would be depth-tested against the scene and lose to whatever stands
//! in front of the selection, and its colour would go through exposure and the
//! tonemap curve. Here it is neither, and it sits under the editor's own UI
//! because the overlay is the node after it.
//!
//! The strokes arrive already in screen space (see `editor::gizmo`), which is
//! what keeps the gizmo one size on screen at any distance, and is also what
//! the editor hit-tests the pointer against — the shape grabbed is the shape
//! drawn.

use std::sync::Arc;

use vulkano::buffer::allocator::SubbufferAllocatorCreateInfo;
use vulkano::buffer::{BufferContents, BufferUsage};
use vulkano::format::{Format, NumericFormat};
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};

use super::context::VkContext;
use super::record::{Arena, Recorder};
use super::rendering;

/// One corner of a stroke's quad, in physical pixels from the top left.
#[derive(BufferContents, Vertex, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct GizmoVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
    /// Distance across the stroke from its centre line, and its half width,
    /// in pixels. What the fragment shader antialiases the edge from.
    #[format(R32G32_SFLOAT)]
    pub edge: [f32; 2],
    /// sRGB, unpremultiplied.
    #[format(R8G8B8A8_UNORM)]
    pub color: [u8; 4],
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct GizmoPush {
    screen_size: [f32; 2],
    output_in_linear_colorspace: i32,
}

pub struct GizmoPass {
    pipeline: Arc<GraphicsPipeline>,
    linear_output: bool,
    vertices: Arena,
}

impl GizmoPass {
    pub fn new(ctx: &VkContext, format: Format) -> Self {
        let vertices = Arena::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::VERTEX_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        Self {
            pipeline: build_pipeline(ctx, format),
            linear_output: format.numeric_format_color() == Some(NumericFormat::SRGB),
            vertices,
        }
    }

    /// Draw `vertices`, a triangle list, into the swapchain image the executor
    /// opened. Nothing is recorded for an empty list — no selection, or a tool
    /// with no handles.
    pub fn record(&self, builder: &mut Recorder, vertices: &[GizmoVertex], extent: [u32; 2]) {
        if vertices.is_empty() {
            return;
        }
        let buffer = self
            .vertices
            .allocate_slice::<GizmoVertex>(vertices.len() as u64);
        buffer.write().unwrap().copy_from_slice(vertices);

        let size = [extent[0] as f32, extent[1] as f32];
        builder
            .set_viewport(
                0,
                &[Viewport {
                    offset: [0.0, 0.0],
                    extent: size,
                    depth_range: 0.0..=1.0,
                }],
            )
            .bind_pipeline_graphics(&self.pipeline)
            .push_constants(
                self.pipeline.layout(),
                0,
                &GizmoPush {
                    screen_size: size,
                    output_in_linear_colorspace: self.linear_output as i32,
                },
            )
            .bind_vertex_buffers(0, buffer);
        builder.draw(vertices.len() as u32, 1, 0, 0);
    }
}

fn build_pipeline(ctx: &VkContext, format: Format) -> Arc<GraphicsPipeline> {
    let device = &ctx.device;
    let vs = vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = fs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let vertex_input_state = GizmoVertex::per_vertex().definition(&vs).unwrap();
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
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                1,
                ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend::alpha()),
                    ..Default::default()
                },
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(rendering::pipeline_info(&[format], None).into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

mod vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/gizmo.vert" }
}

mod fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/gizmo.frag" }
}
