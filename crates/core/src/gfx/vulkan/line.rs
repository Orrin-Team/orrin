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

use crate::scene::DebugLine;

use super::MSAA_SAMPLES;
use super::context::VkContext;
use super::convert::vk_samples;
use super::forward::ForwardTargets;
use super::record::{Arena, Recorder};
use super::taa::FrameView;
use vulkano::buffer::allocator::SubbufferAllocatorCreateInfo;
use vulkano::buffer::{BufferContents, BufferUsage};
use vulkano::image::SampleCount;
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::{Vertex as VertexTrait, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};

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
    /// Indexed `[msaa][subsurface]`, matching the four shapes of
    /// [`ForwardTargets`]: two sample counts, each with and without the
    /// diffusible colour target.
    pipelines: [[Arc<GraphicsPipeline>; 2]; 2],
    subbuffer_allocator: Arena,
}

impl LinePass {
    /// Build one line pipeline per forward target shape.
    pub fn new(ctx: &VkContext, targets: &ForwardTargets) -> Self {
        let subbuffer_allocator = Arena::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::VERTEX_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        Self {
            pipelines: [
                [
                    build_pipeline(ctx, &targets.single, SampleCount::Sample1),
                    build_pipeline(ctx, &targets.single_subsurface, SampleCount::Sample1),
                ],
                [
                    build_pipeline(ctx, &targets.multisampled, vk_samples(MSAA_SAMPLES)),
                    build_pipeline(
                        ctx,
                        &targets.multisampled_subsurface,
                        vk_samples(MSAA_SAMPLES),
                    ),
                ],
            ],
            subbuffer_allocator,
        }
    }

    /// Record this frame's lines into `builder`. Must be called *inside* the
    /// forward render pass, after the scene geometry.
    pub fn record(
        &self,
        builder: &mut Recorder,
        lines: &[DebugLine],
        view: &FrameView,
        extent: [u32; 2],
        subsurface: bool,
        msaa: bool,
    ) {
        if lines.is_empty() {
            return;
        }

        // Whichever shape the executor opened the rendering instance with.
        let pipeline = &self.pipelines[msaa as usize][subsurface as usize];

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
            .allocate_slice::<LineVertex>(vertices.len() as u64);
        buffer.write().unwrap().copy_from_slice(&vertices);

        // The frame's jittered view-projection, so a line lands on the same
        // subpixel as the geometry it is drawn against.
        let view_proj = view.view_proj.to_cols_array_2d();

        builder
            .set_viewport(
                0,
                &[Viewport {
                    offset: [0.0, 0.0],
                    extent: [extent[0] as f32, extent[1] as f32],
                    depth_range: 0.0..=1.0,
                }],
            )
            .bind_pipeline_graphics(pipeline)
            .push_constants(pipeline.layout(), 0, &view_proj)
            .bind_vertex_buffers(0, buffer);

        let vertex_count = vertices.len() as u32;

        // SAFETY: the bound pipeline and vertex buffer cover [0, vertex_count); no
        // index buffer or instancing is used.
        builder.draw(vertex_count, 1, 0, 0);
    }
}

fn build_pipeline(
    ctx: &VkContext,
    target: &PipelineRenderingCreateInfo,
    samples: SampleCount,
) -> Arc<GraphicsPipeline> {
    let device = &ctx.device;
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

    GraphicsPipeline::new(
        device.clone(),
        ctx.pipeline_cache(),
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState {
                topology: PrimitiveTopology::LineList,
                ..Default::default()
            }),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            // Passed in rather than read back off a render pass object: a
            // description carries formats, not a sample count, and a pipeline
            // whose count disagrees with the attachments it is used with is
            // invalid. The two travel together from `ForwardTargets` for that
            // reason.
            multisample_state: Some(MultisampleState {
                rasterization_samples: samples,
                ..Default::default()
            }),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState {
                    write_enable: false,
                    compare_op: CompareOp::Less,
                }),
                ..Default::default()
            }),
            // Masked past the first attachment: this shader declares one output,
            // and the subsurface target shape has two. See `co_tenant_blend_states`.
            color_blend_state: Some(super::forward::co_tenant_blend_states(target)),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(target.clone().into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
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
