//! The selection outline: two graph nodes, after the tonemap and before the
//! gizmo.
//!
//! `outline_mask` draws every selected or hovered mesh, flat and with no depth
//! test, into a two-channel mask: red where a selected one covers, green where
//! the hovered one does. `outline` then reads the mask and paints a band just
//! outside each shape onto the finished frame. With no depth test the outline
//! shows through whatever stands in front of the selection, which is what
//! keeps a selected thing findable behind a wall.
//!
//! A mask of its own rather than the scene's stencil or depth: the outline is
//! the editor's and is drawn after the frame is finished, by which point the
//! scene's targets are resolved, aliased or gone, and a frame without the
//! editor declares neither node.

use std::sync::Arc;

use glam::Mat4;
use vulkano::buffer::BufferContents;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::format::{Format, NumericFormat};
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition, VertexInputState};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};

use super::context::VkContext;
use super::convert::vk_format;
use super::forward::GpuMesh;
use super::mesh::MeshArena;
use super::record::Recorder;
use super::rendering;
use crate::gfx::PositionVertex;
use crate::scene::MeshHandle;

/// Two channels: selected and hovered.
pub const OUTLINE_MASK_FORMAT: crate::gfx::format::Format = crate::gfx::format::Format::Rg8Unorm;

/// The selection's colour, and the hover's, in sRGB.
const SELECTED: [f32; 4] = [1.0, 0.62, 0.16, 1.0];
const HOVERED: [f32; 4] = [0.55, 0.78, 1.0, 0.85];

/// One mesh to outline, and how.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutlineItem {
    pub mesh: MeshHandle,
    pub model: Mat4,
    /// Selected rather than only under the pointer.
    pub selected: bool,
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct MaskPush {
    mvp: [[f32; 4]; 4],
    value: [f32; 4],
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct OutlinePush {
    selected: [f32; 4],
    hovered: [f32; 4],
    output_in_linear_colorspace: i32,
}

pub struct OutlinePass {
    mask: Arc<GraphicsPipeline>,
    outline: Arc<GraphicsPipeline>,
    sampler: Arc<Sampler>,
    linear_output: bool,
}

impl OutlinePass {
    pub fn new(ctx: &VkContext, swapchain_format: Format) -> Self {
        Self {
            mask: build_mask(ctx),
            outline: build_outline(ctx, swapchain_format),
            sampler: Sampler::new(ctx.device.clone(), SamplerCreateInfo::default()).unwrap(),
            linear_output: swapchain_format.numeric_format_color() == Some(NumericFormat::SRGB),
        }
    }

    /// Draw `items` into the mask the executor opened, cleared. `view_proj` is
    /// the unjittered one: this is drawn over the resolved frame.
    pub fn record_mask(
        &self,
        builder: &mut Recorder,
        items: &[OutlineItem],
        meshes: &[GpuMesh],
        arena: &MeshArena,
        view_proj: Mat4,
        extent: [u32; 2],
    ) {
        if items.is_empty() {
            return;
        }
        builder
            .set_viewport(0, &[viewport(extent)])
            .bind_pipeline_graphics(&self.mask);
        arena.bind_positions(builder);
        for item in items {
            let Some(mesh) = meshes.get(item.mesh.0 as usize) else {
                continue;
            };
            let value = if item.selected {
                [1.0, 0.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0, 0.0]
            };
            let push = MaskPush {
                mvp: (view_proj * item.model).to_cols_array_2d(),
                value,
            };
            builder.push_constants(self.mask.layout(), 0, &push);
            builder.draw_indexed(
                mesh.span.index_count,
                1,
                mesh.span.first_index,
                mesh.span.vertex_offset,
                0,
            );
        }
    }

    /// Paint the outline round the mask onto the swapchain image the executor
    /// opened. Nothing is drawn when nothing was outlined.
    pub fn record_outline(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        mask: Arc<ImageView>,
        extent: [u32; 2],
        any: bool,
    ) {
        if !any {
            return;
        }
        let set = ctx.descriptor_sets.get(
            &ctx.descriptor_set_allocator,
            &self.outline.layout().set_layouts()[0],
            [WriteDescriptorSet::image_view_sampler(
                0,
                mask,
                self.sampler.clone(),
            )],
        );
        builder
            .set_viewport(0, &[viewport(extent)])
            .bind_pipeline_graphics(&self.outline)
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.outline.layout(),
                0,
                &[set],
            )
            .push_constants(
                self.outline.layout(),
                0,
                &OutlinePush {
                    selected: SELECTED,
                    hovered: HOVERED,
                    output_in_linear_colorspace: self.linear_output as i32,
                },
            );
        builder.draw(3, 1, 0, 0);
    }
}

fn viewport(extent: [u32; 2]) -> Viewport {
    Viewport {
        offset: [0.0, 0.0],
        extent: [extent[0] as f32, extent[1] as f32],
        depth_range: 0.0..=1.0,
    }
}

fn layout_for(ctx: &VkContext, stages: &[PipelineShaderStageCreateInfo]) -> Arc<PipelineLayout> {
    PipelineLayout::new(
        ctx.device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(stages)
            .into_pipeline_layout_create_info(ctx.device.clone())
            .unwrap(),
    )
    .unwrap()
}

fn build_mask(ctx: &VkContext) -> Arc<GraphicsPipeline> {
    let vs = mask_vs::load(ctx.device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = mask_fs::load(ctx.device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let vertex_input_state = PositionVertex::per_vertex().definition(&vs).unwrap();
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];
    let layout = layout_for(ctx, &stages);
    // Max, so the hovered mesh drawn after a selected one it overlaps cannot
    // take the selected channel back to zero.
    let max = AttachmentBlend {
        src_color_blend_factor: BlendFactor::One,
        dst_color_blend_factor: BlendFactor::One,
        color_blend_op: BlendOp::Max,
        src_alpha_blend_factor: BlendFactor::One,
        dst_alpha_blend_factor: BlendFactor::One,
        alpha_blend_op: BlendOp::Max,
    };
    GraphicsPipeline::new(
        ctx.device.clone(),
        ctx.pipeline_cache(),
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            // Both faces: a mask of coverage, and a mesh seen from inside
            // still covers what it covers.
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                1,
                ColorBlendAttachmentState {
                    blend: Some(max),
                    ..Default::default()
                },
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(rendering::pipeline_info(&[vk_format(OUTLINE_MASK_FORMAT)], None).into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

fn build_outline(ctx: &VkContext, swapchain_format: Format) -> Arc<GraphicsPipeline> {
    let vs = fullscreen_vs::load(ctx.device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = outline_fs::load(ctx.device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];
    let layout = layout_for(ctx, &stages);
    GraphicsPipeline::new(
        ctx.device.clone(),
        ctx.pipeline_cache(),
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(VertexInputState::default()),
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
            subpass: Some(rendering::pipeline_info(&[swapchain_format], None).into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

mod mask_vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/outline_mask.vert" }
}

mod mask_fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/outline_mask.frag" }
}

mod fullscreen_vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/fullscreen.vert" }
}

mod outline_fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/outline.frag" }
}
