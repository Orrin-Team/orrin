//! Transparency: weighted-blended order-independent transparency
//! (McGuire & Bavoil 2013), as two graph nodes.
//!
//! The accumulation pass rasterises every blended surface into two targets whose
//! blend equations commute, so no sort exists to get wrong and nothing pops when
//! two surfaces cross. The composite divides one by its own coverage and mixes
//! it over the lit frame. `shaders/oit.frag` documents the algebra.
//!
//! Two things about the *pass* are worth knowing here.
//!
//! It draws with the forward pass's own pipeline layout. `oit.frag` includes the
//! same `shading.glsl` `forward.frag` does, so its five descriptor sets are the
//! forward pass's five, binding for binding — sharing the layout is what lets
//! the executor build them once and bind them to both, and makes it impossible
//! for the two to drift into shading the same material differently.
//!
//! And it depth-tests against the *prepass* depth rather than the forward pass's
//! multisampled one, which is why `frame::declare` counts transparency among the
//! things that keep the prepass alive. The forward pass's depth is a memoryless
//! attachment — tile-only on Apple hardware, undefined the moment its render
//! pass ends — so a second pass cannot read it without giving up that property
//! and the DRAM it saves. The prepass depth is already resident because SSAO,
//! TAA and the rest sample it.

use std::sync::Arc;

use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::image::sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex as _, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};

use crate::gfx::{DrawList, PositionVertex, SurfaceVertex};

use super::PassCtx;
use super::context::VkContext;
use super::convert::vk_format;
use super::forward::ForwardSets;
use super::hdr::HDR_WIDE_FORMAT;
use super::record::Recorder;
use super::rendering;
use super::swapchain::DEPTH_FORMAT;
use super::taa::FrameView;

/// Side of the composite's compute workgroup.
const TILE: u32 = 8;

/// `rgb` = the sum of weighted premultiplied radiance, `a` = the sum of weighted
/// coverage. Float and wide because both sums are unbounded above: the weight in
/// `oit.frag` is clamped per fragment, not per pixel.
// The wide format: this target carries *weighted* radiance, and the weight is
// in alpha. The packed colour format has no alpha channel, so accumulating into
// it would throw away the divisor the composite needs.
pub(super) const ACCUM_FORMAT: crate::gfx::format::Format = HDR_WIDE_FORMAT;

/// Transmittance: the running product of `1 - alpha`, one channel and eight bits
/// of it. The paper's own recommendation — the value is a fraction in `[0, 1]`
/// that a composite multiplies by, and a quarter of a byte per pixel buys
/// nothing a viewer can see.
pub(super) const REVEAL_FORMAT: crate::gfx::format::Format = crate::gfx::format::Format::R8Unorm;

pub struct OitPass {
    pipeline: Arc<GraphicsPipeline>,
    composite_pipeline: Arc<ComputePipeline>,
    /// Nearest and clamped: the composite reads all three of its inputs at
    /// exactly its own pixel, so there is nothing to filter.
    nearest_clamp: Arc<Sampler>,
}

impl OitPass {
    /// `forward_layout` is the forward pipeline's layout, shared rather than
    /// derived — see the module docs.
    pub fn new(ctx: &VkContext, forward_layout: &Arc<PipelineLayout>) -> Self {
        let device = &ctx.device;
        let pipeline = build_pipeline(ctx, forward_layout);
        let composite_pipeline = build_composite_pipeline(ctx);
        let nearest_clamp = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::default()
            },
        )
        .unwrap();

        Self {
            pipeline,
            composite_pipeline,
            nearest_clamp,
        }
    }

    /// Accumulate every blended surface. Recorded inside this pass's own render
    /// pass; `sets` are the forward pass's, which this pipeline's layout is.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record(
        &self,
        builder: &mut Recorder,
        renderer: &PassCtx<'_>,
        draws: DrawList<'_>,
        sets: &ForwardSets,
        view: &FrameView,
        extent: [u32; 2],
        object_base: u32,
    ) {
        builder
            .set_viewport(
                0,
                &[Viewport {
                    offset: [0.0, 0.0],
                    extent: [extent[0] as f32, extent[1] as f32],
                    depth_range: 0.0..=1.0,
                }],
            )
            .bind_pipeline_graphics(&self.pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout(),
                0,
                &sets.as_vec(),
            );

        // The arena is bound for the pass, not per draw: a mesh is a span into
        // it. The matrix is pushed once for the same reason — what used to vary
        // per run, the material and the first object row, travels with the
        // instance now.
        renderer.arena.bind(builder);
        builder.push_constants(
            self.pipeline.layout(),
            0,
            &super::forward::PushConstants::new(view.view_proj),
        );

        for run in draws.runs() {
            let item = draws.item(run.start);
            let Some(mesh) = renderer.meshes.get(item.mesh.0 as usize) else {
                continue;
            };
            // The transparent rows follow the ones before them in the shared object
            // buffer, so a run's `firstInstance` is its start plus where that
            // block began.
            builder.draw_indexed(
                mesh.span.index_count,
                run.len() as u32,
                mesh.span.first_index,
                mesh.span.vertex_offset,
                object_base + run.start as u32,
            );
        }
    }

    /// Put what the accumulation gathered over the lit frame.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_composite(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        scene: Arc<ImageView>,
        accum: Arc<ImageView>,
        reveal: Arc<ImageView>,
        target: Arc<ImageView>,
    ) {
        let set = ctx.descriptor_sets.get(
            &ctx.descriptor_set_allocator,
            &self.composite_pipeline.layout().set_layouts()[0],
            [
                WriteDescriptorSet::image_view_sampler(0, scene, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(1, accum, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(2, reveal, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view(3, target.clone()),
            ],
        );

        builder
            .bind_pipeline_compute(&self.composite_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.composite_pipeline.layout(),
                0,
                &[set],
            );

        let extent = target.image().extent();
        // SAFETY: the dispatch covers exactly `extent`, and the shader discards
        // invocations past `imageSize`, so nothing writes outside the image. The
        // descriptors bound above match the shader's layout, and the graph
        // declared every resource this pass touches, so its barriers precede it.
        builder.dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1]);
    }
}

fn build_pipeline(ctx: &VkContext, layout: &Arc<PipelineLayout>) -> Arc<GraphicsPipeline> {
    let device = &ctx.device;
    // The forward pass's vertex shader, unchanged: a blended surface is the same
    // geometry with the same per-object rows, and `shading.glsl` reads the same
    // varyings from it.
    let vs = super::forward::vertex_shader(device);
    let fs = fs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();

    let vertex_input_state = [PositionVertex::per_vertex(), SurfaceVertex::per_vertex()]
        .definition(&vs)
        .unwrap();
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];
    GraphicsPipeline::new(
        device.clone(),
        ctx.pipeline_cache(),
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState {
                // Both faces, unlike every other geometry pass. A closed opaque
                // solid hides its own back faces; a transparent one does not,
                // and culling them is what makes a glass sphere look like a
                // decal of one.
                cull_mode: CullMode::None,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: Some(DepthStencilState {
                // Tested against the opaque scene, never written: two blended
                // surfaces must not occlude each other, or the order they were
                // submitted in would decide which survived — the exact
                // dependence this algorithm exists to remove.
                depth: Some(DepthState {
                    write_enable: false,
                    compare_op: CompareOp::Less,
                }),
                ..Default::default()
            }),
            color_blend_state: Some(ColorBlendState {
                attachments: vec![
                    // accum: plain addition. Commutative, which is the property
                    // that makes the draw order irrelevant.
                    ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend {
                            src_color_blend_factor: BlendFactor::One,
                            dst_color_blend_factor: BlendFactor::One,
                            color_blend_op: BlendOp::Add,
                            src_alpha_blend_factor: BlendFactor::One,
                            dst_alpha_blend_factor: BlendFactor::One,
                            alpha_blend_op: BlendOp::Add,
                        }),
                        ..Default::default()
                    },
                    // reveal: `dst = dst * (1 - src)`. Multiplication, which is
                    // commutative for the same reason — and why the target is
                    // cleared to 1.0 rather than to 0.
                    ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend {
                            src_color_blend_factor: BlendFactor::Zero,
                            dst_color_blend_factor: BlendFactor::OneMinusSrcColor,
                            color_blend_op: BlendOp::Add,
                            src_alpha_blend_factor: BlendFactor::Zero,
                            dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
                            alpha_blend_op: BlendOp::Add,
                        }),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(
                rendering::pipeline_info(
                    &[vk_format(ACCUM_FORMAT), vk_format(REVEAL_FORMAT)],
                    Some(vk_format(DEPTH_FORMAT)),
                )
                .into(),
            ),
            ..GraphicsPipelineCreateInfo::layout(layout.clone())
        },
    )
    .unwrap()
}

fn build_composite_pipeline(ctx: &VkContext) -> Arc<ComputePipeline> {
    let device = &ctx.device;
    let stage = PipelineShaderStageCreateInfo::new(
        composite_cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap(),
    );
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    ComputePipeline::new(
        device.clone(),
        ctx.pipeline_cache(),
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/oit.frag",
        include: ["shaders"],
    }
}

mod composite_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/oit_composite.comp" }
}
