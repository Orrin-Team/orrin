//! Screen-space refraction: what a [`BlendMode::Transmissive`] material draws
//! with, as three graph nodes.
//!
//! [`BlendMode::Transmissive`]: crate::gfx::BlendMode::Transmissive
//!
//! Why this is a queue of its own rather than a flag on the blended one is the
//! whole design. Weighted-blended transparency's correctness argument is that
//! its two targets commute, so nothing may sort it and no surface may depend on
//! what any other surface wrote. A refracting surface breaks the second half by
//! construction: it *reads* the frame behind it. Two panes over the same pixel
//! are then not interchangeable — the far one must resolve before the near one
//! can stand in front of it — so this list is depth-ordered, and the transparent
//! one still must not be.
//!
//! Three nodes:
//!
//! - `refraction_scene` reduces the lit frame into a mip pyramid, so a rough
//!   surface can refract a cone rather than a texel. It shares
//!   `shaders/color_pyramid.comp` with the reflection trace, at full resolution
//!   rather than half: a stochastic ray is denoised temporally afterwards and a
//!   clear pane of glass is not, so its background has to be as sharp as the
//!   frame recorded it.
//! - `refraction_draw` rasterises the queue back to front into a premultiplied
//!   target, sampling that pyramid through set 5.
//! - `refraction_composite` puts that target over the lit frame.
//!
//! Two things it shares with [`OitPass`](super::oit::OitPass), for the same
//! reasons documented there: it depth-tests against the *prepass* depth, which
//! is why `frame::declare` counts refraction among the things keeping that pass
//! alive, and it draws with descriptor sets 0 through 4 built once for the
//! forward pass. Its layout is not the forward pass's — it has a sixth set — but
//! the first five are identical binding for binding, which is all Vulkan's set
//! compatibility rule asks.

use std::sync::Arc;

use vulkano::buffer::BufferContents;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::image::sampler::{
    LOD_CLAMP_NONE, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
};
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

/// Levels in the scene pyramid a refractive surface samples its background out
/// of. Seven is what one 64x64 tile reduces to in the one dispatch
/// `color_pyramid.comp` is obliged to do it in.
///
/// Keep in sync with `REFRACTION_MIPS` in `shaders/shading.glsl`, which turns a
/// perceptual roughness into a level of this chain.
pub(super) const SCENE_LEVELS: u32 = 7;

/// Where `refraction.frag` declares the scene pyramid — one past the five sets
/// it shares with the forward pass.
const SCENE_SET: usize = 5;

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct PyramidPush {
    extent: [i32; 2],
    levels: i32,
}

pub struct RefractionPass {
    pipeline: Arc<GraphicsPipeline>,
    pyramid_pipeline: Arc<ComputePipeline>,
    composite_pipeline: Arc<ComputePipeline>,
    /// Linear, because level 0 of the pyramid is a copy of the frame and one
    /// bilinear fetch at a texel centre is exactly that.
    linear_clamp: Arc<Sampler>,
    /// Linear across levels too, because roughness picks a fractional one: a
    /// curved surface whose roughness varies would otherwise step between blurs,
    /// and the step reads as a ring.
    linear_mip: Arc<Sampler>,
    /// Nearest and clamped: the composite reads both inputs at exactly its own
    /// pixel, so there is nothing to filter.
    nearest_clamp: Arc<Sampler>,
}

impl RefractionPass {
    /// `forward_layout` is the forward pipeline's layout. Not shared outright,
    /// as the transparency pass shares it — this pass needs a sixth set — but its
    /// first five set layouts are lifted from it verbatim; see `build_pipeline`.
    pub fn new(ctx: &VkContext, forward_layout: &Arc<PipelineLayout>) -> Self {
        let device = &ctx.device;
        let pipeline = build_pipeline(ctx, forward_layout);

        let clamp = |info: SamplerCreateInfo| {
            Sampler::new(
                device.clone(),
                SamplerCreateInfo {
                    address_mode: [SamplerAddressMode::ClampToEdge; 3],
                    ..info
                },
            )
            .unwrap()
        };

        Self {
            pipeline,
            pyramid_pipeline: build_compute(ctx, pyramid_cs::load(device.clone()).unwrap()),
            composite_pipeline: build_compute(ctx, composite_cs::load(device.clone()).unwrap()),
            linear_clamp: clamp(SamplerCreateInfo::simple_repeat_linear_no_mipmap()),
            linear_mip: clamp(SamplerCreateInfo {
                mipmap_mode: SamplerMipmapMode::Linear,
                lod: 0.0..=LOD_CLAMP_NONE,
                ..SamplerCreateInfo::simple_repeat_linear()
            }),
            nearest_clamp: clamp(SamplerCreateInfo::default()),
        }
    }

    /// Reduce the lit frame into the pyramid a refractive surface samples its
    /// background out of.
    pub(super) fn record_pyramid(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        source: Arc<ImageView>,
        mips: &[Arc<ImageView>],
    ) {
        let levels = mips.len();
        let extent = mips[0].image().extent();
        // The shader's array is sized at `SCENE_LEVELS` whatever the frame's
        // extent produced, so a short chain repeats its last view into the
        // unused slots rather than leaving a descriptor unwritten; `push.levels`
        // is what stops those slots from being stored to.
        let bound = (0..SCENE_LEVELS as usize).map(|level| mips[level.min(levels - 1)].clone());

        let set = ctx.descriptor_sets.get(
            &ctx.descriptor_set_allocator,
            &self.pyramid_pipeline.layout().set_layouts()[0],
            [
                WriteDescriptorSet::image_view_sampler(0, source, self.linear_clamp.clone()),
                WriteDescriptorSet::image_view_array(1, 0, bound),
            ],
        );

        builder
            .bind_pipeline_compute(&self.pyramid_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pyramid_pipeline.layout(),
                0,
                &[set],
            )
            .push_constants(
                self.pyramid_pipeline.layout(),
                0,
                &PyramidPush {
                    extent: [extent[0] as i32, extent[1] as i32],
                    levels: levels as i32,
                },
            );

        // SAFETY: one workgroup owns a 64x64 tile of level 0, and every store
        // the shader makes is bounds-checked against `imageSize` and against
        // `push.levels`. The descriptors bound above match the shader's layout,
        // and the graph declared every resource this pass touches.
        builder.dispatch([extent[0].div_ceil(64), extent[1].div_ceil(64), 1]);
    }

    /// Draw the refractive queue, back to front, into the premultiplied target.
    ///
    /// Recorded inside this pass's own render pass. `sets` are the forward
    /// pass's five; the sixth, carrying the two views of the frame behind the
    /// glass, is built here because both name graph-owned images the executor
    /// only resolves at record time.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record(
        &self,
        builder: &mut Recorder,
        renderer: &PassCtx<'_>,
        draws: DrawList<'_>,
        sets: &ForwardSets,
        blur: Arc<ImageView>,
        sharp: Arc<ImageView>,
        view: &FrameView,
        extent: [u32; 2],
        object_base: u32,
    ) {
        let scene_set = renderer.ctx.descriptor_sets.get(
            &renderer.ctx.descriptor_set_allocator,
            &self.pipeline.layout().set_layouts()[SCENE_SET],
            [
                WriteDescriptorSet::image_view_sampler(0, blur, self.linear_mip.clone()),
                // The frame itself, for the smooth end of the roughness range —
                // no mips to step between, so the plain linear sampler.
                WriteDescriptorSet::image_view_sampler(1, sharp, self.linear_clamp.clone()),
            ],
        );

        let mut bound = sets.as_vec();
        bound.push(scene_set);

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
                &bound,
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
            // The refractive rows follow the ones before them in the shared object
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

    /// Put what the draw pass gathered over the lit frame.
    pub(super) fn record_composite(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        scene: Arc<ImageView>,
        accum: Arc<ImageView>,
        target: Arc<ImageView>,
    ) {
        let set = ctx.descriptor_sets.get(
            &ctx.descriptor_set_allocator,
            &self.composite_pipeline.layout().set_layouts()[0],
            [
                WriteDescriptorSet::image_view_sampler(0, scene, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(1, accum, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view(2, target.clone()),
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

fn build_pipeline(ctx: &VkContext, forward_layout: &Arc<PipelineLayout>) -> Arc<GraphicsPipeline> {
    let device = &ctx.device;
    // The forward pass's vertex shader, unchanged: a refractive surface is the
    // same geometry read from the same per-object rows, and `shading.glsl` reads
    // the same varyings from it.
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

    // Sets 0 through 4 are *the forward pipeline's own objects*, not a second
    // set derived from the same declarations. Deriving them would produce
    // layouts that match binding for binding and still fail to bind: set 3 holds
    // the shadow comparison sampler immutably — part of the layout rather than
    // written into a descriptor set, because MoltenVK cannot accept a written
    // one — and Vulkan compares immutable samplers by identity, so a second
    // `Sampler::new` of the identical description is a different sampler and
    // therefore an incompatible set. Lifting the objects is what makes the five
    // sets the executor already built bind to this pipeline at all.
    let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
    layout_info
        .set_layouts
        .get(SCENE_SET)
        .expect("refraction.frag must declare the scene pyramid in set 5");

    let mut create_info = layout_info
        .into_pipeline_layout_create_info(device.clone())
        .unwrap();
    create_info.set_layouts.splice(
        ..SCENE_SET,
        forward_layout.set_layouts()[..SCENE_SET].iter().cloned(),
    );

    let layout = PipelineLayout::new(device.clone(), create_info).unwrap();

    GraphicsPipeline::new(
        device.clone(),
        ctx.pipeline_cache(),
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState {
                // Front faces only, unlike the transparency accumulation, which
                // draws both. There the two faces of a glass sphere are two
                // surfaces that each scatter light; here the material's
                // *thickness* already stands for the far side, so drawing the
                // back face would refract the same volume a second time and
                // composite it over the first.
                cull_mode: CullMode::Back,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: Some(DepthStencilState {
                // Tested against the opaque scene, never written. Two refractive
                // surfaces are ordered by the back-to-front sort instead, which
                // is what the premultiplied blend below expects — a depth write
                // would let the nearer of two glass panes reject the farther
                // before it had been composited under it.
                depth: Some(DepthState {
                    write_enable: false,
                    compare_op: CompareOp::Less,
                }),
                ..Default::default()
            }),
            color_blend_state: Some(ColorBlendState {
                attachments: vec![ColorBlendAttachmentState {
                    // Premultiplied `over`. The fragment already composited its
                    // own background, so what accumulates here is radiance that
                    // has replaced what was behind it, carrying the coverage
                    // that says how much of the pixel it replaced.
                    blend: Some(AttachmentBlend {
                        src_color_blend_factor: BlendFactor::One,
                        dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                        color_blend_op: BlendOp::Add,
                        src_alpha_blend_factor: BlendFactor::One,
                        dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
                        alpha_blend_op: BlendOp::Add,
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(
                rendering::pipeline_info(
                    &[vk_format(HDR_WIDE_FORMAT)],
                    Some(vk_format(DEPTH_FORMAT)),
                )
                .into(),
            ),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

fn build_compute(
    ctx: &VkContext,
    module: Arc<vulkano::shader::ShaderModule>,
) -> Arc<ComputePipeline> {
    let device = &ctx.device;
    let stage = PipelineShaderStageCreateInfo::new(module.entry_point("main").unwrap());
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

/// The format the accumulation target is created with. Float and wide because
/// what it holds is premultiplied HDR radiance, exactly as the frame it will be
/// composited over holds — and premultiplied means the coverage it was
/// multiplied by has to survive in alpha for `refraction_composite` to put the
/// scene back underneath it. That is what keeps this off the packed colour
/// format the rest of the chain moved to.
pub(super) const ACCUM_FORMAT: crate::gfx::format::Format = HDR_WIDE_FORMAT;

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/refraction.frag",
        include: ["shaders"],
    }
}

/// The same reduction the reflection trace builds its source with, compiled a
/// second time because either pass can run with the other switched off. One
/// module and two pipeline objects, rather than a dependency between two things
/// that are structurally independent.
mod pyramid_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/color_pyramid.comp" }
}

mod composite_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/refraction_composite.comp" }
}
