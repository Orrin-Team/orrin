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

use crate::gfx::rhi::{
    Binding, BindingResource, Blend, ComputePipelineDesc, CullMode, DepthCompare, DepthTest,
    Device, Encoder, GraphicsPipelineDesc, PipelineId, SamplerDesc, SamplerId, TextureViewId,
    Topology, Viewport,
};
use crate::gfx::shaders::ShaderId;
use crate::gfx::{DrawList, PositionVertex, SurfaceVertex};

use super::PassCtx;
use super::forward::ForwardSets;
use super::hdr::HDR_WIDE_FORMAT;
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
    pipeline: PipelineId,
    composite_pipeline: PipelineId,
    /// Nearest and clamped: the composite reads all three of its inputs at
    /// exactly its own pixel, so there is nothing to filter.
    nearest_clamp: SamplerId,
}

impl OitPass {
    /// `forward_layout` is the forward pipeline's layout, shared rather than
    /// derived — see the module docs.
    pub fn new(device: &Device, forward: PipelineId) -> Self {
        let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc {
            // The forward pass's vertex shader, unchanged: a blended surface is
            // the same geometry with the same per-object rows, and `shading.glsl`
            // reads the same varyings from it.
            vertex_streams: &[PositionVertex::STREAM, SurfaceVertex::STREAM],
            // Both faces, unlike every other geometry pass. A closed opaque solid
            // hides its own back faces; a transparent one does not, and culling
            // them is what makes a glass sphere look like a decal of one.
            cull: CullMode::None,
            // Tested against the opaque scene, never written: two blended
            // surfaces must not occlude each other, or the order they were
            // submitted in would decide which survived — the exact dependence
            // this algorithm exists to remove.
            depth_test: Some(DepthTest {
                write: false,
                compare: DepthCompare::Less,
            }),
            depth_format: Some(DEPTH_FORMAT),
            // accum: plain addition, commutative, which is what makes the draw
            // order irrelevant. reveal: `dst = dst * (1 - src)`, commutative for
            // the same reason — and why the target is cleared to 1.0 rather than
            // to 0.
            attachment_blends: &[Blend::Additive, Blend::Revealage],
            topology: Topology::Triangles,
            layout_from: Some(forward),
            layout_prefix_from: None,
            ..GraphicsPipelineDesc::fullscreen(
                ShaderId::ForwardVert,
                ShaderId::OitFrag,
                &[ACCUM_FORMAT, REVEAL_FORMAT],
            )
        });
        let composite_pipeline = device.create_compute_pipeline(&ComputePipelineDesc {
            shader: ShaderId::OitCompositeComp,
        });
        let nearest_clamp = device.create_sampler(SamplerDesc {
            linear: false,
            mipmap_linear: false,
            repeat: false,
            compare: false,
            anisotropy: None,
            max_lod: Some(0.0),
        });

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
        device: &Device,
        builder: &mut Encoder,
        renderer: &PassCtx<'_>,
        draws: DrawList<'_>,
        sets: &ForwardSets,
        view: &FrameView,
        extent: [u32; 2],
        object_base: u32,
    ) {
        let Device::Vulkan(driver) = device;
        builder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: [0.0, 1.0],
        });
        builder.bind_pipeline(self.pipeline);
        for (set, group) in sets.as_vec().into_iter().enumerate() {
            builder.bind_group(set as u32, driver.import_set(group));
        }

        // The arena is bound for the pass, not per draw: a mesh is a span into
        // it. The matrix is pushed once for the same reason — what used to vary
        // per run, the material and the first object row, travels with the
        // instance now.
        builder.bind_geometry(renderer.arena, false);
        let mut words = [0u32; 16];
        for (word, float) in words
            .iter_mut()
            .zip(view.view_proj.to_cols_array_2d().iter().flatten())
        {
            *word = float.to_bits();
        }
        builder.push_constants(&words);

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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_composite(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        scene: TextureViewId,
        accum: TextureViewId,
        reveal: TextureViewId,
        target: TextureViewId,
    ) {
        let sampled = |binding, view| Binding {
            binding,
            resource: BindingResource::SampledTexture {
                view,
                sampler: self.nearest_clamp,
            },
        };
        let set = device.create_bind_group(
            self.composite_pipeline,
            0,
            &[
                sampled(0, scene),
                sampled(1, accum),
                sampled(2, reveal),
                Binding {
                    binding: 3,
                    resource: BindingResource::StorageTexture(target),
                },
            ],
        );

        encoder.bind_pipeline(self.composite_pipeline);
        encoder.bind_group(0, set);

        let extent = device.view_extent(target);
        encoder.dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1]);
    }
}
