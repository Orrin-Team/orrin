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
//!
//! A transmissive material read through a surface shader draws with that
//! shader's twin of the draw pipeline (`surface`). Its set 5 is this pass's
//! two views plus the parameter buffer, so it gets a set of its own, built on
//! the first run that needs it each frame. The queue stays back to front: the
//! pipeline follows the runs, switching to a twin and back as the order asks,
//! rather than the runs being grouped by pipeline.

use crate::gfx::rhi::{
    Binding, BindingResource, Blend, ComputePipelineDesc, CullMode, DepthTest, Device, Encoder,
    GraphicsPipelineDesc, PipelineId, SamplerDesc, SamplerId, ShaderModuleId, ShaderRef,
    TextureViewId, Viewport,
};
use crate::gfx::shaders::ShaderId;
use crate::gfx::{DrawList, PositionVertex, SurfaceVertex};

use super::PassCtx;
use super::forward::ForwardSets;
use super::hdr::HDR_WIDE_FORMAT;
use super::surface::{Surfaces, Twin, params_group};
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

/// The refraction pipeline, built-in or a surface shader's twin of it, with
/// its first five sets lifted from `layout`.
fn pipeline_desc(fragment: ShaderRef, layout: PipelineId) -> GraphicsPipelineDesc<'static> {
    GraphicsPipelineDesc {
        // Front faces only, unlike the transparency accumulation, which draws
        // both. There the two faces of a glass sphere are two surfaces that
        // each scatter light; here the material's *thickness* already stands
        // for the far side, so drawing the back face would refract the same
        // volume a second time and composite it over the first.
        cull: CullMode::Back,
        // Tested against the opaque scene, never written. Two refractive
        // surfaces are ordered by the back-to-front sort instead, which is
        // what the premultiplied blend expects — a depth write would let the
        // nearer of two glass panes reject the farther before it had been
        // composited under it.
        depth_test: Some(DepthTest {
            write: false,
            compare: crate::gfx::depth::NEARER,
        }),
        depth_format: Some(DEPTH_FORMAT),
        vertex_streams: &[PositionVertex::STREAM, SurfaceVertex::STREAM],
        // Premultiplied `over`. The fragment already composited its own
        // background, so what accumulates here is radiance that has replaced
        // what was behind it, carrying the coverage that says how much of the
        // pixel it replaced.
        blend: Blend::Premultiplied,
        // Sets 0 through 4 are *the forward pipeline's own objects*, not a
        // second set derived from the same declarations. Deriving them would
        // produce layouts that match binding for binding and still fail to
        // bind: set 3 holds the shadow comparison sampler immutably, and
        // Vulkan compares immutable samplers by identity. Lifting the objects
        // is what makes the five sets the executor already built bind to this
        // pipeline at all.
        layout_prefix_from: Some((layout, SCENE_SET as u32)),
        // The forward pass's vertex shader, unchanged: a refractive surface is
        // the same geometry read from the same per-object rows.
        ..GraphicsPipelineDesc::fullscreen(ShaderId::ForwardVert, fragment, &[ACCUM_FORMAT])
    }
}

pub struct RefractionPass {
    pipeline: PipelineId,
    /// The same reduction the reflection trace builds its source with, built a
    /// second time because either pass can run with the other switched off. Two
    /// pipeline objects, rather than a dependency between two things that are
    /// structurally independent.
    pyramid_pipeline: PipelineId,
    composite_pipeline: PipelineId,
    /// The pyramid pass's own copy of the clamped linear sampler, on the RHI.
    pyramid_sampler: SamplerId,
    composite_sampler: SamplerId,
    /// Linear across levels too, because roughness picks a fractional one: a
    /// curved surface whose roughness varies would otherwise step between blurs,
    /// and the step reads as a ring.
    mip_sampler: SamplerId,
}

impl RefractionPass {
    /// `forward_layout` is the forward pipeline's layout. Not shared outright,
    /// as the transparency pass shares it — this pass needs a sixth set — but its
    /// first five set layouts are lifted from it verbatim; see `build_pipeline`.
    pub fn new(device: &Device, forward: PipelineId) -> Self {
        let pipeline = device
            .create_graphics_pipeline(&pipeline_desc(ShaderId::RefractionFrag.into(), forward));

        Self {
            pipeline,
            mip_sampler: device.create_sampler(SamplerDesc {
                linear: true,
                mipmap_linear: true,
                repeat: false,
                compare: false,
                anisotropy: None,
                max_lod: None,
            }),
            pyramid_pipeline: device.create_compute_pipeline(&ComputePipelineDesc {
                shader: ShaderId::ColorPyramidComp.into(),
            }),
            composite_pipeline: device.create_compute_pipeline(&ComputePipelineDesc {
                shader: ShaderId::RefractionCompositeComp.into(),
            }),
            pyramid_sampler: device.create_sampler(SamplerDesc {
                linear: true,
                mipmap_linear: false,
                repeat: false,
                compare: false,
                anisotropy: None,
                max_lod: Some(1.0),
            }),
            composite_sampler: device.create_sampler(SamplerDesc {
                linear: false,
                mipmap_linear: false,
                repeat: false,
                compare: false,
                anisotropy: None,
                max_lod: Some(0.0),
            }),
        }
    }

    /// A surface shader's twin of the draw pipeline. Its set 5 is this pass's
    /// two views plus the parameters, so it is lifted with the first five only
    /// and gets a set of its own at record time.
    pub(super) fn build_twin(
        &self,
        device: &Device,
        fragment: ShaderModuleId,
    ) -> Result<PipelineId, String> {
        device.try_create_graphics_pipeline(&pipeline_desc(fragment.into(), self.pipeline))
    }

    /// Reduce the lit frame into the pyramid a refractive surface samples its
    /// background out of.
    pub(super) fn record_pyramid(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        source: TextureViewId,
        mips: &[TextureViewId],
    ) {
        let levels = mips.len();
        let extent = device.view_extent(mips[0]);
        // The shader's array is sized at `SCENE_LEVELS` whatever the frame's
        // extent produced, so a short chain repeats its last view into the
        // unused slots rather than leaving a descriptor unwritten; `push.levels`
        // is what stops those slots from being stored to.
        let mut bindings = vec![Binding {
            binding: 0,
            resource: BindingResource::SampledTexture {
                view: source,
                sampler: self.pyramid_sampler,
            },
        }];
        bindings.extend((0..SCENE_LEVELS as usize).map(|level| Binding {
            binding: 1,
            resource: BindingResource::StorageTexture(mips[level.min(levels - 1)]),
        }));
        let set = device.create_bind_group(self.pyramid_pipeline, 0, &bindings);

        encoder.bind_pipeline(self.pyramid_pipeline);
        encoder.bind_group(0, set);
        encoder.push_constants(&[extent[0], extent[1], levels as u32]);

        // One workgroup owns a 64x64 tile of level 0, and every store the shader
        // makes is bounds-checked against `imageSize` and against `push.levels`.
        encoder.dispatch([extent[0].div_ceil(64), extent[1].div_ceil(64), 1]);
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
        device: &Device,
        builder: &mut Encoder,
        renderer: &PassCtx<'_>,
        draws: DrawList<'_>,
        sets: &ForwardSets,
        blur: TextureViewId,
        sharp: TextureViewId,
        view: &FrameView,
        extent: [u32; 2],
        object_base: u32,
        surfaces: Surfaces<'_>,
    ) {
        let scene = [
            Binding {
                binding: 0,
                resource: BindingResource::SampledTexture {
                    view: blur,
                    sampler: self.mip_sampler,
                },
            },
            // The frame itself, for the smooth end of the roughness range —
            // no mips to step between, so the plain linear sampler.
            Binding {
                binding: 1,
                resource: BindingResource::SampledTexture {
                    view: sharp,
                    sampler: self.pyramid_sampler,
                },
            },
        ];
        let scene_set = device.create_bind_group(self.pipeline, SCENE_SET as u32, &scene);
        // A twin's set 5 is the same two views plus the parameters, built the
        // first time the queue reaches a twin. Not cached: the parameters are a
        // fresh block every frame, so it would miss every frame.
        let mut twin_set = None;

        builder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: [0.0, 1.0],
        });
        // The arena is bound for the pass, not per draw: a mesh is a span into
        // it. The matrix is pushed with each pipeline for the same reason —
        // what used to vary per run, the material and the first object row,
        // travels with the instance now.
        builder.bind_geometry(false);
        let mut words = [0u32; 16];
        for (word, float) in words
            .iter_mut()
            .zip(view.view_proj.to_cols_array_2d().iter().flatten())
        {
            *word = float.to_bits();
        }

        // Back to front, and the order is the point, so the pipeline follows
        // the runs rather than the runs being grouped by pipeline: a run whose
        // material a surface shader reads switches to its twin, and the next
        // run that does not switches back.
        let mut bound = None;
        for run in draws.runs() {
            let item = draws.item(run.start);
            let Some(mesh) = renderer.meshes.get(item.mesh.0 as usize) else {
                continue;
            };
            let twin = surfaces
                .for_material(item.material.0, Twin::Refraction)
                .zip(surfaces.params());
            let pipeline = twin.map_or(self.pipeline, |(pipeline, _)| pipeline);
            if bound != Some(pipeline) {
                builder.bind_pipeline(pipeline);
                for (set, group) in sets.groups().into_iter().enumerate() {
                    builder.bind_group(set as u32, group);
                }
                let set = match twin {
                    Some((pipeline, params)) => *twin_set
                        .get_or_insert_with(|| params_group(device, pipeline, params, &scene)),
                    None => scene_set,
                };
                builder.bind_group(SCENE_SET as u32, set);
                builder.push_constants(&words);
                bound = Some(pipeline);
            }
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
        device: &Device,
        encoder: &mut Encoder,
        scene: TextureViewId,
        accum: TextureViewId,
        target: TextureViewId,
    ) {
        let sampled = |binding, view| Binding {
            binding,
            resource: BindingResource::SampledTexture {
                view,
                sampler: self.composite_sampler,
            },
        };
        let set = device.create_bind_group(
            self.composite_pipeline,
            0,
            &[
                sampled(0, scene),
                sampled(1, accum),
                Binding {
                    binding: 2,
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

/// The format the accumulation target is created with. Float and wide because
/// what it holds is premultiplied HDR radiance, exactly as the frame it will be
/// composited over holds — and premultiplied means the coverage it was
/// multiplied by has to survive in alpha for `refraction_composite` to put the
/// scene back underneath it. That is what keeps this off the packed colour
/// format the rest of the chain moved to.
pub(super) const ACCUM_FORMAT: crate::gfx::format::Format = HDR_WIDE_FORMAT;

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::gfx::format::{Format, SampleCount};
    use crate::gfx::graph::ImageUsage;
    use crate::gfx::rhi::{TextureDesc, TextureDimension};

    /// The pyramid of a flat frame is a flat pyramid at every level, which is
    /// what lets glass read the blur at any roughness and see the same colour.
    #[test]
    #[ignore = "needs a Metal device"]
    fn a_flat_frame_builds_a_flat_pyramid_on_metal() {
        let device = Device::Metal(crate::gfx::metal::MtlDriver::new().expect("a Metal device"));
        device.begin_frame(0);
        let forward = super::super::forward::ForwardPass::new(
            &device,
            super::super::hdr::HDR_FORMAT,
            super::super::hdr::HDR_WIDE_FORMAT,
        );
        let pass = RefractionPass::new(&device, forward.layout_source());
        let source = device.create_texture_2d(
            bytemuck::cast_slice(&[0.5f32, 0.25, 1.0, 1.0].repeat(128 * 128)),
            [128, 128],
            Format::Rgba32Float,
        );
        let pyramid = device.create_texture(&TextureDesc {
            format: Format::Rgba16Float,
            dimension: TextureDimension::D2,
            extent: [128, 128, 1],
            mip_levels: SCENE_LEVELS,
            array_layers: 1,
            samples: SampleCount::Sample1,
            usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
            memoryless: false,
        });
        let mips: Vec<_> = (0..SCENE_LEVELS)
            .map(|l| device.mip_view(pyramid, l))
            .collect();
        device.record_blocking(|encoder| pass.record_pyramid(&device, encoder, source, &mips));
        let Device::Metal(driver) = &device else {
            unreachable!()
        };
        for level in 0..SCENE_LEVELS as usize {
            let bytes = driver.read_texture_slice(pyramid, 0, level);
            let half = |i: usize| {
                let bits = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
                let exponent = i32::from((bits >> 10) & 0x1f);
                let fraction = f32::from(bits & 0x3ff);
                if exponent == 0 {
                    fraction * 2f32.powi(-24)
                } else {
                    (1.0 + fraction / 1024.0) * 2f32.powi(exponent - 15)
                }
            };
            let texel = [half(0), half(2), half(4)];
            assert!(
                (texel[0] - 0.5).abs() < 0.02 && (texel[2] - 1.0).abs() < 0.02,
                "level {level}: {texel:?}"
            );
        }
    }
}
