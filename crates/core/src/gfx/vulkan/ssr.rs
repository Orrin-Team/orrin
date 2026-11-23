//! Screen-space reflections: a depth pyramid, one exact reflection ray per
//! 2x2 quad — and one per pixel on a smooth surface — whose roughness is the
//! blur the hit is read through, and a composite that gathers those rays into
//! each pixel and swaps the environment's guess for what they found.
//!
//! Four dispatches, and each earns its place. The depth pyramid is what makes
//! the march affordable — a ray crossing an empty room steps over it a cell at a
//! time instead of sampling every texel it passes. The colour pyramid is what
//! makes a *rough* reflection: the level whose texel matches the lobe's width
//! at the hit is the lobe's blur, read in one fetch. The trace is half
//! resolution, and the composite full, because it touches the frame's colour and
//! subtracting the environment term wants this pixel's own material.
//!
//! # Exact rays, and roughness as blur
//!
//! Every ray is the exact reflection of the view ray about the surface normal,
//! at every roughness, and the same for the same surface every frame. The
//! roughness is a cone around it: the hit is read from the colour pyramid at
//! the level whose texel is the cone's width there, so a short ray stays sharp
//! and a long one softens — a reflection hardening toward its contact, as a
//! real one does. Nothing in the result varies from frame to frame but what the
//! frame itself does, so there is no history and nothing to denoise. That is
//! the requirement this design answers: no technique that dithers, ghosts or
//! smears, which is what a reflection that leans on accumulating frames to hide
//! its noise does.
//!
//! What it gives up is the lobe's shape. A glossy reflection here is a blurred
//! mirror image, not a scattered one: it has no stretch toward the viewer at
//! grazing angles, an object reflected in a rough surface keeps a soft version
//! of its outline where a real lobe would spread it, because one ray either
//! hits or misses where a lobe would hit in part, and the blur is of the
//! screen, so it can take in whatever sits beside the hit in depth.
//!
//! Not the stochastic design this replaced, though it is the usual one: a ray
//! drawn from the GGX lobe per quad per frame, a bilateral gather, and a
//! history of the reflection's own that averaged the draws into the lobe. A
//! rough surface sees few hits, and a history of rare bright ones never
//! settles: on white metal at roughness 0.5 it left a patch of sparkles that
//! changed every frame, measured at a settled change of 0.257 against 0.186
//! with reflections off, where this design measures 0.201. It also kept four
//! full-resolution images between frames and a pass to fill them.
//!
//! # Two ray rates
//!
//! A glossy surface is traced at one ray per quad, and the composite gathers
//! the four quads around each pixel: the blur at a glossy hit is wider than a
//! quad, so a ray per pixel would buy a sharper version of something the
//! roughness then blurs. That stops at a smooth surface, whose blur is narrower
//! than the quad, where a quarter-rate ray is a quarter-resolution reflection —
//! the soft, stair-stepped mirror the eye picks out first. Below
//! `QUARTER_RATE_ROUGHNESS` in `ssr_march.glsl` each pixel therefore traces its
//! own ray, cross-fading to the gather as roughness rises, so a roughness
//! gradient shows no line. Its cone also reads the frame itself rather than the
//! half-resolution pyramid.
//!
//! The full-rate rays are traced inside the composite, where the pixel that
//! needs one already is, rather than by the trace dispatching over every pixel:
//! a full-resolution trace would idle three lanes in four on every glossy pixel
//! and need a full-resolution target to hand its rays over. Rejected too: a
//! classification pass building ray and tile lists for an indirect dispatch
//! (the FidelityFX shape). The RHI has no indirect dispatch.
//!
//! Where it sits in the frame is not a detail: after shading, so what a ray
//! finds is lit, and before the TAA resolve, so the reflection is antialiased
//! with everything else rather than over the top of it.

use glam::Vec3;

use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, BufferId, ComputePipelineDesc, Device, Encoder, Feature,
    PipelineId, SamplerDesc, SamplerId, TextureViewId,
};
use crate::gfx::shaders::ShaderId;
use crate::scene::SsrSettings;

use super::raytrace::{RayFrame, TEXTURE_SET};
use super::taa::FrameView;

/// Side of the compute workgroup for the trace and the composite.
const TILE: u32 = 8;

/// Side of the tile one workgroup of the pyramid build owns, in full-resolution
/// texels. Fixed by the shader: sixteen invocations square, each reducing a 4x4
/// block, is what lets the first two levels come out of registers and the next
/// four out of one shared array.
const HIZ_TILE: u32 = 64;

/// How deep the pyramid goes above the depth buffer, which the march reads
/// directly as level 0: the image is based at half the frame.
///
/// Six is what a 64-texel tile reduces to on its own, and it is also enough:
/// a cell at level 6 is 64 texels across, so a ray crosses a 1080p frame in
/// about thirty steps. Going deeper would need a second dispatch to combine
/// tiles, and buy a handful of steps on rays that have already left the screen.
///
/// Not a copy of the depth buffer at level 0, though that would let the march
/// read every level from one image: the copy was three quarters of what the
/// build wrote, and the build cost more than the trace it served.
pub(super) const HIZ_LEVELS: u32 = 6;

/// Levels in the pyramid of the lit frame the trace reads its blur out of.
///
/// Seven for the reason [`HIZ_LEVELS`] is six: what a 64-texel tile reduces to
/// on its own, counting the base, since both are built by one dispatch.
/// Level 6 is a 64-texel blur of a half-resolution image, 128 pixels of the
/// frame: the widest blur a reflection can have.
pub(super) const SOURCE_LEVELS: u32 = 7;

/// Single-channel float, because a min-depth pyramid stores exactly one number
/// and `R32_SFLOAT` is a storage format every Vulkan implementation supports.
/// Non-linear NDC depth, not linearised: the march compares against the same
/// values the depth buffer holds, and the comparison only needs monotonicity.
pub(super) const HIZ_FORMAT: crate::gfx::format::Format = crate::gfx::format::Format::R32Float;

/// What one ray leaves behind: the radiance it found, and how much of it to
/// believe. `SFLOAT` because the radiance is HDR and the confidence is a
/// fraction — the same target has to carry both.
pub(super) const RAY_FORMAT: crate::gfx::format::Format = crate::gfx::format::Format::Rgba16Float;

/// Shared by the trace and the composite, because they must agree about the
/// camera, the environment and the settings to the last bit — the composite
/// subtracts a term the trace's own weighting assumes.
/// The reflection block, as the words of its std140 layout: three matrices
/// (`proj`, `inv_proj`, `inv_view`), then `params`, `trace`, `extent`, `env` and
/// `env_tint`, a vec4 each.
const UBO_WORDS: usize = 3 * 16 + 5 * 4;

pub struct SsrPass {
    hiz_pipeline: PipelineId,
    source_pipeline: PipelineId,
    trace_pipeline: PipelineId,
    resolve_pipeline: PipelineId,
    /// Nearest for everything the march reads. The pyramid is sampled at cell
    /// centres and the G-buffer at exact texels; interpolating either invents a
    /// surface between two that are really there, and the march would stop on
    /// it.
    nearest_clamp: SamplerId,
    /// Linear, for the lit frame's reduction into the pyramid and for a smooth
    /// surface's cone, which reads the frame itself.
    linear_clamp: SamplerId,
    /// Linear across levels as well, because the trace picks a fractional one:
    /// a cone that grows smoothly with distance would otherwise step between
    /// blurs, and the step would be visible as a ring on a curved surface.
    linear_mip: SamplerId,
    uniforms: Option<BufferId>,
    /// The trace and the composite again, falling back to a ray traced
    /// against the scene. `None` on a device that cannot trace.
    traced: Option<Traced>,
}

/// The pipelines of the traced pair, and the material textures a hit reads,
/// which each binds through a set of its own for the reason every pass that
/// reads them does: a set belongs to the layout it was made for.
struct Traced {
    trace: PipelineId,
    resolve: PipelineId,
    textures: Option<[BindGroupId; 2]>,
}

impl SsrPass {
    pub fn new(device: &Device) -> Self {
        let pipeline = |shader: ShaderId| {
            device.create_compute_pipeline(&ComputePipelineDesc {
                shader: shader.into(),
            })
        };
        let hiz_pipeline = pipeline(ShaderId::HizBuildComp);
        let source_pipeline = pipeline(ShaderId::ColorPyramidComp);
        let trace_pipeline = pipeline(ShaderId::SsrTraceComp);
        let resolve_pipeline = pipeline(ShaderId::SsrResolveComp);

        // Nearest between levels as well: the march picks a level and means that
        // level's cell, and a value blended with the next level's is a cell size
        // that exists nowhere.
        let nearest_clamp = device.create_sampler(SamplerDesc {
            linear: false,
            mipmap_linear: false,
            repeat: false,
            compare: false,
            anisotropy: None,
            max_lod: None,
        });
        let linear_clamp = device.create_sampler(SamplerDesc {
            linear: true,
            mipmap_linear: false,
            repeat: false,
            compare: false,
            anisotropy: None,
            max_lod: Some(1.0),
        });
        let linear_mip = device.create_sampler(SamplerDesc {
            linear: true,
            mipmap_linear: true,
            repeat: false,
            compare: false,
            anisotropy: None,
            max_lod: None,
        });

        let traced = device.supports(Feature::RayTracing).then(|| {
            let pipeline = |shader: ShaderId| {
                device.create_compute_pipeline_over_textures(
                    &ComputePipelineDesc {
                        shader: shader.into(),
                    },
                    TEXTURE_SET,
                )
            };
            Traced {
                trace: pipeline(ShaderId::SsrTraceCompTraced),
                resolve: pipeline(ShaderId::SsrResolveCompTraced),
                textures: None,
            }
        });

        Self {
            hiz_pipeline,
            source_pipeline,
            trace_pipeline,
            resolve_pipeline,
            nearest_clamp,
            linear_clamp,
            linear_mip,
            uniforms: None,
            traced,
        }
    }

    /// Make sure the traced pair has the material textures bound, before a
    /// frame that traces records. Kept until the textures change.
    pub(super) fn prepare_traced(
        &mut self,
        device: &Device,
        textures: &[TextureViewId],
        sampler: SamplerId,
    ) {
        if let Some(traced) = &mut self.traced
            && traced.textures.is_none()
        {
            traced.textures = Some([traced.trace, traced.resolve].map(|pipeline| {
                device.create_texture_array_group(pipeline, TEXTURE_SET, textures, sampler)
            }));
        }
    }

    /// Let go of the texture sets, which name textures that have changed.
    pub(super) fn forget_textures(&mut self, device: &Device) {
        if let Some(groups) = self
            .traced
            .as_mut()
            .and_then(|traced| traced.textures.take())
        {
            for group in groups {
                device.destroy_bind_group(group);
            }
        }
    }

    /// The traced pipeline and its textures, for a frame that asked for them.
    fn traced(
        &self,
        rays: Option<&RayFrame<'_>>,
        resolve: bool,
    ) -> Option<(PipelineId, BindGroupId)> {
        rays?;
        let traced = self
            .traced
            .as_ref()
            .expect("the graph scheduled traced reflections on a device that cannot trace");
        let textures = traced
            .textures
            .expect("traced reflections were recorded before prepare_traced");
        Some(if resolve {
            (traced.resolve, textures[1])
        } else {
            (traced.trace, textures[0])
        })
    }

    /// Resolve this frame's camera, settings and environment into the block the
    /// trace and the composite read.
    ///
    /// Called before anything records, like the other per-frame `begin_frame`s,
    /// because the passes are separated in the schedule by whatever the graph
    /// decided and must not re-derive the camera between them.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn begin_frame(
        &mut self,
        device: &Device,
        settings: &SsrSettings,
        view: &FrameView,
        extent: [u32; 2],
        levels: u32,
        environment_yaw: f32,
        specular_mips: u32,
        env_tint: Vec3,
    ) {
        // The *jittered* projection, because every screen-space position here is
        // read out of targets the frame rasterised with it. Unjittering would
        // put the reconstructed position half a pixel from the depth it came
        // from, and the march would start beside the surface rather than on it.
        let proj = view.proj;
        let matrices = [
            proj.to_cols_array_2d(),
            proj.inverse().to_cols_array_2d(),
            view.view.inverse().to_cols_array_2d(),
        ];
        let vectors = [
            [
                settings.max_roughness.clamp(0.01, 1.0),
                settings.thickness.max(1e-3),
                settings.intensity.max(0.0),
                0.0,
            ],
            [
                settings.max_steps.max(1) as f32,
                settings.max_distance.max(0.1),
                levels as f32,
                0.0,
            ],
            [
                extent[0] as f32,
                extent[1] as f32,
                1.0 / extent[0].max(1) as f32,
                1.0 / extent[1].max(1) as f32,
            ],
            [
                environment_yaw.to_radians().sin(),
                environment_yaw.to_radians().cos(),
                specular_mips as f32,
                0.0,
            ],
            [env_tint.x, env_tint.y, env_tint.z, 0.0],
        ];
        let mut words = [0u32; UBO_WORDS];
        for (word, float) in words.iter_mut().zip(
            matrices
                .iter()
                .flatten()
                .flatten()
                .chain(vectors.iter().flatten()),
        ) {
            *word = float.to_bits();
        }
        let uniforms = device.upload_uniform(&words);
        self.uniforms = Some(uniforms);
    }

    fn uniforms(&self) -> BufferId {
        self.uniforms
            .expect("the graph scheduled a reflection pass before begin_frame")
    }

    pub(super) fn record_hiz(
        &self,
        device: &Device,
        builder: &mut Encoder,
        depth: TextureViewId,
        mips: &[TextureViewId],
        extent: [u32; 2],
    ) {
        let levels = mips.len() as u32;
        // Every slot in the shader's array has to be written even where the
        // pyramid is shorter than the declaration, so the spares point at level
        // zero. `levels` is what stops anything from storing through them.
        let mut bindings = vec![self.sampled(0, depth, self.nearest_clamp)];
        bindings.extend((0..HIZ_LEVELS as usize).map(|level| Binding {
            binding: 1,
            resource: BindingResource::StorageTexture(mips[level.min(mips.len() - 1)]),
        }));
        let set = device.create_bind_group(self.hiz_pipeline, 0, &bindings);

        builder.bind_pipeline(self.hiz_pipeline);
        builder.bind_group(0, set);
        builder.push_constants(&[extent[0], extent[1], levels]);

        // One workgroup per HIZ_TILE-sized tile covers the frame, and every store
        // is bounds-checked against the level it writes, so nothing lands outside
        // an image.
        builder.dispatch([
            extent[0].div_ceil(HIZ_TILE),
            extent[1].div_ceil(HIZ_TILE),
            1,
        ]);
    }

    pub(super) fn record_source(
        &self,
        device: &Device,
        builder: &mut Encoder,
        source: TextureViewId,
        mips: &[TextureViewId],
    ) {
        let levels = mips.len() as u32;
        let extent = device.view_extent(mips[0]);
        // Linear, because level 0 is a 2:1 reduction of the frame and one bilinear
        // fetch between four texels *is* their average.
        let mut bindings = vec![self.sampled(0, source, self.linear_clamp)];
        bindings.extend((0..SOURCE_LEVELS as usize).map(|level| Binding {
            binding: 1,
            resource: BindingResource::StorageTexture(mips[level.min(mips.len() - 1)]),
        }));
        let set = device.create_bind_group(self.source_pipeline, 0, &bindings);

        builder.bind_pipeline(self.source_pipeline);
        builder.bind_group(0, set);
        builder.push_constants(&[extent[0], extent[1], levels]);

        // One workgroup per HIZ_TILE-sized tile covers level 0, and every store is
        // bounds-checked against the level it writes.
        builder.dispatch([
            extent[0].div_ceil(HIZ_TILE),
            extent[1].div_ceil(HIZ_TILE),
            1,
        ]);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_trace(
        &self,
        device: &Device,
        builder: &mut Encoder,
        hiz: TextureViewId,
        depth: TextureViewId,
        normal: TextureViewId,
        material: TextureViewId,
        source_pyramid: TextureViewId,
        source: TextureViewId,
        target: TextureViewId,
        rays: Option<&RayFrame<'_>>,
    ) {
        let traced = self.traced(rays, false);
        let pipeline = traced.map_or(self.trace_pipeline, |(pipeline, _)| pipeline);
        let set = device.create_transient_bind_group(
            pipeline,
            0,
            &[
                self.sampled(0, hiz, self.nearest_clamp),
                self.sampled(1, depth, self.nearest_clamp),
                self.sampled(2, normal, self.nearest_clamp),
                self.sampled(3, material, self.nearest_clamp),
                self.sampled(4, source_pyramid, self.linear_mip),
                Binding {
                    binding: 5,
                    resource: BindingResource::StorageTexture(target),
                },
                Binding {
                    binding: 6,
                    resource: BindingResource::UniformBuffer(self.uniforms()),
                },
                self.sampled(7, source, self.linear_clamp),
            ],
        );

        builder.bind_pipeline(pipeline);
        builder.bind_group(0, set);
        if let (Some(rays), Some((_, textures))) = (rays, traced) {
            rays.bind(device, builder, pipeline);
            builder.bind_group(TEXTURE_SET, textures);
        }
        dispatch_over(device, builder, target);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_resolve(
        &self,
        device: &Device,
        builder: &mut Encoder,
        source: TextureViewId,
        rays: TextureViewId,
        depth: TextureViewId,
        normal: TextureViewId,
        material: TextureViewId,
        hiz: TextureViewId,
        source_pyramid: TextureViewId,
        environment: TextureViewId,
        environment_sampler: SamplerId,
        target: TextureViewId,
        traced_rays: Option<&RayFrame<'_>>,
    ) {
        let traced = self.traced(traced_rays, true);
        let pipeline = traced.map_or(self.resolve_pipeline, |(pipeline, _)| pipeline);
        let mut bindings = vec![
            // Linear, though the composite reads its own pixel by texel, so
            // a smooth surface's ray samples the frame as the trace does.
            self.sampled(0, source, self.linear_clamp),
            // Fetched at exact texels by the bilateral gather, which does its
            // own interpolation — a linear sampler here would blend across the
            // silhouettes that gather exists to respect.
            self.sampled(1, rays, self.nearest_clamp),
            self.sampled(2, depth, self.nearest_clamp),
            self.sampled(3, normal, self.nearest_clamp),
            self.sampled(4, material, self.nearest_clamp),
            Binding {
                binding: 5,
                resource: BindingResource::StorageTexture(target),
            },
            Binding {
                binding: 8,
                resource: BindingResource::UniformBuffer(self.uniforms()),
            },
            // A smooth surface's own ray, traced here with the trace's
            // samplers for the trace's reasons.
            self.sampled(9, hiz, self.nearest_clamp),
            self.sampled(10, source_pyramid, self.linear_mip),
        ];
        // The traced composite reads the environment through the set every
        // hit is shaded from, and declares it nowhere else.
        if traced.is_none() {
            bindings.extend([
                Binding {
                    binding: 6,
                    resource: BindingResource::Texture(environment),
                },
                Binding {
                    binding: 7,
                    resource: BindingResource::Sampler(environment_sampler),
                },
            ]);
        }
        let set = device.create_transient_bind_group(pipeline, 0, &bindings);

        builder.bind_pipeline(pipeline);
        builder.bind_group(0, set);
        if let (Some(rays), Some((_, textures))) = (traced_rays, traced) {
            rays.bind(device, builder, pipeline);
            builder.bind_group(TEXTURE_SET, textures);
        }
        dispatch_over(device, builder, target);
    }

    fn sampled(&self, binding: u32, view: TextureViewId, sampler: SamplerId) -> Binding {
        Binding {
            binding,
            resource: BindingResource::SampledTexture { view, sampler },
        }
    }
}

fn dispatch_over(device: &Device, builder: &mut Encoder, target: TextureViewId) {
    let extent = device.view_extent(target);
    builder.dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1]);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader reduces a fixed tile with a fixed number of steps, and the
    /// dispatch is sized off the same tile. A pyramid deeper than the tile can
    /// produce would need data from a neighbouring workgroup, which this build
    /// deliberately never reads.
    #[test]
    fn the_pyramid_is_no_deeper_than_one_tile_can_reduce() {
        assert_eq!(
            HIZ_LEVELS,
            HIZ_TILE.trailing_zeros(),
            "a {HIZ_TILE}-texel tile reduces to {} levels above its base, not {HIZ_LEVELS}",
            HIZ_TILE.trailing_zeros(),
        );
    }
}
