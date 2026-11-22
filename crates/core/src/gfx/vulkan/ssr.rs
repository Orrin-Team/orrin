//! Screen-space reflections: a depth pyramid, one importance-sampled ray per
//! half-resolution pixel, a denoiser with a history of its own, and a composite
//! that swaps the environment's guess for what the rays found.
//!
//! Five dispatches, and each earns its place. The depth pyramid is what makes
//! the march affordable — a ray crossing an empty room steps over it a cell at a
//! time instead of sampling every texel it passes. The colour pyramid is what
//! makes a *rough* reflection affordable: a ray from a wide lobe is a cone, and
//! the level whose texel matches the cone's footprint carries the average of
//! what the cone covers instead of one sample of it. The trace is at half
//! resolution because a stochastic ray is denoised either way, so the resolution
//! it is traced at buys much less than the rays it pays for. The temporal pass
//! and the composite are full resolution: the first because a mirror's history
//! resampled at half resolution goes soft under motion, the second because it
//! touches the frame's colour and subtracting the environment term wants this
//! pixel's own material.
//!
//! # The reflection's own history
//!
//! The temporal pass gathers the rays spatially and accumulates them into a
//! history that holds the reflection alone, before the composite and long
//! before TAA. TAA cannot be the denoiser: its neighbourhood clip runs on the
//! composited frame, where a reflection's variance is indistinguishable from
//! texture, so a reflection's noise is either clipped back in every frame —
//! flicker — or let through as grain. Clipped against the gathered rays
//! instead, the box is exactly as wide as the reflection's own spread, so what
//! reaches the composite, and TAA after it, is already stable.
//!
//! The history is reprojected along two candidates per pixel, because a
//! reflection does not move with the surface showing it. It moves with the
//! parallax of what it reflects, which lies a ray's length further down the
//! camera ray, so a mirror's history is found by projecting that virtual point
//! through last frame's camera; the surface's own motion vector, which TAA
//! uses, is the other candidate, and the only one for a pixel whose rays missed.
//! It is trusted by depth: the history keeps the linear depth it was
//! accumulated at, so a disocclusion reads as a surface at the wrong distance.
//!
//! The temporal pass and the composite are two passes, not one, though the
//! graph would accept one and it would save a dispatch. What forbids it is the
//! split frame: an accumulation nothing in the frame reads is left in its
//! storage layout, so the graph's closing transition puts it where next frame's
//! history is declared to be — and that transition is recorded in the trailing
//! graphics segment, which is submitted *after* the next frame's head, where the
//! next frame's temporal pass reads it. The composite reading the accumulation
//! is what moves it inside the frame that wrote it.
//!
//! Where it sits in the frame is not a detail: after shading, so what a ray
//! finds is lit, and before the TAA resolve, so the reflection is antialiased
//! with everything else rather than over the top of it.

use std::sync::atomic::{AtomicBool, Ordering};

use glam::Vec3;

use crate::gfx::format::SampleCount;
use crate::gfx::graph::ImageUsage;
use crate::gfx::rhi::{
    Binding, BindingResource, BufferId, ComputePipelineDesc, Device, Encoder, PipelineId,
    SamplerDesc, SamplerId, TextureDesc, TextureDimension, TextureViewId,
};
use crate::gfx::shaders::ShaderId;
use crate::scene::SsrSettings;

use super::taa::FrameView;

/// Side of the compute workgroup for the trace and the composite.
const TILE: u32 = 8;

/// Side of the tile one workgroup of the pyramid build owns, in full-resolution
/// texels. Fixed by the shader: eight invocations square, each reducing an 8x8
/// block, is what lets the first four levels come out of registers and the next
/// three out of one shared array.
const HIZ_TILE: u32 = 64;

/// How deep the pyramid goes, counting the full-resolution copy at level 0.
///
/// Seven is what a 64-texel tile reduces to on its own, and it is also enough:
/// a cell at level 6 is 64 texels across, so a ray crosses a 1080p frame in
/// about thirty steps. Going deeper would need a second dispatch to combine
/// tiles, and buy a handful of steps on rays that have already left the screen.
pub(super) const HIZ_LEVELS: u32 = 7;

/// Levels in the pyramid of the lit frame the trace samples its cones out of.
///
/// The same number as [`HIZ_LEVELS`] and for the same reason — both are what a
/// 64-texel tile reduces to on its own, since both are built by one dispatch.
/// Level 6 is a 64-texel blur of a half-resolution image, which is wider than
/// any lobe under the roughness cutoff has cause to ask for.
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

/// How far each ray travelled, in metres, zero for a miss: what the temporal pass
/// reprojects a mirror's history by.
///
/// A target of its own rather than squeezed in beside the radiance, whose four
/// channels are all spoken for — packing it there would mean a hand-rolled
/// radiance encoding in two shaders to save a 2 MB write at 1080p.
/// `R32_SFLOAT` because it is in Vulkan's mandatory storage list and the 16-bit
/// one-channel float is not.
pub(super) const RAY_LENGTH_FORMAT: crate::gfx::format::Format =
    crate::gfx::format::Format::R32Float;

/// The reflection history: accumulated radiance, and confidence in alpha.
///
/// Not the frame's packed colour format: its six-bit mantissa cannot represent
/// a step of a thirty-second of the difference, so a long history blended in it
/// stops moving while still a quarter away from the truth.
pub(super) const HISTORY_FORMAT: crate::gfx::format::Format =
    crate::gfx::format::Format::Rgba16Float;

/// The linear depth the history was accumulated at, with the sample count in
/// the low [`COUNT_BITS`] of its mantissa.
///
/// Packed because a second channel would be a second 16-bit float, which is
/// not a mandatory storage format (see `dof_tile` in `frame.rs`), or a second
/// 32-bit one, which doubles a full-resolution image kept twice over to carry
/// six bits. Losing those bits leaves the depth good to one part in 2^17,
/// three orders below the tolerance the temporal pass tests it to.
pub(super) const HISTORY_DEPTH_FORMAT: crate::gfx::format::Format =
    crate::gfx::format::Format::R32Float;

const COUNT_BITS: u32 = 6;

fn count_mask() -> u32 {
    (1 << COUNT_BITS) - 1
}

/// How many frames a history may average, for a mirror and for a surface at
/// the roughness cutoff; the temporal pass interpolates between them. Short for a
/// mirror, whose rays barely vary and where a stale sharp reflection is the
/// visible error; long where the lobe is wide and the noise is.
const MIRROR_HISTORY: u32 = 8;
const ROUGH_HISTORY: u32 = 32;

/// Shared by the trace and the composite, because they must agree about the
/// camera, the environment and the settings to the last bit — the composite
/// subtracts a term the trace's own weighting assumes.
/// The reflection block, as the words of its std140 layout: three matrices
/// (`proj`, `inv_proj`, `inv_view`), then `params`, `trace`, `extent`, `env` and
/// `env_tint`, a vec4 each, then the history's two matrices (`inv_view_proj`,
/// `prev_view_proj`) and its vec4.
const UBO_WORDS: usize = 3 * 16 + 5 * 4 + 2 * 16 + 4;

pub struct SsrPass {
    hiz_pipeline: PipelineId,
    source_pipeline: PipelineId,
    trace_pipeline: PipelineId,
    temporal_pipeline: PipelineId,
    resolve_pipeline: PipelineId,
    /// Nearest for everything the march reads. The pyramid is sampled at cell
    /// centres and the G-buffer at exact texels; interpolating either invents a
    /// surface between two that are really there, and the march would stop on
    /// it.
    nearest_clamp: SamplerId,
    /// Linear, for the lit frame's reduction into the pyramid and for the
    /// reprojected history.
    linear_clamp: SamplerId,
    /// Linear across levels as well, because the trace picks a fractional one:
    /// a cone that grows smoothly with distance would otherwise step between
    /// blurs, and the step would be visible as a ring on a curved surface.
    linear_mip: SamplerId,
    settings: SsrSettings,
    /// Advanced every frame the pass runs, and what decorrelates the sample
    /// sequence frame to frame. Without it every frame would trace the same ray
    /// per pixel and the history would average copies of one sample. It also
    /// picks which half of the history pair is written.
    frame: u64,
    uniforms: Option<BufferId>,
    /// Ping-ponged as TAA's is: `frame & 1` is this frame's accumulation and
    /// the other is the history, each a [`HISTORY_FORMAT`] and
    /// [`HISTORY_DEPTH_FORMAT`] pair. Empty until the first frame that reflects.
    history: Option<[[TextureViewId; 2]; 2]>,
    extent: [u32; 2],
    /// Set whenever the history cannot be trusted — first frame, a resize, or
    /// reflections having been off — and atomic for the reason
    /// `TaaPass::reset` is.
    reset: AtomicBool,
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
        let temporal_pipeline = pipeline(ShaderId::SsrTemporalComp);
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

        Self {
            hiz_pipeline,
            source_pipeline,
            trace_pipeline,
            temporal_pipeline,
            resolve_pipeline,
            nearest_clamp,
            linear_clamp,
            linear_mip,
            settings: SsrSettings::default(),
            frame: 0,
            uniforms: None,
            history: None,
            extent: [0, 0],
            reset: AtomicBool::new(true),
        }
    }

    /// Resolve this frame's camera, settings and environment into the block the
    /// trace, the temporal pass and the composite read, and allocate the
    /// history if the extent has none.
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
        self.settings = *settings;
        if self.history.is_none() || self.extent != extent {
            self.release(device);
            self.history = Some(allocate_history(device, extent));
            self.extent = extent;
        }
        self.frame = self.frame.wrapping_add(1);

        // The *jittered* projection, because every screen-space position here is
        // read out of targets the frame rasterised with it. Unjittering would
        // put the reconstructed position half a pixel from the depth it came
        // from, and the march would start beside the surface rather than on it.
        // The history's pair is the exception, and TAA's: it reprojects against
        // motion vectors that carry no jitter.
        let proj = view.proj;
        let matrices = [
            proj.to_cols_array_2d(),
            proj.inverse().to_cols_array_2d(),
            view.view.inverse().to_cols_array_2d(),
        ];
        let history_matrices = [
            view.unjittered_view_proj.inverse().to_cols_array_2d(),
            view.prev_view_proj.to_cols_array_2d(),
        ];
        let vectors = [
            [
                (self.frame % 64) as f32,
                settings.max_roughness.clamp(0.01, 1.0),
                settings.thickness.max(1e-3),
                settings.intensity.max(0.0),
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
        let history = [
            self.reset.load(Ordering::Relaxed) as u32 as f32,
            MIRROR_HISTORY as f32,
            ROUGH_HISTORY as f32,
            count_mask() as f32,
        ];
        let mut words = [0u32; UBO_WORDS];
        for (word, float) in words.iter_mut().zip(
            matrices
                .iter()
                .flatten()
                .flatten()
                .chain(vectors.iter().flatten())
                .chain(history_matrices.iter().flatten().flatten())
                .chain(&history),
        ) {
            *word = float.to_bits();
        }
        let uniforms = device.upload_uniform(&words);
        self.uniforms = Some(uniforms);
    }

    /// Free the history, for a frame that does not reflect. Freed rather than
    /// kept, as TAA's is: two pairs of full-resolution targets is real memory,
    /// and anything they held is stale the moment a frame renders without them.
    pub(super) fn release(&mut self, device: &Device) {
        for view in self.history.take().into_iter().flatten().flatten() {
            device.destroy_texture(view);
        }
        self.reset.store(true, Ordering::Relaxed);
    }

    /// This frame's accumulation, radiance then depth, which is next frame's
    /// history.
    pub(super) fn accum_views(&self) -> [TextureViewId; 2] {
        self.pair()[(self.frame & 1) as usize]
    }

    pub(super) fn history_views(&self) -> [TextureViewId; 2] {
        self.pair()[((self.frame + 1) & 1) as usize]
    }

    fn pair(&self) -> &[[TextureViewId; 2]; 2] {
        self.history
            .as_ref()
            .expect("the graph scheduled a reflection pass with no history allocated")
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
        source: TextureViewId,
        target: TextureViewId,
        ray_length: TextureViewId,
    ) {
        let set = device.create_transient_bind_group(
            self.trace_pipeline,
            0,
            &[
                self.sampled(0, hiz, self.nearest_clamp),
                self.sampled(1, depth, self.nearest_clamp),
                self.sampled(2, normal, self.nearest_clamp),
                self.sampled(3, material, self.nearest_clamp),
                self.sampled(4, source, self.linear_mip),
                Binding {
                    binding: 5,
                    resource: BindingResource::StorageTexture(target),
                },
                Binding {
                    binding: 6,
                    resource: BindingResource::UniformBuffer(self.uniforms()),
                },
                Binding {
                    binding: 7,
                    resource: BindingResource::StorageTexture(ray_length),
                },
            ],
        );

        builder.bind_pipeline(self.trace_pipeline);
        builder.bind_group(0, set);
        dispatch_over(device, builder, target);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_temporal(
        &self,
        device: &Device,
        builder: &mut Encoder,
        rays: TextureViewId,
        ray_length: TextureViewId,
        depth: TextureViewId,
        normal: TextureViewId,
        material: TextureViewId,
        velocity: TextureViewId,
    ) {
        let [history, history_depth] = self.history_views();
        let [accum, accum_depth] = self.accum_views();
        let set = device.create_transient_bind_group(
            self.temporal_pipeline,
            0,
            &[
                // Fetched at exact texels by the bilateral gather, which does its
                // own interpolation — a linear sampler here would blend across the
                // silhouettes that gather exists to respect.
                self.sampled(0, rays, self.nearest_clamp),
                self.sampled(1, ray_length, self.nearest_clamp),
                self.sampled(2, depth, self.nearest_clamp),
                self.sampled(3, normal, self.nearest_clamp),
                self.sampled(4, material, self.nearest_clamp),
                self.sampled(5, velocity, self.nearest_clamp),
                // Linear, because a reprojected coordinate almost never lands
                // on a texel centre; the depth beside it is fetched exactly,
                // since interpolating it would blend the sample count's bits.
                self.sampled(6, history, self.linear_clamp),
                self.sampled(7, history_depth, self.nearest_clamp),
                Binding {
                    binding: 8,
                    resource: BindingResource::StorageTexture(accum),
                },
                Binding {
                    binding: 9,
                    resource: BindingResource::StorageTexture(accum_depth),
                },
                Binding {
                    binding: 10,
                    resource: BindingResource::UniformBuffer(self.uniforms()),
                },
            ],
        );

        builder.bind_pipeline(self.temporal_pipeline);
        builder.bind_group(0, set);
        dispatch_over(device, builder, accum);

        // The accumulation just written is the history the next frame reads,
        // so whatever made it untrustworthy is over.
        self.reset.store(false, Ordering::Relaxed);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_resolve(
        &self,
        device: &Device,
        builder: &mut Encoder,
        source: TextureViewId,
        normal: TextureViewId,
        material: TextureViewId,
        environment: TextureViewId,
        environment_sampler: SamplerId,
        target: TextureViewId,
    ) {
        let [accum, accum_depth] = self.accum_views();
        let set = device.create_transient_bind_group(
            self.resolve_pipeline,
            0,
            &[
                self.sampled(0, source, self.nearest_clamp),
                self.sampled(1, accum, self.nearest_clamp),
                self.sampled(2, accum_depth, self.nearest_clamp),
                self.sampled(3, normal, self.nearest_clamp),
                self.sampled(4, material, self.nearest_clamp),
                Binding {
                    binding: 5,
                    resource: BindingResource::StorageTexture(target),
                },
                Binding {
                    binding: 6,
                    resource: BindingResource::Texture(environment),
                },
                Binding {
                    binding: 7,
                    resource: BindingResource::Sampler(environment_sampler),
                },
                Binding {
                    binding: 8,
                    resource: BindingResource::UniformBuffer(self.uniforms()),
                },
            ],
        );

        builder.bind_pipeline(self.resolve_pipeline);
        builder.bind_group(0, set);
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

fn allocate_history(device: &Device, extent: [u32; 2]) -> [[TextureViewId; 2]; 2] {
    std::array::from_fn(|_| {
        [HISTORY_FORMAT, HISTORY_DEPTH_FORMAT].map(|format| {
            let texture = device.create_texture(&TextureDesc {
                format,
                dimension: TextureDimension::D2,
                extent: [extent[0], extent[1], 1],
                mip_levels: 1,
                array_layers: 1,
                samples: SampleCount::Sample1,
                usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                memoryless: false,
            });
            // The layout `ssr_history` is declared to enter in, from the first
            // frame on; that frame's contents are still ignored (see `reset`).
            device.initialise_sampled(texture);
            texture
        })
    })
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
            HIZ_TILE.trailing_zeros() + 1,
            "a {HIZ_TILE}-texel tile reduces to {} levels, not {HIZ_LEVELS}",
            HIZ_TILE.trailing_zeros() + 1,
        );
    }

    /// The temporal pass keeps a history's sample count in the low bits of the depth
    /// it was accumulated at, packed as `pack_history_depth` in the shader does.
    /// The longest history has to fit in those bits, or a settled pixel's count
    /// would carry into the depth; and what the bits cost the depth has to stay
    /// far inside the few percent the pass tests it to, or a still surface
    /// would read as a disocclusion.
    #[test]
    fn the_sample_count_fits_beside_the_depth_without_moving_it() {
        let mask = count_mask();
        assert!(MIRROR_HISTORY.max(ROUGH_HISTORY) <= mask);

        let pack = |z: f32, count: u32| f32::from_bits((z.to_bits() & !mask) | count);
        for z in [0.05f32, 1.0, 3.7, 250.0, 10_000.0] {
            for count in 0..=ROUGH_HISTORY {
                let packed = pack(z, count);
                assert_eq!(packed.to_bits() & mask, count, "{z} m lost its count");
                let error = ((packed - z) / z).abs();
                assert!(error < 1e-4, "{count} samples moved {z} m by {error}");
            }
        }
    }
}
