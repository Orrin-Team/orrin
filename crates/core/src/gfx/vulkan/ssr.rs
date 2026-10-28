//! Screen-space reflections: a depth pyramid, one importance-sampled ray per
//! half-resolution pixel, and a composite that swaps the environment's guess for
//! what the ray found.
//!
//! Four dispatches, and each earns its place. The depth pyramid is what makes
//! the march affordable — a ray crossing an empty room steps over it a cell at a
//! time instead of sampling every texel it passes. The colour pyramid is what
//! makes a *rough* reflection affordable: a ray from a wide lobe is a cone, and
//! the level whose texel matches the cone's footprint carries the average of
//! what the cone covers instead of one sample of it. The trace is at half
//! resolution because a stochastic ray is denoised either way, so the resolution
//! it is traced at buys much less than the rays it pays for. The composite is
//! full resolution because it is the pass that touches the frame's colour, and
//! because subtracting the environment term wants this pixel's own material.
//!
//! Where it sits in the frame is not a detail: between the forward pass and the
//! TAA resolve. Before TAA, so the accumulation over the jitter sequence is what
//! denoises the rays; after shading, so what a ray finds is lit.

use glam::Vec3;

use crate::gfx::rhi::{
    Binding, BindingResource, BufferId, ComputePipelineDesc, Device, Encoder, PipelineId,
    SamplerDesc, SamplerId, TextureViewId,
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
    /// Linear, for the lit frame a ray lands in and for the environment cube.
    linear_clamp: SamplerId,
    /// Linear across levels as well, because the trace picks a fractional one:
    /// a cone that grows smoothly with distance would otherwise step between
    /// blurs, and the step would be visible as a ring on a curved surface.
    linear_mip: SamplerId,
    settings: SsrSettings,
    /// Advanced every frame the pass runs, and what decorrelates the sample
    /// sequence frame to frame. Without it every frame would trace the same ray
    /// per pixel and TAA would average eight copies of one sample.
    frame: u64,
    uniforms: Option<BufferId>,
}

impl SsrPass {
    pub fn new(device: &Device) -> Self {
        let pipeline = |shader| device.create_compute_pipeline(&ComputePipelineDesc { shader });
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

        Self {
            hiz_pipeline,
            source_pipeline,
            trace_pipeline,
            resolve_pipeline,
            nearest_clamp,
            linear_clamp,
            linear_mip,
            settings: SsrSettings::default(),
            frame: 0,
            uniforms: None,
        }
    }

    /// Resolve this frame's camera, settings and environment into the block both
    /// dispatches read.
    ///
    /// Called before anything records, like the other per-frame `begin_frame`s,
    /// because the two passes are separated in the schedule by whatever the
    /// graph decided and must not re-derive the camera between them.
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
        self.frame = self.frame.wrapping_add(1);

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
        let Device::Vulkan(driver) = device;
        let uniforms = driver.upload_uniform(&words);
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
        source: TextureViewId,
        target: TextureViewId,
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
            ],
        );

        builder.bind_pipeline(self.trace_pipeline);
        builder.bind_group(0, set);
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
        environment: TextureViewId,
        environment_sampler: SamplerId,
        target: TextureViewId,
    ) {
        let Device::Vulkan(_driver) = device;
        let set = device.create_transient_bind_group(
            self.resolve_pipeline,
            0,
            &[
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
}
