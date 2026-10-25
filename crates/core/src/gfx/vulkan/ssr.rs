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

use std::sync::Arc;

use glam::Vec3;
use vulkano::buffer::allocator::SubbufferAllocatorCreateInfo;
use vulkano::buffer::{BufferContents, BufferUsage};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};

use crate::scene::SsrSettings;

use super::context::VkContext;
use super::record::{Arena, Recorder};
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
pub(super) const HIZ_FORMAT: Format = Format::R32_SFLOAT;

/// What one ray leaves behind: the radiance it found, and how much of it to
/// believe. `SFLOAT` because the radiance is HDR and the confidence is a
/// fraction — the same target has to carry both.
pub(super) const RAY_FORMAT: Format = Format::R16G16B16A16_SFLOAT;

/// Shared by the trace and the composite, because they must agree about the
/// camera, the environment and the settings to the last bit — the composite
/// subtracts a term the trace's own weighting assumes.
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct SsrUbo {
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    /// x = frame index, y = max roughness, z = thickness, w = intensity.
    params: [f32; 4],
    /// x = max steps, y = max distance, z = pyramid levels.
    trace: [f32; 4],
    /// xy = full-resolution extent, zw = its reciprocal.
    extent: [f32; 4],
    /// x = sin(environment yaw), y = cos, z = specular mip count.
    env: [f32; 4],
    env_tint: [f32; 4],
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct HizPush {
    extent: [i32; 2],
    levels: i32,
}

pub struct SsrPass {
    hiz_pipeline: Arc<ComputePipeline>,
    source_pipeline: Arc<ComputePipeline>,
    trace_pipeline: Arc<ComputePipeline>,
    resolve_pipeline: Arc<ComputePipeline>,
    /// Nearest for everything the march reads. The pyramid is sampled at cell
    /// centres and the G-buffer at exact texels; interpolating either invents a
    /// surface between two that are really there, and the march would stop on
    /// it.
    nearest_clamp: Arc<Sampler>,
    /// Linear, for the lit frame a ray lands in and for the environment cube.
    linear_clamp: Arc<Sampler>,
    /// Linear across levels as well, because the trace picks a fractional one:
    /// a cone that grows smoothly with distance would otherwise step between
    /// blurs, and the step would be visible as a ring on a curved surface.
    linear_mip: Arc<Sampler>,
    uniform_allocator: Arena,
    settings: SsrSettings,
    /// Advanced every frame the pass runs, and what decorrelates the sample
    /// sequence frame to frame. Without it every frame would trace the same ray
    /// per pixel and TAA would average eight copies of one sample.
    frame: u64,
    uniforms: Option<vulkano::buffer::Subbuffer<SsrUbo>>,
}

impl SsrPass {
    pub fn new(ctx: &VkContext) -> Self {
        let device = &ctx.device;
        let hiz_pipeline = build_pipeline(
            ctx,
            hiz_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );
        let source_pipeline = build_pipeline(
            ctx,
            source_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );
        let trace_pipeline = build_pipeline(
            ctx,
            trace_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );
        let resolve_pipeline = build_pipeline(
            ctx,
            resolve_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );

        let nearest_clamp = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                // Nearest between levels as well: the march picks a level and
                // means that level's cell, and a value blended with the next
                // level's is a cell size that exists nowhere.
                mipmap_mode: vulkano::image::sampler::SamplerMipmapMode::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                lod: 0.0..=vulkano::image::sampler::LOD_CLAMP_NONE,
                ..SamplerCreateInfo::default()
            },
        )
        .unwrap();
        let linear_clamp = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        )
        .unwrap();
        let linear_mip = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                mipmap_mode: vulkano::image::sampler::SamplerMipmapMode::Linear,
                lod: 0.0..=vulkano::image::sampler::LOD_CLAMP_NONE,
                ..SamplerCreateInfo::simple_repeat_linear()
            },
        )
        .unwrap();

        let uniform_allocator = Arena::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: vulkano::memory::allocator::MemoryTypeFilter::PREFER_DEVICE
                    | vulkano::memory::allocator::MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        Self {
            hiz_pipeline,
            source_pipeline,
            trace_pipeline,
            resolve_pipeline,
            nearest_clamp,
            linear_clamp,
            linear_mip,
            uniform_allocator,
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
        let uniforms = self.uniform_allocator.allocate_sized::<SsrUbo>();
        *uniforms.write().unwrap() = SsrUbo {
            proj: proj.to_cols_array_2d(),
            inv_proj: proj.inverse().to_cols_array_2d(),
            inv_view: view.view.inverse().to_cols_array_2d(),
            params: [
                (self.frame % 64) as f32,
                settings.max_roughness.clamp(0.01, 1.0),
                settings.thickness.max(1e-3),
                settings.intensity.max(0.0),
            ],
            trace: [
                settings.max_steps.max(1) as f32,
                settings.max_distance.max(0.1),
                levels as f32,
                0.0,
            ],
            extent: [
                extent[0] as f32,
                extent[1] as f32,
                1.0 / extent[0].max(1) as f32,
                1.0 / extent[1].max(1) as f32,
            ],
            env: [
                environment_yaw.to_radians().sin(),
                environment_yaw.to_radians().cos(),
                specular_mips as f32,
                0.0,
            ],
            env_tint: [env_tint.x, env_tint.y, env_tint.z, 0.0],
        };
        self.uniforms = Some(uniforms);
    }

    fn uniforms(&self) -> vulkano::buffer::Subbuffer<SsrUbo> {
        self.uniforms
            .clone()
            .expect("the graph scheduled a reflection pass before begin_frame")
    }

    /// Build every level of the depth pyramid in one dispatch. `mips` is one
    /// view per level, because a storage image descriptor takes exactly one.
    pub(super) fn record_hiz(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        depth: Arc<ImageView>,
        mips: &[Arc<ImageView>],
        extent: [u32; 2],
    ) {
        let levels = mips.len() as u32;
        // Every slot in the shader's array has to be written even where the
        // pyramid is shorter than the declaration, so the spares point at level
        // zero. `levels` is what stops anything from storing through them.
        let bound = (0..HIZ_LEVELS as usize).map(|level| mips[level.min(mips.len() - 1)].clone());

        let set = ctx.descriptor_sets.get(
            &ctx.descriptor_set_allocator,
            &self.hiz_pipeline.layout().set_layouts()[0],
            [
                WriteDescriptorSet::image_view_sampler(0, depth, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_array(1, 0, bound),
            ],
        );

        builder
            .bind_pipeline_compute(&self.hiz_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.hiz_pipeline.layout(),
                0,
                &[set],
            )
            .push_constants(
                self.hiz_pipeline.layout(),
                0,
                &HizPush {
                    extent: [extent[0] as i32, extent[1] as i32],
                    levels: levels as i32,
                },
            );

        // SAFETY: one workgroup per HIZ_TILE-sized tile covers the frame, and
        // every store is bounds-checked against the level it writes, so nothing
        // lands outside an image. The descriptors match the shader's layout, and
        // the graph declared every resource this pass touches.
        builder.dispatch([
            extent[0].div_ceil(HIZ_TILE),
            extent[1].div_ceil(HIZ_TILE),
            1,
        ]);
    }

    /// Reduce the lit frame into the pyramid the trace samples cones out of.
    /// Same dispatch shape as the depth pyramid, and for the same reason.
    pub(super) fn record_source(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        source: Arc<ImageView>,
        mips: &[Arc<ImageView>],
    ) {
        let levels = mips.len() as u32;
        let extent = mips[0].image().extent();
        let bound =
            (0..SOURCE_LEVELS as usize).map(|level| mips[level.min(mips.len() - 1)].clone());

        let set = ctx.descriptor_sets.get(
            &ctx.descriptor_set_allocator,
            &self.source_pipeline.layout().set_layouts()[0],
            [
                // Linear, because level 0 is a 2:1 reduction of the frame and
                // one bilinear fetch between four texels *is* their average.
                WriteDescriptorSet::image_view_sampler(0, source, self.linear_clamp.clone()),
                WriteDescriptorSet::image_view_array(1, 0, bound),
            ],
        );

        builder
            .bind_pipeline_compute(&self.source_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.source_pipeline.layout(),
                0,
                &[set],
            )
            .push_constants(
                self.source_pipeline.layout(),
                0,
                &HizPush {
                    extent: [extent[0] as i32, extent[1] as i32],
                    levels: levels as i32,
                },
            );

        // SAFETY: one workgroup per HIZ_TILE-sized tile covers level 0, and
        // every store is bounds-checked against the level it writes, so nothing
        // lands outside an image. The descriptors match the shader's layout, and
        // the graph declared every resource this pass touches.
        builder.dispatch([
            extent[0].div_ceil(HIZ_TILE),
            extent[1].div_ceil(HIZ_TILE),
            1,
        ]);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_trace(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        hiz: Arc<ImageView>,
        depth: Arc<ImageView>,
        normal: Arc<ImageView>,
        material: Arc<ImageView>,
        source: Arc<ImageView>,
        target: Arc<ImageView>,
    ) {
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.trace_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, hiz, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(1, depth, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(2, normal, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(3, material, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(4, source, self.linear_mip.clone()),
                WriteDescriptorSet::image_view(5, target.clone()),
                WriteDescriptorSet::buffer(6, self.uniforms()),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(&self.trace_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.trace_pipeline.layout(),
                0,
                &[set],
            );
        dispatch_over(builder, &target);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_resolve(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        source: Arc<ImageView>,
        rays: Arc<ImageView>,
        depth: Arc<ImageView>,
        normal: Arc<ImageView>,
        material: Arc<ImageView>,
        environment: Arc<ImageView>,
        environment_sampler: Arc<Sampler>,
        target: Arc<ImageView>,
    ) {
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.resolve_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, source, self.linear_clamp.clone()),
                // Fetched at exact texels by the bilateral gather, which does
                // its own interpolation — a linear sampler here would blend
                // across the silhouettes that gather exists to respect.
                WriteDescriptorSet::image_view_sampler(1, rays, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(2, depth, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(3, normal, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(4, material, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view(5, target.clone()),
                WriteDescriptorSet::image_view(6, environment),
                WriteDescriptorSet::sampler(7, environment_sampler),
                WriteDescriptorSet::buffer(8, self.uniforms()),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(&self.resolve_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.resolve_pipeline.layout(),
                0,
                &[set],
            );
        dispatch_over(builder, &target);
    }
}

fn dispatch_over(builder: &mut Recorder, target: &Arc<ImageView>) {
    let extent = target.image().extent();

    // SAFETY: the dispatch covers exactly `extent`, and each shader discards
    // invocations past `imageSize`, so nothing writes outside the image. The
    // descriptors bound above match the shader's layout, and the graph declared
    // every resource this pass touches, so its barriers precede it.
    builder.dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1]);
}

fn build_pipeline(
    ctx: &VkContext,
    entry_point: vulkano::shader::EntryPoint,
) -> Arc<ComputePipeline> {
    let device = &ctx.device;
    let stage = PipelineShaderStageCreateInfo::new(entry_point);
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

mod hiz_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/hiz_build.comp" }
}

mod source_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/color_pyramid.comp" }
}

mod trace_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/ssr_trace.comp" }
}

mod resolve_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/ssr_resolve.comp",
        include: ["shaders"],
    }
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
