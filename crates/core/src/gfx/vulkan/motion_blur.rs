//! Motion blur: a tile-max velocity pyramid, then a reconstruction gather.
//!
//! The two tile passes exist to answer one question cheaply — "is anything near
//! this pixel moving fast?" — because the honest answer is what separates blur
//! that bleeds past a silhouette from blur that stops dead at one. A pixel of
//! still background beside a fast object has no motion of its own, so a gather
//! driven by per-pixel velocity leaves it sharp and the object cut out against
//! it. Driving the gather by the neighbourhood's strongest motion instead, and
//! letting each tap argue for itself, is the reconstruction of McGuire et al.
//! (2012) that both Unreal and HDRP ship.
//!
//! They cost almost nothing: the first pass reads each velocity texel once
//! across the whole frame, and the second runs over an image a two-hundred-and-
//! fifty-sixth the size.

use std::sync::Arc;

use vulkano::buffer::allocator::SubbufferAllocatorCreateInfo;
use vulkano::buffer::{BufferContents, BufferUsage};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};

use crate::scene::{Camera, MotionBlurSettings};

use super::context::VkContext;
use super::record::{Arena, Recorder};
use super::taa::FrameView;

/// Side of the compute workgroup for every pass here.
const TILE: u32 = 8;

/// How many times the frame is halved to reach the tile grid, so a tile is 16
/// pixels across.
///
/// A shift because that is how the graph sizes an image, and 16 rather than the
/// 20 the paper uses for the same reason: the level has to land on exactly the
/// size successive halvings produce, or the tile a pixel computes for itself is
/// not the texel it reads.
pub(super) const TILE_SHIFT: u32 = 4;

/// The longest streak the reconstruction will draw, in pixels.
///
/// Twice a tile, and that factor is load-bearing rather than generous: the
/// gather spreads a vector half its length either side of centre, so a
/// two-tile vector reaches exactly one tile — which is precisely the ring the
/// neighbour-max pass dilates over. Raising this without widening that dilation
/// would let a fast object's blur cross into a tile that never heard about it,
/// and the tile grid would print itself on the frame.
const MAX_BLUR_PIXELS: f32 = (2 << TILE_SHIFT) as f32;

/// Uniforms shared by the tile pass and the gather. A buffer rather than push
/// constants because two mat4s alone fill the guaranteed 128-byte push range.
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct MotionBlurUbo {
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    /// x = shutter fraction, y = max blur radius in pixels, z = near, w = far.
    params: [f32; 4],
}

pub struct MotionBlurPass {
    tile_max_pipeline: Arc<ComputePipeline>,
    neighbour_max_pipeline: Arc<ComputePipeline>,
    gather_pipeline: Arc<ComputePipeline>,
    /// Nearest throughout. Velocity and depth interpolated across a silhouette
    /// describe a surface that is not there, and a tile maximum averaged with
    /// its neighbour is no longer a maximum.
    nearest_clamp: Arc<Sampler>,
    uniform_allocator: Arena,
    settings: MotionBlurSettings,
    near: f32,
    far: f32,
}

impl MotionBlurPass {
    pub fn new(ctx: &VkContext) -> Self {
        let device = &ctx.device;
        let tile_max_pipeline = build_pipeline(
            ctx,
            tile_max_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );
        let neighbour_max_pipeline = build_pipeline(
            ctx,
            neighbour_max_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );
        let gather_pipeline = build_pipeline(
            ctx,
            gather_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );

        let nearest_clamp = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::default()
            },
        )
        .unwrap();

        let uniform_allocator = Arena::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        Self {
            tile_max_pipeline,
            neighbour_max_pipeline,
            gather_pipeline,
            nearest_clamp,
            uniform_allocator,
            settings: MotionBlurSettings::default(),
            near: 0.1,
            far: 1000.0,
        }
    }

    pub fn begin_frame(&mut self, settings: &MotionBlurSettings, camera: &Camera) {
        self.settings = *settings;
        self.near = camera.near;
        self.far = camera.far;
    }

    /// The uniforms both the tile pass and the gather bind.
    ///
    /// One description of the frame's motion, allocated per pass rather than
    /// shared, because the two record at different points in the frame — but
    /// derived from the same [`FrameView`], so they cannot disagree about where
    /// the sky went.
    fn uniforms(&self, view: &FrameView) -> vulkano::buffer::Subbuffer<MotionBlurUbo> {
        let uniforms = self.uniform_allocator.allocate_sized::<MotionBlurUbo>();
        *uniforms.write().unwrap() = MotionBlurUbo {
            // Unjittered, so the sky's reprojection agrees with the motion
            // vectors the prepass wrote — those are unjittered too.
            inv_view_proj: view.unjittered_view_proj.inverse().to_cols_array_2d(),
            prev_view_proj: view.prev_view_proj.to_cols_array_2d(),
            params: [
                self.settings.shutter_fraction(),
                MAX_BLUR_PIXELS,
                self.near,
                self.far,
            ],
        };
        uniforms
    }

    pub(super) fn record_tile_max(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        view: &FrameView,
        velocity: Arc<ImageView>,
        depth: Arc<ImageView>,
        target: Arc<ImageView>,
    ) {
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.tile_max_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, velocity, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(1, depth, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view(2, target.clone()),
                WriteDescriptorSet::buffer(3, self.uniforms(view)),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(&self.tile_max_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.tile_max_pipeline.layout(),
                0,
                &[set],
            );
        dispatch_over(builder, &target);
    }

    pub fn record_neighbour_max(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        tiles: Arc<ImageView>,
        target: Arc<ImageView>,
    ) {
        let set = ctx.descriptor_sets.get(
            &ctx.descriptor_set_allocator,
            &self.neighbour_max_pipeline.layout().set_layouts()[0],
            [
                WriteDescriptorSet::image_view_sampler(0, tiles, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view(1, target.clone()),
            ],
        );

        builder
            .bind_pipeline_compute(&self.neighbour_max_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.neighbour_max_pipeline.layout(),
                0,
                &[set],
            );
        dispatch_over(builder, &target);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_gather(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        view: &FrameView,
        color: Arc<ImageView>,
        velocity: Arc<ImageView>,
        depth: Arc<ImageView>,
        neighbour: Arc<ImageView>,
        target: Arc<ImageView>,
    ) {
        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.gather_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, color, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(1, velocity, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(2, depth, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(3, neighbour, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view(4, target.clone()),
                WriteDescriptorSet::buffer(5, self.uniforms(view)),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(&self.gather_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.gather_pipeline.layout(),
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

mod tile_max_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/motion_blur_tile_max.comp" }
}

mod neighbour_max_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/motion_blur_neighbour_max.comp" }
}

mod gather_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/motion_blur_gather.comp" }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tile shift, the blur cap and the shader's `TILE` are one fact spread
    /// across three files, and the reconstruction is only seamless while they
    /// agree: the gather reaches half a vector, so the cap has to be exactly the
    /// two tiles the neighbour-max pass dilates one ring over.
    #[test]
    fn the_blur_cap_reaches_exactly_the_dilated_ring() {
        let tile = (1u32 << TILE_SHIFT) as f32;
        assert_eq!(tile, 16.0, "the shaders declare a 16-pixel tile");
        assert_eq!(MAX_BLUR_PIXELS * 0.5, tile);
    }
}
