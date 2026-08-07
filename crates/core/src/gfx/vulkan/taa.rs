//! Temporal antialiasing: the camera's subpixel jitter, and the resolve that
//! turns it back into a stable image.
//!
//! Two things live here that do not look alike. The jitter is a property of the
//! *whole frame* — every pass that rasterises geometry must use the same
//! projection or they disagree by a subpixel — so this module produces the
//! [`FrameView`] the frame is drawn with rather than each pass deriving its own
//! matrices. The resolve is an ordinary compute node.
//!
//! The history images are **imported** into the graph, for the reason the
//! exposure buffer is: a transient is `Undefined` at every frame's start by
//! contract, and a history that survives a frame boundary is precisely what this
//! needs. Two allocations, ping-ponged, so the resolve never reads the image it
//! is writing.

use std::sync::Arc;

use glam::{Mat4, Vec2};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{BufferContents, BufferUsage};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};

use crate::scene::{Camera, TaaSettings};

use super::context::VkContext;
use super::hdr::HDR_FORMAT;

/// Side of the resolve's compute workgroup.
const TILE: u32 = 8;

/// Length of the jitter sequence. Halton is not periodic, but the accumulation
/// is: eight frames is long enough to cover a pixel evenly and short enough that
/// a slowly panning camera revisits offsets before the history has forgotten
/// them.
const JITTER_PHASES: u32 = 8;

/// The camera matrices the whole frame is drawn with.
///
/// One struct rather than each pass calling `Camera::view_projection` because
/// the jitter has to be identical across the prepass, the forward pass, the
/// skybox and the debug lines. Two of them disagreeing by a subpixel is not a
/// subtle artifact — the sky would sit still while the geometry shook against
/// it.
#[derive(Clone, Copy)]
pub(super) struct FrameView {
    pub view: Mat4,
    /// Jittered when TAA is on, and the plain projection when it is off.
    pub proj: Mat4,
    pub view_proj: Mat4,
    /// Unjittered, and what `prev_view_proj` is compared against: motion vectors
    /// must not carry the jitter, or the resolve would reproject away the very
    /// offsets it exists to accumulate.
    pub unjittered_view_proj: Mat4,
    /// Last frame's `unjittered_view_proj`; equal to this frame's on the first.
    pub prev_view_proj: Mat4,
    /// The NDC offset baked into `proj`, so a shader can remove it.
    pub jitter: Vec2,
}

/// Uniforms for the resolve. A buffer rather than push constants because two
/// mat4s already fill the guaranteed 128-byte push range on their own.
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct TaaUbo {
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    /// xy = 1/extent, z = history weight, w != 0 to ignore the history.
    params: [f32; 4],
}

pub struct TaaPass {
    pipeline: Arc<ComputePipeline>,
    /// Linear and clamped: the history is resampled at a reprojected coordinate
    /// that almost never lands on a texel centre, and the bicubic fetch in the
    /// shader is built out of bilinear taps.
    linear_clamp: Arc<Sampler>,
    /// Nearest, because velocity and depth are fetched at exact texels and
    /// interpolating either across a silhouette invents a surface.
    nearest_clamp: Arc<Sampler>,
    uniform_allocator: SubbufferAllocator,
    /// Ping-ponged: `frame & 1` is this frame's target and the other is the
    /// history. Empty until the first frame TAA is enabled for.
    history: Option<[Arc<ImageView>; 2]>,
    extent: [u32; 2],
    frame: u64,
    /// Set whenever the history cannot be trusted — first frame, a resize, or
    /// TAA having been off. The resolve then passes the current frame straight
    /// through, which is one aliased frame instead of a frame of garbage.
    reset: bool,
    previous_view_proj: Option<Mat4>,
    settings: TaaSettings,
}

impl TaaPass {
    pub fn new(ctx: &VkContext) -> Self {
        let device = &ctx.device;
        let pipeline = build_pipeline(
            device,
            resolve_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );

        let linear_clamp = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        )
        .unwrap();
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

        let uniform_allocator = SubbufferAllocator::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        Self {
            pipeline,
            linear_clamp,
            nearest_clamp,
            uniform_allocator,
            history: None,
            extent: [0, 0],
            frame: 0,
            reset: true,
            previous_view_proj: None,
            settings: TaaSettings::default(),
        }
    }

    /// Advance one frame: allocate what the resolve will need, pick this frame's
    /// jitter, and hand back the matrices every rasterising pass must use.
    ///
    /// Called whether or not TAA is enabled, because `prev_view_proj` has to
    /// track the camera regardless — otherwise the first frame after it is
    /// switched on reprojects against wherever the camera was when it was
    /// switched off.
    pub(super) fn begin_frame(
        &mut self,
        ctx: &VkContext,
        settings: &TaaSettings,
        camera: &Camera,
        extent: [u32; 2],
    ) -> FrameView {
        self.settings = *settings;

        if settings.enabled {
            if self.history.is_none() || self.extent != extent {
                self.history = Some(allocate_history(ctx, extent));
                self.extent = extent;
                self.reset = true;
            }
            self.frame = self.frame.wrapping_add(1);
        } else {
            // Freed rather than kept: two full-resolution HDR targets is real
            // memory, and anything they held is stale the moment a frame renders
            // without them.
            self.history = None;
            self.reset = true;
        }

        let aspect = extent[0] as f32 / extent[1].max(1) as f32;
        let view = camera.view();
        let projection = camera.projection(aspect);
        let unjittered_view_proj = projection * view;

        let jitter = if settings.enabled {
            jitter_offset(self.frame, extent) * settings.jitter_scale
        } else {
            Vec2::ZERO
        };
        let proj = jittered(projection, jitter);

        let prev_view_proj = self.previous_view_proj.unwrap_or(unjittered_view_proj);
        self.previous_view_proj = Some(unjittered_view_proj);

        FrameView {
            view,
            proj,
            view_proj: proj * view,
            unjittered_view_proj,
            prev_view_proj,
            jitter,
        }
    }

    /// This frame's resolve target, which is next frame's history.
    pub(super) fn output_view(&self) -> Arc<ImageView> {
        self.pair()[(self.frame & 1) as usize].clone()
    }

    fn history_view(&self) -> Arc<ImageView> {
        self.pair()[((self.frame + 1) & 1) as usize].clone()
    }

    fn pair(&self) -> &[Arc<ImageView>; 2] {
        self.history
            .as_ref()
            .expect("the graph scheduled a TAA resolve with no history allocated")
    }

    pub(super) fn record(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        ctx: &VkContext,
        view: &FrameView,
        color: Arc<ImageView>,
        velocity: Arc<ImageView>,
        depth: Arc<ImageView>,
    ) {
        let target = self.output_view();
        let extent = target.image().extent();

        let uniforms = self.uniform_allocator.allocate_sized::<TaaUbo>().unwrap();
        *uniforms.write().unwrap() = TaaUbo {
            inv_view_proj: view.unjittered_view_proj.inverse().to_cols_array_2d(),
            prev_view_proj: view.prev_view_proj.to_cols_array_2d(),
            params: [
                1.0 / extent[0] as f32,
                1.0 / extent[1] as f32,
                self.settings.feedback.clamp(0.0, 0.99),
                self.reset as u32 as f32,
            ],
        };

        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, color, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    self.history_view(),
                    self.linear_clamp.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(2, velocity, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(3, depth, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view(4, target),
                WriteDescriptorSet::buffer(5, uniforms),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                vec![set],
            )
            .unwrap();

        // SAFETY: the dispatch covers exactly the target's extent and the shader
        // discards invocations past `imageSize(u_target)`, so nothing writes
        // outside it. The descriptors bound above match the shader's layout, and
        // the graph declared every resource this pass touches, so its barriers
        // precede it.
        unsafe {
            builder
                .dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1])
                .unwrap()
        };

        // The frame that just resolved is the history the next one reads, so
        // whatever made it untrustworthy is over.
        self.reset = false;
    }
}

/// Where inside the pixel frame `index` samples, as an NDC offset.
///
/// Halton (2,3) rather than a fixed rotated grid: the sequence is
/// low-discrepancy at *every* prefix length, so a history only four frames deep
/// — all a moving camera ever keeps — is still evenly distributed.
fn jitter_offset(index: u64, extent: [u32; 2]) -> Vec2 {
    // From one, because Halton's zeroth element is the origin and a frame with
    // no offset contributes nothing the unjittered image would not have.
    let phase = (index % JITTER_PHASES as u64) as u32 + 1;
    let offset = Vec2::new(halton(phase, 2) - 0.5, halton(phase, 3) - 0.5);
    // NDC spans [-1, 1] across the frame, so one pixel is 2 / extent of it.
    offset * 2.0 / Vec2::new(extent[0].max(1) as f32, extent[1].max(1) as f32)
}

fn halton(mut index: u32, base: u32) -> f32 {
    let mut fraction = 1.0f32;
    let mut result = 0.0f32;
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

/// Offset the projection so the frame rasterises `jitter` NDC units across.
///
/// The third column, not the fourth: clip.xy must move by `jitter * w`, and `w`
/// comes from the view-space z that column multiplies. Adding to the translation
/// column instead would shift the image by a constant in clip space, which is a
/// shear once the perspective divide is done with it.
fn jittered(mut projection: Mat4, jitter: Vec2) -> Mat4 {
    projection.z_axis.x -= jitter.x;
    projection.z_axis.y -= jitter.y;
    projection
}

fn allocate_history(ctx: &VkContext, extent: [u32; 2]) -> [Arc<ImageView>; 2] {
    std::array::from_fn(|_| {
        let image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: HDR_FORMAT,
                extent: [extent[0], extent[1], 1],
                // Written as a storage image by the resolve, read as a sampled
                // one by both the resolve's next frame and everything the frame
                // composites afterwards.
                usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("failed to allocate the TAA history");
        ImageView::new_default(image).unwrap()
    })
}

fn build_pipeline(
    device: &Arc<Device>,
    entry_point: vulkano::shader::EntryPoint,
) -> Arc<ComputePipeline> {
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
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

mod resolve_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/taa_resolve.comp" }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every offset has to land inside the pixel, or the frame is sampling a
    /// neighbour and the resolve accumulates a blur rather than a reconstruction.
    #[test]
    fn the_jitter_stays_within_one_pixel() {
        let extent = [1920, 1080];
        let half = Vec2::new(1.0 / extent[0] as f32, 1.0 / extent[1] as f32);
        for frame in 0..64u64 {
            let offset = jitter_offset(frame, extent);
            assert!(
                offset.x.abs() <= half.x && offset.y.abs() <= half.y,
                "frame {frame} jittered {offset:?} past half a pixel {half:?}",
            );
        }
    }

    /// The point of a sequence is that consecutive frames sample *different*
    /// places; a constant offset would just shift the image.
    #[test]
    fn consecutive_frames_sample_different_points() {
        let extent = [1920, 1080];
        for frame in 0..JITTER_PHASES as u64 {
            assert_ne!(
                jitter_offset(frame, extent),
                jitter_offset(frame + 1, extent),
                "frame {frame} and the next sampled the same point",
            );
        }
    }

    /// The offsets must average out, or the accumulated image sits off-centre
    /// from where the unjittered one would be.
    #[test]
    fn a_full_cycle_is_centred() {
        let extent = [1920, 1080];
        let total: Vec2 = (0..JITTER_PHASES as u64)
            .map(|frame| jitter_offset(frame, extent))
            .sum();
        let mean = total / JITTER_PHASES as f32;
        assert!(mean.length() < 1e-4, "the cycle is biased by {mean:?}");
    }

    /// The jitter must be a clip-space translation of `jitter * w` and nothing
    /// else: the same world point has to keep its depth and its `w`, or the
    /// depth test would disagree between a jittered and an unjittered frame.
    #[test]
    fn jittering_translates_clip_space_and_leaves_depth_alone() {
        use glam::Vec4;

        let camera = Camera::default();
        let projection = camera.projection(16.0 / 9.0);
        let jitter = Vec2::new(0.0013, -0.0021);
        let point = Vec4::new(1.5, -0.5, -7.0, 1.0);

        let plain = projection * point;
        let shaken = jittered(projection, jitter) * point;

        assert!((shaken.z - plain.z).abs() < 1e-6);
        assert!((shaken.w - plain.w).abs() < 1e-6);
        assert!((shaken.x - (plain.x + jitter.x * plain.w)).abs() < 1e-5);
        assert!((shaken.y - (plain.y + jitter.y * plain.w)).abs() < 1e-5);
    }
}
