//! What the previous frame's depth already covered.
//!
//! The compute cull in [`cull`](super::cull) answers "could this be seen from
//! here" with six planes, which is the question a frustum can answer and the
//! only one it can. Most of what a frustum keeps is behind a wall. This module
//! is the second half of the question: a max-depth pyramid over the depth the
//! frame rasterised, and a test in `cull.comp` that drops a box whose nearest
//! point is behind the farthest surface its screen rect covers.
//!
//! # Why the pyramid is a frame old, and why that is the design and not a
//! shortcut
//!
//! The obvious shape is two-phase: draw what was visible last frame, reduce the
//! depth *that* produced, test everything against it, draw the remainder. It
//! cannot be declared here. The cull writes the commands the prepass draws, the
//! prepass writes the depth, and the pyramid reduces it — so a cull that read a
//! pyramid of this frame's depth would close a ring, and
//! [`compile`](crate::gfx::graph::compile) orders every reader after every
//! writer precisely so that it would report the cycle rather than schedule one
//! of the two orders and leave which one to registration order. Two-phase needs
//! versioned resources, which is a change to the compiler and not to this pass.
//!
//! So the cull tests against the pyramid the *last* frame left, and the two
//! allocations ping-pong the way the TAA history does. What that costs is one
//! frame of latency on a disocclusion: something that comes out from behind a
//! wall is drawn one frame late. What it does not cost is correctness in the
//! other direction, which is the one that matters — see the module's tests and
//! `occluded` in `cull.comp` for the four ways the test refuses to answer.
//!
//! # Why it is not [`SsrPass`](super::ssr::SsrPass)'s pyramid
//!
//! That one reduces with `min`, because a reflection ray needs to know it
//! *cannot* have hit anything yet. A visibility test needs the opposite: the
//! frontmost surface that is farthest away is the one that decides whether a
//! box is hidden everywhere its rect lands. Neither pyramid can stand in for the
//! other, and this one is also half the frame at its base and survives the frame
//! boundary, which a transient by contract does not.

use std::sync::Arc;

use vulkano::buffer::BufferContents;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{Image, ImageCreateInfo, ImageSubresourceRange, ImageUsage};
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};

use super::context::VkContext;
use super::convert::vk_format;
use super::record::Recorder;

/// Single-channel float, for the reason the reflection pyramid is: a depth
/// pyramid stores exactly one number, and `R32_SFLOAT` is a storage format every
/// Vulkan implementation supports. Non-linear NDC depth, not linearised — the
/// test is a comparison, and a comparison needs only monotonicity.
pub(super) const OCCLUSION_FORMAT: crate::gfx::format::Format =
    crate::gfx::format::Format::R32Float;

/// How deep the pyramid goes, counting its half-resolution base at level 0.
///
/// Six is what a 64-texel tile of the *depth buffer* reduces to once the base is
/// already halved, and one dispatch cannot go deeper — see `hiz_occlusion.comp`
/// for why it has to be one dispatch. A level-5 texel is 32 base texels across,
/// so the coarsest rect the test can bound is 64 full-resolution pixels square;
/// anything larger on screen is drawn rather than tested, which is the right way
/// round, because something that large is usually the occluder.
pub(super) const OCCLUSION_LEVELS: u32 = 6;

/// Side of the block of the *depth buffer* one workgroup reduces. Fixed by the
/// shader.
const TILE: u32 = 64;

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct BuildPush {
    /// The depth buffer's extent, not the pyramid's.
    extent: [i32; 2],
    levels: i32,
}

/// The two allocations, and the per-level views the build stores through.
struct Pyramid {
    sampled: [Arc<ImageView>; 2],
    mips: [Vec<Arc<ImageView>>; 2],
}

/// What the cull binds and what it is allowed to conclude from it.
#[derive(Clone)]
pub(super) struct OcclusionTest {
    /// The whole pyramid as one sampled view, which is what lets the shader
    /// pick a level with `texelFetch`.
    pub pyramid: Arc<ImageView>,
    pub sampler: Arc<Sampler>,
    /// False when the frame does not test at all, and false for the one frame
    /// after an allocation — the ping-pong has not yet put anything in the half
    /// the cull would read, and a test against uninitialised memory is not a
    /// conservative test.
    pub live: bool,
}

pub(super) struct OcclusionPass {
    pipeline: Arc<ComputePipeline>,
    /// `texelFetch` ignores filtering, so this is a sampler in the sense the
    /// descriptor requires one and in no other. Clamped rather than repeated so
    /// that a coordinate the shader has already bounded cannot wrap if that ever
    /// stops being true.
    nearest_clamp: Arc<Sampler>,
    /// One texel, bound whenever there is no pyramid. The cull's descriptor set
    /// has the binding whether or not the frame tests — a second pipeline for
    /// the sake of one branch that is uniform across the dispatch would buy
    /// nothing and cost a compile — and a descriptor set with a hole in it is
    /// not one Vulkan will accept.
    fallback: Arc<ImageView>,
    pyramid: Option<Pyramid>,
    /// The frame extent `pyramid` was allocated for.
    extent: [u32; 2],
    /// Which half is written this frame. Advanced only on a frame that builds.
    frame: u64,
    /// Frames since the allocation, saturating. The test goes live at one.
    age: u32,
}

impl OcclusionPass {
    pub(super) fn new(ctx: &VkContext) -> Self {
        let stage = PipelineShaderStageCreateInfo::new(
            build_cs::load(ctx.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );
        let layout = PipelineLayout::new(
            ctx.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(ctx.device.clone())
                .unwrap(),
        )
        .unwrap();
        let pipeline = ComputePipeline::new(
            ctx.device.clone(),
            ctx.pipeline_cache(),
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap();

        let nearest_clamp = Sampler::new(
            ctx.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                mipmap_mode: vulkano::image::sampler::SamplerMipmapMode::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                lod: 0.0..=vulkano::image::sampler::LOD_CLAMP_NONE,
                ..SamplerCreateInfo::default()
            },
        )
        .unwrap();

        Self {
            pipeline,
            nearest_clamp,
            fallback: allocate(ctx, [1, 1], 1).0,
            pyramid: None,
            extent: [0, 0],
            frame: 0,
            age: 0,
        }
    }

    /// Advance one frame: allocate or release the pair, and say whether what the
    /// cull is about to read holds a frame of depth.
    pub(super) fn begin_frame(&mut self, ctx: &VkContext, enabled: bool, extent: [u32; 2]) -> bool {
        if !enabled {
            // Released rather than kept, for the reason the TAA history is: two
            // pyramids is real memory, and what they hold is stale the moment a
            // frame renders without them.
            self.pyramid = None;
            self.age = 0;
            return false;
        }

        if self.pyramid.is_none() || self.extent != extent {
            self.pyramid = Some(Pyramid::new(ctx, base_extent(extent)));
            self.extent = extent;
            self.age = 0;
        }
        self.frame = self.frame.wrapping_add(1);
        let live = self.age > 0;
        self.age = self.age.saturating_add(1);
        live
    }

    /// This frame's target, which is next frame's history.
    pub(super) fn hiz_view(&self) -> Arc<ImageView> {
        self.pair().sampled[(self.frame & 1) as usize].clone()
    }

    pub(super) fn history_view(&self) -> Arc<ImageView> {
        self.pair().sampled[((self.frame + 1) & 1) as usize].clone()
    }

    /// What the cull binds. Falls back to the one-texel image whenever there is
    /// no pyramid, so the descriptor set is the same shape either way.
    pub(super) fn test(&self, live: bool) -> OcclusionTest {
        match &self.pyramid {
            Some(_) => OcclusionTest {
                pyramid: self.history_view(),
                sampler: self.nearest_clamp.clone(),
                live,
            },
            None => OcclusionTest {
                pyramid: self.fallback.clone(),
                sampler: self.nearest_clamp.clone(),
                live: false,
            },
        }
    }

    /// Reduce this frame's depth into every level of this frame's half of the
    /// pair, in one dispatch.
    pub(super) fn record(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        depth: Arc<ImageView>,
        extent: [u32; 2],
    ) {
        let mips = &self.pair().mips[(self.frame & 1) as usize];
        // Every slot of the shader's array has to be written even where the
        // pyramid came out shorter than the declaration, so the spares point at
        // level zero. `levels` is what stops anything storing through them.
        let bound =
            (0..OCCLUSION_LEVELS as usize).map(|level| mips[level.min(mips.len() - 1)].clone());

        let set = ctx.descriptor_sets.get(
            &ctx.descriptor_set_allocator,
            &self.pipeline.layout().set_layouts()[0],
            [
                WriteDescriptorSet::image_view_sampler(0, depth, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_array(1, 0, bound),
            ],
        );

        builder
            .bind_pipeline_compute(&self.pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout(),
                0,
                &[set],
            )
            .push_constants(
                self.pipeline.layout(),
                0,
                &BuildPush {
                    extent: [extent[0] as i32, extent[1] as i32],
                    levels: mips.len() as i32,
                },
            );

        // SAFETY: one workgroup per TILE-sized block of the depth buffer covers
        // it, and every store is bounds-checked against the level it writes, so
        // nothing lands outside an image. The descriptors match the shader's
        // layout, and the graph declared every resource this pass touches.
        builder.dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1]);
    }

    fn pair(&self) -> &Pyramid {
        self.pyramid
            .as_ref()
            .expect("the graph scheduled an occlusion pyramid with none allocated")
    }
}

impl Pyramid {
    fn new(ctx: &VkContext, base: [u32; 2]) -> Self {
        let levels = levels_for(base);
        let (first, first_mips) = allocate(ctx, base, levels);
        let (second, second_mips) = allocate(ctx, base, levels);
        Self {
            sampled: [first, second],
            mips: [first_mips, second_mips],
        }
    }
}

/// The pyramid's level 0, which is half the frame each way.
///
/// Halved rather than copied because the base is three quarters of a pyramid's
/// memory and the test never reads it: it picks the coarsest level that still
/// bounds the rect. Floored at one, for the reason
/// [`Extent::FrameDiv`](crate::gfx::graph::Extent::FrameDiv) is — halving a
/// sliver reaches zero, and a zero-extent image is not one Vulkan will create.
/// This must agree with what `frame::declare` imports it as.
fn base_extent(frame: [u32; 2]) -> [u32; 2] {
    [(frame[0] >> 1).max(1), (frame[1] >> 1).max(1)]
}

/// How many levels this base can actually carry. The declaration is a request:
/// halving stops at one texel, and a window small enough to stop early gets a
/// shorter pyramid and a `levels` push that says so.
fn levels_for(base: [u32; 2]) -> u32 {
    let deepest = 32 - base[0].max(base[1]).max(1).leading_zeros();
    OCCLUSION_LEVELS.min(deepest)
}

/// One allocation: the whole-pyramid view the cull samples, and the per-level
/// views the build stores through.
///
/// Two kinds of view over one image because a storage image descriptor takes
/// exactly one mip, so the view that spans the pyramid cannot claim that usage —
/// the same split `GraphImages` makes for the reflection pyramid.
fn allocate(
    ctx: &VkContext,
    extent: [u32; 2],
    levels: u32,
) -> (Arc<ImageView>, Vec<Arc<ImageView>>) {
    let usage = ImageUsage::SAMPLED | ImageUsage::STORAGE;
    let image = Image::new(
        ctx.memory_allocator.clone(),
        ImageCreateInfo {
            format: vk_format(OCCLUSION_FORMAT),
            extent: [extent[0], extent[1], 1],
            usage,
            mip_levels: levels,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .expect("failed to allocate the occlusion depth pyramid");

    let sampled = ImageView::new(
        image.clone(),
        ImageViewCreateInfo {
            view_type: ImageViewType::Dim2d,
            usage: if levels > 1 {
                usage - ImageUsage::STORAGE
            } else {
                usage
            },
            ..ImageViewCreateInfo::from_image(&image)
        },
    )
    .unwrap();

    let mips = (0..levels)
        .map(|level| {
            ImageView::new(
                image.clone(),
                ImageViewCreateInfo {
                    view_type: ImageViewType::Dim2d,
                    subresource_range: ImageSubresourceRange {
                        mip_levels: level..level + 1,
                        ..image.subresource_range()
                    },
                    ..ImageViewCreateInfo::from_image(&image)
                },
            )
            .unwrap()
        })
        .collect();

    (sampled, mips)
}

mod build_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/hiz_occlusion.comp" }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The base has to be what `frame::declare` imports the pair as, or the
    /// graph would barrier an image of one extent and the pass would write one
    /// of another.
    #[test]
    fn the_base_is_the_frame_halved() {
        assert_eq!(base_extent([1920, 1080]), [960, 540]);
        // Floored the way the graph floors it, and by the same rule: halving an
        // odd extent drops the last column.
        assert_eq!(base_extent([1921, 1081]), [960, 540]);
        assert_eq!(base_extent([1, 1]), [1, 1]);
    }

    /// A window too small for the full pyramid gets a shorter one rather than a
    /// mip whose extent Vulkan floored to zero.
    #[test]
    fn a_small_window_carries_a_shorter_pyramid() {
        assert_eq!(levels_for([960, 540]), OCCLUSION_LEVELS);
        // 16 texels across is exactly five halvings.
        assert_eq!(levels_for([16, 9]), 5);
        assert_eq!(levels_for([1, 1]), 1);
    }
}
