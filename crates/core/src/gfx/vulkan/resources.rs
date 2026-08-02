//! Backing store for the compiled graph: the images it owns, and the
//! framebuffers its passes draw into.
//!
//! Nothing here decides *what* to allocate. Format, extent and sample count come
//! from the declaration; usage flags and the memoryless hint are derived by the
//! compiler from what passes said they would do. So an image cannot be created
//! missing a capability something needs, and cannot quietly carry one nothing
//! asked for.

use std::sync::Arc;

use vulkano::command_buffer::RenderPassBeginInfo;
use vulkano::format::ClearValue;
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{Image, ImageCreateInfo, ImageSubresourceRange, ImageType};
use vulkano::memory::MemoryPropertyFlags;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo};

use crate::gfx::graph::{FrameGraph, ResourceId};

use super::forward::ForwardPass;
use super::frame::{Frame, PassBody};
use super::shadow::ShadowPass;
use super::ssao::SsaoPass;

/// Graph-owned images, indexed by [`ResourceId`].
pub(super) struct GraphImages {
    views: Vec<Option<Arc<ImageView>>>,
    /// What [`Extent::Frame`](crate::gfx::graph::Extent::Frame) resolved to when
    /// these were allocated, so a resize is a comparison rather than a flag
    /// somebody has to remember to set.
    extent: [u32; 2],
}

impl GraphImages {
    pub fn allocate(
        memory: &Arc<StandardMemoryAllocator>,
        graph: &FrameGraph,
        extent: [u32; 2],
    ) -> Self {
        // A target that is only ever an attachment never leaves the render pass
        // that wrote it, so ask for lazily-allocated memory: on MoltenVK it
        // becomes tile-only and the 4x MSAA HDR and depth targets cost no DRAM
        // at all. Backends with no lazy memory type fall back silently.
        let lazy = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter {
                preferred_flags: MemoryPropertyFlags::DEVICE_LOCAL
                    | MemoryPropertyFlags::LAZILY_ALLOCATED,
                ..MemoryTypeFilter::PREFER_DEVICE
            },
            ..Default::default()
        };

        let mut views = vec![None; graph.resource_count()];
        for (id, image) in graph.transient_images() {
            let extent = image.desc.extent.resolve(extent);
            let allocated = Image::new(
                memory.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: image.desc.format,
                    extent: [extent[0], extent[1], 1],
                    usage: image.usage,
                    samples: image.desc.samples,
                    array_layers: image.desc.array_layers.unwrap_or(1),
                    ..Default::default()
                },
                if image.memoryless {
                    lazy.clone()
                } else {
                    AllocationCreateInfo::default()
                },
            )
            .unwrap_or_else(|error| {
                panic!(
                    "render graph: could not allocate `{}` ({:?}, {:?}): {error}",
                    graph.resource_name(id),
                    image.desc.format,
                    image.usage,
                )
            });

            // The view type comes from the declaration, not from the layer
            // count: an array image of one layer must still be viewed as an
            // array, because the sampler type is compiled into the pipeline and
            // cannot depend on how many cascades the settings happen to ask for.
            let view = ImageView::new(
                allocated.clone(),
                ImageViewCreateInfo {
                    view_type: if image.desc.array_layers.is_some() {
                        ImageViewType::Dim2dArray
                    } else {
                        ImageViewType::Dim2d
                    },
                    ..ImageViewCreateInfo::from_image(&allocated)
                },
            )
            .unwrap();
            views[id.index()] = Some(view);
        }
        Self { views, extent }
    }

    pub fn is_stale(&self, extent: [u32; 2]) -> bool {
        self.extent != extent
    }

    /// The view for a graph-owned image.
    ///
    /// Panics for a resource the graph does not own, which can only happen if a
    /// pass reads a handle from a graph it was not declared against.
    pub fn view(&self, id: ResourceId) -> Arc<ImageView> {
        self.views[id.index()]
            .clone()
            .expect("render graph resource is not a graph-owned image")
    }

    /// A single-layer 2D view of one layer of an array image.
    ///
    /// A framebuffer attachment has to be one layer, while the same image is
    /// sampled as a whole array — so the cascades need both kinds of view over
    /// the same allocation.
    pub fn layer_view(&self, id: ResourceId, layer: u32) -> Arc<ImageView> {
        let image = self.view(id).image().clone();

        let info = ImageViewCreateInfo {
            view_type: ImageViewType::Dim2d,
            subresource_range: ImageSubresourceRange {
                array_layers: layer..layer + 1,
                ..image.subresource_range()
            },
            ..ImageViewCreateInfo::from_image(&image)
        };

        ImageView::new(image, info).unwrap()
    }
}

/// One framebuffer per pass that draws into graph-owned images, indexed by
/// `PassId`.
///
/// The tonemap and overlay passes have none: they target the acquired swapchain
/// image, which changes every frame, so their framebuffers are the ones the
/// swapchain builds once per image and indexes by acquire.
pub(super) struct PassFramebuffers {
    framebuffers: Vec<Option<Arc<Framebuffer>>>,
}

impl PassFramebuffers {
    pub fn build(
        frame: &Frame,
        images: &GraphImages,
        forward: &ForwardPass,
        ssao: &SsaoPass,
        shadow: &ShadowPass,
    ) -> Self {
        let ids = frame.ids;
        // Only scheduled passes get one. A culled pass's images are never
        // allocated, so building its framebuffer would ask for a view that does
        // not exist.
        let mut framebuffers = vec![None; frame.bodies.len()];
        for &pass_id in frame.graph.order() {
            let body = frame.bodies[pass_id.index()];
            framebuffers[pass_id.index()] = (|| {
                let (render_pass, attachments) = match body {
                    PassBody::SsaoPrepass => {
                        let ssao_ids = ids.ssao.expect("SSAO pass without SSAO resources");
                        (
                            ssao.prepass_rp.clone(),
                            vec![images.view(ssao_ids.normal), images.view(ssao_ids.depth)],
                        )
                    }
                    PassBody::SsaoResolve => {
                        let ssao_ids = ids.ssao.expect("SSAO pass without SSAO resources");
                        (ssao.ssao_rp.clone(), vec![images.view(ssao_ids.raw_ao)])
                    }
                    PassBody::SsaoBlur => {
                        let ssao_ids = ids.ssao.expect("SSAO pass without SSAO resources");
                        (ssao.blur_rp.clone(), vec![images.view(ssao_ids.ao)])
                    }
                    PassBody::Forward => (
                        forward.render_pass.clone(),
                        vec![
                            images.view(ids.msaa_hdr),
                            images.view(ids.msaa_depth),
                            images.view(ids.hdr_color),
                        ],
                    ),
                    // One layer of the cascade array, not the array view: a
                    // framebuffer attachment is a single layer, and handing it
                    // the whole array would make every cascade clear and
                    // overwrite the same pixels without erroring.
                    PassBody::ShadowCascade(cascade) => (
                        shadow.render_pass.clone(),
                        vec![images.layer_view(
                            ids.shadows.expect("shadow pass without a shadow image"),
                            cascade,
                        )],
                    ),
                    PassBody::Tonemap | PassBody::Overlay => return None,
                };
                Some(
                    Framebuffer::new(
                        render_pass,
                        FramebufferCreateInfo {
                            attachments,
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                )
            })();
        }
        Self { framebuffers }
    }

    pub fn get(&self, pass: usize) -> Option<Arc<Framebuffer>> {
        self.framebuffers[pass].clone()
    }
}

/// How each pass's attachments start the frame. `None` means "leave it": for the
/// resolve target and the swapchain image, every pixel is written anyway, so
/// clearing first is bandwidth spent on values nothing reads.
pub(super) fn clear_values(body: PassBody) -> Vec<Option<ClearValue>> {
    match body {
        // Flat +Z in the normal buffer, far in depth.
        PassBody::SsaoPrepass => vec![Some([0.5, 0.5, 1.0, 0.0].into()), Some(1.0.into())],
        // 1.0 = fully unoccluded, so an untouched pixel darkens nothing.
        PassBody::SsaoResolve | PassBody::SsaoBlur => vec![Some([1.0, 0.0, 0.0, 0.0].into())],
        PassBody::Forward => vec![Some([0.02, 0.02, 0.03, 1.0].into()), Some(1.0.into()), None],
        PassBody::Tonemap => vec![None],
        PassBody::Overlay => Vec::new(),
        PassBody::ShadowCascade(_) => vec![Some(1.0.into())],
    }
}

pub(super) fn begin_info(framebuffer: Arc<Framebuffer>, body: PassBody) -> RenderPassBeginInfo {
    RenderPassBeginInfo {
        clear_values: clear_values(body),
        ..RenderPassBeginInfo::framebuffer(framebuffer)
    }
}
