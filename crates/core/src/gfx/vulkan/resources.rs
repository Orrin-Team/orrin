//! Backing store for the compiled graph: the images its passes render into and
//! sample.
//!
//! What each pass *does* with them — which are attachments, how they load and
//! store, what they clear to — is [`rendering`](super::rendering)'s, and was
//! a `RenderPass` and a `Framebuffer` per pass before dynamic rendering.
//!
//! Nothing here decides *what* to allocate, or how much. Format, extent and
//! sample count come from the declaration; usage flags and the memoryless hint
//! are derived by the compiler from what passes said they would do; and which
//! transients are never alive at once, and may therefore be one allocation, is
//! derived from their lifetimes. So an image cannot be created missing a
//! capability something needs, cannot quietly carry one nothing asked for, and
//! cannot share memory with something the frame still wants.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use vulkano::device::Device;
use vulkano::image::sys::RawImage;
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{Image, ImageCreateInfo, ImageSubresourceRange, ImageType, ImageUsage};
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::memory::{
    DeviceMemory, MemoryAllocateInfo, MemoryPropertyFlags, MemoryRequirements, ResourceMemory,
};
use vulkano::sync::Sharing;

use super::convert::{vk_format, vk_image_usage, vk_samples};
use super::driver::VkDriver;
use crate::gfx::graph::{FrameGraph, ResourceId};
use crate::gfx::rhi::TextureViewId;

/// Everything that distinguishes one allocation from another.
///
/// The name is in the key and has to be: two resources can be identical in every
/// other field and still have to be separate images. `subsurface_blur_x` and
/// `subsurface_blur_y` are exactly that pair — same format, same extent, and two
/// allocations precisely because a graph resource is unversioned, so a pass that
/// read and wrote one image would make "readers after writers" point both ways.
/// Keying without the name would silently alias them into one.
#[derive(Clone, PartialEq, Eq, Hash)]
struct ImageKey {
    name: String,
    /// Which frame-in-flight's copy this is. In the key because the copies are
    /// separate allocations by definition — that is the whole point of them —
    /// and a cache that keyed without it would hand every frame the same image
    /// and undo the split. Always zero when the frame is not pipelined.
    slot: usize,
    /// Whether the image was created reachable from both queue families. In the
    /// key because it is baked into the allocation: a graph recompiled with the
    /// split on cannot reuse an image created exclusive.
    concurrent: bool,
    format: vulkano::format::Format,
    extent: [u32; 3],
    usage: ImageUsage,
    samples: vulkano::image::SampleCount,
    array_layers: u32,
    mip_levels: u32,
    memoryless: bool,
    /// The alias group this image was allocated as part of, named by its
    /// members, or `None` for one allocated alone.
    ///
    /// In the key because memory is bound to an image once and for the image's
    /// life: a cached image cannot be re-bound to the block a recompiled graph
    /// would put it in. Keying without this would hand a graph that regrouped
    /// its transients the images from the grouping *before* it, still sharing
    /// memory with resources whose lifetimes now overlap theirs — corruption,
    /// and of the kind that only appears when a checkbox is toggled.
    alias: Option<String>,
}

/// Everything that distinguishes one shared block from another.
///
/// The group's members are in the key, and that is the whole safety argument on
/// this side: a block is only ever handed to images of one alias group, and the
/// compiler has already established that no two members of a group are alive at
/// once. Whether an individual member hits the image cache or is allocated
/// fresh cannot break that — the worst it can do is leave a member on a block of
/// its own, which costs memory rather than correctness.
#[derive(Clone, PartialEq, Eq, Hash)]
struct BlockKey {
    group: String,
    slot: usize,
    /// Both in the key rather than checked, so a regrouped or resized graph
    /// asks for a block that fits instead of being handed one that does not.
    size: vulkano::DeviceSize,
    memory_type_index: u32,
}

/// One allocation's views: the whole image, which is what a pass samples, and
/// for a pyramid one view per level, which is what a storage descriptor takes.
///
/// Built with the image rather than when a pass asks for them, because they
/// change exactly when the image does — and asking per frame was a
/// `vkCreateImageView` per level, per pyramid pass, per frame.
#[derive(Clone)]
struct Views {
    whole: Arc<ImageView>,
    /// The same views as the device knows them, which is what a pass binds.
    whole_id: TextureViewId,
    level_ids: Vec<TextureViewId>,
    /// One single-layer view per layer of an array image, which is what a
    /// framebuffer attachment takes while the whole array is what a shader
    /// samples.
    layer_ids: Vec<TextureViewId>,
}

/// Graph-owned images, indexed by [`ResourceId`].
pub(super) struct GraphImages {
    /// One set of views per frame in flight, indexed `[slot][resource]`.
    ///
    /// One slot unless the frame was split across two queues. With the split,
    /// this frame's compute tail runs while the *next* frame's graphics head
    /// records into the same declarations — which is the overlap the whole
    /// feature is for, and is a race on a single set. Duplicating every
    /// transient rather than only the ones that provably cross is deliberate:
    /// the ones that do are most of them, and "which transients does frame
    /// `n + 1` share with frame `n`" is a derivation whose failure mode is
    /// corruption rather than a compile error.
    views: Vec<Vec<Option<Views>>>,
    /// Which set the frame now recording uses.
    slot: usize,
    /// Every image allocated at the current extent, for any graph, keyed by what
    /// makes it that image.
    ///
    /// A structural toggle — SSAO off, transparency on, the MSAA switch — makes
    /// `ensure_graph` recompile and call straight back here, and without a cache
    /// that means dropping and recreating every image and every framebuffer in
    /// the frame. `declare` and `compile` are device-free and take microseconds;
    /// the allocation is the entire cost, and it is a visible hitch.
    ///
    /// That matters beyond the editor's checkboxes: folding "is this queue
    /// empty?" into the frame's structure means the graph is recompiled whenever
    /// an object drifts on or off screen, which without this would hitch every
    /// few frames.
    ///
    /// Cleared on resize rather than allowed to accumulate: `Extent::Frame`
    /// resolves differently then, so every entry keyed at the old size is dead
    /// weight no lookup will ever hit again.
    cache: HashMap<ImageKey, Views>,
    /// The memory the graph's alias groups share, so that a structural toggle
    /// re-binds the group it already allocated rather than allocating a second
    /// one beside it. Cleared on resize with `cache`, and for the same reason.
    blocks: HashMap<BlockKey, Arc<DeviceMemory>>,
    /// What [`Extent::Frame`](crate::gfx::graph::Extent::Frame) resolved to when
    /// these were allocated, so a resize is a comparison rather than a flag
    /// somebody has to remember to set.
    extent: [u32; 2],
}

impl GraphImages {
    pub fn allocate(
        memory: &Arc<StandardMemoryAllocator>,
        ctx: &super::context::VkContext,
        driver: &VkDriver,
        graph: &FrameGraph,
        extent: [u32; 2],
        slots: usize,
    ) -> Self {
        let mut images = Self {
            views: Vec::new(),
            slot: 0,
            cache: HashMap::new(),
            blocks: HashMap::new(),
            extent,
        };
        images.rebuild(memory, ctx, driver, graph, extent, slots);
        images
    }

    /// Point the accessors at the set frame `frame` records into.
    ///
    /// Taken modulo however many sets there are, so a frame that is not
    /// pipelined always lands on the only one.
    pub fn set_slot(&mut self, frame: u64) {
        self.slot = (frame % self.views.len().max(1) as u64) as usize;
    }

    /// How many frames' worth of images this holds.
    #[allow(dead_code, reason = "counterpart to `set_slot`; reads the same field")]
    pub fn slots(&self) -> usize {
        self.views.len()
    }

    /// Point `views` at the images this graph declares, allocating only the ones
    /// the cache does not already hold.
    pub fn rebuild(
        &mut self,
        memory: &Arc<StandardMemoryAllocator>,
        ctx: &super::context::VkContext,
        driver: &VkDriver,
        graph: &FrameGraph,
        extent: [u32; 2],
        slots: usize,
    ) {
        // A target that is only ever an attachment never leaves the render pass
        // that wrote it, so ask for lazily-allocated memory: on MoltenVK it
        // becomes tile-only and the multisampled HDR and depth targets cost no
        // DRAM at all.
        //
        // "Backends with no lazy memory type fall back silently" is the whole
        // problem with reading this as a general win. *No* desktop driver
        // exposes a lazily-allocated memory type — not AMD's, not NVIDIA's — so
        // there the fallback is ordinary VRAM paying full write-and-resolve
        // traffic, and `msaa_hdr` at 1440p is 118 MB of it. This request is a
        // tiler optimisation that costs nothing to keep and buys nothing off a
        // tiler, which is why `FrameConfig::msaa` now defaults off rather than
        // why this filter changed.
        let lazy = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter {
                preferred_flags: MemoryPropertyFlags::DEVICE_LOCAL
                    | MemoryPropertyFlags::LAZILY_ALLOCATED,
                ..MemoryTypeFilter::PREFER_DEVICE
            },
            ..Default::default()
        };

        if self.extent != extent {
            for views in self.cache.values() {
                driver.destroy_texture(views.whole_id);
                for &level in views.level_ids.iter().chain(&views.layer_ids) {
                    driver.destroy_texture(level);
                }
            }
            self.cache.clear();
            self.blocks.clear();
            self.extent = extent;
        }

        // Which transients the compiler found are never alive at the same time,
        // named by their members so that a group is identified by what is in it
        // rather than by an index a recompile would renumber.
        let groups: Vec<String> = graph
            .alias_groups()
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|&id| graph.resource_name(id))
                    .collect::<Vec<_>>()
                    .join("+")
            })
            .collect();

        let shared_families = ctx.shared_queue_families();
        let mut sets = Vec::with_capacity(slots);
        // Within one graph a key must be claimed at most once, or two resources
        // would be handed the same allocation — see [`ImageKey`] for why that is
        // a correctness bug and not just a surprise. Names are unique per graph,
        // so this only fires on a declaration mistake.
        let mut claimed: std::collections::HashSet<ImageKey> = HashSet::new();
        for slot in 0..slots {
            let mut views = vec![None; graph.resource_count()];
            for (id, image) in graph.transient_images() {
                let extent = image.desc.extent.resolve(extent);
                let mip_levels = image.desc.mip_levels.min(max_mip_levels(extent));
                // Vulkan has no arrayed 3D image, so the two are a declaration
                // mistake together rather than a shape to resolve a winner for.
                assert!(
                    image.desc.depth.is_none() || image.desc.array_layers.is_none(),
                    "render graph: `{}` asked to be both a 3D image and a 2D array",
                    graph.resource_name(id),
                );
                let key = ImageKey {
                    name: graph.resource_name(id).to_string(),
                    slot,
                    // Only where the compiler said both queues reach it, and only
                    // where there is a second family to name — so a single-queue
                    // device creates exactly the images it always did.
                    concurrent: image.concurrent && !shared_families.is_empty(),
                    format: vk_format(image.desc.format),
                    extent: [extent[0], extent[1], image.desc.depth.unwrap_or(1)],
                    usage: vk_image_usage(image.usage),
                    samples: vk_samples(image.desc.samples),
                    array_layers: image.desc.array_layers.unwrap_or(1),
                    mip_levels,
                    memoryless: image.memoryless,
                    alias: image
                        .alias
                        .filter(|_| share_allocations())
                        .map(|group| groups[group as usize].clone()),
                };
                assert!(
                    claimed.insert(key.clone()),
                    "render graph: two resources named `{}` describe the same image",
                    key.name,
                );
                if let Some(view) = self.cache.get(&key) {
                    views[id.index()] = Some(view.clone());
                    continue;
                }

                let create_info = ImageCreateInfo {
                    image_type: match image.desc.depth {
                        Some(_) => ImageType::Dim3d,
                        None => ImageType::Dim2d,
                    },
                    format: vk_format(image.desc.format),
                    extent: [extent[0], extent[1], image.desc.depth.unwrap_or(1)],
                    usage: vk_image_usage(image.usage),
                    samples: vk_samples(image.desc.samples),
                    array_layers: image.desc.array_layers.unwrap_or(1),
                    mip_levels,
                    sharing: if key.concurrent {
                        Sharing::Concurrent(shared_families.iter().copied().collect())
                    } else {
                        Sharing::Exclusive
                    },
                    ..Default::default()
                };

                let allocated = match &key.alias {
                    Some(group) => bind_shared(
                        &ctx.device,
                        memory,
                        &mut self.blocks,
                        group,
                        slot,
                        create_info,
                    ),
                    None => Image::new(
                        memory.clone(),
                        create_info,
                        if image.memoryless {
                            lazy.clone()
                        } else {
                            AllocationCreateInfo::default()
                        },
                    )
                    .map_err(|error| error.to_string()),
                }
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
                        view_type: match (image.desc.depth, image.desc.array_layers) {
                            (Some(_), _) => ImageViewType::Dim3d,
                            (None, Some(_)) => ImageViewType::Dim2dArray,
                            (None, None) => ImageViewType::Dim2d,
                        },
                        // A storage image descriptor takes exactly one level, so a
                        // view spanning the whole pyramid cannot claim that usage —
                        // it is the one a shader samples with `textureLod`, and the
                        // per-level storage views come from `mip_views`.
                        usage: if mip_levels > 1 {
                            vk_image_usage(image.usage) - ImageUsage::STORAGE
                        } else {
                            vk_image_usage(image.usage)
                        },
                        ..ImageViewCreateInfo::from_image(&allocated)
                    },
                )
                .unwrap();
                let levels = if mip_levels > 1 {
                    (0..mip_levels)
                        .map(|level| level_view(&allocated, level))
                        .collect()
                } else {
                    Vec::new()
                };
                let layer_ids = (0..image.desc.array_layers.unwrap_or(0))
                    .map(|layer| driver.adopt_view(layer_view(&allocated, layer)))
                    .collect();
                let entry = Views {
                    layer_ids,
                    whole_id: driver.adopt_view(view.clone()),
                    level_ids: levels
                        .iter()
                        .map(|level| driver.adopt_view(level.clone()))
                        .collect(),
                    whole: view,
                };
                self.cache.insert(key, entry.clone());
                views[id.index()] = Some(entry);
            }
            sets.push(views);
        }
        self.views = sets;
        self.slot = self.slot.min(slots.saturating_sub(1));
    }

    pub fn is_stale(&self, extent: [u32; 2]) -> bool {
        self.extent != extent
    }

    /// The view for a graph-owned image.
    ///
    /// Panics for a resource the graph does not own, which can only happen if a
    /// pass reads a handle from a graph it was not declared against.
    /// The view backing `id`, or `None` for a resource this graph does not own —
    /// an imported image, or a buffer.
    ///
    /// Separate from [`view`](Self::view) because the two callers want opposite
    /// things from a miss: a pass binding a resource it declared has hit a bug,
    /// while the barrier resolver is *asking* which kind of resource this is.
    pub fn try_view(&self, id: ResourceId) -> Option<Arc<ImageView>> {
        self.views[self.slot][id.index()]
            .as_ref()
            .map(|views| views.whole.clone())
    }

    pub fn view(&self, id: ResourceId) -> Arc<ImageView> {
        self.views_of(id).whole.clone()
    }

    /// The image as a pass binds it.
    pub fn texture(&self, id: ResourceId) -> TextureViewId {
        self.views_of(id).whole_id
    }

    /// One id per level, which is what a storage descriptor takes.
    pub fn mip_textures(&self, id: ResourceId) -> &[TextureViewId] {
        let views = self.views_of(id);
        if views.level_ids.is_empty() {
            std::slice::from_ref(&views.whole_id)
        } else {
            &views.level_ids
        }
    }

    fn views_of(&self, id: ResourceId) -> &Views {
        self.views[self.slot][id.index()]
            .as_ref()
            .expect("render graph resource is not a graph-owned image")
    }

    /// The width and height a graph-owned image was allocated at, which is what
    /// a pass drawing into one at less than the frame's extent has to set its
    /// viewport to — the framebuffer's render area follows the attachment, but
    /// the viewport is dynamic state nothing derives.
    pub fn extent(&self, id: ResourceId) -> [u32; 2] {
        let extent = self.view(id).image().extent();
        [extent[0], extent[1]]
    }

    /// A single-layer 2D view of one layer of an array image.
    ///
    /// A framebuffer attachment has to be one layer, while the same image is
    /// sampled as a whole array — so the cascades need both kinds of view over
    /// the same allocation.
    pub fn layer_texture(&self, id: ResourceId, layer: u32) -> TextureViewId {
        self.views_of(id).layer_ids[layer as usize]
    }

    /// How many levels the allocated image actually has, which is what a pass
    /// writing a pyramid must loop over — the declaration is a request, and a
    /// small window cannot honour it.
    pub fn mip_levels(&self, id: ResourceId) -> u32 {
        self.view(id).image().mip_levels()
    }
}

/// One layer of an array image, as a plain 2D view.
fn layer_view(image: &Arc<Image>, layer: u32) -> Arc<ImageView> {
    ImageView::new(
        image.clone(),
        ImageViewCreateInfo {
            view_type: ImageViewType::Dim2d,
            subresource_range: ImageSubresourceRange {
                array_layers: layer..layer + 1,
                ..image.subresource_range()
            },
            ..ImageViewCreateInfo::from_image(image)
        },
    )
    .unwrap()
}

/// Create an image on the block its alias group shares, allocating the block if
/// this is the first member to ask for it.
///
/// The block is one `DeviceMemory` per group rather than a suballocation,
/// because every member of a group has identical create-info and therefore
/// identical memory requirements — so they all sit at offset zero and the block
/// is exactly one image large. A group is a set of equals, not a heap.
///
/// Not `ImageCreateFlags::ALIAS`: that flag is for images that must agree on
/// what the bytes under them *mean*, and these deliberately do not. A transient
/// enters every frame `Undefined`, which is precisely the transition that tells
/// the driver the previous tenant's contents — and its compression metadata —
/// may be thrown away.
fn bind_shared(
    device: &Arc<Device>,
    allocator: &Arc<StandardMemoryAllocator>,
    blocks: &mut HashMap<BlockKey, Arc<DeviceMemory>>,
    group: &str,
    slot: usize,
    create_info: ImageCreateInfo,
) -> Result<Arc<Image>, String> {
    let raw = RawImage::new(device.clone(), create_info).map_err(|error| error.to_string())?;
    let requirements: MemoryRequirements = raw.memory_requirements()[0];
    let memory_type_index = allocator
        .find_memory_type_index(
            requirements.memory_type_bits,
            MemoryTypeFilter::PREFER_DEVICE,
        )
        .ok_or("no device-local memory type accepts this image")?;

    let key = BlockKey {
        group: group.to_string(),
        slot,
        size: requirements.layout.size(),
        memory_type_index,
    };
    let block = match blocks.get(&key) {
        Some(block) => block.clone(),
        None => {
            let block = Arc::new(
                DeviceMemory::allocate(
                    device.clone(),
                    MemoryAllocateInfo {
                        allocation_size: key.size,
                        memory_type_index,
                        ..Default::default()
                    },
                )
                .map_err(|error| error.to_string())?,
            );
            blocks.insert(key, block.clone());
            block
        }
    };

    // SAFETY: this block is only ever bound to members of one alias group, and
    // the compiler established that no two of those are alive in the same slot
    // of the frame — so the resources that share it are never in use at once.
    // The hand-over is synchronised by the barrier `State::aliased` derives,
    // which sources each member's first access from the last access of the
    // member before it.
    let resource = unsafe { ResourceMemory::new_dedicated_unchecked(block) };
    raw.bind_memory([resource])
        .map(Arc::new)
        .map_err(|(error, ..)| error.to_string())
}

/// Whether transients the compiler grouped actually share their memory.
///
/// On unless `ORRIN_ALIAS=0`, which is the control the saving is measured
/// against: the plan is identical either way — the compiler derives the
/// hand-over dependency whether or not the executor takes the sharing up — so
/// this changes how much VRAM the frame holds and nothing else about it.
fn share_allocations() -> bool {
    static SHARE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SHARE.get_or_init(|| !std::env::var("ORRIN_ALIAS").is_ok_and(|value| value.trim() == "0"))
}

/// A 2D view of mip `level` of `image` alone.
fn level_view(image: &Arc<Image>, level: u32) -> Arc<ImageView> {
    let info = ImageViewCreateInfo {
        view_type: ImageViewType::Dim2d,
        subresource_range: ImageSubresourceRange {
            mip_levels: level..level + 1,
            ..image.subresource_range()
        },
        ..ImageViewCreateInfo::from_image(image)
    };
    ImageView::new(image.clone(), info).unwrap()
}

/// The deepest pyramid an extent can carry: halving stops at one texel.
fn max_mip_levels(extent: [u32; 2]) -> u32 {
    32 - extent[0].max(extent[1]).max(1).leading_zeros()
}

#[cfg(test)]
mod tests {
    use crate::gfx::format::Format;
    use crate::gfx::graph::ImageLayout;
    use vulkano::VulkanLibrary;
    use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};

    use super::*;
    use crate::gfx::graph::{Access, GraphBuilder, ImageDesc, PassKind, compile};
    use crate::gfx::vulkan::context::VkContext;

    /// A pyramid written per level by storage and then sampled whole: the
    /// shape the SSR and refraction pyramids take. The import is what keeps
    /// the passes, and so the pyramid, from being culled.
    fn pyramid_graph(levels: u32) -> (FrameGraph, ResourceId) {
        let mut builder = GraphBuilder::new();
        let pyramid = builder.create_image(
            "pyramid",
            ImageDesc::new(Format::Rgba16Float).mip_levels(levels),
        );
        let target = builder.import_image(
            "target",
            ImageDesc::new(Format::Rgba16Float),
            ImageLayout::Undefined,
            ImageLayout::General,
        );
        builder
            .pass("build", PassKind::Compute)
            .access(pyramid, Access::StorageWrite)
            .build();
        builder
            .pass("read", PassKind::Compute)
            .access(pyramid, Access::Sampled)
            .access(target, Access::StorageWrite)
            .build();
        (compile(builder).expect("a pyramid graph"), pyramid)
    }

    /// The per-level storage views are part of the allocation, not of the
    /// frame: every recording sees the same views, and a recompile that hits the
    /// image cache keeps them. Creating them per frame was a `vkCreateImageView`
    /// per level per pass per frame for views that only change on resize.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn a_pyramid_keeps_one_view_per_level_across_frames_and_recompiles() {
        let instance = Instance::new(
            VulkanLibrary::new().expect("a Vulkan library"),
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .expect("a headless instance");
        let ctx = Arc::new(VkContext::new(&instance, None));
        let driver = VkDriver::new(ctx.clone(), 2);
        let (graph, pyramid) = pyramid_graph(5);

        let mut images =
            GraphImages::allocate(&ctx.memory_allocator, &ctx, &driver, &graph, [64, 64], 1);
        let first: Vec<_> = images.mip_textures(pyramid).to_vec();
        assert_eq!(first.len(), 5);
        for (level, id) in first.iter().enumerate() {
            let level = level as u32;
            assert_eq!(
                driver.vk_view(*id).subresource_range().mip_levels,
                level..level + 1
            );
        }

        let (recompiled, pyramid) = pyramid_graph(5);
        images.rebuild(
            &ctx.memory_allocator,
            &ctx,
            &driver,
            &recompiled,
            [64, 64],
            1,
        );
        for (before, after) in first.iter().zip(images.mip_textures(pyramid)) {
            assert_eq!(before, after, "a cache hit rebuilt a mip view");
        }
    }
}
