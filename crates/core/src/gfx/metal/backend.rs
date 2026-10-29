//! What the renderer needs from Metal to draw a frame: the images the graph
//! owns, and a target to render into.
//!
//! The graph's transient images are allocated one texture each. Vulkan aliases
//! memory between transients that are never alive together; Metal could with
//! heaps, but heap resources are untracked (trap T2), so until synchronisation
//! is lowered by hand each one is an ordinary tracked texture and the hazards are
//! Metal's own.

use std::collections::HashMap;

use super::MtlDriver;
use crate::gfx::format::{Format, SampleCount};
use crate::gfx::graph::{FrameGraph, ImageUsage};
use crate::gfx::images::{ImageEntry, ImageTable};
use crate::gfx::rhi::{TextureDesc, TextureDimension, TextureViewId};

/// What makes an allocation that one: the cache returns the same textures for
/// a recompile that did not change them, so a structural toggle does not
/// reallocate every image in the frame.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    name: String,
    format: Format,
    extent: [u32; 3],
    usage: u32,
    samples: SampleCount,
    array_layers: u32,
    mip_levels: u32,
    memoryless: bool,
}

pub(crate) struct MtlBackend {
    pub(crate) extent: [u32; 2],
    pub(crate) format: Format,
    /// The image the frame ends in, which a capture reads back.
    pub(crate) target: TextureViewId,
    images: [u32; 2],
    cache: HashMap<Key, ImageEntry>,
}

impl MtlBackend {
    pub(crate) fn new(
        driver: &MtlDriver,
        format: Format,
        extent: [u32; 2],
        graph: &FrameGraph,
    ) -> (Self, ImageTable) {
        let target = driver.create_texture(&TextureDesc {
            format,
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::Sample1,
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::TRANSFER_SRC,
            memoryless: false,
        });
        let mut backend = Self {
            extent,
            format,
            target,
            images: extent,
            cache: HashMap::new(),
        };
        let table = backend.rebuild(driver, graph);
        (backend, table)
    }

    pub(crate) fn is_stale(&self) -> bool {
        self.images != self.extent
    }

    /// Point a table at the images `graph` declares, allocating only the ones
    /// the cache does not already hold.
    pub(crate) fn rebuild(&mut self, driver: &MtlDriver, graph: &FrameGraph) -> ImageTable {
        if self.images != self.extent {
            for entry in self.cache.values() {
                for id in std::iter::once(&entry.whole)
                    .chain(&entry.levels)
                    .chain(&entry.layers)
                {
                    driver.destroy_texture(*id);
                }
            }
            self.cache.clear();
            self.images = self.extent;
        }

        let mut entries: Vec<Option<ImageEntry>> = vec![None; graph.resource_count()];
        for (id, image) in graph.transient_images() {
            let extent = image.desc.extent.resolve(self.extent);
            let depth = image.desc.depth.unwrap_or(1);
            let largest = extent[0].max(extent[1]).max(depth).max(1);
            let mip_levels = image.desc.mip_levels.min(largest.ilog2() + 1).max(1);
            let layers = image.desc.array_layers.unwrap_or(1);
            let key = Key {
                name: graph.resource_name(id).to_string(),
                format: image.desc.format,
                extent: [extent[0], extent[1], depth],
                usage: image.usage.bits(),
                samples: image.desc.samples,
                array_layers: layers,
                mip_levels,
                memoryless: image.memoryless,
            };
            if let Some(entry) = self.cache.get(&key) {
                entries[id.index()] = Some(entry.clone());
                continue;
            }
            let whole = driver.create_texture(&TextureDesc {
                format: image.desc.format,
                dimension: match (image.desc.depth, image.desc.array_layers) {
                    (Some(_), _) => TextureDimension::D3,
                    (None, Some(_)) => TextureDimension::D2Array,
                    (None, None) => TextureDimension::D2,
                },
                extent: [extent[0], extent[1], depth],
                mip_levels,
                array_layers: layers,
                samples: image.desc.samples,
                usage: image.usage,
                memoryless: image.memoryless,
            });
            let levels = if mip_levels > 1 {
                (0..mip_levels)
                    .map(|level| driver.mip_view(whole, level))
                    .collect()
            } else {
                Vec::new()
            };
            let layer_views = (0..image.desc.array_layers.unwrap_or(0))
                .map(|layer| driver.face_view(whole, layer, 0))
                .collect();
            let entry = ImageEntry {
                whole,
                levels,
                layers: layer_views,
                extent: [extent[0], extent[1]],
                mip_levels,
            };
            self.cache.insert(key, entry.clone());
            entries[id.index()] = Some(entry);
        }
        ImageTable::new(vec![entries])
    }
}
