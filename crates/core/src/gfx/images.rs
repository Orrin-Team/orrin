//! The graph-owned images as a pass names them: ids, extents and levels, with no
//! API object in sight.
//!
//! A driver allocates the images (Vulkan aliases memory between transients,
//! Metal allocates tracked textures one by one) and hands the renderer this
//! table; everything that records a pass reads it. One set of entries per frame
//! in flight, because a frame split across two queues records its head while the
//! previous frame's tail is still running over the same declarations.

use super::graph::ResourceId;
use super::rhi::TextureViewId;

#[derive(Clone, Debug)]
pub struct ImageEntry {
    pub whole: TextureViewId,
    /// One single-level view per mip, empty for an image of one level.
    pub levels: Vec<TextureViewId>,
    /// One single-layer view per layer of an array image.
    pub layers: Vec<TextureViewId>,
    pub extent: [u32; 2],
    pub mip_levels: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ImageTable {
    sets: Vec<Vec<Option<ImageEntry>>>,
    slot: usize,
}

impl ImageTable {
    pub fn new(sets: Vec<Vec<Option<ImageEntry>>>) -> Self {
        Self { sets, slot: 0 }
    }

    /// Point the accessors at the set frame `frame` records into, taken modulo
    /// however many sets there are.
    pub fn set_slot(&mut self, frame: u64) {
        self.slot = (frame % self.sets.len().max(1) as u64) as usize;
    }

    fn entry(&self, id: ResourceId) -> &ImageEntry {
        self.sets[self.slot][id.index()]
            .as_ref()
            .expect("render graph resource is not a graph-owned image")
    }

    pub fn texture(&self, id: ResourceId) -> TextureViewId {
        self.entry(id).whole
    }

    /// One id per level, which is what a storage descriptor takes; the whole
    /// image for one of a single level.
    pub fn mip_textures(&self, id: ResourceId) -> &[TextureViewId] {
        let entry = self.entry(id);
        if entry.levels.is_empty() {
            std::slice::from_ref(&entry.whole)
        } else {
            &entry.levels
        }
    }

    pub fn layer_texture(&self, id: ResourceId, layer: u32) -> TextureViewId {
        self.entry(id).layers[layer as usize]
    }

    /// The width and height a graph-owned image was allocated at.
    pub fn extent(&self, id: ResourceId) -> [u32; 2] {
        self.entry(id).extent
    }

    /// How many levels the allocated image actually has: the declaration is a
    /// request, and a small window cannot honour it.
    pub fn mip_levels(&self, id: ResourceId) -> u32 {
        self.entry(id).mip_levels
    }
}
