use super::sync::ImageLayout;
use crate::gfx::format::{Format, SampleCount};

/// Handle to a resource declared on a [`GraphBuilder`](super::GraphBuilder).
/// Valid only against the graph that issued it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ResourceId(pub(super) u32);

impl ResourceId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// How a graph-owned image is sized.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Extent {
    /// Tracks the frame's render extent, so a swapchain resize reallocates it.
    Frame,
    /// The frame's extent halved `n` times — a level of a downsample chain, and
    /// like [`Frame`](Extent::Frame) it follows a resize.
    ///
    /// A shift rather than a ratio because a bloom chain's levels must land on
    /// exactly the sizes successive halvings produce: a level computed any other
    /// way disagrees with the one the pass above it sampled from by a texel at
    /// odd sizes, and the filter taps drift off centre.
    FrameDiv(u32),
    Fixed([u32; 2]),
}

impl Extent {
    pub fn resolve(self, frame: [u32; 2]) -> [u32; 2] {
        match self {
            Extent::Frame => frame,
            // Floored at one: halving an odd extent far enough reaches zero, and
            // a zero-extent image is not one Vulkan will create.
            Extent::FrameDiv(shift) => [(frame[0] >> shift).max(1), (frame[1] >> shift).max(1)],
            Extent::Fixed(extent) => extent,
        }
    }
}

/// Everything about a graph-owned image that the graph cannot derive. Usage
/// flags are absent on purpose: they come from what passes declare, so an image
/// is never created with a capability nothing asked for, and never missing one
/// something did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ImageDesc {
    pub format: Format,
    pub extent: Extent,
    pub samples: SampleCount,
    /// `Some(n)` makes this a 2D *array* image of `n` layers — how a set of
    /// same-sized targets, the shadow cascades, becomes one resource and
    /// therefore one descriptor binding. The graph tracks state per resource,
    /// not per layer, so passes writing different layers are still ordered
    /// against each other.
    ///
    /// An `Option` rather than a count defaulting to 1 because "array of one"
    /// and "not an array" are different images to a shader: the sampler type is
    /// baked into the pipeline, so a one-cascade frame still has to present a
    /// `texture2DArray`. Deriving the view type from the layer count instead
    /// silently produces a plain 2D view whenever the count happens to be one.
    pub array_layers: Option<u32>,
    /// `Some(n)` makes this a *3D* image `n` texels deep, which
    /// [`extent`](Self::extent) then sizes the other two axes of.
    ///
    /// An `Option` for the reason `array_layers` is one, and the distinction is
    /// sharper here: a 3D image and a 2D array image are laid out alike and are
    /// entirely different things to sample. An array is filtered *within* a
    /// layer and never across layers, while a volume is filtered across all
    /// three axes — which is the whole reason the froxel fog wants one, since
    /// the slice boundaries are exactly where a nearest fetch would band.
    ///
    /// Mutually exclusive with `array_layers`: Vulkan has no arrayed 3D image,
    /// and asking for both is a declaration bug rather than something to
    /// silently pick a winner for.
    pub depth: Option<u32>,
    /// How many mip levels the image carries. More than one makes it a pyramid,
    /// which one pass writes whole and later passes sample with `textureLod`.
    ///
    /// It has to be written whole by a single pass, and that is a property of
    /// this graph rather than of pyramids: resources are unversioned, so a chain
    /// of passes each reading level `n - 1` and writing level `n` of the same
    /// resource makes "readers after all writers" point both ways and `compile`
    /// reports a cycle. A chain that wants a pass per level takes the bloom
    /// shape instead — one image per level.
    ///
    /// Clamped at allocation to what the extent can actually carry, so a window
    /// too small for the requested depth gets a shorter pyramid rather than a
    /// failed allocation. A pass that cares reads the count back off the image.
    pub mip_levels: u32,
}

impl ImageDesc {
    pub fn new(format: Format) -> Self {
        Self {
            format,
            extent: Extent::Frame,
            samples: SampleCount::Sample1,
            array_layers: None,
            depth: None,
            mip_levels: 1,
        }
    }

    pub fn samples(mut self, samples: SampleCount) -> Self {
        self.samples = samples;
        self
    }

    /// Size this image independently of the frame's extent, so a swapchain
    /// resize leaves it alone.
    pub fn extent(mut self, extent: Extent) -> Self {
        self.extent = extent;
        self
    }

    /// Make this a 2D array image of `layers` layers, even when `layers` is 1.
    pub fn array_layers(mut self, layers: u32) -> Self {
        self.array_layers = Some(layers);
        self
    }

    /// Make this a 3D image `depth` texels deep, even when `depth` is 1. See
    /// [`depth`](Self::depth) for why that is not the same request as
    /// [`array_layers`](Self::array_layers).
    pub fn depth(mut self, depth: u32) -> Self {
        self.depth = Some(depth.max(1));
        self
    }

    /// Give this image a mip pyramid `levels` deep. See [`mip_levels`](Self::mip_levels)
    /// for why one pass has to write all of them.
    pub fn mip_levels(mut self, levels: u32) -> Self {
        self.mip_levels = levels.max(1);
        self
    }
}

/// The layouts an imported image is in when the frame starts and must be in
/// when it ends. The graph cannot know either — an acquired swapchain image
/// arrives `Undefined` and has to leave as `PresentSrc` — so the importer
/// states them and the compiler emits whatever transitions that implies.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ImportedLayouts {
    pub entry: ImageLayout,
    pub exit: ImageLayout,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ResourceKind {
    /// Allocated and owned by the graph.
    Transient(ImageDesc),
    Imported(ImageDesc, ImportedLayouts),
    /// v1 has no transient buffers: every buffer in a frame so far is either a
    /// per-frame streamed allocation or an asset upload, both of which outlive
    /// the graph. There is deliberately no `create_buffer` to leave dead
    /// allocation paths behind.
    ImportedBuffer,
}

#[derive(Clone, Debug)]
pub(super) struct ResourceDecl {
    pub name: &'static str,
    pub kind: ResourceKind,
}

impl ResourceDecl {
    pub fn is_image(&self) -> bool {
        !matches!(self.kind, ResourceKind::ImportedBuffer)
    }

    pub fn is_imported(&self) -> bool {
        matches!(
            self.kind,
            ResourceKind::Imported(..) | ResourceKind::ImportedBuffer
        )
    }

    /// The layout this resource is in when the frame begins.
    ///
    /// A transient starts `Undefined` every frame: its contents do not survive
    /// a frame boundary. That is a contract, not an approximation — it is what
    /// makes the barrier plan a pure function of the graph's structure, which
    /// is in turn what makes it something CI can assert. A pass that needs last
    /// frame's pixels must import a resource whose lifetime it owns.
    pub fn entry_layout(&self) -> ImageLayout {
        match self.kind {
            ResourceKind::Transient(_) => ImageLayout::Undefined,
            ResourceKind::Imported(_, layouts) => layouts.entry,
            ResourceKind::ImportedBuffer => ImageLayout::Undefined,
        }
    }

    /// The layout the frame must leave this resource in, if any is required.
    pub fn exit_layout(&self) -> Option<ImageLayout> {
        match self.kind {
            ResourceKind::Imported(_, layouts) => Some(layouts.exit),
            ResourceKind::Transient(_) | ResourceKind::ImportedBuffer => None,
        }
    }
}
