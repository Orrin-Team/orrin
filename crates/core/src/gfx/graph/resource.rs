use vulkano::format::Format;
use vulkano::image::{ImageLayout, SampleCount};

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
    Fixed([u32; 2]),
}

impl Extent {
    pub fn resolve(self, frame: [u32; 2]) -> [u32; 2] {
        match self {
            Extent::Frame => frame,
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
}

impl ImageDesc {
    pub fn new(format: Format) -> Self {
        Self {
            format,
            extent: Extent::Frame,
            samples: SampleCount::Sample1,
        }
    }

    pub fn samples(mut self, samples: SampleCount) -> Self {
        self.samples = samples;
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
