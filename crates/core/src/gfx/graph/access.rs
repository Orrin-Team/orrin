use vulkano::buffer::BufferUsage;
use vulkano::image::{ImageLayout, ImageUsage};
use vulkano::sync::{AccessFlags, PipelineStages};

/// What one pass does with one resource.
///
/// This enum is the entire vocabulary a pass author has, and everything the
/// compiler derives comes out of it: execution order, image layouts, barriers,
/// and the `ImageUsage`/`BufferUsage` the resource is created with. That is the
/// point of the graph — a pass cannot write an incorrect barrier, only an
/// incorrect *declaration*, and a wrong declaration fails at compile time
/// (missing usage flag, read before write) rather than as a driver-specific
/// artifact on someone else's GPU.
///
/// Shader accesses are deliberately imprecise about *which* shader stage: they
/// widen to vertex | fragment | compute. Over-synchronising costs a few
/// pipeline bubbles; under-synchronising is a race that reproduces on one
/// vendor. Narrowing this is a later optimisation, and one that shows up as a
/// diff in the golden barrier plan.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Access {
    ColorAttachment,
    /// The 1-sample destination of an MSAA resolve.
    ///
    /// Not a color attachment with a different name: vulkano's
    /// `single_pass_renderpass!` references a resolve target in
    /// `TransferDstOptimal` and requires `TRANSFER_DST` usage, so that is what
    /// the frame really does and what the plan has to say. The *write* is still
    /// a color-attachment write at color-attachment-output — a render-pass
    /// resolve is not a transfer command, only a transfer-shaped layout.
    ResolveAttachment,
    /// Depth tested *and* written.
    DepthAttachment,
    /// Depth tested only. A read-only depth attachment can stay in
    /// `DepthStencilReadOnlyOptimal`, so a later sampling pass needs no
    /// transition.
    DepthAttachmentRead,
    Sampled,
    UniformRead,
    StorageRead,
    StorageWrite,
    /// Vertex or index buffer read by the input assembler.
    VertexRead,
    IndirectRead,
    TransferRead,
    TransferWrite,
}

impl Access {
    /// The layout the resource must be in for this access. Meaningless for
    /// buffers, which have no layout; the compiler skips layout tracking for
    /// them entirely.
    pub fn layout(self) -> ImageLayout {
        match self {
            Access::ColorAttachment => ImageLayout::ColorAttachmentOptimal,
            Access::ResolveAttachment => ImageLayout::TransferDstOptimal,
            Access::DepthAttachment => ImageLayout::DepthStencilAttachmentOptimal,
            Access::DepthAttachmentRead => ImageLayout::DepthStencilReadOnlyOptimal,
            Access::Sampled => ImageLayout::ShaderReadOnlyOptimal,
            // A storage image is readable and writable in the same layout, and
            // `General` is the only layout that allows both.
            Access::StorageRead | Access::StorageWrite => ImageLayout::General,
            Access::TransferRead => ImageLayout::TransferSrcOptimal,
            Access::TransferWrite => ImageLayout::TransferDstOptimal,
            // Buffer-only accesses; never reached for an image.
            Access::UniformRead | Access::VertexRead | Access::IndirectRead => ImageLayout::Undefined,
        }
    }

    pub fn stages(self) -> PipelineStages {
        let shader = PipelineStages::VERTEX_SHADER
            | PipelineStages::FRAGMENT_SHADER
            | PipelineStages::COMPUTE_SHADER;
        match self {
            Access::ColorAttachment | Access::ResolveAttachment => {
                PipelineStages::COLOR_ATTACHMENT_OUTPUT
            }
            // Depth is touched by both test stages: the early test before the
            // fragment shader, the late test after it (and by the render pass's
            // own load/store operations, which Vulkan attributes to the same
            // two stages).
            Access::DepthAttachment | Access::DepthAttachmentRead => {
                PipelineStages::EARLY_FRAGMENT_TESTS | PipelineStages::LATE_FRAGMENT_TESTS
            }
            Access::Sampled | Access::UniformRead | Access::StorageRead | Access::StorageWrite => {
                shader
            }
            Access::VertexRead => PipelineStages::VERTEX_INPUT,
            Access::IndirectRead => PipelineStages::DRAW_INDIRECT,
            Access::TransferRead | Access::TransferWrite => PipelineStages::ALL_TRANSFER,
        }
    }

    pub fn flags(self) -> AccessFlags {
        match self {
            // Blending reads the attachment, and the graph can't see whether a
            // pipeline blends, so a color write always claims the read too.
            Access::ColorAttachment => {
                AccessFlags::COLOR_ATTACHMENT_READ | AccessFlags::COLOR_ATTACHMENT_WRITE
            }
            Access::ResolveAttachment => AccessFlags::COLOR_ATTACHMENT_WRITE,
            Access::DepthAttachment => {
                AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
            }
            Access::DepthAttachmentRead => AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            Access::Sampled => AccessFlags::SHADER_SAMPLED_READ,
            Access::UniformRead => AccessFlags::UNIFORM_READ,
            Access::StorageRead => AccessFlags::SHADER_STORAGE_READ,
            Access::StorageWrite => {
                AccessFlags::SHADER_STORAGE_READ | AccessFlags::SHADER_STORAGE_WRITE
            }
            Access::VertexRead => {
                AccessFlags::VERTEX_ATTRIBUTE_READ | AccessFlags::INDEX_READ
            }
            Access::IndirectRead => AccessFlags::INDIRECT_COMMAND_READ,
            Access::TransferRead => AccessFlags::TRANSFER_READ,
            Access::TransferWrite => AccessFlags::TRANSFER_WRITE,
        }
    }

    pub fn is_write(self) -> bool {
        matches!(
            self,
            Access::ColorAttachment
                | Access::ResolveAttachment
                | Access::DepthAttachment
                | Access::StorageWrite
                | Access::TransferWrite
        )
    }

    /// True for accesses that happen inside a render pass as an attachment.
    /// An image touched *only* this way never leaves the render pass that wrote
    /// it, so the compiler can mark it transient and let the driver keep it in
    /// tile memory.
    pub fn is_attachment(self) -> bool {
        matches!(
            self,
            Access::ColorAttachment
                | Access::ResolveAttachment
                | Access::DepthAttachment
                | Access::DepthAttachmentRead
        )
    }

    pub fn image_usage(self) -> ImageUsage {
        match self {
            Access::ColorAttachment => ImageUsage::COLOR_ATTACHMENT,
            Access::ResolveAttachment => ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
            Access::DepthAttachment | Access::DepthAttachmentRead => {
                ImageUsage::DEPTH_STENCIL_ATTACHMENT
            }
            Access::Sampled => ImageUsage::SAMPLED,
            Access::StorageRead | Access::StorageWrite => ImageUsage::STORAGE,
            Access::TransferRead => ImageUsage::TRANSFER_SRC,
            Access::TransferWrite => ImageUsage::TRANSFER_DST,
            Access::UniformRead | Access::VertexRead | Access::IndirectRead => ImageUsage::empty(),
        }
    }

    pub fn buffer_usage(self) -> BufferUsage {
        match self {
            Access::UniformRead => BufferUsage::UNIFORM_BUFFER,
            Access::StorageRead | Access::StorageWrite => BufferUsage::STORAGE_BUFFER,
            Access::VertexRead => BufferUsage::VERTEX_BUFFER | BufferUsage::INDEX_BUFFER,
            Access::IndirectRead => BufferUsage::INDIRECT_BUFFER,
            Access::TransferRead => BufferUsage::TRANSFER_SRC,
            Access::TransferWrite => BufferUsage::TRANSFER_DST,
            _ => BufferUsage::empty(),
        }
    }
}
