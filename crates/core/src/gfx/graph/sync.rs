//! The synchronisation vocabulary a pass access is derived into: layouts, stages,
//! access flags and the usage a resource is created with.
//!
//! Engine-owned so the graph compiles without naming a graphics API. The flags
//! carry the Vulkan synchronisation model's names because that is the finest
//! model any driver has to express; one with less to say (Metal has no layouts)
//! lowers the plan to what it can. Only the members the compiler can produce
//! exist — a driver converts each by name, and a member nothing derives is a
//! conversion nobody has tested.

use bitflags::bitflags;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ImageLayout {
    Undefined,
    General,
    ColorAttachmentOptimal,
    DepthStencilAttachmentOptimal,
    DepthStencilReadOnlyOptimal,
    ShaderReadOnlyOptimal,
    TransferSrcOptimal,
    TransferDstOptimal,
    PresentSrc,
}

impl ImageLayout {
    pub const ALL: [ImageLayout; 9] = [
        ImageLayout::Undefined,
        ImageLayout::General,
        ImageLayout::ColorAttachmentOptimal,
        ImageLayout::DepthStencilAttachmentOptimal,
        ImageLayout::DepthStencilReadOnlyOptimal,
        ImageLayout::ShaderReadOnlyOptimal,
        ImageLayout::TransferSrcOptimal,
        ImageLayout::TransferDstOptimal,
        ImageLayout::PresentSrc,
    ];
}

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct PipelineStages: u32 {
        const TOP_OF_PIPE = 1 << 0;
        const DRAW_INDIRECT = 1 << 1;
        const VERTEX_INPUT = 1 << 2;
        const VERTEX_SHADER = 1 << 3;
        const FRAGMENT_SHADER = 1 << 4;
        const EARLY_FRAGMENT_TESTS = 1 << 5;
        const LATE_FRAGMENT_TESTS = 1 << 6;
        const COLOR_ATTACHMENT_OUTPUT = 1 << 7;
        const COMPUTE_SHADER = 1 << 8;
        const ALL_TRANSFER = 1 << 9;
        const BOTTOM_OF_PIPE = 1 << 10;
        const HOST = 1 << 11;
        const ALL_COMMANDS = 1 << 12;
    }
}

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct AccessFlags: u32 {
        const INDIRECT_COMMAND_READ = 1 << 0;
        const INDEX_READ = 1 << 1;
        const VERTEX_ATTRIBUTE_READ = 1 << 2;
        const UNIFORM_READ = 1 << 3;
        const SHADER_SAMPLED_READ = 1 << 4;
        const SHADER_STORAGE_READ = 1 << 5;
        const SHADER_STORAGE_WRITE = 1 << 6;
        const COLOR_ATTACHMENT_READ = 1 << 7;
        const COLOR_ATTACHMENT_WRITE = 1 << 8;
        const DEPTH_STENCIL_ATTACHMENT_READ = 1 << 9;
        const DEPTH_STENCIL_ATTACHMENT_WRITE = 1 << 10;
        const TRANSFER_READ = 1 << 11;
        const TRANSFER_WRITE = 1 << 12;
    }
}

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct ImageUsage: u32 {
        const TRANSFER_SRC = 1 << 0;
        const TRANSFER_DST = 1 << 1;
        const SAMPLED = 1 << 2;
        const STORAGE = 1 << 3;
        const COLOR_ATTACHMENT = 1 << 4;
        const DEPTH_STENCIL_ATTACHMENT = 1 << 5;
        const TRANSIENT_ATTACHMENT = 1 << 6;
    }
}

bitflags! {
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct BufferUsage: u32 {
        const TRANSFER_SRC = 1 << 0;
        const TRANSFER_DST = 1 << 1;
        const UNIFORM_BUFFER = 1 << 2;
        const STORAGE_BUFFER = 1 << 3;
        const INDEX_BUFFER = 1 << 4;
        const VERTEX_BUFFER = 1 << 5;
        const INDIRECT_BUFFER = 1 << 6;
    }
}
