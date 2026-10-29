use super::{BufferId, PipelineId, SamplerId, TextureViewId};
use crate::gfx::format::{Format, SampleCount};
use crate::gfx::graph::{BufferUsage, ImageUsage};
use crate::gfx::shaders::ShaderId;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TextureDimension {
    D2,
    D2Array,
    D3,
    Cube,
}

/// What a driver needs to create a texture. `usage` is the graph's derived
/// union for graph-owned images, never hand-picked.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TextureDesc {
    pub format: Format,
    pub dimension: TextureDimension,
    pub extent: [u32; 3],
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: SampleCount,
    pub usage: ImageUsage,
    /// Never leaves a render pass, so a tiled GPU may keep it in tile memory.
    /// Only the graph's usage flags decide this.
    pub memoryless: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemoryLocation {
    /// Device-preferred memory the CPU may write once, at creation.
    GpuOnly,
    /// Written by the CPU every frame and read by the GPU.
    Upload,
    Readback,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: BufferUsage,
    pub location: MemoryLocation,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SamplerDesc {
    pub linear: bool,
    pub mipmap_linear: bool,
    pub repeat: bool,
    /// Comparison sampler for shadow lookups.
    pub compare: bool,
    pub anisotropy: Option<f32>,
    /// Upper end of the sampled mip range; `None` is unclamped.
    pub max_lod: Option<f32>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VertexFormat {
    Float2,
    Float3,
    Float4,
    /// Four bytes, each an unsigned-normalised channel.
    Unorm8x4,
}

/// One attribute of a vertex stream. Matched to the shader by name, as the
/// vertex shaders declare their inputs by the names of the structs' fields.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VertexAttribute {
    pub name: &'static str,
    pub format: VertexFormat,
    pub offset: u32,
}

/// One vertex buffer binding of a pipeline. Stream `i` binds at Metal buffer
/// index `VERTEX_STREAM_BASE + i`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VertexStream {
    pub stride: u32,
    pub per_instance: bool,
    pub attributes: &'static [VertexAttribute],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CullMode {
    None,
    Back,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Topology {
    Triangles,
    Lines,
}

/// How a draw combines with what is already in every colour attachment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Blend {
    None,
    /// Source-alpha over.
    Alpha,
    /// Per-channel maximum, for a mask where overlapping draws must not take a
    /// channel back down.
    Max,
    /// Plain addition of source and destination, which is commutative — the
    /// property order-independent transparency rests on.
    Additive,
    /// `dst = dst * (1 - src)`: a running product, commutative for the same
    /// reason, and why the target it multiplies into is cleared to one.
    Revealage,
    /// Premultiplied `over`: the fragment already composited its background, so
    /// what accumulates is radiance that replaced what was behind it.
    Premultiplied,
    /// Premultiplied colour over a target whose alpha is coverage accumulated
    /// front to back: what egui's meshes are drawn with.
    PremultipliedCoverage,
}

/// How a texture's channels are read by a shader, for a format whose layout is
/// not the one the shader wants.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TextureSwizzle {
    Identity,
    /// Two channels read as premultiplied white: red into all of colour, green
    /// into alpha. egui's glyph coverage at half the size of four channels.
    CoverageToWhite,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DepthCompare {
    /// The engine's forward-Z convention.
    Less,
    /// Against a depth another pass already wrote, which is how a one-sample
    /// forward pass gets perfect early-Z from the geometry prepass.
    Equal,
    /// For geometry that sits exactly on the far plane, which the depth clear
    /// would otherwise reject.
    LessOrEqual,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DepthTest {
    pub write: bool,
    pub compare: DepthCompare,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Viewport {
    pub origin: [f32; 2],
    pub extent: [f32; 2],
    pub depth_range: [f32; 2],
}

/// A graphics pipeline as the pilot passes need it: two shaders, the formats
/// they render into, and fixed-function state at its defaults. It grows with the
/// passes that need more, never ahead of them.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GraphicsPipelineDesc<'a> {
    pub vertex: ShaderId,
    pub fragment: ShaderId,
    pub color_formats: &'a [Format],
    pub depth_format: Option<Format>,
    pub vertex_streams: &'a [VertexStream],
    pub cull: CullMode,
    pub depth_test: Option<DepthTest>,
    /// The set whose binding 0 is the material texture array, to be created
    /// partially bound where the device allows it.
    pub partial_texture_set: Option<u32>,
    /// Depth bias, scissor and viewport are dynamic rather than viewport alone.
    pub dynamic_depth_bias: bool,
    pub samples: SampleCount,
    /// Spend the samples on a cutout's edge instead of testing it hard.
    pub alpha_to_coverage: bool,
    /// Samplers baked into the set layout, as `(set, binding, sampler)`. Vulkan
    /// compares immutable samplers by identity, so pipelines that must be set
    /// compatible are built around the very same `SamplerId`.
    pub immutable_samplers: &'a [(u32, u32, SamplerId)],
    pub blend: Blend,
    pub topology: Topology,
    /// Write only the first colour attachment. For a pass that shares a render
    /// pass whose later attachments belong to someone else and whose shader
    /// declares one output: an undeclared output left unmasked is undefined data.
    pub mask_extra_attachments: bool,
    /// Blend per colour attachment, overriding `blend` where non-empty.
    pub attachment_blends: &'a [Blend],
    /// Use this pipeline's whole layout instead of deriving one, for a pipeline
    /// that is handed the descriptor sets another pipeline's pass bound.
    pub layout_from: Option<PipelineId>,
    /// Take the first `n` set layouts from this pipeline verbatim and derive the
    /// rest, for a pipeline that extends another's sets with its own.
    pub layout_prefix_from: Option<(PipelineId, u32)>,
}

impl<'a> GraphicsPipelineDesc<'a> {
    /// A full-screen triangle pass: no vertex input, no culling, no depth.
    pub fn fullscreen(vertex: ShaderId, fragment: ShaderId, color_formats: &'a [Format]) -> Self {
        Self {
            vertex,
            fragment,
            color_formats,
            depth_format: None,
            vertex_streams: &[],
            cull: CullMode::None,
            depth_test: None,
            partial_texture_set: None,
            dynamic_depth_bias: false,
            samples: SampleCount::Sample1,
            alpha_to_coverage: false,
            immutable_samplers: &[],
            blend: Blend::None,
            topology: Topology::Triangles,
            mask_extra_attachments: false,
            attachment_blends: &[],
            layout_from: None,
            layout_prefix_from: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ComputePipelineDesc {
    pub shader: ShaderId,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BindingResource {
    SampledTexture {
        view: TextureViewId,
        sampler: SamplerId,
    },
    StorageBuffer(BufferId),
    StorageTexture(TextureViewId),
    /// A texture bound apart from its sampler, as the shader declares them.
    Texture(TextureViewId),
    Sampler(SamplerId),
    UniformBuffer(BufferId),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Binding {
    pub binding: u32,
    pub resource: BindingResource,
}

/// A rectangle of a depth attachment, in texels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DepthRect {
    pub offset: [u32; 2],
    pub extent: [u32; 2],
}

/// What a render pass does with an attachment's contents on the way in.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum LoadOp {
    /// Keep what an earlier pass left there.
    Load,
    Clear,
    /// Every texel is about to be written, or nothing will read the rest.
    DontCare,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StoreOp {
    Store,
    /// Nothing reads it after the pass, so a tiled GPU need never write it out.
    DontCare,
}

/// One colour attachment of a render pass.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ColorAttachment {
    pub target: TextureViewId,
    pub load: LoadOp,
    pub store: StoreOp,
    pub clear: [f32; 4],
    /// A single-sample texture the samples are resolved into at the end.
    pub resolve: Option<TextureViewId>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DepthAttachment {
    pub target: TextureViewId,
    pub load: LoadOp,
    pub store: StoreOp,
    pub clear: f32,
    /// Tested against and never written: a depth another pass wrote, borrowed.
    pub read_only: bool,
}

/// What a graphics pass renders into, stated once for every driver.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct RenderTargets {
    pub colors: Vec<ColorAttachment>,
    pub depth: Option<DepthAttachment>,
}

/// Something a device may or may not be able to do, asked before a pass
/// commits to a path that needs it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Feature {
    /// Blitting a render target of this format down and sampling the result.
    BackdropCapture(Format),
    /// A two-channel texture read as premultiplied white and alpha, which is
    /// what `TextureSwizzle::CoverageToWhite` needs.
    CoverageTexture,
    /// Sampling block-compressed (BC5, BC7) textures.
    BlockCompression,
}

/// Which kind of buffer a per-frame block is bound as.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlockKind {
    Uniform,
    Storage,
    Index,
    /// Host-visible source for a copy into a buffer that is not itself
    /// host-visible; bound as nothing.
    Staging,
}

/// One run of bytes in a buffer-to-buffer copy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BufferCopy {
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
}
