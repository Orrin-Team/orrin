use super::{BufferId, SamplerId, TextureViewId};
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

/// Depth testing is always `Less`, the engine's forward-Z convention.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DepthTest {
    pub write: bool,
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
    UniformBuffer(BufferId),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Binding {
    pub binding: u32,
    pub resource: BindingResource,
}
