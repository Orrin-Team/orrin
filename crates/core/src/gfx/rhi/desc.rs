use crate::gfx::format::{Format, SampleCount};
use crate::gfx::graph::{BufferUsage, ImageUsage};

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
}

/// One vertex buffer binding of a pipeline. Stream `i` binds at Metal buffer
/// index `VERTEX_STREAM_BASE + i`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VertexStream {
    pub stride: u32,
    pub per_instance: bool,
}
