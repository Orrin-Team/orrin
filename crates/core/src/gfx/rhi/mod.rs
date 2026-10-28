//! The renderer's view of a graphics device: typed handles and the descriptions
//! that create them, with no API type anywhere.
//!
//! **Handles, not objects.** A resource is a dense `u32` newtype into a pool the
//! driver owns, freed through a frame-delayed queue (`pool.rs`). That matches the
//! engine's rule that shared state is plain data addressed by ID, and it is what
//! lets one pass module record against Vulkan or Metal without being generic over
//! either: it names a `TextureId`, never an `Arc<Image>` or an `MTLTexture`.
//!
//! **Enum dispatch, not traits.** A `Device` wraps `enum { Vulkan(..), Metal(..) }`
//! and so does the encoder; the cost is one predictable `match` per recorded
//! command. A generic `Renderer<R: Rhi>` would monomorphise every pass twice in
//! the largest crate, and a `Box<dyn Driver>` pays the same dispatch with less
//! inlining.
//!
//! Only the device-free surface exists so far: handles, descriptions and the pool.
//! The drivers and the encoder arrive with the passes that need them.

mod desc;
mod pool;

pub(crate) use pool::Pool;

pub use desc::{
    Binding, BindingResource, Blend, BufferDesc, ComputePipelineDesc, CullMode, DepthCompare,
    DepthRect, DepthTest, GraphicsPipelineDesc, MemoryLocation, SamplerDesc, TextureDesc,
    TextureDimension, Topology, VertexAttribute, VertexFormat, VertexStream, Viewport,
};

use super::format::Format;
use super::vulkan::driver::{GeometryArena, VkDriver, VkEncoder};

/// The device a pass creates its resources on. One variant today; Metal joins it
/// with the pass that first needs it recorded there.
pub enum Device {
    Vulkan(VkDriver),
}

impl Device {
    /// Frees whatever `frame - frames_in_flight` retired. Called once per frame,
    /// before anything records.
    pub fn begin_frame(&self, frame: u64) {
        match self {
            Device::Vulkan(driver) => driver.begin_frame(frame),
        }
    }

    pub fn create_sampler(&self, desc: SamplerDesc) -> SamplerId {
        match self {
            Device::Vulkan(driver) => driver.create_sampler(desc),
        }
    }

    pub fn create_graphics_pipeline(&self, desc: &GraphicsPipelineDesc) -> PipelineId {
        match self {
            Device::Vulkan(driver) => driver.create_graphics_pipeline(desc),
        }
    }

    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> PipelineId {
        match self {
            Device::Vulkan(driver) => driver.create_compute_pipeline(desc),
        }
    }

    /// A buffer that outlives the frame, holding `initial` from creation. The
    /// words are the buffer's whole contents: `desc.size` must be their length.
    pub fn create_buffer_init(&self, desc: BufferDesc, initial: &[u32]) -> BufferId {
        match self {
            Device::Vulkan(driver) => driver.create_buffer_init(desc, initial),
        }
    }

    /// A persistent single-level 2D texture holding `pixels`, tightly packed.
    pub fn create_texture_2d(
        &self,
        pixels: &[u8],
        extent: [u32; 2],
        format: Format,
    ) -> TextureViewId {
        match self {
            Device::Vulkan(driver) => driver.create_texture_2d(pixels, extent, format),
        }
    }

    /// Per-frame uniform data: valid for the frame being recorded, then retired
    /// with it. The words are the block's std140 bytes.
    /// Per-frame vertex data, valid for the frame being recorded.
    pub fn upload_vertices(&self, bytes: &[u8]) -> BufferId {
        match self {
            Device::Vulkan(driver) => driver.upload_vertices(bytes),
        }
    }

    pub fn upload_uniform(&self, words: &[u32]) -> BufferId {
        match self {
            Device::Vulkan(driver) => driver.upload_uniform(words),
        }
    }

    pub fn view_extent(&self, view: TextureViewId) -> [u32; 2] {
        match self {
            Device::Vulkan(driver) => driver.view_extent(view),
        }
    }

    /// Like [`create_bind_group`](Self::create_bind_group) but never cached, for
    /// a set that binds a buffer allocated for this frame alone.
    pub fn create_transient_bind_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
    ) -> BindGroupId {
        match self {
            Device::Vulkan(driver) => driver.create_transient_bind_group(pipeline, set, bindings),
        }
    }

    /// A set of bindings for set `set` of `pipeline`. Equal bindings give back
    /// the same underlying set, as the descriptor cache did before the RHI.
    pub fn create_bind_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
    ) -> BindGroupId {
        match self {
            Device::Vulkan(driver) => driver.create_bind_group(pipeline, set, bindings),
        }
    }
}

/// Records commands. Every method is one `match` over the driver's encoder.
pub enum Encoder<'a> {
    Vulkan(VkEncoder<'a>),
}

impl Encoder<'_> {
    pub fn set_viewport(&mut self, viewport: Viewport) {
        match self {
            Encoder::Vulkan(encoder) => encoder.set_viewport(viewport),
        }
    }

    pub fn set_scissor(&mut self, offset: [u32; 2], extent: [u32; 2]) {
        match self {
            Encoder::Vulkan(encoder) => encoder.set_scissor(offset, extent),
        }
    }

    pub fn set_depth_bias(&mut self, constant: f32, clamp: f32, slope: f32) {
        match self {
            Encoder::Vulkan(encoder) => encoder.set_depth_bias(constant, clamp, slope),
        }
    }

    /// Clear depth to 1.0 over `rects` of the attachment being rendered into.
    pub fn clear_depth(&mut self, rects: &[DepthRect]) {
        match self {
            Encoder::Vulkan(encoder) => encoder.clear_depth(rects),
        }
    }

    pub fn bind_pipeline(&mut self, pipeline: PipelineId) {
        match self {
            Encoder::Vulkan(encoder) => encoder.bind_pipeline(pipeline),
        }
    }

    /// Bind `buffer` as vertex stream 0.
    pub fn bind_vertex_buffer(&mut self, buffer: BufferId) {
        match self {
            Encoder::Vulkan(encoder) => encoder.bind_vertex_buffer(buffer),
        }
    }

    pub fn bind_group(&mut self, set: u32, group: BindGroupId) {
        match self {
            Encoder::Vulkan(encoder) => encoder.bind_group(set, group),
        }
    }

    /// Up to 32 words: a driver's push-constant budget is far above any block
    /// this renderer has, and a fixed bound keeps the call allocation-free.
    pub fn push_constants(&mut self, words: &[u32]) {
        match self {
            Encoder::Vulkan(encoder) => encoder.push_constants(words),
        }
    }

    pub fn dispatch(&mut self, groups: [u32; 3]) {
        match self {
            Encoder::Vulkan(encoder) => encoder.dispatch(groups),
        }
    }

    /// Bind the shared vertex and index buffers every mesh is a span into.
    pub(crate) fn bind_geometry(&mut self, arena: &GeometryArena, positions_only: bool) {
        match self {
            Encoder::Vulkan(encoder) => encoder.bind_geometry(arena, positions_only),
        }
    }

    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instances: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        match self {
            Encoder::Vulkan(encoder) => encoder.draw_indexed(
                index_count,
                instances,
                first_index,
                vertex_offset,
                first_instance,
            ),
        }
    }

    /// Draw every command in `commands`, a buffer of indexed indirect draws.
    pub fn draw_indexed_indirect(&mut self, commands: BufferId) {
        match self {
            Encoder::Vulkan(encoder) => encoder.draw_indexed_indirect(commands),
        }
    }

    pub fn draw(&mut self, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
        match self {
            Encoder::Vulkan(encoder) => {
                encoder.draw(vertices, instances, first_vertex, first_instance)
            }
        }
    }
}

macro_rules! handles {
    ($($(#[$meta:meta])* $name:ident),+ $(,)?) => {$(
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub struct $name(pub(crate) u32);

        impl $name {
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
    )+};
}

handles! {
    TextureId,
    /// A view of part of a texture: one mip level, or a reinterpreted format.
    TextureViewId,
    BufferId,
    SamplerId,
    PipelineId,
    /// One descriptor set / argument buffer's worth of bindings.
    BindGroupId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_of_different_kinds_do_not_convert() {
        let texture = TextureId(3);
        let buffer = BufferId(3);
        assert_eq!(texture.index(), buffer.index());
    }
}
