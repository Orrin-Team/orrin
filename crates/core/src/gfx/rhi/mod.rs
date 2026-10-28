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
    Binding, BindingResource, BufferDesc, ComputePipelineDesc, GraphicsPipelineDesc,
    MemoryLocation, SamplerDesc, TextureDesc, TextureDimension, VertexStream, Viewport,
};

use super::vulkan::driver::{VkDriver, VkEncoder};

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

    pub fn bind_pipeline(&mut self, pipeline: PipelineId) {
        match self {
            Encoder::Vulkan(encoder) => encoder.bind_pipeline(pipeline),
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
