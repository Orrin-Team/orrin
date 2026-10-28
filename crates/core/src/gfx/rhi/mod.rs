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
#[allow(dead_code)]
mod pool;

pub use desc::{
    BufferDesc, MemoryLocation, SamplerDesc, TextureDesc, TextureDimension, VertexStream,
};

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
