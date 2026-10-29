//! The pixel formats and sample counts a frame is declared in.
//!
//! The render graph and the frame declaration are device-free, so they cannot
//! speak any API's format enum. These are the formats a graph-owned or imported
//! image actually takes — named for what a texel holds, not for the API that
//! first spelled it — and each driver converts at the call that creates the
//! image. A format joins this list when a pass needs it; a block-compressed
//! texture is a [`TexelFormat`](super::TexelFormat) and never an image the graph
//! owns.

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Format {
    R8Unorm,
    Rg8Unorm,
    Rgba8Unorm,
    Rgba8Srgb,
    Bgra8Srgb,
    Rg16Float,
    Rgba16Float,
    R32Float,
    Rgba32Float,
    D32Float,
    Rg11B10Float,
}

impl Format {
    pub const ALL: [Format; 11] = [
        Format::R8Unorm,
        Format::Rg8Unorm,
        Format::Rgba8Unorm,
        Format::Rgba8Srgb,
        Format::Bgra8Srgb,
        Format::Rg16Float,
        Format::Rgba16Float,
        Format::R32Float,
        Format::Rgba32Float,
        Format::D32Float,
        Format::Rg11B10Float,
    ];

    pub fn is_depth(self) -> bool {
        matches!(self, Format::D32Float)
    }

    pub fn is_srgb(self) -> bool {
        matches!(self, Format::Rgba8Srgb | Format::Bgra8Srgb)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SampleCount {
    Sample1,
    Sample2,
    Sample4,
    Sample8,
}

impl SampleCount {
    pub const ALL: [SampleCount; 4] = [
        SampleCount::Sample1,
        SampleCount::Sample2,
        SampleCount::Sample4,
        SampleCount::Sample8,
    ];
}
