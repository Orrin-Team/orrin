//! How a texture's texels are laid out in memory, and a texture as it is
//! uploaded: a whole mip chain in one of those layouts.
//!
//! Here rather than with the asset database, which makes these, because the
//! backend is what takes them and the backend does not depend on the database.

/// How a level's texels are laid out, as the Vulkan format the GPU samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexelFormat {
    Rgba8Unorm,
    Rgba8Srgb,
    Bc5Unorm,
    Bc7Unorm,
    Bc7Srgb,
}

impl TexelFormat {
    /// The word an error uses for it.
    pub fn label(self) -> &'static str {
        match self {
            Self::Rgba8Unorm | Self::Rgba8Srgb => "RGBA8",
            Self::Bc5Unorm => "BC5",
            Self::Bc7Unorm | Self::Bc7Srgb => "BC7",
        }
    }

    /// The `VkFormat` value, which is also what a KTX2 header names.
    pub fn vk_format(self) -> u32 {
        match self {
            Self::Rgba8Unorm => 37,
            Self::Rgba8Srgb => 43,
            Self::Bc5Unorm => 141,
            Self::Bc7Unorm => 145,
            Self::Bc7Srgb => 146,
        }
    }

    pub fn from_vk_format(format: u32) -> Option<Self> {
        [
            Self::Rgba8Unorm,
            Self::Rgba8Srgb,
            Self::Bc5Unorm,
            Self::Bc7Unorm,
            Self::Bc7Srgb,
        ]
        .into_iter()
        .find(|candidate| candidate.vk_format() == format)
    }

    pub fn is_srgb(self) -> bool {
        matches!(self, Self::Rgba8Srgb | Self::Bc7Srgb)
    }

    /// The side of the square a block covers, in texels: `1` for RGBA8.
    pub fn block_side(self) -> u32 {
        match self {
            Self::Rgba8Unorm | Self::Rgba8Srgb => 1,
            Self::Bc5Unorm | Self::Bc7Unorm | Self::Bc7Srgb => 4,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Rgba8Unorm | Self::Rgba8Srgb => 4,
            Self::Bc5Unorm | Self::Bc7Unorm | Self::Bc7Srgb => 16,
        }
    }

    /// Bytes a `width` by `height` level takes. A partial block at an edge is
    /// a whole block.
    pub fn level_len(self, width: u32, height: u32) -> usize {
        let side = self.block_side();
        (width.div_ceil(side) as usize) * (height.div_ceil(side) as usize) * self.block_bytes()
    }
}

/// A texture as it is uploaded: its mip chain in the format the GPU samples,
/// level 0 first, each level's texels or blocks tightly packed from the
/// top-left.
#[derive(Clone, Debug, PartialEq)]
pub struct TextureLevels {
    pub format: TexelFormat,
    pub width: u32,
    pub height: u32,
    pub levels: Vec<Vec<u8>>,
}
