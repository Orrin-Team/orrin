//! The engine's vocabulary as Metal's: formats, usages and sampler state.
//!
//! By name, one arm per member, so a format added to `gfx::format` without a
//! Metal spelling is a compile error here rather than a wrong pixel.

use objc2_metal::{
    MTLCompareFunction, MTLPixelFormat, MTLSamplerAddressMode, MTLSamplerMinMagFilter,
    MTLSamplerMipFilter, MTLTextureUsage,
};

use crate::gfx::format::Format;
use crate::gfx::graph::ImageUsage;
use crate::gfx::rhi::SamplerDesc;

pub(super) const fn pixel_format(format: Format) -> MTLPixelFormat {
    match format {
        Format::R8Unorm => MTLPixelFormat::R8Unorm,
        Format::Rg8Unorm => MTLPixelFormat::RG8Unorm,
        Format::Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        Format::Rgba8Srgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        Format::Bgra8Srgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        Format::Rg16Float => MTLPixelFormat::RG16Float,
        Format::Rgba16Float => MTLPixelFormat::RGBA16Float,
        Format::R32Float => MTLPixelFormat::R32Float,
        Format::Rgba32Float => MTLPixelFormat::RGBA32Float,
        Format::D32Float => MTLPixelFormat::Depth32Float,
        Format::Rg11B10Float => MTLPixelFormat::RG11B10Float,
    }
}

/// What a texture is created able to do. A transfer is a blit on Metal, which
/// asks nothing of the texture, so only the shader and attachment uses count.
pub(super) fn texture_usage(usage: ImageUsage) -> MTLTextureUsage {
    let mut out = MTLTextureUsage::empty();
    if usage.contains(ImageUsage::SAMPLED) {
        out |= MTLTextureUsage::ShaderRead;
    }
    if usage.contains(ImageUsage::STORAGE) {
        out |= MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite;
    }
    // A scaling copy is a render pass here (see `backdrop.rs`), so a transfer
    // destination is a render target too.
    if usage.intersects(
        ImageUsage::COLOR_ATTACHMENT
            | ImageUsage::DEPTH_STENCIL_ATTACHMENT
            | ImageUsage::TRANSFER_DST,
    ) {
        out |= MTLTextureUsage::RenderTarget;
    }
    out
}

/// Whether the graph's usage lets a texture live in tile memory only: it is an
/// attachment and nothing reads it back (trap T9).
pub(super) fn may_be_memoryless(usage: ImageUsage) -> bool {
    usage.intersects(ImageUsage::COLOR_ATTACHMENT | ImageUsage::DEPTH_STENCIL_ATTACHMENT)
        && !usage.intersects(
            ImageUsage::SAMPLED
                | ImageUsage::STORAGE
                | ImageUsage::TRANSFER_SRC
                | ImageUsage::TRANSFER_DST,
        )
}

/// A sampler description as Metal states it. `supportArgumentBuffers` is set by
/// the caller, because every sampler here is reached through one.
pub(super) struct SamplerState {
    pub filter: MTLSamplerMinMagFilter,
    pub mip: MTLSamplerMipFilter,
    pub address: MTLSamplerAddressMode,
    pub compare: Option<MTLCompareFunction>,
    pub anisotropy: usize,
    pub max_lod: f32,
}

pub(super) fn sampler_state(desc: &SamplerDesc) -> SamplerState {
    SamplerState {
        filter: if desc.linear {
            MTLSamplerMinMagFilter::Linear
        } else {
            MTLSamplerMinMagFilter::Nearest
        },
        mip: if desc.mipmap_linear {
            MTLSamplerMipFilter::Linear
        } else {
            MTLSamplerMipFilter::Nearest
        },
        address: if desc.repeat {
            MTLSamplerAddressMode::Repeat
        } else {
            MTLSamplerAddressMode::ClampToEdge
        },
        compare: desc.compare.then_some(MTLCompareFunction::Less),
        anisotropy: desc.anisotropy.map_or(1, |a| a.clamp(1.0, 16.0) as usize),
        max_lod: desc.max_lod.unwrap_or(f32::MAX),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A format with no spelling would be a compile error; this holds that the
    /// spellings are distinct, so two engine formats never share a pixel.
    #[test]
    fn every_format_has_its_own_pixel_format() {
        let mut seen = std::collections::HashSet::new();
        for format in Format::ALL {
            assert!(
                seen.insert(pixel_format(format).0),
                "{format:?} shares a pixel format"
            );
        }
    }

    /// A texture a pass samples, stores to or reads back must never be tile
    /// memory: its contents would be undefined.
    #[test]
    fn memoryless_never_covers_a_read_or_a_store() {
        assert!(may_be_memoryless(ImageUsage::COLOR_ATTACHMENT));
        assert!(may_be_memoryless(ImageUsage::DEPTH_STENCIL_ATTACHMENT));
        for read in [
            ImageUsage::SAMPLED,
            ImageUsage::STORAGE,
            ImageUsage::TRANSFER_SRC,
            ImageUsage::TRANSFER_DST,
        ] {
            assert!(!may_be_memoryless(ImageUsage::COLOR_ATTACHMENT | read));
        }
    }

    #[test]
    fn an_attachment_that_is_sampled_is_a_shader_read_and_a_render_target() {
        let usage = texture_usage(ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED);
        assert!(usage.contains(MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget));
        assert!(!usage.contains(MTLTextureUsage::ShaderWrite));
    }
}
