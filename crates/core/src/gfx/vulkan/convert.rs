//! The one place the engine's synchronisation and format vocabulary becomes
//! vulkano's.
//!
//! The graph, the frame declaration and the pass constants are written in
//! `gfx::graph` and `gfx::format` types so they take no API. A driver converts at
//! the call that needs the API's own value — the barrier it emits, the image it
//! creates. Every conversion is by name, so a member added on one side without
//! the other is a compile error here rather than a wrong flag at runtime.

use vulkano::buffer::BufferUsage as VkBufferUsage;
use vulkano::format::Format as VkFormat;
use vulkano::image::{
    ImageLayout as VkLayout, ImageUsage as VkImageUsage, SampleCount as VkSamples,
};
use vulkano::sync::{AccessFlags as VkAccess, PipelineStages as VkStages};

use crate::gfx::format::{Format, SampleCount};
use crate::gfx::graph::{AccessFlags, BufferUsage, ImageLayout, ImageUsage, PipelineStages};

pub(crate) const fn vk_format(format: Format) -> VkFormat {
    match format {
        Format::R8Unorm => VkFormat::R8_UNORM,
        Format::Rg8Unorm => VkFormat::R8G8_UNORM,
        Format::Rgba8Unorm => VkFormat::R8G8B8A8_UNORM,
        Format::Rgba8Srgb => VkFormat::R8G8B8A8_SRGB,
        Format::Bgra8Srgb => VkFormat::B8G8R8A8_SRGB,
        Format::Rg16Float => VkFormat::R16G16_SFLOAT,
        Format::Rgba16Float => VkFormat::R16G16B16A16_SFLOAT,
        Format::R32Float => VkFormat::R32_SFLOAT,
        Format::Rgba32Float => VkFormat::R32G32B32A32_SFLOAT,
        Format::D32Float => VkFormat::D32_SFLOAT,
        Format::Rg11B10Float => VkFormat::B10G11R11_UFLOAT_PACK32,
    }
}

/// `None` for a format no graph image takes, which the caller names in its panic.
pub(crate) fn format_from_vk(format: VkFormat) -> Option<Format> {
    Format::ALL.into_iter().find(|&f| vk_format(f) == format)
}

/// The graph name for a format a surface chose, which no graph image can be
/// declared in unless the engine has a name for it.
pub(crate) fn engine_format(format: VkFormat) -> Format {
    format_from_vk(format).unwrap_or_else(|| {
        panic!("the surface chose {format:?}, which the frame has no format for")
    })
}

pub(crate) const fn vk_samples(samples: SampleCount) -> VkSamples {
    match samples {
        SampleCount::Sample1 => VkSamples::Sample1,
        SampleCount::Sample2 => VkSamples::Sample2,
        SampleCount::Sample4 => VkSamples::Sample4,
        SampleCount::Sample8 => VkSamples::Sample8,
    }
}

#[cfg(test)]
pub(crate) fn samples_from_vk(samples: VkSamples) -> Option<SampleCount> {
    SampleCount::ALL
        .into_iter()
        .find(|&s| vk_samples(s) == samples)
}

pub(crate) fn vk_layout(layout: ImageLayout) -> VkLayout {
    match layout {
        ImageLayout::Undefined => VkLayout::Undefined,
        ImageLayout::General => VkLayout::General,
        ImageLayout::ColorAttachmentOptimal => VkLayout::ColorAttachmentOptimal,
        ImageLayout::DepthStencilAttachmentOptimal => VkLayout::DepthStencilAttachmentOptimal,
        ImageLayout::DepthStencilReadOnlyOptimal => VkLayout::DepthStencilReadOnlyOptimal,
        ImageLayout::ShaderReadOnlyOptimal => VkLayout::ShaderReadOnlyOptimal,
        ImageLayout::TransferSrcOptimal => VkLayout::TransferSrcOptimal,
        ImageLayout::TransferDstOptimal => VkLayout::TransferDstOptimal,
        ImageLayout::PresentSrc => VkLayout::PresentSrc,
    }
}

#[cfg(test)]
pub(crate) fn layout_from_vk(layout: VkLayout) -> Option<ImageLayout> {
    ImageLayout::ALL
        .into_iter()
        .find(|&l| vk_layout(l) == layout)
}

/// A converter pair for one flags type. The reverse direction exists for the
/// round-trip test: nothing at runtime reads a vulkano flag back into the graph.
macro_rules! flag_conversions {
    ($to:ident, $from:ident, $engine:ident, $vk:ident, [$($name:ident),+ $(,)?]) => {
        #[cfg_attr(not(test), allow(dead_code))]
        pub(crate) fn $to(flags: $engine) -> $vk {
            let mut out = $vk::empty();
            $(
                if flags.contains($engine::$name) {
                    out |= $vk::$name;
                }
            )+
            out
        }

        #[cfg(test)]
        pub(crate) fn $from(flags: $vk) -> $engine {
            let mut out = $engine::empty();
            $(
                if flags.contains($vk::$name) {
                    out |= $engine::$name;
                }
            )+
            out
        }
    };
}

flag_conversions!(
    vk_stages,
    stages_from_vk,
    PipelineStages,
    VkStages,
    [
        TOP_OF_PIPE,
        DRAW_INDIRECT,
        VERTEX_INPUT,
        VERTEX_SHADER,
        FRAGMENT_SHADER,
        EARLY_FRAGMENT_TESTS,
        LATE_FRAGMENT_TESTS,
        COLOR_ATTACHMENT_OUTPUT,
        COMPUTE_SHADER,
        ALL_TRANSFER,
        BOTTOM_OF_PIPE,
        HOST,
        ALL_COMMANDS,
    ]
);

flag_conversions!(
    vk_access,
    access_from_vk,
    AccessFlags,
    VkAccess,
    [
        INDIRECT_COMMAND_READ,
        INDEX_READ,
        VERTEX_ATTRIBUTE_READ,
        UNIFORM_READ,
        SHADER_SAMPLED_READ,
        SHADER_STORAGE_READ,
        SHADER_STORAGE_WRITE,
        COLOR_ATTACHMENT_READ,
        COLOR_ATTACHMENT_WRITE,
        DEPTH_STENCIL_ATTACHMENT_READ,
        DEPTH_STENCIL_ATTACHMENT_WRITE,
        TRANSFER_READ,
        TRANSFER_WRITE,
    ]
);

flag_conversions!(
    vk_image_usage,
    image_usage_from_vk,
    ImageUsage,
    VkImageUsage,
    [
        TRANSFER_SRC,
        TRANSFER_DST,
        SAMPLED,
        STORAGE,
        COLOR_ATTACHMENT,
        DEPTH_STENCIL_ATTACHMENT,
        TRANSIENT_ATTACHMENT,
    ]
);

flag_conversions!(
    vk_buffer_usage,
    buffer_usage_from_vk,
    BufferUsage,
    VkBufferUsage,
    [
        TRANSFER_SRC,
        TRANSFER_DST,
        UNIFORM_BUFFER,
        STORAGE_BUFFER,
        INDEX_BUFFER,
        VERTEX_BUFFER,
        INDIRECT_BUFFER,
    ]
);

#[cfg(test)]
mod tests {
    use vulkano::buffer::BufferUsage as VkBufferUsage;
    use vulkano::format::Format as Vk;
    use vulkano::image::{
        ImageLayout as VkLayout, ImageUsage as VkImageUsage, SampleCount as VkSamples,
    };
    use vulkano::sync::{AccessFlags as VkAccess, PipelineStages as VkStages};

    use super::*;
    use crate::gfx::format::{Format, SampleCount};
    use crate::gfx::graph::{AccessFlags, BufferUsage, ImageLayout, ImageUsage, PipelineStages};

    #[test]
    fn every_format_converts_to_the_vulkano_format_of_the_same_name_and_back() {
        let named = [
            (Format::R8Unorm, Vk::R8_UNORM),
            (Format::Rg8Unorm, Vk::R8G8_UNORM),
            (Format::Rgba8Unorm, Vk::R8G8B8A8_UNORM),
            (Format::Rgba8Srgb, Vk::R8G8B8A8_SRGB),
            (Format::Bgra8Srgb, Vk::B8G8R8A8_SRGB),
            (Format::Rg16Float, Vk::R16G16_SFLOAT),
            (Format::Rgba16Float, Vk::R16G16B16A16_SFLOAT),
            (Format::R32Float, Vk::R32_SFLOAT),
            (Format::Rgba32Float, Vk::R32G32B32A32_SFLOAT),
            (Format::D32Float, Vk::D32_SFLOAT),
            (Format::Rg11B10Float, Vk::B10G11R11_UFLOAT_PACK32),
        ];
        assert_eq!(named.len(), Format::ALL.len(), "a format has no row");
        for (format, vk) in named {
            assert_eq!(vk_format(format), vk, "{format:?}");
            assert_eq!(format_from_vk(vk), Some(format), "{vk:?}");
        }
        assert_eq!(format_from_vk(Vk::BC7_UNORM_BLOCK), None);
    }

    #[test]
    fn every_sample_count_converts_and_back() {
        let named = [
            (SampleCount::Sample1, VkSamples::Sample1),
            (SampleCount::Sample2, VkSamples::Sample2),
            (SampleCount::Sample4, VkSamples::Sample4),
            (SampleCount::Sample8, VkSamples::Sample8),
        ];
        assert_eq!(named.len(), SampleCount::ALL.len(), "a count has no row");
        for (count, vk) in named {
            assert_eq!(vk_samples(count), vk);
            assert_eq!(samples_from_vk(vk), Some(count));
        }
        assert_eq!(samples_from_vk(VkSamples::Sample16), None);
    }

    #[test]
    fn every_layout_converts_to_the_vulkano_layout_of_the_same_name_and_back() {
        let named = [
            (ImageLayout::Undefined, VkLayout::Undefined),
            (ImageLayout::General, VkLayout::General),
            (
                ImageLayout::ColorAttachmentOptimal,
                VkLayout::ColorAttachmentOptimal,
            ),
            (
                ImageLayout::DepthStencilAttachmentOptimal,
                VkLayout::DepthStencilAttachmentOptimal,
            ),
            (
                ImageLayout::DepthStencilReadOnlyOptimal,
                VkLayout::DepthStencilReadOnlyOptimal,
            ),
            (
                ImageLayout::ShaderReadOnlyOptimal,
                VkLayout::ShaderReadOnlyOptimal,
            ),
            (
                ImageLayout::TransferSrcOptimal,
                VkLayout::TransferSrcOptimal,
            ),
            (
                ImageLayout::TransferDstOptimal,
                VkLayout::TransferDstOptimal,
            ),
            (ImageLayout::PresentSrc, VkLayout::PresentSrc),
        ];
        assert_eq!(named.len(), ImageLayout::ALL.len(), "a layout has no row");
        for (layout, vk) in named {
            assert_eq!(vk_layout(layout), vk, "{layout:?}");
            assert_eq!(layout_from_vk(vk), Some(layout), "{vk:?}");
        }
    }

    /// Every named flag converts to the vulkano flag of the same name, a set of
    /// them converts to the union, and nothing converts to a flag it was not.
    macro_rules! flag_conversion_test {
        ($test:ident, $engine:ident, $vk:ident, $to:ident, $from:ident, [$($name:ident),+ $(,)?]) => {
            #[test]
            fn $test() {
                let named = [$((stringify!($name), $engine::$name, $vk::$name)),+];
                let defined = $engine::all().iter_names().count();
                assert_eq!(named.len(), defined, "a flag has no row");
                let mut all = $vk::empty();
                for (name, engine, vk) in named {
                    assert_eq!($to(engine), vk, "{name}");
                    assert_eq!($from(vk), engine, "{name}");
                    all |= vk;
                }
                assert_eq!($to($engine::all()), all);
                assert_eq!($from(all), $engine::all());
                assert_eq!($to($engine::empty()), $vk::empty());
                assert_eq!($from($vk::empty()), $engine::empty());
            }
        };
    }

    flag_conversion_test!(
        every_pipeline_stage_converts_by_name_and_back,
        PipelineStages,
        VkStages,
        vk_stages,
        stages_from_vk,
        [
            TOP_OF_PIPE,
            DRAW_INDIRECT,
            VERTEX_INPUT,
            VERTEX_SHADER,
            FRAGMENT_SHADER,
            EARLY_FRAGMENT_TESTS,
            LATE_FRAGMENT_TESTS,
            COLOR_ATTACHMENT_OUTPUT,
            COMPUTE_SHADER,
            ALL_TRANSFER,
            BOTTOM_OF_PIPE,
            HOST,
            ALL_COMMANDS,
        ]
    );

    flag_conversion_test!(
        every_access_flag_converts_by_name_and_back,
        AccessFlags,
        VkAccess,
        vk_access,
        access_from_vk,
        [
            INDIRECT_COMMAND_READ,
            INDEX_READ,
            VERTEX_ATTRIBUTE_READ,
            UNIFORM_READ,
            SHADER_SAMPLED_READ,
            SHADER_STORAGE_READ,
            SHADER_STORAGE_WRITE,
            COLOR_ATTACHMENT_READ,
            COLOR_ATTACHMENT_WRITE,
            DEPTH_STENCIL_ATTACHMENT_READ,
            DEPTH_STENCIL_ATTACHMENT_WRITE,
            TRANSFER_READ,
            TRANSFER_WRITE,
        ]
    );

    flag_conversion_test!(
        every_image_usage_converts_by_name_and_back,
        ImageUsage,
        VkImageUsage,
        vk_image_usage,
        image_usage_from_vk,
        [
            TRANSFER_SRC,
            TRANSFER_DST,
            SAMPLED,
            STORAGE,
            COLOR_ATTACHMENT,
            DEPTH_STENCIL_ATTACHMENT,
            TRANSIENT_ATTACHMENT,
        ]
    );

    flag_conversion_test!(
        every_buffer_usage_converts_by_name_and_back,
        BufferUsage,
        VkBufferUsage,
        vk_buffer_usage,
        buffer_usage_from_vk,
        [
            TRANSFER_SRC,
            TRANSFER_DST,
            UNIFORM_BUFFER,
            STORAGE_BUFFER,
            INDEX_BUFFER,
            VERTEX_BUFFER,
            INDIRECT_BUFFER,
        ]
    );
}
