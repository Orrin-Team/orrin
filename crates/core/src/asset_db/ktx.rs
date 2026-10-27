//! KTX2, the container a texture artifact is stored in.
//!
//! Only the subset this engine writes: one 2D image, no array layers or cube
//! faces, a full mip chain, no supercompression, and one of the
//! [`TexelFormat`]s. Written and read here rather than through a crate, since
//! that subset is a header, a level index and a fixed descriptor per format,
//! and the reader has to refuse everything else anyway. The tests hold both
//! ends to the spec through the `ktx2` crate's reader and Khronos' descriptor
//! table, so a file the cache writes opens in KTX tools as what it is.
//!
//! Levels are stored smallest first, as the spec asks, each starting on a
//! multiple of its block size, so a reader streaming the file gets the small
//! levels, which are drawn first, before the large ones.

use super::cache::TextureArtifact;
use super::texture::{TexelFormat, level_count};

/// The twelve bytes every KTX2 file starts with.
pub const IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

/// The header and the section index; the level index follows.
const HEADER: usize = 80;
const LEVEL_ENTRY: usize = 24;

/// A KTX2 file holding `texture`.
pub fn write(texture: &TextureArtifact) -> Vec<u8> {
    let format = texture.format;
    let levels = texture.levels.len();
    let descriptor = descriptor(format);
    let descriptor_at = HEADER + levels * LEVEL_ENTRY;
    let data: usize = texture.levels.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(descriptor_at + descriptor.len() * 4 + data + levels * 16);

    out.extend_from_slice(&IDENTIFIER);
    for word in [
        format.vk_format(),
        // typeSize: one byte, for an 8-bit or a block-compressed format.
        1,
        texture.width,
        texture.height,
        0, // depth: not a 3D image
        0, // layers: not an array
        1, // faces
        levels as u32,
        0, // no supercompression
        descriptor_at as u32,
        (descriptor.len() * 4) as u32,
        0, // no key/value data
        0,
    ] {
        out.extend_from_slice(&word.to_le_bytes());
    }
    // No supercompression global data: offset and length, both 64-bit.
    out.extend_from_slice(&[0; 16]);
    out.resize(descriptor_at, 0);
    for word in descriptor {
        out.extend_from_slice(&word.to_le_bytes());
    }

    // The spec's mipPadding: every level starts on a multiple of
    // lcm(block size, 4), which for these formats is the block size.
    let align = format.block_bytes();
    let mut index = vec![(0u64, 0u64); levels];
    for level in (0..levels).rev() {
        out.resize(out.len().next_multiple_of(align), 0);
        index[level] = (out.len() as u64, texture.levels[level].len() as u64);
        out.extend_from_slice(&texture.levels[level]);
    }
    for (level, (offset, length)) in index.into_iter().enumerate() {
        let at = HEADER + level * LEVEL_ENTRY;
        out[at..at + 8].copy_from_slice(&offset.to_le_bytes());
        out[at + 8..at + 16].copy_from_slice(&length.to_le_bytes());
        // Uncompressed length: the same, with no supercompression.
        out[at + 16..at + 24].copy_from_slice(&length.to_le_bytes());
    }
    out
}

/// A KTX2 file this engine wrote, read back. Anything outside the subset
/// [`write`] produces is an error naming what it found.
pub fn read(bytes: &[u8]) -> Result<TextureArtifact, String> {
    if !bytes.starts_with(&IDENTIFIER) {
        return Err("not a KTX2 file".to_owned());
    }
    let cut_short = || "the KTX2 file is cut short".to_owned();
    let field = |at: usize, len: usize| bytes.get(at..at + len).ok_or_else(cut_short);
    let word = |at: usize| -> Result<u32, String> {
        Ok(u32::from_le_bytes(field(at, 4)?.try_into().unwrap()))
    };
    let long = |at: usize| -> Result<u64, String> {
        Ok(u64::from_le_bytes(field(at, 8)?.try_into().unwrap()))
    };

    let vk_format = word(12)?;
    let format = TexelFormat::from_vk_format(vk_format).ok_or_else(|| {
        format!("the KTX2 file holds VkFormat {vk_format}, which this engine does not read")
    })?;
    let (width, height) = (word(20)?, word(24)?);
    let (depth, layers, faces) = (word(28)?, word(32)?, word(36)?);
    let levels = word(40)?;
    if word(44)? != 0 {
        return Err("the KTX2 file is supercompressed, which this engine does not read".to_owned());
    }
    if depth != 0 || layers != 0 || faces != 1 || width == 0 || height == 0 {
        return Err("the KTX2 file is not one 2D image".to_owned());
    }
    let most = level_count(width, height);
    if levels == 0 || levels > most {
        return Err(format!(
            "the KTX2 file has {levels} levels, where a {width}x{height} image has 1 to {most}"
        ));
    }

    let levels = (0..levels as usize)
        .map(|level| {
            let at = HEADER + level * LEVEL_ENTRY;
            let (offset, length) = (long(at)?, long(at + 8)?);
            let w = (width >> level).max(1);
            let h = (height >> level).max(1);
            let expected = format.level_len(w, h);
            if length != expected as u64 {
                return Err(format!(
                    "level {level} of the KTX2 file is {length} bytes, where {} at {w}x{h} is \
                     {expected}",
                    format.label()
                ));
            }
            let offset = usize::try_from(offset).map_err(|_| cut_short())?;
            Ok(field(offset, expected)?.to_vec())
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(TextureArtifact {
        format,
        width,
        height,
        levels,
    })
}

/// The Khronos data format descriptor for `format`, as little-endian words,
/// its total size first: one basic block, describing the colour model, the
/// block size and where each channel sits in a block.
fn descriptor(format: TexelFormat) -> Vec<u32> {
    // KHR_DF_TRANSFER_LINEAR or _SRGB; primaries are BT.709 throughout.
    let transfer = if format.is_srgb() { 2 } else { 1 };
    // One sample: where a channel's bits are, which channel, and its range.
    let sample = |offset: u32, bits: u32, channel: u32, upper: u32| {
        [offset | ((bits - 1) << 16) | (channel << 24), 0, 0, upper]
    };
    // KHR_DF_MODEL_RGBSDA, BC5 and BC7; texel block dimensions, each stored
    // as one less than the side; bytes in plane 0.
    let (model, block, bytes, samples) = match format {
        TexelFormat::Rgba8Unorm | TexelFormat::Rgba8Srgb => {
            // Alpha is linear even in an sRGB format: the qualifier says so.
            let alpha = if format.is_srgb() { 0x1F } else { 0x0F };
            let samples = vec![
                sample(0, 8, 0, 255),
                sample(8, 8, 1, 255),
                sample(16, 8, 2, 255),
                sample(24, 8, alpha, 255),
            ];
            (1, 0, 4, samples)
        }
        TexelFormat::Bc5Unorm => {
            let samples = vec![sample(0, 64, 0, u32::MAX), sample(64, 64, 1, u32::MAX)];
            (132, 0x0303, 16, samples)
        }
        TexelFormat::Bc7Unorm | TexelFormat::Bc7Srgb => {
            (134, 0x0303, 16, vec![sample(0, 128, 0, u32::MAX)])
        }
    };
    let block_size = 24 + 16 * samples.len() as u32;
    let mut words = vec![
        4 + block_size,
        0, // vendor Khronos, descriptor type basic
        2 | (block_size << 16),
        model | (1 << 8) | (transfer << 16),
        block,
        bytes,
        0,
    ];
    words.extend(samples.into_iter().flatten());
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [TexelFormat; 5] = [
        TexelFormat::Rgba8Unorm,
        TexelFormat::Rgba8Srgb,
        TexelFormat::Bc5Unorm,
        TexelFormat::Bc7Unorm,
        TexelFormat::Bc7Srgb,
    ];

    #[test]
    fn each_descriptor_is_the_one_khronos_gives() {
        for format in ALL {
            let theirs = vk2dfd::vk2dfd(format.vk_format()).unwrap();
            assert_eq!(descriptor(format), theirs, "{format:?}");
        }
    }

    /// A chain with recognisable bytes in every level.
    fn artifact(format: TexelFormat, width: u32, height: u32) -> TextureArtifact {
        let levels = (0..super::super::texture::level_count(width, height))
            .map(|level| {
                let w = (width >> level).max(1);
                let h = (height >> level).max(1);
                (0..format.level_len(w, h))
                    .map(|i| (i as u32 * 7 + level * 31) as u8)
                    .collect()
            })
            .collect();
        TextureArtifact {
            format,
            width,
            height,
            levels,
        }
    }

    #[test]
    fn a_texture_survives_a_write_and_a_read() {
        for format in ALL {
            let texture = artifact(format, 37, 9);
            assert_eq!(read(&write(&texture)).unwrap(), texture, "{format:?}");
        }
    }

    /// The other end of the spec check: a reader that is not ours accepts the
    /// file and finds the same levels in it.
    #[test]
    fn the_ktx2_crate_reads_what_is_written() {
        for format in ALL {
            let texture = artifact(format, 20, 12);
            let bytes = write(&texture);
            let reader = ktx2::Reader::new(&bytes[..]).unwrap();
            let header = reader.header();
            assert_eq!(
                header.format.map(|f| f.value()),
                Some(format.vk_format()),
                "{format:?}"
            );
            assert_eq!((header.pixel_width, header.pixel_height), (20, 12));
            assert_eq!(header.level_count, 5);
            assert_eq!(header.supercompression_scheme, None);
            let levels: Vec<&[u8]> = reader.levels().map(|level| level.data).collect();
            let ours: Vec<&[u8]> = texture.levels.iter().map(Vec::as_slice).collect();
            assert_eq!(levels, ours, "{format:?}");
        }
    }

    #[test]
    fn levels_are_stored_smallest_first_on_block_boundaries() {
        let bytes = write(&artifact(TexelFormat::Bc7Srgb, 64, 64));
        let entry = |level: usize| {
            let at = HEADER + level * LEVEL_ENTRY;
            u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
        };
        for level in 0..7 {
            assert_eq!(entry(level) % 16, 0, "level {level} is aligned");
        }
        assert!(entry(6) < entry(0), "the 1x1 level comes before the 64x64");
    }

    #[test]
    fn a_file_outside_the_subset_is_refused_by_what_it_is() {
        let bytes = write(&artifact(TexelFormat::Bc7Unorm, 8, 8));
        assert_eq!(
            read(b"PNG and then some more bytes").unwrap_err(),
            "not a KTX2 file"
        );
        assert_eq!(
            read(&bytes[..40]).unwrap_err(),
            "the KTX2 file is cut short"
        );

        let mut other = bytes.clone();
        other[12..16].copy_from_slice(&23u32.to_le_bytes());
        assert_eq!(
            read(&other).unwrap_err(),
            "the KTX2 file holds VkFormat 23, which this engine does not read"
        );

        let mut squeezed = bytes.clone();
        squeezed[44..48].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(
            read(&squeezed).unwrap_err(),
            "the KTX2 file is supercompressed, which this engine does not read"
        );

        let mut short = bytes;
        let at = HEADER + 8;
        let length = u64::from_le_bytes(short[at..at + 8].try_into().unwrap());
        short[at..at + 8].copy_from_slice(&(length - 1).to_le_bytes());
        assert_eq!(
            read(&short).unwrap_err(),
            "level 0 of the KTX2 file is 63 bytes, where BC7 at 8x8 is 64"
        );
    }
}
