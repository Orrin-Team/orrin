//! A texture's levels: the mip chain an import builds on the CPU, and the
//! format each level is stored in.
//!
//! **Mips are made at import, not at upload.** A block-compressed image cannot
//! be the target of a GPU blit, so the chain has to exist before compression,
//! and once it is built here it is filtered the way each [`Usage`] needs
//! rather than the way a blit happens to: colour in linear light, a normal map
//! renormalised at every level so it does not flatten with distance, data as
//! stored. Every level is made from the one above it in `f32`, and rounded to
//! eight bits only as it is written, so error does not build up down the chain.
//!
//! **BC7 for colour and data, BC5 for normal maps.** BC7 holds four channels at
//! one byte a texel, a quarter of RGBA8, with the best quality of the BC
//! formats. BC5 holds two channels at the same rate, each at the precision BC7
//! gives four, which is what a normal map wants: X and Y kept well, and Z
//! rebuilt from them on sample.
//!
//! **Compression runs in 64x64 tiles, on the import's own threads.**
//! `rusty_dds` spreads an image of more than 512 blocks across threads of its
//! own, one per core, which beside the frame's pool cost a stress frame 4 ms
//! at p95 (see [`threads`](crate::threads) on imports). A tile is 256 blocks,
//! under that line, so each is encoded serially, and [`encode_chain`] shares
//! the tiles out across the few threads an import is allowed. Blocks are
//! independent, so the tiles put back together are byte-for-byte the image
//! encoded whole, on any number of threads.

use image::imageops::FilterType;
use image::{Rgba32FImage, RgbaImage};

use super::sidecar::Usage;

/// The side of a compression tile, in texels: 16x16 blocks, 256 of them.
const TILE: u32 = 64;

/// What the device an artifact is for can sample: block-compressed formats, or
/// only plain RGBA8.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    Bc,
    Rgba8,
}

impl Target {
    /// What `backend` samples.
    pub fn of(backend: &impl crate::gfx::RenderBackend) -> Self {
        if backend.samples_bc() {
            Self::Bc
        } else {
            Self::Rgba8
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bc => "bc",
            Self::Rgba8 => "rgba8",
        }
    }
}

pub use crate::gfx::TexelFormat;

impl TexelFormat {
    /// The format a texture of `usage` is stored in for `target`.
    pub fn choose(usage: Usage, compress: bool, target: Target) -> Self {
        match (compress && target == Target::Bc, usage) {
            (true, Usage::Color) => Self::Bc7Srgb,
            (true, Usage::Normal) => Self::Bc5Unorm,
            (true, Usage::Data) => Self::Bc7Unorm,
            (false, Usage::Color) => Self::Rgba8Srgb,
            (false, Usage::Normal | Usage::Data) => Self::Rgba8Unorm,
        }
    }
}

/// How many levels a full chain from `width` by `height` down to one texel
/// has.
pub fn level_count(width: u32, height: u32) -> u32 {
    u32::BITS - width.max(height).max(1).leading_zeros()
}

/// `level0` and every level below it, down to one texel.
///
/// `level0` comes back as it was given; the levels below are filtered for
/// `usage` (see the module docs).
pub fn mip_chain(level0: RgbaImage, usage: Usage) -> Vec<RgbaImage> {
    let (width, height) = level0.dimensions();
    let count = level_count(width, height) as usize;
    let mut chain = Vec::with_capacity(count);
    let mut above = to_float(&level0, usage);
    chain.push(level0);
    for _ in 1..count {
        above = halved(&above, usage);
        chain.push(to_bytes(&above, usage));
    }
    chain
}

/// Scale `image` to `width` by `height`, filtered for `usage`: what `max_size`
/// does to a source before its chain is built.
pub fn resize(image: &RgbaImage, width: u32, height: u32, usage: Usage) -> RgbaImage {
    to_bytes(
        &filtered(&to_float(image, usage), width, height, usage),
        usage,
    )
}

/// Every level of `chain` in `format`, the tiles shared out across `threads`
/// threads: the caller and `threads - 1` started for the call.
///
/// Threads of its own rather than the pool's, for the reason `threads` gives
/// under imports. They take tiles from one counter, so a small level's few
/// tiles and a large level's many balance out over the chain.
pub fn encode_chain(chain: &[RgbaImage], format: TexelFormat, threads: usize) -> Vec<Vec<u8>> {
    let Some(content) = content(format) else {
        return chain.iter().map(|level| level.as_raw().clone()).collect();
    };
    let tiles: Vec<(usize, u32, u32)> = chain
        .iter()
        .enumerate()
        .flat_map(|(level, image)| tiles(image).map(move |(left, top)| (level, left, top)))
        .collect();
    let mut out: Vec<Vec<u8>> = chain
        .iter()
        .map(|level| vec![0; format.level_len(level.width(), level.height())])
        .collect();

    let next = std::sync::atomic::AtomicUsize::new(0);
    let work = || {
        let mut done = Vec::new();
        loop {
            let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Some(&(level, left, top)) = tiles.get(index) else {
                return done;
            };
            done.push((index, encode_tile(&chain[level], content, left, top)));
        }
    };
    let finished: Vec<(usize, Vec<u8>)> = std::thread::scope(|scope| {
        let helpers: Vec<_> = (1..threads.min(tiles.len()))
            .filter_map(|index| {
                std::thread::Builder::new()
                    .name(format!("orrin-encode-{index}"))
                    .spawn_scoped(scope, work)
                    .ok()
            })
            .collect();
        let mut finished = work();
        for helper in helpers {
            finished.extend(
                helper
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
            );
        }
        finished
    });
    for (index, blocks) in finished {
        let (level, left, top) = tiles[index];
        place(&mut out[level], &chain[level], format, left, top, &blocks);
    }
    out
}

/// One level in `format`.
pub fn encode(level: &RgbaImage, format: TexelFormat) -> Vec<u8> {
    let Some(content) = content(format) else {
        return level.as_raw().clone();
    };
    let mut out = vec![0; format.level_len(level.width(), level.height())];
    for (left, top) in tiles(level) {
        let blocks = encode_tile(level, content, left, top);
        place(&mut out, level, format, left, top, &blocks);
    }
    out
}

/// What `rusty_dds` calls a block format; `None` for RGBA8, stored as it is.
fn content(format: TexelFormat) -> Option<rusty_dds::DecodeContent> {
    match format {
        TexelFormat::Rgba8Unorm | TexelFormat::Rgba8Srgb => None,
        TexelFormat::Bc5Unorm => Some(rusty_dds::DecodeContent::Bc5UNorm),
        TexelFormat::Bc7Unorm | TexelFormat::Bc7Srgb => Some(rusty_dds::DecodeContent::Bc7),
    }
}

/// The top-left corner of every tile of `level`, row by row.
fn tiles(level: &RgbaImage) -> impl Iterator<Item = (u32, u32)> {
    let (width, height) = level.dimensions();
    (0..height).step_by(TILE as usize).flat_map(move |top| {
        (0..width)
            .step_by(TILE as usize)
            .map(move |left| (left, top))
    })
}

/// The blocks of the tile at `left`, `top`, row by row.
fn encode_tile(
    level: &RgbaImage,
    content: rusty_dds::DecodeContent,
    left: u32,
    top: u32,
) -> Vec<u8> {
    let (width, height) = level.dimensions();
    let tile_width = TILE.min(width - left);
    let tile_height = TILE.min(height - top);
    let texels = level.as_raw();
    let mut tile = Vec::with_capacity((tile_width * tile_height * 4) as usize);
    for y in top..top + tile_height {
        let start = (y * width + left) as usize * 4;
        tile.extend_from_slice(&texels[start..start + tile_width as usize * 4]);
    }
    let layout = rusty_dds::EncodeLayout::flat_2d(content, tile_width, tile_height);
    rusty_dds::Dds::encode_from_rgba8(&tile, layout)
        .expect("a tile is a valid 2D layout holding all of its texels")
        .data
}

/// Copy a tile's blocks into their rows of the level.
fn place(
    out: &mut [u8],
    level: &RgbaImage,
    format: TexelFormat,
    left: u32,
    top: u32,
    blocks: &[u8],
) {
    let block = format.block_bytes();
    let blocks_x = level.width().div_ceil(4) as usize;
    let row_bytes = TILE.min(level.width() - left).div_ceil(4) as usize * block;
    for (row, run) in blocks.chunks(row_bytes).enumerate() {
        let at = ((top / 4) as usize + row) * blocks_x + (left / 4) as usize;
        out[at * block..at * block + run.len()].copy_from_slice(run);
    }
}

/// The next level down: each texel the average of the 2x2 it covers.
///
/// A box rather than the tent [`resize`] uses: at exactly half size a tent
/// reaches past the footprint and weighs it by phase, so a checkerboard's mip
/// comes out banded instead of flat. An odd side's last row or column is left
/// out: the level below is `side / 2` texels, as Vulkan sizes it, and a 2x2
/// box over that many cannot reach it.
fn halved(image: &Rgba32FImage, usage: Usage) -> Rgba32FImage {
    let (width, height) = image.dimensions();
    let mut out = Rgba32FImage::from_fn((width / 2).max(1), (height / 2).max(1), |x, y| {
        let column = |dx: u32| (2 * x + dx).min(width - 1);
        let row = |dy: u32| (2 * y + dy).min(height - 1);
        let mut sum = [0.0f32; 4];
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let texel = image.get_pixel(column(dx), row(dy)).0;
            for (total, value) in sum.iter_mut().zip(texel) {
                *total += value;
            }
        }
        image::Rgba(sum.map(|total| total * 0.25))
    });
    if usage == Usage::Normal {
        renormalised(&mut out);
    }
    out
}

/// Scale in the space `usage` filters in, for any ratio.
fn filtered(image: &Rgba32FImage, width: u32, height: u32, usage: Usage) -> Rgba32FImage {
    let mut out = image::imageops::resize(image, width, height, FilterType::Triangle);
    if usage == Usage::Normal {
        renormalised(&mut out);
    }
    out
}

/// Averaging directions shortens them; a normal goes back to unit length.
fn renormalised(image: &mut Rgba32FImage) {
    for texel in image.pixels_mut() {
        let [x, y, z, a] = texel.0;
        let length = (x * x + y * y + z * z).sqrt();
        texel.0 = if length > 1e-6 {
            [x / length, y / length, z / length, a]
        } else {
            [0.0, 0.0, 1.0, a]
        };
    }
}

/// Texels as what they stand for: linear light, a direction, or the value.
/// Alpha is linear in every usage.
fn to_float(image: &RgbaImage, usage: Usage) -> Rgba32FImage {
    let linear: [f32; 256] = std::array::from_fn(|code| to_linear(code as u8));
    let channel = |code: u8| -> f32 {
        match usage {
            Usage::Color => linear[code as usize],
            Usage::Normal => f32::from(code) / 255.0 * 2.0 - 1.0,
            Usage::Data => f32::from(code) / 255.0,
        }
    };
    let (width, height) = image.dimensions();
    Rgba32FImage::from_fn(width, height, |x, y| {
        let [r, g, b, a] = image.get_pixel(x, y).0;
        image::Rgba([channel(r), channel(g), channel(b), f32::from(a) / 255.0])
    })
}

/// [`to_float`] undone, rounding to the nearest code.
fn to_bytes(image: &Rgba32FImage, usage: Usage) -> RgbaImage {
    let unit = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    let channel = |value: f32| -> u8 {
        match usage {
            Usage::Color => to_srgb(value),
            Usage::Normal => unit(value * 0.5 + 0.5),
            Usage::Data => unit(value),
        }
    };
    let (width, height) = image.dimensions();
    RgbaImage::from_fn(width, height, |x, y| {
        let [r, g, b, a] = image.get_pixel(x, y).0;
        image::Rgba([channel(r), channel(g), channel(b), unit(a)])
    })
}

fn to_linear(code: u8) -> f32 {
    let c = f32::from(code) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn to_srgb(linear: f32) -> u8 {
    let l = linear.clamp(0.0, 1.0);
    let c = if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (c * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use rusty_dds::DecodeContent;

    use super::*;

    fn flat(width: u32, height: u32, texel: [u8; 4]) -> RgbaImage {
        RgbaImage::from_pixel(width, height, image::Rgba(texel))
    }

    #[test]
    fn a_chain_runs_down_to_one_texel_halving_and_rounding_down() {
        let chain = mip_chain(flat(5, 3, [9, 9, 9, 9]), Usage::Data);
        let sizes: Vec<(u32, u32)> = chain.iter().map(|level| level.dimensions()).collect();
        assert_eq!(sizes, [(5, 3), (2, 1), (1, 1)]);
        assert_eq!(level_count(5, 3), 3);
        assert_eq!(level_count(1, 1), 1);
        assert_eq!(level_count(4096, 16), 13);
    }

    #[test]
    fn a_flat_texture_is_the_same_texel_at_every_level() {
        for usage in [Usage::Color, Usage::Data] {
            let chain = mip_chain(flat(8, 8, [200, 100, 50, 77]), usage);
            for level in &chain {
                assert!(
                    level.pixels().all(|p| p.0 == [200, 100, 50, 77]),
                    "{usage:?} {:?}",
                    level.dimensions()
                );
            }
        }
    }

    fn checker(width: u32, height: u32) -> RgbaImage {
        RgbaImage::from_fn(width, height, |x, y| {
            let v = if (x + y) % 2 == 0 { 0 } else { 255 };
            image::Rgba([v, v, v, 255])
        })
    }

    /// Colour averages light: black and white make 50% luminance, sRGB 188.
    /// Data averages what is stored: 128.
    #[test]
    fn colour_mips_average_light_and_data_mips_average_values() {
        let colour = &mip_chain(checker(4, 4), Usage::Color)[1];
        assert!(
            colour.pixels().all(|p| (186..=190).contains(&p.0[0])),
            "{colour:?}"
        );
        let data = &mip_chain(checker(4, 4), Usage::Data)[1];
        assert!(
            data.pixels().all(|p| (126..=129).contains(&p.0[0])),
            "{data:?}"
        );
    }

    fn decode_normal(texel: [u8; 4]) -> [f32; 3] {
        [0, 1, 2].map(|i| f32::from(texel[i]) / 255.0 * 2.0 - 1.0)
    }

    /// Normals leaning 45° left and right average to straight up, and the
    /// average is renormalised rather than left at the cosine's length.
    #[test]
    fn normal_mips_stay_unit_length() {
        let lean = 0.5f32.sqrt();
        let encode = |v: f32| ((v * 0.5 + 0.5) * 255.0).round() as u8;
        let source = RgbaImage::from_fn(4, 4, |x, _| {
            let x = if x % 2 == 0 { -lean } else { lean };
            image::Rgba([encode(x), encode(0.0), encode(lean), 255])
        });
        let chain = mip_chain(source, Usage::Normal);
        for level in &chain[1..] {
            for texel in level.pixels() {
                let [x, y, z] = decode_normal(texel.0);
                let length = (x * x + y * y + z * z).sqrt();
                assert!((length - 1.0).abs() < 0.02, "{texel:?} has length {length}");
                assert!(z > 0.99, "{texel:?} points up");
            }
        }
    }

    #[test]
    fn each_usage_takes_its_format_for_each_target() {
        use TexelFormat::*;
        use Usage::*;
        let table = [
            (Color, true, Target::Bc, Bc7Srgb),
            (Normal, true, Target::Bc, Bc5Unorm),
            (Data, true, Target::Bc, Bc7Unorm),
            (Color, false, Target::Bc, Rgba8Srgb),
            (Normal, false, Target::Bc, Rgba8Unorm),
            (Color, true, Target::Rgba8, Rgba8Srgb),
            (Normal, true, Target::Rgba8, Rgba8Unorm),
            (Data, true, Target::Rgba8, Rgba8Unorm),
        ];
        for (usage, compress, target, format) in table {
            assert_eq!(
                TexelFormat::choose(usage, compress, target),
                format,
                "{usage:?} compress={compress} {target:?}"
            );
        }
    }

    #[test]
    fn a_level_rounds_partial_blocks_up() {
        assert_eq!(TexelFormat::Bc7Srgb.level_len(5, 3), 2 * 16);
        assert_eq!(TexelFormat::Bc5Unorm.level_len(1, 1), 16);
        assert_eq!(TexelFormat::Bc7Unorm.level_len(9, 9), 9 * 16);
        assert_eq!(TexelFormat::Rgba8Srgb.level_len(5, 3), 60);
    }

    fn gradient(width: u32, height: u32) -> RgbaImage {
        RgbaImage::from_fn(width, height, |x, y| {
            image::Rgba([
                (x * 255 / width.max(2).saturating_sub(1).max(1)) as u8,
                (y * 255 / height.max(2).saturating_sub(1).max(1)) as u8,
                ((x + y) * 3 % 256) as u8,
                255,
            ])
        })
    }

    /// The tiles exist only to keep `rusty_dds` off threads of its own; the
    /// blocks have to be exactly the ones it makes of the whole image, which
    /// at this size it encodes on those threads.
    #[test]
    fn tiled_compression_is_the_whole_image_compressed() {
        let image = gradient(203, 137);
        for (format, content) in [
            (TexelFormat::Bc7Srgb, DecodeContent::Bc7),
            (TexelFormat::Bc5Unorm, DecodeContent::Bc5UNorm),
        ] {
            let whole = rusty_dds::Dds::encode_from_rgba8(
                image.as_raw(),
                rusty_dds::EncodeLayout::flat_2d(content, 203, 137),
            )
            .unwrap()
            .data;
            let tiled = encode(&image, format);
            assert_eq!(tiled.len(), format.level_len(203, 137), "{format:?}");
            assert!(
                tiled == whole,
                "{format:?}: the tiles differ from the whole"
            );
        }
    }

    fn decoded(content: DecodeContent, blocks: &[u8], width: u32, height: u32) -> Vec<u8> {
        rusty_dds::reference::reference_rgba8(content, blocks, width, height, 1).unwrap()
    }

    /// A floor, not a target: a smooth image well inside what BC7 holds, so a
    /// broken encoder or a swapped channel falls far below it.
    #[test]
    fn bc7_keeps_a_smooth_image_above_40_db() {
        let image = gradient(128, 96);
        let blocks = encode(&image, TexelFormat::Bc7Srgb);
        let back = decoded(DecodeContent::Bc7, &blocks, 128, 96);
        let psnr = rusty_dds::psnr_rgba8(image.as_raw(), &back).unwrap();
        assert!(psnr > 40.0, "{psnr:.1} dB");
    }

    #[test]
    fn bc5_keeps_red_and_green() {
        let image = gradient(64, 64);
        let blocks = encode(&image, TexelFormat::Bc5Unorm);
        let back = decoded(DecodeContent::Bc5UNorm, &blocks, 64, 64);
        let worst = image
            .as_raw()
            .chunks(4)
            .zip(back.chunks(4))
            .map(|(a, b)| a[0].abs_diff(b[0]).max(a[1].abs_diff(b[1])))
            .max()
            .unwrap();
        assert!(worst <= 4, "worst channel error {worst}");
    }

    /// Sharing tiles between threads changes how long a chain takes and
    /// nothing about what it is.
    #[test]
    fn a_chain_encoded_across_threads_is_the_chain_encoded_serially() {
        let chain = mip_chain(gradient(203, 137), Usage::Color);
        for format in [
            TexelFormat::Bc7Srgb,
            TexelFormat::Bc5Unorm,
            TexelFormat::Rgba8Srgb,
        ] {
            let serial: Vec<Vec<u8>> = chain.iter().map(|level| encode(level, format)).collect();
            for threads in [1, 3, 4] {
                assert!(
                    encode_chain(&chain, format, threads) == serial,
                    "{format:?} over {threads} threads"
                );
            }
        }
    }

    #[test]
    fn an_rgba8_level_is_its_texels() {
        let image = gradient(3, 2);
        assert_eq!(
            encode(&image, TexelFormat::Rgba8Srgb),
            image.as_raw().clone()
        );
    }
}
