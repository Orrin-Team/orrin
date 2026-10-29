//! Bloom: a downsample chain over the exposed frame, spread back up and blended
//! into the tonemap.
//!
//! Two chains of images, not one. The upsample adds the coarser level's spread
//! to the down-chain level of the same size, and it cannot write that sum back
//! into the level it read: resources here are unversioned, so a level that is
//! both an input and an output of the chain closes a dependency cycle the
//! compiler rejects. The second chain is what a version per write would
//! otherwise buy, and `an_upsample_cannot_accumulate_into_the_level_it_read`
//! is the test that pins the reason.

use crate::gfx::format::Format;
use crate::gfx::rhi::{
    Binding, BindingResource, BufferId, ComputePipelineDesc, Device, Encoder, PipelineId,
    SamplerDesc, SamplerId, TextureViewId,
};
use crate::gfx::shaders::ShaderId;
use crate::scene::{BloomSettings, HdrSettings};

/// Most levels the chain will ever have. Six covers a 4K frame down to about
/// 60x34, past which a level contributes nothing a wider tent filter would not.
pub const MAX_BLOOM_MIPS: usize = 6;

/// Below this the next level would be too small for the 13-tap filter to mean
/// anything, so the chain stops however many levels short of the cap that puts
/// it.
const MIN_BLOOM_EXTENT: u32 = 8;

/// Side of the compute workgroup for every pass in the chain.
const TILE: u32 = 8;

/// How many levels a frame of this size supports.
///
/// Structural: the count decides how many passes the graph registers, so it
/// lives in `FrameConfig` and a resize that changes it recompiles — exactly how
/// the shadow cascade count behaves.
pub fn mip_count(extent: [u32; 2]) -> u8 {
    let mut mips = 0;
    for level in 1..=MAX_BLOOM_MIPS as u32 {
        let width = extent[0] >> level;
        let height = extent[1] >> level;
        if width.min(height) < MIN_BLOOM_EXTENT {
            break;
        }
        mips = level as u8;
    }
    mips
}

pub struct BloomPass {
    prefilter_pipeline: PipelineId,
    downsample_pipeline: PipelineId,
    upsample_pipeline: PipelineId,
    /// Linear and clamped. Linear because every filter in the chain leans on
    /// bilinear taps to halve its sample count; clamped so the taps that fall
    /// outside a level repeat its edge instead of wrapping the glow around to
    /// the opposite side of the screen.
    sampler: SamplerId,
    /// Bound to the tonemap pass when bloom is off, so one pipeline serves both
    /// cases. Black rather than white: at a zero strength the blend ignores it,
    /// and if a strength ever leaks through, black is the harmless answer.
    black: TextureViewId,
    settings: BloomSettings,
    hdr: HdrSettings,
}

impl BloomPass {
    pub fn new(device: &Device) -> Self {
        let pipeline = |shader| device.create_compute_pipeline(&ComputePipelineDesc { shader });
        Self {
            prefilter_pipeline: pipeline(ShaderId::BloomPrefilterComp),
            downsample_pipeline: pipeline(ShaderId::BloomDownsampleComp),
            upsample_pipeline: pipeline(ShaderId::BloomUpsampleComp),
            sampler: device.create_sampler(SamplerDesc {
                linear: true,
                mipmap_linear: false,
                repeat: false,
                compare: false,
                anisotropy: None,
                max_lod: Some(1.0),
            }),
            black: device.create_texture_2d(&[0, 0, 0, 255], [1, 1], Format::Rgba8Unorm),
            settings: BloomSettings::default(),
            hdr: HdrSettings::default(),
        }
    }

    pub fn begin_frame(&mut self, settings: &BloomSettings, hdr: &HdrSettings) {
        self.settings = *settings;
        self.hdr = *hdr;
    }

    pub fn black_view(&self) -> TextureViewId {
        self.black
    }

    /// Zero when bloom is off, which is what makes the tonemap pass's blend a
    /// no-op without a second pipeline.
    pub fn strength(&self) -> f32 {
        if self.settings.enabled {
            self.settings.strength
        } else {
            0.0
        }
    }

    /// Half the frame, exposure applied, fireflies weighted down.
    pub fn record_prefilter(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        source: TextureViewId,
        target: TextureViewId,
        exposure: BufferId,
    ) {
        let set = device.create_bind_group(
            self.prefilter_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::SampledTexture {
                        view: source,
                        sampler: self.sampler,
                    },
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::StorageTexture(target),
                },
                Binding {
                    binding: 2,
                    resource: BindingResource::StorageBuffer(exposure),
                },
            ],
        );

        encoder.bind_pipeline(self.prefilter_pipeline);
        encoder.bind_group(0, set);
        encoder.push_constants(&[
            self.hdr.manual_exposure().to_bits(),
            self.hdr.auto_exposure as u32,
        ]);
        dispatch_over(device, encoder, target);
    }

    pub fn record_downsample(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        source: TextureViewId,
        target: TextureViewId,
    ) {
        let set = device.create_bind_group(
            self.downsample_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::SampledTexture {
                        view: source,
                        sampler: self.sampler,
                    },
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::StorageTexture(target),
                },
            ],
        );

        encoder.bind_pipeline(self.downsample_pipeline);
        encoder.bind_group(0, set);
        dispatch_over(device, encoder, target);
    }

    /// `coarse` is the level above being spread; `same_level` is the down-chain
    /// level of the target's size, which the spread is added to.
    pub fn record_upsample(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        coarse: TextureViewId,
        same_level: TextureViewId,
        target: TextureViewId,
    ) {
        let set = device.create_bind_group(
            self.upsample_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::SampledTexture {
                        view: coarse,
                        sampler: self.sampler,
                    },
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::SampledTexture {
                        view: same_level,
                        sampler: self.sampler,
                    },
                },
                Binding {
                    binding: 2,
                    resource: BindingResource::StorageTexture(target),
                },
            ],
        );

        encoder.bind_pipeline(self.upsample_pipeline);
        encoder.bind_group(0, set);
        encoder.push_constants(&[
            self.settings.radius.to_bits(),
            self.settings.scatter.to_bits(),
        ]);
        dispatch_over(device, encoder, target);
    }
}

/// Cover every texel of `target` with `TILE`-sized groups. Every shader in the
/// chain bounds-checks against `imageSize`, so a partial group at the edge
/// writes nothing extra.
fn dispatch_over(device: &Device, encoder: &mut Encoder, target: TextureViewId) {
    let extent = device.view_extent(target);
    encoder.dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1]);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A compute pass that writes a storage texture, on a real device: the
    /// prefilter of a flat grey frame is a flat grey frame, so anything but
    /// zeros is the pass having run and stored.
    #[test]
    #[ignore = "needs a Metal device"]
    #[cfg(target_os = "macos")]
    fn the_prefilter_stores_into_a_storage_texture_on_metal() {
        use crate::gfx::graph::{BufferUsage, ImageUsage};
        use crate::gfx::rhi::{BufferDesc, MemoryLocation, TextureDesc, TextureDimension};
        let device = Device::Metal(crate::gfx::metal::MtlDriver::new().expect("a Metal device"));
        device.begin_frame(0);
        let pass = BloomPass::new(&device);
        let source = device.create_texture_2d(
            &bytemuck::cast_slice(&[0.5f32, 0.5, 0.5, 1.0].repeat(64)).to_vec(),
            [8, 8],
            Format::Rgba32Float,
        );
        let target = device.create_texture(&TextureDesc {
            format: Format::Rgba16Float,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            mip_levels: 1,
            array_layers: 1,
            samples: crate::gfx::format::SampleCount::Sample1,
            usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
            memoryless: false,
        });
        let exposure = device.create_buffer_bytes(
            BufferDesc {
                size: 12,
                usage: BufferUsage::STORAGE_BUFFER,
                location: MemoryLocation::Upload,
            },
            bytemuck::cast_slice(&[1.0f32, 1.0, 0.0]),
        );
        device.record_blocking(|encoder| {
            pass.record_prefilter(&device, encoder, source, target, exposure);
        });
        let Device::Metal(driver) = &device else {
            unreachable!()
        };
        let pixels = driver.read_texture(target);
        assert!(
            pixels.iter().any(|&b| b != 0),
            "nothing was stored: {pixels:?}"
        );
    }

    /// The chain must stop before a level is too small for a 13-tap filter to
    /// mean anything, and must never exceed the array the ids are carried in.
    #[test]
    fn the_chain_length_suits_the_frame() {
        assert_eq!(mip_count([1920, 1080]), MAX_BLOOM_MIPS as u8);
        assert_eq!(mip_count([3840, 2160]), MAX_BLOOM_MIPS as u8);
        // 1080 >> 6 == 16, still above the floor; 256 >> 5 == 8 is the last that
        // clears it.
        assert_eq!(mip_count([256, 256]), 5);
        assert_eq!(mip_count([64, 64]), 3);
        assert_eq!(mip_count([16, 16]), 1);
    }

    /// A window dragged to nothing must not declare a chain at all, rather than
    /// one whose levels resolve to a single texel.
    #[test]
    fn a_tiny_frame_gets_no_chain() {
        assert_eq!(mip_count([8, 8]), 0);
        assert_eq!(mip_count([1, 1]), 0);
        // Narrow but tall: the smaller side is what limits the chain.
        assert_eq!(mip_count([4096, 12]), 0);
    }
}
