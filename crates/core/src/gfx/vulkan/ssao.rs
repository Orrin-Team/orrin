use std::sync::Arc;

use vulkano::buffer::Subbuffer;
use vulkano::image::view::ImageView;

use super::context::VkContext;
use super::convert::vk_format;
use super::driver::VkDriver;
use super::prepass::{FrameUbo, NORMAL_FORMAT};
use super::texture::MipPolicy;
use crate::gfx::rhi::{
    Binding, BindingResource, BufferId, Device, Encoder, GraphicsPipelineDesc, PipelineId,
    SamplerDesc, SamplerId, TextureViewId, Viewport,
};
use crate::gfx::shaders::ShaderId;

pub(super) const AO_FORMAT: crate::gfx::format::Format = crate::gfx::format::Format::R8Unorm;
const NOISE_SIZE: u32 = 4;
const KERNEL_SIZE: usize = 32;
const KERNEL_MAX: usize = 64;

/// The resolve's tunables, as the words of its std140 block: the kernel, then
/// the noise scale, radius, bias, power and kernel size, padded from 1048 to 1056
/// bytes (a multiple of 16).
const PARAMS_WORDS: usize = KERNEL_MAX * 4 + 8;

fn build_kernel() -> [[f32; 4]; KERNEL_MAX] {
    let mut seed: u64 = 0x2545F491_4F6CDD1D;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / (1u64 << 31) as f32 // [0,1)
    };
    let mut kernel = [[0.0f32; 4]; KERNEL_MAX];
    for (i, k) in kernel.iter_mut().enumerate().take(KERNEL_SIZE) {
        let v = glam::Vec3::new(next() * 2.0 - 1.0, next() * 2.0 - 1.0, next()).normalize_or_zero()
            * next();
        let t = i as f32 / KERNEL_SIZE as f32;
        let v = v * (0.1 + 0.9 * t * t); // cluster samples near the origin
        *k = [v.x, v.y, v.z, 0.0];
    }
    kernel
}

fn build_noise(device: &Device) -> TextureViewId {
    let mut seed: u64 = 0x9E3779B9_7F4A7C15;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as f32 / (1u64 << 31) as f32
    };
    let mut pixels = Vec::with_capacity((NOISE_SIZE * NOISE_SIZE * 4) as usize);
    for _ in 0..(NOISE_SIZE * NOISE_SIZE) {
        let x = next() * 2.0 - 1.0;
        let y = next() * 2.0 - 1.0;
        pixels.extend_from_slice(&[
            ((x * 0.5 + 0.5) * 255.0) as u8,
            ((y * 0.5 + 0.5) * 255.0) as u8,
            0,
            255,
        ]);
    }
    device.create_texture_2d(&pixels, [NOISE_SIZE, NOISE_SIZE], NORMAL_FORMAT)
}

/// What the resolve binds: the tunables, plus the prepass's camera block. The
/// camera half is uploaded by the prepass rather than here, because every other
/// consumer of the prepass reads the same one — the resolve reconstructs view
/// positions from the depth that pass wrote, so it has to use the projection
/// that wrote it, jitter and all.
pub(super) struct SsaoUniforms {
    frame: BufferId,
    params: BufferId,
}

pub struct SsaoPass {
    ssao_pipeline: PipelineId,
    blur_pipeline: PipelineId,
    nearest_clamp: SamplerId,
    nearest_repeat: SamplerId,
    noise_view: TextureViewId,
    /// 1x1 white (=1.0) AO view bound when SSAO is disabled, so the forward
    /// shader samples "no occlusion" without a separate code path. Not a graph
    /// resource: with SSAO off the graph has no AO node at all, so there is
    /// nothing for the forward pass to declare a read of.
    white_view: Arc<ImageView>,
    kernel: [[f32; 4]; KERNEL_MAX],
    pub radius: f32,
    pub bias: f32,
    pub power: f32,
}

impl SsaoPass {
    pub fn new(ctx: &VkContext, device: &Device) -> Self {
        // One description for both, where there were two identical render pass
        // objects: the resolve and the blur write the same format to the same
        // shape of target, and that was always the only thing either declared.
        let pipeline = |fragment| {
            device.create_graphics_pipeline(&GraphicsPipelineDesc {
                vertex: ShaderId::FullscreenVert,
                fragment,
                color_formats: &[AO_FORMAT],
                depth_format: None,
            })
        };
        let ssao_pipeline = pipeline(ShaderId::SsaoFrag);
        let blur_pipeline = pipeline(ShaderId::SsaoBlurFrag);

        let sampler = |repeat| {
            device.create_sampler(SamplerDesc {
                linear: false,
                mipmap_linear: false,
                repeat,
                compare: false,
                anisotropy: None,
                max_lod: Some(0.0),
            })
        };
        let nearest_clamp = sampler(false);
        let nearest_repeat = sampler(true);

        Self {
            ssao_pipeline,
            blur_pipeline,
            nearest_clamp,
            nearest_repeat,
            noise_view: build_noise(device),
            white_view: super::texture::upload_texture(
                ctx,
                &[255u8],
                [1, 1],
                vk_format(AO_FORMAT),
                MipPolicy::None,
            ),
            kernel: build_kernel(),
            radius: 1.0,
            bias: 0.025,
            power: 1.0,
        }
    }

    /// A 1x1 white (=1.0) view, bound when SSAO is disabled.
    pub fn white_view(&self) -> Arc<ImageView> {
        self.white_view.clone()
    }

    pub(super) fn begin_frame(
        &self,
        driver: &VkDriver,
        extent: [u32; 2],
        frame: Subbuffer<FrameUbo>,
    ) -> SsaoUniforms {
        let mut words = [0u32; PARAMS_WORDS];
        for (word, value) in words.iter_mut().zip(self.kernel.iter().flatten()) {
            *word = value.to_bits();
        }
        let tail = KERNEL_MAX * 4;
        words[tail] = (extent[0] as f32 / NOISE_SIZE as f32).to_bits();
        words[tail + 1] = (extent[1] as f32 / NOISE_SIZE as f32).to_bits();
        words[tail + 2] = self.radius.to_bits();
        words[tail + 3] = self.bias.to_bits();
        words[tail + 4] = self.power.to_bits();
        words[tail + 5] = KERNEL_SIZE as u32;

        SsaoUniforms {
            frame: driver.import_buffer(frame.into_bytes()),
            params: driver.upload_uniform(&words),
        }
    }

    /// `depth_view` and `normal_view` are the prepass's graph outputs, declared
    /// as this pass's `Sampled` inputs.
    pub(super) fn record_ao(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        extent: [u32; 2],
        uniforms: &SsaoUniforms,
        depth_view: TextureViewId,
        normal_view: TextureViewId,
    ) {
        let uniform_set = device.create_bind_group(
            self.ssao_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::UniformBuffer(uniforms.frame),
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::UniformBuffer(uniforms.params),
                },
            ],
        );
        let texture_set = device.create_bind_group(
            self.ssao_pipeline,
            1,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::SampledTexture {
                        view: depth_view,
                        sampler: self.nearest_clamp,
                    },
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::SampledTexture {
                        view: normal_view,
                        sampler: self.nearest_clamp,
                    },
                },
                Binding {
                    binding: 2,
                    resource: BindingResource::SampledTexture {
                        view: self.noise_view,
                        sampler: self.nearest_repeat,
                    },
                },
            ],
        );

        set_viewport(encoder, extent);
        encoder.bind_pipeline(self.ssao_pipeline);
        encoder.bind_group(0, uniform_set);
        encoder.bind_group(1, texture_set);
        encoder.draw(3, 1, 0, 0);
    }

    /// The blur is also the upsample when the occlusion was resolved at half the
    /// frame's extent, which is why it takes the prepass depth: it weights each
    /// tap by how close that tap's surface is to the centre's, so the term stops
    /// bleeding across silhouettes. See `ssao_blur.frag`.
    pub(super) fn record_blur(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        extent: [u32; 2],
        raw_ao_view: TextureViewId,
        depth_view: TextureViewId,
        uniforms: &SsaoUniforms,
    ) {
        let input = device.create_bind_group(
            self.blur_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::SampledTexture {
                        view: raw_ao_view,
                        sampler: self.nearest_clamp,
                    },
                },
                // Nearest, like every other read of this depth: interpolating
                // across a silhouette invents a surface, which is precisely what
                // the weighting below exists to stop.
                Binding {
                    binding: 1,
                    resource: BindingResource::SampledTexture {
                        view: depth_view,
                        sampler: self.nearest_clamp,
                    },
                },
            ],
        );
        let frame_set = device.create_bind_group(
            self.blur_pipeline,
            1,
            &[Binding {
                binding: 0,
                resource: BindingResource::UniformBuffer(uniforms.frame),
            }],
        );

        set_viewport(encoder, extent);
        encoder.bind_pipeline(self.blur_pipeline);
        encoder.bind_group(0, input);
        encoder.bind_group(1, frame_set);
        encoder.draw(3, 1, 0, 0);
    }
}

fn set_viewport(encoder: &mut Encoder, extent: [u32; 2]) {
    encoder.set_viewport(Viewport {
        origin: [0.0, 0.0],
        extent: [extent[0] as f32, extent[1] as f32],
        depth_range: [0.0, 1.0],
    });
}
