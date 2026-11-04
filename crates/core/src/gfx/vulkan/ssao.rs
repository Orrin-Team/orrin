use super::prepass::NORMAL_FORMAT;
use crate::gfx::rhi::{
    Binding, BindingResource, BufferId, Device, Encoder, GraphicsPipelineDesc, PipelineId,
    SamplerDesc, SamplerId, TextureViewId, Viewport,
};
use crate::gfx::shaders::ShaderId;

pub(super) const AO_FORMAT: crate::gfx::format::Format = crate::gfx::format::Format::R8Unorm;
const NOISE_SIZE: u32 = 4;

/// The resolve's tunables, as the words of its std140 block: the noise scale,
/// radius, bias and power, padded to a multiple of 16 bytes.
const PARAMS_WORDS: usize = 8;

/// A 4x4 ordered-dither matrix: neighbouring cells as far apart in value as
/// sixteen cells allow, so any 2x2 of the tile already spans the range.
const BAYER: [u32; 16] = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];

/// The noise tile, row-major: red is the slice rotation, green the step jitter,
/// each a fraction of a step in [0, 1).
///
/// Stratified rather than random. The blur after the resolve is a 4x4
/// footprint, the same size as this tile, so it averages exactly these sixteen
/// rotations — and sixteen evenly spaced ones are what make an open surface
/// average to full visibility at any tilt, where sixteen random draws would
/// leave it off by whatever they happened to be. The jitter is the same sixteen
/// values in another order, so a pixel's rotation says nothing about its steps.
fn noise_texels() -> [[u8; 4]; 16] {
    let unorm = |k: u32| ((k as f32 + 0.5) / 16.0 * 255.0).round() as u8;
    std::array::from_fn(|i| {
        let rotation = BAYER[i];
        let jitter = (rotation * 5 + 3) % 16;
        [unorm(rotation), unorm(jitter), 0, 255]
    })
}

fn build_noise(device: &Device) -> TextureViewId {
    let pixels: Vec<u8> = noise_texels().into_iter().flatten().collect();
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
    white_view: TextureViewId,
    pub radius: f32,
    pub bias: f32,
    pub power: f32,
}

impl SsaoPass {
    pub fn new(device: &Device) -> Self {
        // One description for both, where there were two identical render pass
        // objects: the resolve and the blur write the same format to the same
        // shape of target, and that was always the only thing either declared.
        let pipeline = |fragment| {
            device.create_graphics_pipeline(&GraphicsPipelineDesc::fullscreen(
                ShaderId::FullscreenVert,
                fragment,
                &[AO_FORMAT],
            ))
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
            white_view: device.create_texture_2d(&[255u8], [1, 1], AO_FORMAT),
            radius: 1.0,
            bias: 0.025,
            power: 1.0,
        }
    }

    /// A 1x1 white (=1.0) view, bound when SSAO is disabled.
    pub fn white_view(&self) -> TextureViewId {
        self.white_view
    }

    pub(super) fn begin_frame(
        &self,
        device: &Device,
        extent: [u32; 2],
        frame: BufferId,
    ) -> SsaoUniforms {
        let mut words = [0u32; PARAMS_WORDS];
        words[0] = (extent[0] as f32 / NOISE_SIZE as f32).to_bits();
        words[1] = (extent[1] as f32 / NOISE_SIZE as f32).to_bits();
        words[2] = self.radius.to_bits();
        words[3] = self.bias.to_bits();
        words[4] = self.power.to_bits();

        SsaoUniforms {
            frame,
            params: device.upload_uniform(&words),
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
        let uniform_set = device.create_transient_bind_group(
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
        let frame_set = device.create_transient_bind_group(
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

#[cfg(test)]
mod tests {
    use std::f32::consts::{FRAC_PI_2, PI};

    use glam::Vec3;

    use super::*;

    /// Slices per pixel. Lock-stepped with `SLICES` in `ssao.frag`, which keeps
    /// it a constant there so the march unrolls.
    const SLICES: usize = 2;

    /// `slice_frame` in `ssao.frag`: the normal projected into the slice
    /// plane, as its angle `n` from the view vector, its cosine, and its length.
    fn shader_slice_frame(n_view: Vec3, v: Vec3, direction: Vec3) -> (f32, f32, f32) {
        let ortho = direction - direction.dot(v) * v;
        let axis = ortho.cross(v).normalize();
        let projected = n_view - axis * n_view.dot(axis);
        let len = projected.length();
        let cos_n = (projected.dot(v) / len.max(1e-6)).clamp(0.0, 1.0);
        let sign = if ortho.dot(projected) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        (sign * cos_n.acos(), cos_n, len)
    }

    /// `slice_visibility` in `ssao.frag`, the closed-form cosine-weighted
    /// integral between two horizons. The two are a lock-step pair.
    fn shader_slice_visibility(frame: (f32, f32, f32), hc0: f32, hc1: f32) -> f32 {
        let (n, cos_n, len) = frame;
        let h0 = n + (-hc1.clamp(-1.0, 1.0).acos() - n).clamp(-FRAC_PI_2, FRAC_PI_2);
        let h1 = n + (hc0.clamp(-1.0, 1.0).acos() - n).clamp(-FRAC_PI_2, FRAC_PI_2);
        let arc = |h: f32| (cos_n + 2.0 * h * n.sin() - (2.0 * h - n).cos()) * 0.25;
        len * (arc(h0) + arc(h1))
    }

    /// The horizons an open surface leaves: none, so each side's is the
    /// tangent plane, which is where the shader starts its search.
    fn open(frame: (f32, f32, f32)) -> (f32, f32) {
        ((frame.0 + FRAC_PI_2).cos(), (frame.0 - FRAC_PI_2).cos())
    }

    /// The two slices one pixel's noise texel rotates to, as `ssao.frag`
    /// derives them.
    fn slices(rotation: f32) -> [Vec3; SLICES] {
        std::array::from_fn(|s| {
            let phi = (s as f32 + rotation) * PI / SLICES as f32;
            Vec3::new(phi.cos(), phi.sin(), 0.0)
        })
    }

    /// What the blur's 4x4 footprint averages an open surface to: every
    /// rotation in the tile, every slice of each.
    fn tile_visibility(n_view: Vec3) -> f32 {
        let v = Vec3::Z;
        let texels = noise_texels();
        let sum: f32 = texels
            .iter()
            .flat_map(|texel| slices(texel[0] as f32 / 255.0))
            .map(|direction| {
                let frame = shader_slice_frame(n_view, v, direction);
                let (hc0, hc1) = open(frame);
                shader_slice_visibility(frame, hc0, hc1)
            })
            .sum();
        sum / (texels.len() * SLICES) as f32
    }

    /// The property the closed form exists for: nothing in the way reads as
    /// fully visible, whichever way the surface faces the camera — so a wall
    /// seen at a glancing angle is not darkened for being glanced at.
    #[test]
    fn an_open_surface_is_fully_visible_at_any_tilt() {
        for tilt in [0.0f32, 20.0, 45.0, 60.0, 75.0] {
            for azimuth in [0.0f32, 37.0, 90.0, 200.0] {
                let (t, a) = (tilt.to_radians(), azimuth.to_radians());
                let n = Vec3::new(t.sin() * a.cos(), t.sin() * a.sin(), t.cos());
                let visibility = tile_visibility(n);
                assert!(
                    (visibility - 1.0).abs() < 0.01,
                    "{visibility} at tilt {tilt}, azimuth {azimuth}"
                );
            }
        }
    }

    /// What one pixel writes for an open surface: its slices' visibility over
    /// their unoccluded visibility, clamped to the [0, 1] the R8 target holds.
    fn pixel_visibility(n_view: Vec3, rotation: f32, normalised: bool) -> f32 {
        let (mut seen, mut open_sum) = (0.0, 0.0);
        for direction in slices(rotation) {
            let frame = shader_slice_frame(n_view, Vec3::Z, direction);
            let (hc0, hc1) = open(frame);
            seen += shader_slice_visibility(frame, hc0, hc1);
            open_sum += shader_slice_visibility(
                frame,
                (frame.0 + FRAC_PI_2).cos(),
                (frame.0 - FRAC_PI_2).cos(),
            );
        }
        let raw = if normalised {
            seen / open_sum
        } else {
            seen / SLICES as f32
        };
        raw.clamp(0.0, 1.0)
    }

    /// Two slices of a steeply tilted surface read up to a tenth either side
    /// of full visibility, and the target clamps at one *before* the blur
    /// averages — so only the low side survives, and an open surface comes
    /// out darker the more it is tilted, in a pattern that follows its normal.
    /// Normalising each pixel by its own slices' unoccluded integral removes
    /// that before anything is clamped.
    #[test]
    fn clamping_before_the_blur_does_not_darken_an_open_surface() {
        let tilt = 75f32.to_radians();
        let n = Vec3::new(tilt.sin() * 0.6, tilt.sin() * 0.8, tilt.cos());
        let mean = |normalised| {
            let texels = noise_texels();
            texels
                .iter()
                .map(|t| pixel_visibility(n, t[0] as f32 / 255.0, normalised))
                .sum::<f32>()
                / texels.len() as f32
        };
        assert!(
            mean(false) < 0.99,
            "the bias this guards against is gone: {}",
            mean(false)
        );
        assert!((mean(true) - 1.0).abs() < 1e-4, "{}", mean(true));
    }

    /// A surface facing the camera with an occluder rising straight up along
    /// its normal on one side of the slice has lost exactly that side's half.
    #[test]
    fn a_horizon_at_the_normal_closes_its_half_of_the_slice() {
        let frame = shader_slice_frame(Vec3::Z, Vec3::Z, Vec3::X);
        let (_, hc1) = open(frame);
        let half = shader_slice_visibility(frame, 1.0, hc1);
        assert!((half - 0.5).abs() < 1e-5, "{half}");
        let closed = shader_slice_visibility(frame, 1.0, 1.0);
        assert!(closed.abs() < 1e-5, "{closed}");
    }

    /// Stratified, not random: the 4x4 blur sees each of sixteen evenly spaced
    /// rotations exactly once, which is what `an_open_surface_is_fully_visible_
    /// at_any_tilt` relies on. Random rotations would leave a tilted surface's
    /// 4x4 average off by whatever the draw happened to be.
    #[test]
    fn the_noise_tile_holds_sixteen_evenly_spaced_rotations() {
        let mut rotations: Vec<f32> = noise_texels().iter().map(|t| t[0] as f32 / 255.0).collect();
        rotations.sort_by(f32::total_cmp);
        for (k, rotation) in rotations.iter().enumerate() {
            let expected = (k as f32 + 0.5) / 16.0;
            assert!(
                (rotation - expected).abs() < 1.0 / 255.0,
                "{rotation} for {expected}"
            );
        }
    }
}
