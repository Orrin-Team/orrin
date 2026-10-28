//! Contact shadows: a short screen-space march toward the sun, and the mask it
//! leaves for the forward pass to multiply the sun's term by.
//!
//! It exists because a cascade cannot resolve the band it covers. A texel of
//! the nearest cascade is tens of centimetres of world space at any sensible
//! range, and the normal-offset bias that keeps that texel from self-shadowing
//! pushes the lookup a texel's width off the surface — so everything nearer the
//! contact than that width is reported lit whatever the map holds. Screen space
//! has more resolution than the map at exactly that scale and nowhere else,
//! which is what makes a quarter-metre ray worth sixteen depth fetches and a
//! two-metre one not.
//!
//! Its own pass rather than a loop inside `forward.frag`, for the reason the
//! SSAO resolve is its own pass: one march per screen pixel instead of one per
//! shaded fragment, and a target the forward shader samples the way it already
//! samples occlusion. The march reads the *jittered* projection, because the
//! depth it reconstructs from was rasterised with it.

use glam::Vec3;

use crate::gfx::rhi::{
    Binding, BindingResource, BufferId, Device, Encoder, GraphicsPipelineDesc, PipelineId,
    SamplerDesc, SamplerId, TextureViewId, Viewport,
};
use crate::gfx::shaders::ShaderId;
use crate::scene::ContactShadowSettings;

use super::driver::VkDriver;
use super::taa::FrameView;

/// One channel, because visibility is one number. The same format the AO target
/// uses, and for the same reason it is a colour attachment rather than a storage
/// image: `R8_UNORM` is only guaranteed as a storage format behind an optional
/// device feature, while every implementation can render to it.
pub(super) const MASK_FORMAT: crate::gfx::format::Format = crate::gfx::format::Format::R8Unorm;

/// How long the dither sequence runs before repeating.
///
/// Eight, because that is TAA's accumulation window: the resolve averages about
/// that many frames, so a longer sequence would only hand it offsets it never
/// finishes integrating, and a shorter one would hand it the same ray twice.
const DITHER_PHASES: u64 = 8;

pub(super) struct ContactShadowUniforms {
    frame: BufferId,
    params: BufferId,
}

pub struct ContactShadowPass {
    pipeline: PipelineId,
    /// Nearest, because depth and normals are fetched at exact texels: filtering
    /// either invents a surface between two that are really there, and the march
    /// would stop on it.
    nearest_clamp: SamplerId,
    /// 1x1 white (= fully lit), bound when the pass is off so the forward shader
    /// has one path rather than two. Not a graph resource: with contact shadows
    /// off the graph has no node to declare a read of.
    lit_view: TextureViewId,
    /// Advanced every frame the pass runs, and what decorrelates the dither
    /// frame to frame. Without it every frame marches from the same offset and
    /// the temporal resolve averages eight copies of one march.
    frame: u64,
}

impl ContactShadowPass {
    pub fn new(device: &Device) -> Self {
        let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc::fullscreen(
            ShaderId::FullscreenVert,
            ShaderId::ContactShadowsFrag,
            &[MASK_FORMAT],
        ));

        let nearest_clamp = device.create_sampler(SamplerDesc {
            linear: false,
            mipmap_linear: false,
            repeat: false,
            compare: false,
            anisotropy: None,
            max_lod: Some(0.0),
        });

        Self {
            pipeline,
            nearest_clamp,
            lit_view: device.create_texture_2d(&[255u8], [1, 1], MASK_FORMAT),
            frame: 0,
        }
    }

    /// A 1x1 "fully lit" view, bound when the pass is off.
    pub fn lit_view(&self) -> TextureViewId {
        self.lit_view
    }

    /// Resolve this frame's march against the settings and the sun.
    ///
    /// `to_sun` is the world-space direction toward the light, from
    /// [`DirectionalLight::direction_to_light`](crate::gfx::DirectionalLight::direction_to_light)
    /// — the same call the forward pass shades with, so the two cannot disagree
    /// about which side of a surface the sun is on. The view matrix here is the
    /// frame's, not the camera's own, for the reason every rasterising pass
    /// takes its matrices from `FrameView`.
    pub(super) fn begin_frame(
        &mut self,
        driver: &VkDriver,
        settings: &ContactShadowSettings,
        view: &FrameView,
        to_sun: Vec3,
        frame: BufferId,
    ) -> ContactShadowUniforms {
        // A direction, so the view matrix's translation must not apply.
        let light_direction = view.view.transform_vector3(to_sun).normalize_or_zero();

        // Mirrors `ContactShadowSettings::distance_fade`, which is the tested
        // definition of this ramp; the shader is handed the two ends rather than
        // the fraction so it evaluates one smoothstep.
        let fade_end = settings.fade_distance.max(1e-4);
        let fade_start = fade_end * 0.75;

        let march = [
            settings.ray_length.max(0.0),
            settings.bias.max(0.0),
            // The far end of the window, summed here rather than in the
            // shader so the two dials stay independent: a thickness below
            // the bias would otherwise close the window entirely, and the
            // pass would spend a full march per pixel to produce a mask of
            // ones.
            settings.bias.max(0.0) + settings.thickness.max(0.0),
            settings.steps.max(1) as f32,
        ];
        let fade = [
            settings.intensity.clamp(0.0, 1.0),
            fade_start,
            fade_end,
            (self.frame % DITHER_PHASES) as f32,
        ];
        let direction = [light_direction.x, light_direction.y, light_direction.z, 0.0];
        let mut words = [0u32; 12];
        for (word, value) in words
            .iter_mut()
            .zip(direction.iter().chain(&march).chain(&fade))
        {
            *word = value.to_bits();
        }
        let params = driver.upload_uniform(&words);
        self.frame = self.frame.wrapping_add(1);

        ContactShadowUniforms { frame, params }
    }

    /// `depth_view` and `normal_view` are the prepass's graph outputs, declared
    /// as this pass's `Sampled` inputs.
    pub(super) fn record(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        extent: [u32; 2],
        uniforms: &ContactShadowUniforms,
        depth_view: TextureViewId,
        normal_view: TextureViewId,
    ) {
        let uniform_set = device.create_transient_bind_group(
            self.pipeline,
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
            self.pipeline,
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
            ],
        );

        encoder.set_viewport(Viewport {
            origin: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: [0.0, 1.0],
        });
        encoder.bind_pipeline(self.pipeline);
        encoder.bind_group(0, uniform_set);
        encoder.bind_group(1, texture_set);
        encoder.draw(3, 1, 0, 0);
    }
}
