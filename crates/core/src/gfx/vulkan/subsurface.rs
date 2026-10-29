//! Subsurface scattering, the screen-space half: two separable blurs over the
//! radiance that left the surface somewhere other than where it arrived, and a
//! composite that adds it back.
//!
//! The analytic half lives in `shading.glsl` and runs in every queue. This is the
//! part that can only run on the opaque one, and the reason is the reason every
//! screen-space effect here has: it reads the geometry prepass, and blended and
//! refractive surfaces are not in it. What those queues get instead is the wrapped
//! diffuse `subsurface_wrap` switches on precisely when this pass is absent — the
//! two model the same transport, so exactly one of them applies at a time.
//!
//! The split that makes any of this correct happens upstream. `forward_sss.frag`
//! writes the diffusible radiance into a second target and leaves every specular
//! lobe in the first, so what is blurred here is only light that physically moved
//! across the surface. Blurring the lit frame instead — the cheap version of this
//! effect, and the one that gives skin a greasy look — would spread the highlights
//! with it.
//!
//! Where it sits in the frame is not a detail: immediately after shading and
//! before the reflections. After shading because there is nothing to diffuse until
//! the surface has been lit; before the reflections because a mirror beside a face
//! should show the face as the frame will show it, and the trace samples whatever
//! colour it is handed.

use crate::scene::{Camera, SubsurfaceSettings};

use crate::gfx::rhi::{
    Binding, BindingResource, ComputePipelineDesc, Device, Encoder, PipelineId, SamplerDesc,
    SamplerId, TextureViewId,
};
use crate::gfx::shaders::ShaderId;

/// Side of the compute workgroup, matching both shaders' `local_size`.
const TILE: u32 = 8;

/// What the forward pass's second target holds: HDR radiance in `rgb` and, in
/// `a`, the widest channel's mean free path in metres.
///
/// The same format as the colour target beside it, and it has to be a float one
/// for both halves — the radiance is HDR, and the radius is a few thousandths of a
/// metre. Sixteen bits carries both comfortably; the alternative, a separate
/// single-channel target for the radius, would be a second attachment and a second
/// resolve for one number that is already free here.
pub(super) const SUBSURFACE_FORMAT: crate::gfx::format::Format =
    crate::gfx::format::Format::Rgba16Float;

/// Which axis a blur dispatch walks, and everything it needs to size its kernel.
#[derive(Clone, Copy)]
struct BlurPush {
    /// xy = the axis in texels, z = near plane, w = far plane.
    direction: [f32; 4],
    /// rgb = per-channel reach as a fraction of the widest, w = the world-to-pixel
    /// scale at unit depth.
    profile: [f32; 4],
    /// x = the widest kernel a pixel may open, in pixels.
    params: [f32; 4],
}

pub struct SubsurfacePass {
    blur_pipeline: PipelineId,
    composite_pipeline: PipelineId,
    /// Nearest, and it matters. Every fetch in both shaders is a `texelFetch` at
    /// an exact texel — the kernel does its own weighting, and a bilinear tap
    /// would average across the silhouettes the depth term exists to reject
    /// *before* that term ever sees them.
    nearest_clamp: SamplerId,
    push: Option<BlurPush>,
}

impl SubsurfacePass {
    pub fn new(device: &Device) -> Self {
        let pipeline = |shader| device.create_compute_pipeline(&ComputePipelineDesc { shader });
        let blur_pipeline = pipeline(ShaderId::SssBlurComp);
        let composite_pipeline = pipeline(ShaderId::SssCompositeComp);

        let nearest_clamp = device.create_sampler(SamplerDesc {
            linear: false,
            mipmap_linear: false,
            repeat: false,
            compare: false,
            anisotropy: None,
            max_lod: Some(0.0),
        });

        Self {
            blur_pipeline,
            composite_pipeline,
            nearest_clamp,
            push: None,
        }
    }

    /// Resolve this frame's settings and camera into the block both blur
    /// dispatches read.
    ///
    /// Called before anything records, like the other per-frame `begin_frame`s:
    /// the graph decides how far apart the two axes end up in the schedule, and
    /// they must not re-derive the projection between them or the second would
    /// open a kernel of a different width than the first.
    pub(super) fn begin_frame(
        &mut self,
        settings: &SubsurfaceSettings,
        camera: &Camera,
        extent: [u32; 2],
    ) {
        let profile = settings.normalised_profile();
        // The projection's vertical term: a world length perpendicular to the view
        // at depth `z` covers `length * cot(fov/2) / z` of NDC, and NDC spans two
        // units across the frame's height. Taken from the field of view rather than
        // off the matrix so the sign convention — this projection's Y is flipped
        // for Vulkan's clip space — cannot leak into a kernel width.
        let cot_half_fov = 1.0 / (camera.fov_y * 0.5).tan();
        self.push = Some(BlurPush {
            direction: [1.0, 0.0, camera.near, camera.far],
            profile: [
                profile.x,
                profile.y,
                profile.z,
                cot_half_fov * extent[1] as f32 * 0.5,
            ],
            params: [settings.max_radius.max(1.0), 0.0, 0.0, 0.0],
        });
    }

    fn push_for(&self, vertical: bool) -> BlurPush {
        let mut push = self
            .push
            .expect("the graph scheduled a subsurface blur before begin_frame");
        if vertical {
            push.direction[0] = 0.0;
            push.direction[1] = 1.0;
        }
        push
    }

    /// One axis of the diffusion. `vertical` is the only difference between the
    /// two dispatches, which is why they share a pipeline.
    pub(super) fn record_blur(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        source: TextureViewId,
        depth: TextureViewId,
        target: TextureViewId,
        vertical: bool,
    ) {
        let set = device.create_bind_group(
            self.blur_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::SampledTexture {
                        view: source,
                        sampler: self.nearest_clamp,
                    },
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::SampledTexture {
                        view: depth,
                        sampler: self.nearest_clamp,
                    },
                },
                Binding {
                    binding: 2,
                    resource: BindingResource::StorageTexture(target),
                },
            ],
        );

        encoder.bind_pipeline(self.blur_pipeline);
        encoder.bind_group(0, set);
        let push = self.push_for(vertical);
        let mut words = [0u32; 12];
        for (word, value) in words.iter_mut().zip(
            push.direction
                .iter()
                .chain(&push.profile)
                .chain(&push.params),
        ) {
            *word = value.to_bits();
        }
        encoder.push_constants(&words);
        dispatch_over(device, encoder, target);
    }

    pub(super) fn record_composite(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        source: TextureViewId,
        diffused: TextureViewId,
        target: TextureViewId,
    ) {
        let set = device.create_bind_group(
            self.composite_pipeline,
            0,
            &[
                Binding {
                    binding: 0,
                    resource: BindingResource::SampledTexture {
                        view: source,
                        sampler: self.nearest_clamp,
                    },
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::SampledTexture {
                        view: diffused,
                        sampler: self.nearest_clamp,
                    },
                },
                Binding {
                    binding: 2,
                    resource: BindingResource::StorageTexture(target),
                },
            ],
        );

        encoder.bind_pipeline(self.composite_pipeline);
        encoder.bind_group(0, set);
        dispatch_over(device, encoder, target);
    }
}

fn dispatch_over(device: &Device, encoder: &mut Encoder, target: TextureViewId) {
    let extent = device.view_extent(target);
    encoder.dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1]);
}
