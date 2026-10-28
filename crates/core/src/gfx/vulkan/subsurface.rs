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

use std::sync::Arc;

use vulkano::buffer::BufferContents;
use vulkano::descriptor_set::WriteDescriptorSet;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};

use crate::scene::{Camera, SubsurfaceSettings};

use super::context::VkContext;
use super::record::Recorder;

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
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
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
    blur_pipeline: Arc<ComputePipeline>,
    composite_pipeline: Arc<ComputePipeline>,
    /// Nearest, and it matters. Every fetch in both shaders is a `texelFetch` at
    /// an exact texel — the kernel does its own weighting, and a bilinear tap
    /// would average across the silhouettes the depth term exists to reject
    /// *before* that term ever sees them.
    nearest_clamp: Arc<Sampler>,
    push: Option<BlurPush>,
}

impl SubsurfacePass {
    pub fn new(ctx: &VkContext) -> Self {
        let device = &ctx.device;
        let blur_pipeline = build_pipeline(
            ctx,
            blur_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );
        let composite_pipeline = build_pipeline(
            ctx,
            composite_cs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );

        let nearest_clamp = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::default()
            },
        )
        .unwrap();

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
        builder: &mut Recorder,
        ctx: &VkContext,
        source: Arc<ImageView>,
        depth: Arc<ImageView>,
        target: Arc<ImageView>,
        vertical: bool,
    ) {
        let set = ctx.descriptor_sets.get(
            &ctx.descriptor_set_allocator,
            &self.blur_pipeline.layout().set_layouts()[0],
            [
                WriteDescriptorSet::image_view_sampler(0, source, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(1, depth, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view(2, target.clone()),
            ],
        );

        builder
            .bind_pipeline_compute(&self.blur_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.blur_pipeline.layout(),
                0,
                &[set],
            )
            .push_constants(self.blur_pipeline.layout(), 0, &self.push_for(vertical));
        dispatch_over(builder, &target);
    }

    pub(super) fn record_composite(
        &self,
        builder: &mut Recorder,
        ctx: &VkContext,
        source: Arc<ImageView>,
        diffused: Arc<ImageView>,
        target: Arc<ImageView>,
    ) {
        let set = ctx.descriptor_sets.get(
            &ctx.descriptor_set_allocator,
            &self.composite_pipeline.layout().set_layouts()[0],
            [
                WriteDescriptorSet::image_view_sampler(0, source, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(1, diffused, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view(2, target.clone()),
            ],
        );

        builder
            .bind_pipeline_compute(&self.composite_pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.composite_pipeline.layout(),
                0,
                &[set],
            );
        dispatch_over(builder, &target);
    }
}

fn dispatch_over(builder: &mut Recorder, target: &Arc<ImageView>) {
    let extent = target.image().extent();

    // SAFETY: the dispatch covers exactly `extent`, and both shaders discard
    // invocations past `imageSize`, so nothing writes outside the image. The
    // descriptors bound above match the shader's layout, and the graph declared
    // every resource these passes touch, so its barriers precede them.
    builder.dispatch([extent[0].div_ceil(TILE), extent[1].div_ceil(TILE), 1]);
}

fn build_pipeline(
    ctx: &VkContext,
    entry_point: vulkano::shader::EntryPoint,
) -> Arc<ComputePipeline> {
    let device = &ctx.device;
    let stage = PipelineShaderStageCreateInfo::new(entry_point);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    ComputePipeline::new(
        device.clone(),
        ctx.pipeline_cache(),
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

mod blur_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/sss_blur.comp" }
}

mod composite_cs {
    vulkano_shaders::shader! { ty: "compute", path: "shaders/sss_composite.comp" }
}
