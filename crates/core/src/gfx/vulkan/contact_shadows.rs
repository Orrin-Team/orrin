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

use std::sync::Arc;

use glam::Vec3;
use vulkano::buffer::allocator::SubbufferAllocatorCreateInfo;
use vulkano::buffer::{BufferContents, BufferUsage, Subbuffer};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};

use crate::scene::ContactShadowSettings;

use super::PassCtx;
use super::context::VkContext;
use super::convert::vk_format;
use super::prepass::FrameUbo;
use super::record::{Arena, Recorder};
use super::rendering;
use super::taa::FrameView;
use super::texture::MipPolicy;

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

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct ContactShadowUbo {
    /// xyz = view-space direction toward the sun.
    light_direction: [f32; 4],
    /// x = ray length, y = min depth separation, z = max depth separation,
    /// w = steps.
    march: [f32; 4],
    /// x = intensity, y = fade start, z = fade end, w = frame index.
    fade: [f32; 4],
}

/// What the pass binds: the tunables, plus the prepass's camera block. The
/// camera half is uploaded by the prepass rather than here, for the reason the
/// SSAO resolve shares it — the march reconstructs view positions from the depth
/// that pass wrote, so it has to use the projection that wrote it, jitter and
/// all.
pub(super) struct ContactShadowUniforms {
    frame: Subbuffer<FrameUbo>,
    params: Subbuffer<ContactShadowUbo>,
}

pub struct ContactShadowPass {
    pipeline: Arc<GraphicsPipeline>,
    uniform_allocator: Arena,
    /// Nearest, because depth and normals are fetched at exact texels: filtering
    /// either invents a surface between two that are really there, and the march
    /// would stop on it.
    nearest_clamp: Arc<Sampler>,
    /// 1x1 white (= fully lit), bound when the pass is off so the forward shader
    /// has one path rather than two. Not a graph resource: with contact shadows
    /// off the graph has no node to declare a read of.
    lit_view: Arc<ImageView>,
    /// Advanced every frame the pass runs, and what decorrelates the dither
    /// frame to frame. Without it every frame marches from the same offset and
    /// the temporal resolve averages eight copies of one march.
    frame: u64,
}

impl ContactShadowPass {
    pub fn new(ctx: &VkContext) -> Self {
        let device = &ctx.device;
        let pipeline = build_pipeline(ctx);

        let uniform_allocator = Arena::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let nearest_clamp = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap();

        Self {
            pipeline,
            uniform_allocator,
            nearest_clamp,
            lit_view: super::texture::upload_texture(
                ctx,
                &[255u8],
                [1, 1],
                vk_format(MASK_FORMAT),
                MipPolicy::None,
            ),
            frame: 0,
        }
    }

    /// A 1x1 "fully lit" view, bound when the pass is off.
    pub fn lit_view(&self) -> Arc<ImageView> {
        self.lit_view.clone()
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
        settings: &ContactShadowSettings,
        view: &FrameView,
        to_sun: Vec3,
        frame: Subbuffer<FrameUbo>,
    ) -> ContactShadowUniforms {
        // A direction, so the view matrix's translation must not apply.
        let light_direction = view.view.transform_vector3(to_sun).normalize_or_zero();

        // Mirrors `ContactShadowSettings::distance_fade`, which is the tested
        // definition of this ramp; the shader is handed the two ends rather than
        // the fraction so it evaluates one smoothstep.
        let fade_end = settings.fade_distance.max(1e-4);
        let fade_start = fade_end * 0.75;

        let params = self.uniform_allocator.allocate_sized::<ContactShadowUbo>();
        *params.write().unwrap() = ContactShadowUbo {
            light_direction: [light_direction.x, light_direction.y, light_direction.z, 0.0],
            march: [
                settings.ray_length.max(0.0),
                settings.bias.max(0.0),
                // The far end of the window, summed here rather than in the
                // shader so the two dials stay independent: a thickness below
                // the bias would otherwise close the window entirely, and the
                // pass would spend a full march per pixel to produce a mask of
                // ones.
                settings.bias.max(0.0) + settings.thickness.max(0.0),
                settings.steps.max(1) as f32,
            ],
            fade: [
                settings.intensity.clamp(0.0, 1.0),
                fade_start,
                fade_end,
                (self.frame % DITHER_PHASES) as f32,
            ],
        };
        self.frame = self.frame.wrapping_add(1);

        ContactShadowUniforms { frame, params }
    }

    /// `depth_view` and `normal_view` are the prepass's graph outputs, declared
    /// as this pass's `Sampled` inputs.
    pub(super) fn record(
        &self,
        builder: &mut Recorder,
        renderer: &PassCtx<'_>,
        extent: [u32; 2],
        uniforms: &ContactShadowUniforms,
        depth_view: Arc<ImageView>,
        normal_view: Arc<ImageView>,
    ) {
        let uniform_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, uniforms.frame.clone()),
                WriteDescriptorSet::buffer(1, uniforms.params.clone()),
            ],
            [],
        )
        .unwrap();
        let texture_set = renderer.ctx.descriptor_sets.get(
            &renderer.ctx.descriptor_set_allocator,
            &self.pipeline.layout().set_layouts()[1],
            [
                WriteDescriptorSet::image_view_sampler(0, depth_view, self.nearest_clamp.clone()),
                WriteDescriptorSet::image_view_sampler(1, normal_view, self.nearest_clamp.clone()),
            ],
        );

        builder
            .set_viewport(
                0,
                &[Viewport {
                    offset: [0.0, 0.0],
                    extent: [extent[0] as f32, extent[1] as f32],
                    depth_range: 0.0..=1.0,
                }],
            )
            .bind_pipeline_graphics(&self.pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout(),
                0,
                &[uniform_set, texture_set],
            );
        builder.draw(3, 1, 0, 0);
    }
}

fn build_pipeline(ctx: &VkContext) -> Arc<GraphicsPipeline> {
    let device = &ctx.device;
    let vs = fullscreen_vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = contact_fs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();
    GraphicsPipeline::new(
        device.clone(),
        ctx.pipeline_cache(),
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(VertexInputState::default()),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: None,
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                1,
                ColorBlendAttachmentState::default(),
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(rendering::pipeline_info(&[vk_format(MASK_FORMAT)], None).into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

mod fullscreen_vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/fullscreen.vert" }
}
mod contact_fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/contact_shadows.frag" }
}
