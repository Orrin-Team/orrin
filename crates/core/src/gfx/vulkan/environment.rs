//! The environment cubemap and the skybox that draws it.
//!
//! The bake deliberately does not go through the render graph. It runs once at
//! load time, its barrier chain is a straight line, and its output outlives
//! every frame — none of which the graph derives anything useful for. What the
//! frame sees is only the finished cube, which after the bake sits in
//! `ShaderReadOnlyOptimal` and never changes layout again.
//!
//! The skybox is a draw inside the forward render pass rather than a graph node
//! of its own, for the same reason the debug lines are: `msaa_hdr` is declared
//! `store_op: DontCare` and allocated lazily, so on a tile GPU it never reaches
//! DRAM. A separate pass would have to `Store` it and `Load` it back, which
//! trades the whole point of that allocation for a node in the schedule.

use std::sync::Arc;

use glam::{Mat3, Mat4};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo, PrimaryAutoCommandBuffer,
    RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{
    Image, ImageCreateFlags, ImageCreateInfo, ImageSubresourceRange, ImageType, ImageUsage,
    max_mip_levels,
};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
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
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::sync::{self, GpuFuture};

use crate::scene::{Camera, EnvironmentSettings};

use super::MSAA_SAMPLES;
use super::context::VkContext;
use super::hdr::HDR_FORMAT;

/// The environment carries the same radiance the scene does, so it uses the
/// same format the forward target does.
pub const CUBE_FORMAT: Format = HDR_FORMAT;

/// Edge length of one cube face. Enough for a background at ordinary fields of
/// view; not enough for sharp mirror reflections, which is what would drive
/// raising it.
pub const FACE_SIZE: u32 = 512;

/// Per-face `(forward, right, up)`, in cube layer order: +X, -X, +Y, -Y, +Z, -Z.
///
/// Derived from the cube face selection table in the Vulkan spec, inverted: for
/// face coordinates `(u, v)` in `[-1,1]` with `v` increasing *down* the face
/// image, the direction is `forward + u * right + v * up`. Two consequences
/// that look like mistakes and are not — the ±Y faces do not share the other
/// four's handedness, and `up` points along -Y for four of the six, because the
/// convention is left-handed and its `t` axis runs down the image.
///
/// A single transposed or mirrored face shows up only as a discontinuity at a
/// face edge, and would silently corrupt every irradiance and prefilter result
/// derived from the cube later. Changing anything here needs the round trip
/// checked: `direction_for(face, uv)` composed with a hardware cube sample must
/// be the identity.
const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
    ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
    ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]),
    ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
    ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
    ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
];

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct FacePush {
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct SkyboxPush {
    inv_view_rot_proj: [[f32; 4]; 4],
    params: [f32; 4],
}

pub struct EnvironmentPass {
    bake_rp: Arc<RenderPass>,
    bake_pipeline: Arc<GraphicsPipeline>,
    equirect_sampler: Arc<Sampler>,
    skybox_pipeline: Arc<GraphicsPipeline>,
    cube_sampler: Arc<Sampler>,
    /// `None` until an environment is loaded. The skybox is simply not recorded
    /// in that case — there is no fallback cube, because nothing samples the
    /// environment except the skybox until the irradiance and prefilter steps
    /// land and give the forward shader a reason to.
    cube: Option<Arc<ImageView>>,
}

impl EnvironmentPass {
    /// `forward_rp` is the forward pass's render pass: the skybox draws inside
    /// it, after the geometry.
    pub fn new(ctx: &VkContext, forward_rp: &Arc<RenderPass>) -> Self {
        let device = &ctx.device;
        let bake_rp = bake_render_pass(device);
        let bake_pipeline = build_bake_pipeline(device, &bake_rp);
        let skybox_pipeline = build_skybox_pipeline(device, forward_rp);

        // Repeat in u so the seam wraps, clamp in v so the poles do not fold
        // across to the opposite hemisphere.
        let equirect_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [
                    SamplerAddressMode::Repeat,
                    SamplerAddressMode::ClampToEdge,
                    SamplerAddressMode::ClampToEdge,
                ],
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        )
        .unwrap();

        // Mipmapped: the prefiltered roughness chain lands in these levels, and
        // a sampler built without them would have to be replaced then.
        let cube_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::simple_repeat_linear()
            },
        )
        .unwrap();

        Self {
            bake_rp,
            bake_pipeline,
            equirect_sampler,
            skybox_pipeline,
            cube_sampler,
            cube: None,
        }
    }

    /// Project an equirectangular source into a cubemap, replacing whatever was
    /// loaded before. `pixels` is tightly packed RGBA f32, row-major from the
    /// top-left, `width` by `height`.
    ///
    /// Blocks until the bake completes: it is a load-time operation, and the
    /// alternative is a half-written cube visible to the first frame.
    pub fn set_source(&mut self, ctx: &VkContext, pixels: &[f32], extent: [u32; 2]) {
        let expected = extent[0] as usize * extent[1] as usize * 4;
        assert_eq!(
            pixels.len(),
            expected,
            "equirect source is {} floats, expected {expected} for {}x{} RGBA",
            pixels.len(),
            extent[0],
            extent[1],
        );

        let equirect = upload_equirect(ctx, pixels, extent);
        self.cube = Some(bake_cube(
            ctx,
            &self.bake_rp,
            &self.bake_pipeline,
            &self.equirect_sampler,
            equirect,
        ));
    }

    /// Record the skybox. Called at the end of the forward pass body, after the
    /// geometry, so the depth test rejects it everywhere something was drawn.
    pub fn record_skybox(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        ctx: &VkContext,
        camera: &Camera,
        extent: [u32; 2],
        settings: &EnvironmentSettings,
    ) {
        let Some(cube) = self.cube.clone() else {
            return;
        };
        if !settings.show_skybox {
            return;
        }

        let set = DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.skybox_pipeline.layout().set_layouts()[0].clone(),
            [
                WriteDescriptorSet::image_view(0, cube),
                WriteDescriptorSet::sampler(1, self.cube_sampler.clone()),
            ],
            [],
        )
        .unwrap();

        let aspect = extent[0] as f32 / extent[1] as f32;
        // Translation stripped: what the vertex shader unprojects is then a
        // direction, so the ray needs no camera-relative correction and holds
        // its precision arbitrarily far from the origin.
        let view_rotation = Mat4::from_mat3(Mat3::from_mat4(camera.view()));
        let inverse = (camera.projection(aspect) * view_rotation).inverse();
        // Rotating the ray rotates the environment, so the yaw costs nothing in
        // the shader.
        let matrix = Mat4::from_rotation_y(settings.yaw.to_radians()) * inverse;

        builder
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [extent[0] as f32, extent[1] as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_pipeline_graphics(self.skybox_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.skybox_pipeline.layout().clone(),
                0,
                vec![set],
            )
            .unwrap()
            .push_constants(
                self.skybox_pipeline.layout().clone(),
                0,
                SkyboxPush {
                    inv_view_rot_proj: matrix.to_cols_array_2d(),
                    params: [settings.intensity, 0.0, 0.0, 0.0],
                },
            )
            .unwrap();
        unsafe { builder.draw(3, 1, 0, 0).unwrap() };
    }
}

fn upload_equirect(ctx: &VkContext, pixels: &[f32], extent: [u32; 2]) -> Arc<ImageView> {
    let staging = Buffer::from_iter(
        ctx.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        pixels.iter().copied(),
    )
    .expect("failed to allocate equirect staging buffer");

    let image = Image::new(
        ctx.memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R32G32B32A32_SFLOAT,
            extent: [extent[0], extent[1], 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .expect("failed to create equirect image");

    let mut builder = AutoCommandBufferBuilder::primary(
        ctx.command_buffer_allocator.clone(),
        ctx.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, image.clone()))
        .unwrap();
    submit_and_wait(ctx, builder);

    ImageView::new_default(image).expect("failed to create equirect view")
}

fn bake_cube(
    ctx: &VkContext,
    render_pass: &Arc<RenderPass>,
    pipeline: &Arc<GraphicsPipeline>,
    sampler: &Arc<Sampler>,
    equirect: Arc<ImageView>,
) -> Arc<ImageView> {
    // The full chain is allocated now though only level 0 is written: the
    // prefiltered roughness levels go in the rest, and allocating them here
    // means that step changes what fills the image rather than what the image
    // is. Levels nothing has written are never sampled — the skybox asks for
    // level 0 explicitly.
    let mip_levels = max_mip_levels([FACE_SIZE, FACE_SIZE, 1]);

    let image = Image::new(
        ctx.memory_allocator.clone(),
        ImageCreateInfo {
            // Without this the *image* creation succeeds and the cube view
            // fails, which points the error at the wrong line.
            flags: ImageCreateFlags::CUBE_COMPATIBLE,
            image_type: ImageType::Dim2d,
            format: CUBE_FORMAT,
            extent: [FACE_SIZE, FACE_SIZE, 1],
            array_layers: 6,
            mip_levels,
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .expect("failed to create environment cubemap");

    let set = DescriptorSet::new(
        ctx.descriptor_set_allocator.clone(),
        pipeline.layout().set_layouts()[0].clone(),
        [WriteDescriptorSet::image_view_sampler(
            0,
            equirect,
            sampler.clone(),
        )],
        [],
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        ctx.command_buffer_allocator.clone(),
        ctx.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    for (face, (forward, right, up)) in FACES.iter().enumerate() {
        // A framebuffer attachment is one layer of one level, while the same
        // image is sampled as a whole cube — so the bake and the shader need
        // different views over the same allocation.
        let attachment = ImageView::new(
            image.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Dim2d,
                subresource_range: ImageSubresourceRange {
                    array_layers: face as u32..face as u32 + 1,
                    mip_levels: 0..1,
                    ..image.subresource_range()
                },
                ..ImageViewCreateInfo::from_image(&image)
            },
        )
        .unwrap();

        let framebuffer = Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![attachment],
                ..Default::default()
            },
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![None],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [FACE_SIZE as f32, FACE_SIZE as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_pipeline_graphics(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                vec![set.clone()],
            )
            .unwrap()
            .push_constants(
                pipeline.layout().clone(),
                0,
                FacePush {
                    forward: [forward[0], forward[1], forward[2], 0.0],
                    right: [right[0], right[1], right[2], 0.0],
                    up: [up[0], up[1], up[2], 0.0],
                },
            )
            .unwrap();
        unsafe { builder.draw(3, 1, 0, 0).unwrap() };
        builder.end_render_pass(Default::default()).unwrap();
    }

    submit_and_wait(ctx, builder);

    ImageView::new(
        image.clone(),
        ImageViewCreateInfo {
            view_type: ImageViewType::Cube,
            ..ImageViewCreateInfo::from_image(&image)
        },
    )
    .expect("failed to create environment cube view")
}

fn submit_and_wait(ctx: &VkContext, builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
    sync::now(ctx.device.clone())
        .then_execute(ctx.queue.clone(), builder.build().unwrap())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
}

fn bake_render_pass(device: &Arc<Device>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: { format: CUBE_FORMAT, samples: 1, load_op: DontCare, store_op: Store },
        },
        pass: { color: [color], depth_stencil: {} },
    )
    .unwrap()
}

fn build_bake_pipeline(
    device: &Arc<Device>,
    render_pass: &Arc<RenderPass>,
) -> Arc<GraphicsPipeline> {
    let vs = fullscreen_vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = equirect_fs::load(device.clone())
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
    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(VertexInputState::default()),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: None,
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

fn build_skybox_pipeline(
    device: &Arc<Device>,
    render_pass: &Arc<RenderPass>,
) -> Arc<GraphicsPipeline> {
    let vs = skybox_vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = skybox_fs::load(device.clone())
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
    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(VertexInputState::default()),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState {
                rasterization_samples: MSAA_SAMPLES,
                ..Default::default()
            }),
            // The triangle sits exactly on the far plane, so the test has to
            // accept equality or the sky is rejected by the depth clear it is
            // supposed to fill. Writing depth would make it occlude the debug
            // lines drawn after it.
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState {
                    write_enable: false,
                    compare_op: CompareOp::LessOrEqual,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            dynamic_state: [DynamicState::Viewport].into_iter().collect(),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

mod fullscreen_vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/fullscreen.vert" }
}

mod equirect_fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/equirect_to_cube.frag" }
}

mod skybox_vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/skybox.vert" }
}

mod skybox_fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/skybox.frag" }
}

#[cfg(test)]
mod tests {
    use super::FACES;

    /// The cube face selection rules from the Vulkan spec: given a direction,
    /// which layer samples it and at what face coordinates. This is what the
    /// hardware does on a `textureCube` fetch, written out so the bake's
    /// direction table can be checked against it without a GPU.
    fn select_face(dir: [f32; 3]) -> (usize, f32, f32) {
        let [x, y, z] = dir;
        let (face, ma, sc, tc) = if x.abs() >= y.abs() && x.abs() >= z.abs() {
            if x > 0.0 {
                (0, x, -z, -y)
            } else {
                (1, -x, z, -y)
            }
        } else if y.abs() >= z.abs() {
            if y > 0.0 {
                (2, y, x, z)
            } else {
                (3, -y, x, -z)
            }
        } else if z > 0.0 {
            (4, z, x, -y)
        } else {
            (5, -z, -x, -y)
        };
        (face, sc / ma, tc / ma)
    }

    /// The bake writes face `f` by projecting `forward + u * right + v * up`.
    /// If the hardware would not fetch that direction from face `f` at exactly
    /// `(u, v)`, the cube is transposed, mirrored, or on the wrong layer — a
    /// defect that shows up only as a discontinuity at a face edge, and that
    /// silently corrupts every irradiance and prefilter result derived from the
    /// cube afterwards.
    #[test]
    fn every_face_basis_round_trips_through_hardware_cube_selection() {
        // Off-centre and asymmetric on purpose: (0,0) round-trips even for a
        // transposed basis, and a symmetric pair hides a mirrored one.
        let samples = [
            (0.0, 0.0),
            (0.5, 0.25),
            (-0.75, 0.5),
            (0.9, -0.6),
            (-0.3, -0.95),
        ];

        for (face, (forward, right, up)) in FACES.iter().enumerate() {
            for (u, v) in samples {
                let dir = [
                    forward[0] + u * right[0] + v * up[0],
                    forward[1] + u * right[1] + v * up[1],
                    forward[2] + u * right[2] + v * up[2],
                ];
                let (got_face, got_u, got_v) = select_face(dir);

                assert_eq!(
                    got_face, face,
                    "face {face} at ({u}, {v}) is sampled from layer {got_face}",
                );
                assert!(
                    (got_u - u).abs() < 1e-5 && (got_v - v).abs() < 1e-5,
                    "face {face}: wrote ({u}, {v}), hardware samples ({got_u}, {got_v})",
                );
            }
        }
    }
}
