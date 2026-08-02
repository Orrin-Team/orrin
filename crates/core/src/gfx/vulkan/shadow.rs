use std::sync::Arc;

use glam::Mat4;
use vulkano::buffer::{BufferContents, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, ClearDepthStencilImageInfo, CommandBufferUsage,
    PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::ClearDepthStencilValue;
use vulkano::image::sampler::{
    BorderColor, Filter, Sampler, SamplerAddressMode, SamplerCreateInfo,
};
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, DepthBiasState, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex as _, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{RenderPass, Subpass};
use vulkano::sync::GpuFuture;

use crate::gfx::{RenderItem, Vertex};

use super::VulkanRenderer;
use super::context::VkContext;
use super::forward::GpuObject;
use super::swapchain::DEPTH_FORMAT;

/// Per-run push constants: 68 bytes, comfortably inside the 128-byte guaranteed
/// `maxPushConstantsSize`. The cascade's matrix rides here rather than in a
/// uniform buffer because it changes once per pass, not once per draw.
#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct PushConstants {
    light_view_proj: [[f32; 4]; 4],
    object_base: u32,
}

pub struct ShadowPass {
    pub(super) render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    /// 1x1 array depth image cleared to 1.0, bound when shadows are off so the
    /// forward shader samples "fully lit" with no second code path. Not a graph
    /// resource: with shadows off the graph has no cascade image at all, so
    /// there is nothing for the forward pass to declare a read of.
    lit_view: Arc<ImageView>,
    pub constant_bias: f32,
    pub slope_bias: f32,
}

impl ShadowPass {
    pub fn new(ctx: &VkContext) -> Self {
        let device = &ctx.device;
        let render_pass = depth_only_render_pass(device);
        let pipeline = build_pipeline(device, &render_pass);

        Self {
            render_pass,
            pipeline,
            lit_view: build_lit_view(ctx),
            constant_bias: 1.25,
            slope_bias: 2.5,
        }
    }

    /// The "everything is lit" view, bound when shadows are disabled.
    #[allow(dead_code)]
    pub(super) fn lit_view(&self) -> Arc<ImageView> {
        self.lit_view.clone()
    }

    /// Record one cascade's depth pass.
    ///
    /// `resolution` is the shadow map's, not the frame's — every other pass in
    /// the executor takes the swapchain extent, and using it here would render
    /// the cascade into a corner of its own map with nothing reporting an error.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn record(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        renderer: &VulkanRenderer,
        casters: &[RenderItem],
        view_proj: Mat4,
        object_base: u32,
        resolution: u32,
        objects: Subbuffer<[GpuObject]>,
    ) {
        let object_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer(0, objects)],
            [],
        )
        .unwrap();

        builder
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [resolution as f32, resolution as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap();
        // Dynamic so the editor's bias sliders tune acne live instead of
        // rebuilding the pipeline on every drag. `clamp` stays 0.0: a nonzero
        // one needs the `depth_bias_clamp` device feature.
        builder
            .set_depth_bias(self.constant_bias, 0.0, self.slope_bias)
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                vec![object_set],
            )
            .unwrap();

        for run in super::forward::runs(casters) {
            let item = &casters[run.start];
            let Some(mesh) = renderer.meshes.get(item.mesh.0 as usize) else {
                continue;
            };
            let push = PushConstants {
                light_view_proj: view_proj.to_cols_array_2d(),
                object_base: object_base + run.start as u32,
            };
            builder
                .push_constants(self.pipeline.layout().clone(), 0, push)
                .unwrap()
                .bind_vertex_buffers(0, mesh.vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(mesh.index_buffer.clone())
                .unwrap();
            unsafe {
                builder
                    .draw_indexed(mesh.index_count, run.len() as u32, 0, 0, 0)
                    .unwrap();
            }
        }
    }
}

/// The sampler the forward pass reads the cascades through.
///
/// It must be an *immutable* sampler baked into the descriptor set layout, not
/// one written into a set: Metal makes a sampler's compare function a
/// compile-time property, so MoltenVK rejects a written comparison sampler
/// unless `mutable_comparison_samplers` is available, which on a portability
/// subset device it is not.
///
/// The conventions live here because this module owns the map: `Less` against a
/// depth map cleared to 1.0, and a white border so a fragment sampling outside a
/// cascade reads as lit rather than as shadowed.
pub(super) fn comparison_sampler(device: &Arc<Device>) -> Arc<Sampler> {
    Sampler::new(
        device.clone(),
        SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            address_mode: [SamplerAddressMode::ClampToBorder; 3],
            border_color: BorderColor::FloatOpaqueWhite,
            // The comparison happens before the bilinear filter, so a single tap
            // is already a 2x2 percentage-closer result.
            compare: Some(CompareOp::Less),
            ..Default::default()
        },
    )
    .unwrap()
}

fn depth_only_render_pass(device: &Arc<Device>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            depth: {
                format: DEPTH_FORMAT,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: { color: [], depth_stencil: {depth} },
    )
    .unwrap()
}

fn build_pipeline(device: &Arc<Device>, render_pass: &Arc<RenderPass>) -> Arc<GraphicsPipeline> {
    let vs = vs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = fs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();

    let vertex_input_state = Vertex::per_vertex().definition(&vs).unwrap();
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
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(RasterizationState {
                // Back-face culling matches the forward pass, which is only
                // correct because `fit_cascade` applies the same Y flip the
                // camera projection does — otherwise the winding is mirrored
                // here and this would cull front faces instead.
                cull_mode: CullMode::Back,
                // Placeholders; the real values are set dynamically per frame.
                depth_bias: Some(DepthBiasState::default()),
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            // No color attachments to blend into.
            color_blend_state: None,
            dynamic_state: [DynamicState::Viewport, DynamicState::DepthBias]
                .into_iter()
                .collect(),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

/// A 1x1 single-layer depth *array* image cleared to 1.0.
///
/// The array view type is load-bearing: the forward shader declares the
/// cascades as `texture2DArray`, and a plain 2D view would not satisfy that
/// descriptor when shadows are off.
fn build_lit_view(ctx: &VkContext) -> Arc<ImageView> {
    let image = Image::new(
        ctx.memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: DEPTH_FORMAT,
            extent: [1, 1, 1],
            array_layers: 1,
            usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .expect("failed to allocate the shadows-off depth texture");

    let mut builder = AutoCommandBufferBuilder::primary(
        ctx.command_buffer_allocator.clone(),
        ctx.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .clear_depth_stencil_image(ClearDepthStencilImageInfo {
            clear_value: ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
            ..ClearDepthStencilImageInfo::image(image.clone())
        })
        .unwrap();
    vulkano::sync::now(ctx.device.clone())
        .then_execute(ctx.queue.clone(), builder.build().unwrap())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    ImageView::new(
        image.clone(),
        ImageViewCreateInfo {
            view_type: ImageViewType::Dim2dArray,
            ..ImageViewCreateInfo::from_image(&image)
        },
    )
    .unwrap()
}

mod vs {
    vulkano_shaders::shader! { ty: "vertex", path: "shaders/shadow.vert" }
}
mod fs {
    vulkano_shaders::shader! { ty: "fragment", path: "shaders/shadow.frag" }
}
