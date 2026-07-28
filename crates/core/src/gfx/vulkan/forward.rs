use std::sync::Arc;

use glam::{Mat3, Mat4, Vec3};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex as _, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{RenderPass, Subpass};

use crate::gfx::{Material, RenderItem, SceneLighting, Vertex, MAX_POINT_LIGHTS, MAX_TEXTURES};
use crate::scene::Camera;

use super::context::VkContext;
use super::swapchain::DEPTH_FORMAT;
use super::VulkanRenderer;

pub struct GpuMesh {
    pub vertex_buffer: Subbuffer<[Vertex]>,
    pub index_buffer: Subbuffer<[u32]>,
    pub index_count: u32,
}

/// Per-draw push constants. Only the small, per-draw-varying values live here;
/// the fat per-object matrices moved to the set-4 storage buffer so this range
/// stays under the 128-byte guaranteed `maxPushConstantsSize` (it was 196).
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
struct PushConstants {
    mvp: [[f32; 4]; 4],
    material_index: u32,
    /// Row into the set-4 object buffer holding this draw's model/normal matrix.
    object_index: u32,
}

/// Per-object transforms, indexed by [`PushConstants::object_index`] from a
/// storage buffer (set 4). std430 matches this `#[repr(C)]` layout exactly
/// because every field is a 64-byte `mat4` (a multiple of 16).
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
struct GpuObject {
    model: [[f32; 4]; 4],
    /// Inverse-transpose of `model`'s rotation/scale, for transforming normals
    /// correctly under non-uniform scaling. Stored as a mat4; only the upper-left
    /// 3x3 is used in the shader.
    normal_matrix: [[f32; 4]; 4],
}

/// Default texture indices, matching the order `VulkanRenderer::new` seeds them.
const WHITE_TEXTURE: u32 = 0;
const FLAT_NORMAL_TEXTURE: u32 = 1;

#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
pub(crate) struct GpuMaterial {
    base_color: [f32; 4],
    emissive: [f32; 4],
    params: [f32; 4], // metallic, roughness, reflectance
    /// Indices into the set-2 texture array: [albedo, normal, metal-rough, emissive].
    tex_indices: [u32; 4],
}

/// Pack the engine's [`SceneLighting`] into the std140 layout the shader expects.
fn to_gpu_lighting(lighting: &SceneLighting, camera_pos: Vec3, extent: [u32; 2]) -> GpuLighting {
    let (w, h) = (extent[0] as f32, extent[1] as f32);
    let count = lighting.point_lights.len().min(MAX_POINT_LIGHTS);
    let mut point_lights = [GpuPointLight::ZERO; MAX_POINT_LIGHTS];
    for (slot, light) in point_lights
        .iter_mut()
        .zip(lighting.point_lights.iter().take(count))
    {
        slot.position = [
            light.position.x,
            light.position.y,
            light.position.z,
            light.range.max(1e-4),
        ];
        slot.color = [light.color.x, light.color.y, light.color.z, light.intensity];
    }

    // The shader wants the direction *toward* the light, so negate.
    let to_sun = (-lighting.sun.direction).normalize_or_zero();

    GpuLighting {
        camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 0.0],
        ambient: [
            lighting.ambient_color.x,
            lighting.ambient_color.y,
            lighting.ambient_color.z,
            lighting.ambient_intensity,
        ],
        sun_direction: [to_sun.x, to_sun.y, to_sun.z, 0.0],
        sun_color: [
            lighting.sun.color.x,
            lighting.sun.color.y,
            lighting.sun.color.z,
            lighting.sun.intensity,
        ],
        params: [
            count as f32,
            lighting.shininess,
            lighting.specular_strength,
            0.0,
        ],
        viewport: [w, h, 1.0 / w, 1.0 / h],
        point_lights,
    }
}

/// GPU mirror of a [`PointLight`](crate::gfx::PointLight), padded to std140
/// (two `vec4`s).
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
struct GpuPointLight {
    /// xyz = world position, w = range.
    position: [f32; 4],
    /// rgb = color, w = intensity.
    color: [f32; 4],
}

impl GpuPointLight {
    const ZERO: Self = Self {
        position: [0.0; 4],
        color: [0.0; 4],
    };
}

/// GPU layout of the per-frame lighting uniform buffer (set 0, binding 0).
/// Every field is a `vec4` so the Rust `#[repr(C)]` layout matches std140 with
/// no hidden padding.
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
struct GpuLighting {
    /// xyz = camera world position.
    camera_pos: [f32; 4],
    /// rgb = ambient color, w = ambient intensity.
    ambient: [f32; 4],
    /// xyz = normalized direction toward the sun.
    sun_direction: [f32; 4],
    /// rgb = sun color, w = sun intensity.
    sun_color: [f32; 4],
    /// x = point light count, y = shininess, z = specular strength.
    params: [f32; 4],
    /// x=w, y=h, z=1/w, w=1/h
    viewport: [f32; 4],
    point_lights: [GpuPointLight; MAX_POINT_LIGHTS],
}

pub struct ForwardPass {
    pub render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    uniform_buffer_allocator: SubbufferAllocator,
    /// Per-frame streaming allocator for the set-4 per-object transform buffer.
    object_buffer_allocator: SubbufferAllocator,
    sampler: Arc<Sampler>,
    ao_sampler: Arc<Sampler>,
}

impl ForwardPass {
    pub fn new(
        device: &Arc<Device>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        color_format: Format,
    ) -> Self {
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                msaa_color: {
                    format: color_format,
                    samples: 4,
                    load_op: Clear,
                    store_op: DontCare,
                },
                depth: {
                    format: DEPTH_FORMAT,
                    samples: 4,
                    load_op: Clear,
                    store_op: DontCare,
                },

                color: {
                    format: color_format,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
            },
            pass: {
                color: [msaa_color],
                color_resolve: [color],
                depth_stencil: {depth},
            },
        )
        .unwrap();

        let pipeline = build_pipeline(device, &render_pass);

        let uniform_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let object_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo::simple_repeat_linear_no_mipmap(),
        )
        .unwrap();

        let ao_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        ).unwrap();

        Self {
            render_pass,
            pipeline,
            uniform_buffer_allocator,
            object_buffer_allocator,
            sampler,
            ao_sampler
        }
    }

    /// Build the set-1 material storage buffer + descriptor set. Cached by the
    /// renderer and only rebuilt when the material table changes.
    pub fn build_material_set(
        &self,
        ctx: &VkContext,
        materials: &[GpuMaterial],
    ) -> Arc<DescriptorSet> {
        let buffer = Buffer::from_iter(
            ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            materials.iter().copied(),
        )
        .expect("failed to allocate material buffer");

        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[1].clone(),
            [WriteDescriptorSet::buffer(0, buffer)],
            [],
        )
        .unwrap()
    }

    /// Build the set-2 texture array + sampler descriptor set. Cached by the
    /// renderer and only rebuilt when a texture is added.
    pub fn build_texture_set(
        &self,
        ctx: &VkContext,
        textures: &[Arc<ImageView>],
    ) -> Arc<DescriptorSet> {
        let default_view = textures[0].clone();
        let texture_array =
            (0..MAX_TEXTURES).map(|i| textures.get(i).cloned().unwrap_or_else(|| default_view.clone()));
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[2].clone(),
            [
                WriteDescriptorSet::image_view_array(0, 0, texture_array),
                WriteDescriptorSet::sampler(1, self.sampler.clone()),
            ],
            [],
        )
        .unwrap()
    }

    pub fn draw(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            vulkano::command_buffer::PrimaryAutoCommandBuffer,
        >,
        renderer: &VulkanRenderer,
        items: &[RenderItem],
        lighting: &SceneLighting,
        camera: &Camera,
        extent: [u32; 2],
        ao_view: Arc<ImageView>,
        material_set: Arc<DescriptorSet>,
        texture_set: Arc<DescriptorSet>,
    ) {
        let aspect = extent[0] as f32 / extent[1] as f32;
        let view_proj = camera.view_projection(aspect);

        let lighting_buffer = self
            .uniform_buffer_allocator
            .allocate_sized::<GpuLighting>()
            .unwrap();
        *lighting_buffer.write().unwrap() = to_gpu_lighting(lighting, camera.position, extent);

        let lighting_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer(0, lighting_buffer)],
            [],
        )
        .unwrap();

        let ao_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[3].clone(),
            [WriteDescriptorSet::image_view_sampler(0, ao_view, self.ao_sampler.clone())],
            [],
        ).unwrap();

        // Per-object transforms for this frame, one entry per item so the loop's
        // `object_index` is just the `items` index. Built for every item, even
        // ones whose mesh is missing and get skipped below, to keep that mapping
        // trivial.
        let objects: Vec<GpuObject> = items
            .iter()
            .map(|item| {
                let model = item.model;
                // Inverse-transpose so normals stay perpendicular under non-uniform scale.
                let normal_matrix =
                    Mat4::from_mat3(Mat3::from_mat4(model).inverse().transpose());
                GpuObject {
                    model: model.to_cols_array_2d(),
                    normal_matrix: normal_matrix.to_cols_array_2d(),
                }
            })
            .collect();
        // allocate_slice rejects length 0; an empty scene still needs a bindable
        // buffer, so round up to one (unwritten, unread) slot.
        let object_buffer = self
            .object_buffer_allocator
            .allocate_slice::<GpuObject>(objects.len().max(1) as u64)
            .unwrap();
        object_buffer.write().unwrap()[..objects.len()].copy_from_slice(&objects);

        let object_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[4].clone(),
            [WriteDescriptorSet::buffer(0, object_buffer)],
            [],
        )
        .unwrap();

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
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                vec![lighting_set, material_set, texture_set, ao_set, object_set],
            )
            .unwrap();

        for (object_index, item) in items.iter().enumerate() {
            let Some(mesh) = renderer.meshes.get(item.mesh.0 as usize) else {
                continue;
            };
            let push = PushConstants {
                mvp: (view_proj * item.model).to_cols_array_2d(),
                material_index: item.material.0,
                object_index: object_index as u32,
            };

            builder
                .push_constants(self.pipeline.layout().clone(), 0, push)
                .unwrap()
                .bind_vertex_buffers(0, mesh.vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(mesh.index_buffer.clone())
                .unwrap();
            unsafe {
                builder.draw_indexed(mesh.index_count, 1, 0, 0, 0).unwrap();
            }
        }
    }
}

pub fn upload_mesh(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    vertices: &[Vertex],
    indices: &[u32],
) -> GpuMesh {
    let vertex_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertices.iter().copied(),
    )
    .expect("failed to allocate vertex buffer");

    let index_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        indices.iter().copied(),
    )
    .expect("failed to allocate index buffer");

    GpuMesh {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
    }
}

pub(super) fn to_gpu_material(m: &Material) -> GpuMaterial {
    // Missing maps fall back to the default textures, which make the sample a
    // no-op (white = ×1, flat normal = unchanged geometric normal).
    GpuMaterial {
        base_color: [m.base_color.x, m.base_color.y, m.base_color.z, 1.0],
        emissive: [m.emissive.x, m.emissive.y, m.emissive.z, 0.0],
        params: [m.metallic, m.roughness, m.reflectance, 0.0],
        tex_indices: [
            m.albedo_texture.map_or(WHITE_TEXTURE, |h| h.0),
            m.normal_texture.map_or(FLAT_NORMAL_TEXTURE, |h| h.0),
            m.metallic_roughness_texture.map_or(WHITE_TEXTURE, |h| h.0),
            m.emissive_texture.map_or(WHITE_TEXTURE, |h| h.0),
        ],
    }
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
                cull_mode: CullMode::Back,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState {
                rasterization_samples: vulkano::image::SampleCount::Sample4,
                ..Default::default()
            }),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
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

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/forward.vert",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/forward.frag",
    }
}
