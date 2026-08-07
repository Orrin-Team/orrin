use std::sync::Arc;

use glam::{Mat4, Vec3};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex as _, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{RenderPass, Subpass};

use crate::geom::Aabb;
use crate::gfx::sh::SH9;
use crate::gfx::shadows::MAX_CASCADES;
use crate::gfx::{DrawList, MAX_POINT_LIGHTS, MAX_TEXTURES, Material, SceneLighting, Vertex};
use crate::scene::{Camera, EnvironmentSettings};

use super::context::VkContext;
use super::swapchain::DEPTH_FORMAT;
use super::taa::FrameView;
use super::{ShadowFrame, VulkanRenderer};

pub struct GpuMesh {
    pub vertex_buffer: Subbuffer<[Vertex]>,
    pub index_buffer: Subbuffer<[u32]>,
    pub index_count: u32,
    /// Object-space bounds, derived here because upload is the last place the
    /// vertex data exists on the CPU. Culling reads them through
    /// [`RenderBackend::mesh_bounds`](crate::gfx::RenderBackend::mesh_bounds).
    pub bounds: Aabb,
}

/// Per-run push constants. Only the small, per-run-varying values live here; the
/// fat per-object matrices are in the set-4 storage buffer so this range stays
/// under the 128-byte guaranteed `maxPushConstantsSize` (it was 196).
///
/// `view_proj` replaced a pre-multiplied `mvp` when draws became instanced: a
/// run covers many models, so the model half has to be applied in the shader.
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
struct PushConstants {
    view_proj: [[f32; 4]; 4],
    material_index: u32,
    /// First set-4 object row of this run; the shader adds `gl_InstanceIndex`.
    object_base: u32,
}

/// Per-object transforms, indexed by [`PushConstants::object_index`] from a
/// storage buffer (set 4). std430 matches this `#[repr(C)]` layout exactly
/// because every field is a 64-byte `mat4` (a multiple of 16).
#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
pub(super) struct GpuObject {
    model: [[f32; 4]; 4],
    /// Inverse-transpose of `model`'s rotation/scale, for transforming normals
    /// correctly under non-uniform scaling. Stored as a mat4; only the upper-left
    /// 3x3 is used in the shader.
    normal_matrix: [[f32; 4]; 4],
    /// Last frame's `model`, for the motion vector the prepass writes. Uploaded
    /// for every pass rather than only the one that reads it: the buffer is
    /// shared, so the row's stride is shared too, and a second layout for the
    /// passes that ignore this field would be two ways for one object row to be
    /// wrong.
    prev_model: [[f32; 4]; 4],
}

/// Where `forward.frag` declares the cascade comparison sampler. It is bound
/// immutably at pipeline-layout construction, so these have to match the shader
/// by hand rather than being derived from it.
const SHADOW_SET: usize = 3;
const SHADOW_SAMPLER_BINDING: u32 = 2;

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
fn to_gpu_lighting(
    lighting: &SceneLighting,
    camera_pos: Vec3,
    extent: [u32; 2],
    shadows: Option<ShadowFrame<'_>>,
    irradiance: [Vec3; SH9],
    environment_yaw: f32,
    env_specular: Vec3,
) -> GpuLighting {
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

    let mut cascade_view_proj = [[[0.0f32; 4]; 4]; MAX_CASCADES];
    let mut cascade_splits = [0.0f32; MAX_CASCADES];
    let mut cascade_texel_sizes = [0.0f32; MAX_CASCADES];
    // A zero count is what makes every shadow lookup return "lit"; the arrays
    // above are then never indexed.
    let shadow_params = match shadows {
        Some(shadows) => {
            for (slot, cascade) in cascade_view_proj
                .iter_mut()
                .zip(&shadows.cascades.cascades[..shadows.cascades.count])
            {
                *slot = cascade.view_proj.to_cols_array_2d();
            }
            for (index, cascade) in shadows.cascades.cascades[..shadows.cascades.count]
                .iter()
                .enumerate()
            {
                cascade_splits[index] = cascade.split_distance;
                cascade_texel_sizes[index] = cascade.texel_world_size;
            }
            [
                shadows.cascades.count as f32,
                crate::gfx::shadows::OVERLAP,
                shadows.settings.strength,
                if shadows.settings.debug_cascades {
                    1.0
                } else {
                    0.0
                },
            ]
        }
        None => [0.0; 4],
    };

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
        fog_color: [
            lighting.fog_color.x,
            lighting.fog_color.y,
            lighting.fog_color.z,
            lighting.fog_density.max(0.0),
        ],
        fog_params: [lighting.fog_height_falloff, lighting.fog_height, 0.0, 0.0],
        cascade_view_proj,
        cascade_splits,
        cascade_texel_sizes,
        shadow_params,
        point_lights,
        environment: {
            let (sin, cos) = environment_yaw.to_radians().sin_cos();
            [sin, cos, 0.0, 0.0]
        },
        env_specular: [env_specular.x, env_specular.y, env_specular.z, 0.0],
        irradiance: irradiance.map(|c| [c.x, c.y, c.z, 0.0]),
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
    /// rgb = fog color, w = density at the reference height.
    fog_color: [f32; 4],
    /// x = height falloff, y = reference height.
    fog_params: [f32; 4],
    /// Per-cascade light view-projection. std140 lays a `mat4` out as four
    /// `vec4`s with no padding between them, which is exactly what this is.
    cascade_view_proj: [[[f32; 4]; 4]; MAX_CASCADES],
    /// Split distances, as radial distance from the camera.
    cascade_splits: [f32; MAX_CASCADES],
    /// World size of one shadow texel in each cascade, for the normal-offset
    /// bias. It differs per cascade because each fits a different-sized box to
    /// the same number of texels.
    cascade_texel_sizes: [f32; MAX_CASCADES],
    /// x = cascade count, y = blend overlap fraction, z = strength,
    /// w = 1.0 to tint by cascade index.
    shadow_params: [f32; 4],
    point_lights: [GpuPointLight; MAX_POINT_LIGHTS],
    /// x = sin(environment yaw), y = cos(environment yaw). The same rotation
    /// the skybox samples through, so the sky and what it lights agree.
    environment: [f32; 4],
    /// rgb = what sampled environment radiance is multiplied by. Carries the
    /// scene's flat ambient when no environment is loaded, which is what makes
    /// the 1x1 white fallback cube behave as a uniform environment.
    env_specular: [f32; 4],
    /// Diffuse irradiance as nine spherical-harmonic coefficients, already
    /// convolved with the cosine lobe and divided by pi — see `gfx::sh`. `vec4`
    /// rather than `vec3` because std140 pads an array element to 16 bytes
    /// either way, so the padding may as well be visible on both sides.
    irradiance: [[f32; 4]; SH9],
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

        // Material textures are the only ones with a mip chain to sample; the
        // AO and tonemap inputs are screen-space targets read at 1:1.
        let anisotropy = device.enabled_features().sampler_anisotropy.then(|| {
            device
                .physical_device()
                .properties()
                .max_sampler_anisotropy
                .min(16.0)
        });

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                anisotropy,
                ..SamplerCreateInfo::simple_repeat_linear()
            },
        )
        .unwrap();

        let ao_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::simple_repeat_linear_no_mipmap()
            },
        )
        .unwrap();

        Self {
            render_pass,
            pipeline,
            uniform_buffer_allocator,
            object_buffer_allocator,
            sampler,
            ao_sampler,
        }
    }

    /// The set-4 per-object descriptor set for this frame's object buffer.
    ///
    /// Built once per frame by the executor rather than once per pass: the
    /// buffer changes every frame, so nothing here can be cached across frames,
    /// but nothing needs to be rebuilt within one either.
    pub(super) fn build_object_set(
        &self,
        ctx: &VkContext,
        objects: &Subbuffer<[GpuObject]>,
    ) -> Arc<DescriptorSet> {
        DescriptorSet::new(
            ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[4].clone(),
            [WriteDescriptorSet::buffer(0, objects.clone())],
            [],
        )
        .unwrap()
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
        let texture_array = (0..MAX_TEXTURES).map(|i| {
            textures
                .get(i)
                .cloned()
                .unwrap_or_else(|| default_view.clone())
        });
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

    /// Build this frame's per-object rows, written straight into the mapped
    /// subbuffer. Shared by every geometry pass in the frame: they need the same
    /// rows, and the allocator recycles the storage frame to frame.
    ///
    /// One row per item, including items whose mesh is missing, so a run's
    /// object rows stay contiguous and a run's base is just its start.
    ///
    /// `items` goes first so the forward and SSAO passes keep indexing from
    /// zero; each cascade's casters follow, and the returned bases say where.
    /// One buffer rather than one per list is what keeps `object_transforms` a
    /// single resource in the graph rather than a convenient fiction.
    pub(super) fn upload_objects(
        &self,
        visible: DrawList<'_>,
        casters: &[DrawList<'_>],
    ) -> (Subbuffer<[GpuObject]>, [u32; MAX_CASCADES]) {
        let total: usize = visible.len() + casters.iter().map(DrawList::len).sum::<usize>();
        // allocate_slice rejects length 0; an empty scene still needs a bindable
        // buffer, so round up to one (unwritten, unread) slot.
        let buffer = self
            .object_buffer_allocator
            .allocate_slice::<GpuObject>(total.max(1) as u64)
            .unwrap();

        let mut bases = [0u32; MAX_CASCADES];
        {
            let mut rows = buffer.write().unwrap();
            let mut next = 0usize;
            let mut write = |list: &DrawList<'_>, next: &mut usize| {
                for i in 0..list.len() {
                    let item = list.item(i);
                    rows[*next] = GpuObject {
                        model: item.model.to_cols_array_2d(),
                        normal_matrix: Mat4::from_mat3(item.normal_matrix).to_cols_array_2d(),
                        prev_model: item.prev_model.to_cols_array_2d(),
                    };
                    *next += 1;
                }
            };
            write(&visible, &mut next);
            for (base, list) in bases.iter_mut().zip(casters) {
                *base = next as u32;
                write(list, &mut next);
            }
        }
        (buffer, bases)
    }

    pub fn draw(
        &self,
        builder: &mut AutoCommandBufferBuilder<vulkano::command_buffer::PrimaryAutoCommandBuffer>,
        renderer: &VulkanRenderer,
        draws: DrawList<'_>,
        lighting: &SceneLighting,
        camera: &Camera,
        view: &FrameView,
        extent: [u32; 2],
        ao_view: Arc<ImageView>,
        shadow_view: Arc<ImageView>,
        shadows: Option<ShadowFrame<'_>>,
        material_set: Arc<DescriptorSet>,
        texture_set: Arc<DescriptorSet>,
        object_set: Arc<DescriptorSet>,
        environment: &EnvironmentSettings,
    ) {
        // The jittered one, from the frame's shared view: every pass that
        // rasterises geometry has to agree on it to a subpixel.
        let view_proj = view.view_proj;

        let lighting_buffer = self
            .uniform_buffer_allocator
            .allocate_sized::<GpuLighting>()
            .unwrap();
        // Both halves fall back to the scene's flat ambient when nothing is
        // loaded — the diffuse as a band-0-only series, the specular as a tint
        // on a white cube. Two descriptions of the same uniform environment,
        // which is what keeps them from disagreeing.
        let ambient = lighting.ambient_color * lighting.ambient_intensity;
        let irradiance = renderer.environment.irradiance(ambient, environment);
        let env_specular = renderer.environment.specular_tint(ambient, environment);
        *lighting_buffer.write().unwrap() = to_gpu_lighting(
            lighting,
            camera.position,
            extent,
            shadows,
            irradiance,
            environment.yaw,
            env_specular,
        );

        let lighting_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer(0, lighting_buffer)],
            [],
        )
        .unwrap();

        // Set 3 is the screen-space and shadow inputs. The cascades are kept as
        // a separate image and comparison sampler rather than a combined one,
        // for the same reason the texture array is: Metal caps samplers per
        // stage far lower than sampled images.
        let ao_set = DescriptorSet::new(
            renderer.ctx.descriptor_set_allocator.clone(),
            self.pipeline.layout().set_layouts()[3].clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, ao_view, self.ao_sampler.clone()),
                WriteDescriptorSet::image_view(1, shadow_view),
                WriteDescriptorSet::image_view(3, renderer.environment.specular_view()),
                WriteDescriptorSet::sampler(4, renderer.environment.sampler()),
            ],
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

        // `extract_geometry` groups the order by (mesh, material), so each run
        // is one instanced draw: the recording cost stops scaling with entity
        // count and starts scaling with distinct mesh/material pairs.
        for run in draws.runs() {
            let item = draws.item(run.start);
            let Some(mesh) = renderer.meshes.get(item.mesh.0 as usize) else {
                continue;
            };
            let push = PushConstants {
                view_proj: view_proj.to_cols_array_2d(),
                material_index: item.material.0,
                object_base: run.start as u32,
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
        bounds: Aabb::from_points(vertices.iter().map(|v| Vec3::from(v.position))),
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

    // The shadow comparison sampler has to be immutable — part of the layout
    // rather than something written into a descriptor set — because MoltenVK
    // cannot accept a written one. Everything else in the layout is still
    // derived from the shaders' own interface.
    let mut layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
    layout_info.set_layouts[SHADOW_SET]
        .bindings
        .get_mut(&SHADOW_SAMPLER_BINDING)
        .expect("forward.frag must declare the shadow comparison sampler")
        .immutable_samplers = vec![super::shadow::comparison_sampler(device)];

    let layout = PipelineLayout::new(
        device.clone(),
        layout_info
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
