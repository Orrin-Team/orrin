//! The Vulkan implementation of the RHI: pools of vulkano objects behind the
//! typed handles, and an encoder that forwards to the existing `Recorder`.
//!
//! It owns no policy. A pass that used to build a pipeline, a sampler and a
//! descriptor set against vulkano now asks for the same three things by
//! description, and what is built is what was built before — the same sets from
//! the same descriptor cache, the same pipeline cache, the same recording calls.
//! That is what makes a ported pass pixel-identical rather than approximately
//! right.
//!
//! The pools sit behind mutexes because passes record from worker threads; a
//! lock is held only to look a handle up or to insert one, never across a
//! vulkano call.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use vulkano::buffer::allocator::SubbufferAllocatorCreateInfo;
use vulkano::buffer::{Buffer, BufferCreateInfo, Subbuffer};
use vulkano::command_buffer::{ClearAttachment, ClearRect};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::format::Format as VkFormat;
use vulkano::image::sampler::{
    Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
    ColorComponents,
};
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::rasterization::{CullMode as VkCullMode, DepthBiasState};
use vulkano::pipeline::graphics::vertex_input::{
    VertexBufferDescription, VertexDefinition, VertexInputRate, VertexInputState, VertexMemberInfo,
};
use vulkano::pipeline::graphics::viewport::Scissor;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};

use super::context::VkContext;
use super::convert::{vk_buffer_usage, vk_format, vk_samples};
use super::mesh::MeshArena;
use super::record::{Arena, Recorder};
use super::rendering;
use crate::geom::Aabb;
use crate::gfx::Vertex;
use crate::gfx::format::Format;
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, Blend, BlockKind, BufferDesc, BufferId,
    ComputePipelineDesc, CullMode, DepthCompare, DepthRect, GraphicsPipelineDesc, MeshSpan,
    PipelineId, Pool, SamplerDesc, SamplerId, TextureViewId, Topology, VertexFormat, VertexStream,
    Viewport,
};
use crate::gfx::shaders::ShaderId;

#[derive(Clone)]
enum PipelineObject {
    Graphics(Arc<GraphicsPipeline>),
    Compute(Arc<ComputePipeline>),
}

impl PipelineObject {
    fn layout(&self) -> &Arc<PipelineLayout> {
        match self {
            PipelineObject::Graphics(pipeline) => pipeline.layout(),
            PipelineObject::Compute(pipeline) => pipeline.layout(),
        }
    }
}

pub struct VkDriver {
    ctx: Arc<VkContext>,
    frame: AtomicU64,
    views: Mutex<Pool<Arc<ImageView>>>,
    buffers: Mutex<Pool<Subbuffer<[u8]>>>,
    samplers: Mutex<Pool<Arc<Sampler>>>,
    pipelines: Mutex<Pool<PipelineObject>>,
    uniforms: Arena,
    storage: Arena,
    vertices: Arena,
    /// The shared geometry every mesh is a span into. Owned by the device
    /// because every pass binds the same buffers; a pass names a mesh by its
    /// span and never holds the arena.
    arena: RwLock<MeshArena>,
    bind_groups: Mutex<Pool<Arc<DescriptorSet>>>,
}

fn lookup<T: Clone>(pool: &Mutex<Pool<T>>, index: u32, what: &str) -> T {
    pool.lock()
        .expect("a pass panicked while using the device")
        .get(index)
        .unwrap_or_else(|| panic!("{what} {index} was used after it was destroyed"))
        .clone()
}

impl VkDriver {
    pub fn new(ctx: Arc<VkContext>, frames_in_flight: u64) -> Self {
        let uniforms = Arena::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: vulkano::buffer::BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        let storage = Arena::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: vulkano::buffer::BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        let vertices = Arena::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: vulkano::buffer::BufferUsage::VERTEX_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        let arena = RwLock::new(MeshArena::new(&ctx));
        Self {
            ctx,
            frame: AtomicU64::new(0),
            arena,
            views: Mutex::new(Pool::new(frames_in_flight)),
            buffers: Mutex::new(Pool::new(frames_in_flight)),
            samplers: Mutex::new(Pool::new(frames_in_flight)),
            pipelines: Mutex::new(Pool::new(frames_in_flight)),
            uniforms,
            storage,
            vertices,
            bind_groups: Mutex::new(Pool::new(frames_in_flight)),
        }
    }

    /// Put a mesh into the shared arena and say where it landed, with its
    /// object-space bounds.
    pub fn upload_mesh(&self, vertices: &[Vertex], indices: &[u32]) -> (MeshSpan, Aabb) {
        self.arena
            .write()
            .expect("a pass panicked while using the device")
            .upload(&self.ctx, vertices, indices)
    }

    pub fn begin_frame(&self, frame: u64) {
        self.frame.store(frame, Ordering::Relaxed);
        self.views.lock().unwrap().collect(frame);
        self.buffers.lock().unwrap().collect(frame);
        self.samplers.lock().unwrap().collect(frame);
        self.pipelines.lock().unwrap().collect(frame);
        self.bind_groups.lock().unwrap().collect(frame);
    }

    /// Hand a graph-owned view to the RHI for the frame being recorded. The
    /// handle retires with the frame, so a pass that names it later than
    /// `frames_in_flight` frames on is a bug the pool reports.
    pub fn import_view(&self, view: Arc<ImageView>) -> TextureViewId {
        let mut pool = self.views.lock().unwrap();
        let index = pool.insert(view);
        pool.remove(index, self.frame.load(Ordering::Relaxed));
        TextureViewId(index)
    }

    pub fn create_texture_2d(
        &self,
        pixels: &[u8],
        extent: [u32; 2],
        format: Format,
    ) -> TextureViewId {
        let view = super::texture::upload_texture(
            &self.ctx,
            pixels,
            extent,
            vk_format(format),
            super::texture::MipPolicy::None,
        );
        TextureViewId(self.views.lock().unwrap().insert(view))
    }

    pub fn upload_vertices(&self, bytes: &[u8]) -> BufferId {
        let buffer = self.vertices.allocate_slice::<u8>(bytes.len() as u64);
        buffer.write().unwrap().copy_from_slice(bytes);
        self.import_buffer(buffer)
    }

    pub fn upload_block(&self, kind: BlockKind, bytes: &[u8]) -> BufferId {
        let arena = match kind {
            BlockKind::Uniform => &self.uniforms,
            BlockKind::Storage => &self.storage,
        };
        let buffer = arena.allocate_slice::<u8>(bytes.len() as u64);
        buffer.write().unwrap().copy_from_slice(bytes);
        self.import_buffer(buffer)
    }

    pub fn upload_uniform(&self, words: &[u32]) -> BufferId {
        let buffer = self.uniforms.allocate_slice::<u32>(words.len() as u64);
        buffer.write().unwrap().copy_from_slice(words);
        self.import_buffer(buffer.into_bytes())
    }

    pub fn view_extent(&self, view: TextureViewId) -> [u32; 2] {
        let extent = lookup(&self.views, view.0, "texture view").image().extent();
        [extent[0], extent[1]]
    }

    /// A sampler built outside the RHI, for a pass that binds it beside RHI
    /// resources. Valid for the frame being recorded, like the other imports.
    pub fn import_sampler(&self, sampler: Arc<Sampler>) -> SamplerId {
        let mut pool = self.samplers.lock().unwrap();
        let index = pool.insert(sampler);
        pool.remove(index, self.frame.load(Ordering::Relaxed));
        SamplerId(index)
    }

    pub fn import_buffer(&self, buffer: Subbuffer<[u8]>) -> BufferId {
        let mut pool = self.buffers.lock().unwrap();
        let index = pool.insert(buffer);
        pool.remove(index, self.frame.load(Ordering::Relaxed));
        BufferId(index)
    }

    pub fn create_sampler(&self, desc: SamplerDesc) -> SamplerId {
        let filter = if desc.linear {
            Filter::Linear
        } else {
            Filter::Nearest
        };
        let address = if desc.repeat {
            SamplerAddressMode::Repeat
        } else {
            SamplerAddressMode::ClampToEdge
        };
        if desc.compare {
            // The sampler the shading passes read the cascades through. It must be
            // an *immutable* sampler baked into the descriptor set layout, not one
            // written into a set: Metal makes a sampler's compare function a
            // compile-time property, so MoltenVK rejects a written comparison
            // sampler unless `mutable_comparison_samplers` is available, which on
            // a portability subset device it is not.
            //
            // `Less` against a depth map cleared to 1.0, and a white border so a
            // fragment sampling outside a cascade reads as lit rather than as
            // shadowed. The comparison happens before the bilinear filter, so a
            // single tap is already a 2x2 percentage-closer result.
            let sampler = Sampler::new(
                self.ctx.device.clone(),
                SamplerCreateInfo {
                    mag_filter: Filter::Linear,
                    min_filter: Filter::Linear,
                    address_mode: [SamplerAddressMode::ClampToBorder; 3],
                    border_color: vulkano::image::sampler::BorderColor::FloatOpaqueWhite,
                    compare: Some(CompareOp::Less),
                    ..Default::default()
                },
            )
            .expect("failed to create the comparison sampler");
            return SamplerId(self.samplers.lock().unwrap().insert(sampler));
        }
        let info = SamplerCreateInfo {
            mag_filter: filter,
            min_filter: filter,
            mipmap_mode: if desc.mipmap_linear {
                SamplerMipmapMode::Linear
            } else {
                SamplerMipmapMode::Nearest
            },
            address_mode: [address; 3],
            lod: 0.0..=desc
                .max_lod
                .unwrap_or(vulkano::image::sampler::LOD_CLAMP_NONE),
            anisotropy: desc.anisotropy,
            ..Default::default()
        };
        let sampler =
            Sampler::new(self.ctx.device.clone(), info).expect("failed to create a sampler");
        SamplerId(self.samplers.lock().unwrap().insert(sampler))
    }

    pub fn create_graphics_pipeline(&self, desc: &GraphicsPipelineDesc) -> PipelineId {
        let device = &self.ctx.device;
        let vertex = entry_point(&self.ctx, desc.vertex);
        let stages = [
            PipelineShaderStageCreateInfo::new(vertex.clone()),
            PipelineShaderStageCreateInfo::new(entry_point(&self.ctx, desc.fragment)),
        ];
        let vertex_input = if desc.vertex_streams.is_empty() {
            VertexInputState::default()
        } else {
            let descriptions: Vec<_> = desc.vertex_streams.iter().map(vertex_buffer).collect();
            descriptions
                .definition(&vertex)
                .expect("a vertex stream does not match the shader's inputs")
        };
        let lifted = desc.layout_from.map(|source| {
            lookup(&self.pipelines, source.0, "pipeline")
                .layout()
                .clone()
        });
        let mut layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
        if let Some(set) = desc.partial_texture_set {
            self.ctx
                .mark_texture_array_partial(&mut layout_info, set as usize);
        }
        for &(set, binding, sampler) in desc.immutable_samplers {
            layout_info.set_layouts[set as usize]
                .bindings
                .get_mut(&binding)
                .expect("an immutable sampler names a binding the shaders declare")
                .immutable_samplers = vec![lookup(&self.samplers, sampler.0, "sampler")];
        }
        let layout = lifted.unwrap_or_else(|| {
            let mut info = layout_info
                .into_pipeline_layout_create_info(device.clone())
                .unwrap();
            if let Some((source, count)) = desc.layout_prefix_from {
                // The source's own objects, not a second set derived from the
                // same declarations: deriving them would give layouts that match
                // binding for binding and still fail to bind, because an
                // immutable sampler is compared by identity.
                let source = lookup(&self.pipelines, source.0, "pipeline");
                info.set_layouts.splice(
                    ..count as usize,
                    source.layout().set_layouts()[..count as usize]
                        .iter()
                        .cloned(),
                );
            }
            PipelineLayout::new(device.clone(), info).unwrap()
        });
        let colors: Vec<_> = desc.color_formats.iter().map(|&f| vk_format(f)).collect();
        let pipeline = GraphicsPipeline::new(
            device.clone(),
            self.ctx.pipeline_cache(),
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input),
                input_assembly_state: Some(InputAssemblyState {
                    topology: match desc.topology {
                        Topology::Triangles => PrimitiveTopology::TriangleList,
                        Topology::Lines => PrimitiveTopology::LineList,
                    },
                    ..Default::default()
                }),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState {
                    cull_mode: match desc.cull {
                        CullMode::None => VkCullMode::None,
                        CullMode::Back => VkCullMode::Back,
                    },
                    depth_bias: desc.dynamic_depth_bias.then(DepthBiasState::default),
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState {
                    rasterization_samples: vk_samples(desc.samples),
                    alpha_to_coverage_enable: desc.alpha_to_coverage,
                    ..Default::default()
                }),
                depth_stencil_state: desc.depth_test.map(|test| DepthStencilState {
                    depth: Some(DepthState {
                        write_enable: test.write,
                        compare_op: match test.compare {
                            DepthCompare::Less => CompareOp::Less,
                            DepthCompare::Equal => CompareOp::Equal,
                            DepthCompare::LessOrEqual => CompareOp::LessOrEqual,
                        },
                    }),
                    ..Default::default()
                }),
                // A depth-only pass has no attachments to blend into.
                color_blend_state: (!colors.is_empty()).then(|| {
                    let mut state = ColorBlendState::with_attachment_states(
                        colors.len() as u32,
                        ColorBlendAttachmentState::default(),
                    );
                    for (index, attachment) in state.attachments.iter_mut().enumerate() {
                        let blend = desc
                            .attachment_blends
                            .get(index)
                            .copied()
                            .unwrap_or(desc.blend);
                        attachment.blend = attachment_blend(blend);
                        if desc.mask_extra_attachments && index > 0 {
                            attachment.color_write_mask = ColorComponents::empty();
                        }
                    }
                    state
                }),
                dynamic_state: if desc.dynamic_depth_bias {
                    // Scissor as well as viewport, because the atlas aims the
                    // pipeline at one tile of a shared attachment and the
                    // viewport alone does not promise a fragment stays inside it.
                    [
                        DynamicState::Viewport,
                        DynamicState::Scissor,
                        DynamicState::DepthBias,
                    ]
                    .into_iter()
                    .collect()
                } else {
                    [DynamicState::Viewport].into_iter().collect()
                },
                subpass: Some(
                    rendering::pipeline_info(&colors, desc.depth_format.map(vk_format)).into(),
                ),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap();
        PipelineId(
            self.pipelines
                .lock()
                .unwrap()
                .insert(PipelineObject::Graphics(pipeline)),
        )
    }

    /// The set layout `set` of `pipeline` was created with, for a pass that
    /// still builds its own descriptor sets.
    pub fn set_layout(
        &self,
        pipeline: PipelineId,
        set: usize,
    ) -> Arc<vulkano::descriptor_set::layout::DescriptorSetLayout> {
        lookup(&self.pipelines, pipeline.0, "pipeline")
            .layout()
            .set_layouts()[set]
            .clone()
    }

    /// The whole layout of a pipeline, for a pass whose own pipeline must be set
    /// compatible with it.
    pub fn pipeline_layout(&self, pipeline: PipelineId) -> Arc<PipelineLayout> {
        lookup(&self.pipelines, pipeline.0, "pipeline")
            .layout()
            .clone()
    }

    /// A descriptor set built outside the RHI, for the frame being recorded.
    pub fn import_set(&self, set: Arc<DescriptorSet>) -> BindGroupId {
        let mut pool = self.bind_groups.lock().unwrap();
        let index = pool.insert(set);
        pool.remove(index, self.frame.load(Ordering::Relaxed));
        BindGroupId(index)
    }

    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> PipelineId {
        self.create_compute_pipeline_with_samplers(desc, &[])
    }

    /// A compute pipeline with samplers baked into its set layout, as
    /// `(set, binding, sampler)`. Needed where a device must not be handed a
    /// written comparison sampler.
    pub fn create_compute_pipeline_with_samplers(
        &self,
        desc: &ComputePipelineDesc,
        immutable_samplers: &[(u32, u32, SamplerId)],
    ) -> PipelineId {
        let device = &self.ctx.device;
        let stage = PipelineShaderStageCreateInfo::new(entry_point(&self.ctx, desc.shader));
        let mut layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage]);
        for &(set, binding, sampler) in immutable_samplers {
            layout_info.set_layouts[set as usize]
                .bindings
                .get_mut(&binding)
                .expect("an immutable sampler names a binding the shader declares")
                .immutable_samplers = vec![lookup(&self.samplers, sampler.0, "sampler")];
        }
        let layout = PipelineLayout::new(
            device.clone(),
            layout_info
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let pipeline = ComputePipeline::new(
            device.clone(),
            self.ctx.pipeline_cache(),
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap();
        PipelineId(
            self.pipelines
                .lock()
                .unwrap()
                .insert(PipelineObject::Compute(pipeline)),
        )
    }

    pub fn create_buffer_init(&self, desc: BufferDesc, initial: &[u32]) -> BufferId {
        assert_eq!(
            desc.size,
            initial.len() as u64 * 4,
            "a buffer's size is its initial words"
        );
        let buffer = Buffer::from_iter(
            self.ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: vk_buffer_usage(desc.usage),
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            initial.iter().copied(),
        )
        .expect("failed to allocate a buffer");
        BufferId(self.buffers.lock().unwrap().insert(buffer.into_bytes()))
    }

    /// The buffer behind a handle, for a pass still written against vulkano.
    /// Goes away with the last such pass.
    pub fn buffer(&self, id: BufferId) -> Subbuffer<[u8]> {
        lookup(&self.buffers, id.0, "buffer")
    }

    fn writes(&self, bindings: &[Binding]) -> Vec<WriteDescriptorSet> {
        let mut writes: Vec<WriteDescriptorSet> = Vec::with_capacity(bindings.len());
        let mut index = 0;
        while index < bindings.len() {
            let binding = bindings[index];
            // Consecutive storage textures at one binding are the elements of
            // an image array, in order.
            if let BindingResource::StorageTexture(_) = binding.resource {
                let run = bindings[index..]
                    .iter()
                    .take_while(|b| {
                        b.binding == binding.binding
                            && matches!(b.resource, BindingResource::StorageTexture(_))
                    })
                    .count();
                if run > 1 {
                    let views = bindings[index..index + run]
                        .iter()
                        .map(|b| match b.resource {
                            BindingResource::StorageTexture(view) => {
                                lookup(&self.views, view.0, "texture view")
                            }
                            _ => unreachable!(),
                        });
                    writes.push(WriteDescriptorSet::image_view_array(
                        binding.binding,
                        0,
                        views,
                    ));
                    index += run;
                    continue;
                }
            }
            writes.push(match binding.resource {
                BindingResource::SampledTexture { view, sampler } => {
                    WriteDescriptorSet::image_view_sampler(
                        binding.binding,
                        lookup(&self.views, view.0, "texture view"),
                        lookup(&self.samplers, sampler.0, "sampler"),
                    )
                }
                BindingResource::Texture(view) => WriteDescriptorSet::image_view(
                    binding.binding,
                    lookup(&self.views, view.0, "texture view"),
                ),
                BindingResource::Sampler(sampler) => WriteDescriptorSet::sampler(
                    binding.binding,
                    lookup(&self.samplers, sampler.0, "sampler"),
                ),
                BindingResource::StorageTexture(view) => WriteDescriptorSet::image_view(
                    binding.binding,
                    lookup(&self.views, view.0, "texture view"),
                ),
                BindingResource::UniformBuffer(buffer) | BindingResource::StorageBuffer(buffer) => {
                    WriteDescriptorSet::buffer(
                        binding.binding,
                        lookup(&self.buffers, buffer.0, "buffer"),
                    )
                }
            });
            index += 1;
        }
        writes
    }

    /// A bind group the descriptor cache remembers while what it points at is
    /// unchanged. For sets over images, samplers and buffers that outlive the
    /// frame.
    pub fn create_bind_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
    ) -> BindGroupId {
        let pipeline = lookup(&self.pipelines, pipeline.0, "pipeline");
        let set = self.ctx.descriptor_sets.get(
            &self.ctx.descriptor_set_allocator,
            &pipeline.layout().set_layouts()[set as usize],
            self.writes(bindings),
        );
        self.import_set(set)
    }

    /// A bind group built afresh for this frame and not remembered. For a set
    /// that binds a buffer suballocated per frame: it would miss the cache every
    /// frame and pin each frame's buffer for the cache's keep time besides.
    pub fn create_transient_bind_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
    ) -> BindGroupId {
        let pipeline = lookup(&self.pipelines, pipeline.0, "pipeline");
        let set = DescriptorSet::new(
            self.ctx.descriptor_set_allocator.clone(),
            pipeline.layout().set_layouts()[set as usize].clone(),
            self.writes(bindings),
            [],
        )
        .expect("failed to build a descriptor set");
        self.import_set(set)
    }
}

fn attachment_blend(blend: Blend) -> Option<AttachmentBlend> {
    let factors = |src, dst, op| AttachmentBlend {
        src_color_blend_factor: src,
        dst_color_blend_factor: dst,
        color_blend_op: op,
        src_alpha_blend_factor: src,
        dst_alpha_blend_factor: dst,
        alpha_blend_op: op,
    };
    match blend {
        Blend::None => None,
        Blend::Alpha => Some(AttachmentBlend::alpha()),
        Blend::Premultiplied => Some(factors(
            BlendFactor::One,
            BlendFactor::OneMinusSrcAlpha,
            BlendOp::Add,
        )),
        // Max, so the hovered mesh drawn after a selected one it overlaps cannot
        // take the selected channel back to zero.
        Blend::Max => Some(factors(BlendFactor::One, BlendFactor::One, BlendOp::Max)),
        Blend::Additive => Some(factors(BlendFactor::One, BlendFactor::One, BlendOp::Add)),
        // The colour and alpha factors differ here, as the pipeline always had
        // them: `OneMinusSrcColor` on colour and `OneMinusSrcAlpha` on alpha.
        Blend::Revealage => Some(AttachmentBlend {
            src_color_blend_factor: BlendFactor::Zero,
            dst_color_blend_factor: BlendFactor::OneMinusSrcColor,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::Zero,
            dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
            alpha_blend_op: BlendOp::Add,
        }),
    }
}

fn vertex_buffer(stream: &VertexStream) -> VertexBufferDescription {
    VertexBufferDescription {
        members: stream
            .attributes
            .iter()
            .map(|attribute| {
                let format = match attribute.format {
                    VertexFormat::Float2 => VkFormat::R32G32_SFLOAT,
                    VertexFormat::Float3 => VkFormat::R32G32B32_SFLOAT,
                    VertexFormat::Float4 => VkFormat::R32G32B32A32_SFLOAT,
                    VertexFormat::Unorm8x4 => VkFormat::R8G8B8A8_UNORM,
                };
                (
                    attribute.name.to_string(),
                    VertexMemberInfo {
                        offset: attribute.offset,
                        format,
                        num_elements: 1,
                        stride: 0,
                    },
                )
            })
            .collect(),
        stride: stream.stride,
        input_rate: if stream.per_instance {
            VertexInputRate::Instance { divisor: 1 }
        } else {
            VertexInputRate::Vertex
        },
    }
}

fn entry_point(ctx: &VkContext, shader: ShaderId) -> vulkano::shader::EntryPoint {
    let blob = shader.blob();
    let words: Vec<u32> = blob
        .spirv
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    // SAFETY: the words are the output of shaderc for a shader in the build
    // manifest, compiled and cross-translated by `build.rs`, which fails the
    // build on anything SPIR-V-invalid; vulkano validates the module's own
    // capabilities against the device when it creates it.
    let module =
        unsafe { ShaderModule::new(ctx.device.clone(), ShaderModuleCreateInfo::new(&words)) }
            .unwrap_or_else(|e| panic!("{}: {e}", blob.name));
    module
        .entry_point("main")
        .unwrap_or_else(|| panic!("{} has no main", blob.name))
}

/// Forwards to a [`Recorder`], remembering the bound pipeline's layout because
/// binding a set or pushing constants needs it.
pub struct VkEncoder<'a> {
    recorder: &'a mut Recorder,
    driver: &'a VkDriver,
    bound: Option<PipelineObject>,
}

impl<'a> VkEncoder<'a> {
    pub(super) fn new(recorder: &'a mut Recorder, driver: &'a VkDriver) -> Self {
        Self {
            recorder,
            driver,
            bound: None,
        }
    }

    fn layout(&self) -> &Arc<PipelineLayout> {
        self.bound.as_ref().expect("no pipeline is bound").layout()
    }

    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.recorder.set_viewport(
            0,
            &[vulkano::pipeline::graphics::viewport::Viewport {
                offset: viewport.origin,
                extent: viewport.extent,
                depth_range: viewport.depth_range[0]..=viewport.depth_range[1],
            }],
        );
    }

    pub fn set_scissor(&mut self, offset: [u32; 2], extent: [u32; 2]) {
        self.recorder.set_scissor(0, &[Scissor { offset, extent }]);
    }

    pub fn set_depth_bias(&mut self, constant: f32, clamp: f32, slope: f32) {
        self.recorder.set_depth_bias(constant, clamp, slope);
    }

    pub fn clear_depth(&mut self, rects: &[DepthRect]) {
        self.recorder.clear_attachments(
            [ClearAttachment::Depth(1.0)],
            rects.iter().map(|rect| ClearRect {
                offset: rect.offset,
                extent: rect.extent,
                array_layers: 0..1,
            }),
        );
    }

    pub fn bind_pipeline(&mut self, pipeline: PipelineId) {
        let pipeline = lookup(&self.driver.pipelines, pipeline.0, "pipeline");
        match &pipeline {
            PipelineObject::Graphics(graphics) => self.recorder.bind_pipeline_graphics(graphics),
            PipelineObject::Compute(compute) => self.recorder.bind_pipeline_compute(compute),
        };
        self.bound = Some(pipeline);
    }

    pub fn bind_vertex_buffer(&mut self, buffer: BufferId) {
        let buffer = lookup(&self.driver.buffers, buffer.0, "buffer");
        self.recorder.bind_vertex_buffers(0, buffer);
    }

    pub fn bind_group(&mut self, set: u32, group: BindGroupId) {
        let group = lookup(&self.driver.bind_groups, group.0, "bind group");
        let layout = self.layout().clone();
        let point = match self.bound {
            Some(PipelineObject::Compute(_)) => PipelineBindPoint::Compute,
            _ => PipelineBindPoint::Graphics,
        };
        self.recorder
            .bind_descriptor_sets(point, &layout, set, &[group]);
    }

    pub fn push_constants(&mut self, words: &[u32]) {
        macro_rules! by_length {
            ($($n:literal)+) => {
                match words.len() {
                    $($n => self.push_array::<$n>(words),)+
                    n => panic!("{n} words of push constants; the RHI takes 1 to 32"),
                }
            };
        }
        by_length!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32);
    }

    /// vulkano's push-constant call is sized by its argument's type, so the
    /// length has to be a constant by the time it is made.
    fn push_array<const N: usize>(&mut self, words: &[u32]) {
        let mut block = [0u32; N];
        block.copy_from_slice(words);
        let layout = self.layout().clone();
        self.recorder.push_constants(&layout, 0, &block);
    }

    pub fn bind_geometry(&mut self, positions_only: bool) {
        let arena = self
            .driver
            .arena
            .read()
            .expect("a pass panicked while using the device");
        if positions_only {
            arena.bind_positions(self.recorder);
        } else {
            arena.bind(self.recorder);
        }
    }

    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instances: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        self.recorder.draw_indexed(
            index_count,
            instances,
            first_index,
            vertex_offset,
            first_instance,
        );
    }

    pub fn draw_indexed_indirect(&mut self, commands: BufferId) {
        let commands = lookup(&self.driver.buffers, commands.0, "buffer")
            .reinterpret::<[vulkano::command_buffer::DrawIndexedIndirectCommand]>();
        self.recorder.draw_indexed_indirect(commands);
    }

    pub fn dispatch(&mut self, groups: [u32; 3]) {
        self.recorder.dispatch(groups);
    }

    pub fn draw(&mut self, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
        self.recorder
            .draw(vertices, instances, first_vertex, first_instance);
    }
}

#[cfg(test)]
mod tests {
    use vulkano::pipeline::graphics::vertex_input::Vertex as _;

    use super::*;
    use crate::gfx::{PositionVertex, SurfaceVertex};

    /// The RHI's description of a stream is written by hand beside the vertex
    /// struct whose `Vertex` derive is what the pipelines used to take. A field
    /// moved in one and not the other would draw a silently wrong mesh.
    fn agrees(stream: &VertexStream, derived: VertexBufferDescription) {
        let described = vertex_buffer(stream);
        assert_eq!(described.stride, derived.stride);
        // `stride` on a member is the element stride of an array, which the
        // derive fills in for a scalar and nothing reads.
        let layout = |description: &VertexBufferDescription| {
            let mut members: Vec<_> = description
                .members
                .iter()
                .map(|(name, m)| (name.clone(), m.offset, m.format, m.num_elements))
                .collect();
            members.sort_by(|a, b| a.0.cmp(&b.0));
            members
        };
        assert_eq!(layout(&described), layout(&derived));
    }

    #[test]
    fn the_position_stream_matches_its_derived_layout() {
        agrees(&PositionVertex::STREAM, PositionVertex::per_vertex());
    }

    #[test]
    fn the_surface_stream_matches_its_derived_layout() {
        agrees(&SurfaceVertex::STREAM, SurfaceVertex::per_vertex());
    }
}
