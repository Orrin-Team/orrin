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
use vulkano::image::sampler::{ComponentMapping, ComponentSwizzle};
use vulkano::image::sampler::{
    Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
};
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::image::{Image, ImageCreateInfo, ImageSubresourceRange, ImageType};
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
use super::convert::{vk_buffer_usage, vk_format, vk_image_usage, vk_samples};
use super::mesh::MeshArena;
use super::record::{self, Arena, Recorder};
use super::rendering;
use crate::geom::Aabb;
use crate::gfx::Vertex;
use crate::gfx::format::Format;
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, Blend, BlockKind, BufferCopy, BufferDesc, BufferId,
    ComputePipelineDesc, CullMode, DepthCompare, DepthRect, Encoder, Feature, GraphicsPipelineDesc,
    LoadOp, MeshSpan, PipelineId, Pool, RenderTargets, SamplerDesc, SamplerId, ShaderModuleId,
    ShaderRef, StoreOp, TextureDesc, TextureDimension, TextureSwizzle, TextureViewId, Topology,
    VertexFormat, VertexStream, Viewport,
};
use crate::gfx::shaders::{CompiledBindings, CompiledShader};

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
    /// Loaded shaders as SPIR-V and a name: a layout here is reflected from the
    /// SPIR-V, so the MSL and the binding map are dropped at load.
    shaders: Mutex<Pool<Arc<CompiledShader>>>,
    uniforms: Arena,
    storage: Arena,
    staging: Arena,
    indices: Arena,
    vertices: Arena,
    /// The shared geometry every mesh is a span into. Owned by the device
    /// because every pass binds the same buffers; a pass names a mesh by its
    /// span and never holds the arena.
    arena: RwLock<MeshArena>,
    bind_groups: Mutex<Pool<Arc<DescriptorSet>>>,
    /// Texture uploads, none of which the CPU waits for. See `upload`.
    uploads: super::upload::Uploads,
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
        let staging = Arena::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: vulkano::buffer::BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        let indices = Arena::new(
            ctx.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: vulkano::buffer::BufferUsage::INDEX_BUFFER,
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
        let uploads = super::upload::Uploads::new(&ctx);
        Self {
            ctx,
            frame: AtomicU64::new(0),
            arena,
            views: Mutex::new(Pool::new(frames_in_flight)),
            buffers: Mutex::new(Pool::new(frames_in_flight)),
            samplers: Mutex::new(Pool::new(frames_in_flight)),
            pipelines: Mutex::new(Pool::new(frames_in_flight)),
            shaders: Mutex::new(Pool::new(frames_in_flight)),
            uniforms,
            staging,
            indices,
            storage,
            vertices,
            bind_groups: Mutex::new(Pool::new(frames_in_flight)),
            uploads,
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
        self.shaders.lock().unwrap().collect(frame);
        self.bind_groups.lock().unwrap().collect(frame);
    }

    /// Take ownership of a view the driver built for a graph-owned image, for as
    /// long as the image lives: the caller retires it with `destroy_texture`.
    pub fn adopt_view(&self, view: Arc<ImageView>) -> TextureViewId {
        TextureViewId(self.views.lock().unwrap().insert(view))
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

    pub fn create_texture(&self, desc: &TextureDesc) -> TextureViewId {
        self.create_texture_swizzled(desc, TextureSwizzle::Identity)
    }

    pub fn max_anisotropy(&self) -> Option<f32> {
        let device = &self.ctx.device;
        device.enabled_features().sampler_anisotropy.then(|| {
            device
                .physical_device()
                .properties()
                .max_sampler_anisotropy
                .min(16.0)
        })
    }

    pub fn create_texture_mipped(
        &self,
        pixels: &[u8],
        extent: [u32; 2],
        format: Format,
    ) -> TextureViewId {
        let prepared = super::texture::prepare_texture(
            &self.ctx,
            self.uploads.staging(),
            pixels,
            extent,
            vk_format(format),
            super::texture::MipPolicy::Generate,
        );
        self.insert_uploaded(prepared, true)
    }

    pub fn create_texture_levels(&self, levels: &crate::gfx::TextureLevels) -> TextureViewId {
        use crate::gfx::TexelFormat;
        use vulkano::format::Format as VkFormat;
        let format = match levels.format {
            TexelFormat::Rgba8Srgb => VkFormat::R8G8B8A8_SRGB,
            TexelFormat::Rgba8Unorm => VkFormat::R8G8B8A8_UNORM,
            TexelFormat::Bc5Unorm => VkFormat::BC5_UNORM_BLOCK,
            TexelFormat::Bc7Unorm => VkFormat::BC7_UNORM_BLOCK,
            TexelFormat::Bc7Srgb => VkFormat::BC7_SRGB_BLOCK,
        };
        let prepared = super::texture::prepare_levels(
            &self.ctx,
            self.uploads.staging(),
            format,
            [levels.width, levels.height],
            &levels.levels,
        );
        self.insert_uploaded(prepared, true)
    }

    /// Submit an upload without waiting for it and hand back its view, which is
    /// bindable at once — see `upload` for why that is sound on both paths.
    fn insert_uploaded(
        &self,
        prepared: super::texture::Prepared,
        may_transfer: bool,
    ) -> TextureViewId {
        let image = self.uploads.submit(&self.ctx, prepared, may_transfer);
        let view = ImageView::new_default(image).expect("failed to create texture image view");
        TextureViewId(self.views.lock().unwrap().insert(view))
    }

    /// Record, at the top of a frame's command buffer, the ownership acquires
    /// the transfer queue's finished copies are owed, and return the semaphores
    /// the frame's first submission waits on for them. Also releases whatever
    /// uploads have finished since the last frame.
    pub(super) fn record_upload_acquires(
        &self,
        recorder: &mut super::record::Recorder,
    ) -> Vec<(
        Arc<vulkano::sync::semaphore::Semaphore>,
        ash::vk::PipelineStageFlags,
    )> {
        self.uploads.record_acquires(&self.ctx, recorder)
    }

    pub fn supports(&self, feature: Feature) -> bool {
        use vulkano::format::FormatFeatures;
        let physical = self.ctx.device.physical_device();
        let usable = |format, features: FormatFeatures| {
            physical
                .format_properties(format)
                .is_ok_and(|properties| properties.optimal_tiling_features.contains(features))
        };
        match feature {
            Feature::BlockCompression => self.ctx.samples_bc,
            Feature::CoverageTexture => {
                let device = &self.ctx.device;
                let swizzle = !device.enabled_extensions().khr_portability_subset
                    || device.enabled_features().image_view_format_swizzle;
                swizzle
                    && usable(
                        vulkano::format::Format::R8G8_UNORM,
                        FormatFeatures::SAMPLED_IMAGE | FormatFeatures::TRANSFER_DST,
                    )
            }
        }
    }

    pub fn create_texture_swizzled(
        &self,
        desc: &TextureDesc,
        swizzle: TextureSwizzle,
    ) -> TextureViewId {
        let usage = vk_image_usage(desc.usage);
        let image = Image::new(
            self.ctx.memory_allocator.clone(),
            ImageCreateInfo {
                // Without this the *image* creation succeeds and the cube view
                // fails, which points the error at the wrong line.
                flags: if desc.dimension == TextureDimension::Cube {
                    vulkano::image::ImageCreateFlags::CUBE_COMPATIBLE
                } else {
                    vulkano::image::ImageCreateFlags::empty()
                },
                image_type: match desc.dimension {
                    TextureDimension::D3 => ImageType::Dim3d,
                    _ => ImageType::Dim2d,
                },
                format: vk_format(desc.format),
                extent: desc.extent,
                mip_levels: desc.mip_levels,
                array_layers: desc.array_layers,
                samples: vk_samples(desc.samples),
                usage,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("failed to allocate a texture");
        let view = ImageView::new(
            image.clone(),
            ImageViewCreateInfo {
                view_type: match desc.dimension {
                    TextureDimension::D2 => ImageViewType::Dim2d,
                    TextureDimension::D2Array => ImageViewType::Dim2dArray,
                    TextureDimension::D3 => ImageViewType::Dim3d,
                    TextureDimension::Cube => ImageViewType::Cube,
                },
                component_mapping: match swizzle {
                    TextureSwizzle::Identity => ComponentMapping::identity(),
                    TextureSwizzle::CoverageToWhite => ComponentMapping {
                        r: ComponentSwizzle::Red,
                        g: ComponentSwizzle::Red,
                        b: ComponentSwizzle::Red,
                        a: ComponentSwizzle::Green,
                    },
                },
                // A storage descriptor takes exactly one level, so a view
                // spanning a pyramid cannot claim that usage.
                usage: if desc.mip_levels > 1 {
                    usage - vulkano::image::ImageUsage::STORAGE
                } else {
                    usage
                },
                ..ImageViewCreateInfo::from_image(&image)
            },
        )
        .expect("failed to create a texture view");
        TextureViewId(self.views.lock().unwrap().insert(view))
    }

    pub fn mip_view(&self, texture: TextureViewId, level: u32) -> TextureViewId {
        let whole = lookup(&self.views, texture.0, "texture view");
        let image = whole.image().clone();
        let view = ImageView::new(
            image.clone(),
            ImageViewCreateInfo {
                view_type: whole.view_type(),
                subresource_range: ImageSubresourceRange {
                    mip_levels: level..level + 1,
                    ..image.subresource_range()
                },
                ..ImageViewCreateInfo::from_image(&image)
            },
        )
        .expect("failed to create a mip view");
        TextureViewId(self.views.lock().unwrap().insert(view))
    }

    pub fn destroy_texture(&self, view: TextureViewId) {
        self.views
            .lock()
            .unwrap()
            .remove(view.0, self.frame.load(Ordering::Relaxed));
    }

    pub fn fill_depth(&self, texture: TextureViewId, depth: f32) {
        let view = lookup(&self.views, texture.0, "texture view");
        let image = view.image().clone();
        let mut builder = Recorder::new(&self.ctx);
        // Fresh from `create_texture`, so there is nothing to preserve; cleared
        // once and then only ever sampled, so this is the whole of its
        // synchronisation.
        builder
            .image_barrier(super::record::to_transfer_dst(
                image.clone(),
                super::record::whole_image(&image),
                vulkano::image::ImageLayout::Undefined,
            ))
            .clear_depth_stencil_image(vulkano::command_buffer::ClearDepthStencilImageInfo {
                clear_value: vulkano::format::ClearDepthStencilValue { depth, stencil: 0 },
                ..vulkano::command_buffer::ClearDepthStencilImageInfo::image(image.clone())
            })
            .image_barrier(super::record::to_shader_read(
                image.clone(),
                super::record::whole_image(&image),
                vulkano::image::ImageLayout::TransferDstOptimal,
            ));
        builder.submit_and_wait(&self.ctx);
    }

    pub fn initialise_sampled(&self, texture: TextureViewId) {
        let view = lookup(&self.views, texture.0, "texture view");
        let image = view.image().clone();
        let mut builder = Recorder::new(&self.ctx);
        builder.image_barrier(super::record::to_shader_read(
            image.clone(),
            super::record::whole_image(&image),
            vulkano::image::ImageLayout::Undefined,
        ));
        builder.submit_and_wait(&self.ctx);
    }

    /// The vulkano view behind an id, for the Vulkan-side graph executor, which
    /// resolves graph resources to images and hands them to barriers.
    pub fn vk_view(&self, view: TextureViewId) -> Arc<ImageView> {
        lookup(&self.views, view.0, "texture view")
    }

    pub fn create_texture_2d(
        &self,
        pixels: &[u8],
        extent: [u32; 2],
        format: Format,
    ) -> TextureViewId {
        // Engine-owned textures, some of them sampled by one-shot graphics work
        // outside any frame — the environment bake reads its equirect source in
        // a submission of its own, which would never record an acquire. So
        // these stay on the graphics queue, still without a wait.
        let prepared = super::texture::prepare_texture(
            &self.ctx,
            self.uploads.staging(),
            pixels,
            extent,
            vk_format(format),
            super::texture::MipPolicy::None,
        );
        self.insert_uploaded(prepared, false)
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
            BlockKind::Staging => &self.staging,
            BlockKind::Index => &self.indices,
        };
        // Whole words, so a copy out of this block starts on a texel of any
        // format a texture here takes: deltas share one staging arena, and an
        // odd-sized one would otherwise leave the next off a texel boundary.
        let buffer = arena
            .allocate_slice::<u32>((bytes.len() as u64).div_ceil(4))
            .reinterpret::<[u8]>()
            .slice(..bytes.len() as u64);
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

    pub fn create_sampler_wrap_u(&self, desc: SamplerDesc) -> SamplerId {
        let sampler = Sampler::new(
            self.ctx.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Nearest,
                address_mode: [
                    SamplerAddressMode::Repeat,
                    SamplerAddressMode::ClampToEdge,
                    SamplerAddressMode::ClampToEdge,
                ],
                lod: 0.0..=desc.max_lod.unwrap_or(0.0),
                ..Default::default()
            },
        )
        .expect("failed to create a sampler");
        SamplerId(self.samplers.lock().unwrap().insert(sampler))
    }

    pub fn create_texture_cube(&self, pixels: &[u8], face: u32, format: Format) -> TextureViewId {
        let staging = Buffer::from_iter(
            self.ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: vulkano::buffer::BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            pixels.iter().copied(),
        )
        .expect("failed to allocate cube staging");
        let image = Image::new(
            self.ctx.memory_allocator.clone(),
            ImageCreateInfo {
                flags: vulkano::image::ImageCreateFlags::CUBE_COMPATIBLE,
                image_type: ImageType::Dim2d,
                format: vk_format(format),
                extent: [face, face, 1],
                array_layers: 6,
                usage: vulkano::image::ImageUsage::TRANSFER_DST
                    | vulkano::image::ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .expect("failed to create a cube texture");
        // Into the copy's layout and out to the sampler's: a fresh image is
        // `Undefined`, and a copy into it there, or a sample of it left in the
        // copy's layout, is undefined behaviour that happens to work on some
        // drivers.
        let mut recorder = Recorder::new(&self.ctx);
        recorder
            .image_barrier(super::record::to_transfer_dst(
                image.clone(),
                super::record::whole_image(&image),
                vulkano::image::ImageLayout::Undefined,
            ))
            .copy_buffer_to_image(
                vulkano::command_buffer::CopyBufferToImageInfo::buffer_image(
                    staging,
                    image.clone(),
                ),
            )
            .image_barrier(super::record::to_shader_read(
                image.clone(),
                super::record::whole_image(&image),
                vulkano::image::ImageLayout::TransferDstOptimal,
            ));
        recorder.submit_and_wait(&self.ctx);
        let view = ImageView::new(
            image.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Cube,
                ..ImageViewCreateInfo::from_image(&image)
            },
        )
        .expect("failed to create a cube view");
        TextureViewId(self.views.lock().unwrap().insert(view))
    }

    pub fn face_view(&self, texture: TextureViewId, face: u32, level: u32) -> TextureViewId {
        let whole = lookup(&self.views, texture.0, "texture view");
        let image = whole.image().clone();
        let view = ImageView::new(
            image.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Dim2d,
                subresource_range: ImageSubresourceRange {
                    array_layers: face..face + 1,
                    mip_levels: level..level + 1,
                    ..image.subresource_range()
                },
                ..ImageViewCreateInfo::from_image(&image)
            },
        )
        .expect("failed to create a face view");
        TextureViewId(self.views.lock().unwrap().insert(view))
    }

    pub fn record_blocking(&self, record: impl FnOnce(&mut Encoder)) {
        let mut recorder = Recorder::new(&self.ctx);
        {
            let mut encoder = Encoder::Vulkan(VkEncoder::new(&mut recorder, self));
            record(&mut encoder);
        }
        recorder.submit_and_wait(&self.ctx);
    }

    pub fn load_shader(&self, shader: &CompiledShader) -> ShaderModuleId {
        let kept = CompiledShader {
            name: shader.name.clone(),
            stage: shader.stage,
            defines: Vec::new(),
            spirv: shader.spirv.clone(),
            msl: String::new(),
            bindings: CompiledBindings {
                resources: Vec::new(),
                ..shader.bindings.clone()
            },
        };
        ShaderModuleId(self.shaders.lock().unwrap().insert(Arc::new(kept)))
    }

    pub fn destroy_shader(&self, shader: ShaderModuleId) {
        self.shaders
            .lock()
            .unwrap()
            .remove(shader.0, self.frame.load(Ordering::Relaxed));
    }

    pub fn destroy_pipeline(&self, pipeline: PipelineId) {
        self.pipelines
            .lock()
            .unwrap()
            .remove(pipeline.0, self.frame.load(Ordering::Relaxed));
    }

    /// `shader`'s `main` as a fresh module, and the shader's name for errors.
    /// The one way either kind of shader becomes a module.
    fn entry_point(
        &self,
        shader: ShaderRef,
    ) -> Result<(vulkano::shader::EntryPoint, String), String> {
        let loaded: Arc<CompiledShader>;
        let view = match shader {
            ShaderRef::BuiltIn(id) => id.blob().view(),
            ShaderRef::Loaded(id) => {
                loaded = self
                    .shaders
                    .lock()
                    .expect("a pass panicked while using the device")
                    .get(id.0)
                    .cloned()
                    .ok_or_else(|| format!("loaded shader {} was destroyed", id.0))?;
                loaded.view()
            }
        };
        let words = spirv_words(view.spirv).map_err(|e| format!("{}: {e}", view.name))?;
        // SAFETY: vulkano asks for valid SPIR-V. A built-in shader's words are
        // shaderc's output for a manifest entry, which `build.rs` also reflects
        // and translates, failing the build if either cannot parse it. A loaded
        // shader's are the same compiler's output through `orrin-shaderc`, and
        // that they still are exactly that output is the guarantee of whoever
        // made the `CompiledShader`. What is checked here is framing:
        // `spirv_words` refuses anything but a little-endian SPIR-V header and
        // a body its instructions tile, and vulkano then parses every
        // instruction and checks the version and capabilities against the
        // device. Well-formed instructions that mean something invalid are not
        // caught by either, and MoltenVK has been seen to crash on them.
        let module = unsafe {
            ShaderModule::new(self.ctx.device.clone(), ShaderModuleCreateInfo::new(&words))
        }
        .map_err(|e| format!("{}: {}", view.name, reason(e)))?;
        let entry = module
            .entry_point("main")
            .ok_or_else(|| format!("{} has no main", view.name))?;
        Ok((entry, view.name.to_string()))
    }

    /// The persistent pipeline cache, unless a stage was loaded at run time.
    /// A loaded shader is a surface shader's variant, a new module with every
    /// save, while the cache is machine-wide and never pruned — and MoltenVK
    /// compiles everything it holds when it is created, so caching these made
    /// the editor's startup grow with every edit.
    fn cache_for(
        &self,
        stages: &[ShaderRef],
    ) -> Option<Arc<vulkano::pipeline::cache::PipelineCache>> {
        if stages
            .iter()
            .any(|stage| matches!(stage, ShaderRef::Loaded(_)))
        {
            None
        } else {
            self.ctx.pipeline_cache()
        }
    }

    pub fn try_create_graphics_pipeline(
        &self,
        desc: &GraphicsPipelineDesc,
    ) -> Result<PipelineId, String> {
        let device = &self.ctx.device;
        let (vertex, vertex_name) = self.entry_point(desc.vertex)?;
        let (fragment, fragment_name) = self.entry_point(desc.fragment)?;
        let named = |e: &dyn std::fmt::Display| format!("{vertex_name} with {fragment_name}: {e}");
        let stages = [
            PipelineShaderStageCreateInfo::new(vertex.clone()),
            PipelineShaderStageCreateInfo::new(fragment),
        ];
        let vertex_input = if desc.vertex_streams.is_empty() {
            VertexInputState::default()
        } else {
            let descriptions: Vec<_> = desc.vertex_streams.iter().map(vertex_buffer).collect();
            descriptions.definition(&vertex).map_err(|e| {
                format!("{vertex_name}: a vertex stream does not match the shader's inputs: {e}")
            })?
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
            immutable_binding(&mut layout_info, set, binding)
                .map_err(|e| named(&e))?
                .immutable_samplers = vec![lookup(&self.samplers, sampler.0, "sampler")];
        }
        let layout = match lifted {
            Some(layout) => layout,
            None => {
                let mut info = layout_info
                    .into_pipeline_layout_create_info(device.clone())
                    .map_err(|e| named(&format_args!("{e:?}")))?;
                if let Some((source, count)) = desc.layout_prefix_from {
                    // The source's own objects, not a second set derived from the
                    // same declarations: deriving them would give layouts that match
                    // binding for binding and still fail to bind, because an
                    // immutable sampler is compared by identity.
                    let source = lookup(&self.pipelines, source.0, "pipeline");
                    let count = count as usize;
                    if info.set_layouts.len() < count || source.layout().set_layouts().len() < count
                    {
                        return Err(named(&format_args!(
                            "{count} set layouts are borrowed, and the shaders or the source declare fewer"
                        )));
                    }
                    info.set_layouts.splice(
                        ..count,
                        source.layout().set_layouts()[..count].iter().cloned(),
                    );
                }
                PipelineLayout::new(device.clone(), info).map_err(|e| named(&reason(e)))?
            }
        };
        let colors: Vec<_> = desc.color_formats.iter().map(|&f| vk_format(f)).collect();
        let pipeline = GraphicsPipeline::new(
            device.clone(),
            self.cache_for(&[desc.vertex, desc.fragment]),
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
                            DepthCompare::Greater => CompareOp::Greater,
                            DepthCompare::GreaterOrEqual => CompareOp::GreaterOrEqual,
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
                    // Scissor on every pipeline, not only the atlas's: the
                    // overlay clips each mesh with one, and a scissor set on a
                    // pipeline that declares it static is invalid. `set_viewport`
                    // sets the matching one, so a pass that never asks for a
                    // narrower scissor gets its whole viewport.
                    [DynamicState::Viewport, DynamicState::Scissor]
                        .into_iter()
                        .collect()
                },
                subpass: Some(
                    rendering::pipeline_info(&colors, desc.depth_format.map(vk_format)).into(),
                ),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .map_err(|e| named(&reason(e)))?;
        Ok(PipelineId(
            self.pipelines
                .lock()
                .unwrap()
                .insert(PipelineObject::Graphics(pipeline)),
        ))
    }

    /// A descriptor set built outside the RHI, for the frame being recorded.
    pub fn import_set(&self, set: Arc<DescriptorSet>) -> BindGroupId {
        let mut pool = self.bind_groups.lock().unwrap();
        let index = pool.insert(set);
        pool.remove(index, self.frame.load(Ordering::Relaxed));
        BindGroupId(index)
    }

    /// A compute pipeline with samplers baked into its set layout, as
    /// `(set, binding, sampler)`. Needed where a device must not be handed a
    /// written comparison sampler.
    pub fn try_create_compute_pipeline_with_samplers(
        &self,
        desc: &ComputePipelineDesc,
        immutable_samplers: &[(u32, u32, SamplerId)],
    ) -> Result<PipelineId, String> {
        let device = &self.ctx.device;
        let (entry, name) = self.entry_point(desc.shader)?;
        let named = |e: &dyn std::fmt::Display| format!("{name}: {e}");
        let stage = PipelineShaderStageCreateInfo::new(entry);
        let mut layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage]);
        for &(set, binding, sampler) in immutable_samplers {
            immutable_binding(&mut layout_info, set, binding)
                .map_err(|e| named(&e))?
                .immutable_samplers = vec![lookup(&self.samplers, sampler.0, "sampler")];
        }
        let info = layout_info
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| named(&format_args!("{e:?}")))?;
        let layout = PipelineLayout::new(device.clone(), info).map_err(|e| named(&reason(e)))?;
        let pipeline = ComputePipeline::new(
            device.clone(),
            self.cache_for(&[desc.shader]),
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .map_err(|e| named(&reason(e)))?;
        Ok(PipelineId(
            self.pipelines
                .lock()
                .unwrap()
                .insert(PipelineObject::Compute(pipeline)),
        ))
    }

    /// A persistent bind group: built afresh and kept until [`destroy_bind_group`],
    /// for a set whose resources outlive the frame.
    ///
    /// [`destroy_bind_group`]: Self::destroy_bind_group
    pub fn create_persistent_bind_group(
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
        BindGroupId(self.bind_groups.lock().unwrap().insert(set))
    }

    /// A persistent group binding `textures` as the array at binding 0 and
    /// `sampler` at binding 1: the material texture table.
    ///
    /// Padded with the first texture up to the device's array length, because a
    /// descriptor array may only be left partly unwritten where the device allows
    /// it. See [`VkContext::texture_array_len`].
    pub fn create_texture_array_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        textures: &[TextureViewId],
        sampler: SamplerId,
    ) -> BindGroupId {
        let default = textures[0];
        let mut bindings: Vec<Binding> = (0..self.ctx.texture_array_len(textures.len()))
            .map(|index| Binding {
                binding: 0,
                resource: BindingResource::Texture(textures.get(index).copied().unwrap_or(default)),
            })
            .collect();
        bindings.push(Binding {
            binding: 1,
            resource: BindingResource::Sampler(sampler),
        });
        self.create_persistent_bind_group(pipeline, set, &bindings)
    }

    pub fn destroy_bind_group(&self, group: BindGroupId) {
        self.bind_groups
            .lock()
            .unwrap()
            .remove(group.0, self.frame.load(Ordering::Relaxed));
    }

    pub fn destroy_buffer(&self, buffer: BufferId) {
        self.buffers
            .lock()
            .unwrap()
            .remove(buffer.0, self.frame.load(Ordering::Relaxed));
    }

    pub fn create_buffer(&self, desc: BufferDesc) -> BufferId {
        let buffer = Buffer::new_slice::<u8>(
            self.ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: vk_buffer_usage(desc.usage),
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            desc.size,
        )
        .expect("failed to allocate a buffer");
        BufferId(self.buffers.lock().unwrap().insert(buffer.into_bytes()))
    }

    /// A buffer that outlives the frame, holding `bytes` from creation.
    pub fn create_buffer_bytes(&self, desc: BufferDesc, bytes: &[u8]) -> BufferId {
        assert_eq!(
            desc.size,
            bytes.len() as u64,
            "a buffer's size is its initial bytes"
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
            bytes.iter().copied(),
        )
        .expect("failed to allocate a buffer");
        BufferId(self.buffers.lock().unwrap().insert(buffer.into_bytes()))
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

    fn writes(&self, bindings: &[Binding]) -> Vec<WriteDescriptorSet> {
        let mut writes: Vec<WriteDescriptorSet> = Vec::with_capacity(bindings.len());
        let mut index = 0;
        while index < bindings.len() {
            let binding = bindings[index];
            // Consecutive textures at one binding, of one kind, are the elements
            // of an image array, in order.
            let same_kind = |a: &BindingResource, b: &BindingResource| {
                matches!(
                    (a, b),
                    (
                        BindingResource::StorageTexture(_),
                        BindingResource::StorageTexture(_)
                    ) | (BindingResource::Texture(_), BindingResource::Texture(_))
                )
            };
            if let BindingResource::StorageTexture(_) | BindingResource::Texture(_) =
                binding.resource
            {
                let run = bindings[index..]
                    .iter()
                    .take_while(|b| {
                        b.binding == binding.binding && same_kind(&b.resource, &binding.resource)
                    })
                    .count();
                if run > 1 {
                    let views = bindings[index..index + run]
                        .iter()
                        .map(|b| match b.resource {
                            BindingResource::StorageTexture(view)
                            | BindingResource::Texture(view) => {
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
                // The arena's current buffer, so a group made after it grew
                // names the new one — and, being a different buffer, misses
                // the descriptor cache rather than reusing a set over the old.
                BindingResource::Geometry(stream) => WriteDescriptorSet::buffer(
                    binding.binding,
                    self.arena
                        .read()
                        .expect("a pass panicked while using the device")
                        .storage(stream),
                ),
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
        Blend::PremultipliedCoverage => Some(AttachmentBlend {
            src_color_blend_factor: BlendFactor::One,
            src_alpha_blend_factor: BlendFactor::OneMinusDstAlpha,
            dst_alpha_blend_factor: BlendFactor::One,
            ..AttachmentBlend::alpha()
        }),
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
                    VertexFormat::Snorm16x2 => VkFormat::R16G16_SNORM,
                    VertexFormat::Snorm16x4 => VkFormat::R16G16B16A16_SNORM,
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

/// What vulkano refused, rather than its summary that a validation error
/// occurred.
fn reason(error: vulkano::Validated<vulkano::VulkanError>) -> String {
    match error {
        vulkano::Validated::Error(error) => error.to_string(),
        vulkano::Validated::ValidationError(error) => error.to_string(),
    }
}

/// The binding an immutable sampler is baked into, or an error naming it when
/// the shaders do not declare it.
fn immutable_binding(
    layout: &mut PipelineDescriptorSetLayoutCreateInfo,
    set: u32,
    binding: u32,
) -> Result<&mut vulkano::descriptor_set::layout::DescriptorSetLayoutBinding, String> {
    layout
        .set_layouts
        .get_mut(set as usize)
        .and_then(|set_layout| set_layout.bindings.get_mut(&binding))
        .ok_or_else(|| {
            format!(
                "an immutable sampler names set {set} binding {binding}, which no stage declares"
            )
        })
}

const SPIRV_MAGIC: u32 = 0x0723_0203;
const SPIRV_HEADER_WORDS: usize = 5;

/// `bytes` as SPIR-V words, or why they cannot be: not whole words, shorter
/// than a header, not little-endian SPIR-V at all, or a body whose
/// instructions do not tile it.
fn spirv_words(bytes: &[u8]) -> Result<Vec<u32>, String> {
    let (words, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(format!(
            "{} bytes is not a whole number of SPIR-V words",
            bytes.len()
        ));
    }
    if words.len() < SPIRV_HEADER_WORDS {
        return Err(format!(
            "{} words is shorter than a SPIR-V header",
            words.len()
        ));
    }
    let words: Vec<u32> = words.iter().map(|word| u32::from_le_bytes(*word)).collect();
    if words[0] != SPIRV_MAGIC {
        return Err(format!(
            "starts with {:#010x}, not the little-endian SPIR-V magic number",
            words[0]
        ));
    }
    let mut at = SPIRV_HEADER_WORDS;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 || at + count > words.len() {
            return Err(format!(
                "the instruction at word {at} claims {count} words of the {} left",
                words.len() - at
            ));
        }
        at += count;
    }
    Ok(words)
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

    /// The viewport, and a scissor covering exactly it — every pipeline
    /// declares scissor dynamic, so a draw needs one set. A pass that wants a
    /// narrower one sets it after this.
    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.recorder.set_viewport(
            0,
            &[vulkano::pipeline::graphics::viewport::Viewport {
                offset: viewport.origin,
                extent: viewport.extent,
                depth_range: viewport.depth_range[0]..=viewport.depth_range[1],
            }],
        );
        let offset = viewport.origin.map(|o| o.max(0.0).floor() as u32);
        let extent = [0, 1].map(|axis| {
            (viewport.origin[axis] + viewport.extent[axis])
                .ceil()
                .max(0.0) as u32
                - offset[axis]
        });
        self.set_scissor(offset, extent);
    }

    pub fn set_scissor(&mut self, offset: [u32; 2], extent: [u32; 2]) {
        self.recorder.set_scissor(0, &[Scissor { offset, extent }]);
    }

    pub fn set_depth_bias(&mut self, constant: f32, clamp: f32, slope: f32) {
        self.recorder.set_depth_bias(constant, clamp, slope);
    }

    pub fn clear_depth(&mut self, rects: &[DepthRect]) {
        self.recorder.clear_attachments(
            [ClearAttachment::Depth(crate::gfx::depth::SHADOW_CLEAR)],
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

    pub fn copy_buffer(&mut self, src: BufferId, dst: BufferId, regions: &[BufferCopy]) {
        let src = lookup(&self.driver.buffers, src.0, "buffer");
        let dst = lookup(&self.driver.buffers, dst.0, "buffer");
        self.recorder
            .copy_buffer(vulkano::command_buffer::CopyBufferInfo {
                regions: regions
                    .iter()
                    .map(|region| vulkano::command_buffer::BufferCopy {
                        src_offset: region.src_offset,
                        dst_offset: region.dst_offset,
                        size: region.size,
                        ..Default::default()
                    })
                    .collect(),
                ..vulkano::command_buffer::CopyBufferInfo::buffers(src, dst)
            });
    }

    pub fn bind_index_buffer(&mut self, buffer: BufferId) {
        let indices = lookup(&self.driver.buffers, buffer.0, "buffer").reinterpret::<[u32]>();
        self.recorder.bind_index_buffer(indices);
    }

    pub fn write_texture(
        &mut self,
        texture: TextureViewId,
        offset: [u32; 2],
        extent: [u32; 2],
        staging: BufferId,
        initial: bool,
    ) {
        use vulkano::image::ImageLayout;
        let image = lookup(&self.driver.views, texture.0, "texture view")
            .image()
            .clone();
        let staging = lookup(&self.driver.buffers, staging.0, "buffer");
        let range = super::record::whole_image(&image);
        // Before each copy, a wait on earlier fragment reads: a previous frame's
        // overlay may still be sampling a texture this one patches.
        self.recorder
            .image_barrier(vulkano::sync::ImageMemoryBarrier {
                src_stages: vulkano::sync::PipelineStages::FRAGMENT_SHADER,
                ..super::record::to_transfer_dst(
                    image.clone(),
                    range.clone(),
                    if initial {
                        ImageLayout::Undefined
                    } else {
                        ImageLayout::ShaderReadOnlyOptimal
                    },
                )
            })
            .copy_buffer_to_image(vulkano::command_buffer::CopyBufferToImageInfo {
                regions: [vulkano::command_buffer::BufferImageCopy {
                    image_offset: [offset[0], offset[1], 0],
                    image_extent: [extent[0], extent[1], 1],
                    image_subresource: vulkano::image::ImageSubresourceLayers {
                        aspects: vulkano::image::ImageAspects::COLOR,
                        mip_level: 0,
                        array_layers: 0..1,
                    },
                    ..Default::default()
                }]
                .into(),
                ..vulkano::command_buffer::CopyBufferToImageInfo::buffer_image(
                    staging,
                    image.clone(),
                )
            })
            .image_barrier(super::record::to_shader_read(
                image,
                range,
                ImageLayout::TransferDstOptimal,
            ));
    }

    /// Take a swapchain image that arrives presentable into rendering, keeping
    /// what is in it. The semaphore the submission waits on has already made the
    /// frame's writes visible.
    pub fn open_swapchain(&mut self, target: TextureViewId) {
        let image = lookup(&self.driver.views, target.0, "texture view")
            .image()
            .clone();
        self.recorder.image_barrier(present_to_color(image));
    }

    /// Hand a swapchain image back presentable.
    pub fn close_swapchain(&mut self, target: TextureViewId) {
        let image = lookup(&self.driver.views, target.0, "texture view")
            .image()
            .clone();
        self.recorder.image_barrier(color_to_present(image));
    }

    pub fn begin_pass(&mut self, targets: &RenderTargets) {
        use vulkano::command_buffer::{
            RenderingAttachmentInfo, RenderingAttachmentResolveInfo, RenderingInfo,
        };
        use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
        let load = |op: LoadOp| match op {
            LoadOp::Load => AttachmentLoadOp::Load,
            LoadOp::Clear => AttachmentLoadOp::Clear,
            LoadOp::DontCare => AttachmentLoadOp::DontCare,
        };
        let store = |op: StoreOp| match op {
            StoreOp::Store => AttachmentStoreOp::Store,
            StoreOp::DontCare => AttachmentStoreOp::DontCare,
        };
        let view = |id| lookup(&self.driver.views, id, "texture view");
        let colors = targets
            .colors
            .iter()
            .map(|color| {
                Some(RenderingAttachmentInfo {
                    load_op: load(color.load),
                    store_op: store(color.store),
                    clear_value: (color.load == LoadOp::Clear).then(|| color.clear.into()),
                    resolve_info: color
                        .resolve
                        .map(|resolve| RenderingAttachmentResolveInfo::image_view(view(resolve.0))),
                    ..RenderingAttachmentInfo::image_view(view(color.target.0))
                })
            })
            .collect();
        let depth = targets.depth.map(|depth| RenderingAttachmentInfo {
            // `DepthStencilReadOnlyOptimal` is the layout the barrier plan says a
            // borrowed depth is in, so stating it keeps this attachment from
            // transitioning an image nobody wrote.
            image_layout: if depth.read_only {
                vulkano::image::ImageLayout::DepthStencilReadOnlyOptimal
            } else {
                vulkano::image::ImageLayout::DepthStencilAttachmentOptimal
            },
            load_op: load(depth.load),
            store_op: store(depth.store),
            clear_value: (depth.load == LoadOp::Clear).then(|| depth.clear.into()),
            ..RenderingAttachmentInfo::image_view(view(depth.target.0))
        });
        self.recorder.begin_rendering(RenderingInfo {
            color_attachments: colors,
            depth_attachment: depth,
            ..Default::default()
        });
    }

    pub fn begin_rendering_fresh(&mut self, target: TextureViewId) {
        use vulkano::command_buffer::{RenderingAttachmentInfo, RenderingInfo};
        use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
        let view = lookup(&self.driver.views, target.0, "texture view");
        let image = view.image().clone();
        let range = view.subresource_range().clone();
        let level = range.mip_levels.start;
        // Fresh from creation, so there is nothing to preserve.
        self.recorder
            .image_barrier(record::to_color_attachment(image.clone(), range));
        let side = |axis: usize| (image.extent()[axis] >> level).max(1);
        self.recorder.begin_rendering(RenderingInfo {
            color_attachments: vec![Some(RenderingAttachmentInfo {
                load_op: AttachmentLoadOp::DontCare,
                store_op: AttachmentStoreOp::Store,
                ..RenderingAttachmentInfo::image_view(view)
            })],
            // Stated, not derived: left to itself this is taken from the view's
            // image, which is level 0's, and every level below it is then
            // smaller than the area claiming to cover it.
            render_area_extent: [side(0), side(1)],
            ..Default::default()
        });
    }

    pub fn generate_cube_mips(&mut self, texture: TextureViewId) {
        let image = lookup(&self.driver.views, texture.0, "texture view")
            .image()
            .clone();
        generate_cube_mips(self.recorder, &image);
    }

    pub fn publish_render_target(&mut self, texture: TextureViewId) {
        let image = lookup(&self.driver.views, texture.0, "texture view")
            .image()
            .clone();
        let levels = record::levels(&image, 0..image.mip_levels());
        self.recorder
            .image_barrier(record::color_to_shader_read(image, levels));
    }

    /// Open a render pass over `target`, loading what it holds.
    pub fn begin_rendering(&mut self, target: TextureViewId) {
        let view = lookup(&self.driver.views, target.0, "texture view");
        begin(self.recorder, &view);
    }

    /// Vulkan times passes with its own timestamp pairs around each body.
    pub fn set_pass_name(&mut self, _: &'static str) {}

    pub fn end_rendering(&mut self) {
        self.recorder.end_rendering();
    }

    pub fn draw_indexed_indirect(&mut self, commands: BufferId, range: std::ops::Range<u32>) {
        let commands = lookup(&self.driver.buffers, commands.0, "buffer")
            .reinterpret::<[vulkano::command_buffer::DrawIndexedIndirectCommand]>()
            .slice(range.start as u64..range.end as u64);
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

/// Into rendering, keeping what the frame drew. The semaphore the submission
/// waits on has already made the frame's writes visible.
fn present_to_color(image: Arc<Image>) -> vulkano::sync::ImageMemoryBarrier {
    vulkano::sync::ImageMemoryBarrier {
        src_stages: vulkano::sync::PipelineStages::COLOR_ATTACHMENT_OUTPUT,
        src_access: vulkano::sync::AccessFlags::empty(),
        dst_stages: vulkano::sync::PipelineStages::COLOR_ATTACHMENT_OUTPUT,
        dst_access: vulkano::sync::AccessFlags::COLOR_ATTACHMENT_READ
            | vulkano::sync::AccessFlags::COLOR_ATTACHMENT_WRITE,
        old_layout: vulkano::image::ImageLayout::PresentSrc,
        new_layout: vulkano::image::ImageLayout::ColorAttachmentOptimal,
        subresource_range: record::whole_image(&image),
        ..vulkano::sync::ImageMemoryBarrier::image(image)
    }
}

pub(super) fn color_to_present(image: Arc<Image>) -> vulkano::sync::ImageMemoryBarrier {
    vulkano::sync::ImageMemoryBarrier {
        src_stages: vulkano::sync::PipelineStages::COLOR_ATTACHMENT_OUTPUT,
        src_access: vulkano::sync::AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stages: vulkano::sync::PipelineStages::BOTTOM_OF_PIPE,
        dst_access: vulkano::sync::AccessFlags::empty(),
        old_layout: vulkano::image::ImageLayout::ColorAttachmentOptimal,
        new_layout: vulkano::image::ImageLayout::PresentSrc,
        subresource_range: record::whole_image(&image),
        ..vulkano::sync::ImageMemoryBarrier::image(image)
    }
}

fn begin(recorder: &mut Recorder, target: &Arc<ImageView>) {
    recorder.begin_rendering(vulkano::command_buffer::RenderingInfo {
        color_attachments: vec![Some(vulkano::command_buffer::RenderingAttachmentInfo {
            load_op: vulkano::render_pass::AttachmentLoadOp::Load,
            store_op: vulkano::render_pass::AttachmentStoreOp::Store,
            ..vulkano::command_buffer::RenderingAttachmentInfo::image_view(target.clone())
        })],
        ..Default::default()
    });
}

/// Halve each level into the next with a linear blit, all six layers at once.
///
/// Not the same job as the prefilter below: this is a plain box pyramid, and it
/// exists so the prefilter can *read* from a level matched to how densely its
/// samples land. `texture.rs` cannot be reused because its chain walks layer 0
/// alone.
fn generate_cube_mips(builder: &mut Recorder, image: &Arc<Image>) {
    let base = image.extent();
    let aspects = image.subresource_layers().aspects;

    // Level 0 was rendered, not copied, so it enters as a colour attachment;
    // every level below it has never been touched.
    builder
        .image_barrier(record::color_to_transfer_src(
            image.clone(),
            record::levels(image, 0..1),
        ))
        .image_barrier(record::to_transfer_dst(
            image.clone(),
            record::levels(image, 1..image.mip_levels()),
            vulkano::image::ImageLayout::Undefined,
        ));

    for level in 1..image.mip_levels() {
        let region = vulkano::command_buffer::ImageBlit {
            src_subresource: vulkano::image::ImageSubresourceLayers {
                aspects,
                mip_level: level - 1,
                array_layers: 0..6,
            },
            src_offsets: [
                [0, 0, 0],
                vulkano::image::mip_level_extent(base, level - 1).unwrap(),
            ],
            dst_subresource: vulkano::image::ImageSubresourceLayers {
                aspects,
                mip_level: level,
                array_layers: 0..6,
            },
            dst_offsets: [
                [0, 0, 0],
                vulkano::image::mip_level_extent(base, level).unwrap(),
            ],
            ..Default::default()
        };

        // Level `level - 1` must finish being written before it is read, and be
        // in the source layout when it is. Level 0 already is, from the barrier
        // above; the rest arrive here as this loop's previous destination.
        if level > 1 {
            builder.image_barrier(record::transfer_dst_to_src(
                image.clone(),
                record::levels(image, level - 1..level),
            ));
        }

        builder.blit_image(vulkano::command_buffer::BlitImageInfo {
            regions: vec![region].into(),
            filter: Filter::Linear,
            ..vulkano::command_buffer::BlitImageInfo::images(image.clone(), image.clone())
        });
    }

    // What the chain left behind: every level but the last was read as a blit
    // source, the last was only ever written. The prefilter samples the whole
    // pyramid as one cube, so both runs have to reach the same layout.
    let last = image.mip_levels() - 1;
    builder
        .image_barrier(record::to_shader_read(
            image.clone(),
            record::levels(image, 0..last),
            vulkano::image::ImageLayout::TransferSrcOptimal,
        ))
        .image_barrier(record::to_shader_read(
            image.clone(),
            record::levels(image, last..image.mip_levels()),
            vulkano::image::ImageLayout::TransferDstOptimal,
        ));
}

#[cfg(test)]
pub(crate) mod tests {
    use vulkano::pipeline::graphics::vertex_input::Vertex as _;

    use super::*;
    use crate::gfx::format::SampleCount;
    use crate::gfx::graph::ImageUsage;
    use crate::gfx::rhi::Device;
    use crate::gfx::shaders::ShaderId;
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

    #[test]
    fn every_built_in_shaders_bytes_pass_the_spirv_check() {
        for blob in crate::gfx::shaders::SHADERS {
            let words = spirv_words(blob.spirv).unwrap_or_else(|e| panic!("{}: {e}", blob.name));
            assert_eq!(words.len() * 4, blob.spirv.len());
        }
    }

    /// What reaches the `unsafe` module creation has to at least be a whole
    /// SPIR-V word stream; a loaded shader's bytes come from a file.
    #[test]
    fn bytes_that_cannot_be_spirv_are_refused_before_the_driver_sees_them() {
        let spirv = ShaderId::FullscreenVert.blob().spirv;
        assert!(spirv_words(&[]).is_err(), "an empty stream");
        assert!(
            spirv_words(&spirv[..spirv.len() - 1]).is_err(),
            "a stream cut mid-word"
        );
        assert!(spirv_words(&spirv[..8]).is_err(), "shorter than a header");
        let mut no_magic = spirv.to_vec();
        no_magic[0] ^= 0xff;
        assert!(spirv_words(&no_magic).is_err(), "no magic number");
        let big_endian: Vec<u8> = spirv
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|word| word.iter().rev().copied())
            .collect();
        assert!(spirv_words(&big_endian).is_err(), "the wrong byte order");
    }

    /// vulkano asserts every instruction is at least one word long, so a
    /// zeroed word count would panic inside it rather than return an error.
    #[test]
    fn a_body_whose_instructions_do_not_frame_is_refused() {
        let spirv = ShaderId::FullscreenVert.blob().spirv;
        let first = SPIRV_HEADER_WORDS * 4;
        let mut zero_count = spirv.to_vec();
        zero_count[first + 2..first + 4].fill(0);
        assert!(
            spirv_words(&zero_count).is_err(),
            "an instruction of no words"
        );
        assert!(
            spirv_words(&spirv[..first + 4]).is_err(),
            "an instruction that runs past the end"
        );
    }

    // Loaded shaders on a device. The bodies take the device and a readback,
    // so the Metal driver's tests run exactly these against its own device.

    const SIDE: u32 = 8;
    const FORMATS: [Format; 1] = [Format::Rgba8Unorm];
    /// What `outline_mask.frag` writes: its push constant's second vec4.
    const PUSHED: [f32; 4] = [1.0, 0.5, 0.25, 1.0];

    fn vulkan() -> Device {
        let instance = crate::capture::headless_instance();
        let ctx = Arc::new(VkContext::new(&instance, None));
        Device::Vulkan(VkDriver::new(ctx, 2))
    }

    fn read_vulkan(device: &Device, view: TextureViewId) -> Vec<u8> {
        let driver = device.vulkan();
        let image = driver.vk_view(view).image().clone();
        let readback = Buffer::new_slice::<u8>(
            driver.ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: vulkano::buffer::BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            u64::from(SIDE * SIDE * 4),
        )
        .unwrap();
        let mut recorder = Recorder::new(&driver.ctx);
        recorder
            .image_barrier(record::color_to_transfer_src(
                image.clone(),
                record::levels(&image, 0..1),
            ))
            .copy_image_to_buffer(
                vulkano::command_buffer::CopyImageToBufferInfo::image_buffer(
                    image,
                    readback.clone(),
                ),
            );
        recorder.submit_and_wait(&driver.ctx);
        readback.read().unwrap().to_vec()
    }

    pub(crate) fn loaded(device: &Device, shader: ShaderId) -> ShaderModuleId {
        device.load_shader(&CompiledShader::from(shader.blob()))
    }

    fn target(device: &Device) -> TextureViewId {
        device.create_texture(&TextureDesc {
            format: Format::Rgba8Unorm,
            dimension: TextureDimension::D2,
            extent: [SIDE, SIDE, 1],
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::Sample1,
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
            memoryless: false,
        })
    }

    fn fullscreen(
        device: &Device,
        vertex: impl Into<ShaderRef>,
        fragment: impl Into<ShaderRef>,
    ) -> Result<PipelineId, String> {
        device.try_create_graphics_pipeline(&GraphicsPipelineDesc::fullscreen(
            vertex, fragment, &FORMATS,
        ))
    }

    fn draw(
        device: &Device,
        pipeline: PipelineId,
        push: &[u32],
        group: Option<BindGroupId>,
    ) -> TextureViewId {
        let output = target(device);
        device.record_blocking(|encoder| {
            encoder.begin_rendering_fresh(output);
            encoder.bind_pipeline(pipeline);
            encoder.set_viewport(Viewport {
                origin: [0.0; 2],
                extent: [SIDE as f32; 2],
                depth_range: [0.0, 1.0],
            });
            if let Some(group) = group {
                encoder.bind_group(0, group);
            }
            encoder.push_constants(push);
            encoder.draw(3, 1, 0, 0);
            encoder.end_rendering();
        });
        output
    }

    fn pixels(
        device: &Device,
        read: &dyn Fn(&Device, TextureViewId) -> Vec<u8>,
        pipeline: PipelineId,
        push: &[u32],
        group: Option<BindGroupId>,
    ) -> Vec<u8> {
        read(device, draw(device, pipeline, push, group))
    }

    fn mask_push() -> [u32; 20] {
        let mut push = [0; 20];
        for (word, value) in push[16..].iter_mut().zip(PUSHED) {
            *word = value.to_bits();
        }
        push
    }

    /// One selected texel in the middle of the mask, so `outline.frag` draws a
    /// ring around it.
    fn outline_inputs(device: &Device) -> (TextureViewId, SamplerId) {
        let mut mask = vec![0u8; (SIDE * SIDE * 4) as usize];
        let middle = ((SIDE / 2 * SIDE + SIDE / 2) * 4) as usize;
        mask[middle..middle + 4].copy_from_slice(&[255, 0, 0, 255]);
        let mask = device.create_texture_2d(&mask, [SIDE, SIDE], Format::Rgba8Unorm);
        let sampler = device.create_sampler(SamplerDesc {
            linear: false,
            mipmap_linear: false,
            repeat: false,
            compare: false,
            anisotropy: None,
            max_lod: None,
        });
        (mask, sampler)
    }

    fn outline_group(
        device: &Device,
        pipeline: PipelineId,
        mask: TextureViewId,
        sampler: SamplerId,
    ) -> BindGroupId {
        device.create_bind_group(
            pipeline,
            0,
            &[Binding {
                binding: 0,
                resource: BindingResource::SampledTexture {
                    view: mask,
                    sampler,
                },
            }],
        )
    }

    fn outline_push() -> [u32; 9] {
        let mut push = [0; 9];
        for (word, value) in push
            .iter_mut()
            .zip([1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        {
            *word = value.to_bits();
        }
        push[8] = 1;
        push
    }

    /// A built-in shader's own bytes, loaded, draw what the built-in draws:
    /// as the fragment stage beside a built-in vertex stage, and as both.
    pub(crate) fn a_loaded_shader_draws_what_its_built_in_twin_draws(
        device: &Device,
        read: &dyn Fn(&Device, TextureViewId) -> Vec<u8>,
    ) {
        device.begin_frame(0);
        let built_in = fullscreen(device, ShaderId::FullscreenVert, ShaderId::OutlineMaskFrag)
            .expect("the built-in pipeline");
        let fragment = loaded(device, ShaderId::OutlineMaskFrag);
        let vertex = loaded(device, ShaderId::FullscreenVert);
        let half = fullscreen(device, ShaderId::FullscreenVert, fragment)
            .expect("a loaded fragment stage");
        let whole = fullscreen(device, vertex, fragment).expect("two loaded stages");

        let expected = pixels(device, read, built_in, &mask_push(), None);
        for (texel, want) in expected[..4].iter().zip([255u8, 128, 64, 255]) {
            assert!(texel.abs_diff(want) <= 1, "{:?}", &expected[..4]);
        }
        assert_eq!(pixels(device, read, half, &mask_push(), None), expected);
        assert_eq!(pixels(device, read, whole, &mask_push(), None), expected);

        // And one that reads a descriptor set, through a loaded shader's own
        // set layout or argument encoder.
        let (mask, sampler) = outline_inputs(device);
        let built_in = fullscreen(device, ShaderId::FullscreenVert, ShaderId::OutlineFrag).unwrap();
        let fragment = loaded(device, ShaderId::OutlineFrag);
        let outline = fullscreen(device, ShaderId::FullscreenVert, fragment).unwrap();
        let expected = pixels(
            device,
            read,
            built_in,
            &outline_push(),
            Some(outline_group(device, built_in, mask, sampler)),
        );
        assert!(
            expected.chunks(4).any(|texel| texel[0] > 0),
            "the outline drew nothing; the test is not looking at one"
        );
        let got = pixels(
            device,
            read,
            outline,
            &outline_push(),
            Some(outline_group(device, outline, mask, sampler)),
        );
        assert_eq!(got, expected);
    }

    /// `broken` is refused by whatever the backend asks to build it, and the
    /// error names it.
    pub(crate) fn a_broken_loaded_shader_is_an_error(device: &Device, broken: &CompiledShader) {
        device.begin_frame(0);
        let fragment = device.load_shader(broken);
        let error = fullscreen(device, ShaderId::FullscreenVert, fragment)
            .expect_err("a broken shader built a pipeline");
        assert!(error.contains(&broken.name), "{error}");
        println!("refused: {error}");
    }

    /// Retiring a pipeline and a loaded shader frees their slots after the
    /// frames in flight, and nothing built from them reads a recycled slot.
    pub(crate) fn retired_pipelines_and_shaders_are_recycled_safely(
        device: &Device,
        read: &dyn Fn(&Device, TextureViewId) -> Vec<u8>,
    ) {
        let mut frame = 0;
        let mut next_frame = || {
            frame += 1;
            device.begin_frame(frame);
        };
        device.begin_frame(0);
        let (mask, sampler) = outline_inputs(device);
        let built_in = fullscreen(device, ShaderId::FullscreenVert, ShaderId::OutlineFrag).unwrap();
        let expected = pixels(
            device,
            read,
            built_in,
            &outline_push(),
            Some(outline_group(device, built_in, mask, sampler)),
        );
        let outline_shader = loaded(device, ShaderId::OutlineFrag);
        let outline = fullscreen(device, ShaderId::FullscreenVert, outline_shader).unwrap();

        // The pipeline outlives its shader's handle, and the slot that handle
        // named going to a different shader changes nothing about it.
        device.destroy_shader(outline_shader);
        for _ in 0..4 {
            next_frame();
            draw(
                device,
                outline,
                &outline_push(),
                Some(outline_group(device, outline, mask, sampler)),
            );
        }
        let mask_shader = loaded(device, ShaderId::OutlineMaskFrag);
        assert_eq!(mask_shader, outline_shader, "the retired slot is reused");
        let mask_pipeline = fullscreen(device, ShaderId::FullscreenVert, mask_shader).unwrap();
        let got = pixels(
            device,
            read,
            outline,
            &outline_push(),
            Some(outline_group(device, outline, mask, sampler)),
        );
        assert_eq!(got, expected);
        let masked = pixels(device, read, mask_pipeline, &mask_push(), None);
        assert!(masked[0] > 250, "{:?}", &masked[..4]);

        device.destroy_pipeline(outline);
        device.destroy_pipeline(mask_pipeline);
        device.destroy_shader(mask_shader);
        for _ in 0..4 {
            next_frame();
            draw(
                device,
                built_in,
                &outline_push(),
                Some(outline_group(device, built_in, mask, sampler)),
            );
        }
        assert!(
            fullscreen(device, ShaderId::FullscreenVert, mask_shader).is_err(),
            "a retired shader handle is refused, not followed"
        );
        let again = fullscreen(
            device,
            ShaderId::FullscreenVert,
            loaded(device, ShaderId::OutlineMaskFrag),
        )
        .unwrap();
        assert!(
            again == outline || again == mask_pipeline,
            "a retired pipeline slot is reused"
        );
        assert_eq!(pixels(device, read, again, &mask_push(), None), masked);
    }

    #[test]
    #[ignore = "needs a Vulkan device"]
    fn a_loaded_shader_draws_what_its_built_in_twin_draws_on_vulkan() {
        a_loaded_shader_draws_what_its_built_in_twin_draws(&vulkan(), &read_vulkan);
    }

    /// The header and the framing are intact, so the check ahead of the
    /// `unsafe` passes it on; the `OpReturn` before the last `OpFunctionEnd`
    /// has become an opcode SPIR-V does not have, which vulkano's parser
    /// refuses. A body that parses and is still invalid — the same module with
    /// those two instructions cut off — gets past vulkano to MoltenVK, whose
    /// translator reports "Function was not terminated" and then crashes the
    /// process, so that case cannot be a test here.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn an_unknown_opcode_in_a_spirv_body_is_an_error_on_vulkan() {
        let mut broken = CompiledShader::from(ShaderId::OutlineMaskFrag.blob());
        let at = broken.spirv.len() - 8;
        assert_eq!(
            broken.spirv[at..],
            [0x0001_00fd_u32, 0x0001_0038]
                .map(u32::to_le_bytes)
                .concat(),
            "the last two instructions are OpReturn and OpFunctionEnd"
        );
        broken.spirv[at..at + 4].copy_from_slice(&0x0001_ffff_u32.to_le_bytes());
        spirv_words(&broken.spirv).expect("the framing is intact");
        a_broken_loaded_shader_is_an_error(&vulkan(), &broken);
    }

    /// A pipeline with a stage loaded at run time — a surface shader's twin —
    /// adds nothing to the persistent pipeline cache. Every save of a shader is
    /// a new module, the cache is machine-wide and never pruned, and MoltenVK
    /// compiles every shader the file holds when the cache is created: before
    /// this, a cache grown by sessions of shader edits held the editor's window
    /// for over a minute at startup.
    #[cfg(feature = "shader-compiler")]
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn a_loaded_shader_leaves_the_persistent_pipeline_cache_alone_on_vulkan() {
        let device = vulkan();
        let cache = device
            .vulkan()
            .ctx
            .pipeline_cache()
            .expect("a pipeline cache");
        // A constant no earlier run wrote, so no earlier run cached this shader.
        let salt = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let source = format!(
            "#version 450\nlayout(location = 0) out vec4 colour;\n\
             void main() {{ colour = vec4(float({salt}u) * 1e-9); }}\n"
        );
        let compiled = crate::gfx::shaders::compile(
            &source,
            crate::gfx::shaders::Stage::Fragment,
            "salted.frag",
            &[],
            &[],
        )
        .unwrap();
        let fragment = device.load_shader(&compiled);
        let before = cache.get_data().unwrap().len();
        fullscreen(&device, ShaderId::FullscreenVert, fragment).expect("a pipeline");
        assert_eq!(cache.get_data().unwrap().len(), before);
    }

    #[test]
    #[ignore = "needs a Vulkan device"]
    fn retired_pipelines_and_shaders_are_recycled_safely_on_vulkan() {
        retired_pipelines_and_shaders_are_recycled_safely(&vulkan(), &read_vulkan);
    }
}
