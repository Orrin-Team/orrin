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
use std::sync::{Arc, Mutex};

use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::image::sampler::{
    Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
};
use vulkano::image::view::ImageView;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};

use super::context::VkContext;
use super::convert::vk_format;
use super::record::Recorder;
use super::rendering;
use crate::gfx::rhi::{
    BindGroupId, Binding, BindingResource, BufferId, GraphicsPipelineDesc, PipelineId, Pool,
    SamplerDesc, SamplerId, TextureViewId, Viewport,
};
use crate::gfx::shaders::ShaderId;

pub struct VkDriver {
    ctx: Arc<VkContext>,
    frame: AtomicU64,
    views: Mutex<Pool<Arc<ImageView>>>,
    buffers: Mutex<Pool<Subbuffer<[u8]>>>,
    samplers: Mutex<Pool<Arc<Sampler>>>,
    pipelines: Mutex<Pool<Arc<GraphicsPipeline>>>,
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
        Self {
            ctx,
            frame: AtomicU64::new(0),
            views: Mutex::new(Pool::new(frames_in_flight)),
            buffers: Mutex::new(Pool::new(frames_in_flight)),
            samplers: Mutex::new(Pool::new(frames_in_flight)),
            pipelines: Mutex::new(Pool::new(frames_in_flight)),
            bind_groups: Mutex::new(Pool::new(frames_in_flight)),
        }
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
        let info = SamplerCreateInfo {
            mag_filter: filter,
            min_filter: filter,
            mipmap_mode: if desc.mipmap_linear {
                SamplerMipmapMode::Linear
            } else {
                SamplerMipmapMode::Nearest
            },
            address_mode: [address; 3],
            lod: if desc.mipmap_linear {
                0.0..=vulkano::image::sampler::LOD_CLAMP_NONE
            } else {
                0.0..=1.0
            },
            anisotropy: desc.anisotropy,
            ..Default::default()
        };
        let sampler =
            Sampler::new(self.ctx.device.clone(), info).expect("failed to create a sampler");
        SamplerId(self.samplers.lock().unwrap().insert(sampler))
    }

    pub fn create_graphics_pipeline(&self, desc: &GraphicsPipelineDesc) -> PipelineId {
        let device = &self.ctx.device;
        let stages = [
            PipelineShaderStageCreateInfo::new(entry_point(&self.ctx, desc.vertex)),
            PipelineShaderStageCreateInfo::new(entry_point(&self.ctx, desc.fragment)),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let colors: Vec<_> = desc.color_formats.iter().map(|&f| vk_format(f)).collect();
        let pipeline = GraphicsPipeline::new(
            device.clone(),
            self.ctx.pipeline_cache(),
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(VertexInputState::default()),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                depth_stencil_state: None,
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    colors.len() as u32,
                    ColorBlendAttachmentState::default(),
                )),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(
                    rendering::pipeline_info(&colors, desc.depth_format.map(vk_format)).into(),
                ),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap();
        PipelineId(self.pipelines.lock().unwrap().insert(pipeline))
    }

    pub fn create_bind_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
    ) -> BindGroupId {
        let pipeline = lookup(&self.pipelines, pipeline.0, "pipeline");
        let writes: Vec<_> = bindings
            .iter()
            .map(|binding| match binding.resource {
                BindingResource::SampledTexture { view, sampler } => {
                    WriteDescriptorSet::image_view_sampler(
                        binding.binding,
                        lookup(&self.views, view.0, "texture view"),
                        lookup(&self.samplers, sampler.0, "sampler"),
                    )
                }
                BindingResource::StorageBuffer(buffer) => WriteDescriptorSet::buffer(
                    binding.binding,
                    lookup(&self.buffers, buffer.0, "buffer"),
                ),
            })
            .collect();
        let set = self.ctx.descriptor_sets.get(
            &self.ctx.descriptor_set_allocator,
            &pipeline.layout().set_layouts()[set as usize],
            writes,
        );
        let mut pool = self.bind_groups.lock().unwrap();
        let index = pool.insert(set);
        pool.remove(index, self.frame.load(Ordering::Relaxed));
        BindGroupId(index)
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
    bound: Option<Arc<GraphicsPipeline>>,
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

    pub fn bind_pipeline(&mut self, pipeline: PipelineId) {
        let pipeline = lookup(&self.driver.pipelines, pipeline.0, "pipeline");
        self.recorder.bind_pipeline_graphics(&pipeline);
        self.bound = Some(pipeline);
    }

    pub fn bind_group(&mut self, set: u32, group: BindGroupId) {
        let group = lookup(&self.driver.bind_groups, group.0, "bind group");
        let layout = self.layout().clone();
        self.recorder
            .bind_descriptor_sets(PipelineBindPoint::Graphics, &layout, set, &[group]);
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

    pub fn draw(&mut self, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
        self.recorder
            .draw(vertices, instances, first_vertex, first_instance);
    }
}
