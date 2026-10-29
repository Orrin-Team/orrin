//! The native Metal driver, beside the Vulkan one behind the same RHI.
//!
//! Brought up in the order `docs/metal-backend-plan.md` M4 gives: the device and
//! its capability gate first, then resources, then pipelines from the embedded
//! MSL, then the encoder. A method that is still `unimplemented!` names itself
//! in the panic, so a pass that reaches one says which operation it needs next.

mod convert;
mod resources;

use std::marker::PhantomData;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLArgumentBuffersTier, MTLBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLGPUFamily, MTLSamplerState, MTLTexture,
};

use crate::geom::Aabb;
use crate::gfx::Vertex;
use crate::gfx::format::Format;
use crate::gfx::rhi::{
    BindGroupId, Binding, BufferCopy, BufferId, ComputePipelineDesc, DepthRect, Encoder,
    GraphicsPipelineDesc, MeshSpan, PipelineId, Pool, SamplerId, TextureViewId, Viewport,
};

use resources::Arena;

/// Frames the CPU may record ahead of the GPU, which is how many sets of
/// per-frame arenas there are and how late a retired id may be freed.
pub(crate) const FRAME_SLOTS: u64 = 2;

/// A Metal object, held across threads.
///
/// SAFETY: Metal documents devices, queues, buffers, textures and sampler
/// states as safe to use from any thread; only command encoders are not, and an
/// encoder is never wrapped in this.
pub(crate) struct Shared<T>(pub(crate) T);
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}

/// A buffer id's target: a whole buffer, or a range of an arena page.
pub(crate) struct Buf {
    pub(crate) buffer: Shared<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub(crate) offset: u64,
    pub(crate) len: u64,
}

pub struct MtlDriver {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    frame: AtomicU64,
    textures: Mutex<Pool<Shared<Retained<ProtocolObject<dyn MTLTexture>>>>>,
    buffers: Mutex<Pool<Buf>>,
    samplers: Mutex<Pool<Shared<Retained<ProtocolObject<dyn MTLSamplerState>>>>>,
    /// Per-frame data, one arena per kind per frame slot: vertices, uniforms,
    /// storage, indices, staging.
    arenas: Mutex<[[Arena; FRAME_SLOTS as usize]; 5]>,
}

pub struct MtlEncoder<'a> {
    _driver: PhantomData<&'a MtlDriver>,
}

impl MtlDriver {
    /// The system's Metal device, or why this machine cannot be the Metal
    /// backend: it needs Apple GPU family 7 (M1 and later) and tier 2 argument
    /// buffers, and anything less runs Vulkan through MoltenVK instead.
    pub fn new() -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        if !device.supportsFamily(MTLGPUFamily::Apple7) {
            return Err(format!(
                "{} is not Apple GPU family 7 or later",
                device.name()
            ));
        }
        if device.argumentBuffersSupport() != MTLArgumentBuffersTier::Tier2 {
            return Err(format!("{} lacks tier 2 argument buffers", device.name()));
        }
        let queue = device
            .newCommandQueue()
            .ok_or("the device gave no command queue")?;
        Ok(Self {
            device,
            queue,
            frame: AtomicU64::new(0),
            textures: Mutex::new(Pool::new(FRAME_SLOTS)),
            buffers: Mutex::new(Pool::new(FRAME_SLOTS)),
            samplers: Mutex::new(Pool::new(FRAME_SLOTS)),
            arenas: Mutex::new(std::array::from_fn(|_| {
                std::array::from_fn(|_| Arena::new())
            })),
        })
    }

    /// The device's name, for the run banner.
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    /// Frees whatever `frame - FRAME_SLOTS` retired and reuses the arenas that
    /// frame wrote.
    pub fn begin_frame(&self, frame: u64) {
        self.frame.store(frame, Ordering::Relaxed);
        self.textures.lock().unwrap().collect(frame);
        self.buffers.lock().unwrap().collect(frame);
        self.samplers.lock().unwrap().collect(frame);
        let slot = (frame % FRAME_SLOTS) as usize;
        for kind in self.arenas.lock().unwrap().iter_mut() {
            kind[slot].reset();
        }
    }
}

#[allow(
    unused_variables,
    reason = "operations land one at a time; a stub names itself in its panic"
)]
impl MtlDriver {
    pub fn upload_mesh(&self, vertices: &[Vertex], indices: &[u32]) -> (MeshSpan, Aabb) {
        unimplemented!("metal: upload_mesh")
    }

    pub fn create_graphics_pipeline(&self, desc: &GraphicsPipelineDesc) -> PipelineId {
        unimplemented!("metal: create_graphics_pipeline")
    }

    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> PipelineId {
        unimplemented!("metal: create_compute_pipeline")
    }

    pub fn create_compute_pipeline_with_samplers(
        &self,
        desc: &ComputePipelineDesc,
        immutable_samplers: &[(u32, u32, SamplerId)],
    ) -> PipelineId {
        unimplemented!("metal: create_compute_pipeline_with_samplers")
    }

    pub fn create_persistent_bind_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
    ) -> BindGroupId {
        unimplemented!("metal: create_persistent_bind_group")
    }

    pub fn create_texture_array_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        textures: &[TextureViewId],
        sampler: SamplerId,
    ) -> BindGroupId {
        unimplemented!("metal: create_texture_array_group")
    }

    pub fn destroy_bind_group(&self, group: BindGroupId) {
        unimplemented!("metal: destroy_bind_group")
    }

    pub fn create_texture_cube(&self, pixels: &[u8], face: u32, format: Format) -> TextureViewId {
        unimplemented!("metal: create_texture_cube")
    }

    pub fn record_blocking(&self, record: impl FnOnce(&mut Encoder)) {
        unimplemented!("metal: record_blocking")
    }

    pub fn create_texture_mipped(
        &self,
        pixels: &[u8],
        extent: [u32; 2],
        format: Format,
    ) -> TextureViewId {
        unimplemented!("metal: create_texture_mipped")
    }

    pub fn create_texture_levels(&self, levels: &super::TextureLevels) -> TextureViewId {
        unimplemented!("metal: create_texture_levels")
    }

    pub fn fill_depth(&self, texture: TextureViewId, depth: f32) {
        unimplemented!("metal: fill_depth")
    }

    pub fn create_transient_bind_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
    ) -> BindGroupId {
        unimplemented!("metal: create_transient_bind_group")
    }

    pub fn create_bind_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
    ) -> BindGroupId {
        unimplemented!("metal: create_bind_group")
    }
}

#[allow(
    unused_variables,
    reason = "operations land one at a time; a stub names itself in its panic"
)]
impl MtlEncoder<'_> {
    pub fn set_viewport(&mut self, viewport: Viewport) {
        unimplemented!("metal: set_viewport")
    }

    pub fn set_scissor(&mut self, offset: [u32; 2], extent: [u32; 2]) {
        unimplemented!("metal: set_scissor")
    }

    pub fn set_depth_bias(&mut self, constant: f32, clamp: f32, slope: f32) {
        unimplemented!("metal: set_depth_bias")
    }

    pub fn clear_depth(&mut self, rects: &[DepthRect]) {
        unimplemented!("metal: clear_depth")
    }

    pub fn bind_pipeline(&mut self, pipeline: PipelineId) {
        unimplemented!("metal: bind_pipeline")
    }

    pub fn bind_vertex_buffer(&mut self, buffer: BufferId) {
        unimplemented!("metal: bind_vertex_buffer")
    }

    pub fn bind_group(&mut self, set: u32, group: BindGroupId) {
        unimplemented!("metal: bind_group")
    }

    pub fn push_constants(&mut self, words: &[u32]) {
        unimplemented!("metal: push_constants")
    }

    pub fn dispatch(&mut self, groups: [u32; 3]) {
        unimplemented!("metal: dispatch")
    }

    pub fn bind_geometry(&mut self, positions_only: bool) {
        unimplemented!("metal: bind_geometry")
    }

    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instances: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unimplemented!("metal: draw_indexed")
    }

    pub fn open_swapchain(&mut self, target: TextureViewId) {
        unimplemented!("metal: open_swapchain")
    }

    pub fn close_swapchain(&mut self, target: TextureViewId) {
        unimplemented!("metal: close_swapchain")
    }

    pub fn begin_rendering(&mut self, target: TextureViewId) {
        unimplemented!("metal: begin_rendering")
    }

    pub fn end_rendering(&mut self) {
        unimplemented!("metal: end_rendering")
    }

    pub fn capture_backdrop(
        &mut self,
        target: TextureViewId,
        half: TextureViewId,
        quarter: TextureViewId,
    ) {
        unimplemented!("metal: capture_backdrop")
    }

    pub fn begin_rendering_fresh(&mut self, target: TextureViewId) {
        unimplemented!("metal: begin_rendering_fresh")
    }

    pub fn generate_cube_mips(&mut self, texture: TextureViewId) {
        unimplemented!("metal: generate_cube_mips")
    }

    pub fn publish_render_target(&mut self, texture: TextureViewId) {
        unimplemented!("metal: publish_render_target")
    }

    pub fn bind_index_buffer(&mut self, buffer: BufferId) {
        unimplemented!("metal: bind_index_buffer")
    }

    pub fn write_texture(
        &mut self,
        texture: TextureViewId,
        offset: [u32; 2],
        extent: [u32; 2],
        staging: BufferId,
        initial: bool,
    ) {
        unimplemented!("metal: write_texture")
    }

    pub fn copy_buffer(&mut self, src: BufferId, dst: BufferId, regions: &[BufferCopy]) {
        unimplemented!("metal: copy_buffer")
    }

    pub fn draw_indexed_indirect(&mut self, commands: BufferId, range: std::ops::Range<u32>) {
        unimplemented!("metal: draw_indexed_indirect")
    }

    pub fn draw(&mut self, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
        unimplemented!("metal: draw")
    }
}
