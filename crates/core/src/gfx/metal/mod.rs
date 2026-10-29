//! The native Metal driver, beside the Vulkan one behind the same RHI.
//!
//! Brought up in the order `docs/metal-backend-plan.md` M4 gives: the device and
//! its capability gate first, then resources, then pipelines from the embedded
//! MSL, then the encoder. A method that is still `unimplemented!` names itself
//! in the panic, so a pass that reaches one says which operation it needs next.

use std::marker::PhantomData;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice};

use crate::geom::Aabb;
use crate::gfx::Vertex;
use crate::gfx::format::Format;
use crate::gfx::rhi::{
    BindGroupId, Binding, BlockKind, BufferCopy, BufferDesc, BufferId, ComputePipelineDesc,
    DepthRect, Encoder, Feature, GraphicsPipelineDesc, MeshSpan, PipelineId, SamplerDesc,
    SamplerId, TextureDesc, TextureSwizzle, TextureViewId, Viewport,
};

pub struct MtlDriver {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

pub struct MtlEncoder<'a> {
    _driver: PhantomData<&'a MtlDriver>,
}

#[allow(
    unused_variables,
    reason = "operations land one at a time; a stub names itself in its panic"
)]
impl MtlDriver {
    pub fn begin_frame(&self, frame: u64) {
        unimplemented!("metal: begin_frame")
    }

    pub fn upload_mesh(&self, vertices: &[Vertex], indices: &[u32]) -> (MeshSpan, Aabb) {
        unimplemented!("metal: upload_mesh")
    }

    pub fn create_sampler(&self, desc: SamplerDesc) -> SamplerId {
        unimplemented!("metal: create_sampler")
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

    pub fn destroy_buffer(&self, buffer: BufferId) {
        unimplemented!("metal: destroy_buffer")
    }

    pub fn create_buffer(&self, desc: BufferDesc) -> BufferId {
        unimplemented!("metal: create_buffer")
    }

    pub fn create_buffer_bytes(&self, desc: BufferDesc, bytes: &[u8]) -> BufferId {
        unimplemented!("metal: create_buffer_bytes")
    }

    pub fn create_buffer_init(&self, desc: BufferDesc, initial: &[u32]) -> BufferId {
        unimplemented!("metal: create_buffer_init")
    }

    pub fn create_texture(&self, desc: &TextureDesc) -> TextureViewId {
        unimplemented!("metal: create_texture")
    }

    pub fn create_sampler_wrap_u(&self, desc: SamplerDesc) -> SamplerId {
        unimplemented!("metal: create_sampler_wrap_u")
    }

    pub fn create_texture_cube(&self, pixels: &[u8], face: u32, format: Format) -> TextureViewId {
        unimplemented!("metal: create_texture_cube")
    }

    pub fn face_view(&self, texture: TextureViewId, face: u32, level: u32) -> TextureViewId {
        unimplemented!("metal: face_view")
    }

    pub fn record_blocking(&self, record: impl FnOnce(&mut Encoder)) {
        unimplemented!("metal: record_blocking")
    }

    pub fn mip_view(&self, texture: TextureViewId, level: u32) -> TextureViewId {
        unimplemented!("metal: mip_view")
    }

    pub fn max_anisotropy(&self) -> Option<f32> {
        unimplemented!("metal: max_anisotropy")
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

    pub fn supports(&self, feature: Feature) -> bool {
        unimplemented!("metal: supports")
    }

    pub fn create_texture_swizzled(
        &self,
        desc: &TextureDesc,
        swizzle: TextureSwizzle,
    ) -> TextureViewId {
        unimplemented!("metal: create_texture_swizzled")
    }

    pub fn destroy_texture(&self, view: TextureViewId) {
        unimplemented!("metal: destroy_texture")
    }

    pub fn fill_depth(&self, texture: TextureViewId, depth: f32) {
        unimplemented!("metal: fill_depth")
    }

    pub fn create_texture_2d(
        &self,
        pixels: &[u8],
        extent: [u32; 2],
        format: Format,
    ) -> TextureViewId {
        unimplemented!("metal: create_texture_2d")
    }

    pub fn upload_vertices(&self, bytes: &[u8]) -> BufferId {
        unimplemented!("metal: upload_vertices")
    }

    pub fn upload_block(&self, kind: BlockKind, bytes: &[u8]) -> BufferId {
        unimplemented!("metal: upload_block")
    }

    pub fn upload_uniform(&self, words: &[u32]) -> BufferId {
        unimplemented!("metal: upload_uniform")
    }

    pub fn view_extent(&self, view: TextureViewId) -> [u32; 2] {
        unimplemented!("metal: view_extent")
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
