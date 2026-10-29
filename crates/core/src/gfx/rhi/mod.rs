//! The renderer's view of a graphics device: typed handles and the descriptions
//! that create them, with no API type anywhere.
//!
//! **Handles, not objects.** A resource is a dense `u32` newtype into a pool the
//! driver owns, freed through a frame-delayed queue (`pool.rs`). That matches the
//! engine's rule that shared state is plain data addressed by ID, and it is what
//! lets one pass module record against Vulkan or Metal without being generic over
//! either: it names a `TextureId`, never an `Arc<Image>` or an `MTLTexture`.
//!
//! **Enum dispatch, not traits.** A `Device` wraps `enum { Vulkan(..), Metal(..) }`
//! and so does the encoder; the cost is one predictable `match` per recorded
//! command. A generic `Renderer<R: Rhi>` would monomorphise every pass twice in
//! the largest crate, and a `Box<dyn Driver>` pays the same dispatch with less
//! inlining.
//!
//! Only the device-free surface exists so far: handles, descriptions and the pool.
//! The drivers and the encoder arrive with the passes that need them.

mod desc;
mod geometry;
mod pool;

pub use geometry::MeshSpan;
pub(crate) use geometry::{Placement, grown};

pub(crate) use pool::Pool;

pub use desc::{
    Binding, BindingResource, Blend, BlockKind, BufferCopy, BufferDesc, ColorAttachment,
    ComputePipelineDesc, CullMode, DepthAttachment, DepthCompare, DepthRect, DepthTest, Feature,
    GraphicsPipelineDesc, LoadOp, MemoryLocation, RenderTargets, SamplerDesc, StoreOp, TextureDesc,
    TextureDimension, TextureSwizzle, Topology, VertexAttribute, VertexFormat, VertexStream,
    Viewport,
};

use super::format::Format;
use super::vulkan::driver::{VkDriver, VkEncoder};
use crate::geom::Aabb;
use crate::gfx::Vertex;

/// One `match` over the drivers, so adding a driver is adding a variant and
/// every method the drivers share is checked against it at compile time.
macro_rules! device_dispatch {
    ($self:expr, $d:ident => $body:expr) => {
        match $self {
            Device::Vulkan($d) => $body,
            #[cfg(target_os = "macos")]
            Device::Metal($d) => $body,
        }
    };
}

macro_rules! encoder_dispatch {
    ($self:expr, $d:ident => $body:expr) => {
        match $self {
            Encoder::Vulkan($d) => $body,
            #[cfg(target_os = "macos")]
            Encoder::Metal($d) => $body,
        }
    };
}

/// The device a pass creates its resources on: Vulkan everywhere, and on macOS
/// native Metal beside it.
pub enum Device {
    Vulkan(VkDriver),
    #[cfg(target_os = "macos")]
    Metal(super::metal::MtlDriver),
}

impl Device {
    /// The Vulkan driver, for the executor that records the Vulkan frame. A
    /// pass never asks: it holds a `Device`.
    pub(crate) fn vulkan(&self) -> &VkDriver {
        match self {
            Device::Vulkan(driver) => driver,
            #[cfg(target_os = "macos")]
            Device::Metal(_) => unreachable!("the Vulkan executor was handed the Metal device"),
        }
    }

    /// Frees whatever `frame - frames_in_flight` retired. Called once per frame,
    /// before anything records.
    pub fn begin_frame(&self, frame: u64) {
        device_dispatch!(self, driver => driver.begin_frame(frame))
    }

    /// Put a mesh into the device's shared geometry arena and say where it
    /// landed, with its object-space bounds.
    pub fn upload_mesh(&self, vertices: &[Vertex], indices: &[u32]) -> (MeshSpan, Aabb) {
        device_dispatch!(self, driver => driver.upload_mesh(vertices, indices))
    }

    pub fn create_sampler(&self, desc: SamplerDesc) -> SamplerId {
        device_dispatch!(self, driver => driver.create_sampler(desc))
    }

    pub fn create_graphics_pipeline(&self, desc: &GraphicsPipelineDesc) -> PipelineId {
        device_dispatch!(self, driver => driver.create_graphics_pipeline(desc))
    }

    pub fn create_compute_pipeline(&self, desc: &ComputePipelineDesc) -> PipelineId {
        device_dispatch!(self, driver => driver.create_compute_pipeline(desc))
    }

    pub fn create_compute_pipeline_with_samplers(
        &self,
        desc: &ComputePipelineDesc,
        immutable_samplers: &[(u32, u32, SamplerId)],
    ) -> PipelineId {
        device_dispatch!(self, driver => {
            driver.create_compute_pipeline_with_samplers(desc, immutable_samplers)
        })
    }

    /// A bind group kept until [`destroy_bind_group`](Self::destroy_bind_group),
    /// for a set whose resources outlive the frame.
    pub fn create_persistent_bind_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
    ) -> BindGroupId {
        device_dispatch!(self, driver => driver.create_persistent_bind_group(pipeline, set, bindings))
    }

    /// The material texture table: `textures` as the array at binding 0 and
    /// `sampler` at binding 1, padded to the device's array length.
    pub fn create_texture_array_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        textures: &[TextureViewId],
        sampler: SamplerId,
    ) -> BindGroupId {
        device_dispatch!(self, driver => {
            driver.create_texture_array_group(pipeline, set, textures, sampler)
        })
    }

    /// Retire a persistent group once the frames that may still use it are done.
    pub fn destroy_bind_group(&self, group: BindGroupId) {
        device_dispatch!(self, driver => driver.destroy_bind_group(group))
    }

    pub fn destroy_buffer(&self, buffer: BufferId) {
        device_dispatch!(self, driver => driver.destroy_buffer(buffer))
    }

    /// A buffer that outlives the frame, with no defined contents: for one a
    /// dispatch writes before anything reads it.
    pub fn create_buffer(&self, desc: BufferDesc) -> BufferId {
        device_dispatch!(self, driver => driver.create_buffer(desc))
    }

    /// A buffer that outlives the frame, holding `bytes` from creation.
    pub fn create_buffer_bytes(&self, desc: BufferDesc, bytes: &[u8]) -> BufferId {
        device_dispatch!(self, driver => driver.create_buffer_bytes(desc, bytes))
    }

    /// A buffer that outlives the frame, holding `initial` from creation. The
    /// words are the buffer's whole contents: `desc.size` must be their length.
    pub fn create_buffer_init(&self, desc: BufferDesc, initial: &[u32]) -> BufferId {
        device_dispatch!(self, driver => driver.create_buffer_init(desc, initial))
    }

    /// A texture that outlives the frame, owned by whoever holds the id. The
    /// view covers the whole image; a sampled view of a mipped image never
    /// claims storage usage, because a storage descriptor takes one level.
    pub fn create_texture(&self, desc: &TextureDesc) -> TextureViewId {
        device_dispatch!(self, driver => driver.create_texture(desc))
    }

    /// Like [`create_sampler`](Self::create_sampler) but repeating along u only
    /// and clamping v: for an equirectangular map, whose seam wraps and whose
    /// poles must not fold across to the opposite hemisphere.
    pub fn create_sampler_wrap_u(&self, desc: SamplerDesc) -> SamplerId {
        device_dispatch!(self, driver => driver.create_sampler_wrap_u(desc))
    }

    /// A persistent cube texture of `face` texels a side holding `pixels`, six
    /// faces tightly packed in +X, -X, +Y, -Y, +Z, -Z order.
    pub fn create_texture_cube(&self, pixels: &[u8], face: u32, format: Format) -> TextureViewId {
        device_dispatch!(self, driver => driver.create_texture_cube(pixels, face, format))
    }

    /// A single-face, single-level 2D view of `texture`, for rendering into one
    /// face of a cube at one level.
    pub fn face_view(&self, texture: TextureViewId, face: u32, level: u32) -> TextureViewId {
        device_dispatch!(self, driver => driver.face_view(texture, face, level))
    }

    /// Record with `record` on a fresh encoder and wait for the GPU to finish.
    /// For load-time work whose result the next frame must not see half done.
    pub fn record_blocking(&self, record: impl FnOnce(&mut Encoder)) {
        device_dispatch!(self, driver => driver.record_blocking(record))
    }

    /// A view of one mip level of `texture`, with the texture's full usage, for
    /// a pass that writes the levels one at a time.
    pub fn mip_view(&self, texture: TextureViewId, level: u32) -> TextureViewId {
        device_dispatch!(self, driver => driver.mip_view(texture, level))
    }

    /// The most anisotropic filtering the device offers, capped at sixteen, or
    /// `None` where it offers none.
    pub fn max_anisotropy(&self) -> Option<f32> {
        device_dispatch!(self, driver => driver.max_anisotropy())
    }

    /// A persistent 2D colour texture whose full mip chain is generated from
    /// `pixels` at upload.
    pub fn create_texture_mipped(
        &self,
        pixels: &[u8],
        extent: [u32; 2],
        format: Format,
    ) -> TextureViewId {
        device_dispatch!(self, driver => driver.create_texture_mipped(pixels, extent, format))
    }

    /// A persistent texture from levels already authored, which may be block
    /// compressed. Check [`Feature::BlockCompression`] first for those.
    pub fn create_texture_levels(&self, levels: &super::TextureLevels) -> TextureViewId {
        device_dispatch!(self, driver => driver.create_texture_levels(levels))
    }

    pub fn supports(&self, feature: Feature) -> bool {
        device_dispatch!(self, driver => driver.supports(feature))
    }

    /// A texture whose channels a shader reads through `swizzle`.
    pub fn create_texture_swizzled(
        &self,
        desc: &TextureDesc,
        swizzle: TextureSwizzle,
    ) -> TextureViewId {
        device_dispatch!(self, driver => driver.create_texture_swizzled(desc, swizzle))
    }

    /// Retire a texture or a view of one once the frames that may still read it
    /// are done.
    pub fn destroy_texture(&self, view: TextureViewId) {
        device_dispatch!(self, driver => driver.destroy_texture(view))
    }

    /// Fill a depth texture with one value, once, and leave it readable. For the
    /// stand-in a pass samples when the real target does not exist this frame.
    pub fn fill_depth(&self, texture: TextureViewId, depth: f32) {
        device_dispatch!(self, driver => driver.fill_depth(texture, depth))
    }

    /// A persistent single-level 2D texture holding `pixels`, tightly packed.
    pub fn create_texture_2d(
        &self,
        pixels: &[u8],
        extent: [u32; 2],
        format: Format,
    ) -> TextureViewId {
        device_dispatch!(self, driver => driver.create_texture_2d(pixels, extent, format))
    }

    /// Per-frame uniform data: valid for the frame being recorded, then retired
    /// with it. The words are the block's std140 bytes.
    /// Per-frame vertex data, valid for the frame being recorded.
    pub fn upload_vertices(&self, bytes: &[u8]) -> BufferId {
        device_dispatch!(self, driver => driver.upload_vertices(bytes))
    }

    /// A per-frame block of plain data, valid for the frame being recorded. The
    /// bytes are a `#[repr(C)]` struct's, in the layout the shader declares.
    pub fn upload_block(&self, kind: BlockKind, bytes: &[u8]) -> BufferId {
        device_dispatch!(self, driver => driver.upload_block(kind, bytes))
    }

    pub fn upload_uniform(&self, words: &[u32]) -> BufferId {
        device_dispatch!(self, driver => driver.upload_uniform(words))
    }

    pub fn view_extent(&self, view: TextureViewId) -> [u32; 2] {
        device_dispatch!(self, driver => driver.view_extent(view))
    }

    /// Like [`create_bind_group`](Self::create_bind_group) but never cached, for
    /// a set that binds a buffer allocated for this frame alone.
    pub fn create_transient_bind_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
    ) -> BindGroupId {
        device_dispatch!(self, driver => driver.create_transient_bind_group(pipeline, set, bindings))
    }

    /// A set of bindings for set `set` of `pipeline`. Equal bindings give back
    /// the same underlying set, as the descriptor cache did before the RHI.
    pub fn create_bind_group(
        &self,
        pipeline: PipelineId,
        set: u32,
        bindings: &[Binding],
    ) -> BindGroupId {
        device_dispatch!(self, driver => driver.create_bind_group(pipeline, set, bindings))
    }
}

/// Records commands. Every method is one `match` over the driver's encoder.
pub enum Encoder<'a> {
    Vulkan(VkEncoder<'a>),
    #[cfg(target_os = "macos")]
    Metal(super::metal::MtlEncoder<'a>),
}

impl Encoder<'_> {
    pub fn set_viewport(&mut self, viewport: Viewport) {
        encoder_dispatch!(self, encoder => encoder.set_viewport(viewport))
    }

    pub fn set_scissor(&mut self, offset: [u32; 2], extent: [u32; 2]) {
        encoder_dispatch!(self, encoder => encoder.set_scissor(offset, extent))
    }

    pub fn set_depth_bias(&mut self, constant: f32, clamp: f32, slope: f32) {
        encoder_dispatch!(self, encoder => encoder.set_depth_bias(constant, clamp, slope))
    }

    /// Clear depth to 1.0 over `rects` of the attachment being rendered into.
    pub fn clear_depth(&mut self, rects: &[DepthRect]) {
        encoder_dispatch!(self, encoder => encoder.clear_depth(rects))
    }

    pub fn bind_pipeline(&mut self, pipeline: PipelineId) {
        encoder_dispatch!(self, encoder => encoder.bind_pipeline(pipeline))
    }

    /// Bind `buffer` as vertex stream 0.
    pub fn bind_vertex_buffer(&mut self, buffer: BufferId) {
        encoder_dispatch!(self, encoder => encoder.bind_vertex_buffer(buffer))
    }

    pub fn bind_group(&mut self, set: u32, group: BindGroupId) {
        encoder_dispatch!(self, encoder => encoder.bind_group(set, group))
    }

    /// Up to 32 words: a driver's push-constant budget is far above any block
    /// this renderer has, and a fixed bound keeps the call allocation-free.
    pub fn push_constants(&mut self, words: &[u32]) {
        encoder_dispatch!(self, encoder => encoder.push_constants(words))
    }

    pub fn dispatch(&mut self, groups: [u32; 3]) {
        encoder_dispatch!(self, encoder => encoder.dispatch(groups))
    }

    /// Bind the shared vertex and index buffers every mesh is a span into.
    pub fn bind_geometry(&mut self, positions_only: bool) {
        encoder_dispatch!(self, encoder => encoder.bind_geometry(positions_only))
    }

    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instances: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        encoder_dispatch!(self, encoder => encoder.draw_indexed(
            index_count,
            instances,
            first_index,
            vertex_offset,
            first_instance,
        ))
    }

    /// Take a swapchain image that arrives presentable into rendering, keeping
    /// what is in it.
    pub fn open_swapchain(&mut self, target: TextureViewId) {
        encoder_dispatch!(self, encoder => encoder.open_swapchain(target))
    }

    /// Hand a swapchain image back presentable.
    pub fn close_swapchain(&mut self, target: TextureViewId) {
        encoder_dispatch!(self, encoder => encoder.close_swapchain(target))
    }

    /// Open a render pass over `targets`.
    pub fn begin_pass(&mut self, targets: &RenderTargets) {
        encoder_dispatch!(self, encoder => encoder.begin_pass(targets))
    }

    /// Open a render pass over `target`, loading what it holds.
    pub fn begin_rendering(&mut self, target: TextureViewId) {
        encoder_dispatch!(self, encoder => encoder.begin_rendering(target))
    }

    pub fn end_rendering(&mut self) {
        encoder_dispatch!(self, encoder => encoder.end_rendering())
    }

    /// Downsample the render target into `half` and then `quarter`, leaving
    /// `quarter` sampleable and the target ready to render into again. Outside
    /// a render pass; for a pass that blurs what was drawn under it.
    pub fn capture_backdrop(
        &mut self,
        target: TextureViewId,
        half: TextureViewId,
        quarter: TextureViewId,
    ) {
        encoder_dispatch!(self, encoder => encoder.capture_backdrop(target, half, quarter))
    }

    /// Open a render pass over `target` whose contents do not matter and need
    /// not be kept from before, for a texture rendered once. A face view's own
    /// level sets the area.
    pub fn begin_rendering_fresh(&mut self, target: TextureViewId) {
        encoder_dispatch!(self, encoder => encoder.begin_rendering_fresh(target))
    }

    /// Halve each level of a cube into the next with a linear blit, all six
    /// faces at once. Level 0 arrives freshly rendered, and the whole chain
    /// leaves sampleable.
    pub fn generate_cube_mips(&mut self, texture: TextureViewId) {
        encoder_dispatch!(self, encoder => encoder.generate_cube_mips(texture))
    }

    /// Make every level of a texture just rendered into sampleable.
    pub fn publish_render_target(&mut self, texture: TextureViewId) {
        encoder_dispatch!(self, encoder => encoder.publish_render_target(texture))
    }

    pub fn bind_index_buffer(&mut self, buffer: BufferId) {
        encoder_dispatch!(self, encoder => encoder.bind_index_buffer(buffer))
    }

    /// Write `extent` texels at `offset` of `texture` from `staging`, tightly
    /// packed. `initial` is the first write to a texture just created, whose
    /// contents are undefined; a later one patches what is there.
    pub fn write_texture(
        &mut self,
        texture: TextureViewId,
        offset: [u32; 2],
        extent: [u32; 2],
        staging: BufferId,
        initial: bool,
    ) {
        encoder_dispatch!(self, encoder => {
            encoder.write_texture(texture, offset, extent, staging, initial)
        })
    }

    /// Copy runs of bytes from `src` to `dst`, ahead of every pass recorded
    /// after it.
    pub fn copy_buffer(&mut self, src: BufferId, dst: BufferId, regions: &[BufferCopy]) {
        encoder_dispatch!(self, encoder => encoder.copy_buffer(src, dst, regions))
    }

    /// Draw commands `range` of `commands`, a buffer of indexed indirect draws.
    pub fn draw_indexed_indirect(&mut self, commands: BufferId, range: std::ops::Range<u32>) {
        encoder_dispatch!(self, encoder => encoder.draw_indexed_indirect(commands, range))
    }

    pub fn draw(&mut self, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
        encoder_dispatch!(self, encoder => {
            encoder.draw(vertices, instances, first_vertex, first_instance)
        })
    }
}

macro_rules! handles {
    ($($(#[$meta:meta])* $name:ident),+ $(,)?) => {$(
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub struct $name(pub(crate) u32);

        impl $name {
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
    )+};
}

handles! {
    TextureId,
    /// A view of part of a texture: one mip level, or a reinterpreted format.
    TextureViewId,
    BufferId,
    SamplerId,
    PipelineId,
    /// One descriptor set / argument buffer's worth of bindings.
    BindGroupId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_of_different_kinds_do_not_convert() {
        let texture = TextureId(3);
        let buffer = BufferId(3);
        assert_eq!(texture.index(), buffer.index());
    }
}
