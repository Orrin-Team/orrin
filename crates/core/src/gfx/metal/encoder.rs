//! The Metal encoder: one command buffer, and one Metal encoder at a time.
//!
//! A pass opens a render pass (or just dispatches, or copies) and the encoder
//! opens whichever Metal encoder that needs, closing the one before it. Hazards
//! are Metal's own while resources are tracked, so there is no barrier to record
//! (decision D6): the graph's plan is simply not consulted here.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLIndexType, MTLLoadAction,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLRenderStages, MTLScissorRect, MTLSize,
    MTLStoreAction, MTLViewport, MTLWinding,
};

use super::pipeline::Pipe;
use super::{MtlDriver, MtlEncoder};
use crate::gfx::rhi::{
    BindGroupId, BufferCopy, BufferId, DepthRect, PipelineId, TextureViewId, Viewport,
};
use crate::gfx::shaders::PUSH_CONSTANT_BUFFER;

/// Which way a front face winds, in the window space Metal rasterises in. The
/// translator flips clip-space Y so the image lands upright, which leaves
/// window-space winding as Vulkan's; `a_single_sided_quad_is_visible` is the
/// test that holds this (trap T5).
const FRONT_FACING: MTLWinding = MTLWinding::CounterClockwise;

enum Open {
    None,
    Render(Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>),
    Compute(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>),
    Blit(Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>),
}

pub(super) struct State {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    open: Open,
    pipeline: Option<PipelineId>,
    index_buffer: Option<(Retained<ProtocolObject<dyn MTLBuffer>>, u64)>,
}

impl State {
    pub(super) fn new(driver: &MtlDriver) -> Self {
        Self {
            command_buffer: driver
                .queue
                .commandBuffer()
                .expect("the queue gave no command buffer"),
            open: Open::None,
            pipeline: None,
            index_buffer: None,
        }
    }

    fn close(&mut self) {
        match std::mem::replace(&mut self.open, Open::None) {
            Open::None => {}
            Open::Render(encoder) => encoder.endEncoding(),
            Open::Compute(encoder) => encoder.endEncoding(),
            Open::Blit(encoder) => encoder.endEncoding(),
        }
    }

    /// End recording and wait for the GPU.
    pub(super) fn finish_blocking(mut self) {
        self.close();
        self.command_buffer.commit();
        self.command_buffer.waitUntilCompleted();
    }
}

impl<'a> MtlEncoder<'a> {
    pub(super) fn new(driver: &'a MtlDriver, state: &'a mut State) -> Self {
        Self { driver, state }
    }

    fn render(&self) -> &Retained<ProtocolObject<dyn MTLRenderCommandEncoder>> {
        match &self.state.open {
            Open::Render(encoder) => encoder,
            _ => panic!("a draw was recorded outside a render pass"),
        }
    }

    fn compute(&mut self) -> Retained<ProtocolObject<dyn MTLComputeCommandEncoder>> {
        if let Open::Compute(encoder) = &self.state.open {
            return encoder.clone();
        }
        assert!(
            !matches!(self.state.open, Open::Render(_)),
            "a dispatch was recorded inside a render pass"
        );
        self.state.close();
        let encoder = self
            .state
            .command_buffer
            .computeCommandEncoder()
            .expect("failed to open a compute encoder");
        self.state.open = Open::Compute(encoder.clone());
        encoder
    }

    fn blit(&mut self) -> Retained<ProtocolObject<dyn MTLBlitCommandEncoder>> {
        if let Open::Blit(encoder) = &self.state.open {
            return encoder.clone();
        }
        assert!(
            !matches!(self.state.open, Open::Render(_)),
            "a copy was recorded inside a render pass"
        );
        self.state.close();
        let encoder = self
            .state
            .command_buffer
            .blitCommandEncoder()
            .expect("failed to open a blit encoder");
        self.state.open = Open::Blit(encoder.clone());
        encoder
    }

    fn begin(&mut self, target: TextureViewId, load: MTLLoadAction) {
        self.state.close();
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        // SAFETY: attachment 0 of a descriptor is always valid to address.
        let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        attachment.setTexture(Some(&self.driver.texture(target)));
        attachment.setLoadAction(load);
        attachment.setStoreAction(MTLStoreAction::Store);
        attachment.setClearColor(MTLClearColor {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
            alpha: 0.0,
        });
        let encoder = self
            .state
            .command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .expect("failed to open a render encoder");
        self.state.open = Open::Render(encoder);
        self.state.pipeline = None;
    }

    pub fn begin_rendering(&mut self, target: TextureViewId) {
        self.begin(target, MTLLoadAction::Load);
    }

    /// The target's contents do not matter, so they are cleared: a tile GPU
    /// makes that free, and a pass that reads an uncovered texel gets zero, not
    /// whatever was in the memory.
    pub fn begin_rendering_fresh(&mut self, target: TextureViewId) {
        self.begin(target, MTLLoadAction::Clear);
    }

    pub fn end_rendering(&mut self) {
        self.state.close();
    }

    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.render().setViewport(MTLViewport {
            originX: viewport.origin[0] as f64,
            originY: viewport.origin[1] as f64,
            width: viewport.extent[0] as f64,
            height: viewport.extent[1] as f64,
            znear: viewport.depth_range[0] as f64,
            zfar: viewport.depth_range[1] as f64,
        });
    }

    pub fn set_scissor(&mut self, offset: [u32; 2], extent: [u32; 2]) {
        self.render().setScissorRect(MTLScissorRect {
            x: offset[0] as usize,
            y: offset[1] as usize,
            width: extent[0] as usize,
            height: extent[1] as usize,
        });
    }

    pub fn set_depth_bias(&mut self, constant: f32, clamp: f32, slope: f32) {
        self.render()
            .setDepthBias_slopeScale_clamp(constant, slope, clamp);
    }

    pub fn bind_pipeline(&mut self, pipeline: PipelineId) {
        self.state.pipeline = Some(pipeline);
        self.driver.with_pipe(pipeline, |pipe| match pipe {
            Pipe::Render(render) => {
                let encoder = self.render();
                encoder.setRenderPipelineState(&render.state.0);
                if let Some(depth) = &render.depth {
                    encoder.setDepthStencilState(Some(&depth.0));
                }
                encoder.setCullMode(render.cull);
                encoder.setFrontFacingWinding(FRONT_FACING);
            }
            Pipe::Compute(compute) => {
                let encoder = self.compute();
                encoder.setComputePipelineState(&compute.state.0);
            }
        });
    }

    pub fn bind_vertex_buffer(&mut self, buffer: BufferId) {
        let buf = self.driver.buffer(buffer);
        // SAFETY: a live buffer and an offset inside it.
        unsafe {
            self.render().setVertexBuffer_offset_atIndex(
                Some(&buf.buffer.0),
                buf.offset as usize,
                crate::gfx::shaders::VERTEX_STREAM_BASE as usize,
            );
        }
    }

    /// Bind the shared geometry: the positions stream, the surface stream unless
    /// only depth is wanted, and the index buffer.
    pub fn bind_geometry(&mut self, positions_only: bool) {
        let geometry = self
            .driver
            .geometry
            .lock()
            .expect("a pass panicked while using the device");
        let geometry = &geometry.0;
        let base = crate::gfx::shaders::VERTEX_STREAM_BASE as usize;
        let encoder = self.render().clone();
        // SAFETY: live buffers at offset zero.
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(Some(&geometry.positions), 0, base);
            if !positions_only {
                encoder.setVertexBuffer_offset_atIndex(Some(&geometry.surfaces), 0, base + 1);
            }
        }
        self.state.index_buffer = Some((geometry.indices.clone(), 0));
    }

    pub fn bind_index_buffer(&mut self, buffer: BufferId) {
        let buf = self.driver.buffer(buffer);
        self.state.index_buffer = Some((buf.buffer.0.clone(), buf.offset));
    }

    pub fn bind_group(&mut self, set: u32, group: BindGroupId) {
        let (buffer, offset, uses) = self.driver.group(group);
        match &self.state.open {
            Open::Render(encoder) => {
                // SAFETY: a live argument buffer at the index its set names.
                unsafe {
                    encoder.setVertexBuffer_offset_atIndex(
                        Some(&buffer),
                        offset as usize,
                        set as usize,
                    );
                    encoder.setFragmentBuffer_offset_atIndex(
                        Some(&buffer),
                        offset as usize,
                        set as usize,
                    );
                }
                for (resource, usage) in &uses {
                    encoder.useResource_usage_stages(
                        resource,
                        *usage,
                        MTLRenderStages::Vertex | MTLRenderStages::Fragment,
                    );
                }
            }
            Open::Compute(encoder) => {
                // SAFETY: as above.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&buffer), offset as usize, set as usize);
                }
                for (resource, usage) in &uses {
                    encoder.useResource_usage(resource, *usage);
                }
            }
            _ => panic!("a bind group was bound outside a pass"),
        }
    }

    pub fn push_constants(&mut self, words: &[u32]) {
        let bytes = bytemuck::cast_slice::<u32, u8>(words);
        let pointer =
            std::ptr::NonNull::new(bytes.as_ptr() as *mut _).expect("a slice is never null");
        match &self.state.open {
            // SAFETY: a live slice, copied by Metal before the call returns.
            Open::Render(encoder) => unsafe {
                encoder.setVertexBytes_length_atIndex(
                    pointer,
                    bytes.len(),
                    PUSH_CONSTANT_BUFFER as usize,
                );
                encoder.setFragmentBytes_length_atIndex(
                    pointer,
                    bytes.len(),
                    PUSH_CONSTANT_BUFFER as usize,
                );
            },
            // SAFETY: as above.
            Open::Compute(encoder) => unsafe {
                encoder.setBytes_length_atIndex(
                    pointer,
                    bytes.len(),
                    PUSH_CONSTANT_BUFFER as usize,
                );
            },
            _ => panic!("push constants were recorded outside a pass"),
        }
    }

    pub fn dispatch(&mut self, groups: [u32; 3]) {
        let pipeline = self
            .state
            .pipeline
            .expect("a dispatch with no pipeline bound");
        let threads = self.driver.with_pipe(pipeline, |pipe| match pipe {
            Pipe::Compute(compute) => compute.threads,
            Pipe::Render(_) => panic!("a dispatch with a render pipeline bound"),
        });
        self.compute().dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups[0] as usize,
                height: groups[1] as usize,
                depth: groups[2] as usize,
            },
            threads,
        );
    }

    fn topology(&self) -> objc2_metal::MTLPrimitiveType {
        let pipeline = self.state.pipeline.expect("a draw with no pipeline bound");
        self.driver.with_pipe(pipeline, |pipe| match pipe {
            Pipe::Render(render) => render.topology,
            Pipe::Compute(_) => panic!("a draw with a compute pipeline bound"),
        })
    }

    pub fn draw(&mut self, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
        let topology = self.topology();
        // SAFETY: plain counts.
        unsafe {
            self.render()
                .drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                    topology,
                    first_vertex as usize,
                    vertices as usize,
                    instances as usize,
                    first_instance as usize,
                );
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
        let topology = self.topology();
        let (buffer, base) = self
            .state
            .index_buffer
            .clone()
            .expect("an indexed draw with no index buffer bound");
        // SAFETY: the range lies inside the bound index buffer, which the pass
        // recorded the counts against.
        unsafe {
            self.render()
                .drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                    topology,
                    index_count as usize,
                    MTLIndexType::UInt32,
                    &buffer,
                    base as usize + first_index as usize * 4,
                    instances as usize,
                    vertex_offset as isize,
                    first_instance as usize,
                );
        }
    }

    /// One indirect draw per command: Metal has no multi-draw, and the Vulkan
    /// layout of a command is the one Metal reads.
    pub fn draw_indexed_indirect(&mut self, commands: BufferId, range: std::ops::Range<u32>) {
        let topology = self.topology();
        let (index, index_base) = self
            .state
            .index_buffer
            .clone()
            .expect("an indirect draw with no index buffer bound");
        let buf = self.driver.buffer(commands);
        for command in range {
            // SAFETY: each command lies inside the commands buffer.
            unsafe {
                self.render()
                    .drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                        topology,
                        MTLIndexType::UInt32,
                        &index,
                        index_base as usize,
                        &buf.buffer.0,
                        buf.offset as usize + command as usize * 20,
                    );
            }
        }
    }

    pub fn copy_buffer(&mut self, src: BufferId, dst: BufferId, regions: &[BufferCopy]) {
        let (src, dst) = (self.driver.buffer(src), self.driver.buffer(dst));
        let blit = self.blit();
        for region in regions {
            // SAFETY: each region lies inside both buffers.
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &src.buffer.0,
                    (src.offset + region.src_offset) as usize,
                    &dst.buffer.0,
                    (dst.offset + region.dst_offset) as usize,
                    region.size as usize,
                );
            }
        }
    }

    pub fn clear_depth(&mut self, _rects: &[DepthRect]) {
        unimplemented!("metal: clear_depth")
    }

    pub fn open_swapchain(&mut self, _target: TextureViewId) {
        unimplemented!("metal: open_swapchain")
    }

    pub fn close_swapchain(&mut self, _target: TextureViewId) {
        unimplemented!("metal: close_swapchain")
    }

    pub fn capture_backdrop(
        &mut self,
        _target: TextureViewId,
        _half: TextureViewId,
        _quarter: TextureViewId,
    ) {
        unimplemented!("metal: capture_backdrop")
    }

    pub fn generate_cube_mips(&mut self, _texture: TextureViewId) {
        unimplemented!("metal: generate_cube_mips")
    }

    pub fn publish_render_target(&mut self, _texture: TextureViewId) {
        // Tracked hazards: rendering into a texture and then sampling it needs
        // nothing said.
    }

    pub fn write_texture(
        &mut self,
        _texture: TextureViewId,
        _offset: [u32; 2],
        _extent: [u32; 2],
        _staging: BufferId,
        _initial: bool,
    ) {
        unimplemented!("metal: write_texture")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::format::{Format, SampleCount};
    use crate::gfx::graph::ImageUsage;
    use crate::gfx::rhi::{BufferDesc, Device, MemoryLocation, TextureDesc, TextureDimension};
    use crate::gfx::vulkan::hdr::HdrPass;

    fn metal() -> Device {
        Device::Metal(MtlDriver::new().expect("this machine runs the Metal backend"))
    }

    fn target(device: &Device, extent: [u32; 2]) -> TextureViewId {
        device.create_texture(&TextureDesc {
            format: Format::Rgba8Unorm,
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::Sample1,
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
            memoryless: false,
        })
    }

    fn floats(values: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(values).to_vec()
    }

    /// The first pass through the whole stack on a real device: a fullscreen
    /// triangle, a sampled input, a storage buffer and push constants, all
    /// reached through argument buffers. The expected texels are the ACES curve
    /// the shader applies, worked by hand.
    #[test]
    #[ignore = "needs a Metal device"]
    fn the_tonemap_pass_renders_the_aces_curve() {
        let device = metal();
        let Device::Metal(driver) = &device else {
            unreachable!()
        };
        driver.begin_frame(0);

        let hdr = device.create_texture_2d(
            &floats(&[1.0, 0.5, 0.25, 1.0].repeat(16)),
            [4, 4],
            Format::Rgba32Float,
        );
        let bloom = device.create_texture_2d(&[0, 0, 0, 255], [1, 1], Format::Rgba8Unorm);
        let exposure = device.create_buffer_bytes(
            BufferDesc {
                size: 12,
                usage: crate::gfx::graph::BufferUsage::STORAGE_BUFFER,
                location: MemoryLocation::Upload,
            },
            &floats(&[1.0, 1.0, 0.0]),
        );
        let output = target(&device, [4, 4]);

        let mut pass = HdrPass::new(&device, Format::Rgba8Unorm);
        pass.auto_exposure = false;
        pass.manual_exposure = 1.0;
        device.record_blocking(|encoder| {
            encoder.begin_rendering_fresh(output);
            pass.record_tonemap(&device, encoder, [4, 4], hdr, exposure, bloom);
            encoder.end_rendering();
        });

        let pixels = driver.read_texture(output);
        let expect = [205u8, 157, 95, 255];
        for texel in pixels.chunks(4) {
            for (got, want) in texel.iter().zip(expect) {
                assert!(
                    got.abs_diff(want) <= 1,
                    "texel {texel:?}, expected about {expect:?}"
                );
            }
        }
    }

    /// Drawn through the real gizmo pass: a vertex stream matched to its shader
    /// by name, a blend, and the Y orientation the translator fixes. The
    /// triangle covers the top-left half of the target in framebuffer
    /// coordinates, where row zero is the top.
    #[test]
    #[ignore = "needs a Metal device"]
    fn a_gizmo_triangle_lands_where_vulkan_would_put_it() {
        use crate::gfx::vulkan::gizmo::{GizmoPass, GizmoVertex};
        let device = metal();
        let Device::Metal(driver) = &device else {
            unreachable!()
        };
        driver.begin_frame(0);
        let output = target(&device, [8, 8]);
        let pass = GizmoPass::new(&device, Format::Rgba8Unorm);
        let red = |x: f32, y: f32| GizmoVertex {
            position: [x, y],
            edge: [0.0, 100.0],
            color: [255, 0, 0, 255],
        };
        device.record_blocking(|encoder| {
            encoder.begin_rendering_fresh(output);
            pass.record(
                &device,
                encoder,
                &[red(0.0, 0.0), red(8.0, 0.0), red(0.0, 8.0)],
                [8, 8],
            );
            encoder.end_rendering();
        });
        let pixels = driver.read_texture(output);
        let texel = |x: usize, y: usize| &pixels[(y * 8 + x) * 4..][..4];
        assert_eq!(texel(1, 1), [255, 0, 0, 255], "inside, near the top-left");
        assert_eq!(texel(6, 6), [0, 0, 0, 0], "outside, near the bottom-right");
        // Inside only if the top row is the top: a flipped triangle covers the
        // bottom-left half instead, and leaves this texel empty.
        assert_eq!(texel(6, 0), [255, 0, 0, 255], "the top row, far right");
        assert_eq!(texel(7, 2), [0, 0, 0, 0], "beyond the diagonal");

        assert_eq!(
            texel(0, 6),
            [255, 0, 0, 255],
            "inside on the left column, low down"
        );
    }
}
