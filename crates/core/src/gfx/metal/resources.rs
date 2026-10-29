//! Buffers, textures and samplers: what the passes create and bind by id.
//!
//! Unified memory is the premise. On Apple silicon the CPU and GPU share one
//! pool, so a buffer a pass writes every frame is a shared buffer written in
//! place, and nothing needs a staging copy to reach the GPU. A buffer only the
//! GPU touches (a dispatch's output) is private.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion, MTLResourceOptions,
    MTLSamplerAddressMode, MTLSamplerBorderColor, MTLSamplerDescriptor, MTLSize, MTLStorageMode,
    MTLTexture, MTLTextureDescriptor, MTLTextureSwizzle, MTLTextureSwizzleChannels, MTLTextureType,
    MTLTextureUsage,
};

use super::convert::{may_be_memoryless, pixel_format, sampler_state, texture_usage};
use super::{Buf, FRAME_SLOTS, MtlDriver};
use crate::gfx::format::{Format, SampleCount};
use crate::gfx::graph::ImageUsage;
use crate::gfx::rhi::{
    BlockKind, BufferDesc, BufferId, Feature, MemoryLocation, SamplerDesc, SamplerId, TextureDesc,
    TextureDimension, TextureSwizzle, TextureViewId,
};

/// Every offset a buffer is bound at is a multiple of this, which is the
/// strictest alignment Metal asks of one: a constant buffer's.
const BIND_ALIGNMENT: u64 = 256;

impl MtlDriver {
    pub fn max_anisotropy(&self) -> Option<f32> {
        Some(16.0)
    }

    pub fn supports(&self, feature: Feature) -> bool {
        match feature {
            // Every Apple GPU family 7 device blits and filters every format the
            // engine renders to.
            Feature::BackdropCapture(_) => true,
            Feature::CoverageTexture => true,
            Feature::BlockCompression => self.device.supportsBCTextureCompression(),
        }
    }

    pub fn create_sampler(&self, desc: SamplerDesc) -> SamplerId {
        self.build_sampler(&desc, MTLSamplerAddressMode::ClampToEdge, None)
    }

    pub fn create_sampler_wrap_u(&self, desc: SamplerDesc) -> SamplerId {
        self.build_sampler(
            &desc,
            MTLSamplerAddressMode::ClampToEdge,
            Some(MTLSamplerAddressMode::Repeat),
        )
    }

    fn build_sampler(
        &self,
        desc: &SamplerDesc,
        clamp: MTLSamplerAddressMode,
        s_override: Option<MTLSamplerAddressMode>,
    ) -> SamplerId {
        let state = sampler_state(desc);
        let sampler = MTLSamplerDescriptor::new();
        sampler.setMinFilter(state.filter);
        sampler.setMagFilter(state.filter);
        sampler.setMipFilter(state.mip);
        let address = if desc.repeat { state.address } else { clamp };
        sampler.setSAddressMode(s_override.unwrap_or(address));
        sampler.setTAddressMode(address);
        sampler.setRAddressMode(address);
        sampler.setMaxAnisotropy(state.anisotropy);
        sampler.setLodMaxClamp(state.max_lod);
        // Reached through an argument buffer, which only takes a sampler built
        // for it.
        sampler.setSupportArgumentBuffers(true);
        if let Some(compare) = state.compare {
            // `Less` against a depth map cleared to 1.0, and a white border so a
            // fragment sampling outside a cascade reads as lit. A comparison
            // sampler is an ordinary one here; Vulkan's immutable-sampler
            // workaround has no counterpart.
            sampler.setCompareFunction(compare);
            sampler.setSAddressMode(MTLSamplerAddressMode::ClampToBorderColor);
            sampler.setTAddressMode(MTLSamplerAddressMode::ClampToBorderColor);
            sampler.setBorderColor(MTLSamplerBorderColor::OpaqueWhite);
        }
        let sampler = self
            .device
            .newSamplerStateWithDescriptor(&sampler)
            .expect("failed to create a sampler");
        SamplerId(self.samplers.lock().unwrap().insert(super::Shared(sampler)))
    }

    pub fn create_buffer(&self, desc: BufferDesc) -> BufferId {
        let options = match desc.location {
            MemoryLocation::GpuOnly => MTLResourceOptions::StorageModePrivate,
            MemoryLocation::Upload | MemoryLocation::Readback => {
                MTLResourceOptions::StorageModeShared
            }
        };
        let buffer = self
            .device
            .newBufferWithLength_options(desc.size.max(1) as usize, options)
            .expect("failed to allocate a buffer");
        self.insert_buffer(buffer, 0, desc.size)
    }

    pub fn create_buffer_bytes(&self, desc: BufferDesc, bytes: &[u8]) -> BufferId {
        assert_eq!(
            desc.size,
            bytes.len() as u64,
            "a buffer's size is its initial bytes"
        );
        // SAFETY: `bytes` is a live slice, so the pointer is valid for its
        // length, and Metal copies it before returning.
        let buffer = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(bytes.as_ptr() as *mut c_void).expect("a slice is never null"),
                bytes.len().max(1),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("failed to allocate a buffer");
        self.insert_buffer(buffer, 0, desc.size)
    }

    pub fn create_buffer_init(&self, desc: BufferDesc, initial: &[u32]) -> BufferId {
        assert_eq!(
            desc.size,
            initial.len() as u64 * 4,
            "a buffer's size is its initial words"
        );
        self.create_buffer_bytes(desc, bytemuck::cast_slice(initial))
    }

    pub fn destroy_buffer(&self, buffer: BufferId) {
        self.buffers
            .lock()
            .unwrap()
            .remove(buffer.0, self.frame.load(Ordering::Relaxed));
    }

    pub(super) fn insert_buffer(
        &self,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        offset: u64,
        len: u64,
    ) -> BufferId {
        BufferId(self.buffers.lock().unwrap().insert(Buf {
            buffer: super::Shared(buffer),
            offset,
            len,
        }))
    }

    pub fn upload_vertices(&self, bytes: &[u8]) -> BufferId {
        self.upload_frame(0, bytes)
    }

    pub fn upload_block(&self, kind: BlockKind, bytes: &[u8]) -> BufferId {
        let arena = match kind {
            BlockKind::Uniform => 1,
            BlockKind::Storage => 2,
            BlockKind::Index => 3,
            BlockKind::Staging => 4,
        };
        self.upload_frame(arena, bytes)
    }

    pub fn upload_uniform(&self, words: &[u32]) -> BufferId {
        self.upload_block(BlockKind::Uniform, bytemuck::cast_slice(words))
    }

    /// Write `bytes` into this frame's arena `arena` and name where they landed.
    /// The id retires with the frame, and the arena's slot is reset when the
    /// frame that used it has retired, so nothing here is ever freed one by one.
    fn upload_frame(&self, arena: usize, bytes: &[u8]) -> BufferId {
        let frame = self.frame.load(Ordering::Relaxed);
        let slot = (frame % FRAME_SLOTS) as usize;
        let (buffer, offset) = {
            let mut arenas = self.arenas.lock().unwrap();
            arenas[arena][slot].allocate(&self.device, bytes.len() as u64)
        };
        // SAFETY: the allocation is `bytes.len()` bytes of a shared buffer this
        // frame's arena slot owns exclusively until it retires.
        unsafe {
            let base = buffer.contents().as_ptr().cast::<u8>().add(offset as usize);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), base, bytes.len());
        }
        let id = self.insert_buffer(buffer, offset, bytes.len() as u64);
        self.buffers.lock().unwrap().remove(id.0, frame);
        id
    }

    pub fn create_texture(&self, desc: &TextureDesc) -> TextureViewId {
        self.create_texture_swizzled(desc, TextureSwizzle::Identity)
    }

    pub fn create_texture_swizzled(
        &self,
        desc: &TextureDesc,
        swizzle: TextureSwizzle,
    ) -> TextureViewId {
        let texture = self.new_texture(desc);
        let texture = match swizzle {
            TextureSwizzle::Identity => texture,
            TextureSwizzle::CoverageToWhite => {
                let channels = MTLTextureSwizzleChannels {
                    red: MTLTextureSwizzle::Red,
                    green: MTLTextureSwizzle::Red,
                    blue: MTLTextureSwizzle::Red,
                    alpha: MTLTextureSwizzle::Green,
                };
                // SAFETY: the whole level and slice ranges of the texture just
                // made, in a format the swizzle applies to.
                unsafe {
                    texture.newTextureViewWithPixelFormat_textureType_levels_slices_swizzle(
                        texture.pixelFormat(),
                        texture.textureType(),
                        NSRange::new(0, texture.mipmapLevelCount()),
                        NSRange::new(0, texture.arrayLength().max(1)),
                        channels,
                    )
                }
                .expect("failed to create a swizzled view")
            }
        };
        self.insert_texture(texture)
    }

    fn new_texture(&self, desc: &TextureDesc) -> Retained<ProtocolObject<dyn MTLTexture>> {
        let descriptor = MTLTextureDescriptor::new();
        descriptor.setTextureType(match (desc.dimension, desc.samples) {
            (TextureDimension::D2, SampleCount::Sample1) => MTLTextureType::Type2D,
            (TextureDimension::D2, _) => MTLTextureType::Type2DMultisample,
            (TextureDimension::D2Array, _) => MTLTextureType::Type2DArray,
            (TextureDimension::D3, _) => MTLTextureType::Type3D,
            (TextureDimension::Cube, _) => MTLTextureType::TypeCube,
        });
        descriptor.setPixelFormat(pixel_format(desc.format));
        // SAFETY: plain extents, counts and lengths of a description the graph
        // already validated against the format.
        unsafe {
            descriptor.setWidth(desc.extent[0] as usize);
            descriptor.setHeight(desc.extent[1] as usize);
            descriptor.setDepth(desc.extent[2] as usize);
            descriptor.setMipmapLevelCount(desc.mip_levels as usize);
            if !matches!(desc.dimension, TextureDimension::Cube) {
                descriptor.setArrayLength(match desc.dimension {
                    TextureDimension::D2Array => desc.array_layers as usize,
                    _ => 1,
                });
            }
            descriptor.setSampleCount(match desc.samples {
                SampleCount::Sample1 => 1,
                SampleCount::Sample2 => 2,
                SampleCount::Sample4 => 4,
                SampleCount::Sample8 => 8,
            });
        }
        let mut usage = texture_usage(desc.usage);
        // A depth texture that is written by a transfer is cleared by a render
        // pass here, which needs it to be a render target.
        if desc.format.is_depth() && desc.usage.contains(ImageUsage::TRANSFER_DST) {
            usage |= objc2_metal::MTLTextureUsage::RenderTarget;
        }
        descriptor.setUsage(usage);
        // Only what the graph's usage allows: a memoryless texture that is read
        // back is undefined data (trap T9).
        descriptor.setStorageMode(if desc.memoryless && may_be_memoryless(desc.usage) {
            MTLStorageMode::Memoryless
        } else {
            MTLStorageMode::Private
        });
        self.device
            .newTextureWithDescriptor(&descriptor)
            .expect("failed to allocate a texture")
    }

    fn insert_texture(&self, texture: Retained<ProtocolObject<dyn MTLTexture>>) -> TextureViewId {
        TextureViewId(self.textures.lock().unwrap().insert(super::Shared(texture)))
    }

    /// A persistent single-level 2D texture holding `pixels`, tightly packed.
    pub fn create_texture_2d(
        &self,
        pixels: &[u8],
        extent: [u32; 2],
        format: Format,
    ) -> TextureViewId {
        let desc = TextureDesc {
            format,
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::Sample1,
            usage: ImageUsage::SAMPLED,
            memoryless: false,
        };
        let descriptor = MTLTextureDescriptor::new();
        descriptor.setTextureType(MTLTextureType::Type2D);
        descriptor.setPixelFormat(pixel_format(format));
        // SAFETY: as in `new_texture`.
        unsafe {
            descriptor.setWidth(extent[0] as usize);
            descriptor.setHeight(extent[1] as usize);
        }
        descriptor.setUsage(texture_usage(desc.usage));
        // Shared, so the pixels go in directly: this is a load-time upload of
        // something small, and unified memory makes the copy the whole job.
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .expect("failed to allocate a texture");
        let row = pixels.len() / extent[1].max(1) as usize;
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: extent[0] as usize,
                height: extent[1] as usize,
                depth: 1,
            },
        };
        // SAFETY: `pixels` covers `extent` at `row` bytes a row, and Metal
        // copies it before returning.
        unsafe {
            texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                region,
                0,
                0,
                NonNull::new(pixels.as_ptr() as *mut c_void).expect("a slice is never null"),
                row,
                0,
            );
        }
        self.insert_texture(texture)
    }

    /// A 2D colour texture whose whole mip chain is generated from `pixels`.
    pub fn create_texture_mipped(
        &self,
        pixels: &[u8],
        extent: [u32; 2],
        format: Format,
    ) -> TextureViewId {
        let levels = extent[0].max(extent[1]).max(1).ilog2() + 1;
        let descriptor = MTLTextureDescriptor::new();
        descriptor.setTextureType(MTLTextureType::Type2D);
        descriptor.setPixelFormat(pixel_format(format));
        // SAFETY: as in `new_texture`.
        unsafe {
            descriptor.setWidth(extent[0] as usize);
            descriptor.setHeight(extent[1] as usize);
            descriptor.setMipmapLevelCount(levels as usize);
        }
        descriptor.setUsage(MTLTextureUsage::ShaderRead);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .expect("failed to allocate a texture");
        write_level(&texture, 0, extent, pixels, extent[0] as usize * 4);
        self.generate_mips(&texture);
        self.insert_texture(texture)
    }

    /// A texture from levels already authored, which may be block compressed.
    pub fn create_texture_levels(&self, levels: &crate::gfx::TextureLevels) -> TextureViewId {
        use crate::gfx::TexelFormat;
        let (format, block_bytes) = match levels.format {
            TexelFormat::Rgba8Srgb => (MTLPixelFormat::RGBA8Unorm_sRGB, 0),
            TexelFormat::Rgba8Unorm => (MTLPixelFormat::RGBA8Unorm, 0),
            TexelFormat::Bc5Unorm => (MTLPixelFormat::BC5_RGUnorm, 16),
            TexelFormat::Bc7Unorm => (MTLPixelFormat::BC7_RGBAUnorm, 16),
            TexelFormat::Bc7Srgb => (MTLPixelFormat::BC7_RGBAUnorm_sRGB, 16),
        };
        let descriptor = MTLTextureDescriptor::new();
        descriptor.setTextureType(MTLTextureType::Type2D);
        descriptor.setPixelFormat(format);
        // SAFETY: as in `new_texture`.
        unsafe {
            descriptor.setWidth(levels.width as usize);
            descriptor.setHeight(levels.height as usize);
            descriptor.setMipmapLevelCount(levels.levels.len().max(1));
        }
        descriptor.setUsage(MTLTextureUsage::ShaderRead);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .expect("failed to allocate a texture");
        for (level, bytes) in levels.levels.iter().enumerate() {
            let size = [
                (levels.width >> level).max(1),
                (levels.height >> level).max(1),
            ];
            let row = if block_bytes == 0 {
                size[0] as usize * 4
            } else {
                size[0].div_ceil(4) as usize * block_bytes
            };
            write_level(&texture, level, size, bytes, row);
        }
        self.insert_texture(texture)
    }

    /// A cube of `face` texels a side from six tightly packed faces.
    pub fn create_texture_cube(&self, pixels: &[u8], face: u32, format: Format) -> TextureViewId {
        let descriptor = MTLTextureDescriptor::new();
        descriptor.setTextureType(MTLTextureType::TypeCube);
        descriptor.setPixelFormat(pixel_format(format));
        // SAFETY: as in `new_texture`.
        unsafe {
            descriptor.setWidth(face as usize);
            descriptor.setHeight(face as usize);
        }
        descriptor.setUsage(MTLTextureUsage::ShaderRead);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let texture = self
            .device
            .newTextureWithDescriptor(&descriptor)
            .expect("failed to allocate a cube texture");
        let per_face = pixels.len() / 6;
        let row = per_face / face.max(1) as usize;
        for slice in 0..6 {
            let region = MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width: face as usize,
                    height: face as usize,
                    depth: 1,
                },
            };
            let bytes = &pixels[slice * per_face..(slice + 1) * per_face];
            // SAFETY: `bytes` covers one face at `row` bytes a row.
            unsafe {
                texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    region,
                    0,
                    slice,
                    NonNull::new(bytes.as_ptr() as *mut c_void).expect("a slice is never null"),
                    row,
                    0,
                );
            }
        }
        self.insert_texture(texture)
    }

    /// Fill a depth texture with one value, once.
    pub fn fill_depth(&self, texture: TextureViewId, depth: f32) {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
        use objc2_metal::{MTLLoadAction, MTLRenderPassDescriptor, MTLStoreAction};
        let target = self.texture(texture);
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        let attachment = pass.depthAttachment();
        attachment.setTexture(Some(&target));
        attachment.setLoadAction(MTLLoadAction::Clear);
        attachment.setClearDepth(depth as f64);
        attachment.setStoreAction(MTLStoreAction::Store);
        let command_buffer = self.queue.commandBuffer().expect("no command buffer");
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .expect("failed to open a render encoder");
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
    }

    /// Chain-generate a texture's mip levels with the blit engine.
    fn generate_mips(&self, texture: &Retained<ProtocolObject<dyn MTLTexture>>) {
        use objc2_metal::{
            MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
        };
        let command_buffer = self.queue.commandBuffer().expect("no command buffer");
        let blit = command_buffer
            .blitCommandEncoder()
            .expect("failed to open a blit encoder");
        blit.generateMipmapsForTexture(texture);
        blit.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
    }

    pub fn destroy_texture(&self, view: TextureViewId) {
        self.textures
            .lock()
            .unwrap()
            .remove(view.0, self.frame.load(Ordering::Relaxed));
    }

    pub fn view_extent(&self, view: TextureViewId) -> [u32; 2] {
        let texture = self.texture(view);
        [texture.width() as u32, texture.height() as u32]
    }

    /// One mip level of `texture`, as a texture of its own.
    pub fn mip_view(&self, texture: TextureViewId, level: u32) -> TextureViewId {
        let parent = self.texture(texture);
        // SAFETY: `level` names a level the parent has, which the callers take
        // from the same description the parent was made with.
        let view = unsafe {
            parent.newTextureViewWithPixelFormat_textureType_levels_slices(
                parent.pixelFormat(),
                parent.textureType(),
                NSRange::new(level as usize, 1),
                NSRange::new(0, parent.arrayLength().max(1)),
            )
        }
        .expect("failed to create a mip view");
        self.insert_texture(view)
    }

    /// One face of one level of a cube or array texture, as a 2D texture.
    pub fn face_view(&self, texture: TextureViewId, face: u32, level: u32) -> TextureViewId {
        let parent = self.texture(texture);
        // SAFETY: as in `mip_view`, with `face` a slice the parent has.
        let view = unsafe {
            parent.newTextureViewWithPixelFormat_textureType_levels_slices(
                parent.pixelFormat(),
                MTLTextureType::Type2D,
                NSRange::new(level as usize, 1),
                NSRange::new(face as usize, 1),
            )
        }
        .expect("failed to create a face view");
        self.insert_texture(view)
    }

    /// The texels of level 0 of a 2D texture, tightly packed, once the GPU has
    /// finished with it. It stalls, which is what a capture wants.
    pub(crate) fn read_texture(&self, view: TextureViewId) -> Vec<u8> {
        self.read_texture_slice(view, 0, 0)
    }

    /// One slice and level of a texture, tightly packed.
    pub(crate) fn read_texture_slice(
        &self,
        view: TextureViewId,
        slice: usize,
        level: usize,
    ) -> Vec<u8> {
        use objc2_metal::{
            MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
        };
        let texture = self.texture(view);
        let (width, height) = (
            (texture.width() >> level).max(1),
            (texture.height() >> level).max(1),
        );
        let texel = match texture.pixelFormat() {
            f if f == pixel_format(Format::Rgba32Float) => 16,
            f if f == pixel_format(Format::Rgba16Float) => 8,
            f if f == pixel_format(Format::R8Unorm) => 1,
            _ => 4,
        };
        let row = width * texel;
        let buffer = self
            .device
            .newBufferWithLength_options(row * height, MTLResourceOptions::StorageModeShared)
            .expect("failed to allocate a readback buffer");
        let command_buffer = self.queue.commandBuffer().expect("no command buffer");
        let blit = command_buffer
            .blitCommandEncoder()
            .expect("failed to open a blit encoder");
        // SAFETY: the origin and size lie inside level 0, and the buffer holds
        // `row * height` bytes.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                &texture,
                slice,
                level,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width, height, depth: 1 },
                &buffer,
                0,
                row,
                row * height,
            );
        }
        blit.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        // SAFETY: a shared buffer of `row * height` bytes the GPU has finished
        // writing.
        unsafe {
            std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), row * height)
                .to_vec()
        }
    }

    /// Register a texture the driver did not make (a drawable's) for the frame
    /// being recorded; the id retires with the frame.
    pub(crate) fn import_texture(
        &self,
        texture: Retained<ProtocolObject<dyn MTLTexture>>,
    ) -> TextureViewId {
        let mut pool = self.textures.lock().unwrap();
        let index = pool.insert(super::Shared(texture));
        pool.remove(index, self.frame.load(Ordering::Relaxed));
        TextureViewId(index)
    }

    /// Whether a texture lives in tile memory only.
    pub(super) fn is_memoryless(&self, view: TextureViewId) -> bool {
        use objc2_metal::MTLResource;
        self.texture(view).storageMode() == MTLStorageMode::Memoryless
    }

    pub(super) fn texture(&self, view: TextureViewId) -> Retained<ProtocolObject<dyn MTLTexture>> {
        self.textures
            .lock()
            .unwrap()
            .get(view.0)
            .unwrap_or_else(|| panic!("texture {} is not live", view.0))
            .0
            .clone()
    }

    pub(super) fn buffer(&self, buffer: BufferId) -> Buf {
        let pool = self.buffers.lock().unwrap();
        let found = pool
            .get(buffer.0)
            .unwrap_or_else(|| panic!("buffer {} is not live", buffer.0));
        Buf {
            buffer: super::Shared(found.buffer.0.clone()),
            offset: found.offset,
            len: found.len,
        }
    }

    pub(super) fn sampler(
        &self,
        sampler: SamplerId,
    ) -> Retained<ProtocolObject<dyn objc2_metal::MTLSamplerState>> {
        self.samplers
            .lock()
            .unwrap()
            .get(sampler.0)
            .unwrap_or_else(|| panic!("sampler {} is not live", sampler.0))
            .0
            .clone()
    }
}

/// One frame slot's bump allocator over shared buffer pages.
///
/// Pages are kept for the life of the driver and the cursor is reset, so a
/// steady scene allocates nothing after its first few frames; a frame that
/// outgrows its pages adds one, sized to fit.
pub(super) struct Arena {
    pages: Vec<super::Shared<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    page: usize,
    cursor: u64,
}

const PAGE_BYTES: u64 = 4 << 20;

impl Arena {
    pub(super) fn new() -> Self {
        Self {
            pages: Vec::new(),
            page: 0,
            cursor: 0,
        }
    }

    pub(super) fn reset(&mut self) {
        self.page = 0;
        self.cursor = 0;
    }

    pub(super) fn allocate(
        &mut self,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        len: u64,
    ) -> (Retained<ProtocolObject<dyn MTLBuffer>>, u64) {
        let len = len.max(1);
        loop {
            if let Some(page) = self.pages.get(self.page) {
                let start = self.cursor.next_multiple_of(BIND_ALIGNMENT);
                if start + len <= page.0.length() as u64 {
                    self.cursor = start + len;
                    return (page.0.clone(), start);
                }
                self.page += 1;
                self.cursor = 0;
                continue;
            }
            let size = PAGE_BYTES.max(len.next_multiple_of(BIND_ALIGNMENT));
            let page = device
                .newBufferWithLength_options(size as usize, MTLResourceOptions::StorageModeShared)
                .expect("failed to allocate an arena page");
            self.pages.push(super::Shared(page));
        }
    }
}

/// Write one level of a texture from tightly packed bytes.
fn write_level(
    texture: &Retained<ProtocolObject<dyn MTLTexture>>,
    level: usize,
    size: [u32; 2],
    bytes: &[u8],
    row: usize,
) {
    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: size[0] as usize,
            height: size[1] as usize,
            depth: 1,
        },
    };
    // SAFETY: `bytes` covers the level at `row` bytes a row, and Metal copies it
    // before returning.
    unsafe {
        texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
            region,
            level,
            0,
            NonNull::new(bytes.as_ptr() as *mut c_void).expect("a slice is never null"),
            row,
            0,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn driver() -> MtlDriver {
        MtlDriver::new().expect("this machine runs the Metal backend")
    }

    fn read(driver: &MtlDriver, id: BufferId, len: usize) -> Vec<u8> {
        let buf = driver.buffer(id);
        // SAFETY: a shared buffer of at least `len` bytes past `offset`.
        unsafe {
            let base = buf.buffer.0.contents().as_ptr().cast::<u8>();
            std::slice::from_raw_parts(base.add(buf.offset as usize), len).to_vec()
        }
    }

    #[test]
    #[ignore = "needs a Metal device"]
    fn a_per_frame_block_holds_what_was_written_at_an_aligned_offset() {
        let driver = driver();
        driver.begin_frame(0);
        let first = driver.upload_block(BlockKind::Uniform, &[1, 2, 3, 4, 5]);
        let second = driver.upload_block(BlockKind::Uniform, &[9; 300]);
        assert_eq!(read(&driver, first, 5), [1, 2, 3, 4, 5]);
        assert_eq!(read(&driver, second, 300), vec![9; 300]);
        for id in [first, second] {
            assert_eq!(driver.buffer(id).offset % BIND_ALIGNMENT, 0);
        }
        assert_ne!(driver.buffer(first).offset, driver.buffer(second).offset);
    }

    /// The arena slot a frame used is reused two frames on, not before: an id
    /// from frame 0 still reads what it was given in frame 1.
    #[test]
    #[ignore = "needs a Metal device"]
    fn a_frames_block_survives_the_next_frame_and_is_reused_after() {
        let driver = driver();
        driver.begin_frame(0);
        let kept = driver.upload_block(BlockKind::Storage, &[7; 16]);
        let offset = driver.buffer(kept).offset;
        driver.begin_frame(1);
        driver.upload_block(BlockKind::Storage, &[8; 16]);
        assert_eq!(read(&driver, kept, 16), vec![7; 16]);
        driver.begin_frame(2);
        let reused = driver.upload_block(BlockKind::Storage, &[6; 16]);
        assert_eq!(driver.buffer(reused).offset, offset);
    }

    #[test]
    #[ignore = "needs a Metal device"]
    fn a_texture_reports_its_extent_and_a_mip_view_its_level() {
        let driver = driver();
        let id = driver.create_texture(&TextureDesc {
            format: Format::Rgba16Float,
            dimension: TextureDimension::D2,
            extent: [64, 32, 1],
            mip_levels: 4,
            array_layers: 1,
            samples: SampleCount::Sample1,
            usage: ImageUsage::SAMPLED | ImageUsage::STORAGE,
            memoryless: false,
        });
        assert_eq!(driver.view_extent(id), [64, 32]);
        let level = driver.mip_view(id, 2);
        assert_eq!(driver.view_extent(level), [16, 8]);
    }

    #[test]
    #[ignore = "needs a Metal device"]
    fn a_two_pixel_texture_uploads_through_shared_storage() {
        let driver = driver();
        let id = driver.create_texture_2d(&[1, 2, 3, 4, 5, 6, 7, 8], [2, 1], Format::Rgba8Unorm);
        assert_eq!(driver.view_extent(id), [2, 1]);
    }
}
