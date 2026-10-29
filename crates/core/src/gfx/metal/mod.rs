//! The native Metal driver, beside the Vulkan one behind the same RHI.
//!
//! Brought up in the order `docs/metal-backend-plan.md` M4 gives: the device and
//! its capability gate first, then resources, then pipelines from the embedded
//! MSL, then the encoder. A method that is still `unimplemented!` names itself
//! in the panic, so a pass that reaches one says which operation it needs next.

mod bind;
mod convert;
mod encoder;
mod pipeline;
mod resources;

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
use crate::gfx::rhi::{Encoder, MeshSpan, Pool, TextureViewId};

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

/// A shared Metal object, by its protocol.
pub(crate) type Held<P> = Shared<Retained<ProtocolObject<P>>>;
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
    textures: Mutex<Pool<Held<dyn MTLTexture>>>,
    buffers: Mutex<Pool<Buf>>,
    samplers: Mutex<Pool<Held<dyn MTLSamplerState>>>,
    /// Per-frame data, one arena per kind per frame slot: vertices, uniforms,
    /// storage, indices, staging, argument buffers.
    arenas: Mutex<[[Arena; FRAME_SLOTS as usize]; 6]>,
    pipelines: Mutex<Pool<pipeline::Pipe>>,
    functions: Mutex<pipeline::Functions>,
    bind_groups: Mutex<Pool<bind::Group>>,
}

pub struct MtlEncoder<'a> {
    driver: &'a MtlDriver,
    state: &'a mut encoder::State,
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
            pipelines: Mutex::new(Pool::new(FRAME_SLOTS)),
            functions: Mutex::new(Default::default()),
            bind_groups: Mutex::new(Pool::new(FRAME_SLOTS)),
        })
    }

    /// Record with `record` on a fresh encoder and wait for the GPU to finish.
    pub fn record_blocking(&self, record: impl FnOnce(&mut Encoder)) {
        let mut state = encoder::State::new(self);
        {
            let mut encoder = Encoder::Metal(MtlEncoder::new(self, &mut state));
            record(&mut encoder);
        }
        state.finish_blocking();
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
        self.pipelines.lock().unwrap().collect(frame);
        self.bind_groups.lock().unwrap().collect(frame);
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

    pub fn create_texture_cube(&self, pixels: &[u8], face: u32, format: Format) -> TextureViewId {
        unimplemented!("metal: create_texture_cube")
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
}

#[allow(
    unused_variables,
    reason = "operations land one at a time; a stub names itself in its panic"
)]
impl MtlEncoder<'_> {}
