//! AMD FSR through its signed runtime: the five entry points of AMD's C API,
//! the descriptors they are handed, and a context that upscales one frame into
//! the command buffer the frame is recording.
//!
//! **Loaded, not linked.** FSR 4 ships as AMD-signed DLLs and nothing else —
//! it cannot be built from source — and a machine without them still has to
//! run the engine on its own resolve. So the loader DLL is opened at run time,
//! from `ORRIN_FSR_DIR` or beside the executable, and it finds the upscaler
//! DLL beside itself (the lookup order in AMD's `ffx-api.md`).
//!
//! **Every `unsafe` of the integration is here**: the `#[repr(C)]` mirrors of
//! AMD's structs, the function pointers, and the calls through them. Each
//! mirror names the header it is transcribed from, and the test at the bottom
//! pins its size and offsets to what MSVC computed from that header, because a
//! field four bytes out corrupts memory without failing anything.
//!
//! **What a dispatch leaves behind.** The runtime moves each resource from the
//! state it is handed to the one it needs, and back again before it returns
//! (`UnregisterResourcesVK` in the open Vulkan backend). So the graph node
//! declares the accesses whose layouts those states name, and the plan holds
//! on both sides of the call without a barrier of the node's own. It binds
//! its own pipelines and sets, which [`Recorder::record_foreign`] says nothing
//! after it relies on.
//!
//! **One call at a time.** A context must not be entered from two threads at
//! once; [`UpscaleContext`] is `Send` and not `Sync`, and its owner keeps it
//! behind a mutex, which is what lets the dispatch record on a pool worker.

use std::ffi::{CStr, c_char, c_void};
use std::path::Path;
use std::sync::Arc;

use ash::vk::{self, Handle};
use vulkano::VulkanObject;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::{Image, ImageUsage};

use super::record::Recorder;

/// `ffxReturnCode_t`, `ffx_api.h`. Zero is `FFX_API_RETURN_OK`.
type ReturnCode = u32;

/// `ffxContext`, `ffx_api.h`: an opaque pointer the runtime owns.
type RawContext = *mut c_void;

/// `ffxApiMessage`, `ffx_api.h`. `wchar_t` is UTF-16 on Windows.
type MessageFn = unsafe extern "C" fn(kind: u32, message: *const u16);

type CreateContextFn =
    unsafe extern "C" fn(*mut RawContext, *mut Header, *const c_void) -> ReturnCode;
type DestroyContextFn = unsafe extern "C" fn(*mut RawContext, *const c_void) -> ReturnCode;
type ConfigureFn = unsafe extern "C" fn(*mut RawContext, *const Header) -> ReturnCode;
type QueryFn = unsafe extern "C" fn(*mut RawContext, *mut Header) -> ReturnCode;
type DispatchFn = unsafe extern "C" fn(*mut RawContext, *const Header) -> ReturnCode;

// Descriptor types: `ffx_api.h`, `vk/ffx_api_vk.h` and `ffx_upscale.h`, with
// their `MAKE_*_SUB_ID` macros expanded.
const CONFIGURE_GLOBAL_DEBUG: u64 = 0x7;
const QUERY_PROVIDER_VERSION: u64 = 0x6;
const CREATE_BACKEND_VK: u64 = 0x0100_0003;
const CREATE_UPSCALE: u64 = 0x0001_0000;
const DISPATCH_UPSCALE: u64 = 0x0001_0001;
const QUERY_RENDER_RESOLUTION: u64 = 0x0001_0003;
const QUERY_JITTER_PHASE_COUNT: u64 = 0x0001_0004;
const QUERY_GPU_MEMORY: u64 = 0x0001_0008;
const CREATE_UPSCALE_VERSION: u64 = 0x0001_000b;

/// `FFX_UPSCALER_VERSION`, `ffx_upscale.h`: the API this was written against,
/// which the runtime checks a context's creation against.
const UPSCALER_VERSION: u32 = (4 << 22) | (1 << 12) | 2;

/// `FFX_API_EFFECT_ID_UPSCALE` and `FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_WARNINGS`.
const EFFECT_UPSCALE: u64 = 0x0001_0000;
const DEBUG_LEVEL_WARNINGS: u32 = 2;

/// `FfxApiCreateContextUpscaleFlags`, `ffx_upscale.h`.
pub(super) const ENABLE_HIGH_DYNAMIC_RANGE: u32 = 1 << 0;
pub(super) const ENABLE_DEPTH_INVERTED: u32 = 1 << 3;
pub(super) const ENABLE_AUTO_EXPOSURE: u32 = 1 << 5;
pub(super) const ENABLE_DEBUG_CHECKING: u32 = 1 << 7;
pub(super) const ENABLE_DEBUG_VISUALIZATION: u32 = 1 << 9;

/// `FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW`, `ffx_upscale.h`.
const FLAG_DRAW_DEBUG_VIEW: u32 = 1 << 0;

/// `FfxApiResourceState`, `ffx_api_types.h`, and the layouts the open backend
/// maps them to: `SHADER_READ_ONLY_OPTIMAL` and `GENERAL`.
const STATE_COMPUTE_READ: u32 = 1 << 2;
const STATE_UNORDERED_ACCESS: u32 = 1 << 1;

/// `FFX_API_RESOURCE_TYPE_TEXTURE2D` and `FfxApiResourceUsage`,
/// `ffx_api_types.h`.
const RESOURCE_TEXTURE_2D: u32 = 2;
const USAGE_UAV: u32 = 1 << 1;
const USAGE_DEPTH_TARGET: u32 = 1 << 2;

/// `ffxApiHeader`, `ffx_api.h`.
#[repr(C)]
struct Header {
    kind: u64,
    next: *mut Header,
}

impl Header {
    fn new(kind: u64) -> Self {
        Self {
            kind,
            next: std::ptr::null_mut(),
        }
    }
}

/// `ffxConfigureDescGlobalDebug`, `ffx_api.h`.
#[repr(C)]
struct ConfigureGlobalDebug {
    header: Header,
    effect: u64,
    message: Option<MessageFn>,
    level: u32,
}

/// `ffxQueryGetProviderVersion`, `ffx_api.h`.
#[repr(C)]
struct QueryProviderVersion {
    header: Header,
    id: u64,
    name: *const c_char,
}

/// `VkDeviceContext` inside `ffxCreateBackendVKDesc`, `vk/ffx_api_vk.h`.
#[repr(C)]
struct CreateBackendVk {
    header: Header,
    device: vk::Device,
    physical_device: vk::PhysicalDevice,
    get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
}

/// `FfxApiDimensions2D` and `FfxApiFloatCoords2D`, `ffx_api_types.h`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Dimensions {
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Coords {
    x: f32,
    y: f32,
}

/// `ffxCreateContextDescUpscale`, `ffx_upscale.h`.
#[repr(C)]
struct CreateUpscale {
    header: Header,
    flags: u32,
    max_render_size: Dimensions,
    max_upscale_size: Dimensions,
    message: Option<MessageFn>,
}

/// `ffxCreateContextDescUpscaleVersion`, `ffx_upscale.h`.
#[repr(C)]
struct CreateUpscaleVersion {
    header: Header,
    version: u32,
}

/// `FfxApiResourceDescription`, `ffx_api_types.h`. The three unions are
/// written as their texture halves, the only ones this passes.
#[repr(C)]
#[derive(Clone, Copy)]
struct ResourceDescription {
    kind: u32,
    format: u32,
    width: u32,
    height: u32,
    depth: u32,
    mip_count: u32,
    flags: u32,
    usage: u32,
}

/// `FfxApiResource`, `ffx_api_types.h`. On Vulkan `resource` is the `VkImage`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Resource {
    resource: *mut c_void,
    description: ResourceDescription,
    state: u32,
}

impl Resource {
    /// An input left out, as AMD's sample passes one.
    const NONE: Resource = Resource {
        resource: std::ptr::null_mut(),
        description: ResourceDescription {
            kind: 0,
            format: 0,
            width: 0,
            height: 0,
            depth: 0,
            mip_count: 0,
            flags: 0,
            usage: 0,
        },
        state: STATE_COMPUTE_READ,
    };

    /// What `ffxApiGetImageResourceDescriptionVK` in `vk/ffx_api_vk.h` makes
    /// of an image, for the formats the frame hands over.
    fn image(image: &Image, state: u32) -> Self {
        let depth = image.format() == Format::D32_SFLOAT;
        let mut usage = 0;
        if depth {
            usage |= USAGE_DEPTH_TARGET;
        }
        if image.usage().intersects(ImageUsage::STORAGE) {
            usage |= USAGE_UAV;
        }
        let [width, height, _] = image.extent();
        Self {
            resource: image.handle().as_raw() as *mut c_void,
            description: ResourceDescription {
                kind: RESOURCE_TEXTURE_2D,
                format: surface_format(image.format()),
                width,
                height,
                depth: image.array_layers(),
                mip_count: image.mip_levels(),
                flags: 0,
                usage,
            },
            state,
        }
    }
}

/// `FfxApiSurfaceFormat`, `ffx_api_types.h`, for the formats the frame hands
/// over; zero, `UNKNOWN`, for anything else, which the runtime refuses.
fn surface_format(format: Format) -> u32 {
    match format {
        Format::R16G16B16A16_SFLOAT => 4,
        Format::B10G11R11_UFLOAT_PACK32 => 16,
        Format::R16G16_SFLOAT => 18,
        Format::R8_UNORM => 25,
        Format::D32_SFLOAT | Format::R32_SFLOAT => 28,
        _ => 0,
    }
}

/// `ffxDispatchDescUpscale`, `ffx_upscale.h`.
#[repr(C)]
struct DispatchUpscale {
    header: Header,
    command_list: *mut c_void,
    color: Resource,
    depth: Resource,
    motion_vectors: Resource,
    exposure: Resource,
    reactive: Resource,
    transparency_and_composition: Resource,
    output: Resource,
    jitter_offset: Coords,
    motion_vector_scale: Coords,
    render_size: Dimensions,
    upscale_size: Dimensions,
    enable_sharpening: bool,
    sharpness: f32,
    frame_time_delta: f32,
    pre_exposure: f32,
    reset: bool,
    camera_near: f32,
    camera_far: f32,
    camera_fov_angle_vertical: f32,
    view_space_to_meters_factor: f32,
    flags: u32,
}

/// `ffxQueryDescUpscaleGetRenderResolutionFromQualityMode`, `ffx_upscale.h`.
#[repr(C)]
struct QueryRenderResolution {
    header: Header,
    display_width: u32,
    display_height: u32,
    quality_mode: u32,
    render_width: *mut u32,
    render_height: *mut u32,
}

/// `ffxQueryDescUpscaleGetJitterPhaseCount`, `ffx_upscale.h`.
#[repr(C)]
struct QueryJitterPhaseCount {
    header: Header,
    render_width: u32,
    display_width: u32,
    phase_count: *mut i32,
}

/// `FfxApiEffectMemoryUsage`, `ffx_api_types.h`.
#[repr(C)]
#[derive(Default)]
struct MemoryUsage {
    total: u64,
    aliasable: u64,
}

/// `ffxQueryDescUpscaleGetGPUMemoryUsage`, `ffx_upscale.h`.
#[repr(C)]
struct QueryGpuMemory {
    header: Header,
    usage: *mut MemoryUsage,
}

/// Where the runtime's messages go: the engine's log.
///
/// # Safety
///
/// The runtime calls this with a null-terminated UTF-16 string, or null.
unsafe extern "C" fn message(kind: u32, text: *const u16) {
    if text.is_null() {
        return;
    }
    // SAFETY: the runtime hands a null-terminated `wchar_t` string, which on
    // Windows is UTF-16, and it outlives the call; the walk stops at the
    // terminator.
    let text = unsafe {
        let length = (0..).take_while(|&at| *text.add(at) != 0).count();
        String::from_utf16_lossy(std::slice::from_raw_parts(text, length))
    };
    // `FFX_API_MESSAGE_TYPE_ERROR` is zero; everything else is a warning.
    if kind == 0 {
        tracing::error!(target: "fsr", "{text}");
    } else {
        tracing::warn!(target: "fsr", "{text}");
    }
}

/// The loader DLL and its five entry points.
pub(super) struct Runtime {
    create: CreateContextFn,
    destroy: DestroyContextFn,
    query: QueryFn,
    dispatch: DispatchFn,
    /// Held so the entry points above stay mapped; a context holds this.
    _library: libloading::Library,
}

/// The loader's file name, `ffx-api.md`. The upscaler DLL is found by the
/// loader rather than named here.
pub(super) const LOADER_DLL: &str = "amd_fidelityfx_loader_vk.dll";

impl Runtime {
    /// Open the loader in `dir`. The search is confined to that directory and
    /// the system's, so a DLL of the same name in the working directory is
    /// never the one loaded.
    pub(super) fn load(dir: &Path) -> Result<Self, String> {
        use libloading::os::windows::{
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
        };
        let path = std::path::absolute(dir.join(LOADER_DLL))
            .map_err(|e| format!("{}: {e}", dir.display()))?;
        // SAFETY: this runs the DLL's initialisers. The file is AMD's FFX
        // loader, which `ffx-api.md` has an application load with
        // `LoadLibrary` exactly so, and its path is one the user named or
        // the executable's own directory.
        let library: libloading::Library = unsafe {
            libloading::os::windows::Library::load_with_flags(
                &path,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        }
        .map_err(|e| format!("{}: {e}", path.display()))?
        .into();

        // SAFETY: each symbol is looked up under the name `ffx_api.h` exports
        // it as, and read as the signature that header declares for it (the
        // `Pfn*` typedef beside each). The pointers are copied out, and the
        // library they point into is kept in `self` for as long as they are.
        let runtime = unsafe {
            let create = entry::<CreateContextFn>(&library, b"ffxCreateContext\0")?;
            let destroy = entry::<DestroyContextFn>(&library, b"ffxDestroyContext\0")?;
            let configure = entry::<ConfigureFn>(&library, b"ffxConfigure\0")?;
            let query = entry::<QueryFn>(&library, b"ffxQuery\0")?;
            let dispatch = entry::<DispatchFn>(&library, b"ffxDispatch\0")?;

            // The runtime's own complaints — which provider it chose, why it
            // chose none — reach the log in a build that checks, as the
            // context's do.
            if cfg!(debug_assertions) {
                let debug = ConfigureGlobalDebug {
                    header: Header::new(CONFIGURE_GLOBAL_DEBUG),
                    effect: EFFECT_UPSCALE,
                    message: Some(message),
                    level: DEBUG_LEVEL_WARNINGS,
                };
                configure(std::ptr::null_mut(), &debug.header);
            }

            Runtime {
                create,
                destroy,
                query,
                dispatch,
                _library: library,
            }
        };
        Ok(runtime)
    }
}

/// Copy one entry point out of `library`.
///
/// # Safety
///
/// `T` must be the signature the symbol `name` was exported with.
unsafe fn entry<T: Copy>(library: &libloading::Library, name: &[u8]) -> Result<T, String> {
    // SAFETY: the caller's.
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|e| {
            format!(
                "the FSR loader has no {}: {e}",
                String::from_utf8_lossy(&name[..name.len() - 1])
            )
        })
}

/// What a context is created for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct ContextDesc {
    /// The largest extent the frame is rendered at.
    pub max_render: [u32; 2],
    /// The extent it is upscaled to.
    pub output: [u32; 2],
    /// `ENABLE_*` above.
    pub flags: u32,
}

/// One frame's inputs to the upscale.
pub(super) struct Dispatch<'a> {
    pub color: &'a Arc<Image>,
    pub depth: &'a Arc<Image>,
    pub motion: &'a Arc<Image>,
    pub output: &'a Arc<Image>,
    /// In render pixels, as `ffxDispatchDescUpscale::jitterOffset` takes it.
    pub jitter: [f32; 2],
    /// What turns a motion vector into render pixels from this frame to the
    /// last.
    pub motion_scale: [f32; 2],
    pub render: [u32; 2],
    pub upscale: [u32; 2],
    pub frame_ms: f32,
    pub reset: bool,
    pub near: f32,
    pub far: f32,
    pub fov_y: f32,
    pub debug_view: bool,
}

/// An FSR upscale context on the engine's device.
pub(super) struct UpscaleContext {
    raw: RawContext,
    runtime: Arc<Runtime>,
    /// The queue every dispatch is submitted to, and so the one that has to
    /// be idle before the context's resources are freed. Also what keeps the
    /// device alive for as long as the context is.
    queue: Arc<Queue>,
}

// SAFETY: a context is the runtime's heap object, with no affinity to the
// thread that made it — the runtime keeps nothing thread-local for an upscale
// context, and AMD's sample dispatches from its render thread — so it may move
// between threads. It is not `Sync`: its owner serialises every call.
unsafe impl Send for UpscaleContext {}

impl UpscaleContext {
    /// Create a context on `queue`'s device, for frames `desc` describes.
    pub(super) fn new(
        runtime: &Arc<Runtime>,
        queue: &Arc<Queue>,
        desc: ContextDesc,
    ) -> Result<Self, String> {
        let device = queue.device();
        let mut backend = CreateBackendVk {
            header: Header::new(CREATE_BACKEND_VK),
            device: device.handle(),
            physical_device: device.physical_device().handle(),
            get_device_proc_addr: device.instance().fns().v1_0.get_device_proc_addr,
        };
        let mut version = CreateUpscaleVersion {
            header: Header::new(CREATE_UPSCALE_VERSION),
            version: UPSCALER_VERSION,
        };
        version.header.next = &mut backend.header;
        let mut create = CreateUpscale {
            header: Header::new(CREATE_UPSCALE),
            flags: desc.flags,
            max_render_size: Dimensions {
                width: desc.max_render[0],
                height: desc.max_render[1],
            },
            max_upscale_size: Dimensions {
                width: desc.output[0],
                height: desc.output[1],
            },
            message: Some(message),
        };
        create.header.next = &mut version.header;

        let mut raw: RawContext = std::ptr::null_mut();
        // SAFETY: the chain is three descriptors laid out as `ffx_upscale.h`
        // and `vk/ffx_api_vk.h` declare them, each typed by its header and
        // alive across the call; the device and physical device are the
        // engine's live ones, and the proc-address function is the instance's
        // own. Null allocation callbacks mean the runtime's `malloc`.
        let code = unsafe { (runtime.create)(&mut raw, &mut create.header, std::ptr::null()) };
        if code != 0 || raw.is_null() {
            return Err(format!("ffxCreateContext returned {}", describe(code)));
        }
        Ok(Self {
            raw,
            runtime: Arc::clone(runtime),
            queue: Arc::clone(queue),
        })
    }

    /// The name of the implementation the runtime picked, such as
    /// `FSR 4.1.2`: which generation runs is the device's and the driver's
    /// choice, not the engine's.
    pub(super) fn provider(&mut self) -> Option<String> {
        let mut query = QueryProviderVersion {
            header: Header::new(QUERY_PROVIDER_VERSION),
            id: 0,
            name: std::ptr::null(),
        };
        // SAFETY: a query the context answers by writing the two fields after
        // the header, which are laid out as `ffx_api.h` declares them.
        let code = unsafe { (self.runtime.query)(&mut self.raw, &mut query.header) };
        if code != 0 || query.name.is_null() {
            return None;
        }
        // SAFETY: the runtime set `name` to a null-terminated string of its
        // own, which is copied out at once: `ffx-api.md` warns that a later
        // query may overwrite it.
        let name = unsafe { CStr::from_ptr(query.name) };
        Some(format!("{} (0x{:x})", name.to_string_lossy(), query.id))
    }

    /// The render extent the runtime would pick for `output` in quality mode
    /// `mode` (`FfxApiUpscaleQualityMode`).
    pub(super) fn render_extent(&mut self, output: [u32; 2], mode: u32) -> Option<[u32; 2]> {
        let (mut width, mut height) = (0u32, 0u32);
        let mut query = QueryRenderResolution {
            header: Header::new(QUERY_RENDER_RESOLUTION),
            display_width: output[0],
            display_height: output[1],
            quality_mode: mode,
            render_width: &mut width,
            render_height: &mut height,
        };
        // SAFETY: the two out-pointers are to locals alive across the call,
        // and the descriptor is laid out as `ffx_upscale.h` declares it.
        let code = unsafe { (self.runtime.query)(&mut self.raw, &mut query.header) };
        (code == 0).then_some([width, height])
    }

    /// How long the runtime would make the jitter sequence from `render` to
    /// `output` pixels wide.
    pub(super) fn jitter_phases(&mut self, render: u32, output: u32) -> Option<u32> {
        let mut phases = 0i32;
        let mut query = QueryJitterPhaseCount {
            header: Header::new(QUERY_JITTER_PHASE_COUNT),
            render_width: render,
            display_width: output,
            phase_count: &mut phases,
        };
        // SAFETY: as `render_extent`, with one out-pointer.
        let code = unsafe { (self.runtime.query)(&mut self.raw, &mut query.header) };
        (code == 0).then_some(phases.max(0) as u32)
    }

    /// What the context holds on the device, by the runtime's own count.
    pub(super) fn gpu_memory(&mut self) -> Option<u64> {
        let mut usage = MemoryUsage::default();
        let mut query = QueryGpuMemory {
            header: Header::new(QUERY_GPU_MEMORY),
            usage: &mut usage,
        };
        // SAFETY: as `render_extent`, with one out-pointer.
        let code = unsafe { (self.runtime.query)(&mut self.raw, &mut query.header) };
        (code == 0).then_some(usage.total)
    }

    /// Record the upscale of `frame` into `recorder`.
    ///
    /// Each input must be in `SHADER_READ_ONLY_OPTIMAL` and the output in
    /// `GENERAL`, which is what the node declares them as; they are in those
    /// layouts again when this returns.
    pub(super) fn dispatch(
        &mut self,
        recorder: &mut Recorder,
        frame: &Dispatch<'_>,
    ) -> Result<(), String> {
        let mut desc = DispatchUpscale {
            header: Header::new(DISPATCH_UPSCALE),
            command_list: std::ptr::null_mut(),
            color: Resource::image(frame.color, STATE_COMPUTE_READ),
            depth: Resource::image(frame.depth, STATE_COMPUTE_READ),
            motion_vectors: Resource::image(frame.motion, STATE_COMPUTE_READ),
            exposure: Resource::NONE,
            reactive: Resource::NONE,
            transparency_and_composition: Resource::NONE,
            output: Resource::image(frame.output, STATE_UNORDERED_ACCESS),
            jitter_offset: Coords {
                x: frame.jitter[0],
                y: frame.jitter[1],
            },
            motion_vector_scale: Coords {
                x: frame.motion_scale[0],
                y: frame.motion_scale[1],
            },
            render_size: Dimensions {
                width: frame.render[0],
                height: frame.render[1],
            },
            upscale_size: Dimensions {
                width: frame.upscale[0],
                height: frame.upscale[1],
            },
            enable_sharpening: false,
            sharpness: 0.0,
            frame_time_delta: frame.frame_ms,
            pre_exposure: 1.0,
            reset: frame.reset,
            camera_near: frame.near,
            camera_far: frame.far,
            camera_fov_angle_vertical: frame.fov_y,
            view_space_to_meters_factor: 1.0,
            flags: if frame.debug_view {
                FLAG_DRAW_DEBUG_VIEW
            } else {
                0
            },
        };
        let images = [frame.color, frame.depth, frame.motion, frame.output];
        let code = recorder.record_foreign(&images, |command_buffer| {
            desc.command_list = command_buffer.as_raw() as *mut c_void;
            // SAFETY: the command buffer is recording — the recorder owns it
            // and is borrowed for the call — and is in no rendering instance,
            // since this is a compute node. Every image named is held for the
            // frame by the recorder and is in the layout its `state` names,
            // which is the node's declared access. The descriptor is laid out
            // as `ffx_upscale.h` declares it, and the context is not entered
            // anywhere else meanwhile: `&mut self`.
            unsafe { (self.runtime.dispatch)(&mut self.raw, &desc.header) }
        });
        if code != 0 {
            return Err(format!("ffxDispatch returned {}", describe(code)));
        }
        Ok(())
    }
}

impl Drop for UpscaleContext {
    fn drop(&mut self) {
        // Every dispatch went to this queue, so once it is idle nothing the
        // context owns is in use.
        if let Err(e) = self.queue.with(|mut queue| queue.wait_idle()) {
            tracing::error!("FSR: the queue would not idle before the context was destroyed: {e}");
            return;
        }
        // SAFETY: the context was made by `ffxCreateContext` with null
        // callbacks, which are what this passes back; nothing on the device
        // uses it any more, and nothing holds it but `self`.
        let code = unsafe { (self.runtime.destroy)(&mut self.raw, std::ptr::null()) };
        if code != 0 {
            tracing::warn!("FSR: ffxDestroyContext returned {}", describe(code));
        }
    }
}

/// `FfxApiReturnCodes`, `ffx_api.h`.
fn describe(code: ReturnCode) -> String {
    let name = match code {
        0 => "OK",
        1 => "ERROR",
        2 => "ERROR_UNKNOWN_DESCTYPE",
        3 => "ERROR_RUNTIME_ERROR",
        4 => "NO_PROVIDER",
        5 => "ERROR_MEMORY",
        6 => "ERROR_PARAMETER",
        7 => "PROVIDER_NO_SUPPORT_NEW_DESCTYPE",
        _ => "an unknown code",
    };
    format!("{code} ({name})")
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, offset_of, size_of};

    use super::*;

    /// Every struct handed across the boundary has the size and field offsets
    /// AMD's headers give it on x64 Windows.
    ///
    /// The numbers were printed by a C++ file that includes `ffx_api.h`,
    /// `ffx_api_types.h`, `vk/ffx_api_vk.h` and `ffx_upscale.h` from FSR SDK
    /// 2.4 (upscaler 4.1.2) by path and prints `sizeof`, `alignof` and
    /// `offsetof` for each, built with MSVC 2022 for x64 against the Vulkan
    /// SDK 1.4.363 headers. A mismatch here is memory the runtime reads or
    /// writes in the wrong place.
    #[test]
    fn the_descriptors_match_the_headers() {
        let layout = |size, align| (size, align);
        assert_eq!(layout(size_of::<Header>(), align_of::<Header>()), (16, 8));
        assert_eq!(offset_of!(Header, next), 8);

        assert_eq!(size_of::<ConfigureGlobalDebug>(), 40);
        assert_eq!(offset_of!(ConfigureGlobalDebug, effect), 16);
        assert_eq!(offset_of!(ConfigureGlobalDebug, message), 24);
        assert_eq!(offset_of!(ConfigureGlobalDebug, level), 32);

        assert_eq!(size_of::<QueryProviderVersion>(), 32);
        assert_eq!(offset_of!(QueryProviderVersion, id), 16);
        assert_eq!(offset_of!(QueryProviderVersion, name), 24);

        assert_eq!(size_of::<CreateBackendVk>(), 40);
        assert_eq!(offset_of!(CreateBackendVk, device), 16);
        assert_eq!(offset_of!(CreateBackendVk, physical_device), 24);
        assert_eq!(offset_of!(CreateBackendVk, get_device_proc_addr), 32);

        assert_eq!(size_of::<CreateUpscale>(), 48);
        assert_eq!(offset_of!(CreateUpscale, flags), 16);
        assert_eq!(offset_of!(CreateUpscale, max_render_size), 20);
        assert_eq!(offset_of!(CreateUpscale, max_upscale_size), 28);
        assert_eq!(offset_of!(CreateUpscale, message), 40);

        assert_eq!(size_of::<CreateUpscaleVersion>(), 24);
        assert_eq!(offset_of!(CreateUpscaleVersion, version), 16);

        assert_eq!(
            layout(
                size_of::<ResourceDescription>(),
                align_of::<ResourceDescription>()
            ),
            (32, 4)
        );
        assert_eq!(offset_of!(ResourceDescription, format), 4);
        assert_eq!(offset_of!(ResourceDescription, width), 8);
        assert_eq!(offset_of!(ResourceDescription, height), 12);
        assert_eq!(offset_of!(ResourceDescription, depth), 16);
        assert_eq!(offset_of!(ResourceDescription, mip_count), 20);
        assert_eq!(offset_of!(ResourceDescription, flags), 24);
        assert_eq!(offset_of!(ResourceDescription, usage), 28);

        assert_eq!(size_of::<Resource>(), 48);
        assert_eq!(offset_of!(Resource, description), 8);
        assert_eq!(offset_of!(Resource, state), 40);

        assert_eq!(size_of::<DispatchUpscale>(), 432);
        let offsets = [
            (offset_of!(DispatchUpscale, command_list), 16),
            (offset_of!(DispatchUpscale, color), 24),
            (offset_of!(DispatchUpscale, depth), 72),
            (offset_of!(DispatchUpscale, motion_vectors), 120),
            (offset_of!(DispatchUpscale, exposure), 168),
            (offset_of!(DispatchUpscale, reactive), 216),
            (
                offset_of!(DispatchUpscale, transparency_and_composition),
                264,
            ),
            (offset_of!(DispatchUpscale, output), 312),
            (offset_of!(DispatchUpscale, jitter_offset), 360),
            (offset_of!(DispatchUpscale, motion_vector_scale), 368),
            (offset_of!(DispatchUpscale, render_size), 376),
            (offset_of!(DispatchUpscale, upscale_size), 384),
            (offset_of!(DispatchUpscale, enable_sharpening), 392),
            (offset_of!(DispatchUpscale, sharpness), 396),
            (offset_of!(DispatchUpscale, frame_time_delta), 400),
            (offset_of!(DispatchUpscale, pre_exposure), 404),
            (offset_of!(DispatchUpscale, reset), 408),
            (offset_of!(DispatchUpscale, camera_near), 412),
            (offset_of!(DispatchUpscale, camera_far), 416),
            (offset_of!(DispatchUpscale, camera_fov_angle_vertical), 420),
            (
                offset_of!(DispatchUpscale, view_space_to_meters_factor),
                424,
            ),
            (offset_of!(DispatchUpscale, flags), 428),
        ];
        for (index, (rust, c)) in offsets.into_iter().enumerate() {
            assert_eq!(rust, c, "ffxDispatchDescUpscale field {index}");
        }

        assert_eq!(size_of::<QueryRenderResolution>(), 48);
        assert_eq!(offset_of!(QueryRenderResolution, display_width), 16);
        assert_eq!(offset_of!(QueryRenderResolution, display_height), 20);
        assert_eq!(offset_of!(QueryRenderResolution, quality_mode), 24);
        assert_eq!(offset_of!(QueryRenderResolution, render_width), 32);
        assert_eq!(offset_of!(QueryRenderResolution, render_height), 40);

        assert_eq!(size_of::<QueryJitterPhaseCount>(), 32);
        assert_eq!(offset_of!(QueryJitterPhaseCount, render_width), 16);
        assert_eq!(offset_of!(QueryJitterPhaseCount, display_width), 20);
        assert_eq!(offset_of!(QueryJitterPhaseCount, phase_count), 24);

        assert_eq!(size_of::<MemoryUsage>(), 16);
        assert_eq!(offset_of!(MemoryUsage, aliasable), 8);
        assert_eq!(size_of::<QueryGpuMemory>(), 24);
        assert_eq!(offset_of!(QueryGpuMemory, usage), 16);
    }
}
