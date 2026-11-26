//! NVIDIA DLSS through NGX: the C entry points of NVIDIA's NGX SDK, the
//! structures they are handed, and a DLSS feature that upscales one frame into
//! the command buffer the frame is recording.
//!
//! **Linked, not loaded.** The SDK is a static library, `nvsdk_ngx_d.lib`,
//! linked from wherever `ORRIN_DLSS_DIR` names it when the `dlss` feature is
//! built (see `build.rs`). At run time it finds the driver's NGX core, and the
//! DLSS library `nvngx_dlss.dll` beside the executable or, where
//! `ORRIN_DLSS_DIR` is set, in the SDK's release folder: never its development
//! one, which draws a watermark over every frame. A machine with neither runs
//! the engine on its own resolve.
//!
//! **Asked before the instance and the device exist.** NGX names the
//! extensions it needs of each (guide §2.3.1), and an extension is enabled only
//! when its object is made. [`instance_extensions`] runs before the instance
//! is, on every machine, because there is no device yet to ask the vendor of;
//! [`device_extensions`] before the device is, on an NVIDIA device only.
//! Nothing else here is called on a device that is not NVIDIA's.
//!
//! **Every `unsafe` of the integration is here**: the `#[repr(C)]` mirrors of
//! NGX's structs, the declarations of its functions and the calls through
//! them. Each mirror names the header it is transcribed from, and the test at
//! the bottom pins its size and offsets to what MSVC computed from that header.
//! Parameters go through NGX's C map (`NVSDK_NGX_Parameter_Set*`) under the
//! names `nvsdk_ngx_defs.h` gives them; the SDK's helpers are inline C that
//! set those names, so what they do is done here instead.
//!
//! **What an evaluation leaves behind.** NGX takes its inputs in a read state
//! and its output as storage, and puts each back before it returns (§3.4), so
//! the graph node declares those accesses and the plan holds on both sides of
//! the call. It binds its own pipelines and restores none of the command
//! buffer's state (§5.2.5), which [`Recorder::record_foreign`] says nothing
//! after it relies on.
//!
//! **One call at a time, from one thread.** NGX is not thread-safe (§5.2.4).
//! [`Ngx`] and [`Feature`] are `Send` and not `Sync`; their owner keeps them
//! behind mutexes and enters them only from the thread that drives the frame,
//! which records the DLSS node itself (`FrameRecord::record_segment`).

use std::ffi::{CStr, c_char, c_void};
use std::os::windows::ffi::OsStrExt;
use std::path::PathBuf;
use std::sync::Arc;

use ash::vk;
use vulkano::VulkanObject;
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, Queue};
use vulkano::image::ImageUsage;
use vulkano::image::view::ImageView;

use super::context::VkContext;
use super::record::Recorder;

/// NVIDIA's PCI vendor id: the only vendor NGX runs on.
pub(super) const NVIDIA: u32 = 0x10de;

/// `NVSDK_NGX_Result`, `nvsdk_ngx_defs.h`: a failure is `0xBAD00000` with a
/// code in its low bits, and success is one.
type NgxResult = u32;

/// `NVSDK_NGX_Version_API`, `nvsdk_ngx_defs.h`: the API this was written
/// against, which also tells the driver to keep one instance per device.
const SDK_VERSION: u32 = 0x15;

/// `NVSDK_NGX_Feature_SuperSampling`, `NVSDK_NGX_ENGINE_TYPE_CUSTOM`,
/// `NVSDK_NGX_Application_Identifier_Type_Project_Id` and
/// `NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW`.
const FEATURE_SUPER_SAMPLING: u32 = 1;
const ENGINE_TYPE_CUSTOM: u32 = 0;
const IDENTIFIER_PROJECT_ID: u32 = 1;
const RESOURCE_IMAGE_VIEW: u32 = 0;

/// `NVSDK_NGX_Logging_Level`.
const LOGGING_OFF: u32 = 0;
const LOGGING_ON: u32 = 1;

/// `NVSDK_NGX_DLSS_Feature_Flags`, `nvsdk_ngx_defs.h`.
pub(super) const FLAG_IS_HDR: i32 = 1 << 0;
pub(super) const FLAG_MV_LOW_RES: i32 = 1 << 1;
pub(super) const FLAG_DEPTH_INVERTED: i32 = 1 << 3;

/// `NVSDK_NGX_PerfQuality_Value`, `nvsdk_ngx_defs.h`.
pub(super) const PERF_MAX_PERF: i32 = 0;
pub(super) const PERF_BALANCED: i32 = 1;
pub(super) const PERF_MAX_QUALITY: i32 = 2;
pub(super) const PERF_ULTRA_PERFORMANCE: i32 = 3;
pub(super) const PERF_DLAA: i32 = 5;

/// What NGX is initialised with for an engine of its own: a GUID, which the
/// driver checks the shape of, and the engine's version (guide §5.2.1-3).
const PROJECT_ID: &CStr = c"1dd8c65b-0d42-45be-8240-8c1d77bace14";
const ENGINE_VERSION: &CStr =
    match CStr::from_bytes_with_nul(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes()) {
        Ok(version) => version,
        Err(_) => panic!("the crate version has a nul in it"),
    };

/// `NVSDK_NGX_Parameter` and `NVSDK_NGX_Handle`: opaque, NGX's own.
#[repr(C)]
struct Parameter {
    _opaque: [u8; 0],
}

#[repr(C)]
struct Handle {
    _opaque: [u8; 0],
}

/// `NVSDK_NGX_ProjectIdDescription`, `nvsdk_ngx_defs.h`.
#[repr(C)]
struct ProjectIdDescription {
    project_id: *const c_char,
    engine_type: u32,
    engine_version: *const c_char,
}

/// `NVSDK_NGX_Application_Identifier`, `nvsdk_ngx_defs.h`. The union is
/// written as its project half, the larger and the only one passed.
#[repr(C)]
struct ApplicationIdentifier {
    kind: u32,
    project: ProjectIdDescription,
}

/// `NVSDK_NGX_PathListInfo`, `nvsdk_ngx_defs.h`. `wchar_t` is UTF-16 here.
#[repr(C)]
struct PathListInfo {
    path: *const *const u16,
    length: u32,
}

/// `NVSDK_NGX_AppLogCallback`, `nvsdk_ngx_defs.h`.
type LogFn = unsafe extern "C" fn(message: *const c_char, level: u32, source: u32);

/// `NVSDK_NGX_LoggingInfo`, `nvsdk_ngx_defs.h`.
#[repr(C)]
struct LoggingInfo {
    callback: Option<LogFn>,
    minimum_level: u32,
    disable_other_sinks: bool,
}

/// `NVSDK_NGX_FeatureCommonInfo`, `nvsdk_ngx_defs.h`.
#[repr(C)]
struct FeatureCommonInfo {
    path_list: PathListInfo,
    internal: *mut c_void,
    logging: LoggingInfo,
}

/// `NVSDK_NGX_FeatureDiscoveryInfo`, `nvsdk_ngx_defs.h`.
#[repr(C)]
struct FeatureDiscoveryInfo {
    sdk_version: u32,
    feature: u32,
    identifier: ApplicationIdentifier,
    data_path: *const u16,
    feature_info: *const FeatureCommonInfo,
}

/// `NVSDK_NGX_ImageViewInfo_VK`, `nvsdk_ngx_defs_vk.h`.
#[repr(C)]
#[derive(Clone, Copy)]
struct ImageViewInfo {
    view: vk::ImageView,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    format: vk::Format,
    width: u32,
    height: u32,
}

/// `NVSDK_NGX_Resource_VK`, `nvsdk_ngx_defs_vk.h`. The union is written as its
/// image-view half, the larger and the only one NGX reads (guide §5.4.1).
#[repr(C)]
#[derive(Clone, Copy)]
struct Resource {
    view: ImageViewInfo,
    kind: u32,
    read_write: bool,
}

impl Resource {
    /// What `NVSDK_NGX_Create_ImageView_Resource_VK` in
    /// `nvsdk_ngx_helpers_vk.h` makes of a view.
    fn new(view: &ImageView) -> Self {
        let image = view.image();
        let [width, height, _] = image.extent();
        Self {
            view: ImageViewInfo {
                view: view.handle(),
                image: image.handle(),
                range: view.subresource_range().to_vk(),
                format: view.format().into(),
                width,
                height,
            },
            kind: RESOURCE_IMAGE_VIEW,
            read_write: image.usage().intersects(ImageUsage::STORAGE),
        }
    }
}

/// `PFN_NVSDK_NGX_DLSS_GetOptimalSettingsCallback` and
/// `PFN_NVSDK_NGX_DLSS_GetStatsCallback`, `nvsdk_ngx_defs.h`: both read and
/// write the map they are handed.
type MapCallback = unsafe extern "C" fn(*mut Parameter) -> NgxResult;

/// `PFN_NVSDK_NGX_ProgressCallback_C`, `nvsdk_ngx_defs.h`.
type ProgressFn = unsafe extern "C" fn(progress: f32, cancel: *mut bool);

// `nvsdk_ngx_vk.h` and `nvsdk_ngx_params.h`. `NVSDK_CONV` is `__cdecl`, which
// on x64 is the C convention.
unsafe extern "C" {
    fn NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
        info: *const FeatureDiscoveryInfo,
        count: *mut u32,
        properties: *mut *mut vk::ExtensionProperties,
    ) -> NgxResult;
    fn NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
        info: *const FeatureDiscoveryInfo,
        count: *mut u32,
        properties: *mut *mut vk::ExtensionProperties,
    ) -> NgxResult;
    #[allow(clippy::too_many_arguments)]
    fn NVSDK_NGX_VULKAN_Init_with_ProjectID(
        project_id: *const c_char,
        engine_type: u32,
        engine_version: *const c_char,
        data_path: *const u16,
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
        device: vk::Device,
        get_instance_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,
        get_device_proc_addr: Option<vk::PFN_vkGetDeviceProcAddr>,
        feature_info: *const FeatureCommonInfo,
        sdk_version: u32,
    ) -> NgxResult;
    fn NVSDK_NGX_VULKAN_Shutdown1(device: vk::Device) -> NgxResult;
    fn NVSDK_NGX_VULKAN_GetCapabilityParameters(parameters: *mut *mut Parameter) -> NgxResult;
    fn NVSDK_NGX_VULKAN_DestroyParameters(parameters: *mut Parameter) -> NgxResult;
    fn NVSDK_NGX_VULKAN_CreateFeature1(
        device: vk::Device,
        command_buffer: vk::CommandBuffer,
        feature: u32,
        parameters: *mut Parameter,
        handle: *mut *mut Handle,
    ) -> NgxResult;
    fn NVSDK_NGX_VULKAN_EvaluateFeature_C(
        command_buffer: vk::CommandBuffer,
        handle: *const Handle,
        parameters: *const Parameter,
        progress: Option<ProgressFn>,
    ) -> NgxResult;
    fn NVSDK_NGX_VULKAN_ReleaseFeature(handle: *mut Handle) -> NgxResult;

    fn NVSDK_NGX_Parameter_SetF(parameters: *mut Parameter, name: *const c_char, value: f32);
    fn NVSDK_NGX_Parameter_SetUI(parameters: *mut Parameter, name: *const c_char, value: u32);
    fn NVSDK_NGX_Parameter_SetI(parameters: *mut Parameter, name: *const c_char, value: i32);
    fn NVSDK_NGX_Parameter_SetVoidPointer(
        parameters: *mut Parameter,
        name: *const c_char,
        value: *mut c_void,
    );
    fn NVSDK_NGX_Parameter_GetULL(
        parameters: *mut Parameter,
        name: *const c_char,
        value: *mut u64,
    ) -> NgxResult;
    fn NVSDK_NGX_Parameter_GetUI(
        parameters: *mut Parameter,
        name: *const c_char,
        value: *mut u32,
    ) -> NgxResult;
    fn NVSDK_NGX_Parameter_GetI(
        parameters: *mut Parameter,
        name: *const c_char,
        value: *mut i32,
    ) -> NgxResult;
    fn NVSDK_NGX_Parameter_GetVoidPointer(
        parameters: *mut Parameter,
        name: *const c_char,
        value: *mut *mut c_void,
    ) -> NgxResult;
}

/// `NVSDK_NGX_FAILED`, `nvsdk_ngx_defs.h`.
fn failed(result: NgxResult) -> bool {
    result & 0xfff0_0000 == 0xbad0_0000
}

fn check(call: &str, result: NgxResult) -> Result<(), String> {
    if failed(result) {
        Err(format!("{call} returned {}", describe(result)))
    } else {
        Ok(())
    }
}

/// Where NGX's messages go: the engine's log, below its own warnings, since
/// NGX's levels are of verbosity rather than severity.
///
/// # Safety
///
/// NGX calls this with a null-terminated string, or null.
unsafe extern "C" fn log(message: *const c_char, _level: u32, _source: u32) {
    if message.is_null() {
        return;
    }
    // SAFETY: NGX hands a null-terminated string that is valid for the call
    // (`NVSDK_NGX_AppLogCallback`, `nvsdk_ngx_defs.h`), and it is copied out
    // before this returns.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    tracing::debug!(target: "ngx", "{}", text.trim_end());
}

fn wide(path: &std::path::Path) -> Vec<u16> {
    path.as_os_str().encode_wide().chain([0]).collect()
}

/// What NGX is told of the engine whenever it asks who is calling: where it
/// may write its logs, where `nvngx_dlss.dll` may be besides the executable's
/// directory, and where its messages go. The pointers in `common` are into
/// the vectors beside it, whose buffers do not move while this lives.
struct Identity {
    data_path: Vec<u16>,
    _dll_dirs: Vec<Vec<u16>>,
    dll_dir_pointers: Vec<*const u16>,
    common: FeatureCommonInfo,
}

impl Identity {
    fn new() -> Box<Self> {
        // A writable directory of the engine's own, as guide §3.19 suggests;
        // NGX names an unwritable one in its result if this cannot be made.
        let data = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("orrin")
            .join("ngx");
        let _ = std::fs::create_dir_all(&data);
        let dll_dirs: Vec<Vec<u16>> = std::env::var_os("ORRIN_DLSS_DIR")
            .map(|sdk| {
                PathBuf::from(sdk)
                    .join("lib")
                    .join("Windows_x86_64")
                    .join("rel")
            })
            .and_then(|dir| std::path::absolute(dir).ok())
            .iter()
            .map(|dir| wide(dir))
            .collect();
        let dll_dir_pointers: Vec<*const u16> = dll_dirs.iter().map(|dir| dir.as_ptr()).collect();
        let mut identity = Box::new(Self {
            data_path: wide(&data),
            common: FeatureCommonInfo {
                path_list: PathListInfo {
                    path: std::ptr::null(),
                    length: dll_dir_pointers.len() as u32,
                },
                internal: std::ptr::null_mut(),
                logging: LoggingInfo {
                    callback: Some(log),
                    minimum_level: if cfg!(debug_assertions) {
                        LOGGING_ON
                    } else {
                        LOGGING_OFF
                    },
                    disable_other_sinks: false,
                },
            },
            _dll_dirs: dll_dirs,
            dll_dir_pointers,
        });
        if !identity.dll_dir_pointers.is_empty() {
            identity.common.path_list.path = identity.dll_dir_pointers.as_ptr();
        }
        identity
    }

    fn discovery(&self) -> FeatureDiscoveryInfo {
        FeatureDiscoveryInfo {
            sdk_version: SDK_VERSION,
            feature: FEATURE_SUPER_SAMPLING,
            identifier: ApplicationIdentifier {
                kind: IDENTIFIER_PROJECT_ID,
                project: ProjectIdDescription {
                    project_id: PROJECT_ID.as_ptr(),
                    engine_type: ENGINE_TYPE_CUSTOM,
                    engine_version: ENGINE_VERSION.as_ptr(),
                },
            },
            data_path: self.data_path.as_ptr(),
            feature_info: &self.common,
        }
    }
}

/// Copy out the names in an array NGX answered a requirements query with.
///
/// # Safety
///
/// `properties` must be null or point to `count` extension properties that
/// stay valid for the call.
unsafe fn names(properties: *const vk::ExtensionProperties, count: u32) -> Vec<String> {
    if properties.is_null() || count == 0 {
        return Vec::new();
    }
    // SAFETY: the caller's.
    let properties = unsafe { std::slice::from_raw_parts(properties, count as usize) };
    properties
        .iter()
        .filter_map(|property| property.extension_name_as_c_str().ok())
        .map(|name| name.to_string_lossy().into_owned())
        .collect()
}

/// The instance extensions DLSS needs, by NGX's account. Needs nothing to
/// have been initialised (guide §2.3.1).
pub(super) fn instance_extensions() -> Result<Vec<String>, String> {
    let identity = Identity::new();
    let info = identity.discovery();
    let (mut count, mut properties) = (0u32, std::ptr::null_mut());
    // SAFETY: `info` and what it points into live across the call, laid out
    // as `nvsdk_ngx_defs.h` declares them; NGX answers by pointing
    // `properties` at `count` entries of its own, which `names` copies out
    // before anything else calls into it.
    unsafe {
        check(
            "NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements",
            NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
                &info,
                &mut count,
                &mut properties,
            ),
        )?;
        Ok(names(properties, count))
    }
}

/// The device extensions DLSS needs of `physical_device`, by NGX's account.
/// Only ever asked of an NVIDIA device.
pub(super) fn device_extensions(physical_device: &PhysicalDevice) -> Result<Vec<String>, String> {
    debug_assert_eq!(physical_device.properties().vendor_id, NVIDIA);
    let identity = Identity::new();
    let info = identity.discovery();
    let (mut count, mut properties) = (0u32, std::ptr::null_mut());
    // SAFETY: as `instance_extensions`, with the engine's live instance and a
    // physical device enumerated from it.
    unsafe {
        check(
            "NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements",
            NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
                physical_device.instance().handle(),
                physical_device.handle(),
                &info,
                &mut count,
                &mut properties,
            ),
        )?;
        Ok(names(properties, count))
    }
}

/// NGX initialised on a device. Shut down once nothing made through it is
/// left: the parameter map and every feature hold one of these.
struct Session {
    device: Arc<Device>,
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: NGX was initialised on this device, which is alive (held
        // here), and every feature made through it has been released: each
        // holds this session.
        let result = unsafe { NVSDK_NGX_VULKAN_Shutdown1(self.device.handle()) };
        if failed(result) {
            tracing::warn!("NGX: shutdown returned {}", describe(result));
        }
    }
}

/// The render extents DLSS chooses for an output in one mode.
#[derive(Clone, Copy, Debug)]
pub(super) struct Optimal {
    pub render: [u32; 2],
    pub min: [u32; 2],
    pub max: [u32; 2],
}

/// What a feature is created for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct FeatureDesc {
    pub render: [u32; 2],
    pub output: [u32; 2],
    /// `PERF_*` above.
    pub mode: i32,
    /// `FLAG_*` above.
    pub flags: i32,
}

/// One frame's inputs to an evaluation.
pub(super) struct Evaluate<'a> {
    pub color: &'a Arc<ImageView>,
    pub depth: &'a Arc<ImageView>,
    pub motion: &'a Arc<ImageView>,
    /// 1x1, the exposure the tonemap applies.
    pub exposure: &'a Arc<ImageView>,
    pub output: &'a Arc<ImageView>,
    /// In render pixels, as `NVSDK_NGX_Parameter_Jitter_Offset_X` takes it.
    pub jitter: [f32; 2],
    /// What turns a motion vector into render pixels from this frame to the
    /// last.
    pub motion_scale: [f32; 2],
    pub render: [u32; 2],
    pub reset: bool,
    pub frame_ms: f32,
}

/// NGX on the engine's device, and the one parameter map every call is made
/// through.
pub(super) struct Ngx {
    parameters: *mut Parameter,
    session: Arc<Session>,
    /// Kept for as long as NGX might read what it pointed at.
    _identity: Box<Identity>,
}

// SAFETY: the map and the session are NGX's heap objects with no affinity to
// the thread that made them, and the owner enters them from one thread only,
// one call at a time; `Send` is what lets that owner sit behind a mutex.
unsafe impl Send for Ngx {}

impl Ngx {
    /// Initialise NGX on `device`, and check that DLSS can run on it.
    pub(super) fn new(device: &Arc<Device>) -> Result<Self, String> {
        debug_assert_eq!(device.physical_device().properties().vendor_id, NVIDIA);
        let identity = Identity::new();
        let instance = device.instance();
        // SAFETY: the handles are the engine's live instance, physical device
        // and device; the proc-address function is the instance's own; the
        // strings are null-terminated and, with the common info, live in
        // `identity`, which outlives NGX's use of them.
        let result = unsafe {
            NVSDK_NGX_VULKAN_Init_with_ProjectID(
                PROJECT_ID.as_ptr(),
                ENGINE_TYPE_CUSTOM,
                ENGINE_VERSION.as_ptr(),
                identity.data_path.as_ptr(),
                instance.handle(),
                device.physical_device().handle(),
                device.handle(),
                None,
                Some(instance.fns().v1_0.get_device_proc_addr),
                &identity.common,
                SDK_VERSION,
            )
        };
        check("NVSDK_NGX_VULKAN_Init_with_ProjectID", result)?;
        // From here on, a failure shuts NGX down again as this drops.
        let session = Arc::new(Session {
            device: Arc::clone(device),
        });
        let mut parameters = std::ptr::null_mut();
        // SAFETY: NGX is initialised; it points `parameters` at a map of its
        // own, which `Drop` destroys.
        check("NVSDK_NGX_VULKAN_GetCapabilityParameters", unsafe {
            NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut parameters)
        })?;
        if parameters.is_null() {
            return Err("NGX gave no parameter map".to_string());
        }
        let mut ngx = Self {
            parameters,
            session,
            _identity: identity,
        };
        ngx.available()?;
        // What a released feature held is freed rather than cached for the
        // next (§3.14), so a frame the engine resolves holds none of it.
        ngx.set_i(c"FreeMemOnReleaseFeature", 1);
        Ok(ngx)
    }

    /// Whether NGX says DLSS runs here, and why not (guide §5.2.6).
    fn available(&mut self) -> Result<(), String> {
        if self
            .get_i(c"SuperSampling.NeedsUpdatedDriver")
            .is_some_and(|needs| needs != 0)
        {
            let major = self.get_ui(c"SuperSampling.MinDriverVersionMajor");
            let minor = self.get_ui(c"SuperSampling.MinDriverVersionMinor");
            return Err(match major.zip(minor) {
                Some((major, minor)) => {
                    format!("DLSS needs NVIDIA driver {major}.{minor} or later")
                }
                None => "DLSS needs a newer NVIDIA driver".to_string(),
            });
        }
        if self
            .get_i(c"SuperSampling.Available")
            .is_none_or(|available| available == 0)
        {
            let init = self.get_i(c"SuperSampling.FeatureInitResult");
            return Err(format!(
                "NGX says DLSS is not available on this device{}",
                init.map_or(String::new(), |code| format!(
                    " ({})",
                    describe(code as u32)
                ))
            ));
        }
        Ok(())
    }

    /// The render extents DLSS picks for `output` in `mode` (§5.2.8).
    pub(super) fn optimal(&mut self, output: [u32; 2], mode: i32) -> Result<Optimal, String> {
        let callback = self
            .get_pointer(c"DLSSOptimalSettingsCallback")
            .ok_or("the DLSS library has no optimal-settings query")?;
        self.set_ui(c"Width", output[0]);
        self.set_ui(c"Height", output[1]);
        self.set_i(c"PerfQualityValue", mode);
        // Older DLSS libraries read it; `NGX_DLSS_GET_OPTIMAL_SETTINGS` sets it.
        self.set_i(c"RTXValue", 0);
        // SAFETY: NGX stores this name's value as a
        // `PFN_NVSDK_NGX_DLSS_GetOptimalSettingsCallback`
        // (`nvsdk_ngx_helpers.h`), which takes the map it came from.
        let result = unsafe {
            let callback = std::mem::transmute::<*mut c_void, MapCallback>(callback);
            callback(self.parameters)
        };
        check("the optimal-settings query", result)?;
        let pair = |ngx: &mut Self, width, height| ngx.get_ui(width).zip(ngx.get_ui(height));
        let render = pair(self, c"OutWidth", c"OutHeight").map_or([0, 0], |(w, h)| [w, h]);
        let max = pair(
            self,
            c"DLSS.Get.Dynamic.Max.Render.Width",
            c"DLSS.Get.Dynamic.Max.Render.Height",
        )
        .map_or(render, |(w, h)| [w, h]);
        let min = pair(
            self,
            c"DLSS.Get.Dynamic.Min.Render.Width",
            c"DLSS.Get.Dynamic.Min.Render.Height",
        )
        .map_or(render, |(w, h)| [w, h]);
        Ok(Optimal { render, min, max })
    }

    /// What DLSS holds on the device, and whether its library is a
    /// development build, by its own account (`NGX_DLSS_GET_STATS_2`).
    pub(super) fn stats(&mut self) -> Option<(u64, bool)> {
        let callback = self.get_pointer(c"DLSSGetStatsCallback")?;
        // SAFETY: as `optimal`, for `PFN_NVSDK_NGX_DLSS_GetStatsCallback`.
        let result = unsafe {
            let callback = std::mem::transmute::<*mut c_void, MapCallback>(callback);
            callback(self.parameters)
        };
        if failed(result) {
            return None;
        }
        let mut bytes = 0u64;
        // SAFETY: as `get_ui`, for a 64-bit value.
        let read = unsafe {
            NVSDK_NGX_Parameter_GetULL(self.parameters, c"SizeInBytes".as_ptr(), &mut bytes)
        };
        // `NVSDK_NGX_EParameter_IsDevSnippetBranch`.
        let development = self.get_ui(c"#\x45").is_some_and(|dev| dev != 0);
        (!failed(read)).then_some((bytes, development))
    }

    /// Create a feature for `desc`, recorded into a command buffer of its own
    /// and waited for, so it is ready for the frame that first evaluates it.
    pub(super) fn create(
        &mut self,
        ctx: &VkContext,
        desc: &FeatureDesc,
    ) -> Result<Feature, String> {
        self.set_ui(c"CreationNodeMask", 1);
        self.set_ui(c"VisibilityNodeMask", 1);
        self.set_ui(c"Width", desc.render[0]);
        self.set_ui(c"Height", desc.render[1]);
        self.set_ui(c"OutWidth", desc.output[0]);
        self.set_ui(c"OutHeight", desc.output[1]);
        self.set_i(c"PerfQualityValue", desc.mode);
        self.set_i(c"DLSS.Feature.Create.Flags", desc.flags);
        self.set_i(c"DLSS.Enable.Output.Subrects", 0);

        let device = &self.session.device;
        let mut recorder = Recorder::new(ctx);
        let mut handle = std::ptr::null_mut();
        let result = recorder.record_foreign(&[], |command_buffer| {
            // SAFETY: the command buffer is a fresh primary on the engine's
            // device, recording and outside any rendering instance; the map
            // is NGX's own, set above as `NGX_VULKAN_CREATE_DLSS_EXT1` sets
            // it; `handle` is a local NGX writes the feature into.
            unsafe {
                NVSDK_NGX_VULKAN_CreateFeature1(
                    device.handle(),
                    command_buffer,
                    FEATURE_SUPER_SAMPLING,
                    self.parameters,
                    &mut handle,
                )
            }
        });
        check("NVSDK_NGX_VULKAN_CreateFeature1", result)?;
        if handle.is_null() {
            return Err("NGX made no feature".to_string());
        }
        recorder.submit_and_wait(ctx);
        Ok(Feature {
            handle,
            queue: Arc::clone(&ctx.queue),
            _session: Arc::clone(&self.session),
        })
    }

    /// Record `feature`'s upscale of `frame` into `recorder`.
    ///
    /// Each input must be in `SHADER_READ_ONLY_OPTIMAL` and the output in
    /// `GENERAL`, which is what the node declares them as; they are in those
    /// layouts again when this returns.
    pub(super) fn evaluate(
        &mut self,
        feature: &mut Feature,
        recorder: &mut Recorder,
        frame: &Evaluate<'_>,
    ) -> Result<(), String> {
        let mut color = Resource::new(frame.color);
        let mut depth = Resource::new(frame.depth);
        let mut motion = Resource::new(frame.motion);
        let mut exposure = Resource::new(frame.exposure);
        let mut output = Resource::new(frame.output);
        let pointer = |resource: &mut Resource| (resource as *mut Resource).cast::<c_void>();

        // What `NGX_VULKAN_EVALUATE_DLSS_EXT` sets, less the research-only
        // inputs and subrectangle offsets, which are never set in this map
        // and so are already the null and zero it would set them to.
        self.set_pointer(c"Color", pointer(&mut color));
        self.set_pointer(c"Output", pointer(&mut output));
        self.set_pointer(c"Depth", pointer(&mut depth));
        self.set_pointer(c"MotionVectors", pointer(&mut motion));
        self.set_pointer(c"ExposureTexture", pointer(&mut exposure));
        self.set_f(c"Jitter.Offset.X", frame.jitter[0]);
        self.set_f(c"Jitter.Offset.Y", frame.jitter[1]);
        self.set_f(c"Sharpness", 0.0);
        self.set_i(c"Reset", frame.reset as i32);
        self.set_f(c"MV.Scale.X", frame.motion_scale[0]);
        self.set_f(c"MV.Scale.Y", frame.motion_scale[1]);
        self.set_ui(c"DLSS.Render.Subrect.Dimensions.Width", frame.render[0]);
        self.set_ui(c"DLSS.Render.Subrect.Dimensions.Height", frame.render[1]);
        // The frame's colour is the light as lit: nothing scaled it on the
        // way into its format for this to undo.
        self.set_f(c"DLSS.Pre.Exposure", 1.0);
        self.set_f(c"DLSS.Exposure.Scale", 1.0);
        self.set_f(c"FrameTimeDeltaInMsec", frame.frame_ms);

        let views = [
            frame.color,
            frame.depth,
            frame.motion,
            frame.exposure,
            frame.output,
        ];
        let parameters = self.parameters;
        let result = recorder.record_foreign(&views, |command_buffer| {
            // SAFETY: the command buffer is recording — the recorder owns it
            // and is borrowed for the call — and is in no rendering instance,
            // since this is a compute node. Every view the map points at is
            // held for the frame by the recorder and is in the layout the
            // node declared; the `Resource`s it points through are locals
            // alive across the call. The feature and the map are entered
            // nowhere else meanwhile: `&mut` both.
            unsafe {
                NVSDK_NGX_VULKAN_EvaluateFeature_C(command_buffer, feature.handle, parameters, None)
            }
        });
        check("NVSDK_NGX_VULKAN_EvaluateFeature_C", result)
    }

    fn set_f(&mut self, name: &CStr, value: f32) {
        // SAFETY: the map is NGX's live one, the name null-terminated, and NGX
        // copies the value.
        unsafe { NVSDK_NGX_Parameter_SetF(self.parameters, name.as_ptr(), value) }
    }

    fn set_ui(&mut self, name: &CStr, value: u32) {
        // SAFETY: as `set_f`.
        unsafe { NVSDK_NGX_Parameter_SetUI(self.parameters, name.as_ptr(), value) }
    }

    fn set_i(&mut self, name: &CStr, value: i32) {
        // SAFETY: as `set_f`.
        unsafe { NVSDK_NGX_Parameter_SetI(self.parameters, name.as_ptr(), value) }
    }

    /// Point `name` at `value`, which NGX reads at the next call that takes
    /// the map; the caller keeps it alive until then.
    fn set_pointer(&mut self, name: &CStr, value: *mut c_void) {
        // SAFETY: as `set_f`; NGX copies the pointer, not what it points at.
        unsafe { NVSDK_NGX_Parameter_SetVoidPointer(self.parameters, name.as_ptr(), value) }
    }

    fn get_ui(&mut self, name: &CStr) -> Option<u32> {
        let mut value = 0;
        // SAFETY: the map is NGX's live one, the name null-terminated, and
        // `value` a local NGX writes one value of the asked-for type into.
        let result =
            unsafe { NVSDK_NGX_Parameter_GetUI(self.parameters, name.as_ptr(), &mut value) };
        (!failed(result)).then_some(value)
    }

    fn get_i(&mut self, name: &CStr) -> Option<i32> {
        let mut value = 0;
        // SAFETY: as `get_ui`.
        let result =
            unsafe { NVSDK_NGX_Parameter_GetI(self.parameters, name.as_ptr(), &mut value) };
        (!failed(result)).then_some(value)
    }

    fn get_pointer(&mut self, name: &CStr) -> Option<*mut c_void> {
        let mut value = std::ptr::null_mut();
        // SAFETY: as `get_ui`.
        let result = unsafe {
            NVSDK_NGX_Parameter_GetVoidPointer(self.parameters, name.as_ptr(), &mut value)
        };
        (!failed(result) && !value.is_null()).then_some(value)
    }
}

impl Drop for Ngx {
    fn drop(&mut self) {
        // SAFETY: the map came from `GetCapabilityParameters`, which is what
        // `DestroyParameters` is for (guide §6.3), and nothing holds it but
        // `self`. The session it belongs to drops after this.
        let result = unsafe { NVSDK_NGX_VULKAN_DestroyParameters(self.parameters) };
        if failed(result) {
            tracing::warn!(
                "NGX: destroying the parameter map returned {}",
                describe(result)
            );
        }
    }
}

/// A DLSS feature on the engine's device.
pub(super) struct Feature {
    handle: *mut Handle,
    /// The queue every evaluation is submitted to, and so the one that has to
    /// be idle before the feature is released (guide §5.5).
    queue: Arc<Queue>,
    _session: Arc<Session>,
}

// SAFETY: as `Ngx`'s.
unsafe impl Send for Feature {}

impl Drop for Feature {
    fn drop(&mut self) {
        if let Err(e) = self.queue.with(|mut queue| queue.wait_idle()) {
            tracing::error!("DLSS: the queue would not idle before the feature was released: {e}");
            return;
        }
        // SAFETY: the handle came from `CreateFeature1` and is released once;
        // no command buffer that evaluated it is in flight any more, and
        // NGX, which `_session` holds open, is entered from no other thread.
        let result = unsafe { NVSDK_NGX_VULKAN_ReleaseFeature(self.handle) };
        if failed(result) {
            tracing::warn!("DLSS: releasing the feature returned {}", describe(result));
        }
    }
}

/// `NVSDK_NGX_Result`, `nvsdk_ngx_defs.h`.
pub(super) fn describe(result: NgxResult) -> String {
    let name = match result {
        0x1 => "Success",
        0xbad0_0000 => "Fail",
        0xbad0_0001 => "FAIL_FeatureNotSupported",
        0xbad0_0002 => "FAIL_PlatformError",
        0xbad0_0003 => "FAIL_FeatureAlreadyExists",
        0xbad0_0004 => "FAIL_FeatureNotFound",
        0xbad0_0005 => "FAIL_InvalidParameter",
        0xbad0_0006 => "FAIL_ScratchBufferTooSmall",
        0xbad0_0007 => "FAIL_NotInitialized",
        0xbad0_0008 => "FAIL_UnsupportedInputFormat",
        0xbad0_0009 => "FAIL_RWFlagMissing",
        0xbad0_000a => "FAIL_MissingInput",
        0xbad0_000b => "FAIL_UnableToInitializeFeature",
        0xbad0_000c => "FAIL_OutOfDate",
        0xbad0_000d => "FAIL_OutOfGPUMemory",
        0xbad0_000e => "FAIL_UnsupportedFormat",
        0xbad0_000f => "FAIL_UnableToWriteToAppDataPath",
        0xbad0_0010 => "FAIL_UnsupportedParameter",
        0xbad0_0011 => "FAIL_Denied",
        0xbad0_0012 => "FAIL_NotImplemented",
        _ => "an unknown result",
    };
    format!("{result:#x} ({name})")
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, offset_of, size_of};

    use super::*;

    /// Every struct handed across the boundary has the size and field offsets
    /// NVIDIA's headers give it on x64 Windows.
    ///
    /// The numbers were printed by a C++ file that includes `nvsdk_ngx_vk.h`
    /// from DLSS SDK 310.6.0 (NGX API 0x15) by path and prints `sizeof`,
    /// `alignof` and `offsetof` for each, built with MSVC 2022 for x64 against
    /// the Vulkan SDK 1.4.363 headers. A mismatch here is memory NGX reads or
    /// writes in the wrong place.
    #[test]
    fn the_structs_match_the_headers() {
        let layout = |size, align| (size, align);

        assert_eq!(
            layout(size_of::<ImageViewInfo>(), align_of::<ImageViewInfo>()),
            (48, 8)
        );
        assert_eq!(offset_of!(ImageViewInfo, image), 8);
        assert_eq!(offset_of!(ImageViewInfo, range), 16);
        assert_eq!(offset_of!(ImageViewInfo, format), 36);
        assert_eq!(offset_of!(ImageViewInfo, width), 40);
        assert_eq!(offset_of!(ImageViewInfo, height), 44);
        assert_eq!(size_of::<vk::ImageSubresourceRange>(), 20);

        assert_eq!(
            layout(size_of::<Resource>(), align_of::<Resource>()),
            (56, 8)
        );
        assert_eq!(offset_of!(Resource, kind), 48);
        assert_eq!(offset_of!(Resource, read_write), 52);

        assert_eq!(size_of::<vk::ExtensionProperties>(), 260);
        assert_eq!(offset_of!(vk::ExtensionProperties, spec_version), 256);

        assert_eq!(size_of::<ProjectIdDescription>(), 24);
        assert_eq!(offset_of!(ProjectIdDescription, engine_type), 8);
        assert_eq!(offset_of!(ProjectIdDescription, engine_version), 16);
        assert_eq!(size_of::<ApplicationIdentifier>(), 32);
        assert_eq!(offset_of!(ApplicationIdentifier, project), 8);

        assert_eq!(size_of::<PathListInfo>(), 16);
        assert_eq!(offset_of!(PathListInfo, length), 8);
        assert_eq!(size_of::<LoggingInfo>(), 16);
        assert_eq!(offset_of!(LoggingInfo, minimum_level), 8);
        assert_eq!(offset_of!(LoggingInfo, disable_other_sinks), 12);
        assert_eq!(size_of::<FeatureCommonInfo>(), 40);
        assert_eq!(offset_of!(FeatureCommonInfo, internal), 16);
        assert_eq!(offset_of!(FeatureCommonInfo, logging), 24);

        assert_eq!(size_of::<FeatureDiscoveryInfo>(), 56);
        assert_eq!(offset_of!(FeatureDiscoveryInfo, feature), 4);
        assert_eq!(offset_of!(FeatureDiscoveryInfo, identifier), 8);
        assert_eq!(offset_of!(FeatureDiscoveryInfo, data_path), 40);
        assert_eq!(offset_of!(FeatureDiscoveryInfo, feature_info), 48);
    }
}
