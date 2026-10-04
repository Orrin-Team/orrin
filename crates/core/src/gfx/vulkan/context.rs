use std::sync::Arc;

use ash::vk;
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::DescriptorBindingFlags;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::Instance;
use vulkano::memory::MemoryHeapFlags;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::cache::PipelineCache;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::swapchain::Surface;
use vulkano::{Version, VulkanObject};

use super::pipeline_cache::ShaderCache;
use super::vendor::GpuProfile;

pub struct VkContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    /// A queue from a compute-only family, when the device has one.
    ///
    /// What #64 item 8 is submitted to: the frame's compute tail goes here so
    /// that the graphics queue is free to start the *next* frame's head while
    /// it runs. `None` on a device whose only family does everything —
    /// lavapipe, and the software rasteriser CI would use — and there the frame
    /// is planned and submitted exactly as it was before this existed, which is
    /// what makes the two-driver capture comparison still an oracle.
    ///
    /// Compute-*only* rather than any second queue: a second graphics queue on
    /// AMD is the same hardware ring, so it would serialise against the first
    /// and buy nothing. Family 1 on RDNA is the asynchronous compute engine,
    /// which is a different one.
    pub compute_queue: Option<Arc<Queue>>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    /// What the device is and how its memory is shaped. Read by
    /// [`upload_mesh`](super::forward::upload_mesh) to decide whether static
    /// geometry can be written straight into video memory; reported at startup
    /// so a frame-time number has the hardware beside it.
    pub profile: GpuProfile,
    /// The driver's compiled-shader cache, reloaded from disk and written back
    /// when this context is dropped. Passed to every pipeline constructor
    /// through [`pipeline_cache`](Self::pipeline_cache).
    ///
    /// Private because the only thing anything outside wants is the handle, and
    /// the write-back is the destructor's business.
    pipelines: ShaderCache,
    /// Whether a descriptor array may be left partly unwritten.
    ///
    /// The material texture array is [`MAX_TEXTURES`] slots wide and a scene
    /// fills a handful of them. Without this the rest have to be written with a
    /// stand-in view, and the cost of that is not the write: vulkano records a
    /// tracked resource use for every element of every array the bound pipeline
    /// declares, *per draw call*, so a frame of a hundred draws tracks nineteen
    /// thousand image uses and spends milliseconds of CPU doing it. Writing only
    /// the textures that exist is what removes them, and a partly written array
    /// is only legal with this.
    ///
    /// Vulkan 1.2 core. A device without it gets the filled array it always had
    /// — see [`VkContext::texture_array_len`].
    ///
    /// [`MAX_TEXTURES`]: crate::gfx::MAX_TEXTURES
    pub partially_bound: bool,
    /// Whether one recorded command may issue many indirect draws, each with a
    /// `firstInstance` of its own.
    ///
    /// Both halves are what GPU culling draws through: a view's batches are one
    /// multi-draw, and a batch finds its instances through the `firstInstance`
    /// the cull dispatch wrote into its command. A device without them keeps the
    /// CPU sweep, which is the default path anyway — see `read_gpu_culling`.
    pub multi_draw: bool,
}

impl VkContext {
    /// How many elements of the material texture array to write.
    ///
    /// Every slot when the device cannot leave one unwritten, and only the
    /// textures that exist when it can. Paired with
    /// [`mark_texture_array_partial`](Self::mark_texture_array_partial): the
    /// short write is only legal on a layout that declared the binding
    /// partially bound, so the two must read the same flag, which is why
    /// neither takes it as an argument.
    pub fn texture_array_len(&self, loaded: usize) -> usize {
        if self.partially_bound {
            loaded.min(crate::gfx::MAX_TEXTURES)
        } else {
            crate::gfx::MAX_TEXTURES
        }
    }

    /// Declare `set`'s binding 0 — the material texture array, in each of the
    /// three layouts that has one — as partially bound, when the device allows
    /// it.
    ///
    /// A no-op on a pipeline whose shaders never declared the set, so the plain
    /// shadow pipeline (depth only, no material sampling) can be handed the same
    /// call as the masked one it shares a pass with.
    pub fn mark_texture_array_partial(
        &self,
        layout: &mut PipelineDescriptorSetLayoutCreateInfo,
        set: usize,
    ) {
        if !self.partially_bound {
            return;
        }
        if let Some(binding) = layout
            .set_layouts
            .get_mut(set)
            .and_then(|set_layout| set_layout.bindings.get_mut(&0))
        {
            binding.binding_flags |= DescriptorBindingFlags::PARTIALLY_BOUND;
        }
    }

    /// `surface` is `None` for an offscreen context — one that renders into an
    /// ordinary image and never presents. The only thing it changes is device
    /// selection: with no surface there is nothing to ask for presentation
    /// support, and no swapchain extension to require. Everything downstream is
    /// identical, which is the point — an offscreen render has to go through the
    /// same device, the same features and the same passes as a windowed one, or
    /// it would not be evidence about the windowed one.
    pub fn new(instance: &Arc<Instance>, surface: Option<&Arc<Surface>>) -> Self {
        let mut device_extensions = DeviceExtensions {
            khr_swapchain: surface.is_some(),
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            select_physical_device(instance, surface, &device_extensions);
        let compute_family = select_compute_family(&physical_device);

        let profile = GpuProfile::detect(&physical_device);
        tracing::info!(
            "Using device: {} ({:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );
        // On the same line of the log as the device it describes, because these
        // are the facts a recorded frame time is only reproducible against.
        tracing::info!("{}", profile.describe());

        // On portability-subset devices (MoltenVK on macOS) the extension must be
        // enabled if present, and egui's font/texture image views use a
        // non-identity component swizzle, which needs `image_view_format_swizzle`.
        let swizzle = physical_device
            .supported_features()
            .image_view_format_swizzle;

        let anisotropy = physical_device.supported_features().sampler_anisotropy;

        // Two features and one capability: `multi_draw_indirect` is what makes a
        // command buffer's draw count independent of the batch count, and
        // `draw_indirect_first_instance` is what lets each of those draws start
        // at its own place in the instance list. Neither is asserted, because
        // the path that needs them is optional.
        let multi_draw = physical_device.supported_features().multi_draw_indirect
            && physical_device
                .supported_features()
                .draw_indirect_first_instance;

        // Promoted to core in 1.2, and only read there: reaching it through
        // `VK_EXT_descriptor_indexing` on a 1.1 device would drag in that
        // extension's own dependencies for a device old enough that the filled
        // array is the safer shape anyway.
        let partially_bound = physical_device.api_version() >= Version::V1_2
            && physical_device
                .supported_features()
                .descriptor_binding_partially_bound;

        // Without it every attachment of a pipeline must blend identically, and
        // the transparency accumulation's two do not: one sums and the other
        // multiplies, which is exactly what makes its draw order irrelevant.
        // Universally supported on desktop and on Metal; asserted rather than
        // fallen back on, because there is no second blend equation to fall back
        // to.
        assert!(
            physical_device.supported_features().independent_blend,
            "this device cannot blend two attachments differently, which the \
             transparency pass requires",
        );

        if physical_device
            .supported_extensions()
            .khr_portability_subset
        {
            device_extensions.khr_portability_subset = true;
        }

        // VK_EXT_memory_budget exposes live VRAM usage/budget per heap for the
        // performance overlay. Enable it when present; `vram_bytes` falls back to
        // reporting total heap size when it isn't.
        if physical_device.supported_extensions().ext_memory_budget {
            device_extensions.ext_memory_budget = true;
        }

        // Required to submit the graph's barrier plan as it is written. Without
        // it vulkano's `pipeline_barrier` takes its `VK_VERSION_1_0` path, which
        // narrows the 64-bit `AccessFlags2` to 32 bits with an `as u32` — and
        // every bit the plan actually uses for a shader read (`SHADER_SAMPLED_READ`,
        // `SHADER_STORAGE_READ`, `SHADER_STORAGE_WRITE`) sits at 32 or above, so
        // each would truncate to a barrier carrying no access mask at all. See
        // `Access::flags` in `gfx/graph/access.rs` for what is being emitted.
        assert!(
            physical_device.supported_features().synchronization2,
            "this device does not support synchronization2, which submitting the \
             graph's barrier plan requires",
        );
        if physical_device.api_version() < Version::V1_3 {
            device_extensions.khr_synchronization2 = true;
        }

        // Every pass renders without a render pass object; see `rendering.rs`.
        // Promoted to core in 1.3, so the extension is only needed below that —
        // vulkano reads the feature either way. Asserted rather than fallen back
        // on: there is no second recording path to fall back to, and the whole
        // of `gfx/vulkan/` is built around the dynamic form.
        assert!(
            physical_device.supported_features().dynamic_rendering,
            "this device does not support dynamic rendering, which every pass requires",
        );
        if physical_device.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
        }

        // The compute-only family second, so `queues` yields the graphics queue
        // first whether or not there is one.
        let queue_create_infos = [Some(queue_family_index), compute_family]
            .into_iter()
            .flatten()
            .map(|queue_family_index| QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            })
            .collect();

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos,
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    image_view_format_swizzle: swizzle,
                    sampler_anisotropy: anisotropy,
                    independent_blend: true,
                    descriptor_binding_partially_bound: partially_bound,
                    multi_draw_indirect: multi_draw,
                    draw_indirect_first_instance: multi_draw,
                    dynamic_rendering: true,
                    synchronization2: true,
                    ..DeviceFeatures::empty()
                },
                ..Default::default()
            },
        )
        .expect("failed to create device");

        let queue = queues.next().unwrap();
        let compute_queue = compute_family.map(|_| {
            queues
                .next()
                .expect("a compute queue was requested but not created")
        });
        if let Some(compute) = &compute_queue {
            tracing::info!(
                "async compute: queue family {} beside graphics family {}",
                compute.queue_family_index(),
                queue.queue_family_index(),
            );
        }
        assert_packed_color_is_storable(&device);
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo {
                // The default is zero, and a frame recorded across the pool
                // takes one secondary per group of passes. One per group per
                // frame in flight, rounded up to the pool's own reuse block.
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let pipelines = ShaderCache::load(&device);

        Self {
            device,
            queue,
            compute_queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            profile,
            pipelines,
            partially_bound,
            multi_draw,
        }
    }

    /// The queue families a resource both queues touch has to name, or empty
    /// when there is only one queue and nothing to share between.
    ///
    /// Concurrent sharing rather than a derived ownership transfer, for the
    /// reason `gfx/graph/schedule.rs` gives: ownership wraps across the frame
    /// boundary, and a release/acquire pair that falls out of balance corrupts
    /// contents rather than failing a validation check. What concurrent costs
    /// instead is colour compression, on the images the compiler names and no
    /// others.
    ///
    /// Handed back as a `Vec` for the caller to collect into whatever `Sharing`
    /// wants, so this module does not name vulkano's `SmallVec` and the crate
    /// does not take a dependency on it to say two numbers.
    pub fn shared_queue_families(&self) -> Vec<u32> {
        match &self.compute_queue {
            Some(compute) => vec![
                self.queue.queue_family_index(),
                compute.queue_family_index(),
            ],
            None => Vec::new(),
        }
    }

    /// The driver's compiled-shader cache, as every pipeline constructor wants
    /// it. `None` on a device that would not give us one, which is also what
    /// those constructors took before this existed.
    pub fn pipeline_cache(&self) -> Option<Arc<PipelineCache>> {
        self.pipelines.handle()
    }

    /// Live device-local VRAM as `(used, total)` bytes. `used` is `None` when
    /// `VK_EXT_memory_budget` isn't available (e.g. some drivers); `total` is the
    /// summed size of all device-local heaps and is always reported.
    ///
    /// On unified-memory devices (Apple Silicon) the "device-local" heap is system
    /// RAM, so `total` there is the shared pool, not a dedicated VRAM bank.
    pub fn vram_bytes(&self) -> (Option<u64>, u64) {
        let phys = self.device.physical_device();
        let mem_props = phys.memory_properties();

        // Collect device-local heap indices and their summed size up front; the
        // budget extension reports usage per heap against these same indices.
        let mut total = 0u64;
        let device_local: Vec<usize> = mem_props
            .memory_heaps
            .iter()
            .enumerate()
            .filter(|(_, h)| h.flags.intersects(MemoryHeapFlags::DEVICE_LOCAL))
            .map(|(i, h)| {
                total += h.size;
                i
            })
            .collect();

        let instance = self.device.instance();
        if !self.device.enabled_extensions().ext_memory_budget
            || instance.api_version() < Version::V1_1
        {
            return (None, total);
        }

        // Chain the budget struct onto a memory-properties2 query, mirroring how
        // vulkano makes the same call internally.
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        {
            let mut props2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            unsafe {
                (instance.fns().v1_1.get_physical_device_memory_properties2)(
                    phys.handle(),
                    &mut props2,
                );
            }
        }

        let used = device_local.iter().map(|&i| budget.heap_usage[i]).sum();
        (Some(used), total)
    }
}

/// A queue family that computes but does not draw, or `None` under
/// `ORRIN_ASYNC_COMPUTE=0`.
///
/// The asynchronous compute engine on AMD and the compute queues on NVIDIA sit
/// in exactly such a family; a family that also has `GRAPHICS` is the same
/// hardware ring as the graphics queue on the drivers that matter, so taking one
/// would add submissions without adding concurrency. `None` when the device has
/// no such family, and the whole feature switches off with it.
fn select_compute_family(physical_device: &Arc<PhysicalDevice>) -> Option<u32> {
    // The control every A/B of the split is measured with, and the switch that
    // puts a machine back on the single-queue path without a rebuild if a driver
    // turns out to schedule the two badly. `ORRIN_THREADS=1` is the same idea
    // one layer up.
    if std::env::var("ORRIN_ASYNC_COMPUTE").is_ok_and(|value| value.trim() == "0") {
        return None;
    }
    physical_device
        .queue_family_properties()
        .iter()
        .position(|family| {
            family.queue_flags.contains(QueueFlags::COMPUTE)
                && !family.queue_flags.intersects(QueueFlags::GRAPHICS)
                && family.queue_count > 0
        })
        .map(|index| index as u32)
}

fn select_physical_device(
    instance: &Arc<Instance>,
    surface: Option<&Arc<Surface>>,
    extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("failed to enumerate physical devices")
        .filter(|p| p.supported_extensions().contains(extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        // A graphics queue is the whole requirement offscreen.
                        // Presentation support is a property of a surface, and
                        // there is no surface to hold it against.
                        && surface.is_none_or(|surface| {
                            p.surface_support(i as u32, surface).unwrap_or(false)
                        })
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("no suitable physical device found")
}

/// Fail at startup if the frame's packed colour format cannot back a storage
/// image on this device.
///
/// [`HDR_FORMAT`](super::hdr::HDR_FORMAT) is what most of the optical chain
/// reads and writes, and roughly half of those passes are compute passes writing
/// it as a storage image. Vulkan's mandatory-format table guarantees
/// `B10G11R11_UFLOAT_PACK32` as a sampled image and a colour attachment but *not*
/// as a storage image — in practice every desktop driver supports it, and the
/// halved bandwidth is worth more than any other single change in the frame.
///
/// Checked here, once, rather than left to surface as a descriptor-write failure
/// deep inside the first frame that runs bloom. If this ever fires on real
/// hardware the fix is to make the format a device-chosen field of `FrameConfig`
/// and compile a second set of compute shaders for the wide format — which is
/// why the assertion says so.
fn assert_packed_color_is_storable(device: &Arc<Device>) {
    let supported = device
        .physical_device()
        .format_properties(super::hdr::HDR_FORMAT)
        .map(|properties| {
            properties
                .optimal_tiling_features
                .contains(vulkano::format::FormatFeatures::STORAGE_IMAGE)
        })
        .unwrap_or(false);

    assert!(
        supported,
        "this device cannot use {:?} as a storage image, which the frame's \
         colour chain requires. Vulkan does not mandate it, though every \
         desktop driver provides it; supporting such a device means making the \
         colour format a device-chosen `FrameConfig` field and building the \
         compute passes against both formats.",
        super::hdr::HDR_FORMAT,
    );
}
