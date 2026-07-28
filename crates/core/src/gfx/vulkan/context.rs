use std::sync::Arc;

use ash::vk;
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::Instance;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::memory::MemoryHeapFlags;
use vulkano::swapchain::Surface;
use vulkano::{Version, VulkanObject};

pub struct VkContext {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl VkContext {
    pub fn new(instance: &Arc<Instance>, surface: &Arc<Surface>) -> Self {
        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            select_physical_device(instance, surface, &device_extensions);

        println!(
            "Using device: {} ({:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        // On portability-subset devices (MoltenVK on macOS) the extension must be
        // enabled if present, and egui's font/texture image views use a
        // non-identity component swizzle, which needs `image_view_format_swizzle`.
        let swizzle = physical_device
            .supported_features()
            .image_view_format_swizzle;
        if physical_device.supported_extensions().khr_portability_subset {
            device_extensions.khr_portability_subset = true;
        }

        // VK_EXT_memory_budget exposes live VRAM usage/budget per heap for the
        // performance overlay. Enable it when present; `vram_bytes` falls back to
        // reporting total heap size when it isn't.
        if physical_device.supported_extensions().ext_memory_budget {
            device_extensions.ext_memory_budget = true;
        }

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    image_view_format_swizzle: swizzle,
                    ..DeviceFeatures::empty()
                },
                ..Default::default()
            },
        )
        .expect("failed to create device");

        let queue = queues.next().unwrap();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        Self {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        }
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
            let mut props2 =
                vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
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

fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
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
                        && p.surface_support(i as u32, surface).unwrap_or(false)
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
