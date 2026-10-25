//! Descriptor sets reused across frames while what they point at is unchanged.
//!
//! Most of the frame's compute passes bind sets over graph images, the
//! samplers built with the pass and buffers that outlive the frame. None of
//! those move between frames — a graph image changes on resize or recompile and
//! not otherwise — yet every pass built its sets afresh each frame: an
//! allocation, a `vkUpdateDescriptorSets` and the `Arc` traffic of every write,
//! about two dozen times a frame.
//!
//! So a set is looked up by exactly what it would be written with — its layout
//! and the identity of every view, sampler and buffer range — and built only on
//! a miss. Keying by identity is sound because a cached set holds an `Arc` to
//! everything it points at: no address in a live key can be freed and handed to
//! something else. Nothing has to remember to invalidate it either. A resize
//! simply produces keys that were never seen, and the old entries age out.
//!
//! Not every set belongs here. One that binds a buffer suballocated per frame
//! would miss every frame and pin each frame's buffer for [`KEEP_FRAMES`] more,
//! so passes with per-frame uniforms keep `DescriptorSet::new` for those sets.
//!
//! Not a field per pass invalidated in `reallocate`, the way the material and
//! texture sets are: that is two dozen places to keep in step with what each set
//! reads, and an image that changes outside `reallocate` — an environment
//! reload — would be one of them forgotten.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use ash::vk::Handle;
use vulkano::VulkanObject;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet, WriteDescriptorSetElements};

/// How many frames an entry may go unused before it is dropped.
///
/// More than one because a frame split across two queues alternates between
/// two sets of images, so an entry is wanted every other frame; small because
/// an entry holds its images, and the ones from before a resize should go
/// promptly.
const KEEP_FRAMES: u64 = 3;

/// Values keyed by what they were built from, dropped once unused for
/// [`KEEP_FRAMES`].
struct Recent<K, V> {
    entries: HashMap<K, (V, u64)>,
    frame: u64,
}

impl<K, V> Default for Recent<K, V> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            frame: 0,
        }
    }
}

impl<K: Hash + Eq, V: Clone> Recent<K, V> {
    fn get_or_insert_with(&mut self, key: K, build: impl FnOnce() -> V) -> V {
        let frame = self.frame;
        let (value, used) = self.entries.entry(key).or_insert_with(|| (build(), frame));
        *used = frame;
        value.clone()
    }

    fn next_frame(&mut self) {
        self.frame += 1;
        let frame = self.frame;
        self.entries
            .retain(|_, (_, used)| frame - *used <= KEEP_FRAMES);
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// See the module docs.
#[derive(Default)]
pub(super) struct DescriptorCache {
    sets: Mutex<Recent<Vec<u64>, Arc<DescriptorSet>>>,
}

impl DescriptorCache {
    /// The set `layout` written with `writes`: the one built last time these
    /// exact resources were asked for, or a new one.
    pub fn get(
        &self,
        allocator: &Arc<StandardDescriptorSetAllocator>,
        layout: &Arc<DescriptorSetLayout>,
        writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> Arc<DescriptorSet> {
        let writes: Vec<WriteDescriptorSet> = writes.into_iter().collect();
        let key = key(layout, &writes);
        let build = move || {
            DescriptorSet::new(allocator.clone(), layout.clone(), writes, [])
                .expect("failed to write a cached descriptor set")
        };
        match key {
            Some(key) => self
                .sets
                .lock()
                .expect("descriptor cache poisoned")
                .get_or_insert_with(key, build),
            None => build(),
        }
    }

    /// Age every entry by a frame. Called once at the start of each frame.
    pub fn next_frame(&self) {
        self.sets
            .lock()
            .expect("descriptor cache poisoned")
            .next_frame();
    }
}

/// Everything that distinguishes one set from another, or `None` for writes
/// whose content is not an identity — inline uniform data — which are never
/// cached.
fn key(layout: &Arc<DescriptorSetLayout>, writes: &[WriteDescriptorSet]) -> Option<Vec<u64>> {
    let mut key = vec![layout.handle().as_raw()];
    for write in writes {
        key.push(u64::from(write.binding()));
        key.push(u64::from(write.first_array_element()));
        match write.elements() {
            WriteDescriptorSetElements::None(count) => key.extend([0, u64::from(*count)]),
            WriteDescriptorSetElements::Buffer(buffers) => {
                key.push(1);
                for info in buffers {
                    key.extend([
                        info.buffer.buffer().handle().as_raw(),
                        info.buffer.offset(),
                        info.buffer.size(),
                        info.range.start,
                        info.range.end,
                    ]);
                }
            }
            WriteDescriptorSetElements::BufferView(views) => {
                key.push(2);
                key.extend(views.iter().map(|view| view.handle().as_raw()));
            }
            WriteDescriptorSetElements::ImageView(views) => {
                key.push(3);
                for info in views {
                    key.extend([info.image_view.handle().as_raw(), info.image_layout as u64]);
                }
            }
            WriteDescriptorSetElements::ImageViewSampler(pairs) => {
                key.push(4);
                for (info, sampler) in pairs {
                    key.extend([
                        info.image_view.handle().as_raw(),
                        info.image_layout as u64,
                        sampler.handle().as_raw(),
                    ]);
                }
            }
            WriteDescriptorSetElements::Sampler(samplers) => {
                key.push(5);
                key.extend(samplers.iter().map(|sampler| sampler.handle().as_raw()));
            }
            WriteDescriptorSetElements::InlineUniformBlock(_)
            | WriteDescriptorSetElements::AccelerationStructure(_) => return None,
        }
    }
    Some(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A value asked for every frame is built once.
    #[test]
    fn a_key_used_every_frame_is_built_once() {
        let mut recent = Recent::default();
        let mut built = 0;
        for _ in 0..10 {
            recent.get_or_insert_with(vec![1, 2], || {
                built += 1;
                built
            });
            recent.next_frame();
        }
        assert_eq!(built, 1);
    }

    /// With the frame split across two queues the images alternate between two
    /// sets, so an entry is touched every other frame and must survive the
    /// frame it sits out.
    #[test]
    fn an_entry_survives_the_frames_it_sits_out() {
        let mut recent = Recent::default();
        let mut built = 0;
        for frame in 0..10u64 {
            recent.get_or_insert_with(vec![frame % 2], || {
                built += 1;
                built
            });
            recent.next_frame();
        }
        assert_eq!(built, 2);
    }

    /// What makes holding sets safe: a set keeps the views it points at alive,
    /// so an entry nothing asks for any more — the targets from before a
    /// resize — has to go, or it pins their memory for the life of the
    /// renderer.
    #[test]
    fn an_entry_nobody_asks_for_is_dropped() {
        let mut recent = Recent::default();
        let value = Arc::new(());
        recent.get_or_insert_with(vec![7], || value.clone());
        for _ in 0..=KEEP_FRAMES {
            recent.next_frame();
        }
        assert_eq!(
            Arc::strong_count(&value),
            1,
            "the stale entry is still held"
        );
        assert!(recent.is_empty());
    }

    /// The key is the identity of what a set points at: the same views give
    /// back the same set, and a different view — what a resize produces — a
    /// different one, never a stale set over the old image.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn the_same_resources_give_the_same_set_and_others_do_not() {
        use std::collections::BTreeMap;

        use vulkano::VulkanLibrary;
        use vulkano::descriptor_set::layout::{
            DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
        };
        use vulkano::format::Format;
        use vulkano::image::view::ImageView;
        use vulkano::image::{Image, ImageCreateInfo, ImageUsage};
        use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
        use vulkano::memory::allocator::AllocationCreateInfo;
        use vulkano::shader::ShaderStages;

        use crate::gfx::vulkan::context::VkContext;

        let instance = Instance::new(
            VulkanLibrary::new().expect("a Vulkan library"),
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .expect("a headless instance");
        let ctx = VkContext::new(&instance, None);
        let layout = DescriptorSetLayout::new(
            ctx.device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings: BTreeMap::from([(
                    0,
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::COMPUTE,
                        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
                    },
                )]),
                ..Default::default()
            },
        )
        .expect("a one-binding layout");
        let view = || {
            let image = Image::new(
                ctx.memory_allocator.clone(),
                ImageCreateInfo {
                    format: Format::R8G8B8A8_UNORM,
                    extent: [4, 4, 1],
                    usage: ImageUsage::STORAGE,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .expect("a small image");
            ImageView::new_default(image).expect("its view")
        };
        let (a, b) = (view(), view());

        let cache = DescriptorCache::default();
        let get = |view: &Arc<ImageView>| {
            cache.get(
                &ctx.descriptor_set_allocator,
                &layout,
                [WriteDescriptorSet::image_view(0, view.clone())],
            )
        };
        let first = get(&a);
        cache.next_frame();
        assert!(
            Arc::ptr_eq(&first, &get(&a)),
            "the same view built a second set"
        );
        assert!(
            !Arc::ptr_eq(&first, &get(&b)),
            "a different view reused the set"
        );
    }
}
