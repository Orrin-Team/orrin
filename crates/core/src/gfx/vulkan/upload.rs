//! Uploads that do not wait.
//!
//! A texture used to be copied and then waited for — a fence per texture, the
//! CPU idle while the GPU copied, which at a scene's load is the whole of the
//! hitch. Here an upload is recorded, submitted and handed back at once; the
//! fence is only *polled*, once a frame, and what the copy read (its staging
//! buffer) is held until it signals.
//!
//! Two paths, decided by what the device has — see
//! [`VkContext::transfer_queue`](super::context::VkContext::transfer_queue):
//!
//! - **A transfer queue.** The copy runs there, beside the frame rather than in
//!   front of it, and ends with the *release* half of a queue-family ownership
//!   transfer to the graphics family. The image is exclusive to one family at a
//!   time — concurrent sharing would spare the barrier pair and cost colour
//!   compression on every texture for the life of the program. The *acquire*
//!   half is recorded at the top of the next frame's command buffer, which
//!   waits on the semaphore the copy signalled, at the transfer stage only: the
//!   frame's vertex and compute work does not wait for a texture it may never
//!   sample. The mip chain is built there too, after the acquire, because a
//!   blit is graphics-queue work — a transfer queue can copy but not filter.
//! - **No transfer queue.** The whole upload — copy, chain, the move to
//!   shader-read — goes on the graphics queue, still without a wait. Nothing
//!   more is needed: a later submission on the same queue is ordered behind the
//!   upload's last barrier, whose destination covers every stage that samples.
//!   This is the path a one-family device takes, and the one
//!   `ORRIN_TRANSFER_QUEUE=0` forces anywhere.
//!
//! An image is usable — its view bound into the frame's sets — from the moment
//! it is handed back, on either path: the frame that first draws with it is
//! also the first that can, because it either waits on the copy (transfer
//! queue) or is queued behind it (graphics queue). So nothing samples a
//! half-written image and nothing needs a placeholder.

use std::sync::{Arc, Mutex};

use ash::vk;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::allocator::SubbufferAllocatorCreateInfo;
use vulkano::image::Image;
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::sync::semaphore::Semaphore;

use super::context::VkContext;
use super::record::{self, Arena, InFlight, KeepAlive, Pending, Recorder};
use super::texture::{self, Prepared};

/// Uploads submitted and not yet known to be finished, the ownership acquires
/// the next frame owes the transfer queue, and the staging every copy reads.
pub(super) struct Uploads {
    /// Persistently mapped and recycled: a block is reused once every copy that
    /// read from it has finished, and grows to fit the largest texture asked
    /// for. Apart from the frame's own staging arena, so a texture's staging —
    /// held until its copy's fence, which may be frames — never pins a block
    /// the per-frame uploads would otherwise reuse at once.
    staging: Arena,
    in_flight: Mutex<Vec<InFlight>>,
    acquires: Mutex<Vec<Acquire>>,
}

/// An image the transfer queue released and the graphics queue has yet to
/// acquire.
pub(super) struct Acquire {
    image: Arc<Image>,
    generate_mips: bool,
    /// Signalled by the copy; waited exactly once, by the frame that acquires.
    copied: Arc<Semaphore>,
    from_family: u32,
}

impl Uploads {
    pub(super) fn new(ctx: &VkContext) -> Self {
        Self {
            staging: Arena::new(
                ctx.memory_allocator.clone(),
                SubbufferAllocatorCreateInfo {
                    buffer_usage: BufferUsage::TRANSFER_SRC,
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
            ),
            in_flight: Mutex::default(),
            acquires: Mutex::default(),
        }
    }

    /// Where a [`Prepared`] takes its staging from.
    pub(super) fn staging(&self) -> &Arena {
        &self.staging
    }

    /// Submit `prepared` and hand back its image without waiting for the copy.
    ///
    /// `may_transfer` is false for an image graphics-queue work outside the
    /// frame reads straight away — the environment's equirect source, which the
    /// bake samples in a one-shot submission of its own and which is not a
    /// frame, so it would never record the acquire. Those take the graphics
    /// path whatever the device has.
    pub(super) fn submit(
        &self,
        ctx: &VkContext,
        prepared: Prepared,
        may_transfer: bool,
    ) -> Arc<Image> {
        self.retire();
        let graphics = ctx.queue.queue_family_index();
        let image = prepared.image.clone();
        let generate_mips = prepared.generate_mips;

        let mut keep = KeepAlive::default();
        keep.hold(prepared.staging.clone());
        keep.hold(image.clone());

        let submitted = match ctx.transfer_queue.as_ref().filter(|_| may_transfer) {
            Some(queue) => {
                let family = queue.queue_family_index();
                let commands = record::record_one_shot(ctx, family, |recording| {
                    prepared.record_copy(recording);
                    record::image_barrier(
                        recording,
                        record::release_to(
                            image.clone(),
                            record::whole_image(&image),
                            family,
                            graphics,
                        ),
                    );
                });
                let copied = Arc::new(
                    Semaphore::from_pool(ctx.device.clone())
                        .expect("failed to create an upload semaphore"),
                );
                self.acquires
                    .lock()
                    .expect("upload list poisoned")
                    .push(Acquire {
                        image: image.clone(),
                        generate_mips,
                        copied: copied.clone(),
                        from_family: family,
                    });
                record::submit_frame(ctx, queue, commands, keep, &[], &[copied])
            }
            None => {
                let commands = record::record_one_shot(ctx, graphics, |recording| {
                    prepared.record_copy(recording);
                    texture::record_finish(recording, &image, generate_mips);
                });
                record::submit_frame(ctx, &ctx.queue, commands, keep, &[], &[])
            }
        };
        self.in_flight
            .lock()
            .expect("upload list poisoned")
            .push(submitted);
        image
    }

    /// Release every upload the GPU has finished. Asked, never waited.
    pub(super) fn retire(&self) {
        self.in_flight
            .lock()
            .expect("upload list poisoned")
            .retain(|upload| !upload.is_complete());
    }

    /// Record, at the top of a frame's graphics command buffer, every acquire
    /// the transfer queue is owed, and return the semaphores the frame's first
    /// submission must wait on to make them valid.
    ///
    /// Waited at the transfer stage, which is where the acquire, the blits and
    /// the move to shader-read all run; everything the frame does before its
    /// first texture sample proceeds while a copy is still landing.
    pub(super) fn record_acquires(
        &self,
        ctx: &VkContext,
        recorder: &mut Recorder,
    ) -> Vec<(Arc<Semaphore>, vk::PipelineStageFlags)> {
        self.retire();
        let graphics = ctx.queue.queue_family_index();
        let acquires = std::mem::take(&mut *self.acquires.lock().expect("upload list poisoned"));
        acquires
            .into_iter()
            .map(|acquire| {
                let image = acquire.image.clone();
                recorder.record_raw(acquire.image.clone(), |recording| {
                    record::image_barrier(
                        recording,
                        record::acquire_from(
                            image.clone(),
                            record::whole_image(&image),
                            acquire.from_family,
                            graphics,
                        ),
                    );
                    texture::record_finish(recording, &image, acquire.generate_mips);
                });
                (acquire.copied, vk::PipelineStageFlags::TRANSFER)
            })
            .collect()
    }
}

impl Drop for Uploads {
    /// The staging buffers and images a copy still reads must outlive it, and
    /// the device must outlive them; waiting here is what lets the driver be
    /// dropped with uploads still in flight.
    fn drop(&mut self) {
        for upload in self
            .in_flight
            .get_mut()
            .expect("upload list poisoned")
            .drain(..)
        {
            upload.retire();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use vulkano::VulkanLibrary;
    use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};

    use super::*;

    /// `f` against the real upload staging, failing rather than hanging if it
    /// never returns — the failure these tests exist for is a spin.
    fn with_staging(f: impl FnOnce(&VkContext, &Arena) + Send + 'static) {
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let instance = Instance::new(
                VulkanLibrary::new().expect("a Vulkan library"),
                InstanceCreateInfo {
                    flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                    ..Default::default()
                },
            )
            .expect("a headless instance");
            let ctx = VkContext::new(&instance, None);
            let uploads = Uploads::new(&ctx);
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                f(&ctx, uploads.staging())
            }));
            let _ = done.send(outcome);
        });
        match finished.recv_timeout(Duration::from_secs(20)) {
            Ok(Ok(())) => {}
            Ok(Err(panic)) => std::panic::resume_unwind(panic),
            Err(_) => panic!("the staging allocation never returned"),
        }
    }

    /// A buffer-to-image copy's source offset is a whole number of texels into
    /// the *buffer*. An arena buffer need not start at an aligned address in its
    /// memory, so an offset aligned in memory is not enough; the RGBA32F sky,
    /// 16 bytes a texel, is the upload that found this.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn staging_offsets_are_aligned_within_the_buffer() {
        with_staging(|_, staging| {
            let mut held = Vec::new();
            for alignment in [4, 16, 256] {
                for len in [1, 64, 4096, 65_536, 1 << 20] {
                    let staging = staging.allocate_bytes(len, alignment);
                    assert_eq!(
                        staging.offset() % alignment,
                        0,
                        "{len} bytes at alignment {alignment}"
                    );
                    assert_eq!(staging.size(), len);
                    held.push(staging);
                }
            }
        });
    }

    /// The engine's own cold start: the 4 MiB rock maps size the arenas to
    /// exactly 8 MiB, and the 8 MiB sky then has to fit one whole — with any
    /// alignment padding at all, it never did, and the allocator moved to a
    /// fresh arena of the same size forever. The spacer stands in for the rest
    /// of a scene's host allocations, which are what leave that fresh arena at
    /// an address the sky's alignment has to pad.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn a_request_the_size_of_an_arena_is_allocated() {
        with_staging(|ctx, staging| {
            let rock = staging.allocate_bytes(4 << 20, 16);
            let spacer = vulkano::buffer::Buffer::new_slice::<u8>(
                ctx.memory_allocator.clone(),
                vulkano::buffer::BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                vulkano::memory::allocator::AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                4,
            )
            .expect("a spacer buffer");
            let sky = staging.allocate_bytes(8 << 20, 16);
            assert_eq!(sky.size(), 8 << 20);
            drop((rock, spacer));
        });
    }
}
