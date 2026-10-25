//! Recording without vulkano's auto-synchronisation, and submitting without its
//! future chain.
//!
//! Everywhere else the engine would record into an `AutoCommandBufferBuilder`,
//! which tracks every resource on the CPU and derives its own barriers — a
//! second synchronisation compiler beside the one in `gfx/graph/`. [`Recorder`]
//! records raw instead, and [`emit_plan`] submits the barriers the graph
//! actually compiled.
//!
//! Three constraints shape what is here, none of them visible from the API:
//!
//! - **Submission must go through `ash`.** A raw `CommandBuffer` cannot
//!   implement `PrimaryCommandBufferAbstract` from outside vulkano — that trait
//!   needs `CommandBufferResourcesUsage`, whose fields are all `pub(crate)` with
//!   no constructor. `GpuFuture::then_execute` and `QueueGuard::submit` both
//!   take it, so neither will accept one.
//! - **Nothing is kept alive.** Every object a recorded command names must
//!   outlive its execution; that is [`KeepAlive`]'s job.
//! - **Barriers need `synchronization2`.** Without it vulkano narrows
//!   `AccessFlags2` to 32 bits, and every shader-read bit the plan uses sits at
//!   32 or above: they truncate to an empty access mask. `context.rs` requires
//!   the feature for that reason.

use std::any::Any;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use ash::vk;
use vulkano::VulkanObject;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{BufferContents, IndexBuffer, Subbuffer};
use vulkano::command_buffer::{
    BlitImageInfo, ClearAttachment, ClearDepthStencilImageInfo, ClearRect, CommandBuffer,
    CommandBufferBeginInfo, CommandBufferInheritanceInfo, CommandBufferLevel, CommandBufferUsage,
    CopyBufferInfo, CopyBufferToImageInfo, CopyImageToBufferInfo, DrawIndexedIndirectCommand,
    RecordingCommandBuffer, RenderingInfo,
};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::image::{Image, ImageLayout, ImageSubresourceRange};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::vertex_input::VertexBuffersCollection;
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport};
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline, PipelineBindPoint, PipelineLayout};
use vulkano::query::QueryPool;
use vulkano::sync::PipelineStage;
use vulkano::sync::fence::{Fence, FenceCreateInfo};
use vulkano::sync::semaphore::Semaphore;
use vulkano::sync::{
    AccessFlags, DependencyInfo, ImageMemoryBarrier, MemoryBarrier, PipelineStages,
};

use vulkano::DeviceSize;
use vulkano::device::Queue;

use crate::gfx::graph::{Barrier, ResourceId};
use crate::profile_scope;

use super::context::VkContext;

/// Records a one-time command buffer with no auto-synchronisation, submits it,
/// and blocks until the GPU is done with it.
///
/// The wait is what makes this safe to hand a closure that borrows: nothing the
/// closure named can be dropped before the GPU has finished reading it, because
/// this does not return until then. That also makes it the wrong primitive for a
/// frame — see the module docs — and the right one for the load-time uploads,
/// which block on a fence today regardless.
pub(super) fn submit_one_shot(ctx: &VkContext, record: impl FnOnce(&mut RecordingCommandBuffer)) {
    let mut recording = RecordingCommandBuffer::new(
        ctx.command_buffer_allocator.clone(),
        ctx.queue.queue_family_index(),
        CommandBufferLevel::Primary,
        CommandBufferBeginInfo {
            usage: CommandBufferUsage::OneTimeSubmit,
            ..Default::default()
        },
    )
    .expect("failed to begin a raw command buffer");

    record(&mut recording);

    // SAFETY: the closure has returned, so nothing is still recording, and every
    // command it recorded is one this module's callers have paired with the
    // barriers it needs.
    let command_buffer = unsafe { recording.end() }.expect("failed to end a raw command buffer");

    let fence = Fence::new(ctx.device.clone(), FenceCreateInfo::default())
        .expect("failed to create the upload fence");

    submit_and_wait(ctx, &command_buffer, &fence);
}

/// Submit one finished command buffer and block until it has executed.
///
/// Shared by [`submit_one_shot`] and [`Recorder::submit_and_wait`], which differ
/// only in how the buffer was recorded.
fn submit_and_wait(ctx: &VkContext, command_buffer: &CommandBuffer, fence: &Fence) {
    let handles = [command_buffer.handle()];
    let submit = vk::SubmitInfo::default().command_buffers(&handles);

    // `Queue::with` is taken for the lock, not for what the guard offers: a
    // `VkQueue` is externally synchronised, and vulkano's own submissions take
    // the same lock.
    ctx.queue.clone().with(|_guard| {
        let fns = ctx.device.fns();
        // SAFETY: one submission of one command buffer that finished recording
        // above, on the queue whose family it was allocated from, under the
        // queue's lock, with a fresh unsignalled fence.
        unsafe { (fns.v1_0.queue_submit)(ctx.queue.handle(), 1, &submit, fence.handle()) }
            .result()
            .expect("failed to submit a raw command buffer");
    });

    fence
        .wait(None)
        .expect("failed to wait on the upload fence");
}

/// Every mip level and array layer of an image, in whichever aspects its format
/// carries.
///
/// Derived from the format rather than named, because the barrier plan is
/// emitted for depth targets as readily as colour ones and a barrier naming the
/// wrong aspect covers nothing.
pub(super) fn whole_image(image: &Arc<Image>) -> ImageSubresourceRange {
    ImageSubresourceRange {
        aspects: image.format().aspects(),
        mip_levels: 0..image.mip_levels(),
        array_layers: 0..image.array_layers(),
    }
}

/// One image barrier, recorded on its own.
///
/// Verbose by design: the point of this seam is that a transition is written
/// down where it happens rather than derived, so the call site reads as the
/// dependency it is.
pub(super) fn image_barrier(recording: &mut RecordingCommandBuffer, barrier: ImageMemoryBarrier) {
    // SAFETY: the caller is asserting the dependency this expresses is the one
    // the surrounding commands need; vulkano validates the barrier itself.
    unsafe {
        recording.pipeline_barrier(&DependencyInfo {
            image_memory_barriers: [barrier].into_iter().collect(),
            ..Default::default()
        })
    }
    .expect("invalid image barrier");
}

/// The transition an image needs before anything may copy or blit into it,
/// from a state whose contents do not matter.
pub(super) fn to_transfer_dst(
    image: Arc<Image>,
    subresource_range: ImageSubresourceRange,
    old_layout: ImageLayout,
) -> ImageMemoryBarrier {
    ImageMemoryBarrier {
        src_stages: PipelineStages::TOP_OF_PIPE,
        src_access: AccessFlags::empty(),
        dst_stages: PipelineStages::ALL_TRANSFER,
        dst_access: AccessFlags::TRANSFER_WRITE,
        old_layout,
        new_layout: ImageLayout::TransferDstOptimal,
        subresource_range,
        ..ImageMemoryBarrier::image(image)
    }
}

/// The transition a level needs between being written and being read as a blit
/// source.
pub(super) fn transfer_dst_to_src(
    image: Arc<Image>,
    subresource_range: ImageSubresourceRange,
) -> ImageMemoryBarrier {
    ImageMemoryBarrier {
        src_stages: PipelineStages::ALL_TRANSFER,
        src_access: AccessFlags::TRANSFER_WRITE,
        dst_stages: PipelineStages::ALL_TRANSFER,
        dst_access: AccessFlags::TRANSFER_READ,
        old_layout: ImageLayout::TransferDstOptimal,
        new_layout: ImageLayout::TransferSrcOptimal,
        subresource_range,
        ..ImageMemoryBarrier::image(image)
    }
}

/// The transition that hands a finished image to the shaders that sample it.
///
/// `new_layout` is `ShaderReadOnlyOptimal` because that is what vulkano calls an
/// image's canonical layout when its usage is `SAMPLED` plus transfer bits
/// (`RawImage::default_layout`), and every auto-synchronised command buffer that
/// later samples this image will assume it is in that layout on entry.
pub(super) fn to_shader_read(
    image: Arc<Image>,
    subresource_range: ImageSubresourceRange,
    old_layout: ImageLayout,
) -> ImageMemoryBarrier {
    ImageMemoryBarrier {
        src_stages: PipelineStages::ALL_TRANSFER,
        src_access: AccessFlags::TRANSFER_READ | AccessFlags::TRANSFER_WRITE,
        dst_stages: PipelineStages::FRAGMENT_SHADER,
        dst_access: AccessFlags::SHADER_READ,
        old_layout,
        new_layout: ImageLayout::ShaderReadOnlyOptimal,
        subresource_range,
        ..ImageMemoryBarrier::image(image)
    }
}

/// What one [`Barrier`] from the compiled plan actually names.
///
/// The graph tracks resources, not Vulkan objects: a `ResourceId` may be a
/// graph-owned image, an image imported from a pass that owns it across frames
/// (the TAA history, the fog scatter volume), the acquired swapchain image, or
/// an imported buffer. Resolving one to this is the renderer's job, because the
/// graph is deliberately device-free — see `gfx/graph/mod.rs`.
pub(super) enum Target {
    Image(Arc<Image>),
    /// A buffer, covered by a global memory barrier rather than a buffer one.
    ///
    /// Not a shortcut: every buffer barrier the compiler derives spans the whole
    /// buffer, so the range a `BufferMemoryBarrier` would add says nothing the
    /// stage and access pair does not, and the renderer would have to resolve
    /// four imported buffers to their `Subbuffer`s to say it. A memory barrier
    /// carries the same dependency over all buffer memory.
    Memory,
}

/// Record one pass's worth of the compiled barrier plan.
///
/// All of a pass's barriers go into one `DependencyInfo`: barriers submitted
/// together are one dependency the driver satisfies with a single stall, where a
/// sequence of them is a sequence of stalls.
///
/// `resolve` returning `None` drops the barrier — correct only for a resource
/// this configuration does not allocate.
pub(super) fn emit_plan(
    recording: &mut RecordingCommandBuffer,
    barriers: &[Barrier],
    mut resolve: impl FnMut(ResourceId) -> Option<Target>,
) {
    if barriers.is_empty() {
        return;
    }

    let mut image_memory_barriers = Vec::new();
    let mut memory_barriers = Vec::new();

    for barrier in barriers {
        let Some(target) = resolve(barrier.resource) else {
            continue;
        };
        match target {
            Target::Image(image) => {
                let subresource_range = whole_image(&image);
                image_memory_barriers.push(ImageMemoryBarrier {
                    src_stages: barrier.src_stages,
                    src_access: barrier.src_access,
                    dst_stages: barrier.dst_stages,
                    dst_access: barrier.dst_access,
                    old_layout: barrier.old_layout,
                    new_layout: barrier.new_layout,
                    subresource_range,
                    ..ImageMemoryBarrier::image(image)
                });
            }
            // A buffer has no layout, and the compiler knows it: `old_layout`
            // and `new_layout` are both `Undefined` for one, so there is nothing
            // here to drop on the floor.
            Target::Memory => {
                memory_barriers.push(MemoryBarrier {
                    src_stages: barrier.src_stages,
                    src_access: barrier.src_access,
                    dst_stages: barrier.dst_stages,
                    dst_access: barrier.dst_access,
                    ..Default::default()
                });
            }
        }
    }

    if image_memory_barriers.is_empty() && memory_barriers.is_empty() {
        return;
    }

    // SAFETY: every dependency this expresses was derived by the graph compiler
    // from what the passes declared, which is the one place in the engine that
    // knows the whole frame's data flow. Vulkano validates each barrier's own
    // consistency.
    unsafe {
        recording.pipeline_barrier(&DependencyInfo {
            image_memory_barriers: image_memory_barriers.into_iter().collect(),
            memory_barriers: memory_barriers.into_iter().collect(),
            ..Default::default()
        })
    }
    .expect("invalid barrier in the compiled plan");
}

/// A pass's own scratch buffer for the frame's uniforms, shareable while the
/// frame records.
///
/// Nothing but a `SubbufferAllocator` behind a lock. It exists because that
/// type keeps its arena in an `UnsafeCell` and is therefore `!Sync`, and a pass
/// that holds one directly cannot be recorded from a worker at all — the whole
/// pass struct stops being shareable. The lock is never contended: a pass
/// records on one thread, and the passes that share nothing else share no arena
/// either.
pub(super) struct Arena(Mutex<SubbufferAllocator>);

impl Arena {
    pub(super) fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        create_info: SubbufferAllocatorCreateInfo,
    ) -> Self {
        Self(Mutex::new(SubbufferAllocator::new(
            memory_allocator,
            create_info,
        )))
    }

    pub(super) fn allocate_sized<T: BufferContents>(&self) -> Subbuffer<T> {
        self.0
            .lock()
            .expect("a pass panicked while recording")
            .allocate_sized()
            .expect("failed to suballocate a frame uniform")
    }

    pub(super) fn allocate_slice<T: BufferContents>(&self, len: DeviceSize) -> Subbuffer<[T]> {
        self.0
            .lock()
            .expect("a pass panicked while recording")
            .allocate_slice(len)
            .expect("failed to suballocate a frame buffer")
    }
}

/// Everything a recorded command named, held until the GPU is done with it.
///
/// Dropping this before the frame's fence signals is a use-after-free the
/// validation layers will not catch. Untyped because the only operation ever
/// performed on the list is dropping it.
#[derive(Default)]
pub(super) struct KeepAlive(Vec<Box<dyn Any + Send + Sync>>);

impl KeepAlive {
    /// Hold one resource for the frame's lifetime.
    pub(super) fn hold<T: Any + Send + Sync>(&mut self, resource: T) {
        self.0.push(Box::new(resource));
    }

    /// Take over another recording's obligations, for a secondary buffer whose
    /// commands are now the primary's to keep alive.
    pub(super) fn absorb(&mut self, other: KeepAlive) {
        self.0.extend(other.0);
    }
}

/// The engine's recording surface: a raw command buffer plus what it named.
///
/// Two things an auto builder did are argued here once rather than at each of
/// the hundred-odd call sites. Ordering comes from [`Recorder::barriers`] — the
/// plan the graph compiled — not from a tracker inferring it. Lifetimes come
/// from each method holding what it binds in [`KeepAlive`].
///
/// The methods mirror the auto builder's minus the `Result`: a validation
/// failure here is a bug in this crate, as the old `.unwrap()`s said.
pub(super) struct Recorder {
    inner: RecordingCommandBuffer,
    keep: KeepAlive,
}

impl Recorder {
    pub(super) fn new(ctx: &VkContext) -> Self {
        Self::for_family(ctx, ctx.queue.queue_family_index())
    }

    /// A primary recording for a chosen queue family.
    ///
    /// A command buffer may only be submitted to the family it was allocated
    /// from, so a segment bound for the compute queue has to be recorded from
    /// that family's pool. See `gfx/graph/schedule.rs`.
    pub(super) fn for_family(ctx: &VkContext, queue_family_index: u32) -> Self {
        Self::begin(ctx, queue_family_index, CommandBufferLevel::Primary, None)
    }

    /// A recording surface for one group of passes, to be executed by the
    /// primary rather than submitted.
    ///
    /// It inherits no render pass, which is what lets the passes inside it open
    /// and close their own rendering instances exactly as they do in the
    /// primary — the alternative, inheriting one, would mean a secondary per
    /// rendering instance and the attachment formats declared twice.
    pub(super) fn secondary(ctx: &VkContext) -> Self {
        Self::begin(
            ctx,
            ctx.queue.queue_family_index(),
            CommandBufferLevel::Secondary,
            Some(CommandBufferInheritanceInfo::default()),
        )
    }

    fn begin(
        ctx: &VkContext,
        queue_family_index: u32,
        level: CommandBufferLevel,
        inheritance_info: Option<CommandBufferInheritanceInfo>,
    ) -> Self {
        let inner = RecordingCommandBuffer::new(
            ctx.command_buffer_allocator.clone(),
            queue_family_index,
            level,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                inheritance_info,
                ..Default::default()
            },
        )
        .expect("failed to begin a command buffer");
        Self {
            inner,
            keep: KeepAlive::default(),
        }
    }

    /// Run a finished secondary recording here, in the position this is called.
    ///
    /// The buffer and everything it named become this recording's to keep
    /// alive: the primary is what gets submitted, so its fence is the one that
    /// says when either may be dropped.
    pub(super) fn execute(&mut self, secondary: CommandBuffer, keep: KeepAlive) -> &mut Self {
        // SAFETY: `secondary` was recorded at secondary level against this
        // device, is executed exactly once — this recording is submitted once —
        // and inherits no render pass, so no rendering instance is open here.
        unsafe { self.inner.execute_commands(&[&secondary]) }
            .expect("invalid secondary command buffer");
        self.keep.absorb(keep);
        self.keep.hold(secondary);
        self
    }

    /// Finish recording, handing back the buffer and the resources it named.
    ///
    /// The two travel together because they have to: dropping the [`KeepAlive`]
    /// before the buffer's fence signals frees objects the GPU is still reading.
    pub(super) fn end(self) -> (CommandBuffer, KeepAlive) {
        let Self { inner, keep } = self;
        // SAFETY: recording is over — this consumes the recorder — and every
        // command in the buffer was paired with the barriers the compiled plan
        // derived for it.
        let buffer = unsafe { inner.end() }.expect("failed to end the frame's command buffer");
        (buffer, keep)
    }

    /// Submit this recording and block until the GPU has finished with it.
    ///
    /// The blocking wait is what discharges the keep-alive obligation without a
    /// per-frame list: nothing this recorded named can be dropped before the GPU
    /// is done, because the [`KeepAlive`] outlives the wait and is dropped after
    /// it. The right shape for the load-time paths — an environment bake, a mesh
    /// or texture upload — and the wrong one for a frame.
    pub(super) fn submit_and_wait(self, ctx: &VkContext) {
        let (command_buffer, keep) = self.end();
        let fence = Fence::new(ctx.device.clone(), FenceCreateInfo::default())
            .expect("failed to create the upload fence");
        submit_and_wait(ctx, &command_buffer, &fence);
        drop(keep);
    }

    /// Record one image barrier, for the load-time paths that have no compiled
    /// plan to draw on: an upload or a bake is a handful of commands whose
    /// dependencies are written where they happen.
    pub(super) fn image_barrier(&mut self, barrier: ImageMemoryBarrier) -> &mut Self {
        self.keep.hold(barrier.image.clone());
        image_barrier(&mut self.inner, barrier);
        self
    }

    /// Submit one pass's slice of the compiled barrier plan. See [`emit_plan`].
    pub(super) fn barriers(
        &mut self,
        barriers: &[Barrier],
        resolve: impl FnMut(ResourceId) -> Option<Target>,
    ) -> &mut Self {
        emit_plan(&mut self.inner, barriers, resolve);
        self
    }

    pub(super) fn begin_rendering(&mut self, mut info: RenderingInfo) -> &mut Self {
        set_auto_extent_layers(&mut info);
        for attachment in info.color_attachments.iter().flatten() {
            self.keep.hold(attachment.image_view.clone());
            if let Some(resolve) = &attachment.resolve_info {
                self.keep.hold(resolve.image_view.clone());
            }
        }
        for attachment in info.depth_attachment.iter().chain(&info.stencil_attachment) {
            self.keep.hold(attachment.image_view.clone());
        }
        unsafe { self.inner.begin_rendering(&info) }.expect("invalid rendering info");
        self
    }

    pub(super) fn end_rendering(&mut self) -> &mut Self {
        unsafe { self.inner.end_rendering() }
            .expect("ended a rendering instance that was not open");
        self
    }

    pub(super) fn clear_attachments(
        &mut self,
        attachments: impl IntoIterator<Item = ClearAttachment>,
        rects: impl IntoIterator<Item = ClearRect>,
    ) -> &mut Self {
        let attachments: Vec<_> = attachments.into_iter().collect();
        let rects: Vec<_> = rects.into_iter().collect();
        unsafe { self.inner.clear_attachments(&attachments, &rects) }
            .expect("invalid attachment clear");
        self
    }

    pub(super) fn bind_pipeline_graphics(&mut self, pipeline: &Arc<GraphicsPipeline>) -> &mut Self {
        unsafe { self.inner.bind_pipeline_graphics(pipeline) }.expect("invalid graphics pipeline");
        self.keep.hold(pipeline.clone());
        self
    }

    pub(super) fn bind_pipeline_compute(&mut self, pipeline: &Arc<ComputePipeline>) -> &mut Self {
        unsafe { self.inner.bind_pipeline_compute(pipeline) }.expect("invalid compute pipeline");
        self.keep.hold(pipeline.clone());
        self
    }

    pub(super) fn bind_descriptor_sets(
        &mut self,
        bind_point: PipelineBindPoint,
        layout: &Arc<PipelineLayout>,
        first_set: u32,
        sets: &[Arc<DescriptorSet>],
    ) -> &mut Self {
        // Unannotated on purpose: `bind_descriptor_sets` takes `&[&RawDescriptorSet]`
        // and vulkano 0.35 does not export that type, so the elements can be
        // produced and passed but the vector's type cannot be written down.
        let raw: Vec<_> = sets.iter().map(|set| set.as_raw()).collect();
        unsafe {
            self.inner
                .bind_descriptor_sets(bind_point, layout, first_set, &raw, &[])
        }
        .expect("invalid descriptor set binding");
        self.keep.hold(layout.clone());
        for set in sets {
            self.keep.hold(set.clone());
        }
        self
    }

    /// Takes the same collection the auto builder did — a single buffer, or a
    /// tuple of differently-typed ones — so a call site says what it binds
    /// rather than how to erase it.
    pub(super) fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: impl VertexBuffersCollection,
    ) -> &mut Self {
        let buffers = buffers.into_vec();
        unsafe { self.inner.bind_vertex_buffers(first_binding, &buffers) }
            .expect("invalid vertex buffer binding");
        for buffer in buffers {
            self.keep.hold(buffer);
        }
        self
    }

    pub(super) fn bind_index_buffer(&mut self, buffer: impl Into<IndexBuffer>) -> &mut Self {
        let buffer = buffer.into();
        unsafe { self.inner.bind_index_buffer(&buffer) }.expect("invalid index buffer binding");
        self.keep.hold(buffer);
        self
    }

    pub(super) fn push_constants<Pc: BufferContents>(
        &mut self,
        layout: &Arc<PipelineLayout>,
        offset: u32,
        constants: &Pc,
    ) -> &mut Self {
        unsafe { self.inner.push_constants(layout, offset, constants) }
            .expect("invalid push constants");
        self.keep.hold(layout.clone());
        self
    }

    pub(super) fn set_viewport(&mut self, first: u32, viewports: &[Viewport]) -> &mut Self {
        unsafe { self.inner.set_viewport(first, viewports) }.expect("invalid viewport");
        self
    }

    pub(super) fn set_scissor(&mut self, first: u32, scissors: &[Scissor]) -> &mut Self {
        unsafe { self.inner.set_scissor(first, scissors) }.expect("invalid scissor");
        self
    }

    pub(super) fn set_depth_bias(&mut self, constant: f32, clamp: f32, slope: f32) -> &mut Self {
        unsafe { self.inner.set_depth_bias(constant, clamp, slope) }.expect("invalid depth bias");
        self
    }

    pub(super) fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> &mut Self {
        unsafe {
            self.inner
                .draw(vertex_count, instance_count, first_vertex, first_instance)
        }
        .expect("invalid draw");
        self
    }

    pub(super) fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> &mut Self {
        unsafe {
            self.inner.draw_indexed(
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            )
        }
        .expect("invalid indexed draw");
        self
    }

    /// One draw per command in `commands`, which is sliced by the caller to the
    /// batch it is drawing.
    ///
    /// The counts come from a buffer the GPU wrote, so nothing on this side has
    /// seen them. What makes that sound is that the reset dispatch stamps every
    /// command with its batch's own `index_count` and the cull only ever
    /// increments the instance count — so a command names geometry the bound
    /// index buffer really has, and a count of instances that really were
    /// written into the slice this draw's base points at. See `gfx/vulkan/cull`.
    pub(super) fn draw_indexed_indirect(
        &mut self,
        commands: Subbuffer<[DrawIndexedIndirectCommand]>,
    ) -> &mut Self {
        let count = commands.len() as u32;
        unsafe {
            self.inner.draw_indexed_indirect(
                &commands,
                count,
                size_of::<DrawIndexedIndirectCommand>() as u32,
            )
        }
        .expect("invalid indirect draw");
        self
    }

    pub(super) fn dispatch(&mut self, group_counts: [u32; 3]) -> &mut Self {
        unsafe { self.inner.dispatch(group_counts) }.expect("invalid dispatch");
        self
    }

    pub(super) fn copy_buffer(&mut self, info: impl Into<CopyBufferInfo>) -> &mut Self {
        let info = info.into();
        unsafe { self.inner.copy_buffer(&info) }.expect("invalid buffer copy");
        self.keep.hold(info);
        self
    }

    pub(super) fn copy_buffer_to_image(&mut self, info: CopyBufferToImageInfo) -> &mut Self {
        unsafe { self.inner.copy_buffer_to_image(&info) }.expect("invalid buffer-to-image copy");
        self.keep.hold(info);
        self
    }

    pub(super) fn copy_image_to_buffer(&mut self, info: CopyImageToBufferInfo) -> &mut Self {
        unsafe { self.inner.copy_image_to_buffer(&info) }.expect("invalid image-to-buffer copy");
        self.keep.hold(info);
        self
    }

    pub(super) fn clear_depth_stencil_image(
        &mut self,
        info: ClearDepthStencilImageInfo,
    ) -> &mut Self {
        unsafe { self.inner.clear_depth_stencil_image(&info) }.expect("invalid depth clear");
        self.keep.hold(info);
        self
    }

    pub(super) fn blit_image(&mut self, info: BlitImageInfo) -> &mut Self {
        unsafe { self.inner.blit_image(&info) }.expect("invalid blit");
        self.keep.hold(info);
        self
    }

    pub(super) fn reset_query_pool(
        &mut self,
        pool: Arc<QueryPool>,
        queries: Range<u32>,
    ) -> &mut Self {
        // SAFETY: the frame resets its whole pool before opening any rendering
        // instance, so no query in the range is in flight — `timestamps.rs`
        // documents why that has to happen there.
        unsafe { self.inner.reset_query_pool(&pool, queries) }.expect("invalid query pool reset");
        self.keep.hold(pool);
        self
    }

    pub(super) fn write_timestamp(
        &mut self,
        pool: Arc<QueryPool>,
        query: u32,
        stage: PipelineStage,
    ) -> &mut Self {
        unsafe { self.inner.write_timestamp(&pool, query, stage) }
            .expect("invalid timestamp write");
        self.keep.hold(pool);
        self
    }
}

/// One frame's submission, and everything that must outlive it.
///
/// [`wait`](InFlight::wait) consumes it: the fence signalling is the only
/// evidence the GPU is finished with what the frame named.
// Every field but the fence exists to be *held*, not read: dropping any of them
// while the queue is still executing this frame is a use-after-free the
// validation layers do not catch. That is the whole obligation raw recording
// hands back, so the lint is silenced here rather than worked around.
#[allow(dead_code)]
pub(super) struct InFlight {
    /// One per submission. A split frame is three — its graphics head, its
    /// compute tail, and the previous frame's trailing segment — made in one
    /// `render` call and bounded as one, so the run-ahead stays a count of
    /// frames rather than becoming a count of submissions.
    fences: Vec<Fence>,
    keep: Vec<KeepAlive>,
    /// The buffers the queues are executing. Held for the same reason everything
    /// else here is — the allocator recycles a command buffer as soon as it is
    /// dropped, and the GPU is still reading these.
    command_buffers: Vec<CommandBuffer>,
    /// A semaphore a queue is still waiting on or signalling must not be
    /// destroyed either.
    semaphores: Vec<Arc<Semaphore>>,
}

impl InFlight {
    /// Fold another submission from the same `render` call into this one.
    pub(super) fn and(mut self, other: InFlight) -> Self {
        self.fences.extend(other.fences);
        self.keep.extend(other.keep);
        self.command_buffers.extend(other.command_buffers);
        self.semaphores.extend(other.semaphores);
        self
    }
}

/// Work the CPU has handed the queue and cannot yet prove is finished.
///
/// A trait rather than a method on [`InFlight`] so the run-ahead ring can be
/// tested against submissions whose completion the test decides.
pub(super) trait Pending {
    /// Whether the GPU has finished, asked rather than waited for.
    ///
    /// What lets the CPU run ahead: a submission is released when its fence
    /// happens to have signalled, not by blocking until it does.
    fn is_complete(&self) -> bool;

    /// Block until it has, then release what it named.
    fn retire(self);
}

impl Pending for InFlight {
    /// Every submission finished, not the last one: they went to different
    /// queues and do not complete in the order they were made.
    fn is_complete(&self) -> bool {
        self.fences
            .iter()
            .all(|fence| fence.is_signaled().unwrap_or(true))
    }

    fn retire(self) {
        for fence in &self.fences {
            fence.wait(None).expect("failed to wait on the frame fence");
        }
    }
}

/// How far the CPU may record ahead of the GPU, and the submissions that make
/// up the distance.
///
/// Oldest first, which is also the order they complete in: everything here went
/// to one queue. Retiring is opportunistic — a submission is released when its
/// fence happens to have signalled — and the block in
/// [`wait_for_room`](Self::wait_for_room) is the only one there is. Without that
/// block the list would grow until the driver's queue depth, rather than a
/// number this engine chose, decided how far ahead a frame could get.
pub(super) struct RunAhead<T> {
    pending: VecDeque<T>,
    depth: usize,
}

impl<T: Pending> RunAhead<T> {
    pub(super) fn new(depth: usize) -> Self {
        Self {
            pending: VecDeque::with_capacity(depth),
            depth,
        }
    }

    /// Make room for one more submission.
    ///
    /// Releases everything the GPU has already finished, and blocks only if that
    /// left none — which is the run-ahead bound being reached, not a frame going
    /// wrong.
    pub(super) fn wait_for_room(&mut self) {
        while self.pending.front().is_some_and(T::is_complete) {
            self.pending.pop_front();
        }
        while self.pending.len() >= self.depth {
            profile_scope!("wait");
            let oldest = self
                .pending
                .pop_front()
                .expect("the queue is non-empty inside this loop");
            oldest.retire();
        }
    }

    pub(super) fn push(&mut self, pending: T) {
        self.pending.push_back(pending);
    }

    /// Block until the GPU is idle with respect to everything here.
    ///
    /// What a readback needs, and the only place a wait is unconditional.
    pub(super) fn drain(&mut self) {
        while let Some(oldest) = self.pending.pop_front() {
            oldest.retire();
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.pending.len()
    }
}

/// Submit one recorded segment of a frame.
///
/// `queue` because a split frame's segments do not all go to the same one — the
/// compute tail is the whole point of [`VkContext::compute_queue`].
///
/// `signals` needs one semaphore per waiter: a binary semaphore's signal may be
/// waited exactly once. Through `ash` for the reason the module docs give.
pub(super) fn submit_frame(
    ctx: &VkContext,
    queue: &Arc<Queue>,
    command_buffer: CommandBuffer,
    keep: KeepAlive,
    waits: &[(Arc<Semaphore>, vk::PipelineStageFlags)],
    signals: &[Arc<Semaphore>],
) -> InFlight {
    let fence = Fence::new(ctx.device.clone(), FenceCreateInfo::default())
        .expect("failed to create the frame fence");

    let handles = [command_buffer.handle()];
    let wait_handles: Vec<_> = waits.iter().map(|(s, _)| s.handle()).collect();
    let wait_stages: Vec<_> = waits.iter().map(|(_, stage)| *stage).collect();
    let signal_handles: Vec<_> = signals.iter().map(|s| s.handle()).collect();

    let mut submit = vk::SubmitInfo::default().command_buffers(&handles);
    if !wait_handles.is_empty() {
        submit = submit
            .wait_semaphores(&wait_handles)
            .wait_dst_stage_mask(&wait_stages);
    }
    if !signal_handles.is_empty() {
        submit = submit.signal_semaphores(&signal_handles);
    }

    queue.clone().with(|_guard| {
        let fns = ctx.device.fns();
        // SAFETY: one submission of one command buffer that finished recording,
        // on the queue whose family it was allocated from, under the queue's
        // lock, with a fresh unsignalled fence and semaphores this frame owns.
        unsafe { (fns.v1_0.queue_submit)(queue.handle(), 1, &submit, fence.handle()) }
            .result()
            .expect("failed to submit the frame");
    });

    InFlight {
        fences: vec![fence],
        keep: vec![keep],
        command_buffers: vec![command_buffer],
        semaphores: waits
            .iter()
            .map(|(s, _)| s.clone())
            .chain(signals.iter().cloned())
            .collect(),
    }
}

/// Fill in a [`RenderingInfo`]'s render area and layer count from its
/// attachments, as the two zero defaults ask for.
///
/// The smallest of every attachment, resolve targets included, less the render
/// area offset. Reimplemented because `set_auto_extent_layers` is `pub(crate)`:
/// the raw path leaves both fields at zero and then rejects them.
fn set_auto_extent_layers(info: &mut RenderingInfo) {
    let auto_extent = info.render_area_extent[0] == 0 || info.render_area_extent[1] == 0;
    let auto_layers = info.layer_count == 0;
    if !auto_extent && !auto_layers {
        return;
    }

    let mut extent = [u32::MAX, u32::MAX];
    let mut layers = u32::MAX;
    let mut any = false;

    for attachment in info
        .color_attachments
        .iter()
        .flatten()
        .chain(info.depth_attachment.iter())
        .chain(info.stencil_attachment.iter())
        .flat_map(|attachment| {
            Some(&attachment.image_view).into_iter().chain(
                attachment
                    .resolve_info
                    .as_ref()
                    .map(|resolve| &resolve.image_view),
            )
        })
    {
        any = true;
        let view_extent = attachment.image().extent();
        extent[0] = extent[0].min(view_extent[0]);
        extent[1] = extent[1].min(view_extent[1]);
        layers = layers.min(attachment.subresource_range().array_layers.len() as u32);
    }

    if !any {
        return;
    }
    if auto_extent {
        for axis in 0..2 {
            info.render_area_extent[axis] = extent[axis]
                .checked_sub(info.render_area_offset[axis])
                .unwrap_or(1);
        }
    }
    if auto_layers {
        info.layer_count = layers;
    }
}

/// A run of mip levels of a cube or array image, all layers.
pub(super) fn levels(image: &Arc<Image>, mip_levels: Range<u32>) -> ImageSubresourceRange {
    ImageSubresourceRange {
        aspects: image.format().aspects(),
        mip_levels,
        array_layers: 0..image.array_layers(),
    }
}

/// The transition a level needs between being rendered into and being read as a
/// blit source.
pub(super) fn color_to_transfer_src(
    image: Arc<Image>,
    subresource_range: ImageSubresourceRange,
) -> ImageMemoryBarrier {
    ImageMemoryBarrier {
        src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
        src_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stages: PipelineStages::ALL_TRANSFER,
        dst_access: AccessFlags::TRANSFER_READ,
        old_layout: ImageLayout::ColorAttachmentOptimal,
        new_layout: ImageLayout::TransferSrcOptimal,
        subresource_range,
        ..ImageMemoryBarrier::image(image)
    }
}

/// The transition a level needs before it can be rendered into, from a state
/// whose contents do not matter.
pub(super) fn to_color_attachment(
    image: Arc<Image>,
    subresource_range: ImageSubresourceRange,
) -> ImageMemoryBarrier {
    ImageMemoryBarrier {
        src_stages: PipelineStages::TOP_OF_PIPE,
        src_access: AccessFlags::empty(),
        dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
        dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
        old_layout: ImageLayout::Undefined,
        new_layout: ImageLayout::ColorAttachmentOptimal,
        subresource_range,
        ..ImageMemoryBarrier::image(image)
    }
}

/// The transition that hands a rendered image to the shaders that sample it.
pub(super) fn color_to_shader_read(
    image: Arc<Image>,
    subresource_range: ImageSubresourceRange,
) -> ImageMemoryBarrier {
    ImageMemoryBarrier {
        src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
        src_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
        dst_stages: PipelineStages::FRAGMENT_SHADER,
        dst_access: AccessFlags::SHADER_READ,
        old_layout: ImageLayout::ColorAttachmentOptimal,
        new_layout: ImageLayout::ShaderReadOnlyOptimal,
        subresource_range,
        ..ImageMemoryBarrier::image(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    /// A submission whose completion the test decides, logging the one thing
    /// that matters: whether the ring *blocked* on it, as opposed to finding it
    /// already finished.
    struct Fake {
        id: u32,
        complete: Rc<Cell<bool>>,
        waited: Rc<RefCell<Vec<u32>>>,
    }

    impl Pending for Fake {
        fn is_complete(&self) -> bool {
            self.complete.get()
        }

        fn retire(self) {
            self.waited.borrow_mut().push(self.id);
        }
    }

    struct Harness {
        ring: RunAhead<Fake>,
        waited: Rc<RefCell<Vec<u32>>>,
        next: u32,
    }

    impl Harness {
        fn new(depth: usize) -> Self {
            Self {
                ring: RunAhead::new(depth),
                waited: Rc::new(RefCell::new(Vec::new())),
                next: 0,
            }
        }

        fn push(&mut self, complete: bool) {
            let id = self.next;
            self.next += 1;
            self.ring.push(Fake {
                id,
                complete: Rc::new(Cell::new(complete)),
                waited: self.waited.clone(),
            });
        }

        fn waited(&self) -> Vec<u32> {
            self.waited.borrow().clone()
        }
    }

    /// The whole point of a depth greater than one: with room left, opening a
    /// frame costs no wait however busy the GPU is.
    #[test]
    fn runs_ahead_without_blocking_below_the_depth() {
        let mut harness = Harness::new(3);
        harness.push(false);
        harness.push(false);

        harness.ring.wait_for_room();

        assert_eq!(harness.waited(), Vec::<u32>::new());
        assert_eq!(harness.ring.len(), 2);
    }

    /// And the bound that keeps it honest: at the depth the CPU blocks, on the
    /// oldest submission and on that one only.
    #[test]
    fn blocks_on_the_oldest_at_the_depth() {
        let mut harness = Harness::new(3);
        harness.push(false);
        harness.push(false);
        harness.push(false);

        harness.ring.wait_for_room();

        assert_eq!(harness.waited(), vec![0]);
        assert_eq!(harness.ring.len(), 2);
    }

    /// A submission the GPU happens to have finished is released without a wait,
    /// which is what stops the depth from being a lock-step.
    #[test]
    fn releases_finished_submissions_without_waiting() {
        let mut harness = Harness::new(3);
        harness.push(true);
        harness.push(true);
        harness.push(false);

        harness.ring.wait_for_room();

        assert_eq!(harness.waited(), Vec::<u32>::new());
        assert_eq!(harness.ring.len(), 1);
    }

    /// What a readback needs: after this the GPU is idle with respect to
    /// everything the ring held, whatever its fences said on the way in.
    #[test]
    fn drain_waits_for_every_submission() {
        let mut harness = Harness::new(3);
        harness.push(true);
        harness.push(false);
        harness.push(false);

        harness.ring.drain();

        assert_eq!(harness.waited(), vec![0, 1, 2]);
        assert_eq!(harness.ring.len(), 0);
    }
}
