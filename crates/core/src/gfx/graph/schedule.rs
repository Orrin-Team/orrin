//! Splitting a compiled frame across the device's queues.
//!
//! The graph already knows what depends on what, which is the whole input a
//! queue schedule needs, and #64 item 8 is the observation that nothing was
//! using it. This module is that use: it cuts the pass order into contiguous
//! segments, each submitted to one queue, and names the resources that
//! therefore have to be reachable from both.
//!
//! # Why the cut is where it is
//!
//! A frame is very nearly a chain. Measured on the reference frame with every
//! optical stage on, thirty of its thirty-seven dependency levels hold exactly
//! one pass, so there is almost nothing *inside* one frame for a second queue
//! to run alongside — the whole intra-frame gain available is 7%, and it is all
//! in one place. What a second queue is actually worth is the overlap between
//! frames: this frame's compute tail against the *next* frame's graphics head.
//!
//! That is why every interleaved dispatch stays on the graphics queue. Moving
//! one there and back costs a semaphore each way for a pass the graphics queue
//! would have run anyway. The tail is the only place where what follows on the
//! graphics queue belongs to another frame, so it is the only place where
//! crossing pays — which makes the whole schedule one cut, and the plan
//! something a person can read.
//!
//! # The trailing graphics segment
//!
//! The tonemap is a draw, so it cannot move, and it is last. That is the
//! constraint the whole shape follows from: a queue is consumed in order, so a
//! submission waiting on the compute tail blocks everything behind it on the
//! graphics queue — including the next frame's head, which is the thing the
//! split exists to let run early. The renderer's answer is to hold the trailing
//! segment back and submit it *after* the next frame's head
//! (see `gfx/vulkan/mod.rs`), which is why it is a segment of its own here
//! rather than the end of the first one.

use std::ops::Range;

use super::sync::{AccessFlags, PipelineStages};

use super::{Barrier, PassDecl, PassId, PassKind};

/// Which of the device's queues a segment is submitted to.
///
/// Two, and deliberately not a family index: the graph is device-free, so it
/// plans against roles and the renderer maps them onto whatever the physical
/// device turned out to have. A device with no compute-only family gets a frame
/// that never asked for one, because `GraphBuilder::request_async_compute` is
/// the renderer telling the compiler what it found.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Queue {
    Graphics,
    AsyncCompute,
}

/// One contiguous run of the pass order, submitted to one queue.
///
/// Contiguous for the reason the recording partition is: the passes are in the
/// order the compiler derived, and a segment that skipped one would have to
/// hand it to a submission that lands between two of its own.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Segment {
    pub queue: Queue,
    /// Slots into [`FrameGraph::order`](super::FrameGraph::order).
    pub passes: Range<usize>,
}

/// Cut `order` into segments.
///
/// One segment — the whole frame on the graphics queue — unless the frame asked
/// for a second queue *and* has a compute tail to put on it. The split is
/// derived from the frame rather than switched on: a configuration whose last
/// dispatch still has a draw after it has nothing to gain and gets the plan it
/// always had.
pub(super) fn segments(order: &[PassId], passes: &[PassDecl], async_compute: bool) -> Vec<Segment> {
    let whole = || {
        vec![Segment {
            queue: Queue::Graphics,
            passes: 0..order.len(),
        }]
    };
    if !async_compute || order.is_empty() {
        return whole();
    }

    let kind = |slot: usize| passes[order[slot].index()].kind;

    // The tail begins after the last pass that draws and still has a dispatch
    // ahead of it. Everything past that point is dispatches followed by the
    // trailing draw, which is the shape the module docs argue for.
    let last_draw_before_a_dispatch = (0..order.len())
        .filter(|&slot| kind(slot) == PassKind::Inline)
        .rfind(|&slot| (slot + 1..order.len()).any(|later| kind(later) == PassKind::Compute));
    let tail_start = last_draw_before_a_dispatch.map_or(0, |slot| slot + 1);

    // How far the dispatches run. By construction no draw inside the tail has a
    // dispatch after it, so this is the whole of the compute work.
    let dispatches = (tail_start..order.len())
        .take_while(|&slot| kind(slot) == PassKind::Compute)
        .count();
    if dispatches == 0 {
        return whole();
    }
    let tail_end = tail_start + dispatches;

    let mut segments = Vec::with_capacity(3);
    if tail_start > 0 {
        segments.push(Segment {
            queue: Queue::Graphics,
            passes: 0..tail_start,
        });
    }
    segments.push(Segment {
        queue: Queue::AsyncCompute,
        passes: tail_start..tail_end,
    });
    if tail_end < order.len() {
        segments.push(Segment {
            queue: Queue::Graphics,
            passes: tail_end..order.len(),
        });
    }
    segments
}

/// The stages a compute-only queue family can be told about.
///
/// Everything else is a graphics stage, and naming one in a barrier recorded on
/// such a queue is rejected outright
/// (`VUID-vkCmdPipelineBarrier2-srcStageMask-03849`). `DrawIndirect` is on the
/// list because a compute queue reads indirect *dispatch* arguments through it.
const COMPUTE_QUEUE_STAGES: PipelineStages = PipelineStages::TOP_OF_PIPE
    .union(PipelineStages::DRAW_INDIRECT)
    .union(PipelineStages::COMPUTE_SHADER)
    .union(PipelineStages::ALL_TRANSFER)
    .union(PipelineStages::BOTTOM_OF_PIPE)
    .union(PipelineStages::HOST)
    .union(PipelineStages::ALL_COMMANDS);

/// Narrow one barrier recorded on the async queue into stages that queue has.
///
/// The graph derives a frame's dependencies without knowing which queue each
/// pass ends up on, which is the right order to do it in — the dependency is a
/// fact about the data and the queue is a fact about the schedule. This is
/// where the second meets the first.
///
/// `src_crossed_queues` says whether anything the source half describes ran on
/// the *other* queue, and it is the whole distinction:
///
/// - **It did.** What the barrier is really waiting for is the semaphore
///   between the segments, and a semaphore wait makes every write submitted
///   before the signal both available and visible to everything after it. So
///   the source half is already covered, and is dropped to `TopOfPipe` with no
///   access — which is also the only thing that can be said, since the stage
///   that produced the write cannot be named on this queue at all.
/// - **It did not.** The dependency is between two dispatches in this segment
///   and nothing else expresses it, so it is kept. Masking is still needed —
///   [`Access::stages`](super::Access::stages) widens every shader access to
///   vertex | fragment | compute — but what it removes is stages that were
///   never going to run here, and the access flags left are shader flags, which
///   is what `ComputeShader` accepts.
///
/// Getting this wrong in the second direction is silent: a bloom level read
/// with no barrier between it and the dispatch that wrote it is a race that
/// reproduces on someone else's scheduler, which is the bug class the whole
/// graph exists to remove.
pub(super) fn narrow(barrier: &mut Barrier, src_crossed_queues: bool) {
    if src_crossed_queues {
        barrier.src_stages = PipelineStages::TOP_OF_PIPE;
        barrier.src_access = AccessFlags::empty();
    } else {
        barrier.src_stages = barrier.src_stages.intersection(COMPUTE_QUEUE_STAGES);
        debug_assert!(
            !barrier.src_stages.is_empty(),
            "a source that never left this queue must name a stage this queue has",
        );
    }
    barrier.dst_stages = barrier.dst_stages.intersection(COMPUTE_QUEUE_STAGES);
    debug_assert!(
        !barrier.dst_stages.is_empty(),
        "an async pass is a dispatch, so ComputeShader must survive the mask",
    );
}

/// Which queue each slot's pass is submitted to, indexed by slot.
pub(super) fn slot_queues(segments: &[Segment], slots: usize) -> Vec<Queue> {
    let mut queues = vec![Queue::Graphics; slots];
    for segment in segments {
        for slot in segment.passes.clone() {
            queues[slot] = segment.queue;
        }
    }
    queues
}
