//! Per-pass GPU timing, feeding [`Profiler::push_gpu_span`].
//!
//! One timestamp pair per pass, plus a reserved pair per submitted segment of
//! the frame, whose sum is the HUD's single GPU number. Four properties shape
//! the design:
//!
//! - Timestamps are written during recording but readable only once the GPU has
//!   passed them, so a pass's timing belongs to a frame that closed one or more
//!   frames ago. Every slot therefore carries the profiler frame index it
//!   recorded, and spans are filed retroactively against it.
//! - `reset_query_pool` is illegal inside a render pass, so the whole pool is
//!   reset up front — before any pass has declared itself, which is why the
//!   reset covers `2 * MAX_PASSES` queries rather than the ones actually used.
//! - A pool must not be reset while the GPU may still be reading it, which is
//!   what [`SLOTS`] buys.
//! - A split frame is three submissions, and its trailing graphics segment goes
//!   to the queue behind the *next* frame's head (see `PendingTail`). One pair
//!   from its first command to its last would also hold the GPU idling while the
//!   CPU records that next frame, and the neighbouring frames' segments; measured
//!   on an RX 9070 XT, that doubled the frame. So each segment carries its own
//!   pair and the frame is their sum: GPU work across both queues, which can
//!   exceed the wall-clock interval when compute really overlaps graphics. An
//!   unsplit frame is one segment, and so the one pair it always was.
//!
//! Both stamps of a pair are `BottomOfPipe`. A timestamp latches once all prior
//! commands have reached the given stage, so a `TopOfPipe` opening stamp can
//! fire while the previous pass is still running: the start reads too early and
//! adjacent passes appear to overlap. Bottom-of-pipe on both means "everything
//! before this has finished" and "this pass has finished", so durations are
//! disjoint and sum to the frame. The cost is that genuinely overlapping work is
//! attributed to whichever pass finishes last — the right trade for a table
//! whose rows are supposed to add up.

use std::sync::Arc;

use vulkano::query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType};
use vulkano::sync::PipelineStage;

use super::context::VkContext;
use super::record::Recorder;
use crate::profile::{Profiler, Span};

/// Passes timed per frame, including the reserved whole-frame pairs. Costs
/// `2 * MAX_PASSES` queries per slot whether used or not; passes beyond it are
/// dropped rather than mis-attributed.
///
/// The busiest frame that ships is four shadow cascades, the punctual atlas, the
/// prepass, two SSAO passes, the contact-shadow march, the forward pass, three
/// diffusion passes, four reflection passes, two transparency passes, three
/// refraction passes, the temporal resolve, four lens passes, three shutter
/// passes, two metering dispatches, an eleven-pass bloom chain and the tonemap,
/// plus a whole-frame pair per segment. Bloom is what made the old 16 too small,
/// and it grows with `MAX_BLOOM_MIPS`: a chain of `n` levels is `2n - 1` passes,
/// so raising that cap means raising this one.
/// `the_busiest_frame_fits_the_query_pool` is what keeps this number honest —
/// and it is why 40 was not enough: that test used to switch three features off,
/// so the frame it called busiest was one nothing runs, while the editor's own
/// default frame with every feature on overruns the pool and silently drops the
/// passes at the end of the chain.
const MAX_PASSES: usize = 64;

/// Frame slots in rotation: one more than the frames the CPU may be ahead by.
///
/// Two constraints, and the second is the binding one.
///
/// A slot is reset before the frame that writes it records, so it must belong to
/// a frame the GPU has finished. Slot `N % SLOTS` last belonged to `N - SLOTS`,
/// which `RunAhead` has waited on whenever `SLOTS` is at least
/// [`FRAMES_IN_FLIGHT`].
///
/// But a slot must also be *read* before it is reused. `drain_completed`
/// declines any slot whose frame the profiler has not closed yet, so frame `N`
/// is offered on the drains after frames `N + 1` through `N + SLOTS - 1`, and
/// only the last of those is sure to find it finished: that is the drain after
/// the CPU waited for `N` to retire before recording `N + FRAMES_IN_FLIGHT`.
/// The earlier offers lose whenever a CPU frame is shorter than the queue's
/// latency. With `SLOTS` equal to the run-ahead there is no such last offer,
/// which is why two slots silently filed no GPU spans at all rather than filing
/// wrong ones; one spare slot is what buys the readback its window. By then
/// frame `N`'s slot is under the write cursor, so the drain must not skip it.
///
/// Derived from the run-ahead rather than written down beside it, because the
/// two disagreeing costs timings rather than frames, and nothing about a frame
/// looks wrong while it happens.
const SLOTS: usize = super::FRAMES_IN_FLIGHT + 1;

/// Name of the reserved pairs covering the whole frame, one per segment. They
/// take the first queries in segment order, which makes query 0 a
/// guaranteed-written origin for the anchor below.
const WHOLE_FRAME_PASS: &str = "frame";

/// A query pair reserved for one pass. Dropping one without stamping it leaves
/// queries that never get written, which the pair guard in `drain_completed`
/// discards.
///
/// Copied rather than consumed because both halves are stamped from it, and the
/// two may be recorded by different calls on the same worker.
#[derive(Clone, Copy)]
pub struct PassToken {
    base: u32,
}

struct RecordedPass {
    name: &'static str,
    base: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotState {
    /// Never written, or already drained; safe to reset and record into.
    Idle,
    /// Written and submitted; awaiting a non-blocking readback.
    Pending,
}

struct FrameSlot {
    pool: Arc<QueryPool>,
    /// The profiler frame these queries belong to; readback happens later, so
    /// this is the only link back to where the spans go.
    frame_index: u64,
    /// CPU clock at the moment this frame opened. GPU ticks are converted to
    /// durations and laid out from here, which places the lane plausibly without
    /// claiming a calibrated device-to-host mapping.
    anchor_ns: u64,
    passes: Vec<RecordedPass>,
    state: SlotState,
    next_query: u32,
    /// Whole-frame pairs reserved, one per segment; zero when profiling was off
    /// as the frame opened.
    segments: u32,
}

pub struct GpuTimestamps {
    slots: [FrameSlot; SLOTS],
    write: usize,
    /// `VkPhysicalDeviceLimits::timestampPeriod`, nanoseconds per tick.
    period_ns: f32,
    /// Valid low bits of a timestamp for this queue family.
    valid_mask: u64,
    last_frame_ms: f32,
}

/// Ticks-to-nanoseconds scale and the valid-bit mask, or `None` if this queue
/// can't write timestamps (some MoltenVK configurations).
fn probe(ctx: &VkContext) -> Option<(f32, u64)> {
    let phys = ctx.device.physical_device();
    let qfi = ctx.queue.queue_family_index() as usize;
    let valid_bits = phys.queue_family_properties()[qfi].timestamp_valid_bits?;
    let period_ns = phys.properties().timestamp_period;
    if period_ns == 0.0 {
        return None;
    }
    let valid_mask = if valid_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    };
    Some((period_ns, valid_mask))
}

impl GpuTimestamps {
    pub fn new(ctx: &VkContext) -> Option<Self> {
        let (period_ns, valid_mask) = probe(ctx)?;

        let mut slots = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            let pool = QueryPool::new(
                ctx.device.clone(),
                QueryPoolCreateInfo {
                    query_count: 2 * MAX_PASSES as u32,
                    ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
                },
            )
            .ok()?;
            slots.push(FrameSlot {
                pool,
                frame_index: 0,
                anchor_ns: 0,
                passes: Vec::with_capacity(MAX_PASSES),
                state: SlotState::Idle,
                next_query: 0,
                segments: 0,
            });
        }

        Some(Self {
            slots: slots.try_into().ok()?,
            write: 0,
            period_ns,
            valid_mask,
            last_frame_ms: 0.0,
        })
    }

    /// Bind the slot about to be recorded to the profiler frame in progress.
    ///
    /// A slot still `Pending` here was never read back — its results are gone and
    /// its frame has likely aged out of the profiler ring. Clearing is the whole
    /// remedy, provided `next_query` resets with it.
    pub fn begin_frame(&mut self, profiler_frame: u64) {
        let slot = &mut self.slots[self.write];
        slot.passes.clear();
        slot.next_query = 0;
        slot.segments = 0;
        slot.frame_index = profiler_frame;
        slot.anchor_ns = crate::profile::now_ns();
        slot.state = SlotState::Idle;
    }

    /// Reset this frame's queries, reserve a whole-frame pair for each of its
    /// `segments`, and open the first.
    ///
    /// Must be recorded before the first `begin_rendering`: a reset inside a
    /// render pass instance is invalid, and which passes will run isn't known
    /// yet, so the entire pool is reset in one go.
    pub fn record_resets(&mut self, builder: &mut Recorder, segments: usize) {
        let pool = self.slots[self.write].pool.clone();
        // SAFETY: outside any render pass, and this slot is not in flight —
        // there are more slots than frames the CPU may be ahead by, so the frame
        // that last wrote this one has been waited on by
        // `RunAhead::wait_for_room`.
        builder.reset_query_pool(pool, 0..(2 * MAX_PASSES as u32));
        // Reserved first, so segment `i` is always queries `2i`/`2i + 1` and
        // query 0 is guaranteed written — `drain_completed` uses it as the
        // frame's tick origin. The seams and `end_frame` stamp the rest, from
        // those fixed positions.
        //
        // Reserved through `stamp` rather than `begin_pass`, because it must not
        // answer to the per-pass switch: the whole-frame pair is the one number
        // that has to survive turning per-pass timing off, since comparing the
        // two is the entire point of being able to.
        if crate::profile::is_enabled() {
            for _ in 0..segments {
                self.take_pair(WHOLE_FRAME_PASS);
            }
            self.slots[self.write].segments = segments as u32;
            self.open(builder, self.segment(0));
        }
    }

    /// Close segment `index - 1`'s whole-frame pair in the buffer it ends, and
    /// open segment `index`'s in the one that follows.
    pub fn seam(&self, ending: &mut Recorder, starting: &mut Recorder, index: usize) {
        self.close(ending, self.segment(index - 1));
        self.open(starting, self.segment(index));
    }

    /// The whole-frame pair of segment `index`, if one was reserved.
    fn segment(&self, index: usize) -> Option<PassToken> {
        (index < self.slots[self.write].segments as usize).then_some(PassToken {
            base: 2 * index as u32,
        })
    }

    /// Reserve a query pair for `name`, for the pass to stamp when it records.
    ///
    /// Reserving and stamping are separate because a pass may record on a
    /// worker: taking the pair mutates the frame's slot and so belongs to the
    /// thread that owns the frame, while the two writes are commands like any
    /// other and go wherever the pass goes.
    ///
    /// `None` when profiling is off, when per-pass timing is off, or when
    /// `MAX_PASSES` is exhausted — so call sites stay `if let Some(..)` and never
    /// test for support themselves.
    pub fn reserve(&mut self, name: &'static str) -> Option<PassToken> {
        if !crate::profile::is_enabled() || !crate::profile::gpu_passes_enabled() {
            return None;
        }
        self.take_pair(name)
    }

    /// The half of [`reserve`](Self::reserve) past the switches.
    fn take_pair(&mut self, name: &'static str) -> Option<PassToken> {
        let slot = &mut self.slots[self.write];
        if slot.passes.len() >= MAX_PASSES {
            debug_assert!(false, "more than {MAX_PASSES} timed passes in one frame");
            return None;
        }

        let base = slot.next_query;
        slot.next_query += 2;
        slot.passes.push(RecordedPass { name, base });
        Some(PassToken { base })
    }

    /// Stamp the opening half of a reserved pair.
    pub fn open(&self, builder: &mut Recorder, token: Option<PassToken>) {
        let Some(token) = token else {
            return;
        };
        let pool = self.slots[self.write].pool.clone();
        // SAFETY: `base` was reserved from this slot's pool, which was reset
        // this frame and is not in flight.
        builder.write_timestamp(pool, token.base, PipelineStage::BottomOfPipe);
    }

    /// Stamp the closing half of a reserved pair.
    pub fn close(&self, builder: &mut Recorder, token: Option<PassToken>) {
        let Some(token) = token else {
            return;
        };
        let pool = self.slots[self.write].pool.clone();
        // SAFETY: `base + 1` is the closing half of a pair reserved by
        // `reserve` on this slot, reset this frame.
        builder.write_timestamp(pool, token.base + 1, PipelineStage::BottomOfPipe);
    }

    /// Close the last segment's whole-frame pair, mark the slot for readback,
    /// rotate.
    pub fn end_frame(&mut self, builder: &mut Recorder) {
        // No segments means profiling was off when the frame opened, so nothing
        // was stamped and there is no pair to close.
        if let Some(last) = (self.slots[self.write].segments as usize).checked_sub(1) {
            self.close(builder, self.segment(last));
        }
        self.slots[self.write].state = SlotState::Pending;
        self.write = (self.write + 1) % SLOTS;
    }

    /// Non-blocking readback of every retired slot, filing each pass against the
    /// frame it was recorded in.
    ///
    /// A pair that is zero, out of order, or absurd means *that pass* has no span
    /// this frame. Filing a wrong span is worse than filing none, so a bad pair is
    /// dropped rather than substituted.
    pub fn drain_completed(&mut self, profiler: &mut Profiler) {
        for index in 0..SLOTS {
            let slot = &mut self.slots[index];
            if !readable(slot.state, slot.frame_index, profiler.frame_index())
                || slot.passes.is_empty()
            {
                continue;
            }
            let count = slot.next_query as usize;
            let mut results = [0u64; 2 * MAX_PASSES];
            let available = slot
                .pool
                .get_results(
                    0..slot.next_query,
                    &mut results[..count],
                    QueryResultFlags::empty(),
                )
                .unwrap_or(false);
            if !available {
                continue;
            }

            let origin = results[0] & self.valid_mask;
            let mut frame_ms = 0.0f64;
            let mut frame_pairs = 0u32;
            for pass in &slot.passes {
                let start = results[pass.base as usize] & self.valid_mask;
                let end = results[pass.base as usize + 1] & self.valid_mask;
                if start == 0 || end <= start {
                    continue;
                }
                let duration_ns = (end - start) as f64 * self.period_ns as f64;
                // A real pass is never a second long; that's a driver quirk.
                if !duration_ns.is_finite() || duration_ns > 1.0e9 {
                    continue;
                }
                let to_ns = |tick: u64| {
                    slot.anchor_ns
                        + (tick.saturating_sub(origin) as f64 * self.period_ns as f64) as u64
                };
                if pass.name == WHOLE_FRAME_PASS {
                    frame_ms += duration_ns / 1.0e6;
                    frame_pairs += 1;
                }
                profiler.push_gpu_span(
                    slot.frame_index,
                    Span {
                        name: pass.name,
                        depth: 0,
                        start_ns: to_ns(start),
                        end_ns: to_ns(end),
                    },
                );
            }

            // A frame missing a segment would read short, and a short frame
            // looks like a win.
            if frame_pairs == slot.segments {
                self.last_frame_ms = frame_ms as f32;
            }
            slot.state = SlotState::Idle;
        }
    }

    /// Whole-frame GPU milliseconds, the sum of the frame's segment pairs.
    /// Trails the displayed frame.
    pub fn last_frame_ms(&self) -> f32 {
        self.last_frame_ms
    }
}

/// Whether `drain_completed` may read a slot back, with the profiler's frame
/// `open` not yet closed.
///
/// The slot under the write cursor is *not* excluded: it is not reset until the
/// next frame records, and it holds the one frame the run-ahead has certainly
/// retired — see [`SLOTS`].
///
/// A frame reaches the profiler's ring when `end_frame` closes it, and
/// `push_gpu_span` files by matching that index — so a slot recorded in the
/// frame still open has nothing to attach to yet, and waits for the next drain.
/// That case is never reached while the frame is slower than the readback, which
/// is why it went unnoticed: it only matters once results are available in the
/// same frame that recorded them.
fn readable(state: SlotState, recorded: u64, open: u64) -> bool {
    state == SlotState::Pending && recorded < open
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::graph::PassKind;
    use crate::gfx::shadows::MAX_CASCADES;
    use crate::gfx::vulkan::bloom::{MAX_BLOOM_MIPS, mip_count};
    use crate::gfx::vulkan::frame::{FrameConfig, declare};

    /// The latest the GPU may finish frame `g` is when the CPU waits for room to
    /// record `g + FRAMES_IN_FLIGHT`, so that is when this model finishes it. A
    /// CPU frame shorter than the queue's latency gets this close in practice:
    /// the empty scene's 0.66 ms frame lost every whole-frame reading this way.
    #[test]
    fn every_frame_is_read_back_when_the_gpu_finishes_as_late_as_it_may() {
        let in_flight = super::super::FRAMES_IN_FLIGHT as u64;
        let mut state = [SlotState::Idle; SLOTS];
        let mut recorded = [0u64; SLOTS];
        let mut write = 0;
        let mut read = Vec::new();
        let frames = 4 * SLOTS as u64;

        for frame in 0..frames {
            assert!(
                state[write] != SlotState::Pending,
                "frame {} was reset by frame {frame} before it was read back",
                recorded[write],
            );
            recorded[write] = frame;
            state[write] = SlotState::Pending;
            write = (write + 1) % SLOTS;

            for index in 0..SLOTS {
                let finished = recorded[index] + in_flight <= frame;
                if finished && readable(state[index], recorded[index], frame) {
                    state[index] = SlotState::Idle;
                    read.push(recorded[index]);
                }
            }
        }

        let expected: Vec<u64> = (0..=frames - 1 - in_flight).collect();
        assert_eq!(read, expected);
    }

    /// The query pool is sized ahead of knowing what will run, so a frame that
    /// outgrows it drops timings off the end rather than failing to render — the
    /// kind of regression nobody notices until a pass is missing from the
    /// profiler. Bloom is what first made 16 too small; this asserts the busiest
    /// frame that can ship still fits, whatever the chain grows to next.
    ///
    /// Every structural flag is on, and that is the point: three of them used to
    /// be off here, which made the assertion about a frame nobody renders while
    /// the editor's own default — transparency and refraction included — quietly
    /// overran the pool.
    #[test]
    fn the_busiest_frame_fits_the_query_pool() {
        for async_compute in [false, true] {
            busiest_frame_fits(async_compute);
        }
    }

    fn busiest_frame_fits(async_compute: bool) {
        let config = FrameConfig {
            color_format: crate::gfx::format::Format::Bgra8Srgb,
            msaa: false,
            ssao: true,
            ssao_half_res: false,
            contact_shadows: true,
            ssr: true,
            subsurface: true,
            transparency: true,
            refraction: true,
            taa: true,
            auto_exposure: true,
            motion_blur: true,
            dof: true,
            volumetric_fog: false,
            bloom_mips: MAX_BLOOM_MIPS as u8,
            overlay: true,
            outline: true,
            gizmo: true,
            ui: false,
            presents: true,
            shadow_cascades: MAX_CASCADES as u8,
            shadow_resolution: 2048,
            shadow_atlas: 4096,
            async_compute,
            occlusion_culling: false,
        };
        let frame = declare(config).expect("the busiest frame must compile");

        // Raw passes own their submission and are never timed, so they do not
        // draw from the pool. Each segment's whole-frame pair does.
        let timed = frame.graph.segments().len()
            + frame
                .graph
                .order()
                .iter()
                .filter(|&&id| frame.graph.pass_kind(id) != PassKind::Raw)
                .count();

        assert!(
            timed <= MAX_PASSES,
            "the busiest frame (async compute {async_compute}) times {timed} passes \
             but the pool holds {MAX_PASSES}",
        );
    }

    /// The coupling [`SLOTS`] documents, asserted rather than trusted. Too few
    /// slots resets a pool the GPU is still writing; exactly as many leaves the
    /// readback no frame to happen on, which is how three slots behind two
    /// frames in flight became two slots that filed no GPU span at all. Both
    /// failures are silent — the frame renders, the profiler is just wrong.
    #[test]
    #[allow(
        clippy::assertions_on_constants,
        reason = "constant by design: this states a coupling between two consts"
    )]
    fn a_spare_timestamp_slot_beyond_the_frames_in_flight() {
        assert!(
            SLOTS > super::super::FRAMES_IN_FLIGHT,
            "{SLOTS} query slots for {} frames in flight leaves no slot to read \
             back from",
            super::super::FRAMES_IN_FLIGHT,
        );
        assert!(
            SLOTS >= 3,
            "a slot is only eligible to drain a frame after the \
             one it was written in, and never while the cursor sits on it"
        );
    }

    /// A split frame's trailing segment is held back and submitted behind the
    /// next frame's head, so the GPU can sit idle for a whole CPU frame between
    /// the frame's first command and its last. The frame's number is its work,
    /// not that wait: the sleep stands in for the CPU recording the next frame.
    #[test]
    #[ignore = "needs a Vulkan device"]
    fn a_split_frame_is_timed_by_its_work_not_by_the_wait_for_its_tail() {
        use vulkano::VulkanLibrary;
        use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};

        use super::super::record::Pending;

        const IDLE: std::time::Duration = std::time::Duration::from_millis(20);

        let instance = Instance::new(
            VulkanLibrary::new().expect("a Vulkan library"),
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .expect("a headless instance");
        let ctx = VkContext::new(&instance, None);
        let mut timestamps = GpuTimestamps::new(&ctx).expect("a queue that writes timestamps");
        let mut profiler = Profiler::new(4);

        // The renderer's shape: all three segments recorded together, the
        // middle one on the compute queue where there is one.
        let compute_queue = ctx.compute_queue.clone().unwrap_or(ctx.queue.clone());
        timestamps.begin_frame(profiler.frame_index());
        let mut head = Recorder::new(&ctx);
        timestamps.record_resets(&mut head, 3);
        let mut compute = Recorder::for_family(&ctx, compute_queue.queue_family_index());
        timestamps.seam(&mut head, &mut compute, 1);
        let mut tail = Recorder::new(&ctx);
        timestamps.seam(&mut compute, &mut tail, 2);
        timestamps.end_frame(&mut tail);

        head.submit_and_wait(&ctx);
        let (commands, keep) = compute.end();
        super::super::record::submit_frame(&ctx, &compute_queue, commands, keep, &[], &[]).retire();
        std::thread::sleep(IDLE);
        tail.submit_and_wait(&ctx);

        profiler.end_frame();
        timestamps.drain_completed(&mut profiler);
        let frame_ms = timestamps.last_frame_ms();
        assert!(frame_ms > 0.0, "the frame was not read back");
        assert!(
            frame_ms < IDLE.as_secs_f32() * 1.0e3 / 4.0,
            "three empty segments read {frame_ms:.3} ms: the {IDLE:?} the GPU sat \
             idle before the tail was counted as the frame's",
        );
    }

    /// The cap must actually be reachable by a real window, or the chain silently
    /// runs shorter than it was tuned for.
    #[test]
    fn a_common_display_gets_the_full_bloom_chain() {
        assert_eq!(mip_count([2560, 1440]), MAX_BLOOM_MIPS as u8);
    }
}
