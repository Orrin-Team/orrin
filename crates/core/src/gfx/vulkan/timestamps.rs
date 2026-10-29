//! Per-pass GPU timing, feeding [`Profiler::push_gpu_span`].
//!
//! One timestamp pair per pass, plus a reserved pair spanning the whole frame so
//! the HUD's single GPU number keeps its meaning. Three properties shape the
//! design:
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

/// Passes timed per frame, including the reserved whole-frame pair. Costs
/// `2 * MAX_PASSES` queries per slot whether used or not; passes beyond it are
/// dropped rather than mis-attributed.
///
/// The busiest frame that ships is four shadow cascades, the punctual atlas, the
/// prepass, two SSAO passes, the contact-shadow march, the forward pass, three
/// diffusion passes, four reflection passes, two transparency passes, three
/// refraction passes, the temporal resolve, four lens passes, three shutter
/// passes, two metering dispatches, an eleven-pass bloom chain and the tonemap,
/// plus the whole-frame pair. Bloom is what made the old 16 too small, and it
/// grows with `MAX_BLOOM_MIPS`: a chain of `n` levels is `2n - 1` passes, so
/// raising that cap means raising this one.
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
/// But a slot must also be *read* before it is reused, and `drain_completed`
/// declines two: the slot being written, and any slot whose frame the profiler
/// has not closed yet. Frame `N`'s slot is therefore first eligible on frame
/// `N + 1`, when the write cursor sits at `N + 2` — the same slot when `SLOTS`
/// is two, which is why two silently files no GPU spans at all rather than
/// filing wrong ones. One spare slot is what buys the readback its window.
///
/// Derived from the run-ahead rather than written down beside it, because the
/// two disagreeing costs timings rather than frames, and nothing about a frame
/// looks wrong while it happens.
const SLOTS: usize = super::FRAMES_IN_FLIGHT + 1;

/// Name of the reserved pair covering the whole frame. Always query 0/1, which
/// also makes query 0 a guaranteed-written origin for the anchor below.
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
        slot.frame_index = profiler_frame;
        slot.anchor_ns = crate::profile::now_ns();
        slot.state = SlotState::Idle;
    }

    /// Reset this frame's queries and open the reserved whole-frame pair.
    ///
    /// Must be recorded before the first `begin_rendering`: a reset inside a
    /// render pass instance is invalid, and which passes will run isn't known
    /// yet, so the entire pool is reset in one go.
    pub fn record_resets(&mut self, builder: &mut Recorder) {
        let pool = self.slots[self.write].pool.clone();
        // SAFETY: outside any render pass, and this slot is not in flight —
        // there are more slots than frames the CPU may be ahead by, so the frame
        // that last wrote this one has been waited on by
        // `RunAhead::wait_for_room`.
        builder.reset_query_pool(pool, 0..(2 * MAX_PASSES as u32));
        // Reserved first, so it is always query 0/1 and query 0 is guaranteed
        // written — `drain_completed` uses it as the frame's tick origin. Its
        // closing stamp comes from `end_frame`, which knows that fixed position.
        //
        // Reserved through `stamp` rather than `begin_pass`, because it must not
        // answer to the per-pass switch: the whole-frame pair is the one number
        // that has to survive turning per-pass timing off, since comparing the
        // two is the entire point of being able to.
        if crate::profile::is_enabled() {
            let whole_frame = self.take_pair(WHOLE_FRAME_PASS);
            self.open(builder, whole_frame);
        }
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

    /// Close the reserved whole-frame pair, mark the slot for readback, rotate.
    pub fn end_frame(&mut self, builder: &mut Recorder) {
        // Empty means profiling was off when the frame opened, so nothing was
        // stamped and there is no pair to close.
        if !self.slots[self.write].passes.is_empty() {
            self.close(builder, Some(PassToken { base: 0 }));
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
            if index == self.write {
                continue;
            }
            let slot = &mut self.slots[index];
            if slot.state != SlotState::Pending || slot.passes.is_empty() {
                continue;
            }
            // A frame reaches the profiler's ring when `end_frame` closes it, and
            // `push_gpu_span` files by matching that index — so a slot recorded
            // in the frame still open has nothing to attach to yet. Leave it
            // pending and take it on the next drain.
            //
            // Never reached while the frame was slower than the readback, which
            // is why it went unnoticed: this only matters once results are
            // available in the same frame that recorded them.
            if slot.frame_index >= profiler.frame_index() {
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
                    self.last_frame_ms = (duration_ns / 1.0e6) as f32;
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

            slot.state = SlotState::Idle;
        }
    }

    /// Whole-frame GPU milliseconds, from the reserved pair. Trails the displayed
    /// frame.
    pub fn last_frame_ms(&self) -> f32 {
        self.last_frame_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::graph::PassKind;
    use crate::gfx::shadows::MAX_CASCADES;
    use crate::gfx::vulkan::bloom::{MAX_BLOOM_MIPS, mip_count};
    use crate::gfx::vulkan::frame::{FrameConfig, declare};

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
            shadow_cascades: MAX_CASCADES as u8,
            shadow_resolution: 2048,
            shadow_atlas: 4096,
            async_compute: false,
            gpu_culling: false,
            occlusion_culling: false,
        };
        let frame = declare(config).expect("the busiest frame must compile");

        // Raw passes own their submission and are never timed, so they do not
        // draw from the pool. The whole-frame pair does.
        let timed = 1 + frame
            .graph
            .order()
            .iter()
            .filter(|&&id| frame.graph.pass_kind(id) != PassKind::Raw)
            .count();

        assert!(
            timed <= MAX_PASSES,
            "the busiest frame times {timed} passes but the pool holds {MAX_PASSES}",
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

    /// The cap must actually be reachable by a real window, or the chain silently
    /// runs shorter than it was tuned for.
    #[test]
    fn a_common_display_gets_the_full_bloom_chain() {
        assert_eq!(mip_count([2560, 1440]), MAX_BLOOM_MIPS as u8);
    }
}
