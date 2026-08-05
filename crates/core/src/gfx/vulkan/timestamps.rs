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

use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType};
use vulkano::sync::PipelineStage;

use super::context::VkContext;
use crate::profile::{Profiler, Span};

/// Passes timed per frame, including the reserved whole-frame pair. Costs
/// `2 * MAX_PASSES` queries per slot whether used or not; passes beyond it are
/// dropped rather than mis-attributed.
///
/// The busiest frame that ships is four shadow cascades, three SSAO passes, the
/// forward pass, two metering dispatches, an eleven-pass bloom chain and the
/// tonemap — 22, plus the whole-frame pair. Bloom is what made the old 16 too
/// small, and it grows with `MAX_BLOOM_MIPS`: a chain of `n` levels is `2n - 1`
/// passes, so raising that cap means raising this one.
const MAX_PASSES: usize = 32;

/// Frame slots in rotation. Two would be correct only while `previous_frame_end`
/// is a single fence that retires frame N-1 before N records; three removes that
/// coupling, so growing frames-in-flight later can't silently corrupt timings.
const SLOTS: usize = 3;

/// Name of the reserved pair covering the whole frame. Always query 0/1, which
/// also makes query 0 a guaranteed-written origin for the anchor below.
const WHOLE_FRAME_PASS: &str = "frame";

/// A pass whose opening timestamp has been recorded. Consumed by
/// [`GpuTimestamps::end_pass`]; dropping one without ending it leaves a query
/// that never gets written, which the pair guard in `drain_completed` discards.
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
    /// Must be recorded before the first `begin_render_pass`: a reset inside a
    /// render pass is invalid, and which passes will run isn't known yet, so the
    /// entire pool is reset in one go.
    pub fn record_resets(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        let pool = self.slots[self.write].pool.clone();
        // SAFETY: outside any render pass, and this slot is not in flight —
        // `end_frame` only rotates onto a slot that `drain_completed` retired.
        unsafe {
            builder
                .reset_query_pool(pool, 0..(2 * MAX_PASSES as u32))
                .unwrap();
        }
        // Reserved first, so it is always query 0/1 and query 0 is guaranteed
        // written — `drain_completed` uses it as the frame's tick origin. Its
        // closing stamp comes from `end_frame`, which knows that fixed position.
        drop(self.begin_pass(builder, WHOLE_FRAME_PASS));
    }

    /// Stamp the opening timestamp for `name` and reserve its pair.
    ///
    /// `None` when profiling is off or `MAX_PASSES` is exhausted, so call sites
    /// stay `if let Some(..)` and never test for support themselves.
    pub fn begin_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        name: &'static str,
    ) -> Option<PassToken> {
        if !crate::profile::is_enabled() {
            return None;
        }
        let slot = &mut self.slots[self.write];
        if slot.passes.len() >= MAX_PASSES {
            debug_assert!(false, "more than {MAX_PASSES} timed passes in one frame");
            return None;
        }

        let base = slot.next_query;
        slot.next_query += 2;
        slot.passes.push(RecordedPass { name, base });

        let pool = slot.pool.clone();
        // SAFETY: `base` was just reserved from this slot's pool, which was reset
        // this frame and is not in flight.
        unsafe {
            builder
                .write_timestamp(pool, base, PipelineStage::BottomOfPipe)
                .unwrap();
        }
        Some(PassToken { base })
    }

    pub fn end_pass(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        token: Option<PassToken>,
    ) {
        let Some(token) = token else {
            return;
        };
        let pool = self.slots[self.write].pool.clone();
        // SAFETY: `base + 1` is the closing half of a pair reserved by
        // `begin_pass` on this slot, reset this frame.
        unsafe {
            builder
                .write_timestamp(pool, token.base + 1, PipelineStage::BottomOfPipe)
                .unwrap();
        }
    }

    /// Close the reserved whole-frame pair, mark the slot for readback, rotate.
    pub fn end_frame(&mut self, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        // Empty means profiling was off when the frame opened, so nothing was
        // stamped and there is no pair to close.
        if !self.slots[self.write].passes.is_empty() {
            self.end_pass(builder, Some(PassToken { base: 0 }));
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
    #[test]
    fn the_busiest_frame_fits_the_query_pool() {
        let config = FrameConfig {
            color_format: vulkano::format::Format::B8G8R8A8_SRGB,
            ssao: true,
            auto_exposure: true,
            bloom_mips: MAX_BLOOM_MIPS as u8,
            overlay: true,
            shadow_cascades: MAX_CASCADES as u8,
            shadow_resolution: 2048,
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

    /// The cap must actually be reachable by a real window, or the chain silently
    /// runs shorter than it was tuned for.
    #[test]
    fn a_common_display_gets_the_full_bloom_chain() {
        assert_eq!(mip_count([2560, 1440]), MAX_BLOOM_MIPS as u8);
    }
}
