//! Frame profiler: one timing model for CPU phases and GPU passes.
//!
//! A [`Scope`] guard times a region and files it against the frame in progress;
//! [`Profiler::end_frame`] closes that frame and keeps the last few hundred in a
//! ring.
//!
//! The design constraint that shapes everything here is that **GPU spans arrive
//! late**. Timestamp readback trails the displayed frame by one, so a pass's
//! timing is known only after its frame has already closed. Hence
//! [`Profiler::push_gpu_span`], which files a finished span against an *earlier*
//! frame index, and hence a ring of whole frames rather than the running
//! averages in [`FrameStats`](crate::stats::FrameStats) — an average has nowhere
//! to put a measurement that belongs to the past.
//!
//! Collection is per-thread and only the thread that calls `end_frame` is
//! drained, which matches the engine's single-threaded frame loop. Threaded
//! command recording would need each thread's buffer merged here.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::LazyLock;
use std::time::Instant;

/// Frames kept for aggregation; matches `FrameStats::HISTORY` so the table and
/// the graph describe the same window.
const HISTORY: usize = 240;

/// All spans are stamped as nanoseconds from this instant, so a CPU span and a
/// GPU span converted from device ticks share one timeline.
static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

static ENABLED: AtomicBool = AtomicBool::new(true);

/// Nanoseconds since the profiler's epoch. Also the conversion target for GPU
/// timestamps, which arrive in device ticks.
pub fn now_ns() -> u64 {
    EPOCH.elapsed().as_nanos() as u64
}

pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Turning collection off leaves the API in place and costs one relaxed load per
/// scope. Toggling mid-frame is safe but produces one frame of ragged nesting.
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lane {
    Cpu,
    Gpu,
}

/// One timed region. `name` is `&'static str` rather than `String` because a
/// per-scope allocation would put the allocator inside the measurement.
#[derive(Clone, Copy, Debug)]
pub struct Span {
    pub name: &'static str,
    /// Nesting level within its lane; 0 is a phase opened directly by the frame
    /// loop.
    pub depth: u16,
    pub start_ns: u64,
    pub end_ns: u64,
}

impl Span {
    pub fn duration_ms(&self) -> f32 {
        self.end_ns.saturating_sub(self.start_ns) as f32 / 1.0e6
    }
}

#[derive(Default)]
pub struct Frame {
    pub index: u64,
    pub cpu: Vec<Span>,
    pub gpu: Vec<Span>,
}

impl Frame {
    pub fn lane(&self, lane: Lane) -> &[Span] {
        match lane {
            Lane::Cpu => &self.cpu,
            Lane::Gpu => &self.gpu,
        }
    }

    /// Total across top-level spans only, so nested scopes aren't counted twice.
    pub fn root_ms(&self, lane: Lane) -> f32 {
        self.lane(lane)
            .iter()
            .filter(|span| span.depth == 0)
            .map(Span::duration_ms)
            .sum()
    }

    fn clear(&mut self) {
        self.cpu.clear();
        self.gpu.clear();
    }
}

#[derive(Default)]
struct Collector {
    open: u16,
    spans: Vec<Span>,
}

thread_local! {
    static COLLECTOR: RefCell<Collector> = RefCell::new(Collector::default());
}

/// Times its enclosing region and files a [`Span`] on drop. Create one with
/// [`scope`] or the [`profile_scope!`](crate::profile_scope) macro.
pub struct Scope {
    name: &'static str,
    depth: u16,
    start_ns: u64,
}

/// `None` while profiling is off, which makes the guard's drop a no-op.
pub fn scope(name: &'static str) -> Option<Scope> {
    if !is_enabled() {
        return None;
    }
    let depth = COLLECTOR.with(|cell| {
        let mut collector = cell.borrow_mut();
        let depth = collector.open;
        collector.open += 1;
        depth
    });
    Some(Scope {
        name,
        depth,
        start_ns: now_ns(),
    })
}

impl Drop for Scope {
    fn drop(&mut self) {
        let end_ns = now_ns();
        COLLECTOR.with(|cell| {
            let mut collector = cell.borrow_mut();
            // Saturates because enabling mid-frame can create a guard whose
            // matching increment never happened.
            collector.open = collector.open.saturating_sub(1);
            collector.spans.push(Span {
                name: self.name,
                depth: self.depth,
                start_ns: self.start_ns,
                end_ns,
            });
        });
    }
}

/// Time the enclosing block, from the macro call to the end of the scope:
///
/// ```ignore
/// {
///     profile_scope!("collision");
///     collision::run(&mut world);
/// }
/// ```
///
/// The name must be `&'static str`; anything borrowed fails to compile, which is
/// what keeps allocation out of the timed region.
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _orrin_profile_scope = $crate::profile::scope($name);
    };
}

pub struct Profiler {
    ring: VecDeque<Frame>,
    next_index: u64,
    capacity: usize,
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new(HISTORY)
    }
}

impl Profiler {
    pub fn new(capacity: usize) -> Self {
        Self {
            ring: VecDeque::with_capacity(capacity),
            next_index: 0,
            capacity: capacity.max(1),
        }
    }

    /// Index of the frame being recorded right now — the one `end_frame` is
    /// about to close. A pass that writes GPU timestamps stamps itself with this
    /// so the readback, a frame or more later, can find its way home.
    pub fn frame_index(&self) -> u64 {
        self.next_index
    }

    /// Close the frame in progress: drain this thread's finished spans into a
    /// new record and evict the oldest if the ring is full.
    pub fn end_frame(&mut self) {
        let mut frame = if self.ring.len() == self.capacity {
            // Recycled rather than dropped so a steady-state frame allocates
            // nothing; the profiler must stay out of its own measurements.
            let mut oldest = self.ring.pop_front().unwrap_or_default();
            oldest.clear();
            oldest
        } else {
            Frame::default()
        };

        frame.index = self.next_index;
        self.next_index += 1;

        COLLECTOR.with(|cell| {
            let mut collector = cell.borrow_mut();
            frame.cpu.extend_from_slice(&collector.spans);
            collector.spans.clear();
            collector.open = 0;
        });

        self.ring.push_back(frame);
    }

    /// File a finished GPU span against the frame it was recorded in. Returns
    /// `false` if that frame has already aged out of the ring, which is the
    /// normal outcome for a readback that stalled for hundreds of frames.
    pub fn push_gpu_span(&mut self, frame_index: u64, span: Span) -> bool {
        match self.ring.iter_mut().find(|frame| frame.index == frame_index) {
            Some(frame) => {
                frame.gpu.push(span);
                true
            }
            None => false,
        }
    }

    pub fn latest(&self) -> Option<&Frame> {
        self.ring.back()
    }

    pub fn frame(&self, index: u64) -> Option<&Frame> {
        self.ring.iter().find(|frame| frame.index == index)
    }

    pub fn frames(&self) -> impl DoubleEndedIterator<Item = &Frame> {
        self.ring.iter()
    }

    /// Per-name rows over the whole ring, ordered slowest-last-frame first.
    ///
    /// A name appearing several times in one frame is summed within that frame,
    /// so `avg_ms` reads as "cost per frame" rather than "cost per call" — the
    /// question being asked of a phase table.
    ///
    /// "Latest" and the average's denominator both mean *frames that carry spans
    /// in this lane*, not all retained frames. GPU spans arrive a frame or more
    /// late, so the newest frames never have any: counting them would peg the
    /// GPU lane's `last_ms` at zero and drag every average down.
    pub fn aggregate(&self, lane: Lane) -> Vec<Row> {
        let mut rows: Vec<Row> = Vec::new();
        let mut index_of: std::collections::HashMap<&'static str, usize> =
            std::collections::HashMap::new();
        let latest = self
            .ring
            .iter()
            .rev()
            .find(|frame| !frame.lane(lane).is_empty())
            .map(|frame| frame.index);
        let mut populated = 0usize;

        for frame in &self.ring {
            if frame.lane(lane).is_empty() {
                continue;
            }
            populated += 1;
            let mut frame_totals: Vec<(usize, f32, u32)> = Vec::new();
            for span in frame.lane(lane) {
                let row_index = *index_of.entry(span.name).or_insert_with(|| {
                    rows.push(Row {
                        name: span.name,
                        depth: span.depth,
                        last_ms: 0.0,
                        avg_ms: 0.0,
                        max_ms: 0.0,
                        calls: 0,
                    });
                    rows.len() - 1
                });
                rows[row_index].depth = rows[row_index].depth.min(span.depth);
                match frame_totals.iter_mut().find(|(i, _, _)| *i == row_index) {
                    Some((_, ms, calls)) => {
                        *ms += span.duration_ms();
                        *calls += 1;
                    }
                    None => frame_totals.push((row_index, span.duration_ms(), 1)),
                }
            }

            let is_latest = Some(frame.index) == latest;
            for (row_index, ms, calls) in frame_totals {
                let row = &mut rows[row_index];
                row.avg_ms += ms;
                row.max_ms = row.max_ms.max(ms);
                if is_latest {
                    row.last_ms = ms;
                    row.calls = calls;
                }
            }
        }

        let frames = populated.max(1) as f32;
        for row in &mut rows {
            row.avg_ms /= frames;
        }
        rows.sort_by(|a, b| b.last_ms.total_cmp(&a.last_ms));
        rows
    }
}

/// One line of the phase table: a name's cost across the retained window.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub name: &'static str,
    pub depth: u16,
    /// Cost in the most recent frame.
    pub last_ms: f32,
    /// Mean cost per frame that carried spans in this lane, including frames the
    /// name itself never appeared in.
    pub avg_ms: f32,
    pub max_ms: f32,
    /// Times the name was entered in the most recent frame.
    pub calls: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // `ENABLED` is process-global, so tests that toggle it can't run beside
    // tests that record spans.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn span(name: &'static str, depth: u16, start_ns: u64, end_ns: u64) -> Span {
        Span { name, depth, start_ns, end_ns }
    }

    #[test]
    fn scopes_record_nesting_depth() {
        let _guard = SERIAL.lock().unwrap();
        set_enabled(true);
        let mut profiler = Profiler::new(4);
        profiler.end_frame(); // discard anything another test left behind

        {
            profile_scope!("outer");
            {
                profile_scope!("inner");
            }
            {
                profile_scope!("sibling");
            }
        }
        profiler.end_frame();

        let frame = profiler.latest().unwrap();
        let depths: Vec<_> = frame.cpu.iter().map(|s| (s.name, s.depth)).collect();
        // Inner scopes close first, so they land ahead of their parent.
        assert_eq!(
            depths,
            vec![("inner", 1), ("sibling", 1), ("outer", 0)]
        );
    }

    #[test]
    fn disabled_scopes_record_nothing() {
        let _guard = SERIAL.lock().unwrap();
        let mut profiler = Profiler::new(4);
        profiler.end_frame();

        set_enabled(false);
        {
            profile_scope!("ignored");
        }
        profiler.end_frame();
        set_enabled(true);

        assert!(profiler.latest().unwrap().cpu.is_empty());
    }

    #[test]
    fn gpu_spans_file_against_an_earlier_frame() {
        let mut profiler = Profiler::new(4);
        profiler.end_frame(); // frame 0
        let recorded_in = profiler.latest().unwrap().index;
        profiler.end_frame(); // frame 1: the readback arrives here

        assert!(profiler.push_gpu_span(recorded_in, span("forward", 0, 0, 2_000_000)));

        let frame = profiler.frame(recorded_in).unwrap();
        assert_eq!(frame.gpu.len(), 1);
        assert_eq!(frame.gpu[0].duration_ms(), 2.0);
        // The frame the readback happened in stays empty.
        assert!(profiler.latest().unwrap().gpu.is_empty());
    }

    #[test]
    fn gpu_spans_for_evicted_frames_are_dropped() {
        let mut profiler = Profiler::new(2);
        for _ in 0..4 {
            profiler.end_frame();
        }
        assert!(!profiler.push_gpu_span(0, span("forward", 0, 0, 1)));
        assert!(profiler.push_gpu_span(3, span("forward", 0, 0, 1)));
    }

    /// The newest frames never carry GPU spans yet, so a lane's "latest" has to
    /// mean its newest *populated* frame or the column reads zero forever.
    #[test]
    fn gpu_lane_reports_the_newest_frame_that_has_spans() {
        let mut profiler = Profiler::new(8);
        for _ in 0..4 {
            profiler.end_frame();
        }
        profiler.push_gpu_span(1, span("forward", 0, 0, 3_000_000));
        // Frames 2 and 3 exist but their readback hasn't arrived.

        let rows = profiler.aggregate(Lane::Gpu);
        assert_eq!(rows[0].name, "forward");
        assert_eq!(rows[0].last_ms, 3.0);
        // Averaged over the one frame with GPU data, not all four retained.
        assert_eq!(rows[0].avg_ms, 3.0);
    }

    #[test]
    fn ring_evicts_oldest_and_keeps_indices_monotonic() {
        let mut profiler = Profiler::new(3);
        for _ in 0..5 {
            profiler.end_frame();
        }
        let indices: Vec<_> = profiler.frames().map(|f| f.index).collect();
        assert_eq!(indices, vec![2, 3, 4]);
        assert_eq!(profiler.frame_index(), 5);
    }

    #[test]
    fn aggregate_sums_repeats_within_a_frame_and_averages_across_them() {
        let mut profiler = Profiler::new(4);
        profiler.end_frame(); // frame 0
        profiler.end_frame(); // frame 1

        // 1 ms in frame 0; 2 ms split across two calls in frame 1.
        profiler.push_gpu_span(0, span("forward", 0, 0, 1_000_000));
        profiler.push_gpu_span(1, span("forward", 0, 0, 1_500_000));
        profiler.push_gpu_span(1, span("forward", 0, 2_000_000, 2_500_000));
        profiler.push_gpu_span(1, span("ssao", 0, 0, 4_000_000));

        let rows = profiler.aggregate(Lane::Gpu);
        // Ordered by cost in the latest frame, so ssao (4 ms) leads forward (2 ms).
        assert_eq!(rows[0].name, "ssao");
        assert_eq!(rows[1].name, "forward");

        let forward = rows[1];
        assert_eq!(forward.last_ms, 2.0);
        assert_eq!(forward.calls, 2);
        assert_eq!(forward.max_ms, 2.0);
        // (1 ms + 2 ms) over the two retained frames.
        assert_eq!(forward.avg_ms, 1.5);
    }
}
