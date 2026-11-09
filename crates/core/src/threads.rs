//! The engine's one thread pool, and the rule for dispatching into it.
//!
//! Everything parallel in the engine runs on a single rayon pool built here,
//! once, before any work is handed to it. Two things about that are deliberate.
//!
//! **The pool is smaller than the machine.** The default is one thread short of
//! `available_parallelism`, because the thread that calls into the pool is not
//! idle while it waits — rayon runs part of the split on the caller, and outside
//! a parallel region the main thread is the one doing the frame. A pool sized to
//! every core puts a worker on the main thread's core and the two then take
//! turns; sized one short, the caller *is* the missing worker.
//!
//! **`ORRIN_THREADS=1` is genuinely serial**, not a pool of one. Every later
//! parallelism change is judged against its own single-threaded build, so the
//! baseline has to be a build that does not enter rayon at all: [`map`] below
//! takes a plain iterator path rather than paying split, dispatch and join to
//! arrive back at one thread. A pool of one would fold that overhead into the
//! baseline and make every speed-up look better than it is.
//!
//! **Imports are one other place work runs in parallel, and never here.**
//! A texture import compresses on threads of its own, a quarter of the machine
//! and at most four ([`import_threads`]), and stays out of the pool entirely:
//! rayon workers drain their own queued splits before they take new work, so
//! an import's tiles queued on the pool put every parallel step of the frame
//! behind the whole import. Measured on a 7800X3D (16 threads) with a 2048²
//! BC7 chain encoding in a loop beside a 4000-object stress frame: through the
//! pool, frame p95 went from 1.1 to 16.4 ms; on 15 threads of its own, 3.0 ms;
//! on 4, 1.17 ms, where one thread's 200 ms became 54 ms.
//!
//! **Audio is the other, on two threads that are not workers either.** The
//! device's callback runs on a thread the OS audio server schedules against a
//! deadline, and sound files decode on one of their own (`audio::load`) for the
//! import threads' reason. Neither takes frame work.
//!
//! Workers are named `orrin-worker-N`, so a `perf` profile or a RenderDoc
//! capture says which thread a sample came from, and they are taken out of
//! profile collection as they start — see
//! [`profile::suppress_on_this_thread`](crate::profile::suppress_on_this_thread).

use std::sync::OnceLock;

use rayon::prelude::*;

/// Built exactly once; the `OnceLock` is what makes [`init`] safe to call from
/// the engine, a test and a benchmark without any of them knowing about the
/// others.
static POOL: OnceLock<()> = OnceLock::new();

/// How many workers the pool is asked for.
///
/// `ORRIN_THREADS` overrides the default, and is the control every later A/B
/// measurement uses. Zero and unparseable values fall back to the default
/// rather than to rayon's own meaning for zero, which is "all of them" — the one
/// answer this module exists to avoid.
fn requested() -> usize {
    if let Some(threads) = std::env::var("ORRIN_THREADS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&threads| threads > 0)
    {
        return threads;
    }
    let cores = std::thread::available_parallelism().map_or(1, |cores| cores.get());
    cores.saturating_sub(1).max(1)
}

/// Build the global pool. Idempotent, and cheap enough after the first call to
/// sit at the top of anything that might be the first thing to run.
///
/// Failure is not fatal: something else having built the global pool first means
/// the sizing above was not applied, which is worth one line on stderr and
/// nothing more — [`count`] then reports what is actually there.
pub fn init() {
    POOL.get_or_init(|| {
        let threads = requested();
        let built = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("orrin-worker-{index}"))
            .start_handler(|_| crate::profile::suppress_on_this_thread())
            .build_global();
        if let Err(error) = built {
            tracing::warn!(
                "the global thread pool was already built ({error}); \
                 ORRIN_THREADS has no effect in this process"
            );
        }
    });
}

/// Threads available to the pool the caller is in — the global one, unless this
/// is running inside a `ThreadPool::install`, which is how the perf harness
/// sweeps the setting without a fresh process per point.
pub fn count() -> usize {
    init();
    rayon::current_num_threads()
}

/// How many threads an import shares its texture compression across, for a
/// machine of `cores` logical cores whose pool has `pool` workers.
pub fn import_threads_for(cores: usize, pool: usize) -> usize {
    if pool <= 1 {
        return 1;
    }
    (cores / 4).clamp(1, 4)
}

/// [`import_threads_for`] this machine and this pool.
pub fn import_threads() -> usize {
    let cores = std::thread::available_parallelism().map_or(1, |cores| cores.get());
    import_threads_for(cores, count())
}

/// Whether dispatching is worth anything at all. False under `ORRIN_THREADS=1`,
/// which is the contract the serial baseline rests on.
pub fn is_parallel() -> bool {
    count() > 1
}

/// Map `items` across the pool, in order.
///
/// The results keep `items`' order whichever path is taken, so a caller that
/// indexes the output against the input — every one of them so far — does not
/// have to care which ran.
///
/// `f` is called once per element and must not depend on being called in order:
/// the parallel path splits the slice arbitrarily.
pub fn map<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Send + Sync,
{
    // The serial path is a plain iterator rather than a one-thread pool: see
    // the module docs on what `ORRIN_THREADS=1` has to mean.
    if !is_parallel() {
        return items.iter().map(f).collect();
    }
    items.par_iter().map(f).collect()
}

/// `f(y, rows)` for every row of `maps`, across the pool, where `rows` holds
/// row `y` of each map.
///
/// Several maps at once because a generator often writes more than one — an
/// albedo, a normal and a height from the same texel — and each row of all of
/// them is one task. Every map must be the same whole number of `row_len` rows.
pub fn rows_mut<T, F, const N: usize>(maps: [&mut [T]; N], row_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, [&mut [T]; N]) + Send + Sync,
{
    let rows = maps.first().map_or(0, |map| map.len() / row_len.max(1));
    assert!(
        maps.iter().all(|map| map.len() == rows * row_len),
        "every map must be {rows} rows of {row_len}"
    );
    let mut chunks = maps.map(|map| map.chunks_mut(row_len.max(1)));
    let rows: Vec<[&mut [T]; N]> = (0..rows)
        .map(|_| {
            chunks
                .each_mut()
                .map(|chunk| chunk.next().expect("counted above"))
        })
        .collect();
    if !is_parallel() {
        rows.into_iter().enumerate().for_each(|(y, row)| f(y, row));
        return;
    }
    rows.into_par_iter()
        .enumerate()
        .for_each(|(y, row)| f(y, row));
}

/// Work handed to the pool now and collected with [`Deferred::wait`], so the
/// caller can get on with something that has to stay on its own thread — an
/// upload into a backend that is not `Send` — while it runs.
///
/// Under `ORRIN_THREADS=1` nothing is spawned and nothing runs early: the work
/// happens inside `wait`, at the point the code read before it was deferred, so
/// the serial baseline is the program it always was.
///
/// `wait` blocks its thread on the result, so it belongs on a thread outside the
/// pool. A worker that waits holds its slot while the task queues behind it.
pub fn defer<T, F>(work: F) -> Deferred<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    if !is_parallel() {
        return Deferred(Pending::Serial(Box::new(work)));
    }
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    // A panic in a spawned job aborts the process by default; caught here, it is
    // re-raised by `wait` on the thread that asked for the result instead.
    rayon::spawn(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
        let _ = sender.send(outcome);
    });
    Deferred(Pending::Spawned(receiver))
}

/// See [`defer`].
pub struct Deferred<T>(Pending<T>);

enum Pending<T> {
    Serial(Box<dyn FnOnce() -> T + Send>),
    Spawned(std::sync::mpsc::Receiver<std::thread::Result<T>>),
}

impl<T> Deferred<T> {
    /// The work's result, blocking until it is ready.
    pub fn wait(self) -> T {
        match self.0 {
            Pending::Serial(work) => work(),
            Pending::Spawned(receiver) => {
                match receiver.recv().expect("a deferred task always reports") {
                    Ok(value) => value,
                    Err(panic) => std::panic::resume_unwind(panic),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Order is the property every caller depends on, and the one a parallel
    /// split is most likely to lose.
    #[test]
    fn map_preserves_order() {
        let items: Vec<usize> = (0..1_000).collect();
        let doubled = map(&items, |&value| value * 2);
        assert_eq!(doubled.len(), items.len());
        for (index, value) in doubled.iter().enumerate() {
            assert_eq!(*value, index * 2);
        }
    }

    /// A quarter of the machine, never more than four, never none; and
    /// serial when the pool is, so `ORRIN_THREADS=1` stays a serial build.
    #[test]
    fn an_import_takes_a_quarter_of_the_machine_up_to_four_threads() {
        assert_eq!(import_threads_for(16, 15), 4);
        assert_eq!(import_threads_for(8, 7), 2);
        assert_eq!(import_threads_for(4, 3), 1);
        assert_eq!(import_threads_for(2, 1), 1);
        assert_eq!(import_threads_for(64, 63), 4);
        assert_eq!(import_threads_for(16, 1), 1);
    }

    /// An empty input must not reach for a pool at all, which is the shape a
    /// model with no textures takes.
    #[test]
    fn map_over_nothing_is_nothing() {
        let items: [u32; 0] = [];
        assert!(map(&items, |&value| value).is_empty());
    }

    /// A pool of one is the harness's way of measuring the serial baseline, so
    /// `is_parallel` has to read the pool it is *in*, not the global one.
    #[test]
    fn a_single_threaded_pool_takes_the_serial_path() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("a one-thread pool");
        pool.install(|| {
            assert!(!is_parallel());
            let items: Vec<usize> = (0..64).collect();
            assert_eq!(map(&items, |&value| value + 1)[63], 64);
        });
    }

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("a pool")
    }

    /// Every row of every map is handed out exactly once, at its own index, and
    /// rows of the same index arrive together — on both paths.
    #[test]
    fn rows_mut_hands_each_row_its_own_slices() {
        for threads in [1, 4] {
            let (mut a, mut b) = (vec![0u32; 12], vec![0u32; 12]);
            pool(threads).install(|| {
                rows_mut([&mut a[..], &mut b[..]], 3, |y, [ra, rb]| {
                    ra.fill(y as u32 + 1);
                    rb.fill(y as u32 + 101);
                })
            });
            assert_eq!(a, [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4], "under {threads}");
            assert_eq!(b[11], 104, "under {threads}");
        }
    }

    /// The value comes back whichever path ran.
    #[test]
    fn a_deferred_task_returns_its_value_on_either_path() {
        for threads in [1, 4] {
            let value = pool(threads).install(|| defer(|| (0..100u32).sum::<u32>()).wait());
            assert_eq!(value, 4950, "under {threads} threads");
        }
    }

    /// The serial path is the baseline every parallel change is judged
    /// against, so under one thread nothing may run early: the work happens
    /// where it is waited on, exactly as the code read before it was deferred.
    #[test]
    fn under_one_thread_a_deferred_task_runs_where_it_is_waited_on() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        pool(1).install(|| {
            let ran = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&ran);
            let deferred = defer(move || flag.store(true, Ordering::SeqCst));
            assert!(!ran.load(Ordering::SeqCst), "ran before it was waited on");
            deferred.wait();
            assert!(ran.load(Ordering::SeqCst));
        });
    }

    /// A task on the pool that panics would otherwise abort the process,
    /// which is rayon's default for a spawned job; the caller is where the
    /// panic belongs.
    #[test]
    #[should_panic(expected = "deferred failure")]
    fn a_panic_in_a_deferred_task_reaches_the_caller() {
        pool(4).install(|| defer(|| -> u32 { panic!("deferred failure") }).wait());
    }
}
