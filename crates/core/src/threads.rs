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
}
