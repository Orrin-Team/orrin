//! A `VkPipelineCache` that outlives the process.
//!
//! Every `GraphicsPipeline::new` and `ComputePipeline::new` in the renderer hands
//! the driver SPIR-V and gets machine code back, and without a cache it does that
//! from scratch on every launch — NVIDIA's guide lists "use pipeline cache" among
//! its dos for exactly this reason, and it is the one item on that list this
//! renderer was missing outright.
//!
//! The cache is a *machine* artifact, not a project one, which is why it lives
//! under the user's cache directory rather than in `<project>/.orrin/`: the blob
//! is keyed to the driver and the device, so two projects on one machine want the
//! same file, and one project copied to another machine wants neither. Vulkan
//! validates the header itself and silently starts empty when the driver, the
//! device or the blob version has moved, so a stale file costs a cold compile and
//! never a wrong one.
//!
//! **What it is worth, measured.** On an RX 9070 XT (RADV) the offscreen capture
//! suite takes 2.60 s with no cache and 2.16 s with one — about 0.45 s of shader
//! compilation — *when Mesa's own `~/.cache/mesa_shader_cache` is cold. With that
//! warm, both are 2.08 s and this file buys nothing.* Which is the honest shape
//! of the feature: on a Mesa system it is insurance against a cold driver cache
//! (a fresh machine, a driver update, a CI runner, an evicted entry), and it is
//! on the platforms without a comparable system-wide cache that it earns its
//! place every launch. That asymmetry is why the measurement is written down here
//! rather than left as a number someone re-derives and finds to be zero.
//!
//! `ORRIN_PIPELINE_CACHE` overrides the path; setting it empty turns the cache
//! off, which is what a cold-start measurement wants.

use std::path::PathBuf;
use std::sync::Arc;

use vulkano::device::Device;
use vulkano::pipeline::cache::{PipelineCache, PipelineCacheCreateInfo};

const FILE: &str = "pipelines.bin";

/// Owns the cache and writes it back when the renderer goes away.
pub struct ShaderCache {
    /// `None` when the driver refused to create one. Every call site takes an
    /// `Option` anyway — that is vulkano's signature for the cache argument —
    /// so a device without one needs no branch anywhere else.
    cache: Option<Arc<PipelineCache>>,
    /// Where to write on drop, and `None` when there is nowhere sensible to
    /// write or the user turned it off.
    path: Option<PathBuf>,
    /// How many bytes came *out* of the file. Compared against what goes back in
    /// so an unchanged cache is not rewritten every run — the common case, once
    /// a project's shaders have settled.
    loaded_len: usize,
}

impl ShaderCache {
    pub fn load(device: &Arc<Device>) -> Self {
        let path = path();
        let initial_data = path
            .as_ref()
            .and_then(|path| std::fs::read(path).ok())
            .unwrap_or_default();
        let loaded_len = initial_data.len();

        // SAFETY: vulkano requires the blob to have come from `get_data`, and it
        // did — this reads back only what `save` wrote. The file is nonetheless
        // outside our control between runs, so the real guard is the one the
        // driver applies: a `VkPipelineCacheHeaderVersionOne` carries the vendor,
        // device and driver UUIDs, and an implementation that does not recognise
        // its own header ignores the payload and starts empty. That is what makes
        // a truncated or foreign file a slow launch rather than undefined
        // behaviour.
        let cache = unsafe {
            PipelineCache::new(
                device.clone(),
                PipelineCacheCreateInfo {
                    initial_data,
                    ..Default::default()
                },
            )
        };

        let cache = match cache {
            Ok(cache) => Some(cache),
            Err(e) => {
                tracing::warn!("no pipeline cache ({e}); shaders compile from scratch each run");
                None
            }
        };

        Self {
            cache,
            path,
            loaded_len,
        }
    }

    /// What the 22 pipeline constructors pass as their cache argument.
    pub fn handle(&self) -> Option<Arc<PipelineCache>> {
        self.cache.clone()
    }

    /// Write the driver's accumulated blob back out.
    ///
    /// Every failure here is reported and then dropped: a cache that cannot be
    /// written is a slower next launch, and refusing to close the editor over one
    /// would be the worse failure. The write goes to a sibling temporary first,
    /// so a crash or a second instance mid-write leaves the previous cache intact
    /// rather than a truncated file the next run has to discover is unusable.
    fn save(&self) {
        let (Some(cache), Some(path)) = (&self.cache, &self.path) else {
            return;
        };
        let Ok(data) = cache.get_data() else {
            return;
        };
        // Nothing new compiled this run, so there is nothing to write. Byte
        // length is a weak test for "unchanged", but it is free and the cost of
        // being wrong is one redundant write of a file we already hold.
        if data.len() == self.loaded_len {
            return;
        }
        if let Some(dir) = path.parent()
            && let Err(e) = std::fs::create_dir_all(dir)
        {
            tracing::warn!("could not create {}: {e}", dir.display());
            return;
        }

        let temporary = path.with_extension("tmp");
        if let Err(e) = std::fs::write(&temporary, &data) {
            tracing::warn!("could not write {}: {e}", temporary.display());
            return;
        }
        if let Err(e) = std::fs::rename(&temporary, path) {
            tracing::warn!("could not replace {}: {e}", path.display());
            let _ = std::fs::remove_file(&temporary);
        }
    }
}

impl Drop for ShaderCache {
    fn drop(&mut self) {
        self.save();
    }
}

/// Where the blob lives, or `None` to keep it in memory only.
///
/// `ORRIN_PIPELINE_CACHE` wins if set: a path to use, or empty to disable. Set
/// empty is what a cold-start measurement wants, since a warm cache is precisely
/// what it is trying not to measure.
fn path() -> Option<PathBuf> {
    if let Some(override_path) = std::env::var_os("ORRIN_PIPELINE_CACHE") {
        if override_path.is_empty() {
            return None;
        }
        return Some(PathBuf::from(override_path));
    }
    Some(cache_dir()?.join("orrin").join(FILE))
}

/// The platform's per-user cache directory.
///
/// Resolved by hand rather than through a crate because this is the only
/// directory the engine needs and the rules are three lines each. A cache is
/// exactly what these directories are for: losing one costs a slow launch.
fn cache_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    }
    #[cfg(target_os = "macos")]
    {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join("Library").join("Caches"))
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The override is the whole interface the perf harness has to this, and
    /// both of its states have to work: a path to use, and empty for "don't".
    ///
    /// Serialised with the other environment test in this module by being the
    /// only one — `std::env::set_var` is process-wide, so a second test touching
    /// the same variable would race this one.
    #[test]
    fn the_override_names_a_path_or_turns_the_cache_off() {
        // SAFETY: single-threaded within this test, and no other test in this
        // binary reads or writes `ORRIN_PIPELINE_CACHE`.
        unsafe {
            std::env::set_var("ORRIN_PIPELINE_CACHE", "/tmp/orrin-test-cache.bin");
            assert_eq!(path(), Some(PathBuf::from("/tmp/orrin-test-cache.bin")));

            std::env::set_var("ORRIN_PIPELINE_CACHE", "");
            assert_eq!(path(), None);

            std::env::remove_var("ORRIN_PIPELINE_CACHE");
        }
    }
}
