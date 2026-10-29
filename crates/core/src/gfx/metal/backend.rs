//! What the renderer needs from Metal to draw a frame: the images the graph
//! owns, and a target to render into.
//!
//! The graph's transient images are allocated one texture each. Vulkan aliases
//! memory between transients that are never alive together; Metal could with
//! heaps, but a heap's resources are untracked, so until synchronisation is
//! lowered by hand each one is an ordinary tracked texture and the hazards are
//! Metal's own.

use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLPixelFormat};

use super::encoder::Timing;
use super::window::MtlWindow;
use super::{FRAME_SLOTS, MtlDriver};
use crate::gfx::format::{Format, SampleCount};
use crate::gfx::graph::{FrameGraph, ImageUsage};
use crate::gfx::images::{ImageEntry, ImageTable};
use crate::gfx::rhi::{TextureDesc, TextureDimension, TextureViewId};
use crate::scene::PresentSettings;

/// What makes an allocation that one: the cache returns the same textures for
/// a recompile that did not change them, so a structural toggle does not
/// reallocate every image in the frame.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    name: &'static str,
    format: Format,
    extent: [u32; 3],
    usage: u32,
    samples: SampleCount,
    array_layers: u32,
    mip_levels: u32,
    memoryless: bool,
}

/// What a window adds to the backend: the layer a frame ends in, the size the
/// window last asked for, and the frames still on the GPU.
pub(crate) struct Windowed {
    pub(crate) window: MtlWindow,
    pending_extent: [u32; 2],
    present: PresentSettings,
    /// The command buffer of each recent frame, indexed by the frame number the
    /// driver recorded it under, which is the number that picks its arenas, so
    /// the CPU can wait for the one whose arenas it is about to reuse.
    ring: [Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>; FRAME_SLOTS as usize],
}

impl Windowed {
    pub(crate) fn frame<'a>(
        &'a self,
        driver: &'a MtlDriver,
        fallback: TextureViewId,
    ) -> super::window::Frame<'a> {
        super::window::Frame::new(&self.window, driver, fallback)
    }
}

pub(crate) struct MtlBackend {
    pub(crate) extent: [u32; 2],
    pub(crate) format: Format,
    /// The image the frame ends in, which a capture reads back.
    pub(crate) target: TextureViewId,
    images: [u32; 2],
    cache: HashMap<Key, ImageEntry>,
    pub(crate) windowed: Option<Windowed>,
    /// Trails the displayed frame by the pacing depth, as the Vulkan reading does.
    pub(crate) gpu_ms: f32,
    /// Frames whose samples wait on the GPU to finish.
    timed: Vec<(Timing, Retained<ProtocolObject<dyn MTLCommandBuffer>>)>,
    /// Sample buffers whose frames have been read, ready for the next.
    spare_timing: Vec<Timing>,
    /// Set once the device has said it cannot sample timestamps.
    untimeable: bool,
}

/// Seconds in, milliseconds out; a buffer that never ran reports zero for both.
fn span_ms(start: f64, end: f64) -> f32 {
    ((end - start).max(0.0) * 1.0e3) as f32
}

fn create_target(driver: &MtlDriver, format: Format, extent: [u32; 2]) -> TextureViewId {
    driver.create_texture(&TextureDesc {
        format,
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        mip_levels: 1,
        array_layers: 1,
        samples: SampleCount::Sample1,
        usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::TRANSFER_SRC,
        memoryless: false,
    })
}

impl MtlBackend {
    pub(crate) fn new(
        driver: &MtlDriver,
        format: Format,
        extent: [u32; 2],
        graph: &FrameGraph,
    ) -> (Self, ImageTable) {
        let target = create_target(driver, format, extent);
        let mut backend = Self {
            extent,
            format,
            target,
            images: extent,
            cache: HashMap::new(),
            windowed: None,
            gpu_ms: 0.0,
            timed: Vec::new(),
            spare_timing: Vec::new(),
            untimeable: false,
        };
        let table = backend.rebuild(driver, graph);
        (backend, table)
    }

    /// A backend whose frames end in `view`'s window. `target` stays as the
    /// place a frame lands when the layer has no drawable to give.
    pub(crate) fn windowed(
        driver: &MtlDriver,
        view: std::ptr::NonNull<std::ffi::c_void>,
        extent: [u32; 2],
        present: PresentSettings,
        graph: &FrameGraph,
    ) -> (Self, ImageTable) {
        let format = Format::Bgra8Srgb;
        let (mut backend, table) = Self::new(driver, format, extent, graph);
        backend.windowed = Some(Windowed {
            window: MtlWindow::new(
                driver,
                view,
                MTLPixelFormat::BGRA8Unorm_sRGB,
                extent,
                &present,
            ),
            pending_extent: extent,
            present,
            ring: Default::default(),
        });
        (backend, table)
    }

    /// The window's size, as the next frame will adopt it.
    pub(crate) fn resize(&mut self, extent: [u32; 2]) {
        if let Some(windowed) = &mut self.windowed {
            windowed.pending_extent = extent;
        }
    }

    pub(crate) fn set_present(&mut self, present: PresentSettings) {
        if let Some(windowed) = &mut self.windowed
            && windowed.present != present
        {
            windowed.present = present;
            windowed.window.apply(&present);
        }
    }

    /// What the layer is set to, for the run banner; `None` offscreen.
    pub(crate) fn applied_present(&self) -> Option<String> {
        self.windowed.as_ref().map(|windowed| {
            format!(
                "{}, {} drawables",
                if windowed.window.display_sync() {
                    "display sync"
                } else {
                    "immediate"
                },
                windowed.window.drawables()
            )
        })
    }

    /// Whether the window has no area to draw into, so there is no frame to make.
    pub(crate) fn is_collapsed(&self) -> bool {
        self.windowed
            .as_ref()
            .is_some_and(|windowed| windowed.pending_extent.contains(&0))
    }

    /// Adopt the size the window asked for since the last frame: the layer's
    /// drawable, the fallback target, and (through `is_stale`) the graph's images.
    pub(crate) fn adopt_resize(&mut self, driver: &MtlDriver) {
        let Some(windowed) = &self.windowed else {
            return;
        };
        let extent = windowed.pending_extent;
        if extent == self.extent {
            return;
        }
        windowed.window.resize(extent);
        driver.destroy_texture(self.target);
        self.target = create_target(driver, self.format, extent);
        self.extent = extent;
    }

    /// Block until the frame whose arenas `frame` is about to reuse has
    /// finished. Called with the number `begin_frame` is about to be given.
    pub(crate) fn wait_for_room(&mut self, frame: u64) {
        if let Some(windowed) = &mut self.windowed
            && let Some(previous) = windowed.ring[(frame % FRAME_SLOTS) as usize].take()
        {
            previous.waitUntilCompleted();
            super::encoder::report_failure(&previous);
            self.gpu_ms = span_ms(previous.GPUStartTime(), previous.GPUEndTime());
        }
    }

    pub(crate) fn submitted(
        &mut self,
        frame: u64,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ) {
        if let Some(windowed) = &mut self.windowed {
            windowed.ring[(frame % FRAME_SLOTS) as usize] = Some(command_buffer);
        }
    }

    /// A sample buffer to time `frame` into, reusing one already read.
    pub(crate) fn timing(&mut self, driver: &MtlDriver, frame: u64) -> Option<Timing> {
        if let Some(mut timing) = self.spare_timing.pop() {
            timing.reset(frame);
            return Some(timing);
        }
        if self.untimeable {
            return None;
        }
        let timing = Timing::new(driver, frame);
        self.untimeable = timing.is_none();
        timing
    }

    pub(crate) fn timed(
        &mut self,
        timing: Timing,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    ) {
        self.timed.push((timing, command_buffer));
    }

    /// File the samples of every frame the GPU has finished, without waiting.
    pub(crate) fn drain_spans(&mut self, profiler: &mut crate::profile::Profiler) {
        use objc2_metal::MTLCommandBufferStatus;
        let current = profiler.frame_index();
        let mut index = 0;
        while index < self.timed.len() {
            let (timing, command_buffer) = &self.timed[index];
            let status = command_buffer.status();
            // A buffer that failed has no samples to read, but its sample
            // buffer is as reusable as a finished one's.
            if timing.frame >= current || status < MTLCommandBufferStatus::Completed {
                index += 1;
                continue;
            }
            if status == MTLCommandBufferStatus::Completed {
                for (name, start_ns, end_ns) in timing.resolve() {
                    profiler.push_gpu_span(
                        timing.frame,
                        crate::profile::Span {
                            name,
                            depth: 0,
                            start_ns,
                            end_ns,
                        },
                    );
                }
            }
            let (timing, _) = self.timed.swap_remove(index);
            self.spare_timing.push(timing);
        }
    }

    pub(crate) fn is_stale(&self) -> bool {
        self.images != self.extent
    }

    /// Point a table at the images `graph` declares, allocating only the ones
    /// the cache does not already hold and freeing the ones `graph` no longer
    /// declares. A resize changes every key, so it frees everything.
    pub(crate) fn rebuild(&mut self, driver: &MtlDriver, graph: &FrameGraph) -> ImageTable {
        self.images = self.extent;
        let mut previous = std::mem::take(&mut self.cache);
        let mut entries: Vec<Option<ImageEntry>> = vec![None; graph.resource_count()];
        for (id, image) in graph.transient_images() {
            let extent = image.desc.extent.resolve(self.extent);
            let depth = image.desc.depth.unwrap_or(1);
            let largest = extent[0].max(extent[1]).max(depth).max(1);
            let mip_levels = image.desc.mip_levels.min(largest.ilog2() + 1).max(1);
            let layers = image.desc.array_layers.unwrap_or(1);
            let key = Key {
                name: graph.resource_name(id),
                format: image.desc.format,
                extent: [extent[0], extent[1], depth],
                usage: image.usage.bits(),
                samples: image.desc.samples,
                array_layers: layers,
                mip_levels,
                memoryless: image.memoryless,
            };
            if let Some(entry) = previous.remove(&key) {
                entries[id.index()] = Some(entry.clone());
                self.cache.insert(key, entry);
                continue;
            }
            let whole = driver.create_texture(&TextureDesc {
                format: image.desc.format,
                dimension: match (image.desc.depth, image.desc.array_layers) {
                    (Some(_), _) => TextureDimension::D3,
                    (None, Some(_)) => TextureDimension::D2Array,
                    (None, None) => TextureDimension::D2,
                },
                extent: [extent[0], extent[1], depth],
                mip_levels,
                array_layers: layers,
                samples: image.desc.samples,
                usage: image.usage,
                memoryless: image.memoryless,
            });
            let levels = if mip_levels > 1 {
                (0..mip_levels)
                    .map(|level| driver.mip_view(whole, level))
                    .collect()
            } else {
                Vec::new()
            };
            let layer_views = (0..image.desc.array_layers.unwrap_or(0))
                .map(|layer| driver.face_view(whole, layer, 0))
                .collect();
            let entry = ImageEntry {
                whole,
                levels,
                layers: layer_views,
                extent: [extent[0], extent[1]],
                mip_levels,
            };
            self.cache.insert(key, entry.clone());
            entries[id.index()] = Some(entry);
        }
        for entry in previous.into_values() {
            for id in std::iter::once(entry.whole)
                .chain(entry.levels)
                .chain(entry.layers)
            {
                driver.destroy_texture(id);
            }
        }
        ImageTable::new(vec![entries])
    }
}

#[cfg(test)]
mod span_tests {
    use super::span_ms;

    #[test]
    fn a_span_is_in_milliseconds_and_never_negative() {
        assert!((span_ms(1.0, 1.0025) - 2.5).abs() < 1e-3);
        assert_eq!(span_ms(2.0, 0.0), 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::graph::{Access, GraphBuilder, ImageDesc, ImageLayout, PassKind, compile};
    use crate::profile::Profiler;

    /// A target fed by `kept`, and with `extra` a second image feeding it too.
    fn graph(extra: bool) -> FrameGraph {
        let image = || ImageDesc::new(Format::Rgba8Unorm);
        let mut builder = GraphBuilder::new();
        let target = builder.import_image(
            "target",
            image(),
            ImageLayout::Undefined,
            ImageLayout::PresentSrc,
        );
        let kept = builder.create_image("kept", image());
        builder
            .pass("draw", PassKind::Inline)
            .access(kept, Access::ColorAttachment)
            .build();
        let extra = extra.then(|| {
            let extra = builder.create_image("extra", image());
            builder
                .pass("draw_extra", PassKind::Inline)
                .access(extra, Access::ColorAttachment)
                .build();
            extra
        });
        let mut resolve = builder
            .pass("resolve", PassKind::Inline)
            .access(kept, Access::Sampled)
            .access(target, Access::ColorAttachment);
        if let Some(extra) = extra {
            resolve = resolve.access(extra, Access::Sampled);
        }
        resolve.build();
        compile(builder).unwrap()
    }

    fn live_textures(driver: &MtlDriver) -> usize {
        driver.textures.lock().unwrap().live()
    }

    #[test]
    #[ignore = "needs a Metal device"]
    fn an_image_the_graph_stopped_declaring_is_freed() {
        let driver = MtlDriver::new().expect("this machine runs the Metal backend");
        let (mut backend, _) = MtlBackend::new(&driver, Format::Rgba8Unorm, [16, 16], &graph(true));
        let with_extra = live_textures(&driver);

        backend.rebuild(&driver, &graph(false));
        driver.begin_frame(FRAME_SLOTS);
        assert_eq!(live_textures(&driver), with_extra - 1);

        backend.rebuild(&driver, &graph(false));
        driver.begin_frame(2 * FRAME_SLOTS);
        assert_eq!(
            live_textures(&driver),
            with_extra - 1,
            "a rebuild of the same graph frees nothing it still uses"
        );
    }

    #[test]
    #[ignore = "needs a Metal device"]
    fn a_drained_frame_hands_its_sample_buffer_to_the_next() {
        let driver = MtlDriver::new().expect("this machine runs the Metal backend");
        let (mut backend, _) =
            MtlBackend::new(&driver, Format::Rgba8Unorm, [16, 16], &graph(false));
        let mut profiler = Profiler::new(4);

        let first = backend
            .timing(&driver, 0)
            .expect("this device samples timestamps");
        let buffer = Retained::as_ptr(&first.buffer);
        let mut state = super::super::State::new(&driver);
        state.time(first);
        let timing = state.take_timing().unwrap();
        let (_, command_buffer) = state.finish_blocking();
        backend.timed(timing, command_buffer);
        profiler.end_frame();
        backend.drain_spans(&mut profiler);

        let second = backend.timing(&driver, 1).unwrap();
        assert!(std::ptr::eq(Retained::as_ptr(&second.buffer), buffer));
        assert_eq!(second.frame, 1);
    }
}
