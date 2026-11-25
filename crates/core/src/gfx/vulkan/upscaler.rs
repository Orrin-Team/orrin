//! Which upscaler resolves the frame: the engine's own, or a vendor's in its
//! place, and why.
//!
//! [`TaaSettings::upscaler`] is a request. This decides each frame, before the
//! graph is compiled, whether the vendor asked for can run here — the feature
//! built in, Vulkan on Windows, its runtime found, a context made for this
//! frame's extents — and the frame declares the vendor's node only if it can.
//! Anything short of that is the engine's resolve with the reason recorded in
//! an [`UpscalerStatus`], never a failure: the engine's resolve runs
//! everywhere, which is what lets a scene ask for a vendor on every machine.
//!
//! **The context's life.** Made when the vendor first runs, for the output's
//! extent and a render extent as large, so that every mode fits it and a
//! change of mode needs none. A frame dispatches at its own two extents, and
//! one whose extents differ from the last dispatch's drops the history, as
//! a resize or a change of mode drops the engine's. Making one takes a third
//! of a second, so a resize does not make one per frame: an output that
//! still fits the context is resolved in it, and one that has outgrown it,
//! or left it more than twice its area, is given a new one only once it has
//! been asked for unchanged for [`SETTLE_SECONDS`] of frame time, the
//! engine's resolve standing in meanwhile where the old one cannot serve. A
//! drag of the editor's viewport therefore costs one creation when it stops,
//! or none if it only shrank. The context is dropped when the vendor stops
//! running, so a frame resolved by the engine holds none of the vendor's
//! memory, as one the vendor resolves holds none of the engine's history.
//! Dropping it waits for the graphics queue, which is the only one a vendor's
//! dispatch is ever submitted to (see `frame::declare`).
//!
//! **Its threading.** A context takes one call at a time, and the resolve may
//! record on a pool worker. So the context sits behind a mutex: made and
//! dropped here on the thread that drives the frame, before anything records,
//! and entered by the one dispatch a frame has.
//!
//! **Exposure.** The vendor is handed the engine's: the multiplier the tonemap
//! applies, which is what AMD's `exposure` input asks for ("the value the
//! application uses during any subsequent tonemapping"), with a pre-exposure
//! of one, since the frame's colour is the light as lit and not pre-scaled.
//! The metering runs after the resolve, and the graph orders every reader of
//! a resource after its writers, so the resolve cannot read this frame's
//! value: `upscaler_exposure` copies it after the metering into one of two
//! 1x1 images, and the next frame's resolve reads the other, which holds the
//! exposure the last frame was shown at. The pair exists while a vendor
//! resolves and only then. It is seeded with the manual exposure, which the
//! first frame reads; that frame is also its context's first, with no history
//! for the exposure to weigh.
//!
//! **The reactive mask.** The transparency and refraction composites write no
//! depth or motion, so a history reprojects what they drew as though it were
//! the surface behind. Where the frame has either, `reactive_mask` draws how
//! much of each pixel they blended over the opaque frame, which is AMD's
//! `reactive` input. FSR 4.1.2 leaves its output unchanged by it, byte for
//! byte; it is drawn for FSR 3.1, which the same runtime falls back to where
//! FSR 4 does not run, and whose guide asks for it.

use std::sync::{Arc, LazyLock};

use vulkano::device::Queue;
use vulkano::image::Image;

use crate::gfx::format::Format;
use crate::gfx::rhi::{BufferId, Device, Encoder, TextureViewId};
use crate::scene::{TaaSettings, Upscaler, UpscalerStatus};

/// How long an output that has outgrown the vendor's context must be asked
/// for, unchanged, before one is made for it: long enough that a drag pausing
/// under the hand does not make one, short beside the third of a second the
/// making takes.
#[cfg_attr(not(all(feature = "fsr", windows)), allow(dead_code))]
pub(super) const SETTLE_SECONDS: f32 = 0.25;

/// The exposure a vendor reads: one float, as AMD's `exposure` input is.
pub(super) const EXPOSURE_FORMAT: Format = Format::R32Float;

/// The reactive mask: AMD's `reactive` input is an R8 unorm image.
pub(super) const REACTIVE_FORMAT: Format = Format::R8Unorm;

/// What one frame's upscale is handed besides its images. Read only where a
/// vendor's upscaler is built in.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(all(feature = "fsr", windows)), allow(dead_code))]
pub(super) struct FrameInputs {
    /// Where inside its pixel the frame sampled, in render pixels.
    pub jitter: [f32; 2],
    pub render: [u32; 2],
    pub output: [u32; 2],
    pub near: f32,
    pub far: f32,
    /// Vertical, in radians.
    pub fov_y: f32,
    /// Since the previous frame.
    pub frame_ms: f32,
}

/// The images the resolve node declares.
#[cfg_attr(not(all(feature = "fsr", windows)), allow(dead_code))]
pub(super) struct Images {
    pub color: Arc<Image>,
    pub depth: Arc<Image>,
    pub motion: Arc<Image>,
    pub exposure: Arc<Image>,
    pub reactive: Option<Arc<Image>>,
    pub output: Arc<Image>,
}

/// Why the engine's resolve stands in, for the reasons that never change.
/// Made once, so a frame that falls back allocates nothing to say so.
static DLSS_UNBOUND: LazyLock<Arc<str>> = LazyLock::new(|| Arc::from("DLSS is not bound yet"));
#[cfg(not(all(feature = "fsr", windows)))]
static NO_FSR: LazyLock<Arc<str>> = LazyLock::new(|| {
    Arc::from(if cfg!(windows) {
        "this build has no FSR: build with the `fsr` feature"
    } else {
        "FSR runs on Windows only"
    })
});
#[cfg(all(feature = "fsr", windows))]
static FSR_VULKAN_ONLY: LazyLock<Arc<str>> = LazyLock::new(|| Arc::from("FSR runs on Vulkan only"));

pub(super) struct VendorUpscaler {
    status: UpscalerStatus,
    /// Whether the vendor draws its debug view over what it resolves.
    debug_view: bool,
    #[cfg(all(feature = "fsr", windows))]
    fsr: amd::Fsr,
}

impl VendorUpscaler {
    pub(super) fn new() -> Self {
        Self {
            status: UpscalerStatus::default(),
            debug_view: false,
            #[cfg(all(feature = "fsr", windows))]
            fsr: amd::Fsr::default(),
        }
    }

    pub(super) fn status(&self) -> &UpscalerStatus {
        &self.status
    }

    pub(super) fn set_debug_view(&mut self, on: bool) {
        self.debug_view = on;
    }

    /// Decide what resolves the next frame, and make ready what it needs.
    /// `true` when a vendor's upscaler does, and the frame is to declare its
    /// node; `queue` is `None` off Vulkan. `manual_exposure` is what a fresh
    /// exposure pair is seeded with, and `dt` the frame's step in seconds, by
    /// which a resize settles.
    #[cfg_attr(not(all(feature = "fsr", windows)), allow(unused_variables))]
    pub(super) fn prepare(
        &mut self,
        device: &Device,
        queue: Option<&Arc<Queue>>,
        settings: &TaaSettings,
        output: [u32; 2],
        manual_exposure: f32,
        dt: f32,
    ) -> bool {
        let asked = settings.upscaler;
        let mut status = UpscalerStatus {
            asked,
            ..UpscalerStatus::default()
        };
        let vendor = if !settings.resolves() {
            status.running = None;
            false
        } else {
            match asked {
                Upscaler::BuiltIn => false,
                Upscaler::Dlss => {
                    status.reason = Some(DLSS_UNBOUND.clone());
                    false
                }
                Upscaler::Fsr => self.prepare_fsr(queue, settings, output, dt, &mut status),
            }
        };
        if settings.resolves() && !vendor {
            status.running = Some(Upscaler::BuiltIn);
        }
        // A failure is remembered while FSR is still asked for, so it is not
        // retried every frame, and forgotten once it is not.
        #[cfg(all(feature = "fsr", windows))]
        if vendor {
            self.fsr.advance_exposure(device, manual_exposure);
        } else {
            self.fsr
                .release(device, !settings.resolves() || asked != Upscaler::Fsr);
        }
        if status != self.status {
            match (&status.running, &status.reason) {
                (Some(running), None) => tracing::info!(
                    "temporal resolve: {}{}",
                    running.label(),
                    status
                        .provider
                        .as_ref()
                        .map_or(String::new(), |provider| format!(", {provider}")),
                ),
                (Some(running), Some(reason)) => tracing::warn!(
                    "temporal resolve: {} asked for, {} running: {reason}",
                    asked.label(),
                    running.label(),
                ),
                (None, _) => {}
            }
            self.status = status;
        }
        vendor
    }

    #[cfg(not(all(feature = "fsr", windows)))]
    fn prepare_fsr(
        &mut self,
        _queue: Option<&Arc<Queue>>,
        _settings: &TaaSettings,
        _output: [u32; 2],
        _dt: f32,
        status: &mut UpscalerStatus,
    ) -> bool {
        status.reason = Some(NO_FSR.clone());
        false
    }

    #[cfg(all(feature = "fsr", windows))]
    fn prepare_fsr(
        &mut self,
        queue: Option<&Arc<Queue>>,
        settings: &TaaSettings,
        output: [u32; 2],
        dt: f32,
        status: &mut UpscalerStatus,
    ) -> bool {
        let Some(queue) = queue else {
            status.reason = Some(FSR_VULKAN_ONLY.clone());
            return false;
        };
        match self
            .fsr
            .prepare(queue, settings.upscaling, output, self.debug_view, dt)
        {
            Ok(live) => {
                status.running = Some(Upscaler::Fsr);
                status.provider = Some(live.provider.clone());
                status.memory_bytes = live.memory;
                true
            }
            Err(reason) => {
                status.reason = Some(reason);
                false
            }
        }
    }

    /// The exposure images of a frame a vendor resolves: the one
    /// `upscaler_exposure` writes, and the one the resolve reads.
    pub(super) fn exposure_views(&self) -> (TextureViewId, TextureViewId) {
        #[cfg(all(feature = "fsr", windows))]
        return self.fsr.exposure_views();
        #[cfg(not(all(feature = "fsr", windows)))]
        unreachable!("a vendor resolve was scheduled in a build with no vendor upscaler");
    }

    /// Record `upscaler_exposure`: what the tonemap applies, from the metered
    /// buffer or the manual value as `tonemap.frag` chooses.
    #[cfg_attr(not(all(feature = "fsr", windows)), allow(unused_variables))]
    pub(super) fn record_exposure(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        metered: BufferId,
        manual: f32,
        auto: bool,
    ) {
        #[cfg(all(feature = "fsr", windows))]
        self.fsr
            .record_exposure(device, encoder, metered, manual, auto);
        #[cfg(not(all(feature = "fsr", windows)))]
        unreachable!("a vendor resolve was scheduled in a build with no vendor upscaler");
    }

    /// Record `reactive_mask` over `extent`, from the transparency's
    /// revealage and the refraction's coverage, whichever the frame has.
    #[cfg_attr(not(all(feature = "fsr", windows)), allow(unused_variables))]
    pub(super) fn record_reactive(
        &self,
        device: &Device,
        encoder: &mut Encoder,
        reveal: Option<TextureViewId>,
        refraction: Option<TextureViewId>,
        extent: [u32; 2],
    ) {
        #[cfg(all(feature = "fsr", windows))]
        self.fsr
            .record_reactive(device, encoder, reveal, refraction, extent);
        #[cfg(not(all(feature = "fsr", windows)))]
        unreachable!("a vendor resolve was scheduled in a build with no vendor upscaler");
    }

    /// Record the vendor's resolve, the body of the node `prepare` said the
    /// frame should declare. `reset` drops the history it accumulated.
    #[cfg_attr(not(all(feature = "fsr", windows)), allow(unused_variables))]
    pub(super) fn record(
        &self,
        encoder: &mut Encoder,
        images: &Images,
        inputs: &FrameInputs,
        reset: bool,
    ) {
        #[cfg(all(feature = "fsr", windows))]
        {
            let Encoder::Vulkan(encoder) = encoder;
            self.fsr
                .record(encoder.recorder(), images, inputs, reset, self.debug_view);
        }
        #[cfg(not(all(feature = "fsr", windows)))]
        unreachable!("a vendor resolve was scheduled in a build with no vendor upscaler");
    }
}

#[cfg(all(feature = "fsr", windows))]
mod amd {
    use std::path::PathBuf;
    use std::sync::{Arc, LazyLock, Mutex};
    use std::time::Instant;

    use vulkano::device::Queue;

    use super::super::fsr::{self, ContextDesc, Runtime, UpscaleContext};
    use super::super::record::Recorder;
    use super::{EXPOSURE_FORMAT, FrameInputs, Images, REACTIVE_FORMAT, SETTLE_SECONDS};
    use crate::gfx::format::SampleCount;
    use crate::gfx::graph::ImageUsage;
    use crate::gfx::rhi::{
        Binding, BindingResource, BufferId, ComputePipelineDesc, Device, Encoder,
        GraphicsPipelineDesc, PipelineId, SamplerDesc, SamplerId, TextureDesc, TextureDimension,
        TextureViewId, Viewport,
    };
    use crate::gfx::shaders::ShaderId;
    use crate::scene::Upscaling;

    /// Why the engine's resolve stands in while an outgrown context waits.
    static SETTLING: LazyLock<Arc<str>> =
        LazyLock::new(|| Arc::from("FSR waits for the output's extent to settle"));

    /// The context, and what was learnt of it when it was made.
    pub(super) struct Live {
        context: UpscaleContext,
        desc: ContextDesc,
        /// The render and output extents of the last dispatch.
        dispatched: Option<([u32; 2], [u32; 2])>,
        pub provider: Arc<str>,
        pub memory: Option<u64>,
    }

    #[derive(Default)]
    pub(super) struct Fsr {
        /// Loaded on the first frame that asks, and kept: a runtime that is
        /// not there now is not there next frame either.
        runtime: Option<Result<Arc<Runtime>, Arc<str>>>,
        live: Mutex<Option<Live>>,
        /// What creation last failed for, so a frame asking for the same is
        /// not made to wait on it again; a change of extents tries afresh.
        failed: Option<(ContextDesc, Arc<str>)>,
        /// What a dispatch last failed with. Its frame resolved nothing, so
        /// the next falls back.
        dispatch_failed: Mutex<Option<Arc<str>>>,
        /// The exposure images, while FSR resolves: `frame & 1` is written
        /// this frame and the other read.
        exposure: Option<([TextureViewId; 2], u64)>,
        /// An output the context does not suit, and for how many seconds of
        /// frames it has been asked for.
        settling: Option<([u32; 2], f32)>,
        /// Made with the first pair and kept, as is the reactive mask's.
        exposure_pipeline: Option<PipelineId>,
        reactive_pipeline: Option<(PipelineId, SamplerId)>,
    }

    /// Whether `output` has been asked for, unchanged, for long enough to
    /// make a context for it; counts this frame's `dt` towards it.
    fn settled(settling: &mut Option<([u32; 2], f32)>, output: [u32; 2], dt: f32) -> bool {
        match settling {
            Some((asked, waited)) if *asked == output => {
                *waited += dt;
                *waited >= SETTLE_SECONDS
            }
            _ => {
                *settling = Some((output, 0.0));
                false
            }
        }
    }

    /// `FfxApiUpscaleQualityMode`, `ffx_upscale.h`, in the engine's terms.
    fn quality_mode(upscaling: Upscaling) -> u32 {
        match upscaling {
            Upscaling::Native => 0,
            Upscaling::Quality => 1,
            Upscaling::Balanced => 2,
            Upscaling::Performance => 3,
            Upscaling::UltraPerformance => 4,
        }
    }

    /// Where the runtime is looked for: `ORRIN_FSR_DIR`, or the directory the
    /// executable is in.
    fn runtime_dir() -> Result<PathBuf, String> {
        if let Some(dir) = std::env::var_os("ORRIN_FSR_DIR") {
            return Ok(PathBuf::from(dir));
        }
        let exe = std::env::current_exe().map_err(|e| format!("no executable path: {e}"))?;
        exe.parent()
            .map(PathBuf::from)
            .ok_or_else(|| format!("{} has no directory", exe.display()))
    }

    impl Fsr {
        /// Drop the context and the exposure pair, and with `forget` what
        /// failed.
        pub(super) fn release(&mut self, device: &Device, forget: bool) {
            let live = self.live.get_mut().expect("the FSR context was poisoned");
            if live.take().is_some() {
                tracing::info!("FSR: context released");
            }
            for view in self
                .exposure
                .take()
                .into_iter()
                .flat_map(|(views, _)| views)
            {
                device.destroy_texture(view);
            }
            if forget {
                self.failed = None;
                self.settling = None;
                *self
                    .dispatch_failed
                    .get_mut()
                    .expect("the FSR context was poisoned") = None;
            }
        }

        /// Turn the exposure pair over for a frame FSR resolves, making it
        /// if this is the first, seeded with `seed`.
        pub(super) fn advance_exposure(&mut self, device: &Device, seed: f32) {
            if let Some((_, frame)) = &mut self.exposure {
                *frame = frame.wrapping_add(1);
                return;
            }
            let Device::Vulkan(driver) = device;
            let views = std::array::from_fn(|_| {
                let view = device.create_texture(&TextureDesc {
                    format: EXPOSURE_FORMAT,
                    dimension: TextureDimension::D2,
                    extent: [1, 1, 1],
                    mip_levels: 1,
                    array_layers: 1,
                    samples: SampleCount::Sample1,
                    usage: ImageUsage::STORAGE | ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
                    memoryless: false,
                });
                driver.fill_color(view, [seed; 4]);
                view
            });
            self.exposure = Some((views, 0));
            self.exposure_pipeline.get_or_insert_with(|| {
                device.create_compute_pipeline(&ComputePipelineDesc {
                    shader: ShaderId::UpscalerExposureComp.into(),
                })
            });
            self.reactive_pipeline.get_or_insert_with(|| {
                let pipeline = device.create_graphics_pipeline(&GraphicsPipelineDesc::fullscreen(
                    ShaderId::FullscreenVert,
                    ShaderId::ReactiveMaskFrag,
                    &[REACTIVE_FORMAT],
                ));
                let nearest = device.create_sampler(SamplerDesc {
                    linear: false,
                    mipmap_linear: false,
                    repeat: false,
                    compare: false,
                    anisotropy: None,
                    max_lod: Some(0.0),
                    lod_bias: 0.0,
                });
                (pipeline, nearest)
            });
        }

        pub(super) fn record_reactive(
            &self,
            device: &Device,
            encoder: &mut Encoder,
            reveal: Option<TextureViewId>,
            refraction: Option<TextureViewId>,
            extent: [u32; 2],
        ) {
            let (pipeline, sampler) = self
                .reactive_pipeline
                .expect("the reactive pipeline is made with the exposure pair");
            let either = reveal
                .or(refraction)
                .expect("the graph declares the mask only over a blended queue");
            let sampled = |binding, view| Binding {
                binding,
                resource: BindingResource::SampledTexture { view, sampler },
            };
            let set = device.create_transient_bind_group(
                pipeline,
                0,
                &[
                    sampled(0, reveal.unwrap_or(either)),
                    sampled(1, refraction.unwrap_or(either)),
                ],
            );
            encoder.set_viewport(Viewport {
                origin: [0.0, 0.0],
                extent: [extent[0] as f32, extent[1] as f32],
                depth_range: [0.0, 1.0],
            });
            encoder.bind_pipeline(pipeline);
            encoder.bind_group(0, set);
            encoder.push_constants(&[reveal.is_some() as u32, refraction.is_some() as u32]);
            encoder.draw(3, 1, 0, 0);
        }

        /// This frame's exposure image and last frame's.
        pub(super) fn exposure_views(&self) -> (TextureViewId, TextureViewId) {
            let (views, frame) = self
                .exposure
                .expect("the graph scheduled FSR with no exposure made for it");
            (
                views[(frame & 1) as usize],
                views[((frame + 1) & 1) as usize],
            )
        }

        pub(super) fn record_exposure(
            &self,
            device: &Device,
            encoder: &mut Encoder,
            metered: BufferId,
            manual: f32,
            auto: bool,
        ) {
            let pipeline = self
                .exposure_pipeline
                .expect("the exposure pipeline is made with the pair");
            let (target, _) = self.exposure_views();
            let set = device.create_bind_group(
                pipeline,
                0,
                &[
                    Binding {
                        binding: 0,
                        resource: BindingResource::StorageBuffer(metered),
                    },
                    Binding {
                        binding: 1,
                        resource: BindingResource::StorageTexture(target),
                    },
                ],
            );
            encoder.bind_pipeline(pipeline);
            encoder.bind_group(0, set);
            encoder.push_constants(&[manual.to_bits(), auto as u32]);
            encoder.dispatch([1, 1, 1]);
        }

        pub(super) fn prepare(
            &mut self,
            queue: &Arc<Queue>,
            upscaling: Upscaling,
            output: [u32; 2],
            debug_view: bool,
            dt: f32,
        ) -> Result<&Live, Arc<str>> {
            if let Some(reason) = self
                .dispatch_failed
                .get_mut()
                .expect("the FSR context was poisoned")
                .clone()
            {
                *self.live.get_mut().expect("the FSR context was poisoned") = None;
                return Err(reason);
            }
            let runtime = self.runtime.get_or_insert_with(|| {
                let loaded = runtime_dir().and_then(|dir| {
                    let runtime = Runtime::load(&dir)?;
                    tracing::info!("FSR: loaded {}", dir.join(fsr::LOADER_DLL).display());
                    Ok(runtime)
                });
                loaded
                    .map(Arc::new)
                    .map_err(|reason| Arc::from(format!("the FSR runtime did not load: {reason}")))
            });
            let runtime = match runtime {
                Ok(runtime) => runtime.clone(),
                Err(reason) => return Err(reason.clone()),
            };

            let mut flags = fsr::ENABLE_HIGH_DYNAMIC_RANGE | fsr::ENABLE_DEPTH_INVERTED;
            if cfg!(debug_assertions) {
                flags |= fsr::ENABLE_DEBUG_CHECKING;
            }
            if debug_view {
                flags |= fsr::ENABLE_DEBUG_VISUALIZATION;
            }
            let render = upscaling.render_extent(output);
            let desc = ContextDesc {
                max_render: output,
                output,
                flags,
            };

            let live = self.live.get_mut().expect("the FSR context was poisoned");
            let area = |[width, height]: [u32; 2]| width as u64 * height as u64;
            match live.as_ref().map(|live| live.desc) {
                Some(made) if made.flags == flags => {
                    let fits = output[0] <= made.output[0] && output[1] <= made.output[1];
                    let roomy = area(made.output) > 2 * area(output);
                    if fits && !roomy {
                        self.settling = None;
                        return Ok(live.as_ref().unwrap());
                    }
                    if !settled(&mut self.settling, output, dt) {
                        if fits {
                            return Ok(live.as_ref().unwrap());
                        }
                        *live = None;
                        return Err(SETTLING.clone());
                    }
                }
                None if self.settling.is_some() => {
                    if !settled(&mut self.settling, output, dt) {
                        return Err(SETTLING.clone());
                    }
                }
                _ => {}
            }
            self.settling = None;
            if let Some((failed, reason)) = &self.failed
                && *failed == desc
            {
                return Err(reason.clone());
            }
            // Dropped before the new one is made, so the two never hold the
            // device's memory at once.
            *live = None;

            let started = Instant::now();
            let mut context = match UpscaleContext::new(&runtime, queue, desc) {
                Ok(context) => context,
                Err(reason) => {
                    let reason: Arc<str> = Arc::from(format!("FSR: {reason}"));
                    self.failed = Some((desc, reason.clone()));
                    return Err(reason);
                }
            };
            let provider: Arc<str> = Arc::from(
                context
                    .provider()
                    .unwrap_or_else(|| "an unnamed provider".to_string()),
            );
            let memory = context.gpu_memory();

            // The engine computes both itself, to AMD's arithmetic; the
            // runtime is the authority on them.
            let mode = quality_mode(upscaling);
            let theirs = context.render_extent(output, mode);
            debug_assert!(
                theirs.is_none_or(|theirs| theirs == render),
                "FSR renders {output:?} in mode {mode} at {theirs:?}, the engine at {render:?}",
            );
            let phases = context.jitter_phases(render[0], output[0]);
            if phases.is_some_and(|phases| phases != upscaling.jitter_phases()) {
                tracing::warn!(
                    "FSR: a jitter sequence {phases:?} long from {} to {} pixels, the \
                     engine's is {}",
                    render[0],
                    output[0],
                    upscaling.jitter_phases(),
                );
            }

            tracing::info!(
                "FSR: {provider}, up to {:?} to {:?}, {} MiB, created in {:.0} ms",
                desc.max_render,
                output,
                memory.map_or("?".to_string(), |bytes| format!(
                    "{:.1}",
                    bytes as f64 / 1048576.0
                )),
                started.elapsed().as_secs_f64() * 1e3,
            );
            self.failed = None;
            Ok(live.insert(Live {
                context,
                desc,
                dispatched: None,
                provider,
                memory,
            }))
        }

        pub(super) fn record(
            &self,
            recorder: &mut Recorder,
            images: &Images,
            inputs: &FrameInputs,
            reset: bool,
            debug_view: bool,
        ) {
            let mut live = self.live.lock().expect("the FSR context was poisoned");
            let live = live
                .as_mut()
                .expect("the graph scheduled FSR with no context made for it");
            let extents = (inputs.render, inputs.output);
            let resized = live
                .dispatched
                .replace(extents)
                .is_some_and(|last| last != extents);
            let dispatch = fsr::Dispatch {
                color: &images.color,
                depth: &images.depth,
                motion: &images.motion,
                exposure: &images.exposure,
                reactive: images.reactive.as_ref(),
                output: &images.output,
                // FSR's sign: the offset the projection moves the image by, in
                // render pixels, y down, which is the engine's. The opposite
                // sign on either axis measures blurrier against native.
                jitter: inputs.jitter,
                // The prepass writes `now - before` in UV; FSR takes the way
                // back to the last frame, in render pixels. Its `1e3` for a
                // surface behind last frame's camera scales to a vector far
                // off the frame, which FSR takes as no history there, as the
                // engine's resolve does.
                motion_scale: [-(inputs.render[0] as f32), -(inputs.render[1] as f32)],
                render: inputs.render,
                upscale: inputs.output,
                frame_ms: inputs.frame_ms,
                reset: reset || resized,
                near: inputs.near,
                far: inputs.far,
                fov_y: inputs.fov_y,
                debug_view,
            };
            if let Err(reason) = live.context.dispatch(recorder, &dispatch) {
                tracing::error!("FSR: {reason}");
                *self
                    .dispatch_failed
                    .lock()
                    .expect("the FSR context was poisoned") = Some(Arc::from(reason));
            }
        }
    }
}
