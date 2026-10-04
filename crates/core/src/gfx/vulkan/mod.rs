mod bloom;
mod contact_shadows;
mod context;
mod cull;
mod dof;
mod environment;
mod exposure;
mod fog;
mod forward;
pub mod frame;
mod hdr;
mod instances;
mod line;
mod mesh;
mod motion_blur;
mod occlusion;
mod oit;
mod parallel;
mod pipeline_cache;
mod prepass;
mod record;
mod refraction;
mod rendering;
mod resources;
mod shadow;
mod ssao;
mod ssr;
mod subsurface;
mod swapchain;
mod taa;
mod texture;
mod timestamps;
mod vendor;

use std::ops::Range;
use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::CopyImageToBufferInfo;
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::instance::Instance;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::swapchain::{
    AcquireNextImageInfo, PresentInfo, SemaphorePresentInfo, Surface, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::semaphore::Semaphore;
use vulkano::{Validated, VulkanError};

use crate::geom::{Aabb, Frustum};
use crate::gfx::DecalInstance;
use crate::gfx::punctual::ShadowAtlas;
use crate::gfx::shadows::CascadeSet;
use crate::scene::{
    BloomSettings, Camera, ContactShadowSettings, CpuMesh, DofSettings, EnvironmentSettings,
    FogSettings, HdrSettings, MaterialHandle, MeshHandle, MotionBlurSettings, PresentSettings,
    RefractionSettings, ShadowSettings, SsaoSettings, SsrSettings, SubsurfaceSettings, TaaSettings,
    TransparencySettings,
};

use self::contact_shadows::ContactShadowPass;
use self::context::VkContext;
use self::cull::{CullPass, CullView, Draws};
use self::environment::EnvironmentPass;
use self::forward::{ForwardPass, GpuMaterial, GpuMesh};
use self::instances::InstanceStore;
use self::mesh::MeshArena;
use self::occlusion::OcclusionPass;
use self::oit::OitPass;
use self::refraction::RefractionPass;
use crate::gfx::graph::{PassKind, Queue as GraphQueue, ResourceId};

use self::bloom::BloomPass;
use self::dof::DofPass;
use self::exposure::ExposurePass;
use self::fog::FogPass;
use self::frame::{Frame, FrameConfig, PassBody};
use self::hdr::HdrPass;
use self::line::LinePass;
use self::motion_blur::MotionBlurPass;
use self::prepass::GeometryPrepass;
use self::record::{InFlight, Recorder, RunAhead};
use vulkano::command_buffer::CommandBuffer;

/// How far the CPU may record ahead of the GPU.
///
/// Two, which is one frame of overlap. Nothing structural holds it there any
/// more: the depth is a [`RunAhead`] bound and no *resource* is duplicated to
/// raise it. Per-frame lifetimes come from `KeepAlive`, which raw recording made
/// this crate's job; the host-visible data a frame writes comes from
/// `SubbufferAllocator`s, which recycle an arena only once every subbuffer cut
/// from it has been dropped; and the graph's images stay a single set because
/// `frame_chain` keeps the GPU executing one frame at a time whatever the CPU
/// is doing.
///
/// Three was measured, against two, interleaved, and is indistinguishable from
/// it — so two is what stays. `frame_chain` is the reason there is nothing to
/// win: with the GPU executing one frame at a time, a single frame of overlap
/// already keeps the queue fed, and a third recording ahead cannot fill a gap
/// the schedule does not have. A deeper queue is worth revisiting only alongside
/// something that lets frames overlap on the device — an async compute queue, or
/// graph images that are no longer one set.
///
/// The ceiling, should the schedule ever stop serialising, is the timestamp pool
/// rather than memory: `timestamps.rs` rotates one query slot per frame in
/// flight and resets a slot before recording into it, so a frame beyond `SLOTS`
/// would reset a pool the GPU is still writing. `SLOTS` is defined from this
/// constant so the two cannot drift apart.
const FRAMES_IN_FLIGHT: usize = 2;
use self::resources::GraphImages;
use self::shadow::ShadowPass;
use self::ssao::SsaoPass;
use self::ssr::SsrPass;
use self::subsurface::SubsurfacePass;
use self::swapchain::SwapchainState;
use self::taa::TaaPass;
use self::timestamps::GpuTimestamps;

use super::{DrawList, MAX_TEXTURES, Material, RenderBackend, SceneLighting, TextureHandle};
use crate::profile::Profiler;
use crate::profile_scope;
use crate::scene::DebugLine;
use crate::threads;

/// MSAA sample count for the forward pass. One definition, because the forward
/// pipeline, its render pass attachments, and the graph's declaration of the
/// MSAA targets all have to agree or framebuffer creation fails at startup.
pub(crate) const MSAA_SAMPLES: vulkano::image::SampleCount = vulkano::image::SampleCount::Sample4;

/// What an offscreen render targets.
///
/// `R8G8B8A8_SRGB` rather than the `B8G8R8A8_SRGB` a macOS surface usually hands
/// back: both are universally supported as colour attachments, and this one
/// comes out of the readback in the byte order a PNG wants, so the capture path
/// has no channel swap in it to get backwards. The `_SRGB` half matters more —
/// the tonemap pass writes linear values and relies on the format to encode
/// them, exactly as it does on a window.
pub(crate) const OFFSCREEN_FORMAT: Format = Format::R8G8B8A8_SRGB;

/// A hook that draws over the final swapchain image between the tonemap pass and
/// present (the editor UI). Given the future to wait on and that image's view, it
/// returns the future to present. A plain closure, so this module stays free of
/// any UI/egui types.
pub type Overlay<'a> = &'a mut dyn FnMut(Box<dyn GpuFuture>, Arc<ImageView>) -> Box<dyn GpuFuture>;

/// Everything the cascade passes need for one frame.
///
/// Bundled because the three travel together and are meaningless apart: the
/// matrices decide what each pass draws with, the caster lists were culled
/// against those same matrices, and the settings supply the bias the maps are
/// rendered with. `None` means shadows are off, which is what makes the graph
/// drop the passes entirely rather than run them over an empty list.
#[derive(Clone, Copy)]
pub struct ShadowFrame<'a> {
    pub cascades: &'a CascadeSet,
    /// Indexed like `cascades.cascades`. Each is an ordering over the same item
    /// array the camera's list indexes, so an object both visible and casting
    /// exists once on the CPU however many cascades want it.
    pub casters: &'a [DrawList<'a>],
    /// The punctual lights that got atlas tiles, and one caster list per light —
    /// indexed like `atlas.casters`, not like the light arrays, because the
    /// budget means most lights have no list at all.
    pub atlas: &'a ShadowAtlas,
    pub punctual_casters: &'a [DrawList<'a>],
    pub settings: &'a ShadowSettings,
}

/// What a pass body needs of the renderer besides its own state: somewhere to
/// allocate this frame's descriptor sets, and the meshes a draw list indexes.
///
/// Narrower than the `&VulkanRenderer` the passes used to take, because a pass
/// may record on a worker and the renderer as a whole cannot cross a thread —
/// it owns the frames in flight and the overlay's futures, and neither of those
/// is `Send`.
pub(super) struct PassCtx<'a> {
    pub(crate) ctx: &'a VkContext,
    pub(crate) meshes: &'a [GpuMesh],
    /// The shared geometry a pass binds once, before its first draw.
    pub(crate) arena: &'a MeshArena,
    pub(crate) materials: &'a [GpuMaterial],
}

pub struct VulkanRenderer {
    pub(crate) ctx: VkContext,
    swapchain: SwapchainState,
    forward: ForwardPass,
    /// The per-object matrices, which live on the GPU between frames rather than
    /// being repacked into a fresh buffer every one. Owned here rather than by
    /// the forward pass because every geometry pass in the frame reads the same
    /// rows, and no one of them is the owner. See `instances`.
    instances: InstanceStore,
    /// The two dispatches that decide what each opaque view draws, and the
    /// buffers they write. Built whether or not `FrameConfig::gpu_culling` is
    /// set — the pipelines cost a compile at startup and nothing per frame, and
    /// the flag is meant to be switched without a rebuild.
    cull: CullPass,
    /// Owns the ping-ponged depth pyramid the graph imports, and records the
    /// dispatch that fills it. Built whether or not the frame tests, for the
    /// reason `cull` is built whether or not it culls.
    occlusion: OcclusionPass,
    /// Read once. The graph is compiled against it, so a mid-run change would
    /// leave the passes and the plan disagreeing about who writes
    /// `instance_index`.
    gpu_culling: bool,
    /// Read once, for the reason `gpu_culling` is: the graph is compiled
    /// against it, and the pass that samples last frame's pyramid has to be the
    /// pass the plan says samples it.
    occlusion_culling: bool,
    hdr: HdrPass,
    /// Owns the histogram and exposure buffers the graph imports, and records
    /// the two dispatches that fill them.
    exposure: ExposurePass,
    /// Records the bloom chain's dispatches. The levels themselves are
    /// graph-owned transients, so this holds only pipelines and settings.
    bloom: BloomPass,
    /// The one geometry pass in front of shading, shared by SSAO and TAA.
    prepass: GeometryPrepass,
    ssao: SsaoPass,
    /// The short march toward the sun, and the visibility mask the forward pass
    /// multiplies its sun term by. Holds only a pipeline and the dither's frame
    /// counter; the mask is graph-owned.
    contact_shadows: ContactShadowPass,
    /// The air in front of the camera as a froxel grid. Owns the ping-ponged
    /// scattering history the graph imports, and the block describing the medium
    /// — which every shading pass binds whether or not the two dispatches ran,
    /// because the analytic fog past the volume is the same medium.
    fog: FogPass,
    /// The depth pyramid, the reflection rays, and the composite that swaps the
    /// environment's reflection for them. Holds only pipelines and the resolved
    /// settings; every image it works over is graph-owned.
    ssr: SsrPass,
    subsurface: SubsurfacePass,
    /// Weighted-blended transparency: the accumulation pass and the composite
    /// that puts what it gathered over the lit frame. Draws through the forward
    /// pass's own pipeline layout, so it is constructed after it.
    oit: OitPass,
    /// Screen-space refraction: the scene pyramid, the sorted draw, and the
    /// composite. Built with a layout of its own — the same five descriptor sets
    /// the forward pass binds, plus a sixth for that pyramid.
    refraction: RefractionPass,
    /// Owns the ping-ponged history the graph imports, and decides the frame's
    /// subpixel jitter — which is why it is consulted before any pass records.
    taa: TaaPass,
    /// Defocus. Like bloom, it holds only pipelines and the resolved lens: the
    /// three images it works over are graph-owned transients.
    dof: DofPass,
    /// The shutter's reconstruction, and the velocity pyramid it reads.
    motion_blur: MotionBlurPass,
    shadow: ShadowPass,
    /// Debug-line overlay, recorded into the forward pass. Editor-only in
    /// practice: fed lines only through `render_with_overlay`.
    line: LinePass,
    /// The environment cubemap and the skybox that draws it. Also recorded into
    /// the forward pass, for the reason its module documents.
    environment: EnvironmentPass,
    pub(crate) meshes: Vec<GpuMesh>,
    /// The geometry every [`GpuMesh`] is a span into. One allocation per stream
    /// rather than three per mesh, which is what lets a pass bind geometry once
    /// and draw every batch from one command buffer. See `mesh`.
    arena: MeshArena,
    pub(crate) materials: Vec<GpuMaterial>,
    /// Texture views indexed by `TextureHandle`. Index 0 is a 1x1 white texture
    /// and index 1 a flat normal map; materials without a given map point here.
    pub(crate) textures: Vec<Arc<ImageView>>,
    /// The material table, shared by both geometry passes. `None` = dirty;
    /// rebuilt lazily in `render` after a `load_material`.
    material_buffer: Option<vulkano::buffer::Subbuffer<[GpuMaterial]>>,
    /// Cached descriptor sets over that table and the texture array, one pair
    /// per pipeline layout. Two pairs and not one because set compatibility is
    /// a property of the layout each pipeline declares, not of the buffer
    /// written into it — the same reason the object sets are kept apart.
    material_set: Option<Arc<DescriptorSet>>,
    texture_set: Option<Arc<DescriptorSet>>,
    prepass_material_set: Option<Arc<DescriptorSet>>,
    prepass_texture_set: Option<Arc<DescriptorSet>>,
    /// The same two again for the shadow pass's cutout pipeline. A third copy
    /// rather than a shared one for the reason the prepass keeps its own: set
    /// compatibility is a property of the layout a pipeline declares, and these
    /// three declare three. The buffer and the views inside them are the same
    /// objects, which is what keeps the three passes agreeing about a material.
    shadow_material_set: Option<Arc<DescriptorSet>>,
    shadow_texture_set: Option<Arc<DescriptorSet>>,
    /// Frames the GPU has not finished with, oldest first, bounded by
    /// [`FRAMES_IN_FLIGHT`]. Everything a frame named is held here until its
    /// fence signals.
    ///
    /// Their GPU work is still serialised through `frame_chain`: there is one
    /// set of graph images, so frame `n + 1` writes what frame `n` is reading.
    /// What runs ahead is the recording, not the execution.
    in_flight: RunAhead<InFlight>,
    /// Signalled when the last submitted frame finishes, and waited by the next.
    ///
    /// Only on the unsplit path. A split frame's ordering comes from the
    /// per-segment semaphores in [`PendingTail`] instead, because "the whole of
    /// the last frame" is exactly the dependency the split exists to remove.
    frame_chain: Option<Arc<Semaphore>>,
    /// Which frame this is, counted from the first. Picks the frame's set of
    /// graph images, and nothing else.
    frame_index: u64,
    /// The previous frame's trailing graphics segment, recorded and waiting to
    /// be submitted behind *this* frame's head. See [`PendingTail`].
    pending: Option<PendingTail>,
    /// Signalled by the trailing segment submitted last call, waited by the head
    /// submitted next call.
    ///
    /// That segment belongs to the frame two back, which is the frame that last
    /// wrote the set of graph images the next one will — so this is the only
    /// thing a head has to wait for, and the alternative is trusting a queue to
    /// execute submissions in the order they were made, which Vulkan does not
    /// promise.
    head_gate: Option<Arc<Semaphore>>,
    /// The overlay's submissions and presents, when one ran. Nothing waits on
    /// them, but they are bounded like the frames are and for a sharper reason:
    /// `FenceSignalFuture`'s destructor waits on its fence, so holding one slot
    /// and overwriting it each frame is a CPU stall on the previous frame's
    /// present — the run-ahead above, undone one line later.
    overlay_present: RunAhead<FenceSignalFuture<Box<dyn GpuFuture>>>,
    recreate_swapchain: bool,
    pending_extent: [u32; 2],
    /// What the swapchain should be built with. Changing it takes the same route
    /// a resize does — flag the recreation and let the next frame perform it —
    /// because it *is* a recreation, and doing it anywhere else would tear down
    /// images the frame in flight is still presenting from.
    present: PresentSettings,
    /// Per-pass GPU timing; `None` if the device lacks timestamp support.
    timestamps: Option<GpuTimestamps>,
    /// The structure this frame's graph was compiled for. A frame whose config
    /// still matches reuses the compiled graph — recompiling is for a change of
    /// *shape* (SSAO on or off, overlay or not), never for a change of contents.
    config: FrameConfig,
    frame: Frame,
    images: GraphImages,
}

impl VulkanRenderer {
    pub fn new(
        instance: &Arc<Instance>,
        surface: Arc<Surface>,
        extent: [u32; 2],
        present: PresentSettings,
    ) -> Self {
        let ctx = VkContext::new(instance, Some(&surface));
        let format = swapchain_color_format(&ctx, &surface);
        Self::build(ctx, format, extent, present, |ctx, format, extent| {
            SwapchainState::new(ctx, &surface, format, extent, present)
        })
    }

    /// A renderer that draws into an image instead of a window.
    ///
    /// Every pass, every pipeline and every descriptor set is the one the
    /// windowed renderer builds — the target is the only difference, and it has
    /// to be, because the point of rendering offscreen is to have evidence about
    /// what the window shows. A capture taken through a second, simpler path
    /// would only be evidence about that path.
    pub fn offscreen(instance: &Arc<Instance>, extent: [u32; 2]) -> Self {
        let ctx = VkContext::new(instance, None);
        // Presentation settings are inert here: there is no presentation engine
        // to hand an image to, so the default stands and nothing reads it.
        Self::build(
            ctx,
            OFFSCREEN_FORMAT,
            extent,
            PresentSettings::default(),
            SwapchainState::offscreen,
        )
    }

    /// The half of construction that does not know where the frame ends up.
    fn build(
        ctx: VkContext,
        format: Format,
        extent: [u32; 2],
        present: PresentSettings,
        make_target: impl FnOnce(&VkContext, Format, [u32; 2]) -> SwapchainState,
    ) -> Self {
        let forward = ForwardPass::new(&ctx, hdr::HDR_FORMAT, hdr::HDR_WIDE_FORMAT);
        let arena = MeshArena::new(&ctx);
        let hdr = HdrPass::new(&ctx, format);
        let exposure = ExposurePass::new(&ctx);
        let bloom = BloomPass::new(&ctx);
        let prepass = GeometryPrepass::new(&ctx);
        let ssao = SsaoPass::new(&ctx);
        let contact_shadows = ContactShadowPass::new(&ctx);
        let fog = FogPass::new(&ctx);
        let ssr = SsrPass::new(&ctx);
        let subsurface = SubsurfacePass::new(&ctx);
        let oit = OitPass::new(&ctx, forward.pipeline_layout());
        let refraction = RefractionPass::new(&ctx, forward.pipeline_layout());
        let taa = TaaPass::new(&ctx);
        let dof = DofPass::new(&ctx);
        let motion_blur = MotionBlurPass::new(&ctx);
        let shadow = ShadowPass::new(&ctx);
        let line = LinePass::new(&ctx, &forward.targets);
        let environment = EnvironmentPass::new(&ctx, &forward.targets);
        let swapchain = make_target(&ctx, format, extent);
        let timestamps = GpuTimestamps::new(&ctx);

        // Default textures so every material slot resolves to a valid view:
        // index 0 = white (a no-op multiply), index 1 = flat normal (0,0,1).
        let textures = vec![
            texture::upload_texture(
                &ctx,
                &[255, 255, 255, 255],
                [1, 1],
                Format::R8G8B8A8_UNORM,
                texture::MipPolicy::None,
            ),
            texture::upload_texture(
                &ctx,
                &[128, 128, 255, 255],
                [1, 1],
                Format::R8G8B8A8_UNORM,
                texture::MipPolicy::None,
            ),
        ];

        // Compiled for the editor's frame, which is what all but the headless
        // path uses; anything else recompiles on its first render.
        let gpu_culling = read_gpu_culling(&ctx);
        let occlusion_culling = gpu_culling && read_occlusion_culling();
        let config = FrameConfig {
            color_format: format,
            msaa: false,
            ssao: true,
            ssao_half_res: true,
            contact_shadows: true,
            ssr: false,
            subsurface: false,
            transparency: true,
            refraction: true,
            taa: true,
            auto_exposure: true,
            volumetric_fog: false,
            motion_blur: false,
            dof: false,
            bloom_mips: bloom::mip_count(extent),
            overlay: true,
            shadow_cascades: 0,
            shadow_resolution: 1,
            shadow_atlas: 0,
            async_compute: ctx.compute_queue.is_some(),
            gpu_culling,
            occlusion_culling,
        };
        let frame = frame::declare(config).expect("the engine's frame must compile");
        let slots = image_slots(&frame.graph);
        let images =
            GraphImages::allocate(&ctx.memory_allocator, &ctx, &frame.graph, extent, slots);

        Self {
            instances: InstanceStore::new(&ctx),
            cull: CullPass::new(&ctx),
            occlusion: OcclusionPass::new(&ctx),
            gpu_culling,
            occlusion_culling,
            ctx,
            swapchain,
            forward,
            hdr,
            exposure,
            bloom,
            prepass,
            ssao,
            contact_shadows,
            fog,
            ssr,
            subsurface,
            oit,
            refraction,
            taa,
            dof,
            motion_blur,
            shadow,
            line,
            environment,
            meshes: Vec::new(),
            arena,
            materials: vec![forward::to_gpu_material(&Material::default())],
            textures,
            material_buffer: None,
            material_set: None,
            texture_set: None,
            prepass_material_set: None,
            prepass_texture_set: None,
            shadow_material_set: None,
            shadow_texture_set: None,
            in_flight: RunAhead::new(FRAMES_IN_FLIGHT),
            frame_chain: None,
            frame_index: 0,
            pending: None,
            head_gate: None,
            overlay_present: RunAhead::new(FRAMES_IN_FLIGHT),
            recreate_swapchain: false,
            pending_extent: extent,
            present,
            timestamps,
            config,
            frame,
            images,
        }
    }

    /// Recompile the graph if the frame's structure changed, then (re)allocate
    /// what it owns. Also the resize path: a new extent keeps the graph and
    /// replaces only the images it sized against the old one.
    fn ensure_graph(&mut self, config: FrameConfig) {
        if self.config != config {
            self.frame = frame::declare(config).expect("the engine's frame must compile");
            self.config = config;
        } else if !self.images.is_stale(self.swapchain.extent) {
            return;
        }
        self.reallocate();
    }

    /// Copy the last rendered frame back to host memory as tightly packed
    /// `R8G8B8A8_SRGB`, with the extent it was rendered at.
    ///
    /// `None` for a windowed renderer, which has no readable target: a swapchain
    /// image belongs to the presentation engine, and this exists to look at what
    /// the passes produced rather than at what a compositor did with it.
    ///
    /// Waits for the frame first. That is the whole synchronisation story —
    /// `render_frame` already signalled a fence, and the copy is submitted after
    /// it has been reached, so nothing here can read a half-written image.
    pub fn capture(&mut self) -> Option<(Vec<u8>, [u32; 2])> {
        let image = self.swapchain.readback.first()?.clone();

        // A split frame holds its last segment back a frame on purpose, so
        // without this the readback copies the frame *before* the one the caller
        // just rendered. See [`PendingTail`].
        self.flush_pending_tail();
        self.in_flight.drain();
        // The overlay is the last thing to write the target when there is one,
        // so a readback that skipped it would copy the frame underneath it.
        self.overlay_present.drain();

        let extent = self.swapchain.extent;
        let buffer = Buffer::new_slice::<u8>(
            self.ctx.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            u64::from(extent[0]) * u64::from(extent[1]) * 4,
        )
        .expect("failed to allocate the capture buffer");

        let mut builder = self.new_command_buffer();
        builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image, buffer.clone()));
        builder.submit_and_wait(&self.ctx);

        let pixels = buffer.read().unwrap().to_vec();
        Some((pixels, extent))
    }

    /// Draw the overlay over a finished frame and present it.
    ///
    /// Shared by both submission paths, which differ only in *when* they reach
    /// it: unsplit, the frame just submitted; split, the frame before it, whose
    /// trailing segment this call has finally sent.
    fn finish_frame(
        &mut self,
        overlay_ready: Option<Arc<Semaphore>>,
        render_finished: Option<Arc<Semaphore>>,
        image_index: u32,
        raw_passes: &[PassBody],
        overlay: Option<Overlay<'_>>,
    ) {
        if let Some(ready) = overlay_ready {
            let mut future = record::SemaphoreWait::new(
                self.ctx.queue.clone(),
                ready,
                self.swapchain
                    .swapchain
                    .clone()
                    .map(|swapchain| (swapchain, image_index)),
            )
            .boxed();
            let mut overlay = overlay;
            for body in raw_passes {
                match body {
                    PassBody::Overlay => {
                        profile_scope!("overlay");
                        let draw = overlay
                            .take()
                            .expect("the graph scheduled an overlay pass but none was supplied");
                        future = draw(
                            future,
                            self.swapchain.image_views[image_index as usize].clone(),
                        );
                    }
                    other => unreachable!("{other:?} is not a raw pass"),
                }
            }
            // Presented through the overlay's own future rather than by the raw
            // path below: it is the last writer, so it owns the transition to
            // `PresentSrc` that the frame's `final_barriers` deliberately left
            // out. Not waited on — the next frame's acquire is what bounds it,
            // exactly as it did before the renderer recorded raw.
            let presented = match self.swapchain.swapchain.clone() {
                Some(swapchain) => future
                    .then_swapchain_present(
                        self.ctx.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain, image_index),
                    )
                    .boxed()
                    .then_signal_fence_and_flush(),
                None => future.then_signal_fence_and_flush(),
            };
            match presented.map_err(Validated::unwrap) {
                Ok(future) => self.overlay_present.push(future),
                Err(VulkanError::OutOfDate) => self.recreate_swapchain = true,
                Err(e) => tracing::error!("failed to flush the overlay: {e}"),
            }
            return;
        }

        debug_assert!(raw_passes.is_empty());
        let Some(swapchain) = self.swapchain.swapchain.clone() else {
            return;
        };
        // Raw, like the submit — but for the opposite reason: this one is
        // simply what vulkano exposes for a present that waits on a semaphore
        // the caller owns.
        let wait_semaphores = render_finished
            .iter()
            .map(|semaphore| SemaphorePresentInfo::new(semaphore.clone()))
            .collect();
        let present_info = PresentInfo {
            wait_semaphores,
            swapchain_infos: vec![SwapchainPresentInfo::swapchain_image_index(
                swapchain,
                image_index,
            )],
            ..Default::default()
        };
        let result = self.ctx.queue.clone().with(|mut queue| {
            // SAFETY: the image was acquired from this swapchain and has not
            // been presented since; the wait semaphore is the one the
            // submission signals, or none, in which case the overlay path has
            // already waited on the CPU.
            unsafe { queue.present(&present_info) }
                .map(|results| results.into_iter().collect::<Vec<_>>())
        });
        match result.map_err(Validated::unwrap) {
            Ok(results) => {
                for result in results {
                    match result {
                        Ok(suboptimal) => self.recreate_swapchain |= suboptimal,
                        Err(VulkanError::OutOfDate) => self.recreate_swapchain = true,
                        Err(e) => tracing::error!("failed to present: {e}"),
                    }
                }
            }
            Err(VulkanError::OutOfDate) => self.recreate_swapchain = true,
            Err(e) => tracing::error!("failed to present: {e}"),
        }
    }

    /// Submit the trailing segment held back from the last frame, so that the
    /// GPU has the whole of it.
    ///
    /// The readback path's, and the shutdown path's: everywhere else the
    /// segment is submitted behind the next frame's head, which is the entire
    /// reason it waits. Presents nothing — a caller that wanted the image on
    /// screen would have rendered another frame.
    fn flush_pending_tail(&mut self) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        let mut waits = vec![(
            pending.compute_done,
            ash::vk::PipelineStageFlags::ALL_COMMANDS,
        )];
        if let Some(acquired) = pending.acquire {
            waits.push((
                acquired,
                ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ));
        }
        let in_flight = record::submit_frame(
            &self.ctx,
            &self.ctx.queue.clone(),
            pending.command_buffer,
            pending.keep,
            &waits,
            &[],
        );
        self.in_flight.push(in_flight);
    }

    fn new_command_buffer(&self) -> Recorder {
        Recorder::new(&self.ctx)
    }

    /// The queue a segment is submitted to.
    ///
    /// The compiler plans against roles, not families; this is where a role
    /// meets the device. `AsyncCompute` can only be scheduled when there is a
    /// queue for it — `FrameConfig::async_compute` says so — so the fallback
    /// here is unreachable rather than a silent downgrade.
    fn queue_of(&self, queue: GraphQueue) -> &Arc<vulkano::device::Queue> {
        match queue {
            GraphQueue::Graphics => &self.ctx.queue,
            GraphQueue::AsyncCompute => self
                .ctx
                .compute_queue
                .as_ref()
                .expect("the graph scheduled an async segment with no queue to put it on"),
        }
    }

    fn queue_family(&self, queue: GraphQueue) -> u32 {
        self.queue_of(queue).queue_family_index()
    }

    /// The view backing a graph resource, whoever owns the allocation.
    ///
    /// Every image in a frame is graph-owned but one: the TAA resolve's output
    /// is *imported*, because a history has to survive a frame boundary and a
    /// transient by contract does not. Anything downstream that reads the
    /// frame's colour can be handed either, depending on which optical stages
    /// the frame has — so it asks by `ResourceId` and this decides, rather than
    /// each consumer re-deriving which pass ran last.
    fn view_of(&self, id: ResourceId) -> Arc<ImageView> {
        if let Some(fog) = self.frame.ids.fog
            && id == fog.scatter
        {
            // Imported and ping-ponged like the TAA history below, and for the
            // same reason: the scatter pass reads its own previous frame.
            return self.fog.scatter_view();
        }
        if let Some(ids) = self.frame.ids.occlusion {
            // Imported and ping-ponged like the TAA history, and for a stronger
            // version of the same reason: what the cull reads is not merely a
            // history, it is the only thing that keeps the frame acyclic.
            if id == ids.hiz {
                return self.occlusion.hiz_view();
            }
            if id == ids.history {
                return self.occlusion.history_view();
            }
        }
        match self.frame.ids.taa {
            Some(taa) if id == taa.output => self.taa.output_view(),
            _ => self.images.view(id),
        }
    }

    /// Which Vulkan object one resource in the compiled plan names.
    ///
    /// The graph is device-free, so it tracks `ResourceId`s and leaves this to
    /// the renderer: graph-owned images, images imported from the pass that owns
    /// them across a frame boundary, the swapchain image, and buffers.
    fn barrier_target(&self, id: ResourceId, swapchain: &Arc<ImageView>) -> Option<record::Target> {
        let ids = &self.frame.ids;

        if id == ids.swapchain_color {
            return Some(record::Target::Image(swapchain.image().clone()));
        }
        // The imported buffers. See `record::Target::Memory` for why these do
        // not need resolving to their `Subbuffer`s.
        if id == ids.object_transforms
            || id == ids.instance_index
            || id == ids.exposure
            || Some(id) == ids.histogram
        {
            return Some(record::Target::Memory);
        }
        // Imported images: a history has to survive a frame boundary, and a
        // transient by contract does not, so the pass that ping-pongs it owns
        // the allocation.
        if let Some(fog) = ids.fog
            && id == fog.scatter
        {
            return Some(record::Target::Image(
                self.fog.scatter_view().image().clone(),
            ));
        }
        if let Some(occlusion) = ids.occlusion {
            if id == occlusion.hiz {
                return Some(record::Target::Image(
                    self.occlusion.hiz_view().image().clone(),
                ));
            }
            if id == occlusion.history {
                return Some(record::Target::Image(
                    self.occlusion.history_view().image().clone(),
                ));
            }
        }
        if let Some(taa) = ids.taa {
            if id == taa.output {
                return Some(record::Target::Image(
                    self.taa.output_view().image().clone(),
                ));
            }
            if id == taa.history {
                return Some(record::Target::Image(
                    self.taa.history_view().image().clone(),
                ));
            }
        }

        self.images
            .try_view(id)
            .map(|view| record::Target::Image(view.image().clone()))
    }

    fn reallocate(&mut self) {
        // Rebuilt rather than replaced, so the image cache survives: a
        // structural toggle re-points `views` at allocations that already exist
        // instead of recreating every image in the frame. See `GraphImages`.
        self.images.rebuild(
            &self.ctx.memory_allocator,
            &self.ctx,
            &self.frame.graph,
            self.swapchain.extent,
            image_slots(&self.frame.graph),
        );
    }
}

impl RenderBackend for VulkanRenderer {
    fn gpu_culling(&self) -> bool {
        self.gpu_culling
    }

    fn load_mesh(&mut self, mesh: &CpuMesh) -> MeshHandle {
        let (span, bounds) = self.arena.upload(&self.ctx, &mesh.vertices, &mesh.indices);
        let handle = MeshHandle(self.meshes.len() as u32);
        self.meshes.push(GpuMesh { span, bounds });
        handle
    }

    fn mesh_bounds(&self, mesh: MeshHandle) -> Option<Aabb> {
        self.meshes.get(mesh.0 as usize).map(|gpu| gpu.bounds)
    }

    fn load_material(&mut self, material: &Material) -> MaterialHandle {
        let handle = MaterialHandle(self.materials.len() as u32);
        self.materials.push(forward::to_gpu_material(material));
        self.material_buffer = None;
        self.material_set = None;
        self.prepass_material_set = None;
        self.shadow_material_set = None;
        handle
    }

    fn load_texture(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        srgb: bool,
    ) -> TextureHandle {
        // The shader's sampler array and `build_texture_set` only bind the first
        // MAX_TEXTURES views, so a handle past that would index out of range.
        // Clamp to the white default (handle 0) instead of handing back a slot
        // the GPU can't sample.
        if self.textures.len() >= MAX_TEXTURES {
            tracing::warn!(
                "texture cap reached ({MAX_TEXTURES}); ignoring load and using the \
                 white default — material will render untextured"
            );
            return TextureHandle(0);
        }

        // Color maps are authored in sRGB so the GPU decodes them to linear on
        // sample; data maps (normal, metallic-roughness) are already linear.
        let format = if srgb {
            Format::R8G8B8A8_SRGB
        } else {
            Format::R8G8B8A8_UNORM
        };
        let view = texture::upload_texture(
            &self.ctx,
            pixels,
            [width, height],
            format,
            texture::MipPolicy::Generate,
        );
        let handle = TextureHandle(self.textures.len() as u32);
        self.textures.push(view);
        self.texture_set = None;
        self.prepass_texture_set = None;
        self.shadow_texture_set = None;
        handle
    }

    fn load_environment(&mut self, pixels: &[f32], width: u32, height: u32) {
        self.environment
            .set_source(&self.ctx, pixels, [width, height]);
    }

    fn resize(&mut self, extent: [u32; 2]) {
        self.pending_extent = extent;
        self.recreate_swapchain = true;
    }

    fn render(
        &mut self,
        draws: DrawList<'_>,
        transparent: DrawList<'_>,
        refractive: DrawList<'_>,
        decals: &[DecalInstance],
        lighting: &SceneLighting,
        camera: &Camera,
        ssao: &SsaoSettings,
        contact_shadows: &ContactShadowSettings,
        ssr: &SsrSettings,
        subsurface: &SubsurfaceSettings,
        transparency: &TransparencySettings,
        refraction: &RefractionSettings,
        taa: &TaaSettings,
        motion_blur: &MotionBlurSettings,
        dof: &DofSettings,
        bloom: &BloomSettings,
        hdr: &HdrSettings,
        environment: &EnvironmentSettings,
        fog: &FogSettings,
        dt: f32,
    ) {
        // No overlay path (e.g. export/headless) draws no debug lines and no
        // cascades. Contact shadows are not cascades: they need no caster list
        // and no matrices, only the depth buffer, so they run here too.
        self.render_frame(
            draws,
            transparent,
            refractive,
            decals,
            lighting,
            camera,
            ssao,
            contact_shadows,
            ssr,
            subsurface,
            transparency,
            refraction,
            taa,
            motion_blur,
            dof,
            bloom,
            hdr,
            environment,
            fog,
            dt,
            &[],
            None,
            None,
            None,
        );
    }
}

impl VulkanRenderer {
    pub fn queue(&self) -> Arc<Queue> {
        self.ctx.queue.clone()
    }

    pub fn color_format(&self) -> Format {
        self.swapchain.format
    }

    /// The extent the next frame will be drawn at — the swapchain's, not the
    /// window's, which can already be a resize ahead of it. Culling has to use
    /// this one or its side planes won't be the ones on screen.
    pub fn extent(&self) -> [u32; 2] {
        self.swapchain.extent
    }

    /// Whole-frame GPU time in milliseconds, or `None` if the device doesn't
    /// support timestamp queries. Trails the displayed frame by one.
    pub fn gpu_frame_ms(&self) -> Option<f32> {
        self.timestamps.as_ref().map(GpuTimestamps::last_frame_ms)
    }

    /// Ask for a different present mode or image count.
    ///
    /// Takes effect on the next frame, through the same recreation path a resize
    /// takes — a swapchain cannot be replaced while the frame in flight is still
    /// presenting from its images. A call that changes nothing does nothing, so
    /// the app may hand over the resource every frame without recreating one.
    pub fn set_present(&mut self, present: PresentSettings) {
        if self.present != present {
            self.present = present;
            self.recreate_swapchain = true;
        }
    }

    /// What the surface honoured of [`set_present`](Self::set_present), or `None`
    /// offscreen. Not necessarily what was asked for — see
    /// [`SwapchainState::applied_present`].
    pub fn applied_present(&self) -> Option<(vulkano::swapchain::PresentMode, u32)> {
        self.swapchain.applied_present()
    }

    /// File the GPU spans of frames that have completed since the last call.
    /// Separate from rendering because the profiler lives in the world, which
    /// the renderer deliberately can't reach.
    pub fn drain_gpu_spans(&mut self, profiler: &mut Profiler) {
        if let Some(timestamps) = self.timestamps.as_mut() {
            timestamps.drain_completed(profiler);
        }
    }

    /// Live device-local VRAM as `(used, total)` bytes; `used` is `None` when the
    /// driver doesn't expose `VK_EXT_memory_budget`.
    pub fn gpu_memory(&self) -> (Option<u64>, u64) {
        self.ctx.vram_bytes()
    }

    /// Like [`render`](RenderBackend::render) but with the editor's inputs:
    /// shadow cascades, debug lines, GPU timing, and `overlay` — the editor UI,
    /// composited onto the final image before present.
    ///
    /// `overlay` is an `Option` rather than a second entry point because turning
    /// the editor off must not also turn off everything else this path supplies.
    /// `None` is a frame with cascades, lines and timing intact and no UI at all:
    /// the graph drops its `overlay` node, so the cost measured is the scene's
    /// alone. That is what [`Diagnostics::overlay`](crate::scene::Diagnostics)
    /// switches, and the reason it can be switched.
    #[allow(clippy::too_many_arguments)]
    pub fn render_with_overlay(
        &mut self,
        draws: DrawList<'_>,
        transparent: DrawList<'_>,
        refractive: DrawList<'_>,
        decals: &[DecalInstance],
        lighting: &SceneLighting,
        camera: &Camera,
        ssao: &SsaoSettings,
        contact_shadows: &ContactShadowSettings,
        ssr: &SsrSettings,
        subsurface: &SubsurfaceSettings,
        transparency: &TransparencySettings,
        refraction: &RefractionSettings,
        taa: &TaaSettings,
        motion_blur: &MotionBlurSettings,
        dof: &DofSettings,
        bloom: &BloomSettings,
        hdr: &HdrSettings,
        environment: &EnvironmentSettings,
        fog: &FogSettings,
        dt: f32,
        debug_lines: &[DebugLine],
        profiler_frame: u64,
        shadows: Option<ShadowFrame<'_>>,
        overlay: Option<Overlay<'_>>,
    ) {
        self.render_frame(
            draws,
            transparent,
            refractive,
            decals,
            lighting,
            camera,
            ssao,
            contact_shadows,
            ssr,
            subsurface,
            transparency,
            refraction,
            taa,
            motion_blur,
            dof,
            bloom,
            hdr,
            environment,
            fog,
            dt,
            debug_lines,
            Some(profiler_frame),
            shadows,
            overlay,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn render_frame(
        &mut self,
        draws: DrawList<'_>,
        transparent: DrawList<'_>,
        refractive: DrawList<'_>,
        decals: &[DecalInstance],
        lighting: &SceneLighting,
        camera: &Camera,
        ssao: &SsaoSettings,
        contact_shadows: &ContactShadowSettings,
        ssr: &SsrSettings,
        subsurface: &SubsurfaceSettings,
        transparency: &TransparencySettings,
        refraction: &RefractionSettings,
        taa: &TaaSettings,
        motion_blur: &MotionBlurSettings,
        dof: &DofSettings,
        bloom: &BloomSettings,
        hdr: &HdrSettings,
        environment: &EnvironmentSettings,
        fog: &FogSettings,
        // Seconds since the last frame, for exposure adaptation. Zero converges
        // immediately, which is what a one-shot render wants.
        dt: f32,
        debug_lines: &[DebugLine],
        profiler_frame: Option<u64>,
        shadows: Option<ShadowFrame<'_>>,
        overlay: Option<Overlay<'_>>,
    ) {
        if self.pending_extent[0] == 0 || self.pending_extent[1] == 0 {
            return;
        }

        if self.recreate_swapchain {
            // Whatever is held back is held back against the *old* swapchain: it
            // writes an image acquired from it, and the index it was acquired at
            // means nothing in the replacement. Submitted and waited for here so
            // the images it names are finished with, and then dropped
            // unpresented — which is what a resize does to a frame anyway.
            self.flush_pending_tail();
            self.in_flight.drain();
            self.overlay_present.drain();
            if self.swapchain.recreate(self.pending_extent, self.present) {
                self.recreate_swapchain = false;
            } else {
                return;
            }
        }

        // The frame's structure, and the only thing that recompiles the graph.
        // Everything else about this call — how many objects, which camera, what
        // exposure — flows through the same compiled schedule.
        self.ensure_graph(FrameConfig {
            color_format: self.swapchain.format,
            msaa: taa.msaa,
            ssao: ssao.enabled,
            ssao_half_res: ssao.half_resolution,
            contact_shadows: contact_shadows.enabled,
            ssr: ssr.enabled,
            subsurface: subsurface.enabled,
            // Emptiness is part of the frame's structure, not a flag read at
            // record time. With nothing blended and nothing refractive on
            // screen these two chains still cost two full-resolution attachment
            // clears, a seven-level half-res pyramid build, and two full-res
            // composites that are provably identity operations — around 290 MB
            // a frame, roughly a quarter of the whole budget, spent on doing
            // nothing.
            //
            // Safe to vary per frame only because `GraphImages` caches its
            // allocations: an object drifting on and off screen recompiles the
            // graph, and without that cache each crossing would drop and
            // recreate every image in the frame.
            transparency: transparency.enabled && !transparent.is_empty(),
            refraction: refraction.enabled && !refractive.is_empty(),
            taa: taa.enabled,
            auto_exposure: hdr.auto_exposure,
            motion_blur: motion_blur.enabled,
            dof: dof.enabled,
            volumetric_fog: self.fog.enabled(fog),
            // Zero when bloom is off, and also when the window is too small for
            // a chain — so a frame dragged to a sliver drops the passes rather
            // than dispatching over one-texel levels.
            bloom_mips: if bloom.enabled {
                bloom::mip_count(self.swapchain.extent)
            } else {
                0
            },
            overlay: overlay.is_some(),
            // Sourced from the cascade set rather than the setting, so the
            // number of passes the graph declares cannot disagree with the
            // number of matrices there are to draw them with.
            shadow_cascades: shadows.map_or(0, |s| s.cascades.count as u8),
            shadow_resolution: shadows.map_or(1, |s| s.settings.resolution),
            // Sourced from the fitted atlas for the reason the cascade count is
            // sourced from the cascade set: a frame where every punctual light
            // opted out, or where none reached the budget, declares no atlas
            // rather than clearing one nothing reads.
            shadow_atlas: shadows.map_or(0, |s| {
                if s.atlas.faces.is_empty() {
                    0
                } else {
                    s.atlas.resolution
                }
            }),
            // What the device turned out to have, not a preference. Whether the
            // frame is actually split is then the compiler's call — a
            // configuration with no compute tail to move gets the plan it always
            // had, on a device with two queues as much as on one.
            async_compute: self.ctx.compute_queue.is_some(),
            gpu_culling: self.gpu_culling,
            occlusion_culling: self.occlusion_culling,
        });

        // Release what the GPU has already finished and block only if the CPU
        // is [`FRAMES_IN_FLIGHT`] frames ahead: recording raw means this crate
        // owns those lifetimes, and the bound is what caps the list.
        self.in_flight.wait_for_room();
        self.overlay_present.wait_for_room();

        // Split out because under Fifo this blocks until the presentation engine
        // hands back an image — a vsync wait, not work. Folded into a single
        // "render" scope it swamps the numbers and hides real regressions.
        let (image_index, suboptimal, acquire_semaphore) = match self.swapchain.swapchain.clone() {
            Some(swapchain) => {
                profile_scope!("acquire");
                // The raw form, because the frame is submitted raw: the
                // `SwapchainAcquireFuture` the safe one returns owns its
                // semaphore privately, and nothing can read it back out to put
                // in a `VkSubmitInfo`.
                let semaphore = Arc::new(
                    Semaphore::from_pool(self.ctx.device.clone())
                        .expect("failed to create the acquire semaphore"),
                );
                // SAFETY: the semaphore is unsignalled and has no pending wait
                // — it was made a line above, and the frame that ends up waiting
                // on it holds it until its own fence signals. The previous frame
                // may still be executing, which acquire allows: the presentation
                // engine hands back only an image it has finished with.
                let acquired = unsafe {
                    swapchain.acquire_next_image(&AcquireNextImageInfo {
                        semaphore: Some(semaphore.clone()),
                        ..Default::default()
                    })
                };
                match acquired.map_err(Validated::unwrap) {
                    Ok(acquired) => (
                        acquired.image_index,
                        acquired.is_suboptimal,
                        Some(semaphore),
                    ),
                    Err(VulkanError::OutOfDate) => {
                        self.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                }
            }
            // Offscreen: one image, always available, and nothing to wait on —
            // no presentation engine owns it, so there is no hand-back to time.
            None => (0, false, None),
        };
        if suboptimal {
            self.recreate_swapchain = true;
        }

        // Which set of the graph's images this frame writes. One set unless the
        // frame was split, in which case the previous frame's compute tail is
        // still reading the other.
        self.images.set_slot(self.frame_index);
        self.frame_index = self.frame_index.wrapping_add(1);

        // Taken out of `self` for the duration of recording: the pass brackets
        // below interleave with calls that borrow `self` immutably, and a field
        // borrow held across them wouldn't compile. Restored before returning,
        // and only if it was taken — every early return above happens first.
        let mut timestamps = match profiler_frame {
            Some(frame) => {
                let mut taken = self.timestamps.take();
                if let Some(timestamps) = taken.as_mut() {
                    timestamps.begin_frame(frame);
                }
                taken
            }
            None => None,
        };

        let recording = crate::profile::scope("record");
        let mut builder = self.new_command_buffer();

        // Resets are illegal inside a render pass, so they and the reserved
        // whole-frame pair go here, ahead of every pass below.
        if let Some(timestamps) = timestamps.as_mut() {
            timestamps.record_resets(&mut builder);
        }

        // Drive the SSAO tunables from the world resource each frame. When SSAO
        // is disabled we skip the three passes and bind a 1x1 white AO view, so
        // the forward shader samples 1.0 (no occlusion) and is otherwise unchanged.
        self.ssao.radius = ssao.radius;
        self.ssao.bias = ssao.bias;
        self.ssao.power = ssao.power;
        self.hdr.manual_exposure = hdr.manual_exposure();
        self.hdr.auto_exposure = hdr.auto_exposure;
        self.exposure.begin_frame(hdr, dt);
        self.bloom.begin_frame(bloom, hdr);
        // Before anything records: this is where the frame's subpixel jitter is
        // chosen, and every pass that rasterises geometry has to be drawn with
        // the matrices it returns rather than deriving its own from `camera`.
        let view = self
            .taa
            .begin_frame(&self.ctx, taa, camera, self.swapchain.extent);
        // Both resolve their optics against this frame's camera: the lens takes
        // its focal length from the field of view, and the shutter takes the
        // depth range it linearises with.
        // Beside the jitter and for the same reason: this decides which half of
        // the pyramid pair the cull reads and which the build writes, and every
        // pass that touches either has to agree. It also answers whether the
        // half being read holds anything — a freshly allocated pair does not,
        // and the cull is told to test nothing that frame.
        let occlusion_live = self.occlusion.begin_frame(
            &self.ctx,
            self.gpu_culling && self.occlusion_culling,
            self.swapchain.extent,
        );
        self.dof.begin_frame(dof, camera, self.swapchain.extent);
        // Resolved against the camera for the reason the lens is: the kernel's
        // width in pixels is a world length divided by a view depth, so both
        // dispatches have to be handed one projection and one depth range.
        self.subsurface
            .begin_frame(subsurface, camera, self.swapchain.extent);
        // Reflections resolve against the camera *and* the environment: the
        // composite subtracts the environment term the forward pass added, so
        // it has to be handed the same rotation and the same tint that pass
        // will be. The pyramid depth comes off the allocated image rather than
        // the declaration, because a window too small for seven levels gets
        // fewer and the march has to stop at the last one that exists.
        if let Some(ids) = self.frame.ids.ssr {
            let ambient = lighting.ambient_color * lighting.ambient_nits;
            self.ssr.begin_frame(
                ssr,
                &view,
                self.swapchain.extent,
                self.images.mip_levels(ids.hiz),
                environment.yaw,
                environment::SPECULAR_MIPS,
                self.environment.specular_tint(ambient, environment),
            );
        }
        self.motion_blur.begin_frame(motion_blur, camera);
        // Whether or not the two dispatches run, and that is the point: the
        // block resolved here is what every shading pass applies the fog out of,
        // and past the volume's far plane — or in a frame with no volume at all —
        // the analytic integral is described by these very same numbers. The
        // ambient is the same product the lighting block falls back to, so the
        // air and the surfaces standing in it agree about how bright the sky is.
        self.fog.begin_frame(
            &self.ctx,
            fog,
            lighting,
            shadows,
            &view,
            camera.position,
            lighting.ambient_color * lighting.ambient_nits,
            self.swapchain.extent,
        );
        // Sourced from the compiled frame, not from the setting: a window too
        // small for a chain leaves bloom enabled but unbuilt, and a non-zero
        // strength would then blend the 1x1 black stand-in into the image and
        // darken it. The same reason the cascade count comes from the cascade
        // set rather than from `ShadowSettings`.
        self.hdr.bloom_strength = match self.frame.ids.bloom {
            Some(_) => self.bloom.strength(),
            None => 0.0,
        };
        if let Some(shadows) = shadows {
            self.shadow.constant_bias = shadows.settings.constant_bias;
            self.shadow.slope_bias = shadows.settings.slope_bias;
            self.shadow.punctual_constant_bias = shadows.settings.punctual_constant_bias;
            self.shadow.punctual_slope_bias = shadows.settings.punctual_slope_bias;
        }

        // Material table and texture array are static after asset load, so cache
        // their descriptor sets and rebuild only when invalidated (set to None).
        if self.material_buffer.is_none() {
            self.material_buffer = Some(forward::material_buffer(&self.ctx, &self.materials));
        }
        let materials = self.material_buffer.clone().unwrap();
        if self.material_set.is_none() {
            self.material_set = Some(self.forward.build_material_set(&self.ctx, &materials));
        }
        if self.texture_set.is_none() {
            self.texture_set = Some(self.forward.build_texture_set(&self.ctx, &self.textures));
        }
        if self.prepass_material_set.is_none() {
            self.prepass_material_set =
                Some(self.prepass.build_material_set(&self.ctx, &materials));
        }
        if self.prepass_texture_set.is_none() {
            self.prepass_texture_set =
                Some(self.prepass.build_texture_set(&self.ctx, &self.textures));
        }
        if self.shadow_material_set.is_none() {
            self.shadow_material_set = Some(self.shadow.build_material_set(&self.ctx, &materials));
        }
        if self.shadow_texture_set.is_none() {
            self.shadow_texture_set = Some(self.shadow.build_texture_set(
                &self.ctx,
                &self.textures,
                self.prepass.material_sampler(),
            ));
        }
        let material_set = self.material_set.clone().unwrap();
        let texture_set = self.texture_set.clone().unwrap();
        let prepass_material_set = self.prepass_material_set.clone().unwrap();
        let prepass_texture_set = self.prepass_texture_set.clone().unwrap();

        // Every list a frame draws is an ordering over one shared `RenderItem`
        // array, so any non-empty list carries the whole of it. Taking the
        // longest rather than the camera's is what keeps a frame that culled
        // everything on screen — but still casts shadows — from syncing nothing.
        let no_casters: [DrawList<'_>; 0] = [];
        let items = [draws, transparent, refractive]
            .into_iter()
            .chain(
                shadows
                    .map_or(&no_casters[..], |s| s.casters)
                    .iter()
                    .copied(),
            )
            .chain(
                shadows
                    .map_or(&no_casters[..], |s| s.punctual_casters)
                    .iter()
                    .copied(),
            )
            .map(|list| list.items)
            .max_by_key(|items| items.len())
            .unwrap_or(&[]);

        // The matrices themselves, brought up to date in place: only the rows
        // whose object actually moved are written, and the copy that carries
        // them is recorded here — at the top of the frame's command buffer,
        // ahead of every pass that reads them.
        self.instances.sync(&self.ctx, &mut builder, items);

        // The draw orders, as row numbers into those rows. Still per frame,
        // because the orders are: what the camera can see and what casts into
        // each cascade is a different answer every frame even when nothing
        // moved. Four bytes an entry rather than the 192 the matrices took.
        // The camera-visible items come first, so the two screen-space passes
        // still index from zero and the cascades index from
        // `objects.cascade_bases`.
        let objects = self.instances.upload_lists(
            draws,
            transparent,
            refractive,
            shadows.map_or(&no_casters, |s| s.casters),
            shadows.map_or(&no_casters, |s| s.punctual_casters),
        );
        let object_rows = self.instances.rows().clone();

        // The views the compute path culls against, in the numbering `cull.rs`
        // documents: the camera, then each cascade, then each punctual face.
        // Faces individually rather than one view per light, which is the same
        // narrowing `ShadowPass::record_atlas` does on the CPU path and for the
        // same reason — a face frustum has its apex at the light, so nothing
        // outside one can occlude light entering it.
        let cull_views: Vec<CullView> =
            if self.gpu_culling {
                let mut views = vec![CullView::Frustum(Frustum::from_view_projection(
                    view.unjittered_view_proj,
                ))];
                if let Some(shadows) = shadows {
                    views.extend(
                        shadows.cascades.cascades[..shadows.cascades.count]
                            .iter()
                            .map(|cascade| CullView::Cascade {
                                light_view: cascade.light_view,
                                half_extent: cascade.half_extent,
                                depth_range: cascade.depth_range,
                            }),
                    );
                    views.extend(shadows.atlas.faces.iter().map(|face| {
                        CullView::Frustum(Frustum::from_view_projection(face.view_proj))
                    }));
                }
                views
            } else {
                Vec::new()
            };
        let cull_frame = self.gpu_culling.then(|| {
            let meshes = &self.meshes;
            let materials = &self.materials;
            self.cull.prepare(
                &self.ctx,
                draws,
                |mesh| {
                    meshes
                        .get(mesh as usize)
                        .map(|mesh| (mesh.span, mesh.bounds))
                },
                |material| {
                    materials
                        .get(material as usize)
                        .is_some_and(GpuMaterial::is_masked)
                },
                cull_views.len() as u32,
            )
        });

        // One set per pipeline layout per frame, rather than one per pass. The
        // buffer is a fresh subbuffer each frame so none of these can be cached
        // across frames, but every cascade binds the same buffer through the
        // same layout — so the shadow set is built once here instead of once per
        // cascade, which is where the duplication actually was. They are kept
        // separate rather than shared because set compatibility is a property of
        // the layout each pipeline declares, not of the buffer written into it.
        // Which buffer the *opaque* passes index through. Under GPU culling it
        // is the one the dispatch wrote; the blended and refractive queues are
        // still CPU-culled and keep the host-written one, which is why the
        // graph names the two apart. See `blended_index` in `frame::declare`.
        let opaque_indices = match &cull_frame {
            Some(_) => self.cull.indices().clone(),
            None => objects.indices.clone(),
        };
        let forward_object_set =
            self.forward
                .build_object_set(&self.ctx, &object_rows, &opaque_indices);
        // One block for the whole frame, bound by the forward pass's set 0 and
        // the prepass's alike. See `ForwardPass::upload_decals`.
        let decal_block = self.forward.upload_decals(decals);
        let caster_sets = shadows.is_some().then(|| shadow::CasterSets {
            objects: self
                .shadow
                .build_object_set(&self.ctx, &object_rows, &opaque_indices),
            materials: self.shadow_material_set.clone().unwrap(),
            textures: self.shadow_texture_set.clone().unwrap(),
        });
        let prepass_object_set = self.frame.ids.prepass.map(|_| {
            self.prepass
                .build_object_set(&self.ctx, &object_rows, &opaque_indices)
        });

        // Uploaded once even though the prepass and the SSAO resolve both read
        // it — which is what the shared `object_transforms` declaration in
        // `frame::declare` records.
        let frame_uniforms = self
            .frame
            .ids
            .prepass
            .map(|_| self.prepass.begin_frame(&view));
        // The AO extent rather than the frame's: the noise tiles per AO texel,
        // so a half-res target that scaled the pattern by the window would
        // stretch the rotation across two pixels and band the result.
        let ssao_uniforms = self.frame.ids.ssao.map(|ids| {
            self.ssao.begin_frame(
                self.images.extent(ids.raw_ao),
                frame_uniforms
                    .clone()
                    .expect("SSAO reads the geometry prepass"),
            )
        });
        // The frame's view matrix rather than the camera's, for the reason every
        // rasterising pass takes its matrices from `FrameView`: the depth this
        // marches was written under that projection's jitter.
        let contact_shadow_uniforms = self.frame.ids.contact_shadows.map(|_| {
            self.contact_shadows.begin_frame(
                contact_shadows,
                &view,
                lighting.sun.direction_to_light(),
                frame_uniforms
                    .clone()
                    .expect("contact shadows read the geometry prepass"),
            )
        });

        // With SSAO off the graph has no AO node, so the forward pass samples a
        // 1x1 white view instead: "no occlusion" with no second shader path.
        let ao_view = match self.frame.ids.ssao {
            Some(ids) => self.images.view(ids.ao),
            None => self.ssao.white_view(),
        };

        // And again for the contact-shadow mask: with the march off there is no
        // node to read, so the forward pass samples a 1x1 white view and every
        // pixel reports "lit".
        let contact_shadow_view = match self.frame.ids.contact_shadows {
            Some(id) => self.images.view(id),
            None => self.contact_shadows.lit_view(),
        };

        // Same trick for shadows: with them off there is no cascade image to
        // read, so the forward pass samples a 1x1 depth texture of 1.0 and every
        // comparison reports "lit".
        let shadow_view = match self.frame.ids.shadows {
            Some(id) => self.images.view(id),
            None => self.shadow.lit_view(),
        };
        debug_assert_eq!(
            shadow_view.view_type(),
            vulkano::image::view::ImageViewType::Dim2dArray,
            "the forward pipeline binds the cascades as texture2DArray",
        );

        // And the same for the punctual atlas, which is a plain 2D image: with
        // nothing casting there is no atlas to read, so the forward pass samples
        // a 1x1 depth texture of 1.0 and every lookup reports "lit".
        let atlas_view = match self.frame.ids.shadow_atlas {
            Some(id) => self.images.view(id),
            None => self.shadow.lit_atlas_view(),
        };
        // What the forward pass indexes the face table by. Empty when nothing
        // casts, which is the frame where every light's face index is -1.
        let empty_atlas = ShadowAtlas::default();
        let atlas = shadows.map_or(&empty_atlas, |s| s.atlas);

        // What everything past shading composites: whichever image the optical
        // chain left the frame's colour in. `declare` already decided that, and
        // every pass in the chain recorded what it was handed, so nothing here
        // re-derives an order.
        let scene_color = self.view_of(self.frame.ids.scene_color);

        // Built before the walk rather than inside the forward pass's body,
        // because two passes bind these five: the forward pass and the
        // transparency accumulation, which draws through the same pipeline
        // layout. Which of them the compiler scheduled first is not something
        // either may depend on.
        let forward_sets = self.forward.begin_frame(
            self,
            lighting,
            camera,
            self.swapchain.extent,
            ao_view.clone(),
            contact_shadow_view.clone(),
            shadow_view.clone(),
            atlas_view.clone(),
            shadows,
            atlas,
            material_set.clone(),
            texture_set.clone(),
            forward_object_set.clone(),
            decal_block.clone(),
            environment,
            // The same allocation both fog dispatches bind, and the same
            // stand-in volume when they did not run: one description of the
            // medium, three places that read it.
            self.fog.uniforms(),
            self.fog
                .volume_or_fallback(self.frame.ids.fog.map(|ids| self.images.view(ids.volume))),
            self.fog.sampler(),
        );

        // The whole frame, in the order the compiler derived. Nothing below
        // decides what runs next, what an image's layout is, or what has to
        // finish before what — a pass that needs a different place in the frame
        // gets there by changing its declarations, not by being moved here.
        let extent = self.swapchain.extent;
        let swapchain_view = self.swapchain.image_views[image_index as usize].clone();
        // Passes that own their submission, gathered before anything records:
        // they run on the future after this command buffer rather than inside
        // it, and the walk below skips them.
        let raw_passes: Vec<PassBody> = self
            .frame
            .graph
            .order()
            .iter()
            .filter(|pass| self.frame.graph.pass_kind(**pass) == PassKind::Raw)
            .map(|pass| self.frame.bodies[pass.index()])
            .collect();

        // One query pair per timed pass, reserved here rather than as each pass
        // records: taking a pair mutates the frame's slot, and a pass may be
        // recording on a worker.
        let tokens: Vec<Option<timestamps::PassToken>> = match timestamps.as_mut() {
            Some(timestamps) => self
                .frame
                .graph
                .order()
                .iter()
                .map(|pass| match self.frame.graph.pass_kind(*pass) {
                    PassKind::Raw => None,
                    _ => timestamps.reserve(self.frame.graph.pass_name(*pass)),
                })
                .collect(),
            None => vec![None; self.frame.graph.order().len()],
        };

        // The blended queues index through the host-written list whichever path
        // the opaque ones took, which is the same buffer on the CPU path and a
        // different one under GPU culling.
        let blended_sets = forward_sets.with_object_set(match &cull_frame {
            Some(_) => self
                .forward
                .build_object_set(&self.ctx, &object_rows, &objects.indices),
            None => forward_object_set.clone(),
        });

        let record = FrameRecord {
            ctx: &self.ctx,
            meshes: &self.meshes,
            arena: &self.arena,
            materials: &self.materials,
            frame: &self.frame,
            images: &self.images,
            swapchain_view: swapchain_view.clone(),
            shadow_resolution: self.config.shadow_resolution,
            extent,
            timestamps: timestamps.as_ref(),
            tokens: &tokens,
            bloom: &self.bloom,
            contact_shadows: &self.contact_shadows,
            dof: &self.dof,
            environment: &self.environment,
            exposure: &self.exposure,
            fog: &self.fog,
            forward: &self.forward,
            hdr: &self.hdr,
            line: &self.line,
            motion_blur: &self.motion_blur,
            oit: &self.oit,
            prepass: &self.prepass,
            refraction: &self.refraction,
            shadow: &self.shadow,
            ssao: &self.ssao,
            ssr: &self.ssr,
            subsurface: &self.subsurface,
            taa: &self.taa,
            view,
            env: environment,
            draws,
            transparent,
            refractive,
            shadows,
            objects,
            cull: &self.cull,
            object_rows: object_rows.clone(),
            cull_frame,
            cull_views: &cull_views,
            occlusion_test: self.occlusion.test(occlusion_live),
            occlusion: &self.occlusion,
            caster_sets,
            blended_sets,
            forward_sets,
            frame_uniforms,
            ssao_uniforms,
            contact_shadow_uniforms,
            prepass_object_set,
            prepass_material_set,
            prepass_texture_set,
            decal_block,
            scene_color,
            shadow_view,
            debug_lines,
        };

        // One command buffer per compiled segment, each bound for the queue the
        // schedule put it on. One segment — the whole frame on the graphics
        // queue — is what a device with a single queue always gets, and is
        // recorded and submitted exactly as it was before the split existed.
        let ranges: Vec<(GraphQueue, Range<usize>)> = self
            .frame
            .graph
            .segments()
            .iter()
            .map(|segment| (segment.queue, segment.passes.clone()))
            .collect();
        let overlay_pass = raw_passes.contains(&PassBody::Overlay);

        let mut recorded: Vec<(GraphQueue, CommandBuffer, record::KeepAlive)> =
            Vec::with_capacity(ranges.len());
        for (index, (queue, range)) in ranges.iter().enumerate() {
            if index > 0 {
                let (commands, keep) = builder.end();
                recorded.push((ranges[index - 1].0, commands, keep));
                builder = Recorder::for_family(&self.ctx, self.queue_family(*queue));
            }
            // Only the graphics head is worth spreading over the pool: the
            // compute tail is a handful of commands per dispatch, and a
            // secondary would have to come from the other family's pool to say
            // so. See `FrameRecord::cost`, which scores every dispatch zero.
            record.record_segment(&mut builder, range.clone(), *queue == GraphQueue::Graphics);
        }

        // Leave every import in the layout its owner expects — for the
        // swapchain image, the `PresentSrc` the presentation engine requires.
        // In the last segment, because that is where the frame ends.
        //
        // Emitted even when an overlay follows in its own command buffer:
        // vulkano gives a swapchain image a fixed layout requirement of
        // `PresentSrc` (`Image::from_swapchain`), so the overlay's
        // auto-synchronised buffer assumes it arrives in that layout. Leaving it
        // in `ColorAttachmentOptimal` makes that barrier's `oldLayout` a lie,
        // and the contents undefined. Transitioning here presents nothing — the
        // present happens later, on the overlay's own future.
        builder.barriers(self.frame.graph.final_barriers(), |id| {
            self.barrier_target(id, &swapchain_view)
        });

        if let Some(timestamps) = timestamps.as_mut() {
            timestamps.end_frame(&mut builder);
        }
        if timestamps.is_some() {
            self.timestamps = timestamps;
        }

        let (commands, keep) = builder.end();
        recorded.push((
            ranges
                .last()
                .expect("a compiled frame has at least one segment")
                .0,
            commands,
            keep,
        ));
        drop(recording);

        let submitting = crate::profile::scope("submit");

        // The one thing a frame's command buffers do not contain:
        // `Gui::draw_on_image` builds and submits its own, chained onto a
        // [`SemaphoreWait`] on the frame's completion, so the dependency stays
        // on the GPU.
        // Cloned out of `self` because the submission below takes `&mut self`
        // to present, and a closure holding a field borrow across that would
        // not compile.
        let device = self.ctx.device.clone();
        let semaphore = move || {
            Arc::new(
                Semaphore::from_pool(device.clone()).expect("failed to create a frame semaphore"),
            )
        };

        let mut recorded = recorded;
        if recorded.len() == 1 {
            // One queue, one submission, and the frame presented before this
            // call returns — what every device without a compute-only family
            // does, and what every device did before there was a split.
            let (_, command_buffer, keep) = recorded.pop().expect("checked non-empty");

            // One signal per waiter: a binary semaphore's signal may be waited
            // exactly once, so the present and the next frame cannot share one.
            // With an overlay, the present is the overlay's to make, so the
            // frame signals a semaphore for *it* to wait on instead.
            let render_finished =
                (self.swapchain.swapchain.is_some() && !overlay_pass).then(&semaphore);
            let overlay_ready = overlay_pass.then(&semaphore);
            let chain = semaphore();

            let mut waits = Vec::new();
            if let Some(acquired) = acquire_semaphore {
                // The acquired image is only written by the passes that render
                // into it, and the earliest of those is the tonemap's colour
                // write. Waiting at colour-attachment output rather than
                // top-of-pipe lets every shadow, prepass, lighting and post
                // dispatch in the frame run while the presentation engine still
                // owns the image.
                waits.push((
                    acquired,
                    ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                ));
            }
            if let Some(previous) = self.frame_chain.take() {
                // What the old `GpuFuture` chain did, and for the same reason:
                // one set of graph images means this frame writes what the last
                // one is still reading. The dependency is on the GPU, so the CPU
                // does not wait for it.
                waits.push((previous, ash::vk::PipelineStageFlags::ALL_COMMANDS));
            }

            let signals: Vec<_> = render_finished
                .clone()
                .into_iter()
                .chain(overlay_ready.clone())
                .chain(Some(chain.clone()))
                .collect();

            let in_flight = record::submit_frame(
                &self.ctx,
                &self.ctx.queue.clone(),
                command_buffer,
                keep,
                &waits,
                &signals,
            );
            self.frame_chain = Some(chain);
            self.in_flight.push(in_flight);
            self.finish_frame(
                overlay_ready,
                render_finished,
                image_index,
                &raw_passes,
                overlay,
            );
            drop(submitting);
            return;
        }

        // Split. Three submissions, in the one order that lets the compute tail
        // overlap anything: this frame's head first, so it is ahead of the
        // *previous* frame's trailing segment on the graphics queue and does not
        // inherit its wait; that trailing segment second; and this frame's
        // compute tail last, which is now free to run beside whatever the
        // graphics queue does next.
        let tail = recorded.pop().expect("a split frame has three segments");
        let compute = recorded.pop().expect("a split frame has three segments");
        let (_, head_commands, head_keep) =
            recorded.pop().expect("a split frame has three segments");

        // 1. The graphics head. It waits only for the frame that last used this
        //    frame's set of graph images to have finished with them — two frames
        //    back, whose trailing segment signalled this. Named explicitly
        //    rather than left to the queue's submission order, which Vulkan does
        //    not guarantee is execution order.
        let head_done = semaphore();
        let mut head_waits = Vec::new();
        if let Some(gate) = self.head_gate.take() {
            head_waits.push((gate, ash::vk::PipelineStageFlags::ALL_COMMANDS));
        }
        let mut in_flight = record::submit_frame(
            &self.ctx,
            &self.ctx.queue.clone(),
            head_commands,
            head_keep,
            &head_waits,
            std::slice::from_ref(&head_done),
        );

        // 2. The previous frame's trailing segment, behind this frame's head.
        //    Its compute tail has had a whole frame to run.
        let (previous_tail_done, next_head_gate) = match self.pending.take() {
            Some(pending) => {
                let render_finished =
                    (self.swapchain.swapchain.is_some() && !pending.overlay).then(&semaphore);
                let overlay_ready = pending.overlay.then(&semaphore);
                // Two, not one: a binary semaphore's signal may be waited
                // exactly once, and this segment has two waiters — this frame's
                // compute tail, and the head of the frame after it.
                let done = semaphore();
                let gate = semaphore();
                let mut waits = vec![(
                    pending.compute_done,
                    ash::vk::PipelineStageFlags::ALL_COMMANDS,
                )];
                if let Some(acquired) = pending.acquire {
                    waits.push((
                        acquired,
                        ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    ));
                }
                let signals: Vec<_> = render_finished
                    .clone()
                    .into_iter()
                    .chain(overlay_ready.clone())
                    .chain([done.clone(), gate.clone()])
                    .collect();
                in_flight = in_flight.and(record::submit_frame(
                    &self.ctx,
                    &self.ctx.queue.clone(),
                    pending.command_buffer,
                    pending.keep,
                    &waits,
                    &signals,
                ));
                // The overlay drawn over it is the one *this* call supplied, so
                // the editor's UI leads the scene under it by a frame. The cost
                // of holding the segment back, and the reason `PendingTail` says
                // so.
                self.finish_frame(
                    overlay_ready,
                    render_finished,
                    pending.image_index,
                    &raw_passes,
                    overlay,
                );
                (Some(done), Some(gate))
            }
            None => (None, None),
        };

        // 3. The compute tail. Behind this frame's head, and behind the previous
        //    frame's trailing segment — which is not the graphics queue leaking
        //    back in, but the one resource the two really do share: the exposure
        //    the metering dispatch overwrites is the exposure the previous
        //    frame's tonemap reads. It costs nothing, because that segment runs
        //    immediately after the compute work this one already follows.
        let compute_done = semaphore();
        let mut compute_waits = vec![(head_done, ash::vk::PipelineStageFlags::ALL_COMMANDS)];
        if let Some(previous) = previous_tail_done {
            compute_waits.push((previous, ash::vk::PipelineStageFlags::ALL_COMMANDS));
        }
        let (_, compute_commands, compute_keep) = compute;
        in_flight = in_flight.and(record::submit_frame(
            &self.ctx,
            &self.queue_of(GraphQueue::AsyncCompute).clone(),
            compute_commands,
            compute_keep,
            &compute_waits,
            std::slice::from_ref(&compute_done),
        ));
        self.in_flight.push(in_flight);

        // What the *next* frame's head waits on: this call submitted the frame
        // two back's last work, so its completion is when this frame's images
        // become the next-but-one frame's to write.
        self.head_gate = next_head_gate;

        let (_, tail_commands, tail_keep) = tail;
        self.pending = Some(PendingTail {
            command_buffer: tail_commands,
            keep: tail_keep,
            compute_done,
            acquire: acquire_semaphore,
            image_index,
            overlay: overlay_pass,
        });

        drop(submitting);
    }
}

/// Everything one frame records from, in a shape a worker can hold.
///
/// The passes' own state, the compiled schedule, and what this frame resolved
/// before recording opened: the jittered view, the draw orders, the descriptor
/// sets built once and bound by several passes. All of it shared — the pass
/// bodies record through `&self`, which is what lets a run of them go to a
/// worker while the main thread records another.
///
/// Not a `&VulkanRenderer` for the reason [`PassCtx`] is not: the renderer owns
/// the frames in flight and the overlay's futures, neither of which may cross a
/// thread. What is here is the subset that may.
struct FrameRecord<'a> {
    ctx: &'a VkContext,
    meshes: &'a [GpuMesh],
    arena: &'a MeshArena,
    materials: &'a [GpuMaterial],
    frame: &'a Frame,
    images: &'a GraphImages,
    swapchain_view: Arc<ImageView>,
    shadow_resolution: u32,
    extent: [u32; 2],
    /// `None` when the device cannot time passes, or when profiling is off. The
    /// pairs in `tokens` were reserved before any of this recorded.
    timestamps: Option<&'a GpuTimestamps>,
    /// One entry per slot in the compiled order, `None` for a pass this frame
    /// does not time.
    tokens: &'a [Option<timestamps::PassToken>],
    bloom: &'a BloomPass,
    contact_shadows: &'a ContactShadowPass,
    dof: &'a DofPass,
    environment: &'a EnvironmentPass,
    exposure: &'a ExposurePass,
    fog: &'a FogPass,
    forward: &'a ForwardPass,
    hdr: &'a HdrPass,
    line: &'a LinePass,
    motion_blur: &'a MotionBlurPass,
    oit: &'a OitPass,
    prepass: &'a GeometryPrepass,
    refraction: &'a RefractionPass,
    shadow: &'a ShadowPass,
    ssao: &'a SsaoPass,
    ssr: &'a SsrPass,
    subsurface: &'a SubsurfacePass,
    taa: &'a TaaPass,
    view: taa::FrameView,
    /// The environment *settings*; `environment` above is the pass that draws
    /// them.
    env: &'a EnvironmentSettings,
    draws: DrawList<'a>,
    transparent: DrawList<'a>,
    refractive: DrawList<'a>,
    shadows: Option<ShadowFrame<'a>>,
    objects: instances::InstanceLists,
    cull: &'a CullPass,
    /// The persistent object rows, which the cull dispatch reads for the model
    /// matrix that turns a batch's object-space box into a world one.
    object_rows: vulkano::buffer::Subbuffer<[instances::GpuObject]>,
    /// The batch table this frame was prepared with, and `None` on the CPU path.
    /// Every opaque geometry pass reads it to decide whether it draws runs or
    /// indirect commands, so the two cannot half-switch.
    cull_frame: Option<cull::CullFrame>,
    cull_views: &'a [CullView],
    /// The previous frame's pyramid, and whether the cull may conclude anything
    /// from it. Assembled here rather than reached for through `self` because
    /// what the cull binds when the frame does not test is a fallback the pass
    /// owns — see [`occlusion::OcclusionTest`].
    occlusion_test: occlusion::OcclusionTest,
    occlusion: &'a OcclusionPass,
    caster_sets: Option<shadow::CasterSets>,
    forward_sets: forward::ForwardSets,
    /// The same sets with the blended queues' own instance buffer bound. Equal
    /// to `forward_sets` on the CPU path, where there is only one such buffer.
    blended_sets: forward::ForwardSets,
    frame_uniforms: Option<vulkano::buffer::Subbuffer<prepass::FrameUbo>>,
    ssao_uniforms: Option<ssao::SsaoUniforms>,
    contact_shadow_uniforms: Option<contact_shadows::ContactShadowUniforms>,
    prepass_object_set: Option<Arc<DescriptorSet>>,
    prepass_material_set: Arc<DescriptorSet>,
    prepass_texture_set: Arc<DescriptorSet>,
    decal_block: vulkano::buffer::Subbuffer<forward::GpuDecals>,
    scene_color: Arc<ImageView>,
    shadow_view: Arc<ImageView>,
    debug_lines: &'a [DebugLine],
}

/// A frame's trailing graphics segment, held back a frame on purpose.
///
/// The tonemap is a draw, so it belongs on the graphics queue, and it is the
/// last thing the frame does. A queue is consumed in order, so submitting it in
/// its own frame would put a wait for the compute tail *in front of* the next
/// frame's head — and that head is the work the split exists to run early. Held
/// here instead and submitted at the start of the next frame, behind that head,
/// which is the one order in which both can be true.
///
/// What it costs is a frame of latency, and one consequence worth knowing: the
/// overlay drawn over this image is the one the *next* `render` call supplied,
/// so the editor's UI leads the scene under it by a frame.
struct PendingTail {
    command_buffer: vulkano::command_buffer::CommandBuffer,
    keep: record::KeepAlive,
    /// Signalled by the compute segment this waits on.
    compute_done: Arc<Semaphore>,
    /// The acquire for the image the tonemap writes, waited at colour-attachment
    /// output for the reason the unsplit path waits it there.
    acquire: Option<Arc<Semaphore>>,
    image_index: u32,
    /// Whether a raw pass draws over the result before it is presented.
    overlay: bool,
}

/// Whether the frame decides its visible set on the GPU.
///
/// Off unless asked for, which is the opposite of how `ORRIN_ASYNC_COMPUTE`
/// reads and deliberately so: the CPU sweep is still the path every scene has
/// been looked at through, and this is the one being measured against it.
///
/// A device that cannot multi-draw keeps the sweep whatever the variable says,
/// and says so rather than drawing one batch per view and calling it the same
/// thing: the path exists to be measured, and a silently different one is not a
/// measurement. See [`VkContext::multi_draw`].
fn read_gpu_culling(ctx: &VkContext) -> bool {
    let asked = std::env::var("ORRIN_GPU_CULL").is_ok_and(|value| value.trim() == "1");
    if asked && !ctx.multi_draw {
        tracing::warn!(
            "GPU culling: unavailable — this device has no multi-draw indirect \
             with a per-draw first instance; keeping the CPU sweep"
        );
    }
    asked && ctx.multi_draw
}

/// Whether the cull also drops what the previous frame's depth already covered.
///
/// A second variable rather than a level of the first, because the two answer
/// different questions: `ORRIN_GPU_CULL` asks whether the visible set is decided
/// on the GPU at all, and this asks whether that decision is allowed to be
/// *smaller* than the frustum's. It has no effect without the first, which is
/// where it is anded.
fn read_occlusion_culling() -> bool {
    std::env::var("ORRIN_OCCLUSION").is_ok_and(|value| value.trim() == "1")
}

/// How many sets of the graph's transient images the compiled frame needs.
///
/// One, unless the frame was split — with the split, this frame's compute tail
/// is still running when the next frame's graphics head starts recording into
/// the same declarations, which is the whole overlap and a race on one set.
fn image_slots(graph: &crate::gfx::graph::FrameGraph) -> usize {
    if graph.segments().len() > 1 {
        FRAMES_IN_FLIGHT
    } else {
        1
    }
}

/// How many command buffers one frame's recording is worth splitting into.
///
/// The partition has little left to give beyond this: the punctual shadow atlas
/// alone is around 40% of what a frame costs to record, so it is the floor on
/// the critical path however many workers there are, and every extra group
/// costs a secondary command buffer to begin, end and execute.
const RECORD_GROUPS: usize = 3;

impl FrameRecord<'_> {
    /// Record one of the compiled frame's segments into `builder`, in the order
    /// the compiler derived.
    ///
    /// Serially, or — where the pool has threads to spare — into one secondary
    /// command buffer per contiguous run of passes, executed by `builder` in
    /// that same order. The GPU is handed the same stream either way: a run
    /// carries the barriers the plan puts in front of its own passes, and
    /// nothing about a pass's recording depends on which buffer it lands in.
    ///
    /// A segment rather than the whole frame because a split frame is several
    /// command buffers submitted to different queues — see
    /// `gfx/graph/schedule.rs`. The partition is per segment for the same
    /// reason it is contiguous: the runs are executed in order by one primary,
    /// and there is one primary per segment.
    fn record_segment(&self, builder: &mut Recorder, segment: Range<usize>, parallel: bool) {
        let slots = segment.len();
        let groups = threads::count().min(RECORD_GROUPS);
        if !parallel || groups < 2 || slots < 2 {
            for slot in segment {
                self.record_slot(builder, slot);
            }
            return;
        }

        let costs: Vec<u32> = segment.clone().map(|slot| self.cost(slot)).collect();
        let runs = parallel::partition(&costs, groups);
        let recorded = threads::map(&runs, |run| {
            let mut secondary = Recorder::secondary(self.ctx);
            for offset in run.clone() {
                self.record_slot(&mut secondary, segment.start + offset);
            }
            secondary.end()
        });
        for (commands, keep) in recorded {
            builder.execute(commands, keep);
        }
    }

    /// What the pass in `slot` costs to record, near enough to balance the runs
    /// against each other: the draws it will emit.
    ///
    /// Everything not drawing geometry is a handful of commands whatever it
    /// dispatches over, and scores zero — which
    /// [`partition`](parallel::partition) floors at one, so a run of them still
    /// spreads.
    fn cost(&self, slot: usize) -> u32 {
        let pass_id = self.frame.graph.order()[slot];
        match self.frame.bodies[pass_id.index()] {
            PassBody::ShadowCascade(cascade) => self
                .shadows
                .map_or(0, |shadows| shadows.casters[cascade as usize].len() as u32),
            // Each caster is drawn once per face of its light, which is what
            // makes this one pass the frame's most expensive to record.
            PassBody::PunctualShadows => self.shadows.map_or(0, |shadows| {
                shadows
                    .atlas
                    .casters
                    .iter()
                    .zip(shadows.punctual_casters)
                    .map(|(caster, list)| list.len() as u32 * caster.face_count as u32)
                    .sum()
            }),
            PassBody::GeometryPrepass | PassBody::Forward => self.draws.len() as u32,
            PassBody::OitAccumulate => self.transparent.len() as u32,
            PassBody::RefractionDraw => self.refractive.len() as u32,
            _ => 0,
        }
    }

    /// What a pass body gets of the renderer besides its own state.
    /// What an opaque geometry pass draws for `view`: the batches the dispatch
    /// culled, or the ordered list the sweep handed it.
    ///
    /// The view numbering is `cull.rs`'s and is stated once here — the camera,
    /// then each cascade, then each punctual face — because the dispatch wrote
    /// its blocks in that order and a pass reading the wrong one draws another
    /// view's visible set without anything failing.
    fn opaque_draws<'b>(&'b self, view: u32, list: DrawList<'b>, base: u32) -> Draws<'b> {
        match &self.cull_frame {
            Some(frame) => Draws::Gpu {
                frame,
                view,
                commands: self.cull.commands(),
            },
            None => Draws::Cpu { list, base },
        }
    }

    /// Where the punctual faces begin in that numbering.
    fn first_face_view(&self) -> u32 {
        1 + self
            .shadows
            .map_or(0, |shadows| shadows.cascades.count as u32)
    }

    fn pass_ctx(&self) -> PassCtx<'_> {
        PassCtx {
            ctx: self.ctx,
            meshes: self.meshes,
            arena: self.arena,
            materials: self.materials,
        }
    }

    /// The view backing a graph resource, whoever owns the allocation.
    ///
    /// Every image in a frame is graph-owned but two: the TAA resolve's output
    /// and the fog's scattering volume are *imported*, because a history has to
    /// survive a frame boundary and a transient by contract does not. Anything
    /// downstream that reads the frame's colour can be handed either, depending
    /// on which optical stages the frame has — so it asks by `ResourceId` and
    /// this decides, rather than each consumer re-deriving which pass ran last.
    fn view_of(&self, id: ResourceId) -> Arc<ImageView> {
        if let Some(fog) = self.frame.ids.fog
            && id == fog.scatter
        {
            return self.fog.scatter_view();
        }
        match self.frame.ids.taa {
            Some(taa) if id == taa.output => self.taa.output_view(),
            _ => self.images.view(id),
        }
    }

    /// Which Vulkan object one resource in the compiled plan names.
    ///
    /// The graph is device-free, so it tracks `ResourceId`s and leaves this to
    /// the renderer: graph-owned images, images imported from the pass that owns
    /// them across a frame boundary, the swapchain image, and buffers.
    fn barrier_target(&self, id: ResourceId) -> Option<record::Target> {
        let ids = &self.frame.ids;

        if id == ids.swapchain_color {
            return Some(record::Target::Image(self.swapchain_view.image().clone()));
        }
        // The imported buffers. See `record::Target::Memory` for why these do
        // not need resolving to their `Subbuffer`s.
        if id == ids.object_transforms
            || id == ids.instance_index
            || id == ids.exposure
            || Some(id) == ids.histogram
        {
            return Some(record::Target::Memory);
        }
        // Imported images: a history has to survive a frame boundary, and a
        // transient by contract does not, so the pass that ping-pongs it owns
        // the allocation.
        if let Some(fog) = ids.fog
            && id == fog.scatter
        {
            return Some(record::Target::Image(
                self.fog.scatter_view().image().clone(),
            ));
        }
        if let Some(occlusion) = ids.occlusion {
            if id == occlusion.hiz {
                return Some(record::Target::Image(
                    self.occlusion.hiz_view().image().clone(),
                ));
            }
            if id == occlusion.history {
                return Some(record::Target::Image(
                    self.occlusion.history_view().image().clone(),
                ));
            }
        }
        if let Some(taa) = ids.taa {
            if id == taa.output {
                return Some(record::Target::Image(
                    self.taa.output_view().image().clone(),
                ));
            }
            if id == taa.history {
                return Some(record::Target::Image(
                    self.taa.history_view().image().clone(),
                ));
            }
        }

        self.images
            .try_view(id)
            .map(|view| record::Target::Image(view.image().clone()))
    }

    /// Record the pass the compiled order puts in `slot`, and the barriers the
    /// plan puts in front of it.
    fn record_slot(&self, builder: &mut Recorder, slot: usize) {
        let pass_id = self.frame.graph.order()[slot];
        let body = self.frame.bodies[pass_id.index()];

        // What this pass needs finished, and the layouts it needs — derived
        // by `gfx/graph/` from what the passes declared, rather than
        // inferred from what the commands below happen to touch.
        builder.barriers(self.frame.graph.barriers_before(slot), |id| {
            self.barrier_target(id)
        });

        let kind = self.frame.graph.pass_kind(pass_id);
        if kind == PassKind::Raw {
            // Escape-hatch passes own their submission, so they run on the
            // future after this command buffer rather than inside it —
            // `raw_passes` collected them before any of this recorded.
            // `compile` has already established that none of them precedes
            // an inline pass.
            return;
        }

        let timed = self.tokens[slot];
        if let Some(timestamps) = self.timestamps {
            timestamps.open(builder, timed);
        }

        // A dispatch is illegal inside a render pass, so a compute pass is
        // recorded into the same command buffer with no bracket around it.
        // That is the only thing the kind changes: ordering and barriers are
        // derived for it exactly as for a draw.
        if kind == PassKind::Compute {
            match body {
                PassBody::SsrHiz => {
                    let ids = self
                        .frame
                        .ids
                        .ssr
                        .expect("reflections without their images");
                    let prepass = self
                        .frame
                        .ids
                        .prepass
                        .expect("the graph scheduled reflections with no prepass");
                    // One self.view per level, because a storage image descriptor
                    // takes exactly one — the sampled self.view the trace reads
                    // spans the whole pyramid instead.
                    let mips: Vec<_> = (0..self.images.mip_levels(ids.hiz))
                        .map(|level| self.images.mip_view(ids.hiz, level))
                        .collect();
                    self.ssr.record_hiz(
                        builder,
                        self.ctx,
                        self.images.view(prepass.depth),
                        &mips,
                        self.extent,
                    );
                }
                PassBody::SsrSource => {
                    let ids = self
                        .frame
                        .ids
                        .ssr
                        .expect("reflections without their images");
                    let mips: Vec<_> = (0..self.images.mip_levels(ids.source_pyramid))
                        .map(|level| self.images.mip_view(ids.source_pyramid, level))
                        .collect();
                    self.ssr
                        .record_source(builder, self.ctx, self.view_of(ids.source), &mips);
                }
                PassBody::SsrTrace => {
                    let ids = self
                        .frame
                        .ids
                        .ssr
                        .expect("reflections without their images");
                    let prepass = self
                        .frame
                        .ids
                        .prepass
                        .expect("the graph scheduled reflections with no prepass");
                    self.ssr.record_trace(
                        builder,
                        self.ctx,
                        self.images.view(ids.hiz),
                        self.images.view(prepass.depth),
                        self.images.view(prepass.normal),
                        self.images.view(prepass.material),
                        self.images.view(ids.source_pyramid),
                        self.images.view(ids.rays),
                    );
                }
                PassBody::SsrResolve => {
                    let ids = self
                        .frame
                        .ids
                        .ssr
                        .expect("reflections without their images");
                    let prepass = self
                        .frame
                        .ids
                        .prepass
                        .expect("the graph scheduled reflections with no prepass");
                    self.ssr.record_resolve(
                        builder,
                        self.ctx,
                        self.view_of(ids.source),
                        self.images.view(ids.rays),
                        self.images.view(prepass.depth),
                        self.images.view(prepass.normal),
                        self.images.view(prepass.material),
                        self.environment.specular_view(),
                        self.environment.sampler(),
                        self.images.view(ids.output),
                    );
                }
                PassBody::FogScatter => {
                    self.fog
                        .record_scatter(builder, self.ctx, self.shadow_view.clone());
                }
                PassBody::FogIntegrate => {
                    let ids = self
                        .frame
                        .ids
                        .fog
                        .expect("fog passes without their volumes");
                    self.fog
                        .record_integrate(builder, self.ctx, self.images.view(ids.volume));
                }
                PassBody::SubsurfaceBlurHorizontal | PassBody::SubsurfaceBlurVertical => {
                    let ids = self
                        .frame
                        .ids
                        .subsurface
                        .expect("subsurface diffusion without its targets");
                    let prepass = self
                        .frame
                        .ids
                        .prepass
                        .expect("the graph scheduled the diffusion with no prepass");
                    let vertical = matches!(body, PassBody::SubsurfaceBlurVertical);
                    // Mirrors what `declare` said each axis reads: the target
                    // the forward pass resolved, then the other axis's output.
                    let (source, target) = if vertical {
                        (ids.blurred_x, ids.blurred_y)
                    } else {
                        (ids.diffusible, ids.blurred_x)
                    };
                    self.subsurface.record_blur(
                        builder,
                        self.ctx,
                        self.images.view(source),
                        self.images.view(prepass.depth),
                        self.images.view(target),
                        vertical,
                    );
                }
                PassBody::SubsurfaceComposite => {
                    let ids = self
                        .frame
                        .ids
                        .subsurface
                        .expect("subsurface diffusion without its targets");
                    self.subsurface.record_composite(
                        builder,
                        self.ctx,
                        self.view_of(ids.source),
                        self.images.view(ids.blurred_y),
                        self.images.view(ids.output),
                    );
                }
                PassBody::OitComposite => {
                    let ids = self
                        .frame
                        .ids
                        .transparency
                        .expect("transparency without its targets");
                    self.oit.record_composite(
                        builder,
                        self.ctx,
                        self.view_of(ids.source),
                        self.images.view(ids.accum),
                        self.images.view(ids.reveal),
                        self.images.view(ids.output),
                    );
                }
                PassBody::RefractionScene => {
                    let ids = self
                        .frame
                        .ids
                        .refraction
                        .expect("refraction without its images");
                    let mips: Vec<_> = (0..self.images.mip_levels(ids.scene))
                        .map(|level| self.images.mip_view(ids.scene, level))
                        .collect();
                    self.refraction.record_pyramid(
                        builder,
                        self.ctx,
                        self.view_of(ids.source),
                        &mips,
                    );
                }
                PassBody::RefractionComposite => {
                    let ids = self
                        .frame
                        .ids
                        .refraction
                        .expect("refraction without its images");
                    self.refraction.record_composite(
                        builder,
                        self.ctx,
                        self.view_of(ids.source),
                        self.images.view(ids.accum),
                        self.images.view(ids.output),
                    );
                }
                PassBody::TaaResolve => {
                    let ids = self
                        .frame
                        .ids
                        .prepass
                        .expect("the graph scheduled TAA with no prepass");
                    let taa = self.frame.ids.taa.expect("TAA without its images");
                    self.taa.record(
                        builder,
                        self.ctx,
                        &self.view,
                        self.view_of(taa.source),
                        self.images.view(ids.velocity),
                        self.images.view(ids.depth),
                    );
                }
                PassBody::DofPrefilter => {
                    let ids = self
                        .frame
                        .ids
                        .dof
                        .expect("depth of field without its images");
                    let prepass = self
                        .frame
                        .ids
                        .prepass
                        .expect("the graph scheduled depth of field with no prepass");
                    self.dof.record_prefilter(
                        builder,
                        self.ctx,
                        self.view_of(ids.source),
                        self.images.view(prepass.depth),
                        self.images.view(ids.prefiltered),
                    );
                }
                PassBody::DofTileMax => {
                    let ids = self
                        .frame
                        .ids
                        .dof
                        .expect("depth of field without its images");
                    self.dof.record_tile_max(
                        builder,
                        self.ctx,
                        self.images.view(ids.prefiltered),
                        self.images.view(ids.tile),
                    );
                }
                PassBody::DofGather => {
                    let ids = self
                        .frame
                        .ids
                        .dof
                        .expect("depth of field without its images");
                    self.dof.record_gather(
                        builder,
                        self.ctx,
                        self.images.view(ids.prefiltered),
                        self.images.view(ids.tile),
                        self.images.view(ids.near),
                        self.images.view(ids.far),
                    );
                }
                PassBody::DofComposite => {
                    let ids = self
                        .frame
                        .ids
                        .dof
                        .expect("depth of field without its images");
                    let prepass = self
                        .frame
                        .ids
                        .prepass
                        .expect("the graph scheduled depth of field with no prepass");
                    self.dof.record_composite(
                        builder,
                        self.ctx,
                        self.view_of(ids.source),
                        self.images.view(prepass.depth),
                        self.images.view(ids.near),
                        self.images.view(ids.far),
                        self.images.view(ids.output),
                    );
                }
                PassBody::MotionBlurTileMax => {
                    let ids = self
                        .frame
                        .ids
                        .motion_blur
                        .expect("motion blur without its images");
                    let prepass = self
                        .frame
                        .ids
                        .prepass
                        .expect("the graph scheduled motion blur with no prepass");
                    self.motion_blur.record_tile_max(
                        builder,
                        self.ctx,
                        &self.view,
                        self.images.view(prepass.velocity),
                        self.images.view(prepass.depth),
                        self.images.view(ids.tile),
                    );
                }
                PassBody::MotionBlurNeighbourMax => {
                    let ids = self
                        .frame
                        .ids
                        .motion_blur
                        .expect("motion blur without its images");
                    self.motion_blur.record_neighbour_max(
                        builder,
                        self.ctx,
                        self.images.view(ids.tile),
                        self.images.view(ids.neighbour),
                    );
                }
                PassBody::MotionBlurGather => {
                    let ids = self
                        .frame
                        .ids
                        .motion_blur
                        .expect("motion blur without its images");
                    let prepass = self
                        .frame
                        .ids
                        .prepass
                        .expect("the graph scheduled motion blur with no prepass");
                    self.motion_blur.record_gather(
                        builder,
                        self.ctx,
                        &self.view,
                        self.view_of(ids.source),
                        self.images.view(prepass.velocity),
                        self.images.view(prepass.depth),
                        self.images.view(ids.neighbour),
                        self.images.view(ids.output),
                    );
                }
                PassBody::CullReset => {
                    let frame = self
                        .cull_frame
                        .as_ref()
                        .expect("the graph scheduled the cull with no batch table");
                    self.cull.record_reset(builder, self.ctx, frame);
                }
                PassBody::Cull => {
                    let frame = self
                        .cull_frame
                        .as_ref()
                        .expect("the graph scheduled the cull with no batch table");
                    self.cull.record_cull(
                        builder,
                        self.ctx,
                        &self.object_rows,
                        frame,
                        self.cull_views,
                        &self.occlusion_test,
                        self.view.prev_view_proj,
                    );
                }
                PassBody::OcclusionHiz => {
                    let prepass = self
                        .frame
                        .ids
                        .prepass
                        .expect("the graph scheduled the occlusion pyramid with no prepass");
                    self.occlusion.record(
                        builder,
                        self.ctx,
                        self.images.view(prepass.depth),
                        self.extent,
                    );
                }
                PassBody::LuminanceHistogram => self.exposure.record_histogram(
                    builder,
                    self.ctx,
                    self.extent,
                    self.scene_color.clone(),
                ),
                PassBody::LuminanceAverage => {
                    self.exposure.record_average(builder, self.ctx, self.extent)
                }
                PassBody::BloomPrefilter => {
                    let ids = self.frame.ids.bloom.expect("bloom pass without levels");
                    self.bloom.record_prefilter(
                        builder,
                        self.ctx,
                        self.scene_color.clone(),
                        self.images.view(ids.down(0)),
                        self.exposure.exposure_buffer(),
                    );
                }
                PassBody::BloomDownsample(level) => {
                    let ids = self.frame.ids.bloom.expect("bloom pass without levels");
                    let level = level as usize;
                    self.bloom.record_downsample(
                        builder,
                        self.ctx,
                        self.images.view(ids.down(level - 1)),
                        self.images.view(ids.down(level)),
                    );
                }
                PassBody::BloomUpsample(level) => {
                    let ids = self.frame.ids.bloom.expect("bloom pass without levels");
                    let level = level as usize;
                    // Mirrors what `declare` said this pass reads: an
                    // up-chain level where there is one above, and the down
                    // chain's last level at the top of the climb.
                    let coarse = if level + 2 < ids.mips as usize {
                        ids.up(level + 1)
                    } else {
                        ids.down(level + 1)
                    };
                    self.bloom.record_upsample(
                        builder,
                        self.ctx,
                        self.images.view(coarse),
                        self.images.view(ids.down(level)),
                        self.images.view(ids.up(level)),
                    );
                }
                other => unreachable!("{other:?} is not a compute pass"),
            }
            if let Some(timestamps) = self.timestamps {
                timestamps.close(builder, timed);
            }
            return;
        }

        // Attachments, load and store ops and clears together, decided
        // here rather than split between a framebuffer built at allocation
        // time and a positional clear list that had to match its order.
        let rendering = rendering::rendering_info(
            &self.frame.ids,
            self.images,
            &self.swapchain_view,
            body,
            self.env.background,
        )
        .expect("the executor reached a graphics pass with nothing to render into");
        builder.begin_rendering(rendering);

        match body {
            PassBody::ShadowCascade(cascade) => {
                let shadows = self
                    .shadows
                    .expect("the graph scheduled a cascade with no shadows");
                let cascade_index = cascade as usize;
                self.shadow.record(
                    builder,
                    &self.pass_ctx(),
                    self.opaque_draws(
                        1 + cascade,
                        shadows.casters[cascade_index],
                        self.objects.cascade_bases[cascade_index],
                    ),
                    shadows.cascades.cascades[cascade_index].view_proj,
                    self.shadow_resolution,
                    self.caster_sets
                        .as_ref()
                        .expect("the graph scheduled a cascade with no shadows"),
                );
            }
            PassBody::PunctualShadows => {
                let shadows = self
                    .shadows
                    .expect("the graph scheduled the atlas with no shadow frame");
                let gpu = self
                    .cull_frame
                    .as_ref()
                    .map(|frame| (frame, self.cull.commands(), self.first_face_view()));
                self.shadow.record_atlas(
                    builder,
                    &self.pass_ctx(),
                    shadows.atlas,
                    shadows.punctual_casters,
                    &self.objects.punctual_bases,
                    gpu,
                    self.caster_sets
                        .as_ref()
                        .expect("the graph scheduled the atlas with no shadow frame"),
                );
            }
            PassBody::GeometryPrepass => self.prepass.record(
                builder,
                &self.pass_ctx(),
                self.opaque_draws(0, self.draws, 0),
                self.extent,
                self.frame_uniforms.clone().unwrap(),
                self.decal_block.clone(),
                self.prepass_object_set.clone().unwrap(),
                self.prepass_material_set.clone(),
                self.prepass_texture_set.clone(),
            ),
            // Both AO passes take their viewport from the target they draw
            // into rather than from the frame, because that target is the
            // one thing here that is not always the frame's size. They still
            // sample the prepass at full res: the resolve reads depth and
            // normals by UV, and picking one of four texels is what makes
            // the half-res version cheaper.
            PassBody::SsaoResolve => {
                let prepass = self.frame.ids.prepass.unwrap();
                let ids = self.frame.ids.ssao.unwrap();
                self.ssao.record_ao(
                    builder,
                    &self.pass_ctx(),
                    self.images.extent(ids.raw_ao),
                    self.ssao_uniforms.as_ref().unwrap(),
                    self.images.view(prepass.depth),
                    self.images.view(prepass.normal),
                );
            }
            PassBody::SsaoBlur => {
                let ids = self.frame.ids.ssao.unwrap();
                let prepass = self.frame.ids.prepass.unwrap();
                self.ssao.record_blur(
                    builder,
                    &self.pass_ctx(),
                    self.images.extent(ids.ao),
                    self.images.view(ids.raw_ao),
                    self.images.view(prepass.depth),
                    self.ssao_uniforms.as_ref().unwrap(),
                );
            }
            PassBody::ContactShadows => {
                let ids = self
                    .frame
                    .ids
                    .prepass
                    .expect("the graph scheduled contact shadows with no prepass");
                self.contact_shadows.record(
                    builder,
                    &self.pass_ctx(),
                    self.extent,
                    self.contact_shadow_uniforms
                        .as_ref()
                        .expect("contact shadows without their uniforms"),
                    self.images.view(ids.depth),
                    self.images.view(ids.normal),
                );
            }
            PassBody::Forward => {
                // One question, asked once: the graph decided which render
                // pass this frame opens, so the pipeline every draw inside it
                // binds follows from the same answer.
                let subsurface = self.frame.ids.subsurface.is_some();
                let msaa = self.frame.ids.msaa.is_some();
                self.forward.draw(
                    builder,
                    &self.pass_ctx(),
                    self.opaque_draws(0, self.draws, 0),
                    &self.view,
                    self.extent,
                    &self.forward_sets,
                    subsurface,
                    msaa,
                );
                // Between the geometry and the lines, and it has to be:
                // after the geometry so the depth test rejects the sky
                // wherever something was drawn, and before the lines
                // because the sky passes its own test at the depth clear
                // and would otherwise paint over them.
                self.environment.record_skybox(
                    builder,
                    self.ctx,
                    &self.view,
                    self.extent,
                    self.env,
                    subsurface,
                    msaa,
                    self.fog.uniforms(),
                    self.fog.volume_or_fallback(
                        self.frame.ids.fog.map(|ids| self.images.view(ids.volume)),
                    ),
                    self.fog.sampler(),
                );
                // Debug lines share the forward subpass: depth-tested against
                // the scene, drawn on top of it, before the pass ends.
                self.line.record(
                    builder,
                    self.debug_lines,
                    &self.view,
                    self.extent,
                    subsurface,
                    msaa,
                );
            }
            PassBody::OitAccumulate => self.oit.record(
                builder,
                &self.pass_ctx(),
                self.transparent,
                &self.blended_sets,
                &self.view,
                self.extent,
                self.objects.transparent_base,
            ),
            PassBody::RefractionDraw => {
                let ids = self
                    .frame
                    .ids
                    .refraction
                    .expect("refraction without its images");
                self.refraction.record(
                    builder,
                    &self.pass_ctx(),
                    self.refractive,
                    &self.blended_sets,
                    self.images.view(ids.scene),
                    self.view_of(ids.source),
                    &self.view,
                    self.extent,
                    self.objects.refractive_base,
                );
            }
            PassBody::Tonemap => self.hdr.record_tonemap(
                builder,
                self.ctx,
                self.extent,
                self.scene_color.clone(),
                self.exposure.exposure_buffer(),
                // With bloom off the graph has no chain, so the tonemap
                // pass samples a 1x1 black self.view at a zero strength: "no
                // bloom" with no second shader path.
                match self.frame.ids.bloom {
                    Some(ids) => self.images.view(ids.result()),
                    None => self.bloom.black_view(),
                },
            ),
            PassBody::Overlay
            | PassBody::SsrHiz
            | PassBody::SsrSource
            | PassBody::SsrTrace
            | PassBody::SsrResolve
            | PassBody::FogScatter
            | PassBody::FogIntegrate
            | PassBody::SubsurfaceBlurHorizontal
            | PassBody::SubsurfaceBlurVertical
            | PassBody::SubsurfaceComposite
            | PassBody::OitComposite
            | PassBody::RefractionScene
            | PassBody::RefractionComposite
            | PassBody::TaaResolve
            | PassBody::DofPrefilter
            | PassBody::DofTileMax
            | PassBody::DofGather
            | PassBody::DofComposite
            | PassBody::MotionBlurTileMax
            | PassBody::MotionBlurNeighbourMax
            | PassBody::MotionBlurGather
            | PassBody::LuminanceHistogram
            | PassBody::LuminanceAverage
            | PassBody::BloomPrefilter
            | PassBody::BloomDownsample(_)
            | PassBody::BloomUpsample(_)
            | PassBody::CullReset
            | PassBody::Cull
            | PassBody::OcclusionHiz => unreachable!("handled above"),
        }

        builder.end_rendering();
        if let Some(timestamps) = self.timestamps {
            timestamps.close(builder, timed);
        }
    }
}

fn swapchain_color_format(ctx: &VkContext, surface: &Arc<Surface>) -> vulkano::format::Format {
    use vulkano::format::Format;
    use vulkano::swapchain::ColorSpace;
    ctx.device
        .physical_device()
        .surface_formats(surface, Default::default())
        .unwrap()
        .into_iter()
        .find(|(f, c)| {
            matches!(f, Format::B8G8R8A8_SRGB | Format::R8G8B8A8_SRGB)
                && *c == ColorSpace::SrgbNonLinear
        })
        .map(|(f, _)| f)
        .unwrap_or(Format::B8G8R8A8_SRGB)
}
