mod context;
mod environment;
mod forward;
pub mod frame;
mod hdr;
mod line;
mod resources;
mod shadow;
mod ssao;
mod swapchain;
mod texture;
mod timestamps;

use std::sync::Arc;

use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassBeginInfo,
    SubpassContents,
};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::instance::Instance;
use vulkano::swapchain::{Surface, SwapchainPresentInfo, acquire_next_image};
use vulkano::sync::GpuFuture;
use vulkano::sync::{self, future::FenceSignalFuture};
use vulkano::{Validated, VulkanError};

use crate::geom::Aabb;
use crate::gfx::shadows::CascadeSet;
use crate::scene::{
    Camera, CpuMesh, EnvironmentSettings, HdrSettings, MaterialHandle, MeshHandle, ShadowSettings,
    SsaoSettings,
};

use self::context::VkContext;
use self::environment::EnvironmentPass;
use self::forward::{ForwardPass, GpuMaterial, GpuMesh};
use crate::gfx::graph::PassKind;

use self::frame::{Frame, FrameConfig, PassBody};
use self::hdr::HdrPass;
use self::line::LinePass;
use self::resources::{GraphImages, PassFramebuffers, begin_info};
use self::shadow::ShadowPass;
use self::ssao::SsaoPass;
use self::swapchain::SwapchainState;
use self::timestamps::GpuTimestamps;

use super::{MAX_TEXTURES, Material, RenderBackend, RenderItem, SceneLighting, TextureHandle};
use crate::profile::Profiler;
use crate::profile_scope;
use crate::scene::DebugLine;

type FrameFuture = FenceSignalFuture<Box<dyn GpuFuture>>;

/// MSAA sample count for the forward pass. One definition, because the forward
/// pipeline, its render pass attachments, and the graph's declaration of the
/// MSAA targets all have to agree or framebuffer creation fails at startup.
pub(crate) const MSAA_SAMPLES: vulkano::image::SampleCount = vulkano::image::SampleCount::Sample4;

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
    /// Indexed like `cascades.cascades`.
    pub casters: &'a [Vec<RenderItem>],
    pub settings: &'a ShadowSettings,
}

pub struct VulkanRenderer {
    pub(crate) ctx: VkContext,
    swapchain: SwapchainState,
    forward: ForwardPass,
    hdr: HdrPass,
    ssao: SsaoPass,
    shadow: ShadowPass,
    /// Debug-line overlay, recorded into the forward pass. Editor-only in
    /// practice: fed lines only through `render_with_overlay`.
    line: LinePass,
    /// The environment cubemap and the skybox that draws it. Also recorded into
    /// the forward pass, for the reason its module documents.
    environment: EnvironmentPass,
    pub(crate) meshes: Vec<GpuMesh>,
    pub(crate) materials: Vec<GpuMaterial>,
    /// Texture views indexed by `TextureHandle`. Index 0 is a 1x1 white texture
    /// and index 1 a flat normal map; materials without a given map point here.
    pub(crate) textures: Vec<Arc<ImageView>>,
    /// Cached set-1 (materials) and set-2 (textures) descriptor sets. `None` =
    /// dirty; rebuilt lazily in `render` after a `load_material`/`load_texture`.
    material_set: Option<Arc<DescriptorSet>>,
    texture_set: Option<Arc<DescriptorSet>>,
    previous_frame_end: Option<FrameFuture>,
    recreate_swapchain: bool,
    pending_extent: [u32; 2],
    /// Per-pass GPU timing; `None` if the device lacks timestamp support.
    timestamps: Option<GpuTimestamps>,
    /// The structure this frame's graph was compiled for. A frame whose config
    /// still matches reuses the compiled graph — recompiling is for a change of
    /// *shape* (SSAO on or off, overlay or not), never for a change of contents.
    config: FrameConfig,
    frame: Frame,
    images: GraphImages,
    framebuffers: PassFramebuffers,
}

impl VulkanRenderer {
    pub fn new(instance: &Arc<Instance>, surface: Arc<Surface>, extent: [u32; 2]) -> Self {
        let ctx = VkContext::new(instance, &surface);
        let format = swapchain_color_format(&ctx, &surface);
        let forward = ForwardPass::new(&ctx.device, &ctx.memory_allocator, hdr::HDR_FORMAT);
        let hdr = HdrPass::new(&ctx, format);
        let ssao = SsaoPass::new(&ctx);
        let shadow = ShadowPass::new(&ctx);
        let line = LinePass::new(&ctx.device, &ctx.memory_allocator, &forward.render_pass);
        let environment = EnvironmentPass::new(&ctx, &forward.render_pass);
        let swapchain = SwapchainState::new(&ctx, &surface, &hdr.tonemap_rp, format, extent);
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
        let config = FrameConfig {
            color_format: format,
            ssao: true,
            overlay: true,
            shadow_cascades: 0,
            shadow_resolution: 1,
        };
        let frame = frame::declare(config).expect("the engine's frame must compile");
        let images = GraphImages::allocate(&ctx.memory_allocator, &frame.graph, extent);
        let framebuffers = PassFramebuffers::build(&frame, &images, &forward, &ssao, &shadow);

        Self {
            ctx,
            swapchain,
            forward,
            hdr,
            ssao,
            shadow,
            line,
            environment,
            meshes: Vec::new(),
            materials: vec![forward::to_gpu_material(&Material::default())],
            textures,
            material_set: None,
            texture_set: None,
            previous_frame_end: None,
            recreate_swapchain: false,
            pending_extent: extent,
            timestamps,
            config,
            frame,
            images,
            framebuffers,
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

    fn new_command_buffer(&self) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            self.ctx.command_buffer_allocator.clone(),
            self.ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap()
    }

    fn reallocate(&mut self) {
        self.images = GraphImages::allocate(
            &self.ctx.memory_allocator,
            &self.frame.graph,
            self.swapchain.extent,
        );
        self.framebuffers = PassFramebuffers::build(
            &self.frame,
            &self.images,
            &self.forward,
            &self.ssao,
            &self.shadow,
        );
    }
}

impl RenderBackend for VulkanRenderer {
    fn load_mesh(&mut self, mesh: &CpuMesh) -> MeshHandle {
        let gpu = forward::upload_mesh(&self.ctx.memory_allocator, &mesh.vertices, &mesh.indices);
        let handle = MeshHandle(self.meshes.len() as u32);
        self.meshes.push(gpu);
        handle
    }

    fn mesh_bounds(&self, mesh: MeshHandle) -> Option<Aabb> {
        self.meshes.get(mesh.0 as usize).map(|gpu| gpu.bounds)
    }

    fn load_material(&mut self, material: &Material) -> MaterialHandle {
        let handle = MaterialHandle(self.materials.len() as u32);
        self.materials.push(forward::to_gpu_material(material));
        self.material_set = None;
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
            eprintln!(
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
        items: &[RenderItem],
        lighting: &SceneLighting,
        camera: &Camera,
        ssao: &SsaoSettings,
        hdr: &HdrSettings,
        environment: &EnvironmentSettings,
    ) {
        // No overlay path (e.g. export/headless) draws no debug lines and no
        // shadows.
        self.render_frame(
            items,
            lighting,
            camera,
            ssao,
            hdr,
            environment,
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
        self.swapchain.swapchain.image_format()
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

    /// Like [`render`](RenderBackend::render) but composites `overlay` (the
    /// editor UI) onto the final image before present.
    pub fn render_with_overlay(
        &mut self,
        items: &[RenderItem],
        lighting: &SceneLighting,
        camera: &Camera,
        ssao: &SsaoSettings,
        hdr: &HdrSettings,
        environment: &EnvironmentSettings,
        debug_lines: &[DebugLine],
        profiler_frame: u64,
        shadows: Option<ShadowFrame<'_>>,
        overlay: Overlay<'_>,
    ) {
        self.render_frame(
            items,
            lighting,
            camera,
            ssao,
            hdr,
            environment,
            debug_lines,
            Some(profiler_frame),
            shadows,
            Some(overlay),
        );
    }

    fn render_frame(
        &mut self,
        items: &[RenderItem],
        lighting: &SceneLighting,
        camera: &Camera,
        ssao: &SsaoSettings,
        hdr: &HdrSettings,
        environment: &EnvironmentSettings,
        debug_lines: &[DebugLine],
        profiler_frame: Option<u64>,
        shadows: Option<ShadowFrame<'_>>,
        overlay: Option<Overlay<'_>>,
    ) {
        if self.pending_extent[0] == 0 || self.pending_extent[1] == 0 {
            return;
        }

        if self.recreate_swapchain {
            if self
                .swapchain
                .recreate(&self.hdr.tonemap_rp, self.pending_extent)
            {
                self.recreate_swapchain = false;
            } else {
                return;
            }
        }

        // The frame's structure, and the only thing that recompiles the graph.
        // Everything else about this call — how many objects, which camera, what
        // exposure — flows through the same compiled schedule.
        self.ensure_graph(FrameConfig {
            color_format: self.swapchain.swapchain.image_format(),
            ssao: ssao.enabled,
            overlay: overlay.is_some(),
            // Sourced from the cascade set rather than the setting, so the
            // number of passes the graph declares cannot disagree with the
            // number of matrices there are to draw them with.
            shadow_cascades: shadows.map_or(0, |s| s.cascades.count as u8),
            shadow_resolution: shadows.map_or(1, |s| s.settings.resolution),
        });

        // Split out because under Fifo this blocks until the presentation engine
        // hands back an image — a vsync wait, not work. Folded into a single
        // "render" scope it swamps the numbers and hides real regressions.
        let (image_index, suboptimal, acquire_future) = {
            profile_scope!("acquire");
            match acquire_next_image(self.swapchain.swapchain.clone(), None)
                .map_err(Validated::unwrap)
            {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            }
        };
        if suboptimal {
            self.recreate_swapchain = true;
        }

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
        self.hdr.exposure = hdr.exposure;
        if let Some(shadows) = shadows {
            self.shadow.constant_bias = shadows.settings.constant_bias;
            self.shadow.slope_bias = shadows.settings.slope_bias;
        }

        // Material table and texture array are static after asset load, so cache
        // their descriptor sets and rebuild only when invalidated (set to None).
        if self.material_set.is_none() {
            self.material_set = Some(self.forward.build_material_set(&self.ctx, &self.materials));
        }
        if self.texture_set.is_none() {
            self.texture_set = Some(self.forward.build_texture_set(&self.ctx, &self.textures));
        }
        let material_set = self.material_set.clone().unwrap();
        let texture_set = self.texture_set.clone().unwrap();

        // One upload feeding every geometry pass in the frame — the SSAO
        // prepass, the forward pass, and each cascade — because the per-object
        // inverse-transpose is too expensive to compute more than once. The
        // camera-visible items come first, so the two screen-space passes still
        // index from zero and the cascades index from `caster_bases`.
        let no_casters: [Vec<RenderItem>; 0] = [];
        let (object_buffer, caster_bases) = self
            .forward
            .upload_objects(items, shadows.map_or(&no_casters, |s| s.casters));

        // Uploaded once even though two graph nodes read it — which is what the
        // shared `object_transforms` declaration in `frame::declare` records.
        let ssao_uniforms = self
            .frame
            .ids
            .ssao
            .map(|_| self.ssao.begin_frame(camera, self.swapchain.extent));

        // With SSAO off the graph has no AO node, so the forward pass samples a
        // 1x1 white view instead: "no occlusion" with no second shader path.
        let ao_view = match self.frame.ids.ssao {
            Some(ids) => self.images.view(ids.ao),
            None => self.ssao.white_view(),
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

        // The whole frame, in the order the compiler derived. Nothing below
        // decides what runs next, what an image's layout is, or what has to
        // finish before what — a pass that needs a different place in the frame
        // gets there by changing its declarations, not by being moved here.
        let extent = self.swapchain.extent;
        let mut raw_passes = Vec::new();
        for pass_id in self.frame.graph.order().to_vec() {
            let body = self.frame.bodies[pass_id.index()];

            if self.frame.graph.pass_kind(pass_id) == PassKind::Raw {
                // Escape-hatch passes own their submission, so they run on the
                // future after this command buffer rather than inside it.
                // `compile` has already established that none of them precedes
                // an inline pass.
                raw_passes.push(body);
                continue;
            }

            let timed = timestamps.as_mut().and_then(|timestamps| {
                timestamps.begin_pass(&mut builder, self.frame.graph.pass_name(pass_id))
            });

            let framebuffer = self
                .framebuffers
                .get(pass_id.index())
                .unwrap_or_else(|| self.swapchain.framebuffers[image_index as usize].clone());
            builder
                .begin_render_pass(
                    begin_info(framebuffer, body),
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap();

            match body {
                PassBody::ShadowCascade(cascade) => {
                    let shadows = shadows.expect("the graph scheduled a cascade with no shadows");
                    let index = cascade as usize;
                    self.shadow.record(
                        &mut builder,
                        self,
                        &shadows.casters[index],
                        shadows.cascades.cascades[index].view_proj,
                        caster_bases[index],
                        self.config.shadow_resolution,
                        object_buffer.clone(),
                    );
                }
                PassBody::SsaoPrepass => self.ssao.record_prepass(
                    &mut builder,
                    self,
                    items,
                    extent,
                    ssao_uniforms.as_ref().unwrap(),
                    object_buffer.clone(),
                ),
                PassBody::SsaoResolve => {
                    let ids = self.frame.ids.ssao.unwrap();
                    self.ssao.record_ao(
                        &mut builder,
                        self,
                        extent,
                        ssao_uniforms.as_ref().unwrap(),
                        self.images.view(ids.depth),
                        self.images.view(ids.normal),
                    );
                }
                PassBody::SsaoBlur => {
                    let ids = self.frame.ids.ssao.unwrap();
                    self.ssao
                        .record_blur(&mut builder, self, extent, self.images.view(ids.raw_ao));
                }
                PassBody::Forward => {
                    self.forward.draw(
                        &mut builder,
                        self,
                        items,
                        lighting,
                        camera,
                        extent,
                        ao_view.clone(),
                        shadow_view.clone(),
                        shadows,
                        material_set.clone(),
                        texture_set.clone(),
                        object_buffer.clone(),
                        environment,
                    );
                    // Between the geometry and the lines, and it has to be:
                    // after the geometry so the depth test rejects the sky
                    // wherever something was drawn, and before the lines
                    // because the sky passes its own test at the depth clear
                    // and would otherwise paint over them.
                    self.environment.record_skybox(
                        &mut builder,
                        &self.ctx,
                        camera,
                        extent,
                        environment,
                    );
                    // Debug lines share the forward subpass: depth-tested against
                    // the scene, drawn on top of it, before the pass ends.
                    self.line.record(&mut builder, debug_lines, camera, extent);
                }
                PassBody::Tonemap => self.hdr.record_tonemap(
                    &mut builder,
                    &self.ctx,
                    extent,
                    self.images.view(self.frame.ids.hdr_color),
                ),
                PassBody::Overlay => unreachable!("handled above"),
            }

            builder.end_render_pass(Default::default()).unwrap();
            if let Some(timestamps) = timestamps.as_mut() {
                timestamps.end_pass(&mut builder, timed);
            }
        }

        if let Some(timestamps) = timestamps.as_mut() {
            timestamps.end_frame(&mut builder);
        }
        if timestamps.is_some() {
            self.timestamps = timestamps;
        }

        let command_buffer = builder.build().unwrap();
        drop(recording);

        if let Some(prev) = self.previous_frame_end.as_mut() {
            prev.cleanup_finished();
        }

        let mut future = self
            .previous_frame_end
            .take()
            .map(|f| f.boxed())
            .unwrap_or_else(|| sync::now(self.ctx.device.clone()).boxed())
            .join(acquire_future)
            .then_execute(self.ctx.queue.clone(), command_buffer)
            .unwrap()
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
        let before_present = future;

        let submitting = crate::profile::scope("submit");
        let future = before_present
            .then_swapchain_present(
                self.ctx.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.swapchain.clone(),
                    image_index,
                ),
            )
            .boxed()
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(f) => self.previous_frame_end = Some(f),
            Err(VulkanError::OutOfDate) => self.recreate_swapchain = true,
            Err(e) => {
                eprintln!("failed to flush future: {e}");
            }
        }
        drop(submitting);
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
