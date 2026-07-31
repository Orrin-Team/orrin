mod context;
mod forward;
mod hdr;
mod line;
mod swapchain;
mod texture;
mod ssao;
mod timestamps;

use std::sync::Arc;

use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassContents,
};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::instance::Instance;
use vulkano::swapchain::{
    acquire_next_image, Surface, SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::sync::{self, future::FenceSignalFuture};
use vulkano::{Validated, VulkanError};

use crate::scene::{Camera, CpuMesh, HdrSettings, MaterialHandle, MeshHandle, SsaoSettings};

use self::context::VkContext;
use self::forward::{ForwardPass, GpuMesh, GpuMaterial};
use self::line::LinePass;
use self::swapchain::SwapchainState;
use self::ssao::SsaoPass;
use self::hdr::{HdrPass, HDR_FORMAT};
use self::timestamps::GpuTimestamps;

use crate::profile::Profiler;
use crate::profile_scope;
use crate::scene::DebugLine;
use super::{Material, RenderBackend, RenderItem, SceneLighting, TextureHandle, MAX_TEXTURES};

type FrameFuture = FenceSignalFuture<Box<dyn GpuFuture>>;

/// A hook that draws over the final swapchain image between the tonemap pass and
/// present (the editor UI). Given the future to wait on and that image's view, it
/// returns the future to present. A plain closure, so this module stays free of
/// any UI/egui types.
pub type Overlay<'a> =
    &'a mut dyn FnMut(Box<dyn GpuFuture>, Arc<ImageView>) -> Box<dyn GpuFuture>;

pub struct VulkanRenderer {
    pub(crate) ctx: VkContext,
    swapchain: SwapchainState,
    forward: ForwardPass,
    hdr: HdrPass,
    ssao: SsaoPass,
    /// Debug-line overlay, recorded into the forward pass. Editor-only in
    /// practice: fed lines only through `render_with_overlay`.
    line: LinePass,
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
}

impl VulkanRenderer {
    pub fn new(instance: &Arc<Instance>, surface: Arc<Surface>, extent: [u32; 2]) -> Self {
        let ctx = VkContext::new(instance, &surface);
        let format = swapchain_color_format(&ctx, &surface);
        let forward = ForwardPass::new(&ctx.device, &ctx.memory_allocator, HDR_FORMAT);
        let hdr = HdrPass::new(&ctx, &forward.render_pass, format, extent);
        let ssao = SsaoPass::new(&ctx, extent);
        let line = LinePass::new(&ctx.device, &ctx.memory_allocator, &forward.render_pass);
        let swapchain = SwapchainState::new(&ctx, &surface, &hdr.tonemap_rp, format, extent);
        let timestamps = GpuTimestamps::new(&ctx);

        // Default textures so every material slot resolves to a valid view:
        // index 0 = white (a no-op multiply), index 1 = flat normal (0,0,1).
        let textures = vec![
            texture::upload_texture(&ctx, &[255, 255, 255, 255], [1, 1], Format::R8G8B8A8_UNORM),
            texture::upload_texture(&ctx, &[128, 128, 255, 255], [1, 1], Format::R8G8B8A8_UNORM),
        ];

        Self {
            ctx,
            swapchain,
            forward,
            hdr,
            ssao,
            line,
            meshes: Vec::new(),
            materials: vec![forward::to_gpu_material(&Material::default())],
            textures,
            material_set: None,
            texture_set: None,
            previous_frame_end: None,
            recreate_swapchain: false,
            pending_extent: extent,
            timestamps,
        }
    }
}

impl RenderBackend for VulkanRenderer {
    fn load_mesh(&mut self, mesh: &CpuMesh) -> MeshHandle {
        let gpu = forward::upload_mesh(&self.ctx.memory_allocator, &mesh.vertices, &mesh.indices);
        let handle = MeshHandle(self.meshes.len() as u32);
        self.meshes.push(gpu);
        handle
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
        let view = texture::upload_texture(&self.ctx, pixels, [width, height], format);
        let handle = TextureHandle(self.textures.len() as u32);
        self.textures.push(view);
        self.texture_set = None;
        handle
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
    ) {
        // No overlay path (e.g. export/headless) draws no debug lines.
        self.render_frame(items, lighting, camera, ssao, hdr, &[], None, None);
    }
}

impl VulkanRenderer {
    pub fn queue(&self) -> Arc<Queue> {
        self.ctx.queue.clone()
    }

    pub fn color_format(&self) -> Format {
        self.swapchain.swapchain.image_format()
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
        debug_lines: &[DebugLine],
        profiler_frame: u64,
        overlay: Overlay<'_>,
    ) {
        self.render_frame(
            items,
            lighting,
            camera,
            ssao,
            hdr,
            debug_lines,
            Some(profiler_frame),
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
        debug_lines: &[DebugLine],
        profiler_frame: Option<u64>,
        overlay: Option<Overlay<'_>>,
    ) {
        if self.pending_extent[0] == 0 || self.pending_extent[1] == 0 {
            return;
        }

        if self.recreate_swapchain {
            if self.swapchain.recreate(
                &self.hdr.tonemap_rp,
                self.pending_extent,
            ) {
                self.hdr.resize(
                    &self.ctx.memory_allocator,
                    &self.forward.render_pass,
                    self.pending_extent,
                );
                self.ssao.resize(&self.ctx.memory_allocator, self.pending_extent);
                self.recreate_swapchain = false;
            } else {
                return;
            }
        }

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
        let mut builder = AutoCommandBufferBuilder::primary(
            self.ctx.command_buffer_allocator.clone(),
            self.ctx.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

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

        // One upload feeding both the SSAO prepass and the forward pass; the
        // per-object inverse-transpose is too expensive to compute twice.
        let object_buffer = self.forward.upload_objects(items);

        let ao_view = if ssao.enabled {
            let pass = timestamps
                .as_mut()
                .and_then(|timestamps| timestamps.begin_pass(&mut builder, "ssao"));
            self.ssao.record(
                &mut builder,
                self,
                items,
                camera,
                self.swapchain.extent,
                object_buffer.clone(),
            );
            if let Some(timestamps) = timestamps.as_mut() {
                timestamps.end_pass(&mut builder, pass);
            }
            self.ssao.ao_view()
        } else {
            self.ssao.white_view()
        };

        let forward_pass = timestamps
            .as_mut()
            .and_then(|timestamps| timestamps.begin_pass(&mut builder, "forward"));
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.02, 0.02, 0.03, 1.0].into()),
                        Some(1.0.into()),
                        None,
                    ],
                    ..RenderPassBeginInfo::framebuffer(self.hdr.forward_framebuffer())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap();

        self.forward.draw(
            &mut builder,
            self,
            items,
            lighting,
            camera,
            self.swapchain.extent,
            ao_view,
            material_set,
            texture_set,
            object_buffer,
        );

        // Debug lines share the forward subpass: depth-tested against the scene,
        // drawn on top of it, before the pass ends.
        self.line
            .record(&mut builder, debug_lines, camera, self.swapchain.extent);

        builder.end_render_pass(Default::default()).unwrap();
        if let Some(timestamps) = timestamps.as_mut() {
            timestamps.end_pass(&mut builder, forward_pass);
        }

        let tonemap_pass = timestamps
            .as_mut()
            .and_then(|timestamps| timestamps.begin_pass(&mut builder, "tonemap"));
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![None],
                    ..RenderPassBeginInfo::framebuffer(
                        self.swapchain.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap();
        self.hdr
            .record_tonemap(&mut builder, &self.ctx, self.swapchain.extent);
        builder.end_render_pass(Default::default()).unwrap();

        if let Some(timestamps) = timestamps.as_mut() {
            timestamps.end_pass(&mut builder, tonemap_pass);
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

        let after_scene = self
            .previous_frame_end
            .take()
            .map(|f| f.boxed())
            .unwrap_or_else(|| sync::now(self.ctx.device.clone()).boxed())
            .join(acquire_future)
            .then_execute(self.ctx.queue.clone(), command_buffer)
            .unwrap()
            .boxed();

        // Let the overlay (editor UI) draw onto the same swapchain image before
        // present. Without one, present the tonemapped scene directly.
        let before_present = match overlay {
            Some(draw) => {
                profile_scope!("overlay");
                draw(
                    after_scene,
                    self.swapchain.image_views[image_index as usize].clone(),
                )
            }
            None => after_scene,
        };

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
