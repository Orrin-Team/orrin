//! A second window that shows nothing but a UI frame — the storybook's.
//!
//! # Beside the frame, not in it
//!
//! The window is not a node of the frame graph. It has its own surface, its own
//! [`UiPass`] — so its own glyph atlas, uploaded from the painter that laid the
//! story out — and one render pass that starts from nothing and ends
//! presentable: the swapchain image is opened fresh, because the story's
//! backdrop covers every pixel and nothing from before is worth loading. The
//! main frame's graph, its barrier plan and its golden file are untouched by
//! the window existing.
//!
//! # Recorded after the frame, retired with it
//!
//! [`VulkanRenderer::render_ui_window`] runs after the main frame was
//! submitted, against the same frame slot, so the vertices it uploads come out
//! of the per-frame arenas the frame was handed. Those arenas are reused
//! `FRAMES_IN_FLIGHT` (Vulkan) or `FRAME_SLOTS` (Metal) frames later, and the
//! main frame waits for its own submissions at that point — so it waits for the
//! window's there too, before `begin_frame` resets anything: on Vulkan a
//! `RunAhead` of the window's submissions, exactly as the editor overlay keeps
//! one, and on Metal a ring of its command buffers by slot.
//!
//! A swapchain that is out of date is rebuilt at the next frame rather than in
//! the middle of one, after the window's submissions and presents have drained,
//! for the reason the main swapchain is.

use std::sync::Arc;

use vulkano::swapchain::{
    AcquireNextImageInfo, PresentInfo, SemaphorePresentInfo, Surface, SwapchainPresentInfo,
};
use vulkano::sync::semaphore::Semaphore;
use vulkano::{Validated, VulkanError};

#[cfg(target_os = "macos")]
use objc2_metal::MTLCommandBuffer as _;

use super::driver::VkEncoder;
use super::record::{self, InFlight, Recorder, RunAhead};
use super::swapchain::SwapchainState;
use super::ui::UiPass;
use super::{Backend, FRAMES_IN_FLIGHT, VulkanRenderer};
use crate::gfx::rhi::{Device, Encoder};
use crate::gfx::ui::UiFrame;
use crate::scene::PresentSettings;

/// What a window is drawn through: a Vulkan surface, or a macOS view for a
/// Metal layer. It has to match the renderer's backend.
pub enum UiSurface {
    Vulkan(Arc<Surface>),
    #[cfg(target_os = "macos")]
    Metal(std::ptr::NonNull<std::ffi::c_void>),
}

pub(super) struct UiWindow {
    pass: UiPass,
    target: Target,
}

enum Target {
    Vulkan(Box<VkTarget>),
    #[cfg(target_os = "macos")]
    Metal(Box<MtlTarget>),
}

struct VkTarget {
    swapchain: SwapchainState,
    in_flight: RunAhead<InFlight>,
    /// The semaphore each image was last presented behind, held until that
    /// image is presented again — see `VkBackend::present_waits`.
    present_waits: Vec<Option<Arc<Semaphore>>>,
    pending_extent: [u32; 2],
    present: PresentSettings,
    recreate: bool,
}

#[cfg(target_os = "macos")]
struct MtlTarget {
    window: crate::gfx::metal::window::MtlWindow,
    /// Drawn into when the layer has no drawable to give, as the main window's
    /// frame is.
    fallback: crate::gfx::rhi::TextureViewId,
    extent: [u32; 2],
    pending_extent: [u32; 2],
    ring: [Option<
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
    >; crate::gfx::metal::FRAME_SLOTS as usize],
}

impl VulkanRenderer {
    /// Open a window that [`render_ui_window`](Self::render_ui_window) draws
    /// into, replacing any open already. Refused for a surface of the other
    /// backend's kind, or on a renderer with no window of its own.
    pub fn open_ui_window(&mut self, surface: UiSurface, extent: [u32; 2]) -> Result<(), String> {
        self.close_ui_window();
        let extent = extent.map(|side| side.max(1));
        let target = match (surface, &self.backend) {
            (UiSurface::Vulkan(surface), Backend::Vulkan(vk)) => {
                if vk.swapchain.swapchain.is_none() {
                    return Err("an offscreen renderer has no windows to add to".into());
                }
                let present = vk.present;
                let swapchain =
                    SwapchainState::new(&vk.ctx, &surface, vk.swapchain.format, extent, present);
                Target::Vulkan(Box::new(VkTarget {
                    swapchain,
                    in_flight: RunAhead::new(FRAMES_IN_FLIGHT),
                    present_waits: Vec::new(),
                    pending_extent: extent,
                    present,
                    recreate: false,
                }))
            }
            #[cfg(target_os = "macos")]
            (UiSurface::Metal(view), Backend::Metal(mtl)) => {
                let Some(windowed) = &mtl.windowed else {
                    return Err("an offscreen renderer has no windows to add to".into());
                };
                let driver = self.device.metal();
                let window = crate::gfx::metal::window::MtlWindow::new(
                    driver,
                    view,
                    objc2_metal::MTLPixelFormat::BGRA8Unorm_sRGB,
                    extent,
                    &windowed.present(),
                );
                Target::Metal(Box::new(MtlTarget {
                    window,
                    fallback: crate::gfx::metal::backend::create_target(driver, mtl.format, extent),
                    extent,
                    pending_extent: extent,
                    ring: Default::default(),
                }))
            }
            _ => return Err("the window's surface is for the other graphics backend".into()),
        };
        let format = self.backend.format();
        self.ui_window = Some(UiWindow {
            pass: UiPass::new(&self.device, format),
            target,
        });
        Ok(())
    }

    /// Close the UI window once the GPU has finished with it.
    pub fn close_ui_window(&mut self) {
        let Some(mut window) = self.ui_window.take() else {
            return;
        };
        match &mut window.target {
            Target::Vulkan(target) => {
                target.in_flight.drain();
                // A present is fenced by nothing, and the swapchain cannot be
                // destroyed under one still pending.
                if let Backend::Vulkan(vk) = &self.backend {
                    vk.ctx
                        .queue
                        .with(|mut queue| queue.wait_idle())
                        .expect("failed to wait for the queue");
                }
            }
            #[cfg(target_os = "macos")]
            Target::Metal(target) => {
                for command_buffer in target.ring.iter_mut().filter_map(Option::take) {
                    command_buffer.waitUntilCompleted();
                }
                self.device.destroy_texture(target.fallback);
            }
        }
        window.pass.destroy(&self.device);
    }

    pub fn has_ui_window(&self) -> bool {
        self.ui_window.is_some()
    }

    /// The size the UI window will be drawn at from the next frame.
    pub fn resize_ui_window(&mut self, extent: [u32; 2]) {
        match self.ui_window.as_mut().map(|window| &mut window.target) {
            Some(Target::Vulkan(target)) => {
                target.pending_extent = extent;
                target.recreate = true;
            }
            #[cfg(target_os = "macos")]
            Some(Target::Metal(target)) => target.pending_extent = extent,
            None => {}
        }
    }

    /// Wait for the UI window's work in the slot the next frame reuses. Called
    /// beside the main frame's own wait, before `begin_frame`.
    pub(super) fn ui_window_wait_for_room(&mut self) {
        let frame_index = self.frame_index;
        match self.ui_window.as_mut().map(|window| &mut window.target) {
            Some(Target::Vulkan(target)) => target.in_flight.wait_for_room(),
            #[cfg(target_os = "macos")]
            Some(Target::Metal(target)) => {
                let slot = (frame_index % crate::gfx::metal::FRAME_SLOTS) as usize;
                if let Some(previous) = target.ring[slot].take() {
                    previous.waitUntilCompleted();
                }
            }
            None => {}
        }
    }

    /// Forget the window pass's texture table with the main pass's.
    pub(super) fn ui_window_invalidate_textures(&mut self) {
        if let Some(window) = &mut self.ui_window {
            window.pass.invalidate_textures(&self.device);
        }
    }

    /// Draw `frame` into the UI window and present it. After the main frame,
    /// in the same frame slot; a no-op with no window open.
    pub fn render_ui_window(&mut self, frame: UiFrame<'_>) {
        let Some(mut window) = self.ui_window.take() else {
            return;
        };
        match &mut window.target {
            Target::Vulkan(target) => {
                self.render_vk_window(&mut window.pass, target, frame);
            }
            #[cfg(target_os = "macos")]
            Target::Metal(target) => {
                self.render_mtl_window(&mut window.pass, target, frame);
            }
        }
        self.ui_window = Some(window);
    }

    fn render_vk_window(&mut self, pass: &mut UiPass, target: &mut VkTarget, frame: UiFrame<'_>) {
        let Backend::Vulkan(vk) = &self.backend else {
            return;
        };
        let ctx = vk.ctx.clone();
        if target.recreate {
            target.in_flight.drain();
            ctx.queue
                .with(|mut queue| queue.wait_idle())
                .expect("failed to wait for the queue");
            target.present_waits.clear();
            if !target
                .swapchain
                .recreate(target.pending_extent, target.present)
            {
                return;
            }
            target.recreate = false;
        }
        let Some(swapchain) = target.swapchain.swapchain.clone() else {
            return;
        };
        target.in_flight.wait_for_room();

        let acquired_semaphore = Arc::new(
            Semaphore::from_pool(ctx.device.clone()).expect("failed to create a semaphore"),
        );
        // SAFETY: the semaphore was made a line above, so it is unsignalled
        // with no wait pending, and the submission that waits on it holds it
        // until its fence signals.
        let acquired = unsafe {
            swapchain.acquire_next_image(&AcquireNextImageInfo {
                semaphore: Some(acquired_semaphore.clone()),
                ..Default::default()
            })
        };
        let image_index = match acquired.map_err(Validated::unwrap) {
            Ok(acquired) => {
                target.recreate |= acquired.is_suboptimal;
                acquired.image_index
            }
            Err(VulkanError::OutOfDate) => {
                target.recreate = true;
                return;
            }
            Err(e) => {
                tracing::error!("the storybook window could not acquire an image: {e}");
                return;
            }
        };

        let mut recorder = Recorder::new(&ctx);
        {
            let driver = self.device.vulkan();
            let view =
                driver.import_view(target.swapchain.image_views[image_index as usize].clone());
            let mut encoder = Encoder::Vulkan(VkEncoder::new(&mut recorder, driver));
            record_ui(
                &self.device,
                &mut encoder,
                pass,
                frame,
                &self.textures,
                view,
            );
        }
        let (command_buffer, keep) = recorder.end();
        let finished = Arc::new(
            Semaphore::from_pool(ctx.device.clone()).expect("failed to create a semaphore"),
        );
        let in_flight = record::submit_frame(
            &ctx,
            &ctx.queue.clone(),
            command_buffer,
            keep,
            &[(
                // Everything, the fresh target's layout change included: a
                // transition the wait did not cover could run on an image the
                // presentation engine still holds.
                acquired_semaphore,
                ash::vk::PipelineStageFlags::ALL_COMMANDS,
            )],
            std::slice::from_ref(&finished),
        );
        target.in_flight.push(in_flight);

        let slot = image_index as usize;
        if target.present_waits.len() <= slot {
            target.present_waits.resize(slot + 1, None);
        }
        target.present_waits[slot] = Some(finished.clone());
        let present_info = PresentInfo {
            wait_semaphores: vec![SemaphorePresentInfo::new(finished)],
            swapchain_infos: vec![SwapchainPresentInfo::swapchain_image_index(
                swapchain,
                image_index,
            )],
            ..Default::default()
        };
        // SAFETY: the image was acquired from this swapchain above and has not
        // been presented since; the semaphore is the one its only writer, the
        // submission just made, signals.
        let result = ctx.queue.with(|mut queue| {
            unsafe { queue.present(&present_info) }.map(|r| r.collect::<Vec<_>>())
        });
        match result.map_err(Validated::unwrap) {
            Ok(results) => {
                for result in results {
                    match result {
                        Ok(suboptimal) => target.recreate |= suboptimal,
                        Err(VulkanError::OutOfDate) => target.recreate = true,
                        Err(e) => tracing::error!("the storybook window could not present: {e}"),
                    }
                }
            }
            Err(VulkanError::OutOfDate) => target.recreate = true,
            Err(e) => tracing::error!("the storybook window could not present: {e}"),
        }
    }

    #[cfg(target_os = "macos")]
    fn render_mtl_window(&mut self, pass: &mut UiPass, target: &mut MtlTarget, frame: UiFrame<'_>) {
        let driver = self.device.metal();
        if target.pending_extent != target.extent
            && target.pending_extent.iter().all(|&side| side > 0)
        {
            target.window.resize(target.pending_extent);
            self.device.destroy_texture(target.fallback);
            let format = self.backend.format();
            target.fallback =
                crate::gfx::metal::backend::create_target(driver, format, target.pending_extent);
            target.extent = target.pending_extent;
        }
        let mut state = crate::gfx::metal::State::new(driver);
        let drawable = {
            let window_frame =
                crate::gfx::metal::window::Frame::new(&target.window, driver, target.fallback);
            let mut encoder =
                Encoder::Metal(crate::gfx::metal::MtlEncoder::new(driver, &mut state));
            encoder.set_pass_name("storybook");
            record_ui(
                &self.device,
                &mut encoder,
                pass,
                frame,
                &self.textures,
                window_frame.target(),
            );
            window_frame.into_drawable()
        };
        let slot = (self.frame_index % crate::gfx::metal::FRAME_SLOTS) as usize;
        target.ring[slot] = Some(state.finish(drawable));
    }
}

/// The window's one pass: the atlas and the texture table brought up to date,
/// then `frame` over a target whose old contents do not matter, handed back
/// presentable.
fn record_ui(
    device: &Device,
    encoder: &mut Encoder,
    pass: &mut UiPass,
    frame: UiFrame<'_>,
    textures: &[crate::gfx::rhi::TextureViewId],
    target: crate::gfx::rhi::TextureViewId,
) {
    pass.prepare(device, encoder, frame, textures);
    let extent = device.view_extent(target);
    encoder.begin_rendering_fresh(target);
    pass.record(device, encoder, frame, extent);
    encoder.end_rendering();
    encoder.close_swapchain(target);
}
