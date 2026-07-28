use std::sync::Arc;

use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageUsage};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo};
use super::context::VkContext;

pub const DEPTH_FORMAT: Format = Format::D32_SFLOAT;

/// Presentation mode — flip this to toggle vsync:
/// - `Fifo`: vsync ON, capped to refresh, no tearing (always supported).
/// - `Mailbox`: uncapped, no tearing (not always supported).
/// - `Immediate`: uncapped, may tear (not always supported).
///
/// Falls back to `Fifo` automatically if the surface doesn't support the choice.
pub const PRESENT_MODE: PresentMode = PresentMode::Fifo;

pub struct SwapchainState {
    pub swapchain: Arc<Swapchain>,
    pub framebuffers: Vec<Arc<Framebuffer>>,
    /// One view per swapchain image, parallel to `framebuffers`. An overlay
    /// (e.g. the editor UI) draws onto these directly, after the tonemap pass
    /// has written the scene into the same image.
    pub image_views: Vec<Arc<ImageView>>,
    pub extent: [u32; 2],
}

impl SwapchainState {
    pub fn new(
        ctx: &VkContext,
        surface: &Arc<Surface>,
        render_pass: &Arc<RenderPass>,
        format: Format,
        extent: [u32; 2],
    ) -> Self {
        let device = &ctx.device;
        let caps = device
            .physical_device()
            .surface_capabilities(surface, Default::default())
            .expect("failed to query surface capabilities");

        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();

        let present_mode = device
            .physical_device()
            .surface_present_modes(surface, Default::default())
            .map(|modes| {
                if modes.into_iter().any(|m| m == PRESENT_MODE) {
                    PRESENT_MODE
                } else {
                    PresentMode::Fifo
                }
            })
            .unwrap_or(PresentMode::Fifo);
        println!("Present mode: {present_mode:?}");

        // Prefer double-buffering, but stay within the surface's advertised range:
        // never below its minimum, and never above its maximum when it sets one
        // (max_image_count == None means unlimited). A surface whose max is 1
        // would otherwise fail creation against the unconditional `.max(2)`.
        let mut min_image_count = caps.min_image_count.max(2);
        if let Some(max) = caps.max_image_count {
            min_image_count = min_image_count.min(max);
        }

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count,
                image_format: format,
                image_extent: extent,
                image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                present_mode,
                composite_alpha,
                ..Default::default()
            },
        )
        .expect("failed to create swapchain");

        let (framebuffers, image_views) = build_framebuffers(render_pass, &images);

        Self {
            swapchain,
            framebuffers,
            image_views,
            extent,
        }
    }

    // Returns false if the surface has zero area (minimized) and recreation is skipped.
    pub fn recreate(
        &mut self,
        render_pass: &Arc<RenderPass>,
        extent: [u32; 2],
    ) -> bool {
        if extent[0] == 0 || extent[1] == 0 {
            return false;
        }

        let (swapchain, images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: extent,
                ..self.swapchain.create_info()
            })
            .expect("failed to recreate swapchain");

        self.swapchain = swapchain;
        let (framebuffers, image_views) = build_framebuffers(render_pass, &images);
        self.framebuffers = framebuffers;
        self.image_views = image_views;
        self.extent = extent;
        true
    }
}

/// Build a framebuffer and keep its color view for each swapchain image. The
/// views are returned alongside so an overlay can target the same images.
fn build_framebuffers(
    render_pass: &Arc<RenderPass>,
    images: &[Arc<Image>],
) -> (Vec<Arc<Framebuffer>>, Vec<Arc<ImageView>>) {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            let framebuffer = Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view.clone()],
                    ..Default::default()
                },
            )
            .unwrap();
            (framebuffer, view)
        })
        .unzip()
}
