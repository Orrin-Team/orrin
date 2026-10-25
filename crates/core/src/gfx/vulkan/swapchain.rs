use std::sync::Arc;

use super::context::VkContext;
use crate::scene::{PresentSettings, VsyncMode};
use vulkano::device::{Device, DeviceOwned};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageUsage};
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::swapchain::{PresentMode, Surface, Swapchain, SwapchainCreateInfo};

pub const DEPTH_FORMAT: Format = Format::D32_SFLOAT;

/// The one place a [`VsyncMode`] becomes a Vulkan present mode.
fn present_mode(vsync: VsyncMode) -> PresentMode {
    match vsync {
        VsyncMode::Fifo => PresentMode::Fifo,
        VsyncMode::FifoRelaxed => PresentMode::FifoRelaxed,
        VsyncMode::Mailbox => PresentMode::Mailbox,
        VsyncMode::Immediate => PresentMode::Immediate,
    }
}

/// What the surface will actually honour of what was asked for.
///
/// Both halves fall back rather than fail: an unsupported present mode becomes
/// `Fifo`, which every driver has to offer, and an image count outside the
/// advertised range is clamped into it. A setting the surface cannot meet is a
/// worse frame, never a dead window — which matters more here than usual,
/// because these are the two knobs somebody reaches for *while* chasing a
/// number.
fn resolve(
    device: &Arc<Device>,
    surface: &Arc<Surface>,
    want: PresentSettings,
) -> (PresentMode, u32) {
    let physical = device.physical_device();
    let wanted = present_mode(want.vsync);
    let mode = physical
        .surface_present_modes(surface, Default::default())
        .map(|modes| {
            if modes.into_iter().any(|m| m == wanted) {
                wanted
            } else {
                PresentMode::Fifo
            }
        })
        .unwrap_or(PresentMode::Fifo);

    let (min, max) = match physical.surface_capabilities(surface, Default::default()) {
        Ok(caps) => (caps.min_image_count, caps.max_image_count),
        Err(_) => (1, None),
    };

    (mode, image_count(want.images, mode, min, max))
}

/// Images an uncapped present mode needs before it can do anything a capped one
/// cannot: one on screen, one queued, one being drawn.
const UNCAPPED_MINIMUM: u32 = 3;

/// How many images to build the swapchain with, given what was asked for, the
/// mode the surface honoured, and the range it advertises.
///
/// Split out from [`resolve`] because it is the half worth asserting and the
/// half that needs no device.
fn image_count(want: u32, mode: PresentMode, min: u32, max: Option<u32>) -> u32 {
    // `Mailbox` and `Immediate` are the modes that let a frame finish ahead of
    // the display, and with two images they cannot: one is on screen and one is
    // queued, so the acquire blocks until the presentation engine hands one back
    // and the uncapped mode measures as the capped one it was chosen over.
    //
    // A floor rather than a default, and applied here rather than in
    // `PresentSettings`, because it belongs to the mode and not to the setting:
    // somebody switching to `Mailbox` in the performance panel has asked for
    // this, whatever the image count they did not touch still says. The default
    // stays two, so a `Fifo` number taken before this and one taken after are
    // the same measurement.
    let floor = match mode {
        PresentMode::Mailbox | PresentMode::Immediate => UNCAPPED_MINIMUM,
        _ => 1,
    };

    let images = want.max(floor).max(min);
    // Last, so a surface that advertises a smaller maximum than the floor gets a
    // worse frame rather than a failed swapchain — the same fallback the mode
    // itself takes above.
    match max {
        Some(max) => images.min(max),
        None => images,
    }
}

/// What the frame's last pass draws into: a real swapchain, or — offscreen — a
/// single ordinary image nobody presents.
///
/// One type rather than two because everything between the tonemap pass and this
/// point must not be able to tell the difference. An offscreen render exists to
/// be *evidence* about the windowed one, and it stops being that the moment the
/// two take different paths through the renderer. So `swapchain` is an option
/// and the rest of the struct is shared: the frame acquires index 0 instead of
/// waiting on a presentation engine, and skips the present at the end.
pub struct SwapchainState {
    /// `None` offscreen. The only two places that branch on it are the acquire
    /// and the present.
    pub swapchain: Option<Arc<Swapchain>>,
    /// The images behind `image_views`, kept so an offscreen render can copy
    /// the finished frame back out. Empty for a windowed target, which has
    /// nothing to read back.
    pub readback: Vec<Arc<Image>>,
    /// Carried rather than read off the swapchain, because offscreen there is no
    /// swapchain to read it off.
    pub format: Format,
    /// One view per swapchain image, indexed by the acquired image index. The
    /// tonemap pass renders into one of these directly — there is no
    /// framebuffer to bind it to — and an overlay (e.g. the editor UI) draws
    /// onto the same image afterwards.
    pub image_views: Vec<Arc<ImageView>>,
    pub extent: [u32; 2],
    /// Whether the images may be blitted from, which the editor overlay's
    /// frosted glass needs. Asked of the surface, which need not offer it.
    pub capturable: bool,
}

impl SwapchainState {
    pub fn new(
        ctx: &VkContext,
        surface: &Arc<Surface>,
        format: Format,
        extent: [u32; 2],
        present: PresentSettings,
    ) -> Self {
        let device = &ctx.device;
        let caps = device
            .physical_device()
            .surface_capabilities(surface, Default::default())
            .expect("failed to query surface capabilities");

        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let capturable = caps
            .supported_usage_flags
            .contains(ImageUsage::TRANSFER_SRC);
        let mut image_usage = ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST;
        if capturable {
            image_usage |= ImageUsage::TRANSFER_SRC;
        }
        let (present_mode, min_image_count) = resolve(device, surface, present);

        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count,
                image_format: format,
                image_extent: extent,
                image_usage,
                present_mode,
                composite_alpha,
                ..Default::default()
            },
        )
        .expect("failed to create swapchain");

        Self {
            swapchain: Some(swapchain),
            readback: Vec::new(),
            format,
            image_views: build_views(&images),
            extent,
            capturable,
        }
    }

    /// A single image the frame renders into and nobody presents.
    ///
    /// `TRANSFER_SRC` is what lets the finished frame be copied back to host
    /// memory and written out as a PNG. A window asks for it only for the
    /// overlay's glass, and only where the surface offers it.
    pub fn offscreen(ctx: &VkContext, format: Format, extent: [u32; 2]) -> Self {
        let image = Image::new(
            ctx.memory_allocator.clone(),
            ImageCreateInfo {
                format,
                extent: [extent[0], extent[1], 1],
                usage: ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::TRANSFER_DST
                    | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .expect("failed to allocate the offscreen target");

        let images = [image];

        Self {
            swapchain: None,
            image_views: build_views(&images),
            readback: images.to_vec(),
            format,
            extent,
            capturable: true,
        }
    }

    // Returns false if the surface has zero area (minimized) and recreation is skipped.
    //
    // `present` is re-resolved here rather than carried from construction: this is
    // the one path a present-mode or image-count change travels, so the two
    // settings reach the swapchain the same way an extent does and there is no
    // second place for them to be applied from.
    pub fn recreate(&mut self, extent: [u32; 2], present: PresentSettings) -> bool {
        if extent[0] == 0 || extent[1] == 0 {
            return false;
        }
        // An offscreen target has a fixed extent decided by whoever asked for
        // the render; nothing can resize it out from under the frame.
        let Some(current) = self.swapchain.as_ref() else {
            return false;
        };

        let (present_mode, min_image_count) = resolve(current.device(), current.surface(), present);
        let (swapchain, images) = current
            .recreate(SwapchainCreateInfo {
                image_extent: extent,
                present_mode,
                min_image_count,
                ..current.create_info()
            })
            .expect("failed to recreate swapchain");

        self.swapchain = Some(swapchain);
        self.image_views = build_views(&images);
        self.extent = extent;
        true
    }

    /// What the surface actually honoured, as opposed to what was asked for.
    ///
    /// `None` offscreen, where there is no presentation engine to ask. Reported
    /// by the run banner and the performance panel for one reason: a present mode
    /// the driver quietly declined is the last thing a frame-time figure may omit.
    pub fn applied_present(&self) -> Option<(PresentMode, u32)> {
        let swapchain = self.swapchain.as_ref()?;
        Some((
            swapchain.create_info().present_mode,
            swapchain.image_count(),
        ))
    }
}

/// One colour view per target image. The tonemap pass renders into the acquired
/// one and an overlay draws onto the same image after it.
fn build_views(images: &[Arc<Image>]) -> Vec<Arc<ImageView>> {
    images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A capped mode is handed exactly what was asked for: the floor below
    /// exists for the acquire an uncapped mode blocks in, and `Fifo` is supposed
    /// to block there.
    #[test]
    fn a_capped_mode_gets_the_count_it_asked_for() {
        assert_eq!(image_count(2, PresentMode::Fifo, 1, None), 2);
        assert_eq!(image_count(2, PresentMode::FifoRelaxed, 1, None), 2);
    }

    /// The defect this floor exists for: two images and `Mailbox` is `Fifo` with
    /// extra steps, because the acquire waits for the presentation engine either
    /// way.
    #[test]
    fn an_uncapped_mode_gets_a_third_image() {
        assert_eq!(image_count(2, PresentMode::Mailbox, 1, None), 3);
        assert_eq!(image_count(2, PresentMode::Immediate, 1, None), 3);
    }

    /// The floor raises a count, never lowers one.
    #[test]
    fn asking_for_more_than_the_floor_is_honoured() {
        assert_eq!(image_count(5, PresentMode::Mailbox, 1, None), 5);
    }

    /// And the surface has the last word in both directions, because a count it
    /// cannot honour has to be a worse frame rather than a dead window.
    #[test]
    fn the_surface_range_wins_over_both() {
        assert_eq!(image_count(2, PresentMode::Mailbox, 1, Some(2)), 2);
        assert_eq!(image_count(2, PresentMode::Fifo, 4, None), 4);
    }
}
