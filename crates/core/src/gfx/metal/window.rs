//! A window's `CAMetalLayer` as the thing a frame ends in.
//!
//! The frame's last pass renders into a drawable the layer hands out, and the
//! command buffer presents it. The drawable is taken as late as the walk allows
//! (just before the first pass that writes the swapchain image) because holding
//! one idle is holding a slot of the display's queue: with two drawables, a
//! frame that takes its early costs the display a buffer it could have shown.

use std::cell::RefCell;
use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CGSize;
use objc2_metal::{MTLDrawable, MTLPixelFormat};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

use super::MtlDriver;
use crate::gfx::rhi::TextureViewId;
use crate::scene::{PresentSettings, VsyncMode};

/// What a layer is configured with for a set of present settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LayerSettings {
    /// Whether presentation waits for the display's refresh.
    pub display_sync: bool,
    /// `maximumDrawableCount`, which Metal allows to be two or three.
    pub drawables: usize,
}

/// Metal has two behaviours where Vulkan has four: tear-free and capped, or
/// uncapped. `FifoRelaxed` is a way of failing gracefully at a cap, which a
/// layer cannot express, so it takes the cap; `Mailbox` is uncapped without
/// tearing, which a layer cannot express either, so it takes the uncapped mode.
pub(crate) fn layer_settings(present: &PresentSettings) -> LayerSettings {
    let display_sync = matches!(present.vsync, VsyncMode::Fifo | VsyncMode::FifoRelaxed);
    // An uncapped mode with two drawables blocks in `nextDrawable` as soon as
    // one is on screen and one is queued, and then measures the display instead
    // of the frame, so it gets the third.
    let floor = if display_sync { 2 } else { 3 };
    LayerSettings {
        display_sync,
        drawables: (present.images as usize).clamp(floor, 3),
    }
}

/// The layer a window's frames end in.
pub(crate) struct MtlWindow {
    layer: Retained<CAMetalLayer>,
    /// Keeps the layer attached to the view for as long as the window draws.
    _owner: raw_window_metal::Layer,
}

impl MtlWindow {
    /// Attach a layer to the window's `NSView` and configure it for `driver`.
    ///
    /// SAFETY of `view`: it is the window's live `NSView`, which the caller's
    /// window outlives this value by construction (the renderer is dropped with
    /// the window it draws into).
    pub(crate) fn new(
        driver: &MtlDriver,
        view: NonNull<c_void>,
        pixel_format: MTLPixelFormat,
        extent: [u32; 2],
        present: &PresentSettings,
    ) -> Self {
        // SAFETY: `view` is a live NSView, per the contract above.
        let owner = unsafe { raw_window_metal::Layer::from_ns_view(view) };
        // SAFETY: `from_ns_view` returns a `CAMetalLayer` it keeps alive, and the
        // retain makes this value its own owner.
        let layer = unsafe {
            Retained::retain(owner.as_ptr().as_ptr().cast::<CAMetalLayer>())
                .expect("the window has no Metal layer")
        };
        layer.setDevice(Some(&driver.device));
        layer.setPixelFormat(pixel_format);
        // Not framebuffer-only: the overlay's glass samples what is already in
        // the drawable, and a framebuffer-only texture cannot be read.
        layer.setFramebufferOnly(false);
        let window = Self {
            layer,
            _owner: owner,
        };
        window.resize(extent);
        window.apply(present);
        window
    }

    pub(crate) fn resize(&self, extent: [u32; 2]) {
        self.layer.setDrawableSize(CGSize {
            width: f64::from(extent[0]),
            height: f64::from(extent[1]),
        });
    }

    pub(crate) fn apply(&self, present: &PresentSettings) {
        let settings = layer_settings(present);
        self.layer.setDisplaySyncEnabled(settings.display_sync);
        self.layer.setMaximumDrawableCount(settings.drawables);
    }

    pub(crate) fn display_sync(&self) -> bool {
        self.layer.displaySyncEnabled()
    }

    pub(crate) fn drawables(&self) -> usize {
        self.layer.maximumDrawableCount()
    }

    fn next_drawable(&self) -> Option<Retained<ProtocolObject<dyn CAMetalDrawable>>> {
        self.layer.nextDrawable()
    }
}

/// The drawable a frame renders its last passes into, taken the first time a
/// pass asks for it.
pub(crate) struct Frame<'a> {
    window: &'a MtlWindow,
    driver: &'a MtlDriver,
    /// What a frame renders into when the layer has no drawable to give (the
    /// window is occluded or the layer is being resized), so the rest of the
    /// frame still has somewhere to write and is simply not shown.
    fallback: TextureViewId,
    acquired: RefCell<Option<Acquired>>,
}

/// The drawable a frame took, if the layer had one, and the texture to draw into.
type Acquired = (
    Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>,
    TextureViewId,
);

impl<'a> Frame<'a> {
    pub(crate) fn new(
        window: &'a MtlWindow,
        driver: &'a MtlDriver,
        fallback: TextureViewId,
    ) -> Self {
        Self {
            window,
            driver,
            fallback,
            acquired: RefCell::new(None),
        }
    }

    /// The image the frame ends in, acquiring the drawable on first use.
    pub(crate) fn target(&self) -> TextureViewId {
        let mut acquired = self.acquired.borrow_mut();
        acquired
            .get_or_insert_with(|| match self.window.next_drawable() {
                Some(drawable) => {
                    let target = self.driver.import_texture(drawable.texture());
                    (Some(drawable), target)
                }
                None => (None, self.fallback),
            })
            .1
    }

    /// The drawable to present, if a pass ever asked for one.
    pub(crate) fn into_drawable(self) -> Option<Retained<ProtocolObject<dyn MTLDrawable>>> {
        self.acquired
            .into_inner()
            .and_then(|(drawable, _)| drawable)
            .map(ProtocolObject::from_retained)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(vsync: VsyncMode, images: u32) -> PresentSettings {
        PresentSettings { vsync, images }
    }

    #[test]
    fn a_capped_mode_waits_for_the_display() {
        for vsync in [VsyncMode::Fifo, VsyncMode::FifoRelaxed] {
            assert!(
                layer_settings(&settings(vsync, 2)).display_sync,
                "{vsync:?}"
            );
        }
    }

    #[test]
    fn an_uncapped_mode_does_not() {
        for vsync in [VsyncMode::Mailbox, VsyncMode::Immediate] {
            assert!(
                !layer_settings(&settings(vsync, 2)).display_sync,
                "{vsync:?}"
            );
        }
    }

    #[test]
    fn two_drawables_stay_two_when_capped_and_become_three_when_not() {
        assert_eq!(layer_settings(&settings(VsyncMode::Fifo, 2)).drawables, 2);
        assert_eq!(
            layer_settings(&settings(VsyncMode::Immediate, 2)).drawables,
            3
        );
    }

    #[test]
    fn the_count_is_held_to_what_a_layer_allows() {
        assert_eq!(layer_settings(&settings(VsyncMode::Fifo, 0)).drawables, 2);
        assert_eq!(layer_settings(&settings(VsyncMode::Fifo, 9)).drawables, 3);
    }
}
