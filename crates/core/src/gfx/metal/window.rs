//! A window's `CAMetalLayer` as the thing a frame ends in.
//!
//! The frame's last pass renders into a drawable the layer hands out, and the
//! command buffer presents it. The drawable is taken as late as the walk allows
//! (just before the first pass that writes the swapchain image) because holding
//! one idle is holding a slot of the display's queue: with two drawables, a
//! frame that takes its early costs the display a buffer it could have shown.

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

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(vsync: VsyncMode, images: u32) -> PresentSettings {
        PresentSettings { vsync, images }
    }

    #[test]
    fn a_capped_mode_waits_for_the_display() {
        for vsync in [VsyncMode::Fifo, VsyncMode::FifoRelaxed] {
            assert!(layer_settings(&settings(vsync, 2)).display_sync, "{vsync:?}");
        }
    }

    #[test]
    fn an_uncapped_mode_does_not() {
        for vsync in [VsyncMode::Mailbox, VsyncMode::Immediate] {
            assert!(!layer_settings(&settings(vsync, 2)).display_sync, "{vsync:?}");
        }
    }

    #[test]
    fn two_drawables_stay_two_when_capped_and_become_three_when_not() {
        assert_eq!(layer_settings(&settings(VsyncMode::Fifo, 2)).drawables, 2);
        assert_eq!(layer_settings(&settings(VsyncMode::Immediate, 2)).drawables, 3);
    }

    #[test]
    fn the_count_is_held_to_what_a_layer_allows() {
        assert_eq!(layer_settings(&settings(VsyncMode::Fifo, 0)).drawables, 2);
        assert_eq!(layer_settings(&settings(VsyncMode::Fifo, 9)).drawables, 3);
    }
}
