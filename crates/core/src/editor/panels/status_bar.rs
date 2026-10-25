//! The bar across the bottom: what the session is doing, readable without
//! opening a tool — the script build on the left; the frame's cost, the
//! memory it holds and the device it runs on at the right.
//!
//! Nothing here is a control. A status bar that also holds buttons becomes the
//! place commands go when there is nowhere better, and then it is a toolbar.

use orrin_ecs::World;

use crate::editor::icons;
use crate::editor::theme::{self, ThemeSet};
use crate::scene::UnknownComponents;
use crate::stats::FrameStats;

pub const HEIGHT: f32 = 24.0;

pub fn show(ctx: &egui::Context, world: &World, themes: &ThemeSet) {
    let frame = egui::Frame::new()
        .fill(themes.active().bar())
        .inner_margin(egui::Margin::symmetric(10, 0));
    egui::TopBottomPanel::bottom("status_bar")
        .exact_height(HEIGHT)
        .frame(frame)
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 14.0;
                build(ui, world);
                carried(ui, world);

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let Some(stats) = world.get_resource::<FrameStats>() {
                        device(ui, &stats);
                    }
                });
            });
        });
}

/// The frame's cost, the memory it occupies and what it runs on — right to
/// left, so the frame time keeps its place at the corner whatever the device
/// is called.
fn device(ui: &mut egui::Ui, stats: &FrameStats) {
    let gpu = stats
        .gpu_ms()
        .map_or_else(String::new, |ms| format!(" · GPU {ms:.2} ms"));
    ui.label(super::figures(format!(
        "{:.0} FPS · CPU {:.2} ms{gpu}",
        stats.fps(),
        stats.frame_ms()
    )));
    let gib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let vram = match stats.vram_used() {
        Some(used) => format!("VRAM {:.1} / {:.1} GB", gib(used), gib(stats.vram_total())),
        None => format!("VRAM {:.1} GB", gib(stats.vram_total())),
    };
    ui.label(super::figures(vram).weak());
    if let Some(device) = stats.device() {
        ui.label(super::figures(device).weak());
    }
}

/// Components this build could not apply and is carrying through to the next
/// save. Surfaced rather than hidden: silence would make the round trip look
/// lossless when it is merely non-destructive.
fn carried(ui: &mut egui::Ui, world: &World) {
    // A query rather than a walk over every entity: this bar draws every
    // frame, and almost no entity carries anything.
    let mut carried = 0usize;
    world
        .query::<&UnknownComponents>()
        .for_each(|_, unknown| carried += unknown.0.len());
    if carried > 0 {
        let text = format!("carrying {carried} unapplied component(s)");
        ui.horizontal(|ui| icons::labelled(ui, icons::warning(), theme::LOG_WARN, &text))
            .response
            .on_hover_text("Preserved on save. See the console for which.");
    }
}

#[cfg(not(feature = "scripting"))]
fn build(_ui: &mut egui::Ui, _world: &World) {}

#[cfg(feature = "scripting")]
fn build(ui: &mut egui::Ui, world: &World) {
    use crate::build_watcher::BuildStatus;

    let Some(status) = world.get_resource::<BuildStatus>() else {
        return;
    };
    let (color, word) = build_word(&status.state);
    let took = status.last_duration.map_or_else(String::new, |took| {
        format!(" · {:.1} s", took.as_secs_f32())
    });
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        let (dot, _) = ui.allocate_exact_size(egui::Vec2::splat(8.0), egui::Sense::hover());
        ui.painter().circle_filled(dot.center(), 3.5, color);
        ui.label(super::figures(format!("C# {word}{took}")).weak());
    });
}

/// The build's state as a colour and a word, for the bar and the Scripts menu.
#[cfg(feature = "scripting")]
pub fn build_word(state: &crate::build_watcher::BuildState) -> (egui::Color32, &'static str) {
    use crate::build_watcher::BuildState;
    match state {
        BuildState::Succeeded => (theme::OK, "up to date"),
        BuildState::Building => (theme::PENDING, "building"),
        BuildState::Failed => (theme::ERROR, "build failed"),
        BuildState::Unavailable(_) => (theme::ERROR, "no compiler"),
        BuildState::Idle | BuildState::Off(_) => (theme::LOG_DEBUG, "idle"),
    }
}
