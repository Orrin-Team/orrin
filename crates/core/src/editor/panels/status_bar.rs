//! The bar across the bottom: what the session is doing, readable without
//! opening a tool — the script build and any export on the left, the frame's
//! cost at the right.
//!
//! The device the frame runs on and the memory it holds are that item's hover
//! text rather than more of the bar. Neither changes while the editor is open,
//! so on screen they were read once and then were noise; the Profiler shows
//! the memory beside the numbers it explains.
//!
//! Nothing here is a control. A status bar that also holds buttons becomes the
//! place commands go when there is nowhere better, and then it is a toolbar.
//! The agents indicator answers a click, but only by opening the page that
//! says who it is counting: a way to the setting behind a status, not a
//! command. A failed script build and the count of errors logged out of sight
//! answer one the same way, by bringing the Console forward: a way to the tool
//! that explains the status. A build that did not fail has nothing to explain,
//! and is not clickable.

use orrin_ecs::World;

use crate::editor::dock::Tab;
use crate::editor::export::{self, Status};
use crate::editor::icons;
use crate::editor::state::{DockRequest, EditorState};
use crate::editor::theme::{self, ThemeSet};
use crate::scene::UnknownComponents;
use crate::stats::FrameStats;

pub const HEIGHT: f32 = 24.0;

pub fn show(ctx: &egui::Context, world: &World, state: &mut EditorState, themes: &ThemeSet) {
    let frame = egui::Frame::new()
        .fill(themes.active().bar())
        .inner_margin(egui::Margin::symmetric(10, 0));
    egui::TopBottomPanel::bottom("status_bar")
        .exact_height(HEIGHT)
        .frame(frame)
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 14.0;
                build(ui, world, state);
                unseen(ui, state);
                exporting(ui, state.export.status());
                carried(ui, world);
                agents(ui, state);

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let Some(stats) = world.get_resource::<FrameStats>() {
                        frame_cost(ui, &stats);
                    }
                });
            });
        });
}

/// The frame's cost, with what it runs on and the memory it occupies on hover.
fn frame_cost(ui: &mut egui::Ui, stats: &FrameStats) {
    use super::performance::{cpu, fps, gpu, vram};
    let mut figures = format!("{} · {}", fps(stats), cpu(stats));
    if let Some(gpu) = gpu(stats) {
        figures.push_str(" · ");
        figures.push_str(&gpu);
    }
    let hint = match stats.device() {
        Some(device) => format!("{device}\n{}", vram(stats)),
        None => vram(stats),
    };
    ui.label(super::figures(figures)).on_hover_text(hint);
}

/// While the remote listens, how many agents are connected; nothing while it
/// is off. A click opens Settings ▸ Agents, which lists them.
fn agents(ui: &mut egui::Ui, state: &mut EditorState) {
    let Some(Ok(status)) = &state.remote else {
        return;
    };
    let text = match status.peers {
        0 => "Remote on".to_owned(),
        1 => "1 agent".to_owned(),
        n => format!("{n} agents"),
    };
    let colour = if status.peers > 0 {
        theme::OK
    } else {
        theme::PENDING
    };
    let hint = format!(
        "Listening on 127.0.0.1:{} for agents · open Settings ▸ Agents",
        status.port
    );
    let drawn = ui
        .horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            let (dot, _) = ui.allocate_exact_size(egui::Vec2::splat(8.0), egui::Sense::hover());
            ui.painter().circle_filled(dot.center(), 3.5, colour);
            ui.label(super::figures(text).weak());
        })
        .response;
    let response = over(ui, &drawn, "agents").on_hover_text(hint);
    if response.clicked() {
        state.open_settings(Some(super::settings::Page::Agents));
    }
}

/// A click on an item already drawn. A widget of its own, added after the
/// item's text, rather than `Response::interact` on the item: that keeps the
/// row's id, which egui placed beneath the text when the row began, and the
/// text — selectable, so it senses clicks — would take them.
fn over(ui: &egui::Ui, drawn: &egui::Response, name: &str) -> egui::Response {
    ui.interact(drawn.rect, ui.id().with(name), egui::Sense::click())
        .on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Errors logged while the Console was out of sight; nothing when there are
/// none. A click brings the Console forward, which also clears the count.
fn unseen(ui: &mut egui::Ui, state: &mut EditorState) {
    let (errors, _) = state.console.unseen();
    if errors == 0 {
        return;
    }
    let text = match errors {
        1 => "1 error".to_owned(),
        n => format!("{n} errors"),
    };
    let drawn = ui
        .horizontal(|ui| icons::labelled(ui, icons::circle_x(), theme::LOG_ERROR, &text))
        .response;
    let response = over(ui, &drawn, "unseen errors")
        .on_hover_text("Logged while the Console was out of sight · show the Console");
    if response.clicked() {
        state.request_dock(DockRequest::Show(Tab::Console));
    }
}

/// The export File → Export game started, while it runs and once it ends.
fn exporting(ui: &mut egui::Ui, status: &Status) {
    let (color, text) = match status {
        Status::Idle => return,
        Status::Running { platform, step } => (
            theme::PENDING,
            match step {
                Some(step) => format!("Exporting for {} · {step}", export::label(*platform)),
                None => format!("Exporting for {}", export::label(*platform)),
            },
        ),
        Status::Done { platform, .. } => (
            theme::OK,
            format!(
                "Exported for {} · File → Show the game",
                export::label(*platform)
            ),
        ),
        Status::Failed { platform, .. } => (
            theme::ERROR,
            format!(
                "Export for {} failed · see the console",
                export::label(*platform)
            ),
        ),
    };
    let response = ui
        .horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            let (dot, _) = ui.allocate_exact_size(egui::Vec2::splat(8.0), egui::Sense::hover());
            ui.painter().circle_filled(dot.center(), 3.5, color);
            ui.label(super::figures(text).weak());
        })
        .response;
    if let Status::Failed { reason, .. } = status {
        response.on_hover_text(reason);
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
fn build(_ui: &mut egui::Ui, _world: &World, _state: &mut EditorState) {}

#[cfg(feature = "scripting")]
fn build(ui: &mut egui::Ui, world: &World, state: &mut EditorState) {
    use crate::build_watcher::{BuildState, BuildStatus};

    let Some(status) = world.get_resource::<BuildStatus>() else {
        return;
    };
    let (color, word) = build_word(&status.state);
    let took = status.last_duration.map_or_else(String::new, |took| {
        format!(" · {:.1} s", took.as_secs_f32())
    });
    let response = ui
        .horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            let (dot, _) = ui.allocate_exact_size(egui::Vec2::splat(8.0), egui::Sense::hover());
            ui.painter().circle_filled(dot.center(), 3.5, color);
            ui.label(super::figures(format!("C# {word}{took}")).weak());
        })
        .response;
    if matches!(status.state, BuildState::Failed) {
        let clicked = over(ui, &response, "build status")
            .on_hover_text("The compiler's errors are in the Console · show it")
            .clicked();
        if clicked {
            state.request_dock(DockRequest::Show(Tab::Console));
        }
    }
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
