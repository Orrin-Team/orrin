//! The controls that float over the scene: gizmo mode, space and snap at the
//! top left, the frame-stats toggle at the top right.
//!
//! These are the one deliberate exception to the centre painting nothing (see
//! `dock`). They are the commands used while looking at the scene, so they sit
//! on it, where the eye already is — the way every editor the design borrows
//! from places them — rather than in a toolbar row that would take its height
//! from the scene on every frame. Each group is a small `Area`, which is also what
//! makes a click on one belong to the editor rather than to the fly camera:
//! `Dock::wants_pointer` claims any layer above the background.

use orrin_ecs::World;

use crate::editor::icons;
use crate::editor::state::{EditorState, GizmoMode, GizmoSpace, SNAP_STEP};
use crate::editor::theme::{self, ThemeSet};
use crate::editor::widgets;
use crate::stats::FrameStats;

const INSET: f32 = 8.0;

pub fn show(
    ctx: &egui::Context,
    world: &World,
    state: &mut EditorState,
    themes: &ThemeSet,
    viewport: egui::Rect,
) {
    let bar = themes.active().bar();

    egui::Area::new(egui::Id::new("viewport_tools"))
        .fixed_pos(viewport.min + egui::vec2(INSET, INSET))
        .constrain_to(viewport)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                widgets::toolbar_frame(ui.visuals(), bar).show(ui, |ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    gizmo_modes(ui, state);
                });
                widgets::toolbar_frame(ui.visuals(), bar).show(ui, |ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    space_and_snap(ui, state);
                });
            });
        });

    let toggle = egui::Area::new(egui::Id::new("viewport_view"))
        .fixed_pos(egui::pos2(viewport.right() - INSET, viewport.top() + INSET))
        .pivot(egui::Align2::RIGHT_TOP)
        .constrain_to(viewport)
        .show(ctx, |ui| {
            widgets::toolbar_frame(ui.visuals(), bar).show(ui, |ui| {
                let hint = "Frame stats overlay";
                if widgets::toggle_icon(ui, icons::activity(), state.show_stats, hint).clicked() {
                    state.show_stats = !state.show_stats;
                }
            });
        })
        .response
        .rect;

    // An area of its own, pinned under the toggle by its right edge: sharing
    // the toggle's would widen it as the readout opened and drag the button
    // out from under the pointer.
    if state.show_stats
        && let Some(stats) = world.get_resource::<FrameStats>()
    {
        egui::Area::new(egui::Id::new("viewport_stats"))
            .fixed_pos(egui::pos2(toggle.right(), toggle.bottom() + 4.0))
            .pivot(egui::Align2::RIGHT_TOP)
            .constrain_to(viewport)
            .show(ctx, |ui| {
                // Never wrapping: an area keeps the width it settled at, and a
                // readout laid out against a narrower one wraps a character to
                // a line.
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                widgets::toolbar_frame(ui.visuals(), bar)
                    .inner_margin(egui::Margin::symmetric(10, 6))
                    .show(ui, |ui| stats_readout(ui, &stats));
            });
    }
}

fn gizmo_modes(ui: &mut egui::Ui, state: &mut EditorState) {
    for (mode, name, icon) in [
        (GizmoMode::Select, "Select", icons::pointer()),
        (GizmoMode::Move, "Move", icons::move_()),
        (GizmoMode::Rotate, "Rotate", icons::rotate()),
        (GizmoMode::Scale, "Scale", icons::scaling()),
    ] {
        // The mode is real state and is remembered; the handles that would
        // read it are not built, which the hint says rather than the button
        // pretending by being disabled.
        let not_yet = super::menu_bar::NOT_IMPLEMENTED.to_lowercase();
        let hint = format!("{name} — handles {not_yet}");
        if widgets::toggle_icon(ui, icon, state.gizmo_mode == mode, &hint).clicked() {
            state.gizmo_mode = mode;
        }
    }
}

fn space_and_snap(ui: &mut egui::Ui, state: &mut EditorState) {
    let (icon, word, hint) = match state.gizmo_space {
        GizmoSpace::World => (icons::globe(), "World", "Axes follow the scene"),
        GizmoSpace::Local => (icons::box_(), "Local", "Axes follow the entity"),
    };
    if widgets::icon_label_button(ui, icon, word, false, hint).clicked() {
        state.gizmo_space = match state.gizmo_space {
            GizmoSpace::World => GizmoSpace::Local,
            GizmoSpace::Local => GizmoSpace::World,
        };
    }
    let step = format!("{SNAP_STEP}");
    let word = if state.snap { step.as_str() } else { "Snap" };
    let hint = format!("Quantise gizmo drags to {SNAP_STEP} units");
    if widgets::icon_label_button(ui, icons::grid(), word, state.snap, &hint).clicked() {
        state.snap = !state.snap;
    }
}

fn stats_readout(ui: &mut egui::Ui, stats: &FrameStats) {
    ui.spacing_mut().item_spacing.y = 2.0;
    ui.label(super::figures(format!("{:.0} FPS", stats.fps())).strong());
    ui.label(super::figures(format!("CPU {:.2} ms", stats.frame_ms())).color(theme::CPU));
    let gpu = stats
        .gpu_ms()
        .map_or_else(|| "GPU n/a".to_owned(), |ms| format!("GPU {ms:.2} ms"));
    ui.label(super::figures(gpu).color(theme::GPU));
}
