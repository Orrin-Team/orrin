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

use glam::Vec3;
use orrin_ecs::World;

use crate::editor::icons;
use crate::editor::state::{EditorState, GizmoMode, GizmoSpace, SNAP_STEP};
use crate::editor::theme::{self, ThemeSet};
use crate::editor::widgets;
use crate::scene::Camera;
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

/// Take a prefab let go over the scene, and place it on the ground under the
/// pointer — or at the camera's focus when the pointer is above the horizon.
///
/// Read off the pointer rather than through a widget: the centre claims no
/// input (see `dock`), and a widget there would take the fly camera's clicks.
pub fn accept_prefab(
    ctx: &egui::Context,
    world: &World,
    state: &mut EditorState,
    dock: &crate::editor::dock::Dock,
    viewport: egui::Rect,
) {
    if !egui::DragAndDrop::has_payload_of_type::<orrin_registry::AssetId>(ctx) {
        return;
    }
    let Some(pos) = ctx.input(|input| {
        input
            .pointer
            .any_released()
            .then(|| input.pointer.latest_pos())
            .flatten()
    }) else {
        return;
    };
    if !viewport.contains(pos) || !dock.over_scene(ctx, pos) {
        return;
    }
    let Some(prefab) = egui::DragAndDrop::take_payload::<orrin_registry::AssetId>(ctx) else {
        return;
    };
    let Some(camera) = world.get_resource::<Camera>().map(|camera| *camera) else {
        state.request_instantiate(*prefab, None);
        return;
    };
    // The scene is drawn across the whole window, behind the tools.
    let screen = ctx.screen_rect();
    let ndc = (pos - screen.min) / screen.size() * 2.0 - egui::Vec2::splat(1.0);
    let at = ground_point(&camera, screen.aspect_ratio(), ndc).unwrap_or(camera.target);
    state.request_instantiate_at(*prefab, at);
}

/// Where the ray through `ndc` (−1 to 1, y down, as the scene's projection
/// has it) meets the ground plane `y = 0`, if it does in front of the camera.
pub fn ground_point(camera: &Camera, aspect: f32, ndc: egui::Vec2) -> Option<Vec3> {
    let inverse = camera.view_projection(aspect).inverse();
    let near = inverse.project_point3(Vec3::new(ndc.x, ndc.y, 0.0));
    let far = inverse.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
    let direction = far - near;
    if direction.y.abs() < 1e-6 {
        return None;
    }
    let t = -near.y / direction.y;
    (t >= 0.0).then(|| near + direction * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_middle_of_the_screen_lands_on_what_the_camera_looks_at() {
        let camera = Camera::default();
        let at = ground_point(&camera, 1.6, egui::Vec2::ZERO).unwrap();
        assert!((at - camera.target).length() < 1e-3, "{at}");
    }

    #[test]
    fn a_ray_into_the_sky_meets_no_ground() {
        let camera = Camera {
            target: Vec3::new(0.0, 5.0, 0.0),
            ..Camera::default()
        };
        assert_eq!(ground_point(&camera, 1.6, egui::Vec2::ZERO), None);
    }
}
