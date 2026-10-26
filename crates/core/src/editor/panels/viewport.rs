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

use glam::{Mat4, Vec3};
use orrin_ecs::{Entity, World};

use crate::editor::gizmo::{Drag, Gizmo, Target};
use crate::editor::state::{
    EditorState, GizmoMode, GizmoSpace, MOVE_SNAP, ROTATE_SNAP, SCALE_SNAP,
};
use crate::editor::theme::{self, ThemeSet};
use crate::editor::widgets;
use crate::editor::{icons, pick};
use crate::scene::{Camera, LocalTransform, Parent, WorldTransform};
use crate::stats::FrameStats;
use orrin_registry::Reflect;

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
    for (mode, name, key, icon) in [
        (GizmoMode::Select, "Select", 1, icons::pointer()),
        (GizmoMode::Move, "Move", 2, icons::move_()),
        (GizmoMode::Rotate, "Rotate", 3, icons::rotate()),
        (GizmoMode::Scale, "Scale", 4, icons::scaling()),
    ] {
        let hint = format!("{name} ({key})");
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
    // The increment the current tool would snap to, so the button answers
    // "by how much?" without opening anything.
    let step = state.snap_step(true).map(|step| match state.gizmo_mode {
        GizmoMode::Rotate => format!("{step}°"),
        GizmoMode::Scale => format!("×{step}"),
        _ => format!("{step}"),
    });
    let word = match &step {
        Some(step) if state.snap => step.as_str(),
        _ => "Snap",
    };
    let hint = format!(
        "Quantise moves to {MOVE_SNAP} m, turns to {ROTATE_SNAP}° and scales to steps of \
         {SCALE_SNAP}. Hold Ctrl while dragging to snap with this off"
    );
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

/// The transform gizmo on the selection: which handle the pointer is on, a
/// drag of one, and this frame's strokes for the renderer.
///
/// The gizmo stands at the centre of everything selected, turned as the
/// primary is, and a drag moves the whole selection about that centre. Each
/// frame of a drag posts the transforms it has reached as edits, so the drag
/// shows as it happens and, the button being held, stays one gesture until it
/// is let go — one undo step (see `scene::edit`).
pub fn gizmo(
    ctx: &egui::Context,
    world: &World,
    state: &mut EditorState,
    dock: &crate::editor::dock::Dock,
    viewport: egui::Rect,
) {
    state.gizmo_strokes.clear();
    state.gizmo_hot = None;
    let primary = state.selection.primary().filter(|&e| world.is_alive(e));
    let camera = world.get_resource::<Camera>().map(|camera| *camera);
    let targets = targets(world, state.selection.entities());
    let screen = ctx.screen_rect();
    // Under the handles: drawn first.
    if let Some(camera) = camera {
        let view_proj = camera.view_projection(screen.aspect_ratio());
        let forward = (camera.target - camera.position).normalize();
        let ppp = ctx.pixels_per_point();
        pick::each_marker(world, |entity, at, kind| {
            if (at - camera.position).dot(forward) <= camera.near {
                return;
            }
            let on_screen = pick::to_screen(view_proj, screen, at);
            let selected = state.selection.contains(entity);
            crate::editor::gizmo::marker(&mut state.gizmo_strokes, on_screen, kind, selected, ppp);
        });
    }
    let placed = primary
        .zip(camera)
        .and_then(|(primary, camera)| Some((primary, camera, pivot(world, primary, &targets)?)));
    let Some((primary, camera, pivot)) = placed else {
        state.gizmo_drag = None;
        return;
    };
    let Some(handles) = Gizmo::new(&camera, screen, pivot, state.gizmo_space) else {
        state.gizmo_drag = None;
        return;
    };
    let (pointer, down, pressed, ctrl) = ctx.input(|input| {
        (
            input.pointer.latest_pos(),
            input.pointer.primary_down(),
            input.pointer.primary_pressed(),
            input.modifiers.command,
        )
    });
    let ray = |pos: egui::Pos2| {
        let ndc = (pos - screen.min) / screen.size() * 2.0 - egui::Vec2::splat(1.0);
        pick::ray(&camera, screen.aspect_ratio(), ndc)
    };

    match state.gizmo_drag.take() {
        Some((dragged, drag)) if dragged == primary => {
            // Lit through the frame that lets go too, which is the frame a
            // click would otherwise land on whatever is behind the handle.
            state.gizmo_hot = Some(drag.handle);
            if let Some(pos) = pointer.filter(|_| down) {
                for (target, moved) in drag.update(ray(pos), state.snap_step(ctrl)) {
                    state.edit(target, crate::scene::registry::TRANSFORM, moved.to_value());
                }
                state.gizmo_drag = Some((dragged, drag));
            }
        }
        _ => {
            if let Some(pos) = pointer
                && viewport.contains(pos)
                && dock.over_scene(ctx, pos)
            {
                state.gizmo_hot = handles.hit(state.gizmo_mode, pos);
                if pressed && let Some(handle) = state.gizmo_hot {
                    state.gizmo_drag =
                        Drag::begin(&handles, state.gizmo_mode, handle, ray(pos), targets)
                            .map(|drag| (primary, drag));
                }
            }
        }
    }
    handles.strokes(
        state.gizmo_mode,
        state.gizmo_hot,
        ctx.pixels_per_point(),
        &mut state.gizmo_strokes,
    );
}

/// What a drag of the selection moves: each selected entity with a transform,
/// except one under another that is selected — that one already moves with
/// it, through the hierarchy, and moving it as well would move it twice.
fn targets(world: &World, selection: &[Entity]) -> Vec<Target> {
    let selected_above = |entity: Entity| {
        let mut cursor = entity;
        // Bounded: a cycle the hierarchy has not broken yet would otherwise spin.
        for _ in 0..4096 {
            match world.get::<Parent>(cursor).map(|parent| parent.get()) {
                Some(parent) if selection.contains(&parent) => return true,
                Some(parent) => cursor = parent,
                None => return false,
            }
        }
        false
    };
    selection
        .iter()
        .filter(|&&entity| world.is_alive(entity) && !selected_above(entity))
        .filter_map(|&entity| {
            Some(Target {
                entity,
                local: *world.get::<LocalTransform>(entity)?,
                world: world.get::<WorldTransform>(entity)?.0,
                parent: crate::scene::parent_world_matrix(world, entity),
            })
        })
        .collect()
}

/// Where the gizmo stands: at the centre of `targets`, turned as `primary`
/// is — or as the first target is, when the primary is not one of them.
fn pivot(world: &World, primary: Entity, targets: &[Target]) -> Option<Mat4> {
    let first = targets.first()?;
    let centre = targets
        .iter()
        .map(|target| target.world.w_axis.truncate())
        .sum::<Vec3>()
        / targets.len() as f32;
    let turned = world
        .get::<WorldTransform>(primary)
        .map_or(first.world, |world| world.0);
    let (_, rotation, _) = turned.to_scale_rotation_translation();
    Some(Mat4::from_rotation_translation(rotation, centre))
}

/// Select what a plain click on the scene lands on, or nothing when it lands
/// on nothing; with Ctrl, add or remove it instead.
pub fn select_on_click(
    ctx: &egui::Context,
    world: &World,
    state: &mut EditorState,
    dock: &crate::editor::dock::Dock,
    viewport: egui::Rect,
) {
    let Some((pos, toggle)) = ctx.input(|input| {
        let pos = input.pointer.interact_pos()?;
        let toggle = input.modifiers.command;
        input.pointer.primary_clicked().then_some((pos, toggle))
    }) else {
        return;
    };
    if !viewport.contains(pos) || !dock.over_scene(ctx, pos) || state.gizmo_hot.is_some() {
        return;
    }
    let Some(camera) = world.get_resource::<Camera>().map(|camera| *camera) else {
        return;
    };
    // The scene is drawn across the whole window, behind the tools.
    match (
        pick::pick_at(world, &camera, ctx.screen_rect(), pos),
        toggle,
    ) {
        (Some(entity), true) => state.selection.toggle(entity),
        (Some(entity), false) => state.selection.set(entity),
        (None, true) => {}
        (None, false) => state.selection.clear(),
    }
}

/// Where the ray through `ndc` meets the ground plane `y = 0`, if it does in
/// front of the camera.
pub fn ground_point(camera: &Camera, aspect: f32, ndc: egui::Vec2) -> Option<Vec3> {
    let (near, direction) = pick::ray(camera, aspect, ndc);
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
