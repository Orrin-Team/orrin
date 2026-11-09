//! The controls that float over the scene: gizmo mode, space and snap at the
//! top left, the frame-stats toggle at the top right.
//!
//! In UI mode (the toolbar's toggle, or T) a click picks the topmost UI element
//! rather than a mesh, and the selected element gets the rect tool (see
//! `editor::rect_tool`) instead of the transform gizmo.
//!
//! While the game is live — playing or paused — the scene is the game's. The
//! transform tools and the rect tool are hidden, nothing in it is picked, hovered, outlined or
//! marked, and a click there reaches the game alone, which is what lets an
//! in-game button sit over a crate without selecting it. The frame stats and
//! the play border stay: they describe the session, not the scene.
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

use crate::editor::gizmo::{Drag, Emphasis, Gizmo, LENGTH, Target};
use crate::editor::state::{EditorState, GizmoMode, GizmoSpace};
use crate::editor::theme::{self, ThemeSet};
use crate::editor::widgets;
use crate::editor::{icons, pick};
use crate::gfx::vulkan::OutlineItem;
use crate::scene::{
    Camera, Hierarchy, Light, LocalTransform, MeshHandle, Parent, Play, PlayState, WorldTransform,
};
use crate::stats::FrameStats;
use orrin_registry::Reflect;

const INSET: f32 = 8.0;

/// How far above an object its first badge stands, and each badge above the
/// last, in points: a marker's height and a little.
const BADGE_RISE: f32 = 21.0;

pub fn show(
    ctx: &egui::Context,
    world: &World,
    state: &mut EditorState,
    themes: &ThemeSet,
    viewport: egui::Rect,
    live: bool,
) {
    let bar = themes.active().bar();

    if !live {
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
                    widgets::toolbar_frame(ui.visuals(), bar).show(ui, |ui| {
                        let hint = "UI mode (T): pick and lay out UI elements";
                        if widgets::toggle_icon(ui, icons::image(), state.ui_mode, hint).clicked() {
                            state.ui_mode = !state.ui_mode;
                        }
                    });
                });
            });
    }

    let toggle = egui::Area::new(egui::Id::new("viewport_view"))
        .fixed_pos(egui::pos2(viewport.right() - INSET, viewport.top() + INSET))
        .pivot(egui::Align2::RIGHT_TOP)
        .constrain_to(viewport)
        .show(ctx, |ui| {
            widgets::toolbar_frame(ui.visuals(), bar).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 2.0;
                    // Offered where it matters — laying out UI — and kept
                    // while a preview is on, so it can always be turned off.
                    if state.ui_mode || state.preview.resolution.is_some() {
                        resolution(ui, world, state);
                    }
                    let hint = "Frame stats overlay";
                    if widgets::toggle_icon(ui, icons::activity(), state.show_stats, hint).clicked()
                    {
                        state.show_stats = !state.show_stats;
                    }
                });
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

    preview_frame(ctx, world, state, viewport);
    play_frame(ctx, world, themes, viewport);
}

/// The resolution the viewport previews: the button says what it is and how
/// large it is drawn, and opens the choice.
fn resolution(ui: &mut egui::Ui, world: &World, state: &mut EditorState) {
    use crate::editor::preview::{MAX_SIDE, PreviewSize, TEMPLATES};

    let hint = "Preview the game's UI at a screen size";
    // Only an icon at rest: the controls over the scene stay a corner's worth.
    let button = match state.preview.resolution {
        None => widgets::toggle_icon(ui, icons::monitor(), false, hint),
        Some([width, height]) => {
            let percent = world
                .get_resource::<crate::scene::GameScreen>()
                .map_or(100.0, |screen| screen.scale * 100.0);
            let word = format!("{width}×{height} · {percent:.0}%");
            widgets::icon_label_button(ui, icons::monitor(), &word, true, hint)
        }
    };
    let popup = ui.make_persistent_id("preview_resolution");
    if button.clicked() {
        ui.memory_mut(|memory| memory.toggle_popup(popup));
    }
    egui::popup_below_widget(
        ui,
        popup,
        &button,
        egui::PopupCloseBehavior::CloseOnClickOutside,
        |ui| {
            ui.set_min_width(220.0);
            let preview = &mut state.preview;
            ui.selectable_value(&mut preview.resolution, None, "Window");
            ui.separator();
            for &(name, size) in TEMPLATES {
                let label = format!("{name}  {}×{}", size[0], size[1]);
                ui.selectable_value(&mut preview.resolution, Some(size), label);
            }
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Custom");
                let mut custom = preview.resolution.unwrap_or([1920, 1080]);
                let mut changed = false;
                for side in &mut custom {
                    changed |= ui
                        .add(egui::DragValue::new(side).range(1..=MAX_SIDE).speed(4))
                        .changed();
                }
                if icons::button(ui, icons::rotate(), 14.0, ui.visuals().text_color(), false)
                    .on_hover_text("Swap width and height")
                    .clicked()
                {
                    custom.swap(0, 1);
                    changed = true;
                }
                if changed {
                    preview.resolution = Some(custom);
                }
            });
            ui.separator();
            ui.add_enabled_ui(preview.resolution.is_some(), |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut preview.size, PreviewSize::Fit, "Fit to window")
                        .on_hover_text("Scale the screen to the viewport");
                    ui.selectable_value(&mut preview.size, PreviewSize::TrueSize, "True size")
                        .on_hover_text("One game pixel to one screen pixel, cut to the viewport");
                });
            });
        },
    );
}

/// What lies outside the previewed screen, dimmed: there the game has no
/// pixels. Under the viewport's controls, which are areas over this layer.
fn preview_frame(ctx: &egui::Context, world: &World, state: &EditorState, viewport: egui::Rect) {
    if state.preview.resolution.is_none() {
        return;
    }
    let Some(screen) = world.get_resource::<crate::scene::GameScreen>().map(|s| *s) else {
        return;
    };
    let ppp = ctx.pixels_per_point();
    let shown = screen.rect();
    let shown = egui::Rect::from_min_size(
        egui::pos2(shown.x, shown.y) / ppp,
        egui::vec2(shown.width, shown.height) / ppp,
    );
    let painter = ctx
        .layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new("preview_frame"),
        ))
        .with_clip_rect(viewport);
    let dim = egui::Color32::from_black_alpha(150);
    let inner = shown.intersect(viewport);
    for band in [
        egui::Rect::from_min_max(viewport.min, egui::pos2(viewport.max.x, inner.min.y)),
        egui::Rect::from_min_max(egui::pos2(viewport.min.x, inner.max.y), viewport.max),
        egui::Rect::from_min_max(
            egui::pos2(viewport.min.x, inner.min.y),
            egui::pos2(inner.min.x, inner.max.y),
        ),
        egui::Rect::from_min_max(
            egui::pos2(inner.max.x, inner.min.y),
            egui::pos2(viewport.max.x, inner.max.y),
        ),
    ] {
        if band.is_positive() {
            painter.rect_filled(band, 0.0, dim);
        }
    }
    painter.rect_stroke(
        shown,
        0.0,
        egui::Stroke::new(1.0_f32, egui::Color32::from_white_alpha(40)),
        egui::StrokeKind::Outside,
    );
}

/// A border round the scene while the game is live, in the accent while it
/// runs and amber while it is paused, so a change made to the fork is never
/// mistaken for one made to the scene: Stop throws it away.
fn play_frame(ctx: &egui::Context, world: &World, themes: &ThemeSet, viewport: egui::Rect) {
    let colour = match world.get_resource::<Play>().map(|play| play.state()) {
        Some(PlayState::Playing) => themes.active().accent(),
        Some(PlayState::Paused) => theme::PENDING,
        Some(PlayState::Editing) | None => return,
    };
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Middle,
        egui::Id::new("play_frame"),
    ));
    painter.rect_stroke(
        viewport,
        0.0,
        egui::Stroke::new(2.0_f32, colour),
        egui::StrokeKind::Inside,
    );
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
    let steps = state.snap_steps;
    let hint = format!(
        "Quantise moves to {} m, turns to {}° and scales to steps of {}. Hold Ctrl while \
         dragging to snap with this off",
        steps.translate, steps.rotate, steps.scale
    );
    if widgets::icon_label_button(ui, icons::grid(), word, state.snap, &hint).clicked() {
        state.snap = !state.snap;
    }
    // The chevron opens a menu rather than toggling, so it sits apart from the
    // button it belongs to instead of reading as part of its label.
    ui.add_space(6.0);
    snap_steps(ui, state);
}

/// The chevron beside the snap button, and the steps it opens.
fn snap_steps(ui: &mut egui::Ui, state: &mut EditorState) {
    let weak = ui.visuals().weak_text_color();
    let chevron =
        icons::button(ui, icons::chevron_down(), 12.0, weak, false).on_hover_text("Snap steps");
    let popup = ui.make_persistent_id("snap_steps");
    if chevron.clicked() {
        ui.memory_mut(|memory| memory.toggle_popup(popup));
    }
    egui::popup_below_widget(
        ui,
        popup,
        &chevron,
        egui::PopupCloseBehavior::CloseOnClickOutside,
        |ui| {
            ui.set_min_width(170.0);
            let steps = &mut state.snap_steps;
            let row = |ui: &mut egui::Ui, label: &str, drag: egui::DragValue| {
                ui.horizontal(|ui| {
                    ui.label(label);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add(drag);
                    });
                });
            };
            row(
                ui,
                "Move",
                egui::DragValue::new(&mut steps.translate)
                    .speed(0.01)
                    .range(0.001..=100.0)
                    .suffix(" m"),
            );
            row(
                ui,
                "Rotate",
                egui::DragValue::new(&mut steps.rotate)
                    .speed(0.5)
                    .range(0.1..=180.0)
                    .suffix("°"),
            );
            row(
                ui,
                "Scale",
                egui::DragValue::new(&mut steps.scale)
                    .speed(0.01)
                    .range(0.01..=10.0)
                    .prefix("×"),
            );
        },
    );
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

/// Find what the pointer is over in the scene. Run before [`gizmo`], whose
/// handle under the pointer from the last frame keeps a hover from showing
/// through it.
pub fn hover(
    ctx: &egui::Context,
    world: &World,
    state: &mut EditorState,
    dock: &crate::editor::dock::Dock,
    viewport: egui::Rect,
) {
    state.hovered = None;
    if state.gizmo_drag.is_some() || state.gizmo_hot.is_some() || state.marquee.is_some() {
        return;
    }
    let Some(pos) = ctx.input(|input| input.pointer.latest_pos()) else {
        return;
    };
    if !viewport.contains(pos) || !dock.over_scene(ctx, pos) {
        return;
    }
    if let Some(camera) = world.get_resource::<Camera>().map(|camera| *camera) {
        state.hovered = pick::pick_at(world, &camera, ctx.screen_rect(), pos);
    }
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
    outline(world, state);
    // Under the handles: drawn first.
    if let Some(camera) = camera {
        let view_proj = camera.view_projection(screen.aspect_ratio());
        let forward = (camera.target - camera.position).normalize();
        let ppp = ctx.pixels_per_point();
        let (selection, hovered) = (&state.selection, state.hovered);
        let emphasis = |entity| {
            if selection.contains(entity) {
                Emphasis::Selected
            } else if hovered == Some(entity) {
                Emphasis::Hovered
            } else {
                Emphasis::Rest
            }
        };
        let tints = state.icon_tints;
        pick::each_marker(world, |entity, at, kind| {
            if (at - camera.position).dot(forward) <= camera.near {
                return;
            }
            let on_screen = pick::to_screen(view_proj, screen, at);
            crate::editor::gizmo::marker(
                &mut state.gizmo_strokes,
                on_screen,
                kind,
                tints.of(kind),
                emphasis(entity),
                ppp,
            );
        });
        // Stacked above the object's middle, clear of where the transform
        // gizmo stands.
        pick::each_badge(world, |entity, at, kind, slot| {
            if (at - camera.position).dot(forward) <= camera.near {
                return;
            }
            let rise = BADGE_RISE * (slot + 1) as f32;
            let on_screen = pick::to_screen(view_proj, screen, at) - egui::vec2(0.0, rise);
            crate::editor::gizmo::marker(
                &mut state.gizmo_strokes,
                on_screen,
                kind,
                tints.of(kind),
                emphasis(entity),
                ppp,
            );
        });
        pick::each_camera(world, |entity, model, lens, active| {
            let shape = crate::editor::gizmo::CameraShape::new(lens, model, screen.aspect_ratio());
            crate::editor::gizmo::camera(
                &mut state.gizmo_strokes,
                &shape,
                &camera,
                screen,
                tints.camera(),
                emphasis(entity),
                active,
                ppp,
            );
        });
        // A selected sound's reach: where it is at full volume, half, a
        // quarter, and where its rolloff stops.
        let mut lines = Vec::new();
        for &entity in state.selection.entities() {
            let (Some(source), Some(model)) = (
                world
                    .get::<crate::scene::AudioSource>(entity)
                    .map(|source| *source),
                world.get::<WorldTransform>(entity).map(|world| world.0),
            ) else {
                continue;
            };
            lines.clear();
            crate::editor::gizmo::audio_lines(&source, model, &mut lines);
            crate::editor::gizmo::world_lines(
                &mut state.gizmo_strokes,
                &lines,
                &camera,
                screen,
                tints.of(pick::Marker::Audio { playing: false }),
                ppp,
            );
        }
        // A selected light's reach: its range, its cone, its direction.
        for &entity in state.selection.entities() {
            let (Some(light), Some(model)) = (
                world.get::<Light>(entity).map(|light| *light),
                world.get::<WorldTransform>(entity).map(|world| world.0),
            ) else {
                continue;
            };
            let depth = (model.w_axis.truncate() - camera.position).dot(forward);
            let per_point =
                2.0 * depth.max(camera.near) * (camera.fov_y * 0.5).tan() / screen.height();
            lines.clear();
            crate::editor::gizmo::light_lines(&light, model, LENGTH * per_point, &mut lines);
            crate::editor::gizmo::world_lines(
                &mut state.gizmo_strokes,
                &lines,
                &camera,
                screen,
                tints.of(pick::Marker::Light(pick::LightKind::Point)),
                ppp,
            );
        }
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
                let snap = state.snap_step(ctrl);
                for (target, moved) in drag.update(ray(pos), snap) {
                    state.edit(target, crate::scene::registry::TRANSFORM, moved.to_value());
                }
                if let Some(sweep) = drag.sweep(ray(pos), snap) {
                    handles.sweep(sweep, ctx.pixels_per_point(), &mut state.gizmo_strokes);
                }
                if let Some(text) = drag.readout(ray(pos), snap) {
                    readout(ctx, pos, text);
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
        state.gizmo_drag.is_some(),
        ctx.pixels_per_point(),
        &mut state.gizmo_strokes,
    );
}

/// What the drag has done, beside the pointer and above everything else, so
/// the hand on the mouse and the number stay in one place for the eye.
fn readout(ctx: &egui::Context, pointer: egui::Pos2, text: String) {
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Tooltip,
        egui::Id::new("gizmo_readout"),
    ));
    let visuals = ctx.style().visuals.clone();
    let galley = painter.layout_no_wrap(
        text,
        egui::FontId::monospace(12.0),
        visuals.strong_text_color(),
    );
    let at = pointer + egui::vec2(18.0, 14.0);
    let plate = egui::Rect::from_min_size(at, galley.size()).expand2(egui::vec2(6.0, 3.0));
    painter.rect_filled(plate, 4.0, visuals.extreme_bg_color.gamma_multiply(0.9));
    painter.galley(at, galley, visuals.strong_text_color());
}

/// What the outline pass is drawn round: every mesh selected or under a
/// selected entity — so a model selected by its root shows whole — and the
/// hovered mesh, unless it is already among them.
fn outline(world: &World, state: &mut EditorState) {
    state.outline.clear();
    let hierarchy = world.get_resource::<Hierarchy>();
    let item = |entity: Entity, selected: bool| {
        let mesh = *world.get::<MeshHandle>(entity)?;
        let model = world.get::<WorldTransform>(entity)?.0;
        Some(OutlineItem {
            mesh,
            model,
            selected,
        })
    };
    let mut seen = Vec::new();
    let mut stack: Vec<Entity> = state.selection.entities().to_vec();
    while let Some(entity) = stack.pop() {
        if seen.contains(&entity) || !world.is_alive(entity) {
            continue;
        }
        seen.push(entity);
        state.outline.extend(item(entity, true));
        if let Some(hierarchy) = &hierarchy {
            stack.extend_from_slice(hierarchy.children_of(entity));
        }
    }
    if let Some(hovered) = state.hovered
        && !seen.contains(&hovered)
    {
        state.outline.extend(item(hovered, false));
    }
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

/// A drag on the scene that did not start on a handle draws a rectangle, and
/// letting go selects what stands inside it.
pub fn marquee(
    ctx: &egui::Context,
    world: &World,
    state: &mut EditorState,
    dock: &crate::editor::dock::Dock,
    viewport: egui::Rect,
) {
    let (pointer, down, pressed, add) = ctx.input(|input| {
        (
            input.pointer.latest_pos(),
            input.pointer.primary_down(),
            input.pointer.primary_pressed(),
            input.modifiers.command,
        )
    });
    // A press on a handle is the gizmo's, which has already taken it.
    if pressed
        && state.gizmo_drag.is_none()
        && state.gizmo_hot.is_none()
        && let Some(pos) = pointer
        && viewport.contains(pos)
        && dock.over_scene(ctx, pos)
    {
        state.marquee = Some(pos);
    }
    let (Some(from), Some(to)) = (state.marquee, pointer) else {
        return;
    };
    let rect = egui::Rect::from_two_pos(from, to);
    // Short of this it is a click, which `select_on_click` answers.
    let drawn = rect.width().max(rect.height()) > 4.0;
    if down {
        if drawn {
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("marquee"),
            ));
            let selection = ctx.style().visuals.selection;
            painter.rect(
                rect,
                0.0,
                selection.bg_fill.gamma_multiply(0.25),
                selection.stroke,
                egui::StrokeKind::Inside,
            );
        }
        return;
    }
    state.marquee = None;
    let Some(camera) = world.get_resource::<Camera>().map(|camera| *camera) else {
        return;
    };
    if !drawn {
        return;
    }
    let screen = ctx.screen_rect();
    let view_proj = camera.view_projection(screen.aspect_ratio());
    let forward = (camera.target - camera.position).normalize();
    // By where each entity stands, not by any part of it: a floor reaching
    // under every rectangle would otherwise join every selection.
    let mut inside = Vec::new();
    world
        .query::<&WorldTransform>()
        .for_each(|entity, transform| {
            let at = transform.translation();
            if (at - camera.position).dot(forward) > camera.near
                && rect.contains(pick::to_screen(view_proj, screen, at))
            {
                inside.push(entity);
            }
        });
    inside.retain(|&entity| !pick::is_locked(world, entity));
    inside.sort_by_key(|entity| entity.index());
    if !add {
        state.selection.clear();
    }
    for entity in inside {
        if !state.selection.contains(entity) {
            state.selection.toggle(entity);
        }
    }
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

/// The topmost UI element under `pos`, a point, that is not locked: what a
/// click in UI mode selects. Any element counts, raycast target or not — the
/// editor picks what is drawn, not what the game would hit.
fn ui_at(world: &World, layout: &crate::ui::Layout, pos: egui::Pos2, ppp: f32) -> Option<usize> {
    let at = [pos.x * ppp, pos.y * ppp];
    layout.elements().iter().rposition(|element| {
        crate::ui::layout::contains(element.rect, at) && !pick::is_locked(world, element.entity)
    })
}

/// UI mode's rect tool: an outline round the element under the pointer, the
/// tool on the selected one, and a drag of it — each frame of which posts the
/// transform it has reached as an edit, one undo step when the button comes
/// up. The 3D gizmo, hover and marquee stand down in UI mode.
pub fn rect_tool(
    ctx: &egui::Context,
    world: &World,
    state: &mut EditorState,
    dock: &crate::editor::dock::Dock,
    viewport: egui::Rect,
) {
    use crate::editor::rect_tool::{self, Frame, RectDrag, SNAP};

    state.gizmo_strokes.clear();
    state.outline.clear();
    state.hovered = None;
    state.gizmo_drag = None;
    state.rect_hot = None;
    let Some(tree) = world.get_resource::<crate::ui::UiTree>() else {
        state.rect_drag = None;
        return;
    };
    let layout = &tree.layout;
    let ppp = ctx.pixels_per_point();
    let (pointer, down, pressed, free) = ctx.input(|input| {
        (
            input.pointer.latest_pos(),
            input.pointer.primary_down(),
            input.pointer.primary_pressed(),
            input.modifiers.command,
        )
    });
    let over = pointer.filter(|&pos| viewport.contains(pos) && dock.over_scene(ctx, pos));
    let selected = state
        .selection
        .primary()
        .and_then(|entity| layout.index_of(entity));
    if state.rect_drag.is_none()
        && let Some(index) = over.and_then(|pos| ui_at(world, layout, pos, ppp))
        && Some(index) != selected
    {
        rect_tool::hovered(&mut state.gizmo_strokes, layout.elements()[index].rect, ppp);
    }
    let Some(index) = selected else {
        state.rect_drag = None;
        return;
    };
    let element = layout.elements()[index];
    let Some(transform) = world
        .get::<crate::ui::RectTransform>(element.entity)
        .map(|transform| *transform)
    else {
        state.rect_drag = None;
        return;
    };
    let frame = Frame::of(&element, &transform, ppp);
    let pixels = |pos: egui::Pos2| glam::Vec2::new(pos.x * ppp, pos.y * ppp);
    match state.rect_drag.take() {
        Some(drag) if drag.entity == element.entity => {
            state.rect_hot = Some(drag.grip);
            if let Some(pos) = pointer.filter(|_| down) {
                let snap = (!free).then_some(SNAP * ppp);
                let (moved, guides) = drag.update(pixels(pos), snap);
                if moved != transform {
                    state.edit(
                        element.entity,
                        crate::scene::registry::RECT_TRANSFORM,
                        moved.to_value(),
                    );
                }
                let parent = layout.parent_rect(index);
                rect_tool::guides(&mut state.gizmo_strokes, &guides, parent, ppp);
                state.rect_drag = Some(drag);
            }
        }
        _ => {
            if let Some(pos) = over {
                state.rect_hot = frame.hit(pos);
                if pressed && let Some(grip) = state.rect_hot {
                    let parent = layout.parent_rect(index);
                    state.rect_drag = Some(RectDrag::begin(
                        grip,
                        transform,
                        &element,
                        parent,
                        pixels(pos),
                    ));
                }
            }
        }
    }
    frame.strokes(state.rect_hot, &mut state.gizmo_strokes, ppp);
}

/// [`select_on_click`] for UI mode: the topmost element under the click, or
/// nothing. A click on one of the rect tool's handles keeps the selection.
pub fn select_ui_on_click(
    ctx: &egui::Context,
    world: &World,
    state: &mut EditorState,
    dock: &crate::editor::dock::Dock,
    viewport: egui::Rect,
) {
    use crate::editor::rect_tool::Grip;

    let Some((pos, toggle)) = ctx.input(|input| {
        let pos = input.pointer.interact_pos()?;
        let toggle = input.modifiers.command;
        input.pointer.primary_clicked().then_some((pos, toggle))
    }) else {
        return;
    };
    if !viewport.contains(pos)
        || !dock.over_scene(ctx, pos)
        || matches!(state.rect_hot, Some(Grip::Pivot | Grip::Edge { .. }))
    {
        return;
    }
    let Some(tree) = world.get_resource::<crate::ui::UiTree>() else {
        return;
    };
    let hit = ui_at(world, &tree.layout, pos, ctx.pixels_per_point())
        .map(|index| tree.layout.elements()[index].entity);
    match (hit, toggle) {
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
