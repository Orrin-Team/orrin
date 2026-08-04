//! The top bar: quick access, the ribbon, and the scene tab strip.
//!
//! Mounted before the side panels so all three rows claim the full window
//! width, and the only surface here not bound to a selection or a resource — it
//! is where the session itself is presented.
//!
//! Commands the engine cannot yet perform are shown disabled with the phase
//! that will bring them, rather than omitted or faked. That is the same
//! convention the README uses for every claim it makes.

use egui::ImageSource;
use orrin_ecs::World;
use orrin_registry::Registry;

use crate::editor::icons;
use crate::editor::state::{
    EditorState, GizmoMode, GizmoSpace, RibbonTab, SNAP_STEP, SceneRequest, SpawnKind,
};
use crate::editor::theme::ThemeSet;
use crate::scene::{HdrSettings, ShadowSettings, SsaoSettings};

const QUICK_ACCESS_HEIGHT: f32 = 30.0;
const COMMAND_WIDTH: f32 = 76.0;
const COMMAND_HEIGHT: f32 = 54.0;
const CAPTION_SIZE: f32 = 9.0;

pub fn show(
    ctx: &egui::Context,
    world: &mut World,
    state: &mut EditorState,
    registry: &Registry,
    themes: &ThemeSet,
) {
    quick_access(ctx, state, themes);
    ribbon(ctx, world, state, registry);
    scene_tabs(ctx, state);
}

fn quick_access(ctx: &egui::Context, state: &mut EditorState, themes: &ThemeSet) {
    egui::TopBottomPanel::top("quick_access")
        .exact_height(QUICK_ACCESS_HEIGHT)
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.label(egui::RichText::new(&state.project_name).weak().monospace());
                ui.separator();

                // Play mode does not exist, so these say so rather than doing
                // nothing quietly.
                for (icon, name) in [
                    (icons::play(), "Play"),
                    (icons::pause(), "Pause"),
                    (icons::square(), "Stop"),
                ] {
                    ui.add_enabled_ui(false, |ui| {
                        icons::button(ui, icon, 13.0, ui.visuals().text_color(), true)
                            .on_disabled_hover_text(format!("{name} — Phase 3, not implemented"));
                    });
                }
                ui.separator();

                for tab in RibbonTab::ALL {
                    if ui
                        .selectable_label(state.ribbon_tab == tab, tab.label())
                        .clicked()
                    {
                        state.ribbon_tab = tab;
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    theme_picker(ui, state, themes);
                });
            });
        });
}

fn theme_picker(ui: &mut egui::Ui, state: &mut EditorState, themes: &ThemeSet) {
    let active = themes.active().name.clone();
    egui::ComboBox::from_id_salt("theme")
        .selected_text(&active)
        .width(110.0)
        .show_ui(ui, |ui| {
            for name in themes.names() {
                if ui.selectable_label(name == active, name).clicked() {
                    state.request_theme(name);
                }
            }
        });
}

fn ribbon(ctx: &egui::Context, world: &mut World, state: &mut EditorState, registry: &Registry) {
    egui::TopBottomPanel::top("ribbon").show(ctx, |ui| {
        ui.horizontal(|ui| match state.ribbon_tab {
            RibbonTab::Home => {
                gizmo_group(ui, state);
                space_group(ui, state);
                spawn_group(ui, state);
                scene_group(ui, state);
            }
            RibbonTab::Scene => {
                spawn_group(ui, state);
                scene_group(ui, state);
                selection_group(ui, world, state, registry);
            }
            RibbonTab::Render => {
                passes_group(ui, world);
                frame_group(ui, world);
                unbuilt_group(ui);
            }
            RibbonTab::Scripts => scripts_groups(ui, world, state),
            RibbonTab::Assets => assets_group(ui),
        });
    });
}

/// One tab bound to `scene_path`. No ＋: multi-scene needs scene management
/// (Phase 5), and a ＋ that cannot work is worse than no ＋ at all.
fn scene_tabs(ctx: &egui::Context, state: &mut EditorState) {
    egui::TopBottomPanel::top("scene_tabs").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.selectable_label(true, &state.scene_path);
            ui.separator();
            ui.weak("deterministic text format · git-diffable");
        });
    });
}

/// A captioned run of commands. The caption is what makes a ribbon scannable,
/// so it is not optional.
fn group(ui: &mut egui::Ui, caption: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.vertical(|ui| {
        ui.horizontal(|ui| add(ui));
        ui.label(
            egui::RichText::new(caption)
                .size(CAPTION_SIZE)
                .weak()
                .monospace(),
        );
    });
    ui.separator();
}

/// One ribbon command: icon above label, with an optional second line for the
/// value or phase it carries.
///
/// Laid out by hand rather than with nested `Ui`s so the cell is exactly
/// `COMMAND_WIDTH` regardless of how long the label is — a ribbon whose buttons
/// change width as their values change is unreadable, and content-driven width
/// in a panel is what makes a panel grow.
struct Command<'a> {
    icon: ImageSource<'static>,
    label: &'a str,
    sub: Option<String>,
    active: bool,
    enabled: bool,
    hint: Option<String>,
}

impl<'a> Command<'a> {
    fn new(icon: ImageSource<'static>, label: &'a str) -> Self {
        Self {
            icon,
            label,
            sub: None,
            active: false,
            enabled: true,
            hint: None,
        }
    }

    fn sub(mut self, sub: impl Into<String>) -> Self {
        self.sub = Some(sub.into());
        self
    }

    fn active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    /// Disabled, with the phase that will bring it. The engine's honesty
    /// convention, applied to a button.
    fn phase(self, phase: &str) -> Self {
        let label = self.label;
        self.enabled(false)
            .sub(phase.to_owned())
            .hint(format!("{label} — {phase}, not implemented"))
    }
}

fn command(ui: &mut egui::Ui, spec: Command<'_>) -> egui::Response {
    let sense = if spec.enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let (rect, response) = ui.allocate_exact_size(egui::vec2(COMMAND_WIDTH, COMMAND_HEIGHT), sense);

    let visuals = ui.visuals();
    let (fill, stroke) = if !spec.enabled {
        (egui::Color32::TRANSPARENT, egui::Stroke::NONE)
    } else if response.is_pointer_button_down_on() {
        let style = visuals.widgets.active;
        (style.bg_fill, style.bg_stroke)
    } else if spec.active {
        (visuals.selection.bg_fill, visuals.selection.stroke)
    } else if response.hovered() {
        let style = visuals.widgets.hovered;
        (style.bg_fill, style.bg_stroke)
    } else {
        (egui::Color32::TRANSPARENT, egui::Stroke::NONE)
    };
    ui.painter().rect(
        rect,
        visuals.widgets.inactive.corner_radius,
        fill,
        stroke,
        egui::StrokeKind::Inside,
    );

    // Disabled dims, never recolours.
    let text = visuals.text_color();
    let color = if spec.enabled {
        text
    } else {
        text.gamma_multiply(0.4)
    };
    let weak = visuals
        .weak_text_color()
        .gamma_multiply(if spec.enabled { 1.0 } else { 0.4 });

    let icon_rect = egui::Rect::from_center_size(
        egui::pos2(rect.center().x, rect.top() + 6.0 + icons::RIBBON / 2.0),
        egui::Vec2::splat(icons::RIBBON),
    );
    icons::tinted(spec.icon, icons::RIBBON, color).paint_at(ui, icon_rect);

    let painter = ui.painter();
    painter.text(
        egui::pos2(rect.center().x, icon_rect.bottom() + 4.0),
        egui::Align2::CENTER_TOP,
        spec.label,
        egui::TextStyle::Button.resolve(ui.style()),
        color,
    );
    if let Some(sub) = &spec.sub {
        painter.text(
            egui::pos2(rect.center().x, rect.bottom() - CAPTION_SIZE - 4.0),
            egui::Align2::CENTER_TOP,
            sub,
            egui::FontId::monospace(CAPTION_SIZE),
            weak,
        );
    }

    match spec.hint {
        Some(hint) if spec.enabled => response.on_hover_text(hint),
        Some(hint) => response.on_disabled_hover_text(hint),
        None => response,
    }
}

fn gizmo_group(ui: &mut egui::Ui, state: &mut EditorState) {
    group(ui, "Gizmo", |ui| {
        for (mode, label, icon, key) in [
            (GizmoMode::Select, "Select", icons::pointer(), "Q"),
            (GizmoMode::Move, "Move", icons::move_(), "W"),
            (GizmoMode::Rotate, "Rotate", icons::rotate(), "E"),
            (GizmoMode::Scale, "Scale", icons::scaling(), "R"),
        ] {
            // The mode is real state and is remembered; the handles that would
            // read it are Phase 3, which the hint says rather than the button
            // pretending by being disabled.
            let hint = format!("{key} — no handles are drawn yet (Phase 3)");
            if command(
                ui,
                Command::new(icon, label)
                    .active(state.gizmo_mode == mode)
                    .hint(hint),
            )
            .clicked()
            {
                state.gizmo_mode = mode;
            }
        }
    });
}

fn space_group(ui: &mut egui::Ui, state: &mut EditorState) {
    group(ui, "Space & snap", |ui| {
        for (space, label, icon, hint) in [
            (
                GizmoSpace::Local,
                "Local",
                icons::box_(),
                "Axes follow the entity",
            ),
            (
                GizmoSpace::World,
                "World",
                icons::globe(),
                "Axes follow the scene",
            ),
        ] {
            if command(
                ui,
                Command::new(icon, label)
                    .active(state.gizmo_space == space)
                    .hint(hint),
            )
            .clicked()
            {
                state.gizmo_space = space;
            }
        }
        let sub = if state.snap {
            format!("{SNAP_STEP}")
        } else {
            "off".to_owned()
        };
        if command(
            ui,
            Command::new(icons::grid(), "Snap")
                .sub(sub)
                .active(state.snap)
                .hint(format!("Quantise gizmo drags to {SNAP_STEP} units")),
        )
        .clicked()
        {
            state.snap = !state.snap;
        }
    });
}

fn spawn_group(ui: &mut egui::Ui, state: &mut EditorState) {
    group(ui, "Spawn", |ui| {
        menu_command(ui, Command::new(icons::box_(), "Mesh"), |ui| {
            for (label, kind) in [
                ("Cube", SpawnKind::Cube),
                ("Sphere", SpawnKind::Sphere),
                ("Plane", SpawnKind::Plane),
            ] {
                if ui.button(label).clicked() {
                    state.request_spawn(kind);
                    ui.close_menu();
                }
            }
        });
        menu_command(ui, Command::new(icons::lightbulb(), "Light"), |ui| {
            for (label, kind) in [
                ("Point Light", SpawnKind::PointLight),
                ("Directional Light", SpawnKind::DirectionalLight),
            ] {
                if ui.button(label).clicked() {
                    state.request_spawn(kind);
                    ui.close_menu();
                }
            }
        });
    });
}

/// A command that opens a menu. `Ui::menu_button` wants a widget it can size
/// itself, so the popup is anchored to the command's own rect instead.
fn menu_command(ui: &mut egui::Ui, spec: Command<'_>, add: impl FnOnce(&mut egui::Ui)) {
    let id = ui.make_persistent_id(spec.label);
    let response = command(ui, spec);
    if response.clicked() {
        ui.memory_mut(|memory| memory.toggle_popup(id));
    }
    egui::popup_below_widget(
        ui,
        id,
        &response,
        egui::PopupCloseBehavior::CloseOnClick,
        |ui| {
            ui.set_min_width(140.0);
            add(ui);
        },
    );
}

fn scene_group(ui: &mut egui::Ui, state: &mut EditorState) {
    group(ui, "Scene", |ui| {
        if command(
            ui,
            Command::new(icons::save(), "Save").hint(&state.scene_path),
        )
        .clicked()
        {
            state.request_scene(SceneRequest::Save);
        }
        if command(
            ui,
            Command::new(icons::folder_open(), "Load").hint(&state.scene_path),
        )
        .clicked()
        {
            state.request_scene(SceneRequest::Load);
        }
        command(ui, Command::new(icons::file_plus(), "New").phase("Phase 5"));
    });
}

fn selection_group(
    ui: &mut egui::Ui,
    world: &mut World,
    state: &mut EditorState,
    registry: &Registry,
) {
    let selected = state.selected;
    group(ui, "Selection", |ui| {
        if command(
            ui,
            Command::new(icons::trash(), "Delete").enabled(selected.is_some()),
        )
        .clicked()
            && let Some(entity) = selected
        {
            state.request_despawn(entity);
        }
        if command(
            ui,
            Command::new(icons::terminal(), "Dump")
                .sub("to console")
                .enabled(selected.is_some()),
        )
        .clicked()
            && let Some(entity) = selected
        {
            super::inspector::dump(world, registry, entity);
        }
    });
}

fn passes_group(ui: &mut egui::Ui, world: &World) {
    group(ui, "Passes", |ui| {
        let (ssao, shadows, tint) = {
            let s = world.resource::<SsaoSettings>();
            let shadow = world.resource::<ShadowSettings>();
            (s.enabled, shadow.enabled, shadow.debug_cascades)
        };
        if command(ui, Command::new(icons::circle_dot(), "SSAO").active(ssao)).clicked() {
            world.resource_mut::<SsaoSettings>().enabled = !ssao;
        }
        if command(ui, Command::new(icons::sun(), "Shadows").active(shadows)).clicked() {
            world.resource_mut::<ShadowSettings>().enabled = !shadows;
        }
        if command(
            ui,
            Command::new(icons::layers(), "Cascade tint").active(tint),
        )
        .clicked()
        {
            world.resource_mut::<ShadowSettings>().debug_cascades = !tint;
        }
    });
}

fn frame_group(ui: &mut egui::Ui, world: &World) {
    let (exposure, cascades) = {
        let hdr = world.resource::<HdrSettings>();
        let shadow = world.resource::<ShadowSettings>();
        (hdr.exposure, shadow.cascade_count)
    };
    group(ui, "Frame", |ui| {
        command(
            ui,
            Command::new(icons::aperture(), "Exposure")
                .sub(format!("{exposure:.2}"))
                .hint("Edited in the Environment panel"),
        );
        command(
            ui,
            Command::new(icons::layers_2(), "Cascades")
                .sub(cascades.to_string())
                .hint("Edited in the Environment panel"),
        );
    });
}

fn unbuilt_group(ui: &mut egui::Ui) {
    group(ui, "Not built yet", |ui| {
        command(
            ui,
            Command::new(icons::workflow(), "Graph inspector").phase("Phase 4"),
        );
        command(
            ui,
            Command::new(icons::sparkles(), "Path trace").phase("roadmap"),
        );
    });
}

fn assets_group(ui: &mut egui::Ui) {
    group(ui, "Asset pipeline", |ui| {
        command(ui, Command::new(icons::import(), "Import").phase("Phase 3"));
        command(
            ui,
            Command::new(icons::reimport(), "Reimport").phase("Phase 3"),
        );
        command(
            ui,
            Command::new(icons::package(), "Packages").phase("roadmap"),
        );
    });
}

#[cfg(not(feature = "scripting"))]
fn scripts_groups(ui: &mut egui::Ui, _world: &mut World, _state: &mut EditorState) {
    group(ui, "Assembly", |ui| {
        command(
            ui,
            Command::new(icons::refresh(), "Reload")
                .enabled(false)
                .sub("no scripting")
                .hint("This build was compiled without the `scripting` feature"),
        );
    });
}

#[cfg(feature = "scripting")]
fn scripts_groups(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState) {
    use crate::build_watcher::{BuildState, BuildStatus};
    use crate::editor::theme;

    let status = world.get_resource::<BuildStatus>().map(|status| {
        (
            status.state.clone(),
            status.auto_reload,
            status.last_duration,
        )
    });

    group(ui, "Assembly", |ui| {
        let sub = match &status {
            Some((BuildState::Building, ..)) => "building…".to_owned(),
            Some((_, _, Some(took))) => format!("{:.1}s", took.as_secs_f32()),
            _ => String::new(),
        };
        if command(ui, Command::new(icons::refresh(), "Reload").sub(sub)).clicked() {
            state.request_script_reload();
        }

        let auto = status.as_ref().is_some_and(|(_, auto, _)| *auto);
        if command(
            ui,
            Command::new(icons::repeat(), "Auto reload")
                .active(auto)
                .hint("Reload after a successful rebuild"),
        )
        .clicked()
            && let Some(mut status) = world.get_resource_mut::<BuildStatus>()
        {
            status.auto_reload = !auto;
        }
    });

    // The build's own state, in the one place that is always on screen.
    if let Some((state, ..)) = &status {
        let (color, text) = match state {
            BuildState::Succeeded => (theme::OK, "up to date"),
            BuildState::Building => (theme::PENDING, "building"),
            BuildState::Failed => (theme::ERROR, "failed"),
            BuildState::Unavailable(_) => (theme::ERROR, "no compiler"),
            BuildState::Idle | BuildState::Off(_) => (theme::LOG_INFO, "idle"),
        };
        group(ui, "Build", |ui| {
            ui.colored_label(color, text);
        });
    }
}
