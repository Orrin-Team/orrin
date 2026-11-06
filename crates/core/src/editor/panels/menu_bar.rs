//! The bar across the top: every command the editor has, in six menus, the
//! transport in the middle, and the scene file — with the project's name, when
//! there is a project — at the right.
//!
//! Not a ribbon. A ribbon puts every command on screen at once, which makes
//! the commonest no faster to find and spends a sixth of the window on the
//! rest. The commands used while looking at the scene — gizmo mode, space,
//! snap — sit on the scene itself (see `viewport`); the rest are here.
//!
//! Commands the engine cannot yet perform are listed disabled and marked
//! [`NOT_IMPLEMENTED`], rather than omitted or faked: the same convention the
//! README uses for every claim it makes. A menu is where those belong — a
//! disabled entry costs a line in a list the user opened on purpose, where a
//! disabled toolbar button costs space on every frame.

use orrin_ecs::World;
use orrin_registry::Registry;

use crate::editor::catalog;
use crate::editor::dock::{Dock, Tab};
use crate::editor::icons;
use crate::editor::state::{EditorState, SceneRequest, SpawnKind};
use crate::editor::theme::ThemeSet;
use crate::editor::widgets::{self, Tool};
use crate::scene::registry as ids;
use crate::scene::{Play, PlayState, ShadowSettings, SsaoSettings, TaaSettings};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Menu {
    File,
    Edit,
    Entity,
    Render,
    Scripts,
    View,
}

pub const MENUS: [Menu; 6] = [
    Menu::File,
    Menu::Edit,
    Menu::Entity,
    Menu::Render,
    Menu::Scripts,
    Menu::View,
];

impl Menu {
    fn title(self) -> &'static str {
        match self {
            Self::File => "File",
            Self::Edit => "Edit",
            Self::Entity => "Entity",
            Self::Render => "Render",
            Self::Scripts => "Scripts",
            Self::View => "View",
        }
    }
}

const MENU_WIDTH: f32 = 200.0;

pub fn show(
    ctx: &egui::Context,
    world: &mut World,
    registry: &Registry,
    state: &mut EditorState,
    themes: &ThemeSet,
    dock: &mut Dock,
) {
    let theme = themes.active();
    let frame = egui::Frame::new()
        .fill(theme.bar())
        .inner_margin(egui::Margin::symmetric(6, 0));
    egui::TopBottomPanel::top("menu_bar")
        .exact_height(super::BAR_HEIGHT)
        .frame(frame)
        .show(ctx, |ui| {
            let bar = ui.max_rect();
            ui.horizontal_centered(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                ui.add(egui::Image::new(icons::logo()).fit_to_exact_size(egui::Vec2::splat(18.0)));
                ui.add_space(4.0);
                ui.strong("Orrin");
                ui.add_space(8.0);
                ui.scope(|ui| {
                    // Titles are text on the bar until the pointer reaches one:
                    // no resting fill and no outline, but still framed, which
                    // is what keeps the hover and the open state drawn.
                    let idle = &mut ui.visuals_mut().widgets.inactive;
                    idle.weak_bg_fill = egui::Color32::TRANSPARENT;
                    idle.bg_stroke = egui::Stroke::NONE;
                    for menu in MENUS {
                        ui.menu_button(menu.title(), |ui| {
                            ui.set_min_width(MENU_WIDTH);
                            menu_body(menu, ui, world, registry, state, themes, dock);
                        });
                    }
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(6.0);
                    ui.spacing_mut().item_spacing.x = 6.0;
                    if state.scene_dirty() {
                        ui.label(super::figures(format!("{} •", state.scene_path)))
                            .on_hover_text("Unsaved changes");
                    } else {
                        ui.label(super::figures(&state.scene_path));
                    }
                    if let Some(project) = &state.project_name {
                        ui.weak("/");
                        ui.weak(project);
                    }
                });
            });
            let play = world
                .get_resource::<Play>()
                .map_or_else(Default::default, |p| p.state());
            transport(ui, bar, state, play);
        });
}

/// Play, pause and step, centred on the window rather than on whatever space
/// the menus leave, so they stay put as the right-hand label changes length.
/// Play becomes Stop while the game is live, in the accent, so the one button
/// that ends it is the one that started it.
fn transport(ui: &mut egui::Ui, bar: egui::Rect, state: &mut EditorState, play: PlayState) {
    let height = ui.spacing().interact_size.y + 2.0;
    let size = egui::vec2(3.0 * height + 2.0 * 2.0 + 6.0, height + 4.0);
    let rect = egui::Rect::from_center_size(bar.center(), size);
    // A raised pill, not a sunken well: a well reads as a field to type in.
    let fill = ui.visuals().window_fill.to_opaque();
    ui.painter().rect_filled(rect, 6, fill);
    ui.scope_builder(
        egui::UiBuilder::new()
            .max_rect(rect.shrink2(egui::vec2(3.0, 2.0)))
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
        |ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            let live = play != PlayState::Editing;
            let (icon, tool, hint) = if live {
                (
                    icons::stop(),
                    Tool::Selected,
                    "Stop (Ctrl+P) — the scene goes back to how it was before Play",
                )
            } else {
                (icons::play(), Tool::Command, "Play (Ctrl+P)")
            };
            if widgets::tool_button(ui, icon, None, None, tool, hint).clicked() {
                state.request_play();
            }
            let (tool, hint) = match play {
                PlayState::Editing => (Tool::Disabled, "Pause — press Play first"),
                PlayState::Playing => (Tool::Command, "Pause (Ctrl+Shift+P)"),
                PlayState::Paused => (Tool::Selected, "Resume (Ctrl+Shift+P)"),
            };
            if widgets::tool_button(ui, icons::pause(), None, None, tool, hint).clicked() {
                state.request_pause();
            }
            let (tool, hint) = if live {
                (Tool::Command, "Step one frame (Ctrl+Alt+P)")
            } else {
                (Tool::Disabled, "Step — press Play first")
            };
            if widgets::tool_button(ui, icons::step_forward(), None, None, tool, hint).clicked() {
                state.request_step();
            }
        },
    );
}

/// One menu's entries. Public to the panel tests, which draw each body
/// directly rather than clicking it open.
pub fn menu_body(
    menu: Menu,
    ui: &mut egui::Ui,
    world: &mut World,
    registry: &Registry,
    state: &mut EditorState,
    themes: &ThemeSet,
    dock: &mut Dock,
) {
    match menu {
        Menu::File => file(ui, state, dock),
        Menu::Edit => edit(ui, world, registry, state),
        Menu::Entity => entity(ui, state),
        Menu::Render => render(ui, world, state, dock),
        Menu::Scripts => scripts(ui, world, state),
        Menu::View => view(ui, state, themes, dock),
    }
}

/// What every command the engine cannot perform yet says, in one wording, in
/// the place a shortcut would go. Which phase brings it is the roadmap's
/// business, not the menu's.
pub const NOT_IMPLEMENTED: &str = "Not implemented";

fn unbuilt(ui: &mut egui::Ui, label: &str) {
    ui.add_enabled(
        false,
        egui::Button::new(label).shortcut_text(NOT_IMPLEMENTED),
    )
    .on_disabled_hover_text(format!("{label} — {}", NOT_IMPLEMENTED.to_lowercase()));
}

fn command(ui: &mut egui::Ui, label: &str, shortcut: &str) -> bool {
    let clicked = ui
        .add(egui::Button::new(label).shortcut_text(shortcut))
        .clicked();
    if clicked {
        ui.close_menu();
    }
    clicked
}

fn file(ui: &mut egui::Ui, state: &mut EditorState, dock: &mut Dock) {
    let playing = state.playing();
    let recoverable = state.autosave.path(&state.scene_path).is_file();
    ui.add_enabled_ui(!playing, |ui| {
        if command(ui, "Save scene", "Ctrl+S") {
            state.request_scene(SceneRequest::Save);
        }
        if command(ui, "Load scene", "") {
            state.request_scene(SceneRequest::Load);
        }
        ui.add_enabled_ui(recoverable, |ui| {
            if command(ui, "Recover autosave", "") {
                state.request_scene(SceneRequest::Recover);
            }
        })
        .response
        .on_disabled_hover_text("There is no unsaved work to recover for this scene");
    })
    .response
    .on_disabled_hover_text("Stop the game first: the scene is not what is playing");
    if command(ui, "New scene…", "") {
        dock.bring_forward(Tab::Scenes);
    }
    if command(ui, "Scene manager…", "") {
        dock.bring_forward(Tab::Scenes);
    }
    ui.separator();
    // There is no file dialog yet: the path is typed, and both commands above
    // read it. Relative to the working directory.
    ui.weak("Scene file");
    // A fixed width: a field told to fill its space fills a menu's whole
    // allowance and drags the menu out with it.
    ui.add(egui::TextEdit::singleline(&mut state.scene_path).desired_width(MENU_WIDTH - 12.0));
    ui.separator();
    export(ui, state);
    ui.separator();
    unbuilt(ui, "Import…");
    unbuilt(ui, "Reimport all");
    unbuilt(ui, "Packages");
}

/// Export game, one entry per platform, with this machine's marked: the rest
/// cross-compile, which the CLI says is possible or says why not.
fn export(ui: &mut egui::Ui, state: &mut EditorState) {
    use crate::editor::export::{self, Status};
    use orrin_project::export::Platform;

    let open = state.project_root.is_some() && !state.export.is_running();
    ui.add_enabled_ui(open, |ui| {
        ui.menu_button("Export game", |ui| {
            for platform in Platform::ALL {
                let here = if Platform::host() == Some(platform) {
                    "this machine"
                } else {
                    ""
                };
                if command(ui, export::label(platform), here) {
                    state.request_export(platform);
                }
            }
        });
    })
    .response
    .on_disabled_hover_text(if state.export.is_running() {
        "An export is running; the status bar shows its progress"
    } else {
        "Exporting needs a project: open one with `orrin run` or Orrin Hub"
    });
    if let Status::Done { platform, out } = state.export.status() {
        let label = format!("Show the {} game", export::label(*platform));
        if command(ui, &label, "") {
            export::reveal(out);
        }
    }
}

fn edit(ui: &mut egui::Ui, world: &mut World, registry: &Registry, state: &mut EditorState) {
    state.selection.retain_alive(world);
    let primary = state.selection.primary();
    ui.add_enabled_ui(primary.is_some(), |ui| {
        if command(ui, "Delete", "Del") {
            state.request_despawn_selection();
        }
        if command(ui, "Save as prefab", "") {
            state.request_save_prefab();
        }
        if command(ui, "Dump to console", "")
            && let Some(entity) = primary
        {
            super::inspector::dump(world, registry, entity);
        }
    });
    ui.add_enabled_ui(primary.is_some(), |ui| {
        if command(ui, "Select none", "Esc") {
            state.selection.clear();
        }
        if command(ui, "Frame selection", "F") {
            state.request_frame();
        }
    });
    ui.separator();
    // Named by what they would do, so the menu answers "undo what?" before
    // the click rather than after.
    let undo = state
        .history
        .undo_label()
        .map(|label| format!("Undo {label}"));
    ui.add_enabled_ui(undo.is_some(), |ui| {
        if command(ui, undo.as_deref().unwrap_or("Undo"), "Ctrl+Z") {
            state.request_undo();
        }
    });
    let redo = state
        .history
        .redo_label()
        .map(|label| format!("Redo {label}"));
    ui.add_enabled_ui(redo.is_some(), |ui| {
        if command(ui, redo.as_deref().unwrap_or("Redo"), "Ctrl+Shift+Z") {
            state.request_redo();
        }
    });
}

/// What can be spawned. Shared with the hierarchy's add button, so the two
/// lists cannot drift apart.
pub fn spawn_entries(ui: &mut egui::Ui, state: &mut EditorState) {
    for (label, kind) in [
        ("Cube", SpawnKind::Cube),
        ("Sphere", SpawnKind::Sphere),
        ("Plane", SpawnKind::Plane),
    ] {
        if ui
            .add(egui::Button::new(icons::with_label(
                icons::box_(),
                Some(crate::editor::theme::KIND_MESH),
                label,
            )))
            .clicked()
        {
            state.request_spawn(kind);
            ui.close_menu();
        }
    }
    ui.separator();
    for (label, kind, icon) in [
        ("Point light", SpawnKind::PointLight, icons::lightbulb()),
        (
            "Directional light",
            SpawnKind::DirectionalLight,
            icons::sun(),
        ),
    ] {
        if ui
            .add(egui::Button::new(icons::with_label(
                icon,
                Some(crate::editor::theme::KIND_LIGHT),
                label,
            )))
            .clicked()
        {
            state.request_spawn(kind);
            ui.close_menu();
        }
    }
    ui.separator();
    if ui
        .add(egui::Button::new(icons::with_label(
            icons::camera(),
            Some(crate::editor::theme::KIND_CAMERA),
            "Camera",
        )))
        .clicked()
    {
        state.request_spawn(SpawnKind::Camera);
        ui.close_menu();
    }
    ui.separator();
    let ui_icon = catalog::icon(&ids::CANVAS);
    ui.menu_button(icons::with_label(ui_icon, None, "UI"), |ui| {
        use crate::ui::create::UiKind;
        for (kind, icon) in [
            (UiKind::Canvas, catalog::icon(&ids::CANVAS)),
            (UiKind::Panel, icons::rectangle()),
            (UiKind::Image, catalog::icon(&ids::IMAGE)),
            (UiKind::Text, catalog::icon(&ids::TEXT)),
            (UiKind::Button, catalog::icon(&ids::BUTTON)),
            (UiKind::ScrollFrame, catalog::icon(&ids::SCROLL_FRAME)),
        ] {
            let label = icons::with_label(icon, None, kind.name());
            if ui.add(egui::Button::new(label)).clicked() {
                state.request_spawn(SpawnKind::Ui(kind));
                ui.close_menu();
            }
        }
    });
}

fn entity(ui: &mut egui::Ui, state: &mut EditorState) {
    spawn_entries(ui, state);
}

fn render(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState, dock: &mut Dock) {
    super::environment::edit_settings(world, state, |world| {
        ui.checkbox(&mut world.resource_mut::<SsaoSettings>().enabled, "SSAO");
        ui.checkbox(
            &mut world.resource_mut::<TaaSettings>().enabled,
            "Temporal AA",
        );
        let mut shadows = world.resource_mut::<ShadowSettings>();
        ui.checkbox(&mut shadows.enabled, "Shadows");
        ui.checkbox(&mut shadows.debug_cascades, "Tint cascades")
            .on_hover_text("Colour each shadow cascade, to see where the splits fall");
    });
    ui.separator();
    if command(ui, "Environment and exposure…", "") {
        dock.activate(Tab::Environment);
    }
    if command(ui, "Profiler…", "") {
        dock.activate(Tab::Performance);
    }
    ui.separator();
    unbuilt(ui, "Graph inspector");
    unbuilt(ui, "Path trace");
}

#[cfg(not(feature = "scripting"))]
fn scripts(ui: &mut egui::Ui, _world: &mut World, _state: &mut EditorState) {
    ui.add_enabled(false, egui::Button::new("Reload scripts"))
        .on_disabled_hover_text("This build was compiled without the `scripting` feature");
}

#[cfg(feature = "scripting")]
fn scripts(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState) {
    use crate::build_watcher::BuildStatus;

    if command(ui, "Reload scripts", "") {
        state.request_script_reload();
    }
    if let Some(mut status) = world.get_resource_mut::<BuildStatus>() {
        ui.checkbox(&mut status.auto_reload, "Reload after a rebuild");
        ui.separator();
        let (color, word) = super::status_bar::build_word(&status.state);
        icons::labelled(ui, icons::circle_dot(), color, word);
    }
}

fn view(ui: &mut egui::Ui, state: &mut EditorState, themes: &ThemeSet, dock: &mut Dock) {
    for tab in Tab::ALL {
        let mut open = dock.is_open(tab);
        if ui.checkbox(&mut open, tab.title()).changed() {
            dock.toggle(tab);
        }
    }
    ui.separator();
    ui.checkbox(&mut state.show_stats, "Frame stats overlay");
    ui.menu_button("Theme", |ui| {
        let active = themes.active().name.clone();
        for name in themes.names() {
            if ui.radio(name == active, name).clicked() {
                state.request_theme(name);
                ui.close_menu();
            }
        }
    });
    ui.separator();
    if command(ui, "Reset layout", "") {
        dock.reset();
    }
}
