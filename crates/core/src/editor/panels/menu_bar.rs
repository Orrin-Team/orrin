//! The bar across the top: the editor's commands, in six menus, the transport
//! in the middle, and the scene file — with the project's name, when there is
//! a project — at the right.
//!
//! Not a ribbon. A ribbon puts every command on screen at once, which makes
//! the commonest no faster to find and spends a sixth of the window on the
//! rest. The commands used while looking at the scene — gizmo mode, space,
//! snap — sit on the scene itself (see `viewport`); the rest are here, bar
//! the engine developer's own tools, which the command palette finds.
//!
//! A menu lists what can be done. A command the engine cannot perform is not
//! listed at all: an entry that is never enabled teaches nothing but where a
//! feature is missing, in a list the user opened to act. One that cannot run
//! *now* — outside a project, while the game plays, with nothing selected —
//! is listed disabled, and its hover says what would let it run.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use orrin_ecs::World;

use super::settings::Page;
use crate::editor::catalog;
use crate::editor::commands::{self, Command};
use crate::editor::dock::{Dock, Tab};
use crate::editor::icons;
use crate::editor::state::{EditorState, IconTints, SceneRequest, SpawnKind};
use crate::editor::theme::ThemeSet;
use crate::editor::widgets::{self, Tool};
use crate::scene::registry as ids;
use crate::scene::{Play, PlayState, ShadowSettings};

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

/// What the stand-in for the scene file dialog picks, in tests: `None` inside
/// is a dialog closed without a pick.
#[cfg(test)]
pub(super) static SCENE_PICKED: Mutex<Option<Option<PathBuf>>> = Mutex::new(None);

/// Which of the File menu's dialogs is open.
#[derive(Clone, Copy)]
pub(super) enum Pick {
    Open,
    SaveAs,
}

/// A scene file dialog open on a thread of its own, so the editor goes on
/// drawing behind it. Kept in egui's memory, as the asset browser keeps its
/// own, and read every frame by [`show`] whether or not the menu is open.
#[derive(Clone)]
struct SceneDialog {
    pick: Pick,
    /// Empty until the dialog closes; then the file picked, or nothing.
    picked: Arc<Mutex<Option<Option<PathBuf>>>>,
}

fn dialog_id() -> egui::Id {
    egui::Id::new("scene_file_dialog")
}

/// Whether a scene file dialog is open. Public to the panel tests.
pub(super) fn dialog_open(ctx: &egui::Context) -> bool {
    ctx.data(|data| data.get_temp::<SceneDialog>(dialog_id()).is_some())
}

pub(super) fn start_dialog(ctx: &egui::Context, state: &EditorState, pick: Pick) {
    let current = PathBuf::from(&state.scene_path);
    let folder = current
        .parent()
        .filter(|folder| folder.is_dir())
        .map(std::path::Path::to_path_buf)
        .or_else(|| state.project_root.clone());
    let name = format!(
        "{}.{}",
        crate::scene::loaded::name_of(&state.scene_path),
        crate::scene::loaded::EXTENSION
    );
    let picked = Arc::new(Mutex::new(None));
    let (into, repaint) = (picked.clone(), ctx.clone());
    let dialog = std::thread::Builder::new()
        .name("orrin-scene-dialog".to_owned())
        .spawn(move || {
            let file = pick_scene(pick, folder, name);
            if let Ok(mut slot) = into.lock() {
                *slot = Some(file);
            }
            repaint.request_repaint();
        });
    match dialog {
        Ok(_) => ctx.data_mut(|data| data.insert_temp(dialog_id(), SceneDialog { pick, picked })),
        Err(error) => tracing::error!("could not open a file dialog: {error}"),
    }
}

/// Act on a scene file dialog that has closed. Opening goes through the
/// question about unsaved work; saving moves the scene only once a file was
/// picked, and then saves it there.
fn poll_dialog(ctx: &egui::Context, state: &mut EditorState) {
    let Some(dialog) = ctx.data(|data| data.get_temp::<SceneDialog>(dialog_id())) else {
        return;
    };
    let Some(file) = dialog.picked.lock().ok().and_then(|mut slot| slot.take()) else {
        return;
    };
    ctx.data_mut(|data| data.remove::<SceneDialog>(dialog_id()));
    let Some(mut file) = file else {
        return;
    };
    match dialog.pick {
        Pick::Open => state.request_open_scene_checked(file.to_string_lossy()),
        // The menu is shut while the game plays, but the dialog may have
        // been open when Play was pressed: the save would be refused.
        Pick::SaveAs if state.playing() => {
            tracing::warn!("save as: the game is playing; Stop it first");
        }
        Pick::SaveAs => {
            if file.extension().is_none() {
                file.set_extension(crate::scene::loaded::EXTENSION);
            }
            state.scene_path = crate::editor::state::plain(file.to_string_lossy().into_owned());
            state.request_scene(SceneRequest::Save);
        }
    }
}

/// The scene file picked in the operating system's dialog; `None` if it was
/// cancelled. Blocks, so it runs on a thread of its own, through the async
/// dialog for the reason the asset browser's is (`assets::pick_files`).
#[cfg(not(test))]
fn pick_scene(pick: Pick, folder: Option<PathBuf>, name: String) -> Option<PathBuf> {
    let mut dialog =
        rfd::AsyncFileDialog::new().add_filter("Scene", &[crate::scene::loaded::EXTENSION]);
    if let Some(folder) = folder {
        dialog = dialog.set_directory(folder);
    }
    let file = match pick {
        Pick::Open => pollster::block_on(dialog.set_title("Open scene").pick_file()),
        Pick::SaveAs => pollster::block_on(
            dialog
                .set_title("Save scene as")
                .set_file_name(name)
                .save_file(),
        ),
    };
    file.map(|file| file.path().to_owned())
}

#[cfg(test)]
fn pick_scene(_: Pick, _: Option<PathBuf>, _: String) -> Option<PathBuf> {
    SCENE_PICKED.lock().unwrap().take().flatten()
}

pub fn show(
    ctx: &egui::Context,
    world: &mut World,
    state: &mut EditorState,
    themes: &ThemeSet,
    dock: &mut Dock,
) {
    poll_dialog(ctx, state);
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
                            menu_body(menu, ui, world, state, themes, dock);
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
                    if let Some(project) = state.project_name.clone() {
                        ui.weak("/");
                        let weak = ui.visuals().weak_text_color();
                        let name = egui::Label::new(egui::RichText::new(project).color(weak))
                            .sense(egui::Sense::click());
                        if ui
                            .add(name)
                            .on_hover_cursor(egui::CursorIcon::PointingHand)
                            .on_hover_text("Project settings")
                            .clicked()
                        {
                            state.open_settings(Some(Page::General));
                        }
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
            let hint = |command, words: &str| {
                commands::hint(ui.ctx(), &state.settings.keys, command, words)
            };
            let (icon, tool, play_hint) = if live {
                (
                    icons::stop(),
                    Tool::Selected,
                    format!(
                        "{} — the scene goes back to how it was before Play",
                        hint(Command::Play, "Stop")
                    ),
                )
            } else {
                (icons::play(), Tool::Command, hint(Command::Play, "Play"))
            };
            let (pause_tool, pause_hint) = match play {
                PlayState::Editing => (Tool::Disabled, "Pause — press Play first".to_owned()),
                PlayState::Playing => (Tool::Command, hint(Command::Pause, "Pause")),
                PlayState::Paused => (Tool::Selected, hint(Command::Pause, "Resume")),
            };
            let (step_tool, step_hint) = if live {
                (Tool::Command, hint(Command::Step, "Step one frame"))
            } else {
                (Tool::Disabled, "Step — press Play first".to_owned())
            };
            if widgets::tool_button(ui, icon, None, None, tool, &play_hint).clicked() {
                state.request_play();
            }
            let pause =
                widgets::tool_button(ui, icons::pause(), None, None, pause_tool, &pause_hint);
            if pause.clicked() {
                state.request_pause();
            }
            let step =
                widgets::tool_button(ui, icons::step_forward(), None, None, step_tool, &step_hint);
            if step.clicked() {
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
    state: &mut EditorState,
    themes: &ThemeSet,
    dock: &mut Dock,
) {
    match menu {
        Menu::File => file(ui, state, dock),
        Menu::Edit => edit(ui, world, state),
        Menu::Entity => entity(ui, state),
        Menu::Render => render(ui, world, state, dock),
        Menu::Scripts => scripts(ui, world, state),
        Menu::View => view(ui, state, themes, dock),
    }
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

/// An entry for a command in `editor::commands`, beside the chord that runs it.
fn keyed(ui: &mut egui::Ui, state: &EditorState, label: &str, of: Command) -> bool {
    let shortcut = commands::text(ui.ctx(), &state.settings.keys, of);
    command(ui, label, &shortcut)
}

fn file(ui: &mut egui::Ui, state: &mut EditorState, dock: &mut Dock) {
    let playing = state.playing();
    let recoverable = state.autosave.path(&state.scene_path).is_file();
    let picking = dialog_open(ui.ctx());
    ui.add_enabled_ui(!playing, |ui| {
        if keyed(ui, state, "Save scene", Command::Save) {
            state.request_scene(SceneRequest::Save);
        }
        ui.add_enabled_ui(!picking, |ui| {
            if keyed(ui, state, "Save scene as…", Command::SaveAs) {
                start_dialog(ui.ctx(), state, Pick::SaveAs);
            }
            if keyed(ui, state, "Open scene…", Command::Open) {
                start_dialog(ui.ctx(), state, Pick::Open);
            }
        });
        if keyed(ui, state, "Reload scene", Command::Reload) {
            state.request_reload_checked();
        }
        ui.add_enabled_ui(recoverable, |ui| {
            if keyed(ui, state, "Recover autosave", Command::Recover) {
                state.request_scene(SceneRequest::Recover);
            }
        })
        .response
        .on_disabled_hover_text("There is no unsaved work to recover for this scene");
    })
    .response
    .on_disabled_hover_text("Stop the game first: the scene is not what is playing");
    if command(ui, "New scene…", "") {
        state.naming_scene = true;
        dock.bring_forward(Tab::Scenes);
    }
    if command(ui, "Scene manager…", "") {
        dock.bring_forward(Tab::Scenes);
    }
    ui.separator();
    // Shown, not edited: a typed path would quietly move the next Ctrl+S.
    // Save scene as… is the way to move it, through the system's dialog.
    ui.weak("Scene file");
    ui.scope(|ui| {
        ui.set_max_width(MENU_WIDTH);
        let path = &state.scene_path;
        ui.add(egui::Label::new(super::figures(path)).truncate())
            .on_hover_text(path);
    });
    ui.separator();
    ui.add_enabled_ui(state.project_root.is_some(), |ui| {
        if keyed(ui, state, "Project settings…", Command::ProjectSettings) {
            state.open_settings(Some(Page::General));
        }
    })
    .response
    .on_disabled_hover_text(
        "Project settings need a project: open one with `orrin run` or Orrin Hub",
    );
    export(ui, state);
    ui.separator();
    // The Assets tab's + button: what is picked goes into the folder it has
    // open, so the tab is brought forward to show where.
    ui.add_enabled_ui(state.project_root.is_some(), |ui| {
        if command(ui, "Import…", "") {
            state.pick_files = true;
            dock.bring_forward(Tab::Assets);
        }
    })
    .response
    .on_disabled_hover_text("Importing needs a project: open one with `orrin run` or Orrin Hub");
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

fn edit(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState) {
    state.selection.retain_alive(world);
    let primary = state.selection.primary();
    ui.add_enabled_ui(primary.is_some(), |ui| {
        if keyed(ui, state, "Duplicate", Command::Duplicate) {
            state.request_duplicate();
        }
        if keyed(ui, state, "Copy", Command::Copy) {
            state.request_copy();
        }
    });
    ui.add_enabled_ui(state.can_paste(), |ui| {
        if keyed(ui, state, "Paste", Command::Paste) {
            state.request_paste();
        }
    });
    ui.add_enabled_ui(primary.is_some(), |ui| {
        if keyed(ui, state, "Rename", Command::Rename) {
            state.request_rename();
        }
        if keyed(ui, state, "Delete", Command::Delete) {
            state.request_despawn_selection();
        }
        if keyed(ui, state, "Save as prefab", Command::SavePrefab) {
            state.request_save_prefab();
        }
    });
    if keyed(ui, state, "Select all", Command::SelectAll) {
        super::hierarchy::select_all(ui.ctx(), world, state);
    }
    ui.add_enabled_ui(primary.is_some(), |ui| {
        if keyed(ui, state, "Select none", Command::SelectNone) {
            state.selection.clear();
        }
        if keyed(ui, state, "Frame selection", Command::Frame) {
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
        if keyed(ui, state, undo.as_deref().unwrap_or("Undo"), Command::Undo) {
            state.request_undo();
        }
    });
    let redo = state
        .history
        .redo_label()
        .map(|label| format!("Redo {label}"));
    ui.add_enabled_ui(redo.is_some(), |ui| {
        if keyed(ui, state, redo.as_deref().unwrap_or("Redo"), Command::Redo) {
            state.request_redo();
        }
    });
    ui.separator();
    if keyed(ui, state, "Settings…", Command::Settings) {
        state.open_settings(None);
    }
}

/// What can be spawned. Shared with the hierarchy's add button, so the two
/// lists cannot drift apart.
pub fn spawn_entries(ui: &mut egui::Ui, state: &mut EditorState) {
    if ui
        .add(egui::Button::new(icons::with_label(
            icons::folder(),
            None,
            "Folder",
        )))
        .clicked()
    {
        state.request_spawn(SpawnKind::Folder);
        ui.close_menu();
    }
    ui.separator();
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
        ("Spot light", SpawnKind::SpotLight, icons::lightbulb()),
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
        super::scenes::each_switch(world, |label, on| {
            ui.checkbox(on, label);
        });
        ui.checkbox(
            &mut world.resource_mut::<ShadowSettings>().debug_cascades,
            "Tint cascades",
        )
        .on_hover_text("Colour each shadow cascade, to see where the splits fall");
    });
    ui.separator();
    if command(ui, "Environment and exposure…", "") {
        dock.activate(Tab::Environment);
    }
    if command(ui, "Profiler…", "") {
        dock.activate(Tab::Performance);
    }
}

#[cfg(not(feature = "scripting"))]
fn scripts(ui: &mut egui::Ui, _world: &mut World, _state: &mut EditorState) {
    ui.add_enabled(false, egui::Button::new("Reload scripts"))
        .on_disabled_hover_text("This build was compiled without the `scripting` feature");
}

#[cfg(feature = "scripting")]
fn scripts(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState) {
    use crate::build_watcher::BuildStatus;

    if keyed(ui, state, "Reload scripts", Command::ReloadScripts) {
        state.request_script_reload();
    }
    if let Some(mut status) = world.get_resource_mut::<BuildStatus>() {
        ui.checkbox(&mut status.auto_reload, "Reload after a rebuild");
        ui.separator();
        let (color, word) = super::status_bar::build_word(&status.state);
        icons::labelled(ui, icons::circle_dot(), color, word);
    }
}

/// A picker per kind of viewport icon, each drawn inline in a submenu of its
/// own. egui closes a menu at any press outside its own areas, its submenus
/// included, and a colour button's popup is an area of its own, so a picker
/// opened from a button here would close the menu at its first click.
fn icon_colours(ui: &mut egui::Ui, tints: &mut IconTints) {
    let IconTints {
        light,
        sound,
        listener,
        camera,
        empty,
    } = tints;
    for (icon, label, rgb) in [
        (icons::lightbulb(), "Lights", light),
        (icons::speaker_high(), "Sounds", sound),
        (icons::ear(), "Listeners", listener),
        (icons::camera(), "Cameras", camera),
        (icons::circle(), "Empties", empty),
    ] {
        let [r, g, b] = *rgb;
        let swatch = egui::Color32::from_rgb(r, g, b);
        ui.menu_button(icons::with_label(icon, Some(swatch), label), |ui| {
            let mut colour = swatch;
            if egui::color_picker::color_picker_color32(
                ui,
                &mut colour,
                egui::color_picker::Alpha::Opaque,
            ) {
                *rgb = [colour.r(), colour.g(), colour.b()];
            }
        });
    }
    ui.separator();
    let changed = *tints != IconTints::default();
    if ui
        .add_enabled(changed, egui::Button::new("Reset colours"))
        .clicked()
    {
        *tints = IconTints::default();
    }
}

fn view(ui: &mut egui::Ui, state: &mut EditorState, themes: &ThemeSet, dock: &mut Dock) {
    if keyed(ui, state, "Command palette…", Command::Palette) {
        state.palette.toggle();
    }
    ui.separator();
    for tab in Tab::ALL {
        let mut open = dock.is_open(tab);
        if ui.checkbox(&mut open, tab.title()).changed() {
            dock.toggle(tab);
        }
    }
    ui.separator();
    ui.checkbox(&mut state.show_stats, "Frame stats overlay");
    ui.checkbox(&mut state.show_colliders, "Collider outlines");
    ui.menu_button("Draw mode", |ui| {
        use crate::gfx::vulkan::wireframe::Wireframe;
        for (mode, label) in [
            (None, "Shaded"),
            (Some(Wireframe::OverShaded), "Shaded wireframe"),
            (Some(Wireframe::Alone), "Wireframe"),
        ] {
            if ui.radio(state.wireframe == mode, label).clicked() {
                state.wireframe = mode;
                ui.close_menu();
            }
        }
    });
    ui.menu_button("Icon colours", |ui| {
        icon_colours(ui, &mut state.settings.appearance.icons)
    });
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
    if keyed(ui, state, "Reset layout", Command::ResetLayout) {
        dock.reset();
    }
}
