//! The Scene Manager: which scene is open and whether it is saved, where its
//! renderer settings are, what is loaded beside it while the game plays, and
//! the project's other scenes, to open or to start a new one.
//!
//! A tool like any other, so it docks; and one a user reaches for while
//! working in the hierarchy it shares a column with, so it can also leave the
//! dock for a window of its own from the button in its header. Both moves are
//! *requested* (`DockRequest`): the button is drawn inside the dock, which
//! cannot rearrange itself while drawing it.
//!
//! The header answers the two questions the window is opened to ask — which
//! scene, and is it saved — before anything else, and holds the commands on
//! that scene. Everything below it scrolls.
//!
//! It shows the renderer's settings rather than drawing every slider again:
//! the Environment tab already edits them, into the scene when the scene has
//! settings (see `environment::edit_settings`). What only this window says is
//! *whether* the scene has them — whether the evening's tuning is saved with
//! it or lost on the next load — beside the few switches that decide which
//! passes the frame runs at all.
//!
//! Opening another scene over unsaved work asks first, in the list itself
//! rather than in a modal: the question is about the row just clicked, and
//! the answer is usually "save it", which is one of the choices.
//!
//! The list comes from the asset database, which already keeps it current as
//! files are added and renamed; outside a project there is none, and the
//! window says so rather than walking the disk every frame.

use orrin_ecs::World;

use crate::asset_db::{AssetDatabase, AssetKind};
use crate::editor::icons;
use crate::editor::state::{DockRequest, EditorState, OpenChoice, SceneRequest};
use crate::editor::theme;
use crate::editor::widgets::{self, Tool};
use crate::scene::loaded::name_of;
use crate::scene::render_settings;
use crate::scene::{
    BloomSettings, FogSettings, LoadedScenes, ShadowSettings, SsaoSettings, TaaSettings,
};

use super::environment;
use crate::editor::dock::Tab;

/// A row in the project's list: a step taller than a control, so a name and a
/// button beside it both fit with room to click either.
const ROW: f32 = 26.0;

pub fn body(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState, floating: bool) {
    header(ui, state, floating);
    ui.add_space(2.0);
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            renderer(ui, world, state);
            if state.playing() {
                loaded(ui, world);
            }
            project(ui, world, state);
        });
}

/// The open scene's name, file and save state, its commands, and the button
/// that floats or docks the tool.
fn header(ui: &mut egui::Ui, state: &mut EditorState, floating: bool) {
    let weak = ui.visuals().weak_text_color();
    ui.horizontal(|ui| {
        ui.label(icons::text(icons::globe(), icons::ROW, weak));
        let name = name_of(&state.scene_path);
        ui.add(egui::Label::new(egui::RichText::new(name).size(15.0).strong()).truncate());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (icon, hint, request) = if floating {
                (
                    icons::arrow_square_in(),
                    "Dock back beside the Hierarchy",
                    DockRequest::Dock(Tab::Scenes),
                )
            } else {
                (
                    icons::arrow_square_out(),
                    "Open in a window of its own",
                    DockRequest::Float(Tab::Scenes),
                )
            };
            if widgets::tool_button(ui, icon, None, None, Tool::Off, hint).clicked() {
                state.request_dock(request);
            }
            if state.scene_dirty() {
                pill(ui, theme::PENDING, "Unsaved")
                    .on_hover_text("Changes since the last save; Ctrl+S saves them");
            } else {
                pill(ui, theme::OK, "Saved");
            }
        });
    });
    ui.add(egui::Label::new(egui::RichText::new(&state.scene_path).small().color(weak)).truncate())
        .on_hover_text(&state.scene_path);

    let playing = state.playing();
    let recoverable = state.autosave.path(&state.scene_path).is_file();
    let live = |on: bool| {
        if on && !playing {
            Tool::Command
        } else {
            Tool::Disabled
        }
    };
    let why = |hint: &'static str| if playing { "Stop the game first" } else { hint };
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        let text = |word: &str| Some(egui::RichText::new(word));
        if widgets::tool_button(
            ui,
            icons::floppy_disk(),
            None,
            text("Save"),
            live(true),
            why("Save (Ctrl+S)"),
        )
        .clicked()
        {
            state.request_scene(SceneRequest::Save);
        }
        if widgets::tool_button(
            ui,
            icons::refresh(),
            None,
            text("Reload"),
            live(true),
            why("Load the file again, discarding unsaved changes"),
        )
        .clicked()
        {
            state.request_scene(SceneRequest::Load);
        }
        let hint = if recoverable {
            "Load the last autosave in place of the file, as unsaved work"
        } else {
            "There is no unsaved work to recover for this scene"
        };
        if widgets::tool_button(
            ui,
            icons::rotate_ccw(),
            None,
            text("Recover"),
            live(recoverable),
            why(hint),
        )
        .clicked()
        {
            state.request_scene(SceneRequest::Recover);
        }
    });
}

fn renderer(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState) {
    widgets::section(
        ui,
        ui.id().with("renderer"),
        icons::sun(),
        theme::KIND_LIGHT,
        "Renderer",
        None,
        |ui| {
            let source = render_settings::source(world);
            ui.horizontal_wrapped(|ui| match source {
                Some(entity) => {
                    pill(ui, theme::OK, "Saved with the scene");
                    if widgets::tool_button(
                        ui,
                        icons::pointer(),
                        None,
                        Some(egui::RichText::new("Select")),
                        Tool::Command,
                        "Show the Render Settings entity in the inspector",
                    )
                    .clicked()
                    {
                        state.selection.set(entity);
                    }
                }
                None => {
                    pill(ui, theme::PENDING, "This session only").on_hover_text(
                        "The renderer's settings are not part of the scene and are lost on load",
                    );
                    if widgets::tool_button(
                        ui,
                        icons::plus(),
                        None,
                        Some(egui::RichText::new("Save with scene")),
                        Tool::Command,
                        "Add a Render Settings entity holding them, saved with the scene",
                    )
                    .clicked()
                    {
                        state.request_add_render_settings();
                    }
                }
            });
            ui.add_space(2.0);
            environment::edit_settings(world, state, |world| switches(ui, world));
            if widgets::tool_button(
                ui,
                icons::sliders(),
                None,
                Some(egui::RichText::new("All settings…")),
                Tool::Command,
                "Open the Environment tab, which edits every setting",
            )
            .clicked()
            {
                state.request_dock(DockRequest::Show(Tab::Environment));
            }
        },
    );
}

/// The settings that decide which passes the frame runs at all, where the
/// rest only tune a pass already running.
fn switches(ui: &mut egui::Ui, world: &World) {
    let mut row = |label: &str, value: &mut bool| {
        widgets::property(ui, label, |ui| ui.checkbox(value, ""));
    };
    row(
        "Ambient occlusion",
        &mut world.resource_mut::<SsaoSettings>().enabled,
    );
    row(
        "Temporal AA",
        &mut world.resource_mut::<TaaSettings>().enabled,
    );
    row("Bloom", &mut world.resource_mut::<BloomSettings>().enabled);
    row(
        "Shadows",
        &mut world.resource_mut::<ShadowSettings>().enabled,
    );
    row(
        "Volumetric fog",
        &mut world.resource_mut::<FogSettings>().volumetric,
    );
}

fn loaded(ui: &mut egui::Ui, world: &World) {
    let weak = ui.visuals().weak_text_color();
    widgets::section(
        ui,
        ui.id().with("loaded"),
        icons::layers(),
        weak,
        "Loaded",
        None,
        |ui| {
            let Some(loaded) = world.get_resource::<LoadedScenes>() else {
                return;
            };
            let active = loaded.active().handle;
            for scene in loaded.iter() {
                ui.horizontal(|ui| {
                    ui.label(icons::text(icons::globe(), icons::ROW - 2.0, weak));
                    ui.label(&scene.name);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if scene.handle == active {
                            pill(ui, ui.visuals().selection.stroke.color, "Active");
                        } else if scene.persistent {
                            pill(ui, theme::PREFAB, "Persistent");
                        } else {
                            pill(ui, weak, "Additive");
                        }
                    });
                });
            }
        },
    );
}

fn project(ui: &mut egui::Ui, world: &World, state: &mut EditorState) {
    let weak = ui.visuals().weak_text_color();
    widgets::section(
        ui,
        ui.id().with("project"),
        icons::folder(),
        weak,
        "Project scenes",
        None,
        |ui| {
            let Some(db) = world.get_resource::<AssetDatabase>() else {
                empty(
                    ui,
                    "Open a project to list its scenes. The open scene still saves.",
                );
                return;
            };
            let mut scenes: Vec<(&str, orrin_registry::AssetId)> = db
                .iter()
                .filter(|entry| entry.kind == AssetKind::Scene)
                .map(|entry| (entry.path.as_str(), entry.id))
                .collect();
            scenes.sort_unstable();
            if scenes.is_empty() {
                empty(ui, "No scenes yet. Name one below to start it.");
            } else {
                widgets::search(ui, &mut state.scenes_query, "Filter scenes");
                ui.add_space(2.0);
                list(ui, &db, &scenes, state);
            }
            if let Some(path) = state.pending_open().map(str::to_owned) {
                confirm(ui, state, &path);
            }
            ui.add_space(6.0);
            new_scene(ui, world, state);
        },
    );
}

fn list(
    ui: &mut egui::Ui,
    db: &AssetDatabase,
    scenes: &[(&str, orrin_registry::AssetId)],
    state: &mut EditorState,
) {
    let query = state.scenes_query.trim().to_lowercase();
    let current = std::fs::canonicalize(&state.scene_path).ok();
    let persistent = state
        .persistent_scene
        .as_ref()
        .and_then(|path| std::fs::canonicalize(path).ok());
    let chosen_id = ui.id().with("chosen scene");
    let mut chosen: Option<String> = ui.data(|data| data.get_temp(chosen_id));
    let playing = state.playing();
    let mut shown = 0;
    for &(rel, id) in scenes {
        let name = name_of(rel);
        if !query.is_empty() && !name.to_lowercase().contains(&query) {
            continue;
        }
        shown += 1;
        let path = db.dir().join(rel);
        let here = std::fs::canonicalize(&path).ok();
        let open = here.is_some() && here == current;
        let kept = here.is_some() && here == persistent;
        let marked = chosen.as_deref() == Some(rel);
        let folder = rel.rsplit_once('/').map_or("", |(folder, _)| folder);

        let response = row(ui, &name, folder, open, kept, marked);
        if response.clicked() {
            chosen = Some(rel.to_owned());
        }
        let path = path.to_string_lossy().into_owned();
        // A button only on the row in hand, so the list reads as names and the
        // one command a row has is where the pointer already is.
        let offer = !open && !playing && (marked || response.hovered());
        let button = egui::Rect::from_min_max(
            egui::pos2(response.rect.right() - 58.0, response.rect.top() + 3.0),
            egui::pos2(response.rect.right() - 4.0, response.rect.bottom() - 3.0),
        );
        let opened = offer && ui.put(button, egui::Button::new("Open")).clicked();
        if opened || (!open && !playing && response.double_clicked()) {
            state.request_open_scene_checked(path.clone());
        }
        response
            .on_hover_text(format!("{rel}\nDouble-click to open it"))
            .context_menu(|ui| {
                if ui
                    .add_enabled(!open && !playing, egui::Button::new("Open"))
                    .clicked()
                {
                    state.request_open_scene_checked(path.clone());
                    ui.close_menu();
                }
                if ui.button("Show in Assets").clicked() {
                    state.reveal_asset(id);
                    ui.close_menu();
                }
                if ui.button("Copy path").clicked() {
                    ui.ctx().copy_text(path.clone());
                    ui.close_menu();
                }
            });
    }
    if shown == 0 {
        empty(
            ui,
            &format!("No scene matches “{}”.", state.scenes_query.trim()),
        );
    }
    ui.data_mut(|data| match chosen {
        Some(chosen) => data.insert_temp(chosen_id, chosen),
        None => data.remove::<String>(chosen_id),
    });
}

/// One scene: a globe, its name, the folder it is in, and what it is to this
/// session — open now, or kept through every load.
fn row(
    ui: &mut egui::Ui,
    name: &str,
    folder: &str,
    open: bool,
    persistent: bool,
    marked: bool,
) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW), egui::Sense::click());
    let visuals = ui.visuals();
    let radius = visuals.widgets.inactive.corner_radius;
    if marked {
        ui.painter()
            .rect_filled(rect, radius, visuals.selection.bg_fill);
    } else if response.hovered() {
        ui.painter()
            .rect_filled(rect, radius, visuals.widgets.hovered.weak_bg_fill);
    }
    let weak = visuals.weak_text_color();
    let accent = visuals.selection.stroke.color;
    let glyph = egui::Rect::from_center_size(
        egui::pos2(rect.left() + 12.0, rect.center().y),
        egui::Vec2::splat(icons::ROW - 2.0),
    );
    icons::paint(ui, icons::globe(), glyph, if open { accent } else { weak });
    let font = egui::TextStyle::Body.resolve(ui.style());
    let label = ui.painter().text(
        egui::pos2(glyph.right() + 7.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        name,
        font.clone(),
        if open {
            visuals.strong_text_color()
        } else {
            visuals.text_color()
        },
    );
    let mut right = label.right() + 8.0;
    if !folder.is_empty() {
        right = ui
            .painter()
            .text(
                egui::pos2(right, rect.center().y),
                egui::Align2::LEFT_CENTER,
                folder,
                egui::TextStyle::Small.resolve(ui.style()),
                weak,
            )
            .right()
            + 8.0;
    }
    for (shown, colour, word) in [
        (open, accent, "Current"),
        (persistent, theme::PREFAB, "Persistent"),
    ] {
        if shown {
            right = paint_pill(ui, egui::pos2(right, rect.center().y), colour, word).right() + 4.0;
        }
    }
    response
}

/// What to do with the open scene's unsaved work before `path` replaces it.
fn confirm(ui: &mut egui::Ui, state: &mut EditorState, path: &str) {
    let current = name_of(&state.scene_path);
    let next = name_of(path);
    egui::Frame::new()
        .fill(theme::PENDING.gamma_multiply(0.12))
        .stroke(egui::Stroke::new(
            1.0_f32,
            theme::PENDING.gamma_multiply(0.5),
        ))
        .corner_radius(ui.visuals().widgets.inactive.corner_radius)
        .inner_margin(8)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(icons::text(
                    icons::warning_circle(),
                    icons::ROW - 2.0,
                    theme::PENDING,
                ));
                ui.label(format!(
                    "{current} has unsaved changes. Open {next} anyway?"
                ));
            });
            ui.horizontal_wrapped(|ui| {
                if ui.button("Save and open").clicked() {
                    state.resolve_open(OpenChoice::SaveFirst);
                }
                if ui.button("Open without saving").clicked() {
                    state.resolve_open(OpenChoice::Discard);
                }
                if ui.button("Cancel").clicked() {
                    state.resolve_open(OpenChoice::Cancel);
                }
            });
        });
}

/// A field for the new scene's name that says what is wrong with it as it is
/// typed, by the rule the request is checked by.
fn new_scene(ui: &mut egui::Ui, world: &World, state: &mut EditorState) {
    let id = ui.id().with("new scene name");
    let mut name = ui.data_mut(|data| data.get_temp::<String>(id).unwrap_or_default());
    let problem = (!name.trim().is_empty())
        .then(|| crate::editor::state::new_scene_path(world, &name).err())
        .flatten();
    let playing = state.playing();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let width = (ui.available_width() - 80.0).max(80.0);
        let field = ui.add_enabled(
            !playing,
            egui::TextEdit::singleline(&mut name)
                .hint_text("New scene name")
                .desired_width(width),
        );
        let ready = !playing && !name.trim().is_empty() && problem.is_none();
        let entered = field.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
        let hint = if playing {
            "Stop the game first"
        } else {
            "Write the scene under scenes/ and open it"
        };
        let state_of = if ready { Tool::Command } else { Tool::Disabled };
        let create = widgets::tool_button(
            ui,
            icons::file_plus(),
            None,
            Some(egui::RichText::new("Create")),
            state_of,
            hint,
        );
        if ready && (create.clicked() || entered) {
            state.request_new_scene(std::mem::take(&mut name));
        }
    });
    if let Some(problem) = problem {
        ui.colored_label(theme::ERROR, problem);
    }
    ui.data_mut(|data| data.insert_temp(id, name));
}

fn empty(ui: &mut egui::Ui, text: &str) {
    ui.add_space(4.0);
    ui.add(
        egui::Label::new(egui::RichText::new(text).color(ui.visuals().weak_text_color())).wrap(),
    );
    ui.add_space(4.0);
}

/// A word in a rounded tint of `colour`: a state, not a control.
fn pill(ui: &mut egui::Ui, colour: egui::Color32, word: &str) -> egui::Response {
    let galley = ui.painter().layout_no_wrap(
        word.to_owned(),
        egui::TextStyle::Small.resolve(ui.style()),
        colour,
    );
    let size = galley.size() + egui::vec2(12.0, 4.0);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::hover());
    paint_pill(ui, rect.left_center(), colour, word);
    response
}

/// [`pill`] painted with its left edge centred on `at`, for a row that lays
/// itself out by hand.
fn paint_pill(ui: &egui::Ui, at: egui::Pos2, colour: egui::Color32, word: &str) -> egui::Rect {
    let galley = ui.painter().layout_no_wrap(
        word.to_owned(),
        egui::TextStyle::Small.resolve(ui.style()),
        colour,
    );
    let size = galley.size() + egui::vec2(12.0, 4.0);
    let rect = egui::Rect::from_min_size(egui::pos2(at.x, at.y - size.y / 2.0), size);
    ui.painter()
        .rect_filled(rect, size.y / 2.0, colour.gamma_multiply(0.16));
    ui.painter()
        .galley(rect.min + egui::vec2(6.0, 2.0), galley, colour);
    rect
}
