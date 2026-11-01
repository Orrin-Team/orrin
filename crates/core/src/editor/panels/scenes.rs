//! The Scene Manager: which scene is open and where its renderer settings are,
//! what is loaded beside it while the game plays, and the project's other
//! scenes, to open or to start a new one.
//!
//! It shows the renderer's settings rather than drawing every slider again:
//! the Environment tab already edits them, and edits them into the scene when
//! the scene has settings (see `environment::edit_settings`). What only this
//! window can say is *whether* the scene has them — whether the evening's
//! tuning is saved with it or lost on the next load — so that is what it says
//! first, beside the few switches that decide what the frame runs at all.
//!
//! The list comes from the asset database, which already keeps it current as
//! files are added and renamed; outside a project there is none, and the
//! window says so rather than walking the disk every frame.

use orrin_ecs::World;

use super::environment;
use crate::asset_db::{AssetDatabase, AssetKind};
use crate::editor::icons;
use crate::editor::state::{EditorState, SceneRequest};
use crate::editor::theme;
use crate::editor::widgets;
use crate::scene::render_settings;
use crate::scene::{
    BloomSettings, FogSettings, LoadedScenes, ShadowSettings, SsaoSettings, TaaSettings,
};

pub fn body(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState) {
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            current(ui, state);
            renderer(ui, world, state);
            if state.playing() {
                loaded(ui, world);
            }
            project(ui, world, state);
        });
}

fn current(ui: &mut egui::Ui, state: &mut EditorState) {
    let weak = ui.visuals().weak_text_color();
    widgets::section(
        ui,
        ui.id().with("scene"),
        icons::globe(),
        weak,
        "Scene",
        None,
        |ui| {
            let name = crate::scene::loaded::name_of(&state.scene_path);
            if state.scene_dirty() {
                ui.strong(format!("{name} •"))
                    .on_hover_text("Unsaved changes");
            } else {
                ui.strong(name);
            }
            ui.add(
                egui::TextEdit::singleline(&mut state.scene_path)
                    .desired_width(f32::INFINITY)
                    .hint_text("scene file"),
            );
            let recoverable = state.autosave.path(&state.scene_path).is_file();
            ui.add_enabled_ui(!state.playing(), |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        state.request_scene(SceneRequest::Save);
                    }
                    if ui
                        .button("Reload")
                        .on_hover_text("Load the file again, discarding unsaved changes")
                        .clicked()
                    {
                        state.request_scene(SceneRequest::Load);
                    }
                    if ui
                        .add_enabled(recoverable, egui::Button::new("Recover autosave"))
                        .clicked()
                    {
                        state.request_scene(SceneRequest::Recover);
                    }
                });
            })
            .response
            .on_disabled_hover_text("Stop the game first: the scene is not what is playing");
        },
    );
}

fn renderer(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState) {
    let tint = theme::KIND_LIGHT;
    widgets::section(
        ui,
        ui.id().with("renderer"),
        icons::sun(),
        tint,
        "Renderer",
        None,
        |ui| {
            match render_settings::source(world) {
                Some(entity) => {
                    ui.horizontal_wrapped(|ui| {
                        icons::labelled(ui, icons::check(), theme::OK, "Saved with the scene");
                        if ui
                            .small_button("Select")
                            .on_hover_text("Show the Render Settings entity in the inspector")
                            .clicked()
                        {
                            state.selection.set(entity);
                        }
                    });
                }
                None => environment::unsaved_note(ui, state),
            }
            environment::edit_settings(world, state, |world| {
                switches(ui, world);
            });
            if ui
                .button("All settings…")
                .on_hover_text("Open the Environment tab, which edits every setting")
                .clicked()
            {
                state.request_environment_tab();
            }
        },
    );
}

/// The settings that decide which passes the frame runs at all, where the
/// rest only tune a pass already running.
fn switches(ui: &mut egui::Ui, world: &World) {
    ui.checkbox(
        &mut world.resource_mut::<SsaoSettings>().enabled,
        "Ambient occlusion",
    );
    ui.checkbox(
        &mut world.resource_mut::<TaaSettings>().enabled,
        "Temporal AA",
    );
    ui.checkbox(&mut world.resource_mut::<BloomSettings>().enabled, "Bloom");
    ui.checkbox(
        &mut world.resource_mut::<ShadowSettings>().enabled,
        "Shadows",
    );
    ui.checkbox(
        &mut world.resource_mut::<FogSettings>().volumetric,
        "Volumetric fog",
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
                    ui.label(&scene.name);
                    let role = if scene.handle == active {
                        "Active"
                    } else if scene.persistent {
                        "Persistent"
                    } else {
                        "Additive"
                    };
                    ui.weak(role);
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
                ui.weak("Open a project to list its scenes. The scene above still saves.");
                return;
            };
            let current = std::fs::canonicalize(&state.scene_path).ok();
            let persistent = state
                .persistent_scene
                .as_ref()
                .and_then(|path| std::fs::canonicalize(path).ok());
            let mut scenes: Vec<&str> = db
                .iter()
                .filter(|entry| entry.kind == AssetKind::Scene)
                .map(|entry| entry.path.as_str())
                .collect();
            scenes.sort_unstable();
            if scenes.is_empty() {
                ui.weak("No scenes under the assets directory yet.");
            }
            let playing = state.playing();
            for rel in scenes {
                let path = db.dir().join(rel);
                let here = std::fs::canonicalize(&path).ok();
                ui.horizontal(|ui| {
                    ui.label(icons::text(icons::globe(), icons::ROW, weak));
                    let name = crate::scene::loaded::name_of(rel);
                    let open = here.is_some() && here == current;
                    let row = ui
                        .add_enabled(!playing && !open, egui::Button::new(name).frame(false))
                        .on_hover_text(format!("{rel}\nClick to open it"))
                        .on_disabled_hover_text(if open {
                            "Open now"
                        } else {
                            "Stop the game first"
                        });
                    if row.clicked() {
                        state.request_open_scene(path.to_string_lossy());
                    }
                    if open {
                        ui.weak("Open");
                    }
                    if here.is_some() && here == persistent {
                        ui.weak("Persistent").on_hover_text(
                            "Loaded beside every scene while the game runs; set in orrin.toml",
                        );
                    }
                });
            }
            ui.add_space(4.0);
            new_scene(ui, state);
        },
    );
}

fn new_scene(ui: &mut egui::Ui, state: &mut EditorState) {
    let id = ui.id().with("new scene name");
    let mut name = ui.data_mut(|data| data.get_temp::<String>(id).unwrap_or_default());
    ui.add_enabled_ui(!state.playing(), |ui| {
        ui.horizontal(|ui| {
            let field = ui.add(
                egui::TextEdit::singleline(&mut name)
                    .hint_text("New scene name")
                    .desired_width(140.0),
            );
            let entered =
                field.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
            if (ui.button("Create").clicked() || entered) && !name.trim().is_empty() {
                state.request_new_scene(std::mem::take(&mut name));
            }
        });
    });
    ui.data_mut(|data| data.insert_temp(id, name));
}
