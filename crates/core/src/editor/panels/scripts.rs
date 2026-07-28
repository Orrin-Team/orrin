//! Scripting panel: swap in a freshly built game assembly without restarting,
//! and watch what rebuild-on-save is doing.
//!
//! The button only *requests* a reload — the swap destroys and re-creates
//! managed objects, so `App` runs it at the start of the next script phase
//! rather than from inside the UI pass. The watcher's auto-reload toggle takes
//! the same route: it decides whether a green build raises that request, never
//! whether a swap happens here.

use orrin_ecs::World;

use crate::build_watcher::{BuildState, BuildStatus};
use crate::editor::state::EditorState;
use crate::scene::ScriptComponent;

const ERROR: egui::Color32 = egui::Color32::from_rgb(0xE0, 0x5A, 0x4A);
const OK: egui::Color32 = egui::Color32::from_rgb(0x7A, 0xC0, 0x8A);

pub fn show(ctx: &egui::Context, world: &mut World, state: &mut EditorState) {
    let mut live = 0usize;
    let mut faulted = 0usize;
    world.query::<&ScriptComponent>().for_each(|_, script| {
        live += 1;
        if script.faulted {
            faulted += 1;
        }
    });

    egui::Window::new("Scripts")
        .default_pos(egui::pos2(12.0, 232.0))
        .default_size(egui::vec2(280.0, 120.0))
        .show(ctx, |ui| {
            if ui.button("Reload scripts").clicked() {
                state.request_script_reload();
            }
            if let Some(mut status) = world.get_resource_mut::<BuildStatus>() {
                ui.checkbox(&mut status.auto_reload, "Reload after a successful rebuild");
                ui.separator();
                show_build_state(ui, &status);
            }
            ui.separator();
            ui.label(format!("{live} behaviour(s) attached"));
            if faulted > 0 {
                ui.colored_label(ERROR, format!("{faulted} faulted — a reload clears them"));
            }
        });
}

fn show_build_state(ui: &mut egui::Ui, status: &BuildStatus) {
    match &status.state {
        BuildState::Idle => {
            ui.weak("Watching for script changes.");
        }
        BuildState::Building => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Rebuilding…");
            });
        }
        BuildState::Succeeded => {
            let took = status
                .last_duration
                .map_or_else(String::new, |d| format!(" ({:.1}s)", d.as_secs_f32()));
            ui.colored_label(OK, format!("Build up to date{took}"));
        }
        BuildState::Failed => {
            ui.colored_label(
                ERROR,
                format!("Build failed — {} diagnostic(s)", status.diagnostics.len()),
            );
            ui.weak("Still running the previous build.");
            egui::CollapsingHeader::new("Compiler output")
                .default_open(true)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(120.0)
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            for line in &status.diagnostics {
                                ui.colored_label(ERROR, line);
                            }
                        });
                });
        }
        BuildState::Off(reason) => {
            ui.weak("Rebuild-on-save is off.");
            ui.weak(reason);
        }
    }
}
