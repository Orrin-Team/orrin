//! Scripting panel: swap in a freshly built game assembly without restarting.
//!
//! The button only *requests* a reload — the swap destroys and re-creates
//! managed objects, so `App` runs it at the start of the next script phase
//! rather than from inside the UI pass.

use orrin_ecs::World;

use crate::editor::state::EditorState;
use crate::scene::ScriptComponent;

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
        .default_size(egui::vec2(240.0, 72.0))
        .show(ctx, |ui| {
            if ui.button("Reload scripts").clicked() {
                state.request_script_reload();
            }
            ui.weak("Build the game assembly first; reload swaps the built DLL.");
            ui.separator();
            ui.label(format!("{live} behaviour(s) attached"));
            if faulted > 0 {
                ui.colored_label(
                    egui::Color32::from_rgb(0xE0, 0x5A, 0x4A),
                    format!("{faulted} faulted — a reload clears them"),
                );
            }
        });
}
