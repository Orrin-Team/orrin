use orrin_ecs::World;

use crate::editor::state::{EditorState, SceneRequest};
use crate::editor::theme;
use crate::scene::UnknownComponents;

pub fn show(ctx: &egui::Context, world: &World, state: &mut EditorState) {
    egui::Window::new("Scene")
        .default_pos(egui::pos2(12.0, 12.0))
        .default_width(320.0)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("File");
                ui.text_edit_singleline(&mut state.scene_path);
            });

            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    state.request_scene(SceneRequest::Save);
                }
                if ui.button("Load").clicked() {
                    state.request_scene(SceneRequest::Load);
                }
                ui.weak(format!("{} entities", world.entities().count()));
            });

            // Surfaced rather than hidden: these are components this build could
            // not apply and is carrying through to the next save. Silence would
            // make the round trip look lossless when it is merely non-destructive.
            let carried: usize = world
                .entities()
                .filter_map(|e| world.get::<UnknownComponents>(e).map(|u| u.0.len()))
                .sum();
            if carried > 0 {
                ui.separator();
                ui.colored_label(
                    theme::LOG_WARN,
                    format!("⚠ carrying {carried} unapplied component(s)"),
                );
                ui.weak("Preserved on save. See the console for which.");
            }
        });
}
