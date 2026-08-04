use orrin_ecs::World;

use crate::editor::icons;
use crate::editor::state::{EditorState, SceneRequest};
use crate::editor::theme;
use crate::scene::UnknownComponents;

pub fn body(ui: &mut egui::Ui, world: &World, state: &mut EditorState) {
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
        icons::labelled(
            ui,
            icons::warning(),
            theme::LOG_WARN,
            &format!("carrying {carried} unapplied component(s)"),
        );
        ui.weak("Preserved on save. See the console for which.");
    }
}
