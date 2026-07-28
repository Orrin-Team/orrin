use orrin_ecs::World;

use crate::editor::state::{EditorState, SpawnKind};
use crate::scene::Name;

pub fn show(ctx: &egui::Context, world: &mut World, state: &mut EditorState) {
    egui::SidePanel::left("hierarchy")
        .resizable(true)
        .default_width(220.0)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.heading("Hierarchy");
            ui.separator();

            ui.horizontal(|ui| {
                ui.menu_button("➕  Add", |ui| {
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
                    ui.separator();
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

                let has_selection = state.selected.is_some();
                if ui
                    .add_enabled(has_selection, egui::Button::new("🗑  Delete"))
                    .clicked()
                {
                    if let Some(entity) = state.selected {
                        state.request_despawn(entity);
                    }
                }
            });

            ui.separator();

            // Snapshot the entity set so we only read the world while listing it.
            let entities: Vec<_> = world.entities().collect();
            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .show(ui, |ui| {
                    ui.with_layout(
                        egui::Layout::top_down_justified(egui::Align::LEFT),
                        |ui| {
                            for entity in entities {
                                let label = world
                                    .get::<Name>(entity)
                                    .map(|n| n.0.clone())
                                    .unwrap_or_else(|| format!("Entity {}", entity.index()));
                                let selected = state.selected == Some(entity);
                                if ui.selectable_label(selected, label).clicked() {
                                    state.selected = Some(entity);
                                }
                            }
                        },
                    );
                });
        });
}
