//! Right panel: edit the selected entity. Each section takes a fresh,
//! short-lived borrow of the world so the `RefCell` component storage never
//! double-borrows.

use glam::{EulerRot, Quat};

use orrin_ecs::{Entity, World};
use orrin_registry::Registry;

use super::{WIDTH_RANGE, color_row, vec3_row};
use crate::editor::icons;
use crate::editor::state::EditorState;
#[cfg(feature = "scripting")]
use crate::scene::ScriptComponent;
use crate::scene::{
    Assets, Light, LocalTransform, LogBuffer, LogLevel, MaterialHandle, MeshHandle, Name, Time,
    WorldTransform,
};

pub fn show(ctx: &egui::Context, world: &mut World, state: &mut EditorState, registry: &Registry) {
    egui::SidePanel::right("inspector")
        .resizable(true)
        .default_width(280.0)
        .width_range(WIDTH_RANGE)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.heading("Inspector");
            ui.separator();

            let Some(entity) = state.selected else {
                ui.weak("Select an entity in the hierarchy.");
                return;
            };
            if !world.is_alive(entity) {
                state.selected = None;
                return;
            }

            name_section(ui, world, entity);
            transform_section(ui, world, entity);
            mesh_material_section(ui, world, entity);
            light_section(ui, world, entity);
            #[cfg(feature = "scripting")]
            script_section(ui, world, entity);
            actions_section(ui, world, registry, state, entity);
        });
}

/// What can be done to the selected entity as a whole, rather than to one of
/// its components.
fn actions_section(
    ui: &mut egui::Ui,
    world: &mut World,
    registry: &Registry,
    state: &mut EditorState,
    entity: Entity,
) {
    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("Dump to console").clicked() {
            dump(world, registry, entity);
        }
        // The hierarchy deletes the row under the pointer; this deletes the
        // thing being looked at. Both go through the same deferred request —
        // despawning while a panel is iterating the tree is what that defers.
        let name = world.get::<Name>(entity).map_or_else(
            || format!("entity {}", entity.index()),
            |name| name.0.clone(),
        );
        let trash = icons::inline(ui, icons::trash(), icons::ROW);
        if ui
            .add(egui::Button::image_and_text(trash, "Delete"))
            .on_hover_text(format!("Despawn {name}"))
            .clicked()
        {
            state.request_despawn(entity);
        }
    });
}

/// Print the entity exactly as the registry sees it.
///
/// Every section above names a concrete component type; this one names none —
/// it asks each registered type whether it's present and prints whatever comes
/// back. That difference is the point of the registry, and until the inspector
/// itself is registry-driven this button is how a component's registration gets
/// eyeballed.
pub(super) fn dump(world: &mut World, registry: &Registry, entity: Entity) {
    let mut text = String::new();
    orrin_registry::write_entity(&mut text, registry, world, entity);

    // Both resource borrows are taken after the dump is built, so no component
    // borrow is still open when the log is written.
    let frame = world
        .get_resource::<Time>()
        .map_or(0, |time| time.frame_count());
    if let Some(mut log) = world.get_resource_mut::<LogBuffer>() {
        log.push(LogLevel::Info, text, frame);
    }
}

fn name_section(ui: &mut egui::Ui, world: &World, entity: Entity) {
    if let Some(mut name) = world.get_mut::<Name>(entity) {
        ui.horizontal(|ui| {
            ui.label("Name");
            ui.text_edit_singleline(&mut name.0);
        });
    }
    ui.weak(format!("id {}", entity.index()));
}

fn transform_section(ui: &mut egui::Ui, world: &World, entity: Entity) {
    let Some(mut t) = world.get_mut::<LocalTransform>(entity) else {
        return;
    };
    egui::CollapsingHeader::new("Transform")
        .default_open(true)
        .show(ui, |ui| {
            vec3_row(ui, "Position", &mut t.translation, 0.05);

            // Edited as Euler degrees in the same convention as the C# scripting API
            // (`Quaternion.Euler` / `eulerAngles`): Z-X-Y application, i.e. glam's
            // intrinsic `EulerRot::YXZ`. Matching it means the numbers shown here equal
            // a script's `eulerAngles` for the same rotation. Mind the reordering: YXZ
            // takes/yields angles as (yaw=Y, pitch=X, roll=Z), but the three fields are
            // laid out X, Y, Z (pitch, yaw, roll).
            let (yaw, pitch, roll) = t.rotation.to_euler(EulerRot::YXZ);
            let mut euler = [pitch.to_degrees(), yaw.to_degrees(), roll.to_degrees()];
            let mut changed = false;
            ui.horizontal(|ui| {
                ui.label("Rotation");
                for angle in &mut euler {
                    changed |= ui
                        .add(egui::DragValue::new(angle).speed(0.5).suffix("°"))
                        .changed();
                }
            });
            if changed {
                t.rotation = Quat::from_euler(
                    EulerRot::YXZ,
                    euler[1].to_radians(), // yaw   (Y)
                    euler[0].to_radians(), // pitch (X)
                    euler[2].to_radians(), // roll  (Z)
                );
            }

            vec3_row(ui, "Scale", &mut t.scale, 0.05);

            // The fields above are parent-relative. For anything with a parent
            // that makes them baffling on their own — a cube sitting visibly at
            // x=12 reads "Position 0". The world position is the reconciling
            // number, and it is read-only because it is derived.
            if !crate::scene::is_transform_root(world, entity)
                && let Some(world_transform) = world.get::<WorldTransform>(entity)
            {
                let position = world_transform.translation();
                ui.horizontal(|ui| {
                    ui.weak("World");
                    ui.weak(format!(
                        "{:.2}, {:.2}, {:.2}",
                        position.x, position.y, position.z
                    ));
                });
            }
        });
}

fn mesh_material_section(ui: &mut egui::Ui, world: &mut World, entity: Entity) {
    if !world.has::<MeshHandle>(entity) {
        return;
    }
    egui::CollapsingHeader::new("Mesh & Material")
        .default_open(true)
        .show(ui, |ui| {
            if let Some(handle) = mesh_picker(ui, world, entity) {
                world.insert(entity, handle);
            }
            if let Some(handle) = material_picker(ui, world, entity) {
                world.insert(entity, handle);
            }
        });
}

/// Returns a new handle if the user picked a different one. Drops all world
/// borrows before returning, so the caller can safely `insert`.
fn mesh_picker(ui: &mut egui::Ui, world: &World, entity: Entity) -> Option<MeshHandle> {
    let mut options: Vec<(String, MeshHandle)> = world
        .resource::<Assets>()
        .meshes()
        .map(|(n, h)| (n.to_owned(), h))
        .collect();
    options.sort_by(|a, b| a.0.cmp(&b.0));

    let current = world.get::<MeshHandle>(entity).map(|h| *h);
    let mut chosen = current;
    let label = name_of(&options, current);

    egui::ComboBox::from_label("Mesh")
        .selected_text(label)
        .show_ui(ui, |ui| {
            for (name, handle) in &options {
                ui.selectable_value(&mut chosen, Some(*handle), name.as_str());
            }
        });

    (chosen != current).then(|| chosen).flatten()
}

fn material_picker(ui: &mut egui::Ui, world: &World, entity: Entity) -> Option<MaterialHandle> {
    let mut options: Vec<(String, MaterialHandle)> = world
        .resource::<Assets>()
        .materials()
        .map(|(n, h)| (n.to_owned(), h))
        .collect();
    options.sort_by(|a, b| a.0.cmp(&b.0));

    let current = world.get::<MaterialHandle>(entity).map(|h| *h);
    let mut chosen = current;
    let label = name_of(&options, current);

    egui::ComboBox::from_label("Material")
        .selected_text(label)
        .show_ui(ui, |ui| {
            for (name, handle) in &options {
                ui.selectable_value(&mut chosen, Some(*handle), name.as_str());
            }
        });

    (chosen != current).then(|| chosen).flatten()
}

fn name_of<H: Copy + PartialEq>(options: &[(String, H)], handle: Option<H>) -> String {
    handle
        .and_then(|h| options.iter().find(|(_, opt)| *opt == h))
        .map(|(name, _)| name.clone())
        .unwrap_or_else(|| "—".to_owned())
}

#[cfg(feature = "scripting")]
fn script_section(ui: &mut egui::Ui, world: &World, entity: Entity) {
    let Some(mut script) = world.get_mut::<ScriptComponent>(entity) else {
        return;
    };
    egui::CollapsingHeader::new("Script")
        .default_open(true)
        .show(ui, |ui| {
            ui.weak(script.type_name.clone());
            // Only the desired state is written here; the script tick sees the
            // change next frame and dispatches OnEnable/OnDisable itself.
            ui.checkbox(&mut script.enabled, "Enabled");
            if script.faulted {
                // A hook threw and the engine disabled the script (see the log
                // for the exception). Clearing re-arms it from next tick — the
                // manual counterpart to "re-enable on hot reload".
                icons::labelled(ui, icons::warning(), crate::editor::theme::ERROR, "faulted");
                if ui.button("Clear fault").clicked() {
                    script.faulted = false;
                }
            } else {
                ui.weak(match (script.active, script.started) {
                    (true, _) => "active",
                    (false, true) => "inactive",
                    (false, false) => "not started",
                });
            }
        });
}

fn light_section(ui: &mut egui::Ui, world: &World, entity: Entity) {
    let Some(mut light) = world.get_mut::<Light>(entity) else {
        return;
    };
    egui::CollapsingHeader::new("Light")
        .default_open(true)
        .show(ui, |ui| match &mut *light {
            Light::Directional { color, intensity } => {
                color_row(ui, "Color", color);
                ui.add(egui::Slider::new(intensity, 0.0..=20.0).text("Intensity"));
            }
            Light::Point {
                color,
                intensity,
                range,
            } => {
                color_row(ui, "Color", color);
                ui.add(egui::Slider::new(intensity, 0.0..=50.0).text("Intensity"));
                ui.add(egui::Slider::new(range, 0.0..=100.0).text("Range"));
            }
        });
}
