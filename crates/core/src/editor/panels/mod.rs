mod console;
mod environment;
mod hierarchy;
mod inspector;
mod performance;
mod scene;
#[cfg(feature = "scripting")]
mod scripts;

use glam::Vec3;

use orrin_ecs::World;
use orrin_registry::Registry;

use super::state::EditorState;

// Side/bottom panels only — no `CentralPanel`, so the center stays transparent
// and the 3D scene shows through behind the editor.
pub fn draw(
    ctx: &egui::Context,
    world: &mut World,
    state: &mut EditorState,
    registry: &Registry,
) {
    hierarchy::show(ctx, world, state);
    inspector::show(ctx, world, state, registry);
    environment::show(ctx, world);
    performance::show(ctx, world);
    #[cfg(feature = "scripting")]
    scripts::show(ctx, world, state);
    scene::show(ctx, world, state);
    console::show(ctx, world);
}

pub(super) fn vec3_row(ui: &mut egui::Ui, label: &str, v: &mut Vec3, speed: f32) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::DragValue::new(&mut v.x).speed(speed));
        ui.add(egui::DragValue::new(&mut v.y).speed(speed));
        ui.add(egui::DragValue::new(&mut v.z).speed(speed));
    });
}

pub(super) fn color_row(ui: &mut egui::Ui, label: &str, c: &mut Vec3) {
    let mut rgb = [c.x, c.y, c.z];
    ui.horizontal(|ui| {
        ui.label(label);
        if ui.color_edit_button_rgb(&mut rgb).changed() {
            *c = Vec3::from(rgb);
        }
    });
}
