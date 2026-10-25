//! The project's assets — for now its prefabs — each a row to drag into the
//! hierarchy.
//!
//! A row hands over the prefab's [`AssetId`], never its path: the hierarchy
//! and the inspector take that payload the way everything else takes a
//! reference to an asset.
//!
//! [`AssetId`]: orrin_registry::AssetId

use orrin_ecs::World;

use crate::editor::state::EditorState;
use crate::editor::{icons, widgets};
use crate::scene::Prefabs;

const ROW: f32 = 24.0;

pub fn body(ui: &mut egui::Ui, world: &World, state: &mut EditorState) {
    let Some(prefabs) = world.get_resource::<Prefabs>() else {
        ui.weak("No asset index.");
        return;
    };
    let weak = ui.visuals().weak_text_color();

    ui.horizontal(|ui| {
        if icons::button(ui, icons::refresh(), icons::ROW - 2.0, weak, false)
            .on_hover_text("Scan the assets directory again")
            .clicked()
        {
            state.request_prefab_rescan();
        }
        ui.weak(format!("{} prefab(s)", prefabs.iter().count()));
    });
    ui.add_space(4.0);

    if prefabs.iter().next().is_none() {
        let dir = prefabs.dir().map_or_else(
            || "the assets directory".to_owned(),
            |dir| dir.display().to_string(),
        );
        ui.weak(format!(
            "No prefabs under {dir}. A .fprefab is a scene file holding one entity tree."
        ));
        return;
    }

    egui::ScrollArea::vertical()
        .auto_shrink(false)
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            for prefab in prefabs.iter() {
                let folder = prefab
                    .path
                    .rsplit_once('/')
                    .map_or("", |(folder, _)| folder);
                row(ui, prefab.name(), folder, &prefab.path).dnd_set_drag_payload(prefab.id);
            }
        });
}

/// One prefab: its icon and name, and the folder it is in at the far end.
fn row(ui: &mut egui::Ui, name: &str, folder: &str, path: &str) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW), egui::Sense::drag());
    let visuals = ui.visuals();
    if response.hovered() || response.dragged() {
        ui.painter()
            .rect_filled(rect, 0.0, visuals.widgets.hovered.weak_bg_fill);
    }

    let weak = visuals.weak_text_color();
    let glyph = egui::Rect::from_center_size(
        egui::pos2(rect.left() + widgets::PAD + 7.0, rect.center().y),
        egui::Vec2::splat(icons::ROW - 2.0),
    );
    icons::paint(ui, icons::box_(), glyph, weak);
    let font = egui::TextStyle::Body.resolve(ui.style());
    let text = visuals.text_color();
    ui.painter().text(
        egui::pos2(glyph.right() + 6.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        name,
        font.clone(),
        text,
    );
    ui.painter().text(
        egui::pos2(rect.right() - widgets::PAD, rect.center().y),
        egui::Align2::RIGHT_CENTER,
        folder,
        font,
        weak,
    );

    response.on_hover_text(format!("{path}\nDrag onto the hierarchy to place it"))
}
