//! The project's assets — for now its prefabs — each a row to drag into the
//! hierarchy.
//!
//! A row hands over the prefab's [`AssetId`], never its path: the hierarchy
//! and the inspector take that payload the way everything else takes a
//! reference to an asset.
//!
//! [`AssetId`]: orrin_registry::AssetId

use orrin_ecs::World;

use orrin_ecs::FxHashMap;
use orrin_registry::AssetId;

use crate::editor::state::EditorState;
use crate::editor::{icons, widgets};
use crate::scene::{PrefabInstance, Prefabs};

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
    widgets::search(ui, &mut state.assets_query, "Filter prefabs");
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

    let mut in_scene: FxHashMap<AssetId, usize> = FxHashMap::default();
    world
        .query::<&PrefabInstance>()
        .for_each(|_, instance| *in_scene.entry(instance.prefab).or_default() += 1);
    let query = state.assets_query.trim().to_lowercase();
    let shown: Vec<_> = prefabs
        .iter()
        .filter(|prefab| query.is_empty() || prefab.path.to_lowercase().contains(&query))
        .collect();
    if shown.is_empty() {
        ui.weak(format!(
            "No prefab matches “{}”.",
            state.assets_query.trim()
        ));
        return;
    }

    egui::ScrollArea::vertical()
        .auto_shrink(false)
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            for prefab in shown {
                let folder = prefab
                    .path
                    .rsplit_once('/')
                    .map_or("", |(folder, _)| folder);
                let count = in_scene.get(&prefab.id).copied().unwrap_or(0);
                let marked = state.revealed_asset == Some(prefab.id);
                let response = row(ui, prefab.name(), folder, &prefab.path, count, marked);
                if response.clicked() {
                    state.revealed_asset = Some(prefab.id);
                }
                if response.double_clicked() {
                    state.request_instantiate(prefab.id, None);
                }
                if response.drag_started() {
                    response.dnd_set_drag_payload(prefab.id);
                }
                if response.dragged()
                    && let Some(pointer) = ui.ctx().pointer_latest_pos()
                {
                    // What is being carried, beside the pointer, since the
                    // row itself stays put.
                    egui::Area::new(ui.id().with("dragged_prefab"))
                        .order(egui::Order::Tooltip)
                        .fixed_pos(pointer + egui::vec2(14.0, 6.0))
                        .show(ui.ctx(), |ui| {
                            egui::Frame::popup(ui.style()).show(ui, |ui| ui.label(prefab.name()));
                        });
                }
            }
        });
}

/// One prefab: its icon and name, the folder it is in, and how many of it the
/// scene holds at the far end.
fn row(
    ui: &mut egui::Ui,
    name: &str,
    folder: &str,
    path: &str,
    in_scene: usize,
    marked: bool,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), ROW),
        egui::Sense::click_and_drag(),
    );
    let visuals = ui.visuals();
    if marked {
        ui.painter()
            .rect_filled(rect, 0.0, visuals.selection.bg_fill);
    } else if response.hovered() || response.dragged() {
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
    let label = ui.painter().text(
        egui::pos2(glyph.right() + 6.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        name,
        font.clone(),
        text,
    );
    ui.painter().text(
        egui::pos2(label.right() + 8.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        folder,
        font.clone(),
        weak,
    );
    if in_scene > 0 {
        ui.painter().text(
            egui::pos2(rect.right() - widgets::PAD, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            format!("{in_scene} in scene"),
            font,
            weak,
        );
    }

    response.on_hover_text(format!(
        "{path}\nDrag onto the hierarchy or the scene to place it, or double-click"
    ))
}
