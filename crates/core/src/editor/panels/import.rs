//! The Import tab: the asset the browser marks, as the database sees it — what
//! it is, how its last import went and what that made — and its import
//! settings.
//!
//! The settings are a draft until Apply writes them to the asset's sidecar,
//! because every change to a texture's settings is a full reimport, and a
//! toggle passed through on the way to the setting wanted should not cost one.
//! Applying writes a file rather than editing the scene, so it is not on the
//! undo stack: Revert, before Apply, is the way back.

use orrin_ecs::World;
use orrin_registry::AssetId;

use super::assets::log_error;
use crate::asset_db::cache::Summary;
use crate::asset_db::{AssetDatabase, AssetKind, Entry, Settings, Status, Usage};
use crate::editor::previews::Previews;
use crate::editor::state::EditorState;
use crate::editor::{icons, theme};

/// The picture's side: twice a browser tile's, which is as far as a 64 px
/// thumbnail stretches before it reads as blurred rather than large.
const PICTURE: f32 = 128.0;

/// The limits offered for a texture's longest side; `0` is none.
const SIZES: [u32; 7] = [0, 256, 512, 1024, 2048, 4096, 8192];

const USAGES: [(Usage, &str); 3] = [
    (Usage::Color, "Colour (sRGB)"),
    (Usage::Normal, "Normal map"),
    (Usage::Data, "Data (linear)"),
];

/// The settings being edited, and the asset settings they started from: a
/// draft whose asset changed underneath it, by an Apply or by hand, starts
/// again from what the asset has now.
#[derive(Clone, Copy)]
struct Draft {
    id: AssetId,
    from: Settings,
    settings: Settings,
}

enum Action {
    Apply(Settings),
    Reimport,
}

pub fn body(ui: &mut egui::Ui, world: &World, state: &mut EditorState, previews: &mut Previews) {
    let Some(db) = world.get_resource::<AssetDatabase>() else {
        ui.weak("Import settings belong to a project's assets. Open a project to see them.");
        return;
    };
    let Some(entry) = state.revealed_asset.and_then(|id| db.get(id)) else {
        ui.weak(
            "Select an asset in the Assets tab to see how it is imported. Files dropped on the \
             Assets tab from your file explorer are copied into the project.",
        );
        return;
    };
    let id = entry.id;
    let memory = ui.id().with("import_draft");
    let kept: Option<Draft> = ui.data_mut(|data| data.get_temp(memory));
    let mut draft = kept
        .filter(|draft| draft.id == id && draft.from == entry.settings)
        .unwrap_or(Draft {
            id,
            from: entry.settings,
            settings: entry.settings,
        });
    let mut action = None;
    egui::ScrollArea::vertical()
        .auto_shrink(false)
        .show(ui, |ui| {
            overview(ui, &db, entry, previews);
            ui.add_space(10.0);
            settings(ui, entry, &mut draft, &mut action);
        });
    ui.data_mut(|data| data.insert_temp(memory, draft));
    drop(db);

    let mut db = world.resource_mut::<AssetDatabase>();
    let done = match action {
        Some(Action::Apply(settings)) => db.set_settings(id, settings),
        Some(Action::Reimport) => db.reimport(id),
        None => Ok(()),
    };
    drop(db);
    if let Err(error) = done {
        log_error(world, error);
    }
}

/// The picture, the name and where it is, then a line for each fact.
fn overview(ui: &mut egui::Ui, db: &AssetDatabase, entry: &Entry, previews: &mut Previews) {
    let side = PICTURE.min(ui.available_width());
    let (rect, _) = ui.allocate_exact_size(egui::Vec2::splat(side), egui::Sense::hover());
    let visuals = ui.visuals();
    ui.painter().rect_filled(
        rect,
        visuals.widgets.inactive.corner_radius,
        visuals.extreme_bg_color,
    );
    match previews.asset(ui.ctx(), db, entry.id) {
        Some(picture) => egui::Image::new((picture.id(), rect.size())).paint_at(ui, rect),
        None => {
            let glyph = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(48.0));
            icons::paint(ui, glyph_of(entry.kind), glyph, visuals.weak_text_color());
        }
    }
    ui.add_space(6.0);

    let (folder, file) = entry.path.rsplit_once('/').unwrap_or(("", &entry.path));
    ui.label(egui::RichText::new(file).strong().size(15.0));
    if !folder.is_empty() {
        ui.weak(folder);
    }
    ui.add_space(4.0);
    egui::Grid::new("import_facts")
        .num_columns(2)
        .spacing([12.0, 4.0])
        .show(ui, |ui| {
            fact(ui, "Kind");
            ui.label(kind_title(entry.kind));
            ui.end_row();

            fact(ui, "Id");
            ui.add(
                egui::Label::new(egui::RichText::new(entry.id.to_string()).monospace())
                    .selectable(true),
            );
            ui.end_row();

            fact(ui, "Status");
            match &entry.status {
                Status::Failed(message) => {
                    ui.add(
                        egui::Label::new(egui::RichText::new(message).color(theme::ERROR)).wrap(),
                    );
                }
                Status::Queued => {
                    ui.colored_label(theme::PENDING, "Importing…");
                }
                Status::Imported { cached: true, .. } => {
                    ui.label("Imported, found in the cache");
                }
                Status::Imported { .. } => {
                    ui.label("Imported");
                }
                Status::Indexed => {
                    ui.label("Read as it is: nothing to import");
                }
            }
            ui.end_row();

            fact(ui, "Source");
            let mut source = size(entry.bytes);
            if let Some(age) = entry.modified.and_then(ago) {
                source.push_str(&format!(" · changed {age}"));
            }
            ui.label(source);
            ui.end_row();

            if let Some(summary) = entry.summary {
                fact(ui, "Imported as");
                ui.label(match summary {
                    Summary::Texture {
                        format,
                        width,
                        height,
                        levels,
                        bytes,
                    } => format!(
                        "{} · {width} × {height} · {levels} {} · {}",
                        format.label(),
                        if levels == 1 { "level" } else { "levels" },
                        size(bytes)
                    ),
                    Summary::Mesh {
                        vertices,
                        triangles,
                        bytes,
                    } => format!(
                        "{vertices} vertices · {triangles} triangles · {}",
                        size(bytes)
                    ),
                });
                ui.end_row();
            }
        });
}

fn fact(ui: &mut egui::Ui, name: &str) {
    ui.weak(name);
}

/// The draft's fields, then Apply and Revert for the draft and Reimport for
/// the asset as it is.
fn settings(ui: &mut egui::Ui, entry: &Entry, draft: &mut Draft, action: &mut Option<Action>) {
    ui.label(egui::RichText::new("Import settings").strong());
    ui.add_space(2.0);
    match &mut draft.settings {
        Settings::Texture(texture) => {
            egui::Grid::new("import_texture")
                .num_columns(2)
                .spacing([12.0, 6.0])
                .show(ui, |ui| {
                    fact(ui, "Usage");
                    let usage = USAGES
                        .iter()
                        .find(|(usage, _)| *usage == texture.usage)
                        .map_or("", |(_, name)| name);
                    egui::ComboBox::from_id_salt("import_usage")
                        .selected_text(usage)
                        .show_ui(ui, |ui| {
                            for (usage, name) in USAGES {
                                ui.selectable_value(&mut texture.usage, usage, name);
                            }
                        });
                    ui.end_row();

                    fact(ui, "Max size");
                    egui::ComboBox::from_id_salt("import_max_size")
                        .selected_text(size_limit(texture.max_size))
                        .show_ui(ui, |ui| {
                            for limit in SIZES {
                                ui.selectable_value(
                                    &mut texture.max_size,
                                    limit,
                                    size_limit(limit),
                                );
                            }
                        });
                    ui.end_row();

                    ui.label("");
                    ui.checkbox(&mut texture.compress, "Compress").on_hover_text(
                        "Block-compressed where the GPU samples it. Off keeps every texel as it \
                         is, for pixel art and UI, where 4×4 blocks show",
                    );
                    ui.end_row();
                });
        }
        Settings::Mesh(mesh) => {
            egui::Grid::new("import_mesh")
                .num_columns(2)
                .spacing([12.0, 6.0])
                .show(ui, |ui| {
                    fact(ui, "Scale");
                    ui.add(
                        egui::DragValue::new(&mut mesh.scale)
                            .speed(0.01)
                            .range(0.0001..=10_000.0),
                    )
                    .on_hover_text("Multiplies every position, for a file not authored in metres");
                    ui.end_row();
                });
        }
        Settings::None => {
            ui.weak(match entry.kind {
                AssetKind::Material => "A material is read as it is. Edit it in the inspector.",
                _ => "A prefab is read as it is: it has nothing to import.",
            });
            return;
        }
    }
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        let changed = draft.settings != entry.settings;
        if ui
            .add_enabled(changed, egui::Button::new("Apply"))
            .on_hover_text("Write these to the asset's .meta and import it again")
            .clicked()
        {
            *action = Some(Action::Apply(draft.settings));
        }
        if ui
            .add_enabled(changed, egui::Button::new("Revert"))
            .clicked()
        {
            draft.settings = entry.settings;
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .button("Reimport")
                .on_hover_text("Import again as it is, past the cache")
                .clicked()
            {
                *action = Some(Action::Reimport);
            }
        });
    });
}

fn glyph_of(kind: AssetKind) -> icons::Icon {
    match kind {
        AssetKind::Texture => icons::image(),
        AssetKind::Mesh => icons::box_(),
        AssetKind::Material => icons::palette(),
        AssetKind::Prefab => icons::layers(),
        AssetKind::Folder => icons::folder(),
    }
}

fn kind_title(kind: AssetKind) -> String {
    let label = kind.label();
    label[..1].to_ascii_uppercase() + &label[1..]
}

fn size_limit(texels: u32) -> String {
    match texels {
        0 => "No limit".to_owned(),
        texels => format!("{texels} px"),
    }
}

/// A byte count the way a file explorer gives one.
fn size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KB", "MB", "GB", "TB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

/// How long ago `time` was, to the largest whole unit; `None` for a time in
/// the future, which a clock set back can give.
fn ago(time: std::time::SystemTime) -> Option<String> {
    let seconds = time.elapsed().ok()?.as_secs();
    let (count, unit) = match seconds {
        0..60 => return Some("just now".to_owned()),
        60..3_600 => (seconds / 60, "minute"),
        3_600..86_400 => (seconds / 3_600, "hour"),
        _ => (seconds / 86_400, "day"),
    };
    let plural = if count == 1 { "" } else { "s" };
    Some(format!("{count} {unit}{plural} ago"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_read_as_an_explorer_gives_them() {
        assert_eq!(size(512), "512 B");
        assert_eq!(size(1536), "1.5 KB");
        assert_eq!(size(5 * 1024 * 1024), "5.0 MB");
    }
}
