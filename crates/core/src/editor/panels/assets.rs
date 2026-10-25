//! The project's assets, one folder at a time: its subfolders, drawn in the
//! tint each was given, then its assets with their import status. Every asset
//! row can be dragged onto whatever takes that kind — a prefab onto the
//! hierarchy, a mesh onto a mesh renderer, a texture onto a material.
//!
//! A row hands over the asset's [`AssetId`], never its path: the hierarchy
//! and the inspector take that payload the way everything else takes a
//! reference to an asset.
//!
//! Without a project there is no asset database, and the tab lists the
//! prefabs under `assets/` as it always has.
//!
//! [`AssetId`]: orrin_registry::AssetId

use orrin_ecs::World;

use crate::asset_db::{AssetDatabase, AssetKind, Entry, Folder, LoadState, Status, Tint};
use crate::editor::state::EditorState;
use crate::editor::theme;
use crate::editor::{icons, widgets};
use crate::scene::{LogBuffer, LogLevel, Prefabs, Time};

const ROW: f32 = 24.0;

/// The tints a folder can be given: distinct at a glance on the dark theme,
/// and few, so a colour still means something.
const TINTS: [(&str, Tint); 6] = [
    ("Red", Tint([0xe5, 0x53, 0x4b])),
    ("Orange", Tint([0xe8, 0x9a, 0x3c])),
    ("Yellow", Tint([0xe2, 0xc5, 0x4a])),
    ("Green", Tint([0x5f, 0xb5, 0x5a])),
    ("Blue", Tint([0x4f, 0x8e, 0xe8])),
    ("Purple", Tint([0xa7, 0x7b, 0xe8])),
];

pub fn body(ui: &mut egui::Ui, world: &World, state: &mut EditorState) {
    if world.get_resource::<AssetDatabase>().is_some() {
        browser(ui, world);
    } else {
        prefabs(ui, world, state);
    }
}

/// What a click asked for, applied once the database is no longer borrowed
/// for drawing.
enum Action {
    Open(String),
    Tint(String, Option<Tint>),
    Rescan,
}

fn browser(ui: &mut egui::Ui, world: &World) {
    let memory = ui.id().with("asset_folder");
    let mut folder: String = ui.data_mut(|data| data.get_temp(memory).unwrap_or_default());
    let mut action = None;
    {
        let db = world.resource::<AssetDatabase>();
        if !folder.is_empty() && !db.folders().iter().any(|f| f.path == folder) {
            folder.clear();
        }
        header(ui, &db, &folder, &mut action);
        ui.add_space(4.0);

        let (folders, assets) = listing(&db, &folder);
        if folders.is_empty() && assets.is_empty() {
            ui.weak(if folder.is_empty() {
                format!(
                    "Nothing to import under {}. PNG and JPEG textures, glTF meshes and \
                     .fprefab files are assets.",
                    db.dir().display()
                )
            } else {
                "This folder is empty.".to_owned()
            });
        }
        egui::ScrollArea::vertical()
            .auto_shrink(false)
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                for folder in folders {
                    folder_row(ui, folder, &mut action);
                }
                for entry in assets {
                    asset_row(ui, entry, db.state(entry.id)).dnd_set_drag_payload(entry.id);
                }
            });
    }

    match action {
        Some(Action::Open(path)) => folder = path,
        Some(Action::Tint(path, tint)) => {
            let result = world
                .resource_mut::<AssetDatabase>()
                .set_folder_tint(&path, tint);
            if let Err(error) = result {
                log_error(world, error);
            }
        }
        Some(Action::Rescan) => world.resource_mut::<AssetDatabase>().rescan(),
        None => {}
    }
    ui.data_mut(|data| data.insert_temp(memory, folder));
}

/// Rescan and up, where the browser is, and how much is still importing or
/// has failed.
fn header(ui: &mut egui::Ui, db: &AssetDatabase, folder: &str, action: &mut Option<Action>) {
    let weak = ui.visuals().weak_text_color();
    ui.horizontal(|ui| {
        if icons::button(ui, icons::refresh(), icons::ROW - 2.0, weak, false)
            .on_hover_text("Scan the assets directory again")
            .clicked()
        {
            *action = Some(Action::Rescan);
        }
        let up = ui
            .add_enabled_ui(!folder.is_empty(), |ui| {
                icons::button(ui, icons::arrow_up(), icons::ROW - 2.0, weak, false)
            })
            .inner
            .on_hover_text("Up one folder");
        if up.clicked() {
            *action = Some(Action::Open(parent(folder).to_owned()));
        }
        ui.label(if folder.is_empty() { "assets" } else { folder });

        let queued = db.iter().filter(|e| e.status == Status::Queued).count();
        let failed = db
            .iter()
            .filter(|e| matches!(e.status, Status::Failed(_)))
            .count();
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if failed > 0 {
                icons::labelled(
                    ui,
                    icons::warning(),
                    theme::ERROR,
                    &format!("{failed} failed"),
                );
            }
            if queued > 0 {
                icons::labelled(
                    ui,
                    icons::hourglass(),
                    theme::PENDING,
                    &format!("{queued} importing"),
                );
            }
        });
    });
}

/// The folders and assets directly inside `folder`, each in path order.
fn listing<'a>(db: &'a AssetDatabase, folder: &str) -> (Vec<&'a Folder>, Vec<&'a Entry>) {
    let inside = |path: &str| parent(path) == folder && path != folder;
    let folders = db.folders().iter().filter(|f| inside(&f.path)).collect();
    let assets = db.iter().filter(|e| inside(&e.path)).collect();
    (folders, assets)
}

/// The folder `path` is in; `""` is the assets directory itself.
fn parent(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(parent, _)| parent)
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn folder_row(ui: &mut egui::Ui, folder: &Folder, action: &mut Option<Action>) {
    let weak = ui.visuals().weak_text_color();
    let tint = folder
        .tint
        .map_or(weak, |Tint([r, g, b])| egui::Color32::from_rgb(r, g, b));
    let response = row(
        ui,
        egui::Sense::click(),
        icons::folder(),
        tint,
        file_name(&folder.path),
        None,
        "",
    );
    if response.double_clicked() || response.clicked() {
        *action = Some(Action::Open(folder.path.clone()));
    }
    response
        .on_hover_text(format!("{}\nRight-click to tint", folder.path))
        .context_menu(|ui| {
            ui.label("Tint");
            for (name, tint) in TINTS {
                let [r, g, b] = tint.0;
                let swatch = icons::with_label(
                    icons::folder(),
                    Some(egui::Color32::from_rgb(r, g, b)),
                    name,
                );
                if ui
                    .add(egui::Button::new(swatch).selected(folder.tint == Some(tint)))
                    .clicked()
                {
                    *action = Some(Action::Tint(folder.path.clone(), Some(tint)));
                    ui.close_menu();
                }
            }
            ui.separator();
            if ui
                .add_enabled(folder.tint.is_some(), egui::Button::new("No tint"))
                .clicked()
            {
                *action = Some(Action::Tint(folder.path.clone(), None));
                ui.close_menu();
            }
        });
}

fn asset_row(ui: &mut egui::Ui, entry: &Entry, state: LoadState) -> egui::Response {
    let weak = ui.visuals().weak_text_color();
    let (icon, colour, hint) = match entry.kind {
        AssetKind::Mesh => (
            icons::box_(),
            theme::KIND_MESH,
            "Drag onto a mesh renderer to draw it",
        ),
        AssetKind::Texture => (
            icons::image(),
            weak,
            "Drag onto a material's albedo to sample it",
        ),
        AssetKind::Prefab | AssetKind::Folder => {
            (icons::layers(), weak, "Drag onto the hierarchy to place it")
        }
    };
    let (status, detail) = match (&entry.status, state) {
        (Status::Failed(message), _) => (
            Some((icons::warning(), theme::ERROR)),
            format!("Failed: {message}"),
        ),
        (Status::Queued, _) | (_, LoadState::Loading) => (
            Some((icons::hourglass(), theme::PENDING)),
            "Importing".to_owned(),
        ),
        (Status::Imported { cached: true, .. }, _) => (None, "Imported (cached)".to_owned()),
        (Status::Imported { .. }, _) => (None, "Imported".to_owned()),
        (Status::Indexed, _) => (None, "Indexed".to_owned()),
    };
    row(
        ui,
        egui::Sense::drag(),
        icon,
        colour,
        file_name(&entry.path),
        status,
        "",
    )
    .on_hover_text(format!("{}\n{}\n{detail}\n{hint}", entry.path, entry.id))
}

/// One line: a glyph, a name, and at the far end a status glyph or the
/// folder the row is in.
fn row(
    ui: &mut egui::Ui,
    sense: egui::Sense,
    icon: icons::Icon,
    colour: egui::Color32,
    name: &str,
    status: Option<(icons::Icon, egui::Color32)>,
    folder: &str,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW), sense);
    let visuals = ui.visuals();
    if response.hovered() || response.dragged() {
        ui.painter()
            .rect_filled(rect, 0.0, visuals.widgets.hovered.weak_bg_fill);
    }
    let glyph = egui::Rect::from_center_size(
        egui::pos2(rect.left() + widgets::PAD + 7.0, rect.center().y),
        egui::Vec2::splat(icons::ROW - 2.0),
    );
    icons::paint(ui, icon, glyph, colour);
    let font = egui::TextStyle::Body.resolve(ui.style());
    ui.painter().text(
        egui::pos2(glyph.right() + 6.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        name,
        font.clone(),
        visuals.text_color(),
    );
    if !folder.is_empty() {
        ui.painter().text(
            egui::pos2(rect.right() - widgets::PAD, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            folder,
            font,
            visuals.weak_text_color(),
        );
    }
    if let Some((icon, colour)) = status {
        let at = egui::Rect::from_center_size(
            egui::pos2(rect.right() - widgets::PAD - 7.0, rect.center().y),
            egui::Vec2::splat(icons::ROW - 2.0),
        );
        icons::paint(ui, icon, at, colour);
    }
    response
}

fn log_error(world: &World, message: String) {
    let frame = world
        .get_resource::<Time>()
        .map_or(0, |time| time.frame_count());
    if let Some(mut log) = world.get_resource_mut::<LogBuffer>() {
        log.push(LogLevel::Error, message, frame);
    }
}

/// The tab without a project: the prefabs under `assets/`.
fn prefabs(ui: &mut egui::Ui, world: &World, state: &mut EditorState) {
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
                row(
                    ui,
                    egui::Sense::drag(),
                    icons::layers(),
                    weak,
                    prefab.name(),
                    None,
                    parent(&prefab.path),
                )
                .on_hover_text(format!(
                    "{}\nDrag onto the hierarchy to place it",
                    prefab.path
                ))
                .dnd_set_drag_payload(prefab.id);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_db::AssetDatabase;
    use crate::asset_db::tests::Dir;

    /// The browser shows one folder at a time: its direct subfolders, then its
    /// direct assets, each by path. Deeper entries belong to the subfolder.
    #[test]
    fn a_folder_lists_its_own_subfolders_and_assets_only() {
        let dir = Dir::new("browser-listing");
        dir.write("props/door.fprefab", b"orrin-scene 1\n");
        dir.write("props/wood/plank.png", b"");
        dir.write("props/wood/oak/leaf.png", b"");
        dir.write("rock.png", b"");
        let (db, _) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));

        let (folders, assets) = listing(&db, "");
        let folders: Vec<&str> = folders.iter().map(|f| f.path.as_str()).collect();
        let assets: Vec<&str> = assets.iter().map(|a| a.path.as_str()).collect();
        assert_eq!((folders, assets), (vec!["props"], vec!["rock.png"]));

        let (folders, assets) = listing(&db, "props");
        let folders: Vec<&str> = folders.iter().map(|f| f.path.as_str()).collect();
        let assets: Vec<&str> = assets.iter().map(|a| a.path.as_str()).collect();
        assert_eq!(
            (folders, assets),
            (vec!["props/wood"], vec!["props/door.fprefab"])
        );
    }

    #[test]
    fn a_folder_goes_up_to_its_parent_and_the_root_stays_put() {
        assert_eq!(parent("props/wood"), "props");
        assert_eq!(parent("props"), "");
        assert_eq!(parent(""), "");
    }
}
