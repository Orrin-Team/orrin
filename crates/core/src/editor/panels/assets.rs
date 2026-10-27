//! The project's assets, one folder at a time: its subfolders, drawn in the
//! tint each was given, then its assets with their import status. Every asset
//! row can be dragged onto whatever takes that kind — a prefab onto the
//! hierarchy or the scene, a mesh or a material onto a mesh renderer, a
//! texture onto a material asset's slot. Right-clicking a folder, or the open
//! folder below its rows, makes a new material there.
//!
//! A row hands over the asset's [`AssetId`], never its path: the hierarchy
//! and the inspector take that payload the way everything else takes a
//! reference to an asset.
//!
//! The filter is not confined to the open folder: typing searches every
//! asset's path, and each match says which folder it is in.
//!
//! Without a project there is no asset database, and the tab lists the
//! prefabs under `assets/` as it always has.

use orrin_ecs::{FxHashMap, World};
use orrin_registry::AssetId;

use crate::asset_db::{AssetDatabase, AssetKind, Entry, Folder, LoadState, Status, Tint};
use crate::editor::state::EditorState;
use crate::editor::theme;
use crate::editor::{icons, widgets};
use crate::scene::{LogBuffer, LogLevel, PrefabInstance, Prefabs, Time};

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

const PLACE_HINT: &str = "Drag onto the hierarchy or the scene to place it, or double-click";

pub fn body(ui: &mut egui::Ui, world: &World, state: &mut EditorState) {
    if world.get_resource::<AssetDatabase>().is_some() {
        browser(ui, world, state);
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
    /// A material at the default parameters, in this folder.
    NewMaterial(String),
}

/// The browser's place, kept in egui's memory rather than the editor state:
/// which folder is open, and which revealed prefab it last opened a folder for.
#[derive(Clone, Default)]
struct Place {
    folder: String,
    revealed: Option<AssetId>,
}

fn browser(ui: &mut egui::Ui, world: &World, state: &mut EditorState) {
    let memory = ui.id().with("asset_folder");
    let mut place: Place = ui.data_mut(|data| data.get_temp(memory).unwrap_or_default());
    let in_scene = instance_counts(world);
    let mut action = None;
    {
        let db = world.resource::<AssetDatabase>();
        if place.revealed != state.revealed_asset {
            place.revealed = state.revealed_asset;
            if let Some(entry) = state.revealed_asset.and_then(|id| db.get(id)) {
                place.folder = parent(&entry.path).to_owned();
            }
        }
        if !place.folder.is_empty() && !db.folders().iter().any(|f| f.path == place.folder) {
            place.folder.clear();
        }
        header(ui, &db, &place.folder, &mut action);
        widgets::search(ui, &mut state.assets_query, "Filter assets");
        ui.add_space(4.0);

        let query = state.assets_query.trim().to_lowercase();
        let (folders, assets) = if query.is_empty() {
            listing(&db, &place.folder)
        } else {
            let found = db
                .iter()
                .filter(|entry| entry.path.to_lowercase().contains(&query))
                .collect();
            (Vec::new(), found)
        };
        if folders.is_empty() && assets.is_empty() {
            ui.weak(if !query.is_empty() {
                format!("No asset matches “{}”.", state.assets_query.trim())
            } else if place.folder.is_empty() {
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
                    let folder = if query.is_empty() {
                        ""
                    } else {
                        parent(&entry.path)
                    };
                    let count = in_scene.get(&entry.id).copied().unwrap_or(0);
                    asset_row(ui, state, entry, db.state(entry.id), folder, count);
                }
                // The rest of the view, below the rows, is the open folder.
                let rest = (ui.clip_rect().bottom() - ui.cursor().top()).max(ROW);
                let size = egui::vec2(ui.available_width(), rest);
                ui.allocate_response(size, egui::Sense::click())
                    .context_menu(|ui| new_material_item(ui, &place.folder, &mut action));
            });
    }

    match action {
        Some(Action::Open(path)) => place.folder = path,
        Some(Action::Tint(path, tint)) => {
            let result = world
                .resource_mut::<AssetDatabase>()
                .set_folder_tint(&path, tint);
            if let Err(error) = result {
                log_error(world, error);
            }
        }
        Some(Action::Rescan) => world.resource_mut::<AssetDatabase>().rescan(),
        Some(Action::NewMaterial(folder)) => {
            let created = world
                .resource_mut::<AssetDatabase>()
                .create_material(&folder);
            match created {
                Ok(id) => state.revealed_asset = Some(id),
                Err(error) => log_error(world, error),
            }
        }
        None => {}
    }
    ui.data_mut(|data| data.insert_temp(memory, place));
}

fn new_material_item(ui: &mut egui::Ui, folder: &str, action: &mut Option<Action>) {
    if ui.button("New material").clicked() {
        *action = Some(Action::NewMaterial(folder.to_owned()));
        ui.close_menu();
    }
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

/// How many instances of each prefab the scene holds.
fn instance_counts(world: &World) -> FxHashMap<AssetId, usize> {
    let mut in_scene: FxHashMap<AssetId, usize> = FxHashMap::default();
    world
        .query::<&PrefabInstance>()
        .for_each(|_, instance| *in_scene.entry(instance.prefab).or_default() += 1);
    in_scene
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
        "",
        End::Nothing,
        false,
    );
    if response.double_clicked() || response.clicked() {
        *action = Some(Action::Open(folder.path.clone()));
    }
    response
        .on_hover_text(format!(
            "{}\nRight-click to tint, or to make a material",
            folder.path
        ))
        .context_menu(|ui| {
            new_material_item(ui, &folder.path, action);
            ui.separator();
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

/// One asset: dragged to wherever takes its kind; a prefab is also marked by
/// a click and placed by a double-click, as in the prefab list.
fn asset_row(
    ui: &mut egui::Ui,
    state: &mut EditorState,
    entry: &Entry,
    load: LoadState,
    folder: &str,
    in_scene: usize,
) {
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
            "Drag onto a material asset's texture slot to sample it",
        ),
        AssetKind::Material => (
            icons::palette(),
            weak,
            "Drag onto a mesh renderer's material to draw with it. Edit it in the inspector",
        ),
        AssetKind::Prefab | AssetKind::Folder => (icons::layers(), weak, PLACE_HINT),
    };
    let (status, detail) = match (&entry.status, load) {
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
    let count = format!("{in_scene} in scene");
    let end = match status {
        Some((icon, colour)) => End::Glyph(icon, colour),
        None if in_scene > 0 => End::Text(&count),
        None => End::Nothing,
    };
    let prefab = entry.kind == AssetKind::Prefab;
    let sense = if prefab {
        egui::Sense::click_and_drag()
    } else {
        egui::Sense::drag()
    };
    let marked = state.revealed_asset == Some(entry.id);
    let name = file_name(&entry.path);
    let response = row(ui, sense, icon, colour, name, folder, end, marked);
    if prefab && response.clicked() {
        state.revealed_asset = Some(entry.id);
    }
    if prefab && response.double_clicked() {
        state.request_instantiate(entry.id, None);
    }
    carry(ui, &response, entry.id, name);
    response.on_hover_text(format!("{}\n{}\n{detail}\n{hint}", entry.path, entry.id));
}

/// Start a drag of `id`, and while it lasts draw what is being carried beside
/// the pointer, since the row itself stays put.
fn carry(ui: &egui::Ui, response: &egui::Response, id: AssetId, name: &str) {
    if response.drag_started() {
        response.dnd_set_drag_payload(id);
    }
    if response.dragged()
        && let Some(pointer) = ui.ctx().pointer_latest_pos()
    {
        egui::Area::new(ui.id().with("dragged_asset"))
            .order(egui::Order::Tooltip)
            .fixed_pos(pointer + egui::vec2(14.0, 6.0))
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| ui.label(name));
            });
    }
}

/// What a row shows at its far end.
enum End<'a> {
    Nothing,
    Glyph(icons::Icon, egui::Color32),
    Text(&'a str),
}

/// One line: a glyph, a name, the folder it is in when that is not the one
/// open, and at the far end a status glyph or a count.
#[allow(clippy::too_many_arguments)]
fn row(
    ui: &mut egui::Ui,
    sense: egui::Sense,
    icon: icons::Icon,
    colour: egui::Color32,
    name: &str,
    folder: &str,
    end: End,
    marked: bool,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW), sense);
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
    icons::paint(ui, icon, glyph, colour);
    let font = egui::TextStyle::Body.resolve(ui.style());
    let label = ui.painter().text(
        egui::pos2(glyph.right() + 6.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        name,
        font.clone(),
        visuals.text_color(),
    );
    if !folder.is_empty() {
        ui.painter().text(
            egui::pos2(label.right() + 8.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            folder,
            font.clone(),
            weak,
        );
    }
    match end {
        End::Nothing => {}
        End::Glyph(icon, colour) => {
            let at = egui::Rect::from_center_size(
                egui::pos2(rect.right() - widgets::PAD - 7.0, rect.center().y),
                egui::Vec2::splat(icons::ROW - 2.0),
            );
            icons::paint(ui, icon, at, colour);
        }
        End::Text(text) => {
            ui.painter().text(
                egui::pos2(rect.right() - widgets::PAD, rect.center().y),
                egui::Align2::RIGHT_CENTER,
                text,
                font,
                weak,
            );
        }
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

    let in_scene = instance_counts(world);
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
                let count = in_scene.get(&prefab.id).copied().unwrap_or(0);
                let count = format!("{count} in scene");
                let end = if in_scene.contains_key(&prefab.id) {
                    End::Text(&count)
                } else {
                    End::Nothing
                };
                let marked = state.revealed_asset == Some(prefab.id);
                let response = row(
                    ui,
                    egui::Sense::click_and_drag(),
                    icons::box_(),
                    weak,
                    prefab.name(),
                    parent(&prefab.path),
                    end,
                    marked,
                );
                if response.clicked() {
                    state.revealed_asset = Some(prefab.id);
                }
                if response.double_clicked() {
                    state.request_instantiate(prefab.id, None);
                }
                carry(ui, &response, prefab.id, prefab.name());
                response.on_hover_text(format!("{}\n{PLACE_HINT}", prefab.path));
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
