//! The project's assets, one folder at a time: its subfolders, drawn in the
//! tint each was given, then its assets with their import status. Every asset
//! row can be dragged onto whatever takes that kind — a prefab onto the
//! hierarchy or the scene, a mesh or a material onto a mesh renderer, a
//! texture onto a material asset's slot — and an asset or a folder dragged
//! onto a folder moves into it. Right-clicking a folder, or the open folder
//! below its rows, makes a new folder, material or shader there; the open
//! folder's menu also rescans, which a watched directory seldom needs, so it
//! is not on the bar. A double-click does what the kind is for: a prefab or a
//! mesh is placed, a sound plays through the editor's preview (and again
//! stops), a shader opens in the Shader tab, a scene opens, a `.blend` opens
//! in Blender, and a material, a texture or a font brings the Import tab
//! forward on itself.
//!
//! A row hands over the asset's [`AssetId`], never its path: the hierarchy
//! and the inspector take that payload the way everything else takes a
//! reference to an asset. A folder has no id to hand over, so its drag
//! carries its path, which only the browser reads.
//!
//! The bar names the open folder as its path, each folder of which opens
//! itself and takes a drop as a move into it. A path too long for a docked
//! bar gives up its middle, never the filter's room.
//!
//! The filter is not confined to the open folder: typing searches every
//! asset's path, and each match says which folder it is in. `.png`-style
//! words and the kind menu narrow it by type.
//!
//! The folder shows as rows or as tiles, each pictured by its import's
//! thumbnail where it has one ([`Previews::asset`]). Both are drawn only for
//! what is on screen, so a project of thousands of assets costs a screenful.
//! A click marks an asset, which the Import tab shows; Ctrl adds or takes
//! away one and Shift marks the run from the last click, and Delete over the
//! browser, the menu's Delete… and a drag of any of them then act on the
//! whole set. Rename and a move are file operations: the file goes with its
//! sidecar, and neither is on the undo stack, as neither is in an explorer.
//! Files dropped from outside the editor are copied in (see
//! [`AssetDatabase::bring_in`]).
//!
//! Without a project there is no asset database, and the tab lists the
//! prefabs under `assets/` as it always has.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use orrin_ecs::{FxHashMap, World};
use orrin_registry::AssetId;

use crate::asset_db::{AssetDatabase, AssetKind, Entry, Folder, LoadState, Settings, Status, Tint};
use crate::editor::commands::Command;
use crate::editor::previews::Previews;
use crate::editor::state::{AssetLayout, AssetSort, AssetView, EditorState};
use crate::editor::theme;
use crate::editor::{icons, widgets};
use crate::scene::{LogBuffer, LogLevel, PrefabInstance, Prefabs, Time};

/// What the stand-in for the file dialog picks, in tests.
#[cfg(test)]
pub(super) static PICKED: Mutex<Option<Vec<PathBuf>>> = Mutex::new(None);

/// Held by a test for as long as it uses [`PICKED`]: tests run in parallel,
/// and a dialog thread takes whatever any of them put there.
#[cfg(test)]
pub(super) static PICKER: Mutex<()> = Mutex::new(());

/// A row of the project-less prefab list.
const ROW: f32 = 24.0;
/// A row of the browser's list: tall enough for a 28 px thumbnail that reads.
const LIST_ROW: f32 = 32.0;
/// A tile of the browser's grid: its picture, and one line of name under it.
const TILE: egui::Vec2 = egui::vec2(96.0, 96.0);
const TILE_THUMB: f32 = 64.0;
const TILE_GAP: f32 = 4.0;

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

pub fn body(ui: &mut egui::Ui, world: &World, state: &mut EditorState, previews: &mut Previews) {
    let pick = std::mem::take(&mut state.pick_files);
    if world.get_resource::<AssetDatabase>().is_some() {
        browser(ui, world, state, previews, pick);
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
    /// The surface shader template, in this folder.
    NewShader(String),
    /// A folder in this one, named in place once it is made.
    NewFolder(String),
    /// Edit this path's name in place, starting from this text.
    StartRename(String, String),
    Rename(String, String),
    CancelRename,
    /// Move these assets and folders into this folder.
    Move(Vec<String>, String),
    /// Ask whether to delete these assets and folders; see `confirm_delete`.
    Delete(Vec<String>),
    /// Copy these files from outside the project into this folder.
    BringIn(Vec<PathBuf>, String),
    /// Open a file dialog for files to copy into the open folder.
    PickFiles,
    /// Play this sound through the editor's preview, or stop it if it is
    /// the one playing.
    TogglePreview(AssetId),
    /// Start Blender on this `.blend`.
    OpenInBlender(String),
}

/// The browser's place, kept in egui's memory rather than the editor state:
/// which folder is open, which revealed asset it last opened a folder for,
/// what is marked, the name being edited, if one is, and the file dialog, if
/// one is open.
#[derive(Clone, Default)]
struct Place {
    folder: String,
    revealed: Option<AssetId>,
    /// The marked assets and folders, by path, sorted so a cell finds itself
    /// by a binary search. May name what has since gone; each use filters.
    marked: Vec<String>,
    /// Where a Shift-click's range runs from: the last plain or Ctrl click.
    anchor: Option<String>,
    renaming: Option<Renaming>,
    picking: Option<Picking>,
}

impl Place {
    /// What was at `from` is now at `to`: the open folder and the marks go
    /// where it went, and so does whatever was inside it.
    fn follow(&mut self, from: &str, to: &str) {
        let inside = format!("{from}/");
        let moved = |path: &mut String| {
            if path == from {
                *path = to.to_owned();
            } else if let Some(rest) = path.strip_prefix(&inside) {
                *path = format!("{to}/{rest}");
            }
        };
        moved(&mut self.folder);
        self.marked.iter_mut().for_each(moved);
        self.marked.sort();
        if let Some(anchor) = &mut self.anchor {
            moved(anchor);
        }
    }
}

/// A folder being dragged. An asset's drag carries its bare [`AssetId`], which
/// is what every other tool takes; a folder has no id to carry.
struct CarriedFolder(String);

/// A click on `path` with `modifiers` held: Shift marks the listing from the
/// anchor to it, Ctrl adds or takes it away, and a plain click marks it alone.
/// The asset clicked into the set is the one the Import tab shows.
fn mark(
    place: &mut Place,
    state: &mut EditorState,
    db: &AssetDatabase,
    items: &[Item],
    path: String,
    modifiers: egui::Modifiers,
) {
    let at = |wanted: &str| items.iter().position(|item| item.path() == wanted);
    let range = modifiers
        .shift
        .then(|| place.anchor.as_deref().and_then(at).zip(at(&path)))
        .flatten();
    if let Some((from, to)) = range {
        let (first, last) = (from.min(to), from.max(to));
        place.marked = items[first..=last]
            .iter()
            .map(|item| item.path().to_owned())
            .collect();
        place.marked.sort();
    } else if modifiers.command {
        match place.marked.binary_search(&path) {
            Ok(at) => {
                place.marked.remove(at);
            }
            Err(at) => place.marked.insert(at, path.clone()),
        }
        place.anchor = Some(path.clone());
    } else {
        place.marked = vec![path.clone()];
        place.anchor = Some(path.clone());
    }
    let Some(clicked) = db.find(&path).map(|entry| entry.id) else {
        return;
    };
    if place.marked.binary_search(&path).is_ok() {
        state.revealed_asset = Some(clicked);
    } else if state.revealed_asset == Some(clicked) {
        state.revealed_asset = place
            .marked
            .iter()
            .rev()
            .find_map(|path| db.find(path))
            .map(|entry| entry.id);
    }
}

/// Those of `paths` still in the project.
fn existing(db: &AssetDatabase, paths: &[String]) -> Vec<String> {
    paths
        .iter()
        .filter(|path| db.find(path).is_some() || db.folders().iter().any(|f| &f.path == *path))
        .cloned()
        .collect()
}

/// `paths` without any that lies inside another of them, which goes with it.
fn outermost(paths: &[String]) -> Vec<String> {
    paths
        .iter()
        .filter(|path| {
            !paths.iter().any(|other| {
                path.strip_prefix(other.as_str())
                    .is_some_and(|rest| rest.starts_with('/'))
            })
        })
        .cloned()
        .collect()
}

/// What acting on `path` acts on: the whole marked set when it is one of
/// them, else it alone.
fn chosen(db: &AssetDatabase, marked: &[String], path: &str) -> Vec<String> {
    if marked.len() > 1 && marked.binary_search_by(|m| m.as_str().cmp(path)).is_ok() {
        outermost(&existing(db, marked))
    } else {
        vec![path.to_owned()]
    }
}

/// An asset or a folder the browser carries, let go on `response`: a move of
/// it, or of the marked set it is one of, into `folder`. Also outlines
/// `response` while one is held over it.
fn dropped_on(
    ui: &egui::Ui,
    response: &egui::Response,
    db: &AssetDatabase,
    marked: &[String],
    folder: &str,
) -> Option<Action> {
    if !response.contains_pointer() {
        return None;
    }
    let ctx = ui.ctx();
    let carried = match egui::DragAndDrop::payload::<AssetId>(ctx) {
        Some(id) => db.get(*id).map(|entry| entry.path.clone()),
        None => egui::DragAndDrop::payload::<CarriedFolder>(ctx).map(|folder| folder.0.clone()),
    }?;
    // Let go where it was picked up: no move was meant.
    if carried == folder {
        return None;
    }
    if !ui.input(|input| input.pointer.any_released()) {
        let visuals = ui.visuals();
        ui.painter().rect_stroke(
            response.rect,
            visuals.widgets.inactive.corner_radius,
            visuals.selection.stroke,
            egui::StrokeKind::Inside,
        );
        return None;
    }
    egui::DragAndDrop::clear_payload(ctx);
    Some(Action::Move(
        chosen(db, marked, &carried),
        folder.to_owned(),
    ))
}

/// A file dialog open on a thread of its own, so the editor goes on drawing
/// behind it, and the folder what it picks is copied into.
#[derive(Clone)]
struct Picking {
    folder: String,
    /// Empty until the dialog closes; then what was picked, or nothing.
    picked: Arc<Mutex<Option<Vec<PathBuf>>>>,
}

/// A name being edited in place: an asset's without its extension, which the
/// rename keeps, or a folder's.
#[derive(Clone)]
struct Renaming {
    path: String,
    text: String,
    /// The field has been given focus, so losing it ends the edit.
    started: bool,
}

/// `pick` is File ▸ Import… asking for the dialog the bar's + opens.
fn browser(
    ui: &mut egui::Ui,
    world: &World,
    state: &mut EditorState,
    previews: &mut Previews,
    pick: bool,
) {
    let area = ui.max_rect();
    let memory = ui.id().with("asset_folder");
    let mut place: Place = ui.data_mut(|data| data.get_temp(memory).unwrap_or_default());
    let in_scene = instance_counts(world);
    let previewing = world
        .get_resource::<crate::audio::Audio>()
        .and_then(|audio| audio.previewing());
    let mut action = (pick && place.picking.is_none()).then_some(Action::PickFiles);
    let over = ui.rect_contains_pointer(area);
    state.assets_hovered = over;
    {
        let db = world.resource::<AssetDatabase>();
        if place.revealed != state.revealed_asset {
            place.revealed = state.revealed_asset;
            if let Some(entry) = state.revealed_asset.and_then(|id| db.get(id)) {
                place.folder = parent(&entry.path).to_owned();
                // Revealed by another tool: marked alone, unless a click here
                // already made it one of the set.
                if place.marked.binary_search(&entry.path).is_err() {
                    place.marked = vec![entry.path.clone()];
                    place.anchor = Some(entry.path.clone());
                }
            }
        }
        if !place.folder.is_empty() && !db.folders().iter().any(|f| f.path == place.folder) {
            place.folder.clear();
        }
        header(
            ui,
            &db,
            &place.folder,
            &place.marked,
            place.picking.is_some(),
            &mut state.assets_query,
            &mut state.assets_view,
            &mut action,
        );
        ui.add_space(4.0);

        let view = state.assets_view;
        let query = Query::parse(&state.assets_query);
        let searching = !query.is_empty() || view.kind.is_some();
        let (folders, assets) = shown(&db, &place.folder, &query, &view);
        if folders.is_empty() && assets.is_empty() {
            ui.weak(if !query.is_empty() {
                format!("No asset matches “{}”.", state.assets_query.trim())
            } else if let Some(kind) = view.kind {
                format!("The project has no {}.", plural(kind))
            } else if place.folder.is_empty() {
                format!(
                    "Nothing to import under {}. PNG and JPEG textures, glTF meshes, .omat \
                     materials and .oprefab files are assets.",
                    db.dir().display()
                )
            } else {
                "This folder is empty.".to_owned()
            });
        }
        let items: Vec<Item> = folders
            .into_iter()
            .map(Item::Folder)
            .chain(assets.into_iter().map(Item::Asset))
            .collect();
        let keys = &state.settings.keys;
        let rename = ui.rect_contains_pointer(ui.max_rect())
            && !ui.ctx().wants_keyboard_input()
            && ui.input(|input| keys.pressed(input, Command::Rename));
        if rename && let Some(entry) = state.revealed_asset.and_then(|id| db.get(id)) {
            action = Some(Action::StartRename(
                entry.path.clone(),
                entry.name().to_owned(),
            ));
        }
        let keys = &state.settings.keys;
        let delete = over
            && !ui.ctx().wants_keyboard_input()
            && ui.input(|input| keys.pressed(input, Command::Delete));
        if delete {
            // The marked set, or the revealed asset when the listing shows it
            // and nothing is marked, as after coming back to its folder.
            let targets = if place.marked.is_empty() {
                state
                    .revealed_asset
                    .and_then(|id| db.get(id))
                    .filter(|entry| items.iter().any(|item| item.path() == entry.path))
                    .map(|entry| vec![entry.path.clone()])
                    .unwrap_or_default()
            } else {
                outermost(&existing(&db, &place.marked))
            };
            if !targets.is_empty() {
                let keys = &state.settings.keys;
                ui.input_mut(|input| keys.take(input, Command::Delete));
                action = Some(Action::Delete(targets));
            }
        }
        let mut cells = Cells {
            db: &db,
            in_scene: &in_scene,
            previewing,
            layout: view.layout,
            searching,
            state,
            previews,
            renaming: &mut place.renaming,
            marked: &place.marked,
            clicked: None,
            action: &mut action,
            folders: Vec::new(),
        };
        let open = place.folder.as_str();
        match view.layout {
            AssetLayout::List => {
                // Zero spacing before `show_rows` measures; see the hierarchy.
                ui.spacing_mut().item_spacing.y = 0.0;
                egui::ScrollArea::vertical().auto_shrink(false).show_rows(
                    ui,
                    LIST_ROW,
                    items.len(),
                    |ui, range| {
                        let last = range.end == items.len();
                        for item in &items[range] {
                            cells.draw(ui, item);
                        }
                        if last {
                            cells.rest(ui, open);
                        }
                    },
                );
            }
            AssetLayout::Grid => {
                ui.spacing_mut().item_spacing = egui::Vec2::splat(TILE_GAP);
                let width = ui.available_width() - ui.spacing().scroll.bar_width;
                let columns = (((width + TILE_GAP) / (TILE.x + TILE_GAP)).floor() as usize).max(1);
                let rows = items.len().div_ceil(columns);
                egui::ScrollArea::vertical().auto_shrink(false).show_rows(
                    ui,
                    TILE.y,
                    rows,
                    |ui, range| {
                        let last = range.end == rows;
                        for row in range {
                            let end = ((row + 1) * columns).min(items.len());
                            ui.horizontal(|ui| {
                                for item in &items[row * columns..end] {
                                    cells.draw(ui, item);
                                }
                            });
                        }
                        if last {
                            cells.rest(ui, open);
                        }
                    },
                );
            }
        }
        let Cells {
            folders, clicked, ..
        } = cells;
        if let Some(folder) = file_drop(ui, area, &folders, open, &mut action) {
            ui.painter().rect_stroke(
                folder,
                ui.visuals().widgets.inactive.corner_radius,
                ui.visuals().selection.stroke,
                egui::StrokeKind::Inside,
            );
        }
        if let Some((path, modifiers)) = clicked {
            mark(&mut place, state, &db, &items, path, modifiers);
        }
    }

    match action {
        Some(Action::TogglePreview(id)) => {
            if let Some(mut audio) = world.get_resource_mut::<crate::audio::Audio>() {
                if audio.previewing() == Some(id) {
                    audio.stop_preview();
                } else {
                    audio.preview(id);
                }
            }
        }
        Some(Action::OpenInBlender(path)) => {
            let file = world.resource::<AssetDatabase>().dir().join(&path);
            let blender = crate::asset_db::blend::locate(
                state.environment.blender().map(std::ffi::OsString::from),
                state.settings.blender.path.as_deref(),
            );
            let opened = match blender {
                Some(blender) => crate::editor::blender::open(&blender, &file),
                None => Err(format!(
                    "no Blender was found. {}",
                    crate::asset_db::blend::FIX
                )),
            };
            if let Err(why) = opened {
                log_error(
                    world,
                    format!("{path} could not be opened in Blender: {why}"),
                );
            }
        }
        Some(Action::Open(path)) => {
            if place.folder != path {
                place.marked.clear();
                place.anchor = None;
            }
            place.folder = path;
        }
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
        Some(Action::NewShader(folder)) => {
            let created = world.resource_mut::<AssetDatabase>().create_shader(&folder);
            match created {
                Ok(id) => state.revealed_asset = Some(id),
                Err(error) => log_error(world, error),
            }
        }
        Some(Action::NewFolder(folder)) => {
            let created = world.resource_mut::<AssetDatabase>().create_folder(&folder);
            match created {
                Ok(path) => {
                    // Opened, so the folder made in it is drawn to be named.
                    if place.folder != folder {
                        place.folder = folder;
                        place.marked.clear();
                        place.anchor = None;
                    }
                    place.renaming = Some(Renaming {
                        text: file_name(&path).to_owned(),
                        path,
                        started: false,
                    });
                }
                Err(error) => log_error(world, error),
            }
        }
        Some(Action::StartRename(path, text)) => {
            place.renaming = Some(Renaming {
                path,
                text,
                started: false,
            });
        }
        Some(Action::CancelRename) => place.renaming = None,
        Some(Action::Move(paths, into)) => {
            let moved = world
                .resource_mut::<AssetDatabase>()
                .move_all_into(&paths, &into);
            let mut prefabs = false;
            for (from, moved) in paths.iter().zip(moved) {
                match moved {
                    Ok(to) => {
                        // A folder may hold prefabs, and the prefab index is by path.
                        prefabs |= AssetKind::of_path(&to).is_none_or(|k| k == AssetKind::Prefab);
                        place.follow(from, &to);
                    }
                    Err(error) => log_error(world, error),
                }
            }
            if prefabs {
                state.request_prefab_rescan();
            }
        }
        Some(Action::Delete(paths)) => state.request_delete_assets(paths),
        Some(Action::BringIn(files, folder)) => bring_in(world, state, &files, &folder),
        Some(Action::PickFiles) => {
            let picked = Arc::new(Mutex::new(None));
            let (into, ctx) = (picked.clone(), ui.ctx().clone());
            let dialog = std::thread::Builder::new()
                .name("orrin-file-dialog".to_owned())
                .spawn(move || {
                    let files = pick_files();
                    if let Ok(mut slot) = into.lock() {
                        *slot = Some(files);
                    }
                    ctx.request_repaint();
                });
            match dialog {
                Ok(_) => {
                    place.picking = Some(Picking {
                        folder: place.folder.clone(),
                        picked,
                    });
                }
                Err(error) => log_error(world, format!("could not open a file dialog: {error}")),
            }
        }
        Some(Action::Rename(path, name)) => {
            place.renaming = None;
            let renamed = world.resource_mut::<AssetDatabase>().rename(&path, &name);
            match renamed {
                Ok(to) => {
                    // A folder may hold prefabs, and the prefab index is by path.
                    if AssetKind::of_path(&to).is_none_or(|kind| kind == AssetKind::Prefab) {
                        state.request_prefab_rescan();
                    }
                    place.follow(&path, &to);
                }
                Err(error) => log_error(world, error),
            }
        }
        None => {}
    }
    let picked = place.picking.as_ref().and_then(|picking| {
        let files = picking.picked.lock().ok()?.take()?;
        Some((files, picking.folder.clone()))
    });
    if let Some((files, folder)) = picked {
        place.picking = None;
        bring_in(world, state, &files, &folder);
    }
    ui.data_mut(|data| data.insert_temp(memory, place));
}

/// Copy `files` into `folder`, naming what could not come in, and mark the
/// first that did.
fn bring_in(world: &World, state: &mut EditorState, files: &[PathBuf], folder: &str) {
    if files.is_empty() {
        return;
    }
    let (brought, errors) = world
        .resource_mut::<AssetDatabase>()
        .bring_in(files, folder);
    for error in errors {
        log_error(world, error);
    }
    let kind = |rel: &String| AssetKind::of_path(rel);
    if brought
        .iter()
        .any(|rel| kind(rel) == Some(AssetKind::Prefab))
    {
        state.request_prefab_rescan();
    }
    let first = brought.first().and_then(|rel| {
        let db = world.resource::<AssetDatabase>();
        db.find(rel).map(|entry| entry.id)
    });
    if first.is_some() {
        state.revealed_asset = first;
    }
}

/// The files picked in the operating system's dialog; none if it was
/// cancelled. Blocks, so it runs on a thread of its own. The async dialog,
/// because macOS shows one only from its main thread, which the async form
/// arranges itself where the blocking one would fail.
#[cfg(not(test))]
fn pick_files() -> Vec<PathBuf> {
    let dialog = rfd::AsyncFileDialog::new()
        .set_title("Import files into the project")
        .add_filter(
            "Assets",
            &[
                "png", "jpg", "jpeg", "gltf", "glb", "blend", "omat", "oprefab",
            ],
        );
    pollster::block_on(dialog.pick_files())
        .unwrap_or_default()
        .iter()
        .map(|file| file.path().to_owned())
        .collect()
}

#[cfg(test)]
fn pick_files() -> Vec<PathBuf> {
    PICKED.lock().unwrap().take().unwrap_or_default()
}

/// Take files dropped from outside the editor, into the folder drawn under the
/// pointer or else the open one, and outline the browser while files are held
/// over it. Returns the folder to outline as the one they would land in.
///
/// winit reports no pointer motion while an explorer's drag is over the window
/// on Windows, so the pointer is often not known at the drop: a drop then lands
/// in the open folder, wherever it was let go, as long as this tab is drawn.
/// A drop known to be elsewhere is left alone.
fn file_drop(
    ui: &egui::Ui,
    area: egui::Rect,
    folders: &[(egui::Rect, String)],
    open: &str,
    action: &mut Option<Action>,
) -> Option<egui::Rect> {
    let (dropped, held, pointer) = ui.input(|input| {
        let dropped: Vec<_> = input
            .raw
            .dropped_files
            .iter()
            .filter_map(|file| file.path.clone())
            .collect();
        (
            dropped,
            !input.raw.hovered_files.is_empty(),
            input.pointer.hover_pos(),
        )
    });
    if pointer.is_some_and(|at| !area.contains(at)) || (dropped.is_empty() && !held) {
        return None;
    }
    let under = pointer.and_then(|at| folders.iter().find(|(rect, _)| rect.contains(at)));
    if !dropped.is_empty() {
        let folder = under.map_or(open, |(_, path)| path.as_str());
        *action = Some(Action::BringIn(dropped, folder.to_owned()));
        return None;
    }
    let visuals = ui.visuals();
    ui.painter().rect_stroke(
        area,
        visuals.widgets.inactive.corner_radius,
        visuals.selection.stroke,
        egui::StrokeKind::Inside,
    );
    under.map(|(rect, _)| *rect)
}

/// Rename has no undo: it moves files, as a rename in an explorer does.
fn delete_item(ui: &mut egui::Ui, paths: Vec<String>, action: &mut Option<Action>) {
    let label = match paths.len() {
        1 => "Delete…".to_owned(),
        n => format!("Delete {n} items…"),
    };
    if ui.button(label).clicked() {
        *action = Some(Action::Delete(paths));
        ui.close_menu();
    }
}

fn new_folder_item(ui: &mut egui::Ui, folder: &str, action: &mut Option<Action>) {
    if ui.button("New folder").clicked() {
        *action = Some(Action::NewFolder(folder.to_owned()));
        ui.close_menu();
    }
}

fn rename_item(ui: &mut egui::Ui, path: &str, name: &str, action: &mut Option<Action>) {
    if ui
        .button("Rename")
        .on_hover_text("F2. Renames the file and its .meta; not undoable")
        .clicked()
    {
        *action = Some(Action::StartRename(path.to_owned(), name.to_owned()));
        ui.close_menu();
    }
}

fn new_material_item(ui: &mut egui::Ui, folder: &str, action: &mut Option<Action>) {
    if ui.button("New material").clicked() {
        *action = Some(Action::NewMaterial(folder.to_owned()));
        ui.close_menu();
    }
    if ui.button("New surface shader").clicked() {
        *action = Some(Action::NewShader(folder.to_owned()));
        ui.close_menu();
    }
}

/// The least the filter is given beside a long path, in points: enough to read
/// a word or two typed into it.
const FILTER_MIN: f32 = 100.0;

/// The browser's one bar: up, bringing files in, and where the browser is; the
/// filter, in whatever width the path leaves; the view's controls; and how
/// much is still importing or has failed.
#[allow(clippy::too_many_arguments)]
fn header(
    ui: &mut egui::Ui,
    db: &AssetDatabase,
    folder: &str,
    marked: &[String],
    picking: bool,
    query: &mut String,
    view: &mut AssetView,
    action: &mut Option<Action>,
) {
    let weak = ui.visuals().weak_text_color();
    ui.horizontal(|ui| {
        let up = ui
            .add_enabled_ui(!folder.is_empty(), |ui| {
                icons::button(ui, icons::arrow_up(), icons::ROW - 2.0, weak, false)
            })
            .inner
            .on_hover_text("Up one folder");
        if up.clicked() {
            *action = Some(Action::Open(parent(folder).to_owned()));
        }
        let import = ui
            .add_enabled_ui(!picking, |ui| {
                icons::button(ui, icons::plus(), icons::ROW - 2.0, weak, false)
            })
            .inner
            .on_hover_text("Copy files from disk into this folder. Dropping them here does too");
        if import.clicked() {
            *action = Some(Action::PickFiles);
        }

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
            view_controls(ui, view);
            let crumbs = Crumbs::of(ui, folder);
            let left = ui.available_width();
            let filter = (left - crumbs.width - ui.spacing().item_spacing.x)
                .max(FILTER_MIN)
                .min(left);
            let height = ui.spacing().interact_size.y;
            ui.allocate_ui(egui::vec2(filter, height), |ui| {
                widgets::search(ui, query, "Filter assets — words, or .png .omat");
            });
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                crumbs.show(ui, db, marked, action);
            });
        });
    });
}

/// Where the browser is, as a row of folders from the assets directory down,
/// each opening itself and each taking a dropped asset or folder as a move
/// into it.
struct Crumbs {
    /// Each folder's path, `""` being the assets directory.
    paths: Vec<String>,
    /// Each one's width, with the chevron before it for all but the first.
    widths: Vec<f32>,
    width: f32,
    /// The ellipsis that stands for the folders left out, with its chevron.
    ellipsis: f32,
}

/// The chevron between two folders of the path, in points.
const CHEVRON: f32 = icons::ROW - 4.0;
/// The path's folders and chevrons sit closer than the bar's controls: they
/// read as one path, and a docked bar has little width to give them.
const CRUMB_GAP: f32 = 2.0;
const CRUMB_PAD: f32 = 2.0;

impl Crumbs {
    fn of(ui: &egui::Ui, folder: &str) -> Self {
        let mut paths = vec![String::new()];
        let mut end = 0;
        for part in folder.split('/').filter(|part| !part.is_empty()) {
            end += part.len();
            paths.push(folder[..end].to_owned());
            end += 1;
        }
        let spacing = CRUMB_GAP;
        let pad = CRUMB_PAD * 2.0;
        let colour = ui.visuals().text_color();
        let measure = |text: &str, font: egui::FontId| {
            ui.fonts(|fonts| fonts.layout_no_wrap(text.to_owned(), font, colour))
                .size()
                .x
        };
        let font = egui::TextStyle::Button.resolve(ui.style());
        let chevron = measure(
            icons::text(icons::chevron_right(), CHEVRON, colour).text(),
            egui::FontId::new(CHEVRON, icons::family()),
        ) + spacing;
        let widths: Vec<f32> = paths
            .iter()
            .enumerate()
            .map(|(at, path)| {
                let before = if at > 0 { chevron } else { 0.0 };
                measure(Self::name(path), font.clone()) + pad + spacing + before
            })
            .collect();
        Self {
            width: widths.iter().sum(),
            ellipsis: measure("…", font) + pad + spacing + chevron,
            paths,
            widths,
        }
    }

    fn name(path: &str) -> &str {
        if path.is_empty() {
            "assets"
        } else {
            file_name(path)
        }
    }

    /// The path in the width the bar leaves it: whole when it fits, else the
    /// assets directory, an ellipsis and as many of the deepest folders as
    /// fit. The open folder is always shown, cut short if it must be.
    fn show(
        &self,
        ui: &mut egui::Ui,
        db: &AssetDatabase,
        marked: &[String],
        action: &mut Option<Action>,
    ) {
        ui.spacing_mut().item_spacing.x = CRUMB_GAP;
        ui.spacing_mut().button_padding.x = CRUMB_PAD;
        let room = ui.available_width();
        let last = self.paths.len() - 1;
        // The first folder shown after the root; 1 is all of them.
        let mut from = 1;
        if self.width > room && last > 1 {
            from = last;
            let mut used = self.widths[0] + self.ellipsis + self.widths[last];
            while from > 2 && used + self.widths[from - 1] <= room {
                from -= 1;
                used += self.widths[from];
            }
        }
        let root = from == 1 || self.widths[0] + self.ellipsis + self.widths[last] <= room;
        let weak = ui.visuals().weak_text_color();
        let chevron = |ui: &mut egui::Ui| {
            ui.label(icons::text(icons::chevron_right(), CHEVRON, weak));
        };
        if root {
            self.crumb(ui, 0, None, db, marked, action);
        }
        if from > 1 {
            if root {
                chevron(ui);
            }
            // The deepest folder left out, standing for all of them.
            let ellipsis = egui::RichText::new("…").color(weak);
            self.crumb(ui, from - 1, Some(ellipsis), db, marked, action);
        }
        for at in from..=last {
            chevron(ui);
            self.crumb(ui, at, None, db, marked, action);
        }
    }

    fn crumb(
        &self,
        ui: &mut egui::Ui,
        at: usize,
        label: Option<egui::RichText>,
        db: &AssetDatabase,
        marked: &[String],
        action: &mut Option<Action>,
    ) {
        let path = &self.paths[at];
        let name = Self::name(path);
        let text = match label {
            Some(label) => label,
            None if at == self.paths.len() - 1 => egui::RichText::new(name).strong(),
            None => egui::RichText::new(name),
        };
        let response = ui
            .add(
                egui::Button::new(text)
                    .frame(false)
                    .wrap_mode(egui::TextWrapMode::Truncate),
            )
            .on_hover_text(format!(
                "Open {name}. Drop an asset or a folder here to move it in, with its .meta; \
                 not undoable"
            ));
        if let Some(moved) = dropped_on(ui, &response, db, marked, path) {
            *action = Some(moved);
        }
        if response.clicked() {
            *action = Some(Action::Open(path.clone()));
        }
    }
}

/// What the filter box asks for: words found anywhere in an asset's path, and
/// `.ext` words naming the kinds of file wanted — `.png .jpg rock` is every
/// PNG or JPEG with "rock" in its path.
#[derive(Debug, Default, PartialEq)]
struct Query {
    words: Vec<String>,
    extensions: Vec<String>,
}

impl Query {
    fn parse(text: &str) -> Self {
        let mut query = Self::default();
        for word in text.split_whitespace().map(str::to_lowercase) {
            match word.strip_prefix('.') {
                Some(extension) if !extension.is_empty() => {
                    query.extensions.push(extension.to_owned());
                }
                _ => query.words.push(word),
            }
        }
        query
    }

    fn is_empty(&self) -> bool {
        self.words.is_empty() && self.extensions.is_empty()
    }

    fn matches(&self, path: &str) -> bool {
        let path = path.to_lowercase();
        let extension = extension(&path);
        (self.extensions.is_empty() || self.extensions.iter().any(|e| e == extension))
            && self.words.iter().all(|word| path.contains(word.as_str()))
    }
}

/// What the browser lists: the open folder's subfolders and assets, or, while
/// the filter or a kind is set, every matching asset in the project. Assets in
/// the view's order; folders always first, by name.
fn shown<'a>(
    db: &'a AssetDatabase,
    folder: &str,
    query: &Query,
    view: &AssetView,
) -> (Vec<&'a Folder>, Vec<&'a Entry>) {
    let (folders, mut assets) = if query.is_empty() && view.kind.is_none() {
        listing(db, folder)
    } else {
        let found = db
            .iter()
            .filter(|entry| view.kind.is_none_or(|kind| kind == entry.kind))
            .filter(|entry| query.matches(&entry.path))
            .collect();
        (Vec::new(), found)
    };
    let by_name = |a: &&Entry, b: &&Entry| {
        caseless(file_name(&a.path), file_name(&b.path)).then_with(|| a.path.cmp(&b.path))
    };
    match view.sort {
        AssetSort::Name => assets.sort_by(by_name),
        AssetSort::Type => assets.sort_by(|a, b| {
            caseless(extension(&a.path), extension(&b.path)).then_with(|| by_name(a, b))
        }),
        AssetSort::Size => assets.sort_by(|a, b| a.bytes.cmp(&b.bytes).then_with(|| by_name(a, b))),
        AssetSort::Modified => {
            assets.sort_by(|a, b| a.modified.cmp(&b.modified).then_with(|| by_name(a, b)));
        }
    }
    if view.descending {
        assets.reverse();
    }
    (folders, assets)
}

/// `a` against `b` ignoring ASCII case, without lowering either into a copy:
/// this runs on every comparison of every frame's sort.
fn caseless(a: &str, b: &str) -> std::cmp::Ordering {
    a.bytes()
        .map(|byte| byte.to_ascii_lowercase())
        .cmp(b.bytes().map(|byte| byte.to_ascii_lowercase()))
}

fn extension(path: &str) -> &str {
    file_name(path)
        .rsplit_once('.')
        .map_or("", |(_, extension)| extension)
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

const SORTS: [(AssetSort, &str); 4] = [
    (AssetSort::Name, "Name"),
    (AssetSort::Type, "Type"),
    (AssetSort::Size, "Size"),
    (AssetSort::Modified, "Modified"),
];

/// The kinds the browser can be narrowed to. Folders are not among them: a
/// kind lists the whole project, where a folder has nowhere to be opened.
const KINDS: [(AssetKind, &str); 8] = [
    (AssetKind::Texture, "Textures"),
    (AssetKind::Mesh, "Meshes"),
    (AssetKind::Material, "Materials"),
    (AssetKind::Prefab, "Prefabs"),
    (AssetKind::Scene, "Scenes"),
    (AssetKind::Font, "Fonts"),
    (AssetKind::Audio, "Sounds"),
    (AssetKind::Shader, "Surface shaders"),
];

fn plural(kind: AssetKind) -> String {
    KINDS
        .iter()
        .find(|(k, _)| *k == kind)
        .map_or_else(|| kind.label().to_owned(), |(_, name)| name.to_lowercase())
}

/// Rows or tiles, what they are ordered by and which way, and which kind;
/// added right to left, so they read left to right at the bar's end.
fn view_controls(ui: &mut egui::Ui, view: &mut AssetView) {
    let kind = view
        .kind
        .and_then(|kind| KINDS.iter().find(|(k, _)| *k == kind))
        .map_or("All kinds", |k| k.1);
    egui::ComboBox::from_id_salt("asset_kind")
        .selected_text(kind)
        .width(96.0)
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut view.kind, None, "All kinds");
            for (kind, name) in KINDS {
                ui.selectable_value(&mut view.kind, Some(kind), name);
            }
        });
    let (arrow, hint) = if view.descending {
        ("⬇", "Descending. Click for ascending")
    } else {
        ("⬆", "Ascending. Click for descending")
    };
    if ui.button(arrow).on_hover_text(hint).clicked() {
        view.descending = !view.descending;
    }
    let sort = SORTS
        .iter()
        .find(|(s, _)| *s == view.sort)
        .map_or("", |s| s.1);
    egui::ComboBox::from_id_salt("asset_sort")
        .selected_text(format!("By {}", sort.to_lowercase()))
        .width(96.0)
        .show_ui(ui, |ui| {
            for (sort, name) in SORTS {
                ui.selectable_value(&mut view.sort, sort, name);
            }
        });
    let grid = view.layout == AssetLayout::Grid;
    if widgets::toggle_icon(ui, icons::grid(), grid, "Show tiles with large pictures").clicked() {
        view.layout = if grid {
            AssetLayout::List
        } else {
            AssetLayout::Grid
        };
    }
}

/// One thing the browser shows.
enum Item<'a> {
    Folder(&'a Folder),
    Asset(&'a Entry),
}

impl Item<'_> {
    fn path(&self) -> &str {
        match self {
            Item::Folder(folder) => &folder.path,
            Item::Asset(entry) => &entry.path,
        }
    }
}

/// What drawing a row or a tile needs, gathered so each cell takes one
/// argument.
struct Cells<'a> {
    db: &'a AssetDatabase,
    in_scene: &'a FxHashMap<AssetId, usize>,
    /// The sound the editor's preview is playing.
    previewing: Option<AssetId>,
    layout: AssetLayout,
    /// Listing the whole project rather than one folder, so each asset says
    /// which folder it is in.
    searching: bool,
    state: &'a mut EditorState,
    previews: &'a mut Previews,
    renaming: &'a mut Option<Renaming>,
    marked: &'a [String],
    /// The cell clicked this frame and the modifiers held, for [`mark`], which
    /// needs the whole listing.
    clicked: Option<(String, egui::Modifiers)>,
    action: &'a mut Option<Action>,
    /// Each folder drawn, where: what a file dropped from outside lands in.
    folders: Vec<(egui::Rect, String)>,
}

impl Cells<'_> {
    fn draw(&mut self, ui: &mut egui::Ui, item: &Item) {
        match item {
            Item::Folder(folder) => self.folder(ui, folder),
            Item::Asset(entry) => self.asset(ui, entry),
        }
    }

    /// The rest of the view, below the rows, is the open folder.
    fn rest(&mut self, ui: &mut egui::Ui, folder: &str) {
        let rest = (ui.clip_rect().bottom() - ui.cursor().top()).max(ROW);
        let size = egui::vec2(ui.available_width(), rest);
        ui.allocate_response(size, egui::Sense::click())
            .context_menu(|ui| {
                new_folder_item(ui, folder, self.action);
                new_material_item(ui, folder, self.action);
                ui.separator();
                if ui
                    .button("Rescan assets")
                    .on_hover_text(
                        "Read the assets directory again. It is watched, so this is only \
                         for a change the watcher missed",
                    )
                    .clicked()
                {
                    *self.action = Some(Action::Rescan);
                    ui.close_menu();
                }
            });
    }

    fn is_marked(&self, path: &str) -> bool {
        self.marked
            .binary_search_by(|marked| marked.as_str().cmp(path))
            .is_ok()
    }

    /// What a drag of `path` is labelled with beside the pointer: its name,
    /// or how many it carries when it is one of the marked.
    fn carrying(&self, path: &str, name: &str) -> String {
        match self.marked.len() {
            n if n > 1 && self.is_marked(path) => format!("{n} items"),
            _ => name.to_owned(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn cell(
        &self,
        ui: &mut egui::Ui,
        sense: egui::Sense,
        mark: Mark,
        name: &str,
        folder: &str,
        end: End,
        marked: bool,
    ) -> egui::Response {
        match self.layout {
            AssetLayout::List => row(ui, LIST_ROW, sense, mark, name, folder, end, marked),
            AssetLayout::Grid => tile(ui, sense, mark, name, end, marked),
        }
    }

    fn folder(&mut self, ui: &mut egui::Ui, folder: &Folder) {
        let weak = ui.visuals().weak_text_color();
        let tint = folder
            .tint
            .map_or(weak, |Tint([r, g, b])| egui::Color32::from_rgb(r, g, b));
        let name = file_name(&folder.path);
        let response = self.cell(
            ui,
            egui::Sense::click_and_drag(),
            Mark::Glyph(icons::folder(), tint),
            name,
            "",
            End::Nothing,
            self.is_marked(&folder.path),
        );
        self.name_field(ui, &folder.path, response.rect);
        self.folders.push((response.rect, folder.path.clone()));
        if let Some(moved) = dropped_on(ui, &response, self.db, self.marked, &folder.path) {
            *self.action = Some(moved);
        }
        carry(
            ui,
            &response,
            || CarriedFolder(folder.path.clone()),
            &self.carrying(&folder.path, name),
        );
        let modifiers = ui.input(|input| input.modifiers);
        if response.clicked() && (modifiers.command || modifiers.shift) {
            self.clicked = Some((folder.path.clone(), modifiers));
        } else if response.double_clicked() || response.clicked() {
            *self.action = Some(Action::Open(folder.path.clone()));
        }
        let (db, marked) = (self.db, self.marked);
        let action = &mut *self.action;
        response
            .on_hover_text(format!(
                "{}\nRight-click to rename, tint, or make a folder, material or shader in it. \
                 Drag onto a folder to move it, with its .meta; not undoable",
                folder.path
            ))
            .context_menu(|ui| {
                rename_item(ui, &folder.path, name, action);
                new_folder_item(ui, &folder.path, action);
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
                ui.separator();
                delete_item(ui, chosen(db, marked, &folder.path), action);
            });
    }

    /// One asset: marked by a click and dragged to wherever takes its kind;
    /// a double-click does what its kind is for.
    fn asset(&mut self, ui: &mut egui::Ui, entry: &Entry) {
        let weak = ui.visuals().weak_text_color();
        let blend = crate::asset_db::blend::is_blend(std::path::Path::new(&entry.path));
        let (icon, colour, hint) = match entry.kind {
            AssetKind::Mesh if blend => (
                icons::box_(),
                theme::KIND_MESH,
                "Drag into the scene to place it, or onto a mesh renderer to draw it. \
                 Double-click to open it in Blender",
            ),
            AssetKind::Mesh => (
                icons::box_(),
                theme::KIND_MESH,
                "Drag into the scene to place it, or onto a mesh renderer to draw it. \
                 Double-click to place it",
            ),
            AssetKind::Texture => (
                icons::image(),
                weak,
                "Drag onto a material asset's texture slot to sample it. Double-click for \
                 its import settings",
            ),
            AssetKind::Material => (
                icons::palette(),
                weak,
                "Drag onto a mesh renderer's material to draw with it. Double-click to edit \
                 it in the Import tab",
            ),
            AssetKind::Scene => (icons::globe(), weak, "Double-click to open it"),
            AssetKind::Font => (
                icons::text_t(),
                weak,
                "Drag onto a script's font field to set its UI text in it. Double-click to \
                 show it in the Import tab",
            ),
            AssetKind::Audio if self.previewing == Some(entry.id) => (
                icons::speaker_high(),
                theme::KIND_AUDIO,
                "Playing. Double-click to stop it",
            ),
            AssetKind::Audio => (
                icons::file_audio(),
                weak,
                "Double-click to hear it, or drag onto an audio source's clip",
            ),
            AssetKind::Shader => (
                icons::file_code(),
                weak,
                "Double-click to edit it. Assign it to a material asset to draw through it",
            ),
            AssetKind::Prefab | AssetKind::Folder => (icons::layers(), weak, PLACE_HINT),
        };
        // Only a state the user must know about is a line of the hover: an
        // asset that imported, from the cache or not, is the ordinary case.
        let (status, detail) = match (&entry.status, self.db.state(entry.id)) {
            (Status::Failed(message), _) => (
                Some((icons::warning(), theme::ERROR)),
                format!("Failed: {message}\n"),
            ),
            (Status::Queued, _) | (_, LoadState::Loading) => (
                Some((icons::hourglass(), theme::PENDING)),
                "Importing\n".to_owned(),
            ),
            (Status::Imported { .. } | Status::Indexed, _) => (None, String::new()),
        };
        let in_scene = self.in_scene.get(&entry.id).copied().unwrap_or(0);
        let end = match status {
            Some((icon, colour)) => End::Glyph(icon, colour),
            None if in_scene > 0 => End::Count(in_scene),
            None => End::Nothing,
        };
        let mark = match self.previews.asset(ui.ctx(), self.db, entry.id) {
            Some(picture) => Mark::Picture(picture.id()),
            None => Mark::Glyph(icon, colour),
        };
        let marked = self.is_marked(&entry.path) || self.state.revealed_asset == Some(entry.id);
        let name = file_name(&entry.path);
        let folder = if self.searching {
            parent(&entry.path)
        } else {
            ""
        };
        let sense = egui::Sense::click_and_drag();
        let response = self.cell(ui, sense, mark, name, folder, end, marked);
        if response.clicked() {
            self.clicked = Some((entry.path.clone(), ui.input(|input| input.modifiers)));
        }
        if response.double_clicked() {
            match entry.kind {
                AssetKind::Prefab => self.state.request_instantiate(entry.id, None),
                AssetKind::Mesh if blend => {
                    *self.action = Some(Action::OpenInBlender(entry.path.clone()));
                }
                AssetKind::Mesh => self.state.request_instantiate(entry.id, None),
                AssetKind::Audio => *self.action = Some(Action::TogglePreview(entry.id)),
                AssetKind::Shader => self.state.open_shader(entry.id),
                AssetKind::Scene => {
                    let path = self.db.dir().join(&entry.path);
                    self.state
                        .request_open_scene_checked(path.to_string_lossy());
                }
                AssetKind::Material | AssetKind::Texture | AssetKind::Font => {
                    self.state.revealed_asset = Some(entry.id);
                    self.state.request_import_page();
                }
                AssetKind::Folder => {}
            }
        }
        self.name_field(ui, &entry.path, response.rect);
        carry(
            ui,
            &response,
            || entry.id,
            &self.carrying(&entry.path, name),
        );
        let (db, marked) = (self.db, self.marked);
        // Named for what the Import tab does with it. The id is shown there,
        // and copied from here, rather than on every hover.
        let opens = match entry.kind {
            AssetKind::Material => Some("Edit material"),
            _ if entry.settings != Settings::None => Some("Import settings"),
            _ => None,
        };
        response
            .on_hover_text(format!(
                "{}\n{detail}{hint}\nDrag onto a folder to move it, with its .meta; not \
                 undoable",
                entry.path
            ))
            .context_menu(|ui| {
                if entry.kind == AssetKind::Shader && ui.button("Edit").clicked() {
                    self.state.open_shader(entry.id);
                    ui.close_menu();
                }
                if blend && ui.button("Open in Blender").clicked() {
                    *self.action = Some(Action::OpenInBlender(entry.path.clone()));
                    ui.close_menu();
                }
                rename_item(ui, &entry.path, entry.name(), self.action);
                if let Some(opens) = opens
                    && ui.button(opens).clicked()
                {
                    self.state.revealed_asset = Some(entry.id);
                    self.state.request_import_page();
                    ui.close_menu();
                }
                if ui.button("Copy id").clicked() {
                    ui.ctx().copy_text(entry.id.to_string());
                    ui.close_menu();
                }
                ui.separator();
                delete_item(ui, chosen(db, marked, &entry.path), self.action);
            });
    }

    /// The name being edited, laid over the cell's own while `path` is the
    /// one being renamed. Enter renames; Escape, or clicking away, does not.
    fn name_field(&mut self, ui: &mut egui::Ui, path: &str, cell: egui::Rect) {
        let Some(renaming) = self.renaming.as_mut().filter(|r| r.path == path) else {
            return;
        };
        let height = ui.spacing().interact_size.y;
        let field = match self.layout {
            AssetLayout::List => egui::Rect::from_min_max(
                egui::pos2(
                    cell.left() + widgets::PAD + LIST_ROW,
                    cell.center().y - height / 2.0,
                ),
                egui::pos2(cell.right() - widgets::PAD, cell.center().y + height / 2.0),
            ),
            AssetLayout::Grid => {
                egui::Rect::from_min_max(egui::pos2(cell.left(), cell.bottom() - height), cell.max)
            }
        };
        let output = ui
            .scope_builder(egui::UiBuilder::new().max_rect(field), |ui| {
                egui::TextEdit::singleline(&mut renaming.text)
                    .desired_width(field.width())
                    .show(ui)
            })
            .inner;
        let response = output.response;
        if !renaming.started {
            renaming.started = true;
            response.request_focus();
            let mut state = output.state;
            let all = egui::text::CCursorRange::two(
                egui::text::CCursor::new(0),
                egui::text::CCursor::new(renaming.text.chars().count()),
            );
            state.cursor.set_char_range(Some(all));
            state.store(ui.ctx(), response.id);
        } else if response.lost_focus() {
            *self.action = Some(if ui.input(|input| input.key_pressed(egui::Key::Enter)) {
                Action::Rename(path.to_owned(), renaming.text.clone())
            } else {
                Action::CancelRename
            });
        }
    }
}

/// Start a drag of what `payload` makes, and while it lasts draw what is being
/// carried beside the pointer, since the row itself stays put.
fn carry<T: std::any::Any + Send + Sync>(
    ui: &egui::Ui,
    response: &egui::Response,
    payload: impl FnOnce() -> T,
    name: &str,
) {
    if response.drag_started() {
        response.dnd_set_drag_payload(payload());
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

/// What a row or a tile pictures its asset with: the thumbnail once there is
/// one, its kind's glyph until then and for a kind with none.
enum Mark {
    Glyph(icons::Icon, egui::Color32),
    Picture(egui::TextureId),
}

fn paint_mark(ui: &egui::Ui, mark: Mark, rect: egui::Rect, glyph: f32) {
    match mark {
        Mark::Picture(texture) => {
            let visuals = ui.visuals();
            ui.painter().rect_filled(
                rect,
                visuals.widgets.inactive.corner_radius,
                visuals.extreme_bg_color,
            );
            egui::Image::new((texture, rect.size())).paint_at(ui, rect);
        }
        Mark::Glyph(icon, colour) => {
            let at = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(glyph));
            icons::paint(ui, icon, at, colour);
        }
    }
}

/// What a row shows at its far end, and a tile on its picture.
enum End {
    Nothing,
    Glyph(icons::Icon, egui::Color32),
    /// How many instances of a prefab the scene holds.
    Count(usize),
}

fn highlight(ui: &egui::Ui, rect: egui::Rect, response: &egui::Response, marked: bool) {
    let visuals = ui.visuals();
    let radius = visuals.widgets.inactive.corner_radius;
    if marked {
        ui.painter()
            .rect_filled(rect, radius, visuals.selection.bg_fill);
    } else if response.hovered() || response.dragged() {
        ui.painter()
            .rect_filled(rect, radius, visuals.widgets.hovered.weak_bg_fill);
    }
}

/// One line: a picture, a name, the folder it is in when that is not the one
/// open, and at the far end a status glyph or a count.
#[allow(clippy::too_many_arguments)]
fn row(
    ui: &mut egui::Ui,
    height: f32,
    sense: egui::Sense,
    mark: Mark,
    name: &str,
    folder: &str,
    end: End,
    marked: bool,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), height), sense);
    highlight(ui, rect, &response, marked);
    let visuals = ui.visuals();
    let weak = visuals.weak_text_color();
    let side = height - 4.0;
    let picture = egui::Rect::from_min_size(
        egui::pos2(
            rect.left() + widgets::PAD - 3.0,
            rect.center().y - side / 2.0,
        ),
        egui::Vec2::splat(side),
    );
    paint_mark(ui, mark, picture, (side * 0.65).round());
    let font = egui::TextStyle::Body.resolve(ui.style());
    let label = ui.painter().text(
        egui::pos2(picture.right() + 6.0, rect.center().y),
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
        End::Count(count) => {
            ui.painter().text(
                egui::pos2(rect.right() - widgets::PAD, rect.center().y),
                egui::Align2::RIGHT_CENTER,
                format!("{count} in scene"),
                font,
                weak,
            );
        }
    }
    response
}

/// One tile: a large picture with its name under it, cut short with an
/// ellipsis when it is wider than the tile, and a status glyph or a count on
/// the picture's corner.
fn tile(
    ui: &mut egui::Ui,
    sense: egui::Sense,
    mark: Mark,
    name: &str,
    end: End,
    marked: bool,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(TILE, sense);
    highlight(ui, rect, &response, marked);
    let visuals = ui.visuals();
    let picture = egui::Rect::from_min_size(
        egui::pos2(rect.center().x - TILE_THUMB / 2.0, rect.top() + 6.0),
        egui::Vec2::splat(TILE_THUMB),
    );
    paint_mark(ui, mark, picture, 36.0);

    let font = egui::TextStyle::Body.resolve(ui.style());
    let mut job =
        egui::text::LayoutJob::simple_singleline(name.to_owned(), font, visuals.text_color());
    job.wrap = egui::text::TextWrapping::truncate_at_width(rect.width() - 8.0);
    let galley = ui.painter().layout_job(job);
    let at = egui::pos2(
        rect.center().x - galley.size().x / 2.0,
        picture.bottom() + 4.0,
    );
    ui.painter().galley(at, galley, visuals.text_color());

    match end {
        End::Nothing => {}
        End::Glyph(icon, colour) => {
            let centre = picture.right_top() + egui::vec2(-8.0, 8.0);
            ui.painter()
                .circle_filled(centre, 10.0, visuals.extreme_bg_color);
            let at = egui::Rect::from_center_size(centre, egui::Vec2::splat(icons::ROW - 2.0));
            icons::paint(ui, icon, at, colour);
        }
        End::Count(count) => {
            ui.painter().text(
                picture.right_bottom() - egui::vec2(2.0, 2.0),
                egui::Align2::RIGHT_BOTTOM,
                format!("×{count}"),
                egui::TextStyle::Small.resolve(ui.style()),
                visuals.strong_text_color(),
            );
        }
    }
    response
}

pub(super) fn log_error(world: &World, message: String) {
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
            "No prefabs under {dir}. A .oprefab is a scene file holding one entity tree."
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
                let end = match in_scene.get(&prefab.id) {
                    Some(&count) => End::Count(count),
                    None => End::Nothing,
                };
                let marked = state.revealed_asset == Some(prefab.id);
                let response = row(
                    ui,
                    ROW,
                    egui::Sense::click_and_drag(),
                    Mark::Glyph(icons::box_(), weak),
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
                carry(ui, &response, || prefab.id, prefab.name());
                response.on_hover_text(format!("{}\n{PLACE_HINT}", prefab.path));
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_db::AssetDatabase;
    use crate::asset_db::tests::Dir;

    #[test]
    fn a_query_is_words_and_extensions() {
        let query = Query::parse("  .PNG rock .jpg ");
        assert_eq!(
            query,
            Query {
                words: vec!["rock".to_owned()],
                extensions: vec!["png".to_owned(), "jpg".to_owned()],
            }
        );
        assert!(query.matches("textures/Rocks/rock_color.png"));
        assert!(query.matches("rock.JPG"));
        assert!(!query.matches("rock.omat"), "not one of the extensions");
        assert!(!query.matches("bark.png"), "not the word");
        assert!(Query::parse(".omat").matches("materials/brick.omat"));
        assert!(Query::parse("").is_empty());
    }

    fn sized(dir: &Dir, rel: &str, bytes: usize, age: u64) {
        let path = dir.write(rel, &vec![b'x'; bytes]);
        let written = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(age);
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(written)
            .unwrap();
    }

    /// A filter or a kind lists the whole project; the view orders it, and
    /// reverses the order on request. Folders only come with the plain listing.
    #[test]
    fn a_folder_is_listed_filtered_and_ordered_as_the_view_says() {
        let dir = Dir::new("browser-order");
        sized(&dir, "b.png", 30, 3_000);
        sized(&dir, "a.jpg", 10, 1_000);
        sized(&dir, "c.oprefab", 20, 2_000);
        sized(&dir, "deep/d.png", 40, 4_000);
        let (db, _) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let names = |query: &str, view: AssetView| -> (Vec<String>, Vec<String>) {
            let (folders, assets) = shown(&db, "", &Query::parse(query), &view);
            (
                folders.iter().map(|f| f.path.clone()).collect(),
                assets.iter().map(|a| a.path.clone()).collect(),
            )
        };
        let view = |sort, descending| AssetView {
            sort,
            descending,
            ..AssetView::default()
        };

        let (folders, assets) = names("", view(AssetSort::Name, false));
        assert_eq!(folders, ["deep"]);
        assert_eq!(assets, ["a.jpg", "b.png", "c.oprefab"]);
        assert_eq!(
            names("", view(AssetSort::Name, true)).1,
            ["c.oprefab", "b.png", "a.jpg"]
        );
        assert_eq!(
            names("", view(AssetSort::Size, false)).1,
            ["a.jpg", "c.oprefab", "b.png"]
        );
        assert_eq!(
            names("", view(AssetSort::Modified, true)).1,
            ["b.png", "c.oprefab", "a.jpg"]
        );
        assert_eq!(
            names("", view(AssetSort::Type, false)).1,
            ["a.jpg", "c.oprefab", "b.png"],
            "by extension, then name"
        );

        let (folders, assets) = names(".png", view(AssetSort::Name, false));
        assert!(folders.is_empty(), "a search lists assets only");
        assert_eq!(assets, ["b.png", "deep/d.png"]);
        let textures = AssetView {
            kind: Some(AssetKind::Texture),
            ..AssetView::default()
        };
        assert_eq!(names("", textures).1, ["a.jpg", "b.png", "deep/d.png"]);
    }
}
