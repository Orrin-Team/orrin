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
//! asset's path, and each match says which folder it is in. `.png`-style
//! words and the kind menu narrow it by type.
//!
//! The folder shows as rows or as tiles, each pictured by its import's
//! thumbnail where it has one ([`Previews::asset`]). Both are drawn only for
//! what is on screen, so a project of thousands of assets costs a screenful.
//! A click marks an asset, which the Import tab shows; Rename moves the file
//! with its sidecar, and files dropped from outside the editor are copied in
//! (see [`AssetDatabase::bring_in`]).
//!
//! Without a project there is no asset database, and the tab lists the
//! prefabs under `assets/` as it always has.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use orrin_ecs::{FxHashMap, World};
use orrin_registry::AssetId;

use crate::asset_db::{AssetDatabase, AssetKind, Entry, Folder, LoadState, Status, Tint};
use crate::editor::previews::Previews;
use crate::editor::state::{AssetLayout, AssetSort, AssetView, EditorState};
use crate::editor::theme;
use crate::editor::{icons, widgets};
use crate::scene::{LogBuffer, LogLevel, PrefabInstance, Prefabs, Time};

/// What the stand-in for the file dialog picks, in tests.
#[cfg(test)]
pub(super) static PICKED: Mutex<Option<Vec<PathBuf>>> = Mutex::new(None);

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
    if world.get_resource::<AssetDatabase>().is_some() {
        browser(ui, world, state, previews);
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
    /// Edit this path's name in place, starting from this text.
    StartRename(String, String),
    Rename(String, String),
    CancelRename,
    /// Copy these files from outside the project into this folder.
    BringIn(Vec<PathBuf>, String),
    /// Open a file dialog for files to copy into the open folder.
    PickFiles,
}

/// The browser's place, kept in egui's memory rather than the editor state:
/// which folder is open, which revealed asset it last opened a folder for,
/// the name being edited, if one is, and the file dialog, if one is open.
#[derive(Clone, Default)]
struct Place {
    folder: String,
    revealed: Option<AssetId>,
    renaming: Option<Renaming>,
    picking: Option<Picking>,
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

fn browser(ui: &mut egui::Ui, world: &World, state: &mut EditorState, previews: &mut Previews) {
    let area = ui.max_rect();
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
        header(
            ui,
            &db,
            &place.folder,
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
                    "Nothing to import under {}. PNG and JPEG textures, glTF meshes, .fmat \
                     materials and .fprefab files are assets.",
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
        let f2 = ui.rect_contains_pointer(ui.max_rect())
            && !ui.ctx().wants_keyboard_input()
            && ui.input(|input| input.key_pressed(egui::Key::F2));
        if f2 && let Some(entry) = state.revealed_asset.and_then(|id| db.get(id)) {
            action = Some(Action::StartRename(
                entry.path.clone(),
                entry.name().to_owned(),
            ));
        }
        let mut cells = Cells {
            db: &db,
            in_scene: &in_scene,
            layout: view.layout,
            searching,
            state,
            previews,
            renaming: &mut place.renaming,
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
        let folders = cells.folders;
        if let Some(folder) = file_drop(ui, area, &folders, open, &mut action) {
            ui.painter().rect_stroke(
                folder,
                ui.visuals().widgets.inactive.corner_radius,
                ui.visuals().selection.stroke,
                egui::StrokeKind::Inside,
            );
        }
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
        Some(Action::StartRename(path, text)) => {
            place.renaming = Some(Renaming {
                path,
                text,
                started: false,
            });
        }
        Some(Action::CancelRename) => place.renaming = None,
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
                    if place.folder == path {
                        place.folder = to;
                    } else if let Some(rest) = place.folder.strip_prefix(&format!("{path}/")) {
                        place.folder = format!("{to}/{rest}");
                    }
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
            &["png", "jpg", "jpeg", "gltf", "glb", "fmat", "fprefab"],
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
}

/// The browser's one bar: rescan and up, and where the browser is; the
/// filter, in whatever width is left; the view's controls; and how much is
/// still importing or has failed.
fn header(
    ui: &mut egui::Ui,
    db: &AssetDatabase,
    folder: &str,
    picking: bool,
    query: &mut String,
    view: &mut AssetView,
    action: &mut Option<Action>,
) {
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
        let import = ui
            .add_enabled_ui(!picking, |ui| {
                icons::button(ui, icons::plus(), icons::ROW - 2.0, weak, false)
            })
            .inner
            .on_hover_text("Copy files from disk into this folder. Dropping them here does too");
        if import.clicked() {
            *action = Some(Action::PickFiles);
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
            view_controls(ui, view);
            widgets::search(ui, query, "Filter assets — words, or .png .fmat");
        });
    });
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
const KINDS: [(AssetKind, &str); 4] = [
    (AssetKind::Texture, "Textures"),
    (AssetKind::Mesh, "Meshes"),
    (AssetKind::Material, "Materials"),
    (AssetKind::Prefab, "Prefabs"),
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

/// What drawing a row or a tile needs, gathered so each cell takes one
/// argument.
struct Cells<'a> {
    db: &'a AssetDatabase,
    in_scene: &'a FxHashMap<AssetId, usize>,
    layout: AssetLayout,
    /// Listing the whole project rather than one folder, so each asset says
    /// which folder it is in.
    searching: bool,
    state: &'a mut EditorState,
    previews: &'a mut Previews,
    renaming: &'a mut Option<Renaming>,
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
            .context_menu(|ui| new_material_item(ui, folder, self.action));
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
        let response = self.cell(
            ui,
            egui::Sense::click(),
            Mark::Glyph(icons::folder(), tint),
            file_name(&folder.path),
            "",
            End::Nothing,
            false,
        );
        self.name_field(ui, &folder.path, response.rect);
        self.folders.push((response.rect, folder.path.clone()));
        let action = &mut *self.action;
        if response.double_clicked() || response.clicked() {
            *action = Some(Action::Open(folder.path.clone()));
        }
        response
            .on_hover_text(format!(
                "{}\nRight-click to rename, tint, or make a material in it",
                folder.path
            ))
            .context_menu(|ui| {
                rename_item(ui, &folder.path, file_name(&folder.path), action);
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

    /// One asset: marked by a click and dragged to wherever takes its kind; a
    /// prefab is also placed by a double-click, as in the prefab list.
    fn asset(&mut self, ui: &mut egui::Ui, entry: &Entry) {
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
        let (status, detail) = match (&entry.status, self.db.state(entry.id)) {
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
        let prefab = entry.kind == AssetKind::Prefab;
        let marked = self.state.revealed_asset == Some(entry.id);
        let name = file_name(&entry.path);
        let folder = if self.searching {
            parent(&entry.path)
        } else {
            ""
        };
        let sense = egui::Sense::click_and_drag();
        let response = self.cell(ui, sense, mark, name, folder, end, marked);
        if response.clicked() {
            self.state.revealed_asset = Some(entry.id);
        }
        if prefab && response.double_clicked() {
            self.state.request_instantiate(entry.id, None);
        }
        self.name_field(ui, &entry.path, response.rect);
        carry(ui, &response, entry.id, name);
        response
            .on_hover_text(format!("{}\n{}\n{detail}\n{hint}", entry.path, entry.id))
            .context_menu(|ui| {
                rename_item(ui, &entry.path, entry.name(), self.action);
                if ui.button("Import settings").clicked() {
                    self.state.revealed_asset = Some(entry.id);
                    self.state.request_import_page();
                    ui.close_menu();
                }
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
        assert!(!query.matches("rock.fmat"), "not one of the extensions");
        assert!(!query.matches("bark.png"), "not the word");
        assert!(Query::parse(".fmat").matches("materials/brick.fmat"));
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
        sized(&dir, "c.fprefab", 20, 2_000);
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
        assert_eq!(assets, ["a.jpg", "b.png", "c.fprefab"]);
        assert_eq!(
            names("", view(AssetSort::Name, true)).1,
            ["c.fprefab", "b.png", "a.jpg"]
        );
        assert_eq!(
            names("", view(AssetSort::Size, false)).1,
            ["a.jpg", "c.fprefab", "b.png"]
        );
        assert_eq!(
            names("", view(AssetSort::Modified, true)).1,
            ["b.png", "c.fprefab", "a.jpg"]
        );
        assert_eq!(
            names("", view(AssetSort::Type, false)).1,
            ["c.fprefab", "a.jpg", "b.png"],
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

    #[test]
    fn a_folder_goes_up_to_its_parent_and_the_root_stays_put() {
        assert_eq!(parent("props/wood"), "props");
        assert_eq!(parent("props"), "");
        assert_eq!(parent(""), "");
    }
}
