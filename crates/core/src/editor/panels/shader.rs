//! The Shader tab: one surface shader's text, edited in place, beside what its
//! last save compiled to — the material sphere drawn through it, the
//! parameters it declares, and its errors at their lines.
//!
//! **The text is the tab's until it is saved.** The buffer lives in the
//! editor's state and is written only by Save (or Cmd/Ctrl+S while the code has
//! focus), through [`AssetDatabase::write_shader`], which queues the import as it
//! writes. That is not a scene edit: the text edit has its own undo, and a save
//! is a file written, as an external editor's would be. The status in the
//! header reads Compiling from that moment until the import lands, then
//! Compiled or the number of errors.
//!
//! **Errors describe the last save, not the screen.** They come from the
//! database, which only knows what was compiled; once the text has changed
//! since, the list says so and the gutter's marks fade, because a line number
//! from the saved text may not be the line it now points at.
//!
//! **The file is the truth and an external editor is welcome.** The tab
//! watches the size and time the database's scan last saw for the file and
//! rereads it only when they move. A clean buffer follows the file; a dirty
//! one is kept, with a notice offering to reload — typed text is never thrown
//! away unasked, and Save stays off until the user has chosen between their
//! text and the file's, so it never lands on a newer file by accident.
//!
//! **The preview is a material, drawn by the renderer.** The sphere is the
//! material preview's `Shader` viewer, drawn by the same pass as the
//! inspector's and asking for nothing while nothing on it changes. It shows a
//! material of the project's that uses the shader, or, when none does, the
//! database's own material at the defaults with every parameter at the
//! shader's default, so a new shader shows something from its first compile.
//!
//! **Keys typed into the code stay there.** egui reports it wants the keyboard
//! while the code has focus, which keeps every key from the game and the fly
//! camera and every unmodified shortcut from the editor; the two shortcuts
//! that ignore focus elsewhere, save and play, stand aside for it in
//! `panels::shortcuts`, and Cmd/Ctrl+S saves the shader instead.
//!
//! **Shape follows the dock.** Wider than [`WIDE`], the preview and parameters
//! sit in a column beside the code; narrower, below it.

use std::time::SystemTime;

use orrin_ecs::World;
use orrin_registry::AssetId;

use super::assets::log_error;
use super::figures;
use crate::asset_db::{AssetDatabase, AssetKind};
use crate::editor::glsl::Highlighter;
use crate::editor::icons;
use crate::editor::state::EditorState;
use crate::editor::theme::{self, Syntax};
use crate::editor::widgets::{self, Tool};
use crate::gfx::surface::{ColorSpace, DefaultTexture, ParamKind, SurfaceError};
use crate::gfx::vulkan::material_preview::Viewer;
use crate::scene::MaterialHandle;

/// The code's text edit, which `panels::shortcuts` asks about.
pub fn code_id() -> egui::Id {
    egui::Id::new("shader_code")
}

/// From this width the preview and parameters sit beside the code.
const WIDE: f32 = 640.0;
/// The side column's width when they do.
const SIDE: f32 = 260.0;
const ERROR_ROW: f32 = 20.0;

/// What the sphere previews the shader on.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum PreviewOn {
    /// The first material that uses the shader, or its defaults without one.
    #[default]
    First,
    Defaults,
    Material(AssetId),
}

/// The file's size and time as the database's scan last saw them.
type Stamp = (u64, Option<SystemTime>);

/// Everything the tab keeps between frames.
#[derive(Default)]
pub struct ShaderEditor {
    open: Option<AssetId>,
    text: String,
    /// The text as last loaded or saved.
    saved: String,
    dirty: bool,
    stamp: Option<Stamp>,
    /// The file changed on disk while the buffer had unsaved text.
    changed_on_disk: bool,
    /// A shader to open, asked for from elsewhere in the editor.
    request: Option<AssetId>,
    /// One asked for over unsaved text, waiting on what becomes of it.
    pending: Option<AssetId>,
    /// Put the cursor at the start of this line, counted from one.
    goto: Option<u32>,
    preview: PreviewOn,
    highlighter: Highlighter,
}

impl ShaderEditor {
    /// Open shader `id` the next time the tab draws.
    pub fn request(&mut self, id: AssetId) {
        self.request = Some(id);
    }

    #[cfg(all(test, feature = "shader-compiler"))]
    pub fn open(&self) -> Option<AssetId> {
        self.open
    }

    #[cfg(all(test, feature = "shader-compiler"))]
    pub fn text(&self) -> &str {
        &self.text
    }

    #[cfg(all(test, feature = "shader-compiler"))]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    #[cfg(all(test, feature = "shader-compiler"))]
    pub fn changed_on_disk(&self) -> bool {
        self.changed_on_disk
    }

    /// The line the cursor is to be put on when the code next draws.
    #[cfg(all(test, feature = "shader-compiler"))]
    pub fn goto(&self) -> Option<u32> {
        self.goto
    }

    /// How many times the code has been tokenised and laid out.
    #[cfg(all(test, feature = "shader-compiler"))]
    pub fn highlight_runs(&self) -> u32 {
        self.highlighter.runs()
    }

    fn load(&mut self, db: &AssetDatabase, id: AssetId) -> Result<(), String> {
        let text = db.shader_source(id)?;
        self.open = Some(id);
        self.text.clone_from(&text);
        self.saved = text;
        self.dirty = false;
        self.changed_on_disk = false;
        self.stamp = stamp(db, id);
        self.goto = None;
        self.preview = PreviewOn::default();
        Ok(())
    }

    /// Take the file's text when the scan has seen it change: into a clean
    /// buffer, or as a notice beside a dirty one.
    fn follow_disk(&mut self, db: &AssetDatabase, id: AssetId) {
        let now = stamp(db, id);
        if now == self.stamp {
            return;
        }
        self.stamp = now;
        let Ok(disk) = db.shader_source(id) else {
            return;
        };
        if disk == self.saved {
            return;
        }
        if self.dirty {
            self.changed_on_disk = true;
        } else {
            self.text.clone_from(&disk);
            self.saved = disk;
        }
    }
}

fn stamp(db: &AssetDatabase, id: AssetId) -> Option<Stamp> {
    db.get(id).map(|entry| (entry.bytes, entry.modified))
}

/// Where a shader's last save has got to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    Compiling,
    Compiled,
    Errors(usize),
}

impl Status {
    pub fn of(db: &AssetDatabase, id: AssetId) -> Self {
        if db
            .get(id)
            .is_some_and(|entry| entry.status == crate::asset_db::Status::Queued)
        {
            return Self::Compiling;
        }
        match db.shader_errors(id).len() {
            0 => Self::Compiled,
            n => Self::Errors(n),
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::Compiling => "Compiling".to_owned(),
            Self::Compiled => "Compiled".to_owned(),
            Self::Errors(1) => "1 error".to_owned(),
            Self::Errors(n) => format!("{n} errors"),
        }
    }

    pub fn icon(self) -> (icons::Icon, egui::Color32) {
        match self {
            Self::Compiling => (icons::hourglass(), theme::PENDING),
            Self::Compiled => (icons::check_circle(), theme::OK),
            Self::Errors(_) => (icons::warning_circle(), theme::ERROR),
        }
    }
}

/// What a control asked for, done once the database is no longer borrowed.
enum Action {
    Open(AssetId),
    New,
    Save,
    Reload,
    /// Dismiss the notice that the file changed, keeping the typed text.
    KeepMine,
    /// Save the typed text, then open the shader waiting on it.
    SaveThenOpen,
    DiscardThenOpen,
    CancelOpen,
}

pub fn body(ui: &mut egui::Ui, world: &World, state: &mut EditorState) {
    if world.get_resource::<AssetDatabase>().is_none() {
        ui.weak("Surface shaders belong to a project's assets. Open a project to write one.");
        return;
    }
    let mut action = None;
    if let Some(id) = state.shader.request.take()
        && Some(id) != state.shader.open
    {
        action = Some(Action::Open(id));
    }
    // A material is given its slot before the database is borrowed to draw,
    // as the Import tab does: the next service binds it.
    let material = state
        .shader
        .open
        .and_then(|id| preview_material(world, id, state.shader.preview));

    {
        let db = world.resource::<AssetDatabase>();
        let editor = &mut state.shader;
        if let Some(id) = editor.open
            && db.iter().all(|entry| entry.id != id)
            && !editor.dirty
        {
            editor.open = None;
        }
        match editor.open {
            None => empty(ui, &db, &mut action),
            Some(id) => {
                editor.follow_disk(&db, id);
                let previewed = open(ui, &db, editor, id, material, &mut action);
                if previewed.is_some() {
                    state.material_previews[Viewer::Shader as usize] = previewed;
                }
            }
        }
    }
    if let Some(action) = action {
        apply(world, state, action);
    }
}

/// The project's materials that draw through shader `id`.
fn users(db: &AssetDatabase, id: AssetId) -> Vec<(AssetId, String)> {
    db.iter()
        .filter(|entry| entry.kind == AssetKind::Material)
        .filter(|entry| {
            db.material_file(entry.id)
                .is_some_and(|f| f.shader == Some(id))
        })
        .map(|entry| (entry.id, entry.name().to_owned()))
        .collect()
}

fn preview_material(world: &World, id: AssetId, on: PreviewOn) -> Option<MaterialHandle> {
    let mut db = world.get_resource_mut::<AssetDatabase>()?;
    let users = users(&db, id);
    let chosen = match on {
        PreviewOn::Material(material) if users.iter().any(|(m, _)| *m == material) => {
            Some(material)
        }
        PreviewOn::Defaults => None,
        _ => users.first().map(|(material, _)| *material),
    };
    match chosen {
        Some(material) => db.material(material).ok(),
        None => db.shader_preview_material(id).ok(),
    }
}

fn apply(world: &World, state: &mut EditorState, action: Action) {
    let editor = &mut state.shader;
    let open = |editor: &mut ShaderEditor, id: AssetId| {
        let db = world.resource::<AssetDatabase>();
        if let Err(error) = editor.load(&db, id) {
            drop(db);
            log_error(world, error);
        }
    };
    match action {
        Action::Open(id) if editor.dirty => editor.pending = Some(id),
        Action::Open(id) => open(editor, id),
        Action::New => {
            let created = world.resource_mut::<AssetDatabase>().create_shader("");
            match created {
                Ok(id) if editor.dirty => editor.pending = Some(id),
                Ok(id) => open(editor, id),
                Err(error) => log_error(world, error),
            }
        }
        Action::Save => {
            save(world, editor);
        }
        Action::Reload => {
            if let Some(id) = editor.open {
                open(editor, id);
            }
        }
        Action::KeepMine => editor.changed_on_disk = false,
        Action::SaveThenOpen => {
            if save(world, editor)
                && let Some(id) = editor.pending.take()
            {
                open(editor, id);
            }
        }
        Action::DiscardThenOpen => {
            if let Some(id) = editor.pending.take() {
                open(editor, id);
            }
        }
        Action::CancelOpen => editor.pending = None,
    }
}

/// Write the buffer through the database. `false` when it could not be.
fn save(world: &World, editor: &mut ShaderEditor) -> bool {
    let Some(id) = editor.open else {
        return false;
    };
    let written = world
        .resource_mut::<AssetDatabase>()
        .write_shader(id, &editor.text);
    match written {
        Ok(()) => {
            editor.saved.clone_from(&editor.text);
            editor.dirty = false;
            editor.changed_on_disk = false;
            editor.stamp = stamp(&world.resource::<AssetDatabase>(), id);
            true
        }
        Err(error) => {
            log_error(world, error);
            false
        }
    }
}

/// With nothing open: what a surface shader is, and the two ways to start.
fn empty(ui: &mut egui::Ui, db: &AssetDatabase, action: &mut Option<Action>) {
    ui.add_space(24.0);
    ui.vertical_centered(|ui| {
        ui.label(icons::text(
            icons::code(),
            20.0,
            ui.visuals().weak_text_color(),
        ));
        ui.add_space(4.0);
        ui.weak(
            "A surface shader is a short GLSL function that changes what a material's \
             surface is; the engine still lights and draws it.",
        );
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let buttons = 2.0 * 150.0;
            ui.add_space(((ui.available_width() - buttons) / 2.0).max(0.0));
            if ui
                .add(egui::Button::new(icons::with_label(
                    icons::file_plus(),
                    None,
                    "New surface shader",
                )))
                .clicked()
            {
                *action = Some(Action::New);
            }
            picker(ui, db, None, "Open a shader", action);
        });
    });
}

/// A button naming `current`, or saying `prompt`, that lists the project's
/// shaders to open.
fn picker(
    ui: &mut egui::Ui,
    db: &AssetDatabase,
    current: Option<AssetId>,
    prompt: &str,
    action: &mut Option<Action>,
) {
    let name = current
        .and_then(|id| db.get(id))
        .map_or_else(|| prompt.to_owned(), |entry| entry.name().to_owned());
    let response = widgets::tool_button(
        ui,
        icons::file_code(),
        None,
        Some(egui::RichText::new(name).strong()),
        Tool::Command,
        "Open another surface shader",
    );
    let popup = ui.make_persistent_id("shader_picker");
    if response.clicked() {
        ui.memory_mut(|memory| memory.toggle_popup(popup));
    }
    egui::popup_below_widget(
        ui,
        popup,
        &response,
        egui::PopupCloseBehavior::CloseOnClickOutside,
        |ui| {
            ui.set_width(240.0);
            let query_id = popup.with("query");
            let mut query: String = ui.data_mut(|data| data.get_temp(query_id).unwrap_or_default());
            widgets::search(ui, &mut query, "Filter shaders");
            let needle = query.trim().to_lowercase();
            ui.data_mut(|data| data.insert_temp(query_id, query));
            let shaders: Vec<_> = db
                .iter()
                .filter(|entry| entry.kind == AssetKind::Shader)
                .filter(|entry| needle.is_empty() || entry.path.to_lowercase().contains(&needle))
                .collect();
            if shaders.is_empty() {
                ui.weak("The project has no surface shaders.");
            }
            egui::ScrollArea::vertical()
                .max_height(280.0)
                .show(ui, |ui| {
                    for entry in shaders {
                        let selected = Some(entry.id) == current;
                        let button = egui::Button::new(entry.name())
                            .selected(selected)
                            .frame(selected)
                            .min_size(egui::vec2(ui.available_width(), 0.0));
                        if ui.add(button).on_hover_text(&entry.path).clicked() {
                            *action = Some(Action::Open(entry.id));
                            ui.memory_mut(|memory| memory.close_popup());
                        }
                    }
                });
            ui.separator();
            if ui
                .add(egui::Button::new(icons::with_label(
                    icons::file_plus(),
                    None,
                    "New surface shader",
                )))
                .clicked()
            {
                *action = Some(Action::New);
                ui.memory_mut(|memory| memory.close_popup());
            }
        },
    );
}

/// The tab with shader `id` open. Returns what its sphere asks of the renderer.
fn open(
    ui: &mut egui::Ui,
    db: &AssetDatabase,
    editor: &mut ShaderEditor,
    id: AssetId,
    material: Option<MaterialHandle>,
    action: &mut Option<Action>,
) -> Option<crate::gfx::vulkan::material_preview::Subject> {
    let status = Status::of(db, id);
    let errors = db.shader_errors(id);
    if ui.memory(|memory| memory.has_focus(code_id()))
        && ui.input_mut(|input| input.consume_key(egui::Modifiers::COMMAND, egui::Key::S))
        && !editor.changed_on_disk
    {
        *action = Some(Action::Save);
    }

    header(ui, db, editor, id, status, action);
    notices(ui, db, editor, action);
    ui.add_space(4.0);

    let full = ui.available_rect_before_wrap();
    let wide = full.width() >= WIDE;
    let gap = 8.0;
    let (main, side) = if wide {
        let split = full.right() - SIDE;
        (
            egui::Rect::from_min_max(full.min, egui::pos2(split - gap, full.bottom())),
            egui::Rect::from_min_max(egui::pos2(split, full.top()), full.max),
        )
    } else {
        let split = full.top() + (full.height() * 0.6).max(160.0).min(full.height());
        (
            egui::Rect::from_min_max(full.min, egui::pos2(full.right(), split)),
            egui::Rect::from_min_max(egui::pos2(full.left(), split + gap), full.max),
        )
    };
    let syntax = Syntax::of(ui.visuals());
    ui.scope_builder(egui::UiBuilder::new().max_rect(main), |ui| {
        let listed = if errors.is_empty() {
            0.0
        } else {
            let wanted = 24.0 + errors.len() as f32 * ERROR_ROW;
            wanted.min(main.height() * 0.4)
        };
        let code_rect =
            egui::Rect::from_min_max(main.min, egui::pos2(main.right(), main.bottom() - listed));
        ui.scope_builder(egui::UiBuilder::new().max_rect(code_rect), |ui| {
            code(ui, editor, errors, syntax);
        });
        if !errors.is_empty() {
            let list = egui::Rect::from_min_max(
                egui::pos2(main.left(), main.bottom() - listed + 4.0),
                main.max,
            );
            ui.scope_builder(egui::UiBuilder::new().max_rect(list), |ui| {
                error_list(ui, editor, errors);
            });
        }
    });
    let mut previewed = None;
    if side.height() > 0.0 {
        ui.scope_builder(egui::UiBuilder::new().max_rect(side), |ui| {
            egui::ScrollArea::vertical()
                .id_salt("shader_side")
                .auto_shrink(false)
                .show(ui, |ui| {
                    previewed = preview(ui, db, editor, id, material);
                    parameters(ui, db, id);
                });
        });
    }
    ui.advance_cursor_after_rect(full);
    previewed
}

/// Which shader, whether it is saved and how its last save went; Save and
/// Revert at the far end.
fn header(
    ui: &mut egui::Ui,
    db: &AssetDatabase,
    editor: &ShaderEditor,
    id: AssetId,
    status: Status,
    action: &mut Option<Action>,
) {
    ui.horizontal(|ui| {
        picker(ui, db, Some(id), "Open a shader", action);
        if editor.dirty {
            let (rect, response) =
                ui.allocate_exact_size(egui::Vec2::splat(10.0), egui::Sense::hover());
            ui.painter()
                .circle_filled(rect.center(), 3.5, ui.visuals().text_color());
            response.on_hover_text("Unsaved changes");
        }
        ui.add_space(6.0);
        let (icon, colour) = status.icon();
        let shown = ui
            .horizontal(|ui| icons::labelled(ui, icon, colour, &status.label()))
            .response;
        if let Status::Errors(_) = status {
            shown.on_hover_text(
                "Errors in the last save; the last version that compiled is still drawn",
            );
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (save, hint) = match (editor.dirty, editor.changed_on_disk) {
                (_, true) => (
                    Tool::Disabled,
                    "The file changed on disk. Reload it, or keep your text, before saving",
                ),
                (true, false) => (Tool::Command, "Save and compile (Cmd/Ctrl+S in the code)"),
                (false, false) => (Tool::Disabled, "Nothing to save"),
            };
            let text = Some(egui::RichText::new("Save"));
            if widgets::tool_button(ui, icons::floppy_disk(), None, text, save, hint).clicked() {
                *action = Some(Action::Save);
            }
            let revert = if editor.dirty {
                Tool::Command
            } else {
                Tool::Disabled
            };
            let text = Some(egui::RichText::new("Revert"));
            let hint = "Throw away the unsaved text and read the file again";
            if widgets::tool_button(ui, icons::rotate_ccw(), None, text, revert, hint).clicked() {
                *action = Some(Action::Reload);
            }
        });
    });
}

/// What is waiting on the user: the file changing under typed text, or
/// another shader asked for over it.
fn notices(
    ui: &mut egui::Ui,
    db: &AssetDatabase,
    editor: &ShaderEditor,
    action: &mut Option<Action>,
) {
    if editor.changed_on_disk {
        ui.horizontal_wrapped(|ui| {
            icons::labelled(
                ui,
                icons::warning(),
                theme::PENDING,
                "The file changed on disk since you started typing.",
            );
            if ui
                .button("Reload")
                .on_hover_text("Take the file's text")
                .clicked()
            {
                *action = Some(Action::Reload);
            }
            if ui
                .button("Keep mine")
                .on_hover_text("Keep your text; saving writes it over the file")
                .clicked()
            {
                *action = Some(Action::KeepMine);
            }
        });
    }
    if let Some(pending) = editor.pending {
        let name = db
            .get(pending)
            .map_or("another shader", |entry| entry.name());
        ui.horizontal_wrapped(|ui| {
            icons::labelled(
                ui,
                icons::warning(),
                theme::PENDING,
                &format!("Opening {name} would lose your unsaved text."),
            );
            if ui.button("Save first").clicked() {
                *action = Some(Action::SaveThenOpen);
            }
            if ui.button("Discard").clicked() {
                *action = Some(Action::DiscardThenOpen);
            }
            if ui.button("Cancel").clicked() {
                *action = Some(Action::CancelOpen);
            }
        });
    }
}

/// The text edit, in a sunken well with a gutter of line numbers that marks
/// the lines the last save's errors are on.
fn code(ui: &mut egui::Ui, editor: &mut ShaderEditor, errors: &[SurfaceError], syntax: Syntax) {
    let visuals = ui.visuals().clone();
    let font = egui::TextStyle::Monospace.resolve(ui.style());
    let digits = editor.text.lines().count().max(1).ilog10() as f32 + 1.0;
    let digit = ui.fonts(|fonts| fonts.glyph_width(&font, '0'));
    let gutter = (digits.max(2.0) * digit + 18.0).round();
    egui::Frame::new()
        .fill(visuals.extreme_bg_color)
        .stroke(egui::Stroke::new(
            1.0_f32,
            visuals.widgets.noninteractive.bg_stroke.color,
        ))
        .corner_radius(visuals.widgets.inactive.corner_radius)
        .inner_margin(egui::Margin::symmetric(0, 4))
        .show(ui, |ui| {
            ui.set_min_size(ui.available_size());
            egui::ScrollArea::both()
                .id_salt("shader_code_scroll")
                .auto_shrink(false)
                .show(ui, |ui| {
                    ui.horizontal_top(|ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        let left = ui.cursor().left();
                        ui.add_space(gutter);
                        let bands = ui.painter().add(egui::Shape::Noop);
                        let row_height = ui.fonts(|fonts| fonts.row_height(&font));
                        let rows = (ui.available_height() / row_height).floor().max(1.0) as usize;
                        let highlighter = &mut editor.highlighter;
                        let mut layouter = |ui: &egui::Ui, text: &str, _wrap: f32| {
                            highlighter.layout(ui, text, syntax)
                        };
                        let output = egui::TextEdit::multiline(&mut editor.text)
                            .id(code_id())
                            .code_editor()
                            .frame(false)
                            .margin(egui::Margin::symmetric(6, 0))
                            .desired_width(ui.available_width())
                            .desired_rows(rows)
                            .layouter(&mut layouter)
                            .show(ui);
                        if output.response.changed() {
                            editor.dirty = editor.text != editor.saved;
                        }
                        gutter_and_bands(
                            ui,
                            &output,
                            left,
                            gutter,
                            bands,
                            errors,
                            editor.dirty,
                            &font,
                        );
                        if let Some(line) = editor.goto.take() {
                            go_to_line(ui, &output, &editor.text, line);
                        }
                    });
                });
        });
}

#[allow(clippy::too_many_arguments)]
fn gutter_and_bands(
    ui: &egui::Ui,
    output: &egui::text_edit::TextEditOutput,
    left: f32,
    gutter: f32,
    bands: egui::layers::ShapeIdx,
    errors: &[SurfaceError],
    stale: bool,
    font: &egui::FontId,
) {
    let weak = ui.visuals().weak_text_color();
    // A mark from text that has changed since may not point at its line any
    // more, so it fades rather than pretending to.
    let error = if stale {
        theme::ERROR.gamma_multiply(0.45)
    } else {
        theme::ERROR
    };
    let clip = ui.clip_rect();
    let width = output.response.rect.width();
    let mut fills = Vec::new();
    for (index, row) in output.galley.rows.iter().enumerate() {
        let top = output.galley_pos.y + row.rect.min.y;
        let rect =
            egui::Rect::from_min_size(egui::pos2(left, top), egui::vec2(gutter, row.rect.height()));
        if !clip.intersects(rect) {
            continue;
        }
        let line = index as u32 + 1;
        let here: Vec<&str> = errors
            .iter()
            .filter(|e| e.line == Some(line))
            .map(|e| e.message.as_str())
            .collect();
        let colour = if here.is_empty() { weak } else { error };
        ui.painter().text(
            egui::pos2(rect.right() - 8.0, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            line.to_string(),
            font.clone(),
            colour,
        );
        if here.is_empty() {
            continue;
        }
        ui.painter()
            .circle_filled(egui::pos2(rect.left() + 5.0, rect.center().y), 2.5, error);
        fills.push(egui::Shape::rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(rect.right(), top),
                egui::vec2(width, row.rect.height()),
            ),
            0.0,
            error.gamma_multiply(0.12),
        ));
        let mut hint = here.join("\n");
        if stale {
            hint.push_str("\n(from the last save)");
        }
        ui.interact(rect, code_id().with(("gutter", line)), egui::Sense::hover())
            .on_hover_text(hint);
    }
    ui.painter().set(bands, egui::Shape::Vec(fills));
}

/// Put the cursor at the start of `line` and bring it into view.
fn go_to_line(ui: &egui::Ui, output: &egui::text_edit::TextEditOutput, text: &str, line: u32) {
    let index: usize = text
        .split_inclusive('\n')
        .take(line.saturating_sub(1) as usize)
        .map(|line| line.chars().count())
        .sum();
    let mut state = output.state.clone();
    let at = egui::text::CCursor::new(index);
    state
        .cursor
        .set_char_range(Some(egui::text::CCursorRange::one(at)));
    state.store(ui.ctx(), code_id());
    ui.memory_mut(|memory| memory.request_focus(code_id()));
    if let Some(row) = output.galley.rows.get(line.saturating_sub(1) as usize) {
        let rect = row.rect.translate(output.galley_pos.to_vec2());
        ui.scroll_to_rect(rect, Some(egui::Align::Center));
    }
}

/// The last save's errors, each a link to its line.
fn error_list(ui: &mut egui::Ui, editor: &mut ShaderEditor, errors: &[SurfaceError]) {
    let note = if editor.dirty {
        "From the last save; the text has changed since"
    } else {
        "In the last save; the last version that compiled is still drawn"
    };
    ui.label(egui::RichText::new(note).weak().small());
    egui::ScrollArea::vertical()
        .id_salt("shader_errors")
        .auto_shrink(false)
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            for error in errors {
                let mut job = egui::text::LayoutJob::default();
                let font = egui::TextStyle::Monospace.resolve(ui.style());
                let place = error
                    .line
                    .map_or_else(String::new, |line| format!("{line}: "));
                job.append(&place, 0.0, egui::TextFormat::simple(font, theme::ERROR));
                job.append(
                    &error.message,
                    0.0,
                    egui::TextFormat::simple(
                        egui::TextStyle::Body.resolve(ui.style()),
                        ui.visuals().text_color(),
                    ),
                );
                let button = egui::Button::new(job)
                    .frame(false)
                    .wrap_mode(egui::TextWrapMode::Truncate)
                    .min_size(egui::vec2(ui.available_width(), ERROR_ROW));
                let response = ui.add(button).on_hover_text(&error.message);
                if response.clicked()
                    && let Some(line) = error.line
                {
                    editor.goto = Some(line);
                }
            }
        });
}

/// The sphere, and which material it is drawn as.
fn preview(
    ui: &mut egui::Ui,
    db: &AssetDatabase,
    editor: &mut ShaderEditor,
    id: AssetId,
    material: Option<MaterialHandle>,
) -> Option<crate::gfx::vulkan::material_preview::Subject> {
    let weak = ui.visuals().weak_text_color();
    let mut subject = None;
    widgets::section(
        ui,
        ui.id().with("shader_preview"),
        icons::palette(),
        weak,
        "Preview",
        None,
        |ui| {
            let side = ui.available_width().clamp(96.0, 160.0);
            if let Some(material) = material {
                ui.vertical_centered(|ui| {
                    subject = Some(widgets::material_preview(
                        ui,
                        Viewer::Shader,
                        material,
                        side,
                    ));
                });
            }
            ui.add_space(6.0);
            let users = users(db, id);
            let defaults = "The shader's defaults";
            let shown = match editor.preview {
                PreviewOn::Material(chosen) => users.iter().find(|(m, _)| *m == chosen),
                PreviewOn::Defaults => None,
                PreviewOn::First => users.first(),
            }
            .map_or(defaults, |(_, name)| name.as_str());
            widgets::property(ui, "Material", |ui| {
                egui::ComboBox::from_id_salt("shader_preview_material")
                    .selected_text(shown)
                    .width(ui.available_width())
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(shown == defaults, defaults).clicked() {
                            editor.preview = PreviewOn::Defaults;
                        }
                        for (material, name) in &users {
                            if ui.selectable_label(shown == name, name).clicked() {
                                editor.preview = PreviewOn::Material(*material);
                            }
                        }
                    })
                    .response
                    .on_hover_text("Preview on a material that uses this shader")
            });
        },
    );
    subject
}

/// What the last good compile declared, read-only: values are a material's.
fn parameters(ui: &mut egui::Ui, db: &AssetDatabase, id: AssetId) {
    let weak = ui.visuals().weak_text_color();
    widgets::section(
        ui,
        ui.id().with("shader_parameters"),
        icons::sliders(),
        weak,
        "Parameters",
        None,
        |ui| {
            let Some(table) = db.shader_params(id) else {
                widgets::padded(ui, |ui| {
                    ui.weak("They are listed once the shader has compiled.")
                });
                return;
            };
            if table.params.is_empty() {
                widgets::padded(ui, |ui| ui.weak("This shader declares none."));
            }
            for param in &table.params {
                widgets::property(ui, &param.name, |ui| {
                    if let ParamKind::Color { default } = param.kind {
                        let colour = egui::Rgba::from_rgba_unmultiplied(
                            default[0], default[1], default[2], default[3],
                        );
                        egui::color_picker::show_color(ui, colour, egui::vec2(12.0, 12.0));
                    }
                    ui.label(figures(describe(&param.kind)))
                        .on_hover_text("Set per material, in the inspector's Material section");
                });
            }
        },
    );
}

/// A parameter's type, default and hints, as a line.
pub(super) fn describe(kind: &ParamKind) -> String {
    let list = |values: &[f32]| {
        let values: Vec<String> = values.iter().map(f32::to_string).collect();
        format!("({})", values.join(", "))
    };
    let default = match kind {
        ParamKind::Float { default, .. } => default.to_string(),
        ParamKind::Vec2 { default } => list(default),
        ParamKind::Vec3 { default } => list(default),
        ParamKind::Vec4 { default } | ParamKind::Color { default } => list(default),
        ParamKind::Texture { default, .. } => match default {
            DefaultTexture::White => "white",
            DefaultTexture::Black => "black",
            DefaultTexture::Normal => "normal",
        }
        .to_owned(),
    };
    let hint = match kind {
        ParamKind::Float {
            range: Some([min, max]),
            ..
        } => format!(", {min} to {max}"),
        ParamKind::Texture {
            color_space: ColorSpace::Srgb,
            ..
        } => ", sRGB".to_owned(),
        ParamKind::Texture { .. } => ", linear".to_owned(),
        _ => String::new(),
    };
    format!("{} = {default}{hint}", kind.type_name())
}
