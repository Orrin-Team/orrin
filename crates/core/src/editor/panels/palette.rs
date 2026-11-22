//! The command palette: every command in `editor::commands`, and every tool
//! the View menu opens, one search away from anywhere in the editor.
//!
//! **One way to run a command.** [`run`] is what a command's key does, from
//! `panels::shortcuts`, and what Enter or a click does here: each posts the
//! request its menu entry posts. [`blocked`] is why one cannot run now, in its
//! menu entry's words where the entry gives a reason.
//!
//! **Listed, not hidden.** A command that cannot run is drawn faded, and
//! while it is highlighted the line under the list says why: the palette is
//! driven from the keyboard, which has no hover to ask with. A list that
//! hides what is unavailable teaches nothing about what the editor can do.
//!
//! **A modal**, so the rest of the editor is out of the pointer's reach and a
//! click on it, or Escape, closes the list. Its field holds the keyboard
//! every frame it is open, which keeps every unmodified key typing into it;
//! `panels::shortcuts` answers no key but the palette's own while it is open.

use std::borrow::Cow;

use orrin_ecs::World;
use orrin_registry::Registry;

use super::inspector::Filter;
use super::menu_bar;
use super::settings::Page;
use crate::editor::commands::{self, COMMANDS, Command};
use crate::editor::dock::{Dock, Tab};
use crate::editor::state::{DockRequest, EditorState, GizmoMode, SceneRequest};
use crate::editor::widgets;
use crate::scene::{Play, PlayState};

const WIDTH: f32 = 440.0;
const ROW: f32 = 24.0;
/// How far below the top of the window the list hangs. Hung from the top
/// rather than centred, so it does not jump as the filter shortens it.
const DROP: f32 = 72.0;
const LIST_HEIGHT: f32 = 340.0;

/// The palette's own state, kept on `EditorState` across frames.
#[derive(Default)]
pub struct Palette {
    open: bool,
    query: String,
    highlight: usize,
}

impl Palette {
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Open with nothing typed, or close.
    pub fn toggle(&mut self) {
        *self = Self {
            open: !self.open,
            ..Self::default()
        };
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Entry {
    Command(Command),
    Tool(Tab),
}

impl Entry {
    fn label(self) -> Cow<'static, str> {
        match self {
            Self::Command(command) => command.label().into(),
            Self::Tool(tab) => format!("Show {}", tab.title()).into(),
        }
    }

    fn group(self) -> &'static str {
        match self {
            Self::Command(command) => command.group().title(),
            Self::Tool(_) => "View",
        }
    }
}

/// What the File menu says while the game plays.
const PLAYING: &str = "Stop the game first: the scene is not what is playing";

/// Why `command` cannot run now, or `None` when it can.
pub(super) fn blocked(
    ctx: &egui::Context,
    world: &World,
    state: &EditorState,
    command: Command,
) -> Option<&'static str> {
    use Command as C;
    let playing = state.playing();
    let selected = state
        .selection
        .primary()
        .is_some_and(|entity| world.is_alive(entity));
    let live = world
        .get_resource::<Play>()
        .is_some_and(|play| play.state() != PlayState::Editing);
    match command {
        C::Save | C::Reload | C::SaveAs | C::Open | C::Recover if playing => Some(PLAYING),
        C::SaveAs | C::Open if menu_bar::dialog_open(ctx) => Some("A scene file dialog is open"),
        C::Recover if !state.autosave.path(&state.scene_path).is_file() => {
            Some("There is no unsaved work to recover for this scene")
        }
        C::ProjectSettings if state.project_root.is_none() => {
            Some("Project settings need a project: open one with `orrin run` or Orrin Hub")
        }
        C::Undo if state.history.undo_label().is_none() => Some("Nothing to undo"),
        C::Redo if state.history.redo_label().is_none() => Some("Nothing to redo"),
        C::Duplicate
        | C::Copy
        | C::Rename
        | C::Delete
        | C::SavePrefab
        | C::SelectNone
        | C::Frame
        | C::Dump
            if !selected =>
        {
            Some("Nothing is selected")
        }
        C::Paste if !state.can_paste() => Some("Nothing has been copied"),
        C::Pause | C::Step if !live => Some("Press Play first"),
        C::ReloadScripts if !cfg!(feature = "scripting") => {
            Some("This build was compiled without the `scripting` feature")
        }
        _ => None,
    }
}

/// Do what `command`'s key and menu entry do.
pub(super) fn run(
    ctx: &egui::Context,
    world: &mut World,
    registry: &Registry,
    state: &mut EditorState,
    dock: &mut Dock,
    command: Command,
) {
    use Command as C;
    match command {
        C::Save => state.request_scene(SceneRequest::Save),
        C::SaveAs => menu_bar::start_dialog(ctx, state, menu_bar::Pick::SaveAs),
        C::Open => menu_bar::start_dialog(ctx, state, menu_bar::Pick::Open),
        C::Reload => state.request_reload_checked(),
        C::Recover => state.request_scene(SceneRequest::Recover),
        C::ProjectSettings => state.open_settings(Some(Page::General)),
        C::Undo => state.request_undo(),
        C::Redo => state.request_redo(),
        C::Duplicate => state.request_duplicate(),
        C::Copy => {
            state.request_copy();
            // The system clipboard gets the names too: the window sends a
            // paste only while it holds text.
            let names: Vec<String> = state
                .selection
                .entities()
                .iter()
                .filter_map(|&entity| world.get::<crate::scene::Name>(entity).map(|n| n.0.clone()))
                .collect();
            if !names.is_empty() {
                ctx.copy_text(names.join("\n"));
            }
        }
        C::Paste => state.request_paste(),
        C::Rename => state.request_rename(),
        C::Delete => state.request_despawn_selection(),
        C::SavePrefab => state.request_save_prefab(),
        C::SelectAll => super::hierarchy::select_all(ctx, world, state),
        C::SelectNone => state.selection.clear(),
        C::Frame => state.request_frame(),
        C::Dump => {
            if let Some(entity) = state.selection.primary() {
                super::inspector::dump(world, registry, entity);
            }
        }
        C::Settings => state.open_settings(None),
        C::Play => state.request_play(),
        C::Pause => state.request_pause(),
        C::Step => state.request_step(),
        C::SelectTool => state.gizmo_mode = GizmoMode::Select,
        C::MoveTool => state.gizmo_mode = GizmoMode::Move,
        C::RotateTool => state.gizmo_mode = GizmoMode::Rotate,
        C::ScaleTool => state.gizmo_mode = GizmoMode::Scale,
        C::UiMode => state.ui_mode = !state.ui_mode,
        C::Palette => state.palette.toggle(),
        C::ResetLayout => dock.reset(),
        C::ReloadScripts => {
            #[cfg(feature = "scripting")]
            state.request_script_reload();
        }
    }
}

/// The open palette, over everything: drawn after the rest of the editor, so
/// it is the modal in front.
pub(super) fn show(
    ctx: &egui::Context,
    world: &mut World,
    registry: &Registry,
    state: &mut EditorState,
    dock: &mut Dock,
) {
    if !state.palette.open {
        return;
    }
    // Taken before the field is drawn, which would otherwise move its cursor
    // on the arrows and give up the keyboard on Enter.
    let (up, down, enter) = ctx.input_mut(|input| {
        let none = egui::Modifiers::NONE;
        (
            input.consume_key(none, egui::Key::ArrowUp),
            input.consume_key(none, egui::Key::ArrowDown),
            input.consume_key(none, egui::Key::Enter),
        )
    });
    let moved = up || down;
    let mut query = std::mem::take(&mut state.palette.query);
    let mut highlight = state.palette.highlight;
    let mut chosen = None;
    let view: &EditorState = state;
    let keys = &view.settings.keys;
    let area = egui::Modal::default_area(egui::Id::new("command_palette"))
        .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, DROP));
    let modal = egui::Modal::new(egui::Id::new("command_palette"))
        .area(area)
        .show(ctx, |ui| {
            ui.set_width(WIDTH);
            let field = widgets::search(ui, &mut query, "Run a command or show a tool");
            field.request_focus();
            // After the field, so what was typed this frame is what filters.
            let entries = entries(ctx, world, view, &query);
            if field.changed() {
                highlight = 0;
            }
            if down {
                highlight += 1;
            }
            if up {
                highlight = highlight.saturating_sub(1);
            }
            highlight = highlight.min(entries.len().saturating_sub(1));
            if enter {
                chosen = Some(highlight);
            }
            ui.add_space(4.0);
            egui::ScrollArea::vertical()
                .max_height(LIST_HEIGHT)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for (at, (entry, why)) in entries.iter().enumerate() {
                        let shortcut = match entry {
                            Entry::Command(command) => commands::text(ui.ctx(), keys, *command),
                            Entry::Tool(_) => String::new(),
                        };
                        let response = row(ui, *entry, &shortcut, why.is_some(), at == highlight);
                        if moved && at == highlight {
                            response.scroll_to_me(None);
                        }
                        if response.clicked() {
                            highlight = at;
                            chosen = Some(at);
                        }
                    }
                });
            let weak = ui.visuals().weak_text_color();
            match entries.get(highlight) {
                None => {
                    ui.label(egui::RichText::new("No command matches").color(weak));
                }
                Some((_, Some(why))) => {
                    ui.add(egui::Label::new(egui::RichText::new(*why).small().color(weak)).wrap());
                }
                Some(_) => {}
            }
            (field, entries)
        });
    let close = modal.should_close();
    let (field, entries) = modal.inner;
    let palette = &mut state.palette;
    palette.query = query;
    palette.highlight = highlight;

    let run_now = chosen
        .and_then(|at| entries.get(at))
        .filter(|(_, why)| why.is_none());
    if close || run_now.is_some() {
        // Not left holding the keyboard for a field that is gone, which
        // would make the next Escape the field's.
        field.surrender_focus();
        palette.toggle();
    }
    match run_now {
        Some((Entry::Command(command), _)) => run(ctx, world, registry, state, dock, *command),
        Some((Entry::Tool(tab), _)) => state.request_dock(DockRequest::Show(*tab)),
        None => {}
    }
}

/// Every entry whose label or group holds `query`, each with why it cannot
/// run now. Nothing typed, they are in the table's order and then the dock's;
/// otherwise ranked by [`rank`], then what can run before what cannot, then
/// that order — so "console" leads with Show Console rather than with Edit's
/// Dump to console, which is earlier in the table and cannot run with
/// nothing selected.
fn entries(
    ctx: &egui::Context,
    world: &World,
    state: &EditorState,
    query: &str,
) -> Vec<(Entry, Option<&'static str>)> {
    let filter = Filter::new(query);
    let mut entries: Vec<_> = COMMANDS
        .iter()
        .map(|spec| Entry::Command(spec.command))
        .filter(|&entry| entry != Entry::Command(Command::Palette))
        .chain(Tab::ALL.into_iter().map(Entry::Tool))
        .filter(|entry| filter.hits(&entry.label()) || filter.hits(entry.group()))
        .map(|entry| {
            let why = match entry {
                Entry::Command(command) => blocked(ctx, world, state, command),
                Entry::Tool(_) => None,
            };
            (entry, why)
        })
        .collect();
    let needle = query.trim().to_lowercase();
    if !needle.is_empty() {
        // Stable, so the table's order holds inside a rank.
        entries.sort_by_cached_key(|(entry, why)| (rank(*entry, &needle), why.is_some()));
    }
    entries
}

/// How well `entry` answers `needle`, lowercased: 0 when its label — or a
/// tool's own name, past "Show" — starts with it, 1 when a word of its label
/// does, 2 for anything else the filter let through.
fn rank(entry: Entry, needle: &str) -> u8 {
    let label = entry.label().to_lowercase();
    let starts = label.starts_with(needle)
        || matches!(entry, Entry::Tool(tab) if tab.title().to_lowercase().starts_with(needle));
    let word = label.match_indices(needle).any(|(at, _)| {
        label[..at]
            .chars()
            .next_back()
            .is_none_or(|before| !before.is_alphanumeric())
    });
    match (starts, word) {
        (true, _) => 0,
        (false, true) => 1,
        (false, false) => 2,
    }
}

/// One entry: its label, its group, and its chord at the right; faded when
/// it cannot run, filled while it is the one Enter would run.
fn row(
    ui: &mut egui::Ui,
    entry: Entry,
    shortcut: &str,
    blocked: bool,
    highlit: bool,
) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW), egui::Sense::click());
    let visuals = ui.visuals();
    let fill = if highlit {
        Some(visuals.selection.bg_fill)
    } else if response.hovered() {
        Some(visuals.widgets.hovered.weak_bg_fill)
    } else {
        None
    };
    if let Some(fill) = fill {
        ui.painter()
            .rect_filled(rect, visuals.widgets.inactive.corner_radius, fill);
    }
    let weak = visuals.weak_text_color();
    let text = if blocked {
        weak
    } else if highlit {
        visuals.strong_text_color()
    } else {
        visuals.text_color()
    };
    let body = egui::TextStyle::Body.resolve(ui.style());
    let label = ui.painter().text(
        rect.left_center() + egui::vec2(8.0, 0.0),
        egui::Align2::LEFT_CENTER,
        entry.label(),
        body.clone(),
        text,
    );
    ui.painter().text(
        egui::pos2(label.right() + 8.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        entry.group(),
        egui::TextStyle::Small.resolve(ui.style()),
        weak,
    );
    if !shortcut.is_empty() {
        ui.painter().text(
            rect.right_center() - egui::vec2(8.0, 0.0),
            egui::Align2::RIGHT_CENTER,
            shortcut,
            egui::TextStyle::Monospace.resolve(ui.style()),
            weak,
        );
    }
    response
}
