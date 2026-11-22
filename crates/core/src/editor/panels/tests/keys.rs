//! The keys: every command's chord read from one table, changed on
//! Settings ▸ Keys, and every command run by name from the palette.

use egui::{Key, KeyboardShortcut, Modifiers};

use super::*;
use crate::editor::commands::{Chord, Command};
use crate::editor::panels::settings::Page;

const CMD: Modifiers = Modifiers::COMMAND;

fn chord(modifiers: Modifiers, key: Key) -> Chord {
    KeyboardShortcut::new(modifiers, key)
}

/// What egui calls `chord` on this platform: Ctrl+D here, ⌘D on a Mac.
fn shown(editor: &Harness, chord: Chord) -> String {
    editor.ctx.format_shortcut(&chord)
}

fn type_text(editor: &mut Harness, text: &str) {
    editor.input.events.push(egui::Event::Text(text.to_owned()));
    editor.frames(1);
    editor.input.events.clear();
}

fn palette_open(editor: &Harness) -> bool {
    let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("command_palette"));
    editor
        .ctx
        .memory(|memory| memory.areas().visible_last_frame(&layer))
}

fn open_palette(editor: &mut Harness) {
    editor.key_with(CMD, Key::K).frames(1);
    assert!(palette_open(editor), "Ctrl+K opens it");
}

fn duplicates_of(editor: &Harness, name: &str) -> usize {
    editor
        .world
        .entities()
        .filter(|&e| editor.world.get::<Name>(e).is_some_and(|n| n.0 == name))
        .count()
}

/// Every place `needle` was painted inside the Settings tab.
fn in_settings(editor: &Harness, needle: &str) -> Vec<egui::Pos2> {
    let tab = editor.tab(Tab::Settings);
    editor
        .painted_text_rects()
        .into_iter()
        .filter(|(text, rect)| text == needle && tab.contains(rect.center()))
        .map(|(_, rect)| rect.center())
        .collect()
}

fn keys_page(editor: &mut Harness) {
    editor.frames(2);
    editor.state.open_settings(Some(Page::Keys));
    editor.frames(3);
}

/// Where the row labelled `label` shows `chord`.
fn chord_on_row(editor: &Harness, label: &str, chord: Chord) -> egui::Pos2 {
    let row = *in_settings(editor, label)
        .first()
        .unwrap_or_else(|| panic!("{label} is listed: {:?}", editor.painted()));
    let text = shown(editor, chord);
    *in_settings(editor, &text)
        .iter()
        .find(|at| (at.y - row.y).abs() < 8.0 && at.x > row.x)
        .unwrap_or_else(|| panic!("{label} shows {text}: {:?}", editor.painted()))
}

fn reset_glyphs(editor: &Harness) -> Vec<egui::Rect> {
    let tab = editor.tab(Tab::Settings);
    let reset = crate::editor::icons::rotate_ccw().glyph();
    editor
        .painted_icons()
        .into_iter()
        .filter(|(glyph, rect, _)| *glyph == reset && tab.contains(rect.center()))
        .map(|(_, rect, _)| rect)
        .collect()
}

#[test]
fn a_rebound_command_runs_on_its_new_chord_and_no_longer_on_its_old() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let lamp = editor.spawn("Lamp");
    editor.state.selection.set(lamp);
    editor.frames(1);
    editor
        .state
        .settings
        .keys
        .bind(Command::Duplicate, Some(0), chord(Modifiers::NONE, Key::F6))
        .unwrap();

    editor.key_with(CMD, Key::D);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert_eq!(duplicates_of(&editor, "Lamp"), 1, "Ctrl+D runs nothing now");

    editor.key(Key::F6);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert_eq!(duplicates_of(&editor, "Lamp"), 2);

    // Undo given a second chord keeps its first.
    editor
        .state
        .settings
        .keys
        .bind(Command::Undo, None, chord(Modifiers::NONE, Key::F7))
        .unwrap();
    editor.key(Key::F7);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert_eq!(duplicates_of(&editor, "Lamp"), 1, "F7 undid it");
    editor.key_with(CMD | Modifiers::SHIFT, Key::Z);
    editor.state.apply(&mut editor.world, &editor.registry);
    editor.key_with(CMD, Key::Z);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert_eq!(duplicates_of(&editor, "Lamp"), 1, "and so does Ctrl+Z");
}

/// The keys every editor had before there was a table still do what they
/// did. Most have a test of their own; these did not.
#[test]
fn the_default_chords_still_do_what_they_did() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let a = editor.spawn("A");
    let b = editor.spawn("B");
    editor.state.selection.set(a);
    editor.frames(2);

    editor.key_with(CMD, Key::A);
    assert!(editor.state.selection.contains(a) && editor.state.selection.contains(b));
    editor.key(Key::Escape).frames(1);
    assert!(editor.state.selection.is_empty());

    editor.state.selection.set(b);
    editor.key_with(CMD, Key::C);
    editor.state.apply(&mut editor.world, &editor.registry);
    editor.key_with(CMD, Key::V);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert_eq!(duplicates_of(&editor, "B"), 2, "pasted");

    editor.key_with(CMD, Key::P);
    editor.state.apply(&mut editor.world, &editor.registry);
    editor.key_with(CMD | Modifiers::SHIFT, Key::P);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert_eq!(
        editor.world.resource::<crate::scene::Play>().state(),
        crate::scene::PlayState::Paused,
        "Ctrl+Shift+P pauses rather than stopping"
    );
    editor.key_with(CMD | Modifiers::ALT, Key::P);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert_eq!(
        editor.world.resource::<crate::scene::Play>().state(),
        crate::scene::PlayState::Paused,
        "Ctrl+Alt+P steps and stays paused"
    );
}

/// Clicking a chord waits for the next one pressed, which takes its place;
/// the row is then marked changed, and its reset puts the default back.
#[test]
fn clicking_a_chord_captures_the_next_one_and_reset_puts_the_default_back() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    keys_page(&mut editor);
    assert!(reset_glyphs(&editor).is_empty(), "nothing is rebound");

    let at = chord_on_row(&editor, "Duplicate", chord(CMD, Key::D));
    editor.click(at).frames(1);
    assert!(
        editor.painted_contains("Press a key"),
        "{:?}",
        editor.painted()
    );
    let shifted = chord(CMD | Modifiers::SHIFT, Key::D);
    editor.key_with(shifted.modifiers, Key::D).frames(1);
    assert_eq!(
        editor.state.settings.keys.chords(Command::Duplicate),
        [shifted]
    );
    assert!(!editor.painted_contains("Press a key"));
    chord_on_row(&editor, "Duplicate", shifted);
    assert_eq!(editor.state.selection.len(), 0);

    let resets = reset_glyphs(&editor);
    assert_eq!(resets.len(), 1, "{resets:?}");
    editor.click(resets[0].center()).frames(2);
    assert!(editor.state.settings.keys.is_default(Command::Duplicate));
    assert!(reset_glyphs(&editor).is_empty());
}

/// A chord another command has is not taken from it: the row says whose it
/// is, and both keep what they had.
#[test]
fn a_captured_chord_another_command_has_is_refused_and_names_it() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    keys_page(&mut editor);
    let at = chord_on_row(&editor, "Duplicate", chord(CMD, Key::D));
    editor.click(at).frames(1);
    editor.key_with(CMD, Key::C).frames(1);
    assert!(
        editor.painted_contains("already runs Copy"),
        "{:?}",
        editor.painted()
    );
    assert!(editor.state.settings.keys.is_default(Command::Duplicate));
    assert!(editor.state.settings.keys.is_default(Command::Copy));
    assert!(
        editor.state.history.undo_label().is_none(),
        "nothing was copied"
    );
}

/// The palette opens on its chord with its field ready, keeps the commands
/// whose label or group holds what is typed, and runs the highlighted one on
/// Enter.
#[test]
fn the_palette_filters_as_typed_and_runs_the_highlighted_command_on_enter() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let lamp = editor.spawn("Lamp");
    editor.state.selection.set(lamp);
    editor.frames(2);
    open_palette(&mut editor);
    assert!(editor.painted_contains("Duplicate"));
    assert!(editor.painted_contains("Save scene"));
    assert!(
        editor.painted_contains("Show Console"),
        "and the dock's tools"
    );

    type_text(&mut editor, "dupl");
    assert!(editor.painted_contains("Duplicate"));
    assert!(!editor.painted_contains("Save scene"));
    assert!(
        editor
            .painted()
            .contains(&shown(&editor, chord(CMD, Key::D))),
        "with its chord"
    );
    editor.key(Key::Enter);
    editor.state.apply(&mut editor.world, &editor.registry);
    editor.frames(1);
    assert_eq!(duplicates_of(&editor, "Lamp"), 2);
    assert!(!palette_open(&editor), "running it closes the palette");
}

/// A command that cannot run now is listed, not hidden, and says why.
#[test]
fn the_palette_lists_a_command_that_cannot_run_as_disabled_with_its_reason() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.spawn("Lamp");
    editor.frames(2);
    open_palette(&mut editor);
    type_text(&mut editor, "duplicate");
    assert!(
        editor.painted_contains("Duplicate"),
        "{:?}",
        editor.painted()
    );
    assert!(
        editor.painted_contains("Nothing is selected"),
        "{:?}",
        editor.painted()
    );
    editor.key(Key::Enter);
    editor.state.apply(&mut editor.world, &editor.registry);
    editor.frames(1);
    assert_eq!(duplicates_of(&editor, "Lamp"), 1);
    assert!(palette_open(&editor), "Enter on it does nothing");
}
