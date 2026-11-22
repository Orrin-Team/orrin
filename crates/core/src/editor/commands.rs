//! Every command the editor has a key for, in one table: its id, its label,
//! the group it is listed under, its default chords and when it stands aside.
//! The keys `panels::shortcuts` answers, the text beside a menu entry, a
//! tooltip's "(Ctrl+P)", the Keys page in Settings and the command palette are
//! all read from here, so a key and the text that names it cannot disagree.
//!
//! **An id is persisted.** It is how `[keys]` in the user's `settings.toml`
//! names a command, so renaming one silently forgets that binding for everyone
//! who changed it: a loud break, which the test pinning the ids makes so. The
//! label is free to change.
//!
//! **Deviations only.** [`Keys`] holds the commands bound to something other
//! than their defaults, by id, as the rest of the settings file does; a new
//! default reaches everyone who never rebound that command. An id this build
//! does not know is kept, so a newer engine's binding survives an older one
//! saving the file, and a chord that does not read costs that one binding.
//!
//! **The most specific chord wins a press.** egui matches Ctrl+Z whether or not
//! Shift is down, so Ctrl+Shift+Z has to reach redo rather than undo, and the
//! pause and step chords pause and step rather than play. Each press goes to
//! the binding with the most modifiers among those it matches
//! ([`Keys::best`]), whatever order the table lists them in; a binding that
//! stands aside leaves the press where it was rather than handing it to a
//! less specific one.
//!
//! **The window sends copy, cut and paste as events, never as keys** (see
//! `egui_winit`), so [`press`] reads those three as Command+C, X and V with
//! whatever else is held. That is what lets Copy be rebound like any command.
//!
//! **What is not a command stays out.** The fly camera's WASD/QE and mouse
//! buttons, F1 (the overlay toggle, which `app.rs` reads before egui exists in
//! the frame), the hierarchy's arrow keys and every key inside a text field
//! keep their places: none is a command anyone looks up by name, and the
//! arrows come in Shift pairs that would double the table for nothing. A
//! chord the camera, F1 or the arrows answer is refused as a binding
//! ([`reserved`]), so a key never quietly does two things; inside a field, a
//! command's key stands aside instead ([`Focus`]).

use std::collections::BTreeMap;

use egui::{Key, KeyboardShortcut, Modifiers};
use serde::{Deserialize, Serialize};

/// A key with the modifiers held for it. egui's own, so that
/// [`egui::Context::format_shortcut`] names it right on every platform.
pub type Chord = KeyboardShortcut;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Command {
    Save,
    SaveAs,
    Open,
    Reload,
    Recover,
    ProjectSettings,
    Undo,
    Redo,
    Duplicate,
    Copy,
    Paste,
    Rename,
    Delete,
    SavePrefab,
    SelectAll,
    SelectNone,
    Frame,
    Dump,
    Settings,
    Play,
    Pause,
    Step,
    SelectTool,
    MoveTool,
    RotateTool,
    ScaleTool,
    UiMode,
    Palette,
    ResetLayout,
    ReloadScripts,
}

/// Where a command is listed, on the Keys page and in the palette.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    File,
    Edit,
    Play,
    Viewport,
    View,
    Scripts,
}

impl Group {
    pub const ALL: [Self; 6] = [
        Self::File,
        Self::Edit,
        Self::Play,
        Self::Viewport,
        Self::View,
        Self::Scripts,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::File => "File",
            Self::Edit => "Edit",
            Self::Play => "Play",
            Self::Viewport => "Viewport",
            Self::View => "View",
            Self::Scripts => "Scripts",
        }
    }
}

/// Which focused field a command's key stands aside for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Focus {
    /// None: the key means the same in a field, where it types nothing.
    Anywhere,
    /// The Shader tab's code alone, which saves itself on the same key and
    /// is where a stray Play would lose the text's place.
    OutsideCode,
    /// Any text field, where the same key edits text.
    OutsideFields,
}

pub struct Spec {
    pub command: Command,
    pub id: &'static str,
    pub label: &'static str,
    pub group: Group,
    /// The first is the one a menu shows.
    pub keys: &'static [Chord],
    pub focus: Focus,
    /// Read by the tool under the pointer rather than by `panels::shortcuts`:
    /// Rename is the asset browser's over the browser and the tree's over the
    /// tree.
    pub local: bool,
}

const NONE: Modifiers = Modifiers::NONE;
const CMD: Modifiers = Modifiers::COMMAND;
const CMD_SHIFT: Modifiers = Modifiers::COMMAND.plus(Modifiers::SHIFT);
const CMD_ALT: Modifiers = Modifiers::COMMAND.plus(Modifiers::ALT);

const fn chord(modifiers: Modifiers, key: Key) -> Chord {
    KeyboardShortcut::new(modifiers, key)
}

const fn spec(
    command: Command,
    id: &'static str,
    label: &'static str,
    group: Group,
    keys: &'static [Chord],
    focus: Focus,
) -> Spec {
    Spec {
        command,
        id,
        label,
        group,
        keys,
        focus,
        local: false,
    }
}

use Focus::{Anywhere, OutsideCode, OutsideFields};

/// In [`Command`]'s order, which [`Command::spec`] indexes by.
pub const COMMANDS: &[Spec] = &[
    spec(
        Command::Save,
        "file.save",
        "Save scene",
        Group::File,
        &[chord(CMD, Key::S)],
        OutsideCode,
    ),
    spec(
        Command::SaveAs,
        "file.save_as",
        "Save scene as…",
        Group::File,
        &[],
        OutsideFields,
    ),
    spec(
        Command::Open,
        "file.open",
        "Open scene…",
        Group::File,
        &[],
        OutsideFields,
    ),
    spec(
        Command::Reload,
        "file.reload",
        "Reload scene",
        Group::File,
        &[],
        OutsideFields,
    ),
    spec(
        Command::Recover,
        "file.recover",
        "Recover autosave",
        Group::File,
        &[],
        OutsideFields,
    ),
    spec(
        Command::ProjectSettings,
        "file.project_settings",
        "Project settings…",
        Group::File,
        &[],
        OutsideFields,
    ),
    spec(
        Command::Undo,
        "edit.undo",
        "Undo",
        Group::Edit,
        &[chord(CMD, Key::Z)],
        OutsideFields,
    ),
    spec(
        Command::Redo,
        "edit.redo",
        "Redo",
        Group::Edit,
        &[chord(CMD_SHIFT, Key::Z), chord(CMD, Key::Y)],
        OutsideFields,
    ),
    spec(
        Command::Duplicate,
        "edit.duplicate",
        "Duplicate",
        Group::Edit,
        &[chord(CMD, Key::D)],
        OutsideFields,
    ),
    spec(
        Command::Copy,
        "edit.copy",
        "Copy",
        Group::Edit,
        &[chord(CMD, Key::C)],
        OutsideFields,
    ),
    spec(
        Command::Paste,
        "edit.paste",
        "Paste",
        Group::Edit,
        &[chord(CMD, Key::V)],
        OutsideFields,
    ),
    Spec {
        local: true,
        ..spec(
            Command::Rename,
            "edit.rename",
            "Rename",
            Group::Edit,
            &[chord(NONE, Key::F2)],
            OutsideFields,
        )
    },
    spec(
        Command::Delete,
        "edit.delete",
        "Delete",
        Group::Edit,
        &[chord(NONE, Key::Delete)],
        OutsideFields,
    ),
    spec(
        Command::SavePrefab,
        "edit.save_prefab",
        "Save as prefab",
        Group::Edit,
        &[],
        OutsideFields,
    ),
    spec(
        Command::SelectAll,
        "edit.select_all",
        "Select all",
        Group::Edit,
        &[chord(CMD, Key::A)],
        OutsideFields,
    ),
    spec(
        Command::SelectNone,
        "edit.select_none",
        "Select none",
        Group::Edit,
        &[chord(NONE, Key::Escape)],
        OutsideFields,
    ),
    spec(
        Command::Frame,
        "edit.frame",
        "Frame selection",
        Group::Edit,
        &[chord(NONE, Key::F)],
        OutsideFields,
    ),
    spec(
        Command::Dump,
        "edit.dump",
        "Dump to console",
        Group::Edit,
        &[],
        OutsideFields,
    ),
    spec(
        Command::Settings,
        "edit.settings",
        "Settings…",
        Group::Edit,
        &[chord(CMD, Key::Comma)],
        Anywhere,
    ),
    spec(
        Command::Play,
        "play.play",
        "Play or stop",
        Group::Play,
        &[chord(CMD, Key::P)],
        OutsideCode,
    ),
    spec(
        Command::Pause,
        "play.pause",
        "Pause or resume",
        Group::Play,
        &[chord(CMD_SHIFT, Key::P)],
        OutsideCode,
    ),
    spec(
        Command::Step,
        "play.step",
        "Step one frame",
        Group::Play,
        &[chord(CMD_ALT, Key::P)],
        OutsideCode,
    ),
    spec(
        Command::SelectTool,
        "viewport.select",
        "Select tool",
        Group::Viewport,
        &[chord(NONE, Key::Num1)],
        OutsideFields,
    ),
    spec(
        Command::MoveTool,
        "viewport.move",
        "Move tool",
        Group::Viewport,
        &[chord(NONE, Key::Num2)],
        OutsideFields,
    ),
    spec(
        Command::RotateTool,
        "viewport.rotate",
        "Rotate tool",
        Group::Viewport,
        &[chord(NONE, Key::Num3)],
        OutsideFields,
    ),
    spec(
        Command::ScaleTool,
        "viewport.scale",
        "Scale tool",
        Group::Viewport,
        &[chord(NONE, Key::Num4)],
        OutsideFields,
    ),
    spec(
        Command::UiMode,
        "viewport.ui_mode",
        "UI mode",
        Group::Viewport,
        &[chord(NONE, Key::T)],
        OutsideFields,
    ),
    spec(
        Command::Palette,
        "view.palette",
        "Command palette…",
        Group::View,
        &[chord(CMD, Key::K)],
        Anywhere,
    ),
    spec(
        Command::ResetLayout,
        "view.reset_layout",
        "Reset layout",
        Group::View,
        &[],
        OutsideFields,
    ),
    spec(
        Command::ReloadScripts,
        "scripts.reload",
        "Reload scripts",
        Group::Scripts,
        &[],
        OutsideFields,
    ),
];

impl Command {
    pub fn spec(self) -> &'static Spec {
        &COMMANDS[self as usize]
    }

    pub fn id(self) -> &'static str {
        self.spec().id
    }

    pub fn label(self) -> &'static str {
        self.spec().label
    }

    pub fn group(self) -> Group {
        self.spec().group
    }

    pub fn defaults(self) -> &'static [Chord] {
        self.spec().keys
    }

    pub fn from_id(id: &str) -> Option<Self> {
        COMMANDS
            .iter()
            .find(|spec| spec.id == id)
            .map(|spec| spec.command)
    }
}

/// Why a chord was not bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Another command has it, which is unbound there first.
    Held(Command),
    /// Something that is not a command answers it: what the chord does, to
    /// follow its name.
    Reserved(&'static str),
}

/// What a chord is kept for outside the table, if anything: the overlay
/// toggle `app.rs` reads ahead of egui, the fly camera's keys, which it takes
/// with Shift for speed but never with Ctrl, Alt or Cmd, and the hierarchy's
/// arrows, which it takes with Shift to extend the selection.
pub fn reserved(chord: Chord) -> Option<&'static str> {
    let Modifiers {
        alt, ctrl, command, ..
    } = chord.modifiers;
    match chord.logical_key {
        Key::F1 => Some("shows and hides the editor"),
        Key::W | Key::A | Key::S | Key::D | Key::Q | Key::E if !(alt || ctrl || command) => {
            Some("flies the camera while the right button is held")
        }
        Key::ArrowUp | Key::ArrowDown | Key::ArrowLeft | Key::ArrowRight
            if !(alt || ctrl || command) =>
        {
            Some("walks the hierarchy's rows")
        }
        _ => None,
    }
}

/// The command each chord runs, for the commands whose chords are not their
/// defaults. Kept as `[keys]` in `settings.toml`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Keys {
    /// By id; an empty list is a command left with no key.
    rebound: BTreeMap<String, Vec<Chord>>,
}

impl Keys {
    /// What runs `command` now: its rebinding, or its defaults.
    pub fn chords(&self, command: Command) -> &[Chord] {
        self.rebound
            .get(command.id())
            .map_or(command.defaults(), Vec::as_slice)
    }

    /// The chord a menu shows beside `command`.
    pub fn primary(&self, command: Command) -> Option<Chord> {
        self.chords(command).first().copied()
    }

    pub fn is_default(&self, command: Command) -> bool {
        !self.rebound.contains_key(command.id())
    }

    pub fn reset(&mut self, command: Command) {
        self.rebound.remove(command.id());
    }

    fn set(&mut self, command: Command, chords: Vec<Chord>) {
        if chords == command.defaults() {
            self.reset(command);
        } else {
            self.rebound.insert(command.id().to_owned(), chords);
        }
    }

    /// The command `chord` is bound to exactly, if any.
    pub fn holder(&self, chord: Chord) -> Option<Command> {
        COMMANDS
            .iter()
            .map(|spec| spec.command)
            .find(|&command| self.chords(command).contains(&chord))
    }

    /// Put `chord` in `slot` of `command`'s chords, or after them for `None`.
    /// A chord another command holds, or one [`reserved`] for something else,
    /// is refused and nothing changes.
    pub fn bind(
        &mut self,
        command: Command,
        slot: Option<usize>,
        chord: Chord,
    ) -> Result<(), Refusal> {
        if let Some(why) = reserved(chord) {
            return Err(Refusal::Reserved(why));
        }
        match self.holder(chord) {
            Some(other) if other != command => return Err(Refusal::Held(other)),
            Some(_) => return Ok(()),
            None => {}
        }
        let mut chords = self.chords(command).to_vec();
        match slot.filter(|&slot| slot < chords.len()) {
            Some(slot) => chords[slot] = chord,
            None => chords.push(chord),
        }
        self.set(command, chords);
        Ok(())
    }

    /// Take the chord in `slot` off `command`.
    pub fn unbind(&mut self, command: Command, slot: usize) {
        let mut chords = self.chords(command).to_vec();
        if slot < chords.len() {
            chords.remove(slot);
            self.set(command, chords);
        }
    }

    /// The binding a press goes to: of every chord it matches, the one with
    /// the most modifiers, and of those the first in the table.
    pub fn best(&self, press: Chord) -> Option<(Command, Chord)> {
        let mut best: Option<(Command, Chord)> = None;
        for spec in COMMANDS {
            for &chord in self.chords(spec.command) {
                let better = best.is_none_or(|(_, held)| weight(chord) > weight(held));
                if better && matches(press, chord) {
                    best = Some((spec.command, chord));
                }
            }
        }
        best
    }

    /// Whether `command`'s chord was pressed this frame. For a [`Spec::local`]
    /// command, which the tool under the pointer reads without taking.
    pub fn pressed(&self, input: &egui::InputState, command: Command) -> bool {
        input
            .events
            .iter()
            .any(|event| self.runs(event, input.modifiers, command))
    }

    /// Take the first press of `command`'s chord this frame; whether there was one.
    pub fn take(&self, input: &mut egui::InputState, command: Command) -> bool {
        let held = input.modifiers;
        match input
            .events
            .iter()
            .position(|event| self.runs(event, held, command))
        {
            Some(at) => {
                input.events.remove(at);
                true
            }
            None => false,
        }
    }

    fn runs(&self, event: &egui::Event, held: Modifiers, command: Command) -> bool {
        press(event, held)
            .and_then(|press| self.best(press))
            .is_some_and(|(best, _)| best == command)
    }
}

/// Whether a press of `press` is one of `chord`, as egui's `consume_key`
/// decides it: Shift and Alt the chord does not name may be down too.
fn matches(press: Chord, chord: Chord) -> bool {
    press.logical_key == chord.logical_key && press.modifiers.matches_logically(chord.modifiers)
}

fn weight(chord: Chord) -> u8 {
    let Modifiers {
        alt,
        ctrl,
        shift,
        command,
        ..
    } = chord.modifiers;
    [alt, ctrl, shift, command].into_iter().map(u8::from).sum()
}

/// What the key in `event` is as a chord, with `held` being the modifiers
/// down this frame: a key going down, or one of the window's clipboard
/// events. Ctrl is folded into Command where the two are the same key.
pub fn press(event: &egui::Event, held: Modifiers) -> Option<Chord> {
    let clipboard = |key| Some(chord(normalised(held).plus(CMD), key));
    match event {
        egui::Event::Key {
            key,
            pressed: true,
            modifiers,
            ..
        } => Some(chord(normalised(*modifiers), *key)),
        egui::Event::Copy => clipboard(Key::C),
        egui::Event::Cut => clipboard(Key::X),
        egui::Event::Paste(_) => clipboard(Key::V),
        _ => None,
    }
}

/// The modifiers a chord is kept with: Command for the platform's command
/// key, and Ctrl only where that is a different key, as on a Mac.
pub fn normalised(modifiers: Modifiers) -> Modifiers {
    let ctrl_is_command = modifiers.command && !modifiers.mac_cmd;
    Modifiers {
        alt: modifiers.alt,
        ctrl: modifiers.ctrl && !ctrl_is_command,
        shift: modifiers.shift,
        mac_cmd: false,
        command: modifiers.command,
    }
}

/// `command`'s chord as the menus show it, or nothing when it has none.
pub fn text(ctx: &egui::Context, keys: &Keys, command: Command) -> String {
    keys.primary(command)
        .map_or_else(String::new, |chord| ctx.format_shortcut(&chord))
}

/// A tooltip naming the chord: "Play (Ctrl+P)", or the words alone.
pub fn hint(ctx: &egui::Context, keys: &Keys, command: Command, words: &str) -> String {
    match keys.primary(command) {
        Some(chord) => format!("{words} ({})", ctx.format_shortcut(&chord)),
        None => words.to_owned(),
    }
}

/// How a chord is written in the settings file: `Command+Shift+Z`. Command
/// rather than Ctrl or Cmd, because it is the platform's command key on each,
/// and the key by egui's name for it.
pub fn chord_text(chord: Chord) -> String {
    let Modifiers {
        alt,
        ctrl,
        shift,
        command,
        ..
    } = chord.modifiers;
    let mut text = String::new();
    for (held, name) in [
        (command, "Command"),
        (ctrl, "Ctrl"),
        (alt, "Alt"),
        (shift, "Shift"),
    ] {
        if held {
            text.push_str(name);
            text.push('+');
        }
    }
    text.push_str(chord.logical_key.name());
    text
}

/// The reverse of [`chord_text`]. Ctrl reads as Command except on a Mac,
/// where they are two keys.
pub fn parse_chord(text: &str) -> Option<Chord> {
    let mut parts: Vec<&str> = text.split('+').map(str::trim).collect();
    let key = Key::from_name(parts.pop()?)?;
    let mut modifiers = NONE;
    for part in parts {
        modifiers = modifiers.plus(match part {
            "Command" | "Cmd" => CMD,
            "Ctrl" | "Control" if cfg!(target_os = "macos") => Modifiers::CTRL,
            "Ctrl" | "Control" => CMD,
            "Alt" | "Option" => Modifiers::ALT,
            "Shift" => Modifiers::SHIFT,
            _ => return None,
        });
    }
    Some(chord(modifiers, key))
}

impl Serialize for Keys {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let written: BTreeMap<&str, Vec<String>> = self
            .rebound
            .iter()
            .map(|(id, chords)| {
                (
                    id.as_str(),
                    chords.iter().copied().map(chord_text).collect(),
                )
            })
            .collect();
        written.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Keys {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let read = BTreeMap::<String, toml::Value>::deserialize(deserializer)?;
        let mut keys = Self::default();
        for (id, value) in read {
            let texts = match value {
                toml::Value::String(one) => Some(vec![one]),
                many => many.try_into::<Vec<String>>().ok(),
            };
            let chords: Option<Vec<Chord>> =
                texts.and_then(|texts| texts.iter().map(|text| parse_chord(text)).collect());
            match (chords, Command::from_id(&id)) {
                (Some(chords), Some(command)) => keys.set(command, chords),
                (Some(chords), None) => {
                    keys.rebound.insert(id, chords);
                }
                (None, _) => {
                    tracing::warn!("ignoring the key binding {id}: it does not read as chords")
                }
            }
        }
        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_in_the_order_its_commands_index_it() {
        for (at, spec) in COMMANDS.iter().enumerate() {
            assert_eq!(spec.command as usize, at, "{}", spec.id);
            assert_eq!(spec.command.spec().id, spec.id);
            assert_eq!(Command::from_id(spec.id), Some(spec.command));
        }
        assert_eq!(COMMANDS.len(), Command::ReloadScripts as usize + 1);
    }

    /// The ids are what `settings.toml` names a binding by. Renaming one
    /// forgets that binding for whoever changed it, so a rename is made here,
    /// on purpose, or not at all.
    #[test]
    fn the_ids_are_the_ones_settings_files_already_name() {
        let ids: Vec<&str> = COMMANDS.iter().map(|spec| spec.id).collect();
        assert_eq!(
            ids,
            [
                "file.save",
                "file.save_as",
                "file.open",
                "file.reload",
                "file.recover",
                "file.project_settings",
                "edit.undo",
                "edit.redo",
                "edit.duplicate",
                "edit.copy",
                "edit.paste",
                "edit.rename",
                "edit.delete",
                "edit.save_prefab",
                "edit.select_all",
                "edit.select_none",
                "edit.frame",
                "edit.dump",
                "edit.settings",
                "play.play",
                "play.pause",
                "play.step",
                "viewport.select",
                "viewport.move",
                "viewport.rotate",
                "viewport.scale",
                "viewport.ui_mode",
                "view.palette",
                "view.reset_layout",
                "scripts.reload",
            ]
        );
    }

    /// One press, one command: a default chord two commands shared would run
    /// whichever the table lists first and leave the other unreachable.
    #[test]
    fn no_two_commands_share_a_default_chord() {
        let mut seen: Vec<(Chord, &str)> = Vec::new();
        for spec in COMMANDS {
            for &chord in spec.keys {
                if let Some((_, other)) = seen.iter().find(|(held, _)| *held == chord) {
                    panic!("{} and {other} both default to {chord:?}", spec.id);
                }
                assert_eq!(
                    reserved(chord),
                    None,
                    "{} defaults to a reserved key",
                    spec.id
                );
                seen.push((chord, spec.id));
            }
        }
    }

    #[test]
    fn a_rebinding_is_kept_until_it_is_reset_and_the_default_is_not_kept() {
        let mut keys = Keys::default();
        let shifted = chord(CMD_SHIFT, Key::D);
        keys.bind(Command::Duplicate, Some(0), shifted).unwrap();
        assert_eq!(keys.chords(Command::Duplicate), [shifted]);
        assert!(!keys.is_default(Command::Duplicate));
        assert_eq!(keys.holder(shifted), Some(Command::Duplicate));
        assert_eq!(keys.holder(chord(CMD, Key::D)), None);

        keys.reset(Command::Duplicate);
        assert_eq!(keys.chords(Command::Duplicate), [chord(CMD, Key::D)]);
        assert_eq!(keys, Keys::default());

        // Bound back to what it was by hand is no deviation either.
        keys.bind(Command::Duplicate, Some(0), shifted).unwrap();
        keys.bind(Command::Duplicate, Some(0), chord(CMD, Key::D))
            .unwrap();
        assert_eq!(keys, Keys::default());
    }

    #[test]
    fn a_chord_another_command_holds_is_refused_and_names_it() {
        let mut keys = Keys::default();
        assert_eq!(
            keys.bind(Command::Duplicate, Some(0), chord(CMD, Key::C)),
            Err(Refusal::Held(Command::Copy))
        );
        assert_eq!(
            keys.bind(Command::Frame, None, chord(CMD, Key::Y)),
            Err(Refusal::Held(Command::Redo)),
            "a second chord counts"
        );
        assert_eq!(keys, Keys::default(), "nothing changed");

        // Freed first, it can be taken.
        keys.unbind(Command::Copy, 0);
        keys.bind(Command::Duplicate, Some(0), chord(CMD, Key::C))
            .unwrap();
        assert_eq!(keys.holder(chord(CMD, Key::C)), Some(Command::Duplicate));
    }

    /// egui matches Ctrl+Z with Shift down too; the more specific binding
    /// takes the press, wherever the table lists it.
    #[test]
    fn a_press_goes_to_the_most_specific_chord_it_matches() {
        let keys = Keys::default();
        assert_eq!(
            keys.best(chord(CMD_SHIFT, Key::Z)).map(|(c, _)| c),
            Some(Command::Redo)
        );
        assert_eq!(
            keys.best(chord(CMD, Key::Z)).map(|(c, _)| c),
            Some(Command::Undo)
        );
        assert_eq!(
            keys.best(chord(CMD_ALT, Key::P)).map(|(c, _)| c),
            Some(Command::Step)
        );
        assert_eq!(
            keys.best(chord(CMD_SHIFT, Key::P)).map(|(c, _)| c),
            Some(Command::Pause)
        );
        assert_eq!(
            keys.best(chord(CMD, Key::P)).map(|(c, _)| c),
            Some(Command::Play)
        );
        // As egui's own matching does: an extra Shift still reaches Ctrl+D.
        assert_eq!(
            keys.best(chord(CMD_SHIFT, Key::D)).map(|(c, _)| c),
            Some(Command::Duplicate)
        );
        assert_eq!(keys.best(chord(NONE, Key::D)), None);
    }
}
