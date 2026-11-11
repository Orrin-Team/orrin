//! The input vocabulary: one table naming every source a binding can name.
//!
//! The engine's key codes are its own numbering rather than winit's, because
//! winit's discriminants are not a stable ABI and these numbers cross two seams
//! that outlive any dependency — binding files users commit, and the C#
//! `KeyCode` enum. This table is where the three meet: the name a config
//! spells, the code stored and sent across the scripting boundary, and the
//! winit key it arrives as. Adding a key is a row here plus the matching C#
//! variant, and nothing else needs to know.
//!
//! The names are the engine's own. A binding file is a document, and a
//! dependency's rename must not invalidate one — so where the C# enum already
//! spelled a key, that spelling wins and there is exactly one vocabulary either
//! side of the boundary. Aliases cover the spellings people reach for anyway.
//!
//! Keyboard names are bare (`A`, `Space`); every other device is prefixed
//! (`Mouse.Left`, `Gamepad.South`). That is what lets `Gamepad.A` exist
//! alongside the `A` key without either needing a longer name.

use winit::keyboard::KeyCode;

/// One keyboard key: what a config calls it, what the engine stores, and which
/// winit key produces it.
pub struct Key {
    pub name: &'static str,
    pub code: u32,
    pub aliases: &'static [&'static str],
    pub winit: KeyCode,
}

const fn key(name: &'static str, code: u32, winit: KeyCode) -> Key {
    Key {
        name,
        code,
        aliases: &[],
        winit,
    }
}

const fn aliased(
    name: &'static str,
    code: u32,
    winit: KeyCode,
    aliases: &'static [&'static str],
) -> Key {
    Key {
        name,
        code,
        aliases,
        winit,
    }
}

/// Every keyboard key the engine knows. Codes are frozen: they are what C#
/// sends across the scripting boundary and what older binding files contain.
pub const KEYS: &[Key] = &[
    key("A", 1, KeyCode::KeyA),
    key("B", 2, KeyCode::KeyB),
    key("C", 3, KeyCode::KeyC),
    key("D", 4, KeyCode::KeyD),
    key("E", 5, KeyCode::KeyE),
    key("F", 6, KeyCode::KeyF),
    key("G", 7, KeyCode::KeyG),
    key("H", 8, KeyCode::KeyH),
    key("I", 9, KeyCode::KeyI),
    key("J", 10, KeyCode::KeyJ),
    key("K", 11, KeyCode::KeyK),
    key("L", 12, KeyCode::KeyL),
    key("M", 13, KeyCode::KeyM),
    key("N", 14, KeyCode::KeyN),
    key("O", 15, KeyCode::KeyO),
    key("P", 16, KeyCode::KeyP),
    key("Q", 17, KeyCode::KeyQ),
    key("R", 18, KeyCode::KeyR),
    key("S", 19, KeyCode::KeyS),
    key("T", 20, KeyCode::KeyT),
    key("U", 21, KeyCode::KeyU),
    key("V", 22, KeyCode::KeyV),
    key("W", 23, KeyCode::KeyW),
    key("X", 24, KeyCode::KeyX),
    key("Y", 25, KeyCode::KeyY),
    key("Z", 26, KeyCode::KeyZ),
    aliased("Alpha0", 30, KeyCode::Digit0, &["0"]),
    aliased("Alpha1", 31, KeyCode::Digit1, &["1"]),
    aliased("Alpha2", 32, KeyCode::Digit2, &["2"]),
    aliased("Alpha3", 33, KeyCode::Digit3, &["3"]),
    aliased("Alpha4", 34, KeyCode::Digit4, &["4"]),
    aliased("Alpha5", 35, KeyCode::Digit5, &["5"]),
    aliased("Alpha6", 36, KeyCode::Digit6, &["6"]),
    aliased("Alpha7", 37, KeyCode::Digit7, &["7"]),
    aliased("Alpha8", 38, KeyCode::Digit8, &["8"]),
    aliased("Alpha9", 39, KeyCode::Digit9, &["9"]),
    key("LeftArrow", 40, KeyCode::ArrowLeft),
    key("RightArrow", 41, KeyCode::ArrowRight),
    key("UpArrow", 42, KeyCode::ArrowUp),
    key("DownArrow", 43, KeyCode::ArrowDown),
    key("Space", 44, KeyCode::Space),
    aliased("Return", 45, KeyCode::Enter, &["Enter"]),
    aliased("Escape", 46, KeyCode::Escape, &["Esc"]),
    key("Tab", 47, KeyCode::Tab),
    key("Backspace", 48, KeyCode::Backspace),
    aliased("LeftShift", 49, KeyCode::ShiftLeft, &["Shift"]),
    key("RightShift", 50, KeyCode::ShiftRight),
    aliased(
        "LeftControl",
        51,
        KeyCode::ControlLeft,
        &["Control", "Ctrl"],
    ),
    key("RightControl", 52, KeyCode::ControlRight),
    aliased("LeftAlt", 53, KeyCode::AltLeft, &["Alt"]),
    key("RightAlt", 54, KeyCode::AltRight),
    key("F1", 55, KeyCode::F1),
    key("F2", 56, KeyCode::F2),
    key("F3", 57, KeyCode::F3),
    key("F4", 58, KeyCode::F4),
    key("F5", 59, KeyCode::F5),
    key("F6", 60, KeyCode::F6),
    key("F7", 61, KeyCode::F7),
    key("F8", 62, KeyCode::F8),
    key("F9", 63, KeyCode::F9),
    key("F10", 64, KeyCode::F10),
    key("F11", 65, KeyCode::F11),
    key("F12", 66, KeyCode::F12),
    key("Minus", 67, KeyCode::Minus),
    key("Equals", 68, KeyCode::Equal),
    key("LeftBracket", 69, KeyCode::BracketLeft),
    key("RightBracket", 70, KeyCode::BracketRight),
    key("Backslash", 71, KeyCode::Backslash),
    key("Semicolon", 72, KeyCode::Semicolon),
    aliased("Quote", 73, KeyCode::Quote, &["Apostrophe"]),
    key("Comma", 74, KeyCode::Comma),
    key("Period", 75, KeyCode::Period),
    key("Slash", 76, KeyCode::Slash),
    aliased("BackQuote", 77, KeyCode::Backquote, &["Backtick"]),
    key("Keypad0", 78, KeyCode::Numpad0),
    key("Keypad1", 79, KeyCode::Numpad1),
    key("Keypad2", 80, KeyCode::Numpad2),
    key("Keypad3", 81, KeyCode::Numpad3),
    key("Keypad4", 82, KeyCode::Numpad4),
    key("Keypad5", 83, KeyCode::Numpad5),
    key("Keypad6", 84, KeyCode::Numpad6),
    key("Keypad7", 85, KeyCode::Numpad7),
    key("Keypad8", 86, KeyCode::Numpad8),
    key("Keypad9", 87, KeyCode::Numpad9),
    key("KeypadPlus", 88, KeyCode::NumpadAdd),
    key("KeypadMinus", 89, KeyCode::NumpadSubtract),
    key("KeypadMultiply", 90, KeyCode::NumpadMultiply),
    key("KeypadDivide", 91, KeyCode::NumpadDivide),
    key("KeypadPeriod", 92, KeyCode::NumpadDecimal),
    key("KeypadEnter", 93, KeyCode::NumpadEnter),
    key("Insert", 94, KeyCode::Insert),
    key("Delete", 95, KeyCode::Delete),
    key("Home", 96, KeyCode::Home),
    key("End", 97, KeyCode::End),
    key("PageUp", 98, KeyCode::PageUp),
    key("PageDown", 99, KeyCode::PageDown),
    key("CapsLock", 100, KeyCode::CapsLock),
];

/// Mouse buttons, by the number [`super::InputState`] and the C# `MouseButton`
/// enum both address them with. An index, not a mask bit — the mask is private
/// to `state`, which shifts by these.
pub const MOUSE_BUTTONS: &[(&str, u8)] = &[("Left", 0), ("Right", 1), ("Middle", 2)];

/// Mouse movement, in window pixels accumulated over the frame.
pub const MOUSE_AXES: &[(&str, MouseAxis)] = &[("X", MouseAxis::X), ("Y", MouseAxis::Y)];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MouseAxis {
    X,
    Y,
}

/// Face and shoulder buttons, named by position rather than by legend: `South`
/// is the same physical button whether its cap reads A or ✕. The vendor letters
/// are aliases so a config may still spell what is printed on the pad.
pub const PAD_BUTTONS: &[(&str, PadButton, &[&str])] = &[
    ("South", PadButton::South, &["A", "Cross"]),
    ("East", PadButton::East, &["B", "Circle"]),
    ("West", PadButton::West, &["X", "Square"]),
    ("North", PadButton::North, &["Y", "Triangle"]),
    ("LeftBumper", PadButton::LeftBumper, &["LB", "L1"]),
    ("RightBumper", PadButton::RightBumper, &["RB", "R1"]),
    ("LeftStick", PadButton::LeftStick, &["L3"]),
    ("RightStick", PadButton::RightStick, &["R3"]),
    ("DPadUp", PadButton::DPadUp, &[]),
    ("DPadDown", PadButton::DPadDown, &[]),
    ("DPadLeft", PadButton::DPadLeft, &[]),
    ("DPadRight", PadButton::DPadRight, &[]),
    ("Start", PadButton::Start, &[]),
    ("Select", PadButton::Select, &["Back"]),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadButton {
    South,
    East,
    West,
    North,
    LeftBumper,
    RightBumper,
    LeftStick,
    RightStick,
    DPadUp,
    DPadDown,
    DPadLeft,
    DPadRight,
    Start,
    Select,
}

impl PadButton {
    /// Bit position in [`super::InputState`]'s per-pad button mask.
    pub fn index(self) -> u32 {
        self as u32
    }
}

/// Analog pad sources. Triggers are axes rather than buttons; an action that
/// binds one compares it against a threshold.
pub const PAD_AXES: &[(&str, PadAxis)] = &[
    ("LeftStick.X", PadAxis::LeftStickX),
    ("LeftStick.Y", PadAxis::LeftStickY),
    ("RightStick.X", PadAxis::RightStickX),
    ("RightStick.Y", PadAxis::RightStickY),
    ("LeftTrigger", PadAxis::LeftTrigger),
    ("RightTrigger", PadAxis::RightTrigger),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PadAxis {
    LeftStickX,
    LeftStickY,
    RightStickX,
    RightStickY,
    LeftTrigger,
    RightTrigger,
}

impl PadAxis {
    /// How many analog sources a pad has, and so how wide its axis store is.
    pub const COUNT: usize = 6;

    pub fn index(self) -> usize {
        self as usize
    }
}

/// Look a keyboard name up, canonical spelling or alias.
///
/// Matching ignores case: this table is read from a file someone typed, and
/// refusing `space` teaches nothing that accepting it does not. Errors quote the
/// canonical spelling, so the file still converges on one form.
pub fn key_by_name(name: &str) -> Option<u32> {
    KEYS.iter()
        .find(|key| {
            key.name.eq_ignore_ascii_case(name)
                || key
                    .aliases
                    .iter()
                    .any(|alias| alias.eq_ignore_ascii_case(name))
        })
        .map(|key| key.code)
}

/// The engine code for a winit key, or `None` for one the engine has no name
/// for. Linear over the table, on an event that arrives at human speed.
pub fn key_from_winit(code: KeyCode) -> Option<u32> {
    KEYS.iter()
        .find(|key| key.winit == code)
        .map(|key| key.code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_appended_keys_arrive_under_their_codes() {
        let cases = [
            (KeyCode::F1, 55),
            (KeyCode::F12, 66),
            (KeyCode::Minus, 67),
            (KeyCode::Quote, 73),
            (KeyCode::Backquote, 77),
            (KeyCode::Numpad0, 78),
            (KeyCode::Numpad9, 87),
            (KeyCode::NumpadEnter, 93),
            (KeyCode::Insert, 94),
            (KeyCode::PageDown, 99),
            (KeyCode::CapsLock, 100),
        ];
        for (winit, code) in cases {
            assert_eq!(key_from_winit(winit), Some(code), "{winit:?}");
        }
    }

    #[test]
    fn the_appended_keys_are_named_as_unity_names_them() {
        for (name, code) in [
            ("F5", 59),
            ("LeftBracket", 69),
            ("Apostrophe", 73),
            ("Keypad7", 85),
            ("KeypadPlus", 88),
            ("Delete", 95),
        ] {
            assert_eq!(key_by_name(name), Some(code), "{name}");
        }
    }

    /// A row pasted twice is a key that silently answers to the wrong code.
    #[test]
    fn no_two_rows_share_a_code_a_winit_key_or_a_name() {
        for (i, a) in KEYS.iter().enumerate() {
            for b in &KEYS[i + 1..] {
                assert_ne!(a.code, b.code, "{} and {}", a.name, b.name);
                assert_ne!(a.winit, b.winit, "{} and {}", a.name, b.name);
                let spellings = |key: &Key| {
                    std::iter::once(key.name)
                        .chain(key.aliases.iter().copied())
                        .map(str::to_ascii_lowercase)
                        .collect::<Vec<_>>()
                };
                let theirs = spellings(b);
                assert!(
                    spellings(a).iter().all(|name| !theirs.contains(name)),
                    "{} and {} share a spelling",
                    a.name,
                    b.name
                );
            }
        }
    }
}
