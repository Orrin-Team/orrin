//! The input system's behaviour, device-free.
//!
//! Everything here drives `InputState` through its own event entry points and
//! resolves by hand, so a run needs no keyboard, no pad and no window. That is
//! the point of the seam in `scene::input`: the layers above the backend are
//! ordinary data, and the parts worth pinning down — edges, deadzones, player
//! slots, what a bad config file says — are all above it.

use std::path::{Path, PathBuf};

use orrin_core::scene::input::config::{self, ConfigError};
use orrin_core::scene::input::keys::{PadAxis, PadButton};
use orrin_core::scene::input::{Actions, Binding, InputState, Spec, binding, keys};

use winit::event::{DeviceEvent, WindowEvent};

/// The engine key code for a name, for tests that press one.
fn code(name: &str) -> u32 {
    keys::key_by_name(name).unwrap_or_else(|| panic!("`{name}` is a key"))
}

fn specs(text: &str) -> Vec<(String, Spec)> {
    config::parse(text, Path::new("input.toml")).expect("the config parses")
}

fn error(text: &str) -> ConfigError {
    config::parse(text, Path::new("input.toml")).expect_err("the config is refused")
}

/// An `Actions` with `text` applied and a fresh `InputState`, settled by one
/// frame — the same frame the engine runs between reading the config at startup
/// and the player touching anything. Without it every test would spend its first
/// press on the reseed `apply` leaves behind.
fn loaded(text: &str) -> (Actions, InputState) {
    let mut actions = Actions::new();
    actions.apply(specs(text));
    let mut input = InputState::new();
    frame(&mut actions, &mut input);
    (actions, input)
}

/// One frame of the app's input order: resolve what the devices are doing, then
/// expire the one-frame edges. Tests that skip this see last frame's press
/// again, exactly as a frame that forgot `end_frame` would.
fn frame(actions: &mut Actions, input: &mut InputState) {
    actions.resolve(input);
    input.end_frame();
}

// --- The config file -------------------------------------------------------

#[test]
fn an_action_lists_alternative_sources() {
    let specs = specs(
        r#"
        format_version = 1
        [actions]
        Jump = ["Space", "Gamepad.South", "Mouse.Left"]
        "#,
    );
    assert_eq!(
        specs,
        vec![(
            "Jump".to_owned(),
            Spec::Action(vec![
                Binding::Key(code("Space")),
                Binding::PadButton(PadButton::South),
                Binding::MouseButton(0),
            ])
        )]
    );
}

#[test]
fn a_pad_button_accepts_the_legend_printed_on_it() {
    for (name, expected) in [
        ("Gamepad.South", PadButton::South),
        ("Gamepad.A", PadButton::South),
        ("Gamepad.Cross", PadButton::South),
        ("gamepad.l1", PadButton::LeftBumper),
    ] {
        assert_eq!(
            binding::parse(name),
            Ok(Binding::PadButton(expected)),
            "`{name}` names {expected:?}"
        );
    }
}

#[test]
fn a_misspelled_source_suggests_the_canonical_spelling() {
    let error = binding::parse("Retrun").expect_err("`Retrun` names nothing");
    assert_eq!(error.suggestion.as_deref(), Some("Return"));

    // An alias is a spelling the file may use, but a suggestion points at the
    // canonical one so a project converges on a single form.
    assert!(
        binding::parse("Esc").is_ok(),
        "`Esc` is an alias for `Escape`"
    );
    let error = binding::parse("Escpe").expect_err("`Escpe` names nothing");
    assert_eq!(error.suggestion.as_deref(), Some("Escape"));

    // Nothing is within a third of one character, so the message falls back to
    // teaching the prefix rule rather than inventing a neighbour.
    let error = binding::parse("Gamepad.Zzzzzzzzzz").expect_err("names nothing");
    assert_eq!(error.suggestion, None);
}

#[test]
fn a_composed_axis_missing_one_half_says_which_half() {
    let error = error(
        r#"
        format_version = 1
        [actions]
        Right = ["D"]
        [axes.Horizontal]
        positive = "Right"
        "#,
    );
    let message = error.to_string();
    assert!(
        message.contains("negative"),
        "a half-written composed axis should name the missing key, said: {message}"
    );
    assert!(
        !message.contains("not both"),
        "forgetting `negative` is not the same mistake as mixing both shapes, \
         said: {message}"
    );
}

#[test]
fn an_axis_that_mixes_both_shapes_is_still_refused() {
    let error = error(
        r#"
        format_version = 1
        [actions]
        Right = ["D"]
        Left = ["A"]
        [axes.Horizontal]
        positive = "Right"
        negative = "Left"
        source = "Gamepad.LeftStick.X"
        "#,
    );
    assert!(error.to_string().contains("not both"), "{error}");
}

#[test]
fn a_deadzone_on_an_unbounded_source_is_refused() {
    // Mouse movement is a pixel delta with no saturation point, so there is
    // nothing for a deadzone to rescale against. Accepting the key and
    // ignoring it is the silent half of this bug.
    let error = error(
        r#"
        format_version = 1
        [axes.LookX]
        source = "Mouse.X"
        deadzone = 0.2
        "#,
    );
    assert!(matches!(error, ConfigError::AxisShape { .. }), "{error}");
    assert!(error.to_string().contains("deadzone"), "{error}");
}

#[test]
fn a_scale_on_an_unbounded_source_is_still_allowed() {
    let specs = specs(
        r#"
        format_version = 1
        [axes.LookX]
        source = "Mouse.X"
        scale = 0.1
        "#,
    );
    assert!(matches!(
        specs.as_slice(),
        [(_, Spec::Analog { scale, .. })] if *scale == 0.1
    ));
}

#[test]
fn a_deadzone_outside_its_range_is_refused_rather_than_clamped() {
    for value in ["-0.1", "0.95", "1.0", "nan"] {
        let text = format!(
            r#"
            format_version = 1
            [axes.LookX]
            source = "Gamepad.RightStick.X"
            deadzone = {value}
            "#
        );
        let error = error(&text);
        assert!(
            matches!(error, ConfigError::AxisShape { .. }),
            "deadzone {value} should be refused, got {error}"
        );
    }
}

#[test]
fn a_scale_that_is_not_a_number_is_refused() {
    // A NaN scale is the worst kind of accepted value: it reaches a transform,
    // the geometry vanishes, and nothing anywhere reports a bad config.
    for value in ["nan", "inf", "-inf"] {
        let text = format!(
            r#"
            format_version = 1
            [axes.LookX]
            source = "Gamepad.RightStick.X"
            scale = {value}
            "#
        );
        let error = error(&text);
        assert!(
            matches!(error, ConfigError::AxisShape { .. }),
            "scale {value} should be refused, got {error}"
        );
    }
}

#[test]
fn an_axis_component_that_is_not_an_action_is_refused() {
    let error = error(
        r#"
        format_version = 1
        [actions]
        Right = ["D"]
        [axes.Horizontal]
        positive = "Right"
        negative = "Left"
        "#,
    );
    assert!(
        matches!(error, ConfigError::UnknownComponent { ref component, .. } if component == "Left"),
        "{error}"
    );
}

#[test]
fn one_name_cannot_be_both_an_action_and_an_axis() {
    let error = error(
        r#"
        format_version = 1
        [actions]
        Horizontal = ["D"]
        [axes.Horizontal]
        source = "Gamepad.LeftStick.X"
        "#,
    );
    assert!(
        matches!(error, ConfigError::NameCollision { .. }),
        "{error}"
    );
}

#[test]
fn a_file_from_a_newer_engine_is_refused_by_name() {
    let error = error("format_version = 999");
    assert!(
        matches!(error, ConfigError::UnsupportedVersion { found: 999, .. }),
        "{error}"
    );
}

/// The file `orrin new` writes has to parse with the parser that will read it.
///
/// The two live in crates that cannot depend on each other, so the template is
/// read out of the CLI's source the same way `input_vocabulary.rs` reads the C#
/// enum out of `Input.cs`. Without this, renaming a key in `keys.rs` ships every
/// new project a config the engine refuses on the first run.
#[test]
fn the_scaffolded_config_parses() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../orrin-cli/src/new.rs");
    let source = std::fs::read_to_string(&source).expect("new.rs is readable");
    let template = source
        .split_once("const INPUT: &str = r#\"")
        .expect("new.rs declares `const INPUT`")
        .1
        .split_once("\"#;")
        .expect("the raw string is closed")
        .0;

    let specs =
        config::parse(template, Path::new("input.toml")).expect("the scaffolded input.toml parses");
    assert!(
        specs.iter().any(|(name, _)| name == "Jump"),
        "the template should bind something recognisable"
    );
}

// --- Resolution and edges --------------------------------------------------

#[test]
fn an_action_is_held_while_any_of_its_bindings_is() {
    let (mut actions, mut input) = loaded(
        r#"
        format_version = 1
        [actions]
        Jump = ["Space", "Gamepad.South"]
        "#,
    );
    let jump = actions.id("Jump");
    assert!(!actions.held(jump, 0));

    input.set_pad_connected(0, true);
    input.set_pad_button(0, PadButton::South, true);
    frame(&mut actions, &mut input);
    assert!(actions.held(jump, 0));
    assert!(actions.pressed(jump, 0));

    input.set_pad_button(0, PadButton::South, false);
    frame(&mut actions, &mut input);
    assert!(!actions.held(jump, 0));
    assert!(actions.released(jump, 0));
}

#[test]
fn a_second_binding_joining_one_already_held_fires_no_second_press() {
    let (mut actions, mut input) = loaded(
        r#"
        format_version = 1
        [actions]
        Jump = ["Space", "Return"]
        "#,
    );
    let jump = actions.id("Jump");

    input.press(code("Space"));
    frame(&mut actions, &mut input);
    assert!(actions.pressed(jump, 0));

    input.press(code("Return"));
    frame(&mut actions, &mut input);
    assert!(actions.held(jump, 0));
    assert!(
        !actions.pressed(jump, 0),
        "the aggregate never went unheld, so nothing was pressed"
    );

    // Nor is it released while the other key holds it.
    input.release(code("Space"));
    frame(&mut actions, &mut input);
    assert!(actions.held(jump, 0));
    assert!(!actions.released(jump, 0));
}

#[test]
fn a_tap_that_starts_and_ends_inside_one_frame_still_fires() {
    // Actions read level state, so a press and its release landing between two
    // resolves would leave `key_down` false at both — the tap vanishes, while
    // the raw `GetKeyDown` beside it catches the same tap. A hitch is all it
    // takes for a frame to be long enough.
    let (mut actions, mut input) = loaded(
        r#"
        format_version = 1
        [actions]
        Jump = ["Space"]
        Fire = ["Mouse.Left"]
        "#,
    );
    let jump = actions.id("Jump");
    let fire = actions.id("Fire");

    input.press(code("Space"));
    input.release(code("Space"));
    input.press_mouse_button(0);
    input.release_mouse_button(0);
    frame(&mut actions, &mut input);
    assert!(actions.pressed(jump, 0), "the tap fires its press");
    assert!(actions.pressed(fire, 0), "and so does a mouse tap");

    frame(&mut actions, &mut input);
    assert!(actions.released(jump, 0), "and lets go the next frame");
    assert!(actions.released(fire, 0));
    assert!(!actions.held(jump, 0));
}

#[test]
fn rebinding_while_a_key_is_held_fires_no_edge_either_way() {
    let (mut actions, mut input) = loaded(
        r#"
        format_version = 1
        [actions]
        Jump = ["Space"]
        "#,
    );
    let jump = actions.id("Jump");

    input.press(code("Space"));
    frame(&mut actions, &mut input);
    assert!(actions.pressed(jump, 0));

    input.press(code("Return"));
    actions.apply(specs(
        r#"
        format_version = 1
        [actions]
        Jump = ["Return"]
        "#,
    ));
    frame(&mut actions, &mut input);
    assert!(actions.held(jump, 0));
    assert!(
        !actions.pressed(jump, 0),
        "the swap is not a press the player made"
    );
    assert!(!actions.released(jump, 0), "nor a release");
}

#[test]
fn a_composed_axis_reads_minus_one_zero_or_one() {
    let (mut actions, mut input) = loaded(
        r#"
        format_version = 1
        [actions]
        Left = ["A"]
        Right = ["D"]
        [axes.Horizontal]
        positive = "Right"
        negative = "Left"
        "#,
    );
    let horizontal = actions.id("Horizontal");
    assert_eq!(actions.axis(horizontal, 0), 0.0);

    input.press(code("D"));
    frame(&mut actions, &mut input);
    assert_eq!(actions.axis(horizontal, 0), 1.0);

    input.press(code("A"));
    frame(&mut actions, &mut input);
    assert_eq!(actions.axis(horizontal, 0), 0.0, "both ways is neither");

    input.release(code("D"));
    frame(&mut actions, &mut input);
    assert_eq!(actions.axis(horizontal, 0), -1.0);
}

#[test]
fn an_analog_axis_is_deadzoned_and_rescaled_to_full_travel() {
    let (mut actions, mut input) = loaded(
        r#"
        format_version = 1
        [axes.LookX]
        source = "Gamepad.RightStick.X"
        deadzone = 0.5
        "#,
    );
    let look = actions.id("LookX");
    input.set_pad_connected(0, true);

    for (raw, expected) in [
        (0.4, 0.0),
        (-0.5, 0.0),
        (0.75, 0.5),
        (1.0, 1.0),
        (-1.0, -1.0),
    ] {
        input.set_pad_axis(0, PadAxis::RightStickX, raw);
        frame(&mut actions, &mut input);
        let value = actions.axis(look, 0);
        assert!(
            (value - expected).abs() < 1e-6,
            "{raw} past a 0.5 deadzone should read {expected}, read {value}"
        );
    }
}

#[test]
fn an_unbounded_source_is_scaled_but_never_clamped() {
    let (mut actions, mut input) = loaded(
        r#"
        format_version = 1
        [axes.LookX]
        source = "Mouse.X"
        scale = 0.5
        "#,
    );
    let look = actions.id("LookX");

    input.on_device_event(&DeviceEvent::MouseMotion { delta: (40.0, 0.0) });
    actions.resolve(&input);
    assert_eq!(
        actions.axis(look, 0),
        20.0,
        "a flick is worth more than one unit of turn"
    );
}

#[test]
fn an_analog_source_past_the_threshold_reads_as_a_button() {
    let (mut actions, mut input) = loaded(
        r#"
        format_version = 1
        [actions]
        Fire = ["Gamepad.RightTrigger"]
        "#,
    );
    let fire = actions.id("Fire");
    input.set_pad_connected(0, true);

    input.set_pad_axis(0, PadAxis::RightTrigger, 0.2);
    frame(&mut actions, &mut input);
    assert!(!actions.held(fire, 0));

    input.set_pad_axis(0, PadAxis::RightTrigger, 0.9);
    frame(&mut actions, &mut input);
    assert!(actions.held(fire, 0));
    assert!(actions.pressed(fire, 0));
}

// --- Players ---------------------------------------------------------------

#[test]
fn the_keyboard_and_mouse_answer_for_player_zero_only() {
    let (mut actions, mut input) = loaded(
        r#"
        format_version = 1
        [actions]
        Jump = ["Space", "Gamepad.South"]
        "#,
    );
    let jump = actions.id("Jump");

    input.press(code("Space"));
    frame(&mut actions, &mut input);
    assert!(actions.held(jump, 0));
    assert!(
        !actions.held(jump, 1),
        "player two does not jump on player one's spacebar"
    );
}

#[test]
fn a_pad_binding_answers_for_its_own_slot() {
    let (mut actions, mut input) = loaded(
        r#"
        format_version = 1
        [actions]
        Jump = ["Gamepad.South"]
        "#,
    );
    let jump = actions.id("Jump");

    input.set_pad_connected(1, true);
    input.set_pad_button(1, PadButton::South, true);
    frame(&mut actions, &mut input);
    assert!(actions.held(jump, 1));
    assert!(!actions.held(jump, 0));
}

#[test]
fn a_player_past_the_last_slot_reads_as_quiet_rather_than_panicking() {
    let (mut actions, mut input) = loaded(
        r#"
        format_version = 1
        [actions]
        Jump = ["Gamepad.South"]
        [axes.LookX]
        source = "Gamepad.RightStick.X"
        "#,
    );
    let jump = actions.id("Jump");
    let look = actions.id("LookX");
    input.set_pad_connected(0, true);
    input.set_pad_button(0, PadButton::South, true);
    frame(&mut actions, &mut input);

    assert!(!actions.held(jump, 99));
    assert_eq!(actions.axis(look, 99), 0.0);
}

// --- Names that name nothing -----------------------------------------------

#[test]
fn an_id_for_a_name_nothing_binds_warns_once_and_reads_as_unpressed() {
    let (mut actions, mut input) = loaded("format_version = 1");
    let ghost = actions.id("Jump");

    let warnings = actions.take_warnings();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("Jump"), "{warnings:?}");

    assert!(actions.id("Jump") == ghost, "the id is stable");
    assert!(
        actions.take_warnings().is_empty(),
        "asking twice is not two mistakes"
    );

    frame(&mut actions, &mut input);
    assert!(!actions.held(ghost, 0));
    assert_eq!(actions.axis(ghost, 0), 0.0);
}

#[test]
fn a_reload_that_unbinds_a_name_a_script_uses_warns() {
    // Renaming an action in the file and forgetting the script is the mistake
    // live reload invites, and the interning warning alone cannot catch it:
    // the name was bound the first time it was asked for.
    let (mut actions, _) = loaded(
        r#"
        format_version = 1
        [actions]
        Jump = ["Space"]
        "#,
    );
    let _ = actions.id("Jump");
    assert!(actions.take_warnings().is_empty());

    actions.apply(specs(
        r#"
        format_version = 1
        [actions]
        Leap = ["Space"]
        "#,
    ));
    let warnings = actions.take_warnings();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("Jump"), "{warnings:?}");
}

#[test]
fn a_name_nobody_asked_for_is_not_warned_about_on_load() {
    let (mut actions, _) = loaded(
        r#"
        format_version = 1
        [actions]
        Jump = ["Space"]
        "#,
    );
    assert!(
        actions.take_warnings().is_empty(),
        "a config nobody has queried yet is not a mistake"
    );
}

#[test]
fn an_action_id_reads_as_an_empty_axis_and_the_reverse() {
    let (mut actions, mut input) = loaded(
        r#"
        format_version = 1
        [actions]
        Jump = ["Space"]
        [axes.LookX]
        source = "Gamepad.RightStick.X"
        "#,
    );
    let jump = actions.id("Jump");
    let look = actions.id("LookX");

    input.press(code("Space"));
    input.set_pad_connected(0, true);
    input.set_pad_axis(0, PadAxis::RightStickX, 1.0);
    frame(&mut actions, &mut input);

    assert_eq!(actions.axis(jump, 0), 0.0, "an action has no axis value");
    assert!(!actions.held(look, 0), "an axis is never held");
}

// --- Device state ----------------------------------------------------------

#[test]
fn key_repeat_from_the_os_is_not_a_second_press() {
    let mut input = InputState::new();
    input.press(code("Space"));
    assert!(input.key_pressed(code("Space")));

    input.end_frame();
    input.press(code("Space"));
    assert!(input.key_down(code("Space")));
    assert!(
        !input.key_pressed(code("Space")),
        "a repeat is the same press arriving again"
    );
}

#[test]
fn losing_focus_releases_everything_that_was_down() {
    let (mut actions, mut input) = loaded(
        r#"
        format_version = 1
        [actions]
        Jump = ["Space"]
        Fire = ["Mouse.Left"]
        "#,
    );
    let jump = actions.id("Jump");
    let fire = actions.id("Fire");

    input.press(code("Space"));
    input.press_mouse_button(0);
    frame(&mut actions, &mut input);
    assert!(actions.held(jump, 0) && actions.held(fire, 0));

    input.on_window_event(&WindowEvent::Focused(false), false);
    assert!(!input.focused());
    assert!(
        input.key_released(code("Space")),
        "the release is recorded, not dropped — a key held across an alt-tab \
         is let go exactly once"
    );
    assert!(input.mouse_button_released(0));

    frame(&mut actions, &mut input);
    assert!(actions.released(jump, 0));
    assert!(actions.released(fire, 0));
    assert!(!actions.held(jump, 0) && !actions.held(fire, 0));
}

#[test]
fn mouse_motion_accumulates_over_a_frame_and_clears_after_it() {
    let mut input = InputState::new();
    input.on_device_event(&DeviceEvent::MouseMotion { delta: (3.0, -1.0) });
    input.on_device_event(&DeviceEvent::MouseMotion { delta: (2.0, 4.0) });
    assert_eq!(input.mouse_delta(), (5.0, 3.0));

    input.end_frame();
    assert_eq!(input.mouse_delta(), (0.0, 0.0));
}

#[test]
fn detaching_a_pad_zeroes_it_rather_than_leaving_a_stick_pushed() {
    let mut input = InputState::new();
    input.set_pad_connected(0, true);
    input.set_pad_button(0, PadButton::South, true);
    input.set_pad_axis(0, PadAxis::LeftStickX, 1.0);

    input.set_pad_connected(0, false);
    assert!(!input.pad_connected(0));
    assert!(!input.pad_button_down(0, PadButton::South));
    assert_eq!(input.pad_axis(0, PadAxis::LeftStickX), 0.0);
}

/// The wheel is an edge like a press: what turned this frame, summed, and gone
/// by the next.
#[test]
fn the_wheel_sums_a_frame_and_then_reads_still() {
    let (mut actions, mut input) = loaded("format_version = 1\n[actions]\n");
    input.scroll_wheel((0.0, 1.0));
    input.scroll_wheel((0.5, -3.0));
    assert_eq!(input.wheel(), (0.5, -2.0));
    frame(&mut actions, &mut input);
    assert_eq!(input.wheel(), (0.0, 0.0));
}
