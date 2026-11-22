//! Feedback: log rows that can be selected and copied without taking the
//! scene's selection with them, and a log that starts empty with the game.

use super::*;
use crate::scene::{LogBuffer, LogLevel};

fn log(editor: &mut Harness, level: LogLevel, message: &str) {
    editor
        .world
        .resource_mut::<LogBuffer>()
        .push(level, message.to_owned(), 0);
}

fn text_rect(editor: &Harness, whole: &str) -> egui::Rect {
    editor
        .painted_text_rects()
        .into_iter()
        .find(|(text, _)| text == whole)
        .unwrap_or_else(|| panic!("`{whole}` was not painted: {:?}", editor.painted()))
        .1
}

fn copies(editor: &Harness) -> Vec<String> {
    editor
        .output
        .as_ref()
        .expect("a frame has run")
        .platform_output
        .commands
        .iter()
        .filter_map(|command| match command {
            egui::OutputCommand::CopyText(text) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

/// A row's own copy, not the scene's: the key goes to the console while the
/// pointer is over it or it was the last thing clicked, and back to the
/// selection once something else is.
#[test]
fn the_copy_key_after_a_click_in_the_console_copies_its_rows_and_not_the_selection() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let chair = editor.spawn("Chair");
    editor.state.selection.set(chair);
    log(&mut editor, LogLevel::Info, "loaded level");
    editor.frames(2);

    editor.click(text_rect(&editor, "loaded level").center());
    editor.input.events.push(egui::Event::Copy);
    editor.frames(1);
    editor.input.events.clear();
    editor.state.apply(&mut editor.world, &editor.registry);
    assert_eq!(copies(&editor), ["INFO #0\nloaded level"]);
    assert!(
        !editor.state.can_paste(),
        "the scene's selection was not copied"
    );

    editor.click(text_rect(&editor, "Chair").center());
    editor.input.events.push(egui::Event::Copy);
    editor.frames(1);
    editor.input.events.clear();
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(
        copies(&editor).iter().any(|text| text.contains("Chair")),
        "{:?}",
        copies(&editor)
    );
    assert!(editor.state.can_paste());
}

#[test]
fn clear_on_play_empties_the_log_before_the_game_starts() {
    let had = |editor: &Harness| {
        editor
            .world
            .resource::<LogBuffer>()
            .iter()
            .any(|entry| entry.message == "from edit mode")
    };
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    log(&mut editor, LogLevel::Info, "from edit mode");
    editor.state.request_play();
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(editor.state.playing());
    assert!(had(&editor), "off by default");

    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.state.console.options.clear_on_play = true;
    log(&mut editor, LogLevel::Info, "from edit mode");
    editor.state.request_play();
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(editor.state.playing());
    assert!(!had(&editor));
}
