//! Layout: the tools around the scene keep their size in points whatever the
//! window does, and the Environment tool's sky row.

use super::*;
use crate::editor::prefs::{Prefs, PrefsFile};
use crate::scene::{EnvironmentSettings, LogBuffer, LogLevel};

/// The design system's tool sizes: hierarchy, inspector, bottom dock.
const DESIGNED: (f32, f32, f32) = (240.0, 320.0, 220.0);

fn resize(editor: &mut Harness, size: egui::Vec2) -> &mut Harness {
    editor.input.screen_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size));
    editor.frames(3)
}

/// Hierarchy width, inspector width, bottom dock height.
fn tools(editor: &Harness) -> (f32, f32, f32) {
    (
        editor.tab(Tab::Hierarchy).width(),
        editor.tab(Tab::Inspector).width(),
        editor.tab(Tab::Console).height(),
    )
}

fn assert_near(actual: (f32, f32, f32), expected: (f32, f32, f32), slack: f32, at: &str) {
    for (what, a, e) in [
        ("hierarchy", actual.0, expected.0),
        ("inspector", actual.1, expected.1),
        ("bottom dock", actual.2, expected.2),
    ] {
        assert!(
            (a - e).abs() <= slack,
            "at {at} the {what} is {a}, expected {e}"
        );
    }
}

/// The designed sizes are sizes, not shares of a 1280-point window: the
/// tools are as wide on a large display as on the window they were drawn
/// against, and the scene has the rest.
#[test]
fn the_tools_keep_their_designed_size_at_any_window_size() {
    for size in [
        egui::vec2(1280.0, 800.0),
        egui::vec2(1920.0, 1080.0),
        egui::vec2(2560.0, 1440.0),
        egui::vec2(3440.0, 1369.0),
    ] {
        let mut editor = Harness::new(size);
        editor.frames(3);
        assert_near(tools(&editor), DESIGNED, 2.0, &format!("{size:?}"));
    }
}

/// A separator the user drags leaves a size, and that size is the one the
/// next resize keeps.
#[test]
fn a_dragged_separator_keeps_its_size_through_a_resize() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.frames(3);
    let from = editor.left_separator();
    editor.drag(from, from + egui::vec2(100.0, 0.0));
    let from = editor.bottom_separator();
    editor.drag(from, from - egui::vec2(0.0, 80.0));
    let dragged = tools(&editor);
    assert!(dragged.0 > 300.0, "the drag moved it to {}", dragged.0);
    assert!(dragged.2 > 280.0, "the drag moved it to {}", dragged.2);

    resize(&mut editor, egui::vec2(2560.0, 1440.0));
    assert_near(tools(&editor), dragged, 1.0, "2560x1440");
    // Room for 340 + 320 points with the scene still at its share.
    resize(&mut editor, egui::vec2(1200.0, 800.0));
    assert_near(tools(&editor), dragged, 1.0, "1200x800");
}

/// A layout saved in one session comes back with its tools the same size in
/// points in a window of another size. The file is written as every editor
/// since layout version 2 writes it, so one saved before today loads too.
#[test]
fn a_saved_layout_comes_back_at_its_sizes_in_another_window() {
    let mut first = Harness::new(egui::vec2(1280.0, 800.0));
    first.frames(3);
    let from = first.left_separator();
    first.drag(from, from + egui::vec2(60.0, 0.0));
    let saved = tools(&first);
    assert!(saved.0 > 280.0, "the drag moved it to {}", saved.0);

    let dir = crate::asset_db::tests::Dir::new("layout-other-window");
    let layout = ron::to_string(first.dock.state()).expect("serialises");
    dir.write(
        "editor.ron",
        format!("(layout: Some({layout}), layout_version: 2)").as_bytes(),
    );
    let prefs = PrefsFile::new(Some(dir.path().to_owned())).load();
    assert!(prefs.layout.is_some(), "a version 2 layout is kept");

    for size in [egui::vec2(2560.0, 1440.0), egui::vec2(1100.0, 700.0)] {
        let mut second = Harness::new(size);
        second.dock = Dock::from_saved(prefs.layout.clone());
        second.frames(3);
        assert_near(tools(&second), saved, 1.0, &format!("{size:?}"));
    }

    // And what the second session saves is read the same way again.
    let mut second = Harness::new(egui::vec2(2560.0, 1440.0));
    second.dock = Dock::from_saved(prefs.layout);
    second.frames(3);
    let file = PrefsFile::new(Some(dir.path().to_owned()));
    let mut prefs = Prefs::default();
    prefs.layout = Some(second.dock.state().clone());
    file.save(&prefs);
    let mut third = Harness::new(egui::vec2(1280.0, 800.0));
    third.dock = Dock::from_saved(file.load().layout);
    third.frames(3);
    assert_near(tools(&third), saved, 1.0, "back at 1280x800");
}

/// The Environment tab's body alone, `width` wide, drawn as the editor's
/// frame: on a screen tall enough that none of it is culled, and with the
/// harness's input, so clicks and hovers reach it.
fn environment_at(editor: &mut Harness, width: f32) -> &mut Harness {
    editor.input.screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(width + 40.0, 4000.0),
    ));
    let Harness {
        world,
        state,
        ctx,
        input,
        output,
        ..
    } = editor;
    *output = Some(ctx.run(input.clone(), |ctx| {
        egui::Area::new("environment alone".into())
            .fixed_pos(egui::pos2(10.0, 10.0))
            .fade_in(false)
            .show(ctx, |ui| {
                ui.allocate_ui(egui::vec2(width, 3900.0), |ui| {
                    ui.set_width(width);
                    environment::body(ui, world, state);
                });
            });
    }));
    editor
}

fn environment_frames(editor: &mut Harness, width: f32, count: usize) -> &mut Harness {
    for _ in 0..count {
        environment_at(editor, width);
    }
    editor
}

/// One frame of the Environment tab with `events` delivered to it, and two
/// more for what they opened to draw.
fn environment_input(editor: &mut Harness, width: f32, events: Vec<egui::Event>) {
    editor.input.events = events;
    environment_at(editor, width);
    editor.input.events.clear();
    environment_frames(editor, width, 2);
}

fn press(at: egui::Pos2, button: egui::PointerButton) -> Vec<egui::Event> {
    let mut events = vec![egui::Event::PointerMoved(at)];
    for pressed in [true, false] {
        events.push(egui::Event::PointerButton {
            pos: at,
            button,
            pressed,
            modifiers: Default::default(),
        });
    }
    events
}

/// Where the text reading exactly `whole` was painted.
fn painted_at(editor: &Harness, whole: &str) -> Option<egui::Rect> {
    editor
        .painted_text_rects()
        .into_iter()
        .find(|(text, _)| without_icons(text) == whole)
        .map(|(_, rect)| rect)
}

/// Where the HDRI row's file is drawn, beside its label.
fn hdri_value(editor: &Harness, shows: &str) -> egui::Pos2 {
    let row = painted_at(editor, "HDRI").expect("the HDRI row").center().y;
    editor
        .painted_text_rects()
        .into_iter()
        .find(|(text, rect)| without_icons(text) == shows && (rect.center().y - row).abs() < 8.0)
        .unwrap_or_else(|| panic!("no {shows:?} beside HDRI: {:?}", editor.painted()))
        .1
        .center()
}

fn sky(editor: &Harness) -> EnvironmentSettings {
    editor.world.resource::<EnvironmentSettings>().clone()
}

/// A click at `at`, `file` picked in the system's dialog it opens, and the
/// frames until the pick has landed.
fn pick_hdri(editor: &mut Harness, at: egui::Pos2, file: std::path::PathBuf) {
    *environment::HDRI_PICKED.lock().unwrap() = Some(Some(file));
    environment_input(editor, 360.0, press(at, egui::PointerButton::Primary));
    for _ in 0..200 {
        if !environment::picking(&editor.ctx) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
        environment_at(editor, 360.0);
    }
    environment_frames(editor, 360.0, 2);
}

/// The sky is a file in the project, so it is picked as one — in the
/// system's dialog, since the asset database does not list `.hdr` images —
/// and shown, not typed. A pick bakes it; Clear names none; a file outside
/// the assets folder is refused in the console, by name.
#[test]
fn the_sky_is_picked_shown_read_only_and_cleared() {
    let dir = crate::asset_db::tests::Dir::new("environment-hdri");
    dir.write("sky/meadow.hdr", b"#?RADIANCE");
    let outside = crate::asset_db::tests::Dir::new("environment-hdri-outside");
    let stray = outside.write("dusk.hdr", b"#?RADIANCE");
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let db = crate::asset_db::tests::open(&dir);
    let meadow = db.dir().join("sky").join("meadow.hdr");
    editor.world.insert_resource(db);
    environment_frames(&mut editor, 360.0, 2);
    assert!(
        !editor.painted_contains("relative to assets/"),
        "nothing to type into"
    );

    let empty = hdri_value(&editor, "None");
    pick_hdri(&mut editor, empty, meadow);
    assert_eq!(sky(&editor).hdri, "sky/meadow.hdr");
    assert!(sky(&editor).reload_requested, "a pick bakes it");
    editor
        .world
        .resource_mut::<EnvironmentSettings>()
        .reload_requested = false;
    let shown = hdri_value(&editor, "sky/meadow.hdr");

    pick_hdri(&mut editor, shown, stray);
    assert_eq!(
        sky(&editor).hdri,
        "sky/meadow.hdr",
        "a refused pick changes nothing"
    );
    assert!(!sky(&editor).reload_requested);
    let refusal = editor
        .world
        .resource::<LogBuffer>()
        .iter()
        .find(|entry| entry.level == LogLevel::Error)
        .map(|entry| entry.message.clone())
        .expect("the refusal is in the console");
    assert!(
        refusal.contains("dusk.hdr") && refusal.contains("assets"),
        "{refusal}"
    );

    environment_input(
        &mut editor,
        360.0,
        press(shown, egui::PointerButton::Secondary),
    );
    let clear = painted_at(&editor, "Clear")
        .expect("right-click ▸ Clear")
        .center();
    environment_input(
        &mut editor,
        360.0,
        press(clear, egui::PointerButton::Primary),
    );
    assert_eq!(sky(&editor).hdri, "");
    hdri_value(&editor, "None");
}
