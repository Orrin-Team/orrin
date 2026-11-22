//! The Settings tab, driven through the whole editor as the other panel tests
//! are: its entry points, its two scopes, and the manifest it writes.

use super::*;
use crate::asset_db::tests::Dir;
use crate::editor::panels::settings::Page;
use crate::editor::settings::FlyCamera;
use crate::editor::state::DockRequest;

const MANIFEST: &str = r#"# The game's manifest. Edited by hand and by the editor.
format_version = 1

[project]
name = "hello-orrin" # shown in the top bar
version = "0.1.0"

[scripts]
dir = "scripts"
entry = "HelloOrrin.Main, HelloOrrin"

[assets]
dir = "assets"
"#;

/// A project on disk with two scenes under its assets folder, open in the
/// editor with its asset database.
pub(super) fn project(name: &str) -> (Dir, Harness) {
    let dir = Dir::new(name);
    dir.write("orrin.toml", MANIFEST.as_bytes());
    dir.write("assets/scenes/main.orrin", b"");
    dir.write("assets/levels/one.orrin", b"");
    let root = std::fs::canonicalize(dir.path()).unwrap();
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let (db, _) = crate::asset_db::AssetDatabase::open(root.join("assets"), root.join(".cache"));
    editor.world.insert_resource(db);
    editor.state.project_root = Some(root);
    editor.state.project_name = Some("hello-orrin".to_owned());
    (dir, editor)
}

fn manifest(dir: &Dir) -> String {
    std::fs::read_to_string(dir.path().join("orrin.toml")).unwrap()
}

pub(super) fn open(editor: &mut Harness, page: Option<Page>) {
    editor.state.open_settings(page);
    editor.frames(3);
}

/// Every place `needle` was painted inside the Settings tab, in paint order:
/// a popup over the tab comes last.
pub(super) fn in_settings(editor: &Harness, needle: &str) -> Vec<egui::Pos2> {
    let tab = editor.tab(Tab::Settings);
    editor
        .painted_text_rects()
        .into_iter()
        .filter(|(text, rect)| text == needle && tab.contains(rect.center()))
        .map(|(_, rect)| rect.center())
        .collect()
}

fn type_text(editor: &mut Harness, text: &str) {
    editor.input.events.push(egui::Event::Text(text.to_owned()));
    editor.frames(1);
    editor.input.events.clear();
}

/// Click into the field showing `current`, the right-most one (the page sits
/// right of the sidebar), replace what it holds and press Enter.
fn retype(editor: &mut Harness, current: &str, text: &str) {
    let at = *in_settings(editor, current)
        .iter()
        .max_by(|a, b| a.x.total_cmp(&b.x))
        .unwrap_or_else(|| panic!("{current} is painted in the page"));
    editor.click(at).frames(1);
    editor.key_with(egui::Modifiers::COMMAND, egui::Key::A);
    type_text(editor, text);
    editor.key(egui::Key::Enter).frames(2);
}

fn settings_tabs(editor: &Harness) -> usize {
    editor
        .dock
        .state()
        .iter_all_tabs()
        .filter(|(_, tab)| **tab == Tab::Settings)
        .count()
}

fn is_front(editor: &Harness, tab: Tab) -> bool {
    let (surface, node, index) = editor.dock.state().find_tab(&tab).expect("open");
    match &editor.dock.state()[surface][node] {
        egui_dock::Node::Leaf { active, .. } => *active == index,
        _ => false,
    }
}

/// Ctrl+, opens Settings over the scene, in a window of its designed size,
/// not in the dock; pressed again it brings the one that is open forward,
/// wherever the user has put it, rather than opening another.
#[test]
fn ctrl_comma_opens_settings_in_a_window_and_again_brings_it_forward() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.frames(2);
    assert!(
        !editor.dock.is_open(Tab::Settings),
        "not in the default layout"
    );

    editor.key_with(egui::Modifiers::COMMAND, egui::Key::Comma);
    editor.frames(3);
    assert!(editor.dock.is_floating(Tab::Settings));
    assert_eq!(settings_tabs(&editor), 1);
    let window = editor.tab(Tab::Settings);
    assert!(
        (window.width() - 760.0).abs() <= 24.0 && (window.height() - 520.0).abs() <= 48.0,
        "{window:?}"
    );
    let middle = egui::pos2(640.0, 400.0);
    assert!(
        (window.center() - middle).length() < 40.0,
        "centred on the window: {window:?}"
    );
    assert!(editor.painted_contains("Appearance"), "the first time");

    // Docked by the user, behind another tool.
    editor.dock.apply(DockRequest::Dock(Tab::Settings));
    let (surface, node, _) = editor.dock.state().find_tab(&Tab::Settings).unwrap();
    editor.frames(2);
    let neighbour = Tab::ALL
        .into_iter()
        .find(|tab| {
            editor
                .dock
                .state()
                .find_tab(tab)
                .is_some_and(|(s, n, _)| (s, n) == (surface, node))
        })
        .expect("docked beside a tool");
    editor.dock.activate(neighbour);
    editor.frames(2);
    assert!(!is_front(&editor, Tab::Settings));

    editor.key_with(egui::Modifiers::COMMAND, egui::Key::Comma);
    editor.frames(2);
    assert_eq!(settings_tabs(&editor), 1);
    assert!(
        !editor.dock.is_floating(Tab::Settings),
        "left where it was put"
    );
    assert!(is_front(&editor, Tab::Settings));
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

/// A setting away from its default is marked and can be put back; one at
/// its default offers nothing to undo.
#[test]
fn a_changed_setting_offers_a_reset_that_restores_the_default() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.state.settings.viewport.fly_speed = 12.0;
    open(&mut editor, Some(Page::Viewport));

    let row = in_settings(&editor, "Fly speed")[0];
    let resets = reset_glyphs(&editor);
    assert_eq!(resets.len(), 1, "{resets:?}");
    assert!((resets[0].center().y - row.y).abs() < 8.0);
    let dots =
        editor.painted_rects_of(|shape| matches!(shape, egui::Shape::Circle(c) if c.radius <= 3.0));
    assert!(
        dots.iter()
            .any(|dot| dot.center().x < row.x && (dot.center().y - row.y).abs() < 6.0),
        "the changed marker before the label: {dots:?}"
    );

    editor.click(resets[0].center()).frames(2);
    assert_eq!(
        editor.state.settings.viewport.fly_speed,
        FlyCamera::default().fly_speed
    );
    assert!(reset_glyphs(&editor).is_empty());
}

/// A UI that rescales under the pointer mid-drag cannot be dragged, so the
/// scale is held by the tab until the pointer lets go.
#[test]
fn the_ui_scale_applies_when_the_pointer_lets_go() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    open(&mut editor, Some(Page::Appearance));
    let value = editor
        .painted_text_rects()
        .into_iter()
        .find(|(text, _)| text == "1.00×")
        .expect("the scale's value")
        .1;
    let at = egui::pos2(value.left() - 24.0, value.center().y);

    editor.input.events.push(egui::Event::PointerMoved(at));
    editor.input.events.push(egui::Event::PointerButton {
        pos: at,
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: Default::default(),
    });
    editor.frames(1);
    editor.input.events.clear();
    editor
        .input
        .events
        .push(egui::Event::PointerMoved(at + egui::vec2(-12.0, 0.0)));
    editor.frames(2);
    editor.input.events.clear();
    assert_eq!(
        editor.state.settings.appearance.ui_scale, 1.0,
        "not mid-drag"
    );

    editor.input.events.push(egui::Event::PointerButton {
        pos: at + egui::vec2(-12.0, 0.0),
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers: Default::default(),
    });
    editor.frames(2);
    editor.input.events.clear();
    let scale = editor.state.settings.appearance.ui_scale;
    assert!(scale > 1.0, "applied on release: {scale}");
}

#[test]
fn renaming_the_project_writes_orrin_toml_and_the_top_bar() {
    let (dir, mut editor) = project("settings-rename");
    open(&mut editor, Some(Page::General));
    assert!(
        !in_settings(&editor, "Version").is_empty(),
        "the General page"
    );

    retype(&mut editor, "hello-orrin", "space-shooter");
    assert_eq!(
        manifest(&dir),
        MANIFEST.replace(r#""hello-orrin""#, r#""space-shooter""#),
        "the value changed and nothing else, comments included"
    );
    assert_eq!(editor.state.project_name.as_deref(), Some("space-shooter"));
}

#[test]
fn an_entry_that_names_no_assembly_is_refused_and_writes_nothing() {
    let (dir, mut editor) = project("settings-entry");
    open(&mut editor, Some(Page::Scripts));

    retype(
        &mut editor,
        "HelloOrrin.Main, HelloOrrin",
        "HelloOrrin.Main",
    );
    assert_eq!(manifest(&dir), MANIFEST);
    assert!(
        editor.painted_contains("names no assembly"),
        "the error's own words"
    );
    assert!(
        !in_settings(&editor, "HelloOrrin.Main, HelloOrrin").is_empty(),
        "the field shows what is on disk again"
    );
}

/// A scene is picked from the project's scenes, never typed, and is written
/// relative to the project root — not to the assets folder it was listed
/// from — with forward slashes. "None" takes it out again.
#[test]
fn a_start_scene_is_picked_from_the_projects_scenes_and_none_removes_it() {
    let (dir, mut editor) = project("settings-start-scene");
    open(&mut editor, Some(Page::Scenes));

    let start = in_settings(&editor, "None")
        .into_iter()
        .min_by(|a, b| a.y.total_cmp(&b.y))
        .expect("the start scene's picker");
    editor.click(start).frames(2);
    let one = *in_settings(&editor, "assets/levels/one.orrin")
        .last()
        .expect("the picker lists the project's scenes");
    editor.click(one).frames(2);
    let text = manifest(&dir);
    assert!(
        text.ends_with("\n[scenes]\nstart = \"assets/levels/one.orrin\"\n"),
        "{text}"
    );

    let picker = in_settings(&editor, "assets/levels/one.orrin")[0];
    editor.click(picker).frames(2);
    let none = *in_settings(&editor, "None").last().unwrap();
    editor.click(none).frames(2);
    assert_eq!(manifest(&dir), MANIFEST);
}

/// Somebody hand-editing `orrin.toml` gets the loader's error and a way to
/// the file instead of fields that could not be saved; fixed, the fields
/// come back.
#[test]
fn a_manifest_broken_on_disk_shows_its_error_until_it_is_fixed() {
    let (dir, mut editor) = project("settings-broken");
    open(&mut editor, Some(Page::General));
    assert!(!in_settings(&editor, "Name").is_empty());

    dir.write("orrin.toml", b"format_version = 1\n[project\n");
    editor.state.settings_panel.recheck_now();
    editor.frames(2);
    assert!(editor.painted_contains("could not parse"));
    assert!(!in_settings(&editor, "Open orrin.toml").is_empty());
    assert!(in_settings(&editor, "Name").is_empty());

    dir.write("orrin.toml", MANIFEST.as_bytes());
    editor.state.settings_panel.recheck_now();
    editor.frames(2);
    assert!(!in_settings(&editor, "Name").is_empty());
    assert!(in_settings(&editor, "Open orrin.toml").is_empty());
}

/// Pick `option` from the combo box showing `current` on the row `label`.
fn pick(editor: &mut Harness, label: &str, current: &str, option: &str) {
    let row = in_settings(editor, label)[0];
    let combo = *in_settings(editor, current)
        .iter()
        .find(|at| (at.y - row.y).abs() < 6.0)
        .unwrap_or_else(|| panic!("{label}'s picker shows {current}"));
    editor.click(combo).frames(2);
    let choice = *in_settings(editor, option)
        .last()
        .unwrap_or_else(|| panic!("{label}'s picker offers {option}"));
    editor.click(choice).frames(2);
}

fn restart_line(editor: &Harness) -> Option<String> {
    editor
        .painted()
        .into_iter()
        .find(|text| text.starts_with("Restart Orrin to apply"))
}

/// A change waits for a restart, says so once at the top, and taking it
/// back takes the line away.
#[test]
fn a_changed_startup_row_asks_for_a_restart_until_it_is_put_back() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    open(&mut editor, Some(Page::Startup));

    pick(&mut editor, "Log level", "Default", "Debug");
    assert_eq!(
        editor.state.settings.startup.log,
        crate::editor::settings::LogFloor::Debug
    );
    assert_eq!(
        restart_line(&editor).as_deref(),
        Some("Restart Orrin to apply 1 change")
    );
    let line = in_settings(&editor, "Restart Orrin to apply 1 change")[0];
    let first = in_settings(&editor, "Graphics backend")[0];
    assert!(line.y < first.y, "at the top of the page");
    assert!(!in_settings(&editor, "Restart now").is_empty());

    editor.state.settings.startup.validation = crate::editor::settings::Validation::Off;
    editor.frames(2);
    assert_eq!(
        restart_line(&editor).as_deref(),
        Some("Restart Orrin to apply 2 changes")
    );

    editor.state.settings.startup.validation = crate::editor::settings::Validation::Auto;
    editor.frames(1);
    pick(&mut editor, "Log level", "Debug", "Default");
    assert!(restart_line(&editor).is_none());
}

/// Restarting over unsaved work saves it first, and only a save that
/// happened lets the restart go ahead.
#[test]
fn with_unsaved_work_the_button_saves_and_then_restarts() {
    let dir = Dir::new("settings-save-and-restart");
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let scene = dir.path().join("level.orrin");
    editor.state.scene_path = scene.to_string_lossy().into_owned();
    editor.state.settings.startup.threads = Some(2);
    editor
        .state
        .request_spawn(crate::editor::state::SpawnKind::Folder);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(editor.state.scene_dirty());
    open(&mut editor, Some(Page::Startup));
    assert!(in_settings(&editor, "Restart now").is_empty());

    let button = in_settings(&editor, "Save and restart")[0];
    editor.click(button).frames(1);
    assert!(!editor.state.take_restart_request(), "not before the save");
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(scene.is_file(), "saved");
    assert!(!editor.state.scene_dirty());
    assert!(editor.state.take_restart_request());
}

#[test]
fn a_save_that_fails_cancels_the_restart_and_says_so() {
    let dir = Dir::new("settings-save-fails");
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.state.scene_path = dir
        .path()
        .join("missing/level.orrin")
        .to_string_lossy()
        .into_owned();
    editor.state.settings.startup.threads = Some(2);
    editor
        .state
        .request_spawn(crate::editor::state::SpawnKind::Folder);
    editor.state.apply(&mut editor.world, &editor.registry);
    open(&mut editor, Some(Page::Startup));

    let button = in_settings(&editor, "Save and restart")[0];
    editor.click(button).frames(1);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(!editor.state.take_restart_request());
    assert!(editor.state.scene_dirty());
    let log: Vec<String> = editor
        .world
        .resource::<crate::scene::LogBuffer>()
        .iter()
        .map(|entry| entry.message.clone())
        .collect();
    assert!(
        log.iter().any(|line| line.contains("restart cancelled")),
        "{log:?}"
    );
}

/// The strip above `tab`'s body: the tabs of the leaf that holds it.
fn strip_of(editor: &Harness, tab: Tab) -> egui::Rect {
    let leaf = editor.tab(tab);
    let body = editor.tab_body(tab);
    egui::Rect::from_min_max(leaf.min, egui::pos2(leaf.max.x, body.min.y))
}

/// The middle of the ✕ drawn in `tab`'s strip, which is two diagonal strokes
/// and the only diagonals a strip holds.
fn close_button(editor: &Harness, tab: Tab) -> Option<egui::Pos2> {
    let strip = strip_of(editor, tab);
    editor
        .painted_rects_of(|shape| matches!(shape, egui::Shape::LineSegment { .. }))
        .into_iter()
        .filter(|rect| rect.width() > 2.0 && rect.height() > 2.0)
        .filter(|rect| strip.contains(rect.center()))
        .reduce(|a, b| a.union(b))
        .map(|rect| rect.center())
}

/// A field commits on losing focus, and closing the tab is losing focus:
/// what was typed is not dropped with the window.
#[test]
fn closing_settings_keeps_what_was_typed_into_a_project_field() {
    let (dir, mut editor) = project("settings-close-typed");
    open(&mut editor, Some(Page::General));
    let at = *in_settings(&editor, "hello-orrin")
        .iter()
        .max_by(|a, b| a.x.total_cmp(&b.x))
        .expect("the name field");
    editor.click(at).frames(1);
    editor.key_with(egui::Modifiers::COMMAND, egui::Key::A);
    type_text(&mut editor, "space-shooter");

    let close = close_button(&editor, Tab::Settings).expect("a ✕ on the Settings tab");
    editor.click(close).frames(2);
    assert!(!editor.dock.is_open(Tab::Settings));
    assert_eq!(
        manifest(&dir),
        MANIFEST.replace(r#""hello-orrin""#, r#""space-shooter""#)
    );
    assert_eq!(editor.state.project_name.as_deref(), Some("space-shooter"));
}

/// Settings as Ctrl+, opens it, once the window has been measured.
fn open_by_key(editor: &mut Harness) {
    editor.frames(2);
    editor.key_with(egui::Modifiers::COMMAND, egui::Key::Comma);
    editor.frames(3);
    assert!(editor.dock.is_floating(Tab::Settings));
}

fn escape(editor: &mut Harness) {
    editor.key(egui::Key::Escape).frames(2);
}

/// One Escape does one thing: the window in front goes first, and what is
/// selected behind it waits for the next.
#[test]
fn escape_closes_settings_before_it_clears_the_selection() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let cube = editor.spawn("Cube");
    editor.state.selection.set(cube);
    open_by_key(&mut editor);

    escape(&mut editor);
    assert!(!editor.dock.is_open(Tab::Settings));
    assert!(!editor.state.selection.is_empty(), "the selection is kept");

    escape(&mut editor);
    assert!(editor.state.selection.is_empty());
}

/// Escape in a field means the field: it goes back to what it held, and only
/// the Escape after that closes the window.
#[test]
fn escape_in_a_project_field_puts_it_back_and_the_next_closes_settings() {
    let (dir, mut editor) = project("settings-escape-twice");
    open(&mut editor, Some(Page::General));
    let at = *in_settings(&editor, "0.1.0")
        .last()
        .expect("the version field");
    editor.click(at).frames(1);
    editor.key_with(egui::Modifiers::COMMAND, egui::Key::A);
    type_text(&mut editor, "9.9.9");

    escape(&mut editor);
    assert!(editor.dock.is_open(Tab::Settings), "the field's Escape");
    assert!(!in_settings(&editor, "0.1.0").is_empty());

    escape(&mut editor);
    assert!(!editor.dock.is_open(Tab::Settings));
    assert_eq!(manifest(&dir), MANIFEST);
}

/// In the editor Escape is how a running game's cursor is got back, and that
/// press is spent on it.
#[test]
fn the_escape_that_frees_a_games_cursor_does_not_also_close_settings() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    open_by_key(&mut editor);
    editor.state.request_play();
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(editor.state.playing());
    editor
        .world
        .resource_mut::<crate::scene::CursorLock>()
        .requested = true;
    editor.frames(1);

    // The app drops the request on the frame Escape arrives, before the
    // editor draws.
    editor
        .world
        .resource_mut::<crate::scene::CursorLock>()
        .requested = false;
    escape(&mut editor);
    assert!(editor.dock.is_open(Tab::Settings), "the cursor's Escape");

    escape(&mut editor);
    assert!(!editor.dock.is_open(Tab::Settings));
}
