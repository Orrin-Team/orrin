//! The Settings tab, driven through the whole editor as the other panel tests
//! are: its entry points, its two scopes, its search, and the manifest it
//! writes.

use super::*;
use crate::asset_db::tests::Dir;
use crate::editor::panels::settings::Page;
use crate::editor::settings::{Environment, FlyCamera};
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
fn project(name: &str) -> (Dir, Harness) {
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

fn open(editor: &mut Harness, page: Option<Page>) {
    editor.state.open_settings(page);
    editor.frames(3);
}

/// Every place `needle` was painted inside the Settings tab, in paint order:
/// a popup over the tab comes last.
fn in_settings(editor: &Harness, needle: &str) -> Vec<egui::Pos2> {
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

/// At a docked column's width the sidebar folds into one page picker; at the
/// window's own width the two groups are shown beside the page.
#[test]
fn the_sidebar_folds_into_a_picker_when_narrow() {
    for (width, sidebar) in [(300.0, false), (760.0, true)] {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.state.open_settings(Some(Page::Viewport));
        let Harness {
            world,
            state,
            themes,
            ctx,
            input,
            ..
        } = &mut editor;
        let cell = egui::vec2(width, 520.0);
        let mut output = None;
        for _ in 0..3 {
            output = Some(ctx.run(input.clone(), |ctx| {
                egui::Area::new("settings".into())
                    .fixed_pos(egui::Pos2::ZERO)
                    .show(ctx, |ui| {
                        ui.allocate_ui(cell, |ui| {
                            ui.set_min_size(cell);
                            settings::body(ui, world, state, themes);
                        });
                    });
            }));
        }
        let output = output.unwrap();
        let painted = painted_text(&output);
        let shows = |needle: &str| painted.iter().any(|text| text == needle);

        assert!(shows("Fly speed"), "{width}: the page draws");
        assert!(shows("Viewport"), "{width}: the page is named");
        assert_eq!(shows("Orrin"), sidebar, "{width}: {painted:?}");
        assert_eq!(shows("Appearance"), sidebar, "{width}: {painted:?}");

        let mut spilling = Vec::new();
        for clipped in &output.shapes {
            if let egui::Shape::Text(text) = &clipped.shape {
                let rect = text.visual_bounding_rect();
                if rect.right() > width + 1.0 && clipped.clip_rect.right() > width + 1.0 {
                    spilling.push(text.galley.text().to_owned());
                }
            }
        }
        assert!(spilling.is_empty(), "{width}: past the edge: {spilling:?}");
    }
}

/// The sidebar's two headings are where scope is said. A page is its title
/// and its rows: no line under the title repeating the heading, and no path
/// to the file it writes under the rows.
#[test]
fn a_page_is_its_title_and_its_rows() {
    let (_dir, mut editor) = project("settings-plain");
    open(&mut editor, Some(Page::Appearance));
    assert!(!in_settings(&editor, "Orrin").is_empty(), "the heading");
    for gone in ["this computer", "Only you, on this computer"] {
        assert!(in_settings(&editor, gone).is_empty(), "{gone}");
    }
    assert!(
        editor.painted_contains("Not saved: no user directory"),
        "still said: it is a warning, not a place"
    );

    editor.state.settings_file = Some("/somewhere/orrin/settings.toml".into());
    editor.frames(2);
    assert!(!editor.painted_contains("settings.toml"));
    assert!(!editor.painted_contains("Not saved"));
    assert!(in_settings(&editor, "Reveal").is_empty());

    open(&mut editor, Some(Page::General));
    assert!(!editor.painted_contains("Shared with everyone"));
    assert!(!editor.painted_contains("orrin.toml"));
    assert_eq!(
        in_settings(&editor, "Reveal").len(),
        1,
        "the Location row's own"
    );
}

/// A search lists pages from both groups with no sidebar beside them, so each
/// is followed by its group's name, and by nothing more.
#[test]
fn a_search_names_the_group_each_page_is_in() {
    let (_dir, mut editor) = project("settings-search-groups");
    open(&mut editor, None);
    let beside = |editor: &Harness, page: &str, group: &str| {
        let pages = in_settings(editor, page);
        in_settings(editor, group)
            .iter()
            .any(|g| pages.iter().any(|p| (p.y - g.y).abs() < 4.0 && g.x > p.x))
    };

    editor.state.settings_panel.set_query("vsync");
    editor.frames(2);
    assert!(beside(&editor, "Display", "Orrin"));
    assert!(!editor.painted_contains("this computer"));

    editor.state.settings_panel.set_query("semver");
    editor.frames(2);
    assert!(beside(&editor, "General", "Project"));
    assert!(!editor.painted_contains("shared"));
}

/// Search reaches both scopes by what people type, and settings that live
/// in another tool as a way there; a query nothing matches says so.
#[test]
fn search_finds_settings_by_what_people_type_and_says_where_others_live() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.key_with(egui::Modifiers::COMMAND, egui::Key::Comma);
    editor.frames(2);

    type_text(&mut editor, "vsync");
    editor.frames(2);
    assert!(editor.painted_contains("Present mode"));
    let display = in_settings(&editor, "Display");
    let present = in_settings(&editor, "Present mode");
    assert!(
        display.iter().any(|d| present.iter().any(|p| p.y > d.y)),
        "listed under its page: {display:?} {present:?}"
    );
    assert!(!editor.painted_contains("Fly speed"));

    editor.state.settings_panel.set_query("fog");
    editor.frames(2);
    assert!(editor.painted_contains("in Environment (this scene)"));

    editor.state.settings_panel.set_query("zoom");
    editor.frames(2);
    assert!(editor.painted_contains("UI scale"));

    editor.state.settings_panel.set_query("rider");
    editor.frames(2);
    assert!(editor.painted_contains("External editor"));

    editor.state.settings_panel.set_query("qwxz");
    editor.frames(2);
    assert!(editor.painted_contains("No setting matches"));
    assert!(!editor.painted_contains("Present mode"));
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

/// `ORRIN_EDITOR` decides for this run, so the row says so and cannot be
/// changed from under it.
#[test]
fn the_external_editor_is_locked_while_the_environment_chooses_it() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.state.environment = Environment::from_vars(|name| {
        (name == "ORRIN_EDITOR").then(|| "subl {file}:{line}".to_owned())
    });
    open(&mut editor, Some(Page::Scripting));
    assert!(editor.painted_contains("Set by ORRIN_EDITOR for this run"));

    let combo = in_settings(&editor, "Auto")[0];
    editor.click(combo).frames(2);
    assert!(!editor.painted_contains("Rider"), "the picker did not open");
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

#[test]
fn escape_puts_a_project_field_back_and_writes_nothing() {
    let (dir, mut editor) = project("settings-escape");
    open(&mut editor, Some(Page::General));
    let at = *in_settings(&editor, "0.1.0")
        .last()
        .expect("the version field");
    editor.click(at).frames(1);
    editor.key_with(egui::Modifiers::COMMAND, egui::Key::A);
    type_text(&mut editor, "9.9.9");
    editor.key(egui::Key::Escape).frames(2);

    assert_eq!(manifest(&dir), MANIFEST);
    assert!(!in_settings(&editor, "0.1.0").is_empty());
    assert!(!editor.painted_contains("9.9.9"));
}

/// The scenes a game is played with are not changed under it.
#[test]
fn the_scene_pickers_are_closed_while_the_game_plays() {
    let (dir, mut editor) = project("settings-playing");
    editor.state.request_play();
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(editor.state.playing());
    open(&mut editor, Some(Page::Scenes));

    let start = in_settings(&editor, "None")
        .into_iter()
        .min_by(|a, b| a.y.total_cmp(&b.y))
        .expect("the start scene's picker");
    editor.click(start).frames(2);
    assert!(in_settings(&editor, "assets/levels/one.orrin").is_empty());
    assert_eq!(manifest(&dir), MANIFEST);
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

#[test]
fn with_no_project_the_project_group_is_one_line() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    open(&mut editor, None);
    assert!(editor.painted_contains("No project is open. Open one with `orrin run` or Orrin Hub."));
    // "Scenes" and "Scripts" are also docked tools, whose tabs sit behind the
    // window; "General" is only ever a Project page.
    assert!(!editor.painted_contains("General"));
    editor.state.settings_panel.set_query("version");
    editor.frames(2);
    assert!(
        editor.painted_contains("No setting matches"),
        "nor in search"
    );
}

fn file_project_settings(editor: &mut Harness) {
    editor.frames(2);
    let file = text_at(editor, "File").expect("the File menu's title");
    editor.click(file).frames(2);
    let entry = text_at(editor, "Project settings…").expect("File lists it");
    editor.click(entry).frames(3);
}

#[test]
fn file_project_settings_needs_a_project_and_opens_its_general_page() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    file_project_settings(&mut editor);
    assert!(!editor.dock.is_open(Tab::Settings));

    let (_dir, mut editor) = project("settings-file-menu");
    file_project_settings(&mut editor);
    assert!(editor.dock.is_open(Tab::Settings));
    assert!(
        !in_settings(&editor, "Version").is_empty(),
        "the General page"
    );
    assert!(!in_settings(&editor, "hello-orrin").is_empty());
}

#[test]
fn the_project_name_in_the_top_bar_opens_project_settings() {
    let (_dir, mut editor) = project("settings-top-bar");
    editor.frames(2);
    let bar = editor.panel("menu_bar");
    let name = editor
        .painted_text_rects()
        .into_iter()
        .find(|(text, rect)| text == "hello-orrin" && bar.contains(rect.center()))
        .expect("the bar names the project")
        .1;
    editor.click(name.center()).frames(3);
    assert!(editor.dock.is_open(Tab::Settings));
    assert!(
        !in_settings(&editor, "Version").is_empty(),
        "the General page"
    );
}

fn startup_note_count(editor: &Harness) -> usize {
    in_settings(editor, "Applies on restart").len()
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

/// The four choices that were environment variables, each with the one fact
/// about it that matters: nothing happens until the next start.
#[test]
fn the_startup_page_lists_four_rows_that_apply_on_restart() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.state.backend = crate::gfx::BackendChoice::Vulkan;
    open(&mut editor, Some(Page::Startup));
    for label in [
        "Graphics backend",
        "Worker threads",
        "Validation layers",
        "Log level",
    ] {
        assert!(!in_settings(&editor, label).is_empty(), "{label}");
    }
    assert_eq!(startup_note_count(&editor), 4);
    let auto = crate::threads::automatic();
    assert!(editor.painted_contains(&format!("Auto is {auto} on this computer")));
    assert!(restart_line(&editor).is_none(), "nothing changed yet");
    assert!(!editor.painted_contains("this run draws through Metal"));
}

/// Validation is a Vulkan layer, so a run on Metal says it does nothing here.
#[test]
fn validation_says_it_is_vulkans_on_a_metal_run() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.state.backend = crate::gfx::BackendChoice::Metal;
    open(&mut editor, Some(Page::Startup));
    assert!(editor.painted_contains("Validation is Vulkan's; this run draws through Metal"));
}

#[test]
fn a_startup_row_the_environment_sets_says_which_variable_and_shows_its_value() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.state.environment =
        Environment::from_vars(|name| (name == "ORRIN_THREADS").then(|| "3".to_owned()));
    editor.state.settings.startup.threads = Some(5);
    open(&mut editor, Some(Page::Startup));
    assert!(editor.painted_contains("Set by ORRIN_THREADS for this run"));

    let row = in_settings(&editor, "Worker threads")[0];
    let shown = in_settings(&editor, "3")
        .into_iter()
        .find(|at| (at.y - row.y).abs() < 6.0)
        .expect("the row shows the environment's 3, not the saved 5");
    let autos = in_settings(&editor, "Auto").len();
    editor.click(shown).frames(2);
    assert_eq!(
        in_settings(&editor, "Auto").len(),
        autos,
        "the picker did not open"
    );
    assert_eq!(editor.state.settings.startup.threads, Some(5));
    assert!(
        restart_line(&editor).is_none(),
        "a value the environment overrides waits on nothing"
    );
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

#[test]
fn search_finds_worker_threads_under_startup_with_the_restart_line_above() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.state.settings.startup.threads = Some(2);
    open(&mut editor, None);
    editor.state.settings_panel.set_query("cores");
    editor.frames(2);
    let heading = in_settings(&editor, "Startup")[0];
    let row = in_settings(&editor, "Worker threads")[0];
    assert!(row.y > heading.y, "listed under its page");
    let line = in_settings(&editor, "Restart Orrin to apply 1 change")[0];
    assert!(line.y < heading.y, "above the results");

    editor.state.settings_panel.set_query("fly");
    editor.frames(2);
    assert!(
        restart_line(&editor).is_none(),
        "not above unrelated results"
    );
}

#[test]
fn clicking_restart_raises_the_restart_request_and_nothing_else() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.state.settings.startup.backend = crate::editor::settings::Backend::Vulkan;
    open(&mut editor, Some(Page::Startup));
    let settings = editor.state.settings.clone();
    let button = in_settings(&editor, "Restart now")[0];
    editor.click(button).frames(1);
    editor.state.apply(&mut editor.world, &editor.registry);

    assert!(editor.state.take_restart_request());
    assert!(!editor.state.take_restart_request(), "once");
    assert_eq!(editor.state.settings, settings);
    assert!(!editor.state.scene_dirty());
    assert!(!std::path::Path::new(&editor.state.scene_path).exists());
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

/// The scene a game is played in is not what a save would write, so the
/// restart waits for Stop.
#[test]
fn restarting_is_closed_while_the_game_plays() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.state.settings.startup.threads = Some(2);
    editor.state.request_play();
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(editor.state.playing());
    open(&mut editor, Some(Page::Startup));

    let button = in_settings(&editor, "Restart now")[0];
    editor.click(button).frames(1);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(!editor.state.take_restart_request());
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

/// Settings is opened by command and listed in no menu of tools, so its tab
/// carries the one ✕ in the dock; every other tool still closes from View.
#[test]
fn the_settings_tab_closes_from_its_own_close_button_and_no_other_tab_has_one() {
    let (_dir, mut editor) = project("settings-close");
    file_project_settings(&mut editor);
    assert!(editor.dock.is_floating(Tab::Settings));
    let windows = editor.dock.state().iter_surfaces().count();

    for tab in Tab::ALL {
        if editor.dock.is_open(tab) {
            assert_eq!(close_button(&editor, tab), None, "{tab:?} has no ✕");
        }
    }
    let close = close_button(&editor, Tab::Settings).expect("a ✕ on the Settings tab");
    editor.click(close).frames(2);

    assert!(!editor.dock.is_open(Tab::Settings));
    assert_eq!(
        editor.dock.state().iter_surfaces().count(),
        windows - 1,
        "its window went with it"
    );
    assert!(!editor.painted_contains("Search settings"));
    for tab in Tab::ALL {
        assert!(editor.dock.is_open(tab), "{tab:?} stayed open");
    }

    // And it comes back the way it came.
    editor.key_with(egui::Modifiers::COMMAND, egui::Key::Comma);
    editor.frames(3);
    assert!(editor.dock.is_floating(Tab::Settings));
}

/// Docked beside other tools it is still the one tab that closes, and closing
/// it takes nothing else out of the leaf. The window is wide because "Dock
/// back" lands it in the left column, whose strip at 1280 scrolls three tabs
/// and clips the last one's ✕.
#[test]
fn docked_settings_still_closes_and_leaves_its_neighbours() {
    let mut editor = Harness::new(egui::vec2(2560.0, 1000.0));
    open(&mut editor, None);
    editor.dock.apply(DockRequest::Dock(Tab::Settings));
    editor.frames(2);
    assert!(!editor.dock.is_floating(Tab::Settings));

    let close = close_button(&editor, Tab::Settings).expect("a ✕ on the docked tab");
    editor.click(close).frames(2);
    assert!(!editor.dock.is_open(Tab::Settings));
    for tab in Tab::ALL {
        assert!(editor.dock.is_open(tab), "{tab:?} stayed open");
    }
}

/// Where `tab`'s title is painted in its strip.
fn tab_title(editor: &Harness, tab: Tab) -> egui::Pos2 {
    let strip = strip_of(editor, tab);
    editor
        .painted_text_rects()
        .into_iter()
        .find(|(text, rect)| text == tab.title() && strip.contains(rect.center()))
        .unwrap_or_else(|| panic!("{tab:?}'s title is in its strip"))
        .1
        .center()
}

/// The tab's menu says it in words, and only this tab's menu does.
#[test]
fn the_settings_tabs_menu_offers_close_and_another_tools_does_not() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    open(&mut editor, None);

    // The Inspector's strip is clear of the window Settings opened in.
    let inspector = tab_title(&editor, Tab::Inspector);
    editor.right_click(inspector).frames(2);
    assert!(text_at(&editor, "Float in a window").is_some(), "its menu");
    assert_eq!(text_at(&editor, "Close"), None);
    editor.key(egui::Key::Escape).frames(2);

    let settings = tab_title(&editor, Tab::Settings);
    editor.right_click(settings).frames(2);
    let close = text_at(&editor, "Close").expect("the menu lists Close");
    editor.click(close).frames(2);
    assert!(!editor.dock.is_open(Tab::Settings));
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

/// A point on the scene itself, clear of the window Settings is in.
fn on_the_scene(editor: &Harness) -> egui::Pos2 {
    let viewport = editor.dock.viewport().expect("the scene shows");
    let at = viewport.left_center() + egui::vec2(60.0, 0.0);
    assert!(editor.dock.over_scene(&editor.ctx, at), "{at:?}");
    at
}

/// Opened by command, the window is what the next Escape answers to.
#[test]
fn escape_closes_the_window_settings_was_opened_into() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    open_by_key(&mut editor);
    let windows = editor.dock.state().iter_surfaces().count();

    escape(&mut editor);
    assert!(!editor.dock.is_open(Tab::Settings));
    assert_eq!(
        editor.dock.state().iter_surfaces().count(),
        windows - 1,
        "its window went with it"
    );
    for tab in Tab::ALL {
        assert!(editor.dock.is_open(tab), "{tab:?} stayed open");
    }
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

/// A Settings window left open beside the scene does not take the scene's
/// Escape: the key goes to whichever of the two was pressed last.
#[test]
fn after_a_press_on_the_scene_escape_is_the_scenes_until_settings_is_pressed() {
    let mut editor = Harness::new(egui::vec2(2560.0, 1000.0));
    let cube = editor.spawn("Cube");
    open_by_key(&mut editor);

    let scene = on_the_scene(&editor);
    editor.click(scene).frames(2);
    editor.state.selection.set(cube);
    editor.frames(1);
    escape(&mut editor);
    assert!(editor.dock.is_open(Tab::Settings), "not the scene's Escape");
    assert!(editor.state.selection.is_empty());

    let search = in_settings(&editor, "Search settings")[0];
    editor.click(search).frames(2);
    editor.state.selection.set(cube);
    editor.frames(1);
    escape(&mut editor);
    assert!(!editor.dock.is_open(Tab::Settings));
    assert!(!editor.state.selection.is_empty());
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

/// A search with something typed in it is being used, so Escape leaves the
/// field and keeps the window; the empty one Settings opens with is not.
#[test]
fn escape_leaves_a_search_with_text_in_it_before_it_closes_settings() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    open_by_key(&mut editor);
    type_text(&mut editor, "theme");
    editor.frames(1);

    escape(&mut editor);
    assert!(editor.dock.is_open(Tab::Settings), "the search's Escape");
    assert!(
        !in_settings(&editor, "theme").is_empty(),
        "the query is kept"
    );

    escape(&mut editor);
    assert!(!editor.dock.is_open(Tab::Settings));
}

/// An open list is in front of the window, so it is what Escape dismisses.
#[test]
fn escape_dismisses_a_dropdown_before_it_closes_settings() {
    let (dir, mut editor) = project("settings-escape-dropdown");
    open(&mut editor, Some(Page::Scenes));
    let start = in_settings(&editor, "None")
        .into_iter()
        .min_by(|a, b| a.y.total_cmp(&b.y))
        .expect("the start scene's picker");
    editor.click(start).frames(2);
    assert!(!in_settings(&editor, "assets/levels/one.orrin").is_empty());

    escape(&mut editor);
    assert!(editor.dock.is_open(Tab::Settings), "the list's Escape");
    assert!(in_settings(&editor, "assets/levels/one.orrin").is_empty());

    escape(&mut editor);
    assert!(!editor.dock.is_open(Tab::Settings));
    assert_eq!(manifest(&dir), MANIFEST);
}

/// The same for the tab's own menu.
#[test]
fn escape_dismisses_the_tabs_menu_before_it_closes_settings() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    open_by_key(&mut editor);
    let title = tab_title(&editor, Tab::Settings);
    editor.right_click(title).frames(2);
    assert!(text_at(&editor, "Close").is_some(), "its menu");

    escape(&mut editor);
    assert!(editor.dock.is_open(Tab::Settings), "the menu's Escape");
    assert_eq!(text_at(&editor, "Close"), None);

    escape(&mut editor);
    assert!(!editor.dock.is_open(Tab::Settings));
}

/// Docked, Settings is a tool among the others, and none of those closes on
/// Escape.
#[test]
fn escape_does_not_close_settings_docked_among_the_tools() {
    let mut editor = Harness::new(egui::vec2(2560.0, 1000.0));
    open_by_key(&mut editor);
    editor.dock.apply(DockRequest::Dock(Tab::Settings));
    editor.frames(2);
    assert!(!editor.dock.is_floating(Tab::Settings));

    let search = in_settings(&editor, "Search settings")[0];
    editor.click(search).frames(2);
    escape(&mut editor);
    escape(&mut editor);
    assert!(editor.dock.is_open(Tab::Settings));
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

/// Escape in a field is the field's, whatever is selected behind it. egui has
/// let go of the keyboard by the time the editor sees the key, so the
/// selection's Escape once took it, and the field committed what it was
/// meant to drop.
#[test]
fn escape_in_a_project_field_writes_nothing_with_an_entity_selected() {
    let (dir, mut editor) = project("settings-escape-selected");
    let cube = editor.spawn("Cube");
    editor.state.selection.set(cube);
    open(&mut editor, Some(Page::General));
    let at = *in_settings(&editor, "0.1.0")
        .last()
        .expect("the version field");
    editor.click(at).frames(1);
    editor.key_with(egui::Modifiers::COMMAND, egui::Key::A);
    type_text(&mut editor, "9.9.9");

    escape(&mut editor);
    assert_eq!(manifest(&dir), MANIFEST);
    assert!(!in_settings(&editor, "0.1.0").is_empty());
    assert!(!editor.state.selection.is_empty(), "not the selection's");
    assert!(editor.dock.is_open(Tab::Settings));
}
