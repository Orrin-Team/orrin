//! The editor doing what was aimed at: Escape, the question before unsaved
//! work is lost, and the rows a user closed staying closed.

use super::*;

fn painted_named(editor: &Harness, needle: &str) -> bool {
    editor.painted().iter().any(|text| text == needle)
}

/// Escape closes the menu in front of the selection before it clears the
/// selection, here a row's own menu.
#[test]
fn escape_closes_a_row_menu_before_it_clears_the_selection() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let crate_ = editor.spawn("Crate");
    editor.state.selection.set(crate_);
    editor.frames(2);
    editor.right_click(row_of(&editor, "Crate")).frames(1);
    assert!(
        painted_named(&editor, "Save as prefab"),
        "the row menu opened"
    );

    editor.key(egui::Key::Escape).frames(1);
    assert!(
        !painted_named(&editor, "Save as prefab"),
        "the menu is gone"
    );
    assert!(editor.state.selection.contains(crate_), "the menu's Escape");

    editor.key(egui::Key::Escape).frames(1);
    assert!(editor.state.selection.is_empty(), "and the next is its own");
}

/// A scene file holding one entity called `name`, at `rel` in `dir`.
fn scene_file(dir: &crate::asset_db::tests::Dir, rel: &str, name: &str) -> String {
    let mut source = World::new();
    let entity = source.spawn();
    source.insert(entity, Name::new(name));
    let registry = Harness::new(egui::vec2(10.0, 10.0)).registry;
    dir.write(rel, crate::scene::save(&mut source, &registry).as_bytes());
    dir.path().join(rel).to_string_lossy().into_owned()
}

/// The editor with `path` open, then given unsaved work: an entity deleted
/// through the history, and one called "Keep" that a load would throw away.
fn dirty_over(editor: &mut Harness, path: &str) {
    editor.state.request_open_scene(path);
    editor.state.apply(&mut editor.world, &editor.registry);
    delete_by_key(editor, "Old crate");
    editor.spawn("Keep");
    assert!(editor.state.scene_dirty());
}

/// Reloading the open scene over unsaved work asks, wherever it was asked
/// from, and nothing is thrown away until the answer.
#[test]
fn reloading_over_unsaved_work_asks_and_waits_for_the_answer() {
    let dir = crate::asset_db::tests::Dir::new("intent-reload-dirty");
    let hub = scene_file(&dir, "hub.orrin", "Gate");
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    dirty_over(&mut editor, &hub);
    editor.dock.activate(Tab::Scenes);
    editor.frames(2);

    editor.click(text_at(&editor, "Reload").expect("the Scenes tab's Reload"));
    editor.state.apply(&mut editor.world, &editor.registry);
    editor.frames(2);
    assert!(
        named(&editor.world, "Keep").is_some(),
        "nothing was reloaded yet"
    );
    assert!(
        editor.painted_contains("unsaved changes"),
        "{:?}",
        editor.painted()
    );
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(named(&editor.world, "Keep").is_some(), "still waiting");

    editor.click(text_at(&editor, "Reload without saving").expect("a discarding answer"));
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(
        named(&editor.world, "Keep").is_none(),
        "{:?}",
        logged(&editor)
    );
    assert!(named(&editor.world, "Gate").is_some());
    assert!(!editor.state.scene_dirty());
}

/// Closing the window asks only when there is unsaved work, keeps asking
/// however often the close is repeated, and quits on the answer.
#[test]
fn closing_over_unsaved_work_asks_and_a_second_close_does_not_answer() {
    let dir = crate::asset_db::tests::Dir::new("intent-close");
    let hub = scene_file(&dir, "hub.orrin", "Gate");
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.state.request_open_scene(hub.as_str());
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(editor.state.request_close(), "a saved scene closes at once");

    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    dirty_over(&mut editor, &hub);
    assert!(!editor.state.request_close(), "unsaved work is asked about");
    editor.frames(2);
    assert!(
        editor.painted_contains("Save and quit"),
        "{:?}",
        editor.painted()
    );
    assert!(
        !editor.state.request_close(),
        "a second close is not an answer"
    );
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(!editor.state.take_quit_request());

    editor.click(text_at(&editor, "Save and quit").expect("a saving answer"));
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(editor.state.take_quit_request(), "{:?}", logged(&editor));
    let saved = std::fs::read_to_string(&hub).unwrap();
    assert!(saved.contains("Keep"), "the work was saved before quitting");
}

/// Save scene as… moves the scene only once a file was picked, and then
/// saves it there. A dialog closed without a pick changes nothing.
#[test]
fn save_as_moves_the_scene_only_to_a_picked_file() {
    let dir = crate::asset_db::tests::Dir::new("intent-save-as");
    let hub = scene_file(&dir, "hub.orrin", "Gate");
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    dirty_over(&mut editor, &hub);
    let picked = dir.path().join("copy.orrin");
    let save_as = |editor: &mut Harness, pick: Option<std::path::PathBuf>| {
        *menu_bar::SCENE_PICKED.lock().unwrap() = Some(pick);
        editor.frames(2);
        editor.click(row_of(editor, "File")).frames(1);
        editor.click(text_at(editor, "Save scene as…").expect("File ▸ Save scene as…"));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while menu_bar::dialog_open(&editor.ctx) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
            editor.frames(1);
        }
        editor.state.apply(&mut editor.world, &editor.registry);
    };

    save_as(&mut editor, None);
    assert_eq!(editor.state.scene_path, hub, "nothing was picked");
    assert!(editor.state.scene_dirty());

    save_as(&mut editor, Some(picked.clone()));
    assert_eq!(std::path::Path::new(&editor.state.scene_path), picked);
    assert!(std::fs::read_to_string(&picked).unwrap().contains("Keep"));
    assert!(!editor.state.scene_dirty());
}

/// A scene of two parents, each with a child, written to `rel` in `dir`.
fn tree_file(dir: &crate::asset_db::tests::Dir, rel: &str) -> String {
    let mut source = Harness::new(egui::vec2(10.0, 10.0));
    for (parent, child) in [("Rig", "Muzzle"), ("Crate", "Lid")] {
        let parent = source.spawn(parent);
        let child = source.spawn(child);
        crate::scene::reparent(&mut source.world, child, Some(parent), false).unwrap();
    }
    let text = crate::scene::save(&mut source.world, &source.registry);
    dir.write(rel, text.as_bytes());
    dir.path().join(rel).to_string_lossy().into_owned()
}

fn opened(path: &str) -> Harness {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    editor.state.request_open_scene(path);
    editor.state.apply(&mut editor.world, &editor.registry);
    editor.frames(2);
    editor
}

fn close_row(editor: &mut Harness, name: &str) {
    let band = row_band(editor, name);
    let chevron = icon_in(editor, crate::editor::icons::chevron_down(), band)
        .unwrap_or_else(|| panic!("{name} is an open parent"));
    editor.click(chevron).frames(1);
}

/// A row closed in the hierarchy stays closed when the scene is read again
/// and in the next session, which is handed what this one kept.
#[test]
fn a_closed_row_stays_closed_across_a_reload_and_a_restart() {
    let dir = crate::asset_db::tests::Dir::new("intent-closed-rows");
    let path = tree_file(&dir, "rig.orrin");
    let mut editor = opened(&path);
    assert!(editor.painted_contains("Muzzle"));
    close_row(&mut editor, "Rig");
    assert!(!editor.painted_contains("Muzzle"), "closed");
    assert!(editor.painted_contains("Lid"), "only the one row");

    editor
        .state
        .request_scene(crate::editor::state::SceneRequest::Load);
    editor.state.apply(&mut editor.world, &editor.registry);
    editor.frames(2);
    assert!(
        !editor.painted_contains("Muzzle"),
        "still closed after a reload"
    );

    let kept = editor
        .state
        .closed_rows_to_save(&editor.world)
        .expect("a change to keep");
    let prefs_dir = crate::asset_db::tests::Dir::new("intent-closed-rows-prefs");
    let file = crate::editor::prefs::PrefsFile::new(Some(prefs_dir.path().to_owned()));
    let mut prefs = crate::editor::prefs::Prefs::default();
    prefs.collapsed = kept;
    file.save(&prefs);

    let mut next = Harness::new(egui::vec2(1280.0, 800.0));
    next.state.seed_closed_rows(&file.load().collapsed);
    next.state.request_open_scene(path.as_str());
    next.state.apply(&mut next.world, &next.registry);
    next.frames(2);
    assert!(
        !next.painted_contains("Muzzle"),
        "still closed next session"
    );
    assert!(next.painted_contains("Lid"));
}
