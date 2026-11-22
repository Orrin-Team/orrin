//! Browsing the project: making and moving folders, and marking several
//! assets at once.

use super::*;
use crate::asset_db::AssetDatabase;
use crate::asset_db::tests::Dir;

fn browse(editor: &mut Harness, dir: &Dir) {
    let (db, _) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
    editor.world.insert_resource(db);
    editor.dock.activate(Tab::Assets);
    editor.frames(2);
}

fn id_at(editor: &Harness, path: &str) -> Option<orrin_registry::AssetId> {
    let db = editor.world.resource::<AssetDatabase>();
    db.find(path).map(|entry| entry.id)
}

fn has_folder(editor: &Harness, path: &str) -> bool {
    let db = editor.world.resource::<AssetDatabase>();
    db.folders().iter().any(|folder| folder.path == path)
}

/// Where the browser's text `name` was painted, inside its tab.
fn in_browser(editor: &Harness, name: &str) -> Option<egui::Pos2> {
    let body = editor.tab_body(Tab::Assets);
    editor
        .painted_text_rects()
        .into_iter()
        .find(|(text, rect)| text == name && body.contains(rect.center()))
        .map(|(_, rect)| rect.center())
}

/// The open folder's own space, below its rows.
fn empty_space(editor: &Harness) -> egui::Pos2 {
    let body = editor.tab_body(Tab::Assets);
    egui::pos2(body.center().x, body.bottom() - 16.0)
}

fn type_text(editor: &mut Harness, text: &str) {
    editor.input.events.push(egui::Event::Text(text.to_owned()));
    editor.frames(1);
    editor.input.events.clear();
}

/// Press at `from`, carry the pointer over to `to` in steps, and let go.
fn drag_to(editor: &mut Harness, from: egui::Pos2, to: egui::Pos2) {
    pointer(editor, egui::Event::PointerMoved(from));
    pointer(editor, button(from, true));
    for step in 1..=6 {
        let at = from + (to - from) * (step as f32 / 6.0);
        pointer(editor, egui::Event::PointerMoved(at));
    }
    pointer(editor, button(to, false));
    editor.frames(1);
}

// New folders.

/// New folder, on the open folder's space or on a folder, makes one there
/// and puts its name straight into the rename field.
#[test]
fn a_new_folder_is_made_where_it_was_asked_for_and_named_in_place() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let dir = Dir::new("browse-new-folder");
    dir.write("props/door.oprefab", b"orrin-scene 1\n");
    browse(&mut editor, &dir);

    let at = empty_space(&editor);
    secondary_click(&mut editor, at);
    editor.click(text_at(&editor, "New folder").expect("a New folder item"));
    editor.frames(1);
    assert!(has_folder(&editor, "New Folder"));
    type_text(&mut editor, "textures");
    editor.key(egui::Key::Enter).frames(1);
    assert!(has_folder(&editor, "textures"), "named as it was made");
    assert!(!has_folder(&editor, "New Folder"));

    let at = in_browser(&editor, "props").unwrap();
    secondary_click(&mut editor, at);
    editor.click(text_at(&editor, "New folder").unwrap());
    editor.frames(1);
    type_text(&mut editor, "wood");
    editor.key(egui::Key::Enter).frames(1);
    assert!(
        has_folder(&editor, "props/wood"),
        "inside the folder clicked"
    );
    assert!(
        in_browser(&editor, "door.oprefab").is_some(),
        "which opened to show it"
    );
}

// Moving.

/// An asset or a folder dropped on a folder moves into it, the id going
/// along; a move onto a name already taken is refused by name.
#[test]
fn what_is_dragged_onto_a_folder_moves_into_it() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let dir = Dir::new("browse-move");
    dir.write("rock.png", b"");
    dir.write("stone.png", b"");
    dir.write("props/stone.png", b"");
    dir.write("models/tree.glb", b"");
    browse(&mut editor, &dir);
    let rock = id_at(&editor, "rock.png").unwrap();

    let (from, to) = (
        in_browser(&editor, "rock.png").unwrap(),
        in_browser(&editor, "props").unwrap(),
    );
    drag_to(&mut editor, from, to);
    assert_eq!(id_at(&editor, "props/rock.png"), Some(rock));

    let (from, to) = (
        in_browser(&editor, "models").unwrap(),
        in_browser(&editor, "props").unwrap(),
    );
    drag_to(&mut editor, from, to);
    assert!(
        id_at(&editor, "props/models/tree.glb").is_some(),
        "a folder"
    );

    let (from, to) = (
        in_browser(&editor, "stone.png").unwrap(),
        in_browser(&editor, "props").unwrap(),
    );
    drag_to(&mut editor, from, to);
    assert!(id_at(&editor, "stone.png").is_some(), "left where it was");
    let refused = logged(&editor);
    assert!(
        refused
            .iter()
            .any(|line| line.contains("stone.png") && line.contains("already exists")),
        "{refused:?}"
    );
}

// Marking several.

/// Shift extends the mark from the last click, Ctrl toggles one; Delete
/// over the browser asks once for the whole set and leaves the scene's
/// selection alone; and a drag of one of them moves them all.
#[test]
fn several_marked_assets_are_deleted_or_moved_together() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let dir = Dir::new("browse-marks");
    for name in ["a.png", "b.png", "c.png", "d.png"] {
        dir.write(name, b"");
    }
    dir.write("props/door.oprefab", b"orrin-scene 1\n");
    browse(&mut editor, &dir);
    let cube = editor.spawn("Cube");
    editor.state.selection.set(cube);
    let c = id_at(&editor, "c.png").unwrap();

    editor.click(in_browser(&editor, "a.png").unwrap());
    editor.click_with(
        in_browser(&editor, "c.png").unwrap(),
        egui::Modifiers::SHIFT,
    );
    assert_eq!(editor.state.revealed_asset, Some(c), "the last clicked");
    editor.click_with(
        in_browser(&editor, "b.png").unwrap(),
        egui::Modifiers::COMMAND,
    );
    assert_eq!(editor.state.revealed_asset, Some(c));

    let at = empty_space(&editor);
    editor.hover(at).frames(1);
    editor.key(egui::Key::Delete).frames(1);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert_eq!(
        editor.state.pending_delete().map(<[String]>::to_vec),
        Some(vec!["a.png".to_owned(), "c.png".to_owned()])
    );
    assert!(editor.painted_contains("Delete 2 items?"));
    assert!(
        editor.world.is_alive(cube),
        "the scene's Delete is not this"
    );
    editor.click(text_at(&editor, "Cancel").unwrap());
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(editor.state.pending_delete().is_none());

    let (from, to) = (
        in_browser(&editor, "a.png").unwrap(),
        in_browser(&editor, "props").unwrap(),
    );
    drag_to(&mut editor, from, to);
    assert!(id_at(&editor, "props/a.png").is_some());
    assert_eq!(id_at(&editor, "props/c.png"), Some(c), "the whole set");
    assert!(id_at(&editor, "b.png").is_some(), "not marked, not moved");
}

/// Delete… on a marked row asks for the set; on one that is not, for it.
#[test]
fn the_menus_delete_asks_for_the_set_it_was_opened_on() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let dir = Dir::new("browse-marks-menu");
    for name in ["a.png", "b.png", "c.png"] {
        dir.write(name, b"");
    }
    browse(&mut editor, &dir);
    editor.click(in_browser(&editor, "a.png").unwrap());
    editor.click_with(
        in_browser(&editor, "b.png").unwrap(),
        egui::Modifiers::COMMAND,
    );

    let at = in_browser(&editor, "b.png").unwrap();
    secondary_click(&mut editor, at);
    editor.click(text_at(&editor, "Delete 2 items…").expect("the set's item"));
    assert_eq!(
        editor.state.pending_delete().map(<[String]>::to_vec),
        Some(vec!["a.png".to_owned(), "b.png".to_owned()])
    );
    editor.state.resolve_delete(false);
    editor.frames(1);

    let at = in_browser(&editor, "c.png").unwrap();
    secondary_click(&mut editor, at);
    editor.click(text_at(&editor, "Delete…").expect("its own item"));
    assert_eq!(
        editor.state.pending_delete().map(<[String]>::to_vec),
        Some(vec!["c.png".to_owned()])
    );
    editor.state.resolve_delete(true);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(id_at(&editor, "c.png").is_none());
}

// The breadcrumb.

// Double-clicks.

// The scene.
