//! Authoring in the tree and the inspector: duplicating, copying and pasting,
//! renaming in place, one edit made to many entities, a material asset edited
//! where it is marked, and the spot light.

use orrin_registry::EntityId;

use super::*;
use crate::scene::{Light, Parent, PrefabInstance};

/// A component that names another entity, as a game's would.
#[derive(Default, orrin_registry::Reflect)]
struct Follow {
    target: EntityId,
}

const FOLLOW: orrin_registry::ComponentId = orrin_registry::ComponentId::new("game.follow");

fn with_follow(editor: &mut Harness) {
    editor.registry.register::<Follow>(FOLLOW, "Follow");
}

fn follow(editor: &mut Harness, entity: orrin_ecs::Entity, target: orrin_ecs::Entity) {
    let id = crate::scene::edit::ensure_id(&mut editor.world, target);
    editor.world.insert(entity, Follow { target: id });
}

fn target_of(world: &World, entity: orrin_ecs::Entity) -> EntityId {
    world.get::<Follow>(entity).expect("it follows").target
}

fn id_of(world: &World, entity: orrin_ecs::Entity) -> EntityId {
    *world.get::<EntityId>(entity).expect("it has an id")
}

fn all_named(world: &World, name: &str) -> Vec<orrin_ecs::Entity> {
    world
        .entities()
        .filter(|&entity| world.get::<Name>(entity).is_some_and(|n| n.0 == name))
        .collect()
}

fn parent_of(world: &World, entity: orrin_ecs::Entity) -> Option<orrin_ecs::Entity> {
    world.get::<Parent>(entity).map(|parent| parent.get())
}

fn children_named(world: &World, parent: orrin_ecs::Entity, name: &str) -> Vec<orrin_ecs::Entity> {
    all_named(world, name)
        .into_iter()
        .filter(|&child| parent_of(world, child) == Some(parent))
        .collect()
}

fn under(editor: &mut Harness, child: orrin_ecs::Entity, parent: orrin_ecs::Entity) {
    crate::scene::reparent(&mut editor.world, child, Some(parent), false).unwrap();
}

/// A key with Ctrl (Cmd on a Mac) held, then the requests it made applied.
fn command(editor: &mut Harness, key: egui::Key) {
    editor.key_with(egui::Modifiers::COMMAND, key);
    editor.state.apply(&mut editor.world, &editor.registry);
}

fn undo(editor: &mut Harness) {
    editor.state.request_undo();
    editor.state.apply(&mut editor.world, &editor.registry);
}

/// Where the row called `name` was painted inside the tree.
fn tree_row(editor: &Harness, name: &str) -> Option<egui::Rect> {
    let tree = editor.tab_body(Tab::Hierarchy);
    editor
        .painted_text_rects()
        .into_iter()
        .find(|(text, rect)| text == name && tree.contains(rect.center()))
        .map(|(_, rect)| rect)
}

/// Park the pointer over the tree, clear of every row.
fn over_tree(editor: &mut Harness) {
    let tree = editor.tab_body(Tab::Hierarchy);
    editor
        .hover(egui::pos2(tree.center().x, tree.bottom() - 40.0))
        .frames(1);
}

// Duplicate, copy and paste.

/// A duplicate is the whole subtree again, beside the original: fresh ids, a
/// reference inside the copied set pointing at the copy, one to outside it
/// left alone, the copies selected, and all of it one step to undo.
#[test]
fn ctrl_d_duplicates_the_subtree_beside_the_original_in_one_step() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    with_follow(&mut editor);
    let room = editor.spawn("Room");
    let lamp = editor.spawn("Lamp");
    let bulb = editor.spawn("Bulb");
    under(&mut editor, lamp, room);
    under(&mut editor, bulb, lamp);
    follow(&mut editor, lamp, bulb);
    follow(&mut editor, bulb, room);
    editor.state.selection.set(lamp);
    editor.frames(2);

    command(&mut editor, egui::Key::D);

    let lamps = children_named(&editor.world, room, "Lamp");
    assert_eq!(lamps.len(), 2, "a second lamp under the room");
    let copy = *lamps.iter().find(|&&l| l != lamp).unwrap();
    assert_ne!(id_of(&editor.world, copy), id_of(&editor.world, lamp));
    let bulbs = children_named(&editor.world, copy, "Bulb");
    assert_eq!(bulbs.len(), 1, "the copy has its own bulb");
    let copied_bulb = bulbs[0];
    assert_ne!(
        id_of(&editor.world, copied_bulb),
        id_of(&editor.world, bulb)
    );
    assert_eq!(
        target_of(&editor.world, copy),
        id_of(&editor.world, copied_bulb),
        "a reference inside the copy points at the copy"
    );
    assert_eq!(
        target_of(&editor.world, copied_bulb),
        id_of(&editor.world, room),
        "a reference outside it is kept"
    );
    assert_eq!(target_of(&editor.world, lamp), id_of(&editor.world, bulb));
    assert_eq!(editor.state.selection.entities(), [copy]);

    undo(&mut editor);
    assert_eq!(all_named(&editor.world, "Lamp"), [lamp]);
    assert_eq!(all_named(&editor.world, "Bulb"), [bulb]);
    assert_eq!(editor.state.history.undo_label(), None, "one step");
}

/// A prefab with a root and a member, its instance in the scene.
fn door_instance(editor: &mut Harness) -> (orrin_registry::AssetId, orrin_ecs::Entity) {
    let id = orrin_registry::AssetId::new();
    let text = "orrin-scene 1\n\nentity 00000000-0000-4000-8000-000000000001\n  \
                orrin.name = \"Door\"\n\nentity 00000000-0000-4000-8000-000000000002\n  \
                orrin.name = \"Handle\"\n  \
                orrin.parent = entity(00000000-0000-4000-8000-000000000001)\n";
    let prefab = crate::scene::prefab::Prefab::parse(id, "door.oprefab", text).unwrap();
    editor
        .world
        .resource_mut::<crate::scene::Prefabs>()
        .insert(prefab);
    let (root, issues) =
        crate::scene::prefab::instantiate(&mut editor.world, &editor.registry, id).unwrap();
    assert!(issues.is_empty(), "{issues:?}");
    (id, root)
}

/// A copied instance is another instance of the same prefab, its members
/// derived from its own root, so an edit to the prefab reaches it too.
#[test]
fn a_duplicated_prefab_instance_stays_an_instance_of_its_prefab() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    with_follow(&mut editor);
    let (prefab, door) = door_instance(&mut editor);
    let handle = all_named(&editor.world, "Handle")[0];
    let latch = editor.spawn("Latch");
    under(&mut editor, latch, handle);
    follow(&mut editor, latch, handle);
    editor.state.selection.set(door);
    editor.frames(2);

    command(&mut editor, egui::Key::D);

    let doors = all_named(&editor.world, "Door");
    assert_eq!(doors.len(), 2);
    let copy = *doors.iter().find(|&&d| d != door).unwrap();
    assert_eq!(
        editor.world.get::<PrefabInstance>(copy).map(|i| i.prefab),
        Some(prefab)
    );
    let copied_handle = children_named(&editor.world, copy, "Handle")[0];
    let prefabs = editor.world.resource::<crate::scene::Prefabs>();
    assert!(crate::scene::prefab::is_member(
        prefabs.get(prefab).unwrap(),
        id_of(&editor.world, copy),
        id_of(&editor.world, copied_handle)
    ));
    drop(prefabs);
    let copied_latch = children_named(&editor.world, copied_handle, "Latch");
    assert_eq!(
        copied_latch.len(),
        1,
        "a scene entity under a member comes too"
    );
    assert_eq!(
        target_of(&editor.world, copied_latch[0]),
        id_of(&editor.world, copied_handle),
        "and still names the member, the copy's"
    );
}

/// Paste goes beside the primary selection, under its parent, and at the
/// top level when nothing is selected; each paste is one step.
#[test]
fn paste_lands_under_the_primary_selections_parent() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let room = editor.spawn("Room");
    let chair = editor.spawn("Chair");
    under(&mut editor, chair, room);
    let porch = editor.spawn("Porch");
    editor.state.selection.set(chair);
    editor.frames(2);
    command(&mut editor, egui::Key::C);
    assert_eq!(
        all_named(&editor.world, "Chair").len(),
        1,
        "a copy is no edit"
    );

    editor.state.selection.set(porch);
    command(&mut editor, egui::Key::V);
    let top: Vec<_> = all_named(&editor.world, "Chair")
        .into_iter()
        .filter(|&c| parent_of(&editor.world, c).is_none())
        .collect();
    assert_eq!(top.len(), 1, "beside the porch, at the top level");
    assert_eq!(editor.state.selection.entities(), top.as_slice());

    editor.state.selection.set(chair);
    command(&mut editor, egui::Key::V);
    assert_eq!(children_named(&editor.world, room, "Chair").len(), 2);

    editor.state.selection.clear();
    command(&mut editor, egui::Key::V);
    assert_eq!(all_named(&editor.world, "Chair").len(), 4);

    undo(&mut editor);
    undo(&mut editor);
    undo(&mut editor);
    assert_eq!(all_named(&editor.world, "Chair"), [chair]);
    assert_eq!(editor.state.history.undo_label(), None);
}

fn scene_with(dir: &crate::asset_db::tests::Dir, rel: &str, name: &str) -> String {
    let mut source = World::new();
    let entity = source.spawn();
    source.insert(entity, Name::new(name));
    let registry = Harness::new(egui::vec2(10.0, 10.0)).registry;
    dir.write(rel, crate::scene::save(&mut source, &registry).as_bytes());
    dir.path().join(rel).to_string_lossy().into_owned()
}

/// The clipboard is the session's, not the scene's.
#[test]
fn the_clipboard_outlives_opening_another_scene() {
    let dir = crate::asset_db::tests::Dir::new("authoring-clipboard");
    let other = scene_with(&dir, "other.orrin", "Gate");
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let chair = editor.spawn("Chair");
    editor.state.selection.set(chair);
    editor.frames(2);
    command(&mut editor, egui::Key::C);

    editor.state.request_open_scene(&other);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(named(&editor.world, "Gate").is_some());
    assert!(named(&editor.world, "Chair").is_none());

    command(&mut editor, egui::Key::V);
    assert_eq!(all_named(&editor.world, "Chair").len(), 1);
}

/// In a field these keys are the field's.
#[test]
fn the_authoring_keys_stand_aside_while_a_field_has_focus() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let chair = editor.spawn("Chair");
    editor.spawn("Table");
    editor.state.selection.set(chair);
    editor.frames(2);
    // Asked again before each: a focus no widget claims lapses after a frame.
    for key in [egui::Key::D, egui::Key::A, egui::Key::C, egui::Key::V] {
        editor
            .ctx
            .memory_mut(|memory| memory.request_focus(egui::Id::new("some field")));
        command(&mut editor, key);
    }
    assert_eq!(all_named(&editor.world, "Chair"), [chair]);
    assert_eq!(editor.state.selection.entities(), [chair]);
}

// Renaming in the row.

/// Type over the name and press Enter: one name edit, undone in one step.
fn type_over(editor: &mut Harness, text: &str) {
    editor.input.events.push(egui::Event::Text(text.to_owned()));
    editor.frames(1);
    editor.input.events.clear();
}

#[test]
fn f2_renames_the_primary_in_its_row_and_enter_commits_it() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let lamp = editor.spawn("Lamp");
    editor.state.selection.set(lamp);
    editor.frames(2);
    over_tree(&mut editor);

    editor.key(egui::Key::F2).frames(1);
    type_over(&mut editor, "Lantern");
    editor.key(egui::Key::Enter);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert_eq!(editor.world.get::<Name>(lamp).unwrap().0, "Lantern");
    editor.frames(2);
    assert!(tree_row(&editor, "Lantern").is_some(), "the row shows it");

    undo(&mut editor);
    assert_eq!(editor.world.get::<Name>(lamp).unwrap().0, "Lamp");
    assert_eq!(editor.state.history.undo_label(), None, "one step");
}

// The keyboard in the tree.

// A selection made elsewhere shows itself.

// One edit, every selected entity.

fn at(x: f32, y: f32, z: f32) -> LocalTransform {
    LocalTransform::new(crate::scene::Transform::from_translation(glam::Vec3::new(
        x, y, z,
    )))
}

/// Drag the primary's Position X by `by` points through the real widget.
fn drag_position_x(editor: &mut Harness, by: f32) {
    let rects = editor.painted_text_rects();
    let row = rects.iter().find(|(text, _)| text == "Position").unwrap().1;
    let chip = rects
        .iter()
        .find(|(text, rect)| text == "X" && (rect.center().y - row.center().y).abs() < 4.0)
        .expect("the position row has an X chip")
        .1
        .center();
    let frame = |editor: &mut Harness, event: egui::Event| {
        editor.input.events.push(event);
        editor.frames(1);
        editor.input.events.clear();
        editor.state.apply(&mut editor.world, &editor.registry);
    };
    frame(editor, egui::Event::PointerMoved(chip));
    frame(editor, button(chip, true));
    let mut at = chip;
    for _ in 0..3 {
        at += egui::vec2(by / 3.0, 0.0);
        frame(editor, egui::Event::PointerMoved(at));
    }
    frame(editor, button(at, false));
}

/// The field touched is written to every selected entity and nothing else
/// is: each keeps its other values. One step undoes all of them.
#[test]
fn an_inspector_edit_reaches_every_selected_entity_and_only_its_field() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let a = editor.spawn("A");
    editor.world.insert(a, at(1.0, 2.0, 3.0));
    let b = editor.spawn("B");
    editor.state.selection.set(a);
    editor.state.selection.toggle(b);
    editor.frames(2);

    drag_position_x(&mut editor, 60.0);
    let moved = editor.world.get::<LocalTransform>(b).unwrap().translation;
    assert!(moved.x > 0.0, "the primary moved");
    let other = editor.world.get::<LocalTransform>(a).unwrap().translation;
    assert_eq!(other, glam::Vec3::new(moved.x, 2.0, 3.0));

    undo(&mut editor);
    assert_eq!(
        editor.world.get::<LocalTransform>(a).unwrap().translation,
        glam::Vec3::new(1.0, 2.0, 3.0)
    );
    assert_eq!(
        editor.world.get::<LocalTransform>(b).unwrap().translation,
        glam::Vec3::ZERO
    );
    assert_eq!(editor.state.history.undo_label(), None, "one step");
}

/// Adding a component adds it to every selected entity, as one step.
#[test]
fn add_component_reaches_the_whole_selection_in_one_step() {
    let mut editor = Harness::new(egui::vec2(1440.0, 900.0));
    let a = editor.spawn("A");
    let b = editor.spawn("B");
    editor.state.selection.set(a);
    editor.state.selection.toggle(b);
    editor.frames(3);
    open_add_component(&mut editor);
    editor.input.events.push(egui::Event::Text("grad".into()));
    editor.frames(1);
    editor.input.events.clear();
    editor.frames(1);
    editor.key(egui::Key::Enter);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(editor.world.has::<crate::ui::Gradient>(a));
    assert!(editor.world.has::<crate::ui::Gradient>(b));

    undo(&mut editor);
    assert!(!editor.world.has::<crate::ui::Gradient>(a));
    assert!(!editor.world.has::<crate::ui::Gradient>(b));
}

/// A section's Remove takes it from every selected entity that has it.
#[test]
fn remove_component_reaches_the_whole_selection_in_one_step() {
    let mut editor = Harness::new(egui::vec2(1440.0, 900.0));
    let a = editor.spawn("A");
    let b = editor.spawn("B");
    let c = editor.spawn("C");
    for entity in [a, b] {
        editor.world.insert(entity, crate::scene::Tag::new("crate"));
    }
    editor.state.selection.set(c);
    editor.state.selection.toggle(a);
    editor.state.selection.toggle(b);
    editor.frames(3);
    let title = text_at(&editor, "Tag").expect("the Tag section");
    let inspector = editor.tab_body(Tab::Inspector);
    let band =
        egui::Rect::from_x_y_ranges(inspector.x_range(), (title.y - 10.0)..=(title.y + 10.0));
    let menu = icon_in(&editor, crate::editor::icons::ellipsis(), band).expect("its menu");
    editor.click(menu).frames(1);
    editor.click(text_at(&editor, "Remove component").unwrap());
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(!editor.world.has::<crate::scene::Tag>(a));
    assert!(!editor.world.has::<crate::scene::Tag>(b));

    undo(&mut editor);
    assert!(editor.world.has::<crate::scene::Tag>(a));
    assert!(editor.world.has::<crate::scene::Tag>(b));
    assert_eq!(editor.state.history.undo_label(), None, "one step");
}

// A material asset, edited where it is marked.

const BRICK: &[u8] = b"orrin-material 1 min-reader 1\n\nroughness = 0.25\n";

fn brick_project(
    editor: &mut Harness,
    name: &str,
) -> (
    crate::asset_db::tests::Dir,
    orrin_registry::AssetId,
    orrin_registry::AssetId,
) {
    let dir = crate::asset_db::tests::Dir::new(name);
    dir.write("brick.omat", BRICK);
    dir.write("stone.png", b"");
    let (db, _) = crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
    let brick = db.find("brick.omat").unwrap().id;
    let stone = db.find("stone.png").unwrap().id;
    editor.world.insert_resource(db);
    (dir, brick, stone)
}

fn brick_on_disk(dir: &crate::asset_db::tests::Dir) -> crate::asset_db::material::MaterialFile {
    let text = std::fs::read_to_string(dir.path().join("brick.omat")).unwrap();
    crate::asset_db::material::parse(&text, "brick.omat").unwrap()
}

/// The value slot on the row labelled `label`: the widget text at its height,
/// right of the label.
fn slot_beside(editor: &Harness, label: &str, within: egui::Rect) -> egui::Pos2 {
    let rects = editor.painted_text_rects();
    let row = rects
        .iter()
        .find(|(text, rect)| text == label && within.contains(rect.center()))
        .unwrap_or_else(|| panic!("no {label} row"))
        .1;
    rects
        .iter()
        .filter(|(_, rect)| {
            (rect.center().y - row.center().y).abs() < 6.0 && rect.left() > row.right() + 4.0
        })
        .min_by(|a, b| a.1.left().total_cmp(&b.1.left()))
        .unwrap_or_else(|| panic!("nothing beside {label}"))
        .1
        .center()
}

/// The Import tab is where a material nothing draws with yet is edited: the
/// inspector's rows, through the same undoable path, saved to its file.
#[test]
fn the_import_tab_edits_the_marked_material() {
    let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
    let (dir, brick, stone) = brick_project(&mut editor, "authoring-import-material");
    editor.state.revealed_asset = Some(brick);
    editor.dock.activate(Tab::Import);
    editor.frames(2);

    for row in [
        "Base colour",
        "Metallic",
        "Roughness",
        "Emission",
        "Albedo map",
        "Shader",
    ] {
        assert!(
            editor.painted_contains(row),
            "{row}: {:?}",
            editor.painted()
        );
    }
    assert!(!editor.painted_contains("Edit it in the inspector"));

    let slot = slot_beside(&editor, "Albedo map", editor.tab_body(Tab::Import));
    drop_payload(&mut editor, stone, slot);
    assert_eq!(brick_on_disk(&dir).textures.albedo, Some(stone));

    undo(&mut editor);
    assert_eq!(brick_on_disk(&dir).textures.albedo, None);
}

// Slots that open a list.

// The spot light.

/// Pick `kind` from the Light section's Type row.
fn retype(editor: &mut Harness, from: &str, kind: &str) {
    let inspector = editor.tab_body(Tab::Inspector);
    let combo = slot_beside(editor, "Type", inspector);
    let shown = editor
        .painted_text_rects()
        .into_iter()
        .any(|(text, rect)| text == from && (rect.center() - combo).length() < 40.0);
    assert!(shown, "the Type row reads {from}");
    editor.click(combo).frames(1);
    let choice = editor
        .painted_text_rects()
        .into_iter()
        .filter(|(text, _)| text == kind)
        .max_by(|a, b| a.1.top().total_cmp(&b.1.top()))
        .expect("the kind is offered")
        .1
        .center();
    editor.click(choice);
    editor.state.apply(&mut editor.world, &editor.registry);
    editor.frames(2);
}

/// Switching kind keeps what the two kinds share and takes the engine's
/// default for the rest; lux and lumens are never converted into each other.
#[test]
fn the_light_type_row_switches_kind_keeping_what_the_kinds_share() {
    let mut editor = Harness::new(egui::vec2(1440.0, 900.0));
    let lamp = editor.spawn("Lamp");
    let colour = glam::Vec3::new(1.0, 0.5, 0.25);
    editor.world.insert(
        lamp,
        Light::Point {
            color: colour,
            lumens: 1234.0,
            range: 7.0,
            casts_shadows: false,
        },
    );
    editor.state.selection.set(lamp);
    editor.frames(2);

    retype(&mut editor, "Point", "Spot");
    let Some(Light::Spot {
        color,
        lumens,
        range,
        casts_shadows,
        inner_angle,
        outer_angle,
        reflector,
    }) = editor.world.get::<Light>(lamp).map(|l| *l)
    else {
        panic!("it is a spot now");
    };
    assert_eq!(
        (color, lumens, range, casts_shadows),
        (colour, 1234.0, 7.0, false)
    );
    let Light::Spot {
        inner_angle: inner,
        outer_angle: outer,
        reflector: reflects,
        ..
    } = Light::default_spot()
    else {
        unreachable!()
    };
    assert_eq!(
        (inner_angle, outer_angle, reflector),
        (inner, outer, reflects)
    );

    retype(&mut editor, "Spot", "Directional");
    let Some(Light::Directional { color, lux }) = editor.world.get::<Light>(lamp).map(|l| *l)
    else {
        panic!("it is a sun now");
    };
    let Light::Directional {
        lux: default_lux, ..
    } = Light::default()
    else {
        unreachable!()
    };
    assert_eq!((color, lux), (colour, default_lux), "lumens are not lux");

    undo(&mut editor);
    undo(&mut editor);
    assert!(matches!(
        editor.world.get::<Light>(lamp).map(|l| *l),
        Some(Light::Point { lumens, .. }) if lumens == 1234.0
    ));
}
