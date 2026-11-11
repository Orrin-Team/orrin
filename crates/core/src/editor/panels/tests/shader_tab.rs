//! The Shader tab and the material inspector's shader rows, driven through the
//! whole editor as the other panel tests are.

use std::time::Duration;

use orrin_registry::AssetId;

use super::*;
use crate::asset_db::material::{self, MaterialFile, ShaderValue};
use crate::asset_db::tests::Dir;
use crate::asset_db::{AssetDatabase, Status as Import};
use crate::editor::panels::shader::{ShaderEditor, code_id};
use crate::gfx::HeadlessBackend;
use crate::gfx::vulkan::material_preview::Viewer;

const GLOW: &str = "5f0c3a4e-0000-4000-8000-0000000000d1";
const BRICK: &str = "5f0c3a4e-0000-4000-8000-0000000000d2";
const ROCK: &str = "5f0c3a4e-0000-4000-8000-0000000000d3";

const TINT: &str = "\
param color tint = (1, 1, 1, 1);
void surface(in SurfaceIn i, inout SurfaceOut o) {
    o.albedo = tint.rgb;
}
";

/// An undeclared name on line 3.
const BROKEN: &str = "\
param color tint = (1, 1, 1, 1);
void surface(in SurfaceIn i, inout SurfaceOut o) {
    o.albedo = tint.rgb * nothing_by_this_name;
}
";

/// One of every kind of parameter.
const EVERY_KIND: &str = "\
param float wear = 0.5 : range(0, 1);
param float speed = 2.0;
param vec2 offset = (0.25, 0.75);
param vec3 axis = (0, 1, 0);
param vec4 weights = (1, 2, 3, 4);
param color tint = (1.0, 0.5, 0.25, 1.0);
param texture mask = white;
void surface(in SurfaceIn i, inout SurfaceOut o) {
    o.albedo = tint.rgb * wear * sample_param(mask, i.uv + offset).r * speed;
    o.roughness = dot(axis, weights.xyz);
}
";

fn glow() -> AssetId {
    GLOW.parse().unwrap()
}

fn brick() -> AssetId {
    BRICK.parse().unwrap()
}

fn rock() -> AssetId {
    ROCK.parse().unwrap()
}

/// A project holding `fx/glow.oshader` with `shader` in it and, when given,
/// `brick.omat` holding `material`, both under known ids.
fn project(name: &str, shader: &str, material: Option<&MaterialFile>) -> Dir {
    let dir = Dir::new(name);
    dir.write("fx/glow.oshader", shader.as_bytes());
    dir.write(
        "fx/glow.oshader.meta",
        format!("id = \"{GLOW}\"\n").as_bytes(),
    );
    if let Some(file) = material {
        dir.write("brick.omat", material::write(file).as_bytes());
        dir.write("brick.omat.meta", format!("id = \"{BRICK}\"\n").as_bytes());
    }
    dir
}

/// The editor at a size whose bottom dock is wide enough for the code and
/// its side column, on `dir`'s project.
fn editor_on(dir: &Dir) -> Harness {
    let mut editor = Harness::new(egui::vec2(1600.0, 1000.0));
    let db = crate::asset_db::tests::open(dir);
    editor.world.insert_resource(db);
    settle(&mut editor);
    editor
}

/// Let every import finish and land, as frames passing would.
fn settle(editor: &mut Harness) -> Vec<String> {
    let mut backend = HeadlessBackend::new();
    let mut errors = Vec::new();
    for _ in 0..3 {
        assert!(
            editor
                .world
                .resource::<AssetDatabase>()
                .wait_idle(Duration::from_secs(60))
        );
        errors.extend(crate::asset_db::tests::update(&editor.world, &mut backend));
    }
    errors
}

fn shader(editor: &Harness) -> &ShaderEditor {
    &editor.state.shader
}

/// Open the shader in the tab, as the inspector's Edit does.
fn open_glow(editor: &mut Harness) {
    editor.state.open_shader(glow());
    editor.frames(3);
    assert_eq!(shader(editor).open(), Some(glow()));
}

/// Where `label` was painted inside the Shader tab.
fn in_tab(editor: &Harness, label: &str) -> egui::Pos2 {
    let body = editor.tab_body(Tab::Shader);
    editor
        .painted_text_rects()
        .into_iter()
        .find(|(text, rect)| without_icons(text) == label && body.contains(rect.center()))
        .unwrap_or_else(|| panic!("no {label:?} in the Shader tab: {:?}", editor.painted()))
        .1
        .center()
}

fn focus_code(editor: &mut Harness) {
    editor
        .ctx
        .memory_mut(|memory| memory.request_focus(code_id()));
}

fn type_text(editor: &mut Harness, text: &str) {
    focus_code(editor);
    editor.input.events.push(egui::Event::Text(text.to_owned()));
    editor.frames(1);
    editor.input.events.clear();
}

/// Replace the whole text, as select-all and paste would.
fn replace_text(editor: &mut Harness, text: &str) {
    focus_code(editor);
    editor.key_with(egui::Modifiers::COMMAND, egui::Key::A);
    focus_code(editor);
    editor
        .input
        .events
        .push(egui::Event::Paste(text.to_owned()));
    editor.frames(1);
    editor.input.events.clear();
}

fn on_disk(dir: &Dir) -> String {
    std::fs::read_to_string(dir.path().join("fx/glow.oshader")).unwrap()
}

/// Another program's save, heard as the watcher would hear it.
fn save_elsewhere(editor: &mut Harness, dir: &Dir, text: &str) {
    let path = dir.write("fx/glow.oshader", text.as_bytes());
    let mut db = editor.world.resource_mut::<AssetDatabase>();
    db.notice(path);
    db.apply_changes();
}

#[test]
fn with_nothing_open_the_tab_says_what_a_shader_is_and_offers_one() {
    let dir = Dir::new("shader-tab-empty");
    let mut editor = editor_on(&dir);
    editor.dock.bring_forward(Tab::Shader);
    editor.frames(2);
    assert!(editor.painted_contains("A surface shader is a short GLSL function"));
    in_tab(&editor, "Open a shader");

    let at = in_tab(&editor, "New surface shader");
    editor.click(at).frames(2);
    let db = editor.world.resource::<AssetDatabase>();
    let created = db.find("New Surface Shader.oshader").expect("written").id;
    drop(db);
    assert_eq!(shader(&editor).open(), Some(created));
    assert_eq!(shader(&editor).text(), crate::asset_db::shader::NEW_SHADER);
}

#[test]
fn a_shader_double_clicked_in_the_assets_tab_opens_in_the_shader_tab() {
    let dir = project("shader-tab-open", TINT, None);
    let mut editor = editor_on(&dir);
    editor.dock.activate(Tab::Assets);
    editor.frames(2);
    editor.click(row_of(&editor, "fx")).frames(2);
    editor
        .double_click(row_of(&editor, "glow.oshader"))
        .frames(3);

    assert_eq!(shader(&editor).open(), Some(glow()));
    assert!(editor.painted_contains("o.albedo = tint.rgb;"));
    let (surface, node, _) = editor.dock.state().find_tab(&Tab::Shader).unwrap();
    let egui_dock::Node::Leaf { tabs, active, .. } = &editor.dock.state()[surface][node] else {
        panic!("in a leaf");
    };
    assert_eq!(tabs[active.0], Tab::Shader, "brought forward");
}

#[test]
fn a_shaders_menu_in_the_assets_tab_edits_it() {
    let dir = project("shader-tab-menu", TINT, None);
    let mut editor = editor_on(&dir);
    editor.dock.activate(Tab::Assets);
    editor.frames(2);
    editor.click(row_of(&editor, "fx")).frames(2);
    editor
        .right_click(row_of(&editor, "glow.oshader"))
        .frames(1);
    let edit = editor
        .painted_text_rects()
        .into_iter()
        .find(|(text, rect)| text == "Edit" && rect.top() > BAR_HEIGHT)
        .expect("the menu offers Edit")
        .1;
    editor.click(edit.center()).frames(3);
    assert_eq!(shader(&editor).open(), Some(glow()));
}

/// Wherever the browser offers a new material, it offers a new shader.
#[test]
fn the_assets_tab_makes_a_new_shader_where_it_makes_a_new_material() {
    let dir = project("shader-tab-new", TINT, None);
    let mut editor = editor_on(&dir);
    editor.dock.activate(Tab::Assets);
    editor.frames(2);
    editor.right_click(row_of(&editor, "fx")).frames(1);
    let texts = editor.painted();
    assert!(texts.iter().any(|text| text == "New material"), "{texts:?}");
    let new = editor
        .painted_text_rects()
        .into_iter()
        .find(|(text, _)| text == "New surface shader")
        .expect("offered beside New material")
        .1;
    editor.click(new.center()).frames(2);
    let db = editor.world.resource::<AssetDatabase>();
    let entry = db.find("fx/New Surface Shader.oshader").expect("written");
    assert_eq!(editor.state.revealed_asset, Some(entry.id), "and marked");
}

/// Typing marks the buffer, Save writes it through the database, which says
/// Compiling until the import lands and then how it went; Revert reads the
/// file back.
#[test]
fn typing_saving_and_reverting() {
    let dir = project("shader-tab-save", TINT, None);
    let mut editor = editor_on(&dir);
    open_glow(&mut editor);
    assert!(editor.painted_contains("Compiled"));
    assert!(!shader(&editor).is_dirty());

    type_text(&mut editor, "// edited\n");
    assert!(shader(&editor).is_dirty());
    assert!(shader(&editor).text().contains("// edited"));
    editor.frames(1);

    let typed = shader(&editor).text().to_owned();
    editor.click(in_tab(&editor, "Save")).frames(1);
    assert_eq!(on_disk(&dir), typed);
    assert!(!shader(&editor).is_dirty());
    assert_eq!(
        editor
            .world
            .resource::<AssetDatabase>()
            .get(glow())
            .unwrap()
            .status,
        Import::Queued
    );
    assert!(editor.painted_contains("Compiling"));
    settle(&mut editor);
    editor.frames(1);
    assert!(editor.painted_contains("Compiled"));

    type_text(&mut editor, "junk");
    assert!(shader(&editor).is_dirty());
    // A frame for the bar to hear the text changed: egui hits a click
    // against the widgets as the frame before laid them out.
    editor.frames(1);
    editor.click(in_tab(&editor, "Revert")).frames(1);
    assert_eq!(shader(&editor).text(), typed);
    assert!(!shader(&editor).is_dirty());
}

/// A save that does not compile lists its errors under the code, each at its
/// line, and a click on one puts the cursor on that line.
#[test]
fn errors_are_listed_by_line_and_lead_to_it() {
    let dir = project("shader-tab-errors", TINT, None);
    let mut editor = editor_on(&dir);
    open_glow(&mut editor);
    replace_text(&mut editor, BROKEN);
    assert_eq!(shader(&editor).text(), BROKEN);
    focus_code(&mut editor);
    editor.key_with(egui::Modifiers::COMMAND, egui::Key::S);
    assert_eq!(on_disk(&dir), BROKEN, "Cmd+S in the code saves it");
    let logged = settle(&mut editor);
    assert!(logged[0].starts_with("fx/glow.oshader:3: "), "{logged:?}");
    editor
        .ctx
        .memory_mut(|memory| memory.surrender_focus(code_id()));
    editor.frames(2);

    let count = editor
        .world
        .resource::<AssetDatabase>()
        .shader_errors(glow())
        .len();
    let status = if count == 1 {
        "1 error".to_owned()
    } else {
        format!("{count} errors")
    };
    assert!(editor.painted_contains(&status), "{:?}", editor.painted());
    assert!(editor.painted_contains("In the last save"));
    let entry = editor
        .painted_text_rects()
        .into_iter()
        .find(|(text, _)| text.starts_with("3: "))
        .expect("listed at its line")
        .1;

    editor.click(entry.center());
    assert_eq!(shader(&editor).goto(), Some(3));
    editor.frames(1);
    assert_eq!(shader(&editor).goto(), None);
    let cursor = egui::TextEdit::load_state(&editor.ctx, code_id())
        .and_then(|state| state.cursor.char_range())
        .expect("the code has a cursor");
    let line_three = BROKEN
        .lines()
        .take(2)
        .map(|l| l.chars().count() + 1)
        .sum::<usize>();
    assert_eq!(cursor.primary.index, line_three);
    assert!(editor.ctx.memory(|memory| memory.has_focus(code_id())));

    // Once the text moves on, the list says what it describes.
    type_text(&mut editor, " ");
    editor.frames(1);
    assert!(editor.painted_contains("From the last save; the text has changed since"));
}

/// The file is the truth: a clean buffer follows another program's save, and
/// a dirty one keeps what was typed and says the file moved.
#[test]
fn a_clean_buffer_follows_the_file_and_a_dirty_one_keeps_its_text() {
    let dir = project("shader-tab-disk", TINT, None);
    let mut editor = editor_on(&dir);
    open_glow(&mut editor);

    let elsewhere = format!("// from another editor\n{TINT}");
    save_elsewhere(&mut editor, &dir, &elsewhere);
    editor.frames(2);
    assert_eq!(shader(&editor).text(), elsewhere);
    assert!(!shader(&editor).changed_on_disk());

    type_text(&mut editor, "// mine\n");
    let mine = shader(&editor).text().to_owned();
    let again = format!("// and again, longer this time\n{TINT}");
    save_elsewhere(&mut editor, &dir, &again);
    editor
        .ctx
        .memory_mut(|memory| memory.surrender_focus(code_id()));
    editor.frames(2);
    assert_eq!(shader(&editor).text(), mine, "typed text is never dropped");
    assert!(shader(&editor).changed_on_disk());
    assert!(editor.painted_contains("The file changed on disk"));

    // Save waits on a choice rather than landing on the newer file.
    editor.click(in_tab(&editor, "Save")).frames(1);
    assert_eq!(on_disk(&dir), again);

    editor.click(in_tab(&editor, "Reload")).frames(1);
    assert_eq!(shader(&editor).text(), again);
    assert!(!shader(&editor).is_dirty());
    assert!(!shader(&editor).changed_on_disk());
}

/// Another shader asked for over unsaved text waits on what becomes of it.
#[test]
fn opening_another_shader_over_unsaved_text_asks_first() {
    let dir = project("shader-tab-switch", TINT, None);
    dir.write("other.oshader", TINT.as_bytes());
    let mut editor = editor_on(&dir);
    open_glow(&mut editor);
    type_text(&mut editor, "// mine\n");
    let other = editor
        .world
        .resource::<AssetDatabase>()
        .find("other.oshader")
        .unwrap()
        .id;
    editor.state.open_shader(other);
    editor
        .ctx
        .memory_mut(|memory| memory.surrender_focus(code_id()));
    editor.frames(2);
    assert_eq!(shader(&editor).open(), Some(glow()));
    assert!(editor.painted_contains("Opening other would lose your unsaved text."));

    editor.click(in_tab(&editor, "Save first")).frames(1);
    assert!(on_disk(&dir).starts_with("// mine\n") || on_disk(&dir).contains("// mine"));
    assert_eq!(shader(&editor).open(), Some(other));
}

/// The code is laid out once per text, not once per frame.
#[test]
fn the_code_is_highlighted_once_per_text() {
    let dir = project("shader-tab-highlight", TINT, None);
    let mut editor = editor_on(&dir);
    open_glow(&mut editor);
    let settled = shader(&editor).highlight_runs();
    assert!(settled >= 1);
    editor.frames(6);
    assert_eq!(shader(&editor).highlight_runs(), settled);
    type_text(&mut editor, "x");
    let typed = shader(&editor).highlight_runs();
    assert!(typed > settled);
    editor.frames(4);
    assert_eq!(shader(&editor).highlight_runs(), typed);
}

/// The sphere is a material that uses the shader, or the shader's own one at
/// its defaults when none does.
#[test]
fn the_preview_is_a_material_using_the_shader_or_its_defaults() {
    let dir = project("shader-tab-preview", TINT, None);
    let mut editor = editor_on(&dir);
    open_glow(&mut editor);
    settle(&mut editor);
    editor.frames(2);
    let asked = editor.state.material_previews[Viewer::Shader as usize]
        .expect("the tab asks for its sphere")
        .material;
    let own = editor
        .world
        .resource_mut::<AssetDatabase>()
        .shader_preview_material(glow())
        .unwrap();
    assert_eq!(asked, own);
    assert!(editor.painted_contains("The shader's defaults"));

    let users = MaterialFile {
        shader: Some(glow()),
        ..MaterialFile::default()
    };
    dir.write("brick.omat", material::write(&users).as_bytes());
    dir.write("brick.omat.meta", format!("id = \"{BRICK}\"\n").as_bytes());
    editor.world.resource_mut::<AssetDatabase>().rescan();
    editor.frames(2);
    let asked = editor.state.material_previews[Viewer::Shader as usize]
        .unwrap()
        .material;
    let brick = editor
        .world
        .resource_mut::<AssetDatabase>()
        .material(brick())
        .unwrap();
    assert_eq!(asked, brick);
}

/// What the last good compile declared, with each type, default and hint.
#[test]
fn the_tab_lists_what_the_shader_declares() {
    let dir = project("shader-tab-params", EVERY_KIND, None);
    let mut editor = editor_on(&dir);
    editor.input.screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(1600.0, 1800.0),
    ));
    open_glow(&mut editor);
    for line in [
        "float = 0.5, 0 to 1",
        "float = 2",
        "vec2 = (0.25, 0.75)",
        "vec3 = (0, 1, 0)",
        "color = (1, 0.5, 0.25, 1)",
        "texture = white, sRGB",
    ] {
        assert!(
            editor.painted_contains(line),
            "{line}: {:?}",
            editor.painted()
        );
    }
}

/// While the code has focus, keys are the text's: no tool, delete, scene
/// undo, play or scene save — and egui says it wants the keyboard, which is
/// what keeps them from the game and the fly camera.
#[test]
fn keys_typed_into_the_code_stay_in_the_code() {
    let dir = project("shader-tab-keys", TINT, None);
    let mut editor = editor_on(&dir);
    editor.state.scene_path = dir
        .path()
        .join("scene.orrin")
        .to_string_lossy()
        .into_owned();
    let doomed = editor.spawn("Doomed");
    editor.state.selection.set(doomed);
    editor.frames(1).key(egui::Key::Delete);
    editor.state.apply(&mut editor.world, &editor.registry);
    assert!(!editor.world.is_alive(doomed));
    let kept = editor.spawn("Kept");
    editor.state.selection.set(kept);
    open_glow(&mut editor);

    for (modifiers, key) in [
        (egui::Modifiers::NONE, egui::Key::Num2),
        (egui::Modifiers::NONE, egui::Key::Delete),
        (egui::Modifiers::NONE, egui::Key::F),
        (egui::Modifiers::NONE, egui::Key::T),
        (egui::Modifiers::COMMAND, egui::Key::Z),
        (egui::Modifiers::COMMAND, egui::Key::P),
        (egui::Modifiers::COMMAND, egui::Key::S),
    ] {
        focus_code(&mut editor);
        editor.key_with(modifiers, key);
        assert!(editor.ctx.wants_keyboard_input(), "{key:?}");
        editor.state.apply(&mut editor.world, &editor.registry);
    }
    assert_eq!(editor.state.gizmo_mode, GizmoMode::Select);
    assert!(!editor.state.ui_mode);
    assert!(editor.world.is_alive(kept), "not deleted");
    assert!(
        named(&editor.world, "Doomed").is_none(),
        "the scene's undo was not run"
    );
    assert!(!editor.state.playing());
    assert!(
        !dir.path().join("scene.orrin").exists(),
        "the scene was not saved"
    );
}

/// Cmd/Ctrl+Z in the code is the text's own undo.
#[test]
fn undo_in_the_code_undoes_the_typing() {
    let dir = project("shader-tab-undo", TINT, None);
    let mut editor = editor_on(&dir);
    open_glow(&mut editor);
    type_text(&mut editor, "abc");
    assert!(shader(&editor).text().contains("abc"));
    focus_code(&mut editor);
    editor.key_with(egui::Modifiers::COMMAND, egui::Key::Z);
    assert_eq!(shader(&editor).text(), TINT);
}

/// Away from the code, the editor's keys work as before.
#[test]
fn away_from_the_code_the_editors_keys_are_its_own() {
    let dir = project("shader-tab-keys-away", TINT, None);
    let mut editor = editor_on(&dir);
    open_glow(&mut editor);
    editor
        .ctx
        .memory_mut(|memory| memory.surrender_focus(code_id()));
    editor.key(egui::Key::Num2);
    assert_eq!(editor.state.gizmo_mode, GizmoMode::Move);
}

#[test]
fn the_view_menu_opens_the_shader_tab() {
    let dir = Dir::new("shader-tab-view");
    let mut editor = editor_on(&dir);
    editor.dock.toggle(Tab::Shader);
    assert!(!editor.dock.is_open(Tab::Shader));
    let Harness {
        world,
        registry,
        state,
        themes,
        dock,
        ctx,
        input,
        ..
    } = &mut editor;
    let mut draw = |input: egui::RawInput| {
        ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                menu_bar::menu_body(
                    menu_bar::Menu::View,
                    ui,
                    world,
                    registry,
                    state,
                    themes,
                    dock,
                );
            });
        })
    };
    let output = draw(input.clone());
    let output = {
        let _ = output;
        draw(input.clone())
    };
    let at = painted_text_rects_of(&output)
        .into_iter()
        .find(|(text, _)| text == "Shader")
        .expect("listed in View")
        .1
        .center();
    let mut click = input.clone();
    click.events = vec![
        egui::Event::PointerMoved(at),
        egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: Default::default(),
        },
        egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: Default::default(),
        },
    ];
    draw(click);
    assert!(dock.is_open(Tab::Shader));
}

fn painted_text_rects_of(output: &egui::FullOutput) -> Vec<(String, egui::Rect)> {
    fn collect(shape: &egui::Shape, out: &mut Vec<(String, egui::Rect)>) {
        match shape {
            egui::Shape::Text(text) => {
                out.push((text.galley.text().to_owned(), text.visual_bounding_rect()))
            }
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    for clipped in &output.shapes {
        collect(&clipped.shape, &mut out);
    }
    out
}

/// The editor's two promises about the middle of the window hold with the
/// Shader tab open on a shader: it paints nothing there and claims nothing.
#[test]
fn the_shader_tab_leaves_the_scene_alone() {
    let dir = project("shader-tab-viewport", TINT, None);
    let mut editor = editor_on(&dir);
    editor.spawn("Ground plane");
    open_glow(&mut editor);
    editor.frames(3);
    assert!(editor.dock.is_open(Tab::Shader));

    let viewport = editor.viewport();
    let overlays: Vec<egui::Rect> = ["viewport_tools", "viewport_view", "viewport_stats"]
        .into_iter()
        .filter_map(|id| editor.area(id))
        .collect();
    let covering: Vec<_> = editor
        .painted_rects()
        .into_iter()
        .filter(|rect| viewport.contains(rect.center()))
        .filter(|rect| !overlays.iter().any(|overlay| overlay.contains_rect(*rect)))
        .collect();
    assert!(covering.is_empty(), "{covering:?}");

    editor.hover(viewport.center()).frames(2);
    assert!(!editor.dock.wants_pointer(&editor.ctx));
    editor
        .hover(editor.tab_body(Tab::Shader).center())
        .frames(2);
    assert!(editor.dock.wants_pointer(&editor.ctx));
}

/// The inspector's Material section on `brick.omat`, drawn on a selected
/// entity, with the shader compiled.
fn inspecting_brick(name: &str, shader: &str, file: &MaterialFile) -> (Dir, Harness) {
    let dir = project(name, shader, Some(file));
    let mut editor = editor_on(&dir);
    editor.world.insert_resource(crate::scene::Assets::new());
    let handle = editor
        .world
        .resource_mut::<AssetDatabase>()
        .material(brick())
        .unwrap();
    let entity = editor.spawn("Crate");
    editor.world.insert(entity, crate::scene::MeshHandle(0));
    editor.world.insert(entity, handle);
    settle(&mut editor);
    editor.state.selection.set(entity);
    editor.frames(3);
    (dir, editor)
}

fn brick_file(editor: &Harness) -> MaterialFile {
    editor
        .world
        .resource::<AssetDatabase>()
        .material_file(brick())
        .cloned()
        .unwrap()
}

fn in_inspector(editor: &Harness, label: &str) -> egui::Rect {
    let body = editor.tab_body(Tab::Inspector);
    editor
        .painted_text_rects()
        .into_iter()
        .find(|(text, rect)| without_icons(text) == label && body.contains(rect.center()))
        .unwrap_or_else(|| panic!("no {label:?} in the inspector: {:?}", editor.painted()))
        .1
}

fn apply(editor: &mut Harness) {
    editor.state.apply(&mut editor.world, &editor.registry);
}

/// Scroll the inspector to its end, where the Material section's shader rows are.
fn scroll_inspector(editor: &mut Harness) {
    let at = editor.tab_body(Tab::Inspector).center();
    editor.input.events.push(egui::Event::PointerMoved(at));
    editor.input.events.push(egui::Event::MouseWheel {
        unit: egui::MouseWheelUnit::Point,
        delta: egui::vec2(0.0, -4000.0),
        modifiers: Default::default(),
    });
    editor.frames(1);
    editor.input.events.clear();
    editor.frames(2);
}

#[test]
fn the_shader_row_sets_and_clears_the_shader_as_undoable_edits() {
    let (_dir, mut editor) = inspecting_brick("shader-row", TINT, &MaterialFile::default());
    scroll_inspector(&mut editor);
    in_inspector(&editor, "Shader");

    let slot = in_inspector(&editor, "None");
    editor.click(slot.center()).frames(1);
    let choice = editor
        .painted_text_rects()
        .into_iter()
        .find(|(text, _)| text == "glow")
        .expect("the picker offers the project's shaders")
        .1;
    editor.click(choice.center());
    apply(&mut editor);
    assert_eq!(brick_file(&editor).shader, Some(glow()));
    assert_eq!(editor.state.history.undo_label(), Some("Edit brick"));

    editor.state.request_undo();
    apply(&mut editor);
    assert_eq!(brick_file(&editor).shader, None);
    editor.state.request_redo();
    apply(&mut editor);
    assert_eq!(brick_file(&editor).shader, Some(glow()));

    editor.frames(2);
    scroll_inspector(&mut editor);
    editor
        .right_click(in_inspector(&editor, "glow").center())
        .frames(1);
    let clear = editor
        .painted_text_rects()
        .into_iter()
        .find(|(text, _)| text == "Clear shader")
        .expect("the slot's menu clears it")
        .1;
    editor.click(clear.center());
    apply(&mut editor);
    assert_eq!(brick_file(&editor).shader, None);
}

#[test]
fn the_shader_rows_edit_opens_it_in_the_shader_tab() {
    let file = MaterialFile {
        shader: Some(glow()),
        ..MaterialFile::default()
    };
    let (_dir, mut editor) = inspecting_brick("shader-row-edit", TINT, &file);
    scroll_inspector(&mut editor);
    let edit = editor
        .painted_icons()
        .into_iter()
        .find(|(glyph, rect, _)| {
            *glyph == crate::editor::icons::arrow_square_out().glyph()
                && editor.tab_body(Tab::Inspector).contains(rect.center())
        })
        .expect("an Edit button on the Shader row")
        .1;
    editor.click(edit.center()).frames(3);
    assert_eq!(shader(&editor).open(), Some(glow()));
    assert!(editor.dock.is_open(Tab::Shader));
}

/// One row per declared parameter, in declaration order; a reset on a row
/// takes its value out of the file.
#[test]
fn the_parameter_group_follows_the_table_and_reset_removes_a_value() {
    let mut file = MaterialFile {
        shader: Some(glow()),
        ..MaterialFile::default()
    };
    file.set_shader_value("speed", Some(ShaderValue::Float(5.0)));
    let (_dir, mut editor) = inspecting_brick("shader-row-params", EVERY_KIND, &file);
    scroll_inspector(&mut editor);
    let rows: Vec<f32> = ["wear", "speed", "offset", "axis", "weights", "tint", "mask"]
        .into_iter()
        .map(|name| in_inspector(&editor, name).top())
        .collect();
    assert!(rows.windows(2).all(|pair| pair[0] < pair[1]), "{rows:?}");

    let resets: Vec<egui::Rect> = editor
        .painted_icons()
        .into_iter()
        .filter(|(glyph, rect, _)| {
            *glyph == crate::editor::icons::rotate_ccw().glyph()
                && editor.tab_body(Tab::Inspector).contains(rect.center())
        })
        .map(|(_, rect, _)| rect)
        .collect();
    assert_eq!(resets.len(), 1, "only the row with a value set");
    let speed = in_inspector(&editor, "speed");
    assert!((resets[0].center().y - speed.center().y).abs() < 6.0);
    editor.click(resets[0].center());
    apply(&mut editor);
    assert_eq!(brick_file(&editor).shader_value("speed"), None);
    assert!(editor.state.history.undo_label().is_some());
}

/// A texture dropped on a texture parameter's slot is its value.
#[test]
fn a_texture_dropped_on_a_texture_parameter_sets_it() {
    let file = MaterialFile {
        shader: Some(glow()),
        ..MaterialFile::default()
    };
    let dir = project("shader-row-texture", EVERY_KIND, Some(&file));
    dir.write(
        "rock.png",
        &crate::asset_db::import::tests::png(1, 1, |_, _| [10, 20, 30, 255]),
    );
    dir.write("rock.png.meta", format!("id = \"{ROCK}\"\n").as_bytes());
    let mut editor = editor_on(&dir);
    editor.world.insert_resource(crate::scene::Assets::new());
    let handle = editor
        .world
        .resource_mut::<AssetDatabase>()
        .material(brick())
        .unwrap();
    let entity = editor.spawn("Crate");
    editor.world.insert(entity, crate::scene::MeshHandle(0));
    editor.world.insert(entity, handle);
    settle(&mut editor);
    editor.state.selection.set(entity);
    editor.frames(3);
    scroll_inspector(&mut editor);

    let slot = in_inspector(&editor, "white (default)");
    drop_payload(&mut editor, rock(), slot.center());
    assert_eq!(
        brick_file(&editor).shader_value("mask"),
        Some(ShaderValue::Texture(rock()))
    );
}

/// A value the shader no longer declares is listed with a warning and can be
/// taken out of the file.
#[test]
fn an_undeclared_value_is_listed_and_removable() {
    let mut file = MaterialFile {
        shader: Some(glow()),
        ..MaterialFile::default()
    };
    file.set_shader_value("renamed", Some(ShaderValue::Float(1.0)));
    let (_dir, mut editor) = inspecting_brick("shader-row-undeclared", TINT, &file);
    scroll_inspector(&mut editor);
    let row = in_inspector(&editor, "renamed");
    assert!(editor.painted_contains("Not declared by glow"));
    let remove = editor
        .painted_icons()
        .into_iter()
        .find(|(glyph, rect, _)| {
            *glyph == crate::editor::icons::x().glyph()
                && (rect.center().y - row.center().y).abs() < 6.0
        })
        .expect("a remove button on the row")
        .1;
    editor.click(remove.center());
    apply(&mut editor);
    assert_eq!(brick_file(&editor).shader_value("renamed"), None);
}

#[test]
fn a_shader_that_never_compiled_says_so_in_place_of_its_parameters() {
    let file = MaterialFile {
        shader: Some(glow()),
        ..MaterialFile::default()
    };
    let (_dir, mut editor) = inspecting_brick("shader-row-never", BROKEN, &file);
    scroll_inspector(&mut editor);
    assert!(
        editor.painted_contains("glow has not compiled yet"),
        "{:?}",
        editor.painted()
    );
}
