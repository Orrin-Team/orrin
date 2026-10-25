// Visible to the dock, which mounts each tool's `body` into a tab.
pub(super) mod assets;
pub(super) mod console;
pub(super) mod environment;
pub(super) mod hierarchy;
pub(super) mod inspector;
mod menu_bar;
pub(super) mod performance;
#[cfg(feature = "scripting")]
pub(super) mod scripts;
mod status_bar;
mod viewport;

use glam::Vec3;

use orrin_ecs::World;
use orrin_registry::Registry;

use super::dock::Dock;
use super::phases::{self, Glass};
use super::previews::Previews;
use super::state::{EditorState, SceneRequest};
use super::theme::ThemeSet;
use super::widgets;

/// The menu bar across the top of the window.
pub(super) const BAR_HEIGHT: f32 = 32.0;

/// A dock's tab strip. Shorter than the bar above it: that one spans the
/// window and names the session, this one repeats above every tool.
pub(super) const TAB_BAR_HEIGHT: f32 = 26.0;

// The two bars are mounted first so they span the full window width.
// Everything between them belongs to the dock, whose centre paints nothing —
// that is what leaves the 3D scene visible behind the editor — and the
// viewport's toolbars float over that centre last, once the dock has said
// where it is.
pub fn draw(
    ctx: &egui::Context,
    world: &mut World,
    state: &mut EditorState,
    registry: &Registry,
    themes: &ThemeSet,
    dock: &mut Dock,
    previews: &mut Previews,
) {
    shortcuts(ctx, world, state);
    // Before the dock, so a command that rearranges the layout is reflected in
    // the same frame rather than the next one.
    menu_bar::show(ctx, world, registry, state, themes, dock);
    status_bar::show(ctx, world, themes);
    dock.show(ctx, world, state, registry, previews);
    if let Some(rect) = dock.viewport() {
        viewport::show(ctx, world, state, themes, rect);
    }
}

/// Every surface above the panels — the viewport's floating controls, menus,
/// popups, tooltips — has a translucent fill. The renderer blurs what is
/// beneath each into the rects collected here before drawing it, so the fill
/// reads as frosted glass, over a panel as over the scene. See
/// [`phases`](crate::editor::phases) for how "beneath" is drawn first.
///
/// Run after [`draw`], once every surface of the frame has been laid out.
pub fn frost(ctx: &egui::Context, glass: &mut Vec<Glass>) {
    glass.clear();
    let toolbars = [
        egui::Id::new("viewport_tools"),
        egui::Id::new("viewport_view"),
        egui::Id::new("viewport_stats"),
    ];
    let menu = f32::from(ctx.style().visuals.menu_corner_radius.nw);
    ctx.memory(|memory| {
        for layer in memory.layer_ids() {
            if !memory.areas().is_visible(&layer) {
                continue;
            }
            let radius = match layer.order {
                egui::Order::Foreground | egui::Order::Tooltip => menu,
                _ if toolbars.contains(&layer.id) => f32::from(widgets::TOOLBAR_RADIUS),
                _ => continue,
            };
            if let Some(rect) = memory.area_rect(layer.id) {
                glass.push(Glass {
                    phase: phases::of(layer.order),
                    rect,
                    radius,
                });
            }
        }
    });
}

/// The few keys every editor answers the same way. Each one stands aside while
/// a text field has focus, where the same key edits text.
fn shortcuts(ctx: &egui::Context, world: &World, state: &mut EditorState) {
    if ctx.input_mut(|input| input.consume_key(egui::Modifiers::COMMAND, egui::Key::S)) {
        state.request_scene(SceneRequest::Save);
    }
    if ctx.wants_keyboard_input() {
        return;
    }
    let selected = state.selected.filter(|&entity| world.is_alive(entity));
    if let Some(entity) = selected
        && ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Delete))
    {
        state.request_despawn(entity);
    }
    if selected.is_some()
        && ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
    {
        state.selected = None;
    }
}

/// Every number, path, id and log line is monospace, so columns of figures
/// align by eye rather than drifting with the width of each digit.
pub(super) fn figures(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text).monospace()
}

pub(super) fn vec3_row(ui: &mut egui::Ui, label: &str, v: &mut Vec3, speed: f32) -> bool {
    widgets::property(ui, label, |ui| {
        widgets::axes(ui, [&mut v.x, &mut v.y, &mut v.z], speed as f64, "")
    })
}

pub(super) fn color_row(ui: &mut egui::Ui, label: &str, c: &mut Vec3) {
    let mut rgb = [c.x, c.y, c.z];
    widgets::property(ui, label, |ui| {
        if ui.color_edit_button_rgb(&mut rgb).changed() {
            *c = Vec3::from(rgb);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::dock::{Dock, Tab};
    use crate::editor::theme;
    use crate::scene::{LocalTransform, Name};

    /// Icons are glyphs in the font's private-use range; a label that leads
    /// with one is compared by its words.
    fn without_icons(text: &str) -> String {
        text.chars()
            .filter(|c| !('\u{e000}'..='\u{f8ff}').contains(c))
            .collect()
    }

    fn themes_default() -> &'static theme::Theme {
        static THEME: std::sync::OnceLock<theme::Theme> = std::sync::OnceLock::new();
        THEME.get_or_init(theme::Theme::default)
    }

    fn painted_text(output: &egui::FullOutput) -> Vec<String> {
        fn collect(shape: &egui::Shape, out: &mut Vec<String>) {
            match shape {
                egui::Shape::Text(text) => out.push(without_icons(text.galley.text())),
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| collect(shape, out)),
                _ => {}
            }
        }
        let mut out = Vec::new();
        for clipped in &output.shapes {
            collect(&clipped.shape, &mut out);
        }
        out
    }

    /// The whole editor, laid out headlessly at a chosen window size.
    ///
    /// egui needs no GPU to place widgets, and placement is what these tests are
    /// about. Frames are explicit because a panel that sizes its content from
    /// its own width does not misbehave on frame one — it creeps, and only a
    /// later frame can see that.
    struct Harness {
        world: World,
        registry: Registry,
        state: EditorState,
        themes: ThemeSet,
        dock: Dock,
        previews: Previews,
        /// What the last frame asked the renderer to frost.
        glass: Vec<Glass>,
        ctx: egui::Context,
        input: egui::RawInput,
        output: Option<egui::FullOutput>,
    }

    impl Harness {
        fn new(size: egui::Vec2) -> Self {
            Self::build(size, true)
        }

        /// Without the image loaders registered, so every icon fails to load.
        /// egui gives an image-only button the whole available space when that
        /// happens, which a content-sized panel turns into unbounded growth.
        fn without_icons(size: egui::Vec2) -> Self {
            Self::build(size, false)
        }

        fn build(size: egui::Vec2, icons: bool) -> Self {
            let mut world = World::default();
            crate::App::install_default_resources(&mut world);
            let mut registry = Registry::new();
            crate::scene::register_components(&mut registry);

            let themes = ThemeSet::default();
            let ctx = egui::Context::default();
            if icons {
                crate::editor::icons::install(&ctx);
            }
            theme::install_fonts(&ctx);
            theme::apply(&ctx, themes.active());

            Self {
                world,
                registry,
                state: EditorState::default(),
                themes,
                dock: Dock::default(),
                previews: Previews::default(),
                glass: Vec::new(),
                ctx,
                input: egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
                    ..Default::default()
                },
                output: None,
            }
        }

        /// A scene entity as the editor expects one: named, and with the
        /// transform every inspector section past the first reads.
        fn spawn(&mut self, name: &str) -> orrin_ecs::Entity {
            let entity = self.world.spawn();
            self.world.insert(entity, Name(name.to_owned()));
            self.world.insert(entity, LocalTransform::default());
            entity
        }

        fn frames(&mut self, count: usize) -> &mut Self {
            for _ in 0..count {
                let Self {
                    world,
                    registry,
                    state,
                    themes,
                    dock,
                    previews,
                    glass,
                    ctx,
                    input,
                    output,
                } = self;
                *output = Some(ctx.run(input.clone(), |ctx| {
                    draw(ctx, world, state, registry, themes, dock, previews);
                    frost(ctx, glass);
                }));
            }
            self
        }

        /// Every string the last frame actually painted. The editor is drawn,
        /// not returned, so this is the only way to assert that a row reached
        /// the screen rather than merely being iterated over.
        fn painted(&self) -> Vec<String> {
            painted_text(self.output.as_ref().expect("a frame has run"))
        }

        /// Park the pointer, so hover-only affordances draw.
        fn hover(&mut self, at: egui::Pos2) -> &mut Self {
            self.input.events.push(egui::Event::PointerMoved(at));
            self.input.hovered_files.clear();
            self
        }

        /// The bounding rects of every painted shape `keep` accepts.
        fn painted_rects_of(&self, keep: impl Fn(&egui::Shape) -> bool) -> Vec<egui::Rect> {
            fn collect(
                shape: &egui::Shape,
                keep: &dyn Fn(&egui::Shape) -> bool,
                out: &mut Vec<egui::Rect>,
            ) {
                match shape {
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, keep, out)),
                    other if keep(other) => out.push(other.visual_bounding_rect()),
                    _ => {}
                }
            }
            let mut out = Vec::new();
            for clipped in &self.output.as_ref().expect("a frame has run").shapes {
                collect(&clipped.shape, &keep, &mut out);
            }
            out
        }

        /// Every plain filled rect with its fill and its corner radius.
        fn painted_corner_radii(&self) -> Vec<(egui::Rect, egui::Color32, egui::CornerRadius)> {
            fn collect(
                shape: &egui::Shape,
                out: &mut Vec<(egui::Rect, egui::Color32, egui::CornerRadius)>,
            ) {
                match shape {
                    egui::Shape::Rect(rect) if rect.brush.is_none() => {
                        out.push((rect.rect, rect.fill, rect.corner_radius));
                    }
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, out)),
                    _ => {}
                }
            }
            let mut out = Vec::new();
            for clipped in &self.output.as_ref().expect("a frame has run").shapes {
                collect(&clipped.shape, &mut out);
            }
            out
        }

        /// Every plain filled rect with its fill colour.
        fn painted_fills(&self) -> Vec<(egui::Rect, egui::Color32)> {
            fn collect(shape: &egui::Shape, out: &mut Vec<(egui::Rect, egui::Color32)>) {
                match shape {
                    egui::Shape::Rect(rect) if rect.brush.is_none() => {
                        out.push((rect.rect, rect.fill));
                    }
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, out)),
                    _ => {}
                }
            }
            let mut out = Vec::new();
            for clipped in &self.output.as_ref().expect("a frame has run").shapes {
                collect(&clipped.shape, &mut out);
            }
            out
        }

        /// Every icon glyph the frame painted: which, where, and in what colour.
        /// Icons are text in the icon font, so this is how one is found.
        fn painted_icons(&self) -> Vec<(char, egui::Rect, egui::Color32)> {
            fn collect(shape: &egui::Shape, out: &mut Vec<(char, egui::Rect, egui::Color32)>) {
                match shape {
                    egui::Shape::Text(text) => {
                        for section in &text.galley.job.sections {
                            if section.format.font_id.family != crate::editor::icons::family() {
                                continue;
                            }
                            let glyph = text.galley.job.text[section.byte_range.clone()]
                                .chars()
                                .next()
                                .unwrap_or_default();
                            let colour = if section.format.color == egui::Color32::PLACEHOLDER {
                                text.fallback_color
                            } else {
                                section.format.color
                            };
                            out.push((glyph, text.visual_bounding_rect(), colour));
                        }
                    }
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, out)),
                    _ => {}
                }
            }
            let mut out = Vec::new();
            for clipped in &self.output.as_ref().expect("a frame has run").shapes {
                collect(&clipped.shape, &mut out);
            }
            out
        }

        /// The font family a painted string was laid out in. `None` when that
        /// string never reached the screen.
        fn family_of(&self, needle: &str) -> Option<egui::FontFamily> {
            fn find(shape: &egui::Shape, needle: &str, out: &mut Option<egui::FontFamily>) {
                match shape {
                    egui::Shape::Text(text) if text.galley.text().contains(needle) => {
                        *out = text
                            .galley
                            .job
                            .sections
                            .first()
                            .map(|section| section.format.font_id.family.clone());
                    }
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| find(s, needle, out)),
                    _ => {}
                }
            }
            let mut out = None;
            for clipped in &self.output.as_ref().expect("a frame has run").shapes {
                find(&clipped.shape, needle, &mut out);
            }
            out
        }

        /// Every painted string with the rect it landed in.
        fn painted_text_rects(&self) -> Vec<(String, egui::Rect)> {
            fn collect(shape: &egui::Shape, out: &mut Vec<(String, egui::Rect)>) {
                match shape {
                    egui::Shape::Text(text) => {
                        out.push((text.galley.text().to_owned(), text.visual_bounding_rect()));
                    }
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, out)),
                    _ => {}
                }
            }
            let mut out = Vec::new();
            for clipped in &self.output.as_ref().expect("a frame has run").shapes {
                collect(&clipped.shape, &mut out);
            }
            out
        }

        fn painted_contains(&self, needle: &str) -> bool {
            self.painted().iter().any(|text| text.contains(needle))
        }

        /// Where a docked tool's leaf ended up. Only meaningful after a frame:
        /// egui_dock writes each node's rect while it draws.
        fn tab(&self, tab: crate::editor::dock::Tab) -> egui::Rect {
            let (surface, node, _) = self.dock.state().find_tab(&tab).expect("tab is open");
            self.dock.state()[surface][node]
                .rect()
                .expect("node was drawn")
        }

        /// A docked tool's *body*, without the tab strip above it.
        fn tab_body(&self, tab: Tab) -> egui::Rect {
            let (surface, node, _) = self.dock.state().find_tab(&tab).expect("tab is open");
            match &self.dock.state()[surface][node] {
                egui_dock::Node::Leaf { viewport, .. } => *viewport,
                _ => panic!("tab is not in a leaf"),
            }
        }

        /// Every shape the last frame painted, as bounding rects.
        fn painted_rects(&self) -> Vec<egui::Rect> {
            fn collect(shape: &egui::Shape, out: &mut Vec<egui::Rect>) {
                match shape {
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, out)),
                    egui::Shape::Noop => {}
                    other => out.push(other.visual_bounding_rect()),
                }
            }
            let mut out = Vec::new();
            for clipped in &self.output.as_ref().expect("a frame has run").shapes {
                collect(&clipped.shape, &mut out);
            }
            out
        }

        /// The gap the tools leave in the middle of the window — the scene.
        /// Inset, so a neighbour's own border or separator does not count as
        /// part of it. Only meaningful after a frame.
        fn viewport(&self) -> egui::Rect {
            let inset = 6.0;
            let viewport = egui::Rect::from_min_max(
                egui::pos2(
                    self.tab(Tab::Hierarchy).right() + inset,
                    self.panel("menu_bar").bottom() + inset,
                ),
                egui::pos2(
                    self.tab(Tab::Inspector).left() - inset,
                    self.tab(Tab::Console).top() - inset,
                ),
            );
            assert!(
                viewport.width() > 200.0 && viewport.height() > 200.0,
                "the layout left no viewport to check: {viewport:?}"
            );
            viewport
        }

        fn panel(&self, id: &str) -> egui::Rect {
            egui::containers::panel::PanelState::load(&self.ctx, egui::Id::new(id))
                .expect("panel was drawn")
                .rect
        }

        /// Where a floating overlay (an `egui::Area`) was placed last frame.
        fn area(&self, id: &str) -> Option<egui::Rect> {
            self.ctx
                .memory(|memory| memory.area_rect(egui::Id::new(id)))
        }

        /// A press and release at `at`, delivered over the next frame.
        fn click(&mut self, at: egui::Pos2) -> &mut Self {
            self.input.events.push(egui::Event::PointerMoved(at));
            for pressed in [true, false] {
                self.input.events.push(egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                });
            }
            self.frames(1);
            self.input.events.clear();
            self
        }

        fn key(&mut self, key: egui::Key) -> &mut Self {
            for pressed in [true, false] {
                self.input.events.push(egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed,
                    repeat: false,
                    modifiers: Default::default(),
                });
            }
            self.frames(1);
            self.input.events.clear();
            self
        }
    }

    /// The tree is the panel. A toolbar row that takes the panel's whole rect
    /// leaves it nothing to draw into, and nothing about that is a panic — the
    /// editor just comes up empty.
    #[test]
    fn the_hierarchy_draws_its_rows() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.spawn("Ground plane");
        editor.spawn("Key light");
        editor.frames(2);

        assert!(editor.painted_contains("Ground plane"));
        assert!(editor.painted_contains("Key light"));
    }

    /// Every row leads with what the entity is, in that kind's tint — the
    /// difference between finding the light in a tree of a thousand meshes by
    /// colour and finding it by reading.
    #[test]
    fn each_row_shows_its_kind_in_its_tint() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let lamp = editor.spawn("Lamp");
        editor.world.insert(lamp, crate::scene::Light::default());
        let crate_ = editor.spawn("Crate");
        editor.world.insert(crate_, crate::scene::MeshHandle(0));
        editor.frames(2);

        let tree = editor.tab(Tab::Hierarchy);
        let tints: Vec<egui::Color32> = editor
            .painted_icons()
            .into_iter()
            .filter(|(_, rect, _)| tree.contains(rect.center()))
            .map(|(_, _, tint)| tint)
            .collect();
        assert!(tints.contains(&theme::KIND_LIGHT), "{tints:?}");
        assert!(tints.contains(&theme::KIND_MESH), "{tints:?}");
    }

    #[test]
    fn searching_hides_the_rows_that_do_not_match() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.spawn("Ground plane");
        editor.spawn("Key light");
        editor.state.hierarchy_query = "key".to_owned();
        editor.frames(2);

        assert!(editor.painted_contains("Key light"));
        assert!(!editor.painted_contains("Ground plane"));
    }

    /// The reason filtering happens in the draw walk and not in `snapshot`.
    #[test]
    fn a_parent_survives_a_search_its_child_matches() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let parent = editor.spawn("Rig");
        let child = editor.spawn("Muzzle flash");
        crate::scene::reparent(&mut editor.world, child, Some(parent), false).unwrap();
        editor.state.hierarchy_query = "muzzle".to_owned();
        editor.frames(2);

        assert!(editor.painted_contains("Muzzle flash"));
        assert!(editor.painted_contains("Rig"));
    }

    /// The point of the label column: values start on one edge, whatever their
    /// labels say. Measured on the X chips, which every vector row draws and
    /// nothing else fills with that red.
    #[test]
    fn values_line_up_down_the_inspector() {
        let mut editor = Harness::new(egui::vec2(1440.0, 900.0));
        let entity = editor.spawn("Crate");
        editor
            .world
            .insert(entity, crate::scene::Spin::new(glam::Vec3::Y, 1.0));
        editor.state.selected = Some(entity);
        editor.frames(3);

        // Both edges of the vector columns: the X chips start on one line, and
        // so do the Z chips, whether or not a row has a reset button beside it.
        let inspector = editor.tab(Tab::Inspector);
        for (axis, colour) in ["X", "Z"].into_iter().zip([theme::AXIS[0], theme::AXIS[2]]) {
            let lefts: Vec<f32> = editor
                .painted_fills()
                .into_iter()
                .filter(|(rect, fill)| *fill == colour && inspector.contains(rect.center()))
                .map(|(rect, _)| rect.left())
                .collect();
            assert!(
                lefts.len() >= 4,
                "position, rotation, scale and the spin axis: {lefts:?}"
            );
            assert!(
                lefts.iter().all(|x| (x - lefts[0]).abs() < 0.5),
                "the {axis} chips start at different x: {lefts:?}"
            );
        }
    }

    /// A section bar runs edge to edge across the inspector, so the sections
    /// divide the panel rather than sitting as boxes inside it.
    #[test]
    fn inspector_sections_span_the_panel() {
        let mut editor = Harness::new(egui::vec2(1440.0, 900.0));
        let entity = editor.spawn("Crate");
        editor.state.selected = Some(entity);
        editor.frames(3);

        let body = editor.tab_body(Tab::Inspector);
        let bar = editor.ctx.style().visuals.window_fill.to_opaque();
        let widest = editor
            .painted_fills()
            .into_iter()
            .filter(|(rect, fill)| *fill == bar && body.contains(rect.center()))
            .map(|(rect, _)| rect.width())
            .fold(0.0, f32::max);
        assert!(
            (widest - body.width()).abs() <= 1.0,
            "the widest section bar is {widest}px of a {}px panel",
            body.width()
        );
    }

    /// egui reports two widgets sharing an id by painting "First use" and
    /// "Second use" over them in red, and the two then fight over one drag
    /// state. Nothing else about that is an error, so the warning text is what
    /// this looks for — across a scene with every kind of section on screen.
    #[test]
    fn no_two_widgets_share_an_id() {
        let mut editor = Harness::new(egui::vec2(1440.0, 900.0));
        let mut backend = crate::gfx::HeadlessBackend::new();
        crate::scene::entities::build_default_scene(&mut editor.world, &mut backend);
        let entity = editor
            .world
            .entities()
            .find(|&e| editor.world.has::<crate::scene::MaterialHandle>(e))
            .expect("the demo scene draws something");
        editor.state.selected = Some(entity);
        editor.frames(3);

        let clashes: Vec<String> = editor
            .painted()
            .into_iter()
            .filter(|text| text.contains("First use") || text.contains("Second use"))
            .collect();
        assert!(clashes.is_empty(), "{clashes:?}");
    }

    /// What can be done to the selection as a whole lives in the inspector
    /// header's menu — the one place that is always about the thing selected.
    #[test]
    fn the_inspector_menu_offers_delete_for_the_selection() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let entity = editor.spawn("Ground plane");
        editor.state.selected = Some(entity);
        let Harness {
            world,
            registry,
            state,
            ctx,
            input,
            ..
        } = &mut editor;

        let output = ctx.run(input.clone(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                inspector::entity_menu(ui, world, registry, state, entity);
            });
        });
        let painted = painted_text(&output);
        assert!(painted.iter().any(|text| text == "Delete"), "{painted:?}");
        assert!(painted.iter().any(|text| text == "Dump to console"));
    }

    /// The fastest way to delete is the key every other tool uses for it. It
    /// goes through the same deferred request as the menu, so it is applied
    /// after the panels have finished iterating the tree.
    #[test]
    fn the_delete_key_despawns_the_selection() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let entity = editor.spawn("Ground plane");
        editor.state.selected = Some(entity);
        editor.frames(1).key(egui::Key::Delete);
        editor.state.apply(&mut editor.world, &editor.registry);

        assert!(!editor.world.is_alive(entity));
        assert_eq!(editor.state.selected, None);
    }

    /// …but not while the user is typing: Delete in a name field deletes a
    /// character, and taking the entity with it would be data loss.
    #[test]
    fn the_delete_key_is_left_alone_while_a_field_has_focus() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let entity = editor.spawn("Ground plane");
        editor.state.selected = Some(entity);
        editor.frames(1);
        editor
            .ctx
            .memory_mut(|memory| memory.request_focus(egui::Id::new("some field")));
        editor.key(egui::Key::Delete);
        editor.state.apply(&mut editor.world, &editor.registry);

        assert!(editor.world.is_alive(entity));
    }

    /// Put a one-entity door prefab in the harness's index.
    fn door(editor: &mut Harness) -> orrin_registry::AssetId {
        let id = orrin_registry::AssetId::new();
        let text = "orrin-scene 1\n\nentity 00000000-0000-4000-8000-000000000001\n  \
                    orrin.name = \"Door\"\n  orrin.transform\n    rotation = (0.0, 0.0, 0.0, 1.0)\n    \
                    scale = (1.0, 1.0, 1.0)\n    translation = (0.0, 0.0, 0.0)\n";
        let prefab = crate::scene::prefab::Prefab::parse(id, "props/door.fprefab", text).unwrap();
        editor
            .world
            .resource_mut::<crate::scene::Prefabs>()
            .insert(prefab);
        id
    }

    /// Let go of a drag carrying `payload` at `at`, as the asset browser's
    /// rows hand one over.
    fn drop_payload<T: Send + Sync + 'static>(editor: &mut Harness, payload: T, at: egui::Pos2) {
        egui::DragAndDrop::set_payload(&editor.ctx, payload);
        editor.input.events.push(egui::Event::PointerMoved(at));
        editor.input.events.push(egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: Default::default(),
        });
        editor.frames(1);
        editor.input.events.clear();
        editor.state.apply(&mut editor.world, &editor.registry);
    }

    fn the_door(editor: &Harness) -> orrin_ecs::Entity {
        editor
            .world
            .entities()
            .find(|&e| editor.world.get::<Name>(e).is_some_and(|n| n.0 == "Door"))
            .expect("the door was placed")
    }

    #[test]
    fn the_assets_tool_lists_the_projects_prefabs() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        door(&mut editor);
        editor.dock.activate(Tab::Assets);
        editor.frames(2);

        assert!(editor.painted_contains("door"));
        assert!(editor.painted_contains("props"), "the folder it is in");
    }

    /// The gesture the issue asks for: drag a prefab out of the browser and let
    /// go over an entity, and the instance lands under it, selected.
    #[test]
    fn a_prefab_dropped_on_a_row_is_placed_under_it() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let room = editor.spawn("Room");
        let id = door(&mut editor);
        editor.frames(2);
        let row = editor
            .painted_text_rects()
            .into_iter()
            .find(|(text, _)| text == "Room")
            .expect("the row is drawn")
            .1;

        drop_payload(&mut editor, id, row.center());
        let door = the_door(&editor);
        assert_eq!(
            editor
                .world
                .get::<crate::scene::Parent>(door)
                .map(|p| p.get()),
            Some(room)
        );
        assert_eq!(editor.state.selected, Some(door));
    }

    #[test]
    fn a_prefab_dropped_below_the_rows_is_placed_at_the_top_level() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.spawn("Room");
        let id = door(&mut editor);
        editor.frames(2);
        let tree = editor.tab_body(Tab::Hierarchy);

        drop_payload(
            &mut editor,
            id,
            egui::pos2(tree.center().x, tree.bottom() - 60.0),
        );
        let door = the_door(&editor);
        assert!(editor.world.get::<crate::scene::Parent>(door).is_none());
    }

    /// A member of an instance says which prefab it is from and what it
    /// overrides — the field, not just that something changed.
    #[test]
    fn the_inspector_shows_what_an_instance_overrides() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let id = door(&mut editor);
        let (root, _) =
            crate::scene::prefab::instantiate(&mut editor.world, &editor.registry, id).unwrap();
        editor.state.selected = Some(root);
        editor.frames(2);
        assert!(editor.painted_contains("door"), "names its prefab");
        assert!(!editor.painted_contains("translation"));

        editor
            .world
            .get_mut::<LocalTransform>(root)
            .unwrap()
            .translation
            .x = 2.0;
        editor.frames(1);
        assert!(editor.painted_contains("translation"));
    }

    /// Every menu has to lay out: one that borrows a resource the others do not
    /// touch only fails when it is opened.
    #[test]
    fn every_menu_lists_its_commands() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
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

        for (menu, expected) in menu_bar::MENUS.iter().zip([
            "Save scene",
            "Delete",
            "Point light",
            "Shadows",
            "Reload scripts",
            "Reset layout",
        ]) {
            let output = ctx.run(input.clone(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    menu_bar::menu_body(*menu, ui, world, registry, state, themes, dock);
                });
            });
            let painted = painted_text(&output);
            assert!(
                painted.iter().any(|text| text == expected),
                "{menu:?} did not list {expected:?}: {painted:?}"
            );
        }
    }

    /// One bar across the top and one across the bottom. At the 800x600 the
    /// engine opens at, the frame around the tools has to stay a frame.
    #[test]
    fn the_window_chrome_is_one_bar_above_and_one_below() {
        let mut editor = Harness::new(egui::vec2(800.0, 600.0));
        editor.spawn("Ground plane");
        editor.frames(2);

        let chrome = editor.panel("menu_bar").height() + editor.panel("status_bar").height();
        assert!(chrome <= 60.0, "window chrome took {chrome}px of 600");
        assert!(editor.painted_contains("Ground plane"));
    }

    /// The bar names the project only when there is one. Outside a project it
    /// shows the scene file alone, rather than a placeholder standing in for a
    /// name.
    #[test]
    fn the_menu_bar_names_a_project_only_inside_one() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        assert!(editor.painted_contains(&editor.state.scene_path.clone()));
        assert!(!editor.painted_contains("built-in demo"));
        assert!(!editor.painted_contains(" / "));

        editor.state.project_name = Some("Courtyard".to_owned());
        editor.frames(2);
        assert!(editor.painted_contains("Courtyard"));
    }

    /// Menu titles are text on the bar until the pointer reaches one. Drawn as
    /// buttons, six grey slabs sat across the top of the window.
    #[test]
    fn the_menu_titles_sit_flat_on_the_bar() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);

        let bar = editor.panel("menu_bar");
        let transport = egui::Rect::from_center_size(bar.center(), egui::vec2(140.0, bar.height()));
        let button = editor.ctx.style().visuals.widgets.inactive.weak_bg_fill;
        let slabs: Vec<_> = editor
            .painted_fills()
            .into_iter()
            .filter(|(rect, fill)| {
                *fill == button && bar.contains(rect.center()) && !transport.contains(rect.center())
            })
            .collect();
        assert!(slabs.is_empty(), "{slabs:?}");
    }

    /// Play, pause and step are disabled until play mode exists, but they are
    /// the bar's centrepiece: drawn at egui's faded disabled alpha, in a sunken
    /// well, they read as an empty field. Opaque, and light enough to read.
    #[test]
    fn the_transport_is_legible_while_disabled() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);

        let bar = editor.panel("menu_bar");
        let transport = egui::Rect::from_center_size(bar.center(), egui::vec2(140.0, bar.height()));
        let tints: Vec<egui::Color32> = editor
            .painted_icons()
            .into_iter()
            .filter(|(_, rect, _)| transport.contains(rect.center()))
            .map(|(_, _, tint)| tint)
            .collect();
        assert_eq!(tints.len(), 3, "{tints:?}");
        assert!(
            tints.iter().all(|tint| tint.a() == 255 && tint.r() >= 130),
            "{tints:?}"
        );
    }

    /// The status bar reports the frame and the device it runs on. The scene's
    /// counts belong to the hierarchy, beside the tree they count.
    #[test]
    fn the_status_bar_reports_the_frame_and_the_device() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor
            .world
            .resource_mut::<crate::stats::FrameStats>()
            .set_device("Vulkan 1.3 · Test GPU".to_owned());
        editor.frames(2);

        let bar = editor.panel("status_bar");
        let in_bar = |editor: &Harness, needle: &str| {
            editor
                .painted_text_rects()
                .into_iter()
                .any(|(text, rect)| text.contains(needle) && bar.contains(rect.center()))
        };
        assert!(in_bar(&editor, "FPS"));
        assert!(in_bar(&editor, "Vulkan 1.3 · Test GPU"));
        assert!(in_bar(&editor, "VRAM"));
        assert!(!in_bar(&editor, "entities"));
        assert!(bar.height() <= 26.0);
    }

    /// A menu is as wide as its entries need. A field told to fill the width
    /// it is given fills a menu's whole allowance and drags it out with it.
    #[test]
    fn the_file_menu_stays_narrow() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
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
        // Sized the way egui sizes a menu: an area that measures its content
        // on a first pass and holds to that width after.
        let mut width = 0.0;
        for _ in 0..3 {
            let _ = ctx.run(input.clone(), |ctx| {
                let menu_width = ctx.style().spacing.menu_width;
                let area = egui::Area::new("menu".into())
                    .default_width(menu_width)
                    .show(ctx, |ui| {
                        menu_bar::menu_body(
                            menu_bar::Menu::File,
                            ui,
                            world,
                            registry,
                            state,
                            themes,
                            dock,
                        );
                    });
                width = area.response.rect.width();
            });
        }
        assert!(width <= 260.0, "the File menu is {width}px wide");
    }

    /// Every tab carries its tool's icon, open or not, so a strip of tabs reads
    /// at a glance.
    #[test]
    fn every_tab_shows_its_icon() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(3);

        let strip = crate::editor::panels::TAB_BAR_HEIGHT;
        for tab in Tab::ALL {
            let leaf = editor.tab(tab);
            let strip = egui::Rect::from_min_size(leaf.min, egui::vec2(leaf.width(), strip));
            let icons = editor
                .painted_icons()
                .into_iter()
                .filter(|(_, rect, _)| strip.contains(rect.center()))
                .count();
            let tabs = Tab::ALL
                .iter()
                .filter(|other| editor.tab(**other) == leaf)
                .count();
            assert_eq!(icons, tabs, "{}'s strip", tab.title());
        }
    }

    /// The hierarchy's foot counts the scene and the selection.
    #[test]
    fn the_hierarchy_footer_counts_the_scene_and_the_selection() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let ground = editor.spawn("Ground plane");
        editor.spawn("Key light");
        editor.state.selected = Some(ground);
        editor.frames(2);

        let tree = editor.tab(Tab::Hierarchy);
        assert!(
            editor
                .painted_text_rects()
                .into_iter()
                .any(|(text, rect)| text == "2 entities · 1 selected"
                    && tree.contains(rect.center()))
        );
    }

    /// A selected row is filled edge to edge across the panel, and no row is
    /// striped: one fill for the selection, one for the hover, none at rest.
    #[test]
    fn hierarchy_rows_span_the_panel_and_are_not_striped() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let first = editor.spawn("Ground plane");
        for n in 0..6 {
            editor.spawn(&format!("Cube {n}"));
        }
        editor.state.selected = Some(first);
        editor.frames(3);

        let body = editor.tab_body(Tab::Hierarchy);
        let fills: Vec<(egui::Rect, egui::Color32)> = editor
            .painted_fills()
            .into_iter()
            .filter(|(rect, _)| body.contains(rect.center()))
            .collect();
        let selection = editor.ctx.style().visuals.selection.bg_fill;
        let selected = fills
            .iter()
            .find(|(_, fill)| *fill == selection)
            .expect("the selected row is filled");
        assert!(
            (selected.0.width() - body.width()).abs() <= 1.0,
            "the selection is {}px of a {}px panel",
            selected.0.width(),
            body.width()
        );
        assert!(!fills.iter().any(|(_, fill)| *fill == theme::ROW_BAND));
    }

    /// A child's row carries a guide down its parent's column, so how deep a
    /// row sits reads without counting indents: one line per level, unbroken
    /// from the first child to the last.
    #[test]
    fn nested_rows_draw_indent_guides() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let rig = editor.spawn("Rig");
        for name in ["Arm", "Leg"] {
            let child = editor.spawn(name);
            crate::scene::reparent(&mut editor.world, child, Some(rig), false).unwrap();
        }
        editor.spawn("Lamp");
        editor.frames(3);

        let tree = editor.tab_body(Tab::Hierarchy);
        let guides: Vec<egui::Rect> = editor
            .painted_rects_of(|shape| matches!(shape, egui::Shape::LineSegment { .. }))
            .into_iter()
            .filter(|rect| {
                tree.contains(rect.center()) && rect.width() < 1.5 && rect.height() > 10.0
            })
            .collect();
        assert_eq!(
            guides.len(),
            2,
            "one guide segment per child row: {guides:?}"
        );
        assert!(
            (guides[0].left() - guides[1].left()).abs() < 0.5,
            "the guides are not one column: {guides:?}"
        );
        // No gap between one row's segment and the next: a stroke's bounds
        // reach half its width past the row, so neighbours touch or overlap.
        let mut by_top = guides.clone();
        by_top.sort_by(|a, b| a.top().total_cmp(&b.top()));
        assert!(
            by_top[1].top() <= by_top[0].bottom() + 0.01,
            "the guide breaks between rows: {guides:?}"
        );
    }

    /// The pointer arriving on a row changes how it is painted and nothing
    /// about where anything is.
    #[test]
    fn hovering_a_row_moves_nothing() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        for n in 0..6 {
            editor.spawn(&format!("Cube {n}"));
        }
        editor.frames(3);
        let before = editor.painted_text_rects();
        let (_, row) = before
            .iter()
            .find(|(text, _)| text == "Cube 3")
            .cloned()
            .expect("row drawn");

        editor.hover(row.center()).frames(2);
        let after = editor.painted_text_rects();
        for name in ["Cube 2", "Cube 3", "Cube 4"] {
            let at = |rects: &[(String, egui::Rect)]| {
                rects
                    .iter()
                    .find(|(text, _)| text == name)
                    .map(|(_, rect)| *rect)
            };
            assert_eq!(
                at(&before),
                at(&after),
                "{name} moved when a row was hovered"
            );
        }
    }

    /// The gizmo and snap controls float over the scene, top left, rather than
    /// taking a row of chrome — and they stay inside the gap the tools leave.
    #[test]
    fn the_viewport_toolbar_floats_inside_the_viewport() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(3);

        let viewport = editor.viewport().expand(6.0);
        let tools = editor.area("viewport_tools").expect("toolbar drawn");
        assert!(
            viewport.contains_rect(tools),
            "{tools:?} outside {viewport:?}"
        );
        assert!(tools.left() - viewport.left() < 20.0);
        assert!(tools.top() - viewport.top() < 20.0);
    }

    /// A section bar is a band across the panel, square at its ends: rounded,
    /// edge to edge, it read as a button laid across the inspector.
    #[test]
    fn inspector_section_bars_are_square() {
        let mut editor = Harness::new(egui::vec2(1440.0, 900.0));
        let entity = editor.spawn("Crate");
        editor.state.selected = Some(entity);
        editor.frames(3);

        let body = editor.tab_body(Tab::Inspector);
        let bar = editor.ctx.style().visuals.window_fill.to_opaque();
        let radii: Vec<egui::CornerRadius> = editor
            .painted_corner_radii()
            .into_iter()
            .filter(|(rect, fill, _)| *fill == bar && body.contains(rect.center()))
            .map(|(_, _, radius)| radius)
            .collect();
        assert!(!radii.is_empty());
        assert!(
            radii.iter().all(|r| *r == egui::CornerRadius::ZERO),
            "{radii:?}"
        );
    }

    /// The edge of anything floating over the scene is a faint light line —
    /// the rim of a pane of glass — not a drawn border competing with what it
    /// frames.
    #[test]
    fn floating_surfaces_have_faint_edges() {
        let ctx = egui::Context::default();
        theme::apply(&ctx, themes_default());
        let visuals = ctx.style().visuals.clone();
        let toolbar = crate::editor::widgets::toolbar_frame(&visuals, egui::Color32::BLACK);
        for (what, stroke) in [("toolbar", toolbar.stroke), ("menu", visuals.window_stroke)] {
            assert!(stroke.color.a() <= 32, "the {what} outline is {stroke:?}");
            assert!(stroke.width <= 1.0);
        }
    }

    /// The viewport's floating controls are frosted: each reports the rect the
    /// renderer is to blur behind it, and the translucent fill over that blur
    /// is what makes them read as glass rather than as grey boxes.
    #[test]
    fn the_viewport_controls_are_frosted() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.state.show_stats = true;
        editor.frames(3);

        for id in ["viewport_tools", "viewport_view", "viewport_stats"] {
            let area = editor.area(id).expect("drawn");
            assert!(
                editor.glass.iter().any(|glass| {
                    glass.phase == phases::of(egui::Order::Middle)
                        && glass.rect.contains_rect(area.shrink(1.0))
                        && glass.radius == f32::from(crate::editor::widgets::TOOLBAR_RADIUS)
                }),
                "{id} at {area:?} has no glass in {:?}",
                editor.glass
            );
        }
        let fill = editor.ctx.style().visuals.window_fill;
        assert!(fill.a() < 255, "menus are opaque: {fill:?}");
    }

    /// A menu is frosted whole, over a panel as over the scene: the renderer
    /// draws the panels before it blurs, so there is something real to blur
    /// there too.
    #[test]
    fn a_menu_is_frosted_over_the_panels_too() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        let file = editor
            .painted_text_rects()
            .into_iter()
            .find(|(text, _)| text == "File")
            .map(|(_, rect)| rect.center())
            .expect("the File menu's title");
        editor.click(file).frames(3);

        let viewport = editor.dock.viewport().expect("a viewport");
        let menu = editor
            .ctx
            .memory(|memory| {
                memory
                    .layer_ids()
                    .filter(|layer| {
                        layer.order == egui::Order::Foreground && memory.areas().is_visible(layer)
                    })
                    .find_map(|layer| memory.area_rect(layer.id))
            })
            .expect("the File menu is open");
        assert!(
            !viewport.contains_rect(menu),
            "the menu {menu:?} hangs over no panel, so this proves nothing"
        );
        let radius = f32::from(editor.ctx.style().visuals.menu_corner_radius.nw);
        assert!(
            editor.glass.contains(&Glass {
                phase: phases::of(egui::Order::Foreground),
                rect: menu,
                radius,
            }),
            "the menu {menu:?} is not frosted whole: {:?}",
            editor.glass
        );
    }

    /// The frame-stats readout is a few short lines, each on one line. It
    /// floats in an area that otherwise takes the width of the toggle above
    /// it, which wrapped every figure a character at a time.
    /// Opening the readout must not move the button that opened it: a second
    /// click to close it should land where the first one did.
    #[test]
    fn the_stats_toggle_stays_put_when_the_readout_opens() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let toggle = |editor: &Harness| {
            let glyph = crate::editor::icons::activity().glyph();
            let viewport = editor.dock.viewport().expect("a viewport");
            editor
                .painted_icons()
                .into_iter()
                .find(|(icon, rect, _)| *icon == glyph && viewport.contains_rect(*rect))
                .map(|(_, rect, _)| rect)
                .expect("the stats toggle is drawn")
        };
        editor.frames(3);
        let closed = toggle(&editor);
        editor.state.show_stats = true;
        editor.frames(3);
        let open = toggle(&editor);
        assert!(
            (closed.center() - open.center()).length() < 0.5,
            "the toggle moved from {closed:?} to {open:?}"
        );
    }

    /// The hierarchy fits its tab exactly: the tree scrolls, the panel does
    /// not. A footer even a pixel past the bottom turns the dock's own scroll
    /// area on, and the wheel over the counts slides the whole panel.
    #[test]
    fn the_hierarchy_footer_does_not_scroll() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        for index in 0..139 {
            editor.spawn(&format!("Cube {index}"));
        }
        editor.frames(3);
        let footer = |editor: &Harness| {
            editor
                .painted_text_rects()
                .into_iter()
                .find(|(text, _)| text.contains("selected"))
                .map(|(_, rect)| rect)
                .expect("the footer is drawn")
        };
        let before = footer(&editor);

        editor
            .input
            .events
            .push(egui::Event::PointerMoved(before.center()));
        editor.input.events.push(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, -200.0),
            modifiers: Default::default(),
        });
        editor.frames(1);
        editor.input.events.clear();
        editor.frames(3);

        let after = footer(&editor);
        assert_eq!(before, after, "the footer scrolled");
    }

    #[test]
    fn the_stats_overlay_keeps_each_figure_on_one_line() {
        // Switched on after the area has settled at the toggle's width, the
        // way it is used: on from the first frame, the area sizes to the
        // readout and the bug does not show.
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(3);
        editor.state.show_stats = true;
        editor.frames(3);

        let overlay = editor.area("viewport_stats").expect("readout drawn");
        let fps = editor
            .painted_text_rects()
            .into_iter()
            .find(|(text, rect)| text.contains("FPS") && overlay.contains(rect.center()))
            .map(|(_, rect)| rect)
            .expect("the readout drew its FPS line");
        assert!(
            fps.height() < 20.0,
            "the FPS line is {}px tall",
            fps.height()
        );
        assert!(fps.width() > 30.0, "the FPS line is {}px wide", fps.width());
    }

    /// The dock's own idea of where the scene is has to be the gap the tools
    /// actually leave, or the toolbar floats over a tool.
    #[test]
    fn the_dock_knows_where_the_viewport_is() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(3);

        let measured = editor.viewport().expand(6.0);
        let reported = editor.dock.viewport().expect("the default layout has one");
        for (side, a, b) in [
            ("left", measured.left(), reported.left()),
            ("right", measured.right(), reported.right()),
            ("top", measured.top(), reported.top()),
            ("bottom", measured.bottom(), reported.bottom()),
        ] {
            assert!((a - b).abs() <= 4.0, "{side}: measured {a}, reported {b}");
        }
    }

    /// The point of the show/body split: a tool's content draws into whatever
    /// `Ui` it is handed, with no panel or window of its own. Docking hands it a
    /// tab's `Ui`; this hands it a bare cell, which is the same contract.
    #[test]
    fn every_tool_body_draws_without_its_own_container() {
        // Wide enough to lay every body out side by side and short enough that
        // each fits: egui culls shapes outside the screen, so a column of tall
        // cells would report half the tools as drawing nothing. Generous,
        // because a body may overflow the cell it is given — the Environment
        // one does — and that shifts everything after it.
        let mut editor = Harness::new(egui::vec2(4200.0, 700.0));
        let entity = editor.spawn("Ground plane");
        editor.state.selected = Some(entity);

        let Harness {
            world,
            registry,
            state,
            previews,
            ctx,
            input,
            ..
        } = &mut editor;

        // A bounded rect, because a docked tab is bounded: the hierarchy's
        // ScrollArea fills the height it is given, and given none it draws none.
        let cell = egui::vec2(380.0, 640.0);
        let mut run = || {
            ctx.run(input.clone(), |ctx| {
                egui::Area::new("bodies".into()).show(ctx, |ui| {
                    ui.horizontal_top(|ui| {
                        ui.allocate_ui(cell, |ui| hierarchy::body(ui, world, state));
                        ui.allocate_ui(cell, |ui| {
                            inspector::body(ui, world, state, registry, previews)
                        });
                        ui.allocate_ui(cell, |ui| environment::body(ui, world));
                        ui.allocate_ui(cell, |ui| performance::body(ui, world));
                        ui.allocate_ui(cell, |ui| console::body(ui, world, state));
                        ui.allocate_ui(cell, |ui| assets::body(ui, world, state));
                    });
                });
            })
        };
        // An Area settles on its second frame, like every other egui container.
        run();
        let output = run();

        let mut painted = Vec::new();
        for clipped in &output.shapes {
            if let egui::Shape::Text(text) = &clipped.shape {
                painted.push(text.galley.text().to_owned());
            }
        }
        let painted = painted.join("\u{1}");

        // One string from each body, so a tool that silently drew nothing is not
        // mistaken for one that drew.
        for expected in [
            "Ground plane", // hierarchy
            "Transform",    // inspector
            "SSAO",         // environment
            "FPS",          // performance
            "Filter log",   // console
            "prefab(s)",    // assets
        ] {
            assert!(painted.contains(expected), "no body painted {expected:?}");
        }
    }

    /// egui's scroll bars float: they allocate nothing and expand over the
    /// content when the pointer nears them — which is the same moment the row's
    /// delete button appears. Without a reserved gutter the bar lands on top of
    /// the button, and the only way to see that is to look at where things were
    /// actually painted.
    #[test]
    fn the_row_delete_button_clears_the_scroll_bar() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        for n in 0..40 {
            editor.spawn(&format!("Cube {n}"));
        }
        editor.frames(2);

        // Over a row, which is what makes its ✕ draw at all.
        let tree = editor.tab(Tab::Hierarchy);
        let row = egui::pos2(tree.center().x, tree.bottom() - 120.0);
        editor.hover(row).frames(2);

        let bar = egui::vec2(10.0, 0.0).x;
        let delete = crate::editor::icons::x().glyph();
        let cross = editor
            .painted_icons()
            .into_iter()
            .filter(|(glyph, rect, _)| {
                *glyph == delete
                    && tree.contains(rect.center())
                    && (rect.center().y - row.y).abs() < 12.0
            })
            .map(|(_, rect, _)| rect)
            .next()
            .expect("the hovered row painted its delete icon");

        assert!(
            cross.right() <= tree.right() - bar,
            "delete icon at {:?} runs under the scroll bar; tree ends at {}",
            cross,
            tree.right()
        );
    }

    /// A tool used to be a panel that took its width from its content, so
    /// content could widen it — every frame, without settling. A docked leaf is
    /// sized by its split instead, and this is what says so.
    #[test]
    fn a_docked_tool_does_not_widen_itself() {
        for build in [Harness::new, Harness::without_icons] {
            let mut settled = build(egui::vec2(800.0, 600.0));
            settled.frames(3);
            let mut later = build(egui::vec2(800.0, 600.0));
            later.frames(120);

            for tab in [Tab::Hierarchy, Tab::Inspector] {
                assert_eq!(settled.tab(tab).width(), later.tab(tab).width());
            }
        }
    }

    /// The case that used to blow a side panel open across the viewport. A leaf
    /// clips its content now, so the name has nowhere to push.
    #[test]
    fn a_long_entity_name_cannot_widen_its_tool() {
        let mut plain = Harness::new(egui::vec2(1280.0, 800.0));
        plain.frames(3);
        let settled = plain.tab(Tab::Hierarchy).width();

        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.spawn(&"Spawn point ".repeat(40));
        editor.frames(3);

        assert_eq!(editor.tab(Tab::Hierarchy).width(), settled);
    }

    /// The one that can silently break the editor's defining trait. The engine
    /// draws no opaque surface over the middle of the window, so the Vulkan
    /// render shows through behind the whole UI — and `egui_dock` paints a body
    /// for every leaf, including a leaf holding no tabs. Nothing about a grey
    /// rectangle in the middle looks like a bug in the layout code.
    #[test]
    fn only_the_viewport_toolbars_paint_over_the_scene() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.spawn("Ground plane");
        editor.frames(3);

        // The floating toolbars are the deliberate exception, and they are
        // small: a strip of buttons in a corner, not a surface.
        let viewport = editor.viewport();
        let overlays: Vec<egui::Rect> = ["viewport_tools", "viewport_view", "viewport_stats"]
            .into_iter()
            .filter_map(|id| editor.area(id))
            .collect();
        let overlay_area: f32 = overlays.iter().map(|rect| rect.area()).sum();
        assert!(
            overlay_area < viewport.area() * 0.03,
            "the toolbars cover {overlay_area}px² of a {}px² viewport",
            viewport.area()
        );

        let covering: Vec<_> = editor
            .painted_rects()
            .into_iter()
            .filter(|rect| viewport.contains(rect.center()))
            .filter(|rect| !overlays.iter().any(|overlay| overlay.contains_rect(*rect)))
            .collect();
        assert!(
            covering.is_empty(),
            "{} shape(s) painted over the viewport, e.g. {:?}",
            covering.len(),
            covering.first()
        );
    }

    /// The input half of the same trait: the middle of the window paints
    /// nothing *and* claims nothing. egui says otherwise — its
    /// `wants_pointer_input` is true anywhere over the dock's `CentralPanel`,
    /// which is the whole window — so the fly camera's right-click reached it
    /// only while another button was held down. Asserted against egui's own
    /// verdict, since the day that one starts telling the truth is the day the
    /// dock can stop answering for it.
    #[test]
    fn the_viewport_does_not_claim_the_pointer() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(3);

        let scene = editor.viewport().center();
        editor.hover(scene).frames(2);
        assert!(
            editor.ctx.wants_pointer_input(),
            "egui no longer claims the viewport; `Dock::wants_pointer` may be redundant"
        );
        assert!(
            !editor.dock.wants_pointer(&editor.ctx),
            "the scene's own pointer events are being withheld from it"
        );

        let toolbar = editor.area("viewport_tools").expect("toolbar drawn");
        for (what, at) in [
            ("a docked tool", editor.tab(Tab::Hierarchy).center()),
            ("the menu bar", editor.panel("menu_bar").center()),
            ("the status bar", editor.panel("status_bar").center()),
            ("the viewport toolbar", toolbar.center()),
        ] {
            editor.hover(at).frames(2);
            assert!(
                editor.dock.wants_pointer(&editor.ctx),
                "{what} let a pointer event through to the scene"
            );
        }
    }

    /// The default layout is specified in pixels by the design system but built
    /// from fractions, and a split's fraction describes the new node on two of
    /// the four `split_*` calls and the old node on the other two. Getting that
    /// backwards halves a column and looks plausible.
    #[test]
    fn the_default_layout_lands_on_the_specified_widths() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(3);

        // Within a separator's width of the design's numbers.
        for (tab, target) in [(Tab::Hierarchy, 240.0), (Tab::Inspector, 320.0)] {
            let width = editor.tab(tab).width();
            assert!(
                (width - target).abs() <= 2.0,
                "{} is {width}px wide, expected {target}",
                tab.title()
            );
        }
        let bottom = editor.tab(Tab::Console).height();
        assert!(
            (bottom - 220.0).abs() <= 2.0,
            "bottom dock is {bottom}px tall"
        );
        // Environment shares the inspector's leaf rather than taking a column
        // of its own beside it; two right-hand columns starve the scene.
        assert_eq!(editor.tab(Tab::Environment), editor.tab(Tab::Inspector));
    }

    /// `Ui::columns` divides the width and does nothing to make the contents fit
    /// it, so a column too narrow for a slider paints straight over its
    /// neighbour. Docked at 300px this panel shipped four columns of overlapping
    /// labels, and nothing about that is a panic or a missing shape — it is just
    /// unreadable.
    #[test]
    fn the_environment_tool_fits_the_width_it_is_docked_at() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.dock.activate(Tab::Environment);
        editor.frames(3);

        let tool = editor.tab_body(Tab::Environment);
        let spilling: Vec<_> = editor
            .painted_rects()
            .into_iter()
            .filter(|rect| tool.contains(rect.center()) && rect.right() > tool.right())
            .collect();

        assert!(
            spilling.is_empty(),
            "{} shape(s) run past the {}px tool, e.g. {:?}",
            spilling.len(),
            tool.width(),
            spilling.first()
        );
    }

    #[test]
    fn the_console_hides_the_levels_switched_off() {
        use crate::scene::{LogBuffer, LogLevel};

        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        {
            let mut log = editor.world.resource_mut::<LogBuffer>();
            log.push(LogLevel::Debug, "cache hit".to_owned(), 0);
            log.push(LogLevel::Error, "parent cycle".to_owned(), 0);
        }
        editor.frames(2);
        assert!(editor.painted_contains("cache hit"));
        assert!(editor.painted_contains("parent cycle"));

        editor.state.console_levels.set(LogLevel::Debug, false);
        editor.frames(2);
        assert!(!editor.painted_contains("cache hit"));
        assert!(editor.painted_contains("parent cycle"));
    }

    /// A message with several lines — a validation error with its spec
    /// quotation, a dumped entity — keeps to its one row: the first line, cut
    /// short, with the rest a hover away. Painted whole it ran down over the
    /// rows below.
    #[test]
    fn a_multi_line_message_keeps_to_its_row() {
        use crate::scene::{LogBuffer, LogLevel};

        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.world.resource_mut::<LogBuffer>().push(
            LogLevel::Error,
            "VUID-1234: the first line\nThe Vulkan spec states: the second".to_owned(),
            0,
        );
        editor.frames(2);

        let row = editor
            .painted_text_rects()
            .into_iter()
            .find(|(text, _)| text.contains("the first line"))
            .expect("the message drew");
        assert!(!row.0.contains("the second"), "{:?}", row.0);
        assert!(
            row.1.height() < 20.0,
            "the row is {}px tall",
            row.1.height()
        );
    }

    #[test]
    fn the_console_search_keeps_only_matching_lines() {
        use crate::scene::{LogBuffer, LogLevel};

        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        {
            let mut log = editor.world.resource_mut::<LogBuffer>();
            log.push(LogLevel::Info, "loaded 12 entities".to_owned(), 0);
            log.push(
                LogLevel::Warning,
                "Planter_17: material not found".to_owned(),
                3,
            );
        }
        editor.state.console_query = "PLANTER".to_owned();
        editor.frames(2);
        assert!(editor.painted_contains("Planter_17: material not found"));
        assert!(!editor.painted_contains("loaded 12 entities"));
    }

    #[test]
    fn the_console_names_the_script_that_logged() {
        use crate::scene::{LogBuffer, LogEntry, LogLevel};

        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor
            .world
            .resource_mut::<LogBuffer>()
            .push_entry(LogEntry {
                level: LogLevel::Warning,
                message: "too fast".to_owned(),
                source: "Spinner".into(),
                line: 14,
                frame: 0,
            });
        editor.frames(2);
        assert!(editor.painted_contains("Spinner:14: too fast"));
    }

    /// "Every number, path, id and log line is monospace so columns align by
    /// eye." A proportional face gives each digit a different width, so a column
    /// of figures drifts — the rule is about alignment, not flavour.
    #[test]
    fn figures_are_monospace() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let entity = editor.spawn("Ground plane");
        editor.state.selected = Some(entity);
        editor.world.resource_mut::<crate::scene::LogBuffer>().push(
            crate::scene::LogLevel::Info,
            "saved 3 entities".to_owned(),
            0,
        );
        // The console and the profiler share a leaf, so each is looked at in
        // a frame of its own.
        editor.frames(3);
        let mut figures = vec!["saved 3 entities", "id 1", "entities"];
        let check = |editor: &Harness, figures: &mut Vec<&str>| {
            for figure in figures.drain(..) {
                assert_eq!(
                    editor.family_of(figure),
                    Some(egui::FontFamily::Monospace),
                    "{figure:?} is not monospace"
                );
            }
        };
        check(&editor, &mut figures);

        editor.dock.activate(Tab::Performance);
        editor.frames(3);
        figures.extend(["CPU:", "Memory (RSS)"]);
        check(&editor, &mut figures);

        // Prose is not a column, and monospacing a sentence only makes it
        // harder to read.
        assert_eq!(
            editor.family_of("Frustum culling"),
            Some(egui::FontFamily::Proportional)
        );
    }

    /// Closing a tool and reopening it has to put it back somewhere findable.
    #[test]
    fn a_closed_tool_comes_back_where_it_belongs() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        let home = editor.tab(Tab::Performance);

        editor.dock.toggle(Tab::Performance);
        assert!(!editor.dock.is_open(Tab::Performance));
        editor.frames(2);

        editor.dock.toggle(Tab::Performance);
        editor.frames(2);
        assert_eq!(editor.tab(Tab::Performance), home);
    }

    /// Reset undoes whatever the user did to the tree.
    #[test]
    fn reset_layout_restores_the_default_tree() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        let home = editor.tab(Tab::Console);

        editor.dock.toggle(Tab::Console);
        editor.dock.toggle(Tab::Hierarchy);
        assert!(!editor.dock.is_open(Tab::Console));

        editor.dock.reset();
        editor.frames(2);
        assert!(editor.dock.is_open(Tab::Hierarchy));
        assert_eq!(editor.tab(Tab::Console), home);
    }

    /// Not "the editor looks cramped": once the side panels are wider than the
    /// window, the Environment panel between them is handed a negative width,
    /// and egui asserts inside `columns` rather than clamping.
    #[test]
    fn a_window_narrower_than_its_side_panels_still_lays_out() {
        Harness::new(egui::vec2(320.0, 600.0)).frames(4);
    }

    #[test]
    fn a_window_too_narrow_for_any_panel_still_lays_out() {
        Harness::new(egui::vec2(120.0, 400.0)).frames(4);
    }

    /// What the editor's own layout pass costs the CPU each frame — the part of
    /// the frame no `tests/perf.rs` row sees, since that harness has no editor.
    /// Tessellation and the GPU draw are egui_winit_vulkano's and not measured.
    ///
    /// ```text
    /// cargo test --release -p orrin-core --lib editor_frame_cost -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "a measurement, not a check: run it in release"]
    fn editor_frame_cost() {
        const FRAMES: usize = 300;

        let mut editor = Harness::new(egui::vec2(1920.0, 1080.0));
        let mut parent = editor.spawn("Root");
        for n in 0..2000 {
            let entity = editor.spawn(&format!("Entity {n}"));
            if n % 50 == 0 {
                parent = editor.spawn(&format!("Group {n}"));
            } else {
                crate::scene::reparent(&mut editor.world, entity, Some(parent), false).unwrap();
            }
        }
        editor.state.selected = Some(parent);
        editor.frames(20);

        let start = std::time::Instant::now();
        editor.frames(FRAMES);
        let per_frame = start.elapsed() / FRAMES as u32;
        println!("editor layout: {per_frame:?} per frame over {FRAMES} frames, 2 000 entities");
    }
}
