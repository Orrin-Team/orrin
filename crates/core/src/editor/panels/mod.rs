// Visible to the dock, which mounts each tool's `body` into a tab.
pub(super) mod assets;
pub(super) mod console;
pub(super) mod environment;
pub(super) mod hierarchy;
pub(super) mod import;
pub(super) mod inspector;
mod menu_bar;
pub(super) mod performance;
pub(super) mod scenes;
#[cfg(feature = "scripting")]
pub(super) mod scripts;
mod status_bar;
pub(super) mod storybook;
mod ui_sections;
mod viewport;

use glam::Vec3;

use orrin_ecs::World;
use orrin_registry::Registry;

use super::dock::Dock;
use super::previews::Previews;
use super::state::{EditorState, GizmoMode, SceneRequest};
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
    // Asked for by a tool's own controls last frame, while the dock was
    // drawing them and so could not move.
    if let Some(request) = state.take_dock_request() {
        dock.apply(request);
    }
    // Before the dock, so a command that rearranges the layout is reflected in
    // the same frame rather than the next one.
    menu_bar::show(ctx, world, registry, state, themes, dock);
    status_bar::show(ctx, world, state, themes);
    dock.show(ctx, world, state, registry, previews);
    // Before the viewport's tools, which show the screen this sets.
    game_screen(ctx, world, state, dock.viewport());
    if let Some(rect) = dock.viewport() {
        let live = world
            .get_resource::<crate::scene::Play>()
            .is_some_and(|play| play.is_live());
        viewport::show(ctx, world, state, themes, rect, live);
        if !live {
            viewport::accept_prefab(ctx, world, state, dock, rect);
            if state.ui_mode {
                // Before the click, for the reason the gizmo is.
                viewport::rect_tool(ctx, world, state, dock, rect);
                viewport::select_ui_on_click(ctx, world, state, dock, rect);
            } else {
                state.rect_drag = None;
                viewport::hover(ctx, world, state, dock, rect);
                // Before the click: a click on a handle is not a click on the scene.
                viewport::gizmo(ctx, world, state, dock, rect);
                viewport::marquee(ctx, world, state, dock, rect);
                viewport::select_on_click(ctx, world, state, dock, rect);
            }
        } else {
            state.gizmo_strokes.clear();
            state.outline.clear();
            state.rect_drag = None;
        }
    }
    if let Some(rel) = state.pending_delete().map(str::to_owned) {
        confirm_delete(ctx, world, state, &rel);
    }
    // A drag ends with the button and a typed edit with the field, so the
    // gesture they make stays open while either is still held. Read after the
    // widgets, which are what let go of the focus.
    let holding = ctx.input(|input| input.pointer.any_down())
        || ctx.memory(|memory| memory.focused().is_some());
    state.hold(holding);
}

/// The one question every delete asks, whichever tool it came from: what goes,
/// and that it does not come back.
///
/// A modal rather than a line in the tool, because the answer is
/// irreversible and the tool asking may be scrolled, narrow or floating —
/// this is the one place in the editor that takes the whole window to ask.
fn confirm_delete(ctx: &egui::Context, world: &World, state: &mut EditorState, rel: &str) {
    let name = rel.rsplit('/').next().unwrap_or(rel);
    let inside = world
        .get_resource::<crate::asset_db::AssetDatabase>()
        .filter(|db| db.folders().iter().any(|folder| folder.path == rel))
        .map(|db| {
            let prefix = format!("{rel}/");
            db.iter()
                .filter(|entry| entry.path.starts_with(&prefix))
                .count()
        });
    let modal = egui::Modal::new(egui::Id::new("confirm_delete")).show(ctx, |ui| {
        ui.set_width(340.0);
        ui.label(
            egui::RichText::new(format!("Delete {name}?"))
                .size(15.0)
                .strong(),
        );
        ui.add_space(6.0);
        let what = match inside {
            Some(0) => "The folder is removed from disk.".to_owned(),
            Some(1) => "The folder and the asset in it are removed from disk.".to_owned(),
            Some(n) => format!("The folder and the {n} assets in it are removed from disk."),
            None => format!("{rel} and its id are removed from disk."),
        };
        ui.label(what);
        ui.add(
            egui::Label::new(
                egui::RichText::new(
                    "Anything that uses it draws a placeholder until it is replaced. \
                     This cannot be undone.",
                )
                .color(ui.visuals().weak_text_color()),
            )
            .wrap(),
        );
        ui.add_space(10.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let delete = egui::Button::new(
                egui::RichText::new("Delete")
                    .color(super::theme::ERROR)
                    .strong(),
            );
            if ui.add(delete).clicked() {
                state.resolve_delete(true);
            }
            if ui.button("Cancel").clicked() {
                state.resolve_delete(false);
            }
        });
    });
    // Escape, or a click on the dimmed window around it.
    if modal.should_close() && state.pending_delete().is_some() {
        state.resolve_delete(false);
    }
}

/// Tell the game how big its screen is and where in the window it is drawn:
/// the previewed resolution inside the viewport, or the window.
fn game_screen(
    ctx: &egui::Context,
    world: &mut World,
    state: &EditorState,
    viewport: Option<egui::Rect>,
) {
    use crate::scene::{GameScreen, Rect};
    let ppp = ctx.pixels_per_point();
    let window = ctx.screen_rect().size() * ppp;
    let extent = [window.x.round() as u32, window.y.round() as u32];
    let screen = match viewport {
        Some(rect) => crate::editor::preview::screen(
            &state.preview,
            Rect::new(
                rect.min.x * ppp,
                rect.min.y * ppp,
                rect.width() * ppp,
                rect.height() * ppp,
            ),
            extent,
        ),
        None => GameScreen::window(extent),
    };
    if let Some(mut current) = world.get_resource_mut::<GameScreen>() {
        *current = screen;
        return;
    }
    world.insert_resource(screen);
}

/// The few keys every editor answers the same way. Each one stands aside while
/// a text field has focus, where the same key edits text.
fn shortcuts(ctx: &egui::Context, world: &World, state: &mut EditorState) {
    if ctx.input_mut(|input| input.consume_key(egui::Modifiers::COMMAND, egui::Key::S)) {
        state.request_scene(SceneRequest::Save);
    }
    // Most specific first, as with redo below.
    let (alt, shift) = (
        egui::Modifiers::COMMAND | egui::Modifiers::ALT,
        egui::Modifiers::COMMAND | egui::Modifiers::SHIFT,
    );
    if ctx.input_mut(|input| input.consume_key(alt, egui::Key::P)) {
        state.request_step();
    } else if ctx.input_mut(|input| input.consume_key(shift, egui::Key::P)) {
        state.request_pause();
    } else if ctx.input_mut(|input| input.consume_key(egui::Modifiers::COMMAND, egui::Key::P)) {
        state.request_play();
    }
    if ctx.wants_keyboard_input() {
        return;
    }
    // Redo first: egui matches Ctrl+Z whether or not Shift is also down.
    let shift = egui::Modifiers::COMMAND | egui::Modifiers::SHIFT;
    if ctx.input_mut(|input| {
        input.consume_key(shift, egui::Key::Z)
            || input.consume_key(egui::Modifiers::COMMAND, egui::Key::Y)
    }) {
        state.request_redo();
    } else if ctx.input_mut(|input| input.consume_key(egui::Modifiers::COMMAND, egui::Key::Z)) {
        state.request_undo();
    }
    for (key, mode) in [
        (egui::Key::Num1, GizmoMode::Select),
        (egui::Key::Num2, GizmoMode::Move),
        (egui::Key::Num3, GizmoMode::Rotate),
        (egui::Key::Num4, GizmoMode::Scale),
    ] {
        if ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, key)) {
            state.gizmo_mode = mode;
        }
    }
    if ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::T)) {
        state.ui_mode = !state.ui_mode;
    }
    state.selection.retain_alive(world);
    let selected = !state.selection.is_empty();
    if selected
        && ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Delete))
    {
        state.request_despawn_selection();
    }
    if selected
        && ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
    {
        state.selection.clear();
    }
    if selected && ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::F)) {
        state.request_frame();
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
                    ctx,
                    input,
                    output,
                } = self;
                // As the app does every frame, so UI mode sees what is drawn.
                let screen = input
                    .screen_rect
                    .map_or([0, 0], |rect| [rect.width() as u32, rect.height() as u32]);
                crate::ui::update(world, screen);
                *output = Some(ctx.run(input.clone(), |ctx| {
                    draw(ctx, world, state, registry, themes, dock, previews);
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

        /// Two clicks at `at` inside one frame: a double-click.
        fn double_click(&mut self, at: egui::Pos2) -> &mut Self {
            self.input.events.push(egui::Event::PointerMoved(at));
            for _ in 0..2 {
                for pressed in [true, false] {
                    self.input.events.push(egui::Event::PointerButton {
                        pos: at,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    });
                }
            }
            self.frames(1);
            self.input.events.clear();
            self
        }

        /// A secondary press and release at `at`: what opens a context menu.
        fn right_click(&mut self, at: egui::Pos2) -> &mut Self {
            self.input.events.push(egui::Event::PointerMoved(at));
            for pressed in [true, false] {
                self.input.events.push(egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Secondary,
                    pressed,
                    modifiers: Default::default(),
                });
            }
            self.frames(1);
            self.input.events.clear();
            self
        }

        /// [`click`](Self::click) with `modifiers` held.
        fn click_with(&mut self, at: egui::Pos2, modifiers: egui::Modifiers) -> &mut Self {
            self.input.modifiers = modifiers;
            self.input.events.push(egui::Event::PointerMoved(at));
            for pressed in [true, false] {
                self.input.events.push(egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers,
                });
            }
            self.frames(1);
            self.input.events.clear();
            self.input.modifiers = egui::Modifiers::NONE;
            self
        }

        fn key(&mut self, key: egui::Key) -> &mut Self {
            self.key_with(egui::Modifiers::NONE, key)
        }

        fn key_with(&mut self, modifiers: egui::Modifiers, key: egui::Key) -> &mut Self {
            self.input.modifiers = modifiers;
            for pressed in [true, false] {
                self.input.events.push(egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed,
                    repeat: false,
                    modifiers,
                });
            }
            self.frames(1);
            self.input.events.clear();
            self.input.modifiers = egui::Modifiers::NONE;
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
        editor.state.selection.set(entity);
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

    /// The selected entity's material is shown on a sphere the renderer draws,
    /// at the top of its section: the inspector asks for that material and
    /// paints the image the renderer will have drawn it into by the time the
    /// overlay samples it. Nothing selected asks for nothing, which is what
    /// keeps the preview's passes out of every other frame.
    #[test]
    fn the_material_section_shows_its_material_on_the_renderers_sphere() {
        use crate::gfx::vulkan::material_preview::{SIZE, Viewer};
        use crate::scene::{AssetPreviews, MaterialHandle, MeshHandle};

        let mut editor = Harness::new(egui::vec2(1440.0, 900.0));
        let entity = editor.spawn("Crate");
        editor.world.insert(entity, MeshHandle(0));
        editor.world.insert(entity, MaterialHandle(3));
        let mut kept = AssetPreviews::default();
        kept.insert_material(MaterialHandle(3), &crate::gfx::Material::default(), None);
        editor.world.insert_resource(kept);
        editor.world.insert_resource(crate::scene::Assets::new());

        editor.frames(2);
        assert_eq!(editor.state.material_previews, [None, None]);

        editor.state.selection.set(entity);
        editor.frames(2);
        let asked = editor.state.material_previews[Viewer::Inspector as usize];
        assert_eq!(
            asked.map(|subject| subject.material),
            Some(MaterialHandle(3))
        );
        assert_eq!(
            editor.state.material_previews[Viewer::Browser as usize],
            None
        );

        let sphere = sphere_in(&editor, Tab::Inspector, Viewer::Inspector);
        assert!(
            (sphere.width() - sphere.height()).abs() < 0.5 && sphere.width() >= 96.0,
            "{sphere:?}"
        );
        assert_checkerboard_behind(&editor, sphere);

        // Denser than 2x, the picture shrinks rather than stretch past its pixels.
        editor.ctx.set_pixels_per_point(3.0);
        editor.frames(2);
        let sphere = sphere_in(&editor, Tab::Inspector, Viewer::Inspector);
        assert!(sphere.width() * 3.0 <= SIZE as f32 + 0.5, "{sphere:?}");
    }

    /// Dragging the sphere orbits it, and a double-click puts it back.
    #[test]
    fn dragging_the_material_sphere_orbits_it() {
        use crate::gfx::vulkan::material_preview::{Orbit, Viewer};
        use crate::scene::{AssetPreviews, MaterialHandle, MeshHandle};

        let mut editor = Harness::new(egui::vec2(1440.0, 900.0));
        let entity = editor.spawn("Crate");
        editor.world.insert(entity, MeshHandle(0));
        editor.world.insert(entity, MaterialHandle(3));
        let mut kept = AssetPreviews::default();
        kept.insert_material(MaterialHandle(3), &crate::gfx::Material::default(), None);
        editor.world.insert_resource(kept);
        editor.world.insert_resource(crate::scene::Assets::new());
        editor.state.selection.set(entity);
        editor.frames(2);
        let orbit = |editor: &Harness| {
            editor.state.material_previews[Viewer::Inspector as usize]
                .expect("the inspector shows a sphere")
                .orbit
        };
        assert_eq!(orbit(&editor), Orbit::default());

        let centre = sphere_in(&editor, Tab::Inspector, Viewer::Inspector).center();
        let frame = |editor: &mut Harness, event: egui::Event| {
            editor.input.events.push(event);
            editor.frames(1);
            editor.input.events.clear();
        };
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        };
        frame(&mut editor, egui::Event::PointerMoved(centre));
        frame(&mut editor, button(centre, true));
        let mut at = centre;
        for _ in 0..3 {
            at += egui::vec2(10.0, 4.0);
            frame(&mut editor, egui::Event::PointerMoved(at));
        }
        frame(&mut editor, button(at, false));
        let turned = orbit(&editor);
        assert_ne!(turned.yaw, 0.0, "{turned:?}");
        assert_ne!(turned.pitch, 0.0, "{turned:?}");
        editor.frames(2);
        assert_eq!(orbit(&editor), turned, "the orbit is kept between frames");

        editor.double_click(centre);
        editor.frames(1);
        assert_eq!(orbit(&editor), Orbit::default());
    }

    /// A material marked in the browser is drawn on the renderer's sphere in
    /// the Import tab, beside, not instead of, the inspector's.
    #[test]
    fn a_material_marked_in_the_browser_is_previewed_on_the_sphere() {
        use crate::gfx::vulkan::material_preview::Viewer;
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let dir = crate::asset_db::tests::Dir::new("panel-browser-material-preview");
        dir.write("brick.omat", b"orrin-material 1 min-reader 1\n");
        let (db, _) = crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let brick = db.find("brick.omat").unwrap().id;
        editor.world.insert_resource(db);
        editor.dock.activate(Tab::Assets);
        editor.frames(2);
        editor.click(row_of(&editor, "brick.omat"));
        editor.dock.activate(Tab::Import);
        editor.frames(2);

        let handle = editor
            .world
            .resource_mut::<crate::asset_db::AssetDatabase>()
            .material(brick)
            .unwrap();
        let asked = editor.state.material_previews[Viewer::Browser as usize];
        assert_eq!(asked.map(|subject| subject.material), Some(handle));
        let sphere = sphere_in(&editor, Tab::Import, Viewer::Browser);
        assert_checkerboard_behind(&editor, sphere);
    }

    /// Where the sphere `viewer` asked for was painted, inside `tab`.
    fn sphere_in(
        editor: &Harness,
        tab: Tab,
        viewer: crate::gfx::vulkan::material_preview::Viewer,
    ) -> egui::Rect {
        let texture = viewer.texture();
        let spheres = editor.painted_rects_of(|shape| match shape {
            egui::Shape::Rect(rect) => rect
                .brush
                .as_ref()
                .is_some_and(|brush| brush.fill_texture_id == texture),
            egui::Shape::Mesh(mesh) => mesh.texture_id == texture,
            _ => false,
        });
        assert_eq!(spheres.len(), 1, "{spheres:?}");
        let body = editor.tab_body(tab);
        assert!(
            body.contains_rect(spheres[0]),
            "{:?} in {body:?}",
            spheres[0]
        );
        spheres[0]
    }

    /// A cutout's holes read as holes only against something: the sphere's
    /// square is tiled in the theme's darkest background and its window fill.
    fn assert_checkerboard_behind(editor: &Harness, sphere: egui::Rect) {
        let cells: Vec<_> = editor
            .painted_fills()
            .into_iter()
            .filter(|(rect, _)| sphere.expand(0.5).contains_rect(*rect) && rect.area() > 1.0)
            .collect();
        assert!(cells.len() >= 16, "{} cells under {sphere:?}", cells.len());
        let shades: std::collections::HashSet<_> = cells.iter().map(|(_, fill)| *fill).collect();
        let visuals = editor.ctx.style().visuals.clone();
        let theme: std::collections::HashSet<_> = [visuals.extreme_bg_color, visuals.window_fill]
            .into_iter()
            .collect();
        assert_eq!(
            shades, theme,
            "the backdrop is the editor's own backgrounds"
        );
        let covered: f32 = cells.iter().map(|(rect, _)| rect.area()).sum();
        assert!(
            (covered - sphere.area()).abs() < sphere.area() * 0.02,
            "{covered}"
        );
    }

    /// A section bar runs edge to edge across the inspector, so the sections
    /// divide the panel rather than sitting as boxes inside it.
    #[test]
    fn inspector_sections_span_the_panel() {
        let mut editor = Harness::new(egui::vec2(1440.0, 900.0));
        let entity = editor.spawn("Crate");
        editor.state.selection.set(entity);
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

    fn open_add_component(editor: &mut Harness) {
        let at = editor
            .painted_text_rects()
            .into_iter()
            .find(|(text, _)| without_icons(text) == "Add component")
            .map(|(_, rect)| rect.center())
            .expect("the inspector offers to add a component");
        editor.click(at).frames(1);
    }

    /// The add menu files what it offers under headings, each row with the
    /// icon its section wears and its display name alone — never the id.
    #[test]
    fn the_add_menu_groups_its_components_under_headings_with_icons() {
        let mut editor = Harness::new(egui::vec2(1440.0, 900.0));
        let entity = editor.spawn("Crate");
        editor.state.selection.set(entity);
        editor.frames(3);
        open_add_component(&mut editor);

        let menu = editor.area("add_component").expect("the add menu is open");
        let texts = editor.painted_text_rects();
        let row = |wanted: &str| {
            texts
                .iter()
                .find(|(text, rect)| without_icons(text) == wanted && menu.contains(rect.center()))
                .map(|(_, rect)| *rect)
        };
        for heading in ["Rendering", "Physics", "UI"] {
            assert!(row(heading).is_some(), "no {heading} heading: {texts:?}");
        }
        assert!(
            texts.iter().all(|(text, _)| !text.contains("orrin.")),
            "an id is shown: {texts:?}"
        );
        let ui = row("UI").unwrap();
        let gradient = row("Gradient").expect("Gradient is offered");
        let light = row("Light").expect("Light is offered");
        assert!(gradient.top() > ui.top(), "Gradient is under UI");
        assert!(light.top() < ui.top(), "Light is under Rendering, above UI");

        let icons = editor.painted_icons();
        let wears = |glyph: char, beside: egui::Rect| {
            icons.iter().any(|(painted, rect, _)| {
                *painted == glyph && (rect.center().y - beside.center().y).abs() < 4.0
            })
        };
        assert!(wears(crate::editor::icons::gradient().glyph(), gradient));
        assert!(wears(crate::editor::icons::lightbulb().glyph(), light));

        let body = editor.tab_body(Tab::Inspector);
        assert!(
            menu.width() <= 240.0 && menu.width() < body.width(),
            "the add menu is {}px wide in a {}px inspector",
            menu.width(),
            body.width()
        );
    }

    /// The menu opens where it was asked for, not centred under a button that
    /// spans the panel.
    #[test]
    fn the_add_menu_opens_at_the_click() {
        let mut editor = Harness::new(egui::vec2(1440.0, 900.0));
        let entity = editor.spawn("Crate");
        editor.state.selection.set(entity);
        editor.frames(3);
        let label = editor
            .painted_text_rects()
            .into_iter()
            .find(|(text, _)| without_icons(text) == "Add component")
            .map(|(_, rect)| rect)
            .expect("the inspector offers to add a component");
        let at = label.center();
        editor.click(at).frames(1);
        let menu = editor.area("add_component").expect("the add menu is open");
        assert!(
            (menu.left() - at.x).abs() <= 12.0 && menu.top() >= at.y && menu.top() <= at.y + 16.0,
            "clicked at {at:?}, the menu is at {menu:?}"
        );
    }

    /// Typing narrows the menu to what matches, and Enter adds the first.
    #[test]
    fn searching_the_add_menu_narrows_it_and_enter_adds_the_match() {
        let mut editor = Harness::new(egui::vec2(1440.0, 900.0));
        let entity = editor.spawn("Crate");
        editor.state.selection.set(entity);
        editor.frames(3);
        open_add_component(&mut editor);

        editor.input.events.push(egui::Event::Text("grad".into()));
        editor.frames(1);
        editor.input.events.clear();
        editor.frames(1);
        let menu = editor.area("add_component").expect("the add menu is open");
        let shown: Vec<String> = editor
            .painted_text_rects()
            .into_iter()
            .filter(|(_, rect)| menu.contains(rect.center()))
            .map(|(text, _)| without_icons(&text))
            .collect();
        assert!(shown.iter().any(|text| text == "Gradient"), "{shown:?}");
        for gone in ["Corner", "Light", "Rendering"] {
            assert!(
                !shown.iter().any(|text| text == gone),
                "{gone} still shown: {shown:?}"
            );
        }

        editor.key(egui::Key::Enter);
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(editor.world.has::<crate::ui::Gradient>(entity));
        editor.frames(1);
        let open = editor
            .ctx
            .memory(|memory| memory.is_popup_open(egui::Id::new("add_component")));
        assert!(!open, "adding closes the menu");
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
        editor.state.selection.set(entity);
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
        editor.state.selection.set(entity);
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
        editor.state.selection.set(entity);
        editor.frames(1).key(egui::Key::Delete);
        editor.state.apply(&mut editor.world, &editor.registry);

        assert!(!editor.world.is_alive(entity));
        assert!(editor.state.selection.is_empty());
    }

    /// …but not while the user is typing: Delete in a name field deletes a
    /// character, and taking the entity with it would be data loss.
    #[test]
    fn the_delete_key_is_left_alone_while_a_field_has_focus() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let entity = editor.spawn("Ground plane");
        editor.state.selection.set(entity);
        editor.frames(1);
        editor
            .ctx
            .memory_mut(|memory| memory.request_focus(egui::Id::new("some field")));
        editor.key(egui::Key::Delete);
        editor.state.apply(&mut editor.world, &editor.registry);

        assert!(editor.world.is_alive(entity));
    }

    /// The live entity called `name`. Undo respawns into new slots, so a
    /// handle taken before it is no longer the one to look for.
    fn named(world: &World, name: &str) -> Option<orrin_ecs::Entity> {
        world
            .entities()
            .find(|&entity| world.get::<Name>(entity).is_some_and(|n| n.0 == name))
    }

    fn delete_by_key(editor: &mut Harness, name: &str) {
        let entity = editor.spawn(name);
        editor.state.selection.set(entity);
        editor.frames(1).key(egui::Key::Delete);
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(named(&editor.world, name).is_none());
    }

    #[test]
    fn ctrl_z_undoes_and_ctrl_shift_z_or_ctrl_y_redoes() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        delete_by_key(&mut editor, "Ground plane");
        let mut press = |modifiers, key| {
            editor.key_with(modifiers, key);
            editor.state.apply(&mut editor.world, &editor.registry);
            named(&editor.world, "Ground plane").is_some()
        };
        let shift = egui::Modifiers::COMMAND | egui::Modifiers::SHIFT;

        assert!(press(egui::Modifiers::COMMAND, egui::Key::Z), "undone");
        assert!(!press(shift, egui::Key::Z), "redone");
        assert!(
            press(egui::Modifiers::COMMAND, egui::Key::Z),
            "undone again"
        );
        assert!(
            !press(egui::Modifiers::COMMAND, egui::Key::Y),
            "redone again"
        );
    }

    /// A text field has an undo of its own, and Ctrl+Z there is that one.
    #[test]
    fn ctrl_z_is_left_to_a_field_that_has_focus() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        delete_by_key(&mut editor, "Ground plane");
        editor
            .ctx
            .memory_mut(|memory| memory.request_focus(egui::Id::new("some field")));
        editor.key_with(egui::Modifiers::COMMAND, egui::Key::Z);
        editor.state.apply(&mut editor.world, &editor.registry);

        assert!(named(&editor.world, "Ground plane").is_none());
    }

    fn edit_menu(editor: &mut Harness) -> Vec<String> {
        let Harness {
            world,
            registry,
            state,
            themes,
            dock,
            ctx,
            input,
            ..
        } = editor;
        let output = ctx.run(input.clone(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                menu_bar::menu_body(
                    menu_bar::Menu::Edit,
                    ui,
                    world,
                    registry,
                    state,
                    themes,
                    dock,
                );
            });
        });
        painted_text(&output)
    }

    #[test]
    fn the_edit_menu_names_what_undo_and_redo_would_do() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        delete_by_key(&mut editor, "Ground plane");
        let painted = edit_menu(&mut editor);
        assert!(
            painted.iter().any(|t| t == "Undo Delete Ground plane"),
            "{painted:?}"
        );
        assert!(painted.iter().any(|t| t == "Redo"), "{painted:?}");

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        let painted = edit_menu(&mut editor);
        assert!(
            painted.iter().any(|t| t == "Redo Delete Ground plane"),
            "{painted:?}"
        );
    }

    /// The case the pipeline's gestures exist for, driven through the real
    /// widget: a drag shows on every frame and undoes in one step.
    #[test]
    fn dragging_a_transform_field_is_one_undo_step() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let entity = editor.spawn("Cube");
        editor.state.selection.set(entity);
        editor.frames(2);
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
        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        };
        frame(&mut editor, egui::Event::PointerMoved(chip));
        frame(&mut editor, button(chip, true));
        let mut at = chip;
        for _ in 0..3 {
            at += egui::vec2(20.0, 0.0);
            frame(&mut editor, egui::Event::PointerMoved(at));
        }
        let x = |editor: &Harness| {
            editor
                .world
                .get::<LocalTransform>(entity)
                .unwrap()
                .translation
                .x
        };
        assert!(x(&editor) > 0.0, "the drag shows while it is held");
        frame(&mut editor, button(at, false));

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        assert_eq!(x(&editor), 0.0);
        assert_eq!(editor.state.history.undo_label(), None, "one step");
    }

    #[test]
    fn spawning_a_light_can_be_undone() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor
            .state
            .request_spawn(crate::editor::state::SpawnKind::PointLight);
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(named(&editor.world, "Point Light").is_some());

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(named(&editor.world, "Point Light").is_none());
        assert!(editor.state.selection.is_empty());
    }

    /// Where the editor is looking, so it opens on what the author framed.
    #[test]
    fn a_new_camera_stands_where_the_editor_view_does() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let view = *editor.world.resource::<crate::scene::Camera>();
        editor
            .state
            .request_spawn(crate::editor::state::SpawnKind::Camera);
        editor.state.apply(&mut editor.world, &editor.registry);

        let camera = named(&editor.world, "Camera").expect("spawned");
        assert!(editor.world.has::<crate::scene::CameraComponent>(camera));
        let local = *editor.world.get::<LocalTransform>(camera).unwrap();
        assert!(local.translation.distance(view.position) < 1e-4);
        assert!(
            (local.rotation * glam::Vec3::NEG_Z).distance(view.rotation() * glam::Vec3::NEG_Z)
                < 1e-4
        );
        assert_eq!(editor.state.selection.entities(), [camera]);
    }

    #[test]
    fn a_reparent_in_the_tree_can_be_undone() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let parent = editor.spawn("Parent");
        let child = editor.spawn("Child");
        editor.state.request_reparent(child, Some(parent));
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(editor.world.has::<crate::scene::Parent>(child));

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(!editor.world.has::<crate::scene::Parent>(child));
    }

    #[test]
    fn adding_and_removing_a_component_can_be_undone() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let entity = editor.spawn("Cube");
        editor
            .state
            .request_add_component(entity, crate::scene::registry::TAG);
        editor.state.apply(&mut editor.world, &editor.registry);
        editor
            .state
            .request_remove_component(entity, crate::scene::registry::TRANSFORM);
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(editor.world.has::<crate::scene::Tag>(entity));
        assert!(!editor.world.has::<LocalTransform>(entity));

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(editor.world.has::<LocalTransform>(entity));
        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(!editor.world.has::<crate::scene::Tag>(entity));
    }

    /// Put a one-entity door prefab in the harness's index.
    fn door(editor: &mut Harness) -> orrin_registry::AssetId {
        let id = orrin_registry::AssetId::new();
        let text = "orrin-scene 1\n\nentity 00000000-0000-4000-8000-000000000001\n  \
                    orrin.name = \"Door\"\n  orrin.transform\n    rotation = (0.0, 0.0, 0.0, 1.0)\n    \
                    scale = (1.0, 1.0, 1.0)\n    translation = (0.0, 0.0, 0.0)\n";
        let prefab = crate::scene::prefab::Prefab::parse(id, "props/door.oprefab", text).unwrap();
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

    /// With a project, the browser opens on the assets directory's folders and
    /// says how many imports failed, so a broken file is seen without opening
    /// the console.
    #[test]
    fn the_assets_tool_browses_the_projects_folders_and_counts_failures() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let dir = crate::asset_db::tests::Dir::new("panel-browser");
        dir.write("props/door.oprefab", b"orrin-scene 1\n");
        dir.write("rocks/bad.png", b"not a png");
        let (mut db, _) =
            crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        assert!(db.wait_idle(std::time::Duration::from_secs(20)));
        db.update(&editor.world, &mut crate::gfx::HeadlessBackend::new());
        editor.world.insert_resource(db);
        editor.dock.activate(Tab::Assets);
        editor.frames(2);

        assert!(editor.painted_contains("props"));
        assert!(editor.painted_contains("rocks"));
        assert!(editor.painted_contains("1 failed"));
        assert!(
            !editor.painted_contains("door"),
            "inside a folder, not at the root"
        );
    }

    /// A list stacks rows tall enough for a picture beside each name; a grid
    /// lays tiles side by side, each name under its picture.
    #[test]
    fn the_browser_lays_a_folder_out_as_a_list_or_a_grid() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let dir = crate::asset_db::tests::Dir::new("panel-browser-layout");
        dir.write("alpha.oprefab", b"orrin-scene 1\n");
        dir.write("bravo.oprefab", b"orrin-scene 1\n");
        let (db, _) = crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        editor.world.insert_resource(db);
        editor.dock.activate(Tab::Assets);
        editor.frames(2);
        let label = |editor: &Harness, name: &str| {
            editor
                .painted_text_rects()
                .into_iter()
                .find(|(text, _)| text == name)
                .unwrap_or_else(|| panic!("{name} is painted"))
                .1
        };

        let (a, b) = (
            label(&editor, "alpha.oprefab"),
            label(&editor, "bravo.oprefab"),
        );
        assert!(
            (a.left() - b.left()).abs() < 2.0,
            "a list stacks: {a:?} {b:?}"
        );
        assert!(
            b.center().y - a.center().y >= 32.0,
            "rows of 32 or more: {a:?} {b:?}"
        );

        editor.state.assets_view.layout = crate::editor::state::AssetLayout::Grid;
        editor.frames(2);
        let (a, b) = (
            label(&editor, "alpha.oprefab"),
            label(&editor, "bravo.oprefab"),
        );
        assert!(
            (a.center().y - b.center().y).abs() < 1.0,
            "a grid lines up: {a:?} {b:?}"
        );
        assert!(b.left() > a.right(), "side by side: {a:?} {b:?}");
    }

    /// Where the browser is, the filter, and how it lays out, orders and
    /// narrows the folder are one bar, so the folder gets the height.
    #[test]
    fn the_browsers_controls_are_one_row() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let dir = crate::asset_db::tests::Dir::new("panel-browser-bar");
        dir.write("alpha.oprefab", b"orrin-scene 1\n");
        let (db, _) = crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        editor.world.insert_resource(db);
        editor.dock.activate(Tab::Assets);
        editor.frames(2);
        let rects = editor.painted_text_rects();
        let middle = |start: &str| {
            rects
                .iter()
                .find(|(text, _)| text.starts_with(start))
                .unwrap_or_else(|| panic!("{start} is painted: {rects:?}"))
                .1
                .center()
                .y
        };
        let bar = middle("assets");
        for control in ["Filter assets", "By name", "All kinds"] {
            assert!(
                (middle(control) - bar).abs() < 3.0,
                "{control} is off the bar"
            );
        }
    }

    /// A click marks any asset, not only a prefab, in a list and in a grid:
    /// the mark is what the Import tab shows.
    #[test]
    fn a_click_marks_any_kind_of_asset() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let dir = crate::asset_db::tests::Dir::new("panel-browser-mark");
        dir.write("rock.png", b"");
        dir.write("brick.omat", b"orrin-material 1 min-reader 1\n");
        let (db, _) = crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let rock = db.find("rock.png").unwrap().id;
        let brick = db.find("brick.omat").unwrap().id;
        editor.world.insert_resource(db);
        editor.dock.activate(Tab::Assets);
        editor.frames(2);

        editor.click(row_of(&editor, "rock.png"));
        assert_eq!(editor.state.revealed_asset, Some(rock));
        editor.state.assets_view.layout = crate::editor::state::AssetLayout::Grid;
        editor.frames(2);
        editor.click(row_of(&editor, "brick.omat"));
        assert_eq!(editor.state.revealed_asset, Some(brick));
    }

    /// Rename, from a row's menu or F2 over the browser, edits the name in
    /// place with its extension kept aside; Enter renames the file, Escape
    /// leaves it as it was.
    #[test]
    fn an_asset_or_a_folder_is_renamed_in_place() {
        use crate::asset_db::AssetDatabase;
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let dir = crate::asset_db::tests::Dir::new("panel-rename");
        dir.write("rock.png", b"");
        dir.write("props/door.oprefab", b"orrin-scene 1\n");
        let (db, _) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let rock = db.find("rock.png").unwrap().id;
        editor.world.insert_resource(db);
        editor.dock.activate(Tab::Assets);
        editor.frames(2);
        let type_text = |editor: &mut Harness, text: &str| {
            editor.input.events.push(egui::Event::Text(text.to_owned()));
            editor.frames(1);
            editor.input.events.clear();
        };

        let at = row_of(&editor, "rock.png");
        secondary_click(&mut editor, at);
        editor.click(row_of(&editor, "Rename"));
        editor.frames(1);
        type_text(&mut editor, "boulder");
        editor.key(egui::Key::Enter);
        let db = editor.world.resource::<AssetDatabase>();
        assert_eq!(
            db.find("boulder.png").map(|e| e.id),
            Some(rock),
            "the stem was replaced"
        );
        drop(db);

        editor.frames(1);
        let at = row_of(&editor, "props");
        secondary_click(&mut editor, at);
        editor.click(row_of(&editor, "Rename"));
        editor.frames(1);
        type_text(&mut editor, "junk");
        editor.key(egui::Key::Escape);
        let db = editor.world.resource::<AssetDatabase>();
        assert!(
            db.folders().iter().any(|f| f.path == "props"),
            "Escape renames nothing"
        );
        drop(db);

        editor.frames(1);
        editor.state.revealed_asset = Some(rock);
        let at = row_of(&editor, "boulder.png");
        editor.hover(at).frames(1);
        editor.key(egui::Key::F2);
        editor.frames(1);
        type_text(&mut editor, "stone");
        editor.key(egui::Key::Enter);
        let db = editor.world.resource::<AssetDatabase>();
        assert_eq!(
            db.find("stone.png").map(|e| e.id),
            Some(rock),
            "F2 renames the marked one"
        );
    }

    /// A project holding one imported texture, `rocks/rock.png`, 8 by 8.
    fn imported_rock(
        editor: &Harness,
        dir: &crate::asset_db::tests::Dir,
    ) -> crate::asset_db::AssetDatabase {
        let mut png = std::io::Cursor::new(Vec::new());
        image::RgbaImage::from_pixel(8, 8, image::Rgba([200, 40, 40, 255]))
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        dir.write("rocks/rock.png", png.get_ref());
        let (mut db, _) =
            crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        assert!(db.wait_idle(std::time::Duration::from_secs(20)));
        db.update(&editor.world, &mut crate::gfx::HeadlessBackend::new());
        db
    }

    /// The Import tab shows the asset the browser marks — what it is and what
    /// its import made — and its settings, which Apply writes to the sidecar
    /// and Revert puts back.
    #[test]
    fn the_import_tab_shows_the_marked_asset_and_applies_its_settings() {
        use crate::asset_db::{AssetDatabase, Settings};
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let dir = crate::asset_db::tests::Dir::new("panel-import");
        let db = imported_rock(&editor, &dir);
        let rock = db.find("rocks/rock.png").unwrap().id;
        editor.world.insert_resource(db);
        editor.dock.activate(Tab::Import);
        editor.frames(2);
        assert!(
            editor.painted_contains("Select an asset"),
            "{:?}",
            editor.painted()
        );

        editor.state.revealed_asset = Some(rock);
        editor.frames(2);
        let id = rock.to_string();
        for shown in [
            "rock.png", "rocks", &id, "Texture", "Imported", "RGBA8", "8 × 8",
        ] {
            assert!(
                editor.painted_contains(shown),
                "{shown}: {:?}",
                editor.painted()
            );
        }
        let compress = |editor: &Harness| {
            let db = editor.world.resource::<AssetDatabase>();
            match db.get(rock).unwrap().settings {
                Settings::Texture(texture) => texture.compress,
                other => panic!("{other:?}"),
            }
        };

        editor.click(row_of(&editor, "Compress"));
        editor.click(row_of(&editor, "Revert"));
        editor.click(row_of(&editor, "Apply"));
        assert!(compress(&editor), "reverted, so nothing to apply");

        editor.click(row_of(&editor, "Compress"));
        editor.click(row_of(&editor, "Apply"));
        assert!(!compress(&editor));
        let meta = std::fs::read_to_string(dir.path().join("rocks/rock.png.meta")).unwrap();
        assert!(
            meta.contains("compress = false") && meta.contains(&id),
            "{meta}"
        );
    }

    /// Files dropped from a file explorer are copied into the open folder, or
    /// into the folder under the pointer, and the first to come in is marked.
    /// Where the pointer is not known — Windows reports no motion during an
    /// explorer's drag — the open folder takes them.
    #[test]
    fn files_dropped_on_the_browser_are_copied_into_a_folder_and_marked() {
        use crate::asset_db::AssetDatabase;
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let outside = crate::asset_db::tests::Dir::new("panel-drop-source");
        let rock = outside.write("rock.png", b"");
        let door = outside.write("door.oprefab", b"orrin-scene 1\n");
        let dir = crate::asset_db::tests::Dir::new("panel-drop");
        dir.write("props/chair.oprefab", b"orrin-scene 1\n");
        let (db, _) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        editor.world.insert_resource(db);
        editor.dock.activate(Tab::Assets);
        editor.frames(2);
        let drop_file = |editor: &mut Harness, file: &std::path::Path| {
            editor.input.dropped_files = vec![egui::DroppedFile {
                path: Some(file.to_owned()),
                ..Default::default()
            }];
            editor.frames(1);
            editor.input.dropped_files.clear();
        };

        drop_file(&mut editor, &rock);
        let db = editor.world.resource::<AssetDatabase>();
        let copied = db.find("rock.png").expect("copied into the open folder").id;
        assert_eq!(editor.state.revealed_asset, Some(copied));
        drop(db);

        editor.frames(1);
        let props = row_of(&editor, "props");
        editor.hover(props).frames(1);
        drop_file(&mut editor, &door);
        let db = editor.world.resource::<AssetDatabase>();
        let copied = db
            .find("props/door.oprefab")
            .expect("copied into the folder")
            .id;
        assert_eq!(editor.state.revealed_asset, Some(copied));
    }

    /// The bar's import button opens a file dialog, and what is picked there
    /// is copied into the open folder as a drop is. The dialog is the
    /// operating system's, stood in for here by `assets::PICKED`.
    #[test]
    fn files_picked_from_the_bar_are_copied_into_the_open_folder() {
        use crate::asset_db::AssetDatabase;
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let outside = crate::asset_db::tests::Dir::new("panel-pick-source");
        let rock = outside.write("rock.png", b"");
        let dir = crate::asset_db::tests::Dir::new("panel-pick");
        dir.write("props/chair.oprefab", b"orrin-scene 1\n");
        let (db, _) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        editor.world.insert_resource(db);
        editor.dock.activate(Tab::Assets);
        editor.frames(2);
        let browser = editor.tab_body(Tab::Assets);
        let plus = crate::editor::icons::plus().glyph();
        let button = editor
            .painted_icons()
            .into_iter()
            .find(|(glyph, rect, _)| *glyph == plus && browser.contains(rect.center()))
            .expect("the bar has an import button")
            .1;

        *assets::PICKED.lock().unwrap() = Some(vec![rock]);
        editor.click(button.center());
        let mut copied = None;
        for _ in 0..200 {
            editor.frames(1);
            copied = editor
                .world
                .resource::<AssetDatabase>()
                .find("rock.png")
                .map(|e| e.id);
            if copied.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(copied.is_some(), "copied into the open folder");
        assert_eq!(editor.state.revealed_asset, copied);
    }

    /// An old saved layout has no Import tab, so the browser's menu can bring
    /// it back for the asset it was opened on.
    #[test]
    fn an_assets_menu_opens_its_import_settings() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let dir = crate::asset_db::tests::Dir::new("panel-import-menu");
        let db = imported_rock(&editor, &dir);
        let rock = db.find("rocks/rock.png").unwrap().id;
        editor.world.insert_resource(db);
        editor.dock.toggle(Tab::Import);
        assert!(!editor.dock.is_open(Tab::Import));
        editor.dock.activate(Tab::Assets);
        editor.state.revealed_asset = Some(rock);
        editor.frames(2);

        let at = row_of(&editor, "rock.png");
        secondary_click(&mut editor, at);
        editor.click(row_of(&editor, "Import settings"));
        assert!(editor.state.take_import_request());
        editor.dock.show_import();
        assert!(editor.dock.is_open(Tab::Import));
    }

    /// What the prefab list does survives into the project's browser: a
    /// prefab row counts its instances, the filter finds an asset in any
    /// folder, and "Show in Assets" opens the folder the prefab is in.
    #[test]
    fn the_project_browser_counts_filters_and_reveals_prefabs() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let dir = crate::asset_db::tests::Dir::new("panel-browser-prefabs");
        dir.write(
            "props/door.oprefab",
            b"orrin-scene 3 min-reader 3\n\nentity 00000000-0000-4000-8000-000000000001 \"Door\"\n",
        );
        dir.write("rocks/rock.png", b"");
        let (db, _) = crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let (prefabs, issues) = crate::scene::Prefabs::scan(dir.path());
        assert!(issues.is_empty(), "{issues:?}");
        let door = db.find("props/door.oprefab").unwrap().id;
        editor.world.insert_resource(db);
        editor.world.insert_resource(prefabs);
        crate::scene::prefab::instantiate(&mut editor.world, &editor.registry, door).unwrap();
        editor.dock.activate(Tab::Assets);
        editor.frames(2);
        assert!(!editor.painted_contains("1 in scene"), "inside a folder");

        editor.state.assets_query = "door".to_owned();
        editor.frames(1);
        assert!(editor.painted_contains("door"));
        assert!(editor.painted_contains("1 in scene"));
        assert!(!editor.painted_contains("rocks"), "{:?}", editor.painted());

        editor.state.assets_query.clear();
        editor.state.revealed_asset = Some(door);
        editor.frames(2);
        assert!(editor.painted_contains("door"), "opened `props`");
        assert!(!editor.painted_contains("rocks"));
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
        assert_eq!(editor.state.selection.primary(), Some(door));
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
        editor.state.selection.set(root);
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

    /// The layer matrix is a triangle of checkboxes, each pair once, and a
    /// tick goes through the edit pipeline into both of the pair's rows.
    #[test]
    fn the_layer_matrix_unticks_a_pair_in_the_scenes_physics_settings() {
        use crate::scene::PhysicsSettings;

        let mut editor = Harness::new(egui::vec2(1280.0, 1400.0));
        let entity = editor.spawn("Physics");
        editor.world.insert(entity, PhysicsSettings::default());
        editor.state.selection.set(entity);
        editor.frames(2);
        assert!(editor.painted_contains("Physics Settings"));

        let labels = editor.painted_text_rects();
        let digit = |text: &str| -> Vec<egui::Rect> {
            let mut found: Vec<egui::Rect> = labels
                .iter()
                .filter(|(t, _)| t == text)
                .map(|(_, rect)| *rect)
                .collect();
            found.sort_by(|a, b| a.top().total_cmp(&b.top()));
            found
        };
        // The column headers sit level with the row's label; a "0" anywhere
        // else is a transform value. Row "7" is the grid's lowest.
        let layers = digit("Layers")[0];
        let column = digit("0")
            .into_iter()
            .find(|rect| (rect.center().y - layers.center().y).abs() < 4.0)
            .expect("the matrix has a column 0")
            .center()
            .x;
        let row = digit("7").last().unwrap().center().y;
        editor.click(egui::pos2(column, row));
        editor.state.apply(&mut editor.world, &editor.registry);

        let settings = editor.world.get::<PhysicsSettings>(entity).unwrap().clone();
        assert!(!settings.collides(7, 0) && !settings.collides(0, 7));
        assert!(settings.collides(7, 1) && settings.collides(3, 3));
    }

    /// Where the row naming `name` was painted.
    fn row_of(editor: &Harness, name: &str) -> egui::Pos2 {
        editor
            .painted_text_rects()
            .into_iter()
            .find(|(text, _)| text == name)
            .unwrap_or_else(|| panic!("no row for {name}"))
            .1
            .center()
    }

    #[test]
    fn ctrl_click_adds_rows_to_the_selection() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let alpha = editor.spawn("Alpha");
        let beta = editor.spawn("Beta");
        editor.spawn("Gamma");
        editor.frames(2);
        let (a, b) = (row_of(&editor, "Alpha"), row_of(&editor, "Beta"));

        editor
            .click(a)
            .click_with(b, egui::Modifiers::COMMAND)
            .frames(1);
        assert_eq!(editor.state.selection.entities(), [alpha, beta]);
        assert!(editor.painted_contains("3 entities · 2 selected"));

        editor.click_with(a, egui::Modifiers::COMMAND);
        assert_eq!(editor.state.selection.entities(), [beta]);
    }

    #[test]
    fn shift_click_selects_the_rows_between() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.spawn("Alpha");
        let beta = editor.spawn("Beta");
        let gamma = editor.spawn("Gamma");
        editor.frames(2);
        let (a, c) = (row_of(&editor, "Alpha"), row_of(&editor, "Gamma"));

        editor.click(a).click_with(c, egui::Modifiers::SHIFT);
        assert_eq!(editor.state.selection.len(), 3);
        assert!(editor.state.selection.contains(beta));
        assert_eq!(editor.state.selection.primary(), Some(gamma));
    }

    #[test]
    fn delete_takes_every_selected_entity() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let alpha = editor.spawn("Alpha");
        let beta = editor.spawn("Beta");
        let gamma = editor.spawn("Gamma");
        editor.state.selection.set(alpha);
        editor.state.selection.toggle(beta);
        editor.frames(1).key(egui::Key::Delete);
        editor.state.apply(&mut editor.world, &editor.registry);

        assert!(!editor.world.is_alive(alpha));
        assert!(!editor.world.is_alive(beta));
        assert!(editor.world.is_alive(gamma));
        assert!(editor.state.selection.is_empty());
    }

    /// A real drag, frame by frame: press on a selected row, move past egui's
    /// drag threshold, and let go over another row.
    #[test]
    fn dragging_a_selected_row_moves_the_whole_selection() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let room = editor.spawn("Room");
        let alpha = editor.spawn("Alpha");
        let beta = editor.spawn("Beta");
        editor.state.selection.set(alpha);
        editor.state.selection.toggle(beta);
        editor.frames(2);
        let (from, to) = (row_of(&editor, "Alpha"), row_of(&editor, "Room"));

        let button = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        };
        for events in [
            vec![egui::Event::PointerMoved(from), button(from, true)],
            vec![egui::Event::PointerMoved(from + egui::vec2(0.0, 12.0))],
            vec![egui::Event::PointerMoved(to)],
            vec![button(to, false)],
        ] {
            editor.input.events = events;
            editor.frames(1);
        }
        editor.input.events.clear();
        editor.state.apply(&mut editor.world, &editor.registry);

        let parent = |e| editor.world.get::<crate::scene::Parent>(e).map(|p| p.get());
        assert_eq!(parent(alpha), Some(room));
        assert_eq!(parent(beta), Some(room));
    }

    #[test]
    fn save_as_prefab_saves_the_whole_selection() {
        let dir = std::env::temp_dir().join(format!("orrin-save-prefab-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor
            .world
            .insert_resource(crate::scene::Prefabs::scan(&dir).0);
        let alpha = editor.spawn("Alpha");
        let beta = editor.spawn("Beta");
        editor.state.selection.set(alpha);
        editor.state.selection.toggle(beta);

        editor.state.request_save_prefab();
        editor.state.apply(&mut editor.world, &editor.registry);
        let saved = dir.join("Group.oprefab").is_file();
        let _ = std::fs::remove_dir_all(&dir);

        assert!(saved, "the selection was saved under its new root's name");
        let root = editor
            .state
            .selection
            .primary()
            .expect("the new root is selected");
        assert!(editor.world.has::<crate::scene::PrefabInstance>(root));
    }

    /// The colour of every painted string equal to `needle`, as it reaches the
    /// screen: a section's own colour, or the shape's fallback for one with
    /// none. Every match, because a label can appear in more than one place.
    fn colours_of(editor: &Harness, needle: &str) -> Vec<egui::Color32> {
        fn find(shape: &egui::Shape, needle: &str, out: &mut Vec<egui::Color32>) {
            match shape {
                egui::Shape::Text(text) if text.galley.text() == needle => {
                    if let Some(section) = text.galley.job.sections.first() {
                        out.push(if section.format.color == egui::Color32::PLACEHOLDER {
                            text.fallback_color
                        } else {
                            section.format.color
                        });
                    }
                }
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| find(s, needle, out)),
                _ => {}
            }
        }
        let mut out = Vec::new();
        for clipped in &editor.output.as_ref().expect("a frame has run").shapes {
            find(&clipped.shape, needle, &mut out);
        }
        out
    }

    fn secondary_click(editor: &mut Harness, at: egui::Pos2) {
        editor.input.events.push(egui::Event::PointerMoved(at));
        for pressed in [true, false] {
            editor.input.events.push(egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Secondary,
                pressed,
                modifiers: Default::default(),
            });
        }
        editor.frames(1);
        editor.input.events.clear();
        editor.frames(1);
    }

    fn place_door(editor: &mut Harness) -> (orrin_ecs::Entity, orrin_registry::AssetId) {
        let id = door(editor);
        let (root, _) =
            crate::scene::prefab::instantiate(&mut editor.world, &editor.registry, id).unwrap();
        (root, id)
    }

    /// An instance reads as one at a glance: its rows wear the prefab colour.
    #[test]
    fn an_instance_reads_as_one_in_the_tree() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.spawn("Plain");
        place_door(&mut editor);
        editor.frames(2);

        assert_eq!(colours_of(&editor, "Door"), [theme::PREFAB]);
        assert!(!colours_of(&editor, "Plain").contains(&theme::PREFAB));
    }

    #[test]
    fn right_clicking_an_instance_offers_what_can_be_done_to_it() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (door, _) = place_door(&mut editor);
        editor.frames(2);
        let row = row_of(&editor, "Door");

        secondary_click(&mut editor, row);
        assert_eq!(
            editor.state.selection.primary(),
            Some(door),
            "the row is selected"
        );
        for item in [
            "Save as prefab",
            "Apply to prefab",
            "Revert to prefab",
            "Unpack",
            "Show in Assets",
        ] {
            assert!(editor.painted_contains(item), "the menu lacks {item:?}");
        }
    }

    #[test]
    fn an_unpack_request_leaves_plain_entities() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (door, _) = place_door(&mut editor);
        editor
            .state
            .request_prefab_action(crate::editor::state::PrefabAction::Unpack, door);
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(!editor.world.has::<crate::scene::PrefabInstance>(door));

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        let door = named(&editor.world, "Door").unwrap();
        assert!(editor.world.has::<crate::scene::PrefabInstance>(door));
    }

    #[test]
    fn reverting_an_instance_can_be_undone() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (door, _) = place_door(&mut editor);
        editor
            .world
            .insert(door, crate::scene::Tag::new("entrance"));
        editor
            .state
            .request_prefab_action(crate::editor::state::PrefabAction::Revert, door);
        editor.state.apply(&mut editor.world, &editor.registry);
        let door = named(&editor.world, "Door").unwrap();
        assert!(!editor.world.has::<crate::scene::Tag>(door));

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        let door = named(&editor.world, "Door").unwrap();
        assert!(editor.world.has::<crate::scene::Tag>(door));
        assert!(editor.world.has::<crate::scene::PrefabInstance>(door));
    }

    #[test]
    fn show_in_assets_marks_the_prefab_and_asks_for_the_tab() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (door, id) = place_door(&mut editor);
        editor
            .state
            .request_prefab_action(crate::editor::state::PrefabAction::Reveal, door);
        editor.state.apply(&mut editor.world, &editor.registry);
        assert_eq!(editor.state.revealed_asset, Some(id));
        assert!(editor.state.take_reveal_request());
        assert!(!editor.state.take_reveal_request(), "asked once");
    }

    /// A component an instance overrides wears the prefab colour on its bar,
    /// and the Prefab section offers the whole-instance actions.
    #[test]
    fn the_inspector_marks_overridden_components_and_offers_the_actions() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (door, _) = place_door(&mut editor);
        editor
            .world
            .insert(door, crate::scene::Tag::new("entrance"));
        editor.state.selection.set(door);
        editor.frames(2);

        // "Tag" is also the label of the Prefab section's row for the override;
        // it is the section's own title that has to say so.
        assert!(colours_of(&editor, "Tag").contains(&theme::PREFAB));
        assert!(!colours_of(&editor, "Transform").contains(&theme::PREFAB));
        for action in ["Apply", "Revert all", "Unpack"] {
            assert!(editor.painted_contains(action), "no {action:?} button");
        }
    }

    #[test]
    fn the_assets_tool_counts_instances_and_filters() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (_, id) = place_door(&mut editor);
        crate::scene::prefab::instantiate(&mut editor.world, &editor.registry, id).unwrap();
        let crate_ = crate::scene::prefab::Prefab::parse(
            orrin_registry::AssetId::new(),
            "props/crate.oprefab",
            "orrin-scene 1\n\nentity 00000000-0000-4000-8000-000000000002\n  orrin.name = \"Crate\"\n",
        )
        .unwrap();
        editor
            .world
            .resource_mut::<crate::scene::Prefabs>()
            .insert(crate_);
        editor.dock.activate(Tab::Assets);
        editor.frames(2);
        assert!(editor.painted_contains("2 in scene"));
        assert!(editor.painted_contains("crate"));

        editor.state.assets_query = "crate".to_owned();
        editor.frames(1);
        assert!(!editor.painted_contains("door"));
        assert!(editor.painted_contains("crate"));

        editor.state.assets_query = "barrel".to_owned();
        editor.frames(1);
        assert!(editor.painted_contains("No prefab matches"));
    }

    #[test]
    fn double_clicking_a_prefab_places_one() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        door(&mut editor);
        editor.dock.activate(Tab::Assets);
        editor.frames(2);
        let row = row_of(&editor, "door");

        editor.input.events.push(egui::Event::PointerMoved(row));
        for _ in 0..2 {
            for pressed in [true, false] {
                editor.input.events.push(egui::Event::PointerButton {
                    pos: row,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                });
            }
        }
        editor.frames(1);
        editor.input.events.clear();
        editor.state.apply(&mut editor.world, &editor.registry);
        the_door(&editor);
    }

    /// A project of two sounds and a texture, each in a folder of its own so
    /// the browser's root lists only folders. The directory is returned so it
    /// outlives the test.
    fn sound_project(
        editor: &mut Harness,
        name: &str,
    ) -> (crate::asset_db::tests::Dir, [orrin_registry::AssetId; 3]) {
        let dir = crate::asset_db::tests::Dir::new(name);
        let wav = crate::audio::clip::tests::wav(1, 48_000, &[0; 64]);
        dir.write("sfx/door.wav", &wav);
        dir.write("music/theme.wav", &wav);
        dir.write("textures/rock.png", b"");
        let (db, _) = crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let ids = ["sfx/door.wav", "music/theme.wav", "textures/rock.png"]
            .map(|path| db.find(path).unwrap().id);
        editor.world.insert_resource(db);
        (dir, ids)
    }

    fn select_speaker(editor: &mut Harness) -> orrin_ecs::Entity {
        let speaker = editor.spawn("Speaker");
        editor
            .world
            .insert(speaker, crate::scene::AudioSource::default());
        editor.state.selection.set(speaker);
        editor.frames(2);
        speaker
    }

    fn clip_of(editor: &Harness, speaker: orrin_ecs::Entity) -> orrin_registry::AssetId {
        editor
            .world
            .get::<crate::scene::AudioSource>(speaker)
            .unwrap()
            .clip
    }

    fn previewing(editor: &Harness) -> Option<orrin_registry::AssetId> {
        editor.world.resource::<crate::audio::Audio>().previewing()
    }

    /// The clip is a slot that names its sound rather than printing its id,
    /// and takes a sound and nothing else.
    #[test]
    fn the_audio_source_clip_is_a_sound_slot_that_names_its_sound() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (_dir, [door, _, rock]) = sound_project(&mut editor, "panel-audio-slot");
        let speaker = select_speaker(&mut editor);
        let slot = row_of(&editor, "No sound");

        drop_payload(&mut editor, rock, slot);
        assert!(clip_of(&editor, speaker).is_nil(), "a texture is no clip");

        drop_payload(&mut editor, door, slot);
        assert_eq!(clip_of(&editor, speaker), door);
        editor.frames(1);
        assert!(editor.painted_contains("door.wav"));
        assert!(!editor.painted_contains(&door.to_string()));
    }

    #[test]
    fn the_clip_picker_offers_the_projects_sounds_and_nothing_else() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (_dir, [_, theme, _]) = sound_project(&mut editor, "panel-audio-picker");
        let speaker = select_speaker(&mut editor);

        editor.click(row_of(&editor, "No sound"));
        editor.frames(1);
        assert!(editor.painted_contains("sfx/door.wav"));
        assert!(editor.painted_contains("music/theme.wav"));
        assert!(!editor.painted_contains("rock"));

        editor.click(row_of(&editor, "music/theme.wav"));
        editor.state.apply(&mut editor.world, &editor.registry);
        assert_eq!(clip_of(&editor, speaker), theme);
    }

    /// The clip's row carries a play button that auditions the sound in edit
    /// mode, and turns into a stop button while it does.
    #[test]
    fn the_clip_row_previews_its_sound() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (_dir, [door, _, _]) = sound_project(&mut editor, "panel-audio-row");
        let speaker = select_speaker(&mut editor);
        editor
            .world
            .get_mut::<crate::scene::AudioSource>(speaker)
            .unwrap()
            .clip = door;
        editor.frames(1);
        let slot = editor
            .painted_text_rects()
            .into_iter()
            .find(|(text, _)| text == "door.wav")
            .expect("the slot names the sound")
            .1;
        let button = |editor: &Harness, glyph: char| {
            editor
                .painted_icons()
                .into_iter()
                .find(|(g, rect, _)| {
                    *g == glyph
                        && rect.left() > slot.right()
                        && (rect.center().y - slot.center().y).abs() < 24.0
                })
                .map(|(_, rect, _)| rect.center())
        };

        let play = button(&editor, crate::editor::icons::play().glyph()).expect("a play button");
        editor.click(play);
        assert_eq!(previewing(&editor), Some(door));

        editor.frames(1);
        let stop = button(&editor, crate::editor::icons::stop().glyph()).expect("a stop button");
        editor.click(stop);
        assert_eq!(previewing(&editor), None);
    }

    /// The preview button sits in the middle of the clip's slot, and its glyph
    /// in the middle of the frame it wears while playing.
    #[test]
    fn the_preview_button_is_centred_on_the_slot_and_its_glyph_in_its_frame() {
        // At 2x as well: a Retina display snaps glyphs and frames to a finer
        // grid, and an offset hidden by rounding at 1x can show there.
        for scale in [1.0, 2.0] {
            preview_button_is_centred_at(scale);
        }
    }

    fn preview_button_is_centred_at(scale: f32) {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.ctx.set_pixels_per_point(scale);
        let (_dir, [door, _, _]) = sound_project(&mut editor, "panel-audio-centre");
        let speaker = select_speaker(&mut editor);
        editor
            .world
            .get_mut::<crate::scene::AudioSource>(speaker)
            .unwrap()
            .clip = door;
        editor
            .world
            .resource_mut::<crate::audio::Audio>()
            .preview(door);
        editor.frames(2);

        let slot = editor
            .painted_rects_of(|shape| matches!(shape, egui::Shape::Rect(_)))
            .into_iter()
            .filter(|rect| {
                editor
                    .painted_text_rects()
                    .iter()
                    .any(|(text, at)| text == "door.wav" && rect.contains_rect(*at))
            })
            .min_by(|a, b| a.area().total_cmp(&b.area()))
            .expect("the slot's box");
        let stop = crate::editor::icons::stop().glyph();
        let glyph = editor
            .painted_icons()
            .into_iter()
            .find(|(g, rect, _)| *g == stop && rect.left() > slot.right())
            .expect("a stop button")
            .1;
        let frame = editor
            .painted_rects_of(|shape| matches!(shape, egui::Shape::Rect(_)))
            .into_iter()
            .filter(|rect| rect.contains(glyph.center()) && rect.width() < 40.0)
            .min_by(|a, b| a.area().total_cmp(&b.area()))
            .expect("the button's frame");

        // What the eye centres is the glyph's ink, which an icon font places
        // anywhere in its line box; the line box being centred proves nothing.
        fn ink(shape: &egui::Shape, glyph: char, out: &mut Option<egui::Rect>) {
            match shape {
                egui::Shape::Text(text) => {
                    for row in &text.galley.rows {
                        for g in row.glyphs.iter().filter(|g| g.chr == glyph) {
                            let min = text.pos + g.pos.to_vec2() + g.uv_rect.offset;
                            *out = Some(egui::Rect::from_min_size(min, g.uv_rect.size));
                        }
                    }
                }
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| ink(s, glyph, out)),
                _ => {}
            }
        }
        let mut drawn = None;
        for clipped in &editor.output.as_ref().unwrap().shapes {
            let mut found = None;
            ink(&clipped.shape, stop, &mut found);
            if let Some(rect) = found.filter(|rect| frame.contains_rect(*rect)) {
                drawn = Some(rect);
            }
        }
        let drawn = drawn.expect("the stop glyph's ink, inside the frame");

        let off = frame.center() - drawn.center();
        assert!(
            off.x.abs() <= 0.5 && off.y.abs() <= 0.5,
            "at {scale}x, ink {drawn:?} sits {off:?} from the middle of its frame {frame:?}"
        );
        let off = frame.center().y - slot.center().y;
        assert!(
            off.abs() <= 1.0,
            "at {scale}x, button {frame:?} sits {off} from the middle of the slot {slot:?}"
        );
        assert!(
            frame.width() <= orrin_registry::inspect::TRAIL + 0.5,
            "button {frame:?} spills out of the row's {} px trailing slot",
            orrin_registry::inspect::TRAIL
        );
    }

    #[test]
    fn double_clicking_a_sound_in_the_browser_previews_it_and_again_stops_it() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (_dir, [door, _, _]) = sound_project(&mut editor, "panel-audio-browser");
        editor.dock.activate(Tab::Assets);
        editor.frames(2);
        editor.click(row_of(&editor, "sfx"));
        editor.frames(1);

        editor.input.time = Some(0.0);
        editor.double_click(row_of(&editor, "door.wav"));
        assert_eq!(previewing(&editor), Some(door));
        // Far enough apart that egui does not count clicks three and four as a
        // triple-click.
        editor.input.time = Some(2.0);
        editor.double_click(row_of(&editor, "door.wav"));
        assert_eq!(previewing(&editor), None);
    }

    /// Let go of a prefab over the scene, and it stands on the ground where the
    /// pointer was.
    #[test]
    fn a_prefab_dropped_on_the_scene_lands_under_the_pointer() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let id = door(&mut editor);
        editor.frames(2);
        let at = editor.viewport().center();

        drop_payload(&mut editor, id, at);
        let door = the_door(&editor);
        let camera = *editor.world.resource::<crate::scene::Camera>();
        let expected = super::viewport::ground_point(
            &camera,
            1280.0 / 800.0,
            egui::vec2(at.x / 1280.0 * 2.0 - 1.0, at.y / 800.0 * 2.0 - 1.0),
        )
        .expect("the default camera looks down at the ground");
        let placed = editor
            .world
            .get::<LocalTransform>(door)
            .unwrap()
            .translation;
        assert!(
            (placed - expected).length() < 1e-4,
            "{placed} vs {expected}"
        );
        assert!(placed.y.abs() < 1e-4);
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

    /// The entity menu's UI entries sit in a submenu of their own, which lists
    /// every kind the menu can make.
    #[test]
    fn the_entity_menu_offers_ui_in_a_submenu() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let run = |editor: &mut Harness| {
            let Harness {
                state, ctx, input, ..
            } = &mut *editor;
            let output = ctx.run(input.clone(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| menu_bar::spawn_entries(ui, state));
            });
            editor.output = Some(output);
        };
        run(&mut editor);
        run(&mut editor);
        let at = editor
            .painted_text_rects()
            .into_iter()
            .find(|(text, _)| without_icons(text) == "UI")
            .map(|(_, rect)| rect.center())
            .expect("a UI entry");
        editor.input.events.push(egui::Event::PointerMoved(at));
        editor.input.events.push(button(at, true));
        editor.input.events.push(button(at, false));
        run(&mut editor);
        editor.input.events.clear();
        run(&mut editor);
        let painted = editor.painted();
        for kind in ["Canvas", "Panel", "Image", "Text", "Button"] {
            assert!(
                painted.iter().any(|text| text == kind),
                "{kind}: {painted:?}"
            );
        }
    }

    /// The submenu offers the scroll frame beside the other kinds.
    #[test]
    fn the_ui_submenu_offers_a_scroll_frame() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let run = |editor: &mut Harness| {
            let Harness {
                state, ctx, input, ..
            } = &mut *editor;
            let output = ctx.run(input.clone(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| menu_bar::spawn_entries(ui, state));
            });
            editor.output = Some(output);
        };
        run(&mut editor);
        run(&mut editor);
        let at = editor
            .painted_text_rects()
            .into_iter()
            .find(|(text, _)| without_icons(text) == "UI")
            .map(|(_, rect)| rect.center())
            .expect("a UI entry");
        editor.input.events.push(egui::Event::PointerMoved(at));
        editor.input.events.push(button(at, true));
        editor.input.events.push(button(at, false));
        run(&mut editor);
        editor.input.events.clear();
        run(&mut editor);
        let painted = editor.painted();
        assert!(
            painted.iter().any(|text| text == "Scroll Frame"),
            "{painted:?}"
        );
    }

    /// A UI element's row wears the icon of what it is, not a group's dot.
    #[test]
    fn hierarchy_rows_show_what_a_ui_element_is() {
        use crate::ui::create::UiKind;
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        ui_spawn(&mut editor, UiKind::Button);
        editor.dock.activate(Tab::Hierarchy);
        editor.frames(2);
        let tree = editor.tab_body(Tab::Hierarchy);
        let painted: Vec<char> = editor
            .painted_icons()
            .into_iter()
            .filter(|(_, rect, _)| tree.contains(rect.center()))
            .map(|(glyph, ..)| glyph)
            .collect();
        for icon in [
            crate::editor::icons::app_window(),
            crate::editor::icons::cursor_click(),
        ] {
            assert!(painted.contains(&icon.glyph()), "{icon:?} in {painted:?}");
        }
    }

    /// Every entry of the UI submenu wears an icon, the one its component's
    /// inspector section shows where it has one.
    #[test]
    fn the_ui_submenu_shows_an_icon_beside_each_entry() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let run = |editor: &mut Harness| {
            let Harness {
                state, ctx, input, ..
            } = &mut *editor;
            let output = ctx.run(input.clone(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| menu_bar::spawn_entries(ui, state));
            });
            editor.output = Some(output);
        };
        run(&mut editor);
        run(&mut editor);
        let at = editor
            .painted_text_rects()
            .into_iter()
            .find(|(text, _)| without_icons(text) == "UI")
            .map(|(_, rect)| rect.center())
            .expect("a UI entry");
        editor.input.events.push(egui::Event::PointerMoved(at));
        editor.input.events.push(button(at, true));
        editor.input.events.push(button(at, false));
        run(&mut editor);
        editor.input.events.clear();
        run(&mut editor);
        let texts = editor.painted_text_rects();
        let icons = editor.painted_icons();
        use crate::editor::icons as glyphs;
        for (label, icon) in [
            ("Canvas", glyphs::app_window()),
            ("Panel", glyphs::rectangle()),
            ("Image", glyphs::image()),
            ("Text", glyphs::text_t()),
            ("Button", glyphs::cursor_click()),
            ("Scroll Frame", glyphs::mouse_scroll()),
        ] {
            let row = texts
                .iter()
                .find(|(text, _)| without_icons(text) == label)
                .map(|(_, rect)| *rect)
                .unwrap_or_else(|| panic!("no {label} entry"));
            assert!(
                icons.iter().any(|(glyph, rect, _)| *glyph == icon.glyph()
                    && (rect.center().y - row.center().y).abs() < 4.0),
                "{label} has no icon"
            );
        }
    }

    /// A previewed resolution is the game's screen: the UI is told its size,
    /// it is drawn inside the viewport, and the viewport around it is dimmed.
    #[test]
    fn previewing_a_resolution_tells_the_game_its_screen_and_dims_the_rest() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        let window = *editor.world.resource::<crate::scene::GameScreen>();
        assert_eq!(
            window.size,
            [1280.0, 800.0],
            "the window until one is picked"
        );

        editor.state.preview.resolution = Some([1080, 2340]);
        editor.frames(2);
        let screen = *editor.world.resource::<crate::scene::GameScreen>();
        assert_eq!(screen.size, [1080.0, 2340.0]);
        let viewport = editor.dock.viewport().expect("a viewport");
        let shown = screen.rect();
        assert!(
            viewport
                .expand(0.5)
                .contains_rect(egui::Rect::from_min_size(
                    egui::pos2(shown.x, shown.y),
                    egui::vec2(shown.width, shown.height),
                )),
            "{shown:?} is not inside {viewport:?}"
        );
        assert!(
            (shown.height - viewport.height()).abs() < 1.0,
            "fitted to the height"
        );
        let tree = editor.world.resource::<crate::ui::UiTree>();
        assert_eq!(tree.canvas.screen(), [1080.0, 2340.0]);
        drop(tree);
        let dim = egui::Color32::from_black_alpha(150);
        assert!(
            editor.painted_fills().iter().any(|(_, fill)| *fill == dim),
            "the sides of the viewport are dimmed"
        );
    }

    /// In UI mode a button over the viewport opens the templates and the two
    /// sizes.
    #[test]
    fn the_resolution_menu_picks_a_template_and_true_size() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.state.ui_mode = true;
        editor.frames(2);
        let text_at = |editor: &Harness, wanted: &str| {
            editor
                .painted_text_rects()
                .into_iter()
                .find(|(text, _)| without_icons(text).starts_with(wanted))
                .map(|(_, rect)| rect.center())
        };
        let monitor = crate::editor::icons::monitor().glyph();
        let viewport = editor.dock.viewport().expect("a viewport");
        let button = editor
            .painted_icons()
            .into_iter()
            .find(|(glyph, rect, _)| *glyph == monitor && viewport.contains(rect.center()))
            .expect("the resolution button")
            .1;
        editor.click(button.center()).frames(1);
        assert!(
            text_at(&editor, "Window").is_some(),
            "the menu offers the window"
        );
        let full_hd = text_at(&editor, "Full HD").expect("the templates are listed");
        editor.click(full_hd).frames(1);
        assert_eq!(editor.state.preview.resolution, Some([1920, 1080]));

        let true_size = text_at(&editor, "True size").expect("the sizes are listed");
        editor.click(true_size).frames(2);
        assert_eq!(
            editor.state.preview.size,
            crate::editor::preview::PreviewSize::TrueSize
        );
        assert_eq!(
            editor.world.resource::<crate::scene::GameScreen>().scale,
            1.0
        );
    }

    /// The hierarchy's add button holds the Entity menu's own entries, so its
    /// UI entry has to open the submenu there too, rather than close the whole
    /// list and need opening again.
    #[test]
    fn the_hierarchy_add_button_opens_the_ui_submenu_and_builds_from_it() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.dock.activate(Tab::Hierarchy);
        editor.frames(2);
        let tree = editor.tab_body(Tab::Hierarchy);
        let plus = crate::editor::icons::plus().glyph();
        let add = editor
            .painted_icons()
            .into_iter()
            .find(|(glyph, rect, _)| *glyph == plus && tree.contains(rect.center()))
            .expect("the hierarchy has an add button")
            .1;
        editor.click(add.center()).frames(1);
        let text_at = |editor: &Harness, wanted: &str| {
            editor
                .painted_text_rects()
                .into_iter()
                .find(|(text, _)| without_icons(text) == wanted)
                .map(|(_, rect)| rect.center())
        };
        let ui = text_at(&editor, "UI").expect("the add list is open");
        editor.click(ui).frames(1);
        let button = text_at(&editor, "Button").expect("the UI submenu is open");
        assert!(
            text_at(&editor, "Cube").is_some(),
            "and the list behind it still is"
        );

        editor.click(button).frames(1);
        editor.state.apply(&mut editor.world, &editor.registry);
        let buttons = editor.world.query::<&crate::ui::Button>().count();
        assert_eq!(buttons, 1);
        assert!(
            text_at(&editor, "Canvas").is_none(),
            "picking closes the menu"
        );
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

    /// Opens File, then Export game, and says whether each platform's entry
    /// was reached and which export the editor was asked for.
    fn export_from_the_file_menu(editor: &mut Harness, platform: &str) -> bool {
        editor.frames(2);
        let file = text_at(editor, "File").expect("the File menu's title");
        editor.click(file).frames(2);
        let entry = text_at(editor, "Export game").expect("File lists Export game");
        editor.click(entry).frames(2);
        match text_at(editor, platform) {
            Some(at) => {
                editor.click(at).frames(1);
                true
            }
            None => false,
        }
    }

    #[test]
    fn file_export_game_asks_for_the_platform_picked() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.state.project_name = Some("Courtyard".to_owned());
        editor.state.project_root = Some(std::path::PathBuf::from("courtyard"));

        assert!(export_from_the_file_menu(&mut editor, "Windows"));
        assert_eq!(
            editor.state.take_export_request(),
            Some(orrin_project::export::Platform::Windows)
        );
    }

    /// Outside a project there is nothing to export: the entry is listed, so it
    /// can be found, and opens nothing.
    #[test]
    fn file_export_game_is_listed_and_closed_outside_a_project() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));

        assert!(!export_from_the_file_menu(&mut editor, "Windows"));
        assert_eq!(editor.state.take_export_request(), None);
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

    /// Pause and step are disabled until Play is pressed, but they are the
    /// bar's centrepiece: drawn at egui's faded disabled alpha, in a sunken
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
        editor.state.selection.set(ground);
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
        editor.state.selection.set(first);
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

    /// A cube the camera sees at the middle of the viewport, measured so it
    /// can be picked, and where to click to hit it.
    fn cube_under_the_viewport_centre(editor: &mut Harness) -> (orrin_ecs::Entity, egui::Pos2) {
        let mut bounds = crate::scene::MeshBounds::default();
        bounds.insert(
            crate::scene::MeshHandle(0),
            crate::geom::Aabb {
                min: glam::Vec3A::splat(-0.5),
                max: glam::Vec3A::splat(0.5),
            },
        );
        editor.world.insert_resource(bounds);
        let mut assets = crate::scene::Assets::new();
        assets.insert_mesh("cube", crate::scene::MeshHandle(0));
        editor.world.insert_resource(assets);
        editor.frames(2);
        let at = editor.viewport().center();
        let screen = editor.ctx.screen_rect();
        let ndc = (at - screen.min) / screen.size() * 2.0 - egui::Vec2::splat(1.0);
        let camera = *editor.world.resource::<crate::scene::Camera>();
        let (origin, direction) = crate::editor::pick::ray(&camera, screen.aspect_ratio(), ndc);

        let entity = editor.spawn("Cube");
        editor.world.insert(entity, crate::scene::MeshHandle(0));
        editor.world.insert(
            entity,
            LocalTransform::new(crate::scene::Transform::from_translation(
                origin + direction.normalize() * 5.0,
            )),
        );
        crate::scene::propagate_transforms(&mut editor.world);
        (entity, at)
    }

    #[test]
    fn clicking_the_scene_selects_what_is_under_the_pointer() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (cube, at) = cube_under_the_viewport_centre(&mut editor);
        editor.click(at);
        assert_eq!(editor.state.selection.entities(), [cube]);

        let empty = editor.viewport().left_bottom() + egui::vec2(40.0, -40.0);
        editor.click(empty);
        assert!(
            editor.state.selection.is_empty(),
            "empty space selects nothing"
        );
    }

    #[test]
    fn ctrl_clicking_the_scene_adds_to_the_selection() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let other = editor.spawn("Other");
        let (cube, at) = cube_under_the_viewport_centre(&mut editor);
        editor.state.selection.set(other);
        editor.click_with(at, egui::Modifiers::COMMAND);
        assert_eq!(editor.state.selection.entities(), [other, cube]);
        editor.click_with(at, egui::Modifiers::COMMAND);
        assert_eq!(editor.state.selection.entities(), [other]);
    }

    /// Where a point along the gizmo's X shaft is on screen: `fraction` of the
    /// way from the entity's origin to the tip.
    fn x_shaft(editor: &Harness, entity: orrin_ecs::Entity, fraction: f32) -> egui::Pos2 {
        let origin = editor
            .world
            .get::<crate::scene::WorldTransform>(entity)
            .unwrap()
            .translation();
        x_shaft_from(editor, origin, fraction)
    }

    /// [`x_shaft`] for a gizmo standing at `origin`.
    fn x_shaft_from(editor: &Harness, origin: glam::Vec3, fraction: f32) -> egui::Pos2 {
        let camera = *editor.world.resource::<crate::scene::Camera>();
        let screen = editor.ctx.screen_rect();
        let view_proj = camera.view_projection(screen.aspect_ratio());
        let project = |point: glam::Vec3| {
            let ndc = view_proj.project_point3(point);
            screen.min
                + egui::vec2(
                    (ndc.x * 0.5 + 0.5) * screen.width(),
                    (ndc.y * 0.5 + 0.5) * screen.height(),
                )
        };
        let start = project(origin);
        let toward = (project(origin + glam::Vec3::X * 0.01) - start).normalized();
        start + toward * crate::editor::gizmo::LENGTH * fraction
    }

    fn pointer(editor: &mut Harness, event: egui::Event) {
        editor.input.events.push(event);
        editor.frames(1);
        editor.input.events.clear();
        editor.state.apply(&mut editor.world, &editor.registry);
    }

    fn button(pos: egui::Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        }
    }

    /// The gizmo's reason to exist, through the real pointer: a drag on the X
    /// shaft moves the selection along X, shows on every frame, and undoes in
    /// one step.
    #[test]
    fn dragging_the_move_handle_moves_the_selection_in_one_undo_step() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (cube, _) = cube_under_the_viewport_centre(&mut editor);
        let before = *editor.world.get::<LocalTransform>(cube).unwrap();
        editor.state.selection.set(cube);
        editor.state.gizmo_mode = crate::editor::state::GizmoMode::Move;
        editor.frames(2);

        let grab = x_shaft(&editor, cube, 0.5);
        pointer(&mut editor, egui::Event::PointerMoved(grab));
        pointer(&mut editor, button(grab, true));
        let mut at = grab;
        for _ in 0..3 {
            at += (grab - x_shaft(&editor, cube, 0.0)).normalized() * 20.0;
            pointer(&mut editor, egui::Event::PointerMoved(at));
        }
        let during = editor
            .world
            .get::<LocalTransform>(cube)
            .unwrap()
            .translation;
        assert!(
            during.x > before.translation.x + 0.1,
            "{during} from {}",
            before.translation
        );
        assert!((during.y - before.translation.y).abs() < 1e-4);
        assert!((during.z - before.translation.z).abs() < 1e-4);
        pointer(&mut editor, button(at, false));
        assert_eq!(editor.state.selection.entities(), [cube]);

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        let after = editor
            .world
            .get::<LocalTransform>(cube)
            .unwrap()
            .translation;
        assert!((after - before.translation).length() < 1e-5, "{after}");
        assert_eq!(editor.state.history.undo_label(), None, "one step");
    }

    #[test]
    fn dragging_a_scale_tip_scales_the_selection_in_one_undo_step() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (cube, _) = cube_under_the_viewport_centre(&mut editor);
        editor.state.selection.set(cube);
        editor.state.gizmo_mode = crate::editor::state::GizmoMode::Scale;
        editor.frames(2);

        let grab = x_shaft(&editor, cube, 0.95);
        let outward = (grab - x_shaft(&editor, cube, 0.0)).normalized();
        pointer(&mut editor, egui::Event::PointerMoved(grab));
        pointer(&mut editor, button(grab, true));
        let mut at = grab;
        for _ in 0..3 {
            at += outward * 20.0;
            pointer(&mut editor, egui::Event::PointerMoved(at));
        }
        pointer(&mut editor, button(at, false));
        let scale = editor.world.get::<LocalTransform>(cube).unwrap().scale;
        assert!(scale.x > 1.2, "{scale}");
        assert!(
            (scale.y - 1.0).abs() < 1e-5 && (scale.z - 1.0).abs() < 1e-5,
            "{scale}"
        );

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        let scale = editor.world.get::<LocalTransform>(cube).unwrap().scale;
        assert!((scale - glam::Vec3::ONE).length() < 1e-5, "{scale}");
        assert_eq!(editor.state.history.undo_label(), None, "one step");
    }

    /// Drag the X shaft of a gizmo standing at `origin` by three steps.
    fn drag_x_shaft_from(editor: &mut Harness, origin: glam::Vec3) {
        let grab = x_shaft_from(editor, origin, 0.5);
        let outward = (grab - x_shaft_from(editor, origin, 0.0)).normalized();
        pointer(editor, egui::Event::PointerMoved(grab));
        pointer(editor, button(grab, true));
        let mut at = grab;
        for _ in 0..3 {
            at += outward * 20.0;
            pointer(editor, egui::Event::PointerMoved(at));
        }
        pointer(editor, button(at, false));
    }

    fn translation(editor: &Harness, entity: orrin_ecs::Entity) -> glam::Vec3 {
        editor
            .world
            .get::<LocalTransform>(entity)
            .unwrap()
            .translation
    }

    /// With two selected the gizmo stands between them, and moves both.
    #[test]
    fn dragging_with_two_selected_moves_both_in_one_undo_step() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (a, _) = cube_under_the_viewport_centre(&mut editor);
        let b = editor.spawn("Other");
        editor.world.insert(b, crate::scene::MeshHandle(0));
        let beside = translation(&editor, a) + glam::Vec3::new(0.0, 0.6, 0.0);
        editor.world.insert(
            b,
            LocalTransform::new(crate::scene::Transform::from_translation(beside)),
        );
        crate::scene::propagate_transforms(&mut editor.world);
        let (a0, b0) = (translation(&editor, a), translation(&editor, b));
        editor.state.selection.set(a);
        editor.state.selection.toggle(b);
        editor.state.gizmo_mode = crate::editor::state::GizmoMode::Move;
        editor.frames(2);

        drag_x_shaft_from(&mut editor, (a0 + b0) * 0.5);
        let (da, db) = (translation(&editor, a) - a0, translation(&editor, b) - b0);
        assert!(da.x > 0.1, "{da}");
        assert!((da - db).length() < 1e-4, "{da} vs {db}");

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!((translation(&editor, a) - a0).length() < 1e-5);
        assert!((translation(&editor, b) - b0).length() < 1e-5);
        assert_eq!(editor.state.history.undo_label(), None, "one step");
    }

    /// A child selected with its parent already moves with it, through the
    /// hierarchy. Moving it as well would move it twice.
    #[test]
    fn a_child_selected_with_its_parent_moves_only_through_it() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (parent, _) = cube_under_the_viewport_centre(&mut editor);
        let child = editor.spawn("Child");
        crate::scene::reparent(&mut editor.world, child, Some(parent), false).unwrap();
        crate::scene::propagate_transforms(&mut editor.world);
        let (parent0, child0) = (translation(&editor, parent), translation(&editor, child));
        editor.state.selection.set(child);
        editor.state.selection.toggle(parent);
        editor.state.gizmo_mode = crate::editor::state::GizmoMode::Move;
        editor.frames(2);

        drag_x_shaft_from(&mut editor, parent0);
        assert!(translation(&editor, parent).x > parent0.x + 0.1);
        assert!((translation(&editor, child) - child0).length() < 1e-6);
    }

    /// A light where the viewport's middle looks, and where to click on it.
    fn light_under_the_viewport_centre(editor: &mut Harness) -> (orrin_ecs::Entity, egui::Pos2) {
        editor.frames(2);
        let at = editor.viewport().center();
        let screen = editor.ctx.screen_rect();
        let ndc = (at - screen.min) / screen.size() * 2.0 - egui::Vec2::splat(1.0);
        let camera = *editor.world.resource::<crate::scene::Camera>();
        let (origin, direction) = crate::editor::pick::ray(&camera, screen.aspect_ratio(), ndc);
        let lamp = editor.spawn("Lamp");
        editor.world.insert(lamp, crate::scene::Light::default());
        editor.world.insert(
            lamp,
            LocalTransform::new(crate::scene::Transform::from_translation(
                origin + direction.normalize() * 5.0,
            )),
        );
        crate::scene::propagate_transforms(&mut editor.world);
        (lamp, at)
    }

    #[test]
    fn clicking_a_light_s_marker_selects_it() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (lamp, at) = light_under_the_viewport_centre(&mut editor);
        editor.click(at + egui::vec2(3.0, 0.0));
        assert_eq!(editor.state.selection.entities(), [lamp]);
    }

    /// Markers are how a light is found in the scene at all, so they are
    /// there whether or not anything is selected.
    #[test]
    fn markers_are_drawn_with_nothing_selected() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        light_under_the_viewport_centre(&mut editor);
        editor.state.selection.clear();
        editor.frames(1);
        assert!(!editor.state.gizmo_strokes.is_empty());
    }

    /// What the outline pass draws: every selected mesh and every mesh under a
    /// selected entity, in every tool — Select included, where there are no
    /// handles to say what is selected.
    #[test]
    fn a_selection_and_what_is_under_it_are_outlined() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (cube, _) = cube_under_the_viewport_centre(&mut editor);
        let lid = editor.spawn("Lid");
        editor.world.insert(lid, crate::scene::MeshHandle(0));
        crate::scene::reparent(&mut editor.world, lid, Some(cube), false).unwrap();
        crate::scene::propagate_transforms(&mut editor.world);
        editor.frames(1);
        assert!(editor.state.outline.is_empty(), "nothing selected");

        editor.state.selection.set(cube);
        editor.state.gizmo_mode = crate::editor::state::GizmoMode::Select;
        editor.frames(1);
        let outlined: Vec<bool> = editor
            .state
            .outline
            .iter()
            .map(|item| item.selected)
            .collect();
        assert_eq!(outlined, [true, true], "the cube and its lid");
    }

    /// The hovered mesh is outlined too, as hovered — unless it is selected,
    /// which says more.
    #[test]
    fn the_hovered_mesh_is_outlined_as_hovered() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (cube, at) = cube_under_the_viewport_centre(&mut editor);
        editor.hover(at).frames(1);
        let outlined: Vec<bool> = editor
            .state
            .outline
            .iter()
            .map(|item| item.selected)
            .collect();
        assert_eq!(outlined, [false]);

        editor.state.selection.set(cube);
        editor.hover(at).frames(1);
        let outlined: Vec<bool> = editor
            .state
            .outline
            .iter()
            .map(|item| item.selected)
            .collect();
        assert_eq!(outlined, [true], "once, as selected");
    }

    /// A click on a handle is the start of a drag that went nowhere, not a
    /// click on whatever is behind it — here, empty space, which would clear
    /// the selection.
    #[test]
    fn a_click_on_a_handle_keeps_the_selection() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (cube, _) = cube_under_the_viewport_centre(&mut editor);
        editor.state.selection.set(cube);
        editor.state.gizmo_mode = crate::editor::state::GizmoMode::Move;
        editor.frames(2);

        let tip = x_shaft(&editor, cube, 0.9);
        pointer(&mut editor, egui::Event::PointerMoved(tip));
        editor.click(tip);
        editor.state.apply(&mut editor.world, &editor.registry);
        assert_eq!(editor.state.selection.entities(), [cube]);
        assert_eq!(editor.state.history.undo_label(), None, "nothing moved");
    }

    /// F turns the camera on the selection from where it already looks, and
    /// backs off until the selection fits.
    #[test]
    fn f_frames_the_selection() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (cube, _) = cube_under_the_viewport_centre(&mut editor);
        let away = glam::Vec3::new(20.0, 3.0, -15.0);
        editor.world.insert(
            cube,
            LocalTransform::new(crate::scene::Transform::from_translation(away)),
        );
        crate::scene::propagate_transforms(&mut editor.world);
        let before = *editor.world.resource::<crate::scene::Camera>();
        let heading = (before.target - before.position).normalize();
        editor.state.selection.set(cube);
        editor.frames(1).key(egui::Key::F);
        editor.state.apply(&mut editor.world, &editor.registry);

        let after = *editor.world.resource::<crate::scene::Camera>();
        assert!((after.target - away).length() < 1e-4, "{}", after.target);
        let now = (after.target - after.position).normalize();
        assert!(now.dot(heading) > 0.9999, "the heading is kept");
        // A unit cube's bounding sphere, inside the vertical field of view.
        let radius = glam::Vec3::splat(0.5).length();
        let distance = (after.target - after.position).length();
        assert!(distance * (after.fov_y * 0.5).sin() >= radius, "{distance}");
        assert!(distance < 10.0, "near enough to fill the view: {distance}");
        assert!(editor.state.take_camera_moved());
        assert!(!editor.state.take_camera_moved(), "drained");
    }

    #[test]
    fn f_with_nothing_selected_leaves_the_camera_alone() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let before = *editor.world.resource::<crate::scene::Camera>();
        editor.frames(1).key(egui::Key::F);
        editor.state.apply(&mut editor.world, &editor.registry);
        let after = *editor.world.resource::<crate::scene::Camera>();
        assert_eq!(after.position, before.position);
        assert!(!editor.state.take_camera_moved());
    }

    /// Where an icon was painted inside `within`, if it was.
    fn icon_in(
        editor: &Harness,
        icon: crate::editor::icons::Icon,
        within: egui::Rect,
    ) -> Option<egui::Pos2> {
        let glyph = icon.glyph();
        editor
            .painted_icons()
            .into_iter()
            .find(|(g, rect, _)| *g == glyph && within.contains(rect.center()))
            .map(|(_, rect, _)| rect.center())
    }

    fn row_band(editor: &Harness, name: &str) -> egui::Rect {
        let at = row_of(editor, name);
        let tree = editor.tab(Tab::Hierarchy);
        egui::Rect::from_x_y_ranges(tree.x_range(), (at.y - 8.0)..=(at.y + 8.0))
    }

    /// The lock sits on the row, shown while the pointer is there, and is an
    /// edit like any other: undone with Ctrl+Z.
    #[test]
    fn a_row_s_lock_locks_its_entity_undoably() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let crate_ = editor.spawn("Crate");
        editor.frames(2);
        let band = row_band(&editor, "Crate");
        editor.hover(band.center()).frames(1);
        let lock = icon_in(&editor, crate::editor::icons::lock_open(), band)
            .expect("an open lock on the hovered row");
        editor.click(lock);
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(editor.world.has::<crate::scene::Locked>(crate_));

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(!editor.world.has::<crate::scene::Locked>(crate_));
    }

    /// A locked row says so without being hovered, and its children say they
    /// are locked through it.
    #[test]
    fn a_locked_row_and_its_children_show_the_lock() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let crate_ = editor.spawn("Crate");
        let lid = editor.spawn("Lid");
        crate::scene::reparent(&mut editor.world, lid, Some(crate_), false).unwrap();
        editor.world.insert(crate_, crate::scene::Locked);
        editor.frames(2);
        for name in ["Crate", "Lid"] {
            let band = row_band(&editor, name);
            assert!(
                icon_in(&editor, crate::editor::icons::lock(), band).is_some(),
                "{name} shows no lock"
            );
        }
    }

    #[test]
    fn the_entity_under_the_pointer_is_hovered_unless_locked() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (cube, at) = cube_under_the_viewport_centre(&mut editor);
        editor.hover(at).frames(1);
        assert_eq!(editor.state.hovered, Some(cube));

        let empty = editor.viewport().left_bottom() + egui::vec2(40.0, -40.0);
        editor.hover(empty).frames(1);
        assert_eq!(editor.state.hovered, None);

        editor.world.insert(cube, crate::scene::Locked);
        editor.hover(at).frames(1);
        assert_eq!(editor.state.hovered, None);
    }

    /// A handle is in front of everything: over one, nothing behind it is
    /// hovered.
    #[test]
    fn nothing_is_hovered_through_a_handle() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (cube, _) = cube_under_the_viewport_centre(&mut editor);
        editor.state.selection.set(cube);
        editor.state.gizmo_mode = crate::editor::state::GizmoMode::Move;
        editor.frames(2);
        let middle = x_shaft(&editor, cube, 0.0);
        editor.hover(middle).frames(2);
        assert_eq!(
            editor.state.gizmo_hot,
            Some(crate::editor::gizmo::Handle::Centre)
        );
        assert_eq!(editor.state.hovered, None);
    }

    /// Three cubes: two in the middle of the viewport, one well to the left.
    fn three_cubes(editor: &mut Harness) -> [orrin_ecs::Entity; 3] {
        let (a, _) = cube_under_the_viewport_centre(editor);
        let centre = translation(editor, a);
        let mut beside = |name: &str, offset: glam::Vec3| {
            let entity = editor.spawn(name);
            editor.world.insert(entity, crate::scene::MeshHandle(0));
            editor.world.insert(
                entity,
                LocalTransform::new(crate::scene::Transform::from_translation(centre + offset)),
            );
            entity
        };
        let b = beside("Near", glam::Vec3::new(0.0, 0.6, 0.0));
        let c = beside("Far", glam::Vec3::new(-3.5, 0.0, 0.0));
        crate::scene::propagate_transforms(&mut editor.world);
        editor.frames(1);
        [a, b, c]
    }

    /// Press at `from`, drag to `to` in steps, let go — with `modifiers` held.
    fn drag_rectangle(
        editor: &mut Harness,
        from: egui::Pos2,
        to: egui::Pos2,
        modifiers: egui::Modifiers,
    ) {
        editor.input.modifiers = modifiers;
        let press = |pos, pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers,
        };
        pointer(editor, egui::Event::PointerMoved(from));
        pointer(editor, press(from, true));
        for step in 1..=4 {
            pointer(
                editor,
                egui::Event::PointerMoved(from + (to - from) * (step as f32 / 4.0)),
            );
        }
        pointer(editor, press(to, false));
        editor.input.modifiers = egui::Modifiers::NONE;
    }

    #[test]
    fn dragging_on_the_scene_selects_what_the_rectangle_holds() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let [a, b, c] = three_cubes(&mut editor);
        editor.state.selection.set(c);
        let middle = editor.viewport().center();
        let corner = egui::vec2(150.0, 150.0);
        drag_rectangle(
            &mut editor,
            middle - corner,
            middle + corner,
            egui::Modifiers::NONE,
        );

        let mut selected = editor.state.selection.entities().to_vec();
        selected.sort_by_key(|e| e.index());
        assert_eq!(selected, [a, b]);
    }

    #[test]
    fn a_ctrl_drag_adds_and_a_lock_keeps_out() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let [a, b, c] = three_cubes(&mut editor);
        editor.world.insert(b, crate::scene::Locked);
        editor.state.selection.set(c);
        let middle = editor.viewport().center();
        let corner = egui::vec2(150.0, 150.0);
        drag_rectangle(
            &mut editor,
            middle - corner,
            middle + corner,
            egui::Modifiers::COMMAND,
        );

        let mut selected = editor.state.selection.entities().to_vec();
        selected.sort_by_key(|e| e.index());
        assert_eq!(selected, [a, c]);
    }

    /// The rectangle is drawn while it is being dragged out, and gone after.
    #[test]
    fn the_rectangle_shows_while_it_is_drawn() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        three_cubes(&mut editor);
        let middle = editor.viewport().center();
        let from = middle - egui::vec2(150.0, 150.0);
        pointer(&mut editor, egui::Event::PointerMoved(from));
        pointer(&mut editor, button(from, true));
        pointer(
            &mut editor,
            egui::Event::PointerMoved(middle + egui::vec2(150.0, 150.0)),
        );
        let drawn = |editor: &Harness| {
            editor.painted_fills().iter().any(|(rect, _)| {
                (rect.width() - 300.0).abs() < 2.0 && (rect.height() - 300.0).abs() < 2.0
            })
        };
        assert!(drawn(&editor), "no rectangle while dragging");
        pointer(
            &mut editor,
            button(middle + egui::vec2(150.0, 150.0), false),
        );
        assert!(!drawn(&editor), "the rectangle outlived the drag");
    }

    /// Selected, a light shows how far it reaches; unselected it is only its
    /// marker, so a scene of lights is not a scene of spheres.
    #[test]
    fn a_selected_light_shows_its_range() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (lamp, _) = light_under_the_viewport_centre(&mut editor);
        editor.world.insert(
            lamp,
            crate::scene::Light::point(glam::Vec3::ONE, 800.0, 2.0),
        );
        editor.frames(1);
        let resting = editor.state.gizmo_strokes.len();
        editor.state.selection.set(lamp);
        editor.state.gizmo_mode = crate::editor::state::GizmoMode::Select;
        editor.frames(1);
        assert!(
            editor.state.gizmo_strokes.len() > resting + 6 * 3 * 32,
            "{} after, {resting} before",
            editor.state.gizmo_strokes.len()
        );
    }

    #[test]
    fn a_locked_mesh_cannot_be_clicked_in_the_scene() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (cube, at) = cube_under_the_viewport_centre(&mut editor);
        editor.world.insert(cube, crate::scene::Locked);
        editor.click(at);
        assert!(editor.state.selection.is_empty());
    }

    /// Picking the tool is a per-second interaction, so it has keys — numbers,
    /// because the fly camera owns the letters.
    #[test]
    fn the_number_keys_pick_the_tool_unless_a_field_has_focus() {
        use crate::editor::state::GizmoMode;
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(1);
        for (key, mode) in [
            (egui::Key::Num2, GizmoMode::Move),
            (egui::Key::Num3, GizmoMode::Rotate),
            (egui::Key::Num4, GizmoMode::Scale),
            (egui::Key::Num1, GizmoMode::Select),
        ] {
            editor.key(key);
            assert!(editor.state.gizmo_mode == mode, "{key:?}");
        }

        editor
            .ctx
            .memory_mut(|memory| memory.request_focus(egui::Id::new("some field")));
        editor.key(egui::Key::Num3);
        assert!(editor.state.gizmo_mode == GizmoMode::Select);
    }

    /// Each tool snaps by its own increment, and holding Ctrl snaps a drag
    /// whether or not the toggle is on.
    #[test]
    fn the_snap_step_follows_the_tool_and_ctrl_snaps_without_the_toggle() {
        use crate::editor::state::{GizmoMode, MOVE_SNAP, ROTATE_SNAP, SCALE_SNAP};
        let mut state = EditorState::default();
        state.gizmo_mode = GizmoMode::Move;
        assert_eq!(state.snap_step(false), None);
        assert_eq!(state.snap_step(true), Some(MOVE_SNAP));
        state.snap = true;
        assert_eq!(state.snap_step(false), Some(MOVE_SNAP));
        state.gizmo_mode = GizmoMode::Rotate;
        assert_eq!(state.snap_step(false), Some(ROTATE_SNAP));
        state.gizmo_mode = GizmoMode::Scale;
        assert_eq!(state.snap_step(false), Some(SCALE_SNAP));
        state.gizmo_mode = GizmoMode::Select;
        assert_eq!(state.snap_step(true), None);
    }

    #[test]
    fn the_snap_steps_are_the_user_s_to_set() {
        use crate::editor::state::GizmoMode;
        let mut state = EditorState::default();
        state.snap = true;
        state.snap_steps.translate = 0.5;
        state.snap_steps.rotate = 5.0;
        state.gizmo_mode = GizmoMode::Move;
        assert_eq!(state.snap_step(false), Some(0.5));
        state.gizmo_mode = GizmoMode::Rotate;
        assert_eq!(state.snap_step(false), Some(5.0));
    }

    /// The chevron beside the snap button opens the steps for every tool.
    #[test]
    fn the_snap_steps_are_set_from_the_bar() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        let tools = editor.area("viewport_tools").expect("toolbar drawn");
        let chevron = crate::editor::icons::chevron_down().glyph();
        let at = editor
            .painted_icons()
            .into_iter()
            .find(|(glyph, rect, _)| *glyph == chevron && tools.contains(rect.center()))
            .expect("a chevron on the toolbar")
            .1
            .center();
        assert!(!editor.painted_contains("Rotate"));
        editor.click(at);
        editor.frames(1);
        for row in ["Move", "Rotate", "Scale"] {
            assert!(editor.painted_contains(row), "no {row} row");
        }
    }

    /// The snap button says the increment the current tool would use.
    #[test]
    fn the_snap_button_shows_the_current_tools_increment() {
        use crate::editor::state::GizmoMode;
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.state.snap = true;
        editor.state.gizmo_mode = GizmoMode::Rotate;
        editor.frames(2);
        assert!(editor.painted_contains("15°"));
        editor.state.gizmo_mode = GizmoMode::Move;
        editor.frames(1);
        assert!(editor.painted_contains("0.25"));
    }

    /// A section bar is a band across the panel, square at its ends: rounded,
    /// edge to edge, it read as a button laid across the inspector.
    #[test]
    fn inspector_section_bars_are_square() {
        let mut editor = Harness::new(egui::vec2(1440.0, 900.0));
        let entity = editor.spawn("Crate");
        editor.state.selection.set(entity);
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

    /// The edge of anything floating over the scene is a faint light line,
    /// not a drawn border competing with what it frames.
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

    /// Nothing is blurred beneath a floating surface, so each one has to hide
    /// what it floats over: the viewport's toolbars, menus and popups are
    /// opaque, over the scene as over a panel.
    #[test]
    fn floating_surfaces_are_opaque() {
        let ctx = egui::Context::default();
        theme::apply(&ctx, themes_default());
        let visuals = ctx.style().visuals.clone();
        let toolbar = crate::editor::widgets::toolbar_frame(&visuals, themes_default().bar());
        assert_eq!(toolbar.fill.a(), 255, "toolbar {:?}", toolbar.fill);
        assert_eq!(
            visuals.window_fill.a(),
            255,
            "menu {:?}",
            visuals.window_fill
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
        editor.state.selection.set(entity);

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
                        ui.allocate_ui(cell, |ui| environment::body(ui, world, state));
                        ui.allocate_ui(cell, |ui| performance::body(ui, world));
                        ui.allocate_ui(cell, |ui| console::body(ui, world, state));
                        ui.allocate_ui(cell, |ui| assets::body(ui, world, state, previews));
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
        editor.state.selection.set(entity);
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

    /// An editor drawing `brick.omat` on a selected crate, beside a texture and
    /// a mesh, with the material bound as a frame's service would bind it.
    fn material_editor(
        name: &str,
    ) -> (
        Harness,
        crate::asset_db::tests::Dir,
        orrin_ecs::Entity,
        orrin_registry::AssetId,
    ) {
        use crate::asset_db::{AssetDatabase, material};
        let mut editor = Harness::new(egui::vec2(1280.0, 1400.0));
        let dir = crate::asset_db::tests::Dir::new(name);
        dir.write(
            "brick.omat",
            material::write(&Default::default()).as_bytes(),
        );
        dir.write("rock.png", b"");
        dir.write("tree.glb", b"");
        let (db, _) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let brick = db.find("brick.omat").unwrap().id;
        let world = &mut editor.world;
        world.insert_resource(db);
        world.insert_resource(crate::scene::Assets::new());
        world.insert_resource(crate::scene::AssetPreviews::default());
        world.insert_resource(crate::scene::MaterialBlends::default());
        world.insert_resource(crate::scene::MeshBounds::default());
        let handle = world
            .resource_mut::<AssetDatabase>()
            .material(brick)
            .unwrap();
        let entity = editor.spawn("Crate");
        editor.world.insert(entity, crate::scene::MeshHandle(0));
        editor.world.insert(entity, handle);
        let errors = editor
            .world
            .resource_mut::<AssetDatabase>()
            .update(&editor.world, &mut crate::gfx::HeadlessBackend::new());
        assert!(errors.is_empty(), "{errors:?}");
        editor.state.selection.set(entity);
        editor.frames(2);
        (editor, dir, entity, brick)
    }

    /// Where the value of the inspector row labelled `label` is drawn.
    fn value_of(editor: &Harness, label: &str) -> egui::Pos2 {
        let row = row_of(editor, label);
        egui::pos2(editor.tab_body(Tab::Inspector).right() - 40.0, row.y)
    }

    /// The material section of a material asset edits the file, through the
    /// same undo history as the scene: a texture dropped on a slot is saved at
    /// once and undone in one step, and a mesh dropped there is refused.
    #[test]
    fn a_texture_dropped_on_a_material_assets_slot_is_saved_and_undoable() {
        use crate::asset_db::{AssetDatabase, material};
        let (mut editor, dir, _, brick) = material_editor("panel-omat-slot");
        let (rock, tree) = {
            let db = editor.world.resource::<AssetDatabase>();
            (
                db.find("rock.png").unwrap().id,
                db.find("tree.glb").unwrap().id,
            )
        };
        let saved = || {
            let text = std::fs::read_to_string(dir.path().join("brick.omat")).unwrap();
            material::parse(&text, "brick.omat").unwrap().textures
        };

        let slot = value_of(&editor, "Albedo map");
        drop_payload(&mut editor, rock, slot);
        assert_eq!(saved().albedo, Some(rock));
        assert_eq!(editor.state.history.undo_label(), Some("Edit brick"));

        // Away from the slot just dropped on, whose tooltip would otherwise
        // cover the next one down; a real drag comes from the browser.
        editor.hover(egui::pos2(10.0, 400.0)).frames(1);
        let slot = value_of(&editor, "Normal map");
        drop_payload(&mut editor, tree, slot);
        assert_eq!(saved().normal, None);
        let logged: Vec<String> = editor
            .world
            .resource::<crate::scene::LogBuffer>()
            .iter()
            .map(|entry| entry.message.clone())
            .collect();
        assert_eq!(
            logged,
            ["brick.omat: the `normal` slot takes a texture, not `tree.glb`, which is a mesh"]
        );

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        assert_eq!(saved().albedo, None);
        let db = editor.world.resource::<AssetDatabase>();
        assert_eq!(db.material_file(brick).unwrap().textures.albedo, None);
    }

    /// The Material slot of a mesh renderer takes a material asset.
    #[test]
    fn a_material_asset_dropped_on_the_mesh_renderer_draws_with_it() {
        use crate::asset_db::{AssetDatabase, material};
        let (mut editor, dir, entity, _) = material_editor("panel-omat-renderer");
        dir.write(
            "glass.omat",
            material::write(&Default::default()).as_bytes(),
        );
        editor.world.resource_mut::<AssetDatabase>().rescan();
        let glass = editor
            .world
            .resource::<AssetDatabase>()
            .find("glass.omat")
            .unwrap()
            .id;
        editor.frames(1);

        let slot = value_of(&editor, "Material");
        drop_payload(&mut editor, glass, slot);
        let handle = *editor
            .world
            .get::<crate::scene::MaterialHandle>(entity)
            .unwrap();
        assert_eq!(
            editor.world.resource::<AssetDatabase>().material_id(handle),
            Some(glass)
        );
        assert_eq!(editor.state.history.undo_label(), Some("Edit Material"));
    }

    /// Right-clicking the empty part of the open folder, or a folder's row,
    /// offers "New material": a file at the default parameters, indexed as a
    /// material at once and marked in the list.
    #[test]
    fn new_material_in_the_browser_writes_one_and_marks_it() {
        use crate::asset_db::{AssetDatabase, AssetKind};
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let dir = crate::asset_db::tests::Dir::new("panel-new-material");
        dir.write("props/door.oprefab", b"orrin-scene 1\n");
        let (db, _) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        editor.world.insert_resource(db);
        editor.dock.activate(Tab::Assets);
        editor.frames(2);
        let props = row_of(&editor, "props");

        let below = egui::pos2(props.x, editor.tab_body(Tab::Assets).bottom() - 30.0);
        secondary_click(&mut editor, below);
        let item = row_of(&editor, "New material");
        editor.click(item);
        let db = editor.world.resource::<AssetDatabase>();
        let created = db.find("New Material.omat").expect("written and indexed");
        assert_eq!(created.kind, AssetKind::Material);
        assert_eq!(editor.state.revealed_asset, Some(created.id));
        drop(db);
        editor.frames(1);
        assert!(editor.painted_contains("New Material.omat"));

        secondary_click(&mut editor, props);
        let item = row_of(&editor, "New material");
        editor.click(item);
        let db = editor.world.resource::<AssetDatabase>();
        let inside = db
            .find("props/New Material.omat")
            .expect("made in the folder");
        assert_eq!(editor.state.revealed_asset, Some(inside.id));
    }

    /// What the editor's own layout pass costs the CPU each frame — the part of
    /// the frame no `tests/perf.rs` row sees, since that harness has no editor.
    /// Tessellation (`Editor::overlay`) and the GPU draw are not measured. It is
    /// paid only on the frames `editor::cadence` says are due.
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
        editor.state.selection.set(parent);
        editor.frames(20);

        let start = std::time::Instant::now();
        editor.frames(FRAMES);
        let per_frame = start.elapsed() / FRAMES as u32;
        println!("editor layout: {per_frame:?} per frame over {FRAMES} frames, 2 000 entities");
    }

    fn play_state(editor: &Harness) -> crate::scene::PlayState {
        editor.world.resource::<crate::scene::Play>().state()
    }

    /// Ctrl+P, then whatever `apply` does with it, as the editor's frame does.
    fn toggle_play(editor: &mut Harness) {
        editor.key_with(egui::Modifiers::COMMAND, egui::Key::P);
        editor.state.apply(&mut editor.world, &editor.registry);
    }

    /// Play forks the scene and Stop throws the fork away: whatever the game
    /// did to the world — moved, despawned, spawned — the scene after Stop is
    /// the one a save before Play would have written.
    #[test]
    fn stop_puts_the_scene_back_as_it_was_before_play() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let crate_ = editor.spawn("Crate");
        let lamp = editor.spawn("Lamp");
        let before = crate::scene::save(&mut editor.world, &editor.registry);

        toggle_play(&mut editor);
        assert_eq!(play_state(&editor), crate::scene::PlayState::Playing);
        // What a script does: straight at the world, around the pipeline.
        editor
            .world
            .get_mut::<LocalTransform>(crate_)
            .unwrap()
            .0
            .translation
            .x = 9.0;
        editor.world.despawn(lamp);
        editor.spawn("Bullet");

        toggle_play(&mut editor);
        assert_eq!(play_state(&editor), crate::scene::PlayState::Editing);
        assert_eq!(
            crate::scene::save(&mut editor.world, &editor.registry),
            before
        );
        assert!(named(&editor.world, "Bullet").is_none());
    }

    /// The history is the scene's, and the fork is not the scene: an edit made
    /// while playing is gone with the fork, and undo after Stop undoes what it
    /// undid before Play.
    #[test]
    fn edits_while_playing_leave_the_history_alone() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.spawn("Keep");
        delete_by_key(&mut editor, "Old crate");
        let label = editor.state.history.undo_label().map(str::to_owned);
        assert!(label.is_some());

        toggle_play(&mut editor);
        let keep = named(&editor.world, "Keep").unwrap();
        editor.state.selection.set(keep);
        editor.frames(1).key(egui::Key::Delete);
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(named(&editor.world, "Keep").is_none());
        toggle_play(&mut editor);

        assert!(named(&editor.world, "Keep").is_some());
        assert_eq!(editor.state.history.undo_label().map(str::to_owned), label);
        assert!(
            editor.state.history.is_dirty(),
            "the delete is still unsaved"
        );
    }

    fn area_shown(editor: &Harness, id: &str) -> bool {
        let layer = egui::LayerId::new(egui::Order::Middle, egui::Id::new(id));
        editor
            .ctx
            .memory(|memory| memory.areas().visible_last_frame(&layer))
    }

    /// While the game is live the scene is the game's: nothing is selected, no
    /// handle, marker, outline or frustum is drawn over it, and the transform
    /// tools are gone — the panels around it stay, and so do the frame stats.
    #[test]
    fn play_clears_the_selection_and_the_scene_tools() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (cube, _) = cube_under_the_viewport_centre(&mut editor);
        light_under_the_viewport_centre(&mut editor);
        editor.state.selection.set(cube);
        editor.frames(1);
        assert!(!editor.state.gizmo_strokes.is_empty());
        assert!(!editor.state.outline.is_empty());
        assert!(area_shown(&editor, "viewport_tools"));

        toggle_play(&mut editor);
        editor.frames(2);
        assert!(editor.state.selection.entities().is_empty());
        assert!(
            editor.state.gizmo_strokes.is_empty(),
            "no handles or markers"
        );
        assert!(editor.state.outline.is_empty());
        assert!(!area_shown(&editor, "viewport_tools"));
        assert!(
            area_shown(&editor, "viewport_view"),
            "the stats toggle stays"
        );

        editor
            .world
            .resource_mut::<crate::scene::Play>()
            .set(crate::scene::PlayState::Paused);
        editor.frames(1);
        assert!(
            editor.state.gizmo_strokes.is_empty(),
            "a paused game is still the game, markers and all"
        );
        assert!(!area_shown(&editor, "viewport_tools"));
    }

    /// A click in a running game is the game's — a HUD button over a crate
    /// must not select the crate.
    #[test]
    fn a_click_on_the_scene_while_playing_selects_nothing() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (_, at) = cube_under_the_viewport_centre(&mut editor);
        toggle_play(&mut editor);
        editor.click(at);
        assert!(editor.state.selection.entities().is_empty());
        assert!(editor.state.hovered.is_none());
    }

    /// The selection Play cleared is the editor's, so Stop gives it back.
    #[test]
    fn stop_restores_the_selection_play_cleared() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let crate_ = editor.spawn("Crate");
        editor.state.selection.set(crate_);
        toggle_play(&mut editor);
        assert!(editor.state.selection.entities().is_empty());
        toggle_play(&mut editor);
        let restored = named(&editor.world, "Crate").unwrap();
        assert_eq!(editor.state.selection.entities(), [restored]);
    }

    /// A persistent scene holding one entity called `Music`, written to a file
    /// of its own.
    fn persistent_scene(editor: &Harness, tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("orrin-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("persistent.orrin");
        let mut source = World::new();
        let music = source.spawn();
        source.insert(music, Name::new("Music"));
        std::fs::write(&path, crate::scene::save(&mut source, &editor.registry)).unwrap();
        path
    }

    /// Play is the game starting, so the project's persistent scene loads
    /// beside the scene being played, and goes with the fork on Stop.
    #[test]
    fn play_loads_the_persistent_scene_beside_the_scene() {
        use crate::scene::{InScene, LoadedScenes};

        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.spawn("Hub");
        editor.state.persistent_scene = Some(persistent_scene(&editor, "play-persistent"));

        toggle_play(&mut editor);
        let music = named(&editor.world, "Music").expect("the persistent scene loaded");
        assert!(editor.world.get::<InScene>(music).is_some());
        assert!(named(&editor.world, "Hub").is_some());
        let persistent: Vec<String> = editor
            .world
            .resource::<LoadedScenes>()
            .iter()
            .filter(|scene| scene.persistent)
            .map(|scene| scene.name.clone())
            .collect();
        assert_eq!(persistent, ["persistent"]);

        toggle_play(&mut editor);
        assert!(named(&editor.world, "Music").is_none());
        assert_eq!(editor.world.resource::<LoadedScenes>().iter().count(), 1);
    }

    /// Playing the persistent scene itself must not load a second copy of it
    /// beside the first.
    #[test]
    fn playing_the_persistent_scene_loads_it_once() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let path = persistent_scene(&editor, "play-persistent-itself");
        editor.state.request_open_scene(path.to_string_lossy());
        editor.state.apply(&mut editor.world, &editor.registry);
        editor.state.persistent_scene = Some(path);

        toggle_play(&mut editor);
        let copies = editor
            .world
            .entities()
            .filter(|&e| editor.world.get::<Name>(e).is_some_and(|n| n.0 == "Music"))
            .count();
        assert_eq!(copies, 1);
    }

    fn with_database(editor: &mut Harness, dir: &crate::asset_db::tests::Dir) {
        let (db, _) = crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        editor.world.insert_resource(db);
    }

    fn logged(editor: &Harness) -> Vec<String> {
        editor
            .world
            .resource::<crate::scene::LogBuffer>()
            .iter()
            .map(|entry| entry.message.clone())
            .collect()
    }

    /// Where the text `needle` was painted, if it was.
    fn text_at(editor: &Harness, needle: &str) -> Option<egui::Pos2> {
        editor
            .painted_text_rects()
            .into_iter()
            .find(|(text, _)| text == needle)
            .map(|(_, rect)| rect.center())
    }

    /// The Scene Manager can leave the dock for a window of its own, and go
    /// back to where it came from, from a button on the tool itself.
    #[test]
    fn the_scene_manager_pops_out_into_a_window_and_docks_back() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.dock.activate(Tab::Scenes);
        editor.frames(2);
        let docked = editor.tab(Tab::Scenes);
        assert_eq!(docked, editor.tab(Tab::Hierarchy));

        let button = icon_in(
            &editor,
            crate::editor::icons::arrow_square_out(),
            editor.tab_body(Tab::Scenes),
        )
        .expect("pop-out button");
        editor.click(button).frames(2);
        assert!(editor.dock.is_floating(Tab::Scenes));
        let window = editor.tab(Tab::Scenes);
        assert_ne!(window, docked);
        assert!(
            editor.dock.is_open(Tab::Hierarchy),
            "the dock keeps its other tools"
        );
        assert!(
            !editor.dock.over_scene(&editor.ctx, window.center()),
            "a click on the window must not reach the scene behind it"
        );

        let button = icon_in(&editor, crate::editor::icons::arrow_square_in(), window)
            .expect("dock-back button");
        editor.click(button).frames(2);
        assert!(!editor.dock.is_floating(Tab::Scenes));
        assert_eq!(editor.tab(Tab::Scenes), editor.tab(Tab::Hierarchy));
    }

    /// Tab icons are painted by the dock over the strip egui_dock drew. A
    /// window's strip is above the panels, so its icon must be painted there
    /// too, or the window covers it.
    #[test]
    fn a_floating_tools_tab_icon_is_drawn_over_its_window() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor
            .dock
            .apply(crate::editor::state::DockRequest::Float(Tab::Scenes));
        editor.frames(2);
        let Harness {
            world,
            registry,
            state,
            themes,
            dock,
            previews,
            ctx,
            input,
            ..
        } = &mut editor;
        // A marker painted last on the panels' layer: everything before it in
        // the output is beneath every window.
        struct PanelsEnd;
        let output = ctx.run(input.clone(), |ctx| {
            draw(ctx, world, state, registry, themes, dock, previews);
            ctx.layer_painter(egui::LayerId::background())
                .add(egui::Shape::Callback(egui::PaintCallback {
                    rect: egui::Rect::NOTHING,
                    callback: std::sync::Arc::new(PanelsEnd),
                }));
        });
        let window = {
            let (surface, node, _) = dock.state().find_tab(&Tab::Scenes).unwrap();
            dock.state()[surface][node].rect().unwrap()
        };
        let end = output
            .shapes
            .iter()
            .position(|clipped| match &clipped.shape {
                egui::Shape::Callback(callback) => callback.callback.is::<PanelsEnd>(),
                _ => false,
            })
            .expect("the marker was painted");
        let panels = &output.shapes[..end];
        let under = panels.iter().any(|clipped| match &clipped.shape {
            egui::Shape::Text(text) => {
                text.galley.text() == Tab::Scenes.icon().glyph().to_string()
                    && window.contains(text.visual_bounding_rect().center())
            }
            _ => false,
        });
        assert!(!under, "the Scenes tab icon was painted beneath its window");
    }

    /// Where `tab`'s title was painted in its strip.
    fn tab_title(editor: &Harness, tab: Tab) -> egui::Pos2 {
        let leaf = editor.tab(tab);
        let strip = egui::Rect::from_min_size(leaf.min, egui::vec2(leaf.width(), TAB_BAR_HEIGHT));
        editor
            .painted_text_rects()
            .into_iter()
            .find(|(text, rect)| text == tab.title() && strip.contains(rect.center()))
            .map(|(_, rect)| rect.center())
            .unwrap_or_else(|| panic!("no {} tab in its strip", tab.title()))
    }

    /// A press at `from`, a drag to `to` in steps, a pause there and a
    /// release, one frame each, as a hand delivers it. The pause matters:
    /// egui_dock holds on to the target a fast drag passed over for a moment.
    fn drag_tab(editor: &mut Harness, from: egui::Pos2, to: egui::Pos2) {
        pointer(editor, egui::Event::PointerMoved(from));
        pointer(editor, button(from, true));
        for step in 1..=6 {
            let at = from + (to - from) * (step as f32 / 6.0);
            pointer(editor, egui::Event::PointerMoved(at));
        }
        editor.frames(30);
        pointer(editor, button(to, false));
        editor.frames(2);
    }

    /// A tab dragged out over the scene leaves the dock for a window of its
    /// own where it was let go, and the tools it shared a strip with stay.
    #[test]
    fn a_tab_dragged_onto_the_scene_floats_where_it_is_dropped() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        let to = editor.viewport().center();
        let from = tab_title(&editor, Tab::Console);
        drag_tab(&mut editor, from, to);

        assert!(editor.dock.is_floating(Tab::Console));
        let window = editor.tab(Tab::Console);
        assert!(
            under_the_strip(window, to),
            "the window opened at {window:?}, not under the pointer at {to:?}"
        );
        assert!(editor.dock.is_open(Tab::Assets));
        assert!(!editor.dock.is_floating(Tab::Assets));
    }

    /// Dropped on another tool's strip, a tab joins it: the drop onto the
    /// scene must not take drops that were egui_dock's to place.
    #[test]
    fn a_tab_dragged_onto_another_strip_docks_there() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        let strip = editor.tab(Tab::Inspector);
        let to = egui::pos2(strip.right() - 30.0, strip.top() + TAB_BAR_HEIGHT / 2.0);
        let from = tab_title(&editor, Tab::Console);
        drag_tab(&mut editor, from, to);

        assert!(!editor.dock.is_floating(Tab::Console));
        assert_eq!(editor.tab(Tab::Console), editor.tab(Tab::Inspector));
    }

    /// A click on a tab only brings it forward.
    #[test]
    fn a_click_on_a_tab_does_not_float_it() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        let at = tab_title(&editor, Tab::Assets);
        drag_tab(&mut editor, at, at + egui::vec2(1.0, 0.0));
        assert!(!editor.dock.is_floating(Tab::Assets));
    }

    /// A window's tab is also its handle: dragged over the scene, the window
    /// follows rather than staying where it was.
    #[test]
    fn a_floating_tool_dragged_by_its_tab_moves_its_window() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor
            .dock
            .apply(crate::editor::state::DockRequest::Float(Tab::Scenes));
        editor.frames(2);
        let before = editor.tab(Tab::Scenes);
        // High enough that the window still fits below it: egui keeps a
        // window on screen, so a lower drop lands it higher than the pointer.
        let to = egui::pos2(before.right() + 200.0, 150.0);
        assert!(editor.viewport().contains(to));
        let from = tab_title(&editor, Tab::Scenes);
        drag_tab(&mut editor, from, to);

        assert!(editor.dock.is_floating(Tab::Scenes));
        let after = editor.tab(Tab::Scenes);
        assert!(
            under_the_strip(after, to),
            "the window went from {before:?} to {after:?}, not under {to:?}"
        );
        assert_eq!(after.size(), before.size());
    }

    /// Whether `at` is on the tab strip of a window at `window`: where the
    /// pointer that carried the window there would be.
    fn under_the_strip(window: egui::Rect, at: egui::Pos2) -> bool {
        window.x_range().expand(1.0).contains(at.x)
            && (window.top() - 1.0..=window.top() + TAB_BAR_HEIGHT + 1.0).contains(&at.y)
    }

    /// Opening another scene over unsaved work asks first, and offers to save
    /// it, rather than throwing the work away on a click.
    #[test]
    fn opening_a_scene_over_unsaved_changes_asks_first() {
        let dir = crate::asset_db::tests::Dir::new("scene-manager-guard");
        let text = {
            let mut source = World::new();
            let gate = source.spawn();
            source.insert(gate, Name::new("Gate"));
            crate::scene::save(&mut source, &Harness::new(egui::vec2(10.0, 10.0)).registry)
        };
        dir.write("levels/castle.orrin", text.as_bytes());
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);
        editor.state.scene_path = dir.path().join("hub.orrin").to_string_lossy().into_owned();
        delete_by_key(&mut editor, "Old crate");
        editor.spawn("Keep");
        assert!(editor.state.scene_dirty());
        editor.dock.activate(Tab::Scenes);
        editor.frames(2);

        let row = text_at(&editor, "castle").expect("castle is listed");
        editor.click(row).frames(1);
        let open = text_at(&editor, "Open").expect("an Open button on the chosen row");
        editor.click(open);
        editor.state.apply(&mut editor.world, &editor.registry);
        editor.frames(1);
        assert!(
            named(&editor.world, "Keep").is_some(),
            "nothing was opened yet"
        );
        assert!(editor.painted_contains("unsaved changes"));

        let save = text_at(&editor, "Save and open").expect("a save-first choice");
        editor.click(save);
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(
            named(&editor.world, "Gate").is_some(),
            "{:?}",
            logged(&editor)
        );
        let saved = std::fs::read_to_string(dir.path().join("hub.orrin")).unwrap();
        assert!(
            saved.contains("Keep"),
            "the work was saved before the switch"
        );
        assert!(!editor.state.scene_dirty());
    }

    /// Without unsaved work there is nothing to ask about.
    #[test]
    fn opening_a_saved_scene_does_not_ask() {
        let dir = crate::asset_db::tests::Dir::new("scene-manager-clean-open");
        dir.write("levels/castle.orrin", b"orrin-scene 3 min-reader 3\n");
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);
        editor.dock.activate(Tab::Scenes);
        editor.frames(2);
        let row = text_at(&editor, "castle").unwrap();
        editor.click(row).frames(1);
        editor.click(text_at(&editor, "Open").unwrap());
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(editor.state.scene_path.ends_with("castle.orrin"));
    }

    /// A long list narrows as a name is typed.
    #[test]
    fn the_project_scenes_filter_by_name() {
        let dir = crate::asset_db::tests::Dir::new("scene-manager-filter");
        dir.write("levels/castle.orrin", b"");
        dir.write("levels/dungeon.orrin", b"");
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);
        editor.state.scenes_query = "cas".to_owned();
        editor.dock.activate(Tab::Scenes);
        editor.frames(2);
        assert!(editor.painted_contains("castle"));
        assert!(!editor.painted_contains("dungeon"));
    }

    /// A new scene's name is checked before Create is pressed, by the same
    /// rule the request is checked by.
    #[test]
    fn a_new_scene_name_is_checked_before_it_is_used() {
        use crate::editor::state::new_scene_path;

        let dir = crate::asset_db::tests::Dir::new("scene-manager-check");
        dir.write("scenes/castle.orrin", b"");
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);

        assert!(new_scene_path(&editor.world, "  ").is_err());
        let slashed = new_scene_path(&editor.world, "a/b").unwrap_err();
        assert!(slashed.contains("slashes"), "{slashed}");
        let taken = new_scene_path(&editor.world, "castle").unwrap_err();
        assert!(taken.contains("already"), "{taken}");
        let fresh = new_scene_path(&editor.world, "keep").unwrap();
        assert!(fresh.ends_with("scenes/keep.orrin"));
    }

    /// The header names the file the way the project does — relative, with
    /// forward slashes — not by the extended-length absolute path Windows
    /// hands back for a canonicalised directory.
    #[test]
    fn the_scene_manager_shows_the_scene_file_short() {
        let dir = crate::asset_db::tests::Dir::new("scene-manager-short-path");
        dir.write("scenes/new_scene.orrin", b"orrin-scene 3 min-reader 3\n");
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);
        let full = std::fs::canonicalize(dir.path().join("scenes/new_scene.orrin")).unwrap();
        editor.state.scene_path = full.to_string_lossy().into_owned();
        editor.dock.activate(Tab::Scenes);
        editor.frames(2);

        let body = editor.tab_body(Tab::Scenes);
        let shown: Vec<String> = editor
            .painted_text_rects()
            .into_iter()
            .filter(|(_, rect)| body.contains(rect.center()))
            .map(|(text, _)| text)
            .collect();
        assert!(
            shown.iter().any(|text| text == "scenes/new_scene.orrin"),
            "{shown:?}"
        );
        assert!(
            !shown
                .iter()
                .any(|text| text.contains(r"\\?\") || text.contains("Documents")),
            "{shown:?}"
        );
    }

    fn painted_rect_of(editor: &Harness, needle: &str) -> Option<egui::Rect> {
        editor
            .painted_text_rects()
            .into_iter()
            .find(|(text, _)| text == needle)
            .map(|(_, rect)| rect)
    }

    /// The filter has the row to itself bar a plus, and the new scene's name
    /// is asked for below it only once the plus is pressed: two empty wells
    /// side by side read as one field cut in half.
    #[test]
    fn a_new_scene_is_named_below_the_filter_once_asked_for() {
        let dir = crate::asset_db::tests::Dir::new("scene-manager-row");
        dir.write("levels/castle.orrin", b"");
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);
        editor.dock.activate(Tab::Scenes);
        editor.frames(3);

        let body = editor.tab_body(Tab::Scenes);
        let filter = painted_rect_of(&editor, "Filter scenes").expect("a filter");
        assert!(painted_rect_of(&editor, "Scene name").is_none());
        let row = egui::Rect::from_x_y_ranges(
            body.x_range(),
            (filter.center().y - 8.0)..=(filter.center().y + 8.0),
        );
        let plus = icon_in(&editor, crate::editor::icons::plus(), row).expect("a plus");
        assert!(
            plus.x > body.right() - 40.0,
            "the plus ends the filter's row"
        );

        editor.click(plus).frames(1);
        let name = painted_rect_of(&editor, "Scene name").expect("a name field");
        assert!(name.top() > filter.bottom(), "the name is asked for below");
        let create = painted_rect_of(&editor, "Create").expect("a create button");
        assert!(body.contains(create.center()));
    }

    /// Named in place, a scene is made with Enter, the way a rename is.
    #[test]
    fn a_new_scene_is_made_from_its_field_with_enter() {
        let dir = crate::asset_db::tests::Dir::new("scene-manager-enter");
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);
        editor.dock.activate(Tab::Scenes);
        editor.frames(3);
        let filter = painted_rect_of(&editor, "Filter scenes").expect("a filter");
        let row = egui::Rect::from_x_y_ranges(
            editor.tab_body(Tab::Scenes).x_range(),
            (filter.center().y - 8.0)..=(filter.center().y + 8.0),
        );
        let plus = icon_in(&editor, crate::editor::icons::plus(), row).expect("a plus");
        editor.click(plus).frames(1);

        editor
            .input
            .events
            .push(egui::Event::Text("Level 3".to_owned()));
        editor.frames(1);
        editor.input.events.clear();
        editor.key(egui::Key::Enter);
        editor.state.apply(&mut editor.world, &editor.registry);

        assert!(
            dir.path().join("scenes/Level 3.orrin").is_file(),
            "{:?}",
            logged(&editor)
        );
        assert!(std::path::Path::new(&editor.state.scene_path).ends_with("scenes/Level 3.orrin"));
    }

    /// The project's scenes are what the tool is opened for; the renderer's
    /// switches come after them, not between the header and the list.
    #[test]
    fn the_scene_list_comes_before_the_renderer() {
        let dir = crate::asset_db::tests::Dir::new("scene-manager-order");
        dir.write("levels/castle.orrin", b"");
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);
        editor.dock.activate(Tab::Scenes);
        editor.frames(3);
        let castle = painted_rect_of(&editor, "castle").expect("castle is listed");
        let renderer = painted_rect_of(&editor, "Renderer").expect("a renderer section");
        assert!(castle.top() < renderer.top());
    }

    /// Docked at the hierarchy's width, a switch still says which pass it
    /// turns off: a property row cut "Ambient …" and "Volumetri…".
    #[test]
    fn the_renderer_switches_keep_their_names_when_docked() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.dock.activate(Tab::Scenes);
        editor.frames(3);
        fn collect(shape: &egui::Shape, out: &mut Vec<(String, bool)>) {
            match shape {
                egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, out)),
                egui::Shape::Text(text) => {
                    out.push((text.galley.text().to_owned(), text.galley.elided))
                }
                _ => {}
            }
        }
        let mut texts = Vec::new();
        for clipped in &editor.output.as_ref().unwrap().shapes {
            collect(&clipped.shape, &mut texts);
        }
        // An elided galley still reports its whole text; the flag is what
        // says it was cut.
        for name in ["Ambient occlusion", "Temporal AA", "Volumetric fog"] {
            let (_, elided) = texts
                .iter()
                .find(|(text, _)| text == name)
                .unwrap_or_else(|| panic!("{name} was not painted"));
            assert!(!elided, "{name} was cut short");
        }
    }

    /// Whether the scene is saved is said beside its name, where the eye
    /// already is, not at the far edge of a wide window.
    #[test]
    fn the_save_state_sits_beside_the_scene_name() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor
            .dock
            .apply(crate::editor::state::DockRequest::Float(Tab::Scenes));
        editor.frames(3);
        let name = crate::scene::loaded::name_of(&editor.state.scene_path);
        let title = painted_rect_of(&editor, &name).expect("the scene's name");
        let pill = painted_rect_of(&editor, "Saved").expect("the save state");
        assert!((pill.center().y - title.center().y).abs() < 6.0);
        assert!(pill.left() > title.right() && pill.left() - title.right() < 24.0);
    }

    /// The renderer section says where the settings live; choosing the entity
    /// is the hierarchy's job, not a button's here.
    #[test]
    fn the_renderer_section_has_no_select_button() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.state.request_add_render_settings();
        editor.state.apply(&mut editor.world, &editor.registry);
        editor.dock.activate(Tab::Scenes);
        editor.frames(2);
        let body = editor.tab_body(Tab::Scenes);
        assert!(
            !editor
                .painted_text_rects()
                .into_iter()
                .any(|(text, rect)| text == "Select" && body.contains(rect.center()))
        );
    }

    /// A scene is deleted only once the user has said yes to a dialog that
    /// says it cannot be undone.
    #[test]
    fn deleting_a_scene_asks_and_then_removes_it() {
        let dir = crate::asset_db::tests::Dir::new("scene-manager-delete");
        dir.write("levels/castle.orrin", b"orrin-scene 3 min-reader 3\n");
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);
        editor.dock.activate(Tab::Scenes);
        editor.frames(2);

        editor
            .right_click(text_at(&editor, "castle").unwrap())
            .frames(1);
        editor
            .click(text_at(&editor, "Delete…").expect("a delete item"))
            .frames(1);
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(
            dir.path().join("levels/castle.orrin").is_file(),
            "not before the answer"
        );
        assert!(editor.painted_contains("cannot be undone"));

        editor.click(text_at(&editor, "Delete").expect("the confirming button"));
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(!dir.path().join("levels/castle.orrin").exists());
        assert!(!dir.path().join("levels/castle.orrin.meta").exists());
        editor.frames(2);
        let body = editor.tab_body(Tab::Scenes);
        assert!(
            !editor
                .painted_text_rects()
                .into_iter()
                .any(|(text, rect)| text == "castle" && body.contains(rect.center())),
            "the row is gone from the list"
        );
    }

    /// Cancel leaves the file where it was.
    #[test]
    fn a_cancelled_delete_keeps_the_file() {
        let dir = crate::asset_db::tests::Dir::new("scene-manager-delete-cancel");
        dir.write("levels/castle.orrin", b"");
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);
        editor.state.request_delete_asset("levels/castle.orrin");
        editor.frames(2);
        editor.click(text_at(&editor, "Cancel").unwrap());
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(dir.path().join("levels/castle.orrin").is_file());
        assert!(editor.state.pending_delete().is_none());
    }

    /// The open scene is what the editor is showing and would save over; it
    /// is refused by name rather than deleted out from under the session.
    #[test]
    fn the_open_scene_cannot_be_deleted() {
        let dir = crate::asset_db::tests::Dir::new("scene-manager-delete-open");
        dir.write("levels/castle.orrin", b"orrin-scene 3 min-reader 3\n");
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);
        editor
            .state
            .request_open_scene(dir.path().join("levels/castle.orrin").to_string_lossy());
        editor.state.apply(&mut editor.world, &editor.registry);

        editor.state.request_delete_asset("levels/castle.orrin");
        editor.state.resolve_delete(true);
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(dir.path().join("levels/castle.orrin").is_file());
        assert!(
            logged(&editor).iter().any(|m| m.contains("open scene")),
            "{:?}",
            logged(&editor)
        );
    }

    /// The asset browser deletes the same way, from an asset's own menu.
    #[test]
    fn the_asset_browser_deletes_an_asset_after_asking() {
        let dir = crate::asset_db::tests::Dir::new("assets-delete");
        dir.write("rock.png", b"");
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);
        editor.dock.activate(Tab::Assets);
        editor.frames(3);

        let rock = text_at(&editor, "rock.png")
            .or_else(|| text_at(&editor, "rock"))
            .expect("the asset is listed");
        editor.right_click(rock).frames(1);
        editor
            .click(text_at(&editor, "Delete…").expect("a delete item"))
            .frames(1);
        editor.click(text_at(&editor, "Delete").expect("the confirming button"));
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(!dir.path().join("rock.png").exists());
    }

    /// The header says plainly whether the open scene has unsaved work.
    #[test]
    fn the_scene_manager_header_says_whether_the_scene_is_saved() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.dock.activate(Tab::Scenes);
        editor.frames(2);
        assert!(editor.painted_contains("Saved"));
        delete_by_key(&mut editor, "Old crate");
        editor.frames(2);
        assert!(editor.painted_contains("Unsaved"));
    }

    /// The project's scenes are listed by name, and nothing that is not a
    /// scene is.
    #[test]
    fn the_scene_manager_lists_the_projects_scenes() {
        let dir = crate::asset_db::tests::Dir::new("scene-manager-list");
        dir.write("levels/castle.orrin", b"");
        dir.write("levels/dungeon.orrin", b"");
        dir.write("rock.png", b"");
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);
        editor.dock.activate(Tab::Scenes);
        editor.frames(3);

        assert!(editor.painted_contains("castle"));
        assert!(editor.painted_contains("dungeon"));
        assert!(!editor.painted_contains("rock"));
    }

    /// Where the renderer's settings live is the thing the window has to make
    /// plain: in the scene, or nowhere that survives a load.
    #[test]
    fn the_scene_manager_says_whether_the_renderer_is_saved_with_the_scene() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.dock.activate(Tab::Scenes);
        editor.frames(2);
        assert!(editor.painted_contains("Save with scene"));

        editor.state.request_add_render_settings();
        editor.state.apply(&mut editor.world, &editor.registry);
        editor.frames(2);
        assert!(editor.painted_contains("Saved with the scene"));
        assert!(!editor.painted_contains("Save with scene"));
    }

    /// While the game plays, what is loaded is more than the scene file.
    #[test]
    fn the_scene_manager_lists_what_is_loaded_while_playing() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.state.persistent_scene = Some(persistent_scene(&editor, "manager-loaded"));
        toggle_play(&mut editor);
        editor.dock.activate(Tab::Scenes);
        editor.frames(2);
        assert!(editor.painted_contains("Persistent"));
    }

    /// A new scene is a file at once, holding the settings a scene needs to
    /// be drawn and a sun to see it by, and it is what the editor has open.
    #[test]
    fn a_new_scene_is_written_with_render_settings_and_opened() {
        let dir = crate::asset_db::tests::Dir::new("scene-manager-new");
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);
        editor.spawn("Old crate");

        editor.state.request_new_scene("Level 2");
        editor.state.apply(&mut editor.world, &editor.registry);

        let file = dir.path().join("scenes").join("Level 2.orrin");
        assert!(file.is_file(), "{:?}", logged(&editor));
        assert!(named(&editor.world, "Old crate").is_none());
        let settings = named(&editor.world, "Render Settings").expect("settings entity");
        assert!(
            editor
                .world
                .get::<crate::scene::RenderSettings>(settings)
                .is_some()
        );
        assert!(named(&editor.world, "Sun").is_some());
        assert!(std::path::Path::new(&editor.state.scene_path).ends_with("scenes/Level 2.orrin"));
        assert!(!editor.state.scene_dirty());
    }

    /// A new scene looks like the demo a bare `cargo run -p orrin-core` opens
    /// on: its generated sky, and the sun that sky is drawn around. Nothing is
    /// copied into the project for it.
    #[test]
    fn a_new_scene_is_lit_by_the_demos_sky_and_sun() {
        let dir = crate::asset_db::tests::Dir::new("scene-manager-sky");
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);

        editor.state.request_new_scene("Outside");
        editor.state.apply(&mut editor.world, &editor.registry);

        let settings = named(&editor.world, "Render Settings").expect("settings entity");
        let render = editor
            .world
            .get::<crate::scene::RenderSettings>(settings)
            .expect("render settings")
            .clone();
        assert_eq!(
            render.environment.hdri,
            crate::scene::entities::BUILT_IN_SKY
        );
        let sun = named(&editor.world, "Sun").expect("a sun");
        let light = editor
            .world
            .get::<crate::scene::Light>(sun)
            .map(|light| *light);
        assert!(
            matches!(
                light,
                Some(crate::scene::Light::Directional { color, lux })
                    if color == glam::Vec3::new(1.0, 0.97, 0.92) && lux == 250.0
            ),
            "the demo's dusk sun, not a noon one against a dusk sky: {light:?}"
        );
        assert!(
            !dir.path().join("environment").exists(),
            "nothing copied in"
        );
    }

    /// The name becomes a file name, so it has to be one, and must not take
    /// the place of a scene that already exists.
    #[test]
    fn a_new_scene_refuses_a_name_that_is_not_a_new_file() {
        let dir = crate::asset_db::tests::Dir::new("scene-manager-bad-name");
        dir.write("scenes/castle.orrin", b"kept");
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        with_database(&mut editor, &dir);

        for name in ["", "  ", "a/b", "castle"] {
            editor.state.request_new_scene(name);
            editor.state.apply(&mut editor.world, &editor.registry);
        }
        assert_eq!(
            std::fs::read(dir.path().join("scenes/castle.orrin")).unwrap(),
            b"kept"
        );
        assert!(!dir.path().join("scenes/a").exists());
        let errors = logged(&editor)
            .into_iter()
            .filter(|message| message.starts_with("new scene"))
            .count();
        assert_eq!(errors, 4, "{:?}", logged(&editor));
    }

    /// Stop respawns the scene into new slots; what was selected is selected
    /// again, by identity.
    #[test]
    fn the_selection_survives_stop() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.spawn("Other");
        let crate_ = editor.spawn("Crate");
        editor.state.selection.set(crate_);

        toggle_play(&mut editor);
        toggle_play(&mut editor);

        let primary = editor.state.selection.primary();
        assert!(primary.is_some());
        assert_eq!(primary, named(&editor.world, "Crate"));
    }

    /// Saving the fork would write the game's state over the scene. Ctrl+S
    /// while playing says why it did nothing instead.
    #[test]
    fn saving_while_playing_writes_nothing_and_says_so() {
        let dir = std::env::temp_dir().join(format!("orrin-play-save-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("level.orrin");
        let _ = std::fs::remove_file(&path);
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.spawn("Crate");
        editor.state.scene_path = path.to_string_lossy().into_owned();

        toggle_play(&mut editor);
        editor.key_with(egui::Modifiers::COMMAND, egui::Key::S);
        editor.state.apply(&mut editor.world, &editor.registry);

        assert!(!path.exists());
        let warned = editor
            .world
            .resource::<crate::scene::LogBuffer>()
            .iter()
            .any(|entry| {
                entry.level == crate::scene::LogLevel::Warning && entry.message.contains("Stop")
            });
        assert!(warned);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The transport is live: Play is a button, and while the game runs the
    /// same button is Stop, lit in the accent.
    #[test]
    fn the_play_button_plays_and_becomes_stop() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        let play = crate::editor::icons::play().glyph();
        let stop = crate::editor::icons::stop().glyph();
        let (_, rect, _) = editor
            .painted_icons()
            .into_iter()
            .find(|(glyph, _, _)| *glyph == play)
            .expect("a play button");

        editor.click(rect.center());
        editor.state.apply(&mut editor.world, &editor.registry);
        assert_eq!(play_state(&editor), crate::scene::PlayState::Playing);

        editor.frames(1);
        let glyphs: Vec<char> = editor.painted_icons().iter().map(|(g, _, _)| *g).collect();
        assert!(glyphs.contains(&stop), "{glyphs:?}");
        assert!(!glyphs.contains(&play), "{glyphs:?}");
    }

    /// Pause holds the fork where it is; Step lets one frame through and stays
    /// paused. Both are keys as well as buttons.
    #[test]
    fn pause_and_step_hold_the_game() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        toggle_play(&mut editor);
        let shift = egui::Modifiers::COMMAND | egui::Modifiers::SHIFT;
        editor.key_with(shift, egui::Key::P);
        editor.state.apply(&mut editor.world, &editor.registry);
        assert_eq!(play_state(&editor), crate::scene::PlayState::Paused);

        let alt = egui::Modifiers::COMMAND | egui::Modifiers::ALT;
        editor.key_with(alt, egui::Key::P);
        editor.state.apply(&mut editor.world, &editor.registry);
        let mut play = editor.world.resource_mut::<crate::scene::Play>();
        assert!(play.advance());
        assert!(!play.advance());
        assert_eq!(play.state(), crate::scene::PlayState::Paused);
    }

    /// Unsaved work is marked beside the scene's name, so an autosave or a
    /// recovered scene is never mistaken for one on disk.
    #[test]
    fn the_bar_marks_an_unsaved_scene() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        let marked = |editor: &Harness| {
            editor
                .painted()
                .iter()
                .any(|text| text.contains(&editor.state.scene_path) && text.contains('•'))
        };
        assert!(!marked(&editor));
        delete_by_key(&mut editor, "Crate");
        editor.frames(1);
        assert!(marked(&editor));
    }

    fn autosaving(name: &str) -> (Harness, std::path::PathBuf, std::time::Instant) {
        use crate::editor::autosave::Autosave;
        let dir =
            std::env::temp_dir().join(format!("orrin-autosave-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let start = std::time::Instant::now();
        editor.state.autosave = Autosave::new(dir.clone(), start);
        (editor, dir, start)
    }

    fn autosave_at(editor: &mut Harness, at: std::time::Instant) {
        let Harness {
            world,
            registry,
            state,
            ..
        } = editor;
        state.autosave(world, registry, at);
        state.autosave.wait();
    }

    /// The copy is the scene file a save would have written, byte for byte, so
    /// recovering it loses nothing a save would have kept.
    #[test]
    fn an_autosave_is_what_a_save_would_write() {
        let (mut editor, dir, start) = autosaving("bytes");
        editor.spawn("Keep");
        delete_by_key(&mut editor, "Crate");

        autosave_at(&mut editor, start + crate::editor::autosave::INTERVAL);

        let path = editor.state.autosave.path(&editor.state.scene_path);
        let copy = std::fs::read_to_string(&path).expect("an autosave");
        assert_eq!(
            copy,
            crate::scene::save(&mut editor.world, &editor.registry)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The fork is not the scene, and nothing in edit mode changes while it
    /// runs, so play mode is never copied.
    #[test]
    fn nothing_is_autosaved_while_playing() {
        let (mut editor, dir, start) = autosaving("playing");
        delete_by_key(&mut editor, "Crate");
        toggle_play(&mut editor);

        autosave_at(&mut editor, start + crate::editor::autosave::INTERVAL);

        assert!(
            !editor
                .state
                .autosave
                .path(&editor.state.scene_path)
                .exists()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A recovered copy is unsaved work, not the scene file: it opens dirty,
    /// for Save to put back where it belongs.
    #[test]
    fn recovering_an_autosave_opens_it_unsaved() {
        let (mut editor, dir, start) = autosaving("recover");
        editor.spawn("Keep");
        delete_by_key(&mut editor, "Crate");
        autosave_at(&mut editor, start + crate::editor::autosave::INTERVAL);
        editor.spawn("Later");

        editor
            .state
            .request_scene(crate::editor::state::SceneRequest::Recover);
        editor.state.apply(&mut editor.world, &editor.registry);

        assert!(named(&editor.world, "Keep").is_some());
        assert!(named(&editor.world, "Later").is_none());
        assert!(editor.state.history.is_dirty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// While the game is live the scene is framed in the accent, so a change
    /// made to the fork is never mistaken for one made to the scene.
    #[test]
    fn the_viewport_is_framed_while_playing() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let accent = editor.themes.active().accent();
        let framed = |editor: &Harness| {
            let viewport = editor.dock.viewport().expect("a viewport");
            !editor
                .painted_rects_of(|shape| {
                    matches!(shape, egui::Shape::Rect(rect)
                        if rect.stroke.color == accent && rect.stroke.width >= 2.0)
                })
                .into_iter()
                .filter(|rect| rect.intersects(viewport) && rect.width() >= viewport.width() - 4.0)
                .collect::<Vec<_>>()
                .is_empty()
        };
        editor.frames(2);
        assert!(!framed(&editor));
        toggle_play(&mut editor);
        editor.frames(1);
        assert!(framed(&editor));
    }

    // --- the retained UI ---------------------------------------------------

    fn ui_spawn(editor: &mut Harness, kind: crate::ui::create::UiKind) {
        editor
            .state
            .request_spawn(crate::editor::state::SpawnKind::Ui(kind));
        editor.state.apply(&mut editor.world, &editor.registry);
    }

    fn parent_of(world: &World, entity: orrin_ecs::Entity) -> Option<orrin_ecs::Entity> {
        world
            .get::<crate::scene::Parent>(entity)
            .map(|parent| parent.get())
    }

    /// A canvas, made the first time, with an element on it placed by its top
    /// left at `at` and `size` points — one point a pixel in the harness.
    fn ui_box(
        editor: &mut Harness,
        name: &str,
        at: egui::Pos2,
        size: egui::Vec2,
    ) -> orrin_ecs::Entity {
        let canvas = named(&editor.world, "Canvas").unwrap_or_else(|| {
            let canvas = editor.world.spawn();
            editor.world.insert(canvas, Name::new("Canvas"));
            editor.world.insert(canvas, crate::ui::Canvas::default());
            canvas
        });
        let entity = editor.world.spawn();
        editor.world.insert(entity, Name::new(name));
        editor.world.insert(
            entity,
            crate::ui::RectTransform {
                anchor_min: glam::Vec2::ZERO,
                anchor_max: glam::Vec2::ZERO,
                pivot: glam::Vec2::ZERO,
                anchored_position: glam::Vec2::new(at.x, at.y),
                size_delta: glam::Vec2::new(size.x, size.y),
                scale: glam::Vec2::ONE,
            },
        );
        editor.world.insert(entity, crate::ui::Image::default());
        crate::scene::reparent(&mut editor.world, entity, Some(canvas), false).unwrap();
        entity
    }

    fn anchored(editor: &Harness, entity: orrin_ecs::Entity) -> crate::ui::RectTransform {
        *editor
            .world
            .get::<crate::ui::RectTransform>(entity)
            .unwrap()
    }

    /// uGUI's button: an image that tints, and a label under it, on a canvas
    /// made for it — and all of it one undo step.
    #[test]
    fn creating_a_ui_button_builds_its_tree_under_a_new_canvas() {
        use crate::ui::create::UiKind;
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        ui_spawn(&mut editor, UiKind::Button);

        let world = &editor.world;
        let canvas = named(world, "Canvas").expect("a canvas to hold it");
        let button = named(world, "Button").expect("the button");
        let label = named(world, "Text").expect("its label");
        assert!(world.has::<crate::ui::Canvas>(canvas));
        assert_eq!(parent_of(world, canvas), None);
        assert_eq!(parent_of(world, button), Some(canvas));
        assert_eq!(parent_of(world, label), Some(button));
        for component in [
            world.has::<crate::ui::RectTransform>(button),
            world.has::<crate::ui::Image>(button),
            world.has::<crate::ui::Button>(button),
            world.has::<crate::ui::RectTransform>(label),
        ] {
            assert!(component);
        }
        assert!(
            !world.has::<LocalTransform>(button),
            "UI has no 3D transform"
        );
        let text = world.get::<crate::ui::Text>(label).unwrap();
        assert_eq!(text.text, "Button");
        assert_eq!(text.anchor, crate::scene::TextAnchor::MiddleCenter);
        assert_eq!(
            *world.get::<crate::ui::RectTransform>(label).unwrap(),
            crate::ui::RectTransform::stretched()
        );
        drop(text);
        assert_eq!(editor.state.selection.entities(), [button]);

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        assert_eq!(
            editor.world.entities().count(),
            0,
            "one step took all three"
        );
    }

    #[test]
    fn new_ui_goes_under_the_selection_or_else_the_first_canvas() {
        use crate::ui::create::UiKind;
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        ui_spawn(&mut editor, UiKind::Panel);
        let panel = named(&editor.world, "Panel").unwrap();
        let canvas = named(&editor.world, "Canvas").unwrap();
        ui_spawn(&mut editor, UiKind::Text);
        let text = named(&editor.world, "Text").unwrap();
        assert_eq!(parent_of(&editor.world, text), Some(panel));

        let cube = editor.spawn("Cube");
        editor.state.selection.set(cube);
        ui_spawn(&mut editor, UiKind::Image);
        let image = named(&editor.world, "Image").unwrap();
        assert_eq!(parent_of(&editor.world, image), Some(canvas));

        ui_spawn(&mut editor, UiKind::Canvas);
        let canvases = editor
            .world
            .entities()
            .filter(|&e| editor.world.has::<crate::ui::Canvas>(e))
            .count();
        assert_eq!(canvases, 2, "a canvas is always a new one");
    }

    #[test]
    fn t_switches_ui_mode() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(1);
        editor.key(egui::Key::T);
        assert!(editor.state.ui_mode);
        editor.key(egui::Key::T);
        assert!(!editor.state.ui_mode);
    }

    #[test]
    fn in_ui_mode_a_click_selects_the_topmost_element() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        let centre = editor.viewport().center();
        let back = ui_box(
            &mut editor,
            "Back",
            centre - egui::vec2(100.0, 100.0),
            egui::vec2(200.0, 200.0),
        );
        let front = ui_box(
            &mut editor,
            "Front",
            centre - egui::vec2(25.0, 25.0),
            egui::vec2(50.0, 50.0),
        );
        editor.state.ui_mode = true;
        editor.frames(2);

        editor.click(centre);
        assert_eq!(editor.state.selection.entities(), [front]);
        editor.click(centre + egui::vec2(80.0, 80.0));
        assert_eq!(editor.state.selection.entities(), [back]);
        editor.click(centre + egui::vec2(150.0, 0.0));
        assert!(editor.state.selection.is_empty(), "nothing there");
    }

    /// The rect tool's body is the element: a drag moves it, shows as it
    /// goes, and undoes in one step. Ctrl places freely, so where the window's
    /// centre happens to fall cannot snap it.
    #[test]
    fn dragging_a_ui_element_moves_it_in_one_undo_step() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        let centre = editor.viewport().center();
        let panel = ui_box(
            &mut editor,
            "Panel",
            centre - egui::vec2(50.0, 50.0),
            egui::vec2(100.0, 100.0),
        );
        let before = anchored(&editor, panel);
        editor.state.selection.set(panel);
        editor.state.ui_mode = true;
        editor.frames(2);
        assert!(
            !editor.state.gizmo_strokes.is_empty(),
            "the rect tool is drawn"
        );

        editor.input.modifiers = egui::Modifiers::COMMAND;
        pointer(&mut editor, egui::Event::PointerMoved(centre));
        pointer(&mut editor, button(centre, true));
        let mut at = centre;
        for _ in 0..3 {
            at += egui::vec2(10.0, 4.0);
            pointer(&mut editor, egui::Event::PointerMoved(at));
        }
        assert_eq!(
            anchored(&editor, panel).anchored_position,
            before.anchored_position + glam::Vec2::new(30.0, 12.0),
            "the drag shows while it is held"
        );
        pointer(&mut editor, button(at, false));
        editor.input.modifiers = egui::Modifiers::NONE;
        assert_eq!(editor.state.selection.entities(), [panel]);

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        assert_eq!(anchored(&editor, panel), before);
        assert_eq!(editor.state.history.undo_label(), None, "one step");
    }

    #[test]
    fn dragging_a_corner_of_the_rect_tool_resizes_from_the_opposite_one() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        let centre = editor.viewport().center();
        let panel = ui_box(
            &mut editor,
            "Panel",
            centre - egui::vec2(50.0, 50.0),
            egui::vec2(100.0, 100.0),
        );
        let before = anchored(&editor, panel);
        editor.state.selection.set(panel);
        editor.state.ui_mode = true;
        editor.frames(2);

        editor.input.modifiers = egui::Modifiers::COMMAND;
        let corner = centre + egui::vec2(50.0, 50.0);
        pointer(&mut editor, egui::Event::PointerMoved(corner));
        pointer(&mut editor, button(corner, true));
        pointer(
            &mut editor,
            egui::Event::PointerMoved(corner + egui::vec2(20.0, 10.0)),
        );
        pointer(&mut editor, button(corner + egui::vec2(20.0, 10.0), false));
        editor.input.modifiers = egui::Modifiers::NONE;

        let after = anchored(&editor, panel);
        assert_eq!(after.size_delta, glam::Vec2::new(120.0, 110.0));
        assert_eq!(
            after.anchored_position, before.anchored_position,
            "pivoted at its top left, which stays put"
        );
        assert_eq!(
            editor.state.selection.entities(),
            [panel],
            "a handle is not a click"
        );
    }

    #[test]
    fn the_rect_tool_and_ui_picking_are_off_while_the_game_is_live() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        let centre = editor.viewport().center();
        let panel = ui_box(
            &mut editor,
            "Panel",
            centre - egui::vec2(50.0, 50.0),
            egui::vec2(100.0, 100.0),
        );
        editor.state.ui_mode = true;
        editor.state.selection.set(panel);
        editor.frames(2);
        assert!(!editor.state.gizmo_strokes.is_empty());

        toggle_play(&mut editor);
        editor.frames(2);
        assert!(
            editor.state.gizmo_strokes.is_empty(),
            "no rect tool over the game"
        );
        editor.click(centre);
        assert!(editor.state.selection.is_empty(), "a click is the game's");
    }

    /// A scroll frame is made with a backdrop to catch the wheel against and
    /// content taller than itself, so it scrolls as soon as it has children.
    #[test]
    fn creating_a_scroll_frame_makes_one_with_room_to_scroll() {
        use crate::ui::create::UiKind;
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        ui_spawn(&mut editor, UiKind::ScrollFrame);
        let world = &editor.world;
        let frame = named(world, "Scroll Frame").expect("the frame");
        let canvas = named(world, "Canvas").expect("a canvas to hold it");
        assert_eq!(parent_of(world, frame), Some(canvas));
        assert!(world.has::<crate::ui::Image>(frame));
        let rect = *world.get::<crate::ui::RectTransform>(frame).unwrap();
        let scroll = *world.get::<crate::ui::ScrollFrame>(frame).unwrap();
        assert!(scroll.vertical);
        assert!(scroll.content_size.y > rect.size_delta.y, "{scroll:?}");
        assert_eq!(editor.state.selection.entities(), [frame]);
    }

    /// Each modifier has a section of its own, drawn once, and none is drawn
    /// a second time from its vtable.
    #[test]
    fn the_inspector_draws_the_modifier_sections_once_each() {
        use crate::ui::create::UiKind;
        let mut editor = Harness::new(egui::vec2(1280.0, 2400.0));
        ui_spawn(&mut editor, UiKind::Button);
        let button = named(&editor.world, "Button").unwrap();
        editor.world.insert(button, crate::ui::Gradient::default());
        editor.world.insert(button, crate::ui::Corner::default());
        editor.world.insert(button, crate::ui::Stroke::default());
        editor.world.insert(button, crate::ui::Shadow::default());
        editor
            .world
            .insert(button, crate::ui::ScrollFrame::default());
        editor.frames(3);
        let painted = editor.painted();
        let count = |label: &str| {
            painted
                .iter()
                .filter(|text| without_icons(text) == label)
                .count()
        };
        for label in [
            "Gradient",
            "Rotation",
            "Corner",
            "Radius",
            "Squircle",
            "Stroke",
            "Thickness",
            "Placement",
            "Shadow",
            "Blur",
            "Spread",
            "Scroll Frame",
            "Content size",
            "Scroll position",
            "Scrollbars",
        ] {
            assert_eq!(count(label), 1, "{label}: {painted:?}");
        }
        for generic in ["radius_fraction", "scroll_position", "scrollbar_color"] {
            assert_eq!(
                count(generic),
                0,
                "drawn by hand, not from the vtable: {painted:?}"
            );
        }
    }

    #[test]
    fn the_inspector_draws_the_ui_sections_once_each() {
        use crate::ui::create::UiKind;
        let mut editor = Harness::new(egui::vec2(1280.0, 900.0));
        ui_spawn(&mut editor, UiKind::Button);
        editor.frames(3);
        let painted = editor.painted();
        let count = |label: &str| {
            painted
                .iter()
                .filter(|text| without_icons(text) == label)
                .count()
        };
        for label in [
            "Rect Transform",
            "Anchors",
            "Pivot",
            "Normal",
            "Pressed",
            "Interactable",
            "Raycast target",
        ] {
            assert_eq!(count(label), 1, "{label}: {painted:?}");
        }
        for generic in ["Anchored position", "Size delta"] {
            assert_eq!(
                count(generic),
                0,
                "drawn by hand, not from the vtable: {painted:?}"
            );
        }
    }
}
