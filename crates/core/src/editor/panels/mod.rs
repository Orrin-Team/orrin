// Visible to the dock, which mounts each tool's `body` into a tab.
pub(super) mod assets;
pub(super) mod console;
pub(super) mod environment;
pub(super) mod hierarchy;
pub(super) mod import;
pub(super) mod inspector;
mod menu_bar;
pub(super) mod palette;
pub(super) mod performance;
pub(super) mod scenes;
#[cfg(feature = "scripting")]
pub(super) mod scripts;
pub(super) mod settings;
pub(super) mod shader;
mod status_bar;
pub(super) mod storybook;
mod ui_sections;
mod viewport;

use glam::Vec3;

use orrin_ecs::World;
use orrin_registry::Registry;

use super::dock::Dock;
use super::previews::Previews;
use super::state::EditorState;
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
    // Before anything reads it: the tree's eyes, the viewport's reach.
    crate::scene::editor_hidden::refresh(world);
    // Before `shortcuts`, whose Escape is the selection's.
    dock.close_on_escape(ctx);
    // Before `shortcuts` too, which asks it whose the Copy key is.
    state.console.begin(
        ctx,
        world.get_resource::<crate::scene::LogBuffer>().as_deref(),
    );
    shortcuts(ctx, world, registry, state, dock);
    // The browser says so again as it draws; a closed one says nothing.
    state.assets_hovered = false;
    // Asked for by a tool's own controls last frame, while the dock was
    // drawing them and so could not move.
    if let Some(request) = state.take_dock_request() {
        dock.apply(request);
    }
    // Before the dock, so a command that rearranges the layout is reflected in
    // the same frame rather than the next one.
    menu_bar::show(ctx, world, state, themes, dock);
    status_bar::show(ctx, world, state, themes);
    dock.show(ctx, world, state, registry, themes, previews);
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
    if let Some(rels) = state.pending_delete().map(<[String]>::to_vec) {
        confirm_delete(ctx, world, state, &rels);
    }
    if state.pending_close() || state.pending_open().is_some() {
        confirm_unsaved(ctx, state);
    }
    palette::show(ctx, world, registry, state, dock);
    // A drag ends with the button and a typed edit with the field, so the
    // gesture they make stays open while either is still held. Read after the
    // widgets, which are what let go of the focus.
    let typing = ctx.memory(|memory| memory.focused().is_some());
    let holding = ctx.input(|input| input.pointer.any_down()) || typing;
    state.hold(holding);
    state.set_typing(typing);
    let closes = dock.settings_takes_escape() && escape_is_free(ctx, world, state);
    dock.arm_escape(closes);
}

/// Whether an Escape pressed next has nothing more particular to do than
/// close a window: no field to put back or leave, no chord being captured on
/// Settings ▸ Keys, no menu, list or dialog in front to dismiss, and no
/// cursor a running game is holding, which the app frees on the same key
/// before the editor draws (`scene::input::cursor`).
fn escape_is_free(ctx: &egui::Context, world: &World, state: &EditorState) -> bool {
    let typing = (ctx.wants_keyboard_input() && !state.settings_panel.search_is_idle())
        || state.settings_panel.capturing();
    let in_front = ctx.memory(|memory| {
        memory
            .areas()
            .visible_layer_ids()
            .iter()
            .any(|layer| layer.order == egui::Order::Foreground)
    });
    let locked = world
        .get_resource::<crate::scene::Play>()
        .is_some_and(|play| play.is_live())
        && world
            .get_resource::<crate::scene::CursorLock>()
            .is_some_and(|lock| lock.requested);
    !(typing || in_front || locked)
}

/// The one question every delete asks, whichever tool it came from: what goes,
/// and that it does not come back.
///
/// A modal rather than a line in the tool, because the answer is
/// irreversible and the tool asking may be scrolled, narrow or floating —
/// this and [`confirm_unsaved`] are the only places in the editor that take
/// the whole window to ask.
fn confirm_delete(ctx: &egui::Context, world: &World, state: &mut EditorState, rels: &[String]) {
    let name = |rel: &str| rel.rsplit('/').next().unwrap_or(rel).to_owned();
    let (title, what) = {
        let db = world.get_resource::<crate::asset_db::AssetDatabase>();
        // How many assets a folder takes with it; `None` for an asset.
        let inside = |rel: &str| {
            db.as_ref()
                .filter(|db| db.folders().iter().any(|folder| folder.path == rel))
                .map(|db| {
                    let prefix = format!("{rel}/");
                    db.iter()
                        .filter(|entry| entry.path.starts_with(&prefix))
                        .count()
                })
        };
        match rels {
            [rel] => (
                format!("Delete {}?", name(rel)),
                match inside(rel) {
                    Some(0) => "The folder is removed from disk.".to_owned(),
                    Some(1) => "The folder and the asset in it are removed from disk.".to_owned(),
                    Some(n) => {
                        format!("The folder and the {n} assets in it are removed from disk.")
                    }
                    None => format!("{rel} and its id are removed from disk."),
                },
            ),
            _ => {
                const LISTED: usize = 4;
                let mut names: Vec<String> = rels.iter().take(LISTED).map(|r| name(r)).collect();
                if rels.len() > LISTED {
                    names.push(format!("{} more", rels.len() - LISTED));
                }
                let held: usize = rels.iter().filter_map(|rel| inside(rel)).sum();
                let list = names.join(", ");
                (
                    format!("Delete {} items?", rels.len()),
                    match held {
                        0 => format!("{list} and their ids are removed from disk."),
                        1 => format!(
                            "{list}, the asset inside, and their ids are removed from disk."
                        ),
                        n => format!(
                            "{list}, the {n} assets inside, and their ids are removed from disk."
                        ),
                    },
                )
            }
        }
    };
    let modal = egui::Modal::new(egui::Id::new("confirm_delete")).show(ctx, |ui| {
        ui.set_width(340.0);
        ui.label(egui::RichText::new(title).size(15.0).strong());
        ui.add_space(6.0);
        ui.add(egui::Label::new(what).wrap());
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

/// What becomes of the scene's unsaved work before something replaces it:
/// another scene, the same one read again, or the end of the session.
///
/// A modal for the reason the delete question is one, and because the request
/// may come from the File menu, the Scenes tab, the Assets tab or the window's
/// own close button: a question asked inline in one tool goes unseen while
/// that tool is closed or behind another.
fn confirm_unsaved(ctx: &egui::Context, state: &mut EditorState) {
    use crate::editor::state::OpenChoice::{Cancel, Discard, SaveFirst};
    let current = crate::scene::loaded::name_of(&state.scene_path);
    let closing = state.pending_close();
    let (title, choices): (String, &[(&str, _)]) = if closing {
        (
            "Quit anyway?".to_owned(),
            &[
                ("Save and quit", SaveFirst),
                ("Quit without saving", Discard),
            ],
        )
    } else if state.pending_is_reload() {
        (
            format!("Reload {current}?"),
            &[("Reload without saving", Discard)],
        )
    } else {
        let next = state
            .pending_open()
            .map(crate::scene::loaded::name_of)
            .unwrap_or_default();
        (
            format!("Open {next}?"),
            &[
                ("Save and open", SaveFirst),
                ("Open without saving", Discard),
            ],
        )
    };
    let mut answer = None;
    let modal = egui::Modal::new(egui::Id::new("confirm_unsaved")).show(ctx, |ui| {
        ui.set_width(340.0);
        ui.label(egui::RichText::new(title).size(15.0).strong());
        ui.add_space(6.0);
        ui.label(format!("{current} has unsaved changes."));
        ui.add_space(10.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button("Cancel").clicked() {
                answer = Some(Cancel);
            }
            for &(label, choice) in choices.iter().rev() {
                if ui.button(label).clicked() {
                    answer = Some(choice);
                }
            }
        });
    });
    // Escape, or a click on the dimmed window around it.
    if modal.should_close() && answer.is_none() {
        answer = Some(Cancel);
    }
    if let Some(choice) = answer {
        if closing {
            state.resolve_close(choice);
        } else {
            state.resolve_open(choice);
        }
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

/// The keys every editor answers the same way, read from the command table
/// and the user's bindings (`editor::commands`). A press goes to the most
/// specific binding it matches, which runs unless it stands aside: while a
/// text field has focus, where the same key edits text — save and play
/// included while the focus is the Shader tab's code, which saves itself —
/// and while the palette is open, whose field every key but its own is for.
/// A press that stands aside is left where it was.
fn shortcuts(
    ctx: &egui::Context,
    world: &mut World,
    registry: &Registry,
    state: &mut EditorState,
    dock: &mut Dock,
) {
    // A chord being captured on Settings ▸ Keys is not also run.
    if settings::capture(ctx, state) {
        return;
    }
    use crate::editor::commands::{self, Command, Focus};
    state.selection.retain_alive(world);
    let coding = ctx.memory(|memory| memory.has_focus(shader::code_id()));
    let typing = ctx.wants_keyboard_input();
    let palette = state.palette.is_open();
    let selected = !state.selection.is_empty();
    // The focus check cannot see a field on the frame Escape arrives: egui
    // has already taken the keyboard from it. Taking the key here would hide
    // it from the field, which then commits instead of putting back. Nor may
    // it be taken from a menu or list drawn after this, which closes on the
    // same key: a command is the last thing Escape belongs to.
    let escape = !state.was_typing() && escape_is_free(ctx, world, state);
    let mut run: Vec<Command> = Vec::new();
    let keys = &state.settings.keys;
    let assets_hovered = state.assets_hovered;
    let console_claims = state.console.claims_keys();
    ctx.input_mut(|input| {
        let held = input.modifiers;
        input.events.retain(|event| {
            let Some((command, chord)) = commands::press(event, held).and_then(|p| keys.best(p))
            else {
                return true;
            };
            let spec = command.spec();
            let aside = spec.local
                || (palette && command != Command::Palette)
                || match spec.focus {
                    Focus::Anywhere => false,
                    Focus::OutsideCode => coding,
                    Focus::OutsideFields => typing,
                }
                || (chord.logical_key == egui::Key::Escape && !escape)
                || match command {
                    // Over the asset browser, Delete is its marked assets'.
                    Command::Delete => !selected || assets_hovered,
                    // The console's own rows, while it has the pointer or the
                    // last click; it takes the press as it draws.
                    Command::Copy => console_claims,
                    Command::SelectNone | Command::Frame => !selected,
                    _ => false,
                };
            if !aside {
                run.push(command);
            }
            aside
        });
    });
    for command in run {
        palette::run(ctx, world, registry, state, dock, command);
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

    mod authoring;
    mod browsing;
    mod feedback;
    mod intent;
    mod keys;
    mod layout;
    mod settings_tab;
    #[cfg(feature = "shader-compiler")]
    mod shader_tab;

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

        /// A primary press at `from`, dragged to `to` over ten frames and let
        /// go: how a separator is moved.
        fn drag(&mut self, from: egui::Pos2, to: egui::Pos2) -> &mut Self {
            let button = |pos, pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: Default::default(),
            };
            let mut events = vec![egui::Event::PointerMoved(from), button(from, true)];
            events.extend(
                (1..=10).map(|step| egui::Event::PointerMoved(from.lerp(to, step as f32 / 10.0))),
            );
            events.push(button(to, false));
            for event in events {
                self.input.events.push(event);
                self.frames(1);
                self.input.events.clear();
            }
            self.frames(2)
        }

        /// The separator between the left column and the rest of the dock.
        fn left_separator(&self) -> egui::Pos2 {
            let tree = self.tab(Tab::Hierarchy);
            egui::pos2(tree.right() + 0.5, tree.center().y)
        }

        /// The separator along the top of the bottom dock.
        fn bottom_separator(&self) -> egui::Pos2 {
            let dock = self.tab(Tab::Console);
            egui::pos2(dock.center().x, dock.top() - 0.5)
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

    /// A layout test must not depend on whose machine it ran on: the harness
    /// is built from defaults, and only the windowed editor is handed a user
    /// directory to read settings from.
    #[test]
    fn the_harness_reads_no_user_settings() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(1);
        assert_eq!(
            editor.state.settings,
            crate::editor::settings::Settings::default()
        );
        assert_eq!(editor.state.environment.editor_override(), None);
        assert_eq!(
            *editor.world.resource::<crate::scene::PresentSettings>(),
            crate::scene::PresentSettings::default()
        );
        // Nor how the user likes Orrin started, nor the variables that start it.
        assert_eq!(
            editor.state.started_with,
            crate::editor::settings::Startup::default()
        );
        assert_eq!(
            editor.state.environment,
            crate::editor::settings::Environment::default()
        );
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

    fn open_add_component(editor: &mut Harness) {
        let at = editor
            .painted_text_rects()
            .into_iter()
            .find(|(text, _)| without_icons(text) == "Add component")
            .map(|(_, rect)| rect.center())
            .expect("the inspector offers to add a component");
        editor.click(at).frames(1);
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

    /// A tint is an optional field, so setting one adds it to the value and
    /// clearing it takes it away: both an edit, and both undone.
    #[test]
    fn a_folder_s_tint_is_an_edit_both_ways() {
        use orrin_registry::Reflect;
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let folder = editor.world.spawn();
        editor.world.insert(folder, crate::scene::Folder::default());
        let tinted = crate::scene::Folder {
            tint: Some(glam::Vec3::new(0.2, 0.4, 0.6)),
        };
        let tint = |editor: &Harness| {
            editor
                .world
                .get::<crate::scene::Folder>(folder)
                .unwrap()
                .tint
        };
        for (value, expected) in [
            (tinted, tinted.tint),
            (crate::scene::Folder::default(), None),
        ] {
            editor
                .state
                .edit(folder, crate::scene::registry::FOLDER, value.to_value());
            editor.state.apply(&mut editor.world, &editor.registry);
            assert_eq!(tint(&editor), expected);
        }
        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        assert_eq!(tint(&editor), tinted.tint, "undoing the clear");
        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        assert_eq!(tint(&editor), None, "undoing the tint");
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
        let _picker = assets::PICKER
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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

    /// A project holding a mesh, `rock.glb`, and a texture, `stone.png`, with
    /// the clay material a spawned cube is drawn with. The mesh is a `.glb`
    /// rather than a `.blend` so the import worker does not start Blender.
    fn mesh_project(
        editor: &mut Harness,
        name: &str,
    ) -> (crate::asset_db::tests::Dir, [orrin_registry::AssetId; 2]) {
        let dir = crate::asset_db::tests::Dir::new(name);
        dir.write("rock.glb", b"");
        dir.write("stone.png", b"");
        let (db, _) = crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let ids = [
            db.find("rock.glb").unwrap().id,
            db.find("stone.png").unwrap().id,
        ];
        editor.world.insert_resource(db);
        let mut assets = crate::scene::Assets::new();
        assets.insert_material("clay", crate::scene::MaterialHandle(0));
        editor.world.insert_resource(assets);
        (dir, ids)
    }

    /// A mesh let go over the scene is placed as a prefab is: a new entity
    /// named for its file, drawing it with the default material, standing
    /// under the pointer, selected, and one step to undo. Anything else that
    /// is not a prefab is still refused by name.
    #[test]
    fn a_mesh_dropped_on_the_scene_is_placed_drawing_it_and_a_texture_is_refused() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (_dir, [rock, stone]) = mesh_project(&mut editor, "panel-drop-mesh");
        editor.frames(2);
        let at = editor.viewport().center();

        drop_payload(&mut editor, rock, at);
        let placed = named(&editor.world, "rock").expect("the mesh was placed");
        let handle = *editor
            .world
            .get::<crate::scene::MeshHandle>(placed)
            .unwrap();
        assert_eq!(
            editor
                .world
                .resource::<crate::asset_db::AssetDatabase>()
                .mesh_id(handle),
            Some(rock)
        );
        assert_eq!(
            editor
                .world
                .get::<crate::scene::MaterialHandle>(placed)
                .map(|m| *m),
            Some(crate::scene::MaterialHandle(0))
        );
        let camera = *editor.world.resource::<crate::scene::Camera>();
        let expected = super::viewport::ground_point(
            &camera,
            1280.0 / 800.0,
            egui::vec2(at.x / 1280.0 * 2.0 - 1.0, at.y / 800.0 * 2.0 - 1.0),
        )
        .expect("the default camera looks down at the ground");
        let standing = editor
            .world
            .get::<LocalTransform>(placed)
            .unwrap()
            .translation;
        assert!(
            (standing - expected).length() < 1e-4,
            "{standing} vs {expected}"
        );
        assert_eq!(editor.state.selection.primary(), Some(placed));
        assert_eq!(editor.state.history.undo_label(), Some("Place rock"));

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        assert!(!editor.world.is_alive(placed));
        assert_eq!(named(&editor.world, "rock"), None);

        let before = editor.world.entities().count();
        drop_payload(&mut editor, stone, at);
        assert_eq!(editor.world.entities().count(), before);
        let logged: Vec<String> = editor
            .world
            .resource::<crate::scene::LogBuffer>()
            .iter()
            .map(|entry| entry.message.clone())
            .collect();
        assert_eq!(logged, ["`stone.png` is a texture, not a prefab"]);
    }

    /// The Mesh slot's list offers the project's meshes beside the engine's,
    /// and picking one is the same single edit a drop on the slot makes.
    #[test]
    fn the_mesh_picker_offers_the_projects_meshes_and_picking_one_is_one_undoable_edit() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let (_dir, [rock, _]) = mesh_project(&mut editor, "panel-mesh-picker");
        editor
            .world
            .resource_mut::<crate::scene::Assets>()
            .insert_mesh("cube", crate::scene::MeshHandle(0));
        let crate_ = editor.spawn("Crate");
        editor.world.insert(crate_, crate::scene::MeshHandle(0));
        editor.world.insert(crate_, crate::scene::MaterialHandle(0));
        editor.state.selection.set(crate_);
        editor.frames(2);

        editor.click(row_of(&editor, "cube"));
        editor.frames(1);
        assert!(editor.painted_contains("rock.glb"));
        assert!(!editor.painted_contains("stone.png"));

        editor.click(row_of(&editor, "rock.glb"));
        editor.state.apply(&mut editor.world, &editor.registry);
        let handle = *editor
            .world
            .get::<crate::scene::MeshHandle>(crate_)
            .unwrap();
        assert_eq!(
            editor
                .world
                .resource::<crate::asset_db::AssetDatabase>()
                .mesh_id(handle),
            Some(rock)
        );
        assert_eq!(editor.state.history.undo_label(), Some("Edit Mesh"));
        editor.frames(1);
        assert!(editor.painted_contains("rock.glb"), "the slot names it");
        assert!(!editor.painted_contains("cube"));

        editor.state.request_undo();
        editor.state.apply(&mut editor.world, &editor.registry);
        assert_eq!(
            *editor
                .world
                .get::<crate::scene::MeshHandle>(crate_)
                .unwrap(),
            crate::scene::MeshHandle(0)
        );
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

    /// The eye sits on the row beside the lock. Hiding is a view setting,
    /// not an edit: nothing to undo, and the scene is not left unsaved.
    #[test]
    fn a_row_s_eye_hides_its_entity_from_the_view_without_an_edit() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let crate_ = editor.spawn("Crate");
        editor.frames(2);
        let band = row_band(&editor, "Crate");
        editor.hover(band.center()).frames(1);
        let eye = icon_in(&editor, crate::editor::icons::eye(), band)
            .expect("an open eye on the hovered row");
        editor.click(eye);
        editor.state.apply(&mut editor.world, &editor.registry);
        editor.frames(1);
        assert!(crate::scene::editor_hidden::is_hidden(
            &editor.world,
            crate_
        ));
        assert!(!editor.state.scene_dirty(), "hiding left the scene unsaved");

        let band = row_band(&editor, "Crate");
        editor.hover(band.center()).frames(1);
        let eye = icon_in(&editor, crate::editor::icons::eye_slash(), band)
            .expect("a closed eye on the hidden row");
        editor.click(eye);
        editor.state.apply(&mut editor.world, &editor.registry);
        editor.frames(1);
        assert!(!crate::scene::editor_hidden::is_hidden(
            &editor.world,
            crate_
        ));
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
}
