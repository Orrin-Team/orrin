// Visible to the dock, which mounts each tool's `body` into a tab.
pub(super) mod console;
pub(super) mod environment;
pub(super) mod hierarchy;
pub(super) mod inspector;
pub(super) mod performance;
mod ribbon;
pub(super) mod scene;
#[cfg(feature = "scripting")]
pub(super) mod scripts;

use glam::Vec3;

use orrin_ecs::World;
use orrin_registry::Registry;

use super::dock::Dock;
use super::state::EditorState;
use super::theme::ThemeSet;

// The top bar is mounted first so its three rows span the full window width.
// Everything below it belongs to the dock, whose centre paints nothing — that
// is what leaves the 3D scene visible behind the editor.
pub fn draw(
    ctx: &egui::Context,
    world: &mut World,
    state: &mut EditorState,
    registry: &Registry,
    themes: &ThemeSet,
    dock: &mut Dock,
) {
    // Before the dock, so a command that rearranges the layout is reflected in
    // the same frame rather than the next one.
    ribbon::show(ctx, world, state, registry, themes, dock);
    dock.show(ctx, world, state, registry);
}

/// Every number, path, id and log line is monospace, so columns of figures
/// align by eye rather than drifting with the width of each digit.
pub(super) fn figures(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text).monospace()
}

pub(super) fn vec3_row(ui: &mut egui::Ui, label: &str, v: &mut Vec3, speed: f32) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::DragValue::new(&mut v.x).speed(speed));
        ui.add(egui::DragValue::new(&mut v.y).speed(speed));
        ui.add(egui::DragValue::new(&mut v.z).speed(speed));
    });
}

pub(super) fn color_row(ui: &mut egui::Ui, label: &str, c: &mut Vec3) {
    let mut rgb = [c.x, c.y, c.z];
    ui.horizontal(|ui| {
        ui.label(label);
        if ui.color_edit_button_rgb(&mut rgb).changed() {
            *c = Vec3::from(rgb);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::dock::{Dock, Tab};
    use crate::editor::state::RibbonTab;
    use crate::editor::theme;
    use crate::scene::{LocalTransform, Name};

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
            theme::apply(&ctx, themes.active());

            Self {
                world,
                registry,
                state: EditorState::default(),
                themes,
                dock: Dock::default(),
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
                    ctx,
                    input,
                    output,
                } = self;
                *output = Some(ctx.run(input.clone(), |ctx| {
                    draw(ctx, world, state, registry, themes, dock);
                }));
            }
            self
        }

        /// Every string the last frame actually painted. The editor is drawn,
        /// not returned, so this is the only way to assert that a row reached
        /// the screen rather than merely being iterated over.
        fn painted(&self) -> Vec<String> {
            fn collect(shape: &egui::Shape, out: &mut Vec<String>) {
                match shape {
                    egui::Shape::Text(text) => out.push(text.galley.text().to_owned()),
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| collect(shape, out)),
                    _ => {}
                }
            }
            let mut out = Vec::new();
            for clipped in &self.output.as_ref().expect("a frame has run").shapes {
                collect(&clipped.shape, &mut out);
            }
            out
        }

        /// Park the pointer, so hover-only affordances draw.
        fn hover(&mut self, at: egui::Pos2) -> &mut Self {
            self.input.events.push(egui::Event::PointerMoved(at));
            self.input.hovered_files.clear();
            self
        }

        /// Every image the last frame painted, with the rect it landed in.
        /// Icons are images, so this is how an icon's placement is asserted —
        /// `painted` only ever sees text. An unrotated image is a `RectShape`
        /// carrying a texture brush, which is what distinguishes one from the
        /// plain fills the same variant is used for.
        fn painted_images(&self) -> Vec<egui::Rect> {
            fn collect(shape: &egui::Shape, out: &mut Vec<egui::Rect>) {
                match shape {
                    egui::Shape::Rect(rect) if rect.brush.is_some() => out.push(rect.rect),
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

        fn panel(&self, id: &str) -> egui::Rect {
            egui::containers::panel::PanelState::load(&self.ctx, egui::Id::new(id))
                .expect("panel was drawn")
                .rect
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

    /// Delete moved off the Hierarchy toolbar, so the Inspector is now the only
    /// place that can despawn what you are looking at.
    #[test]
    fn the_inspector_offers_delete_for_the_selection() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        let entity = editor.spawn("Ground plane");
        editor.frames(1);
        assert!(!editor.painted_contains("Delete"));

        editor.state.selected = Some(entity);
        editor.frames(1);
        assert!(editor.painted_contains("Delete"));
    }

    /// Every ribbon tab has to lay out: a group that borrows a resource the
    /// other tabs do not touch will only fail on the tab that shows it.
    #[test]
    fn every_ribbon_tab_draws_its_groups() {
        for (tab, caption) in [
            (RibbonTab::Home, "Gizmo"),
            (RibbonTab::Scene, "Selection"),
            (RibbonTab::Render, "Passes"),
            (RibbonTab::Scripts, "Assembly"),
            (RibbonTab::Assets, "Asset pipeline"),
        ] {
            let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
            editor.state.ribbon_tab = tab;
            editor.frames(2);
            assert!(
                editor.painted_contains(caption),
                "{} tab did not draw {caption}",
                tab.label()
            );
        }
    }

    /// The engine opens at 800x600 and the top bar is three stacked panels. If
    /// they take the window, the tree and the viewport are gone and no test that
    /// only checks for panics would notice.
    #[test]
    fn the_top_bar_leaves_the_rest_of_the_window_alone() {
        let mut editor = Harness::new(egui::vec2(800.0, 600.0));
        editor.spawn("Ground plane");
        editor.frames(2);

        let top = editor.panel("quick_access").height()
            + editor.panel("ribbon").height()
            + editor.panel("scene_tabs").height();
        assert!(top < 200.0, "top bar took {top}px of 600");
        assert!(editor.painted_contains("Ground plane"));
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
                        ui.allocate_ui(cell, |ui| inspector::body(ui, world, state, registry));
                        ui.allocate_ui(cell, |ui| environment::body(ui, world));
                        ui.allocate_ui(cell, |ui| performance::body(ui, world));
                        ui.allocate_ui(cell, |ui| scene::body(ui, world, state));
                        ui.allocate_ui(cell, |ui| console::body(ui, world));
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
            "Save",         // scene
            "messages",     // console
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
        let cross = editor
            .painted_images()
            .into_iter()
            .filter(|rect| tree.contains(rect.center()) && (rect.center().y - row.y).abs() < 12.0)
            .max_by(|a, b| a.right().total_cmp(&b.right()))
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
    fn nothing_is_painted_over_the_viewport() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.spawn("Ground plane");
        editor.frames(3);

        // The gap between the tools, inset so a neighbour's own border or
        // separator does not count as covering it.
        let inset = 6.0;
        let viewport = egui::Rect::from_min_max(
            egui::pos2(
                editor.tab(Tab::Hierarchy).right() + inset,
                editor.panel("scene_tabs").bottom() + inset,
            ),
            egui::pos2(
                editor.tab(Tab::Environment).left() - inset,
                editor.tab(Tab::Console).top() - inset,
            ),
        );
        assert!(
            viewport.width() > 200.0 && viewport.height() > 200.0,
            "the layout left no viewport to check: {viewport:?}"
        );

        let covering: Vec<_> = editor
            .painted_rects()
            .into_iter()
            .filter(|rect| viewport.contains(rect.center()))
            .collect();
        assert!(
            covering.is_empty(),
            "{} shape(s) painted over the viewport, e.g. {:?}",
            covering.len(),
            covering.first()
        );
    }

    /// The default layout is specified in pixels by the design system but built
    /// from fractions, and a split's fraction describes the new node on two of
    /// the four `split_*` calls and the old node on the other two. Getting that
    /// backwards halves a column and looks plausible.
    #[test]
    fn the_default_layout_lands_on_the_specified_widths() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(3);

        // Within a separator's width of the design system's numbers.
        for (tab, target) in [
            (Tab::Hierarchy, 220.0),
            (Tab::Environment, 300.0),
            (Tab::Inspector, 280.0),
        ] {
            let width = editor.tab(tab).width();
            assert!(
                (width - target).abs() <= 2.0,
                "{} is {width}px wide, expected {target}",
                tab.title()
            );
        }
        let bottom = editor.tab(Tab::Console).height();
        assert!(
            (bottom - 200.0).abs() <= 2.0,
            "bottom dock is {bottom}px tall"
        );
    }

    /// `Ui::columns` divides the width and does nothing to make the contents fit
    /// it, so a column too narrow for a slider paints straight over its
    /// neighbour. Docked at 300px this panel shipped four columns of overlapping
    /// labels, and nothing about that is a panic or a missing shape — it is just
    /// unreadable.
    #[test]
    fn the_environment_tool_fits_the_width_it_is_docked_at() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
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
        editor.dock.activate(Tab::Performance);
        editor.frames(3);

        for figure in ["[INFO] saved 3 entities", "id 1", "CPU:", "Memory (RSS)"] {
            assert_eq!(
                editor.family_of(figure),
                Some(egui::FontFamily::Monospace),
                "{figure:?} is not monospace"
            );
        }

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

    /// Reset undoes whatever the user did to the tree, including a float.
    #[test]
    fn reset_layout_restores_the_default_tree() {
        let mut editor = Harness::new(egui::vec2(1280.0, 800.0));
        editor.frames(2);
        let home = editor.tab(Tab::Console);

        editor.dock.toggle_float(Tab::Console);
        assert!(editor.dock.is_floating(Tab::Console));

        editor.dock.reset();
        editor.frames(2);
        assert!(!editor.dock.is_floating(Tab::Console));
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
}
