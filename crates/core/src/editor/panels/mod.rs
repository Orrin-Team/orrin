mod console;
mod environment;
mod hierarchy;
mod inspector;
mod performance;
mod ribbon;
mod scene;
#[cfg(feature = "scripting")]
mod scripts;

use glam::Vec3;

use orrin_ecs::World;
use orrin_registry::Registry;

use super::state::EditorState;
use super::theme::ThemeSet;

/// How far a side panel may be dragged. Bounds the drag only — a panel reports
/// the width of its *content*, so keeping one narrow is the content's job.
pub(super) const WIDTH_RANGE: std::ops::RangeInclusive<f32> = 160.0..=420.0;

// Side/bottom panels only — no `CentralPanel`, so the center stays transparent
// and the 3D scene shows through behind the editor. The top bar is mounted
// first so it spans the full window width rather than sitting between the side
// panels.
pub fn draw(
    ctx: &egui::Context,
    world: &mut World,
    state: &mut EditorState,
    registry: &Registry,
    themes: &ThemeSet,
) {
    ribbon::show(ctx, world, state, registry, themes);
    hierarchy::show(ctx, world, state);
    inspector::show(ctx, world, state, registry);
    environment::show(ctx, world);
    performance::show(ctx, world);
    #[cfg(feature = "scripting")]
    scripts::show(ctx, world, state);
    scene::show(ctx, world, state);
    console::show(ctx, world);
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
                    ctx,
                    input,
                    output,
                } = self;
                *output = Some(ctx.run(input.clone(), |ctx| {
                    draw(ctx, world, state, registry, themes);
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

        fn painted_contains(&self, needle: &str) -> bool {
            self.painted().iter().any(|text| text.contains(needle))
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

        // Over a row, which is what makes its ✕ draw at all. Low in the panel,
        // because the floating tool windows still open over the top of it and a
        // covered row does not register a hover.
        let tree = editor.panel("hierarchy");
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

    /// A panel that sizes its content from its own width grows by the overflow
    /// every frame: the content's minimum width becomes the panel's new width,
    /// which becomes a wider content minimum. Nothing shows on frame one.
    #[test]
    fn a_side_panel_does_not_widen_itself() {
        let mut settled = Harness::new(egui::vec2(800.0, 600.0));
        settled.frames(2);
        let mut later = Harness::new(egui::vec2(800.0, 600.0));
        later.frames(120);

        for build in [Harness::new, Harness::without_icons] {
            let mut settled = build(egui::vec2(800.0, 600.0));
            settled.frames(3);
            let mut later = build(egui::vec2(800.0, 600.0));
            later.frames(120);

            for panel in ["hierarchy", "inspector"] {
                assert_eq!(settled.panel(panel).width(), later.panel(panel).width());
            }
        }
    }

    /// The realistic route to a negative width: nobody resizes the window down
    /// to 320px, but a scene does contain an entity with a long name, and a row
    /// is as wide as the name in it.
    #[test]
    fn a_long_entity_name_cannot_push_a_side_panel_open() {
        let mut editor = Harness::new(egui::vec2(800.0, 600.0));
        editor.spawn(&"Spawn point ".repeat(40));
        editor.frames(4);

        assert!(editor.panel("hierarchy").width() <= *WIDTH_RANGE.end());
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
