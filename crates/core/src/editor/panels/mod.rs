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
    ribbon::show(ctx, state, themes);
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
    use crate::editor::theme;
    use crate::scene::Name;

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
            let mut world = World::default();
            crate::App::install_default_resources(&mut world);
            let mut registry = Registry::new();
            crate::scene::register_components(&mut registry);

            let themes = ThemeSet::default();
            let ctx = egui::Context::default();
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

        fn spawn(&mut self, name: &str) -> orrin_ecs::Entity {
            let entity = self.world.spawn();
            self.world.insert(entity, Name(name.to_owned()));
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

    /// A panel that sizes its content from its own width grows by the overflow
    /// every frame: the content's minimum width becomes the panel's new width,
    /// which becomes a wider content minimum. Nothing shows on frame one.
    #[test]
    fn a_side_panel_does_not_widen_itself() {
        let mut settled = Harness::new(egui::vec2(800.0, 600.0));
        settled.frames(2);
        let mut later = Harness::new(egui::vec2(800.0, 600.0));
        later.frames(120);

        for panel in ["hierarchy", "inspector"] {
            assert_eq!(settled.panel(panel).width(), later.panel(panel).width());
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
