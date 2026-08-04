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

    /// Lay the whole editor out at `size`, `frames` times over one context.
    ///
    /// Headless: egui needs no GPU to place widgets, and placement is all that
    /// is under test. Repeating on one context is the point — a panel whose
    /// content width is derived from its own width does not misbehave on the
    /// first frame, it creeps, and only a later frame can see that.
    fn lay_out(size: egui::Vec2, frames: usize) -> egui::Context {
        lay_out_scene(size, frames, |_| {})
    }

    fn lay_out_scene(
        size: egui::Vec2,
        frames: usize,
        populate: impl FnOnce(&mut World),
    ) -> egui::Context {
        let mut world = World::default();
        crate::App::install_default_resources(&mut world);
        populate(&mut world);
        let mut registry = Registry::new();
        crate::scene::register_components(&mut registry);

        let mut state = EditorState::default();
        let themes = ThemeSet::default();
        let ctx = egui::Context::default();
        theme::apply(&ctx, themes.active());

        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
            ..Default::default()
        };
        for _ in 0..frames {
            let _ = ctx.run(input.clone(), |ctx| {
                draw(ctx, &mut world, &mut state, &registry, &themes);
            });
        }
        ctx
    }

    /// The width egui recorded for a panel on the last frame it drew.
    fn panel_width(ctx: &egui::Context, id: &str) -> f32 {
        egui::containers::panel::PanelState::load(ctx, egui::Id::new(id))
            .expect("panel was drawn")
            .rect
            .width()
    }

    /// A panel that sizes its content from its own width grows by the overflow
    /// every frame: the content's minimum width becomes the panel's new width,
    /// which becomes a wider content minimum. Nothing shows on frame one, so
    /// the assertion has to compare a settled layout against a much later one.
    #[test]
    fn a_side_panel_does_not_widen_itself() {
        let settled = lay_out(egui::vec2(800.0, 600.0), 2);
        let later = lay_out(egui::vec2(800.0, 600.0), 120);
        assert_eq!(
            panel_width(&settled, "hierarchy"),
            panel_width(&later, "hierarchy")
        );
        assert_eq!(
            panel_width(&settled, "inspector"),
            panel_width(&later, "inspector")
        );
    }

    /// Not "the editor looks cramped": once the side panels are wider than the
    /// window, the Environment panel between them is handed a negative width,
    /// and egui asserts inside `columns` rather than clamping.
    #[test]
    fn a_window_narrower_than_its_side_panels_still_lays_out() {
        lay_out(egui::vec2(320.0, 600.0), 4);
    }

    #[test]
    fn a_window_too_narrow_for_any_panel_still_lays_out() {
        lay_out(egui::vec2(120.0, 400.0), 4);
    }

    /// The realistic route to the failure above: nobody resizes the window down
    /// to 320px, but a scene does contain an entity with a long name, and a row
    /// is as wide as the name in it.
    #[test]
    fn a_long_entity_name_cannot_push_a_side_panel_open() {
        let ctx = lay_out_scene(egui::vec2(800.0, 600.0), 4, |world| {
            let entity = world.spawn();
            world.insert(entity, crate::scene::Name("Spawn point ".repeat(40)));
        });
        assert!(panel_width(&ctx, "hierarchy") <= *WIDTH_RANGE.end());
    }
}
