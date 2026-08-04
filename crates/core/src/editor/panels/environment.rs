use orrin_ecs::World;

use crate::gfx::shadows::MAX_CASCADES;

use super::{color_row, vec3_row};
use crate::scene::{
    AmbientLight, Camera, EnvironmentSettings, FogSettings, HdrSettings, ShadowSettings,
    SsaoSettings,
};

type Column = fn(&mut egui::Ui, &World);

const COLUMNS: [Column; 4] = [ssao_column, shadow_column, lighting_column, camera_column];

/// A slider plus the label that sits after it. Narrower than this and a column's
/// contents run into its neighbour instead of wrapping — `Ui::columns` divides
/// the width but does nothing to make the contents fit it.
///
/// This was once a full-window strip along the bottom, where four cramped
/// columns were the intended look and folding was only a guard against a
/// negative width. Docked, the same panel is a 300px tab, and four columns there
/// is four overlapping ones. The number is now what it says it is.
const MIN_COLUMN_WIDTH: f32 = 190.0;

/// How many columns `available` can carry. Never zero: `Ui::columns` divides by
/// the count, and it asserts on the negative width a shortfall would produce.
fn column_count(available: f32) -> usize {
    ((available / MIN_COLUMN_WIDTH) as usize).clamp(1, COLUMNS.len())
}

pub fn body(ui: &mut egui::Ui, world: &mut World) {
    // A docked tab is as wide as its split, which can be nothing at all.
    let available = ui.available_width();
    if available > 0.0 {
        let columns = column_count(available);
        ui.columns(columns, |cols| {
            let mut previous = usize::MAX;
            for (index, column) in COLUMNS.iter().enumerate() {
                let target = index * columns / COLUMNS.len();
                if target == previous {
                    cols[target].add_space(10.0);
                    cols[target].separator();
                }
                column(&mut cols[target], world);
                previous = target;
            }
        });
    }
    ui.add_space(4.0);
}

fn ssao_column(ui: &mut egui::Ui, world: &World) {
    ui.strong("SSAO");
    let mut s = world.resource_mut::<SsaoSettings>();
    ui.checkbox(&mut s.enabled, "Enabled");
    ui.add(egui::Slider::new(&mut s.radius, 0.0..=4.0).text("Radius"));
    ui.add(egui::Slider::new(&mut s.bias, 0.0..=0.1).text("Bias"));
    ui.add(egui::Slider::new(&mut s.power, 0.1..=4.0).text("Power"));
}

fn shadow_column(ui: &mut egui::Ui, world: &World) {
    ui.strong("Shadows");
    let mut s = world.resource_mut::<ShadowSettings>();
    ui.checkbox(&mut s.enabled, "Enabled");
    // Cascade count and resolution are frame *structure*: changing either
    // recompiles the graph and reallocates the maps, so they are steppers
    // rather than sliders — a drag would do that once per frame.
    ui.horizontal(|ui| {
        ui.label("Cascades");
        ui.add(egui::DragValue::new(&mut s.cascade_count).range(1..=MAX_CASCADES));
    });
    ui.horizontal(|ui| {
        ui.label("Resolution");
        egui::ComboBox::from_id_salt("shadow_resolution")
            .selected_text(format!("{}", s.resolution))
            .show_ui(ui, |ui| {
                for size in [512u32, 1024, 2048, 4096] {
                    ui.selectable_value(&mut s.resolution, size, format!("{size}"));
                }
            });
    });
    ui.add(
        egui::Slider::new(&mut s.max_distance, 10.0..=500.0)
            .logarithmic(true)
            .text("Distance"),
    );
    ui.add(egui::Slider::new(&mut s.lambda, 0.0..=1.0).text("Split blend"));
    ui.add(egui::Slider::new(&mut s.pullback, 0.0..=200.0).text("Pullback"));
    ui.add(egui::Slider::new(&mut s.constant_bias, 0.0..=8.0).text("Bias"));
    ui.add(egui::Slider::new(&mut s.slope_bias, 0.0..=8.0).text("Slope bias"));
    ui.add(egui::Slider::new(&mut s.strength, 0.0..=1.0).text("Strength"));
    ui.checkbox(&mut s.debug_cascades, "Tint cascades");
}

fn lighting_column(ui: &mut egui::Ui, world: &World) {
    ui.strong("Tonemap");
    {
        let mut hdr = world.resource_mut::<HdrSettings>();
        ui.add(egui::Slider::new(&mut hdr.exposure, 0.1..=5.0).text("Exposure"));
    }
    ui.add_space(6.0);
    ui.strong("Ambient");
    {
        let mut ambient = world.resource_mut::<AmbientLight>();
        color_row(ui, "Color", &mut ambient.color);
        ui.add(egui::Slider::new(&mut ambient.intensity, 0.0..=2.0).text("Intensity"));
    }
    ui.add_space(6.0);
    ui.strong("Sky");
    {
        let mut env = world.resource_mut::<EnvironmentSettings>();
        ui.checkbox(&mut env.show_skybox, "Show skybox");
        ui.add(egui::Slider::new(&mut env.intensity, 0.0..=4.0).text("Intensity"));
        // Rotates the sampling direction, so it needs no rebake and can be
        // dragged.
        ui.add(egui::Slider::new(&mut env.yaw, -180.0..=180.0).text("Rotation"));
    }
    ui.add_space(6.0);
    ui.strong("Fog");
    let mut fog = world.resource_mut::<FogSettings>();
    color_row(ui, "Color", &mut fog.color);
    ui.add(
        egui::Slider::new(&mut fog.density, 0.0..=0.1)
            .logarithmic(true)
            .text("Density"),
    );
    ui.add(egui::Slider::new(&mut fog.height_falloff, 0.0..=1.0).text("Falloff"));
    ui.add(egui::Slider::new(&mut fog.height, -20.0..=20.0).text("Height"));
}

fn camera_column(ui: &mut egui::Ui, world: &World) {
    ui.strong("Camera");
    let mut cam = world.resource_mut::<Camera>();
    vec3_row(ui, "Position", &mut cam.position, 0.1);
    vec3_row(ui, "Target", &mut cam.target, 0.1);

    let mut fov = cam.fov_y.to_degrees();
    if ui
        .add(egui::Slider::new(&mut fov, 20.0..=110.0).text("FOV"))
        .changed()
    {
        cam.fov_y = fov.to_radians();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The width this panel is docked at. One column, because four at 75px each
    /// overlap into an unreadable mess — which is exactly what shipped when this
    /// threshold was still tuned for a full-window strip.
    #[test]
    fn the_docked_width_folds_to_one_column() {
        assert_eq!(column_count(300.0), 1);
    }

    /// Widen it — float it, or drag the split out — and the columns come back.
    #[test]
    fn a_wide_tool_spreads_back_out() {
        assert_eq!(column_count(400.0), 2);
        assert_eq!(column_count(600.0), 3);
        assert_eq!(column_count(800.0), 4);
        assert_eq!(column_count(2000.0), 4);
    }

    #[test]
    fn a_sliver_still_yields_a_usable_count() {
        assert_eq!(column_count(20.0), 1);
        assert_eq!(column_count(0.0), 1);
    }
}
