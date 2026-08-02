use orrin_ecs::World;

use crate::gfx::shadows::MAX_CASCADES;

use super::{color_row, vec3_row};
use crate::scene::{AmbientLight, Camera, FogSettings, HdrSettings, ShadowSettings, SsaoSettings};

pub fn show(ctx: &egui::Context, world: &mut World) {
    egui::TopBottomPanel::bottom("environment")
        .resizable(false)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.heading("Environment");
            ui.separator();

            ui.columns(4, |cols| {
                ssao_column(&mut cols[0], world);
                shadow_column(&mut cols[1], world);
                lighting_column(&mut cols[2], world);
                camera_column(&mut cols[3], world);
            });
            ui.add_space(4.0);
        });
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
