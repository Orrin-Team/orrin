use orrin_ecs::World;

use crate::scene::{LogBuffer, LogLevel};

pub fn show(ctx: &egui::Context, world: &mut World) {
    // Deferred so the immutable borrow taken to render the list is released
    // before the (mutable) clear runs.
    let mut clear = false;

    egui::Window::new("Console")
        .default_pos(egui::pos2(12.0, 320.0))
        .default_size(egui::vec2(440.0, 200.0))
        .show(ctx, |ui| {
            let Some(log) = world.get_resource::<LogBuffer>() else {
                ui.label("No log buffer.");
                return;
            };

            ui.horizontal(|ui| {
                ui.label(format!("{} messages", log.len()));
                clear = ui.button("Clear").clicked();
            });
            ui.separator();

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for entry in log.iter() {
                        let (color, tag) = match entry.level {
                            LogLevel::Info => (egui::Color32::LIGHT_GRAY, "INFO"),
                            LogLevel::Warning => (egui::Color32::from_rgb(255, 200, 80), "WARN"),
                            LogLevel::Error => (egui::Color32::from_rgb(255, 110, 110), "ERROR"),
                        };
                        ui.colored_label(color, format!("[{}] {}", tag, entry.message));
                    }
                });
        });

    if clear {
        if let Some(mut log) = world.get_resource_mut::<LogBuffer>() {
            log.clear();
        }
    }
}
