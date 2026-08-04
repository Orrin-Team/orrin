use orrin_ecs::World;

use crate::editor::theme;
use crate::scene::{LogBuffer, LogLevel};

pub fn body(ui: &mut egui::Ui, world: &mut World) {
    // Deferred so the immutable borrow taken to render the list is released
    // before the (mutable) clear runs.
    let mut clear = false;
    {
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
                        LogLevel::Info => (theme::LOG_INFO, "INFO"),
                        LogLevel::Warning => (theme::LOG_WARN, "WARN"),
                        LogLevel::Error => (theme::LOG_ERROR, "ERROR"),
                    };
                    ui.label(super::figures(format!("[{tag}] {}", entry.message)).color(color));
                }
            });
    }

    if clear && let Some(mut log) = world.get_resource_mut::<LogBuffer>() {
        log.clear();
    }
}
