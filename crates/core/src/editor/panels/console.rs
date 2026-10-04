use orrin_ecs::World;

use crate::editor::state::EditorState;
use crate::editor::theme;
use crate::scene::{LogBuffer, LogLevel};

fn color(level: LogLevel) -> egui::Color32 {
    match level {
        LogLevel::Trace => theme::LOG_TRACE,
        LogLevel::Debug => theme::LOG_DEBUG,
        LogLevel::Info => theme::LOG_INFO,
        LogLevel::Warning => theme::LOG_WARN,
        LogLevel::Error => theme::LOG_ERROR,
    }
}

pub fn body(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState) {
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
            ui.separator();
            for level in LogLevel::ALL {
                let mut shown = state.console_levels.shows(level);
                let label = egui::RichText::new(level.tag()).color(color(level));
                if ui.toggle_value(&mut shown, label).changed() {
                    state.console_levels.set(level, shown);
                }
            }
        });
        ui.separator();

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for entry in log.iter() {
                    if !state.console_levels.shows(entry.level) {
                        continue;
                    }
                    let line = format!("[{}] {}", entry.level.tag(), entry.text());
                    ui.label(super::figures(line).color(color(entry.level)));
                }
            });
    }

    if clear && let Some(mut log) = world.get_resource_mut::<LogBuffer>() {
        log.clear();
    }
}
