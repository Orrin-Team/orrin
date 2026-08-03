//! The top bar. Mounted before the side panels so it claims the full window
//! width, and the only surface here that is not bound to a selection or a
//! resource — it is where the session itself is presented.

use crate::editor::state::EditorState;
use crate::editor::theme::ThemeSet;

const QUICK_ACCESS_HEIGHT: f32 = 30.0;

pub fn show(ctx: &egui::Context, state: &mut EditorState, themes: &ThemeSet) {
    egui::TopBottomPanel::top("quick_access")
        .exact_height(QUICK_ACCESS_HEIGHT)
        .show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.label(egui::RichText::new(&state.project_name).weak().monospace());
                ui.separator();

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    theme_picker(ui, state, themes);
                });
            });
        });
}

fn theme_picker(ui: &mut egui::Ui, state: &mut EditorState, themes: &ThemeSet) {
    let active = themes.active().name.clone();
    egui::ComboBox::from_id_salt("theme")
        .selected_text(&active)
        .width(110.0)
        .show_ui(ui, |ui| {
            for name in themes.names() {
                if ui.selectable_label(name == active, name).clicked() {
                    state.request_theme(name);
                }
            }
        });
}
