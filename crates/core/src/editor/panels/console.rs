//! The log: every engine and script message, newest at the bottom.
//!
//! A row is a level icon, the frame it was logged on, where it came from, and
//! the message — four columns, so a scan down any one of them works. The level
//! is the icon and the colour, and is not spelled out as `[WARN]` as well: on
//! every line, that is the same fact three times.

use orrin_ecs::World;

use crate::editor::state::EditorState;
use crate::editor::widgets::{self, Tool};
use crate::editor::{icons, theme};
use crate::scene::{LogBuffer, LogEntry, LogLevel};

fn color(level: LogLevel) -> egui::Color32 {
    match level {
        LogLevel::Trace => theme::LOG_TRACE,
        LogLevel::Debug => theme::LOG_DEBUG,
        LogLevel::Info => theme::LOG_INFO,
        LogLevel::Warning => theme::LOG_WARN,
        LogLevel::Error => theme::LOG_ERROR,
    }
}

fn icon(level: LogLevel) -> icons::Icon {
    match level {
        LogLevel::Trace | LogLevel::Debug => icons::terminal(),
        LogLevel::Info => icons::info(),
        LogLevel::Warning => icons::warning(),
        LogLevel::Error => icons::circle_x(),
    }
}

const ROW: f32 = 20.0;
const FRAME_COLUMN: f32 = 52.0;

pub fn body(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState) {
    // Deferred so the immutable borrow taken to render the list is released
    // before the (mutable) clear runs.
    let mut clear = false;
    {
        let Some(log) = world.get_resource::<LogBuffer>() else {
            ui.weak("No log buffer.");
            return;
        };

        let mut counts = [0usize; LogLevel::ALL.len()];
        for entry in log.iter() {
            counts[entry.level as usize] += 1;
        }

        ui.horizontal(|ui| {
            let weak = ui.visuals().weak_text_color();
            clear = icons::button(ui, icons::trash(), icons::ROW - 2.0, weak, false)
                .on_hover_text("Clear the log")
                .clicked();
            let toggles = 5.0 * 52.0;
            let width = (ui.available_width() - toggles).clamp(80.0, 320.0);
            ui.allocate_ui(egui::vec2(width, ui.spacing().interact_size.y), |ui| {
                widgets::search(ui, &mut state.console_query, "Filter log");
            });

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                for level in LogLevel::ALL.into_iter().rev() {
                    let shown = state.console_levels.shows(level);
                    // On, not Selected: all five are on by default, and five
                    // accent-tinted buttons would read as five chosen modes.
                    let tool = if shown { Tool::On } else { Tool::Off };
                    let count = super::figures(counts[level as usize].to_string());
                    let hint = format!("Show {} messages", level.tag().to_lowercase());
                    let tint = Some(color(level));
                    if widgets::tool_button(ui, icon(level), tint, Some(count), tool, &hint)
                        .clicked()
                    {
                        state.console_levels.set(level, !shown);
                    }
                }
            });
        });
        ui.add_space(2.0);

        let needle = state.console_query.trim().to_lowercase();
        let rows: Vec<&LogEntry> = log
            .iter()
            .filter(|entry| state.console_levels.shows(entry.level))
            .filter(|entry| needle.is_empty() || entry.text().to_lowercase().contains(&needle))
            .collect();

        let band = theme::ROW_BAND;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show_rows(ui, ROW, rows.len(), |ui, range| {
                ui.spacing_mut().item_spacing.y = 0.0;
                for index in range {
                    let entry = rows[index];
                    let (rect, response) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width(), ROW),
                        egui::Sense::hover(),
                    );
                    if index % 2 == 1 {
                        ui.painter().rect_filled(rect, 0.0, band);
                    }
                    let glyph = egui::Rect::from_center_size(
                        egui::pos2(rect.left() + 9.0, rect.center().y),
                        egui::Vec2::splat(12.0),
                    );
                    icons::paint(ui, icon(entry.level), glyph, color(entry.level));

                    let mono = egui::TextStyle::Monospace.resolve(ui.style());
                    let weak = ui.visuals().weak_text_color();
                    ui.painter().text(
                        egui::pos2(glyph.right() + 8.0, rect.center().y),
                        egui::Align2::LEFT_CENTER,
                        format!("#{}", entry.frame),
                        mono.clone(),
                        weak,
                    );
                    let text = egui::Rect::from_min_max(
                        egui::pos2(glyph.right() + 8.0 + FRAME_COLUMN, rect.top()),
                        rect.max,
                    );
                    // One line per row: a message's first line, cut to the
                    // row's width, with the whole of it on hover. A validation
                    // error or a dumped entity runs to several lines, and
                    // painted whole it ran down over the rows beneath.
                    let full = entry.text();
                    let first = full.lines().next().unwrap_or_default();
                    let more = full.len() > first.len();
                    let shown = if more {
                        format!("{first} …")
                    } else {
                        first.to_owned()
                    };
                    let galley = egui::WidgetText::from(egui::RichText::new(shown).font(mono))
                        .into_galley(
                            ui,
                            Some(egui::TextWrapMode::Truncate),
                            text.width(),
                            egui::TextStyle::Monospace,
                        );
                    let cut = more || galley.elided;
                    ui.painter().galley(
                        egui::pos2(text.left(), rect.center().y - galley.size().y / 2.0),
                        galley,
                        color(entry.level),
                    );
                    if cut {
                        response.on_hover_text(super::figures(full));
                    }
                }
            });
        if rows.is_empty() {
            ui.weak(if log.is_empty() {
                "Nothing logged yet."
            } else {
                "No message matches."
            });
        }
    }

    if clear && let Some(mut log) = world.get_resource_mut::<LogBuffer>() {
        log.clear();
    }
}
