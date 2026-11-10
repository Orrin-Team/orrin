//! The log: every engine and script message, newest at the bottom.
//!
//! A row is a level icon, the frame it was logged on, where it came from, and
//! the message — four columns, so a scan down any one of them works. The level
//! is the icon and the colour, and is not spelled out as `[WARN]` as well: on
//! every line, that is the same fact three times.
//!
//! Where a script's line came from is a link to it, opened in the user's own
//! editor by `editor::source`, when it names exactly one file under the scripts
//! directory. Nothing marks it until it is hovered: an engine module and a
//! script look alike at rest, and only the lookup can tell them apart.

use std::path::PathBuf;
use std::time::Instant;

use orrin_ecs::World;

use crate::editor::source::ScriptFiles;
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

/// The `source:line` a row leads with, while the pointer is on it and it names
/// one script file to open.
///
/// Only the hovered row is measured and looked up, so a console of a thousand
/// lines costs nothing more than it did; one link is live at a time, which is
/// why it can have a fixed id.
fn script_link(
    ui: &egui::Ui,
    files: &mut ScriptFiles,
    entry: &LogEntry,
    text: egui::Rect,
    font: &egui::FontId,
) -> Option<(egui::Rect, PathBuf)> {
    if entry.source.is_empty() || !ui.rect_contains_pointer(text) {
        return None;
    }
    let named = match entry.line {
        0 => entry.source.to_string(),
        line => format!("{}:{line}", entry.source),
    };
    let width = ui.fonts(|fonts| {
        fonts
            .layout_no_wrap(named, font.clone(), egui::Color32::PLACEHOLDER)
            .size()
            .x
    });
    let link = egui::Rect::from_min_max(
        text.min,
        egui::pos2((text.left() + width).min(text.right()), text.bottom()),
    );
    if !ui.rect_contains_pointer(link) {
        return None;
    }
    Some((link, files.find(&entry.source, Instant::now())?))
}

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
                    let link = script_link(ui, &mut state.script_files, entry, text, &mono);
                    let galley = egui::WidgetText::from(egui::RichText::new(shown).font(mono))
                        .into_galley(
                            ui,
                            Some(egui::TextWrapMode::Truncate),
                            text.width(),
                            egui::TextStyle::Monospace,
                        );
                    let cut = more || galley.elided;
                    let top = rect.center().y - galley.size().y / 2.0;
                    let baseline = top + galley.size().y;
                    ui.painter()
                        .galley(egui::pos2(text.left(), top), galley, color(entry.level));
                    if let Some((link, file)) = link {
                        ui.painter().hline(
                            link.x_range(),
                            baseline,
                            egui::Stroke::new(1.0_f32, color(entry.level)),
                        );
                        let name = file.file_name().unwrap_or_default().to_string_lossy();
                        let hint = match entry.line {
                            0 => format!("Open {name}"),
                            line => format!("Open {name} at line {line}"),
                        };
                        let clicked = ui
                            .interact(link, ui.id().with("script link"), egui::Sense::click())
                            .on_hover_cursor(egui::CursorIcon::PointingHand)
                            .on_hover_text(hint)
                            .clicked();
                        if clicked {
                            state.request_open_source(file, entry.line);
                        }
                    } else if cut {
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;

    /// The console alone, in a window of its own, over a scripts directory
    /// holding `Spinner.cs`.
    struct Console {
        ctx: egui::Context,
        world: World,
        state: EditorState,
        dir: PathBuf,
        output: Option<egui::FullOutput>,
    }

    impl Console {
        fn new(name: &str, entries: &[(&'static str, u32, &str)]) -> Self {
            let dir =
                std::env::temp_dir().join(format!("orrin-console-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("Spinner.cs"), b"").unwrap();

            let mut log = LogBuffer::default();
            for &(source, line, message) in entries {
                log.push_entry(LogEntry {
                    level: LogLevel::Info,
                    message: message.to_owned(),
                    source: Cow::Borrowed(source),
                    line,
                    frame: 1,
                });
            }
            let mut world = World::default();
            world.insert_resource(log);
            let mut state = EditorState::default();
            state.script_files = ScriptFiles::new(Some(dir.clone()));
            let ctx = egui::Context::default();
            icons::install(&ctx);
            theme::install_fonts(&ctx);
            Self {
                ctx,
                world,
                state,
                dir,
                output: None,
            }
        }

        fn frame(&mut self, events: Vec<egui::Event>) {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, 300.0),
                )),
                events,
                ..Default::default()
            };
            let Self {
                ctx, world, state, ..
            } = self;
            self.output = Some(ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| body(ui, world, state));
            }));
        }

        /// Where the row reading `text` was painted, and how wide.
        fn row(&self, text: &str) -> egui::Rect {
            fn find(shape: &egui::Shape, text: &str) -> Option<egui::Rect> {
                match shape {
                    egui::Shape::Text(shape) if shape.galley.text() == text => {
                        Some(egui::Rect::from_min_size(shape.pos, shape.galley.size()))
                    }
                    egui::Shape::Vec(shapes) => shapes.iter().find_map(|shape| find(shape, text)),
                    _ => None,
                }
            }
            let output = self.output.as_ref().expect("a frame has run");
            output
                .shapes
                .iter()
                .find_map(|clipped| find(&clipped.shape, text))
                .unwrap_or_else(|| panic!("`{text}` was not painted"))
        }

        /// Hover, press and release at `at`, as a mouse click arrives.
        fn click(&mut self, at: egui::Pos2) {
            let button = |pressed| egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            };
            self.frame(vec![egui::Event::PointerMoved(at)]);
            self.frame(vec![button(true)]);
            self.frame(vec![button(false)]);
            self.frame(Vec::new());
        }
    }

    impl Drop for Console {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    const SCRIPT_ROW: &str = "Spinner:14: spinning up, and saying a great deal more about it";

    #[test]
    fn clicking_a_scripts_attribution_asks_to_open_it_at_that_line() {
        let mut console = Console::new(
            "link",
            &[(
                "Spinner",
                14,
                "spinning up, and saying a great deal more about it",
            )],
        );
        console.frame(Vec::new());
        let row = console.row(SCRIPT_ROW);

        console.click(egui::pos2(row.left() + 3.0, row.center().y));

        let file = std::path::absolute(console.dir.join("Spinner.cs")).unwrap();
        assert_eq!(console.state.take_open_source_request(), Some((file, 14)));
    }

    #[test]
    fn clicking_the_message_opens_nothing() {
        let mut console = Console::new(
            "message",
            &[(
                "Spinner",
                14,
                "spinning up, and saying a great deal more about it",
            )],
        );
        console.frame(Vec::new());
        let row = console.row(SCRIPT_ROW);

        console.click(egui::pos2(row.right() - 3.0, row.center().y));

        assert_eq!(console.state.take_open_source_request(), None);
    }

    /// An engine event names the module it came from, and a Behaviour may be
    /// named apart from its file: neither has a file to open.
    #[test]
    fn an_attribution_with_no_script_behind_it_is_not_a_link() {
        let mut console = Console::new(
            "no-link",
            &[
                ("orrin_core::physics", 0, "stepped"),
                ("Missing", 3, "no such file"),
            ],
        );
        console.frame(Vec::new());

        for text in ["orrin_core::physics: stepped", "Missing:3: no such file"] {
            let row = console.row(text);
            console.click(egui::pos2(row.left() + 3.0, row.center().y));
            assert_eq!(console.state.take_open_source_request(), None, "{text}");
        }
    }
}
