//! The log: every engine and script message, newest at the bottom.
//!
//! A row is a level icon, where it came from, and the message — three
//! columns, so a scan down any one of them works. The level is the icon and
//! the colour, and is not spelled out as `[WARN]` as well: on every line, that
//! is the same fact three times. The frame it was logged on is not a column:
//! a number on every line that is read for one line in a hundred, it is in
//! the detail pane and what is copied, and on hover over a row whose message
//! was cut, above the whole of it. A row that fits has no hover text, so the
//! pointer crossing the list raises nothing.
//!
//! Where a script's line came from is a link to it, opened in the user's own
//! editor by `editor::source`, when it names exactly one file under the scripts
//! directory: the `source:line` prefix is underlined at rest, and a click on it
//! or a double-click anywhere on the row opens the file at the line. An engine
//! module and a script look alike until the lookup tells them apart, so only
//! the rows on screen are looked up, each against `ScriptFiles`' index by name,
//! which walks the directory at most once per its rescan interval and never for
//! a module path: a long log costs what its visible rows cost.
//!
//! Rows are selected as a list's are — click, Ctrl+click, Shift+click — and
//! copied as text, each message whole under a line naming its level, frame and
//! source. The Copy key is the console's while the pointer is over it or it
//! took the last click, and only then, so it never copies the scene's
//! selection as well. One selected message that does not fit its row is shown
//! whole in a pane below the list rather than by growing the row: the list is
//! drawn with `show_rows` at one fixed height, which is what keeps it cheap.
//!
//! What is logged while the console is out of sight is counted for its tab
//! and the status bar from the log's running tally by level
//! ([`LogBuffer::logged`]): a subtraction a frame however long the log, and an
//! entry the ring drops before it is seen still counts.

use std::collections::BTreeSet;
use std::time::Instant;

use orrin_ecs::World;

use crate::editor::commands::{self, Command};
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
/// Where a row's text starts: past the level icon.
const TEXT_LEFT: f32 = 15.0 + 8.0;
/// The width a collapsed run's count takes from the end of its row.
const COUNT_COLUMN: f32 = 44.0;
const DETAIL_HEIGHT: f32 = 90.0;

/// A row of the list: one entry, or with Collapse on, a run of identical
/// ones under the first one's number.
struct Row<'a> {
    number: u64,
    entry: &'a LogEntry,
    count: u32,
}

fn repeats(a: &LogEntry, b: &LogEntry) -> bool {
    a.level == b.level && a.line == b.line && a.source == b.source && a.message == b.message
}

/// How many characters the `source:line` a row leads with takes.
fn prefix_chars(entry: &LogEntry) -> usize {
    let source = entry.source.chars().count();
    match entry.line {
        0 => source,
        line => source + 1 + line.ilog10() as usize + 1,
    }
}

/// One message as pasted text: what it is and where it came from on a line of
/// its own, then all of it.
fn block(row: &Row) -> String {
    let entry = row.entry;
    let mut text = format!("{} #{}", entry.level.tag(), entry.frame);
    if !entry.source.is_empty() {
        text.push(' ');
        text.push_str(&entry.source);
        if entry.line > 0 {
            text.push_str(&format!(":{}", entry.line));
        }
    }
    if row.count > 1 {
        text.push_str(&format!(" ×{}", row.count));
    }
    text.push('\n');
    text.push_str(&entry.message);
    text
}

fn copy<'a, 'b: 'a>(ctx: &egui::Context, rows: impl Iterator<Item = &'a Row<'b>>) {
    let text = rows.map(block).collect::<Vec<_>>().join("\n\n");
    if !text.is_empty() {
        ctx.copy_text(text);
    }
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
        state.console.drawn = true;
        state.console.hovered = ui.rect_contains_pointer(ui.max_rect());
        if state.console.hovered && ui.input(|input| input.pointer.any_pressed()) {
            state.console.pressed_in = true;
        }

        let mut counts = [0usize; LogLevel::ALL.len()];
        for entry in log.iter() {
            counts[entry.level as usize] += 1;
        }

        ui.horizontal(|ui| {
            let weak = ui.visuals().weak_text_color();
            clear = icons::button(ui, icons::trash(), icons::ROW - 2.0, weak, false)
                .on_hover_text("Clear the log")
                .clicked();
            let toggles = 5.0 * 52.0 + 30.0 + 76.0 + 6.0;
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
                ui.add_space(6.0);
                let options = &mut state.console.options;
                toggle(
                    ui,
                    icons::layers(),
                    None,
                    &mut options.collapse,
                    "Collapse a run of the same message into one row",
                );
                // Not the Play glyph, which is the transport's alone.
                toggle(
                    ui,
                    icons::trash(),
                    Some("on Play"),
                    &mut options.clear_on_play,
                    "Clear the log when Play starts",
                );
            });
        });
        ui.add_space(2.0);

        let needle = state.console_query.trim().to_lowercase();
        let collapse = state.console.options.collapse;
        let mut rows: Vec<Row> = Vec::new();
        for (number, entry) in log.since(0) {
            if !state.console_levels.shows(entry.level)
                || !(needle.is_empty() || entry.text().to_lowercase().contains(&needle))
            {
                continue;
            }
            if collapse
                && let Some(last) = rows.last_mut()
                && repeats(last.entry, entry)
            {
                last.count += 1;
                continue;
            }
            rows.push(Row {
                number,
                entry,
                count: 1,
            });
        }

        let ctx = ui.ctx().clone();
        // A field's own copy, and a selection made in the detail pane's text,
        // come first.
        let copy_key = state.console.claims_keys
            && !ctx.wants_keyboard_input()
            && !egui::text_selection::LabelSelectionState::load(&ctx).has_selection()
            && ctx.input_mut(|input| state.settings.keys.take(input, Command::Copy));
        if copy_key {
            let selected = &state.console.selected;
            copy(
                &ctx,
                rows.iter().filter(|row| selected.contains(&row.number)),
            );
        }

        let only = match state.console.selected.len() {
            1 => state.console.selected.first().copied(),
            _ => None,
        };
        let width = ui.available_width();
        let detail = only
            .and_then(|number| rows.iter().find(|row| row.number == number))
            .filter(|row| overflows(ui, row, width));
        if let Some(row) = detail {
            let full = row.entry.text();
            let colour = color(row.entry.level);
            let tallest = (ui.available_height() * 0.7).max(ROW * 2.0);
            egui::TopBottomPanel::bottom(ui.id().with("console detail"))
                .resizable(true)
                .default_height(DETAIL_HEIGHT.min(tallest))
                .height_range(ROW * 2.0..=tallest)
                .frame(egui::Frame::new().inner_margin(egui::Margin {
                    top: 6,
                    ..egui::Margin::ZERO
                }))
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("console detail text")
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.label(frame_line(row.entry));
                            ui.add(
                                egui::Label::new(super::figures(full).color(colour))
                                    .wrap()
                                    .selectable(true),
                            );
                        });
                });
        }

        let now = Instant::now();
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show_rows(ui, ROW, rows.len(), |ui, range| {
                        ui.spacing_mut().item_spacing.y = 0.0;
                        for index in range {
                            row(ui, state, &rows, index, now);
                        }
                    });
                if rows.is_empty() {
                    ui.weak(if log.is_empty() {
                        "Nothing logged yet."
                    } else {
                        "No message matches."
                    });
                }
            });
    }

    if clear && let Some(mut log) = world.get_resource_mut::<LogBuffer>() {
        log.clear();
        state.console.forget(Some(&log));
    }
}

fn toggle(ui: &mut egui::Ui, icon: icons::Icon, word: Option<&str>, on: &mut bool, hint: &str) {
    let tool = if *on { Tool::On } else { Tool::Off };
    let word = word.map(egui::RichText::new);
    if widgets::tool_button(ui, icon, None, word, tool, hint).clicked() {
        *on = !*on;
    }
}

/// Whether `row`'s message is more than its row, `width` wide, can show.
fn overflows(ui: &egui::Ui, row: &Row, width: f32) -> bool {
    let full = row.entry.text();
    if full.contains('\n') {
        return true;
    }
    let room = width - TEXT_LEFT - if row.count > 1 { COUNT_COLUMN } else { 0.0 };
    let mono = egui::TextStyle::Monospace.resolve(ui.style());
    let wide = ui.fonts(|fonts| {
        fonts
            .layout_no_wrap(full, mono, egui::Color32::PLACEHOLDER)
            .size()
            .x
    });
    wide > room
}

fn row(ui: &mut egui::Ui, state: &mut EditorState, rows: &[Row], index: usize, now: Instant) {
    let row = &rows[index];
    let entry = row.entry;
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW), egui::Sense::click());
    let selected = state.console.selected.contains(&row.number);
    if selected {
        ui.painter()
            .rect_filled(rect, 0.0, ui.visuals().selection.bg_fill);
    } else if index % 2 == 1 {
        ui.painter().rect_filled(rect, 0.0, theme::ROW_BAND);
    }
    let glyph = egui::Rect::from_center_size(
        egui::pos2(rect.left() + 9.0, rect.center().y),
        egui::Vec2::splat(12.0),
    );
    icons::paint(ui, icon(entry.level), glyph, color(entry.level));

    let mono = egui::TextStyle::Monospace.resolve(ui.style());
    let weak = ui.visuals().weak_text_color();
    let mut text =
        egui::Rect::from_min_max(egui::pos2(rect.left() + TEXT_LEFT, rect.top()), rect.max);
    if row.count > 1 {
        text.max.x = rect.right() - COUNT_COLUMN;
        ui.painter().text(
            egui::pos2(rect.right() - 6.0, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            format!("×{}", row.count),
            mono.clone(),
            weak,
        );
    }
    // One line per row: a message's first line, cut to the row's width, with
    // the whole of it on hover and, selected, below the list. A validation
    // error or a dumped entity runs to several lines, and painted whole it ran
    // down over the rows beneath.
    let full = entry.text();
    let first = full.lines().next().unwrap_or_default();
    let more = full.len() > first.len();
    let shown = if more {
        format!("{first} …")
    } else {
        first.to_owned()
    };
    let galley = egui::WidgetText::from(egui::RichText::new(shown).font(mono)).into_galley(
        ui,
        Some(egui::TextWrapMode::Truncate),
        text.width(),
        egui::TextStyle::Monospace,
    );
    let cut = more || galley.elided;
    let top = rect.center().y - galley.size().y / 2.0;
    let baseline = top + galley.size().y;
    let linked =
        !entry.source.is_empty() && state.script_files.resolve(&entry.source, now).is_some();
    let link = linked.then(|| {
        let end = galley
            .pos_from_ccursor(egui::text::CCursor::new(prefix_chars(entry)))
            .min
            .x;
        egui::Rect::from_min_max(
            text.min,
            egui::pos2((text.left() + end).min(text.right()), text.bottom()),
        )
    });
    ui.painter()
        .galley(egui::pos2(text.left(), top), galley, color(entry.level));

    let mut open = response.double_clicked() && linked;
    let mut on_link = false;
    if let Some(link) = link {
        on_link = ui.rect_contains_pointer(link);
        let strength = if on_link { 1.0 } else { 0.45 };
        ui.painter().hline(
            link.x_range(),
            baseline,
            egui::Stroke::new(1.0_f32, color(entry.level).gamma_multiply(strength)),
        );
        // Only the link under the pointer is a widget, so it can have a
        // fixed id.
        if on_link && let Some(file) = state.script_files.find(&entry.source, now) {
            let name = file.file_name().unwrap_or_default().to_string_lossy();
            let hint = match entry.line {
                0 => format!("Open {name}"),
                line => format!("Open {name} at line {line}"),
            };
            let clicked = ui
                .interact(link, ui.id().with("script link"), egui::Sense::click())
                .on_hover_cursor(egui::CursorIcon::PointingHand)
                .on_hover_text(hint);
            // The second click of a double-click has opened it once already.
            open |= clicked.clicked() && !clicked.double_clicked();
        }
    }
    if open && let Some(file) = state.script_files.find(&entry.source, now) {
        state.request_open_source(file, entry.line);
    }

    if response.clicked() {
        let modifiers = ui.input(|input| input.modifiers);
        state.console.click(row.number, modifiers, rows);
    }
    if response.secondary_clicked() && !selected {
        state.console.select_only(row.number);
    }
    response.context_menu(|ui| menu(ui, state, rows));
    if cut && !on_link {
        response.on_hover_ui(|ui| {
            ui.set_max_width(ui.spacing().tooltip_width);
            ui.label(frame_line(entry));
            ui.label(super::figures(full));
        });
    }
}

fn frame_line(entry: &LogEntry) -> egui::RichText {
    super::figures(format!("Frame {}", entry.frame)).weak()
}

/// What a right-click on a row offers.
fn menu(ui: &mut egui::Ui, state: &EditorState, rows: &[Row]) {
    let key = commands::text(ui.ctx(), &state.settings.keys, Command::Copy);
    if ui
        .add(egui::Button::new("Copy").shortcut_text(key))
        .clicked()
    {
        let selected = &state.console.selected;
        copy(
            ui.ctx(),
            rows.iter().filter(|row| selected.contains(&row.number)),
        );
        ui.close_menu();
    }
    if ui.button("Copy all").clicked() {
        copy(ui.ctx(), rows.iter());
        ui.close_menu();
    }
}

/// The console's two view choices, kept with the project's editor preferences.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ConsoleOptions {
    /// Draw a run of the same message as one row with a count.
    pub collapse: bool,
    /// Empty the log when Play starts, so what it shows is that run's.
    pub clear_on_play: bool,
}

/// What the console keeps between frames, beyond the filter on `EditorState`.
#[derive(Default)]
pub struct ConsoleView {
    pub options: ConsoleOptions,
    /// By the log's own entry numbers, which hold still while the ring drops
    /// entries and the filter changes what is shown.
    selected: BTreeSet<u64>,
    /// Where a Shift+click extends from.
    anchor: Option<u64>,
    /// The pointer was over the console when it last drew.
    hovered: bool,
    /// The last press anywhere landed in the console.
    pressed_in: bool,
    /// The console drew since [`begin`](Self::begin) last ran.
    drawn: bool,
    /// The Copy key is the console's this frame; see [`claims_keys`](Self::claims_keys).
    claims_keys: bool,
    unseen: Unseen,
}

/// What was logged while the console was out of sight, as the log's running
/// tallies stood when it was last seen and stand now.
#[derive(Default)]
struct Unseen {
    seen: [u64; 2],
    now: [u64; 2],
}

impl Unseen {
    fn read(log: &LogBuffer) -> [u64; 2] {
        [log.logged(LogLevel::Error), log.logged(LogLevel::Warning)]
    }
}

impl ConsoleView {
    /// Settle what the frame before left, ahead of anything reading keys or
    /// counts: whether the console has the Copy key, and what was logged
    /// since. Shown last frame, the console has seen everything so far.
    pub fn begin(&mut self, ctx: &egui::Context, log: Option<&LogBuffer>) {
        let drawn = std::mem::take(&mut self.drawn);
        self.claims_keys = drawn && (self.hovered || self.pressed_in);
        self.hovered = false;
        if !drawn || ctx.input(|input| input.pointer.any_pressed()) {
            self.pressed_in = false;
        }
        let Some(log) = log else {
            return;
        };
        self.unseen.now = Unseen::read(log);
        if drawn {
            self.unseen.seen = self.unseen.now;
        }
    }

    /// Whether the Copy key is the console's rather than the scene
    /// selection's: the console was drawn last frame, and the pointer was
    /// over it or the last press was in it.
    pub fn claims_keys(&self) -> bool {
        self.claims_keys
    }

    /// Errors and warnings logged while the console was out of sight.
    pub fn unseen(&self) -> (u64, u64) {
        let [errors, warnings] = self.unseen.now;
        let [seen_errors, seen_warnings] = self.unseen.seen;
        (errors - seen_errors, warnings - seen_warnings)
    }

    /// Let go of a log that was just cleared: nothing in it is selected or
    /// unseen any more.
    pub fn forget(&mut self, log: Option<&LogBuffer>) {
        self.selected.clear();
        self.anchor = None;
        if let Some(log) = log {
            self.unseen.now = Unseen::read(log);
            self.unseen.seen = self.unseen.now;
        }
    }

    fn click(&mut self, number: u64, modifiers: egui::Modifiers, rows: &[Row]) {
        let at = |number| rows.iter().position(|row| row.number == number);
        if modifiers.shift
            && let Some((from, to)) = self.anchor.and_then(at).zip(at(number))
        {
            let (low, high) = (from.min(to), from.max(to));
            self.selected = rows[low..=high].iter().map(|row| row.number).collect();
            return;
        }
        if modifiers.command {
            if !self.selected.remove(&number) {
                self.selected.insert(number);
            }
        } else {
            self.selected = BTreeSet::from([number]);
        }
        self.anchor = Some(number);
    }

    fn select_only(&mut self, number: u64) {
        self.selected = BTreeSet::from([number]);
        self.anchor = Some(number);
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::path::PathBuf;

    use super::*;
    use crate::editor::source::ScriptFiles;

    /// The console alone, in a window of its own, over a scripts directory
    /// holding `Spinner.cs`.
    struct Console {
        ctx: egui::Context,
        world: World,
        state: EditorState,
        dir: PathBuf,
        output: Option<egui::FullOutput>,
        copies: Vec<String>,
    }

    fn entry(level: LogLevel, source: &'static str, line: u32, message: &str) -> LogEntry {
        LogEntry {
            level,
            message: message.to_owned(),
            source: Cow::Borrowed(source),
            line,
            frame: 1,
        }
    }

    impl Console {
        fn new(name: &str, entries: &[(&'static str, u32, &str)]) -> Self {
            Self::with(
                name,
                entries
                    .iter()
                    .map(|&(source, line, message)| entry(LogLevel::Info, source, line, message))
                    .collect(),
            )
        }

        fn with(name: &str, entries: Vec<LogEntry>) -> Self {
            let dir =
                std::env::temp_dir().join(format!("orrin-console-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("Spinner.cs"), b"").unwrap();

            let mut log = LogBuffer::default();
            for entry in entries {
                log.push_entry(entry);
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
                copies: Vec::new(),
            }
        }

        fn frame(&mut self, events: Vec<egui::Event>) {
            self.frame_with(events, egui::Modifiers::NONE);
        }

        /// One frame, begun as `panels::draw` begins one.
        fn frame_with(&mut self, events: Vec<egui::Event>, modifiers: egui::Modifiers) {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, 300.0),
                )),
                events,
                modifiers,
                ..Default::default()
            };
            let Self {
                ctx, world, state, ..
            } = self;
            let output = ctx.run(input, |ctx| {
                state
                    .console
                    .begin(ctx, world.get_resource::<LogBuffer>().as_deref());
                egui::CentralPanel::default().show(ctx, |ui| body(ui, world, state));
            });
            for command in &output.platform_output.commands {
                if let egui::OutputCommand::CopyText(text) = command {
                    self.copies.push(text.clone());
                }
            }
            self.output = Some(output);
        }

        /// Every painted string.
        fn painted(&self) -> Vec<String> {
            fn collect(shape: &egui::Shape, out: &mut Vec<String>) {
                match shape {
                    egui::Shape::Text(text) => out.push(text.galley.text().to_owned()),
                    egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect(s, out)),
                    _ => {}
                }
            }
            let mut out = Vec::new();
            for clipped in &self.output.as_ref().expect("a frame has run").shapes {
                collect(&clipped.shape, &mut out);
            }
            out
        }

        /// What was last put on the clipboard, by any frame so far.
        fn copied(&self) -> Option<String> {
            self.copies.last().cloned()
        }

        fn press(
            &mut self,
            at: egui::Pos2,
            button: egui::PointerButton,
            modifiers: egui::Modifiers,
        ) {
            let event = |pressed| egui::Event::PointerButton {
                pos: at,
                button,
                pressed,
                modifiers,
            };
            self.arrive(at);
            self.frame_with(vec![event(true)], modifiers);
            self.frame_with(vec![event(false)], modifiers);
            self.frame(Vec::new());
        }

        fn click_with(&mut self, at: egui::Pos2, modifiers: egui::Modifiers) {
            self.press(at, egui::PointerButton::Primary, modifiers);
        }

        fn centre(&self, text: &str) -> egui::Pos2 {
            self.row(text).center()
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
            self.arrive(at);
            self.frame(vec![button(true)]);
            self.frame(vec![button(false)]);
            self.frame(Vec::new());
        }

        /// Bring the pointer to `at` the way a mouse does, moving. One that
        /// appears there has never moved, and egui shows a still pointer's
        /// tooltip at once, over the top row of so short a window.
        fn arrive(&mut self, at: egui::Pos2) {
            self.frame(vec![egui::Event::PointerMoved(at - egui::vec2(4.0, 0.0))]);
            self.frame(vec![egui::Event::PointerMoved(at)]);
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

    /// The whole message, however many lines, under a line saying what it is
    /// and where it came from — the level is a colour on screen and nothing
    /// in a paste.
    #[test]
    fn the_copy_key_puts_the_selected_messages_on_the_clipboard_in_full() {
        let mut console = Console::with(
            "copy",
            vec![
                entry(LogLevel::Error, "Spinner", 14, "first line\nsecond line"),
                entry(LogLevel::Info, "", 0, "beta"),
                entry(LogLevel::Warning, "orrin_core::physics", 0, "gamma"),
            ],
        );
        console.frame(Vec::new());
        console.click(console.centre("Spinner:14: first line …"));
        console.click_with(
            console.centre("orrin_core::physics: gamma"),
            egui::Modifiers::COMMAND,
        );

        console.frame(vec![egui::Event::Copy]);

        assert_eq!(
            console.copied().as_deref(),
            Some(
                "ERROR #1 Spinner:14\nfirst line\nsecond line\n\n\
                 WARN #1 orrin_core::physics\ngamma"
            )
        );
    }

    #[test]
    fn collapsing_folds_a_run_of_repeats_into_one_row_with_a_count() {
        let mut console = Console::with(
            "collapse",
            ["tick", "tick", "tick", "tock", "tick"]
                .into_iter()
                .map(|message| entry(LogLevel::Info, "", 0, message))
                .collect(),
        );
        let ticks = |console: &Console| console.painted().iter().filter(|t| *t == "tick").count();
        console.frame(Vec::new());
        assert_eq!(ticks(&console), 4, "off by default");

        console.state.console.options.collapse = true;
        console.frame(Vec::new());
        assert_eq!(ticks(&console), 2);
        assert!(console.painted().iter().any(|t| t == "×3"));
    }

    /// Counted from the log's own tally, so nothing is counted twice and an
    /// entry the ring dropped before it was seen still counts.
    #[test]
    fn what_is_logged_out_of_sight_is_counted_once_and_forgotten_when_shown() {
        let ctx = egui::Context::default();
        let mut view = ConsoleView::default();
        let mut log = LogBuffer::with_capacity(4);
        log.push(LogLevel::Error, "a".to_owned(), 0);
        log.push(LogLevel::Warning, "b".to_owned(), 0);
        log.push(LogLevel::Info, "c".to_owned(), 0);
        view.begin(&ctx, Some(&log));
        assert_eq!(view.unseen(), (1, 1));

        log.push(LogLevel::Error, "d".to_owned(), 0);
        view.begin(&ctx, Some(&log));
        assert_eq!(view.unseen(), (2, 1));

        for _ in 0..6 {
            log.push(LogLevel::Error, "e".to_owned(), 0);
        }
        view.begin(&ctx, Some(&log));
        assert_eq!(view.unseen(), (8, 1));

        view.drawn = true;
        view.begin(&ctx, Some(&log));
        assert_eq!(view.unseen(), (0, 0));
    }
}
