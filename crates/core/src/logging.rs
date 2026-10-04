//! The engine's one log stream.
//!
//! Engine code logs through `tracing`'s macros; scripts log through the debug
//! ABI in `scripting.rs`, which turns each `Debug.Log*` into the same kind of
//! event. [`Stream`] is the subscriber behind both. It fans an event out to up
//! to three sinks: stderr, the editor console, and — in an exported game — a
//! file beside the executable that keeps warnings and errors only.
//!
//! It implements `Subscriber` directly rather than composing
//! `tracing-subscriber` layers. What that crate brings is a registry for
//! tracking spans, and this stream ignores spans: timing scopes belong to
//! `profile.rs`, which is what the HUD reads. A subscriber for events alone is
//! this file, with no dependency the tree does not already carry.
//!
//! The console cannot be written from here. `LogBuffer` is a world resource and
//! an event may come from a pool worker, so events queue in a [`ConsoleQueue`]
//! and the frame loop drains it into the buffer once a frame — which is also
//! where an entry gets its frame number.
//!
//! Two field names are reserved: `script`, and `line` beside it, attribute an
//! event to game code rather than to the module that relayed it. Every other
//! field is appended to the message as `key=value`.
//!
//! Other crates are held to warnings whatever the floor is. winit and calloop
//! narrate their event loops at trace on this same subscriber, and a floor low
//! enough to show the engine's own trace would otherwise drown it in theirs.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use tracing::field::{Field, Visit};
use tracing::{Event, Level, Metadata, Subscriber, span};

use crate::scene::{LogBuffer, LogEntry, LogLevel};

/// What the log file keeps, on top of the stream's own floor.
const FILE_FLOOR: LogLevel = LogLevel::Warning;

/// What every crate but the engine's own is held to.
const FOREIGN_FLOOR: LogLevel = LogLevel::Warning;

/// The console's own capacity: anything queued past it would fall off the far
/// end of the `LogBuffer` on arrival anyway.
const QUEUE_CAPACITY: usize = 1024;

/// Which sinks a [`Stream`] writes, and how quiet an event it lets through.
pub struct LogConfig {
    pub min_level: LogLevel,
    pub stderr: bool,
    /// Queue events for the editor console. Off where nothing drains it.
    pub console: bool,
    /// Warnings and errors are also written here. Truncated at startup, so the
    /// file describes the run that wrote it.
    pub file: Option<PathBuf>,
}

impl LogConfig {
    /// Everything, to stderr and the console.
    pub fn editor() -> Self {
        Self {
            min_level: LogLevel::Trace,
            stderr: true,
            console: true,
            file: None,
        }
    }

    /// An exported game: nothing below info, no console to feed, and a log
    /// file named after `executable` beside it.
    pub fn export(executable: &Path) -> Self {
        Self {
            min_level: LogLevel::Info,
            stderr: true,
            console: false,
            file: Some(executable.with_extension("log")),
        }
    }

    /// The configuration this build runs with: [`export`](Self::export) under
    /// the `export` feature, [`editor`](Self::editor) otherwise. `ORRIN_LOG`
    /// names a different floor either way, though it cannot bring back the
    /// trace and debug events an export build compiled out.
    pub fn for_this_build() -> Self {
        let mut config = if cfg!(feature = "export") {
            match std::env::current_exe() {
                Ok(executable) => Self::export(&executable),
                Err(_) => Self {
                    file: None,
                    ..Self::export(Path::new(""))
                },
            }
        } else {
            Self::editor()
        };
        if let Ok(Some(level)) = floor_named(std::env::var("ORRIN_LOG").ok().as_deref()) {
            config.min_level = level;
        }
        config
    }
}

/// The floor `ORRIN_LOG` asks for, if it is set — or, for a value that names
/// no level, what to say about it.
fn floor_named(raw: Option<&str>) -> Result<Option<LogLevel>, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(None),
        Some(name) => LogLevel::parse(name).map(Some).ok_or_else(|| {
            format!(
                "ORRIN_LOG: unknown level `{name}` (expected trace, debug, info, warn or \
                 error); keeping this build's default"
            )
        }),
    }
}

/// A poisoned lock is taken anyway: the thread that panicked while logging has
/// its own report to make, and losing every later line would hide it.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The editor console's end of the stream. Bounded, oldest dropped first.
#[derive(Clone, Default)]
pub struct ConsoleQueue(Arc<Mutex<VecDeque<LogEntry>>>);

impl ConsoleQueue {
    fn push(&self, entry: LogEntry) {
        let mut queue = lock(&self.0);
        if queue.len() == QUEUE_CAPACITY {
            queue.pop_front();
        }
        queue.push_back(entry);
    }

    /// Move everything queued into `log`, oldest first, stamped with `frame`.
    pub fn drain_into(&self, log: &mut LogBuffer, frame: u64) {
        for mut entry in lock(&self.0).drain(..) {
            entry.frame = frame;
            log.push_entry(entry);
        }
    }
}

/// The subscriber. Built by [`init`]; public so a test can install one for the
/// duration of a closure with `tracing::subscriber::with_default`.
pub struct Stream {
    min_level: LogLevel,
    stderr: bool,
    console: Option<ConsoleQueue>,
    file: Option<Mutex<File>>,
}

impl Stream {
    /// The stream, the queue it feeds, and — when the log file could not be
    /// opened — what to say about it. That failure costs the file and nothing
    /// else, so it is handed back to be logged rather than returned as an error.
    pub fn new(config: LogConfig) -> (Self, ConsoleQueue, Option<String>) {
        let (file, complaint) = match config.file.map(|path| (File::create(&path), path)) {
            None => (None, None),
            Some((Ok(file), _)) => (Some(Mutex::new(file)), None),
            Some((Err(error), path)) => (
                None,
                Some(format!(
                    "cannot write the log file {} ({error}); warnings and errors \
                     reach stderr only",
                    path.display()
                )),
            ),
        };
        let console = ConsoleQueue::default();
        let stream = Self {
            min_level: config.min_level,
            stderr: config.stderr,
            console: config.console.then(|| console.clone()),
            file,
        };
        (stream, console, complaint)
    }

    fn floor_for(&self, target: &str) -> LogLevel {
        if target.starts_with("orrin") {
            self.min_level
        } else {
            self.min_level.max(FOREIGN_FLOOR)
        }
    }
}

fn level_of(metadata: &Metadata<'_>) -> LogLevel {
    match *metadata.level() {
        Level::ERROR => LogLevel::Error,
        Level::WARN => LogLevel::Warning,
        Level::INFO => LogLevel::Info,
        Level::DEBUG => LogLevel::Debug,
        _ => LogLevel::Trace,
    }
}

/// One line as stderr and the log file carry it.
fn format_line(entry: &LogEntry) -> String {
    format!("{:<5} {}\n", entry.level.tag(), entry.text())
}

/// An event's fields, split into the two that attribute it and the rest.
#[derive(Default)]
struct Fields {
    message: String,
    rest: String,
    script: String,
    line: u32,
}

impl Visit for Fields {
    fn record_str(&mut self, field: &Field, value: &str) {
        match field.name() {
            "script" => self.script.push_str(value),
            "message" => self.message.push_str(value),
            name => {
                let _ = write!(self.rest, " {name}={value}");
            }
        }
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == "line" {
            self.line = u32::try_from(value).unwrap_or(0);
        } else {
            self.record_debug(field, &value);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let _ = match field.name() {
            "message" => write!(self.message, "{value:?}"),
            name => write!(self.rest, " {name}={value:?}"),
        };
    }
}

impl Subscriber for Stream {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.is_event() && level_of(metadata) >= self.floor_for(metadata.target())
    }

    fn event(&self, event: &Event<'_>) {
        let metadata = event.metadata();
        let mut fields = Fields::default();
        event.record(&mut fields);
        fields.message.push_str(&fields.rest);
        let entry = LogEntry {
            level: level_of(metadata),
            message: fields.message,
            source: if fields.script.is_empty() {
                Cow::Borrowed(metadata.target())
            } else {
                Cow::Owned(fields.script)
            },
            line: fields.line,
            frame: 0,
        };

        // Write failures are dropped on purpose. There is nowhere left to
        // report one, and a game with no terminal has no stderr to write to.
        let file = self.file.as_ref().filter(|_| entry.level >= FILE_FLOOR);
        if self.stderr || file.is_some() {
            let line = format_line(&entry);
            if self.stderr {
                let _ = std::io::stderr().lock().write_all(line.as_bytes());
            }
            if let Some(file) = file {
                let _ = lock(file).write_all(line.as_bytes());
            }
        }
        if let Some(console) = &self.console {
            console.push(entry);
        }
    }

    // Spans are never enabled, so none of these is reached; the id is only
    // there because the trait has to return one.
    fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
        span::Id::from_u64(1)
    }
    fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
    fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
    fn enter(&self, _: &span::Id) {}
    fn exit(&self, _: &span::Id) {}
}

static CONSOLE: OnceLock<ConsoleQueue> = OnceLock::new();

/// Install the stream as the process's subscriber and hand back the console's
/// end of it. Idempotent: a later call returns the first one's queue and its
/// `config` is ignored.
pub fn init(config: LogConfig) -> ConsoleQueue {
    CONSOLE
        .get_or_init(|| {
            let (stream, console, complaint) = Stream::new(config);
            if let Err(error) = tracing::subscriber::set_global_default(stream) {
                // Straight to stderr: the stream that would carry this is the
                // thing that failed to install.
                eprintln!("orrin: another log subscriber is already installed ({error})");
            }
            if let Some(complaint) = complaint {
                tracing::warn!("{complaint}");
            }
            if let Err(complaint) = floor_named(std::env::var("ORRIN_LOG").ok().as_deref()) {
                tracing::warn!("{complaint}");
            }
            console
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet(config: LogConfig) -> LogConfig {
        LogConfig {
            stderr: false,
            ..config
        }
    }

    /// Run `emit` against a stream built from `config`, and return what
    /// reached the console as `(level, line, frame)`.
    fn console_of(config: LogConfig, emit: impl FnOnce()) -> Vec<(LogLevel, String, u64)> {
        let (stream, console, _) = Stream::new(quiet(config));
        tracing::subscriber::with_default(stream, emit);
        let mut log = LogBuffer::with_capacity(4096);
        console.drain_into(&mut log, 7);
        log.iter()
            .map(|entry| (entry.level, entry.text(), entry.frame))
            .collect()
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("orrin-log-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn an_event_reaches_the_console_naming_its_module_and_frame() {
        let lines = console_of(LogConfig::editor(), || {
            tracing::warn!("no pipeline cache");
        });
        assert_eq!(
            lines,
            [(
                LogLevel::Warning,
                format!("{}: no pipeline cache", module_path!()),
                7
            )]
        );
    }

    #[test]
    fn a_script_field_takes_the_attribution_and_other_fields_follow_the_message() {
        let lines = console_of(LogConfig::editor(), || {
            tracing::info!(script = "Spinner", line = 14u32, "attached");
            tracing::error!(mesh = "cube", count = 3, "unknown mesh");
        });
        assert_eq!(lines[0].1, "Spinner:14: attached");
        assert_eq!(
            lines[1].1,
            format!("{}: unknown mesh mesh=cube count=3", module_path!())
        );
    }

    #[test]
    fn a_floor_is_named_or_the_misspelling_is() {
        assert_eq!(floor_named(None), Ok(None));
        assert_eq!(floor_named(Some("  ")), Ok(None));
        assert_eq!(floor_named(Some("WARN")), Ok(Some(LogLevel::Warning)));
        let complaint = floor_named(Some("loud")).unwrap_err();
        assert!(complaint.contains("`loud`"), "{complaint}");
    }

    #[test]
    fn events_below_the_floor_are_dropped() {
        let config = LogConfig {
            min_level: LogLevel::Info,
            ..LogConfig::editor()
        };
        let lines = console_of(config, || {
            tracing::trace!("every wakeup");
            tracing::debug!("cache hit");
            tracing::info!("project opened");
        });
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(lines[0].0, LogLevel::Info);
    }

    /// The floor the engine's own trace runs at would otherwise let through
    /// every line winit and calloop write about their event loops.
    #[test]
    fn other_crates_are_held_to_warnings() {
        let lines = console_of(LogConfig::editor(), || {
            tracing::trace!(target: "calloop::loop_logic", "dispatching");
            tracing::info!(target: "winit::platform", "wakeup");
            tracing::warn!(target: "winit::platform", "no primary monitor");
            tracing::info!("the engine's own");
        });
        assert_eq!(
            lines.iter().map(|line| line.0).collect::<Vec<_>>(),
            [LogLevel::Warning, LogLevel::Info],
            "{lines:?}"
        );
    }

    #[test]
    fn an_export_keeps_warnings_and_errors_in_a_file_beside_the_executable() {
        let dir = scratch("export");
        let config = LogConfig::export(&dir.join("game.exe"));
        assert_eq!(config.file.as_deref(), Some(dir.join("game.log").as_path()));

        let lines = console_of(config, || {
            tracing::debug!("cache hit");
            tracing::info!("project opened");
            tracing::warn!("texture missing");
            tracing::error!(script = "Player", line = 9u32, "null entity");
        });
        assert!(lines.is_empty(), "an export has no console: {lines:?}");

        let written = std::fs::read_to_string(dir.join("game.log")).unwrap();
        assert_eq!(
            written,
            format!(
                "WARN  {}: texture missing\nERROR Player:9: null entity\n",
                module_path!()
            )
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_log_file_that_cannot_be_opened_costs_the_file_and_nothing_else() {
        let missing = scratch("unwritable")
            .join("no-such-directory")
            .join("game.log");
        let (stream, console, complaint) = Stream::new(quiet(LogConfig {
            file: Some(missing.clone()),
            ..LogConfig::editor()
        }));
        let complaint = complaint.expect("the failure is reported");
        assert!(
            complaint.contains(&missing.display().to_string()),
            "{complaint}"
        );

        tracing::subscriber::with_default(stream, || tracing::error!("still heard"));
        let mut log = LogBuffer::default();
        console.drain_into(&mut log, 0);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn the_console_queue_drops_its_oldest_past_capacity() {
        let lines = console_of(LogConfig::editor(), || {
            for index in 0..QUEUE_CAPACITY + 5 {
                tracing::info!("{index}");
            }
        });
        assert_eq!(lines.len(), QUEUE_CAPACITY);
        assert!(lines[0].1.ends_with(": 5"), "{}", lines[0].1);
    }

    #[test]
    fn a_line_leads_with_its_level_padded_to_a_column() {
        let entry = |level| LogEntry {
            level,
            message: "attached".to_owned(),
            source: Cow::Borrowed("Spinner"),
            line: 14,
            frame: 0,
        };
        assert_eq!(
            format_line(&entry(LogLevel::Info)),
            "INFO  Spinner:14: attached\n"
        );
        assert_eq!(
            format_line(&entry(LogLevel::Error)),
            "ERROR Spinner:14: attached\n"
        );
    }
}
