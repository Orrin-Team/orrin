//! Editor-facing debug facilities: a bounded log buffer surfaced in the console
//! panel, and a per-frame buffer of world-space lines drawn as a scene overlay.
//! Both are engine resources. The lines are fed by scripts through the debug
//! ABI in `scripting.rs`; the log is fed by the engine's log stream
//! (`logging.rs`), which is where a script's `Debug.Log` arrives too, and by
//! the editor pushing its own lines. Both are inert (or absent) in export
//! builds.

use std::borrow::Cow;
use std::collections::VecDeque;

use glam::Vec3;

/// Declared quietest first: the derived order is what a threshold compares
/// against, so `level >= LogLevel::Warning` reads as it sounds.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
}

impl LogLevel {
    pub const ALL: [Self; 5] = [
        Self::Trace,
        Self::Debug,
        Self::Info,
        Self::Warning,
        Self::Error,
    ];

    /// The fixed-case tag a log line leads with, on stderr and in the console.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
        }
    }

    /// A level by name, as `ORRIN_LOG` spells it. Case-insensitive.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "trace" => Some(Self::Trace),
            "debug" => Some(Self::Debug),
            "info" => Some(Self::Info),
            "warn" | "warning" => Some(Self::Warning),
            "error" => Some(Self::Error),
            _ => None,
        }
    }
}

pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
    /// Who said it: a script's name, or the Rust module an engine event came
    /// from. Empty for a line the editor pushed itself.
    pub source: Cow<'static, str>,
    /// The line in `source`, when a script reported one; `0` otherwise.
    pub line: u32,
    /// The `Time::frame_count` value when the entry was logged.
    pub frame: u64,
}

impl LogEntry {
    /// The line as every sink writes it: `source:line: message`, with whichever
    /// of the first two parts the entry does not have left out.
    pub fn text(&self) -> String {
        match (self.source.is_empty(), self.line) {
            (true, _) => self.message.clone(),
            (false, 0) => format!("{}: {}", self.source, self.message),
            (false, line) => format!("{}:{line}: {}", self.source, self.message),
        }
    }
}

/// Bounded ring buffer of log lines. Oldest entries are dropped once `capacity`
/// is reached, so a chatty script can't grow it without bound.
pub struct LogBuffer {
    entries: VecDeque<LogEntry>,
    capacity: usize,
    /// Every entry ever pushed, so a reader can hold its place by number
    /// across the ring dropping entries and the console being cleared.
    pushed: u64,
    /// `pushed` by level, for a count that the ring dropping entries cannot
    /// make wrong: the console's tally of what was logged out of sight.
    logged: [u64; LogLevel::ALL.len()],
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::with_capacity(1024)
    }
}

impl LogBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity: capacity.max(1),
            pushed: 0,
            logged: [0; LogLevel::ALL.len()],
        }
    }

    pub fn push(&mut self, level: LogLevel, message: String, frame: u64) {
        self.push_entry(LogEntry {
            level,
            message,
            source: Cow::Borrowed(""),
            line: 0,
            frame,
        });
    }

    pub fn push_entry(&mut self, entry: LogEntry) {
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.logged[entry.level as usize] += 1;
        self.entries.push_back(entry);
        self.pushed += 1;
    }

    /// How many entries of `level` were ever pushed, held or not.
    pub fn logged(&self, level: LogLevel) -> u64 {
        self.logged[level as usize]
    }

    /// The entries numbered `from` onwards, each with its number: the first
    /// entry ever pushed is 0. One no longer held is skipped.
    pub fn since(&self, from: u64) -> impl Iterator<Item = (u64, &LogEntry)> {
        let first = self.pushed - self.entries.len() as u64;
        let skip = from.saturating_sub(first).min(self.entries.len() as u64) as usize;
        (first..).zip(&self.entries).skip(skip)
    }

    /// The number the next entry will be given.
    pub fn next_number(&self) -> u64 {
        self.pushed
    }

    pub fn iter(&self) -> impl Iterator<Item = &LogEntry> {
        self.entries.iter()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// One world-space line segment queued for this frame's debug overlay.
#[derive(Clone, Copy)]
pub struct DebugLine {
    pub from: Vec3,
    pub to: Vec3,
    pub color: [f32; 4],
    /// Elapsed-time value (seconds) at which this line stops being drawn. A line
    /// requested with `duration <= 0` gets `expiry == now`, so `sweep` drops it
    /// the following frame — it shows for exactly one frame.
    expiry: f32,
}

/// The per-frame set of debug lines. Scripts push during their tick; the line
/// pass reads it while recording; `sweep` runs once afterwards.
#[derive(Default)]
pub struct DebugLines {
    lines: Vec<DebugLine>,
}

impl DebugLines {
    /// Queue a line. `now` is the current elapsed time; `duration <= 0` requests
    /// a single-frame line.
    pub fn push(&mut self, from: Vec3, to: Vec3, color: [f32; 4], now: f32, duration: f32) {
        self.lines.push(DebugLine {
            from,
            to,
            color,
            expiry: now + duration.max(0.0),
        });
    }

    /// Drop expired lines. Run once per frame *after* the line pass records, with
    /// the current elapsed time: a single-frame line (whose expiry equals its
    /// spawn time) is gone by the next frame, while a timed one survives until
    /// its expiry passes.
    pub fn sweep(&mut self, now: f32) {
        self.lines.retain(|line| line.expiry > now);
    }

    pub fn lines(&self) -> &[DebugLine] {
        &self.lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_leaves_out_the_parts_its_entry_does_not_have() {
        let entry = |source: &'static str, line| LogEntry {
            level: LogLevel::Info,
            message: "attached".to_owned(),
            source: Cow::Borrowed(source),
            line,
            frame: 0,
        };
        assert_eq!(entry("", 0).text(), "attached");
        assert_eq!(
            entry("orrin_core::app", 0).text(),
            "orrin_core::app: attached"
        );
        assert_eq!(entry("Spinner", 14).text(), "Spinner:14: attached");
    }
}
