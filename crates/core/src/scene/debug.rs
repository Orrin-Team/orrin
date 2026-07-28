//! Editor-facing debug facilities: a bounded log buffer surfaced in the console
//! panel, and a per-frame buffer of world-space lines drawn as a scene overlay.
//! Both are engine resources fed by scripts through the debug ABI in
//! `scripting.rs`; both are inert (or absent) in export builds.

use std::collections::VecDeque;

use glam::Vec3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LogLevel {
    Info,
    Warning,
    Error,
}

pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
    /// The `Time::frame_count` value when the entry was logged.
    pub frame: u64,
}

/// Bounded ring buffer of log lines. Oldest entries are dropped once `capacity`
/// is reached, so a chatty script can't grow it without bound.
pub struct LogBuffer {
    entries: VecDeque<LogEntry>,
    capacity: usize,
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
        }
    }

    pub fn push(&mut self, level: LogLevel, message: String, frame: u64) {
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(LogEntry {
            level,
            message,
            frame,
        });
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
