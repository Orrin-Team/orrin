//! A command the hub runs through the `orrin` CLI, with its output streamed
//! back line by line.
//!
//! The hub never builds, runs or exports anything itself. It drives the same
//! CLI a terminal user types, so there is one implementation of each command
//! and the hub's activity panel shows exactly what the terminal would.

use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use orrin_project::export::Platform;

/// What a task is for, which decides what the hub does when it finishes.
#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    /// `orrin new`; the project it makes, and whether to open it after.
    Create {
        dir: PathBuf,
        open_after: bool,
    },
    /// `orrin run`: the editor on a project.
    Edit,
    Build,
    Export {
        platform: Platform,
        out: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Running,
    Done,
    Failed,
    Cancelled,
}

/// The longest output kept: enough to scroll back through a failed build,
/// bounded so an engine left logging for an hour does not grow without end.
const KEEP: usize = 4_000;

pub struct Task {
    pub title: String,
    pub kind: Kind,
    child: Option<Child>,
    lines: Arc<Mutex<Vec<String>>>,
    pub state: State,
}

impl Task {
    /// Start `orrin args…`, streaming its output into the task.
    pub fn spawn(
        cli: &std::path::Path,
        args: &[std::ffi::OsString],
        title: String,
        kind: Kind,
        ctx: &egui::Context,
    ) -> Self {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let mut command = Command::new(cli);
        command
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("CARGO_TERM_COLOR", "never")
            .env("NO_COLOR", "1");
        // Its own process group, so Cancel stops the cargo and dotnet it
        // started too rather than orphaning them.
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut command, 0);
        let (child, state) = match command.spawn() {
            Ok(mut child) => {
                for stream in [
                    child
                        .stdout
                        .take()
                        .map(|s| Box::new(s) as Box<dyn Read + Send>),
                    child
                        .stderr
                        .take()
                        .map(|s| Box::new(s) as Box<dyn Read + Send>),
                ]
                .into_iter()
                .flatten()
                {
                    let lines = lines.clone();
                    let ctx = ctx.clone();
                    std::thread::spawn(move || {
                        for line in BufReader::new(stream).lines().map_while(Result::ok) {
                            let mut lines = lines.lock().unwrap_or_else(|p| p.into_inner());
                            lines.push(line);
                            if lines.len() > KEEP {
                                lines.drain(..KEEP / 4);
                            }
                            ctx.request_repaint();
                        }
                    });
                }
                (Some(child), State::Running)
            }
            Err(error) => {
                lines
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(format!("could not start {}: {error}", cli.display()));
                (None, State::Failed)
            }
        };
        Self {
            title,
            kind,
            child,
            lines,
            state,
        }
    }

    /// Notice the process ending. Returns true on the frame it does.
    pub fn poll(&mut self) -> bool {
        let Some(child) = &mut self.child else {
            return false;
        };
        match child.try_wait() {
            Ok(Some(status)) => {
                self.child = None;
                if self.state == State::Running {
                    self.state = if status.success() {
                        State::Done
                    } else {
                        State::Failed
                    };
                }
                true
            }
            Ok(None) => false,
            Err(_) => {
                self.child = None;
                self.state = State::Failed;
                true
            }
        }
    }

    pub fn cancel(&mut self) {
        if let Some(child) = &mut self.child {
            let pid = child.id().to_string();
            #[cfg(unix)]
            let _ = Command::new("kill")
                .args(["-TERM", &format!("-{pid}")])
                .status();
            #[cfg(windows)]
            let _ = Command::new("taskkill")
                .args(["/T", "/F", "/PID", &pid])
                .status();
            let _ = child.kill();
            self.state = State::Cancelled;
        }
    }

    pub fn lines(&self) -> Vec<String> {
        self.lines.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// The latest step the command announced, if it announces any.
    pub fn step(&self) -> Option<Step> {
        let lines = self.lines.lock().unwrap_or_else(|p| p.into_inner());
        lines.iter().rev().find_map(|line| step(line))
    }

    /// The last thing the command said, for a one-line status.
    pub fn last_line(&self) -> Option<String> {
        let lines = self.lines.lock().unwrap_or_else(|p| p.into_inner());
        lines
            .iter()
            .rev()
            .find(|line| !line.trim().is_empty())
            .cloned()
    }
}

impl Drop for Task {
    fn drop(&mut self) {
        // An editor the hub opened is the user's now; only work the hub is
        // still doing on their behalf stops with it.
        if self.kind != Kind::Edit {
            self.cancel();
        }
    }
}

/// How far a multi-step command has got, from a line like
/// `orrin: [2/4] building the engine for mac (aarch64-apple-darwin)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub index: u32,
    pub of: u32,
    pub what: String,
}

pub fn step(line: &str) -> Option<Step> {
    let rest = line.trim().strip_prefix("orrin: [")?;
    let (counts, what) = rest.split_once("] ")?;
    let (index, of) = counts.split_once('/')?;
    let what = what.split(" for ").next().unwrap_or(what);
    let mut chars = what.chars();
    let what = chars
        .next()
        .map(|first| first.to_uppercase().chain(chars).collect())
        .unwrap_or_default();
    Some(Step {
        index: index.parse().ok()?,
        of: of.parse().ok()?,
        what,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_progress_line_reads_as_a_step_and_anything_else_does_not() {
        assert_eq!(
            step("orrin: [2/4] building the engine for mac (aarch64-apple-darwin)"),
            Some(Step {
                index: 2,
                of: 4,
                what: "Building the engine".to_owned()
            })
        );
        assert_eq!(step("   Compiling orrin-core v0.1.0"), None);
        assert_eq!(step("orrin: [x/4] nonsense"), None);
    }
}
