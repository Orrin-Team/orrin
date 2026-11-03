//! File → Export game: the editor's door to `orrin export`.
//!
//! The editor does not package anything itself. It runs the CLI, the same
//! command a terminal or the hub runs, so a game exports one way wherever the
//! button was. The CLI's output streams into the console — its own step lines
//! as they happen, the compiler's only when something fails, since a release
//! build of the engine is hundreds of lines nobody wants scrolling past — and
//! the status bar carries the step it is on.
//!
//! The CLI is found as `orrin run` leaves it (`$ORRIN_CLI`, set to itself),
//! else beside the engine's executable, as cargo and an install both lay them
//! out, else on the PATH.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};
use std::sync::mpsc::Receiver;

use orrin_project::export::Platform;

use crate::scene::LogLevel;

pub const CLI_ENV: &str = "ORRIN_CLI";

/// Where the last or current export stands.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Status {
    #[default]
    Idle,
    Running {
        platform: Platform,
        /// The step the CLI last announced, as it worded it.
        step: Option<String>,
    },
    Done {
        platform: Platform,
        out: PathBuf,
    },
    Failed {
        platform: Platform,
        reason: String,
    },
}

/// The export this session is running, if any, and how the last one ended.
#[derive(Default)]
pub struct Export {
    job: Option<Job>,
    status: Status,
}

struct Job {
    child: Child,
    lines: Receiver<String>,
    out: PathBuf,
    /// The last lines of output, for the console when the export fails.
    tail: VecDeque<String>,
}

impl Export {
    pub fn status(&self) -> &Status {
        &self.status
    }

    pub fn is_running(&self) -> bool {
        matches!(self.status, Status::Running { .. })
    }

    /// Run `cli export` on the project at `root`, called `name`, for
    /// `platform`.
    pub fn start(
        &mut self,
        cli: &Path,
        root: &Path,
        name: &str,
        platform: Platform,
    ) -> Result<(), String> {
        if self.is_running() {
            return Err("an export is already running; wait for it to finish".to_owned());
        }
        let mut child = std::process::Command::new(cli)
            .arg("export")
            .arg("--project")
            .arg(root)
            .args(["--platform", platform.name()])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("CARGO_TERM_COLOR", "never")
            .spawn()
            .map_err(|error| format!("could not run {}: {error}", cli.display()))?;
        let (sender, lines) = std::sync::mpsc::channel();
        let streams: [Option<Box<dyn Read + Send>>; 2] = [
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
        ];
        for stream in streams.into_iter().flatten() {
            let sender = sender.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stream).lines().map_while(Result::ok) {
                    if sender.send(line).is_err() {
                        return;
                    }
                }
            });
        }
        self.job = Some(Job {
            child,
            lines,
            // Where the CLI puts it without `--out`, which is never passed.
            out: root.join("exports").join(format!("{name}-{platform}")),
            tail: VecDeque::new(),
        });
        self.status = Status::Running {
            platform,
            step: None,
        };
        Ok(())
    }

    /// Take what the CLI has said since the last call, as console lines, and
    /// notice it finishing.
    pub fn service(&mut self) -> Vec<(LogLevel, String)> {
        let mut logged = Vec::new();
        let (Some(job), Status::Running { platform, step }) = (&mut self.job, &mut self.status)
        else {
            return logged;
        };
        let platform = *platform;
        // Finished is the process gone *and* every line it wrote read: the
        // channel disconnects once both readers reach the end of their pipes,
        // which can be a moment after the exit is seen.
        let mut drained = false;
        loop {
            let line = match job.lines.try_recv() {
                Ok(line) => line,
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    drained = true;
                    break;
                }
            };
            if let Some(said) = line.strip_prefix("orrin: ") {
                if said.starts_with('[') {
                    *step = Some(said.split(" for ").next().unwrap_or(said).to_owned());
                }
                logged.push((LogLevel::Info, format!("export: {said}")));
            }
            job.tail.push_back(line);
            if job.tail.len() > TAIL {
                job.tail.pop_front();
            }
        }
        let exit = match job.child.try_wait() {
            Ok(Some(exit)) if drained => exit,
            _ => return logged,
        };
        let job = self.job.take().expect("a running export has a job");
        self.status = if exit.success() {
            logged.push((
                LogLevel::Info,
                format!(
                    "export: the {} game is in {}",
                    label(platform),
                    job.out.display()
                ),
            ));
            Status::Done {
                platform,
                out: job.out,
            }
        } else {
            let reason = job
                .tail
                .iter()
                .rev()
                .find_map(|line| line.strip_prefix("orrin: "))
                .unwrap_or("the CLI failed without saying why")
                .to_owned();
            logged.extend(
                job.tail
                    .iter()
                    .filter(|line| !line.starts_with("orrin: "))
                    .map(|line| (LogLevel::Error, format!("export: {line}"))),
            );
            logged.push((LogLevel::Error, format!("export failed: {reason}")));
            Status::Failed { platform, reason }
        };
        logged
    }
}

/// How many lines of output a failure shows.
const TAIL: usize = 40;

/// The `orrin` CLI: `env` (from [`CLI_ENV`]) when set, else beside the engine
/// in `beside`, else on `path` (a `PATH` value).
pub fn find_cli(
    env: Option<OsString>,
    beside: Option<&Path>,
    path: Option<OsString>,
) -> Result<PathBuf, String> {
    if let Some(cli) = env {
        return Ok(PathBuf::from(cli));
    }
    let name = format!("orrin{}", std::env::consts::EXE_SUFFIX);
    if let Some(cli) = beside
        .map(|dir| dir.join(&name))
        .filter(|cli| cli.is_file())
    {
        return Ok(cli);
    }
    path.and_then(|path| {
        std::env::split_paths(&path)
            .map(|dir| dir.join(&name))
            .find(|cli| cli.is_file())
    })
    .ok_or_else(|| {
        "exporting runs the `orrin` CLI, and it is not beside the engine or on the PATH. \
         Build it with `cargo build -p orrin-cli`, or open the project with `orrin run`"
            .to_owned()
    })
}

/// Show `path` in the platform's file manager.
pub fn reveal(path: &Path) {
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(windows) {
        "explorer"
    } else {
        "xdg-open"
    };
    if let Err(error) = std::process::Command::new(opener).arg(path).spawn() {
        tracing::warn!("could not open {}: {error}", path.display());
    }
}

/// What the menu and status bar call a platform.
pub fn label(platform: Platform) -> &'static str {
    match platform {
        Platform::Mac => "macOS",
        Platform::Linux => "Linux",
        Platform::Windows => "Windows",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// A scratch directory, removed when dropped.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("orrin-export-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn exe(path: &Path) -> PathBuf {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"").unwrap();
        path.to_path_buf()
    }

    #[test]
    fn the_cli_is_found_by_env_then_beside_the_engine_then_on_the_path() {
        let dir = Scratch::new("find");
        let name = format!("orrin{}", std::env::consts::EXE_SUFFIX);
        let named = exe(&dir.path().join("named").join(&name));
        let beside = exe(&dir.path().join("engine").join(&name));
        let on_path = exe(&dir.path().join("bin").join(&name));
        let path = std::env::join_paths([dir.path().join("empty"), dir.path().join("bin")]).ok();

        assert_eq!(
            find_cli(
                Some(named.clone().into()),
                Some(&dir.path().join("engine")),
                path.clone()
            ),
            Ok(named)
        );
        assert_eq!(
            find_cli(None, Some(&dir.path().join("engine")), path.clone()),
            Ok(beside)
        );
        assert_eq!(
            find_cli(None, Some(&dir.path().join("none")), path),
            Ok(on_path)
        );

        let error = find_cli(None, None, None).unwrap_err();
        assert!(error.contains("orrin-cli"), "says how to get one: {error}");
    }

    /// A stand-in for `orrin` that records its arguments and says `lines`.
    #[cfg(unix)]
    fn fake_cli(dir: &Path, lines: &str, status: i32) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("orrin");
        let args = dir.join("args");
        std::fs::write(
            &path,
            format!(
                "#!/bin/sh\necho \"$@\" > '{}'\n{lines}\nexit {status}\n",
                args.display()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    /// Service `export` until it stops running, collecting what it logged.
    fn finish(export: &mut Export) -> Vec<(LogLevel, String)> {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut logged = Vec::new();
        loop {
            logged.extend(export.service());
            if !export.is_running() {
                return logged;
            }
            assert!(Instant::now() < deadline, "the export never finished");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[cfg(unix)]
    #[test]
    fn an_export_runs_the_cli_reports_its_steps_and_says_where_the_game_went() {
        let dir = Scratch::new("done");
        let cli = fake_cli(
            dir.path(),
            "echo 'orrin: [2/4] building the engine for windows (x86_64-pc-windows-msvc)'\n\
             echo '   Compiling orrin-core v0.1.0'\n\
             echo 'orrin: exported `game` for windows -> exports/game-windows (102 MB)'",
            0,
        );
        let root = dir.path().join("game");
        let mut export = Export::default();

        export
            .start(&cli, &root, "game", Platform::Windows)
            .unwrap();
        assert!(export.is_running());
        let logged = finish(&mut export);

        let args = std::fs::read_to_string(dir.path().join("args")).unwrap();
        assert_eq!(
            args.trim(),
            format!("export --project {} --platform windows", root.display())
        );
        assert_eq!(
            export.status(),
            &Status::Done {
                platform: Platform::Windows,
                out: root.join("exports/game-windows"),
            }
        );
        let said: Vec<&str> = logged.iter().map(|(_, line)| line.as_str()).collect();
        assert!(
            said.iter()
                .any(|line| line.contains("[2/4] building the engine")),
            "{said:?}"
        );
        assert!(
            !said.iter().any(|line| line.contains("Compiling")),
            "compiler noise: {said:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_export_says_why_in_the_console_and_the_status() {
        let dir = Scratch::new("failed");
        let cli = fake_cli(
            dir.path(),
            "echo '   Compiling shaderc-sys v0.8' >&2\n\
             echo 'orrin: building for windows from this machine needs `cargo xwin`' >&2",
            1,
        );
        let mut export = Export::default();

        export
            .start(&cli, dir.path(), "game", Platform::Windows)
            .unwrap();
        let logged = finish(&mut export);

        let Status::Failed { reason, .. } = export.status() else {
            panic!("{:?}", export.status());
        };
        assert!(reason.contains("cargo xwin"), "{reason}");
        assert!(
            logged
                .iter()
                .any(|(level, line)| *level == LogLevel::Error && line.contains("shaderc-sys")),
            "the output before a failure reaches the console: {logged:?}"
        );
    }

    #[test]
    fn a_second_export_is_refused_while_one_runs() {
        let mut export = Export {
            job: None,
            status: Status::Running {
                platform: Platform::Mac,
                step: None,
            },
        };
        let error = export
            .start(Path::new("orrin"), Path::new("."), "game", Platform::Linux)
            .unwrap_err();
        assert!(error.contains("already"), "{error}");
    }
}
