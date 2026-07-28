//! Rebuild-on-save for the project's C# scripts, behind the `scripting` feature.
//!
//! `notify` watches the scripts directory on a thread of its own; each source
//! change restarts a short debounce, and the first quiet moment after it hands a
//! `dotnet build` to a worker thread. Nothing in here touches the world's
//! entities or the script host: a green build only raises the same request the
//! "Reload scripts" button raises, so the swap still happens at the one point in
//! the frame where managed objects may be destroyed and re-created.
//!
//! A red build is inert by construction. The staged swap in `Scripting::reload`
//! is never reached, so the session keeps running the code it already had and
//! the failure is just text in the console.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use notify::{RecommendedWatcher, RecursiveMode, Watcher};

use orrin_ecs::World;

use crate::scene::{LogBuffer, LogLevel, Time};

/// How long the scripts directory must stay quiet before a rebuild starts.
/// Long enough to coalesce a "save all" across several files, short enough that
/// a single save doesn't feel like it was ignored.
const DEBOUNCE: Duration = Duration::from_millis(300);

/// Compiler lines pushed to the console per failed build. One broken file can
/// emit hundreds, and the log is a bounded ring — an unfiltered dump would
/// evict every other message in it.
const MAX_DIAGNOSTICS: usize = 12;

/// Log lines shown for a failure the compiler never got as far as diagnosing.
const TAIL_LINES: usize = 5;

/// What the watcher has to report, in the order it happened.
pub enum BuildEvent {
    /// The debounce elapsed and a rebuild started.
    Started,
    Succeeded { duration: Duration },
    /// The compiler rejected the code. `diagnostics` are its error and warning
    /// lines, or the tail of the log when the build died before compiling.
    Failed { diagnostics: Vec<String> },
    /// `dotnet build` could not be run at all.
    Unavailable { reason: String },
}

impl BuildEvent {
    fn log_into(&self, log: &mut LogBuffer, frame: u64) {
        match self {
            Self::Started => {
                log.push(LogLevel::Info, "scripts changed; rebuilding".to_owned(), frame)
            }
            Self::Succeeded { duration } => log.push(
                LogLevel::Info,
                format!("scripts rebuilt in {:.1}s", duration.as_secs_f32()),
                frame,
            ),
            Self::Failed { diagnostics } => {
                log.push(
                    LogLevel::Error,
                    "script build failed; still running the previous build".to_owned(),
                    frame,
                );
                for line in diagnostics.iter().take(MAX_DIAGNOSTICS) {
                    log.push(LogLevel::Error, line.clone(), frame);
                }
                if diagnostics.len() > MAX_DIAGNOSTICS {
                    let hidden = diagnostics.len() - MAX_DIAGNOSTICS;
                    log.push(LogLevel::Error, format!("... and {hidden} more"), frame);
                }
            }
            Self::Unavailable { reason } => {
                log.push(LogLevel::Error, format!("script build could not run: {reason}"), frame)
            }
        }
    }
}

/// Where the watcher is up to, as a world resource so the Scripts panel can
/// show it and flip [`auto_reload`](Self::auto_reload) without reaching into
/// `App`. Present whenever scripting is compiled in, watcher or no watcher.
pub struct BuildStatus {
    /// Whether a green build reloads by itself. Rebuilding happens either way,
    /// so turning this off still gets compiler errors into the console — it
    /// only stops the swap from disturbing a live session unasked.
    pub auto_reload: bool,
    pub state: BuildState,
    /// Diagnostics from the most recent failed build; empty otherwise.
    pub diagnostics: Vec<String>,
    /// How long the last successful build took.
    pub last_duration: Option<Duration>,
}

impl Default for BuildStatus {
    fn default() -> Self {
        Self {
            auto_reload: true,
            state: BuildState::Idle,
            diagnostics: Vec::new(),
            last_duration: None,
        }
    }
}

pub enum BuildState {
    /// Watching; nothing has changed yet this session.
    Idle,
    Building,
    Succeeded,
    Failed,
    /// No watcher is running. The string says why.
    Off(String),
}

impl BuildStatus {
    pub fn disable(&mut self, reason: impl Into<String>) {
        self.state = BuildState::Off(reason.into());
    }

    fn observe(&mut self, event: BuildEvent) {
        self.state = match event {
            BuildEvent::Started => BuildState::Building,
            BuildEvent::Succeeded { duration } => {
                self.last_duration = Some(duration);
                self.diagnostics.clear();
                BuildState::Succeeded
            }
            BuildEvent::Failed { diagnostics } => {
                self.diagnostics = diagnostics;
                BuildState::Failed
            }
            BuildEvent::Unavailable { reason } => BuildState::Off(reason),
        };
    }
}

type BuildResult = Result<orrin_build::Report, orrin_build::Error>;

pub struct BuildWatcher {
    /// Held only for its `Drop`: releasing it stops the OS watch and the thread
    /// behind it. Nothing ever reads this field.
    _watcher: RecommendedWatcher,
    changes: Receiver<()>,
    results: Receiver<BuildResult>,
    finished: Sender<BuildResult>,
    scripts_dir: PathBuf,
    configuration: String,
    bindings: Option<PathBuf>,
    /// When the scripts directory last changed. A further change pushes it
    /// forward rather than starting a second timer, so a burst of saves — or a
    /// "save all" — compiles once, after the last of them.
    pending_since: Option<Instant>,
    /// When the in-flight build started, if there is one.
    building_since: Option<Instant>,
}

impl BuildWatcher {
    /// Start a watcher for the project that produced `game_dll`.
    ///
    /// Both the scripts directory and the configuration come from the loaded
    /// assembly's own path (`<scripts>/bin/<configuration>/<tfm>/Game.dll`)
    /// rather than from the manifest, so a rebuild necessarily writes the exact
    /// file `Scripting::reload` re-reads — the two cannot drift into rebuilding
    /// Debug while the session runs Release. An `$ORRIN_GAME_DLL` pointing
    /// outside that layout therefore gets no watcher: a build that reported
    /// success while leaving the loaded DLL untouched would be worse than not
    /// watching at all.
    pub fn for_game_assembly(game_dll: &Path, bindings: Option<&Path>) -> Result<Self, String> {
        let Some((scripts_dir, configuration)) = build_location(game_dll) else {
            return Err(format!(
                "{} is not inside a `bin/<configuration>/<tfm>/` build output, so there is \
                 no project directory to watch",
                game_dll.display()
            ));
        };

        Self::start(scripts_dir, configuration, bindings)
            .map_err(|err| format!("could not watch {}: {err}", scripts_dir.display()))
    }

    fn start(
        scripts_dir: &Path,
        configuration: &str,
        bindings: Option<&Path>,
    ) -> Result<Self, notify::Error> {
        let (changes_tx, changes) = mpsc::channel();
        let root = scripts_dir.to_path_buf();

        // Filtering here rather than on the main thread keeps the channel down
        // to "something worth compiling changed" — a `dotnet build` writes
        // hundreds of files under `bin/` and `obj/` that must never reach it.
        let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
            let Ok(event) = event else { return };
            if !(event.kind.is_create() || event.kind.is_modify() || event.kind.is_remove()) {
                return;
            }
            if event.paths.iter().any(|path| is_script_source(&root, path)) {
                let _ = changes_tx.send(());
            }
        })?;
        watcher.watch(scripts_dir, RecursiveMode::Recursive)?;

        let (finished, results) = mpsc::channel();
        Ok(Self {
            _watcher: watcher,
            changes,
            results,
            finished,
            scripts_dir: scripts_dir.to_path_buf(),
            configuration: configuration.to_owned(),
            bindings: bindings.map(Path::to_path_buf),
            pending_since: None,
            building_since: None,
        })
    }

    /// Advance the watcher, report into the console and the `BuildStatus`
    /// resource, and return whether a script reload should run this frame.
    ///
    /// Returning the request rather than acting on it is the whole point: the
    /// caller folds it into the editor button's request and performs the swap
    /// at the frame's single reload point.
    pub fn service(&mut self, world: &World) -> bool {
        let events = self.poll();
        if events.is_empty() {
            return false;
        }

        let frame = world.resource::<Time>().frame_count();
        let mut reload = false;
        for event in events {
            event.log_into(&mut world.resource_mut::<LogBuffer>(), frame);

            let mut status = world.resource_mut::<BuildStatus>();
            if status.auto_reload && matches!(event, BuildEvent::Succeeded { .. }) {
                reload = true;
            }
            status.observe(event);
        }
        reload
    }

    fn poll(&mut self) -> Vec<BuildEvent> {
        let mut events = Vec::new();

        // Drained, not taken one at a time: the debounce is measured from the
        // last change, and a single save can produce several notifications.
        let mut changed = false;
        while self.changes.try_recv().is_ok() {
            changed = true;
        }
        if changed {
            self.pending_since = Some(Instant::now());
        }

        if let Ok(result) = self.results.try_recv() {
            let elapsed = self.building_since.take().map_or(Duration::ZERO, |t| t.elapsed());
            events.push(match result {
                Ok(report) if report.success => BuildEvent::Succeeded { duration: elapsed },
                Ok(report) => BuildEvent::Failed {
                    diagnostics: failure_lines(&report),
                },
                Err(err) => BuildEvent::Unavailable {
                    reason: err.to_string(),
                },
            });
        }

        // `pending_since` survives a build in flight, so changes saved while the
        // compiler was running start the next build the moment this one lands —
        // including in this same call, since the branch above just cleared
        // `building_since`.
        if self.building_since.is_none()
            && let Some(since) = self.pending_since
            && since.elapsed() >= DEBOUNCE
        {
            self.pending_since = None;
            self.spawn_build();
            events.push(BuildEvent::Started);
        }

        events
    }

    fn spawn_build(&mut self) {
        let scripts_dir = self.scripts_dir.clone();
        let configuration = self.configuration.clone();
        let bindings = self.bindings.clone();
        let finished = self.finished.clone();

        self.building_since = Some(Instant::now());
        // Detached, and deliberately so: the send is what reports back, and it
        // fails harmlessly if the engine closed while the compiler was running.
        std::thread::spawn(move || {
            let result = orrin_build::Build {
                project_dir: &scripts_dir,
                configuration: &configuration,
                bindings: bindings.as_deref(),
            }
            .run(orrin_build::Output::Capture);
            let _ = finished.send(result);
        });
    }
}

fn failure_lines(report: &orrin_build::Report) -> Vec<String> {
    let diagnostics = report.diagnostics();
    if diagnostics.is_empty() {
        report.tail(TAIL_LINES)
    } else {
        diagnostics
    }
}

/// The project directory and MSBuild configuration a built assembly came from:
/// `<scripts>/bin/Debug/net10.0/Game.dll` -> (`<scripts>`, `Debug`).
fn build_location(game_dll: &Path) -> Option<(&Path, &str)> {
    let tfm = game_dll.parent()?;
    let configuration = tfm.parent()?;
    let bin = configuration.parent()?;
    if bin.file_name()? != "bin" {
        return None;
    }
    Some((bin.parent()?, configuration.file_name()?.to_str()?))
}

/// Whether a changed path is a source file worth recompiling for.
///
/// The `bin`/`obj` exclusion is load-bearing, not tidiness: `dotnet build`
/// writes its output under both, so letting them through would make every build
/// trigger the next one for as long as the editor stayed open.
fn is_script_source(root: &Path, path: &Path) -> bool {
    let relative = path.strip_prefix(root).unwrap_or(path);
    if relative
        .components()
        .any(|part| matches!(part.as_os_str().to_str(), Some("bin" | "obj")))
    {
        return false;
    }

    // Editors save through hidden temp files (`.#Player.cs`, `.Player.cs.swp`).
    // The write that matters always lands on the real name afterwards.
    if path.file_name().and_then(|name| name.to_str()).is_none_or(|name| name.starts_with('.')) {
        return false;
    }

    matches!(path.extension().and_then(|ext| ext.to_str()), Some("cs" | "csproj"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_build_location_comes_from_the_assembly_path() {
        let dll = Path::new("/proj/scripts/bin/Debug/net10.0/MyGame.dll");

        assert_eq!(
            build_location(dll),
            Some((Path::new("/proj/scripts"), "Debug"))
        );
    }

    #[test]
    fn an_assembly_outside_a_bin_layout_has_no_build_location() {
        // What `$ORRIN_GAME_DLL` pointing at a hand-placed DLL looks like: there
        // is no project to rebuild, so no watcher may start.
        assert_eq!(build_location(Path::new("/tmp/MyGame.dll")), None);
        assert_eq!(build_location(Path::new("/tmp/out/Debug/net10.0/MyGame.dll")), None);
    }

    #[test]
    fn sources_are_watched_and_build_output_is_not() {
        let root = Path::new("/proj/scripts");

        assert!(is_script_source(root, &root.join("Player.cs")));
        assert!(is_script_source(root, &root.join("nested/Enemy.cs")));
        assert!(is_script_source(root, &root.join("MyGame.csproj")));

        // The infinite-rebuild loop this exists to prevent.
        assert!(!is_script_source(root, &root.join("obj/Debug/net10.0/MyGame.AssemblyInfo.cs")));
        assert!(!is_script_source(root, &root.join("bin/Debug/net10.0/MyGame.dll")));

        assert!(!is_script_source(root, &root.join("readme.md")));
        assert!(!is_script_source(root, &root.join(".#Player.cs")));
    }

    #[test]
    fn a_bin_directory_above_the_project_does_not_exclude_its_sources() {
        // The exclusion is relative to the watched root: a project that happens
        // to live under a directory called `bin` still gets watched.
        let root = Path::new("/bin/proj/scripts");

        assert!(is_script_source(root, &root.join("Player.cs")));
    }
}
