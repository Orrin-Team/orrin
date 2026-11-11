//! The headless test runner: a project's `[Test]` methods, run against a real
//! world with no window and no GPU. `orrin-core --test [filter]` is the engine
//! binary's door into it, and `orrin test` the CLI's.
//!
//! # The engine owns the frame
//!
//! An iterator test is not a loop that runs frames. The managed side advances
//! it to its next `yield` and returns ([`ScriptHost::begin_test`], then
//! [`ScriptHost::resume_test`] once per frame), and the runner runs the frame
//! in between, outside any dispatch window — exactly as the app runs the
//! script tick between two calls into C#. So test code is game code like any
//! other: it runs inside a window, its spawns and despawns land when the turn
//! ends, and it can never re-enter the frame loop.
//!
//! # A fresh world each
//!
//! Every test gets a world of its own, with the resources the app installs
//! ([`App::install_default_resources`]), the engine's built-in meshes and
//! materials, the project's input bindings and prefabs, and the transport in
//! Playing. It is dropped before the next test starts, which destroys every
//! Behaviour in it — so nothing one test spawned is alive in the next, and no
//! managed handle survives the run into a later reload. Frames are a fixed
//! 1/60 s, so a test's outcome is the same on any machine at any load.
//!
//! What the world does not have is anything that needs a device: no asset
//! database (it decides import formats from the GPU it is opened for), no audio
//! output and no renderer. A test reaches the engine's own named meshes and
//! materials, and a scene file the project's prefabs can expand.
//!
//! # The frame is the app's game half
//!
//! [`Runner::frame`] runs the systems a playing game runs, in the order
//! `app.rs` runs them, and nothing of the editor's. It is a copy of that order
//! rather than a function the app calls too, because the app's frame threads
//! editor work through the middle of it — the script reload has to sit between
//! the physics phase and the tick, beside the watchers and the renderer's
//! extent — and lifting the game half out would restructure the one frame
//! every session runs. A system added to the app's game half belongs here too.
//!
//! [`App::install_default_resources`]: crate::app::App::install_default_resources
//! [`ScriptHost::begin_test`]: orrin_script::ScriptHost::begin_test
//! [`ScriptHost::resume_test`]: orrin_script::ScriptHost::resume_test

use std::fmt::Write as _;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use orrin_ecs::World;
use orrin_registry::Registry;
use orrin_script::{ScriptHost, TestCase, TestStep};

use crate::scene::input::Spec;
use crate::scene::{
    Actions, DebugLines, GameScreen, InputState, Play, PlayState, Prefabs, Time, UiCanvas,
    propagate_transforms,
};
use crate::scripting::{GameAssemblies, Scripting};

/// The frame every test runs at, in seconds.
pub const FRAME: f32 = 1.0 / 60.0;

/// The screen a test's UI is laid out on.
const SCREEN: [u32; 2] = [1280, 720];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    /// An `Assert` failed.
    Assertion,
    /// The test threw something else.
    Exception,
    /// A Behaviour threw from a hook while the test ran.
    Faulted,
    /// Still running when its frame budget ran out.
    OverBudget,
    /// The method cannot run as a test, or yielded something nothing waits on.
    Malformed,
    /// Its `Scene` could not be loaded.
    Scene,
}

impl FailureKind {
    fn label(self) -> &'static str {
        match self {
            Self::Assertion => "assertion",
            Self::Exception => "exception",
            Self::Faulted => "faulted",
            Self::OverBudget => "over-budget",
            Self::Malformed => "malformed",
            Self::Scene => "scene",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    pub kind: FailureKind,
    pub message: String,
    /// The script file, as the compiler saw it, and the line the failure came
    /// from, when it came from one.
    pub location: Option<(String, u32)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Passed,
    Failed(Failure),
}

#[derive(Clone, Debug)]
pub struct TestResult {
    pub name: String,
    pub outcome: Outcome,
    /// Frames run while the test did; none for a `void` test.
    pub frames: u32,
    pub duration: Duration,
}

impl TestResult {
    pub fn passed(&self) -> bool {
        self.outcome == Outcome::Passed
    }
}

/// Runs the tests of one assembly, each in a fresh world.
pub struct Runner<'a> {
    scripting: &'a Scripting,
    registry: &'a Registry,
    assembly: String,
    assets_dir: Option<PathBuf>,
    input: Vec<(String, Spec)>,
    /// Lent to each test's world and taken back after it, so the project is
    /// scanned once rather than once a test.
    prefabs: Option<Prefabs>,
}

impl<'a> Runner<'a> {
    /// The tests of the live assembly named `assembly`: the game's, which
    /// [`Scripting::game_assembly_name`] gives, or `Orrin` for the bindings'
    /// own.
    pub fn new(scripting: &'a Scripting, registry: &'a Registry, assembly: &str) -> Self {
        Self {
            scripting,
            registry,
            assembly: assembly.to_owned(),
            assets_dir: None,
            input: Vec::new(),
            prefabs: None,
        }
    }

    /// Where a test's `Scene` is found.
    pub fn with_assets_dir(mut self, dir: &Path) -> Self {
        self.assets_dir = Some(dir.to_path_buf());
        self
    }

    /// The project's bindings, which every test's input starts with.
    pub fn with_input(mut self, specs: Vec<(String, Spec)>) -> Self {
        self.input = specs;
        self
    }

    pub fn with_prefabs(mut self, prefabs: Prefabs) -> Self {
        self.prefabs = Some(prefabs);
        self
    }

    /// Every test whose name contains `filter`, or every test, in name order.
    pub fn tests(&self, filter: Option<&str>) -> Vec<TestCase> {
        let mut tests = self.scripting.tests(&self.assembly);
        if let Some(filter) = filter {
            tests.retain(|test| test.name.contains(filter));
        }
        tests
    }

    pub fn run(&mut self, filter: Option<&str>) -> Vec<TestResult> {
        self.run_with(filter, |_| {})
    }

    /// [`run`](Self::run), telling `finished` of each result as it lands.
    pub fn run_with(
        &mut self,
        filter: Option<&str>,
        mut finished: impl FnMut(&TestResult),
    ) -> Vec<TestResult> {
        let tests = self.tests(filter);
        let mut results = Vec::with_capacity(tests.len());
        for test in &tests {
            let result = self.run_one(test);
            finished(&result);
            results.push(result);
        }
        results
    }

    fn run_one(&mut self, test: &TestCase) -> TestResult {
        let started = Instant::now();
        let mut world = self.world();
        let opened = match &test.scene {
            Some(scene) => self.open(&mut world, scene),
            None => Ok(()),
        };
        let (outcome, frames) = match opened {
            Ok(()) => self.drive(&mut world, test),
            Err(message) => (
                Outcome::Failed(Failure {
                    kind: FailureKind::Scene,
                    message,
                    location: None,
                }),
                0,
            ),
        };
        self.prefabs = world.remove_resource::<Prefabs>();
        // Here, and not at the end of the scope: every Behaviour the test made
        // is destroyed before the next one starts.
        drop(world);
        TestResult {
            name: test.name.clone(),
            outcome,
            frames,
            duration: started.elapsed(),
        }
    }

    /// Run `test` to its end: its first turn, then a frame and a turn at a
    /// time until it stops running or its budget is spent.
    fn drive(&self, world: &mut World, test: &TestCase) -> (Outcome, u32) {
        let (scripting, registry) = (self.scripting, self.registry);
        let mut step = scripting.test_turn(world, registry, |host| {
            host.begin_test(&self.assembly, &test.name)
        });
        let mut frames = 0;
        while step == TestStep::Running {
            if frames >= test.max_frames {
                step = scripting.test_turn(world, registry, ScriptHost::abandon_test);
                break;
            }
            self.frame(world);
            frames += 1;
            step = scripting.test_turn(world, registry, ScriptHost::resume_test);
        }
        (self.outcome(step, test), frames)
    }

    fn outcome(&self, step: TestStep, test: &TestCase) -> Outcome {
        let kind = match step {
            TestStep::Passed => return Outcome::Passed,
            TestStep::Missing => {
                return Outcome::Failed(Failure {
                    kind: FailureKind::Malformed,
                    message: format!("{} is no longer in {}", test.name, self.assembly),
                    location: None,
                });
            }
            TestStep::AssertionFailed => FailureKind::Assertion,
            TestStep::Threw => FailureKind::Exception,
            TestStep::Faulted => FailureKind::Faulted,
            TestStep::OverBudget => FailureKind::OverBudget,
            TestStep::Running | TestStep::Malformed => FailureKind::Malformed,
        };
        let (location, message) = self.scripting.host().test_failure();
        Outcome::Failed(Failure {
            kind,
            message,
            location,
        })
    }

    /// The world a test starts in.
    fn world(&mut self) -> World {
        let mut world = World::default();
        crate::app::App::install_default_resources(&mut world);
        crate::scene::entities::install_library(
            &mut world,
            &mut crate::gfx::HeadlessBackend::new(),
        );
        world.insert_resource(GameScreen::window(SCREEN));
        world.resource_mut::<Play>().set(PlayState::Playing);
        world.resource_mut::<Actions>().apply(self.input.clone());
        if let Some(prefabs) = self.prefabs.take() {
            world.insert_resource(prefabs);
        }
        world
    }

    /// Load `scene`, relative to the assets directory, as the active scene,
    /// and create the Behaviours it carries.
    fn open(&self, world: &mut World, scene: &str) -> Result<(), String> {
        let Some(dir) = &self.assets_dir else {
            return Err(format!(
                "scene `{scene}`: there is no assets directory to find it in"
            ));
        };
        let relative = Path::new(scene);
        if !relative
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
        {
            return Err(format!(
                "scene `{scene}` must be a path inside the assets directory, with no `..`"
            ));
        }
        let path = dir.join(relative);
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("scene {}: could not read it: {error}", path.display()))?;
        let document = orrin_registry::parse(&text)
            .map_err(|error| format!("scene {}: {error}", path.display()))?;
        let name = crate::scene::loaded::name_of(scene);
        for issue in crate::scene::loaded::load(world, self.registry, &document, name, None) {
            tracing::warn!("scene {}: {issue}", path.display());
        }
        self.scripting.attach_pending(world, self.registry);
        Ok(())
    }

    /// One frame of a playing game, as `app.rs` runs its game half.
    fn frame(&self, world: &mut World) {
        let (scripting, registry) = (self.scripting, self.registry);
        let delta = crate::scene::advance_time(world, FRAME);
        {
            let input = world.resource::<InputState>();
            world.resource_mut::<Actions>().resolve(&input);
        }
        for warning in world.resource_mut::<Actions>().take_warnings() {
            tracing::warn!("{warning}");
        }
        let simulate = world.resource_mut::<Play>().advance();
        if simulate {
            crate::systems::spin(world, delta);
        }
        propagate_transforms(world);
        if simulate {
            scripting.run_physics(world, registry, delta);
            let size = world.resource::<GameScreen>().size;
            world
                .resource_mut::<UiCanvas>()
                .begin(size.map(|side| side.round() as u32));
            crate::ui::events::update(world);
        }
        scripting.attach_pending(world, registry);
        if simulate {
            scripting.tick(world, registry, delta);
            crate::ui::tween::advance(world, delta);
        }
        propagate_transforms(world);
        if world.resource::<Play>().is_live() {
            crate::scene::follow_active_camera(world);
        }
        crate::ui::update(world, SCREEN);
        let now = world.resource::<Time>().elapsed_time();
        world.resource_mut::<DebugLines>().sweep(now);
        world.resource_mut::<InputState>().end_frame();
    }
}

/// How the results are printed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// A line a test as each finishes, and a summary.
    Text,
    /// One JSON document once every test has run, for CI.
    Json,
}

/// What `--test` was given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    pub filter: Option<String>,
    pub format: Format,
}

impl Options {
    /// The arguments after `--test`: `[FILTER] [--format text|json]`.
    pub fn parse(args: &[String]) -> Result<Self, String> {
        let mut options = Self {
            filter: None,
            format: Format::Text,
        };
        let mut args = args.iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--format" => {
                    options.format = match args.next().map(String::as_str) {
                        Some("text") => Format::Text,
                        Some("json") => Format::Json,
                        Some(other) => {
                            return Err(format!("`--format {other}`: use text or json"));
                        }
                        None => return Err("`--format` needs a value: text or json".to_owned()),
                    }
                }
                flag if flag.starts_with('-') => {
                    return Err(format!(
                        "`{flag}` is not a test option; use `--test [FILTER] [--format text|json]`"
                    ));
                }
                filter if options.filter.is_none() => options.filter = Some(filter.to_owned()),
                extra => {
                    return Err(format!(
                        "`{extra}`: one filter at most, and it matches any part of a test's name"
                    ));
                }
            }
        }
        Ok(options)
    }
}

/// 0 when every test that ran passed, 1 when one failed or a filter matched
/// nothing — a filter that runs no test is a typo, and CI should say so.
pub fn exit_code(results: &[TestResult], filter: Option<&str>) -> i32 {
    let failed = results.iter().any(|result| !result.passed());
    let unmatched = filter.is_some() && results.is_empty();
    i32::from(failed || unmatched)
}

/// The engine binary's `--test` mode: boot scripting with no window, run the
/// project's tests, print the results, and return the process's exit code — 2
/// when the tests could not be run at all.
pub fn run_tool(args: &[String]) -> i32 {
    let options = match Options::parse(args) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("orrin-core: {message}");
            return 2;
        }
    };
    let _console = crate::logging::init(crate::logging::LogConfig {
        console: false,
        ..crate::logging::LogConfig::for_this_build()
    });
    crate::threads::init();

    let project = match std::env::current_dir()
        .map_err(|error| format!("cannot read the current directory: {error}"))
        .and_then(|cwd| orrin_project::Project::locate(&cwd).map_err(|error| error.to_string()))
    {
        Ok(project) => project,
        Err(error) => {
            eprintln!("orrin-core: {error}");
            return 2;
        }
    };
    let assemblies = match GameAssemblies::locate(project.as_ref()) {
        Ok(found) => found,
        Err(reason) => {
            eprintln!("orrin-core: cannot run tests: {reason}");
            return 2;
        }
    };
    // `boot` has already said why when it fails.
    let Some(scripting) = Scripting::boot(&assemblies.bindings_dir, &assemblies.game_dll) else {
        return 2;
    };
    let mut registry = Registry::new();
    crate::scene::register_components(&mut registry);
    scripting.register_components(&mut registry);

    let assets_dir = project
        .as_ref()
        .map_or_else(|| PathBuf::from("assets"), |project| project.assets_dir());
    let input = match project
        .as_ref()
        .map(|project| project.root().join(crate::scene::input::config::FILE_NAME))
        .filter(|path| path.exists())
    {
        Some(path) => crate::scene::input::config::load(&path).unwrap_or_else(|error| {
            tracing::error!("{error}");
            Vec::new()
        }),
        None => Vec::new(),
    };
    let (prefabs, issues) = Prefabs::scan(&assets_dir);
    for issue in issues {
        tracing::warn!("{issue}");
    }

    let assembly = scripting
        .game_assembly_name()
        .unwrap_or_default()
        .to_owned();
    let mut runner = Runner::new(&scripting, &registry, &assembly)
        .with_assets_dir(&assets_dir)
        .with_input(input)
        .with_prefabs(prefabs);
    let filter = options.filter.as_deref();
    let count = runner.tests(filter).len();
    let started = Instant::now();
    let results = match options.format {
        Format::Text => {
            println!("running {count} test{} from {assembly}", plural(count));
            runner.run_with(filter, |result| print!("{}", line(result)))
        }
        Format::Json => runner.run(filter),
    };
    let elapsed = started.elapsed();
    drop(runner);

    match options.format {
        Format::Text => println!("{}", summary(&results, filter, elapsed)),
        Format::Json => println!("{}", json(&results)),
    }
    exit_code(&results, filter)
}

fn plural(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

/// `ok` or `FAILED` and the test's name, then for a failure what failed and
/// where.
fn line(result: &TestResult) -> String {
    let mut out = String::new();
    let frames = match result.frames {
        0 => String::new(),
        1 => " (1 frame)".to_owned(),
        n => format!(" ({n} frames)"),
    };
    match &result.outcome {
        Outcome::Passed => {
            let _ = writeln!(out, "ok      {}{frames}", result.name);
        }
        Outcome::Failed(failure) => {
            let _ = writeln!(out, "FAILED  {}{frames}", result.name);
            let _ = write!(out, "        {}", failure.kind.label());
            if let Some((file, line)) = &failure.location {
                let _ = write!(out, " at {}:{line}", shown(file));
            }
            let _ = writeln!(out, ": {}", failure.message);
        }
    }
    out
}

fn summary(results: &[TestResult], filter: Option<&str>, elapsed: Duration) -> String {
    let failed = results.iter().filter(|result| !result.passed()).count();
    let passed = results.len() - failed;
    if let Some(filter) = filter
        && results.is_empty()
    {
        return format!("\nno test matched `{filter}`");
    }
    format!(
        "\ntest result: {}. {passed} passed; {failed} failed; finished in {:.2}s",
        if failed == 0 { "ok" } else { "FAILED" },
        elapsed.as_secs_f64()
    )
}

/// `path` relative to the working directory when it is under it.
fn shown(path: &str) -> String {
    std::env::current_dir()
        .ok()
        .and_then(|cwd| {
            Path::new(path)
                .strip_prefix(cwd)
                .ok()
                .map(|relative| relative.display().to_string())
        })
        .unwrap_or_else(|| path.to_owned())
}

/// Every result as one JSON object. Written by hand: a few strings and numbers
/// do not justify a serializer in the engine's dependencies.
fn json(results: &[TestResult]) -> String {
    let failed = results.iter().filter(|result| !result.passed()).count();
    let mut out = format!(
        "{{\"passed\":{},\"failed\":{failed},\"tests\":[",
        results.len() - failed
    );
    for (index, result) in results.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let _ = write!(
            out,
            "{{\"name\":{},\"frames\":{},\"seconds\":{:.6},",
            quoted(&result.name),
            result.frames,
            result.duration.as_secs_f64()
        );
        match &result.outcome {
            Outcome::Passed => out.push_str("\"outcome\":\"passed\"}"),
            Outcome::Failed(failure) => {
                let _ = write!(
                    out,
                    "\"outcome\":\"failed\",\"kind\":\"{}\",\"message\":{}",
                    failure.kind.label(),
                    quoted(&failure.message)
                );
                if let Some((file, line)) = &failure.location {
                    let _ = write!(out, ",\"file\":{},\"line\":{line}", quoted(file));
                }
                out.push('}');
            }
        }
    }
    out.push_str("]}");
    out
}

fn quoted(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|arg| (*arg).to_owned()).collect()
    }

    fn result(name: &str, outcome: Outcome) -> TestResult {
        TestResult {
            name: name.to_owned(),
            outcome,
            frames: 3,
            duration: Duration::ZERO,
        }
    }

    fn failed(message: &str) -> Outcome {
        Outcome::Failed(Failure {
            kind: FailureKind::Assertion,
            message: message.to_owned(),
            location: Some(("/game/scripts/Tests.cs".to_owned(), 14)),
        })
    }

    #[test]
    fn test_mode_takes_a_filter_and_a_format() {
        assert_eq!(
            Options::parse(&[]).unwrap(),
            Options {
                filter: None,
                format: Format::Text
            }
        );
        assert_eq!(
            Options::parse(&args(&["--format", "json", "Falls"])).unwrap(),
            Options {
                filter: Some("Falls".to_owned()),
                format: Format::Json
            }
        );
        for bad in [
            &["--format"][..],
            &["--format", "xml"],
            &["One", "Two"],
            &["--verbose"],
        ] {
            assert!(Options::parse(&args(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_failure_or_a_filter_that_matches_nothing_exits_non_zero() {
        let pass = result("A", Outcome::Passed);
        assert_eq!(exit_code(&[], None), 0, "a project with no tests passes");
        assert_eq!(exit_code(std::slice::from_ref(&pass), Some("A")), 0);
        assert_eq!(exit_code(&[pass, result("B", failed("no"))], None), 1);
        assert_eq!(exit_code(&[], Some("Typo")), 1);
    }

    #[test]
    fn the_json_report_escapes_what_a_message_can_hold() {
        let report = json(&[
            result("A", Outcome::Passed),
            result("B", failed("expected \"x\"\nwas\t\\y\u{1}")),
        ]);
        assert_eq!(
            report,
            "{\"passed\":1,\"failed\":1,\"tests\":[\
             {\"name\":\"A\",\"frames\":3,\"seconds\":0.000000,\"outcome\":\"passed\"},\
             {\"name\":\"B\",\"frames\":3,\"seconds\":0.000000,\"outcome\":\"failed\",\
             \"kind\":\"assertion\",\"message\":\"expected \\\"x\\\"\\nwas\\t\\\\y\\u0001\",\
             \"file\":\"/game/scripts/Tests.cs\",\"line\":14}]}"
        );
    }

    #[test]
    fn a_failure_prints_its_kind_place_and_message_under_its_name() {
        assert_eq!(
            line(&result("Game.Tests.Falls", Outcome::Passed)),
            "ok      Game.Tests.Falls (3 frames)\n"
        );
        assert_eq!(
            line(&result("Game.Tests.Falls", failed("expected 1, was 2"))),
            "FAILED  Game.Tests.Falls (3 frames)\n        \
             assertion at /game/scripts/Tests.cs:14: expected 1, was 2\n"
        );
    }
}
