//! The headless test runner, through real CoreCLR: each kind of pass and
//! failure a game's `[Test]` can have, the isolation between tests, the name
//! filter, a scene loaded before a test, and a hot reload after a run still
//! unloading the game's load context.
//!
//! The tests run are the bindings' own, `Orrin.Tests.TestProbes` in
//! `TestProbes.cs`, several of which fail on purpose; what is asserted is the
//! runner's structured result for each, never its printed lines. Then the demo
//! game's real tests, which must pass.
//!
//! Needs both C# assemblies built (`dotnet build scripting/Orrin` and
//! `dotnet build scripting/DemoGame`) and skips without them, as
//! `scripted_scene` does. One test, because CoreCLR boots once per process.

#![cfg(feature = "scripting")]

use std::path::{Path, PathBuf};

use orrin_core::logging::{LogConfig, Stream};
use orrin_core::scene::{Name, Tag, register_components, save};
use orrin_core::scripting::{ReloadOutcome, Scripting};
use orrin_core::testing::{FailureKind, Outcome, Runner, TestResult};
use orrin_ecs::World;
use orrin_registry::Registry;

const PROBES: &str = "Orrin.Tests.TestProbes.";

fn assemblies() -> Option<(PathBuf, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bindings = Scripting::find_bindings_dir(&root.join("scripting/Orrin"))?;
    let game = Scripting::find_game_assembly(&root.join("scripting/DemoGame"), "DemoGame")?;
    Some((bindings, game))
}

/// An assets directory holding the scene `SceneIsLoaded` names: one entity,
/// tagged, written by the engine's own save so the file is a real scene.
fn assets_with_a_scene(registry: &Registry) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("test-runner-assets");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("probe-scenes")).unwrap();
    let mut world = World::new();
    world
        .spawn_entity()
        .with(Name::new("Tagged"))
        .with(Tag::new("from-scene".to_owned()));
    std::fs::write(
        dir.join("probe-scenes/tagged.orrin"),
        save(&mut world, registry),
    )
    .unwrap();
    dir
}

fn result<'a>(results: &'a [TestResult], short: &str) -> &'a TestResult {
    let name = format!("{PROBES}{short}");
    results
        .iter()
        .find(|result| result.name == name)
        .unwrap_or_else(|| panic!("no result for {name} in {results:#?}"))
}

fn failure(result: &TestResult) -> &orrin_core::testing::Failure {
    match &result.outcome {
        Outcome::Failed(failure) => failure,
        Outcome::Passed => panic!("{} passed: {result:#?}", result.name),
    }
}

fn passed(result: &TestResult) {
    assert!(
        matches!(result.outcome, Outcome::Passed),
        "{} failed: {result:#?}",
        result.name
    );
}

#[test]
fn the_runner_runs_a_projects_tests_in_fresh_worlds() {
    let Some((bindings, game)) = assemblies() else {
        eprintln!("skipping: build scripting/Orrin and scripting/DemoGame first");
        return;
    };
    // The probes that fail on purpose log their faults; nothing here reads it.
    let (stream, _console, _) = Stream::new(LogConfig {
        stderr: false,
        ..LogConfig::editor()
    });
    let _subscriber = tracing::subscriber::set_default(stream);

    let scripting = Scripting::boot(&bindings, &game).expect("CoreCLR failed to boot");
    let mut registry = Registry::new();
    register_components(&mut registry);
    scripting.register_components(&mut registry);
    let assets = assets_with_a_scene(&registry);

    every_kind_of_result(&scripting, &registry, &assets);
    the_filter_picks_by_name(&scripting, &registry);
    the_demo_games_tests_pass_and_a_reload_after_them_unloads(&scripting, &mut registry);
}

fn every_kind_of_result(scripting: &Scripting, registry: &Registry, assets: &Path) {
    let mut runner = Runner::new(scripting, registry, "Orrin").with_assets_dir(assets);
    let results = runner.run(None);
    let names: Vec<&str> = results
        .iter()
        .map(|result| result.name.strip_prefix(PROBES).unwrap_or(&result.name))
        .collect();
    assert_eq!(
        names,
        [
            "CoroutineRunsFrames",
            "FailsByAFaultedBehaviour",
            "FailsByAssert",
            "FailsByThrowing",
            "IsolationFirstSpawns",
            "IsolationSecondSeesNothing",
            "NotATest",
            "RunsOutOfFrames",
            "SceneIsLoaded",
            "SceneIsMissing",
            "SyncPasses",
        ],
        "every probe, in name order"
    );

    let sync = result(&results, "SyncPasses");
    passed(sync);
    assert_eq!(sync.frames, 0, "a void test runs no frame");

    let coroutine = result(&results, "CoroutineRunsFrames");
    passed(coroutine);
    assert_eq!(coroutine.frames, 61, "one, then thirty, then half a second");

    let assertion = failure(result(&results, "FailsByAssert"));
    assert_eq!(assertion.kind, FailureKind::Assertion);
    assert_eq!(assertion.message, "expected 1, was 2");
    let (file, line) = assertion.location.clone().expect("the assert's line");
    assert!(file.ends_with("TestProbes.cs"), "{file}");
    assert!(line > 0);

    let thrown = failure(result(&results, "FailsByThrowing"));
    assert_eq!(thrown.kind, FailureKind::Exception);
    assert_eq!(
        thrown.message,
        "InvalidOperationException: thrown on purpose"
    );
    assert!(thrown.location.is_some());

    let faulted = result(&results, "FailsByAFaultedBehaviour");
    assert_eq!(faulted.frames, 1, "failed on the frame the Behaviour threw");
    let faulted = failure(faulted);
    assert_eq!(faulted.kind, FailureKind::Faulted);
    assert_eq!(
        faulted.message,
        "ThrowingUpdate.OnUpdate threw InvalidOperationException: update boom"
    );

    let hung = result(&results, "RunsOutOfFrames");
    assert_eq!(hung.frames, 20, "its own budget, not the default");
    assert_eq!(failure(hung).kind, FailureKind::OverBudget);

    assert_eq!(
        failure(result(&results, "NotATest")).kind,
        FailureKind::Malformed
    );

    passed(result(&results, "IsolationFirstSpawns"));
    passed(result(&results, "IsolationSecondSeesNothing"));

    passed(result(&results, "SceneIsLoaded"));
    let missing = failure(result(&results, "SceneIsMissing"));
    assert_eq!(missing.kind, FailureKind::Scene);
    assert!(
        missing.message.contains("probe-scenes/missing.orrin"),
        "{}",
        missing.message
    );
}

fn the_filter_picks_by_name(scripting: &Scripting, registry: &Registry) {
    let mut runner = Runner::new(scripting, registry, "Orrin");
    let names: Vec<String> = runner
        .run(Some("Isolation"))
        .into_iter()
        .map(|result| result.name)
        .collect();
    assert_eq!(
        names,
        [
            format!("{PROBES}IsolationFirstSpawns"),
            format!("{PROBES}IsolationSecondSeesNothing"),
        ]
    );
    assert!(runner.run(Some("NoTestIsCalledThis")).is_empty());
}

/// The runner holds the game's tests only while one runs, so a reload after a
/// run retires the old context completely.
fn the_demo_games_tests_pass_and_a_reload_after_them_unloads(
    scripting: &Scripting,
    registry: &mut Registry,
) {
    let mut runner = Runner::new(scripting, registry, "DemoGame");
    let results = runner.run(None);
    assert!(!results.is_empty(), "the demo game has tests");
    for result in &results {
        passed(result);
    }
    drop(runner);

    registry.clear_game();
    let outcome = scripting.reload(&mut World::new());
    assert_eq!(
        outcome,
        ReloadOutcome::Swapped {
            restored: 0,
            lost: 0,
            leaked: false
        }
    );
    scripting.register_components(registry);
}
