//! End-to-end Behaviour lifecycle test: boots CoreCLR against the built
//! `Orrin` assembly, drives the dispatch entry points the way the engine's
//! script tick does, and asserts hook ordering through a captured log
//! callback (scripts log via the `OrrinApi` table, so the test supplies its
//! own `log` and reads the calls back).
//!
//! Everything lives in one `#[test]`: CoreCLR can only boot once per process.
//!
//! Requires `dotnet build scripting/Orrin` first. Skips with a note when the
//! assembly is missing — this crate is workspace-excluded, so CI never runs it.

use std::ffi::{c_char, CStr, CString};
use std::path::PathBuf;
use std::sync::Mutex;

use orrin_script::{CEntity, GameAssemblyStatus, ScriptHost};

static LOGS: Mutex<Vec<String>> = Mutex::new(Vec::new());

extern "C" fn capture_log(message: *const c_char) {
    if message.is_null() {
        return;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    LOGS.lock().unwrap().push(text.into_owned());
}

/// Drain and return everything logged since the last call.
fn take_logs() -> Vec<String> {
    std::mem::take(&mut *LOGS.lock().unwrap())
}

/// `ORRIN_SCRIPT_DIR` override, else probe the C# build output relative to
/// this crate's root (where `cargo test` runs).
fn assembly_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("ORRIN_SCRIPT_DIR") {
        return Some(PathBuf::from(dir));
    }
    let bin = PathBuf::from("../../scripting/Orrin/bin");
    for config in ["Debug", "Release"] {
        let Ok(entries) = std::fs::read_dir(bin.join(config)) else {
            continue;
        };
        for entry in entries.flatten() {
            let dir = entry.path();
            if dir.join("Orrin.dll").is_file() && dir.join("Orrin.runtimeconfig.json").is_file()
            {
                return Some(dir);
            }
        }
    }
    None
}

/// The demo game assembly, built by `dotnet build scripting/DemoGame`. Absent
/// on a fresh checkout, so the game-assembly assertions skip rather than fail.
fn game_assembly() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("ORRIN_GAME_DLL") {
        return Some(PathBuf::from(path));
    }
    let bin = PathBuf::from("../../scripting/DemoGame/bin");
    for config in ["Debug", "Release"] {
        let Ok(entries) = std::fs::read_dir(bin.join(config)) else {
            continue;
        };
        for entry in entries.flatten() {
            let dll = entry.path().join("DemoGame.dll");
            if dll.is_file() {
                return Some(dll);
            }
        }
    }
    None
}

fn create(host: &ScriptHost, type_name: &str) -> u64 {
    let name = CString::new(type_name).unwrap();
    host.create(
        CEntity {
            index: 1,
            generation: 0,
        },
        &name,
    )
}

#[test]
fn behaviour_lifecycle() {
    let Some(dir) = assembly_dir() else {
        eprintln!("skipping: no built Orrin assembly (run `dotnet build scripting/Orrin`)");
        return;
    };
    let api = orrin_script::OrrinApi {
        log: capture_log,
        ..orrin_script::default_api()
    };
    let host = ScriptHost::boot(&api, &dir).expect("CoreCLR failed to boot");
    take_logs(); // discard boot chatter

    let probe = "Orrin.Tests.LifecycleProbe, Orrin";

    // Full happy path, plus redundant transitions being no-ops. The final
    // destroy happens while inactive, so no second OnDisable is owed.
    let handle = create(&host, probe);
    assert_ne!(handle, 0, "probe type not found in Orrin.dll");
    host.enable(handle);
    host.start(handle);
    host.update(handle, 0.016);
    host.enable(handle); // already active: OnEnable must not re-fire
    host.disable(handle);
    host.disable(handle); // already inactive: OnDisable must not re-fire
    orrin_script::destroy_handle(handle);
    assert_eq!(
        take_logs(),
        [
            "probe:ctor",
            "probe:OnEnable",
            "probe:OnStart",
            "probe:OnUpdate",
            "probe:OnDisable",
            "probe:OnDestroy",
        ],
    );

    // Destroy while still active owes OnDisable first, then OnDestroy, then
    // the free — the ordering guarantee this branch exists to establish.
    let handle = create(&host, probe);
    host.enable(handle);
    orrin_script::destroy_handle(handle);
    assert_eq!(
        take_logs(),
        ["probe:ctor", "probe:OnEnable", "probe:OnDisable", "probe:OnDestroy"],
    );

    // A throwing user constructor is contained: 0 handle, logged, no abort.
    let handle = create(&host, "Orrin.Tests.ThrowingConstructor, Orrin");
    assert_eq!(handle, 0);
    let logs = take_logs();
    assert!(
        logs.iter().any(|l| l.contains("exception during create")),
        "expected contained create exception, got {logs:?}"
    );

    // A throwing OnDestroy is contained; reaching the next assertion at all
    // proves the exception never crossed the native boundary.
    let handle = create(&host, "Orrin.Tests.ThrowingDestroy, Orrin");
    assert_ne!(handle, 0);
    orrin_script::destroy_handle(handle);
    let logs = take_logs();
    assert!(
        logs.iter().any(|l| l.contains("exception during destroy")),
        "expected contained destroy exception, got {logs:?}"
    );

    // The fault channel. A clean OnUpdate reports no fault; a throwing one is
    // contained, reports the fault back (so the engine can disable the script),
    // and logs naming the offending type and hook.
    let handle = create(&host, probe);
    host.enable(handle);
    assert!(!host.update(handle, 0.016), "a clean OnUpdate must not report a fault");
    orrin_script::destroy_handle(handle);
    take_logs(); // discard the probe's own hook chatter

    let handle = create(&host, "Orrin.Tests.ThrowingUpdate, Orrin");
    assert_ne!(handle, 0);
    assert!(host.update(handle, 0.016), "a throwing OnUpdate must report a fault");
    let logs = take_logs();
    assert!(
        logs.iter().any(|l| l.contains("ThrowingUpdate") && l.contains("OnUpdate")),
        "expected a fault log naming the script type and hook, got {logs:?}"
    );
    orrin_script::destroy_handle(handle);

    state_preservation(&host);
    game_assembly_swap(&host);
}

const STATE_PRESERVATION_IMPLEMENTED: bool = true;

/// Behaviour state must survive the destroy/re-create pair a reload performs:
/// capture before the handle is freed, apply onto the fresh instance. Uses a
/// probe in the bindings assembly, so it exercises the property bag without
/// needing a game assembly at all.
fn state_preservation(host: &ScriptHost) {
    let probe = "Orrin.Tests.StatefulProbe, Orrin";

    // Only fields that deviate from their constructor value are captured, so a
    // behaviour that has never run yields no snapshot at all. That is what lets
    // an edited field initializer take effect across a reload instead of being
    // overwritten by the previous build's default.
    let untouched = create(host, probe);
    assert_ne!(untouched, 0, "StatefulProbe not found in Orrin.dll");
    assert_eq!(
        host.capture_state(untouched),
        0,
        "a behaviour still at its constructor values has nothing worth preserving"
    );
    orrin_script::destroy_handle(untouched);
    take_logs();

    let handle = create(host, probe);
    assert_ne!(handle, 0, "StatefulProbe not found in Orrin.dll");
    host.enable(handle);
    host.update(handle, 0.016);
    host.update(handle, 0.016);
    let logs = take_logs();
    assert!(
        logs.iter().any(|l| l.contains("counter=2") && l.contains("scratch=2")),
        "probe should have ticked twice before capture, got {logs:?}"
    );

    let snapshot = host.capture_state(handle);
    orrin_script::destroy_handle(handle);

    if !STATE_PRESERVATION_IMPLEMENTED {
        eprintln!("skipping state preservation: BehaviourState is still a scaffold");
        take_logs();
        return;
    }

    assert_ne!(snapshot, 0, "a probe with capturable fields must yield a snapshot");
    let handle = create(host, probe);
    assert!(host.apply_state(handle, snapshot), "snapshot should restore onto a fresh instance");
    host.enable(handle);
    host.update(handle, 0.016);

    let logs = take_logs();
    let state = logs
        .iter()
        .find(|l| l.starts_with("probe:state"))
        .unwrap_or_else(|| panic!("no state line after reload, got {logs:?}"));

    // Counter, Label and Offset carry over, so the third tick continues from
    // two rather than restarting. Scratch is `[Transient]` and Fixed is
    // readonly: both come back at their constructor values.
    assert!(state.contains("counter=3"), "Counter must survive the swap: {state}");
    assert!(state.contains("label=tick3"), "Label must survive the swap: {state}");
    assert!(state.contains("offset=3"), "Offset (an Orrin value type) must survive: {state}");
    assert!(state.contains("scratch=1"), "[Transient] fields must reset: {state}");
    assert!(state.contains("fixed=7"), "readonly fields are never written back: {state}");

    orrin_script::destroy_handle(handle);
    take_logs();
}

/// The collectible load context: a game assembly loads, its types resolve
/// (sharing the *one* Orrin.dll in the default context), and once every handle
/// into it is freed the context actually unloads — which is what makes the swap
/// repeatable instead of leaking an assembly per reload.
fn game_assembly_swap(host: &ScriptHost) {
    let Some(dll) = game_assembly() else {
        eprintln!("skipping game assembly swap: no built DemoGame (run `dotnet build scripting/DemoGame`)");
        return;
    };
    let path = CString::new(dll.to_str().expect("utf-8 path")).unwrap();
    // Hover rather than Spinner because the snapshot below has to be non-empty:
    // only fields that deviate from their constructor value are captured, and
    // Hover's `_time` accumulates on the first tick while Spinner never mutates
    // anything.
    let hover = "DemoGame.Hover, DemoGame";

    assert_eq!(host.stage_game(&path), GameAssemblyStatus::Ok);
    assert_eq!(
        host.commit_game(),
        GameAssemblyStatus::Ok,
        "the first commit has nothing to retire"
    );

    // Resolving a type out of the collectible context at all proves the load
    // worked; the behaviour running proves it bound to the *same* Orrin.dll
    // the host holds function pointers into, rather than a second copy.
    let first = create(host, hover);
    assert_ne!(first, 0, "{hover} should resolve from the game assembly");
    host.enable(first);
    assert!(!host.update(first, 0.016), "a game behaviour should tick without faulting");
    take_logs();

    // Snapshot before teardown, exactly as a reload does, and deliberately keep
    // it live across the commit below. A snapshot outlives the assembly it came
    // from, so if `BehaviourState` ever stores a value that is an instance of a
    // game-assembly type — a boxed game enum is the easy mistake — it is a live
    // reference into the retired context and the commit reports `Leaked`.
    // Capturing from a *real* game assembly is the only way to catch that; the
    // in-bindings probe cannot, since its types are never collectible.
    let snapshot = host.capture_state(first);
    assert_ne!(snapshot, 0, "{hover} should have a field that deviates after one tick");

    // The handle is the last reference into the old context; free it and the
    // next commit must retire that context cleanly. A `Leaked` here means
    // something still holds an object from the retired assembly — the failure
    // mode that turns hot reload into a slow memory leak.
    orrin_script::destroy_handle(first);
    assert_eq!(host.stage_game(&path), GameAssemblyStatus::Ok);
    assert_eq!(
        host.commit_game(),
        GameAssemblyStatus::Ok,
        "the retired load context must unload once its handles are freed, and a \
         pending snapshot must not pin it"
    );

    let second = create(host, hover);
    assert_ne!(second, 0, "{hover} should resolve from the swapped-in assembly");
    assert!(
        host.apply_state(second, snapshot),
        "a snapshot taken from the previous assembly must restore onto the new one"
    );
    orrin_script::destroy_handle(second);

    assert_eq!(host.unload_game(), GameAssemblyStatus::Ok);
    assert_eq!(
        host.commit_game(),
        GameAssemblyStatus::NothingStaged,
        "committing with nothing staged is a no-op, not a swap"
    );
    take_logs();
}
