//! End-to-end Behaviour lifecycle test: boots CoreCLR against the built
//! `Orrin` assembly, drives the dispatch entry points the way the engine's
//! script tick does, and asserts hook ordering through a captured log
//! callback (scripts log via the `OrrinApi` table, so the test supplies its
//! own `log_event` and reads the calls back).
//!
//! Everything lives in one `#[test]`: CoreCLR can only boot once per process.
//!
//! Requires `dotnet build scripting/Orrin` first. Skips with a note when the
//! assembly is missing — this crate is workspace-excluded, so CI never runs it.

use std::ffi::{CStr, CString, c_char};
use std::path::PathBuf;
use std::sync::Mutex;

use orrin_script::{CEntity, CMeshRenderer, CTransform, GameAssemblyStatus, ScriptHost};

static LOGS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Level, script and line of each entry in `LOGS`, index for index.
static ORIGINS: Mutex<Vec<(u32, String, u32)>> = Mutex::new(Vec::new());

extern "C" fn capture_log(level: u32, source: *const c_char, line: u32, message: *const c_char) {
    if message.is_null() {
        return;
    }
    // SAFETY: C# passes valid, null-terminated UTF-8 buffers.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    let source = if source.is_null() {
        String::new()
    } else {
        // SAFETY: as above.
        unsafe { CStr::from_ptr(source) }
            .to_string_lossy()
            .into_owned()
    };
    LOGS.lock().unwrap().push(text.into_owned());
    ORIGINS.lock().unwrap().push((level, source, line));
}

/// Drain and return everything logged since the last call.
fn take_logs() -> Vec<String> {
    ORIGINS.lock().unwrap().clear();
    std::mem::take(&mut *LOGS.lock().unwrap())
}

/// `take_logs`, with where each line came from.
fn take_attributed_logs() -> Vec<(u32, String, u32, String)> {
    let origins = std::mem::take(&mut *ORIGINS.lock().unwrap());
    let logs = std::mem::take(&mut *LOGS.lock().unwrap());
    origins
        .into_iter()
        .zip(logs)
        .map(|((level, source, line), text)| (level, source, line, text))
        .collect()
}

/// The engine calls the surface probes made, in order, as the recording stubs
/// below describe them.
static CALLS: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn record(call: String) {
    CALLS.lock().unwrap().push(call);
}

fn take_calls() -> Vec<String> {
    std::mem::take(&mut *CALLS.lock().unwrap())
}

fn named(entity: CEntity) -> String {
    format!("{}v{}", entity.index, entity.generation)
}

fn placed(transform: *const CTransform) -> String {
    // SAFETY: C# passes a valid transform for the duration of the call.
    let [x, y, z] = unsafe { (*transform).position };
    format!("{x} {y} {z}")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Every transform read answers at x = 7 locally and x = 4 in the world.
fn answer(out: *mut CTransform, x: f32) -> bool {
    // SAFETY: C# passes a writable transform for the duration of the call.
    unsafe {
        *out = CTransform {
            position: [x, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }
    };
    true
}

extern "C" fn local_read(_: CEntity, out: *mut CTransform) -> bool {
    answer(out, 7.0)
}

extern "C" fn world_read(_: CEntity, out: *mut CTransform) -> bool {
    answer(out, 4.0)
}

extern "C" fn local_write(entity: CEntity, value: *const CTransform) -> bool {
    record(format!("set_transform {} {}", named(entity), placed(value)));
    true
}

extern "C" fn world_write(entity: CEntity, value: *const CTransform) -> bool {
    record(format!(
        "set_world_transform {} {}",
        named(entity),
        placed(value)
    ));
    true
}

extern "C" fn parent_of(_: CEntity) -> CEntity {
    CEntity {
        index: 6,
        generation: 2,
    }
}

extern "C" fn reparent(child: CEntity, parent: CEntity, keep_world: bool) -> bool {
    record(format!(
        "set_parent {} {} {keep_world}",
        named(child),
        named(parent)
    ));
    true
}

extern "C" fn tag_of(_: CEntity, out: *mut c_char, capacity: i32) -> i32 {
    let tag = b"tagged";
    if capacity >= tag.len() as i32 {
        // SAFETY: C# passes `capacity` writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(tag.as_ptr(), out.cast(), tag.len()) };
    }
    tag.len() as i32
}

fn text(value: *const c_char) -> String {
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned()
}

extern "C" fn retag(entity: CEntity, tag: *const c_char) -> bool {
    record(format!("set_tag {} {}", named(entity), text(tag)));
    true
}

extern "C" fn attach_script(entity: CEntity, type_name: *const c_char) -> bool {
    record(format!("add_script {} {}", named(entity), text(type_name)));
    true
}

extern "C" fn despawn(entity: CEntity) -> bool {
    record(format!("despawn {}", named(entity)));
    true
}

const MESH_RENDERER: u32 = 16;
const SPAWNED: CEntity = CEntity {
    index: 9,
    generation: 1,
};

extern "C" fn spawn_at(_: *const CTransform) -> CEntity {
    record("spawn_with_transform".into());
    SPAWNED
}

fn renderer(value: *const u8, size: u32) -> String {
    assert_eq!(size as usize, size_of::<CMeshRenderer>());
    // SAFETY: C# passes `size` readable bytes, just checked to be a renderer.
    let renderer = unsafe { value.cast::<CMeshRenderer>().read_unaligned() };
    format!("{} {}", hex(&renderer.mesh), hex(&renderer.material))
}

extern "C" fn add_component(entity: CEntity, kind: u32, value: *const u8, size: u32) -> bool {
    record(format!(
        "add_component {} {kind} {}",
        named(entity),
        renderer(value, size)
    ));
    true
}

extern "C" fn set_component(entity: CEntity, kind: u32, value: *const u8, size: u32) -> bool {
    record(format!(
        "set_component {} {kind} {}",
        named(entity),
        renderer(value, size)
    ));
    true
}

/// Only the spawned entity draws anything: the probe's mesh and its first
/// material.
extern "C" fn get_component(entity: CEntity, kind: u32, out: *mut u8, size: u32) -> bool {
    record(format!("get_component {} {kind}", named(entity)));
    if kind != MESH_RENDERER || named(entity) != named(SPAWNED) {
        return false;
    }
    assert_eq!(size as usize, size_of::<CMeshRenderer>());
    let drawn = CMeshRenderer {
        mesh: 0x0011_2233_4455_6677_8899_aabb_ccdd_eeff_u128.to_be_bytes(),
        material: 0x1000_0000_0000_0000_0000_0000_0000_0001_u128.to_be_bytes(),
    };
    // SAFETY: C# passes `size` writable bytes, just checked to be a renderer.
    unsafe { out.cast::<CMeshRenderer>().write_unaligned(drawn) };
    true
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
            if dir.join("Orrin.dll").is_file() && dir.join("Orrin.runtimeconfig.json").is_file() {
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
    create_at(host, type_name, 1, 0)
}

fn create_at(host: &ScriptHost, type_name: &str, index: u32, generation: u32) -> u64 {
    let name = CString::new(type_name).unwrap();
    host.create(CEntity { index, generation }, &name).0
}

#[test]
fn behaviour_lifecycle() {
    let Some(dir) = assembly_dir() else {
        eprintln!("skipping: no built Orrin assembly (run `dotnet build scripting/Orrin`)");
        return;
    };
    let api = orrin_script::OrrinApi {
        log_event: capture_log,
        get_transform: local_read,
        set_transform: local_write,
        get_world_transform: world_read,
        set_world_transform: world_write,
        get_parent: parent_of,
        set_parent: reparent,
        get_tag: tag_of,
        set_tag: retag,
        add_script: attach_script,
        despawn,
        spawn_with_transform: spawn_at,
        get_component,
        set_component,
        add_component,
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

    // A line crosses with where it came from: the script file that logged it,
    // and a line in that file.
    let handle = create(&host, probe);
    orrin_script::destroy_handle(handle);
    let logs = take_attributed_logs();
    let (level, source, line, text) = &logs[0];
    assert_eq!(
        (*level, source.as_str(), text.as_str()),
        (2, "LifecycleProbes", "probe:ctor")
    );
    assert!(*line > 0, "no line number crossed with {text:?}");

    // Destroy while still active owes OnDisable first, then OnDestroy, then
    // the free — the ordering guarantee this branch exists to establish.
    let handle = create(&host, probe);
    host.enable(handle);
    orrin_script::destroy_handle(handle);
    assert_eq!(
        take_logs(),
        [
            "probe:ctor",
            "probe:OnEnable",
            "probe:OnDisable",
            "probe:OnDestroy"
        ],
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
    assert!(
        !host.update(handle, 0.016),
        "a clean OnUpdate must not report a fault"
    );
    orrin_script::destroy_handle(handle);
    take_logs(); // discard the probe's own hook chatter

    let handle = create(&host, "Orrin.Tests.ThrowingUpdate, Orrin");
    assert_ne!(handle, 0);
    assert!(
        host.update(handle, 0.016),
        "a throwing OnUpdate must report a fault"
    );
    let logs = take_attributed_logs();
    assert!(
        logs.iter()
            .any(|(_, _, _, l)| l.contains("ThrowingUpdate") && l.contains("OnUpdate")),
        "expected a fault log naming the script type and hook, got {logs:?}"
    );
    // A fault is an error, and it belongs to the Behaviour that threw rather
    // than to the bindings file that caught it.
    assert!(
        logs.iter()
            .any(|(level, source, _, _)| *level == 4 && source == "ThrowingUpdate"),
        "expected the fault as an error attributed to the script, got {logs:?}"
    );
    orrin_script::destroy_handle(handle);

    behaviour_index(&host);
    entity_members(&host);
    asset_references(&host);
    timers(&host);
    pointer_events(&host);
    state_preservation(&host);
    game_assembly_swap(&host);
}

/// One Behaviour reaches another by entity or by type, through an index that
/// holds handles rather than objects: an entry appears at create and is gone
/// at destroy, so nothing resolves a freed handle — including one the runtime
/// has since handed to a different Behaviour.
fn behaviour_index(host: &ScriptHost) {
    let probe = create_at(host, "Orrin.Tests.IndexProbe, Orrin", 1, 0);
    let marker = create_at(host, "Orrin.Tests.IndexMarker, Orrin", 2, 0);
    assert_ne!(probe, 0, "IndexProbe not found in Orrin.dll");
    assert_ne!(marker, 0, "IndexMarker not found in Orrin.dll");
    take_logs();

    assert!(!host.update(probe, 0.016), "the index probe faulted");
    assert_eq!(
        take_logs(),
        ["probe:index get=2 try=True:2 respawned=none all=2 first=2 self=True wrongType=none"],
        "a live Behaviour is found by its entity and by its type, and only at its own generation"
    );

    orrin_script::destroy_handle(marker);
    take_logs();
    assert!(!host.update(probe, 0.016), "the index probe faulted");
    assert_eq!(
        take_logs(),
        [
            "probe:index get=none try=False:none respawned=none all= first=none self=True wrongType=none"
        ],
        "a destroyed Behaviour must leave the index with its handle"
    );

    let elsewhere = create_at(host, "Orrin.Tests.IndexMarker, Orrin", 3, 0);
    take_logs();
    assert!(!host.update(probe, 0.016), "the index probe faulted");
    assert_eq!(
        take_logs(),
        [
            "probe:index get=none try=False:none respawned=none all=3 first=3 self=True wrongType=none"
        ],
        "a new Behaviour is found where it is, never through a stale entry for entity 2"
    );

    orrin_script::destroy_handle(elsewhere);
    orrin_script::destroy_handle(probe);
    take_logs();
}

/// An entity's own members reach the same engine calls the Behaviour's
/// protected ones and `World` do, on whichever entity they are asked of.
fn entity_members(host: &ScriptHost) {
    let handle = create_at(host, "Orrin.Tests.EntityProbe, Orrin", 5, 1);
    assert_ne!(handle, 0, "EntityProbe not found in Orrin.dll");
    take_logs();
    take_calls();

    assert!(!host.update(handle, 0.016), "the entity probe faulted");
    assert_eq!(
        take_logs(),
        [
            "probe:entity local=7 world=4 parent=Entity(6v2) tag=tagged",
            "probe:entity reparent=True tag=True script=True despawn=True",
        ]
    );
    let calls = take_calls();
    assert_eq!(
        calls[..4],
        [
            "set_transform 6v2 1 2 3",
            "set_world_transform 6v2 1 2 3",
            "set_parent 5v1 6v2 false",
            "set_tag 5v1 hero",
        ]
    );
    assert!(
        calls[4].starts_with("add_script 5v1 Orrin.Tests.LifecycleProbe, Orrin,"),
        "AddScript<T> names T assembly-qualified: {}",
        calls[4]
    );
    assert_eq!(calls[5..], ["despawn 5v1"]);

    orrin_script::destroy_handle(handle);
    take_logs();
}

/// The asset-reference spawn and material swap ride on the MeshRenderer
/// component calls: a spawn at a transform plus a queued renderer, and a read
/// of the renderer written back with only its material changed. A null
/// reference spawns nothing at all.
fn asset_references(host: &ScriptHost) {
    let handle = create(host, "Orrin.Tests.AssetProbe, Orrin");
    assert_ne!(handle, 0, "AssetProbe not found in Orrin.dll");
    take_logs();
    take_calls();

    assert!(!host.update(handle, 0.016), "the asset probe faulted");
    assert_eq!(
        take_logs(),
        [
            "World.SpawnRenderable: MeshRenderer.Material is the null reference, so nothing was spawned",
            "probe:asset spawned=Entity(9v1) refused=Entity(0v0) swapped=True bare=False",
        ]
    );
    let mesh = "00112233445566778899aabbccddeeff";
    let first = "10000000000000000000000000000001";
    let second = "20000000000000000000000000000002";
    assert_eq!(
        take_calls(),
        [
            "spawn_with_transform".to_owned(),
            format!("add_component 9v1 16 {mesh} {first}"),
            "get_component 9v1 16".to_owned(),
            format!("set_component 9v1 16 {mesh} {second}"),
            "get_component 4v0 16".to_owned(),
        ]
    );

    orrin_script::destroy_handle(handle);
    take_logs();
}

/// `After` fires once, `Every` repeats — at most once a tick, however long the
/// tick — a cancelled timer never fires, and all of it runs before OnUpdate on
/// the delta the engine passed in. A throwing callback is contained and
/// reported as the script's fault, exactly as a throwing OnUpdate is.
fn timers(host: &ScriptHost) {
    let handle = create(host, "Orrin.Tests.TimerProbe, Orrin");
    assert_ne!(handle, 0, "TimerProbe not found in Orrin.dll");
    host.enable(handle);
    host.start(handle);
    take_logs();
    for delta in [0.25, 0.25, 0.25, 0.25, 1.25] {
        assert!(!host.update(handle, delta), "the timer probe faulted");
    }
    assert_eq!(
        take_logs(),
        [
            "probe:update 1",
            "probe:timer once",
            "probe:timer every",
            "probe:update 2",
            "probe:cancel True False pending=False",
            "probe:update 3",
            "probe:timer every",
            "probe:update 4",
            "probe:timer every",
            "probe:update 5",
        ]
    );
    orrin_script::destroy_handle(handle);
    take_logs();

    let throwing = create(host, "Orrin.Tests.ThrowingTimer, Orrin");
    assert_ne!(throwing, 0, "ThrowingTimer not found in Orrin.dll");
    host.enable(throwing);
    host.start(throwing);
    take_logs();
    assert!(
        host.update(throwing, 0.016),
        "a throwing timer callback must report a fault"
    );
    let logs = take_attributed_logs();
    assert!(
        logs.iter().any(|(level, source, _, text)| *level == 4
            && source == "ThrowingTimer"
            && text.contains("ThrowingTimer.After")
            && text.contains("timer boom")),
        "expected the fault as an error naming the script and its timer, got {logs:?}"
    );
    assert!(
        !logs
            .iter()
            .any(|(_, _, _, text)| text == "probe:throwing-timer update"),
        "OnUpdate ran after its timer faulted the script: {logs:?}"
    );
    orrin_script::destroy_handle(throwing);
    take_logs();
}

/// A pointer event reaches a behaviour that handles it with its payload, is
/// reported unhandled by one that does not — which is what sends it on up the
/// hierarchy — and a throwing handler is contained and reported as a fault.
fn pointer_events(host: &ScriptHost) {
    use orrin_script::{CPointerEvent, Delivery};
    const DOWN: u32 = 2;
    const CLICK: u32 = 4;
    let event = CPointerEvent {
        position: [12.5, 40.0],
        press_position: [10.0, 30.0],
        target: CEntity {
            index: 9,
            generation: 2,
        },
        button: 0,
    };

    let handle = create(host, "Orrin.Tests.PointerProbe, Orrin");
    assert_ne!(handle, 0);
    take_logs();
    assert_eq!(host.pointer(handle, CLICK, &event), Delivery::Handled);
    assert_eq!(take_logs(), ["probe:click 9 12.5 30"]);
    assert_eq!(host.pointer(handle, DOWN, &event), Delivery::Unhandled);
    orrin_script::destroy_handle(handle);

    let plain = create(host, "Orrin.Tests.LifecycleProbe, Orrin");
    assert_eq!(host.pointer(plain, CLICK, &event), Delivery::Unhandled);
    orrin_script::destroy_handle(plain);

    let throwing = create(host, "Orrin.Tests.ThrowingPointer, Orrin");
    take_logs();
    assert_eq!(host.pointer(throwing, DOWN, &event), Delivery::Faulted);
    let logs = take_logs();
    assert!(
        logs.iter()
            .any(|l| l.contains("ThrowingPointer") && l.contains("OnPointerDown")),
        "expected a fault naming the script and its handler, got {logs:?}"
    );
    orrin_script::destroy_handle(throwing);
    take_logs();
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
        logs.iter()
            .any(|l| l.contains("counter=2") && l.contains("scratch=2")),
        "probe should have ticked twice before capture, got {logs:?}"
    );

    let snapshot = host.capture_state(handle);
    orrin_script::destroy_handle(handle);

    if !STATE_PRESERVATION_IMPLEMENTED {
        eprintln!("skipping state preservation: BehaviourState is still a scaffold");
        take_logs();
        return;
    }

    assert_ne!(
        snapshot, 0,
        "a probe with capturable fields must yield a snapshot"
    );
    let handle = create(host, probe);
    assert!(
        host.apply_state(handle, snapshot),
        "snapshot should restore onto a fresh instance"
    );
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
    assert!(
        state.contains("counter=3"),
        "Counter must survive the swap: {state}"
    );
    assert!(
        state.contains("label=tick3"),
        "Label must survive the swap: {state}"
    );
    assert!(
        state.contains("offset=3"),
        "Offset (an Orrin value type) must survive: {state}"
    );
    assert!(
        state.contains("scratch=1"),
        "[Transient] fields must reset: {state}"
    );
    assert!(
        state.contains("fixed=7"),
        "readonly fields are never written back: {state}"
    );

    orrin_script::destroy_handle(handle);
    take_logs();
}

/// The collectible load context: a game assembly loads, its types resolve
/// (sharing the *one* Orrin.dll in the default context), and once every handle
/// into it is freed the context actually unloads — which is what makes the swap
/// repeatable instead of leaking an assembly per reload.
fn game_assembly_swap(host: &ScriptHost) {
    let Some(dll) = game_assembly() else {
        eprintln!(
            "skipping game assembly swap: no built DemoGame (run `dotnet build scripting/DemoGame`)"
        );
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
    assert!(
        !host.update(first, 0.016),
        "a game behaviour should tick without faulting"
    );
    take_logs();

    // Snapshot before teardown, exactly as a reload does, and deliberately keep
    // it live across the commit below. A snapshot outlives the assembly it came
    // from, so if `BehaviourState` ever stores a value that is an instance of a
    // game-assembly type — a boxed game enum is the easy mistake — it is a live
    // reference into the retired context and the commit reports `Leaked`.
    // Capturing from a *real* game assembly is the only way to catch that; the
    // in-bindings probe cannot, since its types are never collectible.
    let snapshot = host.capture_state(first);
    assert_ne!(
        snapshot, 0,
        "{hover} should have a field that deviates after one tick"
    );

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
    assert_ne!(
        second, 0,
        "{hover} should resolve from the swapped-in assembly"
    );
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
