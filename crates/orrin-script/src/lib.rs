//! C# scripting host for the Orrin engine.
//!
//! [`ScriptHost`] boots CoreCLR, loads the `Orrin` assembly, hands it the engine
//! function table ([`OrrinApi`]), and exposes the lifecycle entry points
//! (`Create`/`Enable`/`Start`/`Update`/`Disable`, plus the global
//! [`destroy_handle`]) the engine drives. Generic, World-only ABI functions
//! live here; component-specific ones are supplied by the engine via
//! `OrrinApi { get_transform, set_transform, ..default_api() }`.

use std::cell::Cell;
use std::ffi::{CStr, c_char};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use netcorehost::hostfxr::{HostfxrContext, InitializedForRuntimeConfig};
use netcorehost::pdcstring::PdCString;
use netcorehost::{nethost, pdcstr};
use orrin_ecs::World;

/// Engine function table handed to managed code at startup. Field order and
/// signatures must stay in lock-step with the C# `Orrin.OrrinApi` struct.
#[repr(C)]
pub struct OrrinApi {
    pub log: extern "C" fn(*const c_char),
    pub spawn: extern "C" fn() -> CEntity,
    pub get_transform: extern "C" fn(CEntity, *mut CTransform) -> bool,
    pub set_transform: extern "C" fn(CEntity, *const CTransform) -> bool,
    // Input polling; key/button codes are the engine's stable numbering (see
    // the renderer's `scene::input::map_key` and C# `Orrin.KeyCode`).
    pub key_down: extern "C" fn(u32) -> bool,
    pub key_pressed: extern "C" fn(u32) -> bool,
    pub key_released: extern "C" fn(u32) -> bool,
    pub mouse_button_down: extern "C" fn(u32) -> bool,
    pub cursor_pos: extern "C" fn(*mut f32, *mut f32),
    // Structural ops. The engine queues these and applies them after the
    // dispatch window closes, but `spawn_renderable` reserves and returns a
    // real entity id immediately.
    pub spawn_renderable: extern "C" fn(*const c_char, *const c_char, *const CTransform) -> CEntity,
    pub despawn: extern "C" fn(CEntity) -> bool,
    // Frame timing, read from the engine's `Time` resource. Engine-side (like
    // input), so the renderer supplies the real implementations.
    pub time_delta: extern "C" fn() -> f32,
    pub time_total: extern "C" fn() -> f32,
    pub time_frame_count: extern "C" fn() -> u64,
    // Entity querying. Engine-side (the `Tag` component and the component-kind
    // numbering live in the renderer). `find_all_by_tag` and `get_tag` write
    // into caller-allocated buffers and return the *total* count/byte length —
    // snprintf semantics, so C# can resize and retry when the buffer was too
    // small. `has_component` takes the kind numbering shared with C#
    // `Orrin.ComponentKind`. `set_tag` is a deferred structural change, like
    // `despawn`.
    pub find_by_tag: extern "C" fn(*const c_char, *mut CEntity) -> bool,
    pub find_all_by_tag: extern "C" fn(*const c_char, *mut CEntity, i32) -> i32,
    pub has_component: extern "C" fn(CEntity, u32) -> bool,
    pub get_tag: extern "C" fn(CEntity, *mut c_char, i32) -> i32,
    pub set_tag: extern "C" fn(CEntity, *const c_char) -> bool,
    // Collision + composition. All four are deferred structural changes (like
    // `set_tag`): they validate eagerly and apply after the dispatch window.
    // Box half extents are passed as three floats to keep the signature
    // blittable-trivial; `bool` is one byte on both sides (C# `byte`).
    pub add_box_collider: extern "C" fn(CEntity, f32, f32, f32, bool) -> bool,
    pub add_sphere_collider: extern "C" fn(CEntity, f32, bool) -> bool,
    pub set_material: extern "C" fn(CEntity, *const c_char) -> bool,
    pub add_script: extern "C" fn(CEntity, *const c_char) -> bool,
    // Developer debug utilities. `log_warn`/`log_error` mirror `log` at higher
    // severity; the engine routes all three into its editor console. Logging is
    // always live. `debug_draw_line` records one world-space line into the
    // engine's per-frame overlay buffer: endpoints, RGBA, then a lifetime in
    // seconds (<= 0 = a single frame). Editor-only — no-op in export builds.
    // The line is passed as loose floats to stay blittable-trivial, the same
    // choice as the collider extents.
    pub log_warn: extern "C" fn(*const c_char),
    pub log_error: extern "C" fn(*const c_char),
    pub debug_draw_line: extern "C" fn(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32),
    // Hierarchy. Appended after `debug_draw_line`; never reordered above it.
    //
    // The world transform is a matrix engine-side, so `get_world_transform`
    // hands back the closest translation/rotation/scale fit to it. That is
    // exact for any chain of rotations and uniform scales, and lossy once a
    // non-uniformly scaled ancestor introduces shear — the same admission
    // Unity's `lossyScale` makes. `set_world_transform` composes with the
    // inverse of the parent's world transform engine-side, so a script never
    // handles a matrix.
    //
    // `get_parent` returns `CEntity::NULL` for a root. `set_parent` is a
    // *structural* change and so is deferred like `set_tag`: it validates
    // eagerly (rejecting a self-parent, a cycle, or a stale handle) and applies
    // after the dispatch window. Passing `CEntity::NULL` as the parent detaches.
    // `keep_world` preserves the child's world transform across the move.
    pub get_world_transform: extern "C" fn(CEntity, *mut CTransform) -> bool,
    pub set_world_transform: extern "C" fn(CEntity, *const CTransform) -> bool,
    pub get_parent: extern "C" fn(CEntity) -> CEntity,
    pub set_parent: extern "C" fn(CEntity, CEntity, bool) -> bool,
    // Named input. Appended after `set_parent`; never reordered above it.
    //
    // `action_id` interns a name and hands back a handle that is stable for the
    // life of the process — reloading a binding file changes what a handle
    // points at, never what it is, so C# may cache one and keep it across a
    // reload. A name nothing binds still gets a handle; it simply answers no.
    // The name is the only string here, crossed once per action rather than on
    // every query.
    //
    // The trailing `u32` is the player slot. Keyboard and mouse bindings answer
    // for slot 0 only; a gamepad binding answers for the pad assigned to that
    // slot. `action_held` and friends read false for an id naming an axis, and
    // `axis_value` reads zero for one naming an action.
    pub action_id: extern "C" fn(*const c_char) -> u32,
    pub action_held: extern "C" fn(u32, u32) -> bool,
    pub action_pressed: extern "C" fn(u32, u32) -> bool,
    pub action_released: extern "C" fn(u32, u32) -> bool,
    pub axis_value: extern "C" fn(u32, u32) -> f32,
    // The mouse's edges, appended after the named input. `mouse_button_down`
    // above is level state; these two are the one-frame edges its keyboard
    // counterparts have had since the beginning.
    pub mouse_button_pressed: extern "C" fn(u32) -> bool,
    pub mouse_button_released: extern "C" fn(u32) -> bool,
    // Structured logging, appended after the mouse edges. One entry for every
    // level, and the one the bindings call: `log`, `log_warn` and `log_error`
    // above predate it and stay only because the table never shrinks.
    //
    // Level (0 = trace … 4 = error, C# `Orrin.LogLevel`), the script's name or
    // null, its line or 0, and the message.
    pub log_event: extern "C" fn(u32, *const c_char, u32, *const c_char),
    // Assets, appended after `log_event`; never reordered above it.
    //
    // An asset id crosses as 16 bytes in canonical UUID order, through a buffer
    // the caller owns. `find_asset` resolves a path relative to the assets
    // directory, writes the id and returns the asset's kind (`ASSET_*`), or
    // returns `ASSET_NONE` and writes nothing. `instantiate_prefab` reserves and
    // returns the root at once and builds the tree after the dispatch window,
    // like `spawn_renderable`; it returns `CEntity::NULL` for an id that names
    // no prefab.
    pub find_asset: extern "C" fn(*const c_char, *mut u8) -> u32,
    pub instantiate_prefab: extern "C" fn(*const u8) -> CEntity,
    // Load state, appended after `instantiate_prefab`; never reordered above
    // it. Takes an id's 16 bytes and returns `LOAD_*`: whether what the id
    // draws is the asset yet, a placeholder while it imports, or a placeholder
    // for good.
    pub asset_state: extern "C" fn(*const u8) -> u32,
    // Scenes, appended after `asset_state`; never reordered above it.
    //
    // A scene is named by its asset id's 16 bytes, a loaded one by a `u32`
    // handle that is never zero and never reused. All three changes are
    // deferred structural changes, validated at the call: `load_scene` returns
    // false and `load_scene_additive` zero for an id that names no scene, and
    // `unload_scene` false for the active scene or one that is not loaded.
    // `load_scene_additive` reserves the handle at the call, like
    // `instantiate_prefab` its root, so the same tick can hold and unload it.
    pub load_scene: extern "C" fn(*const u8) -> bool,
    pub load_scene_additive: extern "C" fn(*const u8) -> u32,
    pub unload_scene: extern "C" fn(u32) -> bool,
    pub active_scene: extern "C" fn() -> u32,
    // The camera the frame is drawn from, as a position and rotation (scale is
    // ignored and read back as one). Both return false when there is no camera
    // to read or write.
    pub get_camera: extern "C" fn(*mut CTransform) -> bool,
    pub set_camera: extern "C" fn(*const CTransform) -> bool,
    // Camera entities, appended after `set_camera`; never reordered above it.
    //
    // `add_camera` is structural and deferred like `add_box_collider`; the
    // other two read and write a camera's lens in place. All three refuse a
    // lens that cannot project (see `CameraComponent::check`). While a camera
    // entity is active, `get_camera`/`set_camera` above address its world
    // transform rather than the view, which follows it.
    pub add_camera: extern "C" fn(CEntity, *const CCamera) -> bool,
    pub get_camera_lens: extern "C" fn(CEntity, *mut CCamera) -> bool,
    pub set_camera_lens: extern "C" fn(CEntity, *const CCamera) -> bool,
    // The camera the game draws from, or `CEntity::NULL` for none. Selecting
    // `CEntity::NULL` hands the choice back to the cameras' priorities. A
    // camera `add_camera` queued this tick may be selected; anything else that
    // is not a camera is refused.
    pub get_active_camera: extern "C" fn() -> CEntity,
    pub set_active_camera: extern "C" fn(CEntity) -> bool,
    // In-game UI, appended after `set_active_camera`; never reordered above it.
    //
    // Immediate mode: each call draws for the frame it is made in. A rect is
    // four loose floats — x, y, width, height in physical pixels from the
    // window's top left, the cursor's space — and a colour four more, sRGB with
    // straight alpha: the blittable-trivial choice `debug_draw_line` made. Text
    // crosses as a UTF-8 pointer and its byte length, so C# encodes it on the
    // stack with no terminator. A font or texture is its id's 16 bytes, and a
    // null font is the engine's own. `ui_text`'s `u32` after the size is C#
    // `Orrin.TextAnchor`. `ui_screen_size` reports the window the tick began
    // with.
    pub ui_panel: extern "C" fn(f32, f32, f32, f32, f32, f32, f32, f32),
    pub ui_text:
        extern "C" fn(*const u8, u32, *const u8, f32, u32, f32, f32, f32, f32, f32, f32, f32, f32),
    pub ui_image: extern "C" fn(*const u8, f32, f32, f32, f32, f32, f32, f32, f32),
    pub ui_screen_size: extern "C" fn(*mut f32, *mut f32),
    // Engine components as blittable structs, appended after `ui_screen_size`;
    // never reordered above it.
    //
    // One generic pair rather than a pair per component: `kind` is C#
    // `Orrin.ComponentKind`, the numbering `has_component` takes, and the
    // pointer is that kind's `C…` struct below. `size` is the byte size of the
    // struct C# built, checked against the engine's on every call — the table's
    // own size handshake cannot see a struct grow, so a stale bindings build is
    // refused by name here instead of reading past the end of C#'s buffer.
    // `get` and `set` act on the component in place; `add` is structural and
    // deferred like `add_camera`, and a `set` made later in the same tick lands
    // on the queued one. An asset id inside a struct is its 16 bytes in
    // canonical UUID order, all zeroes for none; a `bool` is a `u32`, nonzero
    // for true. Text, which no blittable struct can hold, crosses through
    // `get_component_text` (the `get_tag` protocol: returns the full UTF-8 byte
    // length, writes at most `capacity`, -1 for none) and `set_component_text`
    // (a pointer and a byte length). `ui_pointer_over` is whether the pointer
    // was over a retained raycast target when the tick began.
    pub get_component: extern "C" fn(CEntity, u32, *mut u8, u32) -> bool,
    pub set_component: extern "C" fn(CEntity, u32, *const u8, u32) -> bool,
    pub add_component: extern "C" fn(CEntity, u32, *const u8, u32) -> bool,
    pub get_component_text: extern "C" fn(CEntity, u32, *mut u8, i32) -> i32,
    pub set_component_text: extern "C" fn(CEntity, u32, *const u8, u32) -> bool,
    pub ui_pointer_over: extern "C" fn() -> bool,
    // Tweens, appended after `ui_pointer_over`; never reordered above it.
    //
    // `tween_play` takes a `CTween` and its byte size, checked like a
    // component's, and returns the tween's id as bits — never zero, which is
    // the refusal: a stale entity, a property, style or direction this build
    // cannot name, or a duration or delay that is negative or not finite. A
    // tween is engine data the moment it is started, so unlike `add_component`
    // nothing about it is deferred; it first moves after the tick. Completion
    // comes back through the managed `Orrin.Tweens.Complete`, at the start of
    // the next tick.
    pub tween_play: extern "C" fn(CEntity, *const CTween, u32) -> u64,
    pub tween_cancel: extern "C" fn(u64) -> bool,
    pub tween_playing: extern "C" fn(u64) -> bool,
    // Story controls, appended after the tweens. A story asks for a control
    // by name — UTF-8 and a byte length — with its default, and is handed the
    // value the storybook holds for it, which is the default the first time.
    // Outside a story build each returns its default.
    pub story_float: extern "C" fn(*const u8, u32, f32, f32, f32) -> f32,
    pub story_bool: extern "C" fn(*const u8, u32, bool) -> bool,
    // Physics, appended after the story controls.
    //
    // `add_collider` takes a `CCollider` and its byte size, checked like a
    // component's, and queues the add like `add_box_collider` does; it is what
    // every collider shape goes through, the two older calls included on the C#
    // side. The pushes take a vector and a `ForceMode` (C# `Orrin.ForceMode`)
    // and refuse anything that is not, and will not be, a dynamic body.
    // Velocities pass `angular` as a flag choosing which of the two; the getter
    // writes three floats. `physics_raycast` takes an origin, a direction, a
    // length, a layer mask and whether triggers count, and writes a
    // `CRaycastHit` of the size given.
    pub add_collider: extern "C" fn(CEntity, *const CCollider, u32) -> bool,
    pub physics_add_force: extern "C" fn(CEntity, f32, f32, f32, u32) -> bool,
    pub physics_add_torque: extern "C" fn(CEntity, f32, f32, f32, u32) -> bool,
    pub physics_get_velocity: extern "C" fn(CEntity, bool, *mut [f32; 3]) -> bool,
    pub physics_set_velocity: extern "C" fn(CEntity, bool, f32, f32, f32) -> bool,
    pub physics_raycast:
        extern "C" fn(f32, f32, f32, f32, f32, f32, f32, u32, bool, *mut CRaycastHit, u32) -> bool,
    // Mouse look, appended after physics. `mouse_delta` writes this frame's raw
    // motion, x right and y down, in the device's own units — what a
    // first-person camera turns by, and present whether or not the cursor
    // moves. `set_cursor_locked` is a request; see `scene::input::cursor` for
    // when the window honours it.
    pub mouse_delta: extern "C" fn(*mut [f32; 2]),
    pub set_cursor_locked: extern "C" fn(bool),
    pub cursor_locked: extern "C" fn() -> bool,
    // `spawn` with a transform, appended after mouse look: an empty entity
    // that stands somewhere — a camera, a pivot. Immediate, as `spawn` is.
    pub spawn_with_transform: extern "C" fn(*const CTransform) -> CEntity,
    // Audio, appended after `spawn_with_transform`; never reordered above it.
    //
    // `audio_play` takes a `CSound` and its byte size, checked like a
    // component's, and returns the sound's id as bits — never zero, which is
    // the refusal: a size the engine does not expect, an emitter or rolloff it
    // cannot name, an entity that is not alive, or a clip that is not a sound
    // in the project. A sound is accepted at once and starts once its clip has
    // decoded; a game with no output device accepts it and ends it the next
    // frame. `audio_play_source` plays an entity's `AudioSource` from the
    // start (zero for none, or one with no clip); `audio_play_music` takes a
    // clip's 16 bytes and replaces what the music channel was playing. The
    // volumes are linear gains, clamped at zero.
    pub audio_play: extern "C" fn(*const CSound, u32) -> u64,
    pub audio_stop: extern "C" fn(u64) -> bool,
    pub audio_set_volume: extern "C" fn(u64, f32) -> bool,
    pub audio_playing: extern "C" fn(u64) -> bool,
    pub audio_play_source: extern "C" fn(CEntity) -> u64,
    pub audio_stop_source: extern "C" fn(CEntity) -> bool,
    pub audio_play_music: extern "C" fn(*const u8, bool) -> u64,
    pub audio_stop_music: extern "C" fn(),
    pub audio_set_master_volume: extern "C" fn(f32),
    pub audio_set_music_volume: extern "C" fn(f32),
}

/// The handle `action_id` returns when there is no engine behind the table.
/// Distinct from a real id so a stubbed query cannot be mistaken for action
/// zero.
pub const NO_ACTION: u32 = u32::MAX;

/// What `find_asset` found. Lock-step with C# `Orrin.AssetKinds`: append,
/// never renumber.
pub const ASSET_NONE: u32 = 0;
pub const ASSET_PREFAB: u32 = 1;
pub const ASSET_MESH: u32 = 2;
pub const ASSET_TEXTURE: u32 = 3;
pub const ASSET_MATERIAL: u32 = 4;
pub const ASSET_SCENE: u32 = 5;
pub const ASSET_FONT: u32 = 6;
pub const ASSET_AUDIO: u32 = 7;

/// What `asset_state` reports. Lock-step with C# `Orrin.AssetState`: append,
/// never renumber.
pub const LOAD_LOADING: u32 = 0;
pub const LOAD_READY: u32 = 1;
pub const LOAD_FAILED: u32 = 2;

/// A table with the generic functions wired and the rest stubbed; the engine
/// overrides `get_transform`/`set_transform` and the input functions with real
/// implementations (they touch engine-side resources).
pub fn default_api() -> OrrinApi {
    OrrinApi {
        log: orrin_log,
        spawn: orrin_spawn,
        get_transform: stub_get_transform,
        set_transform: stub_set_transform,
        key_down: stub_key_query,
        key_pressed: stub_key_query,
        key_released: stub_key_query,
        mouse_button_down: stub_key_query,
        cursor_pos: stub_cursor_pos,
        spawn_renderable: stub_spawn_renderable,
        despawn: stub_despawn,
        time_delta: stub_time_seconds,
        time_total: stub_time_seconds,
        time_frame_count: stub_time_frame_count,
        find_by_tag: stub_find_by_tag,
        find_all_by_tag: stub_find_all_by_tag,
        has_component: stub_has_component,
        get_tag: stub_get_tag,
        set_tag: stub_set_tag,
        add_box_collider: stub_add_box_collider,
        add_sphere_collider: stub_add_sphere_collider,
        set_material: stub_set_material,
        add_script: stub_add_script,
        log_warn: orrin_log_warn,
        log_error: orrin_log_error,
        debug_draw_line: stub_debug_draw_line,
        get_world_transform: stub_get_transform,
        set_world_transform: stub_set_transform,
        get_parent: stub_get_parent,
        set_parent: stub_set_parent,
        action_id: stub_action_id,
        action_held: stub_action_query,
        action_pressed: stub_action_query,
        action_released: stub_action_query,
        axis_value: stub_axis_value,
        mouse_button_pressed: stub_key_query,
        mouse_button_released: stub_key_query,
        log_event: default_log_event,
        find_asset: stub_find_asset,
        instantiate_prefab: stub_instantiate_prefab,
        asset_state: stub_asset_state,
        load_scene: stub_load_scene,
        load_scene_additive: stub_load_scene_additive,
        unload_scene: stub_unload_scene,
        active_scene: stub_active_scene,
        get_camera: stub_get_camera,
        set_camera: stub_set_camera,
        add_camera: stub_camera_lens,
        get_camera_lens: stub_get_camera_lens,
        set_camera_lens: stub_camera_lens,
        get_active_camera: stub_get_active_camera,
        set_active_camera: stub_set_active_camera,
        ui_panel: stub_ui_panel,
        ui_text: stub_ui_text,
        ui_image: stub_ui_image,
        ui_screen_size: stub_ui_screen_size,
        get_component: stub_get_component,
        set_component: stub_put_component,
        add_component: stub_put_component,
        get_component_text: stub_get_component_text,
        set_component_text: stub_set_component_text,
        ui_pointer_over: stub_ui_pointer_over,
        tween_play: stub_tween_play,
        tween_cancel: stub_tween_query,
        tween_playing: stub_tween_query,
        story_float: stub_story_float,
        story_bool: stub_story_bool,
        add_collider: stub_add_collider,
        physics_add_force: stub_physics_push,
        physics_add_torque: stub_physics_push,
        physics_get_velocity: stub_physics_get_velocity,
        physics_set_velocity: stub_physics_set_velocity,
        physics_raycast: stub_physics_raycast,
        mouse_delta: stub_mouse_delta,
        set_cursor_locked: stub_set_cursor_locked,
        cursor_locked: stub_cursor_locked,
        spawn_with_transform: stub_spawn_with_transform,
        audio_play: stub_audio_play,
        audio_stop: stub_tween_query,
        audio_set_volume: stub_audio_set_volume,
        audio_playing: stub_tween_query,
        audio_play_source: stub_audio_play_source,
        audio_stop_source: stub_audio_stop_source,
        audio_play_music: stub_audio_play_music,
        audio_stop_music: stub_audio_stop_music,
        audio_set_master_volume: stub_audio_volume,
        audio_set_music_volume: stub_audio_volume,
    }
}

extern "C" fn stub_audio_play(_: *const CSound, _: u32) -> u64 {
    0
}

extern "C" fn stub_audio_set_volume(_: u64, _: f32) -> bool {
    false
}

extern "C" fn stub_audio_play_source(_: CEntity) -> u64 {
    0
}

extern "C" fn stub_audio_stop_source(_: CEntity) -> bool {
    false
}

extern "C" fn stub_audio_play_music(_: *const u8, _: bool) -> u64 {
    0
}

extern "C" fn stub_audio_stop_music() {}

extern "C" fn stub_audio_volume(_: f32) {}

extern "C" fn stub_spawn_with_transform(_: *const CTransform) -> CEntity {
    CEntity::NULL
}

extern "C" fn stub_mouse_delta(out: *mut [f32; 2]) {
    if !out.is_null() {
        // SAFETY: C# passes two writable floats.
        unsafe { out.write_unaligned([0.0; 2]) };
    }
}

extern "C" fn stub_set_cursor_locked(_: bool) {}

extern "C" fn stub_cursor_locked() -> bool {
    false
}

extern "C" fn stub_tween_play(_: CEntity, _: *const CTween, _: u32) -> u64 {
    0
}

extern "C" fn stub_tween_query(_: u64) -> bool {
    false
}

extern "C" fn stub_story_float(_: *const u8, _: u32, default: f32, _: f32, _: f32) -> f32 {
    default
}

extern "C" fn stub_story_bool(_: *const u8, _: u32, default: bool) -> bool {
    default
}

extern "C" fn stub_add_collider(_: CEntity, _: *const CCollider, _: u32) -> bool {
    false
}

extern "C" fn stub_physics_push(_: CEntity, _: f32, _: f32, _: f32, _: u32) -> bool {
    false
}

extern "C" fn stub_physics_get_velocity(_: CEntity, _: bool, _: *mut [f32; 3]) -> bool {
    false
}

extern "C" fn stub_physics_set_velocity(_: CEntity, _: bool, _: f32, _: f32, _: f32) -> bool {
    false
}

#[allow(clippy::too_many_arguments)]
extern "C" fn stub_physics_raycast(
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: u32,
    _: bool,
    _: *mut CRaycastHit,
    _: u32,
) -> bool {
    false
}

extern "C" fn stub_get_component(_: CEntity, _: u32, _: *mut u8, _: u32) -> bool {
    false
}

extern "C" fn stub_put_component(_: CEntity, _: u32, _: *const u8, _: u32) -> bool {
    false
}

extern "C" fn stub_get_component_text(_: CEntity, _: u32, _: *mut u8, _: i32) -> i32 {
    -1
}

extern "C" fn stub_set_component_text(_: CEntity, _: u32, _: *const u8, _: u32) -> bool {
    false
}

extern "C" fn stub_ui_pointer_over() -> bool {
    false
}

extern "C" fn stub_camera_lens(_entity: CEntity, _value: *const CCamera) -> bool {
    false
}

extern "C" fn stub_get_camera_lens(_entity: CEntity, _out: *mut CCamera) -> bool {
    false
}

extern "C" fn stub_get_active_camera() -> CEntity {
    CEntity::NULL
}

extern "C" fn stub_set_active_camera(_entity: CEntity) -> bool {
    false
}

extern "C" fn stub_ui_panel(_: f32, _: f32, _: f32, _: f32, _: f32, _: f32, _: f32, _: f32) {}

#[allow(clippy::too_many_arguments)]
extern "C" fn stub_ui_text(
    _text: *const u8,
    _len: u32,
    _font: *const u8,
    _size: f32,
    _anchor: u32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
) {
}

#[allow(clippy::too_many_arguments)]
extern "C" fn stub_ui_image(
    _texture: *const u8,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
) {
}

extern "C" fn stub_ui_screen_size(width: *mut f32, height: *mut f32) {
    for out in [width, height] {
        if !out.is_null() {
            // SAFETY: C# passes valid, writable f32 pointers.
            unsafe { *out = 0.0 };
        }
    }
}

extern "C" fn stub_load_scene(_id: *const u8) -> bool {
    false
}

extern "C" fn stub_load_scene_additive(_id: *const u8) -> u32 {
    0
}

extern "C" fn stub_unload_scene(_handle: u32) -> bool {
    false
}

extern "C" fn stub_active_scene() -> u32 {
    0
}

extern "C" fn stub_get_camera(_out: *mut CTransform) -> bool {
    false
}

extern "C" fn stub_set_camera(_value: *const CTransform) -> bool {
    false
}

extern "C" fn stub_find_asset(_path: *const c_char, _out: *mut u8) -> u32 {
    ASSET_NONE
}

extern "C" fn stub_instantiate_prefab(_id: *const u8) -> CEntity {
    CEntity::NULL
}

extern "C" fn stub_asset_state(_id: *const u8) -> u32 {
    LOAD_FAILED
}

/// With no engine behind the table there is no stream to join, so the level
/// picks one of the three plain sinks below and the attribution is dropped.
extern "C" fn default_log_event(
    level: u32,
    _source: *const c_char,
    _line: u32,
    message: *const c_char,
) {
    match level {
        0..=2 => orrin_log(message),
        3 => orrin_log_warn(message),
        _ => orrin_log_error(message),
    }
}

extern "C" fn stub_action_id(_name: *const c_char) -> u32 {
    NO_ACTION
}

extern "C" fn stub_action_query(_id: u32, _player: u32) -> bool {
    false
}

extern "C" fn stub_axis_value(_id: u32, _player: u32) -> f32 {
    0.0
}

extern "C" fn stub_debug_draw_line(
    _fx: f32,
    _fy: f32,
    _fz: f32,
    _tx: f32,
    _ty: f32,
    _tz: f32,
    _r: f32,
    _g: f32,
    _b: f32,
    _a: f32,
    _duration: f32,
) {
}

extern "C" fn stub_add_box_collider(
    _entity: CEntity,
    _hx: f32,
    _hy: f32,
    _hz: f32,
    _is_trigger: bool,
) -> bool {
    false
}

extern "C" fn stub_add_sphere_collider(_entity: CEntity, _radius: f32, _is_trigger: bool) -> bool {
    false
}

extern "C" fn stub_set_material(_entity: CEntity, _material: *const c_char) -> bool {
    false
}

extern "C" fn stub_get_parent(_entity: CEntity) -> CEntity {
    CEntity::NULL
}

extern "C" fn stub_set_parent(_child: CEntity, _parent: CEntity, _keep_world: bool) -> bool {
    false
}

extern "C" fn stub_add_script(_entity: CEntity, _type_name: *const c_char) -> bool {
    false
}

extern "C" fn stub_find_by_tag(_tag: *const c_char, _out: *mut CEntity) -> bool {
    false
}

extern "C" fn stub_find_all_by_tag(_tag: *const c_char, _out: *mut CEntity, _capacity: i32) -> i32 {
    0
}

extern "C" fn stub_has_component(_entity: CEntity, _kind: u32) -> bool {
    false
}

extern "C" fn stub_get_tag(_entity: CEntity, _out: *mut c_char, _capacity: i32) -> i32 {
    -1
}

extern "C" fn stub_set_tag(_entity: CEntity, _tag: *const c_char) -> bool {
    false
}

extern "C" fn stub_time_seconds() -> f32 {
    0.0
}

extern "C" fn stub_time_frame_count() -> u64 {
    0
}

extern "C" fn stub_spawn_renderable(
    _mesh: *const c_char,
    _material: *const c_char,
    _transform: *const CTransform,
) -> CEntity {
    CEntity::NULL
}

extern "C" fn stub_despawn(_entity: CEntity) -> bool {
    false
}

extern "C" fn stub_get_transform(_entity: CEntity, _out: *mut CTransform) -> bool {
    false
}

extern "C" fn stub_set_transform(_entity: CEntity, _value: *const CTransform) -> bool {
    false
}

extern "C" fn stub_key_query(_code: u32) -> bool {
    false
}

extern "C" fn stub_cursor_pos(x: *mut f32, y: *mut f32) {
    if !x.is_null() {
        // SAFETY: C# passes valid, writable f32 pointers.
        unsafe { *x = 0.0 };
    }
    if !y.is_null() {
        // SAFETY: as above.
        unsafe { *y = 0.0 };
    }
}

/// Logging callback C# invokes through [`OrrinApi::log`].
//
// The three log sinks below take a raw pointer without being `unsafe fn`, which
// clippy denies by default. They cannot be `unsafe fn`: `OrrinApi` declares
// these fields as safe `extern "C" fn` pointers, and that struct is the frozen
// ABI the managed side reads. The pointer is never supplied by Rust — CoreCLR
// is the only caller — and each one null-checks before it reads.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn orrin_log(message: *const c_char) {
    if message.is_null() {
        return;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    println!("[c#] {text}");
}

/// Default warning sink (the engine overrides it to route into the editor
/// console). Prints to stderr so a script's diagnostics are still visible when
/// no console is present.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn orrin_log_warn(message: *const c_char) {
    if message.is_null() {
        return;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    eprintln!("[c# WARN] {text}");
}

/// Default error sink; see [`orrin_log_warn`].
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn orrin_log_error(message: *const c_char) {
    if message.is_null() {
        return;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    eprintln!("[c# ERROR] {text}");
}

/// Spawn a new, empty entity and return its handle.
pub extern "C" fn orrin_spawn() -> CEntity {
    with_world(CEntity::NULL, |world| {
        let entity = world.spawn();
        CEntity {
            index: entity.index,
            generation: entity.generation,
        }
    })
}

thread_local! {
    static ACTIVE_WORLD: Cell<*mut World> = const { Cell::new(std::ptr::null_mut()) };
}

/// Install `world` as the active world while `dispatch` runs, then clear it.
///
/// The engine must hold no other borrow of this `World` for the duration of
/// `dispatch` — the ABI functions reconstruct an exclusive `&mut World` from the
/// stored pointer. (Hence the tick collects handles first, then dispatches.)
pub fn with_active_world<R>(world: &mut World, dispatch: impl FnOnce() -> R) -> R {
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            ACTIVE_WORLD.with(|w| w.set(std::ptr::null_mut()));
        }
    }
    ACTIVE_WORLD.with(|w| w.set(world as *mut World));
    let _clear = Clear;
    dispatch()
}

/// Run `op` against the active world, or return `default` if none is installed
/// (an ABI call outside a dispatch window).
pub fn with_world<R>(default: R, op: impl FnOnce(&mut World) -> R) -> R {
    let ptr = ACTIVE_WORLD.with(|w| w.get());
    if ptr.is_null() {
        return default;
    }
    // SAFETY: within a dispatch window `ptr` is a valid, uniquely-owned
    // `&mut World`; scripts are single-threaded and this borrow lives only for
    // the leaf `op` call, so it never aliases another.
    op(unsafe { &mut *ptr })
}

/// Signature of the C# `Orrin.Behaviours.Destroy(nint)` entry point; `extern
/// "system"` matches the convention netcorehost hands back.
type DestroyFn = extern "system" fn(u64);

static DESTROY_HANDLE: AtomicUsize = AtomicUsize::new(0);

/// Install the managed teardown callback (called once by [`ScriptHost::boot`]).
pub fn set_destroy_handle(destroy: DestroyFn) {
    DESTROY_HANDLE.store(destroy as usize, Ordering::Release);
}

/// Tear down the behaviour behind `handle`: the managed side fires
/// OnDisable (if still active) and OnDestroy, then frees the `GCHandle` — in
/// that order, enforced in one place (C# `Behaviours.Destroy`). A no-op before
/// the host is booted, so it's safe to call unconditionally from
/// `ScriptComponent::drop`.
///
/// This usually runs from `Drop`, *outside* a dispatch window: `ACTIVE_WORLD`
/// is null, so any engine API the callbacks touch returns defaults instead of
/// aliasing whatever `&mut World` triggered the despawn. Safe, but it means
/// OnDestroy cannot usefully read the world — document that as a script-facing
/// limitation (Unity has an equivalent one).
pub fn destroy_handle(handle: u64) {
    let ptr = DESTROY_HANDLE.load(Ordering::Acquire);
    if ptr != 0 {
        // SAFETY: only ever set by `set_destroy_handle` from a valid C# `DestroyFn`.
        let destroy: DestroyFn = unsafe { std::mem::transmute::<usize, DestroyFn>(ptr) };
        destroy(handle);
    }
}

/// The four `Orrin.PropertyBag` entry points, held as process globals for the
/// same reason [`destroy_handle`] is: the component registry's script entries
/// are bare `fn` pointers reached from a `ComponentVtable`, with no `&ScriptHost`
/// anywhere in scope. A registry that had to borrow the host would have to be
/// owned by it, and the registry deliberately outlives the world *and* the
/// scripting layer — a scene loads before either exists.
///
/// Every one takes a caller-owned buffer and returns the size the value needs:
/// non-negative means "this many bytes", written only if they fit, so a caller
/// that guessed too small retries with the number it got back. Negative is a
/// failure the managed side has already logged.
mod bag {
    use super::*;

    pub type ReadFn = extern "system" fn(u64, *mut u8, i32) -> i32;
    pub type WriteFn = extern "system" fn(u64, *const u8, i32) -> i32;
    pub type DefaultFn = extern "system" fn(*const c_char, *mut u8, i32) -> i32;
    pub type ComponentsFn = extern "system" fn(*mut u8, i32) -> i32;

    pub static READ: AtomicUsize = AtomicUsize::new(0);
    pub static WRITE: AtomicUsize = AtomicUsize::new(0);
    pub static DEFAULT: AtomicUsize = AtomicUsize::new(0);
    pub static COMPONENTS: AtomicUsize = AtomicUsize::new(0);
}

/// What most property bags fit in, so the common path is one managed call.
///
/// A component is a handful of scalars; 512 bytes is roughly forty of them.
/// Overshooting costs a second call and an exact-sized allocation, which is
/// still correct — this number only decides how often that happens.
const BAG_HINT: usize = 512;

/// Install the managed property-bag entry points (called once by
/// [`ScriptHost::boot`]).
fn set_bag_entry_points(
    read: bag::ReadFn,
    write: bag::WriteFn,
    default: bag::DefaultFn,
    components: bag::ComponentsFn,
) {
    bag::READ.store(read as usize, Ordering::Release);
    bag::WRITE.store(write as usize, Ordering::Release);
    bag::DEFAULT.store(default as usize, Ordering::Release);
    bag::COMPONENTS.store(components as usize, Ordering::Release);
}

/// Run the size-then-fill protocol against one managed entry point.
///
/// `None` when the entry point is not installed (no host booted) or the managed
/// side reported a failure — which it has already logged, naming the type.
fn fetch_bag(call: impl Fn(*mut u8, i32) -> i32) -> Option<Vec<u8>> {
    let mut buffer = vec![0u8; BAG_HINT];
    let needed = call(buffer.as_mut_ptr(), BAG_HINT as i32);
    if needed < 0 {
        return None;
    }
    let needed = needed as usize;
    if needed > buffer.len() {
        buffer.resize(needed, 0);
        // The second call must agree with the first about the size. It cannot
        // disagree unless the behaviour mutated itself between them, which
        // nothing on this path does — but a shrinking answer would leave the
        // tail of the buffer as stale bytes, so it is refused rather than
        // trusted.
        if call(buffer.as_mut_ptr(), needed as i32) != needed as i32 {
            return None;
        }
    }
    buffer.truncate(needed);
    Some(buffer)
}

/// Encode the fields of the behaviour behind `handle` as one property bag.
pub fn read_bag(handle: u64) -> Option<Vec<u8>> {
    let ptr = bag::READ.load(Ordering::Acquire);
    if ptr == 0 {
        return None;
    }
    // SAFETY: only ever set by `set_bag_entry_points` from a valid C# `ReadFn`.
    let read: bag::ReadFn = unsafe { std::mem::transmute::<usize, bag::ReadFn>(ptr) };
    fetch_bag(|buffer, capacity| read(handle, buffer, capacity))
}

/// Overwrite that behaviour's fields from `bytes`. Fields the buffer does not
/// name keep their current values.
pub fn write_bag(handle: u64, bytes: &[u8]) -> bool {
    let ptr = bag::WRITE.load(Ordering::Acquire);
    if ptr == 0 {
        return false;
    }
    // SAFETY: only ever set by `set_bag_entry_points` from a valid C# `WriteFn`.
    let write: bag::WriteFn = unsafe { std::mem::transmute::<usize, bag::WriteFn>(ptr) };
    write(handle, bytes.as_ptr(), bytes.len() as i32) >= 0
}

/// The bag a freshly constructed `type_name` produces — the component's default.
pub fn default_bag(type_name: &CStr) -> Option<Vec<u8>> {
    let ptr = bag::DEFAULT.load(Ordering::Acquire);
    if ptr == 0 {
        return None;
    }
    // SAFETY: only ever set by `set_bag_entry_points` from a valid C# `DefaultFn`.
    let default: bag::DefaultFn = unsafe { std::mem::transmute::<usize, bag::DefaultFn>(ptr) };
    fetch_bag(|buffer, capacity| default(type_name.as_ptr(), buffer, capacity))
}

/// Every registry-visible Behaviour in the live game assemblies, encoded as a
/// list of `{ id, type, name }`. The game assembly's half of
/// `register_components`, re-read after each swap.
pub fn component_bags() -> Option<Vec<u8>> {
    let ptr = bag::COMPONENTS.load(Ordering::Acquire);
    if ptr == 0 {
        return None;
    }
    // SAFETY: only ever set by `set_bag_entry_points` from a valid C# `ComponentsFn`.
    let components: bag::ComponentsFn =
        unsafe { std::mem::transmute::<usize, bag::ComponentsFn>(ptr) };
    fetch_bag(|buffer, capacity| components(buffer, capacity))
}

/// C ABI mirror of `orrin_ecs::Entity` (blittable: two `u32`s).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CEntity {
    pub index: u32,
    pub generation: u32,
}

impl CEntity {
    /// Returned when an ABI call can't produce a real entity.
    ///
    /// All-zeroes, because that is what C# produces for an `Entity` field left
    /// uninitialized — a struct's `default` cannot be customized there, and the
    /// commonest source of one is a Behaviour field a hot reload could not
    /// restore. Making the sentinel coincide with it means those two cases are
    /// the same value instead of two different "not an entity" handles, only
    /// one of which `IsValid` knew about.
    ///
    /// This is safe only because `orrin_ecs` reserves slot 0 and never
    /// allocates it (see `EntityAllocator::default`): `is_alive` is false for
    /// `{0, 0}` and every `SparseSet` lookup misses, so it cannot alias a live
    /// entity. Do not restore that reservation's removal without moving this
    /// sentinel back.
    ///
    /// C# mirrors it as `Orrin.Entity.Null`; keep the two in sync, and note
    /// the `sizeof` handshake in `Bootstrap.Init` cannot catch a mismatch here
    /// — the layout is unchanged, only the value.
    pub const NULL: Self = Self {
        index: 0,
        generation: 0,
    };
}

/// C ABI transform: position, rotation (quaternion `xyzw`), scale. Matches the
/// C# `Orrin.Transform`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CTransform {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

/// C ABI camera lens: vertical field of view in degrees, near and far planes
/// in metres, and the priority that picks a camera when none is selected.
/// Matches the C# `Orrin.CameraComponent`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CCamera {
    pub fov_degrees: f32,
    pub near: f32,
    pub far: f32,
    pub priority: i32,
}

/// C ABI `ui::Canvas`; matches C# `Orrin.Canvas`. `scale_mode` is C#
/// `Orrin.CanvasScaleMode`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CCanvas {
    pub scale_mode: u32,
    pub reference_resolution: [f32; 2],
    pub match_width_or_height: f32,
    pub sort_order: i32,
}

/// C ABI `ui::RectTransform`; matches C# `Orrin.RectTransform`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CRectTransform {
    pub anchor_min: [f32; 2],
    pub anchor_max: [f32; 2],
    pub pivot: [f32; 2],
    pub anchored_position: [f32; 2],
    pub size_delta: [f32; 2],
    pub scale: [f32; 2],
}

/// C ABI `ui::Image`; matches C# `Orrin.Image`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CImage {
    pub texture: [u8; 16],
    pub color: [f32; 4],
    pub slice: [f32; 4],
    pub raycast_target: u32,
}

/// C ABI `ui::Text` without its string; matches C# `Orrin.Text`. `anchor` is
/// C# `Orrin.TextAnchor`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CText {
    pub font: [u8; 16],
    pub color: [f32; 4],
    pub size: f32,
    pub anchor: u32,
    pub wrap: u32,
    pub raycast_target: u32,
}

/// C ABI `ui::Button`; matches C# `Orrin.Button`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CButton {
    pub normal: [f32; 4],
    pub hover: [f32; 4],
    pub pressed: [f32; 4],
    pub disabled: [f32; 4],
    pub interactable: u32,
}

/// C ABI `ui::Gradient`; matches C# `Orrin.Gradient`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CGradient {
    pub start: [f32; 4],
    pub end: [f32; 4],
    pub rotation: f32,
    pub offset: [f32; 2],
}

/// C ABI `ui::Corner`; matches C# `Orrin.Corner`. `style` is C#
/// `Orrin.CornerStyle`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CCorner {
    pub radius: f32,
    pub radius_fraction: f32,
    pub style: u32,
    pub exponent: f32,
}

/// C ABI `ui::Stroke`; matches C# `Orrin.Stroke`. `placement` is C#
/// `Orrin.StrokePlacement`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CStroke {
    pub color: [f32; 4],
    pub thickness: f32,
    pub placement: u32,
}

/// C ABI `ui::Shadow`; matches C# `Orrin.Shadow`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CShadow {
    pub color: [f32; 4],
    pub offset: [f32; 2],
    pub blur: f32,
    pub spread: f32,
}

/// C ABI `scene::RigidBody`; matches C# `Orrin.Rigidbody`. Flags are whole
/// words, as everywhere in this ABI, so neither side pads differently.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CRigidBody {
    pub kinematic: u32,
    pub mass: f32,
    pub gravity_scale: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub continuous: u32,
    pub lock_rotation: u32,
}

/// C ABI `scene::AudioSource`; matches C# `Orrin.AudioSource`. `rolloff` is
/// the C# `Orrin.AudioRolloff` numbering: inverse, linear.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CAudioSource {
    pub clip: [u8; 16],
    pub volume: f32,
    pub looping: u32,
    pub play_on_start: u32,
    pub spatial: u32,
    pub min_distance: f32,
    pub max_distance: f32,
    pub rolloff: u32,
}

/// C ABI of an entity's `MeshHandle` and `MaterialHandle` together; matches C#
/// `Orrin.MeshRenderer`. Each is the [`AssetId`](orrin_registry::AssetId)
/// bytes of what the slot draws rather than the slot, which means nothing
/// outside this session.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CMeshRenderer {
    pub mesh: [u8; 16],
    pub material: [u8; 16],
}

/// A request to `audio_play`; matches C# `Orrin.SoundRequest`. `emitter` is
/// where it is heard from — 0 flat, 1 at `position`, 2 on `entity` — and the
/// distance fields are read only by the last two, as an `AudioSource`'s are.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CSound {
    pub clip: [u8; 16],
    pub entity: CEntity,
    pub position: [f32; 3],
    pub volume: f32,
    pub looping: u32,
    pub emitter: u32,
    pub min_distance: f32,
    pub max_distance: f32,
    pub rolloff: u32,
}

/// C ABI `scene::Collider`, for `add_collider`; matches C# `Orrin.ColliderData`.
/// `shape` is the C# `Orrin.ColliderShape` numbering — box, sphere, capsule,
/// cylinder, mesh, convex hull — and `size` is read per shape: a box's half
/// extents; a sphere's radius in `x`; a capsule's or cylinder's radius in `x`
/// and half height in `y`; nothing for the two read from the entity's mesh.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CCollider {
    pub shape: u32,
    pub size: [f32; 3],
    pub is_trigger: u32,
    pub friction: f32,
    pub restitution: f32,
    pub layer: u32,
}

/// What `physics_raycast` hit; matches C# `Orrin.RaycastHit`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CRaycastHit {
    pub entity: CEntity,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub distance: f32,
}

/// C ABI `ui::ScrollFrame`; matches C# `Orrin.ScrollFrame`. `scrollbar` is C#
/// `Orrin.ScrollbarVisibility`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CScrollFrame {
    pub content_size: [f32; 2],
    pub scroll_position: [f32; 2],
    pub scroll_speed: f32,
    pub scrollbar_thickness: f32,
    pub scrollbar_color: [f32; 4],
    pub scrollbar: u32,
    pub horizontal: u32,
    pub vertical: u32,
}

/// C ABI `ui::tween::Tween` without its target; matches C# `Orrin.TweenRequest`.
/// `property`, `style` and `direction` are C# `Orrin.TweenProperty`,
/// `Orrin.EaseStyle` and `Orrin.EaseDirection`. A value uses as many lanes as
/// its property has, from the first; `from` is read only when `has_from` is
/// nonzero.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CTween {
    pub property: u32,
    pub style: u32,
    pub direction: u32,
    pub repeat: i32,
    pub duration: f32,
    pub delay: f32,
    pub reverses: u32,
    pub has_from: u32,
    pub from: [f32; 4],
    pub to: [f32; 4],
}

/// C ABI pointer event payload: where the pointer is, where its button went
/// down, the element it hit and the button. Matches C# `Orrin.PointerEvent`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CPointerEvent {
    pub position: [f32; 2],
    pub press_position: [f32; 2],
    pub target: CEntity,
    pub button: u32,
}

/// What a behaviour did with a pointer event. The bytes are lock-step with C#
/// `Orrin.Behaviours`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delivery {
    Handled,
    /// It threw, which counts as handled: the event does not go on up.
    Faulted,
    /// It implements no handler for the event, which goes on to the parent.
    Unhandled,
}

/// C ABI collision event payload, as seen by one participant: the *other*
/// entity, the world-space contact point, the contact normal pointing from
/// the receiving entity toward `other`, and the receiver's velocity at the
/// contact less `other`'s, in m/s. Matches the C# `Orrin.Collision`.
///
/// Handed over by pointer and never size-checked per call, so it only ever
/// grows at the end: a stale bindings build reads a prefix of it.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CCollision {
    pub other: CEntity,
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub relative_velocity: [f32; 3],
}

/// The optional hooks a Behaviour's type overrides, as `Behaviours.Create`
/// reports them. The bits are lock-step with the constants in C#
/// `Orrin.Behaviours`.
///
/// A hook the engine may call per physics step, or per touching pair per step,
/// is a managed transition per call whether or not the type wrote a body for
/// it — thousands of resting crates would pay for an empty base method. So the
/// managed side reads the type's overrides once, when it builds the instance,
/// and the engine skips the call for a hook that is not here. Read at creation
/// rather than per call because a type cannot change while it is loaded: a hot
/// reload re-creates every instance, and with it this answer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Hooks(u32);

impl Hooks {
    pub const NONE: Self = Self(0);
    pub const FIXED_UPDATE: Self = Self(1);
    pub const COLLISION_STAY: Self = Self(1 << 1);
    pub const TRIGGER_STAY: Self = Self(1 << 2);

    pub fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    pub fn bits(self) -> u32 {
        self.0
    }

    /// Whether every hook in `other` is overridden.
    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether any hook in `other` is overridden.
    pub fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

impl std::ops::BitOr for Hooks {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl std::ops::BitOrAssign for Hooks {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

/// Outcome of a game-assembly operation. The discriminants are lock-step with
/// the constants in C# `Orrin.GameAssembly`; append, never renumber — same
/// rule as the ABI table and the key codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameAssemblyStatus {
    Ok,
    /// The path was null or not valid UTF-8.
    BadArgument,
    /// No file at that path — usually a build that hasn't run yet.
    NotFound,
    /// The image was rejected: a corrupt or half-written DLL, a missing
    /// dependency, the wrong target framework. Managed side logged the detail.
    LoadFailed,
    /// `Commit`/`Rollback`/`Unload` with nothing to act on.
    NothingStaged,
    /// The swap succeeded, but the retired context outlived its unload: some
    /// reference into the old assembly is still reachable. The new code is live
    /// regardless — this is a diagnostic, not a failure.
    Leaked,
    Unknown(i32),
}

impl GameAssemblyStatus {
    fn from_raw(code: i32) -> Self {
        match code {
            0 => Self::Ok,
            1 => Self::BadArgument,
            2 => Self::NotFound,
            3 => Self::LoadFailed,
            4 => Self::NothingStaged,
            5 => Self::Leaked,
            other => Self::Unknown(other),
        }
    }

    /// Whether the operation did what was asked. `Leaked` counts: the swap
    /// happened, and refusing to proceed over a leaked context would strand the
    /// session on stale code for a condition it cannot fix.
    pub fn succeeded(self) -> bool {
        matches!(self, Self::Ok | Self::Leaked)
    }
}

impl std::fmt::Display for GameAssemblyStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ok => f.write_str("ok"),
            Self::BadArgument => f.write_str("invalid assembly path"),
            Self::NotFound => f.write_str("assembly not found"),
            Self::LoadFailed => f.write_str("assembly could not be loaded"),
            Self::NothingStaged => f.write_str("no assembly staged"),
            Self::Leaked => {
                f.write_str("the previous assembly did not unload (something still references it)")
            }
            Self::Unknown(code) => write!(f, "unknown status {code}"),
        }
    }
}

/// The booted .NET runtime plus the managed lifecycle entry points. Holding the
/// `HostfxrContext` keeps the runtime resident for the host's lifetime.
pub struct ScriptHost {
    _context: HostfxrContext<InitializedForRuntimeConfig>,
    // Writes the type's `Hooks` bits through the out-pointer.
    create_fn: extern "system" fn(CEntity, *const c_char, *mut u32) -> u64,
    // The lifecycle hooks return a fault byte (0 = clean, nonzero = the C# hook
    // threw and was contained): the managed side never lets an exception cross
    // into these native frames, and reports it here so the engine can disable
    // the offending script. Signatures stay in lock-step with C#
    // `Orrin.Behaviours` (which returns `byte`).
    start_fn: extern "system" fn(u64) -> u8,
    update_fn: extern "system" fn(u64, f32) -> u8,
    enable_fn: extern "system" fn(u64) -> u8,
    disable_fn: extern "system" fn(u64) -> u8,
    collision_enter_fn: extern "system" fn(u64, *const CCollision) -> u8,
    collision_exit_fn: extern "system" fn(u64, *const CCollision) -> u8,
    trigger_enter_fn: extern "system" fn(u64, *const CCollision) -> u8,
    trigger_exit_fn: extern "system" fn(u64, *const CCollision) -> u8,
    fixed_update_fn: extern "system" fn(u64, f32) -> u8,
    collision_stay_fn: extern "system" fn(u64, *const CCollision) -> u8,
    trigger_stay_fn: extern "system" fn(u64, *const CCollision) -> u8,
    // Returns 0 handled, 1 faulted, 2 not handled; see `Delivery`.
    pointer_fn: extern "system" fn(u64, u32, *const CPointerEvent) -> u8,
    // Runs and forgets whatever the game asked to happen when tween `id`
    // finished. Contains its own exceptions, like the hooks.
    tween_complete_fn: extern "system" fn(u64),
    // The game assembly's stories, by the `get_tag` protocol: every name, one
    // per line, as UTF-8 into a caller-owned buffer, returning the full byte
    // length.
    story_names_fn: extern "system" fn(*mut u8, i32) -> i32,
    // Build the story named by the UTF-8 bytes onto the canvas entity; returns
    // 0, 1 when it threw, 2 when no story has that name.
    story_build_fn: extern "system" fn(*const u8, u32, CEntity) -> u8,
    // Game-assembly swap. These live in `Orrin.dll`, which is in the default
    // (uncollectible) load context, so the pointers stay valid across every
    // reload of the game assembly — the whole reason the bindings and the game
    // are separate assemblies.
    game_load_fn: extern "system" fn(*const c_char) -> i32,
    game_commit_fn: extern "system" fn() -> i32,
    game_rollback_fn: extern "system" fn() -> i32,
    game_unload_fn: extern "system" fn() -> i32,
    state_capture_fn: extern "system" fn(u64) -> u64,
    state_apply_fn: extern "system" fn(u64, u64) -> u8,
    state_discard_fn: extern "system" fn(),
    /// Every hook call made into managed code, so a test can tell a call that
    /// was skipped from one that was made and returned at once.
    dispatched: Cell<u64>,
}

/// The .NET runtime an exported game carries, beside its `Orrin.dll`: a
/// `DOTNET_ROOT` holding one `host/fxr` and one `shared/Microsoft.NETCore.App`.
///
/// It is a private install rather than a self-contained publish because
/// hostfxr refuses `initialize_for_runtime_config` for a self-contained app,
/// and that call is the one component hosting — and so the collectible game
/// context — is built on. A framework-dependent `Orrin.dll` pointed at its own
/// root boots exactly the way it does against a machine-wide SDK.
pub const APP_LOCAL_RUNTIME: &str = "dotnet";

impl ScriptHost {
    /// Boot CoreCLR and load `Orrin.dll` from `assembly_dir`, handing C# the
    /// `api` table. `assembly_dir` must contain `Orrin.dll` and
    /// `Orrin.runtimeconfig.json`, and may contain an [`APP_LOCAL_RUNTIME`],
    /// which is then used instead of any installed .NET.
    pub fn boot(api: &OrrinApi, assembly_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        // hostfxr resolves paths relative to the working directory, and
        // netcorehost's runtime path type isn't public here, so load with the cwd
        // pointed at the assembly dir, then restore it.
        let previous = std::env::current_dir()?;
        let runtime = assembly_dir.join(APP_LOCAL_RUNTIME);
        let runtime = runtime
            .join("host")
            .is_dir()
            .then(|| runtime.canonicalize())
            .transpose()?;
        std::env::set_current_dir(assembly_dir)?;
        let result = Self::load(api, runtime.as_deref());
        let _ = std::env::set_current_dir(previous);
        result
    }

    fn load(api: &OrrinApi, runtime: Option<&Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let config = pdcstr!("Orrin.runtimeconfig.json");
        let context = match runtime {
            Some(root) => {
                let root = PdCString::from_os_str(root.as_os_str())?;
                nethost::load_hostfxr_with_dotnet_root(&root)?
                    .initialize_for_runtime_config_with_dotnet_root(config, &root)?
            }
            None => nethost::load_hostfxr()?.initialize_for_runtime_config(config)?,
        };

        // Deref each `ManagedFunction` to its raw `extern "system"` fn pointer.
        // The loader is scoped so its borrow of `context` ends before the move.
        #[rustfmt::skip]
        let (
            init, create_fn, start_fn, update_fn, enable_fn, disable_fn,
            collision_enter_fn, collision_exit_fn, trigger_enter_fn, trigger_exit_fn,
            fixed_update_fn, collision_stay_fn, trigger_stay_fn, pointer_fn,
            tween_complete_fn, story_names_fn, story_build_fn, destroy,
            game_load_fn, game_commit_fn, game_rollback_fn, game_unload_fn,
            state_capture_fn, state_apply_fn, state_discard_fn,
            bag_read, bag_write, bag_default, bag_components,
        ) = {
            let loader = context.get_delegate_loader_for_assembly(pdcstr!("Orrin.dll"))?;
            (
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(*const OrrinApi, i32) -> i32>(
                    pdcstr!("Orrin.Bootstrap, Orrin"),
                    pdcstr!("Init"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(CEntity, *const c_char, *mut u32) -> u64>(
                    pdcstr!("Orrin.Behaviours, Orrin"),
                    pdcstr!("Create"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64) -> u8>(
                    pdcstr!("Orrin.Behaviours, Orrin"),
                    pdcstr!("Start"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64, f32) -> u8>(
                    pdcstr!("Orrin.Behaviours, Orrin"),
                    pdcstr!("Update"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64) -> u8>(
                    pdcstr!("Orrin.Behaviours, Orrin"),
                    pdcstr!("Enable"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64) -> u8>(
                    pdcstr!("Orrin.Behaviours, Orrin"),
                    pdcstr!("Disable"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64, *const CCollision) -> u8>(
                    pdcstr!("Orrin.Behaviours, Orrin"),
                    pdcstr!("CollisionEnter"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64, *const CCollision) -> u8>(
                    pdcstr!("Orrin.Behaviours, Orrin"),
                    pdcstr!("CollisionExit"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64, *const CCollision) -> u8>(
                    pdcstr!("Orrin.Behaviours, Orrin"),
                    pdcstr!("TriggerEnter"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64, *const CCollision) -> u8>(
                    pdcstr!("Orrin.Behaviours, Orrin"),
                    pdcstr!("TriggerExit"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64, f32) -> u8>(
                    pdcstr!("Orrin.Behaviours, Orrin"),
                    pdcstr!("FixedUpdate"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64, *const CCollision) -> u8>(
                    pdcstr!("Orrin.Behaviours, Orrin"),
                    pdcstr!("CollisionStay"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64, *const CCollision) -> u8>(
                    pdcstr!("Orrin.Behaviours, Orrin"),
                    pdcstr!("TriggerStay"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64, u32, *const CPointerEvent) -> u8>(
                    pdcstr!("Orrin.Behaviours, Orrin"),
                    pdcstr!("Pointer"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64)>(
                    pdcstr!("Orrin.Tweens, Orrin"),
                    pdcstr!("Complete"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(*mut u8, i32) -> i32>(
                    pdcstr!("Orrin.Stories, Orrin"),
                    pdcstr!("Names"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(*const u8, u32, CEntity) -> u8>(
                    pdcstr!("Orrin.Stories, Orrin"),
                    pdcstr!("Build"),
                )?,
                // Teardown is a process-wide global, not a `ScriptHost` method:
                // it must stay reachable from `ScriptComponent::drop`, which
                // has no `&Scripting` in scope.
                *loader.get_function_with_unmanaged_callers_only::<DestroyFn>(
                    pdcstr!("Orrin.Behaviours, Orrin"),
                    pdcstr!("Destroy"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(*const c_char) -> i32>(
                    pdcstr!("Orrin.GameAssembly, Orrin"),
                    pdcstr!("Load"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn() -> i32>(
                    pdcstr!("Orrin.GameAssembly, Orrin"),
                    pdcstr!("Commit"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn() -> i32>(
                    pdcstr!("Orrin.GameAssembly, Orrin"),
                    pdcstr!("Rollback"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn() -> i32>(
                    pdcstr!("Orrin.GameAssembly, Orrin"),
                    pdcstr!("Unload"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64) -> u64>(
                    pdcstr!("Orrin.BehaviourState, Orrin"),
                    pdcstr!("Capture"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64, u64) -> u8>(
                    pdcstr!("Orrin.BehaviourState, Orrin"),
                    pdcstr!("Apply"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn()>(
                    pdcstr!("Orrin.BehaviourState, Orrin"),
                    pdcstr!("Discard"),
                )?,
                // The property bag. Like `Destroy`, these become process
                // globals rather than `ScriptHost` methods — see `mod bag`.
                *loader.get_function_with_unmanaged_callers_only::<bag::ReadFn>(
                    pdcstr!("Orrin.PropertyBag, Orrin"),
                    pdcstr!("Read"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<bag::WriteFn>(
                    pdcstr!("Orrin.PropertyBag, Orrin"),
                    pdcstr!("Write"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<bag::DefaultFn>(
                    pdcstr!("Orrin.PropertyBag, Orrin"),
                    pdcstr!("Default"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<bag::ComponentsFn>(
                    pdcstr!("Orrin.PropertyBag, Orrin"),
                    pdcstr!("Components"),
                )?,
            )
        };

        // Hand C# the byte size of our OrrinApi as an ABI handshake: it refuses
        // to initialize if its own struct is a different size (a stale-rebuild
        // mismatch), turning silent UB into a clean startup error.
        let status = init(api, std::mem::size_of::<OrrinApi>() as i32);
        if status != 0 {
            return Err(format!(
                "Orrin.Bootstrap.Init rejected initialization (status {status}); the \
                 Orrin assembly is out of sync with the engine ABI — rebuild \
                 scripting/Orrin"
            )
            .into());
        }
        set_destroy_handle(destroy);
        set_bag_entry_points(bag_read, bag_write, bag_default, bag_components);

        Ok(Self {
            _context: context,
            create_fn,
            start_fn,
            update_fn,
            enable_fn,
            disable_fn,
            collision_enter_fn,
            collision_exit_fn,
            trigger_enter_fn,
            trigger_exit_fn,
            fixed_update_fn,
            collision_stay_fn,
            trigger_stay_fn,
            pointer_fn,
            tween_complete_fn,
            story_names_fn,
            story_build_fn,
            game_load_fn,
            game_commit_fn,
            game_rollback_fn,
            game_unload_fn,
            state_capture_fn,
            state_apply_fn,
            state_discard_fn,
            dispatched: Cell::new(0),
        })
    }

    // Game-assembly swap. Loading is *staged*: `stage_game` builds a new
    // collectible load context without disturbing the running one, so a bad
    // build can be rejected by `rollback_game` while every live behaviour is
    // still intact. Only `commit_game` retires the old context, and it must not
    // be called until every `GCHandle` into the old assembly has been freed —
    // a surviving handle pins its type and the context never unloads.

    /// Stage the game assembly at `dll_path`. Follow with `commit_game` (after
    /// tearing down the old behaviours) or `rollback_game`.
    pub fn stage_game(&self, dll_path: &CStr) -> GameAssemblyStatus {
        GameAssemblyStatus::from_raw((self.game_load_fn)(dll_path.as_ptr()))
    }

    /// Retire the live load context and promote the staged one.
    pub fn commit_game(&self) -> GameAssemblyStatus {
        GameAssemblyStatus::from_raw((self.game_commit_fn)())
    }

    /// Drop a staged assembly, leaving the running one untouched.
    pub fn rollback_game(&self) -> GameAssemblyStatus {
        GameAssemblyStatus::from_raw((self.game_rollback_fn)())
    }

    /// Unload the live game assembly without replacing it.
    pub fn unload_game(&self) -> GameAssemblyStatus {
        GameAssemblyStatus::from_raw((self.game_unload_fn)())
    }

    /// Snapshot a behaviour's reloadable fields before its handle is freed.
    /// Returns a snapshot id, or `0` when there was nothing to keep — pass a
    /// nonzero id to `apply_state` once the behaviour has been re-created.
    pub fn capture_state(&self, handle: u64) -> u64 {
        (self.state_capture_fn)(handle)
    }

    /// Restore a snapshot onto a freshly created behaviour. The snapshot is
    /// consumed either way; `false` means nothing was restored (the type was
    /// renamed, or every field changed shape).
    pub fn apply_state(&self, handle: u64, snapshot: u64) -> bool {
        (self.state_apply_fn)(handle, snapshot) != 0
    }

    /// Drop snapshots no `apply_state` claimed — behaviours whose type
    /// disappeared in the edit that triggered the reload.
    pub fn discard_states(&self) {
        (self.state_discard_fn)()
    }

    /// Instantiate a `Behaviour` by assembly-qualified type name, attached to
    /// `entity`. Returns its `GCHandle` (as `u64`), or `0` on failure, and the
    /// optional hooks its type overrides.
    pub fn create(&self, entity: CEntity, type_name: &CStr) -> (u64, Hooks) {
        let mut hooks = 0u32;
        let handle = (self.create_fn)(entity, type_name.as_ptr(), &mut hooks);
        (handle, Hooks::from_bits(hooks))
    }

    /// How many hook calls have been made into managed code since boot.
    pub fn dispatched(&self) -> u64 {
        self.dispatched.get()
    }

    fn dispatch<R>(&self, call: impl FnOnce() -> R) -> R {
        self.dispatched.set(self.dispatched.get() + 1);
        call()
    }

    // Each hook returns `true` if the C# side caught an exception (the script is
    // now faulted and should not be dispatched to again until the fault clears).

    pub fn start(&self, handle: u64) -> bool {
        self.dispatch(|| (self.start_fn)(handle) != 0)
    }

    pub fn update(&self, handle: u64, delta_time: f32) -> bool {
        self.dispatch(|| (self.update_fn)(handle, delta_time) != 0)
    }

    pub fn enable(&self, handle: u64) -> bool {
        self.dispatch(|| (self.enable_fn)(handle) != 0)
    }

    pub fn disable(&self, handle: u64) -> bool {
        self.dispatch(|| (self.disable_fn)(handle) != 0)
    }

    /// Only for a behaviour whose [`Hooks`] has [`Hooks::FIXED_UPDATE`].
    pub fn fixed_update(&self, handle: u64, step: f32) -> bool {
        self.dispatch(|| (self.fixed_update_fn)(handle, step) != 0)
    }

    pub fn collision_enter(&self, handle: u64, collision: &CCollision) -> bool {
        self.dispatch(|| (self.collision_enter_fn)(handle, collision) != 0)
    }

    /// Only for a behaviour whose [`Hooks`] has [`Hooks::COLLISION_STAY`].
    pub fn collision_stay(&self, handle: u64, collision: &CCollision) -> bool {
        self.dispatch(|| (self.collision_stay_fn)(handle, collision) != 0)
    }

    pub fn collision_exit(&self, handle: u64, collision: &CCollision) -> bool {
        self.dispatch(|| (self.collision_exit_fn)(handle, collision) != 0)
    }

    /// The trigger half of the same pair: a contact where either side is a
    /// trigger goes here instead.
    pub fn trigger_enter(&self, handle: u64, collision: &CCollision) -> bool {
        self.dispatch(|| (self.trigger_enter_fn)(handle, collision) != 0)
    }

    /// Only for a behaviour whose [`Hooks`] has [`Hooks::TRIGGER_STAY`].
    pub fn trigger_stay(&self, handle: u64, collision: &CCollision) -> bool {
        self.dispatch(|| (self.trigger_stay_fn)(handle, collision) != 0)
    }

    pub fn trigger_exit(&self, handle: u64, collision: &CCollision) -> bool {
        self.dispatch(|| (self.trigger_exit_fn)(handle, collision) != 0)
    }

    /// Offer pointer event `kind` (C# `Orrin.PointerEventKind`) to a behaviour.
    pub fn pointer(&self, handle: u64, kind: u32, event: &CPointerEvent) -> Delivery {
        match self.dispatch(|| (self.pointer_fn)(handle, kind, event)) {
            0 => Delivery::Handled,
            1 => Delivery::Faulted,
            _ => Delivery::Unhandled,
        }
    }

    /// Tell the game tween `id` finished.
    pub fn tween_complete(&self, id: u64) {
        (self.tween_complete_fn)(id);
    }

    /// Every story the live game assembly declares, in its declared order.
    pub fn story_names(&self) -> Vec<String> {
        let mut buffer = vec![0u8; 1024];
        loop {
            let length = (self.story_names_fn)(buffer.as_mut_ptr(), buffer.len() as i32);
            if length < 0 {
                return Vec::new();
            }
            let length = length as usize;
            if length <= buffer.len() {
                buffer.truncate(length);
                break;
            }
            buffer.resize(length, 0);
        }
        String::from_utf8_lossy(&buffer)
            .lines()
            .filter(|name| !name.is_empty())
            .map(str::to_owned)
            .collect()
    }

    /// Run story `name`'s builder onto `canvas`, inside a dispatch window the
    /// caller has opened.
    pub fn build_story(&self, name: &str, canvas: CEntity) -> StoryBuild {
        match (self.story_build_fn)(name.as_ptr(), name.len() as u32, canvas) {
            0 => StoryBuild::Built,
            1 => StoryBuild::Faulted,
            _ => StoryBuild::Missing,
        }
    }
}

/// What a story's builder did. The bytes are lock-step with C# `Orrin.Stories`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoryBuild {
    Built,
    /// It threw; the managed side logged why.
    Faulted,
    /// The game declares no story by that name.
    Missing,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sizes `Orrin.MathTests` asserts of the C# mirrors. Either side
    /// changing alone is a layout break the engine's per-call size check would
    /// then refuse at run time.
    #[test]
    fn the_ui_structs_are_the_sizes_the_bindings_check() {
        use std::mem::size_of;
        assert_eq!(size_of::<CCanvas>(), 20);
        assert_eq!(size_of::<CRectTransform>(), 48);
        assert_eq!(size_of::<CImage>(), 52);
        assert_eq!(size_of::<CText>(), 48);
        assert_eq!(size_of::<CButton>(), 68);
        assert_eq!(size_of::<CPointerEvent>(), 28);
        assert_eq!(size_of::<CGradient>(), 44);
        assert_eq!(size_of::<CCorner>(), 16);
        assert_eq!(size_of::<CStroke>(), 24);
        assert_eq!(size_of::<CShadow>(), 32);
        assert_eq!(size_of::<CScrollFrame>(), 52);
        assert_eq!(size_of::<CTween>(), 64);
        assert_eq!(size_of::<CAudioSource>(), 44);
        assert_eq!(size_of::<CSound>(), 60);
        assert_eq!(size_of::<CMeshRenderer>(), 32);
        assert_eq!(size_of::<CCollision>(), 44);
    }
}
