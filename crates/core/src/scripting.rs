//! C# scripting integration (engine side), behind the `scripting` feature.
//!
//! The generic ABI lives in `orrin-script`; the transform functions need the
//! engine's own `LocalTransform`, so they're defined here and assembled into the
//! table with `..orrin_script::default_api()`.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use glam::{Mat4, Quat, Vec3};

use orrin_ecs::{Entity, FxHashMap, World};
use orrin_script::{CCollision, CEntity, CTransform, GameAssemblyStatus, OrrinApi, ScriptHost};

use crate::collision::{CollisionEvent, CollisionEventKind, CollisionState};
use crate::scene::{
    Assets, Collider, ColliderShape, DebugLines, InputState, LocalTransform, LogBuffer, LogLevel,
    MaterialHandle, MeshHandle, Name, Parent, ScriptComponent, Tag, Time, Transform,
    WorldTransform,
};

extern "C" fn get_transform(entity: CEntity, out: *mut CTransform) -> bool {
    if out.is_null() {
        return false;
    }
    orrin_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        let Some(transform) = world.get::<LocalTransform>(entity) else {
            return false;
        };
        // SAFETY: `out` is a valid, writable `CTransform` supplied by C#.
        unsafe {
            *out = CTransform {
                position: transform.translation.to_array(),
                rotation: transform.rotation.to_array(),
                scale: transform.scale.to_array(),
            };
        }
        true
    })
}

extern "C" fn set_transform(entity: CEntity, value: *const CTransform) -> bool {
    if value.is_null() {
        return false;
    }
    // SAFETY: `value` is a valid `CTransform` supplied by C#.
    let value = unsafe { *value };
    orrin_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        let Some(mut transform) = world.get_mut::<LocalTransform>(entity) else {
            return false;
        };
        transform.translation = Vec3::from_array(value.position);
        transform.rotation = Quat::from_array(value.rotation);
        transform.scale = Vec3::from_array(value.scale);
        true
    })
}

// The world transform is a `Mat4` engine-side and a `CTransform` across the
// ABI, so this hands back the closest translation/rotation/scale fit to it.
// Exact for any chain of rotations and uniform scales; lossy once a
// non-uniformly scaled ancestor has introduced shear, which no TRS triple can
// spell. Scripts that need an exact world quantity should read the position,
// which is always the matrix's translation column.
extern "C" fn get_world_transform(entity: CEntity, out: *mut CTransform) -> bool {
    if out.is_null() {
        return false;
    }
    orrin_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        let Some(world_transform) = world.get::<WorldTransform>(entity) else {
            return false;
        };
        let (scale, rotation, translation) = world_transform.0.to_scale_rotation_translation();
        // SAFETY: `out` is a valid, writable `CTransform` supplied by C#.
        unsafe {
            *out = CTransform {
                position: translation.to_array(),
                rotation: rotation.to_array(),
                scale: scale.to_array(),
            };
        }
        true
    })
}

/// Write a world-space transform by composing it with the inverse of the
/// parent's, so a script never has to know it has a parent at all.
///
/// Reads a `WorldTransform` produced by the last propagation, which for a script
/// is the one after `spin`. A script that moves a parent and then world-places
/// its child in the same tick composes against the parent's previous position.
extern "C" fn set_world_transform(entity: CEntity, value: *const CTransform) -> bool {
    if value.is_null() {
        return false;
    }
    // SAFETY: `value` is a valid `CTransform` supplied by C#.
    let value = unsafe { *value };
    orrin_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        let target = Mat4::from_scale_rotation_translation(
            Vec3::from_array(value.scale),
            Quat::from_array(value.rotation),
            Vec3::from_array(value.position),
        );
        let parent_world = crate::scene::parent_world_matrix(world, entity);
        let local = parent_world.inverse() * target;

        let Some(mut transform) = world.get_mut::<LocalTransform>(entity) else {
            return false;
        };
        let (scale, rotation, translation) = local.to_scale_rotation_translation();
        transform.translation = translation;
        transform.rotation = rotation;
        transform.scale = scale;
        true
    })
}

extern "C" fn get_parent(entity: CEntity) -> CEntity {
    orrin_script::with_world(CEntity::NULL, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        match world.get::<Parent>(entity).map(|p| p.get()) {
            Some(parent) if world.is_alive(parent) => CEntity {
                index: parent.index,
                generation: parent.generation,
            },
            _ => CEntity::NULL,
        }
    })
}

/// Validate now, apply after the dispatch window.
///
/// Reparenting attaches or detaches a component, so it is a structural change
/// and cannot run mid-dispatch. Validating eagerly is what keeps the return
/// value meaningful: a script learns immediately that it asked for a cycle,
/// rather than finding out by the move silently not happening.
extern "C" fn set_parent(child: CEntity, parent: CEntity, keep_world: bool) -> bool {
    orrin_script::with_world(false, |world| {
        let child = Entity {
            index: child.index,
            generation: child.generation,
        };
        // `CEntity::NULL` is all-zeroes and slot 0 is reserved, so a zero index
        // is the null sentinel — the same test as C# `Entity.IsValid`.
        let parent = (parent.index != 0).then_some(Entity {
            index: parent.index,
            generation: parent.generation,
        });
        if crate::scene::can_reparent(world, child, parent).is_err() {
            return false;
        }
        COMMANDS.with(|commands| {
            commands.borrow_mut().push(Command::Reparent {
                child,
                parent,
                keep_world,
            })
        });
        true
    })
}

// The `InputState` resource is engine-side, so these live here (like the
// transform functions) and read it through the active-world seam. Outside a
// dispatch window, or before the resource exists, they report "nothing held".

fn with_input(query: impl FnOnce(&InputState) -> bool) -> bool {
    orrin_script::with_world(false, |world| {
        world
            .get_resource::<InputState>()
            .is_some_and(|input| query(&input))
    })
}

extern "C" fn key_down(code: u32) -> bool {
    with_input(|input| input.key_down(code))
}

extern "C" fn key_pressed(code: u32) -> bool {
    with_input(|input| input.key_pressed(code))
}

extern "C" fn key_released(code: u32) -> bool {
    with_input(|input| input.key_released(code))
}

extern "C" fn mouse_button_down(button: u32) -> bool {
    with_input(|input| input.mouse_button_down(button))
}

extern "C" fn cursor_pos(x: *mut f32, y: *mut f32) {
    let (cx, cy) = orrin_script::with_world((0.0, 0.0), |world| {
        world
            .get_resource::<InputState>()
            .map_or((0.0, 0.0), |input| input.cursor())
    });
    if !x.is_null() {
        // SAFETY: C# passes valid, writable f32 pointers.
        unsafe { *x = cx };
    }
    if !y.is_null() {
        // SAFETY: as above.
        unsafe { *y = cy };
    }
}

// The `Time` resource is engine-side, so these live here (like the input
// functions) and read it through the active-world seam. Outside a dispatch
// window, or before the resource exists, they report zero.

fn with_time<R: Default>(query: impl FnOnce(&Time) -> R) -> R {
    orrin_script::with_world(R::default(), |world| {
        world
            .get_resource::<Time>()
            .map_or_else(R::default, |time| query(&time))
    })
}

extern "C" fn time_delta() -> f32 {
    with_time(|time| time.delta_time())
}

extern "C" fn time_total() -> f32 {
    with_time(|time| time.elapsed_time())
}

extern "C" fn time_frame_count() -> u64 {
    with_time(|time| time.frame_count())
}

// Read-only world inspection for scripts. These are leaf calls: any RefCell
// borrow a query takes lives only inside the `with_world` closure and is
// released before control returns to C#, so no storage borrow is ever held
// across a dispatch. The find functions copy results into caller-owned memory
// for the same reason — results outlive the query borrow, never vice versa.
//
// `kind` numbering is lock-step with C# `Orrin.ComponentKind` (same rule as
// key codes: append, never renumber): 0 = Transform (LocalTransform), 1 = Tag.

extern "C" fn find_by_tag(tag: *const c_char, out: *mut CEntity) -> bool {
    if tag.is_null() || out.is_null() {
        return false;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let tag = unsafe { CStr::from_ptr(tag) }.to_string_lossy();
    orrin_script::with_world(false, |world| {
        let found = world
            .query::<&Tag>()
            .find(|_, t| t.as_str() == tag.as_ref());

        match found {
            Some(e) => {
                // SAFETY: `out` was null-checked above; C# passes a pointer to
                // a single stack-allocated Entity slot (see Native.FindByTag).
                unsafe {
                    *out = CEntity {
                        index: e.index,
                        generation: e.generation,
                    }
                }
                true
            }
            None => false,
        }
    })
}

extern "C" fn find_all_by_tag(tag: *const c_char, out: *mut CEntity, capacity: i32) -> i32 {
    if tag.is_null() || (out.is_null() && capacity > 0) {
        return 0;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let tag = unsafe { CStr::from_ptr(tag) }.to_string_lossy();
    orrin_script::with_world(0, |world| {
        let mut matches: Vec<CEntity> = Vec::new();
        world.query::<&Tag>().for_each(|e, t| {
            if t.as_str() == tag.as_ref() {
                matches.push(CEntity {
                    index: e.index,
                    generation: e.generation,
                });
            }
        });

        let n = matches.len().min(capacity.max(0) as usize);
        if n > 0 {
            // SAFETY: C# guarantees `out` points at `capacity` writable CEntity slots
            // (pinned managed Entity[] in Native.FindAllByTag); src is our own Vec,
            // so the ranges cannot overlap.
            unsafe { std::ptr::copy_nonoverlapping(matches.as_ptr(), out, n) };
        }
        matches.len() as i32
    })
}

extern "C" fn has_component(entity: CEntity, kind: u32) -> bool {
    orrin_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };

        match kind {
            0 => world.has::<LocalTransform>(entity),
            1 => world.has::<Tag>(entity),
            2 => world.has::<Collider>(entity),
            _ => false,
        }
    })
}

// String out-param protocol: returns the tag's UTF-8 byte length (no nul
// terminator — the return value carries the size), or -1 if the entity has no
// Tag. Writes min(len, capacity) bytes; C# retries with an exact-size buffer
// when len > capacity.
extern "C" fn get_tag(entity: CEntity, out: *mut c_char, capacity: i32) -> i32 {
    if out.is_null() && capacity > 0 {
        return -1;
    }
    orrin_script::with_world(-1, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };

        match world.get::<Tag>(entity) {
            Some(tag) => {
                let bytes = tag.as_str().as_bytes();
                let n = bytes.len().min(capacity.max(0) as usize);
                if n > 0 {
                    // SAFETY: C# guarantees `out` points at `capacity` writable bytes
                    // (stackalloc'd or pinned in Native.GetTag); src is the tag's own
                    // storage, so the ranges cannot overlap.
                    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out as *mut u8, n) };
                }
                bytes.len() as i32
            }
            None => -1,
        }
    })
}

extern "C" fn set_tag(entity: CEntity, tag: *const c_char) -> bool {
    if tag.is_null() {
        return false;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let tag = unsafe { CStr::from_ptr(tag) }
        .to_string_lossy()
        .into_owned();
    orrin_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        if !world.is_alive(entity) {
            return false;
        }
        COMMANDS.with(|commands| commands.borrow_mut().push(Command::SetTag { entity, tag }));
        true
    })
}

fn queue_add_collider(entity: CEntity, collider: Collider) -> bool {
    orrin_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        if !world.is_alive(entity) {
            return false;
        }
        COMMANDS.with(|commands| {
            commands
                .borrow_mut()
                .push(Command::AddCollider { entity, collider })
        });
        true
    })
}

extern "C" fn add_box_collider(
    entity: CEntity,
    hx: f32,
    hy: f32,
    hz: f32,
    is_trigger: bool,
) -> bool {
    queue_add_collider(
        entity,
        Collider {
            shape: ColliderShape::Box {
                half_extents: Vec3::new(hx, hy, hz),
            },
            is_trigger,
        },
    )
}

extern "C" fn add_sphere_collider(entity: CEntity, radius: f32, is_trigger: bool) -> bool {
    queue_add_collider(
        entity,
        Collider {
            shape: ColliderShape::Sphere { radius },
            is_trigger,
        },
    )
}

extern "C" fn set_material(entity: CEntity, material: *const c_char) -> bool {
    if material.is_null() {
        return false;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let name = unsafe { CStr::from_ptr(material) }.to_string_lossy();
    orrin_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        if !world.is_alive(entity) {
            return false;
        }
        // Resolve now so a bad name fails loudly at the call site (same rule
        // as spawn_renderable).
        let Some(material) = world
            .get_resource::<Assets>()
            .and_then(|assets| assets.material(&name))
        else {
            eprintln!("[script] set_material: unknown material {name:?}");
            return false;
        };
        COMMANDS.with(|commands| {
            commands
                .borrow_mut()
                .push(Command::SetMaterial { entity, material })
        });
        true
    })
}

extern "C" fn add_script(entity: CEntity, type_name: *const c_char) -> bool {
    if type_name.is_null() {
        return false;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let type_name = unsafe { CStr::from_ptr(type_name) }
        .to_string_lossy()
        .into_owned();
    orrin_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        if !world.is_alive(entity) {
            return false;
        }
        COMMANDS.with(|commands| {
            commands
                .borrow_mut()
                .push(Command::AttachScript { entity, type_name })
        });
        true
    })
}

// Structural edits requested from inside a script dispatch are queued and
// applied by `apply_commands` once the dispatch window closes. Direct mutation
// happens to be safe today (the tick holds no borrows while dispatching), but
// deferral keeps two hazards off the table for good: a despawn dropping a
// `ScriptComponent` (and freeing its GCHandle) while this tick's handle list
// still references it, and any future engine code that holds borrows during
// dispatch. Entity ids are still reserved eagerly — the allocator touches no
// component storage — so scripts get a real handle back synchronously.

enum Command {
    SpawnRenderable {
        entity: Entity,
        mesh: MeshHandle,
        material: MaterialHandle,
        transform: CTransform,
    },
    Despawn(Entity),
    // Adding a component changes which entities queries match, so it defers
    // like the other structural edits; a same-tick FindByTag won't see it.
    SetTag {
        entity: Entity,
        tag: String,
    },
    AddCollider {
        entity: Entity,
        collider: Collider,
    },
    SetMaterial {
        entity: Entity,
        material: MaterialHandle,
    },
    // Applied by `Scripting::apply_commands` (needs the host to create the
    // managed instance); the new behaviour gets OnEnable/OnStart next tick.
    AttachScript {
        entity: Entity,
        type_name: String,
    },
    // `None` detaches. Validated at the call, applied here — see `set_parent`.
    Reparent {
        child: Entity,
        parent: Option<Entity>,
        keep_world: bool,
    },
}

thread_local! {
    static COMMANDS: RefCell<Vec<Command>> = const { RefCell::new(Vec::new()) };
}

extern "C" fn spawn_renderable(
    mesh: *const c_char,
    material: *const c_char,
    transform: *const CTransform,
) -> CEntity {
    if mesh.is_null() || material.is_null() || transform.is_null() {
        return CEntity::NULL;
    }
    // SAFETY: C# passes valid, null-terminated UTF-8 buffers and a valid transform.
    let mesh_name = unsafe { CStr::from_ptr(mesh) }.to_string_lossy();
    let material_name = unsafe { CStr::from_ptr(material) }.to_string_lossy();
    let transform = unsafe { *transform };

    orrin_script::with_world(CEntity::NULL, |world| {
        // Resolve asset handles now so a bad name fails loudly at the call
        // site instead of silently when the queue drains.
        let (mesh, material) = {
            let Some(assets) = world.get_resource::<Assets>() else {
                eprintln!("[script] spawn_renderable: no Assets resource");
                return CEntity::NULL;
            };
            match (assets.mesh(&mesh_name), assets.material(&material_name)) {
                (Some(mesh), Some(material)) => (mesh, material),
                (mesh, material) => {
                    if mesh.is_none() {
                        eprintln!("[script] spawn_renderable: unknown mesh {mesh_name:?}");
                    }
                    if material.is_none() {
                        eprintln!("[script] spawn_renderable: unknown material {material_name:?}");
                    }
                    return CEntity::NULL;
                }
            }
        };

        let entity = world.spawn();
        COMMANDS.with(|commands| {
            commands.borrow_mut().push(Command::SpawnRenderable {
                entity,
                mesh,
                material,
                transform,
            })
        });
        CEntity {
            index: entity.index,
            generation: entity.generation,
        }
    })
}

extern "C" fn despawn(entity: CEntity) -> bool {
    orrin_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        if !world.is_alive(entity) {
            return false;
        }
        COMMANDS.with(|commands| commands.borrow_mut().push(Command::Despawn(entity)));
        true
    })
}

// Debug logging routes into the engine's `LogBuffer` resource (surfaced by the
// editor console), stamped with the current frame. Like input/time, the sink is
// an engine-side resource, so the real impls live here and reach it through the
// active-world seam.
//
// Outside a dispatch window there is no world to reach, and the cases that hit
// that path — OnDestroy during a despawn, and everything C# reports during a
// hot reload — are exactly the ones worth hearing about, so the message falls
// back to stderr instead of being dropped. It just doesn't reach the console
// panel.
fn log_at(level: LogLevel, message: *const c_char) {
    if message.is_null() {
        return;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let text = unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned();
    let buffered = orrin_script::with_world(false, |world| {
        let frame = world
            .get_resource::<Time>()
            .map_or(0, |time| time.frame_count());
        match world.get_resource_mut::<LogBuffer>() {
            Some(mut log) => {
                log.push(level, text.clone(), frame);
                true
            }
            None => false,
        }
    });
    if !buffered {
        match level {
            LogLevel::Info => println!("[c#] {text}"),
            LogLevel::Warning => eprintln!("[c# WARN] {text}"),
            LogLevel::Error => eprintln!("[c# ERROR] {text}"),
        }
    }
}

extern "C" fn log_info(message: *const c_char) {
    log_at(LogLevel::Info, message);
}

extern "C" fn log_warn(message: *const c_char) {
    log_at(LogLevel::Warning, message);
}

extern "C" fn log_error(message: *const c_char) {
    log_at(LogLevel::Error, message);
}

// Debug lines are collected into the engine's per-frame `DebugLines` resource;
// the line pass reads them, and they expire by `duration` (<= 0 = one frame).
// Editor-only: in export builds C# strips the call and this resource simply
// isn't fed.
extern "C" fn debug_draw_line(
    fx: f32,
    fy: f32,
    fz: f32,
    tx: f32,
    ty: f32,
    tz: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
    duration: f32,
) {
    orrin_script::with_world((), |world| {
        let now = world
            .get_resource::<Time>()
            .map_or(0.0, |time| time.elapsed_time());
        if let Some(mut lines) = world.get_resource_mut::<DebugLines>() {
            lines.push(
                Vec3::new(fx, fy, fz),
                Vec3::new(tx, ty, tz),
                [r, g, b, a],
                now,
                duration,
            );
        }
    });
}

fn build_api() -> OrrinApi {
    OrrinApi {
        get_transform,
        set_transform,
        key_down,
        key_pressed,
        key_released,
        mouse_button_down,
        cursor_pos,
        spawn_renderable,
        despawn,
        time_delta,
        time_total,
        time_frame_count,
        find_by_tag,
        find_all_by_tag,
        has_component,
        get_tag,
        set_tag,
        add_box_collider,
        add_sphere_collider,
        set_material,
        add_script,
        log: log_info,
        log_warn,
        log_error,
        debug_draw_line,
        get_world_transform,
        set_world_transform,
        get_parent,
        set_parent,
        ..orrin_script::default_api()
    }
}

/// What a reload did, for the editor console.
///
/// Failures are deliberately non-fatal and non-destructive: a reload that can't
/// find or load the new assembly leaves the session running exactly the code it
/// already had, because the swap is staged and nothing is torn down until the
/// new image has been accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReloadOutcome {
    /// The swap happened. `restored` counts behaviours re-created from the new
    /// assembly, `lost` those whose type no longer exists in it (renamed or
    /// deleted — the entity keeps its other components, just not the script).
    Swapped {
        restored: usize,
        lost: usize,
        /// The retired assembly outlived its unload. The new code is live; the
        /// old one just never unmapped.
        leaked: bool,
    },
    /// The new assembly was rejected; the previous one is still running.
    Rejected(GameAssemblyStatus),
}

impl std::fmt::Display for ReloadOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Swapped {
                restored,
                lost,
                leaked,
            } => {
                write!(f, "scripts reloaded: {restored} restored")?;
                if *lost > 0 {
                    write!(f, ", {lost} dropped (type no longer in the assembly)")?;
                }
                if *leaked {
                    write!(f, "; the previous assembly did not unload")?;
                }
                Ok(())
            }
            Self::Rejected(status) => {
                write!(
                    f,
                    "reload rejected ({status}); still running the previous build"
                )
            }
        }
    }
}

impl ReloadOutcome {
    /// Console severity: anything that didn't swap is a warning, since the
    /// developer asked for new code and is still looking at the old.
    pub fn level(&self) -> LogLevel {
        match self {
            Self::Swapped {
                lost: 0,
                leaked: false,
                ..
            } => LogLevel::Info,
            Self::Swapped { .. } => LogLevel::Warning,
            Self::Rejected(_) => LogLevel::Warning,
        }
    }
}

/// One script's lifecycle state for the duration of a tick, lifted out of the
/// world so no storage borrow is held while C# runs. Written back at the end of
/// the tick.
struct Pending {
    entity: Entity,
    handle: u64,
    started: bool,
    enabled: bool,
    active: bool,
    /// Set the moment a hook throws this tick, so the later phases (collision,
    /// update) skip a script that faulted during activation.
    faulted: bool,
}

pub struct Scripting {
    host: ScriptHost,
    /// The game DLL this session loaded, kept so a reload can re-read the same
    /// path after the build tool has overwritten it.
    game_dll: PathBuf,
    /// `tick`'s scratch, kept between frames for the allocation and nothing
    /// else — every tick clears it before use and it is never read in between.
    ///
    /// `tick` *takes* these rather than holding a `RefMut`, because the buffers
    /// are live across the dispatch window: a borrow held there would turn any
    /// future re-entry into a `RefCell` panic instead of a compile error, and
    /// the whole point of the window is that C# may call back in.
    pending: RefCell<Vec<Pending>>,
    /// `entity -> index into pending`, so routing a collision event costs a
    /// hash lookup instead of a scan over every scripted entity. Built only on
    /// ticks that actually have events.
    routing: RefCell<FxHashMap<Entity, usize>>,
}

impl Scripting {
    /// Locate the engine's own bindings assembly (`Orrin.dll` plus its
    /// runtimeconfig) by probing `bin/{Debug,Release}/net*` under `bindings_dir`
    /// and picking the most recently built.
    ///
    /// This is engine-owned and never comes from a project: `Orrin.dll` boots
    /// CoreCLR and holds every ABI entry point, so it lives in the default load
    /// context for the life of the process. A project supplies only its *game*
    /// assembly — see [`find_game_assembly`](Self::find_game_assembly).
    pub fn find_bindings_dir(bindings_dir: &Path) -> Option<PathBuf> {
        Self::newest_under(bindings_dir, |dir| {
            let dll = dir.join("Orrin.dll");
            (dir.join("Orrin.runtimeconfig.json").is_file() && dll.is_file()).then_some(dll)
        })
        .map(|(dir, _)| dir)
    }

    /// The bindings sitting directly beside the engine executable — the layout
    /// an exported build ships, and the only one that resolves when the engine
    /// is launched from inside a project directory rather than the repo root.
    pub fn bindings_beside_executable() -> Option<PathBuf> {
        let dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
        (dir.join("Orrin.dll").is_file() && dir.join("Orrin.runtimeconfig.json").is_file())
            .then_some(dir)
    }

    /// Locate a project's built game assembly: `<scripts_dir>/bin/{Debug,
    /// Release}/net*/<assembly_name>.dll`, most recently built wins.
    ///
    /// `assembly_name` comes from the assembly half of the manifest's
    /// `scripts.entry` (`"HelloOrrin.Main, HelloOrrin"` → `HelloOrrin`), so
    /// the manifest needs no separate field for it and the two can't drift.
    pub fn find_game_assembly(scripts_dir: &Path, assembly_name: &str) -> Option<PathBuf> {
        let file = format!("{assembly_name}.dll");
        Self::newest_under(scripts_dir, |dir| {
            let dll = dir.join(&file);
            dll.is_file().then_some(dll)
        })
        .map(|(_, dll)| dll)
    }

    /// Probe `root/bin/{Debug,Release}/*` for directories `accept` recognizes,
    /// returning the one whose named file is newest.
    fn newest_under(
        root: &Path,
        accept: impl Fn(&Path) -> Option<PathBuf>,
    ) -> Option<(PathBuf, PathBuf)> {
        let mut best: Option<(SystemTime, PathBuf, PathBuf)> = None;
        for config in ["Debug", "Release"] {
            let Ok(entries) = std::fs::read_dir(root.join("bin").join(config)) else {
                continue;
            };
            for entry in entries.flatten() {
                let dir = entry.path();
                let Some(file) = accept(&dir) else {
                    continue;
                };
                let modified = file
                    .metadata()
                    .and_then(|meta| meta.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                if best.as_ref().is_none_or(|(t, _, _)| modified > *t) {
                    best = Some((modified, dir, file));
                }
            }
        }
        best.map(|(_, dir, file)| (dir, file))
    }

    /// The assembly half of an assembly-qualified type name
    /// (`"HelloOrrin.Main, HelloOrrin"` → `"HelloOrrin"`), or `None` for a
    /// bare type name.
    pub fn assembly_of(entry_type: &str) -> Option<&str> {
        entry_type
            .split_once(',')
            .map(|(_, assembly)| assembly.trim())
            .filter(|assembly| !assembly.is_empty())
    }

    /// Boot the runtime from the bindings in `bindings_dir`, then load the
    /// project's game assembly from `game_dll`. Returns `None` (with a logged
    /// reason) if either step fails — the engine then runs without scripting
    /// rather than half-initialized.
    pub fn boot(bindings_dir: &Path, game_dll: &Path) -> Option<Self> {
        let host = match ScriptHost::boot(&build_api(), bindings_dir) {
            Ok(host) => host,
            Err(err) => {
                // Named, and with the fix attached, because scripting is a
                // default feature: this is the first thing a machine with no
                // .NET prints, and the bare hostfxr error underneath it is
                // "No such file or directory (os error 2)".
                eprintln!(
                    "scripting disabled: could not host the .NET runtime from {} ({err})\n\
                     \x20 the engine runs without it. Install the .NET SDK and \
                     `dotnet build scripting/Orrin`, or build with \
                     `--no-default-features` to leave scripting out entirely.",
                    bindings_dir.display(),
                );
                return None;
            }
        };

        // The initial load is the same staged swap a reload performs, with
        // nothing live to retire: stage, then commit.
        let scripting = Self {
            host,
            game_dll: game_dll.to_path_buf(),
            pending: RefCell::new(Vec::new()),
            routing: RefCell::new(FxHashMap::default()),
        };
        match scripting.stage() {
            Ok(()) => {}
            Err(status) => {
                eprintln!(
                    "scripting disabled: could not load {} ({status})",
                    game_dll.display()
                );
                return None;
            }
        }
        let status = scripting.host.commit_game();
        if !status.succeeded() {
            eprintln!("scripting disabled: {status}");
            return None;
        }
        Some(scripting)
    }

    /// Stage `self.game_dll` in a fresh load context without retiring the live
    /// one. On `Err` nothing has changed and no rollback is owed — the managed
    /// side only records a pending context on success.
    fn stage(&self) -> Result<(), GameAssemblyStatus> {
        let Some(path) = self.game_dll.to_str().and_then(|p| CString::new(p).ok()) else {
            return Err(GameAssemblyStatus::BadArgument);
        };
        let status = self.host.stage_game(&path);
        if status.succeeded() {
            Ok(())
        } else {
            Err(status)
        }
    }

    /// Swap in a freshly built game assembly without restarting the engine.
    ///
    /// None of the four phases commute:
    ///
    /// - **Staging is first** because it is the only step that can abort for
    ///   free, while every live behaviour is still intact. Past it there is no
    ///   way back, so an early return added below would owe a `rollback_game()`.
    /// - **Collection is separate from teardown** because `query` holds a
    ///   `RefCell` borrow of the storage the teardown loop needs mutably — a
    ///   runtime panic, not a compile error.
    /// - **Capture precedes destroy, and commit follows both.** Dropping the
    ///   component frees its `GCHandle`, leaving nothing to snapshot; and
    ///   `commit_game` retires the old load context only once *every* handle
    ///   into it is gone. Faulted scripts are included for that second reason:
    ///   skipping them strands a handle and leaks a context per reload after.
    /// - **Re-creation follows commit**, since `Behaviours.ResolveType` searches
    ///   the live context — attach earlier and every behaviour is faithfully
    ///   rebuilt from the *old* code.
    pub fn reload(&self, world: &mut World) -> ReloadOutcome {
        if let Err(status) = self.stage() {
            return ReloadOutcome::Rejected(status);
        }

        struct Reloading {
            entity: Entity,
            handle: u64,
            type_name: String,
            enabled: bool,
            started: bool,
            snapshot: u64,
        }

        let mut pending: Vec<Reloading> = Vec::new();
        world
            .query::<&ScriptComponent>()
            .for_each(|entity, script| {
                pending.push(Reloading {
                    entity,
                    handle: script.handle,
                    type_name: script.type_name.clone(),
                    enabled: script.enabled,
                    started: script.started,
                    snapshot: 0,
                })
            });

        for script in pending.iter_mut() {
            script.snapshot = self.host.capture_state(script.handle);
            // Dropped here and not bound to anything: the `Drop` impl is what
            // fires OnDisable/OnDestroy (against the old code, correctly) and
            // frees the handle. A binding that outlived the commit below would
            // pin the retired assembly.
            let _ = world.remove::<ScriptComponent>(script.entity);
        }

        let status = self.host.commit_game();
        if !status.succeeded() {
            return ReloadOutcome::Rejected(status);
        }
        // Not a failure: the new code is live regardless, and a leak is not
        // something the engine can act on from here — only report.
        let leaked = status == GameAssemblyStatus::Leaked;

        let mut restored = 0usize;
        let mut lost = 0usize;

        for script in &pending {
            let handle = self.attach_with(
                world,
                script.entity,
                &script.type_name,
                script.started,
                script.enabled,
            );
            // The type is no longer in the assembly — renamed or deleted, an
            // ordinary thing to do between builds. The entity keeps every other
            // component; `attach_with` has already named the missing type.
            if handle == 0 {
                lost += 1;
                continue;
            }
            if script.snapshot != 0 {
                self.host.apply_state(handle, script.snapshot);
            }
            restored += 1;
        }

        // Last, so it only drops the snapshots belonging to the `lost` scripts.
        self.host.discard_states();

        ReloadOutcome::Swapped {
            restored,
            lost,
            leaked,
        }
    }

    /// Attach a C# `Behaviour` (by assembly-qualified type name) to `entity`.
    /// A no-op when the type isn't in the loaded game assembly.
    pub fn attach(&self, world: &mut World, entity: Entity, type_name: &str) {
        self.attach_with(world, entity, type_name, false, true);
    }

    /// `attach`, with the lifecycle position a hot reload has to carry over.
    /// Returns the new managed handle, or `0` if the type could not be created.
    pub fn attach_with(
        &self,
        world: &mut World,
        entity: Entity,
        type_name: &str,
        started: bool,
        enabled: bool,
    ) -> u64 {
        let Ok(name) = CString::new(type_name) else {
            return 0;
        };
        let handle = self.host.create(
            CEntity {
                index: entity.index,
                generation: entity.generation,
            },
            &name,
        );
        if handle == 0 {
            // Silence here used to mean "the scene came up with no scripts and
            // nothing said why" — the commonest cause is a typo'd or renamed
            // entry type, which is invisible otherwise.
            eprintln!(
                "[script] no Behaviour `{type_name}` in the loaded game assembly \
                 (check the type name and that the project has been rebuilt)"
            );
        }
        if handle != 0 {
            world.insert(
                entity,
                ScriptComponent {
                    handle,
                    type_name: type_name.to_owned(),
                    started,
                    // Desired-on from birth; `active` stays false until the
                    // first tick actually dispatches OnEnable.
                    enabled,
                    active: false,
                    faulted: false,
                },
            );
        }
        handle
    }

    /// The managed host, for the steps of a reload that talk to C# directly
    /// (state capture, the context commit).
    pub fn host(&self) -> &ScriptHost {
        &self.host
    }

    /// Request an activation change; the transition (OnEnable/OnDisable) is
    /// dispatched by the next `tick`, not here.
    pub fn set_enabled(&self, world: &mut World, entity: Entity, enabled: bool) {
        if let Some(mut script) = world.get_mut::<ScriptComponent>(entity) {
            script.enabled = enabled;
        }
    }

    /// Tick every script. Collect handles first, drop the world borrow, then
    /// dispatch — so the ABI's `&mut World` reconstruction never aliases.
    ///
    /// Dispatch order inside the window mirrors Unity: activation transitions
    /// (OnEnable/OnStart/OnDisable), then this frame's collision callbacks,
    /// then OnUpdate.
    pub fn tick(&self, world: &mut World, delta_time: f32) {
        let mut pending = self.pending.take();
        pending.clear();
        world
            .query::<&ScriptComponent>()
            .for_each(|entity, script| {
                // Already-faulted scripts are inert: never collected, never
                // dispatched to, until something clears the flag.
                if script.faulted {
                    return;
                }
                pending.push(Pending {
                    entity,
                    handle: script.handle,
                    started: script.started,
                    enabled: script.enabled,
                    active: script.active,
                    faulted: false,
                })
            });
        if pending.is_empty() {
            self.pending.replace(pending);
            return;
        }

        // Take this frame's collision events before the dispatch window opens
        // — the resource borrow must not be held while C# can re-enter the
        // world. Leftovers on an early return are cleared by the next
        // `collision::run`.
        let events: Vec<CollisionEvent> = world
            .get_resource_mut::<CollisionState>()
            .map_or_else(Vec::new, |mut state| std::mem::take(&mut state.events));

        // Built here and not inside the window, and only when there is
        // something to route: a tick with no collisions should not pay for an
        // index nothing reads.
        let mut routing = self.routing.take();
        routing.clear();
        if !events.is_empty() {
            routing.extend(
                pending
                    .iter()
                    .enumerate()
                    .map(|(i, script)| (script.entity, i)),
            );
        }

        orrin_script::with_active_world(world, || {
            for script in &mut pending {
                if script.enabled && !script.active {
                    // Mirror the managed side, which flips its own `Active`
                    // before invoking the hook: if OnEnable/OnStart throws, the
                    // state still advances and we just stop dispatching.
                    script.active = true;
                    if self.host.enable(script.handle) {
                        script.faulted = true;
                        continue;
                    }
                    if !script.started {
                        script.started = true;
                        if self.host.start(script.handle) {
                            script.faulted = true;
                        }
                    }
                } else if !script.enabled && script.active {
                    script.active = false;
                    if self.host.disable(script.handle) {
                        script.faulted = true;
                    }
                }
            }

            // Route each event to both participants' scripts (if any). The
            // stored normal points a → b, so b's callback sees it negated —
            // "from me toward the other", both sides.
            for event in &events {
                for (target, other, flip) in [(event.a, event.b, false), (event.b, event.a, true)] {
                    // An entity holds at most one `ScriptComponent`, so the
                    // index names the same entry the linear scan used to find.
                    // The activation flags are still re-read per event, since a
                    // fault raised by an earlier one must stop delivery here.
                    let Some(&index) = routing.get(&target) else {
                        continue;
                    };
                    let script = &mut pending[index];
                    if !script.active || script.faulted {
                        continue;
                    }
                    let normal = if flip { -event.normal } else { event.normal };
                    let collision = CCollision {
                        other: CEntity {
                            index: other.index,
                            generation: other.generation,
                        },
                        point: event.point.to_array(),
                        normal: normal.to_array(),
                    };
                    let faulted = match event.kind {
                        CollisionEventKind::Enter => {
                            self.host.collision_enter(script.handle, &collision)
                        }
                        CollisionEventKind::Exit => {
                            self.host.collision_exit(script.handle, &collision)
                        }
                    };
                    if faulted {
                        script.faulted = true;
                    }
                }
            }

            for script in &mut pending {
                if script.active && !script.faulted && self.host.update(script.handle, delta_time) {
                    script.faulted = true;
                }
            }
        });

        // Structural changes the scripts queued land now, after every script
        // has run — so this frame's extraction already sees new renderables.
        self.apply_commands(world);

        for script in &pending {
            if let Some(mut component) = world.get_mut::<ScriptComponent>(script.entity) {
                component.active = script.active;
                component.started = script.started;
                // Persist a fault raised this tick so every future tick skips
                // it; a script that threw during OnUpdate does not run again.
                component.faulted = script.faulted;
            }
        }

        self.pending.replace(pending);
        self.routing.replace(routing);
    }

    /// Apply the structural changes scripts queued during a dispatch. Runs
    /// with no other world borrows held, so a despawn can drop a
    /// `ScriptComponent` (and free its GCHandle) safely.
    fn apply_commands(&self, world: &mut World) {
        let commands: Vec<Command> = COMMANDS.with(|c| c.borrow_mut().drain(..).collect());
        for command in commands {
            match command {
                Command::SpawnRenderable {
                    entity,
                    mesh,
                    material,
                    transform,
                } => {
                    world.insert(entity, Name::new(format!("Scripted {}", entity.index)));
                    world.insert(
                        entity,
                        LocalTransform::from(Transform {
                            translation: Vec3::from_array(transform.position),
                            rotation: Quat::from_array(transform.rotation),
                            scale: Vec3::from_array(transform.scale),
                        }),
                    );
                    world.insert(entity, mesh);
                    world.insert(entity, material);
                }
                Command::Despawn(entity) => {
                    // Takes the subtree with it, so a script that despawns a
                    // parent does not leave its children behind.
                    crate::scene::despawn_recursive(world, entity);
                }
                Command::SetTag { entity, tag } => {
                    // `insert` is a stale-handle no-op, so a despawn queued earlier
                    // this tick wins — same rule as the other commands.
                    world.insert(entity, Tag::new(tag));
                }
                Command::AddCollider { entity, collider } => {
                    world.insert(entity, collider);
                }
                Command::SetMaterial { entity, material } => {
                    world.insert(entity, material);
                }
                Command::Reparent {
                    child,
                    parent,
                    keep_world,
                } => {
                    // Re-validated by `reparent` itself: the world has moved on
                    // since the call, and the parent may have been despawned by
                    // an earlier command in this same drain.
                    let _ = crate::scene::reparent(world, child, parent, keep_world);
                }
                Command::AttachScript { entity, type_name } => {
                    // Creates the managed instance now; the enable/start pair
                    // dispatches on the next tick, outside any window.
                    self.attach(world, entity, &type_name);
                }
            }
        }
    }
}
