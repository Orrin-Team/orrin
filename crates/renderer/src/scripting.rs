//! C# scripting integration (engine side), behind the `scripting` feature.
//!
//! The generic ABI lives in `ferron-script`; the transform functions need the
//! engine's own `LocalTransform`, so they're defined here and assembled into the
//! table with `..ferron_script::default_api()`.

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use glam::{Quat, Vec3};

use ferron_ecs::{Entity, World};
use ferron_script::{CCollision, CEntity, CTransform, FerronApi, ScriptHost};

use crate::collision::{CollisionEvent, CollisionEventKind, CollisionState};
use crate::scene::{
    Assets, Collider, ColliderShape, DebugLines, InputState, LocalTransform, LogBuffer, LogLevel,
    MaterialHandle, MeshHandle, Name, ScriptComponent, Tag, Time, Transform,
};

extern "C" fn get_transform(entity: CEntity, out: *mut CTransform) -> bool {
    if out.is_null() {
        return false;
    }
    ferron_script::with_world(false, |world| {
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
    ferron_script::with_world(false, |world| {
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

// The `InputState` resource is engine-side, so these live here (like the
// transform functions) and read it through the active-world seam. Outside a
// dispatch window, or before the resource exists, they report "nothing held".

fn with_input(query: impl FnOnce(&InputState) -> bool) -> bool {
    ferron_script::with_world(false, |world| {
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
    let (cx, cy) = ferron_script::with_world((0.0, 0.0), |world| {
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
    ferron_script::with_world(R::default(), |world| {
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
// `kind` numbering is lock-step with C# `Ferron.ComponentKind` (same rule as
// key codes: append, never renumber): 0 = Transform (LocalTransform), 1 = Tag.

extern "C" fn find_by_tag(tag: *const c_char, out: *mut CEntity) -> bool {
    if tag.is_null() || out.is_null() {
        return false;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let tag = unsafe { CStr::from_ptr(tag) }.to_string_lossy();
    ferron_script::with_world(false, |world| {
        let found = world.query::<&Tag>().find(|_, t| t.as_str() == tag.as_ref());

        match found {
            Some(e) => {
                // SAFETY: `out` was null-checked above; C# passes a pointer to
                // a single stack-allocated Entity slot (see Native.FindByTag).
                unsafe { *out = CEntity { index: e.index, generation: e.generation } }
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
    ferron_script::with_world(0, |world| {
        let mut matches: Vec<CEntity> = Vec::new();
        world.query::<&Tag>().for_each(|e, t| {
            if t.as_str() == tag.as_ref() {
                matches.push(CEntity { index: e.index, generation: e.generation });
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
    ferron_script::with_world(false, |world| {
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
    ferron_script::with_world(-1, |world| {
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
    let tag = unsafe { CStr::from_ptr(tag) }.to_string_lossy().into_owned();
    ferron_script::with_world(false, |world| {
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
    ferron_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        if !world.is_alive(entity) {
            return false;
        }
        COMMANDS.with(|commands| {
            commands.borrow_mut().push(Command::AddCollider { entity, collider })
        });
        true
    })
}

extern "C" fn add_box_collider(entity: CEntity, hx: f32, hy: f32, hz: f32, is_trigger: bool) -> bool {
    queue_add_collider(
        entity,
        Collider {
            shape: ColliderShape::Box { half_extents: Vec3::new(hx, hy, hz) },
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
    ferron_script::with_world(false, |world| {
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
            commands.borrow_mut().push(Command::SetMaterial { entity, material })
        });
        true
    })
}

extern "C" fn add_script(entity: CEntity, type_name: *const c_char) -> bool {
    if type_name.is_null() {
        return false;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let type_name = unsafe { CStr::from_ptr(type_name) }.to_string_lossy().into_owned();
    ferron_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        if !world.is_alive(entity) {
            return false;
        }
        COMMANDS.with(|commands| {
            commands.borrow_mut().push(Command::AttachScript { entity, type_name })
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
    SetTag { entity: Entity, tag: String },
    AddCollider { entity: Entity, collider: Collider },
    SetMaterial { entity: Entity, material: MaterialHandle },
    // Applied by `Scripting::apply_commands` (needs the host to create the
    // managed instance); the new behaviour gets OnEnable/OnStart next tick.
    AttachScript { entity: Entity, type_name: String },
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

    ferron_script::with_world(CEntity::NULL, |world| {
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
    ferron_script::with_world(false, |world| {
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
// active-world seam. Outside a dispatch window (no active world) the message is
// dropped — a script can only log while it is ticking.
fn log_at(level: LogLevel, message: *const c_char) {
    if message.is_null() {
        return;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned();
    ferron_script::with_world((), |world| {
        let frame = world.get_resource::<Time>().map_or(0, |time| time.frame_count());
        if let Some(mut log) = world.get_resource_mut::<LogBuffer>() {
            log.push(level, text, frame);
        }
    });
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
    ferron_script::with_world((), |world| {
        let now = world.get_resource::<Time>().map_or(0.0, |time| time.elapsed_time());
        if let Some(mut lines) = world.get_resource_mut::<DebugLines>() {
            lines.push(Vec3::new(fx, fy, fz), Vec3::new(tx, ty, tz), [r, g, b, a], now, duration);
        }
    });
}

fn build_api() -> FerronApi {
    FerronApi {
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
        ..ferron_script::default_api()
    }
}

pub struct Scripting {
    host: ScriptHost,
}

impl Scripting {
    /// Locate the built managed assembly under `scripts_dir` by probing
    /// `bin/{Debug,Release}/net*` and picking the most recently built.
    ///
    /// The caller decides where `scripts_dir` comes from (project manifest or
    /// the built-in default) and handles the `FERRON_SCRIPT_DIR` override.
    pub fn find_assembly_dir(scripts_dir: &Path) -> Option<PathBuf> {
        let mut best: Option<(SystemTime, PathBuf)> = None;
        for config in ["Debug", "Release"] {
            let bin = scripts_dir.join("bin").join(config);
            let Ok(entries) = std::fs::read_dir(&bin) else {
                continue;
            };
            for entry in entries.flatten() {
                let dir = entry.path();
                let dll = dir.join("Ferron.dll");
                if !dir.join("Ferron.runtimeconfig.json").is_file() || !dll.is_file() {
                    continue;
                }
                let modified = dll
                    .metadata()
                    .and_then(|meta| meta.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                if best.as_ref().map_or(true, |(t, _)| modified > *t) {
                    best = Some((modified, dir));
                }
            }
        }
        best.map(|(_, dir)| dir)
    }

    /// Boot the runtime, loading the managed assembly from `assembly_dir`.
    /// Returns `None` (with a logged reason) if the runtime can't start.
    pub fn boot(assembly_dir: &Path) -> Option<Self> {
        match ScriptHost::boot(&build_api(), assembly_dir) {
            Ok(host) => Some(Self { host }),
            Err(err) => {
                eprintln!("scripting disabled: {err}");
                None
            }
        }
    }

    /// Attach a C# `Behaviour` (by assembly-qualified type name) to `entity`.
    pub fn attach(&self, world: &mut World, entity: Entity, type_name: &str) {
        let Ok(name) = CString::new(type_name) else {
            return;
        };
        let handle = self.host.create(
            CEntity {
                index: entity.index,
                generation: entity.generation,
            },
            &name,
        );
        if handle != 0 {
            world.insert(
                entity,
                ScriptComponent {
                    handle,
                    started: false,
                    // Desired-on from birth; `active` stays false until the
                    // first tick actually dispatches OnEnable.
                    enabled: true,
                    active: false,
                    faulted: false,
                },
            );
        }
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
        struct Pending {
            entity: Entity,
            handle: u64,
            started: bool,
            enabled: bool,
            active: bool,
            // Set the moment a hook throws this tick, so the later phases
            // (collision, update) skip a script that faulted during activation.
            faulted: bool,
        }

        let mut pending: Vec<Pending> = Vec::new();
        world.query::<&ScriptComponent>().for_each(|entity, script| {
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
            return;
        }

        // Take this frame's collision events before the dispatch window opens
        // — the resource borrow must not be held while C# can re-enter the
        // world. Leftovers on an early return are cleared by the next
        // `collision::run`.
        let events: Vec<CollisionEvent> = world
            .get_resource_mut::<CollisionState>()
            .map_or_else(Vec::new, |mut state| std::mem::take(&mut state.events));

        ferron_script::with_active_world(world, || {
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
                    let Some(script) = pending
                        .iter_mut()
                        .find(|script| script.entity == target && script.active && !script.faulted)
                    else {
                        continue;
                    };
                    let normal = if flip { -event.normal } else { event.normal };
                    let collision = CCollision {
                        other: CEntity { index: other.index, generation: other.generation },
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
                    world.despawn(entity);
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
                Command::AttachScript { entity, type_name } => {
                    // Creates the managed instance now; the enable/start pair
                    // dispatches on the next tick, outside any window.
                    self.attach(world, entity, &type_name);
                }
            }
        }
    }
}
