//! C# scripting integration (engine side), behind the `scripting` feature.
//!
//! The generic ABI lives in `orrin-script`; the transform functions need the
//! engine's own `LocalTransform`, so they're defined here and assembled into the
//! table with `..orrin_script::default_api()`.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use glam::{Mat4, Quat, Vec3};

use orrin_ecs::{Entity, FxHashMap, World};
use orrin_registry::{AssetId, ComponentId, Registry, ScriptBridge, Value, ValueError};
use orrin_script::{
    CCamera, CCollision, CEntity, CTransform, GameAssemblyStatus, OrrinApi, ScriptHost,
};

use crate::asset_db::{AssetDatabase, AssetKind, LoadState};
use crate::collision::{CollisionEvent, CollisionEventKind, CollisionState};
use crate::scene::prefab::{self, Prefabs};
use crate::scene::{
    ActionId, Actions, ActiveCamera, Assets, CameraComponent, Collider, ColliderShape, DebugLines,
    InputState, LoadedScenes, LocalTransform, LogLevel, MaterialHandle, MeshHandle, Name, Parent,
    PendingScripts, Rect, SceneHandle, ScriptComponent, Tag, TextAnchor, TextStyle, Time,
    Transform, UiCanvas, WorldTransform, color_bytes,
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

extern "C" fn mouse_button_pressed(button: u32) -> bool {
    with_input(|input| input.mouse_button_pressed(button))
}

extern "C" fn mouse_button_released(button: u32) -> bool {
    with_input(|input| input.mouse_button_released(button))
}

// Named input. `action_id` needs `&mut Actions` to intern, which is why it is
// the one input entry point that takes the resource mutably; the queries below
// only read.

extern "C" fn action_id(name: *const c_char) -> u32 {
    if name.is_null() {
        return orrin_script::NO_ACTION;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    orrin_script::with_world(orrin_script::NO_ACTION, |world| {
        world
            .get_resource_mut::<Actions>()
            .map_or(orrin_script::NO_ACTION, |mut actions| actions.id(&name).0)
    })
}

fn with_actions<R: Default>(query: impl FnOnce(&Actions) -> R) -> R {
    orrin_script::with_world(R::default(), |world| {
        world
            .get_resource::<Actions>()
            .map_or_else(R::default, |actions| query(&actions))
    })
}

extern "C" fn action_held(id: u32, player: u32) -> bool {
    with_actions(|actions| actions.held(ActionId(id), player as usize))
}

extern "C" fn action_pressed(id: u32, player: u32) -> bool {
    with_actions(|actions| actions.pressed(ActionId(id), player as usize))
}

extern "C" fn action_released(id: u32, player: u32) -> bool {
    with_actions(|actions| actions.released(ActionId(id), player as usize))
}

extern "C" fn axis_value(id: u32, player: u32) -> f32 {
    with_actions(|actions| actions.axis(ActionId(id), player as usize))
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
            3 => world.has::<CameraComponent>(entity),
            KIND_CANVAS => world.has::<crate::ui::Canvas>(entity),
            KIND_RECT_TRANSFORM => world.has::<crate::ui::RectTransform>(entity),
            KIND_IMAGE => world.has::<crate::ui::Image>(entity),
            KIND_TEXT => world.has::<crate::ui::Text>(entity),
            KIND_BUTTON => world.has::<crate::ui::Button>(entity),
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
            tracing::warn!("set_material: unknown material {name:?}");
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
    AddCamera {
        entity: Entity,
        camera: CameraComponent,
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
    AddUi {
        entity: Entity,
        component: UiComponent,
    },
    // `root` was reserved at the call; the instance is built into it here.
    InstantiatePrefab {
        root: Entity,
        prefab: AssetId,
    },
    // Despawns the scripts that queued it, which is why scene changes wait
    // here rather than happening at the call. See `scene::loaded`.
    LoadScene(AssetId),
    // `handle` was reserved at the call.
    LoadSceneAdditive {
        handle: SceneHandle,
        scene: AssetId,
    },
    UnloadScene(SceneHandle),
}

/// A UI component a script added, waiting for the commands to drain.
#[derive(Clone)]
enum UiComponent {
    Canvas(crate::ui::Canvas),
    RectTransform(crate::ui::RectTransform),
    Image(crate::ui::Image),
    Text(crate::ui::Text),
    Button(crate::ui::Button),
}

impl UiComponent {
    fn kind(&self) -> u32 {
        match self {
            Self::Canvas(_) => KIND_CANVAS,
            Self::RectTransform(_) => KIND_RECT_TRANSFORM,
            Self::Image(_) => KIND_IMAGE,
            Self::Text(_) => KIND_TEXT,
            Self::Button(_) => KIND_BUTTON,
        }
    }

    fn insert(self, world: &mut World, entity: Entity) {
        match self {
            Self::Canvas(c) => {
                world.insert(entity, c);
            }
            Self::RectTransform(c) => {
                world.insert(entity, c);
            }
            Self::Image(c) => {
                world.insert(entity, c);
            }
            Self::Text(c) => {
                world.insert(entity, c);
            }
            Self::Button(c) => {
                world.insert(entity, c);
            }
        }
    }
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
                tracing::warn!("spawn_renderable: no Assets resource");
                return CEntity::NULL;
            };
            match (assets.mesh(&mesh_name), assets.material(&material_name)) {
                (Some(mesh), Some(material)) => (mesh, material),
                (mesh, material) => {
                    if mesh.is_none() {
                        tracing::warn!("spawn_renderable: unknown mesh {mesh_name:?}");
                    }
                    if material.is_none() {
                        tracing::warn!("spawn_renderable: unknown material {material_name:?}");
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

extern "C" fn find_asset(path: *const c_char, out: *mut u8) -> u32 {
    if path.is_null() || out.is_null() {
        return orrin_script::ASSET_NONE;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let path = unsafe { CStr::from_ptr(path) }.to_string_lossy();
    orrin_script::with_world(orrin_script::ASSET_NONE, |world| {
        let in_database = world.get_resource::<AssetDatabase>().and_then(|db| {
            let entry = db.find(&path)?;
            let kind = match entry.kind {
                AssetKind::Prefab => orrin_script::ASSET_PREFAB,
                AssetKind::Mesh => orrin_script::ASSET_MESH,
                AssetKind::Texture => orrin_script::ASSET_TEXTURE,
                AssetKind::Material => orrin_script::ASSET_MATERIAL,
                AssetKind::Scene => orrin_script::ASSET_SCENE,
                AssetKind::Font => orrin_script::ASSET_FONT,
                AssetKind::Folder => return None,
            };
            Some((entry.id, kind))
        });
        // Without a project there is no database, and prefabs are indexed on
        // their own.
        let found = in_database.or_else(|| {
            let id = world.get_resource::<Prefabs>()?.find(&path)?;
            Some((id, orrin_script::ASSET_PREFAB))
        });
        let Some((id, kind)) = found else {
            return orrin_script::ASSET_NONE;
        };
        // SAFETY: `out` is the 16-byte buffer C# reserved for the id.
        unsafe { std::ptr::copy_nonoverlapping(id.to_bytes().as_ptr(), out, 16) };
        kind
    })
}

extern "C" fn asset_state(id: *const u8) -> u32 {
    if id.is_null() {
        return orrin_script::LOAD_FAILED;
    }
    // SAFETY: `id` is the 16 bytes of an `AssetRef` C# holds.
    let id = AssetId::from_bytes(unsafe { *id.cast::<[u8; 16]>() });
    orrin_script::with_world(orrin_script::LOAD_FAILED, |world| {
        let engine_built = world.get_resource::<Assets>().is_some_and(|assets| {
            assets.mesh_by_id(id).is_some() || assets.material_by_id(id).is_some()
        });
        let indexed = world
            .get_resource::<Prefabs>()
            .is_some_and(|prefabs| prefabs.get(id).is_some());
        if engine_built || indexed {
            return orrin_script::LOAD_READY;
        }
        match world.get_resource::<AssetDatabase>().map(|db| db.state(id)) {
            Some(LoadState::Loading) => orrin_script::LOAD_LOADING,
            Some(LoadState::Ready) => orrin_script::LOAD_READY,
            Some(LoadState::Failed) | None => orrin_script::LOAD_FAILED,
        }
    })
}

extern "C" fn instantiate_prefab(id: *const u8) -> CEntity {
    if id.is_null() {
        return CEntity::NULL;
    }
    // SAFETY: `id` is the 16 bytes of an `AssetRef` C# holds.
    let prefab = AssetId::from_bytes(unsafe { *id.cast::<[u8; 16]>() });
    orrin_script::with_world(CEntity::NULL, |world| {
        // Checked now so a bad reference fails loudly at the call, the rule
        // `spawn_renderable` follows for a bad name.
        let known = world
            .get_resource::<Prefabs>()
            .is_some_and(|prefabs| prefabs.get(prefab).is_some());
        if !known {
            tracing::warn!("Prefab.Instantiate: no prefab in the project has the id {prefab}");
            return CEntity::NULL;
        }
        let root = world.spawn();
        COMMANDS.with(|commands| {
            commands
                .borrow_mut()
                .push(Command::InstantiatePrefab { root, prefab })
        });
        CEntity {
            index: root.index,
            generation: root.generation,
        }
    })
}

/// Build a queued instance into the root its script was handed.
fn build_prefab(world: &mut World, registry: &Registry, root: Entity, prefab: AssetId) {
    match prefab::instantiate_into(world, registry, prefab, root) {
        Ok(issues) => {
            for issue in issues {
                tracing::warn!("Prefab.Instantiate: {issue}");
            }
        }
        Err(error) => {
            tracing::warn!("Prefab.Instantiate: {error}");
            world.despawn(root);
        }
    }
}

/// Whether `id` names a scene file the project has.
fn is_scene(world: &World, id: AssetId) -> bool {
    world
        .get_resource::<AssetDatabase>()
        .and_then(|db| db.get(id).map(|entry| entry.kind == AssetKind::Scene))
        .unwrap_or(false)
}

fn scene_id(id: *const u8) -> Option<AssetId> {
    // SAFETY: a non-null `id` is the 16 bytes of an `AssetRef` C# holds.
    (!id.is_null()).then(|| AssetId::from_bytes(unsafe { *id.cast::<[u8; 16]>() }))
}

extern "C" fn load_scene(id: *const u8) -> bool {
    let Some(scene) = scene_id(id) else {
        return false;
    };
    orrin_script::with_world(false, |world| {
        if !is_scene(world, scene) {
            tracing::warn!("SceneManager.Load: no scene in the project has the id {scene}");
            return false;
        }
        COMMANDS.with(|commands| commands.borrow_mut().push(Command::LoadScene(scene)));
        true
    })
}

extern "C" fn load_scene_additive(id: *const u8) -> u32 {
    let Some(scene) = scene_id(id) else {
        return 0;
    };
    orrin_script::with_world(0, |world| {
        if !is_scene(world, scene) {
            tracing::warn!("SceneManager.LoadAdditive: no scene in the project has the id {scene}");
            return 0;
        }
        if world.get_resource::<LoadedScenes>().is_none() {
            world.insert_resource(LoadedScenes::default());
        }
        let handle = world.resource_mut::<LoadedScenes>().reserve();
        COMMANDS.with(|commands| {
            commands
                .borrow_mut()
                .push(Command::LoadSceneAdditive { handle, scene })
        });
        handle.get()
    })
}

extern "C" fn unload_scene(handle: u32) -> bool {
    let Some(handle) = SceneHandle::from_raw(handle) else {
        tracing::warn!("SceneManager.Unload: that handle names no scene");
        return false;
    };
    orrin_script::with_world(false, |world| {
        let (active, listed) =
            world
                .get_resource::<LoadedScenes>()
                .map_or((false, false), |loaded| {
                    (
                        loaded.active().handle == handle,
                        loaded.get(handle).is_some(),
                    )
                });
        if active {
            tracing::warn!(
                "SceneManager.Unload: that is the active scene; Load another in its place"
            );
            return false;
        }
        let queued = COMMANDS.with(|commands| {
            commands.borrow().iter().any(|command| {
                matches!(command, Command::LoadSceneAdditive { handle: h, .. } if *h == handle)
            })
        });
        if !listed && !queued {
            tracing::warn!(
                "SceneManager.Unload: scene {} is not loaded; it was unloaded already, or a \
                 Load replaced it",
                handle.get()
            );
            return false;
        }
        COMMANDS.with(|commands| commands.borrow_mut().push(Command::UnloadScene(handle)));
        true
    })
}

extern "C" fn active_scene() -> u32 {
    orrin_script::with_world(0, |world| {
        world
            .get_resource::<LoadedScenes>()
            .map_or(0, |loaded| loaded.active().handle.get())
    })
}

/// The active camera entity's world position and rotation as of this call:
/// its own local transform, composed with its parent's last propagated world
/// one. Not its `WorldTransform`, which predates this tick's script writes,
/// so a script setting the rotation and then the position keeps both.
fn camera_entity_pose(world: &World, entity: Entity) -> (Vec3, Quat) {
    let local = world
        .get::<LocalTransform>(entity)
        .map_or(Mat4::IDENTITY, |transform| transform.matrix());
    let (_, rotation, position) =
        (crate::scene::parent_world_matrix(world, entity) * local).to_scale_rotation_translation();
    (position, rotation)
}

extern "C" fn get_camera(out: *mut CTransform) -> bool {
    if out.is_null() {
        return false;
    }
    orrin_script::with_world(false, |world| {
        let (position, rotation) = match crate::scene::active_camera(world) {
            Some(entity) => camera_entity_pose(world, entity),
            None => {
                let Some(camera) = world.get_resource::<crate::scene::Camera>() else {
                    return false;
                };
                (camera.position, camera.rotation())
            }
        };
        // SAFETY: `out` is a valid, writable `CTransform` supplied by C#.
        unsafe {
            *out = CTransform {
                position: position.to_array(),
                rotation: rotation.to_array(),
                scale: [1.0; 3],
            };
        }
        true
    })
}

/// With a camera entity active this moves the entity, keeping its scale, and
/// the view follows it at the end of the frame; writing the view directly
/// would be overwritten by that follow before anything drew it.
extern "C" fn set_camera(value: *const CTransform) -> bool {
    if value.is_null() {
        return false;
    }
    // SAFETY: `value` is a valid `CTransform` supplied by C#.
    let value = unsafe { *value };
    let position = Vec3::from_array(value.position);
    let rotation = Quat::from_array(value.rotation).normalize();
    orrin_script::with_world(false, |world| {
        if let Some(entity) = crate::scene::active_camera(world) {
            let local = crate::scene::parent_world_matrix(world, entity).inverse()
                * Mat4::from_rotation_translation(rotation, position);
            let Some(mut transform) = world.get_mut::<LocalTransform>(entity) else {
                tracing::warn!(
                    "Camera.Place: the active camera (entity {}) has no Transform to move",
                    entity.index
                );
                return false;
            };
            let (_, rotation, translation) = local.to_scale_rotation_translation();
            transform.translation = translation;
            transform.rotation = rotation;
            return true;
        }
        let Some(mut camera) = world.get_resource_mut::<crate::scene::Camera>() else {
            return false;
        };
        camera.place(position, rotation);
        true
    })
}

/// A lens from C#, or `None` with the reason logged against `call`.
fn checked_lens(value: *const CCamera, call: &str) -> Option<CameraComponent> {
    if value.is_null() {
        return None;
    }
    // SAFETY: `value` is a valid `CCamera` supplied by C#.
    let value = unsafe { *value };
    let camera = CameraComponent {
        fov_degrees: value.fov_degrees,
        near: value.near,
        far: value.far,
        priority: value.priority,
    };
    match camera.check() {
        Ok(()) => Some(camera),
        Err(error) => {
            tracing::warn!("{call}: {error}");
            None
        }
    }
}

extern "C" fn add_camera(entity: CEntity, value: *const CCamera) -> bool {
    let Some(camera) = checked_lens(value, "World.AddCamera") else {
        return false;
    };
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
                .push(Command::AddCamera { entity, camera })
        });
        true
    })
}

extern "C" fn get_camera_lens(entity: CEntity, out: *mut CCamera) -> bool {
    if out.is_null() {
        return false;
    }
    orrin_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        let Some(camera) = world.get::<CameraComponent>(entity) else {
            return false;
        };
        // SAFETY: `out` is a valid, writable `CCamera` supplied by C#.
        unsafe {
            *out = CCamera {
                fov_degrees: camera.fov_degrees,
                near: camera.near,
                far: camera.far,
                priority: camera.priority,
            };
        }
        true
    })
}

extern "C" fn set_camera_lens(entity: CEntity, value: *const CCamera) -> bool {
    let Some(lens) = checked_lens(value, "World.SetCamera") else {
        return false;
    };
    orrin_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        let Some(mut camera) = world.get_mut::<CameraComponent>(entity) else {
            return false;
        };
        *camera = lens;
        true
    })
}

extern "C" fn get_active_camera() -> CEntity {
    orrin_script::with_world(CEntity::NULL, |world| {
        crate::scene::active_camera(world).map_or(CEntity::NULL, |entity| CEntity {
            index: entity.index,
            generation: entity.generation,
        })
    })
}

extern "C" fn set_active_camera(entity: CEntity) -> bool {
    orrin_script::with_world(false, |world| {
        let selected = (entity.index != 0).then_some(Entity {
            index: entity.index,
            generation: entity.generation,
        });
        if let Some(entity) = selected {
            let queued = COMMANDS.with(|commands| {
                commands.borrow().iter().any(
                    |command| matches!(command, Command::AddCamera { entity: e, .. } if *e == entity),
                )
            });
            if !queued && !world.has::<CameraComponent>(entity) {
                tracing::warn!(
                    "Camera.Active: entity {} is not a camera; give it one with \
                     World.AddCamera first",
                    entity.index
                );
                return false;
            }
        }
        if world.get_resource::<ActiveCamera>().is_none() {
            world.insert_resource(ActiveCamera::default());
        }
        world.resource_mut::<ActiveCamera>().select(selected);
        true
    })
}

// The in-game UI's calls record into the `UiCanvas` resource the frame's
// painter reads; see `scene::ui`. With no canvas (a tool, a test) they do
// nothing, as a debug line does with no `DebugLines`.

fn with_canvas(draw: impl FnOnce(&mut UiCanvas)) {
    orrin_script::with_world((), |world| {
        if let Some(mut canvas) = world.get_resource_mut::<UiCanvas>() {
            draw(&mut canvas);
        }
    });
}

#[allow(clippy::too_many_arguments)]
extern "C" fn ui_panel(x: f32, y: f32, width: f32, height: f32, r: f32, g: f32, b: f32, a: f32) {
    with_canvas(|canvas| canvas.panel(Rect::new(x, y, width, height), color_bytes([r, g, b, a])));
}

#[allow(clippy::too_many_arguments)]
extern "C" fn ui_text(
    text: *const u8,
    len: u32,
    font: *const u8,
    size: f32,
    anchor: u32,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
) {
    if text.is_null() || len == 0 {
        return;
    }
    // SAFETY: C# passes `len` bytes it keeps pinned for the call.
    let bytes = unsafe { std::slice::from_raw_parts(text, len as usize) };
    let text = String::from_utf8_lossy(bytes);
    // SAFETY: a non-null font is the 16 bytes of an `AssetRef` C# holds.
    let font = (!font.is_null()).then(|| AssetId::from_bytes(unsafe { *font.cast::<[u8; 16]>() }));
    let style = TextStyle {
        font,
        size,
        anchor: TextAnchor::from_u32(anchor).unwrap_or_default(),
        color: color_bytes([r, g, b, a]),
        wrap: true,
    };
    with_canvas(|canvas| canvas.text(Rect::new(x, y, width, height), style, &text));
}

#[allow(clippy::too_many_arguments)]
extern "C" fn ui_image(
    texture: *const u8,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
) {
    if texture.is_null() {
        return;
    }
    // SAFETY: `texture` is the 16 bytes of an `AssetRef` C# holds.
    let texture = AssetId::from_bytes(unsafe { *texture.cast::<[u8; 16]>() });
    with_canvas(|canvas| {
        canvas.image(
            Rect::new(x, y, width, height),
            texture,
            color_bytes([r, g, b, a]),
        );
    });
}

extern "C" fn ui_screen_size(width: *mut f32, height: *mut f32) {
    let screen = orrin_script::with_world([0.0; 2], |world| {
        world
            .get_resource::<UiCanvas>()
            .map_or([0.0; 2], |canvas| canvas.screen())
    });
    for (out, side) in [width, height].into_iter().zip(screen) {
        if !out.is_null() {
            // SAFETY: C# passes valid, writable f32 pointers.
            unsafe { *out = side };
        }
    }
}

// Engine components as blittable structs, through one generic pair; see
// `OrrinApi::get_component`. The `kind`s are C# `Orrin.ComponentKind`, the
// numbering `has_component` matches on.

const KIND_CANVAS: u32 = 4;
const KIND_RECT_TRANSFORM: u32 = 5;
const KIND_IMAGE: u32 = 6;
const KIND_TEXT: u32 = 7;
const KIND_BUTTON: u32 = 8;

fn entity_of(entity: CEntity) -> Entity {
    Entity {
        index: entity.index,
        generation: entity.generation,
    }
}

/// The C# name of component `kind`, for a message.
fn kind_name(kind: u32) -> &'static str {
    match kind {
        KIND_CANVAS => "Canvas",
        KIND_RECT_TRANSFORM => "RectTransform",
        KIND_IMAGE => "Image",
        KIND_TEXT => "Text",
        KIND_BUTTON => "Button",
        _ => "an unknown component",
    }
}

/// Whether C# built `T` at the size the engine did, refusing by name if not.
fn sized<T>(call: &str, kind: u32, size: u32) -> bool {
    let expected = std::mem::size_of::<T>() as u32;
    if size != expected {
        let name = kind_name(kind);
        tracing::warn!(
            "World.{call}<{name}>: C# passed {size} bytes and the engine's {name} is \
             {expected}; rebuild scripting/Orrin so the bindings match the engine"
        );
    }
    size == expected
}

fn asset_bytes(id: Option<AssetId>) -> [u8; 16] {
    id.unwrap_or(AssetId::NIL).to_bytes()
}

fn asset_of(bytes: [u8; 16]) -> Option<AssetId> {
    Some(AssetId::from_bytes(bytes)).filter(|id| !id.is_nil())
}

fn canvas_to_c(canvas: &crate::ui::Canvas) -> orrin_script::CCanvas {
    orrin_script::CCanvas {
        scale_mode: canvas.scale_mode as u32,
        reference_resolution: canvas.reference_resolution.to_array(),
        match_width_or_height: canvas.match_width_or_height,
        sort_order: canvas.sort_order,
    }
}

fn canvas_from_c(c: orrin_script::CCanvas) -> Option<crate::ui::Canvas> {
    use crate::ui::ScaleMode;
    let scale_mode = match c.scale_mode {
        0 => ScaleMode::ConstantPixel,
        1 => ScaleMode::ScaleWithScreen,
        other => {
            tracing::warn!("Canvas: {other} is not a CanvasScaleMode");
            return None;
        }
    };
    Some(crate::ui::Canvas {
        scale_mode,
        reference_resolution: glam::Vec2::from_array(c.reference_resolution),
        match_width_or_height: c.match_width_or_height,
        sort_order: c.sort_order,
    })
}

fn rect_to_c(rect: &crate::ui::RectTransform) -> orrin_script::CRectTransform {
    orrin_script::CRectTransform {
        anchor_min: rect.anchor_min.to_array(),
        anchor_max: rect.anchor_max.to_array(),
        pivot: rect.pivot.to_array(),
        anchored_position: rect.anchored_position.to_array(),
        size_delta: rect.size_delta.to_array(),
        scale: rect.scale.to_array(),
    }
}

fn rect_from_c(c: orrin_script::CRectTransform) -> crate::ui::RectTransform {
    use glam::Vec2;
    crate::ui::RectTransform {
        anchor_min: Vec2::from_array(c.anchor_min),
        anchor_max: Vec2::from_array(c.anchor_max),
        pivot: Vec2::from_array(c.pivot),
        anchored_position: Vec2::from_array(c.anchored_position),
        size_delta: Vec2::from_array(c.size_delta),
        scale: Vec2::from_array(c.scale),
    }
}

fn image_to_c(image: &crate::ui::Image) -> orrin_script::CImage {
    orrin_script::CImage {
        texture: asset_bytes(image.texture),
        color: image.color,
        slice: image.slice,
        raycast_target: image.raycast_target.into(),
    }
}

fn image_from_c(c: orrin_script::CImage) -> crate::ui::Image {
    crate::ui::Image {
        texture: asset_of(c.texture),
        color: c.color,
        slice: c.slice,
        raycast_target: c.raycast_target != 0,
    }
}

fn text_to_c(text: &crate::ui::Text) -> orrin_script::CText {
    orrin_script::CText {
        font: asset_bytes(text.font),
        color: text.color,
        size: text.size,
        anchor: text.anchor as u32,
        wrap: text.wrap.into(),
        raycast_target: text.raycast_target.into(),
    }
}

/// Everything but the string, which the struct does not carry.
fn text_from_c(c: orrin_script::CText, text: String) -> Option<crate::ui::Text> {
    let Some(anchor) = TextAnchor::from_u32(c.anchor) else {
        tracing::warn!("Text: {} is not a TextAnchor", c.anchor);
        return None;
    };
    Some(crate::ui::Text {
        text,
        font: asset_of(c.font),
        size: c.size,
        color: c.color,
        anchor,
        wrap: c.wrap != 0,
        raycast_target: c.raycast_target != 0,
    })
}

fn button_to_c(button: &crate::ui::Button) -> orrin_script::CButton {
    orrin_script::CButton {
        normal: button.normal,
        hover: button.hover,
        pressed: button.pressed,
        disabled: button.disabled,
        interactable: button.interactable.into(),
    }
}

fn button_from_c(c: orrin_script::CButton) -> crate::ui::Button {
    crate::ui::Button {
        normal: c.normal,
        hover: c.hover,
        pressed: c.pressed,
        disabled: c.disabled,
        interactable: c.interactable != 0,
    }
}

/// `value` as C# sent it, if it is the size of `T`.
fn read_c<T: Copy>(call: &str, kind: u32, value: *const u8, size: u32) -> Option<T> {
    if value.is_null() || !sized::<T>(call, kind, size) {
        return None;
    }
    // SAFETY: C# passes a pointer to `size` readable bytes, just checked to be
    // a whole `T`; unaligned, since nothing promises the buffer's alignment.
    Some(unsafe { value.cast::<T>().read_unaligned() })
}

/// `value` decoded as component `kind`, keeping `text` for a `Text`.
fn decode(
    call: &str,
    kind: u32,
    value: *const u8,
    size: u32,
    text: impl FnOnce() -> String,
) -> Option<UiComponent> {
    use orrin_script::{CButton, CCanvas, CImage, CRectTransform, CText};
    match kind {
        KIND_CANVAS => {
            canvas_from_c(read_c::<CCanvas>(call, kind, value, size)?).map(UiComponent::Canvas)
        }
        KIND_RECT_TRANSFORM => Some(UiComponent::RectTransform(rect_from_c(read_c::<
            CRectTransform,
        >(
            call, kind, value, size,
        )?))),
        KIND_IMAGE => Some(UiComponent::Image(image_from_c(read_c::<CImage>(
            call, kind, value, size,
        )?))),
        KIND_TEXT => {
            text_from_c(read_c::<CText>(call, kind, value, size)?, text()).map(UiComponent::Text)
        }
        KIND_BUTTON => Some(UiComponent::Button(button_from_c(read_c::<CButton>(
            call, kind, value, size,
        )?))),
        _ => {
            tracing::warn!("World.{call}: component kind {kind} has no blittable form");
            None
        }
    }
}

/// The add of `kind` to `entity` this tick has queued, if there is one.
fn with_queued<R>(
    entity: Entity,
    kind: u32,
    change: impl FnOnce(&mut UiComponent) -> R,
) -> Option<R> {
    COMMANDS.with(|commands| {
        commands
            .borrow_mut()
            .iter_mut()
            .rev()
            .find_map(|command| match command {
                Command::AddUi {
                    entity: queued,
                    component,
                } if *queued == entity && component.kind() == kind => Some(component),
                _ => None,
            })
            .map(change)
    })
}

extern "C" fn get_component(entity: CEntity, kind: u32, out: *mut u8, size: u32) -> bool {
    fn write<T: Copy>(out: *mut u8, size: u32, kind: u32, value: T) -> bool {
        if !sized::<T>("GetComponent", kind, size) {
            return false;
        }
        // SAFETY: C# passes `size` writable bytes, just checked to be a whole
        // `T`; unaligned, as in `read_c`.
        unsafe { out.cast::<T>().write_unaligned(value) };
        true
    }
    if out.is_null() {
        return false;
    }
    orrin_script::with_world(false, |world| {
        let entity = entity_of(entity);
        let value = match kind {
            KIND_CANVAS => world
                .get::<crate::ui::Canvas>(entity)
                .map(|c| canvas_to_c(&c))
                .map(|c| write(out, size, kind, c)),
            KIND_RECT_TRANSFORM => world
                .get::<crate::ui::RectTransform>(entity)
                .map(|c| write(out, size, kind, rect_to_c(&c))),
            KIND_IMAGE => world
                .get::<crate::ui::Image>(entity)
                .map(|c| write(out, size, kind, image_to_c(&c))),
            KIND_TEXT => world
                .get::<crate::ui::Text>(entity)
                .map(|c| write(out, size, kind, text_to_c(&c))),
            KIND_BUTTON => world
                .get::<crate::ui::Button>(entity)
                .map(|c| write(out, size, kind, button_to_c(&c))),
            _ => {
                tracing::warn!("World.GetComponent: component kind {kind} has no blittable form");
                None
            }
        };
        value.unwrap_or(false)
    })
}

extern "C" fn set_component(entity: CEntity, kind: u32, value: *const u8, size: u32) -> bool {
    orrin_script::with_world(false, |world| {
        let entity = entity_of(entity);
        let current_text = || {
            world
                .get::<crate::ui::Text>(entity)
                .map(|text| text.text.clone())
                .or_else(|| {
                    with_queued(entity, KIND_TEXT, |queued| match queued {
                        UiComponent::Text(text) => text.text.clone(),
                        _ => String::new(),
                    })
                })
                .unwrap_or_default()
        };
        let Some(component) = decode("SetComponent", kind, value, size, current_text) else {
            return false;
        };
        let in_place = match &component {
            UiComponent::Canvas(c) => world
                .get_mut::<crate::ui::Canvas>(entity)
                .map(|mut slot| *slot = *c),
            UiComponent::RectTransform(c) => world
                .get_mut::<crate::ui::RectTransform>(entity)
                .map(|mut slot| *slot = *c),
            UiComponent::Image(c) => world
                .get_mut::<crate::ui::Image>(entity)
                .map(|mut slot| *slot = *c),
            UiComponent::Text(c) => world
                .get_mut::<crate::ui::Text>(entity)
                .map(|mut slot| *slot = c.clone()),
            UiComponent::Button(c) => world
                .get_mut::<crate::ui::Button>(entity)
                .map(|mut slot| *slot = *c),
        };
        in_place.is_some() || with_queued(entity, kind, |queued| *queued = component).is_some()
    })
}

extern "C" fn add_component(entity: CEntity, kind: u32, value: *const u8, size: u32) -> bool {
    let Some(component) = decode("AddComponent", kind, value, size, String::new) else {
        return false;
    };
    orrin_script::with_world(false, |world| {
        let entity = entity_of(entity);
        if !world.is_alive(entity) {
            return false;
        }
        COMMANDS.with(|commands| {
            commands
                .borrow_mut()
                .push(Command::AddUi { entity, component })
        });
        true
    })
}

// The `get_tag` protocol; see `OrrinApi::get_component_text`.
extern "C" fn get_component_text(entity: CEntity, kind: u32, out: *mut u8, capacity: i32) -> i32 {
    if kind != KIND_TEXT || (out.is_null() && capacity > 0) {
        return -1;
    }
    orrin_script::with_world(-1, |world| {
        let entity = entity_of(entity);
        let text = world
            .get::<crate::ui::Text>(entity)
            .map(|text| text.text.clone())
            .or_else(|| {
                with_queued(entity, KIND_TEXT, |queued| match queued {
                    UiComponent::Text(text) => text.text.clone(),
                    _ => String::new(),
                })
            });
        let Some(text) = text else {
            return -1;
        };
        let bytes = text.as_bytes();
        let n = bytes.len().min(capacity.max(0) as usize);
        if n > 0 {
            // SAFETY: C# passes `capacity` writable bytes; the source is our own
            // string, so the ranges cannot overlap.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, n) };
        }
        bytes.len() as i32
    })
}

extern "C" fn set_component_text(entity: CEntity, kind: u32, text: *const u8, len: u32) -> bool {
    if kind != KIND_TEXT || (text.is_null() && len > 0) {
        return false;
    }
    let text = if len == 0 {
        String::new()
    } else {
        // SAFETY: C# passes `len` bytes it keeps pinned for the call.
        String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(text, len as usize) })
            .into_owned()
    };
    orrin_script::with_world(false, |world| {
        let entity = entity_of(entity);
        if let Some(mut component) = world.get_mut::<crate::ui::Text>(entity) {
            component.text = text;
            return true;
        }
        with_queued(entity, KIND_TEXT, |queued| {
            if let UiComponent::Text(queued) = queued {
                queued.text = text;
            }
        })
        .is_some()
    })
}

extern "C" fn ui_pointer_over() -> bool {
    orrin_script::with_world(false, |world| {
        world
            .get_resource::<crate::ui::events::UiPointer>()
            .is_some_and(|pointer| pointer.is_over_ui())
    })
}

/// Apply a queued scene change. Every other command is `apply_commands`' own.
fn scene_command(world: &mut World, registry: &Registry, command: Command) {
    use crate::scene::loaded;

    let (call, issues) = match command {
        Command::LoadScene(scene) => (
            "SceneManager.Load",
            loaded::read(world, scene).map(|(document, name)| {
                loaded::load(world, registry, &document, name, Some(scene))
            }),
        ),
        Command::LoadSceneAdditive { handle, scene } => (
            "SceneManager.LoadAdditive",
            loaded::read(world, scene).map(|(document, name)| {
                loaded::load_additive(world, registry, handle, &document, name, Some(scene), false)
            }),
        ),
        Command::UnloadScene(handle) => (
            "SceneManager.Unload",
            loaded::unload(world, handle).map(|()| Vec::new()),
        ),
        _ => unreachable!("not a scene command"),
    };
    match issues {
        Ok(issues) => {
            for issue in issues {
                tracing::warn!("{call}: {issue}");
            }
        }
        Err(error) => tracing::warn!("{call}: {error}"),
    }
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

// A script's log line becomes an event on the engine's stream (`logging.rs`),
// so it reaches stderr, the console and an export's log file like any other.
// That needs no world, which is what lets the lines C# writes outside a
// dispatch window — OnDestroy during a despawn, everything a hot reload
// reports — reach the console too.
//
// `level` counts up from trace, matching `LogLevel` and C# `Orrin.LogLevel`.
// One this build does not know is logged as an error rather than dropped: the
// caller meant to be heard. `source` names the script and may be null.
extern "C" fn log_event(level: u32, source: *const c_char, line: u32, message: *const c_char) {
    if message.is_null() {
        return;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    let script = if source.is_null() {
        std::borrow::Cow::Borrowed("")
    } else {
        // SAFETY: as above.
        unsafe { CStr::from_ptr(source) }.to_string_lossy()
    };
    let script: &str = &script;
    match level {
        0 => tracing::trace!(script, line, "{text}"),
        1 => tracing::debug!(script, line, "{text}"),
        2 => tracing::info!(script, line, "{text}"),
        3 => tracing::warn!(script, line, "{text}"),
        _ => tracing::error!(script, line, "{text}"),
    }
}

// The unattributed sinks the table has carried since before `log_event`. The
// bindings no longer call them; they stay because the table only grows.
extern "C" fn log_info(message: *const c_char) {
    log_event(LogLevel::Info as u32, std::ptr::null(), 0, message);
}

extern "C" fn log_warn(message: *const c_char) {
    log_event(LogLevel::Warning as u32, std::ptr::null(), 0, message);
}

extern "C" fn log_error(message: *const c_char) {
    log_event(LogLevel::Error as u32, std::ptr::null(), 0, message);
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

/// Answers the component registry's questions about a C# Behaviour by crossing
/// into `Orrin.PropertyBag`.
///
/// Zero-sized: the managed entry points are process globals (see
/// `orrin_script::read_bag` for why), so one of these serves every Behaviour
/// type and carries nothing but the impl.
///
/// Every method takes the handle out of the `ScriptComponent` and drops the
/// world borrow before calling managed code. The bag entry points do not
/// re-enter the engine, but the rule that no world borrow is held across the
/// boundary is worth keeping unconditionally — it is the one that stops being
/// true silently.
struct PropertyBags;

impl PropertyBags {
    /// The GCHandle for `entity`'s behaviour, if it is the type this vtable
    /// entry belongs to. An entity carries at most one `ScriptComponent`, so
    /// two script components on one entity is simply not a state that exists.
    fn handle(world: &World, entity: Entity, type_name: &str) -> Option<u64> {
        world
            .get::<ScriptComponent>(entity)
            .filter(|script| script.type_name == type_name)
            .map(|script| script.handle)
    }
}

impl ScriptBridge for PropertyBags {
    fn has(&self, world: &World, entity: Entity, type_name: &str) -> bool {
        Self::handle(world, entity, type_name).is_some()
    }

    fn read(&self, world: &World, entity: Entity, type_name: &str) -> Option<Value> {
        let bytes = orrin_script::read_bag(Self::handle(world, entity, type_name)?)?;
        match orrin_registry::wire::decode(&bytes) {
            Ok(value) => Some(value),
            Err(error) => {
                tracing::error!("{type_name}'s property bag did not decode: {error}");
                None
            }
        }
    }

    fn write(
        &self,
        world: &mut World,
        entity: Entity,
        type_name: &str,
        value: &Value,
    ) -> Result<(), ValueError> {
        let Some(handle) = Self::handle(world, entity, type_name) else {
            return Err(ValueError::invalid(
                "a live script component",
                type_name.to_owned(),
            ));
        };
        let mut bytes = Vec::new();
        orrin_registry::wire::encode(value, &mut bytes);
        if orrin_script::write_bag(handle, &bytes) {
            Ok(())
        } else {
            // The managed side has already logged what went wrong, naming the
            // field; this only says the write did not land.
            Err(ValueError::invalid(
                "fields the behaviour accepted",
                type_name.to_owned(),
            ))
        }
    }

    fn remove(&self, world: &mut World, entity: Entity, type_name: &str) {
        if Self::handle(world, entity, type_name).is_some() {
            // Dropping the component is what tears the managed object down; see
            // `ScriptComponent`'s `Drop`.
            let _ = world.remove::<ScriptComponent>(entity);
        }
    }

    /// An empty struct rather than a panic when the managed side cannot answer:
    /// a default is asked for while building a prefab or an inspector row, and
    /// a game assembly that failed to load should leave those blank rather than
    /// take the editor with it.
    fn default(&self, type_name: &str) -> Value {
        let empty = Value::Struct(Vec::new());
        let Ok(name) = CString::new(type_name) else {
            return empty;
        };
        let Some(bytes) = orrin_script::default_bag(&name) else {
            return empty;
        };
        match orrin_registry::wire::decode(&bytes) {
            Ok(value) => value,
            Err(error) => {
                tracing::error!("{type_name}'s defaults did not decode: {error}");
                empty
            }
        }
    }
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
        action_id,
        action_held,
        action_pressed,
        action_released,
        axis_value,
        mouse_button_pressed,
        mouse_button_released,
        log_event,
        find_asset,
        instantiate_prefab,
        asset_state,
        load_scene,
        load_scene_additive,
        unload_scene,
        active_scene,
        get_camera,
        set_camera,
        add_camera,
        get_camera_lens,
        set_camera_lens,
        get_active_camera,
        set_active_camera,
        ui_panel,
        ui_text,
        ui_image,
        ui_screen_size,
        get_component,
        set_component,
        add_component,
        get_component_text,
        set_component_text,
        ui_pointer_over,
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
    /// This tick's pointer events, swapped out of the world for the dispatch
    /// and back, cleared, so neither side reallocates.
    pointer_events: RefCell<crate::ui::events::UiEvents>,
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
                tracing::warn!(
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
            pointer_events: RefCell::new(crate::ui::events::UiEvents::default()),
        };
        match scripting.stage() {
            Ok(()) => {}
            Err(status) => {
                tracing::warn!(
                    "scripting disabled: could not load {} ({status})",
                    game_dll.display()
                );
                return None;
            }
        }
        let status = scripting.host.commit_game();
        if !status.succeeded() {
            tracing::warn!("scripting disabled: {status}");
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

    /// Create the Behaviours a load left waiting (see [`PendingScripts`]) and
    /// give each its saved fields.
    ///
    /// Run outside any dispatch window, since it creates managed objects, and
    /// before the tick, so they get their OnEnable and OnStart on it. One that
    /// cannot be created is kept as data, never dropped: a type renamed since
    /// the save, a game assembly that is not loaded, or a second Behaviour on an
    /// entity that can carry only one.
    pub fn attach_pending(&self, world: &mut World, registry: &Registry) {
        let mut waiting = Vec::new();
        world
            .query::<&PendingScripts>()
            .for_each(|entity, _| waiting.push(entity));
        for entity in waiting {
            let Some(PendingScripts(scripts)) = world.remove::<PendingScripts>(entity) else {
                continue;
            };
            for (id, value) in scripts {
                let binding = registry
                    .get(&id)
                    .and_then(|vtable| vtable.script.as_ref().map(|b| (vtable, b)));
                let Some((vtable, binding)) = binding else {
                    crate::scene::persist::keep_unapplied(world, entity, (id, value));
                    continue;
                };
                if world.has::<ScriptComponent>(entity) {
                    tracing::warn!(
                        "entity {} already has a Behaviour, and carries one at most; `{id}` was \
                         kept but not attached",
                        entity.index()
                    );
                    crate::scene::persist::keep_unapplied(world, entity, (id, value));
                    continue;
                }
                if self.attach_with(world, entity, &binding.type_name, false, true) == 0 {
                    crate::scene::persist::keep_unapplied(world, entity, (id, value));
                    continue;
                }
                if let Err(error) = vtable.write(world, entity, &value) {
                    tracing::warn!("`{id}` on entity {}: {error}", entity.index());
                }
            }
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
            tracing::error!(
                "no Behaviour `{type_name}` in the loaded game assembly \
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

    /// Describe every `[Component]`-attributed Behaviour in the live game
    /// assembly to `registry`, so its fields save, load and inspect exactly as a
    /// Rust component's do.
    ///
    /// The game assembly's half of `register_components`, and the counterpart to
    /// [`Registry::clear_game`]: call `clear_game` before the swap and this
    /// after it, or the previous build's entries keep a retired type reachable
    /// and its load context never unloads.
    ///
    /// Returns how many were registered. A managed failure is already logged by
    /// the C# side and leaves the registry as it was — the engine runs on with
    /// script components uninspectable rather than not at all.
    pub fn register_components(&self, registry: &mut Registry) -> usize {
        let Some(bytes) = orrin_script::component_bags() else {
            return 0;
        };
        let listing = match orrin_registry::wire::decode(&bytes) {
            Ok(value) => value,
            Err(error) => {
                tracing::error!("the component listing did not decode: {error}");
                return 0;
            }
        };
        let Value::List(entries) = listing else {
            tracing::error!("the component listing was not a list");
            return 0;
        };

        let bridge: Arc<dyn ScriptBridge> = Arc::new(PropertyBags);
        let mut registered = 0;
        for entry in entries {
            let (
                Some(Value::String(id)),
                Some(Value::String(type_name)),
                Some(Value::String(name)),
            ) = (
                entry.field("id").cloned(),
                entry.field("type").cloned(),
                entry.field("name").cloned(),
            )
            else {
                tracing::error!("a component listing entry was missing id, type or name");
                continue;
            };
            // The `orrin.` namespace is the engine's, and a game claiming one of
            // its ids would shadow a component in every scene it saved. Refused
            // by name rather than by the duplicate-id panic below, which would
            // take the editor down over a typo in someone's game code.
            if id.starts_with("orrin.") {
                tracing::error!(
                    "`{type_name}` claims the reserved id `{id}`; \
                     the `orrin.` prefix belongs to the engine's own components"
                );
                continue;
            }
            let component_id = ComponentId::owned(id);
            if registry.get(&component_id).is_some() {
                tracing::error!(
                    "`{type_name}` claims `{component_id}`, which is already registered"
                );
                continue;
            }
            registry.register_script(component_id, name, type_name, Arc::clone(&bridge));
            registered += 1;
        }
        registered
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
    /// then its pointer events, then OnUpdate.
    pub fn tick(&self, world: &mut World, registry: &Registry, delta_time: f32) {
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
        let mut pointer_events = self.pointer_events.take();
        if let Some(mut queued) = world.get_resource_mut::<crate::ui::events::UiEvents>() {
            std::mem::swap(&mut *queued, &mut pointer_events);
        }

        let mut routing = self.routing.take();
        routing.clear();
        if !events.is_empty() || !pointer_events.is_empty() {
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

            // Offered up each event's chain to the first behaviour that
            // handles it. One that is inactive, faulted or has no handler
            // passes it on, as a disabled component does in uGUI.
            crate::ui::events::deliver(&pointer_events, |entity, event| {
                let Some(&index) = routing.get(&entity) else {
                    return false;
                };
                let script = &mut pending[index];
                if !script.active || script.faulted {
                    return false;
                }
                let payload = orrin_script::CPointerEvent {
                    position: event.position,
                    press_position: event.press_position,
                    target: CEntity {
                        index: event.target.index,
                        generation: event.target.generation,
                    },
                    button: 0,
                };
                match self
                    .host
                    .pointer(script.handle, event.kind as u32, &payload)
                {
                    orrin_script::Delivery::Handled => true,
                    orrin_script::Delivery::Unhandled => false,
                    orrin_script::Delivery::Faulted => {
                        script.faulted = true;
                        true
                    }
                }
            });

            for script in &mut pending {
                if script.active && !script.faulted && self.host.update(script.handle, delta_time) {
                    script.faulted = true;
                }
            }
        });
        pointer_events.clear();

        // Structural changes the scripts queued land now, after every script
        // has run — so this frame's extraction already sees new renderables.
        self.apply_commands(world, registry);

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
        self.pointer_events.replace(pointer_events);
    }

    /// Apply the structural changes scripts queued during a dispatch. Runs
    /// with no other world borrows held, so a despawn can drop a
    /// `ScriptComponent` (and free its GCHandle) safely.
    fn apply_commands(&self, world: &mut World, registry: &Registry) {
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
                Command::AddCamera { entity, camera } => {
                    world.insert(entity, camera);
                }
                Command::AddUi { entity, component } => component.insert(world, entity),
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
                Command::InstantiatePrefab { root, prefab } => {
                    build_prefab(world, registry, root, prefab);
                }
                command @ (Command::LoadScene(_)
                | Command::LoadSceneAdditive { .. }
                | Command::UnloadScene(_)) => scene_command(world, registry, command),
                Command::AttachScript { entity, type_name } => {
                    // Creates the managed instance now; the enable/start pair
                    // dispatches on the next tick, outside any window.
                    self.attach(world, entity, &type_name);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::{LogConfig, Stream};
    use crate::scene::LogBuffer;
    use crate::scene::prefab::{Prefab, PrefabInstance};
    use orrin_registry::AssetId;

    /// What `emit` put on the stream, as `(level, line)`.
    fn logged(emit: impl FnOnce()) -> Vec<(LogLevel, String)> {
        let (stream, console, _) = Stream::new(LogConfig {
            stderr: false,
            ..LogConfig::editor()
        });
        tracing::subscriber::with_default(stream, emit);
        let mut log = LogBuffer::default();
        console.drain_into(&mut log, 0);
        log.iter()
            .map(|entry| (entry.level, entry.text()))
            .collect()
    }

    #[test]
    fn a_script_log_keeps_its_level_and_names_its_script() {
        let lines = logged(|| {
            for (level, text) in [c"t", c"d", c"i", c"w", c"e"].into_iter().enumerate() {
                log_event(level as u32, c"Spinner".as_ptr(), 14, text.as_ptr());
            }
        });
        // An export build compiles the two quietest levels out at the call
        // site, so there they never reach the stream to be filtered.
        let floor = if cfg!(feature = "export") {
            LogLevel::Info
        } else {
            LogLevel::Trace
        };
        assert_eq!(
            lines,
            LogLevel::ALL
                .into_iter()
                .zip(["t", "d", "i", "w", "e"])
                .filter(|(level, _)| *level >= floor)
                .map(|(level, text)| (level, format!("Spinner:14: {text}")))
                .collect::<Vec<_>>()
        );
    }

    /// The pre-`log_event` sinks pass no source, and a level from a newer
    /// bindings assembly is one this build cannot name.
    #[test]
    fn an_unattributed_or_unknown_level_log_is_still_heard() {
        let lines = logged(|| {
            log_warn(c"legacy".as_ptr());
            log_event(99, std::ptr::null(), 0, c"from the future".as_ptr());
            log_event(2, std::ptr::null(), 0, std::ptr::null());
        });
        assert_eq!(
            lines,
            [
                (
                    LogLevel::Warning,
                    "orrin_core::scripting: legacy".to_owned()
                ),
                (
                    LogLevel::Error,
                    "orrin_core::scripting: from the future".to_owned()
                ),
            ]
        );
    }

    /// A world whose prefab index holds one crate, as the FFI sees it.
    fn prefab_world() -> (World, Registry, AssetId) {
        let id = AssetId::new();
        let mut prefabs = Prefabs::default();
        prefabs.insert(
            Prefab::parse(
                id,
                "props/crate.fprefab",
                "orrin-scene 1\n\nentity 00000000-0000-4000-8000-000000000001\n  orrin.name = \"Crate\"\n",
            )
            .unwrap(),
        );
        let mut world = World::new();
        world.insert_resource(prefabs);
        let mut registry = Registry::new();
        crate::scene::register_components(&mut registry);
        (world, registry, id)
    }

    #[test]
    fn a_script_finds_a_prefab_by_its_path() {
        let (mut world, _, id) = prefab_world();
        let mut out = [0u8; 16];
        let kind = orrin_script::with_active_world(&mut world, || {
            find_asset(c"props/crate.fprefab".as_ptr(), out.as_mut_ptr())
        });
        assert_eq!(kind, orrin_script::ASSET_PREFAB);
        assert_eq!(AssetId::from_bytes(out), id);

        let kind = orrin_script::with_active_world(&mut world, || {
            find_asset(c"props/barrel.fprefab".as_ptr(), out.as_mut_ptr())
        });
        assert_eq!(kind, orrin_script::ASSET_NONE);
    }

    /// The root comes back at once, as `spawn_renderable`'s entity does; the
    /// tree under it is built when the dispatch's commands drain.
    #[test]
    fn a_script_gets_the_root_at_once_and_the_tree_after_the_dispatch() {
        let (mut world, registry, id) = prefab_world();
        let bytes = id.to_bytes();
        let root =
            orrin_script::with_active_world(&mut world, || instantiate_prefab(bytes.as_ptr()));
        let root = Entity {
            index: root.index,
            generation: root.generation,
        };
        assert!(world.is_alive(root));
        assert!(world.get::<Name>(root).is_none());

        let commands: Vec<Command> = COMMANDS.with(|c| c.borrow_mut().drain(..).collect());
        assert_eq!(commands.len(), 1);
        let Command::InstantiatePrefab {
            root: queued,
            prefab,
        } = commands[0]
        else {
            panic!("expected the instantiation to be queued");
        };
        assert_eq!((queued, prefab), (root, id));

        build_prefab(&mut world, &registry, root, prefab);
        assert_eq!(world.get::<Name>(root).unwrap().0, "Crate");
        assert_eq!(world.get::<PrefabInstance>(root).unwrap().prefab, id);
    }

    #[test]
    fn an_unknown_prefab_gives_a_script_the_null_entity() {
        let (mut world, _, _) = prefab_world();
        let bytes = AssetId::new().to_bytes();
        let root =
            orrin_script::with_active_world(&mut world, || instantiate_prefab(bytes.as_ptr()));
        assert_eq!((root.index, root.generation), (0, 0));
        assert_eq!(world.entities().count(), 0);
        assert!(COMMANDS.with(|c| c.borrow().is_empty()));
    }

    /// A project with two scenes and the active one holding `Hub`, as a script
    /// running in it sees the world.
    fn scene_world(
        tag: &str,
    ) -> (
        World,
        Registry,
        AssetId,
        AssetId,
        crate::asset_db::tests::Dir,
    ) {
        let mut registry = Registry::new();
        crate::scene::register_components(&mut registry);
        let scene_text = |name: &str| {
            let mut source = World::new();
            let entity = source.spawn();
            source.insert(entity, Name::new(name));
            crate::scene::save(&mut source, &registry)
        };
        let dir = crate::asset_db::tests::Dir::new(tag);
        dir.write("levels/castle.orrin", scene_text("Gate").as_bytes());
        dir.write("levels/shop.orrin", scene_text("Shopkeeper").as_bytes());
        let (db, _) = crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let castle = db.find("levels/castle.orrin").unwrap().id;
        let shop = db.find("levels/shop.orrin").unwrap().id;
        let mut world = World::new();
        world.insert_resource(db);
        world.insert_resource(crate::scene::LoadedScenes::default());
        let hub = world.spawn();
        world.insert(hub, Name::new("Hub"));
        (world, registry, castle, shop, dir)
    }

    fn names(world: &World) -> Vec<String> {
        let mut names: Vec<String> = world
            .entities()
            .filter_map(|entity| world.get::<Name>(entity).map(|name| name.0.clone()))
            .collect();
        names.sort();
        names
    }

    fn drain(world: &mut World, registry: &Registry) {
        let commands: Vec<Command> = COMMANDS.with(|c| c.borrow_mut().drain(..).collect());
        for command in commands {
            scene_command(world, registry, command);
        }
    }

    /// A load despawns the scripts that asked for it, so it waits for the
    /// dispatch to close like every other structural change.
    #[test]
    fn a_script_loads_a_scene_after_the_dispatch() {
        let (mut world, registry, castle, _, _dir) = scene_world("script-scene-load");
        let bytes = castle.to_bytes();
        let queued = orrin_script::with_active_world(&mut world, || load_scene(bytes.as_ptr()));
        assert!(queued);
        assert_eq!(names(&world), ["Hub"]);

        drain(&mut world, &registry);
        assert_eq!(names(&world), ["Gate"]);
        let loaded = world.resource::<crate::scene::LoadedScenes>();
        assert_eq!(loaded.active().name, "castle");
        assert_eq!(loaded.active().asset, Some(castle));
    }

    /// The handle comes back from the call, so the same tick can hold on to it
    /// — and unload it — before the scene has even loaded.
    #[test]
    fn an_additive_load_hands_back_a_handle_the_same_tick_can_unload() {
        let (mut world, registry, _, shop, _dir) = scene_world("script-scene-additive");
        let bytes = shop.to_bytes();
        let handle =
            orrin_script::with_active_world(&mut world, || load_scene_additive(bytes.as_ptr()));
        assert_ne!(handle, 0);
        drain(&mut world, &registry);
        assert_eq!(names(&world), ["Hub", "Shopkeeper"]);

        let handle = orrin_script::with_active_world(&mut world, || {
            let again = load_scene_additive(bytes.as_ptr());
            assert!(unload_scene(again));
            again
        });
        assert_ne!(handle, 0);
        drain(&mut world, &registry);
        assert_eq!(names(&world), ["Hub", "Shopkeeper"]);
    }

    /// A reference to anything but a scene, or to nothing, fails at the call
    /// and queues nothing — the rule a bad prefab reference follows.
    #[test]
    fn a_reference_that_names_no_scene_is_refused_at_the_call() {
        let (mut world, _, _, _, dir) = scene_world("script-scene-refused");
        dir.write("rock.png", b"");
        world.resource_mut::<AssetDatabase>().rescan();
        let rock = world
            .resource::<AssetDatabase>()
            .find("rock.png")
            .unwrap()
            .id;
        for id in [rock, AssetId::new()] {
            let bytes = id.to_bytes();
            let (single, additive) = orrin_script::with_active_world(&mut world, || {
                (
                    load_scene(bytes.as_ptr()),
                    load_scene_additive(bytes.as_ptr()),
                )
            });
            assert!(!single);
            assert_eq!(additive, 0);
        }
        assert!(COMMANDS.with(|c| c.borrow().is_empty()));
    }

    /// There is always an active scene; a script replaces it with a load.
    #[test]
    fn a_script_cannot_unload_the_active_scene() {
        let (mut world, _, _, _, _dir) = scene_world("script-scene-active");
        let unloaded = orrin_script::with_active_world(&mut world, || {
            let active = active_scene();
            assert_ne!(active, 0);
            unload_scene(active)
        });
        assert!(!unloaded);
        assert!(!orrin_script::with_active_world(&mut world, || {
            unload_scene(9999)
        }));
        assert!(COMMANDS.with(|c| c.borrow().is_empty()));
    }

    /// A project's textures and meshes are found by path like its prefabs, and
    /// a script can ask whether one is drawable yet.
    #[test]
    fn a_script_finds_a_mesh_or_texture_by_path_and_reads_its_load_state() {
        let dir = crate::asset_db::tests::Dir::new("script-assets");
        dir.write("models/tree.glb", b"not a glb");
        let mut png = std::io::Cursor::new(Vec::new());
        image::RgbaImage::new(1, 1)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        dir.write("rocks/rock.png", png.get_ref());
        let (db, _) = crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let mut world = World::new();
        world.insert_resource(db);

        let find = |world: &mut World, path: &std::ffi::CStr| {
            let mut out = [0u8; 16];
            let kind = orrin_script::with_active_world(world, || {
                find_asset(path.as_ptr(), out.as_mut_ptr())
            });
            (kind, AssetId::from_bytes(out))
        };
        let state = |world: &mut World, id: AssetId| {
            let bytes = id.to_bytes();
            orrin_script::with_active_world(world, || asset_state(bytes.as_ptr()))
        };

        let (kind, tree) = find(&mut world, c"models/tree.glb");
        assert_eq!(kind, orrin_script::ASSET_MESH);
        let (kind, rock) = find(&mut world, c"rocks/rock.png");
        assert_eq!(kind, orrin_script::ASSET_TEXTURE);

        let db = world.resource::<crate::asset_db::AssetDatabase>();
        assert!(db.wait_idle(std::time::Duration::from_secs(20)));
        drop(db);
        world
            .resource_mut::<crate::asset_db::AssetDatabase>()
            .update(&world, &mut crate::gfx::HeadlessBackend::new());

        assert_eq!(state(&mut world, rock), orrin_script::LOAD_READY);
        assert_eq!(
            state(&mut world, tree),
            orrin_script::LOAD_FAILED,
            "not a glb"
        );
        assert_eq!(state(&mut world, AssetId::new()), orrin_script::LOAD_FAILED);
    }

    #[test]
    fn a_script_finds_a_font_by_path() {
        let dir = crate::asset_db::tests::Dir::new("script-font");
        dir.write("fonts/title.ttf", b"glyphs");
        let (db, _) = crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let mut world = World::new();
        world.insert_resource(db);
        let mut out = [0u8; 16];
        let kind = orrin_script::with_active_world(&mut world, || {
            find_asset(c"fonts/title.ttf".as_ptr(), out.as_mut_ptr())
        });
        assert_eq!(kind, orrin_script::ASSET_FONT);
        let id = world
            .resource::<crate::asset_db::AssetDatabase>()
            .find("fonts/title.ttf")
            .unwrap()
            .id;
        assert_eq!(AssetId::from_bytes(out), id);
    }

    /// A `.fmat` is found by path as a material, Ready once it reads and
    /// Failed while it does not; an engine-built material is always Ready.
    #[test]
    fn a_script_finds_a_material_by_path_and_reads_its_load_state() {
        let dir = crate::asset_db::tests::Dir::new("script-materials");
        let brick = crate::asset_db::material::write(&Default::default());
        dir.write("props/brick.fmat", brick.as_bytes());
        dir.write("props/broken.fmat", b"orrin-scene 3 min-reader 3\n");
        let (db, _) = crate::asset_db::AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        let mut assets = Assets::default();
        assets.insert_material("gold", MaterialHandle(7));
        let mut world = World::new();
        world.insert_resource(db);
        world.insert_resource(assets);

        let find = |world: &mut World, path: &std::ffi::CStr| {
            let mut out = [0u8; 16];
            let kind = orrin_script::with_active_world(world, || {
                find_asset(path.as_ptr(), out.as_mut_ptr())
            });
            (kind, AssetId::from_bytes(out))
        };
        let state = |world: &mut World, id: AssetId| {
            let bytes = id.to_bytes();
            orrin_script::with_active_world(world, || asset_state(bytes.as_ptr()))
        };

        let (kind, brick) = find(&mut world, c"props/brick.fmat");
        assert_eq!(kind, orrin_script::ASSET_MATERIAL);
        let (kind, broken) = find(&mut world, c"props/broken.fmat");
        assert_eq!(kind, orrin_script::ASSET_MATERIAL);

        assert_eq!(state(&mut world, brick), orrin_script::LOAD_READY);
        assert_eq!(state(&mut world, broken), orrin_script::LOAD_FAILED);
        let gold = AssetId::named("material", "gold");
        assert_eq!(state(&mut world, gold), orrin_script::LOAD_READY);
    }

    fn c(entity: Entity) -> CEntity {
        CEntity {
            index: entity.index,
            generation: entity.generation,
        }
    }

    fn lens(priority: i32) -> CCamera {
        CCamera {
            fov_degrees: 70.0,
            near: 0.2,
            far: 300.0,
            priority,
        }
    }

    fn camera_world() -> World {
        let mut world = World::new();
        world.insert_resource(crate::scene::Camera::default());
        world.insert_resource(ActiveCamera::default());
        world
    }

    /// Structural, so it lands when the dispatch's commands drain; a lens that
    /// cannot project never reaches the queue.
    #[test]
    fn a_script_adds_a_camera_after_the_dispatch_and_never_a_broken_one() {
        let mut world = camera_world();
        let entity = world.spawn();
        let broken = CCamera {
            near: 0.0,
            ..lens(0)
        };
        let (added, refused) = orrin_script::with_active_world(&mut world, || {
            (
                add_camera(c(entity), &lens(4)),
                add_camera(c(entity), &broken),
            )
        });
        assert!(added);
        assert!(!refused);
        assert!(!world.has::<CameraComponent>(entity), "deferred");

        let commands: Vec<Command> = COMMANDS.with(|c| c.borrow_mut().drain(..).collect());
        let [
            Command::AddCamera {
                entity: queued,
                camera,
            },
        ] = commands.as_slice()
        else {
            panic!("expected exactly the one valid camera to be queued");
        };
        assert_eq!(*queued, entity);
        assert_eq!(camera.priority, 4);
    }

    #[test]
    fn a_script_reads_and_retunes_a_lens_in_place() {
        let mut world = camera_world();
        let entity = world.spawn();
        world.insert(entity, CameraComponent::default());

        let (read, written, refused) = orrin_script::with_active_world(&mut world, || {
            let mut read = lens(0);
            get_camera_lens(c(entity), &mut read);
            let written = set_camera_lens(c(entity), &lens(2));
            let refused = set_camera_lens(
                c(entity),
                &CCamera {
                    far: 0.1,
                    ..lens(9)
                },
            );
            (read, written, refused)
        });
        let default = CameraComponent::default();
        assert_eq!(
            (read.fov_degrees, read.near, read.far, read.priority),
            (
                default.fov_degrees,
                default.near,
                default.far,
                default.priority
            )
        );
        assert!(written);
        assert!(!refused);
        let camera = *world.get::<CameraComponent>(entity).unwrap();
        assert_eq!(
            (camera.fov_degrees, camera.far, camera.priority),
            (70.0, 300.0, 2)
        );
        assert!(orrin_script::with_active_world(&mut world, || {
            has_component(c(entity), 3)
        }));
    }

    #[test]
    fn a_script_switches_between_cameras_and_back_to_priority() {
        let mut world = camera_world();
        let first = world.spawn();
        world.insert(first, CameraComponent::default());
        let second = world.spawn();
        world.insert(second, CameraComponent::default());
        let not_a_camera = world.spawn();

        orrin_script::with_active_world(&mut world, || {
            assert_eq!(
                get_active_camera().index,
                first.index,
                "priority tie: lower slot"
            );
            assert!(set_active_camera(c(second)));
            assert_eq!(get_active_camera().index, second.index);
            assert!(!set_active_camera(c(not_a_camera)));
            assert_eq!(
                get_active_camera().index,
                second.index,
                "a refusal changes nothing"
            );
            assert!(set_active_camera(CEntity::NULL));
            assert_eq!(get_active_camera().index, first.index);
        });
    }

    /// A camera a script adds is one it means to look through, usually at once.
    #[test]
    fn a_camera_queued_this_tick_can_be_selected_this_tick() {
        let mut world = camera_world();
        let other = world.spawn();
        world.insert(other, CameraComponent::default());
        let fresh = world.spawn();
        let selected = orrin_script::with_active_world(&mut world, || {
            add_camera(c(fresh), &lens(0)) && set_active_camera(c(fresh))
        });
        assert!(selected);
        COMMANDS.with(|c| c.borrow_mut().clear());
        assert_eq!(world.resource::<ActiveCamera>().selected(), Some(fresh));
    }

    /// `Camera.position` and friends move the camera the game draws from, so
    /// a script written before camera entities keeps working with one active.
    /// Read back through the local transform, not last frame's world one, so
    /// setting the rotation and then the position keeps both.
    #[test]
    fn the_view_api_moves_the_active_camera_entity_and_reads_this_tick() {
        let mut world = camera_world();
        let player = world.spawn();
        world.insert(
            player,
            LocalTransform::new(Transform {
                translation: Vec3::new(10.0, 0.0, 0.0),
                ..Transform::default()
            }),
        );
        let camera = world.spawn();
        world.insert(camera, LocalTransform::default());
        world.insert(camera, CameraComponent::default());
        crate::scene::reparent(&mut world, camera, Some(player), false).unwrap();
        crate::scene::propagate_transforms(&mut world);

        let facing = Quat::from_rotation_y(0.5);
        let read = orrin_script::with_active_world(&mut world, || {
            let mut view = CTransform {
                position: [0.0; 3],
                rotation: [0.0; 4],
                scale: [1.0; 3],
            };
            assert!(get_camera(&mut view));
            view.rotation = facing.to_array();
            assert!(set_camera(&view));
            assert!(get_camera(&mut view));
            view.position = [12.0, 1.0, 0.0];
            assert!(set_camera(&view));
            assert!(get_camera(&mut view));
            view
        });
        assert!(Vec3::from_array(read.position).distance(Vec3::new(12.0, 1.0, 0.0)) < 1e-4);
        assert!(Quat::from_array(read.rotation).angle_between(facing) < 1e-4);

        let local = *world.get::<LocalTransform>(camera).unwrap();
        assert!(
            local.translation.distance(Vec3::new(2.0, 1.0, 0.0)) < 1e-4,
            "{local:?}"
        );
        assert!(local.rotation.angle_between(facing) < 1e-4);
        assert_eq!(
            world.resource::<crate::scene::Camera>().position,
            crate::scene::Camera::default().position,
            "the view itself follows the entity at the end of the frame"
        );
    }

    /// What C# hands across for each kind, with every field away from its
    /// default so a dropped one shows.
    fn c_image() -> orrin_script::CImage {
        orrin_script::CImage {
            texture: AssetId::new().to_bytes(),
            color: [0.1, 0.2, 0.3, 0.4],
            slice: [1.0, 2.0, 3.0, 4.0],
            raycast_target: 0,
        }
    }

    fn bytes_of<T>(value: &T) -> *const u8 {
        (value as *const T).cast()
    }

    fn size_of<T>() -> u32 {
        std::mem::size_of::<T>() as u32
    }

    /// Every kind reads back what was written to it, and an add is deferred
    /// like any structural change.
    #[test]
    fn each_ui_component_round_trips_through_the_generic_pair() {
        use orrin_script::{CButton, CCanvas, CRectTransform, CText};

        let mut world = World::new();
        let entity = world.spawn();
        let canvas = CCanvas {
            scale_mode: 1,
            reference_resolution: [1280.0, 720.0],
            match_width_or_height: 0.5,
            sort_order: -2,
        };
        let rect = CRectTransform {
            anchor_min: [0.1, 0.2],
            anchor_max: [0.3, 0.4],
            pivot: [0.5, 0.6],
            anchored_position: [7.0, 8.0],
            size_delta: [9.0, 10.0],
            scale: [2.0, 3.0],
        };
        let image = c_image();
        let text = CText {
            font: AssetId::new().to_bytes(),
            color: [1.0, 0.0, 0.5, 1.0],
            size: 40.0,
            anchor: 4,
            wrap: 0,
            raycast_target: 1,
        };
        let button = CButton {
            normal: [0.1; 4],
            hover: [0.2; 4],
            pressed: [0.3; 4],
            disabled: [0.4; 4],
            interactable: 0,
        };
        let added = orrin_script::with_active_world(&mut world, || {
            [
                add_component(
                    c(entity),
                    KIND_CANVAS,
                    bytes_of(&canvas),
                    size_of::<CCanvas>(),
                ),
                add_component(
                    c(entity),
                    KIND_RECT_TRANSFORM,
                    bytes_of(&rect),
                    size_of::<CRectTransform>(),
                ),
                add_component(
                    c(entity),
                    KIND_IMAGE,
                    bytes_of(&image),
                    size_of::<orrin_script::CImage>(),
                ),
                add_component(c(entity), KIND_TEXT, bytes_of(&text), size_of::<CText>()),
                add_component(
                    c(entity),
                    KIND_BUTTON,
                    bytes_of(&button),
                    size_of::<CButton>(),
                ),
            ]
        });
        assert_eq!(added, [true; 5]);
        assert!(!world.has::<crate::ui::Image>(entity), "deferred");
        let commands: Vec<Command> = COMMANDS.with(|c| c.borrow_mut().drain(..).collect());
        assert_eq!(commands.len(), 5);
        for command in commands {
            let Command::AddUi { entity, component } = command else {
                panic!("not a UI add");
            };
            component.insert(&mut world, entity);
        }

        fn read<T: Default>(world: &mut World, entity: Entity, kind: u32) -> T {
            let mut out = T::default();
            let read = orrin_script::with_active_world(world, || {
                get_component(c(entity), kind, (&mut out as *mut T).cast(), size_of::<T>())
            });
            assert!(read, "kind {kind}");
            out
        }
        assert_eq!(read::<CCanvas>(&mut world, entity, KIND_CANVAS), canvas);
        assert_eq!(
            read::<CRectTransform>(&mut world, entity, KIND_RECT_TRANSFORM),
            rect
        );
        assert_eq!(
            read::<orrin_script::CImage>(&mut world, entity, KIND_IMAGE),
            image
        );
        assert_eq!(read::<CText>(&mut world, entity, KIND_TEXT), text);
        assert_eq!(read::<CButton>(&mut world, entity, KIND_BUTTON), button);

        let moved = CRectTransform {
            anchored_position: [-1.0, -2.0],
            ..rect
        };
        assert!(orrin_script::with_active_world(&mut world, || {
            set_component(
                c(entity),
                KIND_RECT_TRANSFORM,
                bytes_of(&moved),
                size_of::<CRectTransform>(),
            )
        }));
        assert_eq!(
            world
                .get::<crate::ui::RectTransform>(entity)
                .unwrap()
                .anchored_position,
            glam::Vec2::new(-1.0, -2.0)
        );
        assert!(orrin_script::with_active_world(&mut world, || {
            (0..=KIND_BUTTON).all(|kind| kind < KIND_CANVAS || has_component(c(entity), kind))
        }));
    }

    /// A bindings build whose struct is not the engine's size is refused by
    /// name rather than read past the end of.
    #[test]
    fn a_struct_of_the_wrong_size_is_refused_by_name() {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, crate::ui::Image::default());
        let image = c_image();
        let lines = logged(|| {
            orrin_script::with_active_world(&mut world, || {
                let mut out = c_image();
                let short = size_of::<orrin_script::CImage>() - 4;
                assert!(!get_component(
                    c(entity),
                    KIND_IMAGE,
                    (&mut out as *mut orrin_script::CImage).cast(),
                    short
                ));
                assert!(!set_component(
                    c(entity),
                    KIND_IMAGE,
                    bytes_of(&image),
                    short
                ));
                assert!(!add_component(
                    c(entity),
                    KIND_IMAGE,
                    bytes_of(&image),
                    short
                ));
            });
        });
        assert_eq!(lines.len(), 3, "{lines:?}");
        for (level, line) in &lines {
            assert_eq!(*level, LogLevel::Warning);
            assert!(
                line.contains("Image") && line.contains("48") && line.contains("52"),
                "{line}"
            );
        }
        assert_eq!(
            *world.get::<crate::ui::Image>(entity).unwrap(),
            crate::ui::Image::default()
        );
        assert!(COMMANDS.with(|c| c.borrow().is_empty()));
    }

    /// The `get_tag` protocol: the full length comes back whatever the
    /// buffer, so C# can retry once with one that fits.
    #[test]
    fn a_text_is_read_whole_after_a_retry_and_written_by_length() {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(
            entity,
            crate::ui::Text {
                text: "Start the game".into(),
                ..crate::ui::Text::default()
            },
        );
        let bare = world.spawn();
        orrin_script::with_active_world(&mut world, || {
            let mut small = [0u8; 4];
            let total = get_component_text(c(entity), KIND_TEXT, small.as_mut_ptr(), 4);
            assert_eq!(total, 14);
            assert_eq!(&small, b"Star");
            let mut whole = vec![0u8; total as usize];
            assert_eq!(
                get_component_text(c(entity), KIND_TEXT, whole.as_mut_ptr(), total),
                14
            );
            assert_eq!(whole, b"Start the game");
            assert_eq!(
                get_component_text(c(bare), KIND_TEXT, small.as_mut_ptr(), 4),
                -1
            );
            assert_eq!(
                get_component_text(c(entity), KIND_IMAGE, small.as_mut_ptr(), 4),
                -1
            );

            let text = "Quit — now";
            assert!(set_component_text(
                c(entity),
                KIND_TEXT,
                text.as_ptr(),
                text.len() as u32
            ));
            assert!(!set_component_text(
                c(bare),
                KIND_TEXT,
                text.as_ptr(),
                text.len() as u32
            ));
        });
        assert_eq!(
            world.get::<crate::ui::Text>(entity).unwrap().text,
            "Quit — now"
        );
    }

    /// A set made in the same tick as the add lands on the queued component,
    /// so `AddComponent` then `SetText` is not a silent no-op.
    #[test]
    fn a_set_in_the_tick_of_the_add_lands_on_the_queued_component() {
        let mut world = World::new();
        let entity = world.spawn();
        let text = orrin_script::CText {
            size: 30.0,
            wrap: 1,
            ..Default::default()
        };
        orrin_script::with_active_world(&mut world, || {
            let size = size_of::<orrin_script::CText>();
            assert!(add_component(c(entity), KIND_TEXT, bytes_of(&text), size));
            let label = "Ready";
            assert!(set_component_text(
                c(entity),
                KIND_TEXT,
                label.as_ptr(),
                label.len() as u32
            ));
            let bigger = orrin_script::CText { size: 50.0, ..text };
            assert!(set_component(c(entity), KIND_TEXT, bytes_of(&bigger), size));
            let mut back = [0u8; 8];
            assert_eq!(
                get_component_text(c(entity), KIND_TEXT, back.as_mut_ptr(), 8),
                5
            );
        });
        let commands: Vec<Command> = COMMANDS.with(|c| c.borrow_mut().drain(..).collect());
        let [
            Command::AddUi {
                entity: queued,
                component,
            },
        ] = commands.as_slice()
        else {
            panic!("expected the one queued add");
        };
        let mut placed = World::new();
        let target = placed.spawn();
        assert_eq!(*queued, entity);
        component.clone().insert(&mut placed, target);
        let added = placed.get::<crate::ui::Text>(target).unwrap();
        assert_eq!((added.text.as_str(), added.size), ("Ready", 50.0));
    }

    #[test]
    fn a_script_asks_whether_the_pointer_is_over_the_ui() {
        use crate::ui::events::{PointerInput, UiEvents, UiPointer, step};

        let mut world = World::new();
        assert!(!orrin_script::with_active_world(&mut world, || {
            ui_pointer_over()
        }));
        let canvas = world.spawn();
        world.insert(canvas, crate::ui::Canvas::default());
        let panel = world.spawn();
        world.insert(
            panel,
            crate::ui::RectTransform {
                anchor_min: glam::Vec2::ZERO,
                anchor_max: glam::Vec2::ZERO,
                pivot: glam::Vec2::ZERO,
                ..crate::ui::RectTransform::default()
            },
        );
        world.insert(panel, crate::ui::Image::default());
        crate::scene::reparent(&mut world, panel, Some(canvas), false).unwrap();
        crate::ui::update(&mut world, [800, 600]);

        for (x, over) in [(50.0, true), (500.0, false)] {
            let mut pointer = UiPointer::default();
            let input = PointerInput {
                position: [x, 50.0],
                ..PointerInput::default()
            };
            let tree = world.resource::<crate::ui::UiTree>();
            step(
                &world,
                &tree.layout,
                &mut pointer,
                input,
                &mut UiEvents::default(),
            );
            drop(tree);
            world.insert_resource(pointer);
            assert_eq!(
                orrin_script::with_active_world(&mut world, || ui_pointer_over()),
                over
            );
        }
    }

    fn ui_world() -> World {
        let mut world = World::new();
        let mut canvas = UiCanvas::default();
        canvas.begin([1920, 1080]);
        world.insert_resource(canvas);
        world
    }

    #[test]
    fn a_script_draws_into_the_canvas_in_call_order() {
        use crate::scene::{Rect, TextAnchor, TextStyle, UiCommand};

        let mut world = ui_world();
        let icon = AssetId::new();
        let text = "héllo";
        orrin_script::with_active_world(&mut world, || {
            ui_panel(10.0, 20.0, 300.0, 40.0, 0.0, 0.0, 0.0, 0.5);
            ui_text(
                text.as_ptr(),
                text.len() as u32,
                std::ptr::null(),
                24.0,
                4,
                10.0,
                20.0,
                300.0,
                40.0,
                1.0,
                1.0,
                1.0,
                1.0,
            );
            ui_image(
                icon.to_bytes().as_ptr(),
                0.0,
                0.0,
                64.0,
                64.0,
                1.0,
                1.0,
                1.0,
                1.0,
            );
        });

        let canvas = world.resource::<UiCanvas>();
        let rect = Rect::new(10.0, 20.0, 300.0, 40.0);
        let commands = canvas.commands();
        assert_eq!(commands.len(), 3, "{commands:?}");
        assert_eq!(
            commands[0],
            UiCommand::Panel {
                rect,
                color: [0, 0, 0, 128]
            }
        );
        let UiCommand::Text {
            rect: text_rect,
            style,
            text: range,
        } = &commands[1]
        else {
            panic!("{:?}", commands[1]);
        };
        assert_eq!(*text_rect, rect);
        assert_eq!(
            *style,
            TextStyle {
                font: None,
                size: 24.0,
                anchor: TextAnchor::MiddleCenter,
                color: [255; 4],
                wrap: true,
            }
        );
        assert_eq!(canvas.str(range), "héllo");
        assert_eq!(
            commands[2],
            UiCommand::Image {
                rect: Rect::new(0.0, 0.0, 64.0, 64.0),
                texture: icon,
                uv: [0.0, 0.0, 1.0, 1.0],
                tint: [255; 4],
            }
        );
    }

    /// The anchor crosses as a number C# owns the meaning of; one past the end
    /// is a version mismatch and sets text as an unanchored label would.
    #[test]
    fn a_script_names_its_font_and_an_unknown_anchor_falls_back() {
        use crate::scene::{TextAnchor, UiCommand};

        let mut world = ui_world();
        let font = AssetId::new();
        orrin_script::with_active_world(&mut world, || {
            ui_text(
                "a".as_ptr(),
                1,
                font.to_bytes().as_ptr(),
                12.0,
                99,
                0.0,
                0.0,
                10.0,
                10.0,
                1.0,
                1.0,
                1.0,
                1.0,
            );
        });
        let canvas = world.resource::<UiCanvas>();
        let UiCommand::Text { style, .. } = &canvas.commands()[0] else {
            panic!("{:?}", canvas.commands());
        };
        assert_eq!(style.font, Some(font));
        assert_eq!(style.anchor, TextAnchor::UpperLeft);
    }

    #[test]
    fn a_script_reads_the_screen_its_tick_began_with() {
        let mut world = ui_world();
        let (mut width, mut height) = (0.0, 0.0);
        orrin_script::with_active_world(&mut world, || ui_screen_size(&mut width, &mut height));
        assert_eq!((width, height), (1920.0, 1080.0));
    }

    /// A tool or a test may tick scripts with no canvas; the calls do nothing.
    #[test]
    fn ui_calls_without_a_canvas_are_ignored() {
        let mut world = World::new();
        let (mut width, mut height) = (1.0, 1.0);
        orrin_script::with_active_world(&mut world, || {
            ui_panel(0.0, 0.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0);
            ui_text(
                std::ptr::null(),
                0,
                std::ptr::null(),
                12.0,
                0,
                0.0,
                0.0,
                10.0,
                10.0,
                1.0,
                1.0,
                1.0,
                1.0,
            );
            ui_screen_size(&mut width, &mut height);
        });
        assert_eq!((width, height), (0.0, 0.0));
    }
}
