//! What the remote answers, method by method. The reads here take what the
//! frame's thread holds and change nothing; each answer that lists things is
//! bounded by a `limit` with a default, and says where the next page starts.
//! The two that change things are in [`edit`](super::edit) and
//! [`command`](super::command).
//!
//! | method | params | result |
//! | --- | --- | --- |
//! | `client_info` | `name`, `version` | nothing: who the client is, for Settings ▸ Agents, answered by its connection's own thread |
//! | `editor_state` | none | project, agents connected, permissions, loaded scenes and the project's start and persistent ones, open scene, unsaved, play state, world read, selection, camera |
//! | `scene_read` | `ids`, `root`, `name`, `scene`, `blocks`, `limit`, `offset` | an outline, or entity blocks in the scene grammar; each entity's scene when several are loaded |
//! | `component_list` | `query`, `defaults`, `limit`, `offset` | registered components, each with its default as text |
//! | `asset_find` | `kind`, `path`, `limit`, `offset` | indexed assets: id, path, kind, load state, import error |
//! | `console_read` | `cursor`, `level`, `limit` | console entries numbered from `cursor`, and the next cursor |
//! | `frame_stats` | `limit` | the run banner, frame time, and the profiler's CPU phases and GPU passes |
//! | `screenshot` | `view`, `max_size` | the scene as a PNG in a blob frame, then its size and timings ([`shot`](super::shot)) |
//! | `scene_edit` | `label`, `ops` | the step's label, the ops applied, ids spawned and given |
//! | `editor_command` | `command` and its own | what the command did, once it can be seen |
//!
//! An entity is named by its [`EntityId`], or by `#slot` when it has none —
//! one a script spawned, or one never saved. A block `scene_read` returns for
//! an entity with an id is exactly what a save would write for it; one with
//! none is the debug dump's form, which the scene parser refuses on purpose.

use orrin_ecs::{Entity, FxBuildHasher, FxHashMap, World};
use orrin_registry::{EntityId, Registry, SceneDocument, SceneEntity};
use orrin_remote::wire::Request;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use super::command::Held;
use crate::asset_db::{AssetDatabase, AssetKind, LoadState, Status};
use crate::editor::state::EditorState;
use crate::scene::loaded::{LoadedScenes, scene_of};
use crate::scene::{Camera, LogBuffer, LogLevel, Name, Play, PlayState, persist};
use orrin_remote::permissions::Access;

type FxHashSet<T> = std::collections::HashSet<T, FxBuildHasher>;

const METHODS: [&str; 9] = [
    "editor_state",
    "scene_read",
    "component_list",
    "asset_find",
    "console_read",
    "frame_stats",
    "screenshot",
    "scene_edit",
    "editor_command",
];

/// An answer now, a command held until its effect can be seen, or a
/// screenshot to be drawn this size at most.
pub(super) enum Reply {
    Now(Result<Value, String>),
    Held(Held),
    Screenshot(u32, crate::gfx::vulkan::frame::ShotView),
}

pub(super) fn answer(
    request: &Request,
    world: &mut World,
    registry: &Registry,
    state: &mut EditorState,
) -> Reply {
    let params = &request.params;
    let method = request.method.as_str();
    // The one check of what an agent may do: every method, and every command
    // by name, passes here before anything reads or changes a thing.
    if let Err(refused) = state.settings.agents.allow.check(method, params) {
        return Reply::Now(Err(refused));
    }
    let answer = match method {
        "screenshot" => match super::shot::asked(params) {
            Ok((size, view)) => return Reply::Screenshot(size, view),
            Err(refused) => Err(refused),
        },
        "scene_edit" => super::edit::scene_edit(world, registry, state, params),
        "editor_command" => match super::command::editor_command(world, registry, state, params) {
            Ok(Err(held)) => return Reply::Held(held),
            Ok(Ok(done)) => Ok(done),
            Err(refused) => Err(refused),
        },
        _ => read(method, world, registry, state, params),
    };
    Reply::Now(answer.map_err(|why| format!("`{method}`: {why}")))
}

fn read(
    method: &str,
    world: &World,
    registry: &Registry,
    state: &EditorState,
    params: &Value,
) -> Result<Value, String> {
    match method {
        "editor_state" => params_of::<Nothing>(params).map(|_| editor_state(world, state)),
        "scene_read" => scene_read(world, registry, state, params),
        "component_list" => component_list(registry, params),
        "asset_find" => asset_find(world, state, params),
        "console_read" => console_read(world, params),
        "frame_stats" => frame_stats(world, params),
        other => Err(format!(
            "the editor has no such method; it answers {} (asked for `{other}`)",
            METHODS.join(", ")
        )),
    }
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Nothing {}

/// Absent params are every default.
pub(super) fn params_of<T: DeserializeOwned + Default>(params: &Value) -> Result<T, String> {
    if params.is_null() {
        return Ok(T::default());
    }
    params_of_required(params)
}

pub(super) fn params_of_required<T: DeserializeOwned>(params: &Value) -> Result<T, String> {
    T::deserialize(params).map_err(|error| format!("its params do not read: {error}"))
}

/// `limit` within `1..=max`, `default` when not given.
fn bounded(limit: Option<usize>, default: usize, max: usize) -> usize {
    limit.unwrap_or(default).clamp(1, max)
}

fn page<T>(all: Vec<T>, offset: usize, limit: usize) -> (usize, Vec<T>, Option<usize>) {
    let total = all.len();
    let next = (offset.saturating_add(limit) < total).then_some(offset + limit);
    (
        total,
        all.into_iter().skip(offset).take(limit).collect(),
        next,
    )
}

/// Why every folder is refused: the folder rules did not read.
fn refused_folders(why: &str) -> String {
    format!(
        "{why}; agents are refused every folder until it reads. Fix it, or reset the folder \
         rules in Settings ▸ Permissions"
    )
}

/// Whether the folder rules let an agent go as far as `need` at `path`:
/// absolute, or relative to the project root.
pub(super) fn permit(
    folders: &Result<orrin_remote::permissions::Folders, String>,
    root: Option<&std::path::Path>,
    path: &std::path::Path,
    need: Access,
) -> Result<(), String> {
    let folders = folders.as_ref().map_err(|why| refused_folders(why))?;
    let relative = match root {
        Some(root) if path.is_absolute() => path.strip_prefix(root).map_err(|_| {
            format!(
                "{} is outside the project, where agents have no access",
                path.display()
            )
        })?,
        _ => path,
    };
    folders.permit(&relative.to_string_lossy(), need)
}

pub(super) fn world_read(state: &EditorState) -> &'static str {
    if state.playing() { "play" } else { "edit" }
}

fn reference(world: &World, entity: Entity) -> String {
    match world.get::<EntityId>(entity) {
        Some(id) => id.to_string(),
        None => format!("#{}", entity.index()),
    }
}

fn resolve(world: &World, state: &EditorState, text: &str) -> Result<Entity, String> {
    find(world, text, world_read(state))
}

/// The live entity `text` names: an [`EntityId`], or `#slot` for one that has
/// none. `read` names the world in an error.
pub(super) fn find(world: &World, text: &str, read: &str) -> Result<Entity, String> {
    if let Some(slot) = text.strip_prefix('#') {
        let slot: u32 = slot
            .parse()
            .map_err(|_| format!("`{text}` is not a `#slot`"))?;
        return world
            .entities()
            .find(|&entity| entity.index() == slot && world.get::<EntityId>(entity).is_none())
            .ok_or_else(|| {
                format!("no entity without an id is in slot {slot} of the {read} world")
            });
    }
    let id: EntityId = text
        .parse()
        .map_err(|_| format!("`{text}` is neither an entity id nor a `#slot`"))?;
    let mut found = None;
    world.query::<&EntityId>().for_each(|entity, have| {
        if *have == id {
            found = Some(entity);
        }
    });
    found.ok_or_else(|| format!("no entity in the {read} world has id {id}"))
}

/// Every loaded scene, the active one first: its handle, name, file, role,
/// how many entities belong to it by the membership rule, and for the active
/// one whether it has unsaved changes.
fn loaded_scenes(world: &World, state: &EditorState) -> Vec<Value> {
    let Some(loaded) = world.get_resource::<LoadedScenes>() else {
        return Vec::new();
    };
    let mut counts: FxHashMap<u32, usize> = FxHashMap::default();
    for entity in world.entities() {
        *counts.entry(scene_of(world, entity).get()).or_default() += 1;
    }
    let db = world.get_resource::<AssetDatabase>();
    let active = loaded.active().handle;
    loaded
        .iter()
        .map(|scene| {
            let path = scene
                .asset
                .and_then(|asset| Some(db.as_ref()?.get(asset)?.path.clone()))
                .map(|path| match state.agent_assets.as_str() {
                    "" => path,
                    assets => format!("{assets}/{path}"),
                });
            let role = match scene {
                _ if scene.handle == active => "active",
                scene if scene.persistent => "persistent",
                _ => "additive",
            };
            let mut entry = json!({
                "handle": scene.handle.get(),
                "name": scene.name,
                "path": path,
                "role": role,
                "entities": counts.get(&scene.handle.get()).copied().unwrap_or(0),
            });
            if scene.handle == active {
                entry["unsaved"] = Value::Bool(state.scene_dirty());
            }
            entry
        })
        .collect()
}

pub(super) fn selection(world: &World, state: &EditorState) -> Vec<String> {
    state
        .selection
        .entities()
        .iter()
        .filter(|&&entity| world.is_alive(entity))
        .map(|&entity| reference(world, entity))
        .collect()
}

fn editor_state(world: &World, state: &EditorState) -> Value {
    let play = match world.get_resource::<Play>().map(|play| play.state()) {
        Some(PlayState::Playing) => "playing",
        Some(PlayState::Paused) => "paused",
        Some(PlayState::Editing) | None => "editing",
    };
    let selection = selection(world, state);
    let scenes = loaded_scenes(world, state);
    let manifest = state
        .project_root
        .as_ref()
        .and_then(|root| orrin_project::Project::load(&root.join("orrin.toml")).ok());
    let shown = |path: Option<&std::path::Path>| path.map(|path| path.display().to_string());
    let camera = world.get_resource::<Camera>().map(|camera| {
        json!({
            "position": camera.position.to_array(),
            "target": camera.target.to_array(),
        })
    });
    let agents: Vec<Value> = state
        .remote_peers
        .iter()
        .map(|peer| {
            json!({
                "connection": peer.id,
                "client": peer.client.as_deref().unwrap_or("unknown client"),
                "connected_secs": peer.since.elapsed().as_secs(),
                "requests": peer.requests,
                "last": peer.last,
            })
        })
        .collect();
    json!({
        "project": state.project_name,
        "agents": agents,
        "scenes": scenes,
        "start_scene": manifest.as_ref().and_then(|p| shown(p.start_scene_relative())),
        "persistent_scene": manifest
            .as_ref()
            .and_then(|p| shown(p.persistent_scene_relative())),
        "permissions": orrin_remote::permissions::summary(
            &state.settings.agents.allow,
            &state.agent_folders,
        ),
        "scene": state.scene_path,
        "unsaved": state.scene_dirty(),
        "play": play,
        "world": world_read(state),
        "selection": selection,
        "camera": camera,
    })
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct SceneRead {
    /// Only these entities.
    ids: Vec<String>,
    /// Only this entity and everything under it.
    root: Option<String>,
    /// Only entities whose name holds this, ignoring case.
    name: Option<String>,
    /// Whole blocks in the scene grammar rather than an outline.
    blocks: bool,
    /// Only the entities of the loaded scene with this handle.
    scene: Option<u32>,
    limit: Option<usize>,
    offset: usize,
}

fn scene_read(
    world: &World,
    registry: &Registry,
    state: &EditorState,
    params: &Value,
) -> Result<Value, String> {
    let read: SceneRead = params_of(params)?;
    let wanted: Option<FxHashSet<Entity>> = match read.ids.is_empty() {
        true => None,
        false => Some(
            read.ids
                .iter()
                .map(|id| resolve(world, state, id))
                .collect::<Result<_, _>>()?,
        ),
    };
    let root = read
        .root
        .as_deref()
        .map(|root| resolve(world, state, root))
        .transpose()?;
    let name = read.name.as_deref().map(str::to_lowercase);
    let order = persist::save_order(world);
    let depth = order.len();
    let matched: Vec<Entity> = order
        .into_iter()
        .filter(|&entity| {
            wanted
                .as_ref()
                .is_none_or(|wanted| wanted.contains(&entity))
                && root.is_none_or(|root| under(world, entity, root, depth))
                && name.as_ref().is_none_or(|name| {
                    world
                        .get::<Name>(entity)
                        .is_some_and(|have| have.0.to_lowercase().contains(name))
                })
                && read
                    .scene
                    .is_none_or(|scene| scene_of(world, entity).get() == scene)
        })
        .collect();
    // With one scene loaded every entity is in it, and saying so is noise.
    let several = world
        .get_resource::<LoadedScenes>()
        .is_some_and(|loaded| loaded.iter().nth(1).is_some());
    let limit = match read.blocks {
        true => bounded(read.limit, 20, 200),
        false => bounded(read.limit, 100, 1000),
    };
    let (total, entities, next) = page(matched, read.offset, limit);
    let mut answer = json!({
        "world": world_read(state),
        "total": total,
        "offset": read.offset,
        "next": next,
    });
    if read.blocks {
        let (scene, unsaved, notes) = blocks(world, registry, &entities);
        if !scene.is_empty() {
            answer["scene"] = Value::String(scene);
        }
        if !unsaved.is_empty() {
            answer["unsaved"] = Value::String(unsaved);
        }
        if !notes.is_empty() {
            answer["notes"] = json!(notes);
        }
        if several {
            let in_scene: Map<String, Value> = entities
                .iter()
                .map(|&entity| {
                    let handle = scene_of(world, entity).get();
                    (reference(world, entity), Value::from(handle))
                })
                .collect();
            answer["in_scene"] = Value::Object(in_scene);
        }
    } else {
        let outline: Vec<Value> = entities
            .iter()
            .map(|&entity| {
                let components: Vec<&str> = registry
                    .components()
                    .filter(|component| component.has(world, entity))
                    .map(|component| component.id.as_str())
                    .collect();
                let mut entry = json!({
                    "id": reference(world, entity),
                    "name": world.get::<Name>(entity).map(|name| name.0.clone()),
                    "parent": persist::live_parent(world, entity)
                        .map(|parent| reference(world, parent)),
                    "components": components,
                });
                if several {
                    entry["in_scene"] = Value::from(scene_of(world, entity).get());
                }
                entry
            })
            .collect();
        answer["entities"] = Value::Array(outline);
    }
    Ok(answer)
}

/// Whether `entity` is `root` or below it. Bounded by the entity count, since
/// the hierarchy breaks a cycle only at its next propagation.
fn under(world: &World, mut entity: Entity, root: Entity, steps: usize) -> bool {
    for _ in 0..=steps {
        if entity == root {
            return true;
        }
        match persist::live_parent(world, entity) {
            Some(parent) => entity = parent,
            None => return false,
        }
    }
    false
}

/// The save's own blocks for `entities` that have an id, as one document; the
/// debug dump of those that have none; and what a reader should know about
/// either.
fn blocks(
    world: &World,
    registry: &Registry,
    entities: &[Entity],
) -> (String, String, Vec<String>) {
    let saved: FxHashMap<EntityId, SceneEntity> = persist::as_saved(world, registry)
        .into_iter()
        .map(|block| (block.id, block))
        .collect();
    let mut chosen = Vec::new();
    let mut unsaved = String::new();
    let mut notes = Vec::new();
    for &entity in entities {
        let Some(id) = world.get::<EntityId>(entity).map(|id| *id) else {
            if !unsaved.is_empty() {
                unsaved.push('\n');
            }
            orrin_registry::write_entity(&mut unsaved, registry, world, entity);
            continue;
        };
        match saved.get(&id) {
            Some(block) => chosen.push(block.clone()),
            None => notes.push(format!(
                "{id} is inside a prefab instance: a save writes it into the instance root's \
                 `orrin.prefab` record, not as a block of its own"
            )),
        }
        if let Some(parent) = persist::live_parent(world, entity)
            && world.get::<EntityId>(parent).is_none()
        {
            notes.push(format!(
                "{id}'s parent #{} has no id yet; a save gives it one and writes it as this \
                 block's `orrin.parent`",
                parent.index()
            ));
        }
    }
    if !unsaved.is_empty() {
        notes.push(
            "`unsaved` holds entities with no id: the live dump, headed by slot, which no save \
             has written"
                .to_owned(),
        );
    }
    let mut scene = String::new();
    if !chosen.is_empty() {
        orrin_registry::write_document(&mut scene, &SceneDocument { entities: chosen });
    }
    (scene, unsaved, notes)
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ComponentList {
    /// Only components whose id or name holds this, ignoring case.
    query: Option<String>,
    /// Each component's default value, as text; on unless `false`.
    defaults: Option<bool>,
    limit: Option<usize>,
    offset: usize,
}

fn component_list(registry: &Registry, params: &Value) -> Result<Value, String> {
    let list: ComponentList = params_of(params)?;
    let query = list.query.as_deref().map(str::to_lowercase);
    let mut components: Vec<_> = registry
        .components()
        .filter(|component| {
            query.as_ref().is_none_or(|query| {
                component.id.as_str().to_lowercase().contains(query)
                    || component.name.to_lowercase().contains(query)
            })
        })
        .collect();
    components.sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));
    let limit = bounded(list.limit, 200, 1000);
    let (total, components, next) = page(components, list.offset, limit);
    let defaults = list.defaults.unwrap_or(true);
    let components: Vec<Value> = components
        .into_iter()
        .map(|component| {
            let mut entry = json!({
                "id": component.id.as_str(),
                "name": component.name,
                "script": component.script.is_some(),
            });
            if defaults {
                let mut text = String::new();
                orrin_registry::write_component(
                    &mut text,
                    component.id.as_str(),
                    &component.default(),
                );
                entry["default"] = Value::String(text);
            }
            entry
        })
        .collect();
    Ok(json!({
        "total": total,
        "offset": list.offset,
        "next": next,
        "components": components,
    }))
}

const KINDS: [(&str, AssetKind); 8] = [
    ("texture", AssetKind::Texture),
    ("mesh", AssetKind::Mesh),
    ("prefab", AssetKind::Prefab),
    ("material", AssetKind::Material),
    ("scene", AssetKind::Scene),
    ("font", AssetKind::Font),
    ("audio", AssetKind::Audio),
    ("shader", AssetKind::Shader),
];

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AssetFind {
    kind: Option<String>,
    /// Only assets whose path holds this, ignoring case.
    path: Option<String>,
    limit: Option<usize>,
    offset: usize,
}

fn asset_find(world: &World, state: &EditorState, params: &Value) -> Result<Value, String> {
    let find: AssetFind = params_of(params)?;
    let kind = find
        .kind
        .as_deref()
        .map(|asked| {
            KINDS
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(asked))
                .map(|&(_, kind)| kind)
                .ok_or_else(|| {
                    let names: Vec<&str> = KINDS.iter().map(|(name, _)| *name).collect();
                    format!("`{asked}` is not a kind; one of {}", names.join(", "))
                })
        })
        .transpose()?;
    let db = world.get_resource::<AssetDatabase>().ok_or_else(|| {
        "no project is open, so there is no asset index; open one with `orrin run`".to_owned()
    })?;
    let path = find.path.as_deref().map(str::to_lowercase);
    // What the folder rules keep from agents is not listed: not even its name.
    let folders = state
        .agent_folders
        .as_ref()
        .map_err(|why| refused_folders(why))?;
    let readable = |path: &str| {
        let in_project = match state.agent_assets.as_str() {
            "" => path.to_owned(),
            assets => format!("{assets}/{path}"),
        };
        folders.permit(&in_project, Access::Read).is_ok()
    };
    let mut entries: Vec<_> = db
        .iter()
        .filter(|entry| readable(&entry.path))
        .filter(|entry| kind.is_none_or(|kind| entry.kind == kind))
        .filter(|entry| {
            path.as_ref()
                .is_none_or(|path| entry.path.to_lowercase().contains(path))
        })
        .collect();
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    let limit = bounded(find.limit, 50, 500);
    let (total, entries, next) = page(entries, find.offset, limit);
    let assets: Vec<Value> = entries
        .into_iter()
        .map(|entry| {
            let kind = KINDS
                .iter()
                .find(|(_, kind)| *kind == entry.kind)
                .map_or(entry.kind.label(), |(name, _)| name);
            let state = match db.state(entry.id) {
                LoadState::Loading => "loading",
                LoadState::Ready => "ready",
                LoadState::Failed => "failed",
            };
            let error = match &entry.status {
                Status::Failed(error) => Some(error.as_str()),
                _ => None,
            };
            json!({
                "id": entry.id.to_string(),
                "path": entry.path,
                "kind": kind,
                "state": state,
                "error": error,
            })
        })
        .collect();
    Ok(json!({
        "total": total,
        "offset": find.offset,
        "next": next,
        "assets": assets,
    }))
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ConsoleRead {
    /// The number of the first entry wanted; 0 is the oldest ever logged.
    cursor: u64,
    /// The quietest level returned: trace, debug, info, warn or error.
    level: Option<String>,
    limit: Option<usize>,
}

fn console_read(world: &World, params: &Value) -> Result<Value, String> {
    let read: ConsoleRead = params_of(params)?;
    let floor = match read
        .level
        .as_deref()
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        None | Some("info") => LogLevel::Info,
        Some("trace") => LogLevel::Trace,
        Some("debug") => LogLevel::Debug,
        Some("warn" | "warning") => LogLevel::Warning,
        Some("error") => LogLevel::Error,
        Some(other) => {
            return Err(format!(
                "`{other}` is not a level; one of trace, debug, info, warn, error"
            ));
        }
    };
    let log = world
        .get_resource::<LogBuffer>()
        .ok_or_else(|| "this session has no console".to_owned())?;
    let limit = bounded(read.limit, 100, 1000);
    let oldest = log.next_number() - log.len() as u64;
    // Past the end is a cursor from an editor that has since restarted.
    let cursor = match read.cursor > log.next_number() {
        true => oldest,
        false => read.cursor,
    };
    let mut next = log.next_number();
    let mut entries = Vec::new();
    for (number, entry) in log.since(cursor) {
        if entries.len() == limit {
            next = number;
            break;
        }
        if entry.level < floor {
            continue;
        }
        let level = match entry.level {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warning => "warn",
            LogLevel::Error => "error",
        };
        entries.push(json!({
            "n": number,
            "level": level,
            "message": entry.message,
            "source": entry.source,
            "line": (entry.line != 0).then_some(entry.line),
            "frame": entry.frame,
        }));
    }
    Ok(json!({
        "entries": entries,
        "next": next,
        "lost": oldest.saturating_sub(cursor),
    }))
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FrameStatsParams {
    /// Rows a lane, slowest in the last frame first.
    limit: Option<usize>,
}

/// What the Profiler tab shows, with the banner that says how it was
/// measured. Read, never toggled: the switches that move a frame time are the
/// developer's.
fn frame_stats(world: &World, params: &Value) -> Result<Value, String> {
    let read: FrameStatsParams = params_of(params)?;
    let limit = bounded(read.limit, 20, 100);
    let stats = world
        .get_resource::<crate::stats::FrameStats>()
        .ok_or_else(|| "this session keeps no frame statistics".to_owned())?;
    let (min_ms, max_ms) = stats.min_max_ms();
    let lane = |lane: crate::profile::Lane| -> Vec<Value> {
        world
            .get_resource::<crate::profile::Profiler>()
            .map(|profiler| profiler.aggregate(lane))
            .unwrap_or_default()
            .into_iter()
            .take(limit)
            .map(|row| {
                json!({
                    "name": row.name,
                    "depth": row.depth,
                    "last_ms": row.last_ms,
                    "avg_ms": row.avg_ms,
                    "max_ms": row.max_ms,
                    "calls": row.calls,
                })
            })
            .collect()
    };
    let per_pass = crate::profile::is_enabled() && crate::profile::gpu_passes_enabled();
    let mut answer = json!({
        "banner": stats.run_banner(),
        "frame_ms": stats.frame_ms(),
        "fps": stats.fps(),
        "min_ms": min_ms,
        "max_ms": max_ms,
        "gpu_ms": stats.gpu_ms(),
        "memory_bytes": stats.memory_bytes(),
        "vram_used": stats.vram_used(),
        "vram_total": stats.vram_total(),
        "overlay": world
            .get_resource::<crate::scene::Diagnostics>()
            .is_some_and(|diagnostics| diagnostics.overlay),
        "profiling": crate::profile::is_enabled(),
        "gpu_pass_timings": per_pass,
        "cpu": lane(crate::profile::Lane::Cpu),
        "gpu": lane(crate::profile::Lane::Gpu),
    });
    if per_pass {
        answer["note"] = Value::String(
            "Time each GPU pass is on: a timestamp pair around every pass makes neighbouring \
             passes run in turn, so the GPU rows can sum to more than the frame. Compare whole \
             frames with it off"
                .to_owned(),
        );
    }
    Ok(answer)
}

#[cfg(test)]
mod tests {
    use orrin_remote::permissions::{Capability, FolderRule, Folders};

    use super::*;
    use crate::asset_db::tests::Dir;

    fn call(
        world: &mut World,
        state: &mut EditorState,
        method: &str,
        params: Value,
    ) -> Result<Value, String> {
        let mut registry = Registry::new();
        crate::scene::register_components(&mut registry);
        let request = Request {
            id: 1,
            method: method.into(),
            params,
        };
        match answer(&request, world, &registry, state) {
            Reply::Now(answer) => answer,
            _ => panic!("{method} was held"),
        }
    }

    /// The folder rules govern what the tools do with a project path: an
    /// asset in a folder agents may not read is not even listed, and a scene
    /// saved into a folder they may only read is refused before anything is
    /// asked of the editor. A capability that is off refuses its method by
    /// name, here, before either.
    #[test]
    fn the_folder_rules_hide_what_agents_may_not_read_and_refuse_a_write_outside() {
        let dir = Dir::new("remote-folders");
        dir.write("assets/levels/one.orrin", b"orrin-scene 4 min-reader 4\n");
        dir.write("assets/props/door.oprefab", b"orrin-scene 1\n");
        let mut world = World::default();
        let (db, _) = AssetDatabase::open(dir.path().join("assets"), dir.path().join(".cache"));
        world.insert_resource(db);
        let mut state = EditorState::default();
        state.project_root = Some(dir.path().to_owned());
        state.agent_assets = "assets".to_owned();
        state.agent_folders = Ok(Folders {
            unlisted: orrin_remote::permissions::Access::None,
            rules: vec![
                FolderRule {
                    path: "assets/levels".into(),
                    access: orrin_remote::permissions::Access::Read,
                },
                FolderRule {
                    path: "assets/props".into(),
                    access: orrin_remote::permissions::Access::None,
                },
            ],
        });

        let found = call(&mut world, &mut state, "asset_find", Value::Null).unwrap();
        let paths: Vec<&str> = found["assets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|asset| asset["path"].as_str().unwrap())
            .collect();
        assert_eq!(paths, ["levels/one.orrin"]);

        let save = json!({"command": "save", "path": "assets/levels/two.orrin"});
        let refused = call(&mut world, &mut state, "editor_command", save).unwrap_err();
        assert!(
            refused.contains("may not write to assets/levels/two.orrin"),
            "{refused}"
        );

        state
            .settings
            .agents
            .allow
            .set(Capability::ReadScene, false);
        let refused = call(&mut world, &mut state, "asset_find", Value::Null).unwrap_err();
        assert!(refused.contains("Read scene and assets"), "{refused}");
    }

    /// The membership rule, as an agent reads it: an entity a script parented
    /// under an additive scene's entity carries no tag of its own and is in
    /// that scene, not the active one. Reported wrong, an agent would edit or
    /// save it as the open scene's.
    #[test]
    fn an_entity_under_an_additive_scenes_entity_is_reported_in_that_scene() {
        use crate::scene::loaded;
        let mut registry = Registry::new();
        crate::scene::register_components(&mut registry);
        let mut world = World::default();
        crate::App::install_default_resources(&mut world);
        let scene = |name: &str, id: &str| {
            orrin_registry::parse(&format!(
                "orrin-scene 4 min-reader 4\n\nentity {id} \"{name}\"\n"
            ))
            .unwrap()
        };
        let floor = scene("Floor", "1163793b-df57-4e23-9844-00124ce2fce0");
        loaded::load(&mut world, &registry, &floor, "base", None);
        let lamp = scene("Lamp", "14758fa1-eba1-429c-965b-c7b647529bcd");
        let handle = world.resource_mut::<LoadedScenes>().reserve();
        loaded::load_additive(&mut world, &registry, handle, &lamp, "extra", None, false);
        let lamp = world
            .entities()
            .find(|&e| world.get::<Name>(e).is_some_and(|name| name.0 == "Lamp"))
            .unwrap();
        let shade = world.spawn();
        world.insert(shade, Name::new("Shade"));
        crate::scene::reparent(&mut world, shade, Some(lamp), false).unwrap();

        let mut state = EditorState::default();
        let outline = call(&mut world, &mut state, "scene_read", Value::Null).unwrap();
        let in_scene = |name: &str| {
            outline["entities"]
                .as_array()
                .unwrap()
                .iter()
                .find(|entity| entity["name"] == name)
                .map(|entity| entity["in_scene"].clone())
                .unwrap()
        };
        let active = world.resource::<LoadedScenes>().active().handle.get();
        assert_eq!(in_scene("Floor"), active);
        assert_eq!(in_scene("Lamp"), handle.get());
        assert_eq!(in_scene("Shade"), handle.get());

        let only = json!({"scene": handle.get()});
        let extra = call(&mut world, &mut state, "scene_read", only).unwrap();
        assert_eq!(extra["total"], 2);
    }
}
