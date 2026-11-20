//! `scene_edit`: an agent's change to the open scene, as one labelled batch
//! through [`SceneEditor`] — one step in the Edit menu, applied whole or not
//! at all ([`SceneEditor::batch_whole`] puts back a batch that fails part-way).
//! There is no other way the remote changes a scene.
//!
//! **Values are text in the scene grammar**, read by the scene parser: a
//! component is written exactly as an entity block holds it — the id, then
//! its value or its fields one level in — which is the form `component_list`
//! and `scene_read` print. A `set` replaces the fields it names and leaves the
//! rest as they are; a component that is one value, or an enum of another
//! variant, is replaced whole. A component added or spawned takes the fields
//! it names over the registry's default, so a transform can be given by its
//! translation alone.
//!
//! **The open scene only.** A spawned entity is in its parent's scene by the
//! membership rule (`scene::loaded`), so a parent in any other loaded scene is
//! refused with that reason; in edit mode the open scene is the only one
//! loaded, and the case cannot arise.
//!
//! **Refused, not merged or split:** an edit while the game plays, since that
//! world is a fork Stop throws away; and an edit while the developer is mid
//! gesture — a drag, or a field being typed into — since a batch closes the
//! open gesture first, which would cut theirs in two. That one is answered as
//! busy rather than held, because the gesture can last as long as a hand on
//! the mouse does.

use orrin_ecs::{Entity, World};
use orrin_registry::{AssetId, ComponentId, EntityId, Registry, Value};
use serde::Deserialize;
use serde_json::{Map, Value as Json, json};

use super::answer;
use crate::asset_db::AssetDatabase;
use crate::editor::state::EditorState;
use crate::scene::edit::{EditError, SceneEditor, ensure_id};
use crate::scene::{Name, persist};

/// What an agent's step is called in the Edit menu, so the developer can
/// tell it from their own.
const PREFIX: &str = "Agent: ";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Edit {
    label: String,
    ops: Vec<EditOp>,
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum EditOp {
    /// One or more components' fields, as `text`.
    Set {
        entity: String,
        text: String,
    },
    AddComponent {
        entity: String,
        component: String,
        /// The component as text; its registered default when absent.
        text: Option<String>,
    },
    RemoveComponent {
        entity: String,
        component: String,
    },
    Spawn {
        /// An id of the caller's choosing, for later ops in the batch to name.
        id: Option<String>,
        name: Option<String>,
        parent: Option<String>,
        /// Its components as text; none is an entity with nothing but a name.
        text: Option<String>,
        /// A prefab or mesh asset, by id or path, to place instead.
        prefab: Option<String>,
    },
    Despawn {
        entities: Vec<String>,
    },
    Reparent {
        entity: String,
        parent: Option<String>,
        #[serde(default = "keep")]
        keep_world: bool,
    },
}

fn keep() -> bool {
    true
}

impl EditOp {
    fn name(&self) -> &'static str {
        match self {
            Self::Set { .. } => "set",
            Self::AddComponent { .. } => "add_component",
            Self::RemoveComponent { .. } => "remove_component",
            Self::Spawn { .. } => "spawn",
            Self::Despawn { .. } => "despawn",
            Self::Reparent { .. } => "reparent",
        }
    }
}

/// What a batch did that its caller cannot work out alone.
#[derive(Default)]
struct Done {
    spawned: Vec<String>,
    /// Each `#slot` an op named, and the id that op gave it.
    named: Vec<(String, Entity)>,
}

pub(super) fn scene_edit(
    world: &mut World,
    registry: &Registry,
    state: &mut EditorState,
    params: &Json,
) -> Result<Json, String> {
    let edit: Edit = answer::params_of_required(params)?;
    if edit.label.trim().is_empty() {
        return Err("give the batch a `label`: it names the step in the Edit menu".to_owned());
    }
    if state.playing() {
        return Err(
            "the game is playing: the world is a fork that Stop throws away, so an edit to it \
             would be lost. Stop it first (editor_command stop)"
                .to_owned(),
        );
    }
    if state.history.in_gesture() {
        return Err(
            "the editor is busy: the developer is part-way through a change of their own (a drag, \
             or a field being typed into). Try again shortly"
                .to_owned(),
        );
    }
    let label = format!("{PREFIX}{}", edit.label.trim());
    let count = edit.ops.len();
    let mut done = Done::default();
    let folders = Folders {
        rules: &state.agent_folders,
        assets: &state.agent_assets,
    };
    let mut editor = SceneEditor::new(world, registry, &mut state.history);
    editor
        .batch_whole(label.clone(), |editor| {
            for (index, op) in edit.ops.iter().enumerate() {
                apply(editor, registry, &folders, op, &mut done).map_err(|why| {
                    EditError(format!(
                        "op {index} ({}) refused, so nothing in the batch was applied: {why}",
                        op.name()
                    ))
                })?;
            }
            Ok(())
        })
        .map_err(|error| error.to_string())?;
    let mut ids = Map::new();
    for (slot, entity) in done.named {
        if let Some(id) = world.get::<EntityId>(entity) {
            ids.insert(slot, Json::String(id.to_string()));
        }
    }
    state.selection.retain_alive(world);
    Ok(json!({
        "label": label,
        "applied": count,
        "spawned": done.spawned,
        "ids": ids,
        "unsaved": state.scene_dirty(),
    }))
}

/// The folder rules, for the one op that reads a project path: placing a
/// prefab.
struct Folders<'a> {
    rules: &'a Result<orrin_remote::permissions::Folders, String>,
    /// The assets folder relative to the project root.
    assets: &'a str,
}

fn apply(
    editor: &mut SceneEditor,
    registry: &Registry,
    folders: &Folders,
    op: &EditOp,
    done: &mut Done,
) -> Result<(), String> {
    let named = &mut done.named;
    let mut find = |editor: &SceneEditor, text: &str| -> Result<Entity, String> {
        let entity = answer::find(editor.world(), text, "edit")?;
        if text.starts_with('#') {
            named.push((text.to_owned(), entity));
        }
        Ok(entity)
    };
    match op {
        EditOp::Set { entity, text } => {
            let target = find(editor, entity)?;
            for (component, given) in components(registry, text)? {
                let current = persist::read_component(editor.world(), registry, target, &component)
                    .ok_or_else(|| {
                        format!("{entity} has no {component}; add it with add_component")
                    })?;
                let value = merged(&component, current, given)?;
                editor
                    .set(target, &component, value)
                    .map_err(|error| error.to_string())?;
            }
            Ok(())
        }
        EditOp::AddComponent {
            entity,
            component,
            text,
        } => {
            let target = find(editor, entity)?;
            let id = ComponentId::owned(component.clone());
            let vtable = registry.get(&id).ok_or_else(|| unregistered(&id))?;
            let value = match text {
                None => vtable.default(),
                Some(text) => {
                    let mut given = components(registry, text)?;
                    match (given.pop(), given.is_empty()) {
                        (Some((named, value)), true) if named == id => {
                            merged(&id, vtable.default(), value)?
                        }
                        _ => {
                            return Err(format!(
                                "`text` must hold {component} alone, written as \
                                 component_list prints it"
                            ));
                        }
                    }
                }
            };
            editor
                .add_component(target, &id, value)
                .map_err(|error| format!("{entity}: {error}"))
        }
        EditOp::RemoveComponent { entity, component } => {
            let target = find(editor, entity)?;
            let id = ComponentId::owned(component.clone());
            editor
                .remove_component(target, &id)
                .map_err(|error| format!("{entity}: {error}"))
        }
        EditOp::Spawn {
            id,
            name,
            parent,
            text,
            prefab,
        } => {
            let parent = match parent {
                Some(text) => {
                    let parent = find(editor, text)?;
                    in_active_scene(editor.world(), parent, text)?;
                    Some(parent)
                }
                None => None,
            };
            let root = match prefab {
                Some(prefab) => {
                    if id.is_some() || text.is_some() {
                        return Err(
                            "a placed prefab has the ids and components its file gives it, so \
                             `id` and `text` go with a plain spawn only"
                                .to_owned(),
                        );
                    }
                    place(editor, folders, prefab, name.as_deref(), parent)?
                }
                None => spawn(
                    editor,
                    registry,
                    id.as_deref(),
                    name.as_deref(),
                    parent,
                    text,
                )?,
            };
            if let Some(id) = editor.world().get::<EntityId>(root) {
                done.spawned.push(id.to_string());
            }
            Ok(())
        }
        EditOp::Despawn { entities } => {
            let roots = entities
                .iter()
                .map(|entity| find(editor, entity))
                .collect::<Result<Vec<_>, _>>()?;
            editor.despawn(&roots).map_err(|error| error.to_string())
        }
        EditOp::Reparent {
            entity,
            parent,
            keep_world,
        } => {
            let child = find(editor, entity)?;
            let parent = match parent {
                Some(text) => {
                    let parent = find(editor, text)?;
                    in_active_scene(editor.world(), parent, text)?;
                    Some(parent)
                }
                None => None,
            };
            if *keep_world {
                editor.propagate();
            }
            editor
                .reparent(child, parent, *keep_world)
                .map_err(|error| error.to_string())
        }
    }
}

/// Whether `entity` is in the active scene, the one an edit changes. Edit
/// mode has that scene alone loaded, so this refuses nothing today; it is
/// what keeps a child out of a scene no save of the open scene would write,
/// should another ever be loaded while editing — the membership rule would
/// otherwise put the child in its parent's scene.
fn in_active_scene(world: &orrin_ecs::World, entity: Entity, text: &str) -> Result<(), String> {
    use crate::scene::loaded::{LoadedScenes, scene_of};
    let Some(loaded) = world.get_resource::<LoadedScenes>() else {
        return Ok(());
    };
    let scene = scene_of(world, entity);
    if scene == loaded.active().handle {
        return Ok(());
    }
    let name = loaded.get(scene).map_or("", |scene| scene.name.as_str());
    Err(format!(
        "{text} is in the loaded scene `{name}` (handle {}), not the open one; a child is in \
         its parent's scene, and an edit changes the open scene only",
        scene.get()
    ))
}

fn unregistered(id: &ComponentId) -> String {
    format!("no component `{id}` is registered; component_list names those there are")
}

/// `text` as the components it writes, each one the registry knows. The
/// parser numbers lines from the top of what it was given, which is two lines
/// of header above the caller's first.
pub(super) fn components(
    registry: &Registry,
    text: &str,
) -> Result<Vec<(ComponentId, Value)>, String> {
    const KIND: &str = "orrin-components";
    let document = format!("{KIND} 1 min-reader 1\n\n{text}");
    let fields = orrin_registry::parse_value_document(&document, KIND, 1).map_err(|error| {
        format!(
            "`text` line {}: {}",
            error.line.saturating_sub(2),
            error.message
        )
    })?;
    fields
        .into_iter()
        .map(|(id, value)| {
            let id = ComponentId::owned(id);
            if id == crate::scene::registry::PARENT {
                return Err(
                    "name a parent with `parent` or a reparent op, not as a component".to_owned(),
                );
            }
            registry.get(&id).ok_or_else(|| unregistered(&id))?;
            Ok((id, value))
        })
        .collect()
}

/// `given` over `current`: the fields it names replace theirs, and a value
/// that is not a set of fields — or an enum of another variant — replaces it
/// whole.
fn merged(component: &ComponentId, current: Value, given: Value) -> Result<Value, String> {
    let fill = |mut fields: Vec<(String, Value)>, given: Vec<(String, Value)>| {
        for (name, value) in given {
            let Some(slot) = fields.iter_mut().find(|(have, _)| *have == name) else {
                let known: Vec<&str> = fields.iter().map(|(name, _)| name.as_str()).collect();
                return Err(format!(
                    "{component} has no field `{name}`; its fields are {}",
                    known.join(", ")
                ));
            };
            slot.1 = value;
        }
        Ok(fields)
    };
    match (current, given) {
        (Value::Struct(fields), Value::Struct(given)) => fill(fields, given).map(Value::Struct),
        (
            Value::Enum { variant, fields },
            Value::Enum {
                variant: asked,
                fields: given,
            },
        ) if variant == asked => fill(fields, given).map(|fields| Value::Enum { variant, fields }),
        (_, given) => Ok(given),
    }
}

/// A new entity holding exactly what it was given, recorded as spawned.
fn spawn(
    editor: &mut SceneEditor,
    registry: &Registry,
    id: Option<&str>,
    name: Option<&str>,
    parent: Option<Entity>,
    text: &Option<String>,
) -> Result<Entity, String> {
    let id = match id {
        Some(id) => {
            let id: EntityId = id
                .parse()
                .map_err(|_| format!("`{id}` is not an entity id (a UUID)"))?;
            if answer::find(editor.world(), &id.to_string(), "edit").is_ok() {
                return Err(format!("an entity with id {id} is already in the scene"));
            }
            id
        }
        None => EntityId::new(),
    };
    let components = match text {
        Some(text) => components(registry, text)?
            .into_iter()
            .map(|(id, given)| {
                let default = registry.get(&id).map(|vtable| vtable.default());
                let value = match default {
                    Some(default) => merged(&id, default, given)?,
                    None => given,
                };
                Ok((id, value))
            })
            .collect::<Result<Vec<_>, String>>()?,
        None => Vec::new(),
    };
    if name.is_some()
        && components
            .iter()
            .any(|(c, _)| *c == crate::scene::registry::NAME)
    {
        return Err("name it with `name` or with orrin.name in `text`, not both".to_owned());
    }
    let label = name.unwrap_or("entity").to_owned();
    editor.replace(format!("Create {label}"), &[], |world, registry| {
        // A parent without an id would be recorded as no parent at all, and
        // an undo then a redo would bring the child back as a root.
        if let Some(parent) = parent {
            ensure_id(world, parent);
        }
        let entity = world.spawn();
        world.insert(entity, id);
        if let Some(name) = name {
            world.insert(entity, Name::new(name));
        }
        let mut made = Ok(entity);
        for (component, value) in &components {
            if let Err(error) = persist::write_component(world, registry, entity, component, value)
            {
                made = Err(format!("{component}: {error}"));
                break;
            }
        }
        if made.is_ok()
            && let Some(parent) = parent
            && let Err(error) = crate::scene::reparent(world, entity, Some(parent), false)
        {
            made = Err(format!("parenting it: {error}"));
        }
        (made, vec![entity])
    })
}

/// A prefab, or a mesh, by asset id or path, placed as the editor's own
/// Place does.
fn place(
    editor: &mut SceneEditor,
    folders: &Folders,
    asset: &str,
    name: Option<&str>,
    parent: Option<Entity>,
) -> Result<Entity, String> {
    let (id, path) = {
        let db = editor
            .world()
            .get_resource::<AssetDatabase>()
            .ok_or_else(|| "no project is open, so there are no prefabs".to_owned())?;
        let entry = match asset.parse::<AssetId>() {
            Ok(id) => db.get(id),
            Err(_) => db.find(asset),
        }
        .ok_or_else(|| format!("the project has no asset at `{asset}`"))?;
        (entry.id, entry.path.clone())
    };
    let in_project = match folders.assets {
        "" => path,
        assets => format!("{assets}/{path}"),
    };
    super::answer::permit(
        folders.rules,
        None,
        std::path::Path::new(&in_project),
        orrin_remote::permissions::Access::Read,
    )?;
    editor.replace(format!("Place {asset}"), &[], |world, registry| {
        match crate::editor::state::place(world, registry, id, parent, None) {
            Ok(root) => {
                if let Some(name) = name {
                    world.insert(root, Name::new(name));
                }
                (Ok(root), vec![root])
            }
            Err(error) => (Err(error), Vec::new()),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCENE: &str = "orrin-scene 4 min-reader 4

entity 1163793b-df57-4e23-9844-00124ce2fce0 \"Ball\"
  orrin.transform
    rotation = (0.0, 0.0, 0.0, 1.0)
    scale = (1.0, 1.0, 1.0)
    translation = (1.0, 0.0, 0.0)

entity 14758fa1-eba1-429c-965b-c7b647529bcd \"Crate\"
  orrin.transform
    rotation = (0.0, 0.0, 0.0, 1.0)
    scale = (2.0, 2.0, 2.0)
    translation = (0.0, 0.0, -3.0)
";

    const BALL: &str = "1163793b-df57-4e23-9844-00124ce2fce0";
    const CRATE: &str = "14758fa1-eba1-429c-965b-c7b647529bcd";
    const LAMP: &str = "5a1d9a43-0000-4000-8000-00000000a001";

    struct Session {
        world: World,
        registry: Registry,
        state: EditorState,
    }

    impl Session {
        fn new() -> Self {
            let mut world = World::default();
            crate::App::install_default_resources(&mut world);
            let mut registry = Registry::new();
            crate::scene::register_components(&mut registry);
            crate::scene::load(SCENE, &mut world, &registry).unwrap();
            let mut session = Self {
                world,
                registry,
                state: EditorState::default(),
            };
            assert_eq!(
                session.saved(),
                SCENE,
                "the scene round-trips before any edit"
            );
            session
        }

        fn saved(&mut self) -> String {
            crate::scene::save(&mut self.world, &self.registry)
        }

        fn edit(&mut self, params: Json) -> Result<Json, String> {
            scene_edit(&mut self.world, &self.registry, &mut self.state, &params)
        }
    }

    /// Every kind of op once: two entities spawned, one under the other, a
    /// field set, a component added and another taken away, a reparent and a
    /// despawn.
    fn every_op() -> Vec<Json> {
        vec![
            json!({"op": "spawn", "id": LAMP, "name": "Lamp",
                   "text": "orrin.transform\n  translation = (0.0, 2.0, 0.0)"}),
            json!({"op": "spawn", "name": "Shade", "parent": LAMP}),
            json!({"op": "set", "entity": BALL,
                   "text": "orrin.transform\n  translation = (4.0, 5.0, 6.0)"}),
            json!({"op": "add_component", "entity": BALL, "component": "orrin.spin"}),
            json!({"op": "remove_component", "entity": CRATE, "component": "orrin.transform"}),
            json!({"op": "reparent", "entity": BALL, "parent": LAMP}),
            json!({"op": "set", "entity": BALL, "text": "orrin.name = \"Kept ball\""}),
            json!({"op": "despawn", "entities": [CRATE]}),
        ]
    }

    #[test]
    fn a_batch_then_its_undo_leaves_what_a_save_writes_as_it_was() {
        let mut session = Session::new();
        let done = session
            .edit(json!({"label": "Light the room", "ops": every_op()}))
            .unwrap();
        assert_eq!(done["applied"], 8);
        assert_eq!(done["label"], "Agent: Light the room");
        let edited = session.saved();
        assert_ne!(edited, SCENE);
        assert!(edited.contains("\"Shade\"") && !edited.contains("\"Crate\""));
        assert!(
            edited.contains("\"Kept ball\"\n  orrin.parent")
                && edited.contains("translation = (4.0, 3.0, 6.0)"),
            "moved under the lamp where it stood: {edited}"
        );
        assert_eq!(
            session.state.history.undo_label(),
            Some("Agent: Light the room"),
            "one step"
        );

        session.state.request_undo();
        session.state.apply(&mut session.world, &session.registry);
        assert_eq!(session.saved(), SCENE);
        assert_eq!(session.state.history.undo_label(), None);
    }

    #[test]
    fn a_batch_whose_last_op_is_refused_changes_nothing_and_names_it() {
        let mut session = Session::new();
        let mut ops = every_op();
        ops.push(json!({"op": "set", "entity": BALL,
                        "text": "orrin.transform\n  colour = (1.0, 0.0, 0.0)"}));
        let refused = session
            .edit(json!({"label": "Light the room", "ops": ops}))
            .unwrap_err();
        assert!(
            refused.starts_with("op 8 (set) refused, so nothing in the batch was applied:")
                && refused.contains("orrin.transform has no field `colour`"),
            "{refused}"
        );
        assert_eq!(session.saved(), SCENE);
        assert_eq!(session.state.history.undo_label(), None, "no step");
    }

    #[test]
    fn an_edit_while_the_game_plays_is_refused() {
        let mut session = Session::new();
        session.state.request_play();
        session.state.apply(&mut session.world, &session.registry);
        assert!(session.state.playing());
        let refused = session
            .edit(json!({"label": "Nudge", "ops": [
                {"op": "set", "entity": BALL, "text": "orrin.transform\n  translation = (9.0, 9.0, 9.0)"}
            ]}))
            .unwrap_err();
        assert!(refused.starts_with("the game is playing"), "{refused}");
    }
}
