//! `editor_command`: the transport, saving and opening, undo and redo, a
//! script reload, the selection and framing it — each raised as the same
//! `EditorState::request_*` its button raises, so Play and Stop still go
//! through the fork and a reload still lands at the frame's one safe point.
//!
//! **The answer says what happened, not what was asked.** A request is carried
//! out later — the UI pass applies most of them, the script phase of the next
//! frame a reload — so the command is [`Held`] and answered on a later
//! [`serve`](super::Remote::serve) once its effect can be seen: the play state
//! changed, the scene saved or opened, undo's step undone, the reload's
//! outcome known. One whose effect never shows is answered at its deadline
//! with what can be seen, and says so.
//!
//! **Scenes.** `new_scene` is the editor's own request, under the unsaved-work
//! rule `open_scene` keeps. `load_scene` and `unload_scene` act on a playing
//! game only — the editor opens one scene at a time, and additive scenes are a
//! running game's — and run at once, by the functions a script's
//! `SceneManager` call ends in: the remote is served after the frame's script
//! tick and outside any dispatch window, where a load may despawn script
//! instances, so they need not wait in a queue for the next tick as a
//! script's must.
//!
//! **Nothing waits on the developer.** A command that would open a dialog is
//! refused instead: opening a scene over unsaved work needs `save` or
//! `discard`, and saving a scene that has never had a file needs a `path`.
//! The UI pass is what carries a command out, so with the overlay off (F1)
//! one that needs it is refused rather than left to time out.

use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use orrin_ecs::World;
use orrin_registry::Registry;
use serde::Deserialize;
use serde_json::{Value as Json, json};

use super::answer;
use crate::editor::state::{DEFAULT_SCENE_PATH, EditorState, OpenChoice, SceneRequest, plain};
use crate::scene::loaded::{self, LoadedScenes, SceneHandle};
use crate::scene::{Camera, Diagnostics, LogBuffer, LogLevel, Play, PlayState, Time};
use orrin_remote::permissions::Access;

/// How long a held command may take before it is answered as timed out. In
/// time rather than frames, because what makes one late is a frame that stalls
/// — a large scene loads, and a reload runs, synchronously on the frame's
/// thread — and the frame count is lowest exactly then. Twenty seconds covers
/// a load of seconds with room to spare, and stays under the client's own
/// thirty (`orrin_remote::client`), so the editor's account of what it saw
/// reaches the client before the client gives up on it.
pub(super) const DEADLINE: Duration = Duration::from_secs(20);

#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    Play,
    Pause,
    Step,
    Stop,
    Save {
        /// Relative to the project, for a scene that has no file yet.
        path: Option<String>,
    },
    OpenScene {
        path: String,
        #[serde(default)]
        save: bool,
        #[serde(default)]
        discard: bool,
    },
    Undo,
    Redo,
    ReloadScripts,
    Select {
        ids: Vec<String>,
    },
    Frame,
    NewScene {
        name: String,
        #[serde(default)]
        save: bool,
        #[serde(default)]
        discard: bool,
    },
    LoadScene {
        path: String,
        #[serde(default)]
        additive: bool,
    },
    UnloadScene {
        handle: u32,
    },
}

/// A command raised and not yet seen to land.
pub(super) struct Held {
    wait: Wait,
    /// `EditorState::applied` when it was raised: the UI pass that carries
    /// it out moves it on.
    applied: u64,
    /// The console's next number when it was raised, so the answer can quote
    /// what the editor said about it.
    said_from: u64,
    raised: Instant,
}

enum Wait {
    Transport,
    Step { frame: u64 },
    Save,
    Open { path: String },
    Undo { label: String, version: u64 },
    Reload { count: u64 },
    Frame,
}

/// Run `params`' command: answered now when it is refused or immediate, held
/// otherwise.
pub(super) fn editor_command(
    world: &mut World,
    registry: &Registry,
    state: &mut EditorState,
    params: &Json,
) -> Result<Result<Json, Held>, String> {
    let command: Command = answer::params_of_required(params)?;
    let live = state.playing();
    let needs_ui = !matches!(
        command,
        Command::ReloadScripts
            | Command::Select { .. }
            | Command::LoadScene { .. }
            | Command::UnloadScene { .. }
    );
    if needs_ui
        && world
            .get_resource::<Diagnostics>()
            .is_some_and(|diagnostics| !diagnostics.overlay)
    {
        return Err(
            "the editor's overlay is off (F1), and its UI pass is what carries a command out; \
             ask the developer to turn it back on"
                .to_owned(),
        );
    }
    let said_from = world
        .get_resource::<LogBuffer>()
        .map_or(0, |log| log.next_number());
    let hold = |wait: Wait, state: &EditorState| {
        Ok(Err(Held {
            wait,
            applied: state.applied(),
            said_from,
            raised: Instant::now(),
        }))
    };
    match command {
        Command::Play if live => Err("the game is already playing".to_owned()),
        Command::Stop | Command::Pause | Command::Step if !live => {
            Err("the game is not playing; play it first".to_owned())
        }
        Command::Play | Command::Stop => {
            state.request_play();
            hold(Wait::Transport, state)
        }
        Command::Pause => {
            state.request_pause();
            hold(Wait::Transport, state)
        }
        Command::Step => {
            state.request_step();
            let frame = world
                .get_resource::<Time>()
                .map_or(0, |time| time.frame_count());
            hold(Wait::Step { frame }, state)
        }
        Command::Save { .. } | Command::OpenScene { .. } if live => Err(
            "the game is playing: the world is the game's, not the scene's. Stop it first"
                .to_owned(),
        ),
        Command::Save { path } => {
            if let Some(path) = path {
                let file = in_project(state, &path)?;
                writable(state, &file)?;
                if file.exists() && plain(file.display().to_string()) != state.scene_path {
                    return Err(format!(
                        "{path} already exists and is not the open scene; open it, or choose \
                         another path"
                    ));
                }
                state.scene_path = plain(file.display().to_string());
            } else if state.scene_path == DEFAULT_SCENE_PATH {
                return Err(
                    "the scene has never been saved, so it has no file; give a `path` relative \
                     to the project, ending .orrin"
                        .to_owned(),
                );
            } else {
                writable(state, Path::new(&state.scene_path))?;
            }
            state.request_scene(SceneRequest::Save);
            hold(Wait::Save, state)
        }
        Command::OpenScene {
            path,
            save,
            discard,
        } => {
            let file = in_project(state, &path)?;
            answer::permit(
                &state.agent_folders,
                state.project_root.as_deref(),
                &file,
                Access::Read,
            )?;
            if !file.is_file() {
                return Err(format!("the project has no scene at {path}"));
            }
            let file = plain(file.display().to_string());
            if state.scene_dirty() {
                match (save, discard) {
                    (true, true) => return Err("`save` or `discard`, not both".to_owned()),
                    (false, false) => {
                        return Err(
                            "the open scene has unsaved changes: pass `save: true` to keep them \
                             first, or `discard: true` to lose them"
                                .to_owned(),
                        );
                    }
                    (true, false) if state.scene_path == DEFAULT_SCENE_PATH => {
                        return Err(
                            "the open scene has never been saved, so it has no file to save to; \
                             save it with a `path` first, or pass `discard: true`"
                                .to_owned(),
                        );
                    }
                    // The answers the editor's own dialog gives, chosen here
                    // so it never opens.
                    (save, _) => {
                        if save {
                            writable(state, Path::new(&state.scene_path))?;
                        }
                        state.request_open_scene_checked(file.clone());
                        state.resolve_open(if save {
                            OpenChoice::SaveFirst
                        } else {
                            OpenChoice::Discard
                        });
                    }
                }
            } else {
                state.request_open_scene(file.clone());
            }
            hold(Wait::Open { path: file }, state)
        }
        Command::Undo | Command::Redo => {
            let undo = matches!(command, Command::Undo);
            let label = match undo {
                true => state.history.undo_label(),
                false => state.history.redo_label(),
            };
            let Some(label) = label.map(str::to_owned) else {
                return Err(format!(
                    "there is nothing to {}",
                    if undo { "undo" } else { "redo" }
                ));
            };
            let version = state.history.version();
            if undo {
                state.request_undo();
            } else {
                state.request_redo();
            }
            hold(Wait::Undo { label, version }, state)
        }
        Command::ReloadScripts => {
            reload(state).and_then(|count| hold(Wait::Reload { count }, state))
        }
        Command::NewScene { .. } if live => Err(
            "the game is playing: a new scene would replace the game's world. Stop it first"
                .to_owned(),
        ),
        Command::NewScene {
            name,
            save,
            discard,
        } => {
            if save {
                return Err(
                    "a new scene opens on the frame its request lands, before a save asked for \
                     with it could; save first (editor_command save), then ask for the new scene"
                        .to_owned(),
                );
            }
            if state.scene_dirty() && !discard {
                return Err(
                    "the open scene has unsaved changes: save it first, or pass `discard: true` \
                     to lose them"
                        .to_owned(),
                );
            }
            let file = crate::editor::state::new_scene_path(world, &name)?;
            writable(state, &file)?;
            state.request_new_scene(name);
            hold(
                Wait::Open {
                    path: plain(file.display().to_string()),
                },
                state,
            )
        }
        Command::LoadScene { .. } | Command::UnloadScene { .. } if !live => Err(
            "scenes load beside each other only while the game plays: the editor has one scene \
             open at a time. Open another with open_scene, or play first"
                .to_owned(),
        ),
        Command::LoadScene { path, additive } => {
            load_scene(world, registry, state, &path, additive).map(Ok)
        }
        Command::UnloadScene { handle } => {
            let handle =
                SceneHandle::from_raw(handle).ok_or_else(|| "no scene has handle 0".to_owned())?;
            loaded::unload(world, handle)?;
            state.selection.retain_alive(world);
            Ok(Ok(json!({"unloaded": handle.get()})))
        }
        Command::Select { ids } => {
            let entities = ids
                .iter()
                .map(|id| answer::find(world, id, answer::world_read(state)))
                .collect::<Result<Vec<_>, _>>()?;
            state.selection.clear();
            for entity in entities {
                if !state.selection.contains(entity) {
                    state.selection.toggle(entity);
                }
            }
            Ok(Ok(json!({"selection": answer::selection(world, state)})))
        }
        Command::Frame => {
            if state.selection.is_empty() {
                return Err("nothing is selected to frame; select something first".to_owned());
            }
            state.request_frame();
            hold(Wait::Frame, state)
        }
    }
}

#[cfg(feature = "scripting")]
fn reload(state: &mut EditorState) -> Result<u64, String> {
    let count = state.reloads().0;
    state.request_script_reload();
    Ok(count)
}

#[cfg(not(feature = "scripting"))]
fn reload(_: &mut EditorState) -> Result<u64, String> {
    Err("this editor was built without scripting, so there are no scripts to reload".to_owned())
}

/// Load the scene at `path` into the playing game, in place of the active one
/// or beside it, by the functions a script's `SceneManager` call ends in.
/// Done here and now rather than queued as a script's is: the remote is
/// served outside any dispatch window, after the frame's script tick, which
/// is where a load may despawn script instances.
fn load_scene(
    world: &mut World,
    registry: &Registry,
    state: &mut EditorState,
    path: &str,
    additive: bool,
) -> Result<Json, String> {
    let file = in_project(state, path)?;
    answer::permit(
        &state.agent_folders,
        state.project_root.as_deref(),
        &file,
        Access::Read,
    )?;
    let scene = loaded::asset_at(world, &file)
        .ok_or_else(|| format!("{path} is not a scene asset the project indexes"))?;
    let (document, name) = loaded::read(world, scene)?;
    let (handle, issues) = if additive {
        let handle = world.resource_mut::<LoadedScenes>().reserve();
        let issues = loaded::load_additive(
            world,
            registry,
            handle,
            &document,
            name.clone(),
            Some(scene),
            false,
        );
        (handle, issues)
    } else {
        let issues = loaded::load(world, registry, &document, name.clone(), Some(scene));
        (world.resource::<LoadedScenes>().active().handle, issues)
    };
    state.selection.retain_alive(world);
    let issues: Vec<String> = issues.iter().map(ToString::to_string).collect();
    Ok(json!({
        "loaded": name,
        "handle": handle.get(),
        "additive": additive,
        "issues": issues,
    }))
}

/// Whether the folder rules let an agent write `file`: what a save of it
/// needs, the open scene's own included.
fn writable(state: &EditorState, file: &Path) -> Result<(), String> {
    answer::permit(
        &state.agent_folders,
        state.project_root.as_deref(),
        file,
        Access::ReadWrite,
    )
}

/// `path`, relative to the project and inside it, as a scene file.
fn in_project(state: &EditorState, path: &str) -> Result<PathBuf, String> {
    let root = state
        .project_root
        .as_deref()
        .ok_or_else(|| "no project is open".to_owned())?;
    let relative = Path::new(path);
    let inside = relative
        .components()
        .all(|part| matches!(part, Component::Normal(_) | Component::CurDir));
    if !inside || relative.as_os_str().is_empty() {
        return Err(format!(
            "`{path}` is not a path inside the project: give one relative to it, with no `..`"
        ));
    }
    if relative
        .extension()
        .is_none_or(|ext| ext != crate::scene::loaded::EXTENSION)
    {
        return Err(format!("`{path}` is not a scene: a scene file ends .orrin"));
    }
    Ok(root.join(relative))
}

impl Held {
    /// The answer, once the command's effect can be seen; `None` until then.
    pub(super) fn poll(&self, world: &World, state: &EditorState) -> Option<Result<Json, String>> {
        let carried = state.applied() > self.applied;
        match &self.wait {
            Wait::Transport | Wait::Save | Wait::Open { .. } | Wait::Undo { .. } | Wait::Frame
                if !carried =>
            {
                None
            }
            Wait::Step { frame } => {
                let now = world
                    .get_resource::<Time>()
                    .map_or(0, |time| time.frame_count());
                (carried && now > frame + 1).then(|| Ok(self.play(world)))
            }
            Wait::Reload { count } => {
                let (now, last) = state.reloads();
                (now > *count).then(|| match last {
                    Some((true, message)) => Ok(json!({"swapped": true, "outcome": message})),
                    Some((false, message)) => Err(message.clone()),
                    None => Err("the reload left no outcome".to_owned()),
                })
            }
            _ => Some(self.landed(world, state)),
        }
    }

    /// The answer to a command the UI pass has carried out.
    fn landed(&self, world: &World, state: &EditorState) -> Result<Json, String> {
        let said = self.said(world);
        let failed = |what: &str| {
            Err(match said.is_empty() {
                true => what.to_owned(),
                false => format!("{what}: {}", said.join("; ")),
            })
        };
        match &self.wait {
            Wait::Transport => Ok(self.play(world)),
            _ if self.went_wrong(world) => failed("it failed"),
            Wait::Save if state.scene_dirty() => failed("the scene was not saved"),
            Wait::Save => Ok(json!({"saved": state.scene_path, "said": said})),
            Wait::Open { path } if state.scene_path != *path || state.scene_dirty() => {
                failed("the scene was not opened")
            }
            Wait::Open { path } => Ok(json!({
                "opened": path,
                "entities": world.entities().count(),
                "said": said,
            })),
            Wait::Undo { version, .. } if state.history.version() == *version => {
                failed("nothing changed")
            }
            Wait::Undo { label, .. } => Ok(json!({"step": label, "unsaved": state.scene_dirty()})),
            Wait::Frame => {
                let camera = world.get_resource::<Camera>().map(|camera| {
                    json!({
                        "position": camera.position.to_array(),
                        "target": camera.target.to_array(),
                    })
                });
                Ok(json!({"camera": camera}))
            }
            Wait::Step { .. } | Wait::Reload { .. } => Ok(self.play(world)),
        }
    }

    /// Whether it has waited past [`DEADLINE`].
    pub(super) fn overdue(&self) -> bool {
        self.raised.elapsed() >= DEADLINE
    }

    /// The answer at the deadline: what can be seen, and that it timed out.
    pub(super) fn expired(&self, world: &World) -> String {
        let said = self.said(world);
        let mut answer = format!(
            "the command was raised but no effect was seen within {} s; play is {}",
            DEADLINE.as_secs(),
            self.play(world)["play"]
        );
        if !said.is_empty() {
            answer.push_str(&format!("; the editor said: {}", said.join("; ")));
        }
        answer
    }

    fn play(&self, world: &World) -> Json {
        let play = match world.get_resource::<Play>().map(|play| play.state()) {
            Some(PlayState::Playing) => "playing",
            Some(PlayState::Paused) => "paused",
            Some(PlayState::Editing) | None => "editing",
        };
        let frame = world.get_resource::<Time>().map(|time| time.frame_count());
        json!({"play": play, "frame": frame})
    }

    /// Whether the editor logged an error since the command was raised: a
    /// save that could not write, a scene that did not parse.
    fn went_wrong(&self, world: &World) -> bool {
        world.get_resource::<LogBuffer>().is_some_and(|log| {
            log.since(self.said_from)
                .any(|(_, entry)| entry.level >= LogLevel::Error)
        })
    }

    /// What the editor itself logged since the command was raised: the save's
    /// line, the load's, a refusal, and any warning.
    fn said(&self, world: &World) -> Vec<String> {
        let Some(log) = world.get_resource::<LogBuffer>() else {
            return Vec::new();
        };
        log.since(self.said_from)
            .filter(|(_, entry)| entry.source.is_empty() || entry.level >= LogLevel::Warning)
            .take(10)
            .map(|(_, entry)| entry.text())
            .collect()
    }
}
