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
//! **Nothing waits on the developer.** A command that would open a dialog is
//! refused instead: opening a scene over unsaved work needs `save` or
//! `discard`, and saving a scene that has never had a file needs a `path`.
//! The UI pass is what carries a command out, so with the overlay off (F1)
//! one that needs it is refused rather than left to time out.

use std::path::{Component, Path, PathBuf};

use orrin_ecs::World;
use serde::Deserialize;
use serde_json::{Value as Json, json};

use super::answer;
use crate::editor::state::{DEFAULT_SCENE_PATH, EditorState, OpenChoice, SceneRequest, plain};
use crate::scene::{Camera, Diagnostics, LogBuffer, LogLevel, Play, PlayState, Time};

/// How many frames a held command may take before it is answered as timed
/// out. Everything here lands within two or three; the rest is room for a
/// frame that stalls, and at a thousand frames a second it is still most of a
/// second.
pub(super) const DEADLINE: u32 = 600;

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
    pub(super) frames: u32,
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
    world: &World,
    state: &mut EditorState,
    params: &Json,
) -> Result<Result<Json, Held>, String> {
    let command: Command = answer::params_of_required(params)?;
    let live = state.playing();
    let needs_ui = !matches!(command, Command::ReloadScripts | Command::Select { .. });
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
    let hold = |wait: Wait, state: &EditorState| {
        Ok(Err(Held {
            wait,
            applied: state.applied(),
            said_from: world
                .get_resource::<LogBuffer>()
                .map_or(0, |log| log.next_number()),
            frames: 0,
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

    /// The answer at the deadline: what can be seen, and that it timed out.
    pub(super) fn expired(&self, world: &World) -> String {
        let said = self.said(world);
        let mut answer = format!(
            "the command was raised but no effect was seen within {DEADLINE} frames; play is \
             {}",
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
