//! What an agent may do through Orrin's tools: which kinds of thing
//! ([`Capabilities`], one switch each, chosen by [`Preset`]) and where in the
//! project ([`Folders`]). One definition, read by the editor — which checks
//! every live method at one point in its dispatch — and by `orrin mcp`, which
//! checks its offline tools against the same files. A refusal names what is
//! off and says where to turn it on, and nothing is hidden: every tool stays
//! listed and is refused, so an agent can learn the rules from the refusal or,
//! before it tries, from `project_info` and `editor_state`.
//!
//! **What this does not do.** It governs Orrin's tools — the remote's methods
//! and `orrin mcp`'s own. It is not a sandbox: an agent that can already run a
//! shell or write files by other means is not stopped by any rule here, and a
//! folder rule does not stop the build watcher reloading what such an agent
//! wrote. It is the developer saying what Orrin will do on an agent's behalf.
//!
//! **Where it lives.** The capabilities follow the person, so they are the
//! `[agents.allow]` table of the user's `settings.toml` beside `[agents]
//! remote` ([`USER_TABLE`]); the editor writes that table and this reads it.
//! Folder rules name a project's folders, so they are per project, in its
//! editor directory ([`FOLDERS`]) — never in `orrin.toml`, which is shared
//! and checked in, where one person's rules for their agent do not belong.
//! Both are read at the request, so a change governs the very next one.

use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// The table of the user's `settings.toml` that holds [`Capabilities`].
pub const USER_TABLE: [&str; 2] = ["agents", "allow"];

/// The file in a project's editor directory that holds its [`Folders`].
pub const FOLDERS: &str = "agent-folders.toml";

/// Where both are changed, as a refusal says.
const WHERE: &str = "Settings ▸ Permissions";

/// One kind of thing an agent may be refused, at the grain a person would
/// want to deny.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    ReadScene,
    Console,
    Screenshot,
    FrameStats,
    EditScene,
    PlayControl,
    GameInput,
    UndoRedo,
    Selection,
    Scenes,
    ReloadScripts,
    BuildScripts,
    RunTests,
    OpenEditor,
}

impl Capability {
    pub const ALL: [Self; 14] = [
        Self::ReadScene,
        Self::Console,
        Self::Screenshot,
        Self::FrameStats,
        Self::EditScene,
        Self::PlayControl,
        Self::GameInput,
        Self::UndoRedo,
        Self::Selection,
        Self::Scenes,
        Self::ReloadScripts,
        Self::BuildScripts,
        Self::RunTests,
        Self::OpenEditor,
    ];

    /// Its key in `[agents.allow]`, and in what the tools report.
    pub fn key(self) -> &'static str {
        match self {
            Self::ReadScene => "read_scene",
            Self::Console => "console",
            Self::Screenshot => "screenshot",
            Self::FrameStats => "frame_stats",
            Self::EditScene => "edit_scene",
            Self::PlayControl => "play_control",
            Self::GameInput => "game_input",
            Self::UndoRedo => "undo_redo",
            Self::Selection => "selection",
            Self::Scenes => "scenes",
            Self::ReloadScripts => "reload_scripts",
            Self::BuildScripts => "build_scripts",
            Self::RunTests => "run_tests",
            Self::OpenEditor => "open_editor",
        }
    }

    /// What Settings ▸ Permissions calls it.
    pub fn label(self) -> &'static str {
        match self {
            Self::ReadScene => "Read scene and assets",
            Self::Console => "Read the console",
            Self::Screenshot => "Screenshots",
            Self::FrameStats => "Frame stats",
            Self::EditScene => "Edit the scene",
            Self::PlayControl => "Play, pause and stop",
            Self::GameInput => "Send game input",
            Self::UndoRedo => "Undo and redo",
            Self::Selection => "Select and frame",
            Self::Scenes => "Save, open and create scenes",
            Self::ReloadScripts => "Reload scripts",
            Self::BuildScripts => "Build scripts",
            Self::RunTests => "Run tests",
            Self::OpenEditor => "Open the editor",
        }
    }
}

/// One switch per [`Capability`], every one on by default: Full control,
/// which is what the tools did before there were permissions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Capabilities {
    pub read_scene: bool,
    pub console: bool,
    pub screenshot: bool,
    pub frame_stats: bool,
    pub edit_scene: bool,
    pub play_control: bool,
    pub game_input: bool,
    pub undo_redo: bool,
    pub selection: bool,
    pub scenes: bool,
    pub reload_scripts: bool,
    pub build_scripts: bool,
    pub run_tests: bool,
    pub open_editor: bool,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self::of(Preset::FullControl)
    }
}

/// A named set of [`Capabilities`]. Custom is not chosen: it is what the
/// switches are when they match no other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    ReadOnly,
    EditScene,
    FullControl,
    Custom,
}

impl Preset {
    pub const CHOSEN: [Self; 3] = [Self::ReadOnly, Self::EditScene, Self::FullControl];

    pub fn label(self) -> &'static str {
        match self {
            Self::ReadOnly => "Read only",
            Self::EditScene => "Edit scene",
            Self::FullControl => "Full control",
            Self::Custom => "Custom",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Self::ReadOnly => "read_only",
            Self::EditScene => "edit_scene",
            Self::FullControl => "full_control",
            Self::Custom => "custom",
        }
    }

    fn allows(self, capability: Capability) -> bool {
        use Capability::*;
        let read = matches!(capability, ReadScene | Console | Screenshot | FrameStats);
        let edit = matches!(
            capability,
            EditScene | PlayControl | GameInput | UndoRedo | Selection
        );
        match self {
            Self::ReadOnly => read,
            Self::EditScene => read || edit,
            Self::FullControl | Self::Custom => true,
        }
    }
}

impl Capabilities {
    pub fn of(preset: Preset) -> Self {
        let mut capabilities = Self {
            read_scene: false,
            console: false,
            screenshot: false,
            frame_stats: false,
            edit_scene: false,
            play_control: false,
            game_input: false,
            undo_redo: false,
            selection: false,
            scenes: false,
            reload_scripts: false,
            build_scripts: false,
            run_tests: false,
            open_editor: false,
        };
        for capability in Capability::ALL {
            *capability_mut(&mut capabilities, capability) = preset.allows(capability);
        }
        capabilities
    }

    pub fn allows(&self, capability: Capability) -> bool {
        let mut copy = *self;
        *capability_mut(&mut copy, capability)
    }

    pub fn set(&mut self, capability: Capability, on: bool) {
        *capability_mut(self, capability) = on;
    }

    /// The preset these switches are, or Custom.
    pub fn preset(&self) -> Preset {
        Preset::CHOSEN
            .into_iter()
            .find(|&preset| Self::of(preset) == *self)
            .unwrap_or(Preset::Custom)
    }

    /// Whether `method`, with `params`, may run; why not, naming the switch
    /// and where it is, when it may not.
    pub fn check(&self, method: &str, params: &Value) -> Result<(), String> {
        match required(method, params) {
            Some(capability) if !self.allows(capability) => Err(format!(
                "`{method}` needs \"{}\", which is off for agents (preset: {}). Turn it on in \
                 {WHERE}",
                capability.label(),
                self.preset().label()
            )),
            _ => Ok(()),
        }
    }

    /// The capabilities in the user directory `user_dir`, as the editor last
    /// saved them: the defaults when there is no file or no table, and when
    /// the file does not read — which is what the editor itself runs on then.
    pub fn of_user(user_dir: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(user_dir.join("settings.toml")) else {
            return Self::default();
        };
        let Ok(table) = text.parse::<toml::Table>() else {
            return Self::default();
        };
        let [agents, allow] = USER_TABLE;
        table
            .get(agents)
            .and_then(|agents| agents.get(allow))
            .and_then(|allow| allow.clone().try_into().ok())
            .unwrap_or_default()
    }
}

fn capability_mut(capabilities: &mut Capabilities, capability: Capability) -> &mut bool {
    match capability {
        Capability::ReadScene => &mut capabilities.read_scene,
        Capability::Console => &mut capabilities.console,
        Capability::Screenshot => &mut capabilities.screenshot,
        Capability::FrameStats => &mut capabilities.frame_stats,
        Capability::EditScene => &mut capabilities.edit_scene,
        Capability::PlayControl => &mut capabilities.play_control,
        Capability::GameInput => &mut capabilities.game_input,
        Capability::UndoRedo => &mut capabilities.undo_redo,
        Capability::Selection => &mut capabilities.selection,
        Capability::Scenes => &mut capabilities.scenes,
        Capability::ReloadScripts => &mut capabilities.reload_scripts,
        Capability::BuildScripts => &mut capabilities.build_scripts,
        Capability::RunTests => &mut capabilities.run_tests,
        Capability::OpenEditor => &mut capabilities.open_editor,
    }
}

/// The capability `method` needs, by the method and, for `editor_command`,
/// the command; `None` for one every agent may call: what an agent needs to
/// learn the rules (`editor_state`, `project_info`) and to say who it is.
pub fn required(method: &str, params: &Value) -> Option<Capability> {
    use Capability::*;
    Some(match method {
        "scene_read" | "component_list" | "asset_find" => ReadScene,
        "console_read" => Console,
        "screenshot" => Screenshot,
        "frame_stats" => FrameStats,
        "scene_edit" => EditScene,
        "game_input" => GameInput,
        "build_scripts" => BuildScripts,
        "run_tests" => RunTests,
        "editor_open" => OpenEditor,
        "editor_command" => match params.get("command").and_then(Value::as_str)? {
            "play" | "pause" | "step" | "stop" | "load_scene" | "unload_scene" => PlayControl,
            "undo" | "redo" => UndoRedo,
            "select" | "frame" => Selection,
            "save" | "open_scene" | "new_scene" => Scenes,
            "reload_scripts" => ReloadScripts,
            _ => return None,
        },
        _ => return None,
    })
}

/// How far an agent may go in a folder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    None,
    Read,
    ReadWrite,
}

impl Access {
    pub const ALL: [Self; 3] = [Self::None, Self::Read, Self::ReadWrite];

    pub fn label(self) -> &'static str {
        match self {
            Self::None => "No access",
            Self::Read => "Read",
            Self::ReadWrite => "Read and write",
        }
    }
}

/// A folder of the project, relative to its root with `/` separators, and
/// how far an agent may go in it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FolderRule {
    pub path: String,
    pub access: Access,
}

/// A project's folder rules: the one whose path is the longest prefix of a
/// path decides it, and `unlisted` decides a path no rule covers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Folders {
    pub unlisted: Access,
    #[serde(rename = "folder")]
    pub rules: Vec<FolderRule>,
}

impl Default for Folders {
    /// The whole project, read and written: what the tools did before.
    fn default() -> Self {
        Self {
            unlisted: Access::ReadWrite,
            rules: Vec::new(),
        }
    }
}

/// A named set of [`Folders`]; Custom is what any other set is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FolderPreset {
    WholeProject,
    AssetsOnly,
    ReadOnly,
    Custom,
}

impl FolderPreset {
    pub const CHOSEN: [Self; 3] = [Self::WholeProject, Self::AssetsOnly, Self::ReadOnly];

    pub fn label(self) -> &'static str {
        match self {
            Self::WholeProject => "Whole project",
            Self::AssetsOnly => "Assets folder only",
            Self::ReadOnly => "Read only everywhere",
            Self::Custom => "Custom",
        }
    }
}

impl Folders {
    /// `preset`'s rules for a project whose assets folder is `assets`,
    /// relative to its root.
    pub fn of(preset: FolderPreset, assets: &str) -> Self {
        match preset {
            FolderPreset::WholeProject | FolderPreset::Custom => Self::default(),
            FolderPreset::AssetsOnly => Self {
                unlisted: Access::None,
                rules: vec![FolderRule {
                    path: assets.to_owned(),
                    access: Access::ReadWrite,
                }],
            },
            FolderPreset::ReadOnly => Self {
                unlisted: Access::Read,
                rules: Vec::new(),
            },
        }
    }

    pub fn preset(&self, assets: &str) -> FolderPreset {
        FolderPreset::CHOSEN
            .into_iter()
            .find(|&preset| Self::of(preset, assets) == *self)
            .unwrap_or(FolderPreset::Custom)
    }

    /// The rules in `editor_dir`: every folder read and written when there is
    /// no file, and an error naming the file when it does not read — which the
    /// callers treat as every folder refused, since a rule the developer wrote
    /// and the editor could not read must not quietly become no rule.
    pub fn load(editor_dir: &Path) -> Result<Self, String> {
        let file = editor_dir.join(FOLDERS);
        let text = match std::fs::read_to_string(&file) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => return Err(format!("{} could not be read: {error}", file.display())),
        };
        let mut folders: Self = toml::from_str(&text)
            .map_err(|error| format!("{} does not read: {error}", file.display()))?;
        for rule in &mut folders.rules {
            rule.path =
                normalise(&rule.path).map_err(|why| format!("{}: {why}", file.display()))?;
        }
        Ok(folders)
    }

    /// Write the rules to `editor_dir`, or remove the file for the default.
    pub fn save(&self, editor_dir: &Path) -> Result<(), String> {
        let file = editor_dir.join(FOLDERS);
        if *self == Self::default() {
            return match std::fs::remove_file(&file) {
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                    Err(format!("could not remove {}: {error}", file.display()))
                }
                _ => Ok(()),
            };
        }
        let text = toml::to_string(self).map_err(|error| error.to_string())?;
        std::fs::create_dir_all(editor_dir)
            .and_then(|()| std::fs::write(&file, text))
            .map_err(|error| format!("could not write {}: {error}", file.display()))
    }

    /// How far an agent may go at `path`, relative to the project root.
    pub fn access(&self, path: &str) -> Result<Access, String> {
        let path = normalise(path)?;
        let covers = |rule: &&FolderRule| {
            path == rule.path
                || rule.path.is_empty()
                || path
                    .strip_prefix(&rule.path)
                    .is_some_and(|rest| rest.starts_with('/'))
        };
        Ok(self
            .rules
            .iter()
            .filter(covers)
            .max_by_key(|rule| rule.path.len())
            .map_or(self.unlisted, |rule| rule.access))
    }

    /// Whether an agent may go as far as `need` at `path`; why not, naming
    /// the path and where to change it, when it may not.
    pub fn permit(&self, path: &str, need: Access) -> Result<(), String> {
        let have = self.access(path)?;
        if have >= need {
            return Ok(());
        }
        let what = match need {
            Access::ReadWrite => "write to",
            _ => "read",
        };
        Err(format!(
            "agents may not {what} {path} ({} there). Change the folder rules in {WHERE}",
            have.label().to_lowercase()
        ))
    }
}

/// `path` as a rule holds it: relative to the project root, `/`-separated,
/// with no `.` parts and no trailing `/`. Absolute paths and `..` are refused,
/// as `orrin.toml` refuses them, so no rule can name a place outside the
/// project.
pub fn normalise(path: &str) -> Result<String, String> {
    let as_path = Path::new(path);
    let mut parts = Vec::new();
    for part in as_path.components() {
        match part {
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            Component::CurDir => {}
            _ => {
                return Err(format!(
                    "`{path}` is not a folder inside the project: give one relative to it, with \
                     no `..`"
                ));
            }
        }
    }
    Ok(parts.join("/"))
}

/// What the tools report of the permissions in force.
pub fn summary(capabilities: &Capabilities, folders: &Result<Folders, String>) -> Value {
    let off: Vec<&str> = Capability::ALL
        .into_iter()
        .filter(|&capability| !capabilities.allows(capability))
        .map(Capability::key)
        .collect();
    let folders = match folders {
        Ok(folders) => json!({
            "unlisted": folders.unlisted,
            "rules": folders.rules,
        }),
        Err(why) => json!({"error": why, "every_folder": "refused until it reads"}),
    };
    json!({
        "preset": capabilities.preset().key(),
        "off": off,
        "folders": folders,
        "change_in": WHERE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every method and command an agent can call, with the capability it
    /// needs. A new method that lands here unmapped is callable whatever the
    /// switches say, so this list is the review.
    const CALLS: &[(&str, Option<&str>, Option<Capability>)] = &[
        ("editor_state", None, None),
        ("project_info", None, None),
        ("client_info", None, None),
        ("input_names", None, None),
        ("scene_read", None, Some(Capability::ReadScene)),
        ("component_list", None, Some(Capability::ReadScene)),
        ("asset_find", None, Some(Capability::ReadScene)),
        ("console_read", None, Some(Capability::Console)),
        ("screenshot", None, Some(Capability::Screenshot)),
        ("frame_stats", None, Some(Capability::FrameStats)),
        ("scene_edit", None, Some(Capability::EditScene)),
        ("game_input", None, Some(Capability::GameInput)),
        (
            "editor_command",
            Some("play"),
            Some(Capability::PlayControl),
        ),
        (
            "editor_command",
            Some("pause"),
            Some(Capability::PlayControl),
        ),
        (
            "editor_command",
            Some("step"),
            Some(Capability::PlayControl),
        ),
        (
            "editor_command",
            Some("stop"),
            Some(Capability::PlayControl),
        ),
        (
            "editor_command",
            Some("load_scene"),
            Some(Capability::PlayControl),
        ),
        (
            "editor_command",
            Some("unload_scene"),
            Some(Capability::PlayControl),
        ),
        ("editor_command", Some("undo"), Some(Capability::UndoRedo)),
        ("editor_command", Some("redo"), Some(Capability::UndoRedo)),
        (
            "editor_command",
            Some("select"),
            Some(Capability::Selection),
        ),
        ("editor_command", Some("frame"), Some(Capability::Selection)),
        ("editor_command", Some("save"), Some(Capability::Scenes)),
        (
            "editor_command",
            Some("open_scene"),
            Some(Capability::Scenes),
        ),
        (
            "editor_command",
            Some("new_scene"),
            Some(Capability::Scenes),
        ),
        (
            "editor_command",
            Some("reload_scripts"),
            Some(Capability::ReloadScripts),
        ),
        ("build_scripts", None, Some(Capability::BuildScripts)),
        ("run_tests", None, Some(Capability::RunTests)),
        ("editor_open", None, Some(Capability::OpenEditor)),
    ];

    fn params(command: Option<&str>) -> Value {
        command.map_or(Value::Null, |command| json!({ "command": command }))
    }

    /// Each capability turned off alone refuses exactly its own calls, by
    /// name and with where to turn it back on, and leaves every other call
    /// alone; the ones that need none are never refused.
    #[test]
    fn turning_one_capability_off_refuses_its_calls_and_no_others() {
        for off in Capability::ALL {
            let mut capabilities = Capabilities::default();
            capabilities.set(off, false);
            for &(method, command, needs) in CALLS {
                let checked = capabilities.check(method, &params(command));
                if needs == Some(off) {
                    let refused = checked.expect_err(method);
                    assert!(
                        refused.contains(off.label()) && refused.contains(WHERE),
                        "{refused}"
                    );
                } else {
                    assert!(checked.is_ok(), "{method} {command:?} with {off:?} off");
                }
            }
        }
        assert!(
            Capability::ALL
                .iter()
                .all(|&c| CALLS.iter().any(|&(_, _, needs)| needs == Some(c))),
            "every capability governs something"
        );
    }

    #[test]
    fn a_preset_changed_by_one_switch_is_custom_and_back_again() {
        for preset in Preset::CHOSEN {
            assert_eq!(Capabilities::of(preset).preset(), preset);
        }
        let mut capabilities = Capabilities::of(Preset::EditScene);
        capabilities.set(Capability::Scenes, true);
        assert_eq!(capabilities.preset(), Preset::Custom);
        capabilities.set(Capability::Scenes, false);
        assert_eq!(capabilities.preset(), Preset::EditScene);
        assert_eq!(Capabilities::default().preset(), Preset::FullControl);
    }

    #[test]
    fn the_longest_folder_rule_wins_and_the_unlisted_rule_takes_the_rest() {
        let folders = Folders {
            unlisted: Access::None,
            rules: vec![
                FolderRule {
                    path: "assets".into(),
                    access: Access::ReadWrite,
                },
                FolderRule {
                    path: "assets/levels".into(),
                    access: Access::Read,
                },
            ],
        };
        let access = |path| folders.access(path).unwrap();
        assert_eq!(access("assets/props/crate.oprefab"), Access::ReadWrite);
        assert_eq!(access("assets/levels/one.orrin"), Access::Read);
        assert_eq!(access("assets/levels"), Access::Read);
        assert_eq!(access("assets/levels2/one.orrin"), Access::ReadWrite);
        assert_eq!(access("./assets/levels/./one.orrin"), Access::Read);
        assert_eq!(access("scripts/Main.cs"), Access::None);
        assert_eq!(access("assetsbis/x"), Access::None);

        for outside in ["../other/x.orrin", "assets/../../x", "/etc/passwd"] {
            assert!(folders.access(outside).is_err(), "{outside}");
            assert!(folders.permit(outside, Access::Read).is_err(), "{outside}");
        }
        let refused = folders
            .permit("assets/levels/one.orrin", Access::ReadWrite)
            .unwrap_err();
        assert!(
            refused.contains("may not write to assets/levels/one.orrin"),
            "{refused}"
        );
        assert!(
            folders
                .permit("assets/props/a.orrin", Access::ReadWrite)
                .is_ok()
        );

        assert_eq!(
            Folders::of(FolderPreset::AssetsOnly, "assets").preset("assets"),
            FolderPreset::AssetsOnly
        );
        assert_eq!(folders.preset("assets"), FolderPreset::Custom);
    }
}
