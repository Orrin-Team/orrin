//! The Settings tab: one window over two scopes, because the question people
//! cannot answer from a setting's name is who it affects.
//!
//! **Two scopes, said once.** *Project* pages edit `orrin.toml`, which is
//! checked in and shared; *Orrin* pages edit `EditorState::settings`, which
//! follows this person across projects (see `editor::settings`). The sidebar
//! groups pages under those two names, and that heading is the whole
//! statement: a page is its title and its rows. Search crosses both with no
//! sidebar in sight, so there each page is followed by its group's name.
//!
//! **Rows are data.** A `Row` is a page, a label, help and the words people
//! type for it, and the `Field` it edits; search, the changed marker, reset
//! and the notes are written once against that, so a page is added by adding
//! rows. Only Orrin rows have a default: a project's name or entry is the
//! project's, not a deviation from anything, so Project rows have no marker
//! and no reset.
//!
//! **Settings that live elsewhere are found here and left there.** Fog,
//! exposure and the rest belong to the scene and are edited in Environment;
//! search lists them as `Link`s to the tool that owns them rather than
//! growing a second copy of its controls.
//!
//! **Nothing to confirm.** No Apply and no OK: an Orrin row edits the state
//! `SettingsStore::settle` applies and saves at the end of the frame, and a
//! Project row commits through `ManifestEdit` on Enter, on losing focus, or on
//! a pick — Escape puts a field back. Neither goes in the undo history, which
//! is the scene's. With nothing to put back, Escape closes the window the tab
//! floats in (`Dock::close_on_escape`). The one held value is the UI scale: a UI that rescales
//! under the pointer cannot be dragged, so the tab keeps the dragged value and
//! hands it over when the pointer lets go.
//!
//! **The manifest is read, never assumed.** The tab loads `orrin.toml` itself
//! and re-reads it when its modification time changes — checked at most once
//! a second, and only while a Project page or a search is on screen — so a
//! hand edit shows up, and a file that no longer loads shows the loader's
//! error and a way to the file instead of fields that could not be saved. A scene is picked from
//! the project's scene assets and never typed, which is what keeps an
//! absolute or `..` path out of the manifest; the picker writes it relative
//! to the project root, as the manifest spells every path, not to the assets
//! folder the asset database lists it from.
//!
//! **Startup rows apply on restart.** They decide what the `ORRIN_*`
//! variables read once at startup decided alone, so a change is kept at once
//! and taken by the next start. The page counts what differs from what this
//! run started with and offers the restart, saving the scene first when it has
//! unsaved work. A row the environment sets shows the environment's value and
//! is not counted: the next run inherits the same environment, so it would not
//! apply that change either.
//!
//! **Any width.** Docked into a column the sidebar folds into one page picker.
//!
//! **Permissions are the person's, folders the project's.** The Permissions
//! page's switches are `[agents.allow]` in the user's settings and its folder
//! rules the project's `.orrin/agent-folders.toml` (`orrin_remote::permissions`
//! holds both, and is what the editor and `orrin mcp` check against). They
//! govern what Orrin's tools do for an agent. They do not confine an agent that
//! can already write the project's files some other way, and the page does not
//! claim to.
//!
//! **Closed, it costs nothing:** the dock does not call [`body`]. Open, it is
//! a handful of rows a frame and, on a Project page, three file stats a
//! second: the manifest and the two scenes it names. The Blender page looks
//! for Blender, which searches the PATH, once a second.

use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use orrin_ecs::World;
use orrin_project::{ManifestEdit, Project};
use orrin_registry::inspect::{compact, label_width};

use super::inspector::Filter;
use super::performance::{VSYNC_MODES, vsync_label};
use crate::asset_db::{AssetDatabase, AssetKind};
use crate::editor::dock::Tab;
use crate::editor::icons;
use crate::editor::settings::{
    AUTOSAVE_SECS, BOOST, Backend, ExternalEditor, FLY_SPEED, LogFloor, SCROLL_SPEED, SENSITIVITY,
    SWAPCHAIN_IMAGES, Settings, Startup, UI_SCALE, Validation,
};
use crate::editor::state::{DockRequest, EditorState, IconTints};
use crate::editor::theme::{self, ThemeSet};
use crate::editor::widgets;
use orrin_remote::permissions::{
    Access, Capabilities, Capability, FolderPreset, FolderRule, Folders, Preset,
};

/// Below this the sidebar folds into a picker: a 168-point sidebar beside a
/// property row's label column leaves too little for a slider.
const NARROW: f32 = 520.0;
const SIDEBAR: f32 = 168.0;
/// How often a visible Project page asks whether `orrin.toml` changed.
const RECHECK: Duration = Duration::from_secs(1);
/// What the File menu says while the game plays.
const PLAYING: &str = "Stop the game first: the scene is not what is playing";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Page {
    General,
    Scenes,
    Scripts,
    #[default]
    Appearance,
    Viewport,
    Scripting,
    Blender,
    Agents,
    Permissions,
    Saving,
    Display,
    Startup,
}

impl Page {
    const PROJECT: [Self; 3] = [Self::General, Self::Scenes, Self::Scripts];
    const ORRIN: [Self; 9] = [
        Self::Appearance,
        Self::Viewport,
        Self::Scripting,
        Self::Blender,
        Self::Agents,
        Self::Permissions,
        Self::Saving,
        Self::Display,
        Self::Startup,
    ];

    fn title(self) -> &'static str {
        match self {
            Self::General => "General",
            Self::Scenes => "Scenes",
            Self::Scripts => "Scripts",
            Self::Appearance => "Appearance",
            Self::Viewport => "Viewport",
            Self::Scripting => "Scripting",
            Self::Blender => "Blender",
            Self::Agents => "Agents",
            Self::Permissions => "Permissions",
            Self::Saving => "Saving",
            Self::Display => "Display",
            Self::Startup => "Startup",
        }
    }

    fn is_project(self) -> bool {
        Self::PROJECT.contains(&self)
    }

    /// The sidebar heading the page is under, beside its name in search
    /// results.
    fn group(self) -> &'static str {
        if self.is_project() {
            "Project"
        } else {
            "Orrin"
        }
    }
}

/// What a row edits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Field {
    Name,
    Version,
    Location,
    Start,
    Persistent,
    Entry,
    ScriptsFolder,
    AssetsFolder,
    Theme,
    UiScale,
    LightIcons,
    SoundIcons,
    ListenerIcons,
    CameraIcons,
    EmptyIcons,
    FlySpeed,
    Boost,
    Sensitivity,
    ScrollSpeed,
    Editor,
    /// The external editor's own command, while it is Custom.
    Command,
    Reload,
    BlenderPath,
    LiveLink,
    LinkPort,
    /// Not a setting but an action, found by search like one.
    AddOn,
    Remote,
    /// An action too: the client's configuration, to the clipboard.
    McpConfig,
    /// Which of the presets the switches below it are.
    Preset,
    Allow(Capability),
    /// The folder a new folder rule is for, while it is typed.
    AddFolder,
    Autosave,
    Interval,
    Present,
    Images,
    Backend,
    Threads,
    Validation,
    Log,
}

impl Field {
    fn is_startup(self) -> bool {
        matches!(
            self,
            Self::Backend | Self::Threads | Self::Validation | Self::Log
        )
    }
}

struct Row {
    page: Page,
    field: Field,
    label: &'static str,
    help: &'static str,
    /// What people type looking for it that the label and help do not say.
    keywords: &'static str,
}

const fn row(
    page: Page,
    field: Field,
    label: &'static str,
    help: &'static str,
    keywords: &'static str,
) -> Row {
    Row {
        page,
        field,
        label,
        help,
        keywords,
    }
}

const ROWS: &[Row] = &[
    row(
        Page::General,
        Field::Name,
        "Name",
        "What the project is called: the top bar, and the exported game's file and window",
        "title rename",
    ),
    row(
        Page::General,
        Field::Version,
        "Version",
        "Written into an exported game",
        "release semver",
    ),
    row(
        Page::General,
        Field::Location,
        "Location",
        "The folder holding orrin.toml",
        "folder path directory root",
    ),
    row(
        Page::Scenes,
        Field::Start,
        "Start scene",
        "The scene an exported game opens on",
        "first main boot opening level",
    ),
    row(
        Page::Scenes,
        Field::Persistent,
        "Persistent scene",
        "Loaded beside every other scene for as long as the game runs",
        "always global hud manager",
    ),
    row(
        Page::Scripts,
        Field::Entry,
        "Entry behaviour",
        "The Behaviour attached at start, as Namespace.Type, Assembly",
        "entry point main class c# csharp dll",
    ),
    row(
        Page::Scripts,
        Field::ScriptsFolder,
        "Scripts folder",
        "Where the game's C# lives, relative to the project",
        "code source csproj directory",
    ),
    row(
        Page::Scripts,
        Field::AssetsFolder,
        "Assets folder",
        "Where the project's assets live, relative to the project",
        "content directory",
    ),
    row(
        Page::Appearance,
        Field::Theme,
        "Theme",
        "The editor's colours",
        "dark light colour color palette skin",
    ),
    row(
        Page::Appearance,
        Field::UiScale,
        "UI scale",
        "A multiple of the display's own scale",
        "zoom font size text size dpi bigger smaller",
    ),
    row(
        Page::Appearance,
        Field::LightIcons,
        "Light icons",
        "The colour lights are marked in, in the viewport and the hierarchy",
        "icon colour color tint gizmo marker",
    ),
    row(
        Page::Appearance,
        Field::SoundIcons,
        "Sound icons",
        "The colour sound sources are marked in",
        "icon colour color tint audio speaker marker",
    ),
    row(
        Page::Appearance,
        Field::ListenerIcons,
        "Listener icons",
        "The colour audio listeners are marked in",
        "icon colour color tint ear marker",
    ),
    row(
        Page::Appearance,
        Field::CameraIcons,
        "Camera icons",
        "The colour cameras and their wireframes are drawn in",
        "icon colour color tint frustum marker",
    ),
    row(
        Page::Appearance,
        Field::EmptyIcons,
        "Empty icons",
        "The colour entities with nothing to draw are marked in",
        "icon colour color tint marker",
    ),
    row(
        Page::Viewport,
        Field::FlySpeed,
        "Fly speed",
        "How fast the camera flies with the right button held",
        "camera move movement wasd navigation",
    ),
    row(
        Page::Viewport,
        Field::Boost,
        "Boost",
        "What holding Shift multiplies the fly speed by",
        "camera fast sprint shift",
    ),
    row(
        Page::Viewport,
        Field::Sensitivity,
        "Look sensitivity",
        "How far the view turns for a move of the mouse",
        "mouse camera turn rotate",
    ),
    row(
        Page::Viewport,
        Field::ScrollSpeed,
        "Scroll speed",
        "Metres the camera moves per notch of the wheel",
        "wheel zoom dolly camera",
    ),
    row(
        Page::Scripting,
        Field::Editor,
        "External editor",
        "What opens a script at a line from the console",
        "ide vscode vs code rider zed sublime open code",
    ),
    row(
        Page::Scripting,
        Field::Reload,
        "Reload on rebuild",
        "Swap in the game's scripts after a rebuild that succeeds",
        "hot reload auto build dotnet",
    ),
    row(
        Page::Blender,
        Field::BlenderPath,
        "Blender",
        "The Blender that exports .blend assets and installs the add-on; empty finds an \
         installed one",
        "path executable application app exe location",
    ),
    row(
        Page::Blender,
        Field::LiveLink,
        "Live link",
        "Let the Blender add-on stream the file it has open into the viewport as it is edited, \
         before it is saved",
        "stream sync realtime real-time meshsync edit mode",
    ),
    row(
        Page::Blender,
        Field::LinkPort,
        "Port",
        "Where on 127.0.0.1 the live link listens; the add-on's preferences name the same port",
        "network tcp socket localhost live link",
    ),
    row(
        Page::Blender,
        Field::AddOn,
        "Add-on",
        "Install the Orrin Live Link add-on into that Blender and enable it",
        "addon install add-on extension plugin live link",
    ),
    row(
        Page::Agents,
        Field::Remote,
        "Remote",
        "Let an AI agent read the open scene, the assets and the console through `orrin mcp`, \
         over 127.0.0.1 with a token only this project's .orrin/remote holds",
        "mcp ai agent claude llm assistant tool remote server automation",
    ),
    row(
        Page::Agents,
        Field::McpConfig,
        "MCP config",
        "Copy a .mcp.json entry that runs `orrin mcp` for this project, for an MCP client such \
         as Claude Code",
        "mcp json copy clipboard client claude code configure setup",
    ),
    row(
        Page::Permissions,
        Field::Preset,
        "Preset",
        "What an agent may do through Orrin's tools; changing a switch below makes it Custom",
        "agent permission permissions access preset read only edit full control custom mcp",
    ),
    row(
        Page::Permissions,
        Field::Allow(Capability::ReadScene),
        "Read scene and assets",
        "Read the open scene, the registered components and the asset index",
        "agent permission allow deny read outline blocks components assets index",
    ),
    row(
        Page::Permissions,
        Field::Allow(Capability::Console),
        "Read the console",
        "Read the editor console's lines",
        "agent permission allow deny log logs messages errors warnings",
    ),
    row(
        Page::Permissions,
        Field::Allow(Capability::Screenshot),
        "Screenshots",
        "Take a picture of the game or the scene",
        "agent permission allow deny image picture png view",
    ),
    row(
        Page::Permissions,
        Field::Allow(Capability::FrameStats),
        "Frame stats",
        "Read frame times, the profiler and the run banner",
        "agent permission allow deny performance profiler fps timings",
    ),
    row(
        Page::Permissions,
        Field::Allow(Capability::EditScene),
        "Edit the scene",
        "Change the open scene through scene_edit, each call one undo step",
        "agent permission allow deny edit spawn despawn set component reparent modify",
    ),
    row(
        Page::Permissions,
        Field::Allow(Capability::PlayControl),
        "Play, pause and stop",
        "Play, pause, step and stop the game, and load scenes while it plays",
        "agent permission allow deny play pause stop step run game load unload",
    ),
    row(
        Page::Permissions,
        Field::Allow(Capability::GameInput),
        "Send game input",
        "Press keys, buttons and actions in the playing game, as a playtest",
        "agent permission allow deny input keyboard mouse gamepad press playtest",
    ),
    row(
        Page::Permissions,
        Field::Allow(Capability::UndoRedo),
        "Undo and redo",
        "Undo and redo the editor's steps, the developer's own included",
        "agent permission allow deny undo redo history",
    ),
    row(
        Page::Permissions,
        Field::Allow(Capability::Selection),
        "Select and frame",
        "Change the selection and frame it in the camera",
        "agent permission allow deny select selection frame camera focus",
    ),
    row(
        Page::Permissions,
        Field::Allow(Capability::Scenes),
        "Save, open and create scenes",
        "Save the open scene, open another, or make a new one",
        "agent permission allow deny save open new scene file write",
    ),
    row(
        Page::Permissions,
        Field::Allow(Capability::ReloadScripts),
        "Reload scripts",
        "Swap in the game's last build",
        "agent permission allow deny hot reload scripts assembly",
    ),
    row(
        Page::Permissions,
        Field::Allow(Capability::BuildScripts),
        "Build scripts",
        "Compile the project's C# with orrin build",
        "agent permission allow deny compile dotnet build csharp",
    ),
    row(
        Page::Permissions,
        Field::Allow(Capability::RunTests),
        "Run tests",
        "Run the project's tests with orrin test",
        "agent permission allow deny test tests unit",
    ),
    row(
        Page::Permissions,
        Field::Allow(Capability::OpenEditor),
        "Open the editor",
        "Launch an editor on the project with its remote on",
        "agent permission allow deny launch start editor_open run",
    ),
    row(
        Page::Saving,
        Field::Autosave,
        "Autosave",
        "Keep a copy of unsaved work, recoverable from the File menu",
        "backup recover crash",
    ),
    row(
        Page::Saving,
        Field::Interval,
        "Interval",
        "How long between copies of unsaved work",
        "autosave every frequency period",
    ),
    row(
        Page::Display,
        Field::Present,
        "Present mode",
        "How finished frames reach the screen; Immediate is the one to measure in",
        "vsync fifo mailbox immediate tearing frame cap fps",
    ),
    row(
        Page::Display,
        Field::Images,
        "Swapchain images",
        "How many images the window cycles through; three suits an uncapped mode",
        "buffering triple double latency",
    ),
    row(
        Page::Startup,
        Field::Backend,
        "Graphics backend",
        "The graphics API the editor draws through",
        "vulkan metal gpu graphics api renderer moltenvk",
    ),
    row(
        Page::Startup,
        Field::Threads,
        "Worker threads",
        "How many threads the engine's parallel work runs on",
        "threads cores cpu parallel pool workers",
    ),
    row(
        Page::Startup,
        Field::Validation,
        "Validation layers",
        "Vulkan's checks on every GPU call: slower, and names what went wrong",
        "validation debug layer vulkan khronos gpu errors",
    ),
    row(
        Page::Startup,
        Field::Log,
        "Log level",
        "The quietest message the log keeps",
        "log verbose logging trace debug quiet console",
    ),
];

/// Metal is a macOS API, so elsewhere it is not offered.
const BACKENDS: &[Backend] = if cfg!(target_os = "macos") {
    &[Backend::Auto, Backend::Vulkan, Backend::Metal]
} else {
    &[Backend::Auto, Backend::Vulkan]
};
const VALIDATIONS: [Validation; 3] = [Validation::Auto, Validation::On, Validation::Off];
const LOG_FLOORS: [LogFloor; 6] = [
    LogFloor::Default,
    LogFloor::Trace,
    LogFloor::Debug,
    LogFloor::Info,
    LogFloor::Warn,
    LogFloor::Error,
];

/// A setting that lives in another tool, found by search and left there.
struct Link {
    label: &'static str,
    keywords: &'static str,
    tab: Tab,
    place: &'static str,
}

const fn link(label: &'static str, keywords: &'static str, tab: Tab, place: &'static str) -> Link {
    Link {
        label,
        keywords,
        tab,
        place,
    }
}

const SCENE: &str = "in Environment (this scene)";

const LINKS: &[Link] = &[
    link(
        "Fog",
        "volumetric mist haze atmosphere",
        Tab::Environment,
        SCENE,
    ),
    link(
        "Exposure",
        "tonemap ev100 brightness hdr auto exposure",
        Tab::Environment,
        SCENE,
    ),
    link("Bloom", "glow", Tab::Environment, SCENE),
    link(
        "Shadows",
        "cascades shadow map contact",
        Tab::Environment,
        SCENE,
    ),
    link("Ambient occlusion", "ssao gtao ao", Tab::Environment, SCENE),
    link(
        "Anti-aliasing",
        "taa temporal msaa antialiasing aliasing jaggies",
        Tab::Environment,
        SCENE,
    ),
    link("Reflections", "ssr screen space", Tab::Environment, SCENE),
    link(
        "Sky",
        "hdri environment map ambient light",
        Tab::Environment,
        SCENE,
    ),
    link(
        "Depth of field",
        "dof lens focus blur",
        Tab::Environment,
        SCENE,
    ),
    link("Motion blur", "shutter", Tab::Environment, SCENE),
    link(
        "Frame timing",
        "profiler fps performance cpu gpu ms passes",
        Tab::Performance,
        "in Profiler",
    ),
    link(
        "Scenes list",
        "open new scene manager levels",
        Tab::Scenes,
        "in Scenes",
    ),
];

/// The tab's own state, kept on `EditorState` across frames.
#[derive(Default)]
pub struct SettingsPanel {
    query: String,
    page: Page,
    /// Where Edit ▸ Settings… opens: the Orrin page last viewed this session.
    last_orrin: Page,
    focus_search: bool,
    /// Whether the keyboard is in the search field with nothing typed, which
    /// is how the tab opens and not a field in use.
    search_idle: bool,
    /// The UI scale while its slider is held.
    scale: Option<f32>,
    /// Text fields with focus, and what they hold until they commit.
    typing: Vec<(Field, String)>,
    manifest: Manifest,
    /// The Blender the Blender page would run, as last looked for.
    found: Option<Found>,
    /// Why the last folder typed into Permissions was not added.
    folder_error: Option<String>,
}

/// `blend::locate` searches the PATH, so the page asks it at most once a
/// second, and again at once when the path it was asked with changes.
struct Found {
    at: Instant,
    configured: Option<PathBuf>,
    env: Option<String>,
    path: Option<PathBuf>,
    exists: bool,
}

/// The project's `orrin.toml` as the tab last read it.
#[derive(Default)]
struct Manifest {
    /// The project root it was read for.
    root: Option<PathBuf>,
    loaded: Option<Result<Project, String>>,
    /// The last commit's refusal, shown under the row that made it.
    error: Option<(Field, String)>,
    stamp: Option<(SystemTime, u64)>,
    checked: Option<Instant>,
    /// Whether the start and the persistent scene the manifest names exist.
    missing: [bool; 2],
}

impl SettingsPanel {
    /// Show `page`, or the Orrin page last viewed for `None`, with the search
    /// field empty and focused.
    pub fn show(&mut self, page: Option<Page>) {
        self.select(page.unwrap_or(self.last_orrin));
        self.query.clear();
        self.focus_search = true;
    }

    /// Whether the search field holds the keyboard and no text: nothing an
    /// Escape would be putting back or leaving.
    pub fn search_is_idle(&self) -> bool {
        self.search_idle
    }

    #[cfg(test)]
    pub fn set_query(&mut self, query: &str) {
        self.query = query.to_owned();
    }

    #[cfg(test)]
    pub fn recheck_now(&mut self) {
        self.manifest.checked = None;
    }

    /// The Blender `blend::locate` finds for these settings and environment,
    /// and whether it is there.
    fn blender(&mut self, state: &EditorState) -> (Option<PathBuf>, bool) {
        let configured = state.settings.blender.path.as_ref();
        let env = state.environment.blender();
        let fresh = self.found.as_ref().is_some_and(|found| {
            found.at.elapsed() < RECHECK
                && found.configured.as_ref() == configured
                && found.env.as_deref() == env
        });
        if !fresh {
            let path = crate::asset_db::blend::locate(
                env.map(std::ffi::OsString::from),
                configured.map(PathBuf::as_path),
            );
            self.found = Some(Found {
                at: Instant::now(),
                configured: configured.cloned(),
                env: env.map(str::to_owned),
                exists: path.as_deref().is_some_and(Path::is_file),
                path,
            });
        }
        self.found
            .as_ref()
            .map_or((None, false), |found| (found.path.clone(), found.exists))
    }

    fn select(&mut self, page: Page) {
        self.page = page;
        if !page.is_project() {
            self.last_orrin = page;
        }
    }
}

/// A change to the manifest a Project row asked for.
enum Change {
    Name(String),
    Version(String),
    Entry(String),
    Start(Option<PathBuf>),
    Persistent(Option<PathBuf>),
}

impl Change {
    fn field(&self) -> Field {
        match self {
            Self::Name(_) => Field::Name,
            Self::Version(_) => Field::Version,
            Self::Entry(_) => Field::Entry,
            Self::Start(_) => Field::Start,
            Self::Persistent(_) => Field::Persistent,
        }
    }

    fn apply(self, edit: &mut ManifestEdit) {
        match self {
            Self::Name(name) => edit.set_name(&name),
            Self::Version(version) => edit.set_version(&version),
            Self::Entry(entry) => edit.set_entry(&entry),
            Self::Start(scene) => edit.set_start_scene(scene.as_deref()),
            Self::Persistent(scene) => edit.set_persistent_scene(scene.as_deref()),
        }
    }
}

impl Manifest {
    /// Read `<root>/orrin.toml` if it is a different project, or if it changed
    /// on disk since the last look a second or more ago.
    fn sync(&mut self, root: &Path, state: &mut EditorState) {
        let same = self.root.as_deref() == Some(root);
        if same && self.checked.is_some_and(|at| at.elapsed() < RECHECK) {
            return;
        }
        self.checked = Some(Instant::now());
        let path = root.join("orrin.toml");
        let stamp = stamp(&path);
        if same && stamp == self.stamp && self.loaded.is_some() {
            self.missing = self.project().map_or([false; 2], scenes_missing);
            return;
        }
        self.root = Some(root.to_owned());
        self.stamp = stamp;
        self.error = None;
        match Project::load(&path) {
            Ok(project) => self.adopt(project, state),
            Err(error) => self.loaded = Some(Err(error.to_string())),
        }
    }

    fn project(&self) -> Option<&Project> {
        self.loaded.as_ref()?.as_ref().ok()
    }

    /// What the editor holds about the project follows the file.
    fn adopt(&mut self, project: Project, state: &mut EditorState) {
        state.project_name = Some(project.name().to_owned());
        state.persistent_scene = project.persistent_scene();
        self.missing = scenes_missing(&project);
        self.stamp = stamp(&project.manifest_path());
        self.loaded = Some(Ok(project));
    }

    /// Nothing is written unless all of it validates; a refusal is shown under
    /// its row and the row shows what is on disk again.
    fn commit(&mut self, change: Change, state: &mut EditorState) {
        let Some(project) = self.project() else {
            return;
        };
        let field = change.field();
        let committed = project.edit().and_then(|mut edit| {
            change.apply(&mut edit);
            edit.commit()
        });
        match committed {
            Ok(project) => {
                self.error = None;
                self.adopt(project, state);
            }
            Err(error) => self.error = Some((field, error.to_string())),
        }
    }
}

fn stamp(path: &Path) -> Option<(SystemTime, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

fn scenes_missing(project: &Project) -> [bool; 2] {
    [project.start_scene(), project.persistent_scene()]
        .map(|scene| scene.is_some_and(|scene| !scene.is_file()))
}

/// `path` as the manifest spells paths: its plain components, joined by
/// forward slashes on every platform.
fn slashed(path: &Path) -> String {
    path.components()
        .filter_map(|part| match part {
            Component::Normal(part) => Some(part.to_string_lossy()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// The manifest's spelling of an asset at `asset`, which the asset database
/// gives relative to the assets folder: relative to the project root instead.
fn in_manifest(project: &Project, asset: &str) -> String {
    let assets = project.assets_dir();
    let base = assets.strip_prefix(project.root()).unwrap_or(Path::new(""));
    slashed(&base.join(asset))
}

pub fn body(ui: &mut egui::Ui, world: &World, state: &mut EditorState, themes: &ThemeSet) {
    let mut panel = std::mem::take(&mut state.settings_panel);
    let search = widgets::search(ui, &mut panel.query, "Search settings");
    if std::mem::take(&mut panel.focus_search) {
        search.request_focus();
    }
    panel.search_idle = search.has_focus() && panel.query.is_empty();
    ui.add_space(6.0);

    let root = state.project_root.clone();
    if root.is_none() && panel.page.is_project() {
        panel.page = panel.last_orrin;
    }
    let searching = !panel.query.trim().is_empty();
    if let Some(root) = &root
        && (searching || panel.page.is_project())
    {
        panel.manifest.sync(root, state);
    }
    let mut cx = Cx {
        world,
        state,
        themes,
        defaults: Settings::default(),
    };
    if searching {
        results(ui, &mut cx, &mut panel);
    } else if ui.available_width() < NARROW {
        picker(ui, cx.state, &mut panel);
        ui.add_space(6.0);
        page(ui, &mut cx, &mut panel);
    } else {
        let height = ui.available_height();
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(
                egui::vec2(SIDEBAR, height),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    ui.set_min_size(egui::vec2(SIDEBAR, height));
                    sidebar(ui, cx.state, &mut panel);
                },
            );
            // The one divider: everything else is grouped by space.
            let x = ui.cursor().left() + 6.0;
            let top = ui.cursor().top();
            ui.painter().vline(
                x,
                top..=top + height,
                ui.visuals().widgets.noninteractive.bg_stroke,
            );
            ui.add_space(14.0);
            ui.vertical(|ui| page(ui, &mut cx, &mut panel));
        });
    }
    state.settings_panel = panel;
}

/// What every row draws against.
struct Cx<'a> {
    world: &'a World,
    state: &'a mut EditorState,
    themes: &'a ThemeSet,
    defaults: Settings,
}

const NO_PROJECT: &str = "No project is open. Open one with `orrin run` or Orrin Hub.";

fn sidebar(ui: &mut egui::Ui, state: &EditorState, panel: &mut SettingsPanel) {
    let weak = ui.visuals().weak_text_color();
    ui.label(egui::RichText::new("Project").strong());
    match (&state.project_root, &state.project_name) {
        (Some(_), name) => {
            let name = name.as_deref().unwrap_or("this project");
            ui.add(egui::Label::new(egui::RichText::new(name).color(weak)).truncate());
            ui.add_space(2.0);
            for page in Page::PROJECT {
                entry(ui, panel, page);
            }
        }
        (None, _) => {
            ui.add(egui::Label::new(egui::RichText::new(NO_PROJECT).color(weak)).wrap());
        }
    }
    ui.add_space(14.0);
    ui.label(egui::RichText::new("Orrin").strong());
    ui.add_space(2.0);
    for page in Page::ORRIN {
        entry(ui, panel, page);
    }
}

/// A page in the sidebar: the selection fill when it is the page shown, the
/// hover fill under the pointer, and nothing otherwise.
fn entry(ui: &mut egui::Ui, panel: &mut SettingsPanel, page: Page) {
    let height = ui.spacing().interact_size.y;
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::click(),
    );
    let visuals = ui.visuals();
    let selected = panel.page == page;
    let fill = if selected {
        Some(visuals.selection.bg_fill)
    } else if response.hovered() {
        Some(visuals.widgets.hovered.weak_bg_fill)
    } else {
        None
    };
    if let Some(fill) = fill {
        ui.painter()
            .rect_filled(rect, visuals.widgets.inactive.corner_radius, fill);
    }
    let colour = if selected {
        visuals.strong_text_color()
    } else {
        visuals.text_color()
    };
    ui.painter().text(
        rect.left_center() + egui::vec2(8.0, 0.0),
        egui::Align2::LEFT_CENTER,
        page.title(),
        egui::TextStyle::Body.resolve(ui.style()),
        colour,
    );
    if response.clicked() {
        panel.select(page);
    }
}

/// The sidebar folded into one control, for a narrow dock.
fn picker(ui: &mut egui::Ui, state: &EditorState, panel: &mut SettingsPanel) {
    let mut chosen = None;
    let weak = ui.visuals().weak_text_color();
    egui::ComboBox::from_id_salt("settings page picker")
        .width(ui.available_width())
        .selected_text(panel.page.title())
        .show_ui(ui, |ui| {
            ui.label(egui::RichText::new("Project").color(weak));
            if state.project_root.is_some() {
                for page in Page::PROJECT {
                    if ui
                        .selectable_label(panel.page == page, page.title())
                        .clicked()
                    {
                        chosen = Some(page);
                    }
                }
            } else {
                ui.add(egui::Label::new(egui::RichText::new(NO_PROJECT).color(weak)).wrap());
            }
            ui.add_space(4.0);
            ui.label(egui::RichText::new("Orrin").color(weak));
            for page in Page::ORRIN {
                if ui
                    .selectable_label(panel.page == page, page.title())
                    .clicked()
                {
                    chosen = Some(page);
                }
            }
        });
    if let Some(page) = chosen {
        panel.select(page);
    }
}

/// The selected page: its title and its rows.
fn page(ui: &mut egui::Ui, cx: &mut Cx, panel: &mut SettingsPanel) {
    let page = panel.page;
    let weak = ui.visuals().weak_text_color();
    ui.label(egui::RichText::new(page.title()).strong());
    ui.add_space(10.0);
    if page == Page::Startup {
        restart_line(ui, cx.state);
    }
    let unread = match &panel.manifest.loaded {
        Some(Err(error)) if page.is_project() => Some(error.clone()),
        _ => None,
    };
    egui::ScrollArea::vertical()
        .id_salt("settings page")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            match unread {
                Some(error) => unloadable(ui, &error, cx.state),
                None if page.is_project() && panel.manifest.project().is_none() => {}
                None => {
                    #[cfg(not(feature = "scripting"))]
                    if page == Page::Scripting {
                        note(
                            ui,
                            "This build was compiled without the `scripting` feature; these \
                             are kept for a build that has it",
                            weak,
                        );
                    }
                    rows_of(ui, cx, panel, |row| row.page == page);
                    if page == Page::Permissions {
                        folders(ui, cx.state, panel);
                    }
                }
            }
            if !page.is_project() && cx.state.settings_file.is_none() {
                ui.add_space(16.0);
                ui.label(egui::RichText::new("Not saved: no user directory").color(weak));
            }
        });
}

/// `orrin.toml` does not load: what the loader said, and the way to the file.
fn unloadable(ui: &mut egui::Ui, error: &str, state: &EditorState) {
    ui.add(egui::Label::new(egui::RichText::new(error).color(theme::ERROR)).wrap());
    ui.add_space(6.0);
    if ui.button("Open orrin.toml").clicked()
        && let Some(root) = &state.project_root
    {
        crate::editor::export::reveal(&root.join("orrin.toml"));
    }
}

/// The rows `keep` accepts, in the order they are declared.
fn rows_of(ui: &mut egui::Ui, cx: &mut Cx, panel: &mut SettingsPanel, keep: impl Fn(&Row) -> bool) {
    for row in ROWS.iter().filter(|row| keep(row)) {
        if row.page.is_project() {
            project_row(ui, cx, panel, row);
        } else {
            orrin_row(ui, cx, panel, row);
        }
    }
}

/// Every row that matches, under its page's name and group, then the
/// settings other tools own.
fn results(ui: &mut egui::Ui, cx: &mut Cx, panel: &mut SettingsPanel) {
    let filter = Filter::new(&panel.query);
    let project = cx.state.project_root.is_some() && panel.manifest.project().is_some();
    let matches = |row: &Row| {
        (project || !row.page.is_project())
            && [row.label, row.help, row.keywords, row.page.title()]
                .into_iter()
                .any(|text| filter.hits(text))
    };
    let weak = ui.visuals().weak_text_color();
    egui::ScrollArea::vertical()
        .id_salt("settings results")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if ROWS
                .iter()
                .any(|row| row.page == Page::Startup && matches(row))
            {
                restart_line(ui, cx.state);
            }
            let mut any = false;
            for page in Page::PROJECT.into_iter().chain(Page::ORRIN) {
                if !ROWS.iter().any(|row| row.page == page && matches(row)) {
                    continue;
                }
                any = true;
                heading(ui, page.title(), page.group());
                rows_of(ui, cx, panel, |row| row.page == page && matches(row));
                ui.add_space(10.0);
            }
            let mut links = LINKS
                .iter()
                .filter(|link| filter.hits(link.label) || filter.hits(link.keywords))
                .peekable();
            if links.peek().is_some() {
                any = true;
                heading(ui, "Elsewhere", "edited in another tool");
            }
            for link in links {
                let hint = format!("Open {}", link.tab.title());
                let mut arrow = false;
                let place = widgets::property_with(
                    ui,
                    link.label,
                    |ui| {
                        let place = egui::RichText::new(link.place).color(weak);
                        ui.add(egui::Button::new(place).frame(false))
                            .on_hover_text(&hint)
                            .clicked()
                    },
                    |ui| {
                        arrow = icons::button(ui, icons::arrow_square_out(), 12.0, weak, false)
                            .on_hover_text(&hint)
                            .clicked();
                    },
                );
                if place || arrow {
                    cx.state.request_dock(DockRequest::Show(link.tab));
                }
            }
            if !any {
                ui.label(egui::RichText::new("No setting matches").color(weak));
            }
        });
}

/// What a Startup change is waiting on, and the way to it: weak text and an
/// ordinary button, since saturated colour is the diagnostics'.
fn restart_line(ui: &mut egui::Ui, state: &mut EditorState) {
    let pending = state
        .settings
        .startup
        .changes_from(&state.started_with, &state.environment);
    if pending == 0 {
        return;
    }
    let text = match pending {
        1 => "Restart Orrin to apply 1 change".to_owned(),
        n => format!("Restart Orrin to apply {n} changes"),
    };
    let label = if state.scene_dirty() {
        "Save and restart"
    } else {
        "Restart now"
    };
    let playing = state.playing();
    let weak = ui.visuals().weak_text_color();
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(text).color(weak));
        let button = ui
            .add_enabled(!playing, egui::Button::new(label))
            .on_disabled_hover_text(PLAYING);
        if button.clicked() {
            state.request_restart();
        }
    });
    ui.add_space(10.0);
}

fn heading(ui: &mut egui::Ui, title: &str, scope: &str) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(title).strong());
        ui.label(egui::RichText::new(scope).color(ui.visuals().weak_text_color()));
    });
    ui.add_space(2.0);
}

/// A line under a row, level with its label, for a consequence the row
/// cannot show: an override, when a change takes effect, a refusal.
fn note(ui: &mut egui::Ui, text: &str, colour: egui::Color32) {
    ui.horizontal(|ui| {
        ui.add_space(8.0);
        ui.add(egui::Label::new(egui::RichText::new(text).small().color(colour)).wrap());
    });
    ui.add_space(2.0);
}

/// Hover help over a row's label, which `widgets::property` paints rather
/// than adds as a widget. Starts past the changed marker, which says its own.
fn help(ui: &egui::Ui, top: egui::Pos2, label: f32, key: &str, text: &str) {
    let height = ui.spacing().interact_size.y;
    let rect = egui::Rect::from_min_max(
        egui::pos2(top.x + 8.0, top.y),
        egui::pos2(top.x + label, top.y + height),
    );
    ui.interact(
        rect,
        ui.id().with(("settings help", key)),
        egui::Sense::hover(),
    )
    .on_hover_text(text);
}

fn orrin_row(ui: &mut egui::Ui, cx: &mut Cx, panel: &mut SettingsPanel, row: &Row) {
    let field = row.field;
    let env = &cx.state.environment;
    let locked = match field {
        Field::Editor => env.editor_override(),
        Field::BlenderPath => env.blender_override(),
        Field::Remote => env.remote_override(),
        Field::Backend => env.backend_override(),
        Field::Threads => env.threads_override(),
        Field::Validation => env.validation_override(),
        Field::Log => env.log_override(),
        _ => None,
    };
    let built = cfg!(feature = "scripting") || !matches!(field, Field::Editor | Field::Reload);
    let live = locked.is_none() && built;
    let changed = !is_default(field, &cx.state.settings, &cx.defaults);
    let default = default_text(field, &cx.defaults);
    let weak = ui.visuals().weak_text_color();
    let top = ui.cursor().min;
    let label = label_width(ui);
    let mut reset = false;
    widgets::property_with(
        ui,
        row.label,
        |ui| {
            ui.add_enabled_ui(live, |ui| {
                if locked.is_some() && field.is_startup() {
                    let mut chosen = cx.state.environment.over(&cx.state.settings.startup);
                    startup_control(ui, field, &mut chosen);
                } else if field == Field::Remote
                    && let Some(mut on) = locked.and(cx.state.environment.remote())
                {
                    ui.checkbox(&mut on, "");
                } else if field == Field::BlenderPath
                    && let Some(path) = locked.and(cx.state.environment.blender())
                {
                    let mut shown = path.to_owned();
                    ui.add(egui::TextEdit::singleline(&mut shown).desired_width(f32::INFINITY));
                } else {
                    control(ui, cx, panel, field);
                }
            });
        },
        |ui| {
            if changed && live {
                reset = icons::button(ui, icons::rotate_ccw(), 12.0, weak, false)
                    .on_hover_text(format!("Reset to {default}"))
                    .clicked();
            }
        },
    );
    help(ui, top, label, row.label, row.help);
    if changed {
        let height = ui.spacing().interact_size.y;
        let centre = egui::pos2(top.x + 3.5, top.y + height / 2.0);
        ui.painter()
            .circle_filled(centre, 2.5, ui.visuals().selection.stroke.color);
        let rect = egui::Rect::from_center_size(centre, egui::vec2(7.0, height));
        ui.interact(
            rect,
            ui.id().with(("settings changed", row.label)),
            egui::Sense::hover(),
        )
        .on_hover_text(format!("Changed; the default is {default}"));
    }
    if reset {
        if field == Field::Theme {
            cx.state.request_theme(theme::BUILT_IN);
        } else {
            if field == Field::UiScale {
                panel.scale = None;
            }
            restore(field, &mut cx.state.settings, &cx.defaults);
        }
    }
    if let Some(var) = locked {
        note(ui, &format!("Set by {var} for this run"), weak);
    }
    if field.is_startup() {
        note(ui, "Applies on restart", weak);
    }
    if field == Field::Threads {
        let auto = crate::threads::automatic();
        note(ui, &format!("Auto is {auto} on this computer"), weak);
    }
    if field == Field::Validation && cx.state.backend == crate::gfx::BackendChoice::Metal {
        note(
            ui,
            "Validation is Vulkan's; this run draws through Metal",
            weak,
        );
    }
    if row.page == Page::Blender {
        blender_note(ui, cx, panel, field, locked.is_some());
    }
    if field == Field::Remote {
        let (line, colour) = remote_line(cx.state);
        note(ui, &line, colour.unwrap_or(weak));
        connections(ui, cx.state);
    }
    if field == Field::Editor
        && let ExternalEditor::Custom(command) = &cx.state.settings.scripting.editor
    {
        let current = command.clone();
        let top = ui.cursor().min;
        let label = label_width(ui);
        let typed = widgets::property(ui, "Command", |ui| {
            ui.add_enabled_ui(live, |ui| {
                committed_text(ui, &mut panel.typing, Field::Command, &current)
            })
            .inner
        });
        help(
            ui,
            top,
            label,
            "Command",
            "Run with the script, split on spaces",
        );
        if let Some(command) = typed {
            cx.state.settings.scripting.editor = ExternalEditor::Custom(command);
        }
        note(
            ui,
            "{file} and {line} become the script and the line; the file is added at the end \
             when nothing names it",
            weak,
        );
    }
}

/// What only a Blender row can say: which Blender runs, what the link is
/// doing, how the last install went.
fn blender_note(ui: &mut egui::Ui, cx: &Cx, panel: &mut SettingsPanel, field: Field, locked: bool) {
    let weak = ui.visuals().weak_text_color();
    match field {
        Field::BlenderPath => match panel.blender(cx.state) {
            (None, _) => note(
                ui,
                "No Blender was found: set its path, or install Blender",
                theme::ERROR,
            ),
            (Some(path), false) => note(
                ui,
                &format!("Not found: {} is not a file", path.display()),
                theme::ERROR,
            ),
            (Some(path), true) if !locked && cx.state.settings.blender.path.is_none() => {
                note(ui, &format!("Using {}", path.display()), weak);
            }
            _ => {}
        },
        Field::LiveLink => {
            let (line, colour) = link_line(cx.world, cx.state);
            note(ui, &line, colour.unwrap_or(weak));
        }
        Field::AddOn => match cx.state.addon.last() {
            Some(Ok(done)) => note(ui, &sentence(done), weak),
            Some(Err(failed)) => note(ui, &sentence(failed), theme::ERROR),
            None => {}
        },
        _ => {}
    }
}

/// The live link's state in a line, and a colour when it is a failure.
fn link_line(world: &World, state: &EditorState) -> (String, Option<egui::Color32>) {
    if !state.settings.blender.live_link {
        return ("Off".to_owned(), None);
    }
    if let Some(error) = &state.link_error {
        return (sentence(error), Some(theme::ERROR));
    }
    let Some(db) = world.get_resource::<AssetDatabase>() else {
        return ("Listens while a project is open".to_owned(), None);
    };
    let line = match db.link_status() {
        None => "Not listening".to_owned(),
        Some(status) if status.peers == 0 => format!("Listening on 127.0.0.1:{}", status.port),
        Some(status) => {
            let who = match status.peers {
                1 => "1 Blender connected".to_owned(),
                n => format!("{n} Blenders connected"),
            };
            match status.live.as_slice() {
                [] => who,
                live => format!("{who}, streaming {}", live.join(", ")),
            }
        }
    };
    (line, None)
}

/// The remote's state in a line, and a colour when it is a failure.
fn remote_line(state: &EditorState) -> (String, Option<egui::Color32>) {
    match &state.remote {
        None => ("Off".to_owned(), None),
        Some(Err(error)) => (sentence(error), Some(theme::ERROR)),
        Some(Ok(status)) => {
            let line = match status.peers {
                0 => format!("Listening on 127.0.0.1:{}", status.port),
                1 => format!("1 agent connected, on 127.0.0.1:{}", status.port),
                n => format!("{n} agents connected, on 127.0.0.1:{}", status.port),
            };
            (line, None)
        }
    }
}

/// Each agent connected: what client it says it is, for how long, how busy
/// it has been, and a way to end that one connection.
fn connections(ui: &mut egui::Ui, state: &mut EditorState) {
    let weak = ui.visuals().weak_text_color();
    let mut close = None;
    for peer in &state.remote_peers {
        let last = peer
            .last
            .as_deref()
            .map_or_else(String::new, |method| format!(" · last {method}"));
        let requests = match peer.requests {
            1 => "1 request".to_owned(),
            n => format!("{n} requests"),
        };
        let line = format!("{} · {requests}{last}", for_how_long(peer.since.elapsed()));
        let client = peer.client.as_deref().unwrap_or("unknown client");
        ui.horizontal(|ui| {
            ui.add_space(8.0);
            // The button first, from the right, so it is whole at any width
            // and the text gives way to it.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("Disconnect").clicked() {
                    close = Some(peer.id);
                }
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    ui.add(egui::Label::new(egui::RichText::new(client).strong()).truncate());
                    let detail = egui::RichText::new(&line).small().color(weak);
                    ui.add(egui::Label::new(detail).truncate());
                });
            });
        });
    }
    if close.is_some() {
        state.remote_disconnect = close;
    }
}

/// How long a connection has been open, to the coarsest unit that says it.
fn for_how_long(open: Duration) -> String {
    match open.as_secs() {
        seconds @ 0..60 => format!("{seconds} s"),
        seconds @ 60..3600 => format!("{} min", seconds / 60),
        seconds => format!("{} h {} min", seconds / 3600, seconds % 3600 / 60),
    }
}

/// What `.mcp.json` holds to run `orrin mcp` on the project at `root`: the
/// `orrin` this editor would run an export with when it can find one, the
/// one on the PATH otherwise.
fn mcp_config(root: &Path) -> String {
    let beside = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    let cli = crate::editor::export::find_cli(
        std::env::var_os(crate::editor::export::CLI_ENV),
        beside.as_deref(),
        std::env::var_os("PATH"),
    )
    .map_or_else(|_| "orrin".to_owned(), |cli| cli.display().to_string());
    let config = serde_json::json!({
        "mcpServers": {
            "orrin": {
                "command": cli,
                "args": ["mcp", "--project", root.display().to_string()],
            }
        }
    });
    serde_json::to_string_pretty(&config).unwrap_or_default()
}

/// A console line as a note on the page: its first letter capitalised.
fn sentence(line: &str) -> String {
    let mut chars = line.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().chain(chars).collect()
    })
}

/// A text field that commits on Enter or on losing focus and gives back what
/// it held on Escape. Returns the text to commit.
fn committed_text(
    ui: &mut egui::Ui,
    typing: &mut Vec<(Field, String)>,
    field: Field,
    current: &str,
) -> Option<String> {
    let held = typing.iter().position(|(typed, _)| *typed == field);
    let mut text = held.map_or_else(|| current.to_owned(), |at| typing.swap_remove(at).1);
    let response =
        ui.add(egui::TextEdit::singleline(&mut text).desired_width(ui.available_width()));
    if response.has_focus() {
        typing.push((field, text));
        return None;
    }
    let escaped = ui.input(|input| input.key_pressed(egui::Key::Escape));
    (response.lost_focus() && !escaped && text != current).then_some(text)
}

fn control(ui: &mut egui::Ui, cx: &mut Cx, panel: &mut SettingsPanel, field: Field) {
    let width = ui.available_width();
    let settings = &mut cx.state.settings;
    match field {
        Field::Theme => {
            let current = settings.appearance.theme_name().to_owned();
            let mut picked = None;
            egui::ComboBox::from_id_salt("settings theme")
                .width(width)
                .truncate()
                .selected_text(&current)
                .show_ui(ui, |ui| {
                    for name in cx.themes.names() {
                        if ui.selectable_label(name == current, name).clicked() {
                            picked = Some(name.to_owned());
                        }
                    }
                });
            if let Some(name) = picked {
                cx.state.request_theme(name);
            }
        }
        Field::UiScale => {
            let mut value = panel.scale.unwrap_or(settings.appearance.ui_scale);
            let slider = egui::Slider::new(&mut value, UI_SCALE)
                .step_by(0.05)
                .fixed_decimals(2)
                .suffix("×");
            if ui.add(slider).changed() {
                panel.scale = Some(value);
            }
            if !ui.input(|input| input.pointer.any_down())
                && let Some(value) = panel.scale.take()
            {
                settings.appearance.ui_scale = value;
            }
        }
        Field::LightIcons
        | Field::SoundIcons
        | Field::ListenerIcons
        | Field::CameraIcons
        | Field::EmptyIcons => {
            if let Some(rgb) = tint_mut(&mut settings.appearance.icons, field) {
                ui.color_edit_button_srgb(rgb);
            }
        }
        Field::FlySpeed => {
            let speed = &mut settings.viewport.fly_speed;
            ui.add(
                egui::Slider::new(speed, FLY_SPEED)
                    .logarithmic(true)
                    .suffix(" m/s"),
            );
        }
        Field::Boost => {
            ui.add(egui::Slider::new(&mut settings.viewport.boost, BOOST).suffix("×"));
        }
        Field::Sensitivity => {
            let sensitivity = &mut settings.viewport.sensitivity;
            ui.add(
                egui::Slider::new(sensitivity, SENSITIVITY)
                    .logarithmic(true)
                    .suffix("×"),
            );
        }
        Field::ScrollSpeed => {
            let scroll = &mut settings.viewport.scroll_speed;
            ui.add(
                egui::Slider::new(scroll, SCROLL_SPEED)
                    .logarithmic(true)
                    .suffix(" m"),
            );
        }
        Field::Editor => {
            let editor = &mut settings.scripting.editor;
            egui::ComboBox::from_id_salt("settings editor")
                .width(width)
                .selected_text(editor_label(editor))
                .show_ui(ui, |ui| {
                    for preset in [
                        ExternalEditor::Auto,
                        ExternalEditor::VsCode,
                        ExternalEditor::Rider,
                        ExternalEditor::Zed,
                    ] {
                        let label = editor_label(&preset);
                        if ui.selectable_label(*editor == preset, label).clicked() {
                            *editor = preset;
                        }
                    }
                    let custom = matches!(editor, ExternalEditor::Custom(_));
                    if ui.selectable_label(custom, "Custom").clicked() && !custom {
                        *editor = ExternalEditor::Custom(String::new());
                    }
                });
        }
        Field::Reload => {
            ui.checkbox(&mut settings.scripting.reload_after_rebuild, "");
        }
        Field::BlenderPath => {
            let current = settings
                .blender
                .path
                .as_ref()
                .map_or_else(String::new, |path| path.display().to_string());
            if let Some(typed) = committed_text(ui, &mut panel.typing, field, &current) {
                let typed = typed.trim();
                settings.blender.path = (!typed.is_empty()).then(|| PathBuf::from(typed));
            }
        }
        Field::LiveLink => {
            ui.checkbox(&mut settings.blender.live_link, "");
        }
        Field::LinkPort => {
            let current = settings.blender.port.to_string();
            if let Some(port) = committed_text(ui, &mut panel.typing, field, &current)
                .and_then(|typed| typed.trim().parse::<u16>().ok())
                .filter(|&port| port != 0)
            {
                settings.blender.port = port;
            }
        }
        Field::AddOn => {
            let (blender, exists) = panel.blender(cx.state);
            let blender = blender.filter(|_| exists);
            let running = cx.state.addon.is_running();
            let label = if running {
                "Installing…"
            } else {
                "Install add-on"
            };
            let button = ui
                .add_enabled(!running && blender.is_some(), egui::Button::new(label))
                .on_disabled_hover_text("No Blender was found to install it into");
            if button.clicked()
                && let Some(blender) = blender
            {
                cx.state.addon.start(blender);
            }
        }
        Field::Remote => {
            ui.checkbox(&mut settings.agents.remote, "");
        }
        Field::McpConfig => {
            let root = cx.state.project_root.clone();
            let button = ui
                .add_enabled(root.is_some(), egui::Button::new("Copy MCP config"))
                .on_disabled_hover_text("Open a project first: the config names it");
            if button.clicked()
                && let Some(root) = root
            {
                ui.ctx().copy_text(mcp_config(&root));
            }
        }
        Field::Autosave => {
            ui.checkbox(&mut settings.saving.autosave, "");
        }
        Field::Interval => {
            let on = settings.saving.autosave;
            let every = &mut settings.saving.autosave_interval_secs;
            ui.add_enabled(
                on,
                egui::Slider::new(every, AUTOSAVE_SECS)
                    .logarithmic(true)
                    .suffix(" s"),
            );
        }
        Field::Present => {
            let vsync = &mut settings.display.vsync;
            egui::ComboBox::from_id_salt("settings present")
                .width(width)
                .truncate()
                .selected_text(vsync_label(*vsync))
                .show_ui(ui, |ui| {
                    for mode in VSYNC_MODES {
                        ui.selectable_value(vsync, mode, vsync_label(mode));
                    }
                });
        }
        Field::Images => {
            ui.add(egui::Slider::new(
                &mut settings.display.images,
                SWAPCHAIN_IMAGES,
            ));
        }
        Field::Backend | Field::Threads | Field::Validation | Field::Log => {
            startup_control(ui, field, &mut settings.startup);
        }
        Field::Name
        | Field::Version
        | Field::Location
        | Field::Start
        | Field::Persistent
        | Field::Entry
        | Field::ScriptsFolder
        | Field::AssetsFolder
        | Field::Command
        | Field::AddFolder => {}
        Field::Preset => {
            let allow = &mut settings.agents.allow;
            let current = allow.preset();
            egui::ComboBox::from_id_salt("settings agents preset")
                .width(width)
                .selected_text(current.label())
                .show_ui(ui, |ui| {
                    for preset in Preset::CHOSEN {
                        if ui
                            .selectable_label(current == preset, preset.label())
                            .clicked()
                        {
                            *allow = Capabilities::of(preset);
                        }
                    }
                });
        }
        Field::Allow(capability) => {
            let mut on = settings.agents.allow.allows(capability);
            if ui.checkbox(&mut on, "").changed() {
                settings.agents.allow.set(capability, on);
            }
        }
    }
}

/// The project's folder rules for agents, under the switches: a preset, what
/// every folder no rule names gets, and one rule a folder, the longest that
/// matches a path deciding it. Kept in the project's `.orrin`, not in
/// `orrin.toml`: they are this person's rules for their agent, not the
/// project's.
fn folders(ui: &mut egui::Ui, state: &mut EditorState, panel: &mut SettingsPanel) {
    let weak = ui.visuals().weak_text_color();
    ui.add_space(14.0);
    heading(ui, "Folders", "this project, kept in .orrin");
    if state.project_root.is_none() {
        note(ui, "Open a project to give its folders rules", weak);
        return;
    }
    let assets = state.agent_assets.clone();
    let folders = match &mut state.agent_folders {
        Ok(folders) => folders,
        Err(why) => {
            note(ui, &sentence(why), theme::ERROR);
            note(ui, "Agents are refused every folder until it reads", weak);
            if ui.button("Reset to the whole project").clicked() {
                state.agent_folders = Ok(Folders::default());
                state.agent_folders_changed = true;
            }
            return;
        }
    };
    let mut changed = false;
    let current = folders.preset(&assets);
    widgets::property(ui, "Folder preset", |ui| {
        egui::ComboBox::from_id_salt("settings agents folders")
            .width(ui.available_width())
            .selected_text(current.label())
            .show_ui(ui, |ui| {
                for preset in FolderPreset::CHOSEN {
                    if ui
                        .selectable_label(current == preset, preset.label())
                        .clicked()
                    {
                        *folders = Folders::of(preset, &assets);
                        changed = true;
                    }
                }
            });
    });
    widgets::property(ui, "Everything else", |ui| {
        changed |= access_picker(ui, "unlisted", &mut folders.unlisted);
    });
    let mut gone = None;
    for (at, rule) in folders.rules.iter_mut().enumerate() {
        let shown = if rule.path.is_empty() {
            "."
        } else {
            &rule.path
        };
        widgets::property_with(
            ui,
            shown,
            |ui| changed |= access_picker(ui, &rule.path, &mut rule.access),
            |ui| {
                if icons::button(ui, icons::x(), 12.0, weak, false)
                    .on_hover_text("Remove this rule")
                    .clicked()
                {
                    gone = Some(at);
                }
            },
        );
    }
    if let Some(at) = gone {
        folders.rules.remove(at);
        changed = true;
    }
    let typed = widgets::property(ui, "Add folder", |ui| {
        committed_text(ui, &mut panel.typing, Field::AddFolder, "")
    });
    if let Some(typed) = typed.filter(|typed| !typed.trim().is_empty()) {
        match orrin_remote::permissions::normalise(typed.trim()) {
            Ok(path) if folders.rules.iter().any(|rule| rule.path == path) => {
                panel.folder_error = Some(format!("{path} has a rule already"));
            }
            Ok(path) => {
                folders.rules.push(FolderRule {
                    path,
                    access: Access::Read,
                });
                folders.rules.sort_by(|a, b| a.path.cmp(&b.path));
                panel.folder_error = None;
                changed = true;
            }
            Err(why) => panel.folder_error = Some(sentence(&why)),
        }
    }
    if let Some(error) = &panel.folder_error {
        note(ui, error, theme::ERROR);
    }
    note(
        ui,
        "Relative to the project, longest match first. These govern Orrin's tools only, not a \
         process that can already write the project's files",
        weak,
    );
    if changed {
        state.agent_folders_changed = true;
    }
}

/// An access level, picked; whether it changed.
fn access_picker(ui: &mut egui::Ui, salt: &str, access: &mut Access) -> bool {
    let before = *access;
    egui::ComboBox::from_id_salt(("settings agents access", salt.to_owned()))
        .width(ui.available_width())
        .selected_text(access.label())
        .show_ui(ui, |ui| {
            for option in Access::ALL {
                ui.selectable_value(access, option, option.label());
            }
        });
    *access != before
}

/// A Startup row's picker over `startup`, which a locked row hands in as a
/// copy holding the environment's values.
fn startup_control(ui: &mut egui::Ui, field: Field, startup: &mut Startup) {
    let combo = |salt: &str, shown: &str| {
        egui::ComboBox::from_id_salt(("settings startup", salt.to_owned()))
            .width(ui.available_width())
            .selected_text(shown.to_owned())
    };
    match field {
        Field::Backend => {
            let backend = &mut startup.backend;
            combo("backend", backend_label(*backend)).show_ui(ui, |ui| {
                for &option in BACKENDS {
                    ui.selectable_value(backend, option, backend_label(option));
                }
            });
        }
        Field::Threads => {
            let threads = &mut startup.threads;
            combo("threads", &threads_label(*threads)).show_ui(ui, |ui| {
                ui.selectable_value(threads, None, "Auto");
                let cores = u32::try_from(crate::threads::cores()).unwrap_or(u32::MAX);
                for count in 1..=cores {
                    ui.selectable_value(threads, Some(count), count.to_string());
                }
            });
        }
        Field::Validation => {
            let validation = &mut startup.validation;
            combo("validation", validation_label(*validation)).show_ui(ui, |ui| {
                for option in VALIDATIONS {
                    ui.selectable_value(validation, option, validation_label(option));
                }
            });
        }
        Field::Log => {
            let log = &mut startup.log;
            combo("log", log_label(*log)).show_ui(ui, |ui| {
                for option in LOG_FLOORS {
                    ui.selectable_value(log, option, log_label(option));
                }
            });
        }
        _ => {}
    }
}

fn backend_label(backend: Backend) -> &'static str {
    match backend {
        Backend::Auto => "Auto",
        Backend::Vulkan => "Vulkan",
        Backend::Metal => "Metal",
    }
}

fn threads_label(threads: Option<u32>) -> String {
    threads.map_or_else(|| "Auto".to_owned(), |count| count.to_string())
}

fn validation_label(validation: Validation) -> &'static str {
    match validation {
        Validation::Auto => "Auto",
        Validation::On => "On",
        Validation::Off => "Off",
    }
}

fn log_label(log: LogFloor) -> &'static str {
    match log {
        LogFloor::Default => "Default",
        LogFloor::Trace => "Trace",
        LogFloor::Debug => "Debug",
        LogFloor::Info => "Info",
        LogFloor::Warn => "Warn",
        LogFloor::Error => "Error",
    }
}

fn editor_label(editor: &ExternalEditor) -> &'static str {
    match editor {
        ExternalEditor::Auto => "Auto",
        ExternalEditor::VsCode => "VS Code",
        ExternalEditor::Rider => "Rider",
        ExternalEditor::Zed => "Zed",
        ExternalEditor::Custom(_) => "Custom",
    }
}

fn tint_mut(tints: &mut IconTints, field: Field) -> Option<&mut [u8; 3]> {
    Some(match field {
        Field::LightIcons => &mut tints.light,
        Field::SoundIcons => &mut tints.sound,
        Field::ListenerIcons => &mut tints.listener,
        Field::CameraIcons => &mut tints.camera,
        Field::EmptyIcons => &mut tints.empty,
        _ => return None,
    })
}

fn tint(mut tints: IconTints, field: Field) -> Option<[u8; 3]> {
    tint_mut(&mut tints, field).copied()
}

fn is_default(field: Field, s: &Settings, d: &Settings) -> bool {
    match field {
        Field::Theme => s.appearance.theme.is_none(),
        Field::UiScale => s.appearance.ui_scale == d.appearance.ui_scale,
        Field::FlySpeed => s.viewport.fly_speed == d.viewport.fly_speed,
        Field::Boost => s.viewport.boost == d.viewport.boost,
        Field::Sensitivity => s.viewport.sensitivity == d.viewport.sensitivity,
        Field::ScrollSpeed => s.viewport.scroll_speed == d.viewport.scroll_speed,
        Field::Editor => s.scripting.editor == d.scripting.editor,
        Field::Reload => s.scripting.reload_after_rebuild == d.scripting.reload_after_rebuild,
        Field::Autosave => s.saving.autosave == d.saving.autosave,
        Field::Interval => s.saving.autosave_interval_secs == d.saving.autosave_interval_secs,
        Field::Present => s.display.vsync == d.display.vsync,
        Field::Images => s.display.images == d.display.images,
        Field::Backend => s.startup.backend == d.startup.backend,
        Field::Threads => s.startup.threads == d.startup.threads,
        Field::Validation => s.startup.validation == d.startup.validation,
        Field::Log => s.startup.log == d.startup.log,
        Field::BlenderPath => s.blender.path == d.blender.path,
        Field::LiveLink => s.blender.live_link == d.blender.live_link,
        Field::LinkPort => s.blender.port == d.blender.port,
        Field::AddOn | Field::McpConfig => true,
        Field::Preset => s.agents.allow == d.agents.allow,
        Field::Allow(capability) => {
            s.agents.allow.allows(capability) == d.agents.allow.allows(capability)
        }
        Field::Remote => s.agents.remote == d.agents.remote,
        _ => tint(s.appearance.icons, field) == tint(d.appearance.icons, field),
    }
}

/// Put `field` back to its default. The theme is not here: a pick goes
/// through `EditorState::request_theme`.
fn restore(field: Field, s: &mut Settings, d: &Settings) {
    match field {
        Field::UiScale => s.appearance.ui_scale = d.appearance.ui_scale,
        Field::FlySpeed => s.viewport.fly_speed = d.viewport.fly_speed,
        Field::Boost => s.viewport.boost = d.viewport.boost,
        Field::Sensitivity => s.viewport.sensitivity = d.viewport.sensitivity,
        Field::ScrollSpeed => s.viewport.scroll_speed = d.viewport.scroll_speed,
        Field::Editor => s.scripting.editor = d.scripting.editor.clone(),
        Field::Reload => s.scripting.reload_after_rebuild = d.scripting.reload_after_rebuild,
        Field::Autosave => s.saving.autosave = d.saving.autosave,
        Field::Interval => s.saving.autosave_interval_secs = d.saving.autosave_interval_secs,
        Field::Present => s.display.vsync = d.display.vsync,
        Field::Images => s.display.images = d.display.images,
        Field::Backend => s.startup.backend = d.startup.backend,
        Field::Threads => s.startup.threads = d.startup.threads,
        Field::Validation => s.startup.validation = d.startup.validation,
        Field::Log => s.startup.log = d.startup.log,
        Field::BlenderPath => s.blender.path.clone_from(&d.blender.path),
        Field::LiveLink => s.blender.live_link = d.blender.live_link,
        Field::LinkPort => s.blender.port = d.blender.port,
        Field::AddOn | Field::McpConfig => {}
        Field::Preset => s.agents.allow = d.agents.allow,
        Field::Allow(capability) => s
            .agents
            .allow
            .set(capability, d.agents.allow.allows(capability)),
        Field::Remote => s.agents.remote = d.agents.remote,
        _ => {
            if let (Some(to), Some(from)) = (
                tint_mut(&mut s.appearance.icons, field),
                tint(d.appearance.icons, field),
            ) {
                *to = from;
            }
        }
    }
}

/// The default with its unit, as the changed marker and the reset name it.
fn default_text(field: Field, d: &Settings) -> String {
    let on = |value: bool| if value { "on" } else { "off" }.to_owned();
    match field {
        Field::Theme => theme::BUILT_IN.to_owned(),
        Field::UiScale => format!("{:.2}×", d.appearance.ui_scale),
        Field::FlySpeed => format!("{} m/s", compact(d.viewport.fly_speed)),
        Field::Boost => format!("{}×", compact(d.viewport.boost)),
        Field::Sensitivity => format!("{}×", compact(d.viewport.sensitivity)),
        Field::ScrollSpeed => format!("{} m", compact(d.viewport.scroll_speed)),
        Field::Editor => editor_label(&d.scripting.editor).to_owned(),
        Field::Reload => on(d.scripting.reload_after_rebuild),
        Field::Autosave => on(d.saving.autosave),
        Field::Interval => format!("{} s", d.saving.autosave_interval_secs),
        Field::Present => vsync_label(d.display.vsync).to_owned(),
        Field::Images => d.display.images.to_string(),
        Field::Backend => backend_label(d.startup.backend).to_owned(),
        Field::Threads => threads_label(d.startup.threads),
        Field::Validation => validation_label(d.startup.validation).to_owned(),
        Field::Log => log_label(d.startup.log).to_owned(),
        Field::BlenderPath => "empty, an installed Blender".to_owned(),
        Field::LiveLink => on(d.blender.live_link),
        Field::LinkPort => d.blender.port.to_string(),
        Field::AddOn | Field::McpConfig => String::new(),
        Field::Preset => d.agents.allow.preset().label().to_owned(),
        Field::Allow(capability) => on(d.agents.allow.allows(capability)),
        Field::Remote => on(d.agents.remote),
        _ => tint(d.appearance.icons, field)
            .map_or_else(String::new, |[r, g, b]| format!("#{r:02X}{g:02X}{b:02X}")),
    }
}

fn project_row(ui: &mut egui::Ui, cx: &mut Cx, panel: &mut SettingsPanel, row: &Row) {
    let Some(project) = panel.manifest.project() else {
        return;
    };
    let field = row.field;
    let weak = ui.visuals().weak_text_color();
    let playing = cx.state.playing();
    let missing = match field {
        Field::Start => panel.manifest.missing[0],
        Field::Persistent => panel.manifest.missing[1],
        _ => false,
    };
    let top = ui.cursor().min;
    let label = label_width(ui);
    let typing = &mut panel.typing;
    let change = widgets::property(ui, row.label, |ui| match field {
        Field::Name => committed_text(ui, typing, field, project.name()).map(Change::Name),
        Field::Version => committed_text(ui, typing, field, project.version()).map(Change::Version),
        Field::Entry => committed_text(ui, typing, field, project.entry_type()).map(Change::Entry),
        Field::Location => {
            let root = project.root();
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Reveal").clicked() {
                    crate::editor::export::reveal(root);
                }
                let path = crate::editor::state::plain(root.display().to_string());
                ui.add(egui::Label::new(super::figures(&path).color(weak)).truncate())
                    .on_hover_text(path);
            });
            None
        }
        Field::ScriptsFolder | Field::AssetsFolder => {
            let dir = if field == Field::ScriptsFolder {
                project.scripts_dir()
            } else {
                project.assets_dir()
            };
            let rel = slashed(dir.strip_prefix(project.root()).unwrap_or(&dir));
            let shown = if rel.is_empty() { ".".to_owned() } else { rel };
            ui.add(egui::Label::new(super::figures(shown).color(weak)).truncate());
            None
        }
        Field::Start | Field::Persistent => {
            let current = if field == Field::Start {
                project.start_scene_relative()
            } else {
                project.persistent_scene_relative()
            };
            let picker = ui.add_enabled_ui(!playing, |ui| {
                scene_picker(ui, cx.world, project, field, current, missing)
            });
            picker.response.on_disabled_hover_text(PLAYING);
            picker.inner.map(|scene| {
                if field == Field::Start {
                    Change::Start(scene)
                } else {
                    Change::Persistent(scene)
                }
            })
        }
        _ => None,
    });
    help(ui, top, label, row.label, row.help);

    if missing {
        note(
            ui,
            "Missing: the file is not there; pick another or None",
            theme::ERROR,
        );
    }
    match field {
        Field::Start => note(
            ui,
            "Read by orrin export; the editor does not open it",
            weak,
        ),
        Field::Persistent => note(ui, "Loaded beside the scene from the next Play", weak),
        Field::Entry => note(ui, "Applies the next time the project is opened", weak),
        _ => {}
    }
    if let Some((failed, error)) = &panel.manifest.error
        && *failed == field
    {
        note(ui, error, theme::ERROR);
    }
    if let Some(change) = change {
        panel.manifest.commit(change, cx.state);
    }
}

/// A scene from the project's scene assets, or None; never a typed path.
/// Returns the pick, if one was made.
fn scene_picker(
    ui: &mut egui::Ui,
    world: &World,
    project: &Project,
    field: Field,
    current: Option<&Path>,
    missing: bool,
) -> Option<Option<PathBuf>> {
    let shown = current.map(slashed);
    let text = match (&shown, missing) {
        (None, _) => "None".to_owned(),
        (Some(scene), false) => scene.clone(),
        (Some(scene), true) => format!("{scene} (missing)"),
    };
    let mut picked = None;
    egui::ComboBox::from_id_salt(("settings scene", field == Field::Start))
        .width(ui.available_width())
        .truncate()
        .selected_text(text)
        .show_ui(ui, |ui| {
            if ui.selectable_label(shown.is_none(), "None").clicked() {
                picked = Some(None);
            }
            let Some(db) = world.get_resource::<AssetDatabase>() else {
                return;
            };
            let mut scenes: Vec<String> = db
                .iter()
                .filter(|entry| entry.kind == AssetKind::Scene)
                .map(|entry| in_manifest(project, &entry.path))
                .collect();
            scenes.sort_unstable();
            for scene in scenes {
                let on = shown.as_deref() == Some(scene.as_str());
                if ui.selectable_label(on, &scene).clicked() {
                    picked = Some(Some(PathBuf::from(scene)));
                }
            }
        });
    picked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_db::tests::Dir;

    fn project(assets: &str) -> (Dir, Project) {
        let dir = Dir::new(&format!("settings-paths-{}", assets.replace('/', "-")));
        dir.write(
            "orrin.toml",
            format!(
                "format_version = 1\n[project]\nname = \"p\"\nversion = \"1\"\n\
                 [scripts]\ndir = \"scripts\"\nentry = \"P.Main, P\"\n[assets]\ndir = \"{assets}\"\n"
            )
            .as_bytes(),
        );
        let project = Project::load(&dir.path().join("orrin.toml")).unwrap();
        (dir, project)
    }

    /// The asset database lists a scene relative to the assets folder; the
    /// manifest names it relative to the project root, wherever the assets
    /// folder is.
    #[test]
    fn a_scene_asset_is_named_relative_to_the_project_root() {
        for (assets, expected) in [
            ("assets", "assets/levels/one.orrin"),
            ("content/art", "content/art/levels/one.orrin"),
            (".", "levels/one.orrin"),
            ("./assets", "assets/levels/one.orrin"),
        ] {
            let (_dir, project) = project(assets);
            assert_eq!(
                in_manifest(&project, "levels/one.orrin"),
                expected,
                "{assets}"
            );
        }
    }

    #[test]
    fn every_orrin_row_resets_to_what_it_calls_its_default() {
        let defaults = Settings::default();
        let mut moved = Settings::default();
        moved.appearance.ui_scale = 1.5;
        moved.appearance.icons = IconTints {
            light: [1, 2, 3],
            sound: [1, 2, 3],
            listener: [1, 2, 3],
            camera: [1, 2, 3],
            empty: [1, 2, 3],
        };
        moved.viewport.fly_speed = 20.0;
        moved.viewport.boost = 9.0;
        moved.viewport.sensitivity = 3.0;
        moved.viewport.scroll_speed = 7.0;
        moved.scripting.editor = ExternalEditor::Zed;
        moved.scripting.reload_after_rebuild = !defaults.scripting.reload_after_rebuild;
        moved.saving.autosave = !defaults.saving.autosave;
        moved.saving.autosave_interval_secs = 600;
        moved.display.vsync = crate::scene::VsyncMode::Immediate;
        moved.display.images = 4;
        moved.startup = crate::editor::settings::Startup {
            backend: crate::editor::settings::Backend::Vulkan,
            threads: Some(3),
            validation: crate::editor::settings::Validation::On,
            log: crate::editor::settings::LogFloor::Warn,
        };
        moved.blender = crate::editor::settings::Blender {
            path: Some("/opt/blender/blender".into()),
            live_link: !defaults.blender.live_link,
            port: 47200,
        };
        moved.agents.remote = !defaults.agents.remote;
        for capability in orrin_remote::permissions::Capability::ALL {
            moved.agents.allow.set(capability, false);
        }
        for row in ROWS.iter().filter(|row| !row.page.is_project()) {
            // A pick goes through a request; installing and copying are
            // actions; a preset is every switch below it at once.
            if matches!(
                row.field,
                Field::Theme | Field::AddOn | Field::McpConfig | Field::Preset
            ) {
                continue;
            }
            assert!(!is_default(row.field, &moved, &defaults), "{}", row.label);
            restore(row.field, &mut moved, &defaults);
            assert!(is_default(row.field, &moved, &defaults), "{}", row.label);
            assert!(
                !default_text(row.field, &defaults).is_empty(),
                "{}",
                row.label
            );
        }
        assert_eq!(moved, defaults, "every row together is every setting");
    }
}
