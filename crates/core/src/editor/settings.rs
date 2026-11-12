//! How the person at this computer likes Orrin, in every project: the theme,
//! the fly camera, which editor opens a script. Not what a project *is* — that
//! is `orrin.toml`, checked in and shared — and not where this project's tools
//! were left, which is `prefs` and lives in `<project>/.orrin/editor.ron`.
//!
//! TOML where `editor.ron` is RON, because this file is meant to be read and
//! edited by hand and holds no trees of enums. It is
//! `<user dir>/settings.toml`, with the user's own themes in `themes/` beside
//! it (see `orrin_project::user`).
//!
//! **Deviations only.** A value equal to its default is not written, as a
//! material's shader values and a Behaviour's fields across a reload are not,
//! so a default the engine changes later reaches everyone who never set it.
//! All defaults is an empty file, or none.
//!
//! **The directory is handed in, never looked up.** `app.rs` resolves it for
//! the windowed editor and nothing else does, so a test, a capture or the perf
//! harness never runs under the settings of whoever started it. `None` reads
//! and writes nothing. The engine's own defaults — `PresentSettings` above all,
//! which every recorded measurement assumes — are therefore unchanged; only the
//! editor applies a saved choice over them.
//!
//! **The environment still wins**, as it does everywhere else in the engine:
//! env, then settings, then the built-in default. [`Environment`] is what the
//! UI asks to say that a setting is overridden this run, and by what.
//!
//! **A file that does not parse is somebody's hand edit.** It is named, the
//! session runs on defaults, and before the next save it is moved aside to
//! `settings.toml.bad` rather than written over. A key this build does not know
//! is ignored, so an older engine still opens after a newer one wrote the file.
//!
//! **Panels edit `EditorState::settings`** and [`SettingsStore::settle`], at
//! the end of every editor frame, applies what changed and keeps it. The UI
//! scale is egui's zoom factor rather than its pixels per point, because the
//! zoom multiplies the display's own scale, which `egui_winit` refreshes when
//! the window moves to another screen; a fixed pixels per point would be wrong
//! on the second one.
//!
//! **Startup is read before there is an editor.** The backend, the worker
//! threads, validation and the log floor were environment variables alone,
//! read once as the process starts, so `App::run` opens the store first —
//! before the log stream exists, which is why [`SettingsStore::open`] hands
//! back what it has to say instead of logging it. They apply on restart:
//! `EditorState::started_with` keeps what this run began with, and the
//! Settings tab counts the difference.

use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::time::Duration;

use orrin_ecs::World;
use serde::{Deserialize, Serialize};

use super::state::EditorState;
pub use super::state::IconTints;
use super::theme::ThemeSet;
pub use crate::camera_controller::FlyCamera;
use crate::gfx::BackendChoice;
use crate::scene::{LogLevel, PresentSettings, VsyncMode};

const FILE: &str = "settings.toml";
const BAD: &str = "settings.toml.bad";
const THEMES: &str = "themes";

pub const UI_SCALE: RangeInclusive<f32> = 0.75..=2.0;
pub const FLY_SPEED: RangeInclusive<f32> = 0.1..=500.0;
pub const BOOST: RangeInclusive<f32> = 1.0..=20.0;
pub const SENSITIVITY: RangeInclusive<f32> = 0.1..=10.0;
pub const SCROLL_SPEED: RangeInclusive<f32> = 0.1..=50.0;
pub const AUTOSAVE_SECS: RangeInclusive<u32> = 15..=3600;
pub const SWAPCHAIN_IMAGES: RangeInclusive<u32> = 2..=4;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub appearance: Appearance,
    pub viewport: FlyCamera,
    pub scripting: Scripting,
    pub saving: Saving,
    pub display: Display,
    pub startup: Startup,
}

/// What a run is started with. Each default is what the engine does with no
/// `ORRIN_*` variable set.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Startup {
    #[serde(deserialize_with = "or_default")]
    pub backend: Backend,
    /// The pool's workers; `None` is one short of the machine.
    #[serde(deserialize_with = "or_default")]
    pub threads: Option<u32>,
    #[serde(deserialize_with = "or_default")]
    pub validation: Validation,
    #[serde(deserialize_with = "or_default")]
    pub log: LogFloor,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// `BackendChoice::DEFAULT`, or Vulkan where Metal cannot start.
    #[default]
    Auto,
    Vulkan,
    Metal,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Validation {
    /// On in a debug build.
    #[default]
    Auto,
    On,
    Off,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFloor {
    /// This build's own floor.
    #[default]
    Default,
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl From<LogLevel> for LogFloor {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => Self::Trace,
            LogLevel::Debug => Self::Debug,
            LogLevel::Info => Self::Info,
            LogLevel::Warning => Self::Warn,
            LogLevel::Error => Self::Error,
        }
    }
}

impl Startup {
    pub fn backend_choice(&self) -> Option<BackendChoice> {
        match self.backend {
            Backend::Auto => None,
            Backend::Vulkan => Some(BackendChoice::Vulkan),
            Backend::Metal => Some(BackendChoice::Metal),
        }
    }

    pub fn threads(&self) -> Option<usize> {
        self.threads.map(|threads| threads as usize)
    }

    pub fn validation(&self) -> Option<bool> {
        match self.validation {
            Validation::Auto => None,
            Validation::On => Some(true),
            Validation::Off => Some(false),
        }
    }

    pub fn log_floor(&self) -> Option<LogLevel> {
        Some(match self.log {
            LogFloor::Default => return None,
            LogFloor::Trace => LogLevel::Trace,
            LogFloor::Debug => LogLevel::Debug,
            LogFloor::Info => LogLevel::Info,
            LogFloor::Warn => LogLevel::Warning,
            LogFloor::Error => LogLevel::Error,
        })
    }

    /// How many values differ from `started` that the next run will take,
    /// which leaves out every one `env` decides.
    pub fn changes_from(&self, started: &Startup, env: &Environment) -> usize {
        let (now, then) = (env.over(self), env.over(started));
        [
            now.backend != then.backend,
            now.threads != then.threads,
            now.validation != then.validation,
            now.log != then.log,
        ]
        .into_iter()
        .filter(|&changed| changed)
        .count()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Appearance {
    /// A theme's name; `None` is the built-in.
    pub theme: Option<String>,
    /// A multiple of the display's own scale, within [`UI_SCALE`].
    pub ui_scale: f32,
    pub icons: IconTints,
}

impl Default for Appearance {
    fn default() -> Self {
        Self {
            theme: None,
            ui_scale: 1.0,
            icons: IconTints::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Scripting {
    #[serde(deserialize_with = "or_default")]
    pub editor: ExternalEditor,
    /// Whether a green rebuild reloads by itself. `BuildStatus`'s own default.
    pub reload_after_rebuild: bool,
}

impl Default for Scripting {
    fn default() -> Self {
        Self {
            editor: ExternalEditor::Auto,
            reload_after_rebuild: true,
        }
    }
}

/// What opens a script at a line.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExternalEditor {
    /// VS Code when it is on the PATH, else whatever the platform opens a
    /// `.cs` file with, which knows no line.
    #[default]
    Auto,
    VsCode,
    Rider,
    Zed,
    /// A command in `ORRIN_EDITOR`'s grammar: split on whitespace, `{file}`
    /// and `{line}` replaced, the file appended when nothing names it.
    Custom(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Saving {
    pub autosave: bool,
    /// Seconds between copies of unsaved work, within [`AUTOSAVE_SECS`].
    pub autosave_interval_secs: u32,
}

impl Default for Saving {
    fn default() -> Self {
        Self {
            autosave: true,
            autosave_interval_secs: super::autosave::INTERVAL.as_secs() as u32,
        }
    }
}

impl Saving {
    /// How often a copy is owed; `None` when autosave is off.
    pub fn every(&self) -> Option<Duration> {
        self.autosave
            .then(|| Duration::from_secs(self.autosave_interval_secs.into()))
    }
}

/// The windowed editor's swapchain. The engine's own default is
/// [`PresentSettings::default`], unchanged, so a capture or a measurement
/// harness never runs under somebody's saved choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Display {
    #[serde(deserialize_with = "or_default")]
    pub vsync: VsyncMode,
    /// Within [`SWAPCHAIN_IMAGES`].
    pub images: u32,
}

impl Default for Display {
    fn default() -> Self {
        PresentSettings::default().into()
    }
}

impl From<PresentSettings> for Display {
    fn from(present: PresentSettings) -> Self {
        Self {
            vsync: present.vsync,
            images: present.images,
        }
    }
}

impl Display {
    pub fn present(&self) -> PresentSettings {
        PresentSettings {
            vsync: self.vsync,
            images: self.images,
        }
    }
}

impl Settings {
    /// Brings every value back inside the range it may take; one that is not a
    /// number at all goes back to its default.
    pub fn sanitise(&mut self) {
        let fly = FlyCamera::default();
        within(&mut self.appearance.ui_scale, &UI_SCALE, 1.0);
        within(&mut self.viewport.fly_speed, &FLY_SPEED, fly.fly_speed);
        within(&mut self.viewport.boost, &BOOST, fly.boost);
        within(
            &mut self.viewport.sensitivity,
            &SENSITIVITY,
            fly.sensitivity,
        );
        within(
            &mut self.viewport.scroll_speed,
            &SCROLL_SPEED,
            fly.scroll_speed,
        );
        let saving = &mut self.saving.autosave_interval_secs;
        *saving = (*saving).clamp(*AUTOSAVE_SECS.start(), *AUTOSAVE_SECS.end());
        let images = &mut self.display.images;
        *images = (*images).clamp(*SWAPCHAIN_IMAGES.start(), *SWAPCHAIN_IMAGES.end());
        // Rayon reads zero as every core, which is what Auto exists to avoid.
        self.startup.threads = self.startup.threads.filter(|&threads| threads > 0);
    }

    /// Hands the world the settings it holds as resources.
    pub fn apply_to_world(&self, world: &World) {
        if let Some(mut present) = world.get_resource_mut::<PresentSettings>() {
            *present = self.display.present();
        }
        #[cfg(feature = "scripting")]
        if let Some(mut status) = world.get_resource_mut::<crate::build_watcher::BuildStatus>() {
            status.auto_reload = self.scripting.reload_after_rebuild;
        }
    }

    fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        let mut settings: Self = toml::from_str(text)?;
        settings.sanitise();
        Ok(settings)
    }

    /// Only what differs from the defaults: an empty string for none.
    fn to_toml(&self) -> Option<String> {
        let tables = toml::Table::try_from(self)
            .and_then(|table| Ok((table, toml::Table::try_from(Self::default())?)));
        let (mut table, defaults) = match tables {
            Ok(tables) => tables,
            Err(error) => {
                tracing::error!("cannot serialise the editor's settings: {error}");
                return None;
            }
        };
        prune(&mut table, &defaults);
        toml::to_string(&table)
            .inspect_err(|error| tracing::error!("cannot serialise the editor's settings: {error}"))
            .ok()
    }
}

impl Appearance {
    pub fn theme_name(&self) -> &str {
        self.theme.as_deref().unwrap_or(super::theme::BUILT_IN)
    }
}

fn within(value: &mut f32, range: &RangeInclusive<f32>, default: f32) {
    *value = if value.is_finite() {
        value.clamp(*range.start(), *range.end())
    } else {
        default
    };
}

fn prune(table: &mut toml::Table, defaults: &toml::Table) {
    table.retain(|key, value| match (value, defaults.get(key)) {
        (toml::Value::Table(inner), Some(toml::Value::Table(defaults))) => {
            prune(inner, defaults);
            !inner.is_empty()
        }
        (value, Some(default)) => value != default,
        (_, None) => true,
    });
}

/// A value an older engine cannot read — an editor preset added since, say —
/// costs that one setting rather than the file.
fn or_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned + Default,
{
    let value = toml::Value::deserialize(deserializer)?;
    Ok(value.clone().try_into().unwrap_or_else(|error| {
        tracing::warn!("ignoring the setting {value}: {error}");
        T::default()
    }))
}

/// What the process environment decides for this run in place of a setting.
/// Read once, by the windowed editor; a default one overrides nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Environment {
    editor: Option<String>,
    backend: Option<Backend>,
    threads: Option<u32>,
    validation: Option<Validation>,
    log: Option<LogFloor>,
}

impl Environment {
    pub fn of_process() -> Self {
        Self::from_vars(|name| std::env::var(name).ok())
    }

    /// Each variable read the way the engine reads it at startup, so a row is
    /// locked exactly when the variable decides: a thread count of zero or a
    /// misspelt level is no override, and a backend name that cannot be
    /// honoured runs Vulkan, which is what its row then shows.
    pub fn from_vars(var: impl Fn(&str) -> Option<String>) -> Self {
        let set = |name| var(name).filter(|value: &String| !value.trim().is_empty());
        Self {
            editor: set(super::source::EDITOR_ENV),
            backend: set(BackendChoice::ENV).map(|name| match BackendChoice::parse(Some(&name)) {
                Ok(BackendChoice::Metal) => Backend::Metal,
                Ok(BackendChoice::Vulkan) | Err(_) => Backend::Vulkan,
            }),
            threads: crate::threads::named(set(crate::threads::ENV).as_deref())
                .map(|threads| u32::try_from(threads).unwrap_or(u32::MAX)),
            validation: crate::app::validation_named(set(crate::app::VALIDATION_ENV).as_deref())
                .map(|on| if on { Validation::On } else { Validation::Off }),
            log: crate::logging::floor_named(set(crate::logging::ENV).as_deref())
                .ok()
                .flatten()
                .map(LogFloor::from),
        }
    }

    /// `ORRIN_EDITOR`'s value, when it holds anything.
    pub fn editor(&self) -> Option<&str> {
        self.editor.as_deref()
    }

    /// The variable that chooses the external editor this run, if one does.
    pub fn editor_override(&self) -> Option<&'static str> {
        self.editor.is_some().then_some(super::source::EDITOR_ENV)
    }

    pub fn backend_override(&self) -> Option<&'static str> {
        self.backend.is_some().then_some(BackendChoice::ENV)
    }

    pub fn threads_override(&self) -> Option<&'static str> {
        self.threads.is_some().then_some(crate::threads::ENV)
    }

    pub fn validation_override(&self) -> Option<&'static str> {
        self.validation
            .is_some()
            .then_some(crate::app::VALIDATION_ENV)
    }

    pub fn log_override(&self) -> Option<&'static str> {
        self.log.is_some().then_some(crate::logging::ENV)
    }

    /// `startup` with what this run's environment chose in place of it.
    pub fn over(&self, startup: &Startup) -> Startup {
        Startup {
            backend: self.backend.unwrap_or(startup.backend),
            threads: self.threads.or(startup.threads),
            validation: self.validation.unwrap_or(startup.validation),
            log: self.log.unwrap_or(startup.log),
        }
    }
}

/// `settings.toml` in the per-user directory, and what the session runs with.
pub struct SettingsStore {
    dir: Option<PathBuf>,
    /// What the session runs with: the file's contents once saved.
    applied: Settings,
    /// Whether there was a file to read, readable or not. Only a run with none
    /// is seeded from the project.
    found: bool,
    /// The file could not be read, so it is moved aside before the first save.
    malformed: bool,
    unsaved: bool,
    /// Whether the world has been handed `applied` yet. Until it has, a world
    /// resource that differs is the engine's default rather than an edit.
    pushed: bool,
}

impl SettingsStore {
    /// `None` reads and writes nothing, and every setting is its default. A
    /// missing file is the defaults; one that cannot be read is named and is
    /// the defaults too, since refusing to open the editor over a preference
    /// would be the worse failure.
    ///
    /// Hands back the warnings reading it raised rather than logging them: the
    /// windowed editor opens it before the log stream exists.
    pub fn open(user_dir: Option<PathBuf>) -> (Self, Vec<String>) {
        let mut store = Self {
            dir: user_dir,
            applied: Settings::default(),
            found: false,
            malformed: false,
            unsaved: false,
            pushed: false,
        };
        let Some(path) = store.path() else {
            return (store, Vec::new());
        };
        let (read, mut said) = crate::logging::heard(|| {
            std::fs::read_to_string(&path)
                .map_err(|error| error.to_string())
                .and_then(|text| Settings::from_toml(&text).map_err(|error| error.to_string()))
        });
        match read {
            Ok(settings) => {
                store.found = true;
                store.applied = settings;
            }
            Err(_) if !path.exists() => {}
            Err(error) => {
                said.push(format!(
                    "ignoring {} — {error}; it is kept as {BAD} when a setting next changes",
                    path.display()
                ));
                store.found = true;
                store.malformed = true;
            }
        }
        (store, said)
    }

    fn path(&self) -> Option<PathBuf> {
        self.dir.as_ref().map(|dir| dir.join(FILE))
    }

    /// Where the settings are kept, whether or not anything has been written
    /// there yet; `None` when nothing is saved.
    pub fn file(&self) -> Option<PathBuf> {
        self.path()
    }

    pub fn current(&self) -> &Settings {
        &self.applied
    }

    /// Where the user's own themes are kept, beside the settings.
    pub fn themes_dir(&self) -> Option<PathBuf> {
        self.dir.as_ref().map(|dir| dir.join(THEMES))
    }

    /// On a first run, take the theme and icon colours a project's
    /// `editor.ron` held from when they were kept there, so nobody loses what
    /// they picked; after that the project's are never read.
    pub fn seed(&mut self, theme: Option<String>, icons: Option<IconTints>) {
        if self.found {
            return;
        }
        if let Some(theme) = theme.filter(|theme| theme != super::theme::BUILT_IN) {
            self.applied.appearance.theme = Some(theme);
        }
        if let Some(icons) = icons {
            self.applied.appearance.icons = icons;
        }
        self.save();
    }

    /// Makes what the session runs with what the editor state says, then
    /// keeps it once the pointer lets go rather than on every frame a slider
    /// is dragged through.
    ///
    /// Two resource reads and one comparison a frame, of `Settings` against the
    /// copy last applied. Derived equality over numbers, strings and enums
    /// never allocates; the copy is cloned only when the two differ, and
    /// `sanitise` runs first so a NaN cannot keep them different forever.
    pub(super) fn settle(
        &mut self,
        ctx: &egui::Context,
        world: &World,
        state: &mut EditorState,
        themes: &mut ThemeSet,
    ) {
        if !self.pushed {
            self.applied.apply_to_world(world);
            self.pushed = true;
        }
        // Restyling mid-frame would leave the panels already drawn on the old
        // palette, so a pick is a request the editor drains after them.
        if let Some(name) = state.take_theme_request()
            && themes.names().any(|known| known == name)
        {
            state.settings.appearance.theme = (name != super::theme::BUILT_IN).then_some(name);
        }
        let settings = &mut state.settings;
        // The Profiler's present mode and the Scripts panel's checkbox edit
        // their world resources, as they did before there were settings.
        if let Some(present) = world.get_resource::<PresentSettings>() {
            let display = Display::from(*present);
            if display != self.applied.display {
                settings.display = display;
            }
        }
        #[cfg(feature = "scripting")]
        if let Some(status) = world.get_resource::<crate::build_watcher::BuildStatus>()
            && status.auto_reload != self.applied.scripting.reload_after_rebuild
        {
            settings.scripting.reload_after_rebuild = status.auto_reload;
        }

        if *settings != self.applied {
            settings.sanitise();
            let (old, new) = (&self.applied.appearance, &settings.appearance);
            if new.theme != old.theme && themes.select(new.theme_name()) {
                super::theme::apply(ctx, themes.active());
            }
            if new.ui_scale != old.ui_scale {
                ctx.set_zoom_factor(new.ui_scale);
            }
            state.autosave.configure(settings.saving.every());
            settings.apply_to_world(world);
            self.applied.clone_from(settings);
            self.unsaved = true;
        }
        if self.unsaved && !ctx.input(|input| input.pointer.any_down()) {
            self.save();
        }
    }

    /// What applies before the first frame: the theme, the scale and autosave.
    /// The world's half waits for the first [`settle`](Self::settle).
    pub(super) fn begin(
        &self,
        ctx: &egui::Context,
        state: &mut EditorState,
        themes: &mut ThemeSet,
    ) {
        state.settings.clone_from(&self.applied);
        state.started_with.clone_from(&self.applied.startup);
        themes.select(self.applied.appearance.theme_name());
        ctx.set_zoom_factor(self.applied.appearance.ui_scale);
        state.autosave.configure(self.applied.saving.every());
    }

    /// Keeps a change still waiting on the pointer, as the editor closes.
    pub(super) fn flush(&mut self) {
        if self.unsaved {
            self.save();
        }
    }

    fn save(&mut self) {
        self.unsaved = false;
        let Some(path) = self.path() else {
            return;
        };
        let Some(text) = self.applied.to_toml() else {
            return;
        };
        if text.is_empty() && !self.found {
            return;
        }
        if let Some(dir) = path.parent()
            && let Err(error) = std::fs::create_dir_all(dir)
        {
            tracing::error!("cannot create {}: {error}", dir.display());
            return;
        }
        // A hand-edited file that did not parse is somebody's work: kept, once,
        // rather than replaced.
        if self.malformed {
            let aside = path.with_file_name(BAD);
            match std::fs::rename(&path, &aside) {
                Ok(()) => tracing::warn!(
                    "{} could not be read; it is kept as {} and a new one written",
                    path.display(),
                    aside.display()
                ),
                Err(_) if !path.exists() => {}
                Err(error) => {
                    tracing::error!(
                        "cannot move {} aside ({error}), so it is left as it is and not saved over",
                        path.display()
                    );
                    return;
                }
            }
            self.malformed = false;
        }
        let partial = path.with_file_name(format!("{FILE}.partial"));
        match std::fs::write(&partial, text).and_then(|()| std::fs::rename(&partial, &path)) {
            Ok(()) => self.found = true,
            Err(error) => tracing::error!("cannot write {}: {error}", path.display()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::state::EditorState;
    use crate::editor::theme::{self, ThemeSet};

    /// A scratch directory, removed when dropped.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("orrin-settings-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn file(&self) -> PathBuf {
            self.0.join(FILE)
        }

        fn store(&self) -> (SettingsStore, Vec<String>) {
            SettingsStore::open(Some(self.0.clone()))
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn world() -> World {
        let mut world = World::default();
        crate::App::install_default_resources(&mut world);
        world
    }

    /// One editor frame's worth of settling, with the pointer up or down.
    fn settle(
        store: &mut SettingsStore,
        world: &World,
        state: &mut EditorState,
        themes: &mut ThemeSet,
        pointer_down: bool,
    ) {
        let ctx = egui::Context::default();
        let pos = egui::pos2(10.0, 10.0);
        let mut events = vec![egui::Event::PointerMoved(pos)];
        if pointer_down {
            events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: Default::default(),
            });
        }
        ctx.begin_pass(egui::RawInput {
            events,
            ..Default::default()
        });
        store.settle(&ctx, world, state, themes);
        let _ = ctx.end_pass();
    }

    fn unusual() -> Settings {
        let mut settings = Settings::default();
        settings.appearance.theme = Some("Void".into());
        settings.appearance.ui_scale = 1.25;
        settings.appearance.icons.light = [1, 2, 3];
        settings.viewport.fly_speed = 12.0;
        settings.viewport.sensitivity = 0.5;
        settings.scripting.editor = ExternalEditor::Custom("subl {file}:{line}".into());
        settings.scripting.reload_after_rebuild = false;
        settings.saving.autosave = false;
        settings.saving.autosave_interval_secs = 300;
        settings.display.vsync = VsyncMode::Immediate;
        settings.display.images = 3;
        settings
    }

    fn save(store: &mut SettingsStore, settings: Settings) {
        let world = world();
        let mut state = EditorState::default();
        state.settings = settings;
        settle(store, &world, &mut state, &mut ThemeSet::default(), false);
    }

    #[test]
    fn settings_round_trip() {
        let dir = Scratch::new("round-trip");
        save(&mut dir.store().0, unusual());
        assert_eq!(*dir.store().0.current(), unusual());
    }

    /// Deviations only, as a material's shader values and a Behaviour's
    /// fields across a reload are: a default the engine changes later reaches
    /// everyone who never set it.
    #[test]
    fn a_value_left_at_its_default_is_not_written() {
        let dir = Scratch::new("deviations");
        let mut settings = Settings::default();
        settings.viewport.fly_speed = 12.0;
        settings.appearance.icons.camera = [9, 9, 9];
        save(&mut dir.store().0, settings);

        let text = std::fs::read_to_string(dir.file()).unwrap();
        assert!(text.contains("fly_speed = 12"), "{text}");
        assert!(text.contains("camera = [9, 9, 9]"), "{text}");
        for absent in [
            "boost",
            "sensitivity",
            "ui_scale",
            "light",
            "scripting",
            "saving",
        ] {
            assert!(!text.contains(absent), "{absent} in\n{text}");
        }
    }

    #[test]
    fn settings_that_are_all_defaults_write_nothing() {
        let dir = Scratch::new("all-defaults");
        let mut changed = Settings::default();
        changed.saving.autosave = false;
        assert_eq!(
            changed.to_toml().as_deref(),
            Some("[saving]\nautosave = false\n")
        );
        assert_eq!(Settings::default().to_toml().as_deref(), Some(""));

        let mut store = dir.store().0;
        store.seed(None, None);
        save(&mut store, Settings::default());
        assert!(!dir.file().exists(), "nothing to say, so no file");

        save(&mut store, changed);
        save(&mut store, Settings::default());
        assert_eq!(
            std::fs::read_to_string(dir.file()).unwrap(),
            "",
            "back to every default, the file stays and says nothing"
        );
    }

    #[test]
    fn an_unknown_key_is_ignored() {
        let dir = Scratch::new("unknown-key");
        std::fs::write(
            dir.file(),
            "[appearance]\ntheme = \"Void\"\nsparkle = true\n\n[telemetry]\nlevel = 3\n",
        )
        .unwrap();
        let store = dir.store().0;
        assert_eq!(store.current().appearance.theme.as_deref(), Some("Void"));
    }

    /// A newer engine's editor preset costs that one value, not the file.
    #[test]
    fn an_editor_this_build_does_not_know_falls_back_to_auto() {
        let dir = Scratch::new("unknown-variant");
        std::fs::write(
            dir.file(),
            "[scripting]\neditor = \"fleet\"\nreload_after_rebuild = false\n",
        )
        .unwrap();
        let settings = dir.store().0.current().clone();
        assert_eq!(settings.scripting.editor, ExternalEditor::Auto);
        assert!(!settings.scripting.reload_after_rebuild);
    }

    #[test]
    fn a_malformed_file_yields_defaults_and_is_kept_aside_rather_than_destroyed() {
        let dir = Scratch::new("malformed");
        let typed = "[viewport]\nfly_speed = \"fast\n";
        std::fs::write(dir.file(), typed).unwrap();

        let mut store = dir.store().0;
        assert_eq!(*store.current(), Settings::default());
        assert_eq!(
            std::fs::read_to_string(dir.file()).unwrap(),
            typed,
            "reading alone changes nothing"
        );

        let mut settings = Settings::default();
        settings.viewport.boost = 2.0;
        save(&mut store, settings.clone());
        assert_eq!(
            std::fs::read_to_string(dir.0.join("settings.toml.bad")).unwrap(),
            typed
        );
        assert_eq!(*dir.store().0.current(), settings);

        settings.viewport.boost = 3.0;
        save(&mut store, settings);
        assert_eq!(
            std::fs::read_to_string(dir.0.join("settings.toml.bad")).unwrap(),
            typed,
            "kept aside once: the next save does not move the good file over it"
        );
    }

    #[test]
    fn a_value_out_of_range_is_brought_back_into_it() {
        let dir = Scratch::new("ranges");
        std::fs::write(
            dir.file(),
            "[appearance]\nui_scale = 9.0\n[viewport]\nfly_speed = nan\nboost = 0.0\n\
             [saving]\nautosave_interval_secs = 1\n[display]\nimages = 40\n",
        )
        .unwrap();
        let settings = dir.store().0.current().clone();
        assert_eq!(settings.appearance.ui_scale, *UI_SCALE.end());
        assert_eq!(settings.viewport.fly_speed, FlyCamera::default().fly_speed);
        assert_eq!(settings.viewport.boost, *BOOST.start());
        assert_eq!(
            settings.saving.autosave_interval_secs,
            *AUTOSAVE_SECS.start()
        );
        assert_eq!(settings.display.images, *SWAPCHAIN_IMAGES.end());
    }

    #[test]
    fn with_no_directory_nothing_is_read_or_written() {
        let mut store = SettingsStore::open(None).0;
        assert_eq!(*store.current(), Settings::default());
        assert_eq!(store.themes_dir(), None);
        save(&mut store, unusual());
        assert_eq!(
            *store.current(),
            unusual(),
            "the session still runs with it"
        );
    }

    #[test]
    fn the_users_themes_sit_beside_the_settings() {
        let dir = Scratch::new("themes-dir");
        assert_eq!(dir.store().0.themes_dir(), Some(dir.0.join("themes")));
    }

    /// The bug this module exists to fix: a theme picked in one project was
    /// gone in the next, because it was kept in the project's `editor.ron`.
    #[test]
    fn a_theme_chosen_with_one_project_open_is_the_theme_with_another_open() {
        use crate::editor::prefs::PrefsFile;

        let user = Scratch::new("two-projects-user");
        let (one, other) = (Scratch::new("project-one"), Scratch::new("project-two"));
        theme_file(&user.0.join("themes"), "Void");
        // What the old editor left in the second project: it must not win.
        std::fs::write(other.0.join("editor.ron"), "(theme: Some(\"orrin-dark\"))").unwrap();

        let world = world();
        let mut store = user.store().0;
        store.seed(PrefsFile::new(Some(one.0.clone())).load().theme, None);
        let mut themes = ThemeSet::load([user.0.join("themes")]);
        let mut state = EditorState::default();
        state.request_theme("Void");
        settle(&mut store, &world, &mut state, &mut themes, false);
        assert_eq!(themes.active().name, "Void");

        let mut store = user.store().0;
        store.seed(PrefsFile::new(Some(other.0.clone())).load().theme, None);
        assert_eq!(store.current().appearance.theme.as_deref(), Some("Void"));
        assert!(
            !one.0.join("editor.ron").exists(),
            "the project is not written"
        );
    }

    #[test]
    fn picking_the_built_in_theme_is_the_default_rather_than_a_deviation() {
        let user = Scratch::new("built-in");
        theme_file(&user.0.join("themes"), "Void");
        let world = world();
        let mut store = user.store().0;
        let mut themes = ThemeSet::load([user.0.join("themes")]);
        let mut state = EditorState::default();
        state.request_theme("Void");
        settle(&mut store, &world, &mut state, &mut themes, false);
        state.request_theme(theme::BUILT_IN);
        settle(&mut store, &world, &mut state, &mut themes, false);
        assert_eq!(state.settings.appearance.theme, None);
        assert_eq!(themes.active().name, theme::BUILT_IN);
    }

    fn theme_file(dir: &std::path::Path, name: &str) {
        std::fs::create_dir_all(dir).unwrap();
        let theme = theme::Theme {
            name: name.to_owned(),
            ..theme::Theme::default()
        };
        std::fs::write(
            dir.join(format!("{name}.toml")),
            toml::to_string(&theme).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn a_first_run_takes_the_theme_and_icons_an_old_editor_ron_held() {
        use crate::editor::prefs::PrefsFile;

        let user = Scratch::new("seed-user");
        let project = Scratch::new("seed-project");
        std::fs::write(
            project.0.join("editor.ron"),
            "(theme: Some(\"Void\"), icons: Some((light: (1, 2, 3))))",
        )
        .unwrap();
        let prefs = PrefsFile::new(Some(project.0.clone())).load();

        let mut store = user.store().0;
        store.seed(prefs.theme, prefs.icons);
        let seeded = store.current().clone();
        assert_eq!(seeded.appearance.theme.as_deref(), Some("Void"));
        assert_eq!(seeded.appearance.icons.light, [1, 2, 3]);
        assert_eq!(*user.store().0.current(), seeded, "and kept at once");

        let mut store = user.store().0;
        store.seed(Some("Another".into()), None);
        assert_eq!(
            store.current().appearance.theme.as_deref(),
            Some("Void"),
            "only a first run is seeded"
        );
    }

    #[test]
    fn a_change_is_kept_once_the_pointer_lets_go() {
        let dir = Scratch::new("pointer");
        let world = world();
        let mut store = dir.store().0;
        let mut themes = ThemeSet::default();
        let mut state = EditorState::default();
        state.settings.viewport.fly_speed = 9.0;
        settle(&mut store, &world, &mut state, &mut themes, true);
        assert!(!dir.file().exists(), "not while a slider is held");
        settle(&mut store, &world, &mut state, &mut themes, false);
        assert_eq!(dir.store().0.current().viewport.fly_speed, 9.0);
    }

    #[test]
    fn a_change_reaches_the_world_and_the_editors_own_state() {
        let world = world();
        let mut store = SettingsStore::open(None).0;
        let mut themes = ThemeSet::default();
        let mut state = EditorState::default();
        state.settings.display.vsync = VsyncMode::Mailbox;
        state.settings.saving.autosave = false;
        state.settings.scripting.reload_after_rebuild = false;
        settle(&mut store, &world, &mut state, &mut themes, false);

        assert_eq!(
            world.resource::<PresentSettings>().vsync,
            VsyncMode::Mailbox
        );
        assert!(
            !state
                .autosave
                .due(std::time::Instant::now() + Duration::from_secs(1 << 20), 1)
        );
        #[cfg(feature = "scripting")]
        assert!(
            !world
                .resource::<crate::build_watcher::BuildStatus>()
                .auto_reload
        );
    }

    /// The Profiler's present mode and the Scripts panel's checkbox edit their
    /// world resources, as they always have; the store reads them back.
    #[test]
    fn a_resource_edited_by_its_own_panel_is_kept() {
        let dir = Scratch::new("from-world");
        let world = world();
        let mut store = dir.store().0;
        let mut themes = ThemeSet::default();
        let mut state = EditorState::default();
        settle(&mut store, &world, &mut state, &mut themes, false);

        world.resource_mut::<PresentSettings>().vsync = VsyncMode::Immediate;
        #[cfg(feature = "scripting")]
        {
            world
                .resource_mut::<crate::build_watcher::BuildStatus>()
                .auto_reload = false;
        }
        settle(&mut store, &world, &mut state, &mut themes, false);

        let kept = dir.store().0.current().clone();
        assert_eq!(kept.display.vsync, VsyncMode::Immediate);
        #[cfg(feature = "scripting")]
        assert!(!kept.scripting.reload_after_rebuild);
    }

    /// The store's first frame pushes what was loaded into the world rather
    /// than reading the world's engine defaults back over it.
    #[test]
    fn the_first_frame_gives_the_world_the_saved_settings() {
        let dir = Scratch::new("first-frame");
        save(&mut dir.store().0, unusual());

        let world = world();
        let mut store = dir.store().0;
        let mut state = EditorState::default();
        state.settings = store.current().clone();
        settle(
            &mut store,
            &world,
            &mut state,
            &mut ThemeSet::default(),
            false,
        );
        assert_eq!(
            *world.resource::<PresentSettings>(),
            unusual().display.present()
        );
        assert_eq!(*dir.store().0.current(), unusual());
    }

    #[cfg(feature = "scripting")]
    #[test]
    fn reload_after_rebuild_defaults_to_what_the_build_status_does() {
        assert_eq!(
            Scripting::default().reload_after_rebuild,
            crate::build_watcher::BuildStatus::default().auto_reload
        );
    }

    #[test]
    fn the_display_defaults_to_the_engines_own_present_settings() {
        assert_eq!(Display::default().present(), PresentSettings::default());
    }

    #[test]
    fn the_editor_variable_overrides_and_says_so() {
        let set = Environment::from_vars(|name| {
            (name == super::super::source::EDITOR_ENV).then(|| "zed {file}".to_owned())
        });
        assert_eq!(set.editor(), Some("zed {file}"));
        assert_eq!(set.editor_override(), Some("ORRIN_EDITOR"));

        let blank = Environment::from_vars(|_| Some("  ".to_owned()));
        assert_eq!(blank.editor(), None);
        assert_eq!(blank.editor_override(), None);
        assert_eq!(Environment::default().editor_override(), None);
    }

    /// No `[startup]` is today's behaviour with no variable set, and a default
    /// is never written; a choice is, by name.
    #[test]
    fn startup_defaults_write_nothing_and_a_change_round_trips() {
        let dir = Scratch::new("startup");
        std::fs::write(dir.file(), "[viewport]\nfly_speed = 12.0\n").unwrap();
        assert_eq!(dir.store().0.current().startup, Startup::default());
        assert_eq!(Startup::default().backend_choice(), None);
        assert_eq!(Startup::default().threads(), None);
        assert_eq!(Startup::default().validation(), None);
        assert_eq!(Startup::default().log_floor(), None);

        let mut settings = Settings::default();
        settings.viewport.fly_speed = 12.0;
        save(&mut dir.store().0, settings.clone());
        let text = std::fs::read_to_string(dir.file()).unwrap();
        assert!(!text.contains("startup"), "{text}");

        settings.startup = Startup {
            backend: Backend::Vulkan,
            threads: Some(4),
            validation: Validation::Off,
            log: LogFloor::Debug,
        };
        save(&mut dir.store().0, settings.clone());
        let text = std::fs::read_to_string(dir.file()).unwrap();
        for line in [
            "[startup]",
            "backend = \"vulkan\"",
            "threads = 4",
            "validation = \"off\"",
            "log = \"debug\"",
        ] {
            assert!(text.contains(line), "{line} in\n{text}");
        }
        let read = dir.store().0.current().startup.clone();
        assert_eq!(read, settings.startup);
        assert_eq!(
            read.backend_choice(),
            Some(crate::gfx::BackendChoice::Vulkan)
        );
        assert_eq!(read.threads(), Some(4));
        assert_eq!(read.validation(), Some(false));
        assert_eq!(read.log_floor(), Some(crate::scene::LogLevel::Debug));
    }

    /// A hand edit that names something this build does not know costs that
    /// one value; zero threads is rayon's "all of them" and is Auto here.
    #[test]
    fn a_startup_value_this_build_cannot_read_is_auto() {
        let dir = Scratch::new("startup-unknown");
        std::fs::write(
            dir.file(),
            "[startup]\nbackend = \"dx12\"\nthreads = 0\nvalidation = \"maybe\"\nlog = \"debug\"\n",
        )
        .unwrap();
        let (store, said) = dir.store();
        let startup = &store.current().startup;
        assert_eq!(startup.backend, Backend::Auto);
        assert_eq!(startup.threads, None);
        assert_eq!(startup.validation, Validation::Auto);
        assert_eq!(startup.log, LogFloor::Debug);
        assert!(said.iter().any(|line| line.contains("dx12")), "{said:?}");
    }

    /// The store is opened before the log stream exists, so what it has to
    /// say comes back to be logged once it does.
    #[test]
    fn what_the_store_says_while_opening_is_handed_back() {
        let dir = Scratch::new("startup-heard");
        std::fs::write(dir.file(), "[viewport\n").unwrap();
        let (_, said) = dir.store();
        assert_eq!(said.len(), 1, "{said:?}");
        assert!(
            said[0].contains(&dir.file().display().to_string()),
            "{said:?}"
        );
        assert!(SettingsStore::open(None).1.is_empty());
    }

    #[test]
    fn every_startup_variable_overrides_and_says_so() {
        let vars = |name: &str| {
            Some(
                match name {
                    "ORRIN_BACKEND" => "Vulkan",
                    "ORRIN_THREADS" => "3",
                    "ORRIN_VALIDATION" => "0",
                    "ORRIN_LOG" => "warn",
                    _ => return None,
                }
                .to_owned(),
            )
        };
        let set = Environment::from_vars(vars);
        assert_eq!(set.backend_override(), Some("ORRIN_BACKEND"));
        assert_eq!(set.threads_override(), Some("ORRIN_THREADS"));
        assert_eq!(set.validation_override(), Some("ORRIN_VALIDATION"));
        assert_eq!(set.log_override(), Some("ORRIN_LOG"));
        assert_eq!(
            set.over(&Startup::default()),
            Startup {
                backend: Backend::Vulkan,
                threads: Some(3),
                validation: Validation::Off,
                log: LogFloor::Warn,
            }
        );

        for unset in [
            Environment::default(),
            Environment::from_vars(|_| Some("  ".to_owned())),
        ] {
            assert_eq!(unset.backend_override(), None);
            assert_eq!(unset.threads_override(), None);
            assert_eq!(unset.validation_override(), None);
            assert_eq!(unset.log_override(), None);
        }
        // What the engine would not take is no override either, so the row
        // stays the user's.
        let nonsense = Environment::from_vars(|name| {
            matches!(name, "ORRIN_THREADS" | "ORRIN_LOG").then(|| "lots".to_owned())
        });
        assert_eq!(nonsense.threads_override(), None);
        assert_eq!(nonsense.log_override(), None);
    }

    /// The restart line counts what the user changed that the next run will
    /// actually pick up: a row the environment decides is not one.
    #[test]
    fn a_change_the_environment_overrides_is_not_waiting_on_a_restart() {
        let started = Startup::default();
        let mut now = started.clone();
        assert_eq!(now.changes_from(&started, &Environment::default()), 0);
        now.threads = Some(2);
        now.log = LogFloor::Info;
        assert_eq!(now.changes_from(&started, &Environment::default()), 2);
        let threads = Environment::from_vars(|name| (name == "ORRIN_THREADS").then(|| "6".into()));
        assert_eq!(now.changes_from(&started, &threads), 1);
    }

    #[test]
    fn the_editors_presets_are_written_by_name() {
        let mut settings = Settings::default();
        settings.scripting.editor = ExternalEditor::VsCode;
        let text = settings.to_toml().unwrap();
        assert!(text.contains("editor = \"vscode\""), "{text}");
        settings.display.vsync = VsyncMode::FifoRelaxed;
        let text = settings.to_toml().unwrap();
        assert!(text.contains("vsync = \"fifo-relaxed\""), "{text}");
    }
}
