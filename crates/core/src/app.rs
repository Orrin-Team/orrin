use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use vulkano::VulkanLibrary;
use vulkano::instance::debug::{
    DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
    DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo,
};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::swapchain::Surface;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::camera_controller::CameraController;
use crate::editor::Editor;
use crate::gfx::BackendChoice;
use crate::gfx::punctual::ShadowAtlas;
use crate::gfx::shadows::{CascadeSet, cascades};
use crate::gfx::vulkan::ShadowFrame;
use crate::gfx::vulkan::VulkanRenderer;
use crate::gfx::{RenderBackend, SceneLighting};
use crate::profile::Profiler;
use crate::profile_scope;
use crate::scene::entities::{SceneChoice, StressSpec, spawn_stress_scene};
use crate::scene::{
    Actions, AmbientLight, BloomSettings, Camera, ContactShadowSettings, Culling, DebugLine,
    DebugLines, DecalSettings, Diagnostics, DofSettings, EnvironmentSettings, FogSettings,
    GameScreen, HdrSettings, InputState, LogBuffer, LogLevel, MotionBlurSettings, PresentSettings,
    RefractionSettings, ShadowSettings, SsaoSettings, SsrSettings, SubsurfaceSettings, TaaSettings,
    Time, TransparencySettings, UiCanvas,
};
use crate::stats::FrameStats;
use crate::systems;
use crate::systems::FrameGeometry;
use orrin_ecs::World;
use orrin_registry::Registry;

struct Active {
    window: Arc<Window>,
    renderer: VulkanRenderer,
    editor: Editor,
}

pub struct App {
    /// `None` on a run that does not draw through Vulkan.
    instance: Option<Arc<Instance>>,
    backend: BackendChoice,
    active: Option<Active>,
    world: World,
    /// Not a world resource: it has to survive the world being cleared, and a
    /// scene load needs it before there is a world to read it out of.
    registry: Registry,
    camera_controller: CameraController,
    /// Timestamp of the previous rendered frame; `None` until the first frame
    /// establishes a baseline. Delta is `now - last_instant`, never a
    /// difference of two large "seconds since start" floats (which quantizes).
    last_instant: Option<Instant>,
    /// This frame's renderables and the draw orders over them. Kept on the app
    /// so its `Vec`s keep their capacity across frames instead of reallocating.
    geometry: FrameGeometry,
    lighting: SceneLighting,
    /// This frame's decals. Kept alongside `lighting` for the same reason it is
    /// a field rather than a local: extraction refills it every frame and the
    /// allocation is reused.
    decals: Vec<crate::gfx::DecalInstance>,
    /// This frame's cascade matrices. The caster orders live in `geometry`,
    /// which was culled against exactly these.
    cascades: CascadeSet,
    /// Which punctual lights got atlas tiles this frame, where those tiles are,
    /// and the matrix each face is drawn with. Fitted between extracting the
    /// lighting and extracting the geometry, because the caster lists are culled
    /// against the very lights it chose.
    atlas: ShadowAtlas,
    /// This frame's debug lines, copied out of the `DebugLines` resource so the
    /// renderer borrow doesn't overlap the world borrow.
    debug_lines: Vec<DebugLine>,
    /// The game's UI laid out, with the glyph cache it is laid out from.
    ui: crate::gfx::ui::Painter,
    /// The log stream's end for the console panel, drained once a frame.
    console: crate::logging::ConsoleQueue,
    #[cfg(feature = "scripting")]
    scripting: Option<crate::scripting::Scripting>,
    /// Rebuild-on-save. `None` when the project's layout gave nothing to watch
    /// — see `BuildWatcher::for_game_assembly`.
    #[cfg(feature = "scripting")]
    build_watcher: Option<crate::build_watcher::BuildWatcher>,
    /// The open story's world. Beside `scripting` rather than in the editor
    /// because building one runs game code.
    #[cfg(feature = "scripting")]
    stories: crate::storybook::StoryRunner,
    /// The window the open story is drawn in, and its painter, made the first
    /// time one opens so a session that never opens one pays nothing for it.
    #[cfg(feature = "scripting")]
    story_window: Option<StoryWindow>,
    #[cfg(feature = "scripting")]
    story_ui: Option<crate::gfx::ui::Painter>,
    /// Reload-on-save for the project's binding file. `None` without a project,
    /// without the file, or in a build with no `notify`.
    #[cfg(feature = "scripting")]
    input_watcher: Option<crate::scene::input::ConfigWatcher>,
    /// Rescan-on-save for the prefab index. `None` without an assets
    /// directory.
    prefab_watcher: Option<crate::scene::prefab::Watcher>,
    /// The gamepad backend. `None` when the machine has no gamepad subsystem
    /// to open, which is a session on keyboard and mouse rather than a failure.
    gamepads: Option<crate::scene::input::Gamepads>,
    /// The Orrin project this run was launched inside, if any. `None` means
    /// the engine is running standalone on its built-in demo scene — and, for
    /// the editor, that there is nowhere to keep themes or a layout.
    project: Option<orrin_project::Project>,
    /// Which built-in scene this run opens, from `ORRIN_SCENE`.
    scene: SceneChoice,
    /// An exported game rather than the editor: see [`App::shipped_project`].
    shipped: bool,
    /// Whether the window holds the cursor now, so a change is applied once.
    cursor_grabbed: bool,
    /// Extra profiling load from `ORRIN_STRESS`; `None` for a normal run.
    stress: Option<StressSpec>,
    /// Kept alive for the process: dropping the messenger stops validation
    /// output, which is exactly when you need it most.
    _debug_messenger: Option<DebugUtilsMessenger>,
}

/// What `boot_scripting` produced: the live host, and the watcher that keeps it
/// fed. The watcher is optional and independent — failing to start one leaves
/// scripting perfectly usable, just without rebuild-on-save.
#[cfg(feature = "scripting")]
struct Scripts {
    scripting: crate::scripting::Scripting,
    watcher: Option<crate::build_watcher::BuildWatcher>,
}

impl App {
    pub fn run() {
        // First, so that everything below it — and every exit it takes — is
        // heard.
        let console = crate::logging::init(crate::logging::LogConfig::for_this_build());

        // Before anything that might dispatch into it — the scene build decodes
        // textures across it, and that happens on the first `resumed`.
        crate::threads::init();

        let event_loop = EventLoop::new().unwrap();
        event_loop.set_control_flow(ControlFlow::Poll);

        let backend = BackendChoice::from_env();
        let (instance, debug_messenger) = match backend {
            BackendChoice::Vulkan => {
                let (instance, messenger) = create_vulkan_instance(&event_loop);
                (Some(instance), messenger)
            }
            BackendChoice::Metal => (None, None),
        };

        let shipped = cfg!(feature = "export");
        if shipped
            && !crate::profile::shipped_game_profiles(
                std::env::var("ORRIN_PROFILE").ok().as_deref(),
            )
        {
            crate::profile::set_enabled(false);
        }
        let project = if shipped {
            match Self::shipped_project() {
                Ok(project) => Some(project),
                Err(err) => {
                    tracing::error!("{err}");
                    std::process::exit(1);
                }
            }
        } else {
            let cwd = std::env::current_dir().unwrap_or_else(|err| {
                tracing::error!("cannot read the current directory: {err}");
                std::process::exit(1);
            });

            // A manifest that exists but doesn't load is fatal: silently falling
            // back to the demo scene would hide the user's real project.
            match orrin_project::Project::locate(&cwd) {
                Ok(project) => project,
                Err(err) => {
                    tracing::error!("{err}");
                    std::process::exit(1);
                }
            }
        };
        if let Some(project) = &project {
            tracing::info!(
                "project `{}` at {}",
                project.name(),
                project.root().display()
            );
        }

        // Opened here rather than with the world's resources, because
        // enumerating input devices is an OS scan and the cold-start gate
        // measures `install_default_resources`. Its cost belongs to the run,
        // not to the number that guards scene construction.
        let gamepads = match crate::scene::input::Gamepads::new() {
            Ok(gamepads) => Some(gamepads),
            Err(error) => {
                tracing::warn!("{error}");
                None
            }
        };

        let mut app = App {
            instance,
            backend,
            active: None,
            world: World::default(),
            registry: Registry::new(),
            camera_controller: CameraController::new(),
            last_instant: None,
            geometry: FrameGeometry::default(),
            lighting: SceneLighting::default(),
            decals: Vec::new(),
            cascades: CascadeSet::default(),
            atlas: ShadowAtlas::default(),
            debug_lines: Vec::new(),
            ui: crate::gfx::ui::Painter::new(),
            console,
            #[cfg(feature = "scripting")]
            scripting: None,
            #[cfg(feature = "scripting")]
            build_watcher: None,
            #[cfg(feature = "scripting")]
            stories: crate::storybook::StoryRunner::default(),
            #[cfg(feature = "scripting")]
            story_window: None,
            #[cfg(feature = "scripting")]
            story_ui: None,
            #[cfg(feature = "scripting")]
            input_watcher: None,
            prefab_watcher: None,
            gamepads,
            project,
            scene: SceneChoice::from_env(),
            shipped,
            cursor_grabbed: false,
            stress: StressSpec::from_env().filter(|spec| !spec.is_empty()),
            _debug_messenger: debug_messenger,
        };

        crate::scene::register_components(&mut app.registry);
        Self::install_default_resources(&mut app.world);
        // Here rather than in `install_default_resources` for the gamepads'
        // reason, though it returns at once: the device opens on a thread of
        // its own, and a test world has no business making a sound.
        app.world
            .resource_mut::<crate::audio::Audio>()
            .open_device();
        app.load_input_config();
        app.load_prefabs();

        event_loop.run_app(&mut app).unwrap();
    }

    /// The game this executable was exported as, from the `orrin.json` beside
    /// it. Never the working directory: a player starts a game from a desktop
    /// shortcut, a launcher or a terminal anywhere, and none of them is the
    /// game's directory.
    fn shipped_project() -> Result<orrin_project::Project, String> {
        let exe = std::env::current_exe()
            .map_err(|err| format!("cannot find this executable's own location: {err}"))?;
        let manifest = exe
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join(orrin_project::export::FILE_NAME);
        if !manifest.is_file() {
            return Err(format!(
                "{} is missing, so this game does not know what to run. Copy the whole \
                 exported directory, not only the executable",
                manifest.display()
            ));
        }
        orrin_project::Project::from_export(&manifest).map_err(|err| err.to_string())
    }

    /// Every world resource the engine starts with, before any window or device
    /// exists.
    ///
    /// Split out of [`run`](App::run) so the cold-start guard
    /// (`tests/cold_start.rs`) measures the list the engine actually installs
    /// rather than a copy of it — a resource added here and not there would
    /// otherwise be invisible to the very benchmark meant to catch startup
    /// creep.
    pub fn install_default_resources(world: &mut World) {
        world.insert_resource(Camera::default());
        world.insert_resource(crate::scene::ActiveCamera::default());
        world.insert_resource(Time::new());
        world.insert_resource(AmbientLight::default());
        world.insert_resource(SsaoSettings::default());
        world.insert_resource(SsrSettings::default());
        world.insert_resource(SubsurfaceSettings::default());
        world.insert_resource(TransparencySettings::default());
        world.insert_resource(DecalSettings::default());
        world.insert_resource(RefractionSettings::default());
        world.insert_resource(ShadowSettings::default());
        world.insert_resource(ContactShadowSettings::default());
        world.insert_resource(HdrSettings::default());
        world.insert_resource(BloomSettings::default());
        world.insert_resource(TaaSettings::default());
        world.insert_resource(MotionBlurSettings::default());
        world.insert_resource(DofSettings::default());
        world.insert_resource(FogSettings::default());
        // `ORRIN_HDRI` names an environment relative to the assets directory,
        // the same env-var-over-default shape the scripts directory and entry
        // type resolve by. It exists because nothing persists the choice yet:
        // a scene file cannot name an environment until resources are part of
        // what a scene saves, and until then this is how a run gets one without
        // a click.
        let hdri = std::env::var("ORRIN_HDRI").unwrap_or_default();
        world.insert_resource(EnvironmentSettings {
            reload_requested: !hdri.trim().is_empty(),
            hdri,
            ..Default::default()
        });
        world.insert_resource(PresentSettings::default());
        world.insert_resource(Diagnostics::default());
        world.insert_resource(crate::scene::Play::default());
        world.insert_resource(crate::scene::LoadedScenes::default());
        world.insert_resource(FrameStats::new());
        world.insert_resource(Profiler::default());
        world.insert_resource(InputState::new());
        world.insert_resource(crate::scene::CursorLock::default());
        world.insert_resource(Actions::new());
        world.insert_resource(crate::physics::CollisionState::default());
        world.insert_resource(crate::physics::PhysicsState::default());
        world.insert_resource(crate::audio::Audio::default());
        world.insert_resource(LogBuffer::default());
        world.insert_resource(DebugLines::default());
        world.insert_resource(UiCanvas::default());
        world.insert_resource(crate::ui::UiTree::default());
        world.insert_resource(crate::ui::events::UiPointer::default());
        world.insert_resource(crate::ui::events::UiEvents::default());
        world.insert_resource(crate::ui::tween::UiTweens::default());
        world.insert_resource(crate::ui::story::Storybook::default());
        world.insert_resource(Culling::default());
        // Empty until `load_prefabs` scans the project: the scan reads the disk,
        // and its cost belongs to the run rather than to the cold-start gate.
        world.insert_resource(crate::scene::Prefabs::default());
        // Inserted whether or not a watcher ever starts: the Scripts panel reads
        // it either way, and `Off` is how it explains itself.
        #[cfg(feature = "scripting")]
        world.insert_resource(crate::build_watcher::BuildStatus::default());
    }

    /// Read the project's `input.toml`, if it has one.
    ///
    /// A project without the file is normal and silent: bindings are optional,
    /// and an unbound action already reports itself the first time a script
    /// asks for one. A file that exists and does not parse is loud but not
    /// fatal — the session runs with nothing bound rather than refusing to
    /// start, because the fix is a text edit away and the editor is where it
    /// will be made.
    fn load_input_config(&mut self) {
        let Some(root) = self.project.as_ref().map(|project| project.root()) else {
            return;
        };
        let path = root.join(crate::scene::input::config::FILE_NAME);
        if !path.exists() {
            return;
        }
        match crate::scene::input::config::load(&path) {
            Ok(specs) => self.world.resource_mut::<Actions>().apply(specs),
            Err(error) => tracing::error!("{error}"),
        }

        // Watched even when the first read failed: the file is on disk, and
        // fixing it is exactly the save the watcher exists to notice.
        #[cfg(feature = "scripting")]
        if !self.shipped {
            self.input_watcher = match crate::scene::input::ConfigWatcher::new(&path) {
                Ok(watcher) => Some(watcher),
                Err(error) => {
                    tracing::warn!("{error}");
                    None
                }
            };
        }
    }

    /// Index the prefabs under the assets directory — the project's, or
    /// `assets/` beside the working directory without one, as the environment
    /// map resolves.
    ///
    /// A prefab that cannot be indexed is named and left out; the rest load.
    fn load_prefabs(&mut self) {
        let dir = self
            .project
            .as_ref()
            .map_or_else(|| PathBuf::from("assets"), |project| project.assets_dir());
        let (prefabs, issues) = crate::scene::Prefabs::scan(&dir);
        for issue in issues {
            tracing::warn!("{issue}");
        }
        let count = prefabs.iter().count();
        if count > 0 {
            tracing::info!("{count} prefab(s) under {}", dir.display());
        }
        self.world.insert_resource(prefabs);

        if dir.is_dir() && !self.shipped {
            self.prefab_watcher = match crate::scene::prefab::Watcher::new(&dir) {
                Ok(watcher) => Some(watcher),
                Err(error) => {
                    tracing::warn!("{error}");
                    None
                }
            };
        }
    }

    /// Index and start importing the project's assets, and reimport them as
    /// they change.
    ///
    /// Only for a project: the engine's own `assets/` holds the demo's
    /// environment and the fetched Sponza, which the demo loads by path, and
    /// indexing it would write a sidecar beside every file in a directory the
    /// engine does not own. Opened once the renderer exists and before the
    /// scene is built: the device decides what format every texture is imported
    /// in, and the scene is the first thing to ask for one.
    fn open_asset_database(&mut self, target: crate::asset_db::Target) {
        let Some(project) = &self.project else {
            return;
        };
        let dir = project.assets_dir();
        let (mut db, issues) = if self.shipped {
            crate::asset_db::AssetDatabase::open_packed(&dir, target)
        } else {
            crate::asset_db::AssetDatabase::open_for(
                &dir,
                crate::asset_db::cache_dir(Some(project)),
                target,
            )
        };
        for issue in issues {
            tracing::warn!("{issue}");
        }
        if dir.is_dir()
            && !self.shipped
            && let Err(error) = db.watch()
        {
            tracing::warn!("{error}; asset changes will need the browser's rescan button");
        }
        let count = db.iter().count();
        if count > 0 {
            tracing::info!("{count} asset(s) under {}", dir.display());
        }
        self.world.insert_resource(db);
    }

    /// An exported game's opening: its start scene, and its persistent scene
    /// beside it, in place of the built-in demo.
    fn open_game(&mut self) {
        let Some(project) = &self.project else {
            return;
        };
        let start = project.start_scene();
        let persistent = project.persistent_scene();
        let issues = crate::scene::loaded::open_start(
            &mut self.world,
            &self.registry,
            start.as_deref(),
            persistent.as_deref(),
        );
        for issue in issues {
            tracing::error!("{issue}");
        }
    }

    /// Boot the script host and attach the project's entry Behaviour.
    ///
    /// Two assemblies, resolved separately because they have different owners:
    /// the *bindings* (`Orrin.dll`) belong to the engine and stay in the
    /// default load context forever, while the *game* assembly belongs to the
    /// project and is what a hot reload swaps. Each follows env > manifest >
    /// built-in default, with its own env override.
    #[cfg(feature = "scripting")]
    fn boot_scripting(&mut self) -> Option<Scripts> {
        use crate::build_watcher::{BuildStatus, BuildWatcher};
        use crate::scripting::Scripting;
        use std::path::{Path, PathBuf};

        const BINDINGS: &str = "scripting/Orrin";
        const DEMO_SCRIPTS: &str = "scripting/DemoGame";
        const DEMO_ENTRY: &str = "DemoGame.Game, DemoGame";

        let bindings_dir = match std::env::var("ORRIN_SCRIPT_DIR") {
            Ok(dir) => Some(PathBuf::from(dir)),
            // Next to the executable is the shipped layout, and it is the only
            // one that works from inside a project directory. The cwd-relative
            // probe is the repo checkout, where the bindings are still an
            // unbuilt source tree.
            Err(_) => Scripting::bindings_beside_executable()
                .or_else(|| Scripting::find_bindings_dir(Path::new(BINDINGS))),
        };
        let Some(bindings_dir) = bindings_dir else {
            tracing::warn!(
                "scripting disabled: no Orrin bindings next to the engine, and none \
                 under {BINDINGS} (run `dotnet build {BINDINGS}`, or set ORRIN_SCRIPT_DIR)"
            );
            return None;
        };

        let entry = std::env::var("ORRIN_ENTRY")
            .ok()
            .or_else(|| {
                self.project
                    .as_ref()
                    .map(|project| project.entry_type().to_string())
            })
            .unwrap_or_else(|| DEMO_ENTRY.to_string());

        // Without a project this is the engine's own demo game, which is a
        // normal game assembly like any other — one loading path, so the demo
        // exercises hot reload exactly as a user's project does.
        let scripts_dir = self
            .project
            .as_ref()
            .map(|project| project.scripts_dir())
            .unwrap_or_else(|| PathBuf::from(DEMO_SCRIPTS));

        let shipped_dll = self
            .project
            .as_ref()
            .and_then(orrin_project::Project::game_assembly);
        let game_dll = match std::env::var("ORRIN_GAME_DLL") {
            Ok(path) => Some(PathBuf::from(path)),
            Err(_) if shipped_dll.is_some() => shipped_dll,
            Err(_) => match Scripting::assembly_of(&entry) {
                Some(assembly) => Scripting::find_game_assembly(&scripts_dir, assembly),
                None => {
                    tracing::warn!(
                        "scripting disabled: entry `{entry}` names no assembly — it must be \
                         assembly-qualified (`MyGame.Main, MyGame`) so the engine knows which \
                         DLL to load"
                    );
                    return None;
                }
            },
        };
        let Some(game_dll) = game_dll else {
            tracing::warn!(
                "scripting disabled: no built game assembly for `{entry}` under {} \
                 (run `dotnet build` there, or set ORRIN_GAME_DLL)",
                scripts_dir.display()
            );
            return None;
        };

        let scripting = Scripting::boot(&bindings_dir, &game_dll)?;

        // The game assembly's half of `register_components`, run at the same
        // point in the load as the engine's own — before anything can attach a
        // Behaviour, so the first scene to be saved already knows about them.
        scripting.register_components(&mut self.registry);

        // One entry Behaviour; it finds or spawns everything else itself
        // through the script API.
        let entity = self
            .world
            .spawn_entity()
            .with(crate::scene::Name::new("Script Entry"))
            .id();
        scripting.attach(&mut self.world, entity, &entry);

        // Stress scripts are just the entry Behaviour again: one more dispatch
        // target each tick, through the identical attach path, so the load is
        // representative rather than a special case.
        let stress_scripts = self.stress.map_or(0, |spec| spec.scripts);
        for index in 0..stress_scripts {
            let entity = self
                .world
                .spawn_entity()
                .with(crate::scene::Name::new(format!("Stress Script {index}")))
                .id();
            scripting.attach(&mut self.world, entity, &entry);
        }
        if stress_scripts > 0 {
            tracing::info!("stress load added — {stress_scripts} scripted entities");
        }

        let watcher = match BuildWatcher::for_game_assembly(&game_dll, Some(&bindings_dir)) {
            _ if self.shipped => None,
            Ok(watcher) => Some(watcher),
            Err(reason) => {
                tracing::warn!("rebuild-on-save is off: {reason}");
                self.world.resource_mut::<BuildStatus>().disable(reason);
                None
            }
        };

        Some(Scripts { scripting, watcher })
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.active.is_some() {
            return;
        }

        let title = match &self.project {
            Some(project) if self.shipped => project.name().to_owned(),
            _ => "Orrin".to_owned(),
        };
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title(title))
                .unwrap(),
        );
        let size = window.inner_size();
        let present = *self.world.resource::<PresentSettings>();
        let mut renderer = match self.instance.as_ref() {
            Some(instance) => {
                let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
                VulkanRenderer::new(instance, surface, [size.width, size.height], present)
            }
            None => metal_renderer(&window, [size.width, size.height], present),
        };

        let device = renderer.device_label();
        self.world.resource_mut::<FrameStats>().set_device(device);
        self.open_asset_database(crate::asset_db::Target::of(&renderer));
        if self.shipped {
            crate::scene::entities::install_library(&mut self.world, &mut renderer);
            self.open_game();
        } else {
            self.scene.begin(&mut self.world, &mut renderer);
        }
        if let Some(spec) = self.stress {
            spawn_stress_scene(&mut self.world, &spec);
        }
        self.camera_controller
            .sync_from(&self.world.resource::<Camera>());

        // Attach one entry Behaviour; it finds or spawns everything else itself
        // through the script API. Both the scripts directory and the entry type
        // resolve as env override > project manifest > built-in default, so a
        // developer can always point a run at something else without editing
        // the manifest.
        #[cfg(feature = "scripting")]
        if let Some(scripts) = self.boot_scripting() {
            self.stories.refresh(&self.world, &scripts.scripting);
            self.scripting = Some(scripts.scripting);
            self.build_watcher = scripts.watcher;
        }

        let editor = Editor::new(
            event_loop,
            window.clone(),
            renderer.max_texture_side(),
            self.project.as_ref(),
        );

        if self.shipped {
            // A game is played, not edited: there is no transport to press, and
            // no editor to press it in.
            self.world
                .resource_mut::<crate::scene::Play>()
                .set(crate::scene::PlayState::Playing);
            self.world.resource_mut::<Diagnostics>().overlay = false;
        }

        print_run_banner(&self.world, &renderer, self.backend, self.scene);

        self.active = Some(Active {
            window,
            renderer,
            editor,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        #[cfg_attr(not(feature = "scripting"), allow(unused_variables))] window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(active) = self.active.as_mut() else {
            return;
        };

        // The storybook's window is the story's alone: none of what follows —
        // the editor, the fly camera, the game's input — hears of it.
        #[cfg(feature = "scripting")]
        if self
            .story_window
            .as_ref()
            .is_some_and(|story| story.window.id() == window_id)
        {
            match &event {
                WindowEvent::CloseRequested => {
                    self.world
                        .resource_mut::<crate::ui::story::Storybook>()
                        .open = None;
                }
                WindowEvent::Resized(size) => {
                    active.renderer.resize_ui_window([size.width, size.height]);
                }
                _ => {}
            }
            self.stories.on_window_event(&event);
            return;
        }

        // Ahead of egui, and it has to be: the switch this answers is the one that
        // takes egui out of the frame, so it cannot be a key egui is asked about
        // first. F1 rather than a checkbox for the same reason — a UI that is off
        // cannot offer the control that turns it back on.
        if !self.shipped && pressed_overlay_toggle(&event) {
            let mut diagnostics = self.world.resource_mut::<Diagnostics>();
            diagnostics.overlay = !diagnostics.overlay;
        }
        let overlay_on = self.world.resource::<Diagnostics>().overlay;

        // The editor sees events first; when it doesn't want one, the camera
        // controller and the script-facing InputState may. All three apply the
        // same egui gate.
        //
        // Skipped entirely with the overlay off, and not merely ignored: egui
        // accumulates window events into the input it hands over at the start of
        // each frame, so feeding a UI that never runs one is an unbounded queue as
        // well as work nobody reads. It also means "overlay off" measures the
        // editor's absence rather than its silence.
        let egui_wants = if overlay_on {
            active.editor.on_window_event(&event)
        } else {
            false
        };
        self.world
            .resource_mut::<InputState>()
            .on_window_event(&event, egui_wants);
        // The fly camera is the editor's; in a game the camera is the scripts'.
        if !self.shipped {
            self.camera_controller
                .process_window_event(&event, egui_wants);
        }

        match event {
            WindowEvent::CloseRequested => {
                // Its surface before its window, which the renderer cannot
                // promise once the app is being torn down.
                active.renderer.close_ui_window();
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                active.renderer.resize([size.width, size.height]);
            }
            WindowEvent::RedrawRequested => {
                // First frame gets a zero delta — its "interval" would otherwise
                // be the whole startup (Vulkan init + scene build + CoreCLR boot,
                // 1–3 s with scripting), spiking every Spin and the first script
                // OnUpdate. Later deltas are clamped so a hitch (breakpoint,
                // window drag) can't teleport the sim on the frame after.
                const MAX_DELTA: f32 = 0.25;
                let now = Instant::now();
                let delta = match self.last_instant.replace(now) {
                    Some(prev) => (now - prev).as_secs_f32().min(MAX_DELTA),
                    None => 0.0,
                };
                self.world.resource_mut::<Time>().update(delta);
                self.world.resource_mut::<FrameStats>().record(delta);

                // Before the resolve, so a binding saved this frame is the one
                // the frame runs on. `Actions::apply` seeds the previous frame
                // from the current one, so the swap itself fires no edges.
                #[cfg(feature = "scripting")]
                if let Some(watcher) = &mut self.input_watcher {
                    profile_scope!("input watcher");
                    watcher.service(&self.world);
                }

                // Before anything reads the scene, so a prefab saved this frame
                // is what the frame draws.
                if let Some(watcher) = &mut self.prefab_watcher
                    && watcher.due()
                {
                    profile_scope!("prefab rescan");
                    for issue in crate::scene::prefab::rescan(&mut self.world, &self.registry) {
                        tracing::warn!("{issue}");
                    }
                }

                // Pads are polled, so their frame starts here — before the
                // resolve that reads them, and gated on focus so a stick held
                // in a background window drives nothing.
                if let Some(gamepads) = &mut self.gamepads {
                    profile_scope!("gamepads");
                    let mut input = self.world.resource_mut::<InputState>();
                    let focused = input.focused();
                    gamepads.pump(&mut input, focused);
                }

                // Actions resolve here and nowhere else: after this frame's
                // events have landed, before anything reads an action. Not
                // gated on an event arriving or on scripts running, because an
                // action's edges are derived by comparing this frame against
                // the last one — a frame that skipped the resolve would leave
                // the previous frame's press standing for a second frame.
                {
                    profile_scope!("input actions");
                    let input = self.world.resource::<InputState>();
                    self.world.resource_mut::<Actions>().resolve(&input);
                }
                // Drained into a local first: the `Actions` borrow a `for` over
                // the call would hold lives to the end of the loop, and the
                // resources here are one `RefCell` each.
                let frame = self.world.resource::<Time>().frame_count();
                let warnings = self.world.resource_mut::<Actions>().take_warnings();
                if !warnings.is_empty() {
                    let mut log = self.world.resource_mut::<LogBuffer>();
                    for warning in warnings {
                        log.push(LogLevel::Warning, warning, frame);
                    }
                }

                // A scene still loading on its own thread lands on the first frame
                // after it is ready; until then the loop keeps presenting.
                if crate::scene::entities::finish_pending_scene(
                    &mut self.world,
                    &mut active.renderer,
                ) {
                    self.camera_controller
                        .sync_from(&self.world.resource::<Camera>());
                }

                // Asked once, here: it consumes a paused game's step. Edit mode
                // runs no game code — see `scene::play`.
                let simulate = self.world.resource_mut::<crate::scene::Play>().advance();
                apply_cursor_grab(
                    &self.world,
                    self.shipped,
                    self.camera_controller.looking(),
                    &mut self.cursor_grabbed,
                    &active.window,
                );

                if simulate {
                    profile_scope!("spin");
                    systems::spin(&self.world, delta);
                }

                // Collision reads world transforms, so they have to be current
                // as of the last thing that wrote a local one — `spin`.
                {
                    profile_scope!("propagate");
                    crate::scene::propagate_transforms(&mut self.world);
                }

                // After the transform-mutating systems and before the script
                // tick, so the events scripts receive match the positions
                // they'll read this frame.
                if simulate {
                    profile_scope!("physics");
                    crate::physics::run(&mut self.world, delta);
                }

                // The UI is immediate: what this tick draws replaces the last
                // tick's. A paused game ticks nothing and keeps its UI on
                // screen; edit mode runs no game code and shows none.
                //
                // The game's screen is the window, unless the editor is drawn
                // and previewing a resolution — then it is whatever the editor
                // set last time it drew (see `editor::preview`).
                let extent = active.renderer.extent();
                if !overlay_on || self.shipped || self.world.get_resource::<GameScreen>().is_none()
                {
                    self.world.insert_resource(GameScreen::window(extent));
                }
                if simulate {
                    let size = self.world.resource::<GameScreen>().size;
                    let size = size.map(|side| side.round() as u32);
                    self.world.resource_mut::<UiCanvas>().begin(size);
                } else if !self.world.resource::<crate::scene::Play>().is_live() {
                    self.world.resource_mut::<UiCanvas>().clear();
                }
                // Against what the last frame drew, which is what the player
                // pointed at. Edit mode has no pointer state, so a button does
                // not come back from Stop still lit.
                if simulate {
                    profile_scope!("ui events");
                    crate::ui::events::update(&mut self.world);
                    crate::scene::entities::spawn_from_clicks(&mut self.world);
                } else if !self.world.resource::<crate::scene::Play>().is_live() {
                    self.world
                        .resource_mut::<crate::ui::events::UiPointer>()
                        .reset();
                    self.world
                        .resource_mut::<crate::ui::events::UiEvents>()
                        .clear();
                }

                // A reload requested from the editor last frame lands here:
                // before the tick, after physics, with no dispatch window
                // open and no world borrow held — the only point in the frame
                // where managed objects may be destroyed and re-created.
                #[cfg(feature = "scripting")]
                {
                    let mut reload_requested = active.editor.take_script_reload_request();
                    // A green rebuild raises the button's own request rather
                    // than reloading itself, so both routes converge on the one
                    // safe point below and a failed build never reaches it.
                    if let Some(watcher) = &mut self.build_watcher {
                        profile_scope!("build watcher");
                        reload_requested |= watcher.service(&self.world);
                    }
                    if let Some(scripting) = &self.scripting {
                        if reload_requested {
                            profile_scope!("script reload");
                            self.stories.close();
                            let outcome = scripting.reload(&mut self.world);
                            let frame = self.world.resource::<Time>().frame_count();
                            self.world.resource_mut::<LogBuffer>().push(
                                outcome.level(),
                                outcome.to_string(),
                                frame,
                            );
                            // Only a swap that happened means the session is
                            // running what was built; a rejected one leaves the
                            // compiled code still ahead of the live code.
                            if matches!(outcome, crate::scripting::ReloadOutcome::Swapped { .. }) {
                                // Only after a swap that actually happened: a
                                // rejected reload leaves the previous assembly
                                // live, and its registry entries with it.
                                self.registry.clear_game();
                                scripting.register_components(&mut self.registry);
                                self.world
                                    .resource_mut::<crate::build_watcher::BuildStatus>()
                                    .reloaded();
                            }
                            // A rejected build rebuilds the story it closed
                            // from the code still running.
                            self.stories.refresh(&self.world, scripting);
                        }
                        scripting.attach_pending(&mut self.world, &self.registry);
                        if simulate {
                            profile_scope!("script tick");
                            scripting.tick(&mut self.world, &self.registry, delta);
                        }
                    }
                }

                // After the tick, so a tween started in it moves this frame,
                // and before the layout, which draws where it moved to. A
                // paused game's tweens hold; a stopped one's are gone with the
                // scene they were driving.
                if simulate {
                    profile_scope!("tweens");
                    crate::ui::tween::advance(&self.world, delta);
                } else if !self.world.resource::<crate::scene::Play>().is_live() {
                    self.world
                        .resource_mut::<crate::ui::tween::UiTweens>()
                        .clear();
                }

                // Everything the stream has heard since the last frame, this
                // frame's script tick included, into the buffer the console
                // is about to draw.
                self.console
                    .drain_into(&mut self.world.resource_mut::<LogBuffer>(), frame);

                // Before extraction: the UI may spawn/despawn/edit entities.
                //
                // Not run at all with the overlay off. That is deliberate and it
                // is the whole measurement: `Editor::run` opens an egui pass that
                // only `Editor::draw` closes, so running one without drawing it
                // would leave the pass unbalanced — and the UI's own layout and
                // tessellation are most of what the editor costs, so skipping the
                // draw alone would answer a question nobody asked.
                // And not every frame with it on: the UI is rebuilt at its own
                // cadence and redrawn as it was in between — see
                // `editor::cadence`. Decided once, here, because `run` and
                // `overlay` are one egui pass and have to agree.
                let ui_due = overlay_on && active.editor.ui_due(std::time::Instant::now());
                if ui_due {
                    profile_scope!("editor");
                    active.editor.run(&mut self.world, &self.registry);
                    if active.editor.take_camera_moved() {
                        self.camera_controller
                            .sync_from(&self.world.resource::<Camera>());
                    }
                }

                // After the UI, so the editor's own camera edits are the
                // baseline the controller builds on.
                if !self.shipped {
                    let before = *self.world.resource::<Camera>();
                    self.camera_controller
                        .update(&mut self.world.resource_mut::<Camera>(), delta);
                    let after = *self.world.resource::<Camera>();
                    if overlay_on
                        && (before.position != after.position
                            || before.target != after.target
                            || before.up != after.up)
                    {
                        active.editor.camera_moved();
                    }
                }

                // A second pass, because the physics write-back, the script tick,
                // and the editor have all written local transforms since the
                // first one. Extraction reads world transforms, so without this
                // the frame would draw everything one frame behind.
                {
                    profile_scope!("propagate");
                    crate::scene::propagate_transforms(&mut self.world);
                }
                // After the propagation, so an outline is where its body is
                // drawn this frame.
                if overlay_on && !self.shipped {
                    active.editor.outline_colliders(&self.world);
                }

                // After the propagation, so the view is where its camera is this
                // frame. Edit mode keeps the editor's fly camera — see
                // `scene::camera`.
                if self.world.resource::<crate::scene::Play>().is_live() {
                    crate::scene::follow_active_camera(&self.world);
                }
                // After the camera follows, so the listener is where the frame
                // is drawn from, and every emitter where it is drawn.
                {
                    profile_scope!("audio");
                    crate::audio::update(&self.world);
                }

                // After the editor and the script tick, either of which can load
                // a scene or edit its settings, and before anything reads them.
                {
                    profile_scope!("render settings");
                    crate::scene::render_settings::sync(&mut self.world);
                }

                // After everything that can issue an asset handle — a scene the
                // editor loaded, a mesh dropped in the inspector, the script
                // tick — and before extraction, so no pending handle reaches a
                // draw and a reimported asset draws from this frame on.
                {
                    profile_scope!("assets");
                    crate::asset_db::service(&self.world, &mut active.renderer);
                }
                // After the service, with the backend in hand, for the same
                // reason: an image the tick drew first gets its slot here. The
                // retained trees are resolved first, every frame and in edit
                // mode too, against the window the frame is about to fill.
                {
                    profile_scope!("ui");
                    crate::ui::update(&mut self.world, active.renderer.extent());
                    crate::gfx::ui::paint(&self.world, &mut active.renderer, &mut self.ui);
                }
                // The open story, laid out in its own world for its own
                // window, which is drawn after the frame below.
                #[cfg(feature = "scripting")]
                {
                    sync_story_window(
                        event_loop,
                        &self.world,
                        &mut self.story_window,
                        &mut self.stories,
                        self.instance.as_ref(),
                        &mut active.renderer,
                    );
                    if self.story_window.is_some()
                        && let Some(scripting) = &self.scripting
                    {
                        profile_scope!("story");
                        self.stories
                            .update(&mut self.world, scripting, &self.registry, delta);
                        let painter = self
                            .story_ui
                            .get_or_insert_with(crate::gfx::ui::Painter::new);
                        let renderer = &mut active.renderer;
                        self.stories.paint(&mut self.world, |story| {
                            crate::gfx::ui::paint(story, renderer, painter)
                        });
                    }
                }

                {
                    profile_scope!("extract");
                    // The frustum has to be the one this frame draws with, so
                    // the aspect comes from the swapchain the pass will use.
                    let extent = active.renderer.extent();
                    let aspect = extent[0] as f32 / extent[1].max(1) as f32;
                    systems::extract_lighting(&self.world, &mut self.lighting);
                    // Beside the lighting and for the same reason: a decal is
                    // scene data two existing passes read, not a queue.
                    systems::extract_decals(&self.world, aspect, &mut self.decals);
                    // Cascades are fitted before extraction because the caster
                    // lists are culled against them: a shadow pass needs what
                    // reaches its box, which is not what the camera can see.
                    let shadow_settings = *self.world.resource::<ShadowSettings>();
                    self.cascades = if shadow_settings.enabled {
                        cascades(
                            &self.world.resource::<Camera>().clone(),
                            aspect,
                            self.lighting.sun.direction,
                            &shadow_settings.cascade_config(),
                        )
                    } else {
                        CascadeSet::default()
                    };
                    // Same reason the cascades are fitted first, and it has to
                    // be after `extract_lighting`: the atlas spends its tiles on
                    // the lights that frame produced, and the caster lists are
                    // culled against the reach of exactly those.
                    self.atlas = match shadow_settings.atlas_config() {
                        Some(config) => crate::gfx::punctual::fit(
                            &self.lighting,
                            self.world.resource::<Camera>().position,
                            &config,
                        ),
                        None => ShadowAtlas::default(),
                    };
                    systems::extract_geometry(&self.world, aspect, &mut self.geometry);
                    // Copy this frame's debug lines out (they're Copy) so the render
                    // borrow below doesn't overlap the world borrow.
                    self.debug_lines.clear();
                    self.debug_lines
                        .extend_from_slice(self.world.resource::<DebugLines>().lines());
                }
                let camera = *self.world.resource::<Camera>();
                let ssao = *self.world.resource::<SsaoSettings>();
                let contact_shadows = *self.world.resource::<ContactShadowSettings>();
                let ssr = *self.world.resource::<SsrSettings>();
                let subsurface = *self.world.resource::<SubsurfaceSettings>();
                let transparency = *self.world.resource::<TransparencySettings>();
                let refraction = *self.world.resource::<RefractionSettings>();
                let taa = *self.world.resource::<TaaSettings>();
                let motion_blur = *self.world.resource::<MotionBlurSettings>();
                let dof = *self.world.resource::<DofSettings>();
                let shadow_settings = *self.world.resource::<ShadowSettings>();
                let bloom = *self.world.resource::<BloomSettings>();
                let hdr = *self.world.resource::<HdrSettings>();
                let environment = self.world.resource::<EnvironmentSettings>().clone();
                let fog = *self.world.resource::<FogSettings>();
                if environment.reload_requested {
                    self.world
                        .resource_mut::<EnvironmentSettings>()
                        .reload_requested = false;
                    load_environment(
                        &mut active.renderer,
                        self.project.as_ref(),
                        &environment.hdri,
                    );
                }
                // Between the editor's pass and this frame's recording, so a
                // material dragged in the inspector draws changed this frame.
                crate::scene::apply_material_edits(&mut self.world, &mut active.renderer);
                // Stamped into this frame's GPU queries so the readback, some
                // frames later, can file its spans against the right frame.
                let profiler_frame = self.world.resource::<Profiler>().frame_index();
                let dt = self.world.resource::<Time>().delta_time();

                // Asked for every frame rather than only on a change: the renderer
                // compares against what it applied and recreates the swapchain
                // only when the two differ, so this is a comparison and not a
                // teardown.
                active
                    .renderer
                    .set_present(*self.world.resource::<PresentSettings>());

                let Active {
                    renderer, editor, ..
                } = active;
                let overlay: Option<crate::gfx::vulkan::Overlay<'_>> = overlay_on.then(|| {
                    if ui_due {
                        editor.overlay()
                    } else {
                        editor.cached_overlay()
                    }
                });
                {
                    profile_scope!("render submit");
                    renderer.render_with_overlay(
                        self.geometry.visible(),
                        self.geometry.transparent(),
                        self.geometry.refractive(),
                        &self.decals,
                        &self.lighting,
                        &camera,
                        &ssao,
                        &contact_shadows,
                        &ssr,
                        &subsurface,
                        &transparency,
                        &refraction,
                        &taa,
                        &motion_blur,
                        &dof,
                        &bloom,
                        &hdr,
                        &environment,
                        &fog,
                        dt,
                        &self.debug_lines,
                        profiler_frame,
                        // Either half is reason enough to hand one over: the sun
                        // and the punctual lights are independent, and a scene
                        // with cascades off and a lamp casting is an ordinary
                        // thing to want.
                        (self.cascades.count > 0 || !self.atlas.faces.is_empty()).then_some({
                            ShadowFrame {
                                cascades: &self.cascades,
                                atlas: &self.atlas,
                                settings: &shadow_settings,
                            }
                        }),
                        Some(self.ui.frame()),
                        overlay,
                    );
                }

                #[cfg(feature = "scripting")]
                if renderer.has_ui_window()
                    && let Some(painter) = &self.story_ui
                {
                    profile_scope!("storybook window");
                    renderer.render_ui_window(painter.frame());
                }

                // Before `gpu_frame_ms`, which reads the whole-frame pair this
                // drain refreshes.
                renderer.drain_gpu_spans(&mut self.world.resource_mut::<Profiler>());

                let gpu_ms = renderer.gpu_frame_ms();
                let (vram_used, vram_total) = renderer.gpu_memory();
                self.world
                    .resource_mut::<FrameStats>()
                    .set_gpu_stats(gpu_ms, vram_used, vram_total);

                // Expire debug lines now that this frame has drawn them: a
                // one-frame line (expiry == its spawn time) is dropped, a timed
                // one survives until its lifetime elapses.
                let now = self.world.resource::<Time>().elapsed_time();
                self.world.resource_mut::<DebugLines>().sweep(now);

                // Clear the one-frame pressed/released edges now that scripts
                // have observed them during the tick above.
                self.world.resource_mut::<InputState>().end_frame();
                #[cfg(feature = "scripting")]
                self.stories.end_frame();

                // Last, so every scope guard above has dropped and this frame's
                // CPU spans are complete. Its GPU spans arrive later and are
                // filed against this frame by index.
                self.world.resource_mut::<Profiler>().end_frame();
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if !self.shipped {
            self.camera_controller.process_device_event(&event);
        }
        self.world
            .resource_mut::<InputState>()
            .on_device_event(&event);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(active) = self.active.as_ref() {
            active.window.request_redraw();
        }
    }
}

/// Hold or release the cursor for whichever of the fly camera and the game
/// wants it — see `scene::input::cursor`. Once a frame, so the two cannot
/// release each other's grab, and before the scripts tick, so a game that asks
/// this frame looks from the next.
fn apply_cursor_grab(
    world: &World,
    shipped: bool,
    fly_camera_looking: bool,
    grabbed: &mut bool,
    window: &Window,
) {
    /// `Escape` in the engine's key numbering; see `scene::input::keys`.
    const ESCAPE: u32 = 46;
    let live = world.resource::<crate::scene::Play>().is_live();
    let (focused, escaped) = {
        let input = world.resource::<InputState>();
        (input.focused(), input.key_pressed(ESCAPE))
    };
    let lock = {
        let mut lock = world.resource_mut::<crate::scene::CursorLock>();
        if !live || (escaped && !shipped) {
            lock.requested = false;
        }
        *lock
    };
    let grab = crate::scene::input::wants_grab(fly_camera_looking, live, focused, lock);
    if grab == *grabbed {
        return;
    }
    *grabbed = grab;
    window.set_cursor_visible(!grab);
    let mode = if grab {
        CursorGrabMode::Locked
    } else {
        CursorGrabMode::None
    };
    // Locked is unsupported on some platforms; fall back to Confined.
    let _ = window.set_cursor_grab(mode).or_else(|_| {
        window.set_cursor_grab(if grab {
            CursorGrabMode::Confined
        } else {
            CursorGrabMode::None
        })
    });
}

/// Whether this event is the press that toggles the editor overlay.
///
/// Read off the raw winit event rather than through [`InputState`], because that
/// table is the one C# `KeyCode` mirrors field for field — an editor keybind is
/// no reason to grow a scripting ABI.
fn pressed_overlay_toggle(event: &WindowEvent) -> bool {
    use winit::event::{ElementState, KeyEvent};
    use winit::keyboard::{KeyCode, PhysicalKey};

    matches!(
        event,
        WindowEvent::KeyboardInput {
            event: KeyEvent {
                physical_key: PhysicalKey::Code(KeyCode::F1),
                state: ElementState::Pressed,
                repeat: false,
                ..
            },
            ..
        }
    )
}

/// The Vulkan instance a Vulkan run draws through, with the validation
/// messenger that has to live as long as it does.
fn create_vulkan_instance(
    event_loop: &EventLoop<()>,
) -> (Arc<Instance>, Option<DebugUtilsMessenger>) {
    let library = VulkanLibrary::new().expect("failed to load vulkan library");
    let mut enabled_extensions = Surface::required_extensions(event_loop).unwrap();

    // Architecture §3.5: validation on in every dev build. Silently skipped
    // when the layer isn't installed, so a machine without the Vulkan SDK
    // (and CI) still runs — a GPU crash with no named cause is the thing
    // this exists to prevent.
    let validation = should_validate() && has_validation_layer(&library);
    if validation {
        enabled_extensions.ext_debug_utils = true;
    }

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions,
            enabled_layers: if validation {
                vec![VALIDATION_LAYER.to_owned()]
            } else {
                Vec::new()
            },
            ..Default::default()
        },
    )
    .expect("failed to create instance");

    let debug_messenger = validation.then(|| attach_debug_messenger(&instance));
    if validation {
        tracing::info!("Vulkan validation layer enabled");
    } else if should_validate() {
        tracing::warn!(
            "validation requested but `{VALIDATION_LAYER}` isn't installed; \
             GPU errors will not be named (install the Vulkan SDK)"
        );
    }
    (instance, debug_messenger)
}

/// A renderer whose frames end in `window`'s Metal layer.
#[cfg(target_os = "macos")]
fn metal_renderer(window: &Window, extent: [u32; 2], present: PresentSettings) -> VulkanRenderer {
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    let RawWindowHandle::AppKit(handle) = window
        .window_handle()
        .expect("the window has no native handle")
        .as_raw()
    else {
        unreachable!("a macOS window is an AppKit one");
    };
    VulkanRenderer::windowed_metal(handle.ns_view, extent, present).unwrap_or_else(|reason| {
        tracing::error!("ORRIN_BACKEND=metal cannot start: {reason}");
        std::process::exit(1);
    })
}

#[cfg(not(target_os = "macos"))]
fn metal_renderer(_: &Window, _: [u32; 2], _: PresentSettings) -> VulkanRenderer {
    unreachable!("BackendChoice never picks Metal off macOS")
}

/// The storybook's window, and the story it was opened or retitled for.
#[cfg(feature = "scripting")]
struct StoryWindow {
    window: Arc<Window>,
    story: String,
}

/// Open, retitle or close the storybook's window to match the story the
/// editor has open. A window that cannot be made closes the story rather than
/// leaving it open with nowhere to draw.
#[cfg(feature = "scripting")]
fn sync_story_window(
    event_loop: &ActiveEventLoop,
    world: &World,
    shown: &mut Option<StoryWindow>,
    runner: &mut crate::storybook::StoryRunner,
    instance: Option<&Arc<Instance>>,
    renderer: &mut VulkanRenderer,
) {
    use crate::storybook::{WindowChange, window_change};
    use crate::ui::story::Storybook;

    let wanted = world.resource::<Storybook>().open.clone();
    let title = |story: &str| format!("Storybook — {story}");
    match window_change(wanted.as_deref(), shown.as_ref().map(|w| w.story.as_str())) {
        WindowChange::Keep => {}
        WindowChange::Retitle => {
            if let (Some(shown), Some(story)) = (shown.as_mut(), wanted) {
                shown.window.set_title(&title(&story));
                shown.story = story;
            }
        }
        WindowChange::Close => {
            renderer.close_ui_window();
            *shown = None;
            runner.close();
        }
        WindowChange::Open => {
            let story = wanted.unwrap_or_default();
            let attributes = Window::default_attributes()
                .with_title(title(&story))
                .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
            let opened = event_loop
                .create_window(attributes)
                .map_err(|e| e.to_string())
                .and_then(|window| {
                    let window = Arc::new(window);
                    let size = window.inner_size();
                    let surface = match instance {
                        Some(instance) => Surface::from_window(instance.clone(), window.clone())
                            .map(crate::gfx::vulkan::UiSurface::Vulkan)
                            .map_err(|e| e.to_string()),
                        None => metal_surface(&window),
                    }?;
                    renderer.open_ui_window(surface, [size.width, size.height])?;
                    runner.set_extent([size.width, size.height]);
                    Ok(window)
                });
            match opened {
                Ok(window) => *shown = Some(StoryWindow { window, story }),
                Err(reason) => {
                    tracing::warn!("the storybook window could not open: {reason}");
                    world.resource_mut::<Storybook>().open = None;
                }
            }
        }
    }
}

#[cfg(all(feature = "scripting", target_os = "macos"))]
fn metal_surface(window: &Window) -> Result<crate::gfx::vulkan::UiSurface, String> {
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    match window.window_handle().map_err(|e| e.to_string())?.as_raw() {
        RawWindowHandle::AppKit(handle) => Ok(crate::gfx::vulkan::UiSurface::Metal(handle.ns_view)),
        _ => Err("a macOS window is an AppKit one".into()),
    }
}

#[cfg(all(feature = "scripting", not(target_os = "macos")))]
fn metal_surface(_: &Window) -> Result<crate::gfx::vulkan::UiSurface, String> {
    unreachable!("BackendChoice never picks Metal off macOS")
}

/// Print what this run measures with, once, at startup.
///
/// Every one of these changes a frame-time number without changing a pixel, so a
/// figure quoted without them is not reproducible: a debug build with validation
/// on and a release build with it off differ by more than most of the work in the
/// renderer. The banner exists so a pasted log is a complete description of the
/// run it came from.
fn print_run_banner(
    world: &World,
    renderer: &VulkanRenderer,
    backend: BackendChoice,
    scene: SceneChoice,
) {
    let present = renderer
        .applied_present()
        .unwrap_or_else(|| "offscreen".to_string());
    let extent = renderer.extent();

    tracing::info!(
        "run config: {} build | {} | debug assertions {} | validation {} | present {} | \
         {}x{} | MSAA {} | overlay {} | GPU pass timings {} | scene {}",
        if cfg!(debug_assertions) {
            "unoptimised"
        } else {
            "optimised"
        },
        renderer.device_label(),
        on_off(cfg!(debug_assertions)),
        on_off(should_validate() && backend == BackendChoice::Vulkan),
        present,
        extent[0],
        extent[1],
        // What the frame actually rasterises at, not the constant it *would*
        // use: MSAA is off by default now, and a banner quoting `Sample4` on a
        // one-sample run would misdescribe the number beside it.
        if world.resource::<TaaSettings>().msaa {
            format!("{:?}", crate::gfx::vulkan::MSAA_SAMPLES)
        } else {
            "off".to_string()
        },
        on_off(world.resource::<Diagnostics>().overlay),
        on_off(crate::profile::is_enabled() && crate::profile::gpu_passes_enabled()),
        scene.label(),
    );
}

fn on_off(value: bool) -> &'static str {
    if value { "on" } else { "off" }
}

const VALIDATION_LAYER: &str = "VK_LAYER_KHRONOS_validation";

/// On by default in dev builds; `ORRIN_VALIDATION=0`/`1` overrides either way,
/// so a release build can be checked without rebuilding it as debug.
fn should_validate() -> bool {
    match std::env::var("ORRIN_VALIDATION").ok().as_deref() {
        Some("0") | Some("false") => false,
        Some(_) => true,
        None => cfg!(debug_assertions),
    }
}

fn has_validation_layer(library: &Arc<VulkanLibrary>) -> bool {
    library
        .layer_properties()
        .map(|mut layers| layers.any(|layer| layer.name() == VALIDATION_LAYER))
        .unwrap_or(false)
}

/// Print validation messages with the offending object named, per architecture
/// §3.5. Errors go to stderr so a crash log carries them.
fn attach_debug_messenger(instance: &Arc<Instance>) -> DebugUtilsMessenger {
    // SAFETY: the callback only formats and prints; it makes no Vulkan calls.
    let callback = unsafe {
        DebugUtilsMessengerCallback::new(|severity, message_type, data| {
            let id = data
                .message_id_name
                .map(|name| format!("{name}: "))
                .unwrap_or_default();
            let message = data.message;
            if severity.intersects(DebugUtilsMessageSeverity::ERROR) {
                tracing::error!("vulkan: {id}{message}");
            } else if severity.intersects(DebugUtilsMessageSeverity::WARNING) {
                tracing::warn!("vulkan: {id}{message}");
            } else {
                tracing::info!("vulkan: {id}{message}");
            }
            let _ = message_type;
        })
    };

    // `ext_debug_utils` was enabled on the instance above, which is the only
    // precondition beyond the callback's.
    DebugUtilsMessenger::new(
        instance.clone(),
        DebugUtilsMessengerCreateInfo {
            message_severity: DebugUtilsMessageSeverity::ERROR | DebugUtilsMessageSeverity::WARNING,
            message_type: DebugUtilsMessageType::GENERAL
                | DebugUtilsMessageType::VALIDATION
                | DebugUtilsMessageType::PERFORMANCE,
            ..DebugUtilsMessengerCreateInfo::user_callback(callback)
        },
    )
    .expect("failed to create the validation messenger")
}

/// Bake the environment from `hdri`, and report what happened in one line.
///
/// The path resolves against the project's assets directory, or `assets/`
/// beside the working directory when there is no project — the same
/// manifest-else-built-in-default shape the scripts directory resolves by.
///
/// A failure changes nothing: the session keeps the environment it already had,
/// the way a rejected script build keeps the code it had. That matters more
/// here than it looks, because the alternative to "keep the old sky" is "no sky
/// and black metals" for a typo in a filename.
fn load_environment(
    renderer: &mut VulkanRenderer,
    project: Option<&orrin_project::Project>,
    hdri: &str,
) {
    let hdri = hdri.trim();
    if hdri.is_empty() {
        tracing::warn!("no environment file named; give a path relative to the assets directory");
        return;
    }

    let assets_dir =
        project.map_or_else(|| PathBuf::from("assets"), |project| project.assets_dir());

    match crate::scene::sky_pixels(&assets_dir, hdri) {
        Ok((pixels, width, height)) => {
            renderer.load_environment(&pixels, width, height);
            tracing::info!("environment baked from `{hdri}` ({width}x{height})");
        }
        Err(error) => tracing::error!("{error}"),
    }
}
