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
use crate::gfx::shadows::{CascadeSet, MAX_CASCADES, cascades};
use crate::gfx::vulkan::ShadowFrame;
use crate::gfx::vulkan::VulkanRenderer;
use crate::gfx::{RenderBackend, RenderItem, SceneLighting};
use crate::profile::Profiler;
use crate::profile_scope;
use crate::scene::entities::{StressSpec, build_default_scene, spawn_stress_scene};
use crate::scene::{
    AmbientLight, BloomSettings, Camera, Culling, DebugLine, DebugLines, EnvironmentSettings,
    FogSettings, HdrSettings, InputState, LogBuffer, LogLevel, ShadowSettings, SsaoSettings, Time,
    load_hdri,
};
use crate::stats::FrameStats;
use crate::systems;
use orrin_ecs::World;
use orrin_registry::Registry;

struct Active {
    window: Arc<Window>,
    renderer: VulkanRenderer,
    editor: Editor,
}

pub struct App {
    instance: Arc<Instance>,
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
    render_items: Vec<RenderItem>,
    lighting: SceneLighting,
    /// This frame's cascade matrices and the caster list each was culled
    /// against. Kept on the app so the per-cascade `Vec`s keep their capacity
    /// across frames instead of reallocating.
    cascades: CascadeSet,
    shadow_casters: [Vec<RenderItem>; MAX_CASCADES],
    /// This frame's debug lines, copied out of the `DebugLines` resource so the
    /// renderer borrow doesn't overlap the world borrow.
    debug_lines: Vec<DebugLine>,
    #[cfg(feature = "scripting")]
    scripting: Option<crate::scripting::Scripting>,
    /// Rebuild-on-save. `None` when the project's layout gave nothing to watch
    /// — see `BuildWatcher::for_game_assembly`.
    #[cfg(feature = "scripting")]
    build_watcher: Option<crate::build_watcher::BuildWatcher>,
    /// The Orrin project this run was launched inside, if any. `None` means
    /// the engine is running standalone on its built-in demo scene — and, for
    /// the editor, that there is nowhere to keep themes or a layout.
    project: Option<orrin_project::Project>,
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
        let event_loop = EventLoop::new().unwrap();
        event_loop.set_control_flow(ControlFlow::Poll);

        let library = VulkanLibrary::new().expect("failed to load vulkan library");
        let mut enabled_extensions = Surface::required_extensions(&event_loop).unwrap();

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
            println!("orrin: Vulkan validation layer enabled");
        } else if should_validate() {
            eprintln!(
                "orrin: validation requested but `{VALIDATION_LAYER}` isn't installed; \
                 GPU errors will not be named (install the Vulkan SDK)"
            );
        }

        let cwd = std::env::current_dir().unwrap_or_else(|err| {
            eprintln!("orrin: cannot read the current directory: {err}");
            std::process::exit(1);
        });

        // A manifest that exists but doesn't load is fatal: silently falling
        // back to the demo scene would hide the user's real project.
        let project = match orrin_project::Project::locate(&cwd) {
            Ok(project) => project,
            Err(err) => {
                eprintln!("orrin: {err}");
                std::process::exit(1);
            }
        };
        if let Some(project) = &project {
            println!(
                "orrin: project `{}` at {}",
                project.name(),
                project.root().display()
            );
        }

        let mut app = App {
            instance,
            active: None,
            world: World::default(),
            registry: Registry::new(),
            camera_controller: CameraController::new(),
            last_instant: None,
            render_items: Vec::new(),
            lighting: SceneLighting::default(),
            cascades: CascadeSet::default(),
            shadow_casters: Default::default(),
            debug_lines: Vec::new(),
            #[cfg(feature = "scripting")]
            scripting: None,
            #[cfg(feature = "scripting")]
            build_watcher: None,
            project,
            stress: StressSpec::from_env().filter(|spec| !spec.is_empty()),
            _debug_messenger: debug_messenger,
        };

        crate::scene::register_components(&mut app.registry);
        Self::install_default_resources(&mut app.world);

        event_loop.run_app(&mut app).unwrap();
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
        world.insert_resource(Time::new());
        world.insert_resource(AmbientLight::default());
        world.insert_resource(SsaoSettings::default());
        world.insert_resource(ShadowSettings::default());
        world.insert_resource(HdrSettings::default());
        world.insert_resource(BloomSettings::default());
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
        world.insert_resource(FrameStats::new());
        world.insert_resource(Profiler::default());
        world.insert_resource(InputState::new());
        world.insert_resource(crate::collision::CollisionState::default());
        world.insert_resource(LogBuffer::default());
        world.insert_resource(DebugLines::default());
        world.insert_resource(Culling::default());
        // Inserted whether or not a watcher ever starts: the Scripts panel reads
        // it either way, and `Off` is how it explains itself.
        #[cfg(feature = "scripting")]
        world.insert_resource(crate::build_watcher::BuildStatus::default());
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
            eprintln!(
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

        let game_dll = match std::env::var("ORRIN_GAME_DLL") {
            Ok(path) => Some(PathBuf::from(path)),
            Err(_) => match Scripting::assembly_of(&entry) {
                Some(assembly) => Scripting::find_game_assembly(&scripts_dir, assembly),
                None => {
                    eprintln!(
                        "scripting disabled: entry `{entry}` names no assembly — it must be \
                         assembly-qualified (`MyGame.Main, MyGame`) so the engine knows which \
                         DLL to load"
                    );
                    return None;
                }
            },
        };
        let Some(game_dll) = game_dll else {
            eprintln!(
                "scripting disabled: no built game assembly for `{entry}` under {} \
                 (run `dotnet build` there, or set ORRIN_GAME_DLL)",
                scripts_dir.display()
            );
            return None;
        };

        let scripting = Scripting::boot(&bindings_dir, &game_dll)?;

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
            println!("orrin: stress load added — {stress_scripts} scripted entities");
        }

        let watcher = match BuildWatcher::for_game_assembly(&game_dll, Some(&bindings_dir)) {
            Ok(watcher) => Some(watcher),
            Err(reason) => {
                eprintln!("rebuild-on-save is off: {reason}");
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

        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Orrin"))
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let size = window.inner_size();
        let mut renderer =
            VulkanRenderer::new(&self.instance, surface.clone(), [size.width, size.height]);

        build_default_scene(&mut self.world, &mut renderer);
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
            self.scripting = Some(scripts.scripting);
            self.build_watcher = scripts.watcher;
        }

        let editor = Editor::new(
            event_loop,
            surface,
            renderer.queue(),
            renderer.color_format(),
            self.project.as_ref(),
        );

        self.active = Some(Active {
            window,
            renderer,
            editor,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(active) = self.active.as_mut() else {
            return;
        };

        // The editor sees events first; when it doesn't want one, the camera
        // controller and the script-facing InputState may. All three apply the
        // same egui gate.
        let egui_wants = active.editor.on_window_event(&event);
        self.world
            .resource_mut::<InputState>()
            .on_window_event(&event, egui_wants);
        let was_looking = self.camera_controller.looking();
        self.camera_controller
            .process_window_event(&event, egui_wants);
        if self.camera_controller.looking() != was_looking {
            let looking = self.camera_controller.looking();
            active.window.set_cursor_visible(!looking);
            let grab = if looking {
                CursorGrabMode::Locked
            } else {
                CursorGrabMode::None
            };
            // Locked is unsupported on some platforms; fall back to Confined.
            let _ = active.window.set_cursor_grab(grab).or_else(|_| {
                active.window.set_cursor_grab(if looking {
                    CursorGrabMode::Confined
                } else {
                    CursorGrabMode::None
                })
            });
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
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

                {
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
                {
                    profile_scope!("collision");
                    crate::collision::run(&mut self.world);
                }

                // A reload requested from the editor last frame lands here:
                // before the tick, after collision, with no dispatch window
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
                                self.world
                                    .resource_mut::<crate::build_watcher::BuildStatus>()
                                    .reloaded();
                            }
                        }
                        {
                            profile_scope!("script tick");
                            scripting.tick(&mut self.world, delta);
                        }
                    }
                }

                // Before extraction: the UI may spawn/despawn/edit entities.
                {
                    profile_scope!("editor");
                    active.editor.run(&mut self.world, &self.registry);
                }

                // After the UI, so the editor's own camera edits are the
                // baseline the controller builds on.
                self.camera_controller
                    .update(&mut self.world.resource_mut::<Camera>(), delta);

                // A second pass, because collision resolution, the script tick,
                // and the editor have all written local transforms since the
                // first one. Extraction reads world transforms, so without this
                // the frame would draw everything one frame behind.
                {
                    profile_scope!("propagate");
                    crate::scene::propagate_transforms(&mut self.world);
                }

                {
                    profile_scope!("extract");
                    // The frustum has to be the one this frame draws with, so
                    // the aspect comes from the swapchain the pass will use.
                    let extent = active.renderer.extent();
                    let aspect = extent[0] as f32 / extent[1].max(1) as f32;
                    systems::extract_renderables(&self.world, aspect, &mut self.render_items);
                    systems::extract_lighting(&self.world, &mut self.lighting);
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
                    systems::extract_shadow_casters(
                        &self.world,
                        &self.cascades,
                        &mut self.shadow_casters,
                    );
                    // Copy this frame's debug lines out (they're Copy) so the render
                    // borrow below doesn't overlap the world borrow.
                    self.debug_lines.clear();
                    self.debug_lines
                        .extend_from_slice(self.world.resource::<DebugLines>().lines());
                }
                let camera = *self.world.resource::<Camera>();
                let ssao = *self.world.resource::<SsaoSettings>();
                let shadow_settings = *self.world.resource::<ShadowSettings>();
                let bloom = *self.world.resource::<BloomSettings>();
                let hdr = *self.world.resource::<HdrSettings>();
                let environment = self.world.resource::<EnvironmentSettings>().clone();
                if environment.reload_requested {
                    self.world
                        .resource_mut::<EnvironmentSettings>()
                        .reload_requested = false;
                    let message = load_environment(
                        &mut active.renderer,
                        self.project.as_ref(),
                        &environment.hdri,
                    );
                    let frame = self.world.resource::<Time>().frame_count();
                    self.world
                        .resource_mut::<LogBuffer>()
                        .push(message.0, message.1, frame);
                }
                // Stamped into this frame's GPU queries so the readback, some
                // frames later, can file its spans against the right frame.
                let profiler_frame = self.world.resource::<Profiler>().frame_index();
                let dt = self.world.resource::<Time>().delta_time();

                let Active {
                    renderer, editor, ..
                } = active;
                let mut overlay = |before, image| editor.draw(before, image);
                {
                    profile_scope!("render submit");
                    renderer.render_with_overlay(
                        &self.render_items,
                        &self.lighting,
                        &camera,
                        &ssao,
                        &bloom,
                        &hdr,
                        &environment,
                        dt,
                        &self.debug_lines,
                        profiler_frame,
                        (self.cascades.count > 0).then(|| ShadowFrame {
                            cascades: &self.cascades,
                            casters: &self.shadow_casters,
                            settings: &shadow_settings,
                        }),
                        &mut overlay,
                    );
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
        self.camera_controller.process_device_event(&event);
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
            let label = if severity.intersects(DebugUtilsMessageSeverity::ERROR) {
                "error"
            } else if severity.intersects(DebugUtilsMessageSeverity::WARNING) {
                "warning"
            } else {
                "info"
            };
            eprintln!(
                "[vulkan {label}] {}{}",
                data.message_id_name
                    .map(|name| format!("{name}: "))
                    .unwrap_or_default(),
                data.message
            );
            let _ = message_type;
        })
    };

    // SAFETY: `ext_debug_utils` was enabled on the instance above, which is the
    // only precondition beyond the callback's.
    unsafe {
        DebugUtilsMessenger::new(
            instance.clone(),
            DebugUtilsMessengerCreateInfo {
                message_severity: DebugUtilsMessageSeverity::ERROR
                    | DebugUtilsMessageSeverity::WARNING,
                message_type: DebugUtilsMessageType::GENERAL
                    | DebugUtilsMessageType::VALIDATION
                    | DebugUtilsMessageType::PERFORMANCE,
                ..DebugUtilsMessengerCreateInfo::user_callback(callback)
            },
        )
    }
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
) -> (LogLevel, String) {
    let hdri = hdri.trim();
    if hdri.is_empty() {
        return (
            LogLevel::Warning,
            "no environment file named; give a path relative to the assets directory".to_string(),
        );
    }

    let assets_dir =
        project.map_or_else(|| PathBuf::from("assets"), |project| project.assets_dir());

    match load_hdri(&assets_dir, hdri) {
        Ok(image) => {
            renderer.load_environment(&image.pixels, image.width, image.height);
            (
                LogLevel::Info,
                format!(
                    "environment baked from `{hdri}` ({}x{})",
                    image.width, image.height
                ),
            )
        }
        Err(error) => {
            // Also to the terminal: a run without the editor open has no
            // console panel to read, and a silently missing environment looks
            // identical to one that loaded and happened to be dark.
            eprintln!("orrin: {error}");
            (LogLevel::Error, error.to_string())
        }
    }
}
