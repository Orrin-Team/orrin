use std::sync::Arc;
use std::time::Instant;

use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::swapchain::Surface;
use vulkano::VulkanLibrary;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::camera_controller::CameraController;
use crate::editor::Editor;
use crate::gfx::vulkan::VulkanRenderer;
use crate::gfx::{RenderBackend, RenderItem, SceneLighting};
use crate::scene::entities::build_default_scene;
use crate::scene::{
    AmbientLight, Camera, DebugLine, DebugLines, HdrSettings, InputState, LogBuffer, SsaoSettings,
    Time,
};
use crate::stats::FrameStats;
use crate::systems;
use orrin_ecs::World;

struct Active {
    window: Arc<Window>,
    renderer: VulkanRenderer,
    editor: Editor,
}

pub struct App {
    instance: Arc<Instance>,
    active: Option<Active>,
    world: World,
    camera_controller: CameraController,
    /// Timestamp of the previous rendered frame; `None` until the first frame
    /// establishes a baseline. Delta is `now - last_instant`, never a
    /// difference of two large "seconds since start" floats (which quantizes).
    last_instant: Option<Instant>,
    render_items: Vec<RenderItem>,
    lighting: SceneLighting,
    /// This frame's debug lines, copied out of the `DebugLines` resource so the
    /// renderer borrow doesn't overlap the world borrow.
    debug_lines: Vec<DebugLine>,
    #[cfg(feature = "scripting")]
    scripting: Option<crate::scripting::Scripting>,
    /// The Orrin project this run was launched inside, if any. `None` means
    /// the engine is running standalone on its built-in demo scene.
    #[cfg(feature = "scripting")]
    project: Option<orrin_project::Project>,
}

impl App {
    pub fn run() {
        let event_loop = EventLoop::new().unwrap();
        event_loop.set_control_flow(ControlFlow::Poll);

        let library = VulkanLibrary::new().expect("failed to load vulkan library");
        let required_extensions = Surface::required_extensions(&event_loop).unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

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
            camera_controller: CameraController::new(),
            last_instant: None,
            render_items: Vec::new(),
            lighting: SceneLighting::default(),
            debug_lines: Vec::new(),
            #[cfg(feature = "scripting")]
            scripting: None,
            #[cfg(feature = "scripting")]
            project,
        };

        app.world.insert_resource(Camera::default());
        app.world.insert_resource(Time::new());
        app.world.insert_resource(AmbientLight::default());
        app.world.insert_resource(SsaoSettings::default());
        app.world.insert_resource(HdrSettings::default());
        app.world.insert_resource(FrameStats::new());
        app.world.insert_resource(InputState::new());
        app.world.insert_resource(crate::collision::CollisionState::default());
        app.world.insert_resource(LogBuffer::default());
        app.world.insert_resource(DebugLines::default());

        event_loop.run_app(&mut app).unwrap();
    }

    /// Boot the script host and attach the project's entry Behaviour.
    ///
    /// Two assemblies, resolved separately because they have different owners:
    /// the *bindings* (`Orrin.dll`) belong to the engine and stay in the
    /// default load context forever, while the *game* assembly belongs to the
    /// project and is what a hot reload swaps. Each follows env > manifest >
    /// built-in default, with its own env override.
    #[cfg(feature = "scripting")]
    fn boot_scripting(&mut self) -> Option<crate::scripting::Scripting> {
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
        Some(scripting)
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
        self.camera_controller
            .sync_from(&self.world.resource::<Camera>());

        // Attach one entry Behaviour; it finds or spawns everything else itself
        // through the script API. Both the scripts directory and the entry type
        // resolve as env override > project manifest > built-in default, so a
        // developer can always point a run at something else without editing
        // the manifest.
        #[cfg(feature = "scripting")]
        {
            self.scripting = self.boot_scripting();
        }

        let editor = Editor::new(event_loop, surface, renderer.queue(), renderer.color_format());

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
        self.camera_controller.process_window_event(&event, egui_wants);
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

                systems::spin(&self.world, delta);

                // After the transform-mutating systems and before the script
                // tick, so the events scripts receive match the positions
                // they'll read this frame.
                crate::collision::run(&mut self.world);

                // A reload requested from the editor last frame lands here:
                // before the tick, after collision, with no dispatch window
                // open and no world borrow held — the only point in the frame
                // where managed objects may be destroyed and re-created.
                #[cfg(feature = "scripting")]
                {
                    let reload_requested = active.editor.take_script_reload_request();
                    if let Some(scripting) = &self.scripting {
                        if reload_requested {
                            let outcome = scripting.reload(&mut self.world);
                            let frame = self.world.resource::<Time>().frame_count();
                            self.world.resource_mut::<LogBuffer>().push(
                                outcome.level(),
                                outcome.to_string(),
                                frame,
                            );
                        }
                        scripting.tick(&mut self.world, delta);
                    }
                }

                // Before extraction: the UI may spawn/despawn/edit entities.
                active.editor.run(&mut self.world);

                // After the UI, so the editor's own camera edits are the
                // baseline the controller builds on.
                self.camera_controller
                    .update(&mut self.world.resource_mut::<Camera>(), delta);

                systems::extract_renderables(&self.world, &mut self.render_items);
                systems::extract_lighting(&self.world, &mut self.lighting);
                // Copy this frame's debug lines out (they're Copy) so the render
                // borrow below doesn't overlap the world borrow.
                self.debug_lines.clear();
                self.debug_lines
                    .extend_from_slice(self.world.resource::<DebugLines>().lines());
                let camera = *self.world.resource::<Camera>();
                let ssao = *self.world.resource::<SsaoSettings>();
                let hdr = *self.world.resource::<HdrSettings>();

                let Active {
                    renderer, editor, ..
                } = active;
                let mut overlay = |before, image| editor.draw(before, image);
                renderer.render_with_overlay(
                    &self.render_items,
                    &self.lighting,
                    &camera,
                    &ssao,
                    &hdr,
                    &self.debug_lines,
                    &mut overlay,
                );

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
        self.world.resource_mut::<InputState>().on_device_event(&event);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(active) = self.active.as_ref() {
            active.window.request_redraw();
        }
    }
}
