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
use ferron_ecs::World;

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
    /// The Ferron project this run was launched inside, if any. `None` means
    /// the engine is running standalone on its built-in demo scene.
    #[cfg(feature = "scripting")]
    project: Option<ferron_project::Project>,
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
            eprintln!("ferron: cannot read the current directory: {err}");
            std::process::exit(1);
        });

        // A manifest that exists but doesn't load is fatal: silently falling
        // back to the demo scene would hide the user's real project.
        let project = match ferron_project::Project::locate(&cwd) {
            Ok(project) => project,
            Err(err) => {
                eprintln!("ferron: {err}");
                std::process::exit(1);
            }
        };
        if let Some(project) = &project {
            println!(
                "ferron: project `{}` at {}",
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
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.active.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("renderer-prototype"))
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
            let scripts_dir = self
                .project
                .as_ref()
                .map(|project| project.scripts_dir())
                .unwrap_or_else(|| std::path::PathBuf::from("scripting/Ferron"));

            let assembly_dir = match std::env::var("FERRON_SCRIPT_DIR") {
                Ok(dir) => Some(std::path::PathBuf::from(dir)),
                Err(_) => crate::scripting::Scripting::find_assembly_dir(&scripts_dir),
            };

            let scripting = match assembly_dir {
                Some(dir) => crate::scripting::Scripting::boot(&dir),
                None => {
                    eprintln!(
                        "scripting disabled: no built managed assembly under {} \
                         (run `dotnet build` there, or set FERRON_SCRIPT_DIR)",
                        scripts_dir.display()
                    );
                    None
                }
            };
            if let Some(scripting) = &scripting {
                let entry = std::env::var("FERRON_ENTRY")
                    .ok()
                    .or_else(|| {
                        self.project
                            .as_ref()
                            .map(|project| project.entry_type().to_string())
                    })
                    .unwrap_or_else(|| "Ferron.Demo.Game, Ferron".to_string());
                let entity = self
                    .world
                    .spawn_entity()
                    .with(crate::scene::Name::new("Script Entry"))
                    .id();
                scripting.attach(&mut self.world, entity, &entry);
            }
            self.scripting = scripting;
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

                #[cfg(feature = "scripting")]
                if let Some(scripting) = &self.scripting {
                    scripting.tick(&mut self.world, delta);
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
