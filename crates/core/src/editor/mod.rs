//! In-window editor UI (egui overlay). New tools are added as `panels` modules.
//!
//! egui lays the editor out and tessellates it; the renderer draws it, in the
//! phases `phases` cuts it into (see `gfx/vulkan/overlay.rs`).

mod dock;
mod gizmo;
mod icons;
mod kind;
mod panels;
mod phases;
mod pick;
mod prefs;
mod previews;
mod selection;
mod state;
mod theme;
mod widgets;

use std::sync::Arc;

use vulkano::device::Queue;
use vulkano::swapchain::Surface;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

use orrin_ecs::World;
use orrin_registry::Registry;

use self::dock::Dock;
use self::prefs::{Prefs, PrefsFile};
use self::previews::Previews;
use self::state::EditorState;
use self::theme::ThemeSet;
use crate::gfx::vulkan::{Glass, OverlayFrame, OverlayPhase};

pub struct Editor {
    ctx: egui::Context,
    input: egui_winit::State,
    surface: Arc<Surface>,
    state: EditorState,
    themes: ThemeSet,
    dock: Dock,
    prefs: Prefs,
    prefs_file: PrefsFile,
    previews: Previews,
    /// The last frame's translucent surfaces, from `panels::frost`.
    glass: Vec<phases::Glass>,
    /// Kept between frames for its allocations.
    frame: OverlayFrame,
    last_layout_check: std::time::Instant,
}

/// The last chance to keep a layout: a drag in the final two seconds of a
/// session would otherwise be thrown away on exit.
impl Drop for Editor {
    fn drop(&mut self) {
        self.save_layout();
    }
}

impl Editor {
    pub fn new(
        event_loop: &ActiveEventLoop,
        surface: Arc<Surface>,
        queue: Arc<Queue>,
        project: Option<&orrin_project::Project>,
    ) -> Self {
        let ctx = egui::Context::default();
        let max_texture_side = queue
            .device()
            .physical_device()
            .properties()
            .max_image_dimension2_d as usize;
        let input = egui_winit::State::new(
            ctx.clone(),
            ctx.viewport_id(),
            event_loop,
            Some(window(&surface).scale_factor() as f32),
            // egui's own default, as the system theme it starts from: the
            // editor's palette is applied over it either way.
            Some(winit::window::Theme::Dark),
            Some(max_texture_side),
        );

        let mut themes = project.map_or_else(ThemeSet::default, |project| {
            ThemeSet::load(&project.themes_dir())
        });
        let prefs_file = PrefsFile::new(project.map(|project| project.editor_dir()));
        let mut prefs = prefs_file.load();
        if let Some(name) = &prefs.theme {
            themes.select(name);
        }
        let mut state = EditorState::new(project);
        if let Some(steps) = prefs.snap {
            state.snap_steps = steps;
        }
        icons::install(&ctx);
        theme::install_fonts(&ctx);
        theme::apply(&ctx, themes.active());

        Self {
            ctx,
            input,
            surface,
            state,
            themes,
            dock: Dock::from_saved(prefs.layout.take()),
            prefs,
            prefs_file,
            previews: Previews::default(),
            glass: Vec::new(),
            frame: OverlayFrame::default(),
            last_layout_check: std::time::Instant::now(),
        }
    }

    /// Returns `true` if the editor wants the event, so the caller can withhold
    /// it from game/camera input.
    ///
    /// egui's own verdict stands for everything but the pointer buttons and the
    /// wheel; those it would claim across the whole window, for the reason
    /// [`Dock::wants_pointer`] describes.
    pub fn on_window_event(&mut self, event: &WindowEvent) -> bool {
        let consumed = self
            .input
            .on_window_event(window(&self.surface), event)
            .consumed;
        match event {
            WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. } => {
                self.dock.wants_pointer(&self.ctx)
            }
            _ => consumed,
        }
    }

    pub fn run(&mut self, world: &mut World, registry: &Registry) {
        let Editor {
            ctx,
            input,
            surface,
            state,
            themes,
            dock,
            prefs,
            prefs_file,
            previews,
            glass,
            ..
        } = self;
        ctx.begin_pass(input.take_egui_input(window(surface)));
        panels::draw(ctx, world, state, registry, themes, dock, previews);
        panels::frost(ctx, glass);
        phases::mark(ctx);
        state.apply(world, registry);
        if state.take_reveal_request() {
            dock.activate(dock::Tab::Assets);
        }

        // Kept once the pointer lets go, rather than on every frame a step is
        // dragged through.
        let steps = state.snap_steps;
        if prefs.snap.unwrap_or_default() != steps && !ctx.input(|input| input.pointer.any_down()) {
            prefs.snap = Some(steps);
            prefs_file.save(prefs);
        }

        // Restyling mid-frame would leave the panels already drawn on the old
        // palette, so a pick takes effect on the next one.
        if let Some(name) = state.take_theme_request()
            && themes.select(&name)
        {
            theme::apply(ctx, themes.active());
            prefs.theme = Some(name);
            prefs_file.save(prefs);
        }

        self.save_layout_if_settled();
    }

    /// Write the dock tree out when it has stopped moving.
    ///
    /// A layout changes by dragging, so there is no single moment to save at and
    /// no cheap way to ask `DockState` whether it differs. Comparing the
    /// serialised form catches every change including a drag, and throttling it
    /// keeps that comparison off the frame budget — at this interval it is a few
    /// microseconds a minute, and the most a crash can cost is the last drag.
    fn save_layout_if_settled(&mut self) {
        const INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

        if self.last_layout_check.elapsed() < INTERVAL {
            return;
        }
        self.last_layout_check = std::time::Instant::now();
        self.save_layout();
    }

    fn save_layout(&mut self) {
        let current = self.dock.state();
        if self
            .prefs
            .layout
            .as_ref()
            .is_some_and(|saved| ron::to_string(saved).ok() == ron::to_string(current).ok())
        {
            return;
        }
        self.prefs.layout = Some(current.clone());
        self.prefs_file.save(&self.prefs);
    }

    /// Whether the editor moved the camera since the last call — see
    /// `EditorState::take_camera_moved`.
    pub fn take_camera_moved(&mut self) -> bool {
        self.state.take_camera_moved()
    }

    /// Whether the user asked for a script reload since the last call. Drained
    /// by the app at the start of the script phase — see
    /// `EditorState::script_reload_request`.
    #[cfg(feature = "scripting")]
    pub fn take_script_reload_request(&mut self) -> bool {
        self.state.take_script_reload_request()
    }

    /// Finish the frame [`run`](Self::run) began and hand it to the renderer.
    ///
    /// Tessellated here rather than in `run`, so the editor's profile scope
    /// stays the layout alone and this lands in the overlay's.
    pub fn overlay(&mut self) -> &mut OverlayFrame {
        let output = self.ctx.end_pass();
        self.input
            .handle_platform_output(window(&self.surface), output.platform_output);

        let ppp = output.pixels_per_point;
        let frame = &mut self.frame;
        std::mem::swap(&mut frame.gizmo, &mut self.state.gizmo_strokes);
        self.state.gizmo_strokes.clear();
        std::mem::swap(&mut frame.outline, &mut self.state.outline);
        self.state.outline.clear();
        frame.pixels_per_point = ppp;
        frame.textures.append(output.textures_delta);
        frame
            .phases
            .resize_with(phases::COUNT, OverlayPhase::default);
        for (index, (phase, shapes)) in frame
            .phases
            .iter_mut()
            .zip(phases::split(output.shapes))
            .enumerate()
        {
            phase.primitives = self.ctx.tessellate(shapes, ppp);
            phase.glass.clear();
            phase
                .glass
                .extend(
                    self.glass
                        .iter()
                        .filter(|glass| glass.phase == index)
                        .map(|glass| Glass {
                            rect: glass.rect,
                            radius: glass.radius,
                        }),
                );
        }
        &mut self.frame
    }
}

fn window(surface: &Surface) -> &Window {
    surface
        .object()
        .and_then(|object| object.downcast_ref::<Window>())
        .expect("the editor's surface is a winit window")
}
