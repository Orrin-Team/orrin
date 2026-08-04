//! In-window editor UI (egui overlay). New tools are added as `panels` modules.

mod dock;
mod icons;
mod panels;
mod prefs;
mod state;
mod theme;

use std::sync::Arc;

use egui_winit_vulkano::{Gui, GuiConfig};
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::swapchain::Surface;
use vulkano::sync::GpuFuture;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;

use orrin_ecs::World;
use orrin_registry::Registry;

use self::dock::Dock;
use self::prefs::{Prefs, PrefsFile};
use self::state::EditorState;
use self::theme::ThemeSet;

pub struct Editor {
    gui: Gui,
    state: EditorState,
    themes: ThemeSet,
    dock: Dock,
    prefs: Prefs,
    prefs_file: PrefsFile,
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
        format: Format,
        project: Option<&orrin_project::Project>,
    ) -> Self {
        let gui = Gui::new(
            event_loop,
            surface,
            queue,
            format,
            GuiConfig {
                // Load (don't clear) so the UI draws over the rendered scene.
                is_overlay: true,
                allow_srgb_render_target: true,
                ..Default::default()
            },
        );

        let mut themes = project.map_or_else(ThemeSet::default, |project| {
            ThemeSet::load(&project.themes_dir())
        });
        let prefs_file = PrefsFile::new(project.map(|project| project.editor_dir()));
        let mut prefs = prefs_file.load();
        if let Some(name) = &prefs.theme {
            themes.select(name);
        }
        icons::install(&gui.context());
        theme::apply(&gui.context(), themes.active());

        Self {
            gui,
            state: EditorState::new(project),
            themes,
            dock: Dock::from_saved(prefs.layout.take()),
            prefs,
            prefs_file,
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
        let consumed = self.gui.update(event);
        match event {
            WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. } => {
                self.dock.wants_pointer(&self.gui.context())
            }
            _ => consumed,
        }
    }

    pub fn run(&mut self, world: &mut World, registry: &Registry) {
        // Destructure for disjoint borrows: `gui` drives egui while the closure
        // edits `state`/`world`.
        let Editor {
            gui,
            state,
            themes,
            dock,
            prefs,
            prefs_file,
            ..
        } = self;
        gui.immediate_ui(|gui| {
            let ctx = gui.context();
            panels::draw(&ctx, world, state, registry, themes, dock);
        });
        state.apply(world, registry);

        // Restyling mid-frame would leave the panels already drawn on the old
        // palette, so a pick takes effect on the next one.
        if let Some(name) = state.take_theme_request()
            && themes.select(&name)
        {
            theme::apply(&gui.context(), themes.active());
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

    /// Whether the user asked for a script reload since the last call. Drained
    /// by the app at the start of the script phase — see
    /// `EditorState::script_reload_request`.
    #[cfg(feature = "scripting")]
    pub fn take_script_reload_request(&mut self) -> bool {
        self.state.take_script_reload_request()
    }

    pub fn draw(
        &mut self,
        before: Box<dyn GpuFuture>,
        image: Arc<ImageView>,
    ) -> Box<dyn GpuFuture> {
        self.gui.draw_on_image(before, image)
    }
}
