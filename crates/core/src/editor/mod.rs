//! In-window editor UI (egui overlay). New tools are added as `panels` modules.

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

use self::prefs::{Prefs, PrefsFile};
use self::state::EditorState;
use self::theme::ThemeSet;

pub struct Editor {
    gui: Gui,
    state: EditorState,
    themes: ThemeSet,
    prefs: Prefs,
    prefs_file: PrefsFile,
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
        let prefs = prefs_file.load();
        if let Some(name) = &prefs.theme {
            themes.select(name);
        }
        icons::install(&gui.context());
        theme::apply(&gui.context(), themes.active());

        Self {
            gui,
            state: EditorState::new(project),
            themes,
            prefs,
            prefs_file,
        }
    }

    /// Returns `true` if egui wants the event, so the caller can withhold it
    /// from game/camera input.
    pub fn on_window_event(&mut self, event: &WindowEvent) -> bool {
        self.gui.update(event)
    }

    pub fn run(&mut self, world: &mut World, registry: &Registry) {
        // Destructure for disjoint borrows: `gui` drives egui while the closure
        // edits `state`/`world`.
        let Editor {
            gui,
            state,
            themes,
            prefs,
            prefs_file,
        } = self;
        gui.immediate_ui(|gui| {
            let ctx = gui.context();
            panels::draw(&ctx, world, state, registry, themes);
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
