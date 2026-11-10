//! In-window editor UI (egui overlay). New tools are added as `panels` modules.
//!
//! egui lays the editor out and tessellates it; the renderer draws it (see
//! `gfx/vulkan/overlay.rs`).

mod autosave;
mod cadence;
pub(crate) mod catalog;
mod dock;
mod export;
mod gizmo;
mod icons;
mod kind;
mod panels;
mod pick;
mod prefs;
mod preview;
mod previews;
mod rect_tool;
mod selection;
mod state;
mod theme;
mod widgets;

use std::sync::Arc;

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
use crate::gfx::vulkan::OverlayFrame;

pub struct Editor {
    ctx: egui::Context,
    input: egui_winit::State,
    window: Arc<Window>,
    state: EditorState,
    themes: ThemeSet,
    dock: Dock,
    prefs: Prefs,
    prefs_file: PrefsFile,
    previews: Previews,
    /// Kept between frames for its allocations.
    frame: OverlayFrame,
    last_layout_check: std::time::Instant,
    /// Whether this frame rebuilds the UI or redraws the last one. See
    /// `cadence`.
    cadence: cadence::Cadence,
    /// Kept between frames for its allocations.
    outline: Vec<(glam::Vec3, glam::Vec3)>,
}

/// The last chance to keep a layout: a drag in the final two seconds of a
/// session would otherwise be thrown away on exit.
impl Drop for Editor {
    fn drop(&mut self) {
        self.save_layout();
        self.state.autosave.wait();
    }
}

/// Run `orrin export` on the open project, from File → Export game.
fn start_export(state: &mut EditorState, platform: orrin_project::export::Platform) {
    let (Some(root), Some(name)) = (state.project_root.clone(), state.project_name.clone()) else {
        return;
    };
    let beside = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf));
    let started = export::find_cli(
        std::env::var_os(export::CLI_ENV),
        beside.as_deref(),
        std::env::var_os("PATH"),
    )
    .and_then(|cli| state.export.start(&cli, &root, &name, platform));
    match started {
        Ok(()) => tracing::info!("export: {name} for {}", export::label(platform)),
        Err(error) => tracing::error!("export: {error}"),
    }
}

impl Editor {
    pub fn new(
        event_loop: &ActiveEventLoop,
        window: Arc<Window>,
        max_texture_side: usize,
        project: Option<&orrin_project::Project>,
    ) -> Self {
        let ctx = egui::Context::default();
        let input = egui_winit::State::new(
            ctx.clone(),
            ctx.viewport_id(),
            event_loop,
            Some(window.scale_factor() as f32),
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
        if let Some(preview) = prefs.preview {
            state.preview = preview;
        }
        if let Some(view) = prefs.assets {
            state.assets_view = view;
        }
        if let Some(tints) = prefs.icons {
            state.icon_tints = tints;
        }
        icons::install(&ctx);
        theme::install_fonts(&ctx);
        theme::apply(&ctx, themes.active());

        Self {
            ctx,
            input,
            window,
            state,
            themes,
            dock: Dock::from_saved(prefs.layout.take()),
            prefs,
            prefs_file,
            previews: Previews::default(),
            frame: OverlayFrame::default(),
            last_layout_check: std::time::Instant::now(),
            cadence: cadence::Cadence::default(),
            outline: Vec::new(),
        }
    }

    /// Returns `true` if the editor wants the event, so the caller can withhold
    /// it from game/camera input.
    ///
    /// egui's own verdict stands for everything but the pointer buttons and the
    /// wheel; those it would claim across the whole window, for the reason
    /// [`Dock::wants_pointer`] describes.
    pub fn on_window_event(&mut self, event: &WindowEvent) -> bool {
        // Every frame brings a redraw request; only what egui can react to
        // makes the next one rebuild the UI.
        if !matches!(event, WindowEvent::RedrawRequested) {
            self.cadence.touch();
        }
        let consumed = self.input.on_window_event(&self.window, event).consumed;
        match event {
            WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. } => {
                self.dock.wants_pointer(&self.ctx)
            }
            _ => consumed,
        }
    }

    /// Whether this frame rebuilds the UI. When it does not, neither
    /// [`run`](Self::run) nor [`overlay`](Self::overlay) is called and the
    /// renderer is handed [`cached_overlay`](Self::cached_overlay) instead.
    pub fn ui_due(&self, now: std::time::Instant) -> bool {
        self.cadence.due(now)
    }

    /// The scene camera moved, and with it everything the editor projects
    /// through it: the gizmo and the selection outline.
    pub fn camera_moved(&mut self) {
        self.cadence.touch();
    }

    /// The last frame [`overlay`](Self::overlay) produced, to be drawn again.
    /// Its texture deltas were drained when it was first drawn, so nothing is
    /// uploaded twice.
    pub fn cached_overlay(&mut self) -> &mut OverlayFrame {
        &mut self.frame
    }

    pub fn run(&mut self, world: &mut World, registry: &Registry) {
        let Editor {
            ctx,
            input,
            window,
            state,
            themes,
            dock,
            prefs,
            prefs_file,
            previews,
            ..
        } = self;
        ctx.begin_pass(input.take_egui_input(window));
        panels::draw(ctx, world, state, registry, themes, dock, previews);
        state.apply(world, registry);
        state.autosave(world, registry, std::time::Instant::now());
        if state.take_reveal_request() {
            dock.activate(dock::Tab::Assets);
        }
        if state.take_import_request() {
            dock.show_import();
        }
        if let Some(platform) = state.take_export_request() {
            start_export(state, platform);
        }
        for (level, line) in state.export.service() {
            if level >= crate::scene::LogLevel::Error {
                tracing::error!("{line}");
            } else {
                tracing::info!("{line}");
            }
        }

        // Kept once the pointer lets go, rather than on every frame a step is
        // dragged through.
        let steps = state.snap_steps;
        if prefs.snap.unwrap_or_default() != steps && !ctx.input(|input| input.pointer.any_down()) {
            prefs.snap = Some(steps);
            prefs_file.save(prefs);
        }
        let view = self::state::AssetView {
            kind: None,
            ..state.assets_view
        };
        if prefs.assets.unwrap_or_default() != view {
            prefs.assets = Some(view);
            prefs_file.save(prefs);
        }
        // Kept once a typed size has stopped being dragged.
        if prefs.preview.unwrap_or_default() != state.preview
            && !ctx.input(|input| input.pointer.any_down())
        {
            prefs.preview = Some(state.preview);
            prefs_file.save(prefs);
        }

        // Kept once a colour has stopped being dragged in its picker.
        if prefs.icons.unwrap_or_default() != state.icon_tints
            && !ctx.input(|input| input.pointer.any_down())
        {
            prefs.icons = Some(state.icon_tints);
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

    /// Outline colliders into this frame's debug lines: the selection's
    /// always, as Unity does, and every one while the viewport toggle is on.
    /// Every frame rather than with the UI, whose rebuilds are spaced out —
    /// see `cadence` — because a debug line lasts one frame.
    pub fn outline_colliders(&mut self, world: &World) {
        const SOLID: [f32; 4] = [0.45, 0.95, 0.55, 1.0];
        const TRIGGER: [f32; 4] = [0.95, 0.75, 0.3, 1.0];
        let all = self.state.show_colliders;
        let selection = &self.state.selection;
        if !all && selection.is_empty() {
            return;
        }
        let meshes = world.get_resource::<crate::scene::MeshBounds>();
        let Some(mut lines) = world.get_resource_mut::<crate::scene::DebugLines>() else {
            return;
        };
        let now = world
            .get_resource::<crate::scene::Time>()
            .map_or(0.0, |time| time.elapsed_time());
        let outline = &mut self.outline;
        world
            .query::<(&crate::scene::WorldTransform, &crate::scene::Collider)>()
            .for_each(|entity, (transform, collider)| {
                if !all && !selection.contains(entity) {
                    return;
                }
                let geometry = collider
                    .shape
                    .uses_mesh()
                    .then(|| world.get::<crate::scene::MeshHandle>(entity))
                    .flatten()
                    .and_then(|handle| meshes.as_ref()?.geometry(*handle).cloned());
                outline.clear();
                crate::physics::outline::outline(
                    collider.shape,
                    transform.0,
                    geometry.as_deref(),
                    outline,
                );
                let color = if collider.is_trigger { TRIGGER } else { SOLID };
                for &(from, to) in outline.iter() {
                    lines.push(from, to, color, now, 0.0);
                }
            });
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
            .handle_platform_output(&self.window, output.platform_output);
        let repaint = output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map_or(std::time::Duration::MAX, |viewport| viewport.repaint_delay);
        self.cadence.ran(std::time::Instant::now(), repaint);

        let ppp = output.pixels_per_point;
        let frame = &mut self.frame;
        std::mem::swap(&mut frame.gizmo, &mut self.state.gizmo_strokes);
        self.state.gizmo_strokes.clear();
        std::mem::swap(&mut frame.outline, &mut self.state.outline);
        self.state.outline.clear();
        frame.wireframe = self.state.wireframe;
        frame.material_previews = std::mem::take(&mut self.state.material_previews);
        frame.pixels_per_point = ppp;
        frame.textures.append(output.textures_delta);
        frame.primitives = self.ctx.tessellate(output.shapes, ppp);
        &mut self.frame
    }
}
