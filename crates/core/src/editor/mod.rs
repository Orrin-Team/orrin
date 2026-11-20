//! In-window editor UI (egui overlay). New tools are added as `panels` modules.
//!
//! egui lays the editor out and tessellates it; the renderer draws it (see
//! `gfx/vulkan/overlay.rs`).

mod autosave;
pub(crate) mod blender;
mod cadence;
pub(crate) mod catalog;
mod dock;
mod export;
mod gizmo;
mod glsl;
mod icons;
mod kind;
mod panels;
mod pick;
mod prefs;
mod preview;
mod previews;
mod rect_tool;
mod remote;
mod selection;
pub mod settings;
pub(crate) mod source;
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
    settings: settings::SettingsStore,
    previews: Previews,
    /// Kept between frames for its allocations.
    frame: OverlayFrame,
    last_layout_check: std::time::Instant,
    /// Whether this frame rebuilds the UI or redraws the last one. See
    /// `cadence`.
    cadence: cadence::Cadence,
    /// Kept between frames for its allocations.
    outline: Vec<(glam::Vec3, glam::Vec3)>,
    /// The remote, while it listens.
    remote: Option<remote::Remote>,
    /// The project's editor directory, where the remote's file goes, and the
    /// name it greets a client with. `None` with no project open.
    remote_home: Option<(std::path::PathBuf, String)>,
    /// Whether `app.rs` has offered this editor a remote at all, which it
    /// does only for the windowed editor and never for an exported game.
    remote_offered: bool,
    /// Whether the remote should be listening, as last decided.
    remote_wanted: bool,
}

/// The last chance to keep a layout: a drag in the final two seconds of a
/// session would otherwise be thrown away on exit.
impl Drop for Editor {
    fn drop(&mut self) {
        self.save_layout();
        self.settings.flush();
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
    .and_then(|cli| {
        let blender = state.settings.blender.path.as_deref();
        state.export.start(&cli, &root, &name, platform, blender)
    });
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
        scripts_dir: Option<std::path::PathBuf>,
        mut settings: settings::SettingsStore,
        backend: crate::gfx::BackendChoice,
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

        let prefs_file = PrefsFile::new(project.map(|project| project.editor_dir()));
        let mut prefs = prefs_file.load();
        settings.seed(prefs.theme.take(), prefs.icons.take());
        let mut themes = ThemeSet::load(
            settings
                .themes_dir()
                .into_iter()
                .chain(project.map(|project| project.themes_dir())),
        );
        let mut state = EditorState::new(project);
        if let Some(project) = project {
            state.agent_folders = orrin_remote::permissions::Folders::load(&project.editor_dir());
            let assets = project.assets_dir();
            let assets = assets.strip_prefix(project.root()).unwrap_or(&assets);
            state.agent_assets =
                orrin_remote::permissions::normalise(&assets.to_string_lossy()).unwrap_or_default();
        }
        state.script_files = source::ScriptFiles::new(scripts_dir);
        state.environment = settings::Environment::of_process();
        state.settings_file = settings.file();
        state.backend = backend;
        settings.begin(&ctx, &mut state, &mut themes);
        if let Some(steps) = prefs.snap {
            state.snap_steps = steps;
        }
        if let Some(preview) = prefs.preview {
            state.preview = preview;
        }
        if let Some(view) = prefs.assets {
            state.assets_view = view;
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
            settings,
            previews: Previews::default(),
            frame: OverlayFrame::default(),
            last_layout_check: std::time::Instant::now(),
            cadence: cadence::Cadence::default(),
            outline: Vec::new(),
            remote: None,
            remote_home: project.map(|project| (project.editor_dir(), project.name().to_owned())),
            remote_offered: false,
            remote_wanted: false,
        }
    }

    /// Let the remote listen while Settings ▸ Agents or `ORRIN_REMOTE` says
    /// it should, from now on. The windowed editor's alone, for the reason
    /// `SettingsStore::drive_blender` is.
    pub fn offer_remote(&mut self) {
        self.remote_offered = true;
        self.sync_remote();
        // A remote that started has written its own file over any stale one.
        // Otherwise one a killed editor left is cleared here, off the frame's
        // thread, since telling whether it is stale means connecting to it.
        if self.remote.is_none()
            && let Some((dir, _)) = &self.remote_home
        {
            let dir = dir.clone();
            let swept = std::thread::Builder::new()
                .name("orrin-remote-sweep".to_owned())
                .spawn(move || {
                    if let Some(removed) = orrin_remote::client::sweep(&dir) {
                        tracing::info!("{removed}");
                    }
                });
            if let Err(error) = swept {
                tracing::warn!("could not look for a stale remote file: {error}");
            }
        }
    }

    /// Write the folder rules Settings ▸ Agents changed, once the pointer
    /// lets go. They are read at every request, so the next one is governed.
    fn save_agent_folders(&mut self) {
        if !self.state.agent_folders_changed || self.ctx.input(|input| input.pointer.any_down()) {
            return;
        }
        self.state.agent_folders_changed = false;
        let (Some((dir, _)), Ok(folders)) = (&self.remote_home, &self.state.agent_folders) else {
            return;
        };
        if let Err(error) = folders.save(dir) {
            tracing::error!("{error}");
        }
    }

    /// Stop the remote now and take its file back, for a way out that will
    /// not reach the editor's `Drop`.
    pub fn stop_remote(&mut self) {
        if self.remote.take().is_some() {
            tracing::info!("the remote stopped listening");
        }
        self.state.remote_peers.clear();
    }

    /// Answer what an agent asked since the last frame. See `remote`. A
    /// command it raised is carried out by the UI pass, so that pass is made
    /// due rather than left to its cadence.
    pub fn serve_remote(&mut self, world: &mut World, registry: &Registry) {
        if let Some(remote) = &mut self.remote
            && remote.serve(world, registry, &mut self.state)
        {
            self.cadence.touch();
        }
    }

    /// This frame of an agent's playtest, fed before the actions resolve; with
    /// no remote, any sequence it left is dropped and let go of.
    pub fn feed_game_input(&mut self, world: &mut World, extent: [u32; 2]) {
        if self.remote.is_none() {
            self.state.game_input.cancel();
        }
        self.state.game_input.feed(world, extent);
    }

    /// What screenshot the remote wants the next frame to draw too, if any.
    /// A check of an `Option` with the remote off.
    pub fn take_screenshot_request(&mut self) -> Option<crate::gfx::vulkan::frame::ShotView> {
        self.remote.as_mut()?.wants_screenshot()
    }

    /// The screenshot the renderer was asked for, read back in `readback`.
    pub fn screenshot_taken(
        &mut self,
        taken: Result<(Vec<u8>, [u32; 2]), String>,
        readback: std::time::Duration,
    ) {
        if let Some(remote) = &mut self.remote {
            remote.screenshot_taken(taken, readback);
        }
    }

    /// A script reload landed, for the remote waiting on it.
    pub fn script_reloaded(&mut self, swapped: bool, message: String) {
        self.state.script_reloaded(swapped, message);
    }

    /// Start or stop the remote when what is wanted changed; otherwise
    /// refresh what the settings page shows of it, which is one atomic load.
    fn sync_remote(&mut self) {
        if !self.remote_offered {
            return;
        }
        let want = self.state.environment.remote();
        let want = want.unwrap_or(self.state.settings.agents.remote);
        if want == self.remote_wanted {
            if let Some(remote) = &self.remote {
                if let Some(Ok(status)) = &mut self.state.remote {
                    *status = remote.status();
                }
                let state = &mut self.state;
                remote.peers(&mut state.remote_peers_seen, &mut state.remote_peers);
                if let Some(id) = state.remote_disconnect.take() {
                    remote.disconnect(id);
                }
            }
            return;
        }
        self.remote_wanted = want;
        self.stop_remote();
        // A new remote counts its changes from zero; this makes its first
        // count one the list has not seen.
        self.state.remote_peers_seen = u64::MAX;
        self.state.remote = want.then(|| {
            let Some((dir, name)) = &self.remote_home else {
                return Err("the remote needs a project open: its port and token are \
                            written to the project's .orrin/remote"
                    .to_owned());
            };
            let started = remote::Remote::start(dir, name).inspect_err(|error| {
                tracing::warn!("{error}");
            })?;
            let status = started.status();
            tracing::info!(
                "the remote is listening on 127.0.0.1:{} for agents, found through {}",
                status.port,
                orrin_remote::discovery::path(dir).display()
            );
            self.remote = Some(started);
            Ok(status)
        });
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
            settings,
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
        if let Some((file, line)) = state.take_open_source_request() {
            source::open(
                &file,
                line,
                &state.settings.scripting.editor,
                &state.environment,
            );
        }
        for (level, line) in state.export.service() {
            if level >= crate::scene::LogLevel::Error {
                tracing::error!("{line}");
            } else {
                tracing::info!("{line}");
            }
        }
        match state.addon.service() {
            Some(Ok(done)) => tracing::info!("{done}"),
            Some(Err(failed)) => tracing::error!("{failed}"),
            None => {}
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
        settings.settle(ctx, world, state, themes);
        self.sync_remote();
        self.save_agent_folders();

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
                if (!all && !selection.contains(entity))
                    || crate::scene::editor_hidden::is_hidden(world, entity)
                {
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

    /// How the user likes the fly camera, for the controller `App` owns.
    pub fn fly_camera(&self) -> crate::camera_controller::FlyCamera {
        self.state.settings.viewport
    }

    /// Whether the editor moved the camera since the last call — see
    /// `EditorState::take_camera_moved`.
    pub fn take_camera_moved(&mut self) -> bool {
        self.state.take_camera_moved()
    }

    /// Whether the user asked to restart Orrin since the last call; see
    /// `EditorState::request_restart`.
    pub fn take_restart_request(&mut self) -> bool {
        self.state.take_restart_request()
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
