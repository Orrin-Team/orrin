use std::path::PathBuf;
use std::time::Instant;

use glam::Vec3;

use orrin_ecs::{Entity, World};
use orrin_registry::{AssetId, EntityId, Registry};

use super::autosave::Autosave;
use super::gizmo::{Drag, Handle};
use super::selection::Selection;
use crate::asset_db::material::MaterialFile;
use crate::scene::edit::{EditError, History, SceneEditor};
use crate::scene::{
    self, Assets, LoadedScenes, LogBuffer, LogLevel, MaterialHandle, Name, Play, PlayState,
    Prefabs, Time, Transform, entities,
};

/// Where the editor's Save/Load buttons read and write. Relative to the
/// process's working directory — a real file picker and a project-relative
/// `scenes/` directory belong with the asset database.
pub const DEFAULT_SCENE_PATH: &str = "scene.orrin";

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SceneRequest {
    Save,
    /// Replaces the current contents: every live entity is despawned first.
    Load,
    /// Load the scene's autosave in its place, as unsaved work.
    Recover,
}

/// What the transform gizmo edits. Select has no handles; the rest do (see
/// `editor::gizmo`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoMode {
    Select,
    Move,
    Rotate,
    Scale,
}

/// Whether gizmo axes follow the entity or the scene.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GizmoSpace {
    Local,
    World,
}

/// What a move drag quantises to when snapping, in metres.
pub const MOVE_SNAP: f32 = 0.25;

/// What a rotate drag quantises to when snapping, in degrees.
pub const ROTATE_SNAP: f32 = 15.0;

/// What a scale drag quantises its factor to when snapping.
pub const SCALE_SNAP: f32 = 0.1;

/// How the asset browser shows a folder. The user's to set from the
/// browser's bar, and remembered with their other preferences.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AssetView {
    pub layout: AssetLayout,
    pub sort: AssetSort,
    pub descending: bool,
    /// Only assets of this kind; `None` lists every kind. A filter for the
    /// session, so not remembered.
    #[serde(skip)]
    pub kind: Option<crate::asset_db::AssetKind>,
}

/// Rows with a small picture, or tiles with a large one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AssetLayout {
    #[default]
    List,
    Grid,
}

/// What a folder's assets are ordered by. Folders always come first, by name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AssetSort {
    #[default]
    Name,
    /// By extension, then name: every texture together, every mesh together.
    Type,
    /// The source file's size.
    Size,
    /// When the source file was last written.
    Modified,
}

/// What each tool snaps to. The user's to set from the snap button's menu,
/// and remembered with their other preferences.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SnapSteps {
    /// Metres.
    pub translate: f32,
    /// Degrees.
    pub rotate: f32,
    /// A plain ratio.
    pub scale: f32,
}

impl Default for SnapSteps {
    fn default() -> Self {
        Self {
            translate: MOVE_SNAP,
            rotate: ROTATE_SNAP,
            scale: SCALE_SNAP,
        }
    }
}

/// Which log levels the console shows: one bit per [`LogLevel`], all set to
/// begin with.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct LevelFilter(u8);

impl Default for LevelFilter {
    fn default() -> Self {
        Self(u8::MAX)
    }
}

impl LevelFilter {
    pub fn shows(self, level: LogLevel) -> bool {
        self.0 & (1 << level as u8) != 0
    }

    pub fn set(&mut self, level: LogLevel, shown: bool) {
        if shown {
            self.0 |= 1 << level as u8;
        } else {
            self.0 &= !(1 << level as u8);
        }
    }
}

/// Where a tool should go. Requested by a tool's own controls, which are drawn
/// inside the dock and so cannot rearrange it while it is drawing them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DockRequest {
    /// Bring it forward, reopening it if it was closed.
    Show(super::dock::Tab),
    /// Into a window of its own.
    Float(super::dock::Tab),
    /// Back where the default layout keeps it.
    Dock(super::dock::Tab),
}

/// What becomes of unsaved work when another scene is opened over it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenChoice {
    SaveFirst,
    Discard,
    Cancel,
}

/// What can be done to a prefab instance as a whole, from the tree's menu or
/// the inspector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefabAction {
    /// Write the instance back into its prefab.
    Apply,
    /// Throw away everything it overrides.
    Revert,
    /// Make it plain entities.
    Unpack,
    /// Point at its prefab in the asset browser.
    Reveal,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SpawnKind {
    Cube,
    Sphere,
    Plane,
    PointLight,
    DirectionalLight,
    Camera,
    Ui(crate::ui::create::UiKind),
}

impl SpawnKind {
    /// What the new entity is called.
    pub fn name(self) -> &'static str {
        match self {
            SpawnKind::Cube => "Cube",
            SpawnKind::Sphere => "Sphere",
            SpawnKind::Plane => "Plane",
            SpawnKind::PointLight => "Point Light",
            SpawnKind::DirectionalLight => "Directional Light",
            SpawnKind::Camera => "Camera",
            SpawnKind::Ui(kind) => kind.name(),
        }
    }
}

/// Spawn/despawn are *requested* by the panels during UI building and applied
/// by `apply` afterwards, so we never mutate the entity set while a panel is
/// iterating it.
pub struct EditorState {
    pub selection: Selection,
    pub scene_path: String,
    /// The project's persistent scene, loaded beside the scene while it plays.
    pub persistent_scene: Option<PathBuf>,
    /// What the top bar calls this session. Not the window title and not a
    /// path — just the name, so a run says which project it is inside. `None`
    /// outside a project, where the bar shows the scene file alone rather
    /// than a placeholder.
    pub project_name: Option<String>,
    /// The project's directory, which File → Export game hands the CLI.
    pub project_root: Option<PathBuf>,
    /// The export File → Export game started, and how the last one ended.
    pub export: super::export::Export,
    export_request: Option<orrin_project::export::Platform>,
    /// The hierarchy's search box. Lives here rather than in the panel so it
    /// survives the frame, and so a filtered tree stays filtered.
    pub hierarchy_query: String,
    /// The console's level toggles. Here for the reason the search box is.
    pub console_levels: LevelFilter,
    pub console_query: String,
    /// The inspector's property filter. Kept across selections, the way a
    /// search in a property panel usually is: looking for one field on several
    /// entities is the case it exists for.
    pub inspector_query: String,
    /// The asset browser's filter, kept for the reason the others are.
    pub assets_query: String,
    /// The Scene Manager's filter over the project's scenes.
    pub scenes_query: String,
    pub assets_view: AssetView,
    /// The asset the browser marks, and the Import tab shows: the last one
    /// clicked there, or the prefab "Show in Assets" pointed at.
    pub revealed_asset: Option<AssetId>,
    pub gizmo_mode: GizmoMode,
    pub gizmo_space: GizmoSpace,
    pub snap: bool,
    pub snap_steps: SnapSteps,
    /// Whether dragging one scale axis moves all three by the same ratio.
    pub scale_linked: bool,
    /// The frame-time readout floating in the viewport's corner.
    pub show_stats: bool,
    /// The handle being dragged, and the primary selection it was grabbed
    /// under. Dropped when the button is released or the primary changes.
    pub gizmo_drag: Option<(Entity, Drag)>,
    /// The handle under the pointer, or being dragged: drawn lit, and what
    /// keeps a click on a handle from also selecting what is behind it.
    pub gizmo_hot: Option<Handle>,
    /// What the pointer is over in the scene, picked as a click would pick
    /// it: outlined, so the click's result is known before it is made.
    pub hovered: Option<Entity>,
    /// Where a drag on the scene began, while it is drawing a selection
    /// rectangle.
    pub marquee: Option<egui::Pos2>,
    /// This frame's outlined meshes, for the renderer. Swapped into the
    /// overlay frame rather than copied, as the strokes are.
    pub outline: Vec<crate::gfx::vulkan::OutlineItem>,
    /// This frame's handles, for the renderer. Swapped into the overlay frame
    /// rather than copied.
    pub gizmo_strokes: Vec<crate::gfx::vulkan::GizmoVertex>,
    /// UI mode: clicks in the scene pick UI elements, and the selected one
    /// gets the rect tool instead of the transform gizmo.
    pub ui_mode: bool,
    /// The resolution the viewport shows the game's UI at.
    pub preview: super::preview::Preview,
    /// A drag of the rect tool, while the button is held.
    pub rect_drag: Option<super::rect_tool::RectDrag>,
    /// The part of the rect tool under the pointer, or being dragged.
    pub rect_hot: Option<super::rect_tool::Grip>,
    /// Every scene edit the panels make goes through this, by way of the
    /// requests below; see `scene::edit`. While the game plays it is a
    /// history of the fork, thrown away with it, and the scene's waits in
    /// [`Fork`].
    pub history: History,
    /// Component values the panels edited on detached copies this frame,
    /// applied by `apply` as part of the open gesture.
    edits: Vec<(Entity, orrin_registry::ComponentId, orrin_registry::Value)>,
    /// Material assets the inspector edited this frame, the same way.
    material_edits: Vec<(AssetId, MaterialFile)>,
    /// Whether the pointer is down or a field has focus — while it is, the
    /// open gesture stays open. Set by the panels once a frame's widgets
    /// have run.
    holding: bool,
    undo_request: bool,
    redo_request: bool,
    /// Not an edit: the camera is the editor's, not the scene's, so framing
    /// writes it directly rather than through the pipeline.
    frame_request: bool,
    camera_moved: bool,
    remove_request: Option<(Entity, orrin_registry::ComponentId)>,
    /// Overrides to give back to the prefab: entity, component, path.
    revert_request: Vec<(
        Entity,
        orrin_registry::ComponentId,
        orrin_registry::FieldPath,
    )>,
    spawn_request: Option<SpawnKind>,
    /// Give the scene a Render Settings entity, holding what it is drawn with.
    render_settings_request: bool,
    /// Write a new scene file under this name and open it. Serviced by
    /// `apply` with the other scene requests, for the reason they are.
    new_scene_request: Option<String>,
    /// A change to where a tool sits: the dock's, drained before the next
    /// frame lays it out.
    dock_request: Option<DockRequest>,
    /// A scene the Scene Manager was asked to open over unsaved work, waiting
    /// for the user to say what happens to that work.
    pending_open: Option<String>,
    /// Save the open scene, then open this one — the save-first answer to
    /// `pending_open`.
    open_after_save: Option<String>,
    /// An asset or folder, relative to the assets directory, waiting for the
    /// user to confirm it should be deleted.
    pending_delete: Option<String>,
    /// One the user confirmed. Serviced by `apply`, after the panels, since a
    /// deleted prefab rescans the index the asset browser draws from.
    delete_request: Option<String>,
    despawn_request: Vec<Entity>,
    /// A registered component to add to an entity, at its default. Deferred
    /// like the rest: the inspector is mid-way through drawing that entity's
    /// components when the menu is clicked.
    add_request: Option<(Entity, orrin_registry::ComponentId)>,
    /// `(child, new parent)`, with `None` meaning "detach to a root". Requested
    /// rather than applied inline for the same reason as the two above: the
    /// hierarchy panel is iterating a snapshot of the tree while it builds, and
    /// a reparent reshapes exactly that.
    reparent_request: Vec<(Entity, Option<Entity>)>,
    /// A prefab to place under an entity, or at the top level for `None`.
    /// Requested for the reason a reparent is: it reshapes the tree the
    /// hierarchy is iterating.
    instantiate_request: Option<(AssetId, Option<Entity>, Option<Vec3>)>,
    /// A whole-instance action, on the instance this entity is part of.
    prefab_request: Option<(PrefabAction, Entity)>,
    /// Not serviced by `apply`: bringing the asset browser forward is the
    /// dock's, which the editor drains after the UI pass.
    reveal_request: bool,
    /// Bring the Import tab forward: the dock's, like `reveal_request`.
    import_request: bool,
    /// Serviced by `apply`, because a rescan replaces the index the asset
    /// browser is drawing from.
    rescan_request: bool,
    /// Save the selection as a new prefab. Deferred because it can give the
    /// selection a new root, which reshapes the tree.
    save_prefab_request: bool,
    /// Not serviced by `apply` either: restyling has to happen between frames,
    /// so the editor drains it after the UI pass rather than during it.
    theme_request: Option<String>,
    /// Serviced by `apply` for the same reason spawn/despawn are: loading
    /// despawns every entity, and a `ScriptComponent`'s `Drop` tears down a
    /// managed object, which may only happen outside a dispatch window.
    scene_request: Option<SceneRequest>,
    /// Unlike the two above, this one is *not* serviced by `apply`: a script
    /// reload destroys and re-creates managed objects, which may only happen at
    /// the start of the frame's script phase, outside any dispatch window. The
    /// app drains it there.
    #[cfg(feature = "scripting")]
    script_reload_request: bool,
    /// Serviced by `apply`: Stop despawns the fork, for the reason a load is
    /// deferred.
    transport_request: Option<Transport>,
    /// What Stop goes back to. `Some` exactly while the game is live.
    fork: Option<Fork>,
    pub autosave: Autosave,
}

/// A press of one of the transport's buttons.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Transport {
    PlayStop,
    Pause,
    Step,
}

/// The scene as it was when Play was pressed, and the edit-mode state that
/// belongs to it rather than to the fork.
///
/// The scene is held as its document — what a save would write — because
/// that is the one form of it the engine already knows how to put back
/// exactly, scripts and prefab instances included: Stop is a load of it.
struct Fork {
    document: orrin_registry::SceneDocument,
    camera: scene::Camera,
    history: History,
    /// What was selected when Play was pressed, by identity: the scene comes
    /// back in new slots.
    selection: Vec<EntityId>,
}

impl Default for EditorState {
    fn default() -> Self {
        Self {
            selection: Selection::default(),
            scene_path: DEFAULT_SCENE_PATH.to_owned(),
            persistent_scene: None,
            project_name: None,
            project_root: None,
            export: super::export::Export::default(),
            export_request: None,
            hierarchy_query: String::new(),
            console_levels: LevelFilter::default(),
            console_query: String::new(),
            inspector_query: String::new(),
            assets_query: String::new(),
            scenes_query: String::new(),
            assets_view: AssetView::default(),
            revealed_asset: None,
            gizmo_mode: GizmoMode::Select,
            gizmo_space: GizmoSpace::World,
            snap: false,
            snap_steps: SnapSteps::default(),
            scale_linked: false,
            show_stats: false,
            gizmo_drag: None,
            gizmo_hot: None,
            hovered: None,
            marquee: None,
            gizmo_strokes: Vec::new(),
            ui_mode: false,
            preview: Default::default(),
            rect_drag: None,
            rect_hot: None,
            outline: Vec::new(),
            history: History::default(),
            edits: Vec::new(),
            material_edits: Vec::new(),
            holding: false,
            undo_request: false,
            redo_request: false,
            frame_request: false,
            camera_moved: false,
            remove_request: None,
            revert_request: Vec::new(),
            spawn_request: None,
            render_settings_request: false,
            new_scene_request: None,
            dock_request: None,
            pending_open: None,
            open_after_save: None,
            pending_delete: None,
            delete_request: None,
            despawn_request: Vec::new(),
            add_request: None,
            reparent_request: Vec::new(),
            instantiate_request: None,
            prefab_request: None,
            reveal_request: false,
            import_request: false,
            rescan_request: false,
            save_prefab_request: false,
            theme_request: None,
            scene_request: None,
            #[cfg(feature = "scripting")]
            script_reload_request: false,
            transport_request: None,
            fork: None,
            autosave: Autosave::new(PathBuf::from(".orrin/autosave"), Instant::now()),
        }
    }
}

impl EditorState {
    pub fn new(project: Option<&orrin_project::Project>) -> Self {
        Self {
            project_name: project.map(|project| project.name().to_owned()),
            project_root: project.map(|project| project.root().to_path_buf()),
            persistent_scene: project.and_then(|project| project.persistent_scene()),
            autosave: Autosave::new(
                project
                    .map_or_else(|| PathBuf::from(".orrin"), |p| p.editor_dir())
                    .join("autosave"),
                Instant::now(),
            ),
            ..Self::default()
        }
    }

    /// Export the project for `platform`, from File → Export game.
    pub fn request_export(&mut self, platform: orrin_project::export::Platform) {
        self.export_request = Some(platform);
    }

    pub fn take_export_request(&mut self) -> Option<orrin_project::export::Platform> {
        self.export_request.take()
    }

    /// Play from edit mode; Stop from play.
    pub fn request_play(&mut self) {
        self.transport_request = Some(Transport::PlayStop);
    }

    /// Pause a playing game, or resume a paused one.
    pub fn request_pause(&mut self) {
        self.transport_request = Some(Transport::Pause);
    }

    /// One frame of a live game, leaving it paused.
    pub fn request_step(&mut self) {
        self.transport_request = Some(Transport::Step);
    }

    /// Whether the game is live: the world is the fork.
    pub fn playing(&self) -> bool {
        self.fork.is_some()
    }

    /// Whether the scene has changes its file does not, the fork's aside.
    pub fn scene_dirty(&self) -> bool {
        self.fork
            .as_ref()
            .map_or(&self.history, |fork| &fork.history)
            .is_dirty()
    }

    /// Write a copy of unsaved work if one is owed, and report the last one
    /// that finished. See `editor::autosave`.
    pub fn autosave(&mut self, world: &mut World, registry: &Registry, now: Instant) {
        match self.autosave.finished() {
            Some(Ok(path)) => log(
                world,
                LogLevel::Info,
                format!("autosaved to {}", path.display()),
            ),
            Some(Err(error)) => log(world, LogLevel::Warning, error),
            None => {}
        }
        let version = self.history.version();
        if self.playing()
            || self.holding
            || !self.history.is_dirty()
            || !self.autosave.due(now, version)
        {
            return;
        }
        let document = scene::to_document(world, registry);
        let path = self.autosave.path(&self.scene_path);
        self.autosave.write(document, path, version, now);
    }

    pub fn request_spawn(&mut self, kind: SpawnKind) {
        self.spawn_request = Some(kind);
    }

    pub fn request_add_render_settings(&mut self) {
        self.render_settings_request = true;
    }

    /// Write a new scene called `name` beside the project's others, and open it.
    pub fn request_new_scene(&mut self, name: impl Into<String>) {
        self.new_scene_request = Some(name.into());
    }

    pub fn request_dock(&mut self, request: DockRequest) {
        self.dock_request = Some(request);
    }

    pub fn take_dock_request(&mut self) -> Option<DockRequest> {
        self.dock_request.take()
    }

    /// Open `path`, or, over unsaved work, ask first: see
    /// [`pending_open`](Self::pending_open).
    pub fn request_open_scene_checked(&mut self, path: impl Into<String>) {
        if self.scene_dirty() {
            self.pending_open = Some(path.into());
        } else {
            self.request_open_scene(path);
        }
    }

    /// Ask to delete the asset or folder at `rel`, relative to the assets
    /// directory. Nothing happens until [`resolve_delete`](Self::resolve_delete).
    pub fn request_delete_asset(&mut self, rel: impl Into<String>) {
        self.pending_delete = Some(rel.into());
    }

    /// What is waiting to be confirmed for deletion.
    pub fn pending_delete(&self) -> Option<&str> {
        self.pending_delete.as_deref()
    }

    pub fn resolve_delete(&mut self, confirmed: bool) {
        let pending = self.pending_delete.take();
        if confirmed {
            self.delete_request = pending;
        }
    }

    /// The scene waiting on an answer about the open one's unsaved work.
    pub fn pending_open(&self) -> Option<&str> {
        self.pending_open.as_deref()
    }

    pub fn resolve_open(&mut self, choice: OpenChoice) {
        let Some(path) = self.pending_open.take() else {
            return;
        };
        match choice {
            OpenChoice::SaveFirst => self.open_after_save = Some(path),
            OpenChoice::Discard => self.request_open_scene(path),
            OpenChoice::Cancel => {}
        }
    }

    /// Make `path` the scene file and load it.
    pub fn request_open_scene(&mut self, path: impl Into<String>) {
        self.scene_path = plain(path.into());
        self.scene_request = Some(SceneRequest::Load);
    }

    pub fn request_scene(&mut self, request: SceneRequest) {
        self.scene_request = Some(request);
    }

    pub fn request_despawn(&mut self, entity: Entity) {
        self.despawn_request.push(entity);
    }

    pub fn request_despawn_selection(&mut self) {
        self.despawn_request
            .extend_from_slice(self.selection.entities());
    }

    pub fn request_reparent(&mut self, child: Entity, parent: Option<Entity>) {
        self.reparent_request.push((child, parent));
    }

    pub fn request_save_prefab(&mut self) {
        self.save_prefab_request = true;
    }

    pub fn request_instantiate(&mut self, prefab: AssetId, parent: Option<Entity>) {
        self.instantiate_request = Some((prefab, parent, None));
    }

    /// Place a prefab at the top level, standing at `at`.
    pub fn request_instantiate_at(&mut self, prefab: AssetId, at: Vec3) {
        self.instantiate_request = Some((prefab, None, Some(at)));
    }

    pub fn request_prefab_action(&mut self, action: PrefabAction, entity: Entity) {
        self.prefab_request = Some((action, entity));
    }

    /// Point the asset browser at `id`, and bring it forward.
    pub fn reveal_asset(&mut self, id: AssetId) {
        self.revealed_asset = Some(id);
        self.reveal_request = true;
    }

    pub fn take_reveal_request(&mut self) -> bool {
        std::mem::take(&mut self.reveal_request)
    }

    pub fn request_import_page(&mut self) {
        self.import_request = true;
    }

    pub fn take_import_request(&mut self) -> bool {
        std::mem::take(&mut self.import_request)
    }

    pub fn request_prefab_rescan(&mut self) {
        self.rescan_request = true;
    }

    /// Give `component` the value a panel edited a detached copy of.
    pub fn edit(
        &mut self,
        entity: Entity,
        component: orrin_registry::ComponentId,
        value: orrin_registry::Value,
    ) {
        self.edits.push((entity, component, value));
    }

    /// Give material asset `id` the parameters and slots of `file`, saved to
    /// its file when the gesture ends.
    pub fn edit_material(&mut self, id: AssetId, file: MaterialFile) {
        self.material_edits.push((id, file));
    }

    /// The increment a drag of the current tool snaps to: when the toggle is
    /// on, or while `held` (Ctrl) is down.
    pub fn snap_step(&self, held: bool) -> Option<f32> {
        if !(self.snap || held) {
            return None;
        }
        match self.gizmo_mode {
            GizmoMode::Move => Some(self.snap_steps.translate),
            GizmoMode::Rotate => Some(self.snap_steps.rotate),
            GizmoMode::Scale => Some(self.snap_steps.scale),
            GizmoMode::Select => None,
        }
    }

    /// Keep the open gesture open through this frame, or let it close.
    pub fn hold(&mut self, holding: bool) {
        self.holding = holding;
    }

    /// Point the camera at the selection, near enough to fill the view.
    pub fn request_frame(&mut self) {
        self.frame_request = true;
    }

    /// Whether the editor moved the camera since the last call. The app
    /// drains it to bring the fly camera's own heading in line, which would
    /// otherwise turn the view straight back.
    pub fn take_camera_moved(&mut self) -> bool {
        std::mem::take(&mut self.camera_moved)
    }

    pub fn request_undo(&mut self) {
        self.undo_request = true;
    }

    pub fn request_redo(&mut self) {
        self.redo_request = true;
    }

    pub fn request_remove_component(&mut self, entity: Entity, id: orrin_registry::ComponentId) {
        self.remove_request = Some((entity, id));
    }

    pub fn request_revert(
        &mut self,
        entity: Entity,
        component: orrin_registry::ComponentId,
        path: orrin_registry::FieldPath,
    ) {
        self.revert_request.push((entity, component, path));
    }

    pub fn request_add_component(&mut self, entity: Entity, id: orrin_registry::ComponentId) {
        self.add_request = Some((entity, id));
    }

    pub fn request_theme(&mut self, name: impl Into<String>) {
        self.theme_request = Some(name.into());
    }

    pub fn take_theme_request(&mut self) -> Option<String> {
        self.theme_request.take()
    }

    #[cfg(feature = "scripting")]
    pub fn request_script_reload(&mut self) {
        self.script_reload_request = true;
    }

    #[cfg(feature = "scripting")]
    pub fn take_script_reload_request(&mut self) -> bool {
        std::mem::take(&mut self.script_reload_request)
    }

    pub fn apply(&mut self, world: &mut World, registry: &Registry) {
        self.apply_edits(world, registry);
        if let Some((action, entity)) = self.prefab_request.take()
            && world.is_alive(entity)
        {
            self.prefab_action(world, registry, action, entity);
        }
        if std::mem::take(&mut self.save_prefab_request) {
            self.save_prefab(world, registry);
        }
        if std::mem::take(&mut self.frame_request)
            && let Some(camera) = framed(world, self.selection.entities())
        {
            *world.resource_mut::<scene::Camera>() = camera;
            self.camera_moved = true;
        }
        // Not an edit: the prefabs changed on disk, and their instances follow.
        if std::mem::take(&mut self.rescan_request) {
            for issue in scene::prefab::rescan(world, registry) {
                log(world, LogLevel::Warning, issue);
            }
        }
        if let Some(request) = self.transport_request.take() {
            self.transport(world, registry, request);
        }
        if let Some(name) = self.new_scene_request.take() {
            let (level, message) = if self.playing() {
                (
                    LogLevel::Warning,
                    "new scene: the game is playing; Stop it first".to_owned(),
                )
            } else {
                self.new_scene(world, registry, &name)
            };
            log(world, level, message);
        }
        if let Some(rel) = self.delete_request.take() {
            self.delete_asset(world, &rel);
        }
        if let Some(request) = self.scene_request.take() {
            let (level, message) = self.scene(world, registry, request);
            log(world, level, message);
        }
        if let Some(path) = self.open_after_save.take() {
            let (level, message) = self.scene(world, registry, SceneRequest::Save);
            let saved = level != LogLevel::Error;
            log(world, level, message);
            if saved {
                self.scene_path = path;
                let (level, message) = self.scene(world, registry, SceneRequest::Load);
                log(world, level, message);
            }
        }
    }

    fn scene(
        &mut self,
        world: &mut World,
        registry: &Registry,
        request: SceneRequest,
    ) -> (LogLevel, String) {
        match request {
            _ if self.playing() => (
                LogLevel::Warning,
                format!(
                    "the game is playing: Stop it first, or the game's state would \
                         replace {}",
                    self.scene_path
                ),
            ),
            SceneRequest::Save => {
                SceneEditor::new(world, registry, &mut self.history).commit();
                let saved = save_scene(world, registry, &self.scene_path);
                if saved.0 != LogLevel::Error {
                    self.history.mark_saved();
                    self.autosave
                        .saved(&self.scene_path, self.history.version());
                }
                saved
            }
            SceneRequest::Recover => {
                self.selection.clear();
                let path = self.autosave.path(&self.scene_path);
                let loaded = load_scene(world, registry, &path.to_string_lossy());
                if loaded.0 != LogLevel::Error {
                    self.history.clear_unsaved();
                    self.opened(world);
                }
                loaded
            }
            SceneRequest::Load => {
                self.selection.clear();
                let loaded = load_scene(world, registry, &self.scene_path);
                if loaded.0 != LogLevel::Error {
                    self.history.clear();
                    self.opened(world);
                }
                loaded
            }
        }
    }

    /// What the panels asked of the scene this frame, through the edit
    /// pipeline.
    fn apply_edits(&mut self, world: &mut World, registry: &Registry) {
        let mut editor = SceneEditor::new(world, registry, &mut self.history);
        let mut failed: Vec<(LogLevel, String)> = Vec::new();
        let fail = |error: EditError| (LogLevel::Error, error.to_string());

        for (entity, component, value) in self.edits.drain(..) {
            failed.extend(editor.set(entity, &component, value).err().map(fail));
        }
        for (id, file) in self.material_edits.drain(..) {
            failed.extend(editor.set_material(id, file).err().map(fail));
        }
        if !self.holding {
            failed.extend(editor.try_commit().err().map(fail));
        }
        if std::mem::take(&mut self.undo_request) {
            failed.extend(editor.undo().err().map(fail));
        }
        if std::mem::take(&mut self.redo_request) {
            failed.extend(editor.redo().err().map(fail));
        }

        if let Some(kind) = self.spawn_request.take() {
            let label = format!("Create {}", kind.name());
            let selected = self.selection.primary();
            let spawned = editor.replace(label, &[], |world, _| {
                let spawned = spawn(world, kind, selected);
                let root = spawned.map(|(_, root)| root);
                (
                    spawned.map(|(entity, _)| entity),
                    root.into_iter().collect(),
                )
            });
            if let Some(entity) = spawned {
                self.selection.set(entity);
            }
        }
        if std::mem::take(&mut self.render_settings_request)
            && scene::render_settings::source(editor.world()).is_none()
        {
            let settings = scene::render_settings::capture(editor.world());
            let entity = editor.replace("Add render settings", &[], |world, _| {
                let entity = world.spawn();
                world.insert(entity, Name::new("Render Settings"));
                world.insert(entity, settings);
                (entity, vec![entity])
            });
            self.selection.set(entity);
        }
        if !self.despawn_request.is_empty() {
            let doomed = std::mem::take(&mut self.despawn_request);
            failed.extend(editor.despawn(&doomed).err().map(fail));
        }
        if let Some((entity, id)) = self.add_request.take()
            && editor.world().is_alive(entity)
            && let Some(vtable) = registry.get(&id)
            && !vtable.has(editor.world(), entity)
        {
            let added = editor.add_component(entity, &id, vtable.default());
            failed.extend(added.err().map(fail));
        }
        if let Some((entity, id)) = self.remove_request.take() {
            failed.extend(editor.remove_component(entity, &id).err().map(fail));
        }
        for (entity, component, path) in std::mem::take(&mut self.revert_request) {
            let reverted =
                editor.changed("Revert override", entity, &component, |world, registry| {
                    scene::prefab::revert(world, registry, entity, &component, &path)
                });
            if let Err(error) = reverted {
                let message = format!(
                    "reverting {component} on entity {}: {error}",
                    entity.index()
                );
                failed.push((LogLevel::Error, message));
            }
        }

        let moves = std::mem::take(&mut self.reparent_request);
        // One dragged along with an ancestor that is also moving keeps its
        // place under that ancestor, as a file does in a moved folder.
        let moves: Vec<(Entity, Option<Entity>)> = moves
            .iter()
            .copied()
            .filter(|&(child, _)| {
                !moves
                    .iter()
                    .any(|&(other, _)| other != child && is_under(editor.world(), child, other))
            })
            .collect();
        let mut reparent_all = |editor: &mut SceneEditor| {
            for &(child, parent) in &moves {
                // `keep_world`: a drag in the tree changes who owns an object,
                // not where it is. Without it, dropping something onto a
                // distant parent would fling it across the scene.
                if let Err(error) = editor.reparent(child, parent, true) {
                    failed.push((LogLevel::Warning, error.to_string()));
                }
            }
        };
        match moves.len() {
            0 => {}
            1 => reparent_all(&mut editor),
            n => editor.batch(format!("Reparent {n} entities"), reparent_all),
        }

        if let Some((prefab, parent, at)) = self.instantiate_request.take() {
            let name = editor
                .world()
                .get_resource::<Prefabs>()
                .and_then(|prefabs| prefabs.get(prefab).map(|p| p.name().to_owned()))
                .unwrap_or_else(|| prefab.to_string());
            let placed =
                editor.replace(format!("Place {name}"), &[], |world, registry| match place(
                    world, registry, prefab, parent, at,
                ) {
                    Ok(root) => (Ok(root), vec![root]),
                    Err(error) => (Err(error), Vec::new()),
                });
            match placed {
                Ok(root) => self.selection.set(root),
                Err(error) => failed.push((LogLevel::Error, error)),
            }
        }

        for (level, message) in failed {
            log(editor.world(), level, message);
        }
        self.selection.retain_alive(editor.world());
    }
}

/// The camera turned on `selection` from where it already looks, and backed
/// off until the sphere around the selection's bounds fits the view — or
/// `None` when nothing selected has a place.
///
/// A mesh is measured by its bounds and anything else by its origin, given a
/// half-metre of room so a lone light is not framed from a hair's breadth.
fn framed(world: &World, selection: &[Entity]) -> Option<scene::Camera> {
    let camera = *world.get_resource::<scene::Camera>()?;
    let meshes = world.get_resource::<scene::MeshBounds>();
    let mut bounds = crate::geom::Aabb::EMPTY;
    for &entity in selection {
        let Some(model) = world.get::<scene::WorldTransform>(entity).map(|w| w.0) else {
            continue;
        };
        let measured = meshes
            .as_ref()
            .zip(world.get::<scene::MeshHandle>(entity))
            .and_then(|(meshes, mesh)| meshes.get(*mesh))
            .filter(|bounds| bounds.is_valid());
        let piece = measured.map_or_else(
            || {
                let at = glam::Vec3A::from(model.w_axis.truncate());
                crate::geom::Aabb { min: at, max: at }
            },
            |bounds| bounds.transformed(&model),
        );
        bounds = bounds.union(&piece);
    }
    if !bounds.is_valid() {
        return None;
    }
    let centre = Vec3::from(bounds.center());
    let radius = bounds.half_extents().length().max(0.5);
    let heading = (camera.target - camera.position).normalize();
    // The sphere's tangent at the edge of the vertical field of view, and a
    // little more, so the selection is not drawn touching the frame.
    let distance = radius / (camera.fov_y * 0.5).sin() * 1.15;
    Some(scene::Camera {
        position: centre - heading * distance,
        target: centre,
        ..camera
    })
}

/// Spawn an instance of `prefab`, under `parent` when there is one and
/// standing at `at` when that is given.
fn place(
    world: &mut World,
    registry: &Registry,
    prefab: AssetId,
    parent: Option<Entity>,
    at: Option<Vec3>,
) -> Result<Entity, String> {
    let (root, issues) = scene::prefab::instantiate(world, registry, prefab)?;
    for issue in issues {
        log(world, LogLevel::Warning, issue.to_string());
    }
    // Not `keep_world`: a prefab sits where it was authored to, relative to
    // whatever it was dropped on.
    if let Some(parent) = parent
        && let Err(error) = scene::reparent(world, root, Some(parent), false)
    {
        log(world, LogLevel::Warning, error.to_string());
    }
    if let Some(at) = at {
        let mut local = world
            .get::<scene::LocalTransform>(root)
            .map(|local| *local)
            .unwrap_or_default();
        local.translation = at;
        world.insert(root, local);
    }
    Ok(root)
}

impl EditorState {
    /// Revert and unpack are edits to this scene and can be undone. Apply is
    /// not: it writes the prefab's file, and every other instance follows.
    fn prefab_action(
        &mut self,
        world: &mut World,
        registry: &Registry,
        action: PrefabAction,
        entity: Entity,
    ) {
        let Some((root, prefab)) = scene::prefab::instance_of(world, entity) else {
            log(
                world,
                LogLevel::Warning,
                "that entity is not part of a prefab instance".to_owned(),
            );
            return;
        };
        let path = world
            .get_resource::<Prefabs>()
            .and_then(|prefabs| prefabs.get(prefab).map(|p| p.path.clone()))
            .unwrap_or_else(|| prefab.to_string());
        let outcome = match action {
            PrefabAction::Apply => {
                scene::prefab::apply(world, registry, entity).map(|issues| (issues, "applied to"))
            }
            PrefabAction::Revert => SceneEditor::new(world, registry, &mut self.history)
                .replace("Revert to prefab", &[root], |world, registry| {
                    let reverted = scene::prefab::revert_all(world, registry, entity);
                    (reverted, vec![root])
                })
                .map(|issues| (issues, "reverted to")),
            PrefabAction::Unpack => {
                SceneEditor::new(world, registry, &mut self.history).replace(
                    "Unpack prefab",
                    &[root],
                    |world, _| (scene::prefab::unpack(world, entity), vec![root]),
                );
                Ok((Vec::new(), "unpacked from"))
            }
            PrefabAction::Reveal => {
                self.revealed_asset = Some(prefab);
                self.reveal_request = true;
                return;
            }
        };
        match outcome {
            Ok((issues, done)) => {
                for issue in issues {
                    log(world, LogLevel::Warning, issue);
                }
                log(world, LogLevel::Info, format!("{done} {path}"));
            }
            Err(error) => log(world, LogLevel::Error, error),
        }
    }

    fn transport(&mut self, world: &mut World, registry: &Registry, request: Transport) {
        if world.get_resource::<Play>().is_none() {
            world.insert_resource(Play::default());
        }
        match request {
            Transport::PlayStop if self.playing() => self.stop(world, registry),
            Transport::PlayStop => {
                // The gesture is the scene's, so it is closed into the scene's
                // history rather than carried into the fork's.
                SceneEditor::new(world, registry, &mut self.history).commit();
                // After the document, which is what gives an entity its id.
                let document = scene::to_document(world, registry);
                let selection = self.selected_ids(world);
                self.fork = Some(Fork {
                    document,
                    camera: *world.resource::<scene::Camera>(),
                    history: std::mem::take(&mut self.history),
                    selection,
                });
                // The scene is the game's while it runs: see `panels::viewport`.
                self.release_scene();
                world.resource_mut::<Play>().set(PlayState::Playing);
                self.load_persistent(world, registry);
            }
            Transport::Pause => {
                let mut play = world.resource_mut::<Play>();
                match play.state() {
                    PlayState::Playing => play.set(PlayState::Paused),
                    PlayState::Paused => play.set(PlayState::Playing),
                    PlayState::Editing => {}
                }
            }
            Transport::Step => world.resource_mut::<Play>().step(),
        }
    }

    fn selected_ids(&self, world: &World) -> Vec<EntityId> {
        self.selection
            .entities()
            .iter()
            .filter_map(|&entity| world.get::<EntityId>(entity).map(|id| *id))
            .collect()
    }

    /// Let go of everything the editor holds in the scene: the selection and
    /// any pointer gesture over it.
    fn release_scene(&mut self) {
        self.selection.clear();
        self.gizmo_drag = None;
        self.gizmo_hot = None;
        self.hovered = None;
        self.marquee = None;
    }

    /// Throw the fork away and put the scene back, with the camera and the
    /// selection Play cleared. The selection is carried by identity, since
    /// every entity comes back in a new slot.
    fn stop(&mut self, world: &mut World, registry: &Registry) {
        let Some(fork) = self.fork.take() else {
            return;
        };
        let selected = fork.selection;
        self.release_scene();

        for issue in replace_scene(world, registry, &fork.document) {
            log(world, LogLevel::Warning, issue.to_string());
        }
        *world.resource_mut::<scene::Camera>() = fork.camera;
        self.camera_moved = true;
        self.history = fork.history;
        self.opened(world);
        // Contacts between entities that no longer exist would arrive as exits
        // on the first frame of the next play.
        if world
            .get_resource::<crate::collision::CollisionState>()
            .is_some()
        {
            world.insert_resource(crate::collision::CollisionState::default());
        }
        world.resource_mut::<Play>().set(PlayState::Editing);

        let by_id: std::collections::HashMap<EntityId, Entity> = world
            .entities()
            .filter_map(|entity| world.get::<EntityId>(entity).map(|id| (*id, entity)))
            .collect();
        for id in selected {
            if let Some(&entity) = by_id.get(&id) {
                self.selection.toggle(entity);
            }
        }
    }

    /// Write a scene called `name` under the assets directory's `scenes/`,
    /// holding render settings that name a sky and a sun, and open it in place
    /// of this one.
    ///
    /// Written before it is opened, so the file exists from the moment it is
    /// listed and a crash before the first save loses nothing but the name.
    fn new_scene(
        &mut self,
        world: &mut World,
        registry: &Registry,
        name: &str,
    ) -> (LogLevel, String) {
        let path = match new_scene_path(world, name) {
            Ok(path) => path,
            Err(problem) => return (LogLevel::Error, format!("new scene: {problem}")),
        };
        let dir = path.parent().unwrap_or(&path).to_owned();

        let mut render = scene::RenderSettings::default();
        render.environment.hdri = entities::BUILT_IN_SKY.to_owned();
        let mut fresh = World::new();
        let settings = fresh.spawn();
        fresh.insert(settings, Name::new("Render Settings"));
        fresh.insert(settings, render);
        entities::spawn_built_in_sun(&mut fresh);
        let text = scene::save(&mut fresh, registry);
        if let Err(error) = std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(&path, text))
        {
            return (
                LogLevel::Error,
                format!("new scene: could not write {}: {error}", path.display()),
            );
        }

        self.scene_path = plain(path.to_string_lossy().into_owned());
        self.selection.clear();
        let loaded = load_scene(world, registry, &self.scene_path);
        if loaded.0 != LogLevel::Error {
            self.history.clear();
            self.opened(world);
        }
        loaded
    }

    /// Delete what the user confirmed, unless it is or holds the open scene:
    /// that is what the editor is showing and would save straight back.
    fn delete_asset(&mut self, world: &mut World, rel: &str) {
        let Some(dir) = world
            .get_resource::<crate::asset_db::AssetDatabase>()
            .map(|db| db.dir().to_owned())
        else {
            let message = format!("could not delete {rel}: there is no asset database");
            log(world, LogLevel::Error, message);
            return;
        };
        let target = std::fs::canonicalize(dir.join(rel)).ok();
        let open = std::fs::canonicalize(&self.scene_path).ok();
        if let (Some(target), Some(open)) = (target, open)
            && open.starts_with(&target)
        {
            let message =
                format!("could not delete {rel}: it holds the open scene; open another first");
            log(world, LogLevel::Error, message);
            return;
        }
        let deleted = world
            .resource_mut::<crate::asset_db::AssetDatabase>()
            .delete(rel);
        match deleted {
            Ok(count) => {
                // A folder may hold prefabs, and the prefab index is by path.
                let kind = crate::asset_db::AssetKind::of_path(rel);
                if kind.is_none_or(|kind| kind == crate::asset_db::AssetKind::Prefab) {
                    self.rescan_request = true;
                }
                let gone = self.revealed_asset.is_some_and(|id| {
                    world
                        .resource::<crate::asset_db::AssetDatabase>()
                        .iter()
                        .all(|entry| entry.id != id)
                });
                if gone {
                    self.revealed_asset = None;
                }
                let what = match count {
                    1 => String::new(),
                    n => format!(" and the {n} assets in it"),
                };
                log(world, LogLevel::Info, format!("deleted {rel}{what}"));
            }
            Err(error) => log(world, LogLevel::Error, error),
        }
    }

    /// The scene file is what is loaded now, and it alone.
    fn opened(&self, world: &mut World) {
        let path = std::path::Path::new(&self.scene_path);
        let asset = scene::loaded::asset_at(world, path);
        let name = scene::loaded::name_of(&self.scene_path);
        if world.get_resource::<LoadedScenes>().is_none() {
            world.insert_resource(LoadedScenes::default());
        }
        world.resource_mut::<LoadedScenes>().reset(name, asset);
    }

    /// Load the project's persistent scene beside the one just forked, as a
    /// game's start would — unless that is the scene being played.
    fn load_persistent(&self, world: &mut World, registry: &Registry) {
        let Some(path) = &self.persistent_scene else {
            return;
        };
        if same_file(path, std::path::Path::new(&self.scene_path)) {
            return;
        }
        let shown = path.display();
        let document = match std::fs::read_to_string(path) {
            Ok(text) => match orrin_registry::parse(&text) {
                Ok(document) => document,
                Err(error) => {
                    log(
                        world,
                        LogLevel::Error,
                        format!("persistent scene {shown}: {error}"),
                    );
                    return;
                }
            },
            Err(error) => {
                log(
                    world,
                    LogLevel::Error,
                    format!(
                        "persistent scene {shown}: could not read it: {error}; check \
                         `[scenes] persistent` in orrin.toml"
                    ),
                );
                return;
            }
        };
        if world.get_resource::<LoadedScenes>().is_none() {
            world.insert_resource(LoadedScenes::default());
        }
        let asset = scene::loaded::asset_at(world, path);
        let name = scene::loaded::name_of(&path.to_string_lossy());
        let handle = world.resource_mut::<LoadedScenes>().reserve();
        let issues =
            scene::loaded::load_additive(world, registry, handle, &document, name, asset, true);
        for issue in issues {
            log(world, LogLevel::Warning, issue.to_string());
        }
    }

    /// Save the selection as a prefab in the assets directory, named after the
    /// entity when there is one and `Group` when there are several, and select
    /// the instance it became.
    ///
    /// Undoing it gives the selection back as it was; the file stays written.
    fn save_prefab(&mut self, world: &mut World, registry: &Registry) {
        self.selection.retain_alive(world);
        let selection = self.selection.entities().to_vec();
        let name = match selection[..] {
            [] => return,
            [one] => world
                .get::<Name>(one)
                .map_or_else(|| "Prefab".to_owned(), |name| name.0.clone()),
            _ => "Group".to_owned(),
        };
        let Some(dir) = world
            .get_resource::<Prefabs>()
            .and_then(|prefabs| prefabs.dir().map(std::path::Path::to_owned))
        else {
            log(
                world,
                LogLevel::Error,
                "there is no assets directory to save a prefab into".to_owned(),
            );
            return;
        };
        let rel = scene::prefab::unused_path(&dir, &name);
        let label = format!("Save {name} as prefab");
        let created = SceneEditor::new(world, registry, &mut self.history).replace(
            label,
            &selection,
            |world, registry| match scene::prefab::create(world, registry, &selection, &rel) {
                Ok((root, _)) => (Ok(root), vec![root]),
                Err(error) => (Err(error), selection.clone()),
            },
        );
        match created {
            Ok(root) => {
                self.selection.set(root);
                log(world, LogLevel::Info, format!("saved {rel}"));
            }
            Err(error) => log(world, LogLevel::Error, error),
        }
    }
}

/// Where a new scene called `name` would be written, or why it cannot be.
///
/// One rule for the Scene Manager's field, which shows the problem as the name
/// is typed, and for the request, which checks again in case the disk changed
/// in between.
pub fn new_scene_path(world: &World, name: &str) -> Result<PathBuf, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("give it a name".to_owned());
    }
    if name.starts_with('.')
        || name
            .chars()
            .any(|c| matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|'))
    {
        return Err(format!(
            "`{name}` is not a file name; leave out slashes and punctuation"
        ));
    }
    let path = assets_dir(world)
        .join("scenes")
        .join(format!("{name}.{}", scene::loaded::EXTENSION));
    if path.exists() {
        return Err(format!(
            "`{name}` already exists; open it from the list instead"
        ));
    }
    Ok(path)
}

/// Where the project's assets are: the database's directory, or `assets/`
/// beside the working directory without one, as prefabs resolve.
fn assets_dir(world: &World) -> PathBuf {
    if let Some(db) = world.get_resource::<crate::asset_db::AssetDatabase>() {
        return db.dir().to_owned();
    }
    world
        .get_resource::<Prefabs>()
        .and_then(|prefabs| prefabs.dir().map(std::path::Path::to_owned))
        .unwrap_or_else(|| PathBuf::from("assets"))
}

/// `path` without the extended-length prefix Windows puts on a canonicalised
/// path. It is still the same file, but `\\?\` is noise anywhere a person
/// reads it, and the scene path is shown in the File menu and the Scene
/// Manager.
pub fn plain(path: String) -> String {
    match path.strip_prefix(r"\\?\") {
        Some(rest) if rest.starts_with("UNC\\") => format!(r"\\{}", &rest[4..]),
        Some(rest) => rest.to_owned(),
        None => path,
    }
}

/// Whether two paths name one file, however each is spelled.
fn same_file(a: &std::path::Path, b: &std::path::Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// Whether `ancestor` is above `entity` in the tree.
fn is_under(world: &World, entity: Entity, ancestor: Entity) -> bool {
    let mut cursor = entity;
    // Bounded: a cycle the hierarchy has not broken yet would otherwise spin.
    for _ in 0..4096 {
        match world
            .get::<scene::Parent>(cursor)
            .map(|parent| parent.get())
        {
            Some(parent) if parent == ancestor => return true,
            Some(parent) => cursor = parent,
            None => return false,
        }
    }
    false
}

fn save_scene(world: &mut World, registry: &Registry, path: &str) -> (LogLevel, String) {
    let text = scene::save(world, registry);
    let entities = world.entities().count();
    match std::fs::write(path, &text) {
        Ok(()) => (
            LogLevel::Info,
            format!("saved {entities} entities to {path}"),
        ),
        Err(e) => (LogLevel::Error, format!("could not write {path}: {e}")),
    }
}

fn load_scene(world: &mut World, registry: &Registry, path: &str) -> (LogLevel, String) {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) => return (LogLevel::Error, format!("could not read {path}: {e}")),
    };

    // Parsed before anything is despawned, so a malformed file costs nothing.
    let document = match orrin_registry::parse(&text) {
        Ok(document) => document,
        Err(e) => return (LogLevel::Error, format!("{path}: {e}")),
    };

    let issues = replace_scene(world, registry, &document);
    let count = document.entities.len();
    for issue in &issues {
        log(world, LogLevel::Warning, issue.to_string());
    }
    match issues.len() {
        0 => (
            LogLevel::Info,
            format!("loaded {count} entities from {path}"),
        ),
        n => (
            LogLevel::Warning,
            format!("loaded {count} entities from {path}; {n} component(s) kept but not applied"),
        ),
    }
}

/// Every live entity despawned, and `document`'s spawned in their place.
fn replace_scene(
    world: &mut World,
    registry: &Registry,
    document: &orrin_registry::SceneDocument,
) -> Vec<scene::LoadIssue> {
    let existing: Vec<Entity> = world.entities().collect();
    for entity in existing {
        world.despawn(entity);
    }
    scene::instantiate(document, world, registry)
}

fn log(world: &World, level: LogLevel, message: String) {
    let frame = world
        .get_resource::<Time>()
        .map_or(0, |time| time.frame_count());
    if let Some(mut buffer) = world.get_resource_mut::<LogBuffer>() {
        buffer.push(level, message, frame);
    }
}

/// Spawn `kind`, returning what to select and the root of what was spawned —
/// which for UI can be a canvas made to hold it.
fn spawn(world: &mut World, kind: SpawnKind, selected: Option<Entity>) -> Option<(Entity, Entity)> {
    if let SpawnKind::Ui(kind) = kind {
        let created = crate::ui::create::create(world, kind, selected);
        return Some((created.element, created.root));
    }
    spawn_object(world, kind).map(|entity| (entity, entity))
}

fn spawn_object(world: &mut World, kind: SpawnKind) -> Option<Entity> {
    match kind {
        SpawnKind::Ui(_) => None,
        SpawnKind::Cube | SpawnKind::Sphere | SpawnKind::Plane => {
            let mesh_name = match kind {
                SpawnKind::Cube => "cube",
                SpawnKind::Sphere => "sphere",
                _ => "plane",
            };
            // Copy the handles out before mutating the world (drops the borrows).
            let mesh = world.resource::<Assets>().mesh(mesh_name)?;
            let material = default_material(world)?;
            Some(entities::spawn_mesh(
                world,
                kind.name(),
                Transform::default(),
                mesh,
                material,
            ))
        }
        // Physical fixtures, matching `Light::default`: a bright shop light and an
        // overcast sun. Both are large numbers and neither is arbitrary — a light
        // spawned at the old unitless 8.0 would now be an 8 lumen indicator lamp,
        // which is a feature that looks broken rather than a light that looks dim.
        SpawnKind::PointLight => Some(entities::spawn_point_light(
            world,
            kind.name(),
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::ONE,
            4000.0,
            10.0,
        )),
        SpawnKind::DirectionalLight => Some(entities::spawn_directional_light(
            world,
            kind.name(),
            Vec3::new(-0.4, -1.0, -0.6),
            Vec3::ONE,
            20_000.0,
        )),
        // Where the editor is looking, so it opens on what the author framed.
        SpawnKind::Camera => {
            let view = *world.get_resource::<scene::Camera>()?;
            Some(entities::spawn_camera(
                world,
                kind.name(),
                view.position,
                view.rotation(),
            ))
        }
    }
}

/// "clay" if present, else whatever the registry has first.
fn default_material(world: &World) -> Option<MaterialHandle> {
    let assets = world.resource::<Assets>();
    assets
        .material("clay")
        .or_else(|| assets.materials().next().map(|(_, h)| h))
}
