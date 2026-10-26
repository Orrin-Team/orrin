use glam::Vec3;

use orrin_ecs::{Entity, World};
use orrin_registry::{AssetId, Registry};

use super::gizmo::{Drag, Handle};
use super::selection::Selection;
use crate::scene::edit::{EditError, History, SceneEditor};
use crate::scene::{
    self, Assets, LogBuffer, LogLevel, MaterialHandle, Name, Prefabs, Time, Transform, entities,
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
        }
    }
}

/// Spawn/despawn are *requested* by the panels during UI building and applied
/// by `apply` afterwards, so we never mutate the entity set while a panel is
/// iterating it.
pub struct EditorState {
    pub selection: Selection,
    pub scene_path: String,
    /// What the top bar calls this session. Not the window title and not a
    /// path — just the name, so a run says which project it is inside. `None`
    /// outside a project, where the bar shows the scene file alone rather
    /// than a placeholder.
    pub project_name: Option<String>,
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
    /// The prefab the asset browser marks: the last one clicked there, or the
    /// one "Show in Assets" pointed at.
    pub revealed_asset: Option<AssetId>,
    pub gizmo_mode: GizmoMode,
    pub gizmo_space: GizmoSpace,
    pub snap: bool,
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
    /// This frame's handles, for the renderer. Swapped into the overlay frame
    /// rather than copied.
    pub gizmo_strokes: Vec<crate::gfx::vulkan::GizmoVertex>,
    /// Every scene edit the panels make goes through this, by way of the
    /// requests below; see `scene::edit`.
    pub history: History,
    /// Component values the panels edited on detached copies this frame,
    /// applied by `apply` as part of the open gesture.
    edits: Vec<(Entity, orrin_registry::ComponentId, orrin_registry::Value)>,
    /// Whether the pointer is down or a field has focus — while it is, the
    /// open gesture stays open. Set by the panels once a frame's widgets
    /// have run.
    holding: bool,
    undo_request: bool,
    redo_request: bool,
    remove_request: Option<(Entity, orrin_registry::ComponentId)>,
    /// Overrides to give back to the prefab: entity, component, path.
    revert_request: Vec<(
        Entity,
        orrin_registry::ComponentId,
        orrin_registry::FieldPath,
    )>,
    spawn_request: Option<SpawnKind>,
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
}

impl Default for EditorState {
    fn default() -> Self {
        Self {
            selection: Selection::default(),
            scene_path: DEFAULT_SCENE_PATH.to_owned(),
            project_name: None,
            hierarchy_query: String::new(),
            console_levels: LevelFilter::default(),
            console_query: String::new(),
            inspector_query: String::new(),
            assets_query: String::new(),
            revealed_asset: None,
            gizmo_mode: GizmoMode::Select,
            gizmo_space: GizmoSpace::World,
            snap: false,
            scale_linked: false,
            show_stats: false,
            gizmo_drag: None,
            gizmo_hot: None,
            gizmo_strokes: Vec::new(),
            history: History::default(),
            edits: Vec::new(),
            holding: false,
            undo_request: false,
            redo_request: false,
            remove_request: None,
            revert_request: Vec::new(),
            spawn_request: None,
            despawn_request: Vec::new(),
            add_request: None,
            reparent_request: Vec::new(),
            instantiate_request: None,
            prefab_request: None,
            reveal_request: false,
            rescan_request: false,
            save_prefab_request: false,
            theme_request: None,
            scene_request: None,
            #[cfg(feature = "scripting")]
            script_reload_request: false,
        }
    }
}

impl EditorState {
    pub fn new(project: Option<&orrin_project::Project>) -> Self {
        Self {
            project_name: project.map(|project| project.name().to_owned()),
            ..Self::default()
        }
    }

    pub fn request_spawn(&mut self, kind: SpawnKind) {
        self.spawn_request = Some(kind);
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

    pub fn take_reveal_request(&mut self) -> bool {
        std::mem::take(&mut self.reveal_request)
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

    /// The increment a drag of the current tool snaps to: when the toggle is
    /// on, or while `held` (Ctrl) is down.
    pub fn snap_step(&self, held: bool) -> Option<f32> {
        if !(self.snap || held) {
            return None;
        }
        match self.gizmo_mode {
            GizmoMode::Move => Some(MOVE_SNAP),
            GizmoMode::Rotate => Some(ROTATE_SNAP),
            GizmoMode::Scale => Some(SCALE_SNAP),
            GizmoMode::Select => None,
        }
    }

    /// Keep the open gesture open through this frame, or let it close.
    pub fn hold(&mut self, holding: bool) {
        self.holding = holding;
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
        // Not an edit: the prefabs changed on disk, and their instances follow.
        if std::mem::take(&mut self.rescan_request) {
            for issue in scene::prefab::rescan(world, registry) {
                log(world, LogLevel::Warning, issue);
            }
        }
        if let Some(request) = self.scene_request.take() {
            let (level, message) = match request {
                SceneRequest::Save => {
                    SceneEditor::new(world, registry, &mut self.history).commit();
                    let saved = save_scene(world, registry, &self.scene_path);
                    if saved.0 != LogLevel::Error {
                        self.history.mark_saved();
                    }
                    saved
                }
                SceneRequest::Load => {
                    self.selection.clear();
                    let loaded = load_scene(world, registry, &self.scene_path);
                    if loaded.0 != LogLevel::Error {
                        self.history.clear();
                    }
                    loaded
                }
            };
            log(world, level, message);
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
        if !self.holding {
            editor.commit();
        }
        if std::mem::take(&mut self.undo_request) {
            failed.extend(editor.undo().err().map(fail));
        }
        if std::mem::take(&mut self.redo_request) {
            failed.extend(editor.redo().err().map(fail));
        }

        if let Some(kind) = self.spawn_request.take() {
            let label = format!("Create {}", kind.name());
            let spawned = editor.replace(label, &[], |world, _| {
                let entity = spawn(world, kind);
                (entity, entity.into_iter().collect())
            });
            if let Some(entity) = spawned {
                self.selection.set(entity);
            }
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

    let existing: Vec<Entity> = world.entities().collect();
    for entity in existing {
        world.despawn(entity);
    }

    let issues = scene::instantiate(&document, world, registry);
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

fn log(world: &World, level: LogLevel, message: String) {
    let frame = world
        .get_resource::<Time>()
        .map_or(0, |time| time.frame_count());
    if let Some(mut buffer) = world.get_resource_mut::<LogBuffer>() {
        buffer.push(level, message, frame);
    }
}

fn spawn(world: &mut World, kind: SpawnKind) -> Option<Entity> {
    match kind {
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
    }
}

/// "clay" if present, else whatever the registry has first.
fn default_material(world: &World) -> Option<MaterialHandle> {
    let assets = world.resource::<Assets>();
    assets
        .material("clay")
        .or_else(|| assets.materials().next().map(|(_, h)| h))
}
