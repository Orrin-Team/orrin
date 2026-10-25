use glam::Vec3;

use orrin_ecs::{Entity, World};
use orrin_registry::{AssetId, Registry};

use super::selection::Selection;
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

/// What a transform gizmo would edit. The mode is real state and is remembered;
/// the handles that would read it are Phase 3, so today it drives a readout and
/// nothing else.
#[derive(Clone, Copy, PartialEq, Eq)]
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

/// What a gizmo drag would quantise to, when snapping is on.
pub const SNAP_STEP: f32 = 0.25;

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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SpawnKind {
    Cube,
    Sphere,
    Plane,
    PointLight,
    DirectionalLight,
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
    pub gizmo_mode: GizmoMode,
    pub gizmo_space: GizmoSpace,
    pub snap: bool,
    /// Whether dragging one scale axis moves all three by the same ratio.
    pub scale_linked: bool,
    /// The frame-time readout floating in the viewport's corner.
    pub show_stats: bool,
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
    instantiate_request: Option<(AssetId, Option<Entity>)>,
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
            gizmo_mode: GizmoMode::Select,
            gizmo_space: GizmoSpace::World,
            snap: false,
            scale_linked: false,
            show_stats: false,
            spawn_request: None,
            despawn_request: Vec::new(),
            add_request: None,
            reparent_request: Vec::new(),
            instantiate_request: None,
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
        self.instantiate_request = Some((prefab, parent));
    }

    pub fn request_prefab_rescan(&mut self) {
        self.rescan_request = true;
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
        if let Some(kind) = self.spawn_request.take()
            && let Some(entity) = spawn(world, kind)
        {
            self.selection.set(entity);
        }
        if !self.despawn_request.is_empty() {
            for entity in std::mem::take(&mut self.despawn_request) {
                crate::scene::despawn_recursive(world, entity);
            }
            self.selection.retain_alive(world);
        }
        if let Some((entity, id)) = self.add_request.take()
            && world.is_alive(entity)
            && let Some(vtable) = registry.get(&id)
            && !vtable.has(world, entity)
            && let Err(error) = vtable.write(world, entity, &vtable.default())
        {
            log(
                world,
                LogLevel::Error,
                format!("adding {id} to entity {}: {error}", entity.index()),
            );
        }
        let moves = std::mem::take(&mut self.reparent_request);
        for &(child, parent) in &moves {
            // One dragged along with an ancestor that is also moving keeps its
            // place under that ancestor, as a file does in a moved folder.
            if moves
                .iter()
                .any(|&(other, _)| other != child && is_under(world, child, other))
            {
                continue;
            }
            // `keep_world`: a drag in the tree changes who owns an object, not
            // where it is. Without it, dropping something onto a distant parent
            // would fling it across the scene.
            if let Err(error) = crate::scene::reparent(world, child, parent, true) {
                log(world, LogLevel::Warning, error.to_string());
            }
        }
        if let Some((prefab, parent)) = self.instantiate_request.take() {
            match scene::prefab::instantiate(world, registry, prefab) {
                Ok((root, issues)) => {
                    for issue in issues {
                        log(world, LogLevel::Warning, issue.to_string());
                    }
                    // Not `keep_world`: a prefab sits where it was authored to,
                    // relative to whatever it was dropped on.
                    if let Some(parent) = parent
                        && let Err(error) = scene::reparent(world, root, Some(parent), false)
                    {
                        log(world, LogLevel::Warning, error.to_string());
                    }
                    self.selection.set(root);
                }
                Err(error) => log(world, LogLevel::Error, error),
            }
        }
        if std::mem::take(&mut self.save_prefab_request) {
            self.save_prefab(world, registry);
        }
        if std::mem::take(&mut self.rescan_request) {
            for issue in scene::prefab::rescan(world, registry) {
                log(world, LogLevel::Warning, issue);
            }
        }
        if let Some(request) = self.scene_request.take() {
            let (level, message) = match request {
                SceneRequest::Save => save_scene(world, registry, &self.scene_path),
                SceneRequest::Load => {
                    self.selection.clear();
                    load_scene(world, registry, &self.scene_path)
                }
            };
            log(world, level, message);
        }
    }
}

impl EditorState {
    /// Save the selection as a prefab in the assets directory, named after the
    /// entity when there is one and `Group` when there are several, and select
    /// the instance it became.
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
        match scene::prefab::create(world, registry, &selection, &rel) {
            Ok((root, _)) => {
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
            let (mesh_name, label) = match kind {
                SpawnKind::Cube => ("cube", "Cube"),
                SpawnKind::Sphere => ("sphere", "Sphere"),
                _ => ("plane", "Plane"),
            };
            // Copy the handles out before mutating the world (drops the borrows).
            let mesh = world.resource::<Assets>().mesh(mesh_name)?;
            let material = default_material(world)?;
            Some(entities::spawn_mesh(
                world,
                label,
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
            "Point Light",
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::ONE,
            4000.0,
            10.0,
        )),
        SpawnKind::DirectionalLight => Some(entities::spawn_directional_light(
            world,
            "Directional Light",
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
