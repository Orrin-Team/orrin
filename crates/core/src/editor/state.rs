use glam::Vec3;

use orrin_ecs::{Entity, World};
use orrin_registry::Registry;

use crate::scene::{
    self, entities, Assets, LogBuffer, LogLevel, MaterialHandle, Time, Transform,
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
    pub selected: Option<Entity>,
    pub scene_path: String,
    spawn_request: Option<SpawnKind>,
    despawn_request: Option<Entity>,
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
            selected: None,
            scene_path: DEFAULT_SCENE_PATH.to_owned(),
            spawn_request: None,
            despawn_request: None,
            scene_request: None,
            #[cfg(feature = "scripting")]
            script_reload_request: false,
        }
    }
}

impl EditorState {
    pub fn request_spawn(&mut self, kind: SpawnKind) {
        self.spawn_request = Some(kind);
    }

    pub fn request_scene(&mut self, request: SceneRequest) {
        self.scene_request = Some(request);
    }

    pub fn request_despawn(&mut self, entity: Entity) {
        self.despawn_request = Some(entity);
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
        if let Some(kind) = self.spawn_request.take() {
            if let Some(entity) = spawn(world, kind) {
                self.selected = Some(entity);
            }
        }
        if let Some(entity) = self.despawn_request.take() {
            world.despawn(entity);
            if self.selected == Some(entity) {
                self.selected = None;
            }
        }
        if let Some(request) = self.scene_request.take() {
            let (level, message) = match request {
                SceneRequest::Save => save_scene(world, registry, &self.scene_path),
                SceneRequest::Load => {
                    self.selected = None;
                    load_scene(world, registry, &self.scene_path)
                }
            };
            log(world, level, message);
        }
    }
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
        0 => (LogLevel::Info, format!("loaded {count} entities from {path}")),
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
        SpawnKind::PointLight => Some(entities::spawn_point_light(
            world,
            "Point Light",
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::ONE,
            8.0,
            10.0,
        )),
        SpawnKind::DirectionalLight => Some(entities::spawn_directional_light(
            world,
            "Directional Light",
            Vec3::new(-0.4, -1.0, -0.6),
            Vec3::ONE,
            1.0,
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
