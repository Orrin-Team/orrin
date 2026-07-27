use glam::Vec3;

use ferron_ecs::{Entity, World};

use crate::scene::{entities, Assets, MaterialHandle, Transform};

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
#[derive(Default)]
pub struct EditorState {
    pub selected: Option<Entity>,
    spawn_request: Option<SpawnKind>,
    despawn_request: Option<Entity>,
    /// Unlike the two above, this one is *not* serviced by `apply`: a script
    /// reload destroys and re-creates managed objects, which may only happen at
    /// the start of the frame's script phase, outside any dispatch window. The
    /// app drains it there.
    #[cfg(feature = "scripting")]
    script_reload_request: bool,
}

impl EditorState {
    pub fn request_spawn(&mut self, kind: SpawnKind) {
        self.spawn_request = Some(kind);
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

    pub fn apply(&mut self, world: &mut World) {
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
