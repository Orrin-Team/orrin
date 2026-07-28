mod scene;
mod textures;

pub use scene::build_default_scene;

use glam::{Quat, Vec3};

use orrin_ecs::{Entity, World};

use crate::scene::{Light, LocalTransform, MaterialHandle, MeshHandle, Name, Transform};

pub fn spawn_mesh(
    world: &mut World,
    name: impl Into<String>,
    transform: Transform,
    mesh: MeshHandle,
    material: MaterialHandle,
) -> Entity {
    world
        .spawn_entity()
        .with(Name::new(name))
        .with(LocalTransform::from(transform))
        .with(mesh)
        .with(material)
        .id()
}

pub fn spawn_point_light(
    world: &mut World,
    name: impl Into<String>,
    position: Vec3,
    color: Vec3,
    intensity: f32,
    range: f32,
) -> Entity {
    world
        .spawn_entity()
        .with(Name::new(name))
        .with(LocalTransform::from(Transform::from_translation(position)))
        .with(Light::point(color, intensity, range))
        .id()
}

/// The direction is stored as the entity's rotation (forward = `-Z`), so it can
/// be reoriented like any other transform.
pub fn spawn_directional_light(
    world: &mut World,
    name: impl Into<String>,
    direction: Vec3,
    color: Vec3,
    intensity: f32,
) -> Entity {
    let rotation = Quat::from_rotation_arc(Vec3::NEG_Z, direction.normalize_or_zero());
    world
        .spawn_entity()
        .with(Name::new(name))
        .with(LocalTransform::from(Transform {
            rotation,
            ..Default::default()
        }))
        .with(Light::directional(color, intensity))
        .id()
}
