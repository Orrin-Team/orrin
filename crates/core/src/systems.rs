use glam::Vec3;

use orrin_ecs::World;

use crate::gfx::{PointLight, RenderItem, SceneLighting, MAX_POINT_LIGHTS};
use crate::scene::{AmbientLight, Light, LocalTransform, MaterialHandle, MeshHandle, Spin};

pub fn spin(world: &World, dt: f32) {
    world
        .query::<(&mut LocalTransform, &Spin)>()
        .for_each(|_entity, (transform, spin)| spin.apply(transform, dt));
}

pub fn extract_renderables(world: &World, out: &mut Vec<RenderItem>) {
    out.clear();
    world
        .query::<(&LocalTransform, &MeshHandle, Option<&MaterialHandle>)>()
        .for_each(|_entity, (transform, mesh, material)| {
            out.push(RenderItem {
                model: transform.matrix(),
                mesh: *mesh,
                material: material.copied().unwrap_or(MaterialHandle(0)),
            });
        });
}

pub fn extract_lighting(world: &World, out: &mut SceneLighting) {
    let defaults = SceneLighting::default();
    out.ambient_color = defaults.ambient_color;
    out.ambient_intensity = defaults.ambient_intensity;
    out.sun = defaults.sun;
    out.shininess = defaults.shininess;
    out.specular_strength = defaults.specular_strength;
    out.point_lights.clear();

    if let Some(ambient) = world.get_resource::<AmbientLight>() {
        out.ambient_color = ambient.color;
        out.ambient_intensity = ambient.intensity;
    }

    let mut has_sun = false;
    world
        .query::<(&LocalTransform, &Light)>()
        .for_each(|_entity, (transform, light)| match *light {
            Light::Directional { color, intensity } => {
                // The shader supports one directional light, so the first wins.
                if !has_sun {
                    let direction = (transform.rotation * Vec3::NEG_Z).normalize_or_zero();
                    out.sun.direction = direction;
                    out.sun.color = color;
                    out.sun.intensity = intensity;
                    has_sun = true;
                }
            }
            Light::Point {
                color,
                intensity,
                range,
            } => {
                if out.point_lights.len() < MAX_POINT_LIGHTS {
                    out.point_lights.push(PointLight {
                        position: transform.translation,
                        color,
                        intensity,
                        range,
                    });
                }
            }
        });
}
