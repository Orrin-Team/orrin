//! Choosing an entity by pointing at it: the cursor's ray against each mesh's
//! bounds.
//!
//! Bounds rather than an ID buffer, the open choice #60 left: no render
//! target, no GPU readback and the frame of latency it brings, and the bounds
//! already exist for culling. The price is precision on a mesh that fills
//! little of its box — an arch picks through its opening.
//!
//! Each box is tested in its mesh's own space, with the ray carried in by the
//! inverse model matrix, so a rotated box stays the box rather than growing
//! into the world-aligned one culling uses. An affine map keeps the ray's
//! parameter, so the hit distances still compare across entities.

use glam::Vec3;
use orrin_ecs::{Entity, World};

use crate::scene::{Camera, MeshBounds, MeshHandle, WorldTransform};

/// The ray through `ndc` (−1 to 1, y down, as the scene's projection has it):
/// its start on the near plane, and a direction reaching the far plane.
pub fn ray(camera: &Camera, aspect: f32, ndc: egui::Vec2) -> (Vec3, Vec3) {
    let inverse = camera.view_projection(aspect).inverse();
    let near = inverse.project_point3(Vec3::new(ndc.x, ndc.y, 0.0));
    let far = inverse.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
    (near, far - near)
}

/// The nearest entity whose mesh bounds the ray from `origin` along
/// `direction` enters, if any.
pub fn pick(world: &World, origin: Vec3, direction: Vec3) -> Option<Entity> {
    let bounds = world.get_resource::<MeshBounds>()?;
    let mut nearest: Option<(f32, Entity)> = None;
    world
        .query::<(&WorldTransform, &MeshHandle)>()
        .for_each(|entity, (model, mesh)| {
            let Some(aabb) = bounds.get(*mesh) else {
                return;
            };
            let inverse = model.0.inverse();
            let hit = aabb.ray_entry(
                inverse.transform_point3(origin),
                inverse.transform_vector3(direction),
            );
            if let Some(t) = hit
                && nearest.is_none_or(|(best, _)| t < best)
            {
                nearest = Some((t, entity));
            }
        });
    nearest.map(|(_, entity)| entity)
}

#[cfg(test)]
mod tests {
    use glam::{Mat4, Quat, Vec3A};

    use super::*;
    use crate::geom::Aabb;

    fn world() -> World {
        let mut world = World::new();
        let mut bounds = MeshBounds::default();
        bounds.insert(
            MeshHandle(0),
            Aabb {
                min: Vec3A::splat(-0.5),
                max: Vec3A::splat(0.5),
            },
        );
        world.insert_resource(bounds);
        world
    }

    fn mesh(world: &mut World, model: Mat4) -> Entity {
        let entity = world.spawn();
        world.insert(entity, MeshHandle(0));
        world.insert(entity, WorldTransform(model));
        entity
    }

    #[test]
    fn the_nearest_box_along_the_ray_is_picked() {
        let mut world = world();
        let far = mesh(
            &mut world,
            Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)),
        );
        let near = mesh(
            &mut world,
            Mat4::from_translation(Vec3::new(0.0, 0.0, -2.0)),
        );
        let ray = (Vec3::new(0.0, 0.0, 3.0), Vec3::NEG_Z);
        assert_eq!(pick(&world, ray.0, ray.1), Some(near));
        world.despawn(near);
        assert_eq!(pick(&world, ray.0, ray.1), Some(far));
        assert_eq!(pick(&world, ray.0, Vec3::Z), None, "behind the ray");
    }

    /// A long thin box turned 45°: its world-aligned box covers the point the
    /// ray passes through, the box itself does not.
    #[test]
    fn a_rotated_box_is_tested_as_itself() {
        let mut world = world();
        let model = Mat4::from_scale_rotation_translation(
            Vec3::new(4.0, 0.1, 0.1),
            Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
            Vec3::ZERO,
        );
        let plank = mesh(&mut world, model);
        assert_eq!(pick(&world, Vec3::new(1.0, 5.0, 1.0), Vec3::NEG_Y), None);
        assert_eq!(
            pick(&world, Vec3::new(1.0, 5.0, -1.0), Vec3::NEG_Y),
            Some(plank)
        );
    }

    #[test]
    fn an_entity_without_measured_bounds_is_not_picked() {
        let mut world = world();
        let unknown = world.spawn();
        world.insert(unknown, MeshHandle(7));
        world.insert(unknown, WorldTransform(Mat4::IDENTITY));
        assert_eq!(pick(&world, Vec3::new(0.0, 0.0, 3.0), Vec3::NEG_Z), None);
    }
}
