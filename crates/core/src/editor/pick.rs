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

use glam::{Mat4, Vec3};
use orrin_ecs::{Entity, World};

use crate::scene::{Camera, Light, Locked, MeshBounds, MeshHandle, Parent, WorldTransform};

/// The ray through `ndc` (−1 to 1, y down, as the scene's projection has it):
/// its start on the near plane, and a direction reaching the far plane.
pub fn ray(camera: &Camera, aspect: f32, ndc: egui::Vec2) -> (Vec3, Vec3) {
    let inverse = camera.view_projection(aspect).inverse();
    let near = inverse.project_point3(Vec3::new(ndc.x, ndc.y, 0.0));
    let far = inverse.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
    (near, far - near)
}

/// How an entity with nothing drawn at its place is shown, and picked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Marker {
    Light,
    /// Anything else with no mesh: a group, an anchor, a script's holder.
    Empty,
}

/// Whether `entity` is out of reach in the viewport: [`Locked`] itself, or
/// under something that is.
pub fn is_locked(world: &World, entity: Entity) -> bool {
    let mut cursor = entity;
    // Bounded: a cycle the hierarchy has not broken yet would otherwise spin.
    for _ in 0..4096 {
        if world.has::<Locked>(cursor) {
            return true;
        }
        match world.get::<Parent>(cursor).map(|parent| parent.get()) {
            Some(parent) => cursor = parent,
            None => return false,
        }
    }
    false
}

/// How far from a marker's centre a click still picks it, in points.
pub const MARKER_REACH: f32 = 9.0;

/// Every entity shown as a marker, with where it stands: it has a place in
/// the world and no mesh to see or pick it by.
///
/// A walk over every placed entity each time it is asked. The editor asks
/// once a frame to draw them and once a click to pick them, and the cost is a
/// component check per entity; an index kept beside the world would have to
/// be kept right through every path that adds or removes a mesh.
pub fn each_marker(world: &World, mut visit: impl FnMut(Entity, Vec3, Marker)) {
    world
        .query::<&WorldTransform>()
        .for_each(|entity, transform| {
            if world.has::<MeshHandle>(entity) {
                return;
            }
            let marker = if world.has::<Light>(entity) {
                Marker::Light
            } else {
                Marker::Empty
            };
            visit(entity, transform.translation(), marker);
        });
}

/// Where a world point lands on `screen`, in points, through `view_proj`.
pub fn to_screen(view_proj: Mat4, screen: egui::Rect, point: Vec3) -> egui::Pos2 {
    let ndc = view_proj.project_point3(point);
    screen.min
        + egui::vec2(
            (ndc.x * 0.5 + 0.5) * screen.width(),
            (ndc.y * 0.5 + 0.5) * screen.height(),
        )
}

/// What is under `pointer`: the nearest of the meshes the ray from it enters
/// and the markers within [`MARKER_REACH`] of it.
pub fn pick_at(
    world: &World,
    camera: &Camera,
    screen: egui::Rect,
    pointer: egui::Pos2,
) -> Option<Entity> {
    let ndc = (pointer - screen.min) / screen.size() * 2.0 - egui::Vec2::splat(1.0);
    let (origin, direction) = ray(camera, screen.aspect_ratio(), ndc);
    let view_proj = camera.view_projection(screen.aspect_ratio());
    let mut nearest = mesh_hit(world, origin, direction);
    each_marker(world, |entity, at, _| {
        // In the ray's own units, so it compares with a mesh's entry.
        let t = (at - origin).dot(direction) / direction.length_squared();
        let reached = (to_screen(view_proj, screen, at) - pointer).length() <= MARKER_REACH;
        if t >= 0.0
            && reached
            && nearest.is_none_or(|(best, _)| t < best)
            && !is_locked(world, entity)
        {
            nearest = Some((t, entity));
        }
    });
    nearest.map(|(_, entity)| entity)
}

/// The nearest entity whose mesh bounds the ray from `origin` along
/// `direction` enters, and how far along the ray, in its own units.
fn mesh_hit(world: &World, origin: Vec3, direction: Vec3) -> Option<(f32, Entity)> {
    let bounds = world.get_resource::<MeshBounds>()?;
    let mut nearest: Option<(f32, Entity)> = None;
    world
        .query::<(&WorldTransform, &MeshHandle)>()
        .for_each(|entity, (model, mesh)| {
            let Some(aabb) = bounds.get(*mesh) else {
                return;
            };
            if is_locked(world, entity) {
                return;
            }
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
    nearest
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3A};

    use super::*;
    use crate::geom::Aabb;

    fn pick(world: &World, origin: Vec3, direction: Vec3) -> Option<Entity> {
        mesh_hit(world, origin, direction).map(|(_, entity)| entity)
    }

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

    fn screen() -> egui::Rect {
        egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1600.0, 900.0))
    }

    /// Where `point` lands on screen through the default camera.
    fn on_screen(point: Vec3) -> egui::Pos2 {
        let screen = screen();
        let ndc = Camera::default()
            .view_projection(screen.aspect_ratio())
            .project_point3(point);
        egui::pos2(
            (ndc.x * 0.5 + 0.5) * screen.width(),
            (ndc.y * 0.5 + 0.5) * screen.height(),
        )
    }

    fn light(world: &mut World, at: Vec3) -> Entity {
        let entity = world.spawn();
        world.insert(entity, crate::scene::Light::default());
        world.insert(entity, WorldTransform(Mat4::from_translation(at)));
        entity
    }

    #[test]
    fn a_light_is_picked_by_its_marker() {
        let mut world = world();
        let lamp = light(&mut world, Vec3::ZERO);
        let camera = Camera::default();
        let near = on_screen(Vec3::ZERO) + egui::vec2(4.0, 0.0);
        assert_eq!(pick_at(&world, &camera, screen(), near), Some(lamp));
        let far = on_screen(Vec3::ZERO) + egui::vec2(30.0, 0.0);
        assert_eq!(pick_at(&world, &camera, screen(), far), None);
    }

    #[test]
    fn markers_are_what_has_a_place_and_no_mesh() {
        let mut world = world();
        let lamp = light(&mut world, Vec3::ZERO);
        let group = world.spawn();
        world.insert(group, WorldTransform(Mat4::IDENTITY));
        mesh(&mut world, Mat4::IDENTITY);
        world.spawn();
        let mut seen = Vec::new();
        each_marker(&world, |entity, _, marker| seen.push((entity, marker)));
        seen.sort_by_key(|(entity, _)| entity.index());
        assert_eq!(seen, [(lamp, Marker::Light), (group, Marker::Empty)]);
    }

    /// A marker is behind or in front of a mesh like anything else: the
    /// nearer of the two along the ray is what the click meant.
    #[test]
    fn the_nearer_of_a_mesh_and_a_marker_is_picked() {
        let camera = Camera::default();
        let toward = (camera.position - camera.target).normalize();
        let middle = on_screen(Vec3::ZERO);

        let mut world = world();
        let cube = mesh(&mut world, Mat4::IDENTITY);
        let lamp = light(&mut world, toward * 2.0);
        assert_eq!(
            pick_at(&world, &camera, screen(), middle),
            Some(lamp),
            "in front"
        );

        world.despawn(lamp);
        light(&mut world, -toward * 2.0);
        assert_eq!(
            pick_at(&world, &camera, screen(), middle),
            Some(cube),
            "behind"
        );
    }

    /// A locked mesh lets the click through to what is behind it, and so does
    /// anything under it in the hierarchy, and a locked light's marker.
    #[test]
    fn a_click_passes_through_what_is_locked() {
        let camera = Camera::default();
        let toward = (camera.position - camera.target).normalize();
        let middle = on_screen(Vec3::ZERO);

        let mut world = world();
        let behind = mesh(&mut world, Mat4::from_translation(-toward * 3.0));
        let front = mesh(&mut world, Mat4::IDENTITY);
        assert_eq!(pick_at(&world, &camera, screen(), middle), Some(front));
        world.insert(front, Locked);
        assert_eq!(pick_at(&world, &camera, screen(), middle), Some(behind));

        world.remove::<Locked>(front);
        let holder = world.spawn();
        world.insert(holder, Locked);
        crate::scene::reparent(&mut world, front, Some(holder), false).unwrap();
        assert!(is_locked(&world, front));
        assert_eq!(
            pick_at(&world, &camera, screen(), middle),
            Some(behind),
            "under a lock"
        );

        world.despawn(behind);
        world.despawn(front);
        let lamp = light(&mut world, Vec3::ZERO);
        world.insert(lamp, Locked);
        assert_eq!(pick_at(&world, &camera, screen(), middle), None);
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
