//! The view a frame is drawn from, and which scene camera supplies it.
//!
//! [`Camera`] is a resource, not a component: the renderer, culling, the
//! shadow cascades and the editor's picking all read one view, and every frame
//! has exactly one. In edit mode the editor's fly camera owns it. While the
//! game runs, [`follow_active_camera`] copies the active [`CameraComponent`]
//! entity into it once a frame, after the last transform propagation, so a
//! camera parented to a moving player draws from where the player is now.
//!
//! Which camera is active is [`ActiveCamera`]'s selection when that names a
//! live camera, and otherwise the camera with the highest `priority`. The
//! fallback is what lets a scene author the camera it opens on, and what keeps
//! a game drawing when the selected camera is despawned or a scene load
//! replaces it. Ties go to the lower entity slot: stable, but not an order
//! anyone authored, so a scene that wants a particular camera says so with a
//! priority.
//!
//! A world with no camera entity leaves the resource alone, so a game that
//! places the view directly through `Camera.Place` keeps working unchanged.

use glam::{Mat4, Quat, Vec3};
use orrin_ecs::{Entity, World};

use super::{CameraComponent, WorldTransform};
use crate::geom::Frustum;

/// The scene camera a script chose to draw from, if any. See the module docs
/// for what happens when it names no live camera.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActiveCamera(Option<Entity>);

impl ActiveCamera {
    /// `None` hands the choice back to the cameras' priorities.
    pub fn select(&mut self, camera: Option<Entity>) {
        self.0 = camera;
    }

    pub fn selected(&self) -> Option<Entity> {
        self.0
    }
}

/// The camera entity the game draws from this frame, if there is any camera.
pub fn active_camera(world: &World) -> Option<Entity> {
    let selected = world.get_resource::<ActiveCamera>().and_then(|a| a.0);
    if let Some(entity) = selected
        && world.has::<CameraComponent>(entity)
    {
        return Some(entity);
    }
    let mut best: Option<(i32, Entity)> = None;
    world
        .query::<&CameraComponent>()
        .for_each(|entity, camera| {
            let wins = best.is_none_or(|(priority, held)| {
                (camera.priority, std::cmp::Reverse(entity.index))
                    > (priority, std::cmp::Reverse(held.index))
            });
            if wins {
                best = Some((camera.priority, entity));
            }
        });
    best.map(|(_, entity)| entity)
}

/// Copy the active camera's world transform and lens into the [`Camera`]
/// resource, and say which camera that was. Leaves the resource untouched when
/// there is no camera entity.
pub fn follow_active_camera(world: &World) -> Option<Entity> {
    let entity = active_camera(world)?;
    let lens = *world.get::<CameraComponent>(entity)?;
    let placed = world
        .get::<WorldTransform>(entity)
        .map_or(Mat4::IDENTITY, |transform| transform.0);
    let (_, rotation, position) = placed.to_scale_rotation_translation();
    let mut view = world.get_resource_mut::<Camera>()?;
    view.place(position, rotation);
    view.fov_y = lens.fov_degrees.to_radians();
    view.near = lens.near;
    view.far = lens.far;
    Some(entity)
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 1.5, 4.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y: 60f32.to_radians(),
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl Camera {
    #[inline]
    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    /// [`Self::projection`] over `near..far` rather than the camera's own range.
    #[inline]
    pub fn projection_range(&self, aspect: f32, near: f32, far: f32) -> Mat4 {
        // Reversed by handing the range over backwards: `perspective_rh` maps
        // its first plane to 0 and its second to 1, and the result is exactly
        // `1 - d` of the forward matrix.
        let mut proj = Mat4::perspective_rh(self.fov_y, aspect, far, near);
        proj.y_axis.y *= -1.0;
        proj
    }

    // Right-handed perspective for Vulkan's [0,1] depth range, with Y flipped for
    // its +Y-down clip space. Reverse-Z: near maps to 1, far to 0, paired with a
    // GREATER depth test and a depth clear of 0.0 — `gfx::depth` holds those, and
    // says why, and why the far plane is still finite. Every shader that reads
    // this depth goes through `depth.glsl` for the same reason.
    #[inline]
    pub fn projection(&self, aspect: f32) -> Mat4 {
        self.projection_range(aspect, self.near, self.far)
    }

    #[inline]
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view()
    }

    /// The six clip planes of this camera's volume, for culling. `aspect` must
    /// be the one the frame is drawn with, or the side planes will not match
    /// what ends up on screen.
    #[inline]
    pub fn frustum(&self, aspect: f32) -> Frustum {
        Frustum::from_view_projection(self.view_projection(aspect))
    }

    /// Which way the camera faces, as a rotation of -Z (forward) and +Y (up):
    /// the form a script holds an orientation in.
    pub fn rotation(&self) -> Quat {
        let forward = (self.target - self.position).normalize_or(Vec3::NEG_Z);
        let right = forward.cross(self.up).normalize_or(Vec3::X);
        let up = right.cross(forward);
        Quat::from_mat3(&glam::Mat3::from_cols(right, up, -forward)).normalize()
    }

    /// Stand at `position` facing `rotation`. The point looked at keeps its
    /// distance, because the editor orbits and frames around it.
    pub fn place(&mut self, position: Vec3, rotation: Quat) {
        let distance = self.target.distance(self.position).max(1e-3);
        let rotation = rotation.normalize();
        self.position = position;
        self.target = position + rotation * Vec3::NEG_Z * distance;
        self.up = rotation * Vec3::Y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{LocalTransform, Transform, propagate_transforms, reparent};

    fn close(a: Vec3, b: Vec3) -> bool {
        a.distance(b) < 1e-4
    }

    /// Depth after the divide of a point `distance` metres straight ahead.
    fn depth_at(camera: &Camera, distance: f32) -> f32 {
        let forward = (camera.target - camera.position).normalize();
        camera
            .view_projection(16.0 / 9.0)
            .project_point3(camera.position + forward * distance)
            .z
    }

    /// Reversed: the near plane is 1 and the far plane 0, so a float depth
    /// buffer spends its exponent range on the distance rather than on the
    /// first few centimetres in front of the lens.
    #[test]
    fn the_projection_maps_near_to_one_and_far_to_zero() {
        let camera = Camera::default();
        assert!((depth_at(&camera, camera.near) - 1.0).abs() < 1e-6);
        assert!(depth_at(&camera, camera.far).abs() < 1e-6);
    }

    #[test]
    fn depth_decreases_with_distance() {
        let camera = Camera::default();
        let depths: Vec<f32> = [0.2, 1.0, 10.0, 100.0, 900.0]
            .into_iter()
            .map(|d| depth_at(&camera, d))
            .collect();
        assert!(depths.windows(2).all(|w| w[0] > w[1]), "{depths:?}");
    }

    /// The same distance the shaders' `linear_depth` recovers, in its closed
    /// form: `depth.glsl` and this are a lock-step pair.
    #[test]
    fn linear_depth_inverts_the_projection() {
        let camera = Camera::default();
        let (n, f) = (camera.near, camera.far);
        for distance in [0.1, 0.5, 3.0, 40.0, 999.0] {
            let d = depth_at(&camera, distance);
            let linear = n * f / (n + d * (f - n));
            assert!(
                (linear - distance).abs() / distance < 1e-3,
                "{distance} m read back as {linear} m"
            );
        }
    }

    /// What reversing is for. Two surfaces five centimetres apart at half a
    /// kilometre land on the same f32 depth forward — they would fight — and
    /// on depths hundreds of representable values apart reversed.
    #[test]
    fn reversed_depth_separates_surfaces_far_from_the_camera() {
        let camera = Camera::default();
        let (a, b) = (500.0, 500.05);
        assert_ne!(depth_at(&camera, a), depth_at(&camera, b));
        let forward = Camera {
            near: camera.far,
            far: camera.near,
            ..camera
        };
        let (fa, fb) = (depth_at(&forward, a), depth_at(&forward, b));
        assert!(
            (fa.to_bits() as i64 - fb.to_bits() as i64).abs() <= 1,
            "forward-Z already separates {fa} and {fb}, so this test proves nothing"
        );
        let ulps = depth_at(&camera, a).to_bits() as i64 - depth_at(&camera, b).to_bits() as i64;
        assert!(ulps > 100, "only {ulps} representable depths apart");
    }

    /// Reversing the depth swaps which of the two depth planes is near and
    /// which is far, but keeps both: the volume culled is the one drawn.
    #[test]
    fn the_frustum_keeps_exactly_what_lies_between_near_and_far() {
        use crate::geom::Aabb;
        let camera = Camera::default();
        let forward = (camera.target - camera.position).normalize();
        let frustum = camera.frustum(16.0 / 9.0);
        let at = |distance: f32| {
            let c = camera.position + forward * distance;
            Aabb {
                min: (c - Vec3::splat(0.01)).into(),
                max: (c + Vec3::splat(0.01)).into(),
            }
        };
        assert!(!frustum.intersects(&at(0.05)), "in front of the near plane");
        assert!(frustum.intersects(&at(500.0)));
        assert!(!frustum.intersects(&at(1100.0)), "beyond the far plane");
    }

    #[test]
    fn the_rotation_faces_the_target_with_the_up_kept_up() {
        let camera = Camera::default();
        let rotation = camera.rotation();
        assert!(close(
            rotation * Vec3::NEG_Z,
            (camera.target - camera.position).normalize()
        ));
        assert!((rotation * Vec3::Y).dot(Vec3::Y) > 0.9, "upright");
    }

    #[test]
    fn a_placed_camera_reads_back_where_and_how_it_was_placed() {
        let mut camera = Camera::default();
        let distance = camera.target.distance(camera.position);
        let rotation = Quat::from_euler(glam::EulerRot::YXZ, 0.6, -0.3, 0.0);
        let position = Vec3::new(2.0, 1.0, -3.0);

        camera.place(position, rotation);

        assert_eq!(camera.position, position);
        assert!(close(
            camera.rotation() * Vec3::NEG_Z,
            rotation * Vec3::NEG_Z
        ));
        assert!(close(camera.up, rotation * Vec3::Y));
        assert!((camera.target.distance(position) - distance).abs() < 1e-4);
    }

    fn camera_at(world: &mut World, translation: Vec3, priority: i32) -> Entity {
        let entity = world.spawn();
        world.insert(
            entity,
            LocalTransform::new(Transform {
                translation,
                ..Transform::default()
            }),
        );
        world.insert(
            entity,
            CameraComponent {
                priority,
                ..CameraComponent::default()
            },
        );
        entity
    }

    fn game_world() -> World {
        let mut world = World::new();
        world.insert_resource(Camera::default());
        world.insert_resource(ActiveCamera::default());
        world
    }

    #[test]
    fn with_nothing_selected_the_highest_priority_camera_draws() {
        let mut world = game_world();
        assert_eq!(active_camera(&world), None);
        let low = camera_at(&mut world, Vec3::ZERO, 0);
        let high = camera_at(&mut world, Vec3::ZERO, 5);
        let _also_low = camera_at(&mut world, Vec3::ZERO, 0);
        assert_eq!(active_camera(&world), Some(high));
        world.despawn(high);
        assert_eq!(
            active_camera(&world),
            Some(low),
            "ties go to the lower slot"
        );
    }

    #[test]
    fn a_selection_outranks_every_priority_until_it_stops_being_a_camera() {
        let mut world = game_world();
        let chosen = camera_at(&mut world, Vec3::ZERO, -1);
        let high = camera_at(&mut world, Vec3::ZERO, 5);
        world.resource_mut::<ActiveCamera>().select(Some(chosen));
        assert_eq!(active_camera(&world), Some(chosen));

        world.remove::<CameraComponent>(chosen);
        assert_eq!(active_camera(&world), Some(high), "no longer a camera");

        world.despawn(chosen);
        let reused = camera_at(&mut world, Vec3::ZERO, -1);
        assert_eq!(reused.index, chosen.index, "the slot is reused");
        assert_eq!(
            active_camera(&world),
            Some(high),
            "a stale handle selects nothing, not the slot's next occupant"
        );
    }

    #[test]
    fn the_view_follows_the_active_camera_lens_and_all() {
        let mut world = game_world();
        let entity = camera_at(&mut world, Vec3::new(1.0, 2.0, 3.0), 0);
        let facing = Quat::from_rotation_y(0.7);
        {
            let mut transform = world.get_mut::<LocalTransform>(entity).unwrap();
            transform.rotation = facing;
            // A scaled camera still looks along a unit forward.
            transform.scale = Vec3::splat(3.0);
        }
        *world.get_mut::<CameraComponent>(entity).unwrap() = CameraComponent {
            fov_degrees: 90.0,
            near: 0.5,
            far: 40.0,
            priority: 0,
        };
        propagate_transforms(&mut world);

        assert_eq!(follow_active_camera(&world), Some(entity));
        let view = *world.resource::<Camera>();
        assert!(close(view.position, Vec3::new(1.0, 2.0, 3.0)));
        assert!(close(view.rotation() * Vec3::NEG_Z, facing * Vec3::NEG_Z));
        assert!(close(view.up, Vec3::Y));
        assert!((view.fov_y - 90f32.to_radians()).abs() < 1e-6);
        assert_eq!((view.near, view.far), (0.5, 40.0));
    }

    /// The world transform, not the local one: a camera on a player's
    /// shoulder is where the player is.
    #[test]
    fn a_parented_camera_draws_from_where_its_parent_put_it() {
        let mut world = game_world();
        let player = world.spawn();
        world.insert(
            player,
            LocalTransform::new(Transform {
                translation: Vec3::new(10.0, 0.0, 0.0),
                ..Transform::default()
            }),
        );
        let camera = camera_at(&mut world, Vec3::new(0.0, 1.5, 2.0), 0);
        reparent(&mut world, camera, Some(player), false).unwrap();
        propagate_transforms(&mut world);

        follow_active_camera(&world);
        assert!(close(
            world.resource::<Camera>().position,
            Vec3::new(10.0, 1.5, 2.0)
        ));
    }

    /// What keeps `Camera.Place` working for a game with no camera entity.
    #[test]
    fn with_no_camera_entity_the_view_is_left_where_it_was() {
        let world = game_world();
        let placed = Vec3::new(4.0, 5.0, 6.0);
        world.resource_mut::<Camera>().position = placed;
        assert_eq!(follow_active_camera(&world), None);
        assert_eq!(world.resource::<Camera>().position, placed);
    }
}
