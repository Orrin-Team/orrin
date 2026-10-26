//! The transform gizmo: where its handles are on screen, which one the pointer
//! is on, and what dragging one does to the entity.
//!
//! **Screen space.** The handles are built as strokes in points, from the
//! entity's origin and axes projected through the camera, and scaled so an
//! axis is always [`LENGTH`] points long however far away the entity is — at
//! the far end of a scene a world-sized gizmo is a speck. Hit-testing walks the
//! same segments the strokes are built from, so the shape the pointer grabs is
//! the shape drawn, and the renderer only fills them in (see
//! `gfx::vulkan::gizmo`).
//!
//! **Parent space.** A drag computes the entity's new *world* placement — along
//! a world axis, or about one — and writes it back as the [`LocalTransform`]
//! that produces it under the entity's parent. Writing the world transform
//! instead would fight the hierarchy, which derives it from the local one every
//! frame. The axes themselves are the scene's in world space and the entity's
//! own in local space.
//!
//! **Rings seen edge on.** A turn normally follows the pointer round the ring's
//! plane. Seen edge on, that plane is a line on screen and following the pointer
//! round it is ill-conditioned, so a drag begun there measures the pointer's
//! travel along the ring's screen tangent instead — which is what keeps turning
//! about X possible from the default camera, which sees the X ring edge on.
//!
//! One entity: the selection is a set, and the gizmo honours its primary.

use glam::{Mat4, Quat, Vec3};

use crate::editor::state::{GizmoMode, GizmoSpace};
use crate::gfx::vulkan::GizmoVertex;
use crate::scene::{Camera, LocalTransform};

/// How long an axis is on screen, and the radius of a ring, in points.
pub const LENGTH: f32 = 90.0;

/// How far from a handle the pointer may be and still grab it, in points.
const REACH: f32 = 7.0;

/// The colour of the handle under the pointer, or being dragged.
const HOT: egui::Color32 = egui::Color32::from_rgb(255, 214, 64);

/// Below this, a ring's plane is too near edge on to follow the pointer round.
const EDGE_ON: f32 = 0.25;

const RING_SEGMENTS: usize = 64;

/// One handle: an axis to move along, or the ring to turn about it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handle(pub usize);

/// The gizmo as one frame sees it: placed on an entity, through a camera.
pub struct Gizmo {
    origin: Vec3,
    axes: [Vec3; 3],
    /// [`LENGTH`] points, in world units at the origin's depth.
    length: f32,
    view_proj: Mat4,
    camera: Vec3,
    screen: egui::Rect,
}

impl Gizmo {
    /// Placed on an entity whose world matrix is `world`. `None` when the
    /// entity is behind the camera, where there is nothing to draw or grab.
    pub fn new(
        camera: &Camera,
        screen: egui::Rect,
        world: Mat4,
        space: GizmoSpace,
    ) -> Option<Self> {
        let (_, rotation, origin) = world.to_scale_rotation_translation();
        let forward = (camera.target - camera.position).normalize();
        let depth = forward.dot(origin - camera.position);
        if depth <= camera.near {
            return None;
        }
        // What one point spans at that depth, which is what keeps the gizmo
        // one size on screen.
        let per_point = 2.0 * depth * (camera.fov_y * 0.5).tan() / screen.height();
        let axes = match space {
            GizmoSpace::World => [Vec3::X, Vec3::Y, Vec3::Z],
            GizmoSpace::Local => [rotation * Vec3::X, rotation * Vec3::Y, rotation * Vec3::Z],
        };
        Some(Self {
            origin,
            axes,
            length: LENGTH * per_point,
            view_proj: camera.view_projection(screen.aspect_ratio()),
            camera: camera.position,
            screen,
        })
    }

    pub fn axis(&self, handle: Handle) -> Vec3 {
        self.axes[handle.0]
    }

    /// Where a world point lands on screen, in points.
    fn project(&self, point: Vec3) -> egui::Pos2 {
        to_screen(self.view_proj, self.screen, point)
    }

    /// The handle `pointer` is on, nearest first.
    pub fn hit(&self, mode: GizmoMode, pointer: egui::Pos2) -> Option<Handle> {
        let mut nearest: Option<(f32, Handle)> = None;
        self.each_segment(mode, |handle, a, b, _| {
            let distance = segment_distance(pointer, a, b);
            if distance < REACH && nearest.is_none_or(|(best, _)| distance < best) {
                nearest = Some((distance, handle));
            }
        });
        nearest.map(|(_, handle)| handle)
    }

    /// Append the handles of `mode` to `out` as triangles in physical pixels.
    pub fn strokes(
        &self,
        mode: GizmoMode,
        hot: Option<Handle>,
        pixels_per_point: f32,
        out: &mut Vec<GizmoVertex>,
    ) {
        self.each_segment(mode, |handle, a, b, part| {
            let (colour, width) = if hot == Some(handle) {
                (HOT, 1.75)
            } else {
                (orrin_registry::inspect::AXIS[handle.0], 1.25)
            };
            let [red, green, blue, _] = colour.to_array();
            let (alpha, from, to) = match part {
                Part::Shaft(alpha) => (alpha, width, width),
                Part::Head(alpha) => (alpha, 5.5, 0.0),
                Part::Ring => (1.0, width, width),
            };
            let colour = [red, green, blue, (alpha * 255.0).round() as u8];
            stroke(out, a, b, from, to, colour, pixels_per_point);
        });
    }

    /// Every piece of `mode`'s handles as a screen-space segment, in draw
    /// order: the one walk both drawing and hit-testing read.
    fn each_segment(
        &self,
        mode: GizmoMode,
        mut visit: impl FnMut(Handle, egui::Pos2, egui::Pos2, Part),
    ) {
        let origin = self.project(self.origin);
        match mode {
            GizmoMode::Move => {
                for (index, axis) in self.axes.iter().enumerate() {
                    let tip = self.project(self.origin + *axis * self.length);
                    // An axis pointing at the camera is a dot, which can be
                    // neither read nor dragged along, so it fades out instead.
                    let alpha = (((tip - origin).length() / LENGTH - 0.2) / 0.15).clamp(0.0, 1.0);
                    if alpha <= 0.0 {
                        continue;
                    }
                    let neck = origin + (tip - origin) * 0.8;
                    visit(Handle(index), origin, neck, Part::Shaft(alpha));
                    visit(Handle(index), neck, tip, Part::Head(alpha));
                }
            }
            GizmoMode::Rotate => {
                let toward = self.camera - self.origin;
                for (index, axis) in self.axes.iter().enumerate() {
                    // Only the half facing the camera: the far half crosses
                    // the near one, and two crossing rings read as neither.
                    for step in 0..RING_SEGMENTS {
                        let (a, b) = (self.rim(*axis, step), self.rim(*axis, step + 1));
                        if (a - self.origin).dot(toward) < 0.0
                            || (b - self.origin).dot(toward) < 0.0
                        {
                            continue;
                        }
                        visit(Handle(index), self.project(a), self.project(b), Part::Ring);
                    }
                }
            }
            GizmoMode::Select | GizmoMode::Scale => {}
        }
    }

    /// Point `step` of [`RING_SEGMENTS`] round the ring about `axis`.
    fn rim(&self, axis: Vec3, step: usize) -> Vec3 {
        let (u, v) = axis.any_orthonormal_pair();
        let angle = step as f32 / RING_SEGMENTS as f32 * std::f32::consts::TAU;
        self.origin + (u * angle.cos() + v * angle.sin()) * self.length
    }
}

#[derive(Clone, Copy)]
enum Part {
    /// With how far the axis has faded.
    Shaft(f32),
    Head(f32),
    Ring,
}

/// A drag in progress: everything about the entity as it was when the handle
/// was grabbed, so each frame computes from the start rather than from the
/// last frame, and a drag back to where it began lands exactly there.
#[derive(Clone, Copy, Debug)]
pub struct Drag {
    pub handle: Handle,
    grip: Grip,
    origin: Vec3,
    axis: Vec3,
    local: LocalTransform,
    world_rotation: Quat,
    parent: Mat4,
}

/// What a drag measures the pointer against.
#[derive(Clone, Copy, Debug)]
enum Grip {
    /// A move: how far along the axis the grab was.
    Along(f32),
    /// A turn seen face enough on: from the origin to where the grab met the
    /// ring's plane.
    Around(Vec3),
    /// A turn seen edge on: the grab's place on screen, and the ring's screen
    /// tangent there, scaled to radians per point of travel.
    Across {
        view_proj: Mat4,
        screen: egui::Rect,
        from: egui::Pos2,
        tangent: egui::Vec2,
    },
}

impl Drag {
    /// Grab `handle` with the pointer's ray. `parent` is the parent's world
    /// matrix, identity for a root; `world` the entity's own.
    pub fn begin(
        gizmo: &Gizmo,
        mode: GizmoMode,
        handle: Handle,
        ray: (Vec3, Vec3),
        local: LocalTransform,
        parent: Mat4,
        world: Mat4,
    ) -> Option<Self> {
        let origin = gizmo.origin;
        let axis = gizmo.axis(handle);
        let grip = match mode {
            GizmoMode::Move => Grip::Along(along_axis(ray, origin, axis)?),
            GizmoMode::Rotate => {
                let facing = (gizmo.camera - origin).normalize().dot(axis).abs();
                match plane_hit(ray, origin, axis) {
                    Some(hit) if facing >= EDGE_ON => Grip::Around(hit - origin),
                    _ => {
                        let from = to_screen(gizmo.view_proj, gizmo.screen, ray.0 + ray.1);
                        // The near half only, as drawn: seen edge on, each
                        // point of it shares its place on screen with one of
                        // the far half, whose tangent runs the other way.
                        let toward = gizmo.camera - origin;
                        let grabbed = (0..RING_SEGMENTS)
                            .map(|step| gizmo.rim(axis, step))
                            .filter(|point| (*point - origin).dot(toward) >= 0.0)
                            .min_by(|a, b| {
                                let away = |p: &Vec3| (gizmo.project(*p) - from).length_sq();
                                away(a).total_cmp(&away(b))
                            })
                            .expect("a ring has points");
                        let nudge = axis.cross(grabbed - origin).normalize() * gizmo.length * 0.01;
                        let along = gizmo.project(grabbed + nudge) - gizmo.project(grabbed);
                        Grip::Across {
                            view_proj: gizmo.view_proj,
                            screen: gizmo.screen,
                            from,
                            // A ring's screen radius is `LENGTH`, so this many
                            // points round it is one radian.
                            tangent: along.normalized() / LENGTH,
                        }
                    }
                }
            }
            GizmoMode::Select | GizmoMode::Scale => return None,
        };
        Some(Self {
            handle,
            grip,
            origin,
            axis,
            local,
            world_rotation: world.to_scale_rotation_translation().1,
            parent,
        })
    }

    /// The local transform the drag has reached at the pointer's ray, with the
    /// distance or angle rounded to `snap` — metres for a move, degrees for a
    /// turn — when one is given.
    pub fn update(&self, ray: (Vec3, Vec3), snap: Option<f32>) -> LocalTransform {
        let round = |value: f32| match snap {
            Some(step) if step > 0.0 => (value / step).round() * step,
            _ => value,
        };
        let mut local = self.local;
        let angle = match self.grip {
            Grip::Along(start) => {
                if let Some(now) = along_axis(ray, self.origin, self.axis) {
                    let moved = self.origin + self.axis * round(now - start);
                    local.translation = self.parent.inverse().transform_point3(moved);
                }
                return local;
            }
            Grip::Around(from) => {
                let Some(hit) = plane_hit(ray, self.origin, self.axis) else {
                    return local;
                };
                let to = hit - self.origin;
                self.axis.dot(from.cross(to)).atan2(from.dot(to))
            }
            Grip::Across {
                view_proj,
                screen,
                from,
                tangent,
            } => (to_screen(view_proj, screen, ray.0 + ray.1) - from).dot(tangent),
        };
        let turn = Quat::from_axis_angle(self.axis, round(angle.to_degrees()).to_radians());
        let (_, parent, _) = self.parent.to_scale_rotation_translation();
        local.rotation = (parent.inverse() * turn * self.world_rotation).normalize();
        local
    }
}

fn to_screen(view_proj: Mat4, screen: egui::Rect, point: Vec3) -> egui::Pos2 {
    let ndc = view_proj.project_point3(point);
    screen.min
        + egui::vec2(
            (ndc.x * 0.5 + 0.5) * screen.width(),
            (ndc.y * 0.5 + 0.5) * screen.height(),
        )
}

/// How far along the line through `origin` in `axis` the point nearest the
/// ray is. `None` when the ray runs along the axis, where every point of it is
/// equally near.
fn along_axis((start, direction): (Vec3, Vec3), origin: Vec3, axis: Vec3) -> Option<f32> {
    let direction = direction.normalize();
    let facing = direction.dot(axis);
    let denominator = 1.0 - facing * facing;
    if denominator < 1e-6 {
        return None;
    }
    let offset = start - origin;
    Some((axis.dot(offset) - facing * direction.dot(offset)) / denominator)
}

/// Where the ray crosses the plane through `origin` facing `normal`, ahead of
/// the ray's start.
fn plane_hit((start, direction): (Vec3, Vec3), origin: Vec3, normal: Vec3) -> Option<Vec3> {
    let facing = direction.dot(normal);
    if facing.abs() < 1e-6 {
        return None;
    }
    let t = (origin - start).dot(normal) / facing;
    (t >= 0.0).then(|| start + direction * t)
}

/// How far `point` is from the segment `a`–`b`.
fn segment_distance(point: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let along = b - a;
    let t = ((point - a).dot(along) / along.length_sq().max(f32::EPSILON)).clamp(0.0, 1.0);
    (point - (a + along * t)).length()
}

/// One segment as a quad, `from` points half wide at `a` and `to` at `b`,
/// written in physical pixels and a pixel wider each side for the antialiasing
/// the fragment shader fades over.
fn stroke(
    out: &mut Vec<GizmoVertex>,
    a: egui::Pos2,
    b: egui::Pos2,
    from: f32,
    to: f32,
    color: [u8; 4],
    pixels_per_point: f32,
) {
    let (a, b) = (
        a.to_vec2() * pixels_per_point,
        b.to_vec2() * pixels_per_point,
    );
    let along = b - a;
    let length = along.length();
    if length < 1e-3 {
        return;
    }
    let normal = egui::vec2(-along.y, along.x) / length;
    let corner = |at: egui::Vec2, half: f32, side: f32| {
        let half = half * pixels_per_point;
        let reach = (half + 1.0) * side;
        GizmoVertex {
            position: (at + normal * reach).into(),
            edge: [reach, half],
            color,
        }
    };
    let (a0, a1) = (corner(a, from, -1.0), corner(a, from, 1.0));
    let (b0, b1) = (corner(b, to, -1.0), corner(b, to, 1.0));
    out.extend([a0, a1, b1, a0, b1, b0]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::pick;
    use crate::scene::Transform;

    fn screen() -> egui::Rect {
        egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1600.0, 900.0))
    }

    /// The pointer's ray at a screen point, as the viewport casts it.
    fn ray_at(camera: &Camera, at: egui::Pos2) -> (Vec3, Vec3) {
        let screen = screen();
        let ndc = (at - screen.min) / screen.size() * 2.0 - egui::Vec2::splat(1.0);
        pick::ray(camera, screen.aspect_ratio(), ndc)
    }

    fn at(position: Vec3) -> Mat4 {
        Mat4::from_translation(position)
    }

    fn gizmo(camera: &Camera, world: Mat4) -> Gizmo {
        Gizmo::new(camera, screen(), world, GizmoSpace::World).expect("in front of the camera")
    }

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-3
    }

    #[test]
    fn the_gizmo_is_the_same_size_on_screen_near_and_far() {
        let camera = Camera::default();
        let along = (camera.target - camera.position).normalize();
        let screen_length = |distance: f32| {
            let origin = camera.position + along * distance;
            let gizmo = gizmo(&camera, at(origin));
            (gizmo.project(origin + Vec3::X * gizmo.length) - gizmo.project(origin)).length()
        };
        let near = screen_length(3.0);
        let far = screen_length(60.0);
        assert!((near - LENGTH).abs() < 2.0, "{near}");
        assert!((far - near).abs() < 1.0, "{near} near, {far} far");
    }

    #[test]
    fn nothing_behind_the_camera_has_a_gizmo() {
        let camera = Camera::default();
        let behind = camera.position * 2.0 - camera.target;
        assert!(Gizmo::new(&camera, screen(), at(behind), GizmoSpace::World).is_none());
    }

    #[test]
    fn a_move_handle_is_grabbed_along_its_shaft_and_not_beside_it() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let middle = gizmo.project(Vec3::X * gizmo.length * 0.5);
        assert_eq!(gizmo.hit(GizmoMode::Move, middle), Some(Handle(0)));
        let up = gizmo.project(Vec3::Y * gizmo.length * 0.6);
        assert_eq!(gizmo.hit(GizmoMode::Move, up), Some(Handle(1)));
        let beside = middle + egui::vec2(0.0, 30.0);
        assert_eq!(gizmo.hit(GizmoMode::Move, beside), None);
        assert_eq!(gizmo.hit(GizmoMode::Select, middle), None);
    }

    #[test]
    fn a_ring_is_grabbed_on_its_rim() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        // The Y ring lies in the ground plane, and this point of it is on
        // the half toward the camera, the half that is drawn — and on no
        // other ring, as the point straight toward the camera would be.
        let toward = Vec3::new(camera.position.x, 0.0, camera.position.z).normalize();
        let aside = Quat::from_rotation_y(45f32.to_radians()) * toward;
        let rim = gizmo.project(aside * gizmo.length);
        assert_eq!(gizmo.hit(GizmoMode::Rotate, rim), Some(Handle(1)));
        let outside = gizmo.project(Vec3::ZERO) + egui::vec2(LENGTH * 2.0, 0.0);
        assert_eq!(gizmo.hit(GizmoMode::Rotate, outside), None);
    }

    #[test]
    fn local_axes_follow_the_entity() {
        let camera = Camera::default();
        let world = Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let local = Gizmo::new(&camera, screen(), world, GizmoSpace::Local).unwrap();
        assert!(
            close(local.axis(Handle(0)), Vec3::NEG_Z),
            "{}",
            local.axis(Handle(0))
        );
        let global = Gizmo::new(&camera, screen(), world, GizmoSpace::World).unwrap();
        assert!(close(global.axis(Handle(0)), Vec3::X));
    }

    /// Grab the X shaft and drag the pointer to where `to` projects.
    fn drag_x(
        camera: &Camera,
        gizmo: &Gizmo,
        parent: Mat4,
        local: LocalTransform,
        to: Vec3,
        snap: Option<f32>,
    ) -> LocalTransform {
        let grab = gizmo.project(gizmo.origin + Vec3::X * gizmo.length * 0.5);
        let world = parent * local.matrix();
        let drag = Drag::begin(
            gizmo,
            GizmoMode::Move,
            Handle(0),
            ray_at(camera, grab),
            local,
            parent,
            world,
        )
        .expect("the shaft was grabbed");
        let offset = to - gizmo.origin;
        let release = gizmo.project(gizmo.origin + Vec3::X * gizmo.length * 0.5 + offset);
        drag.update(ray_at(camera, release), snap)
    }

    #[test]
    fn dragging_the_x_handle_moves_along_x_only() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let moved = drag_x(
            &camera,
            &gizmo,
            Mat4::IDENTITY,
            LocalTransform::default(),
            Vec3::new(1.3, 0.0, 0.0),
            None,
        );
        assert!(
            close(moved.translation, Vec3::new(1.3, 0.0, 0.0)),
            "{}",
            moved.translation
        );
        assert_eq!(moved.rotation, Quat::IDENTITY);
    }

    #[test]
    fn a_move_snaps_its_distance_to_the_step() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let moved = drag_x(
            &camera,
            &gizmo,
            Mat4::IDENTITY,
            LocalTransform::default(),
            Vec3::new(0.6, 0.0, 0.0),
            Some(0.25),
        );
        assert!(
            close(moved.translation, Vec3::new(0.5, 0.0, 0.0)),
            "{}",
            moved.translation
        );
    }

    /// The parent is twice the size, so a metre in the world is half a unit of
    /// the child's own translation.
    #[test]
    fn a_move_under_a_parent_edits_the_parent_relative_translation() {
        let camera = Camera::default();
        let parent = Mat4::from_scale_rotation_translation(
            Vec3::splat(2.0),
            Quat::IDENTITY,
            Vec3::new(-1.0, 0.0, 0.0),
        );
        let local = LocalTransform::new(Transform::from_translation(Vec3::new(0.5, 0.0, 0.0)));
        let gizmo = gizmo(&camera, parent * local.matrix());
        let moved = drag_x(
            &camera,
            &gizmo,
            parent,
            local,
            Vec3::new(1.0, 0.0, 0.0),
            None,
        );
        assert!(
            close(moved.translation, Vec3::new(1.0, 0.0, 0.0)),
            "{}",
            moved.translation
        );
        assert!(close(
            (parent * moved.matrix()).w_axis.truncate(),
            Vec3::new(1.0, 0.0, 0.0)
        ));
    }

    /// Turn the Y ring by the angle between two points on its rim.
    fn turn_y(
        camera: &Camera,
        parent: Mat4,
        local: LocalTransform,
        degrees: f32,
        snap: Option<f32>,
    ) -> LocalTransform {
        let world = parent * local.matrix();
        let gizmo = gizmo(camera, world);
        let toward = Vec3::new(camera.position.x, 0.0, camera.position.z).normalize();
        let rim = |direction: Vec3| gizmo.project(gizmo.origin + direction * gizmo.length);
        let drag = Drag::begin(
            &gizmo,
            GizmoMode::Rotate,
            Handle(1),
            ray_at(camera, rim(toward)),
            local,
            parent,
            world,
        )
        .expect("the ring was grabbed");
        let turned = Quat::from_rotation_y(degrees.to_radians()) * toward;
        drag.update(ray_at(camera, rim(turned)), snap)
    }

    #[test]
    fn turning_a_ring_rotates_about_its_axis_by_the_angle_dragged() {
        let camera = Camera::default();
        let turned = turn_y(
            &camera,
            Mat4::IDENTITY,
            LocalTransform::default(),
            40.0,
            None,
        );
        let expected = Quat::from_rotation_y(40f32.to_radians());
        assert!(
            turned.rotation.angle_between(expected) < 1e-3,
            "{:?}",
            turned.rotation
        );
    }

    #[test]
    fn a_turn_snaps_its_angle_to_the_step() {
        let camera = Camera::default();
        let turned = turn_y(
            &camera,
            Mat4::IDENTITY,
            LocalTransform::default(),
            40.0,
            Some(15.0),
        );
        let expected = Quat::from_rotation_y(45f32.to_radians());
        assert!(
            turned.rotation.angle_between(expected) < 1e-3,
            "{:?}",
            turned.rotation
        );
    }

    /// Under a parent turned on X, a turn about the *world* Y axis is not a
    /// turn about the child's own Y: the result has to be what composes, under
    /// that parent, to the world rotation asked for.
    #[test]
    fn a_turn_under_a_parent_composes_to_the_world_rotation_asked_for() {
        let camera = Camera::default();
        let parent = Mat4::from_rotation_x(30f32.to_radians());
        let local = LocalTransform::default();
        let turned = turn_y(&camera, parent, local, 40.0, None);
        let world = Quat::from_mat4(&(parent * turned.matrix()));
        let expected =
            Quat::from_rotation_y(40f32.to_radians()) * Quat::from_rotation_x(30f32.to_radians());
        assert!(
            world.angle_between(expected) < 1e-3,
            "{world:?} vs {expected:?}"
        );
    }

    /// The default camera sees the X ring edge on, a line through the middle
    /// of the gizmo. Dragging across that line still turns about X, and the
    /// grabbed point of the rim follows the pointer.
    #[test]
    fn an_edge_on_ring_turns_by_the_pointer_s_travel_across_it() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let toward = Vec3::new(0.0, camera.position.y, camera.position.z).normalize();
        let grabbed = toward * gizmo.length;
        let from = gizmo.project(grabbed);
        let drag = Drag::begin(
            &gizmo,
            GizmoMode::Rotate,
            Handle(0),
            ray_at(&camera, from),
            LocalTransform::default(),
            Mat4::IDENTITY,
            Mat4::IDENTITY,
        )
        .expect("the ring was grabbed");
        let to = from + egui::vec2(0.0, -20.0);
        let turned = drag.update(ray_at(&camera, to), None);

        let (axis, angle) = turned.rotation.to_axis_angle();
        assert!(angle > 0.05, "it turned: {angle}");
        assert!(axis.dot(Vec3::X).abs() > 0.999, "about X: {axis}");
        let followed = gizmo.project(turned.rotation * grabbed);
        assert!(
            (followed - to).length() < (from - to).length(),
            "{followed:?}"
        );
    }

    #[test]
    fn the_handles_are_triangles_in_physical_pixels_and_the_hot_one_is_lit() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let mut points = Vec::new();
        gizmo.strokes(GizmoMode::Move, None, 1.0, &mut points);
        assert!(
            !points.is_empty() && points.len() % 3 == 0,
            "{}",
            points.len()
        );
        let mut pixels = Vec::new();
        gizmo.strokes(GizmoMode::Move, Some(Handle(0)), 2.0, &mut pixels);
        assert_eq!(pixels.len(), points.len());
        // Doubled, give or take the pixel of antialiasing fringe, which is a
        // pixel at any scale.
        let right = |vertices: &[GizmoVertex]| {
            vertices
                .iter()
                .map(|v| v.position[0])
                .fold(f32::MIN, f32::max)
        };
        assert!((right(&pixels) - right(&points) * 2.0).abs() < 2.0);

        let colours = |vertices: &[GizmoVertex]| -> Vec<[u8; 4]> {
            vertices.iter().map(|v| v.color).collect()
        };
        let x = orrin_registry::inspect::AXIS[0].to_array();
        let y = orrin_registry::inspect::AXIS[1].to_array();
        assert!(colours(&pixels).contains(&y));
        assert!(
            !colours(&pixels).contains(&x),
            "the hot X handle is lit instead"
        );
        assert!(colours(&pixels).contains(&HOT.to_array()));

        let mut none = Vec::new();
        gizmo.strokes(GizmoMode::Select, None, 1.0, &mut none);
        assert!(none.is_empty());
    }
}
