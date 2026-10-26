//! The transform gizmo: where its handles are on screen, which one the pointer
//! is on, and what dragging one does to the entity.
//!
//! **Screen space.** The handles are built as shapes in points, from the
//! entity's origin and axes projected through the camera, and scaled so an
//! axis is always [`LENGTH`] points long however far away the entity is — at
//! the far end of a scene a world-sized gizmo is a speck. Hit-testing walks the
//! same shapes the strokes are built from, so the shape the pointer grabs is
//! the shape drawn, and the renderer only fills them in (see
//! `gfx::vulkan::gizmo`).
//!
//! **The handles.** Move has an arrow per axis, a square per plane to move in
//! two axes at once, and a centre that moves in the plane of the screen. Rotate
//! has a ring per axis. Scale has a box-tipped axis to stretch along, and a
//! centre that scales evenly.
//!
//! **Parent space.** A drag computes the entity's new *world* placement — along
//! a world axis, or about one — and writes it back as the [`LocalTransform`]
//! that produces it under the entity's parent. Writing the world transform
//! instead would fight the hierarchy, which derives it from the local one every
//! frame. Move and rotate use the scene's axes in world space and the entity's
//! own in local space. Scale always uses the entity's own: a stretch along a
//! world axis of a turned entity is a shear, which a transform cannot hold.
//!
//! **Seen edge on.** A turn normally follows the pointer round the ring's
//! plane. Seen edge on, that plane is a line on screen and following the pointer
//! round it is ill-conditioned, so a drag begun there measures the pointer's
//! travel along the ring's screen tangent instead — which is what keeps turning
//! about X possible from the default camera, which sees the X ring edge on. A
//! move square seen edge on has no such fallback and is hidden instead.
//!
//! **The selection.** The gizmo stands at the centre of everything selected,
//! turned as the primary is, and a drag moves each selected entity about that
//! centre (see [`Drag`]).

use glam::{Mat4, Quat, Vec3};

use orrin_ecs::Entity;

use crate::editor::pick::{Marker, to_screen};
use crate::editor::state::{GizmoMode, GizmoSpace};
use crate::gfx::vulkan::GizmoVertex;
use crate::scene::{Camera, LocalTransform};

/// How long an axis is on screen, and the radius of a ring, in points.
pub const LENGTH: f32 = 90.0;

/// How far from a handle the pointer may be and still grab it, in points.
const REACH: f32 = 7.0;

/// The colour of the handle under the pointer, or being dragged.
const HOT: egui::Color32 = egui::Color32::from_rgb(255, 214, 64);

/// Where a plane's square sits, as fractions of [`LENGTH`] out along each of
/// its two axes: clear of the centre, inside the arrowheads.
const PLANE: (f32, f32) = (0.3, 0.5);

/// The colour of the centre handle, which belongs to no axis.
const CENTRE: egui::Color32 = egui::Color32::from_rgb(220, 222, 228);

/// Half the side of the centre's box, and of a scale axis's tip, in points.
const CENTRE_BOX: f32 = 6.0;
const TIP_BOX: f32 = 4.5;

/// Below this, a ring's plane is too near edge on to follow the pointer round.
const EDGE_ON: f32 = 0.25;

/// The smallest factor a scale drag reaches: zero would make the transform
/// singular, and everything that inverts it with it.
const MIN_FACTOR: f32 = 0.01;

const RING_SEGMENTS: usize = 64;

/// One handle of the gizmo.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handle {
    /// An axis: to move or stretch along, or the ring to turn about it.
    Axis(usize),
    /// The square between the other two axes: to move in the plane this axis
    /// is the normal of.
    Plane(usize),
    /// The middle: to move in the plane of the screen, or to scale evenly.
    Centre,
}

/// The gizmo as one frame sees it: placed on an entity, through a camera.
pub struct Gizmo {
    origin: Vec3,
    /// What move and rotate use, which the space decides.
    axes: [Vec3; 3],
    /// The entity's own, which scale always uses.
    own: [Vec3; 3],
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
        let own = [rotation * Vec3::X, rotation * Vec3::Y, rotation * Vec3::Z];
        let axes = match space {
            GizmoSpace::World => [Vec3::X, Vec3::Y, Vec3::Z],
            GizmoSpace::Local => own,
        };
        Some(Self {
            origin,
            axes,
            own,
            length: LENGTH * per_point,
            view_proj: camera.view_projection(screen.aspect_ratio()),
            camera: camera.position,
            screen,
        })
    }

    /// The axis `handle` belongs to under `mode`: its own for an axis, the
    /// normal for a plane, and the way the camera looks for the centre.
    pub fn axis_of(&self, mode: GizmoMode, handle: Handle) -> Vec3 {
        let axes = match mode {
            GizmoMode::Scale => &self.own,
            _ => &self.axes,
        };
        match handle {
            Handle::Axis(index) | Handle::Plane(index) => axes[index],
            Handle::Centre => (self.origin - self.camera).normalize(),
        }
    }

    /// Where a world point lands on screen, in points.
    fn project(&self, point: Vec3) -> egui::Pos2 {
        to_screen(self.view_proj, self.screen, point)
    }

    /// The handle `pointer` is on, nearest first, and a filled shape the
    /// pointer is inside before anything merely near it.
    pub fn hit(&self, mode: GizmoMode, pointer: egui::Pos2) -> Option<Handle> {
        let mut nearest: Option<(f32, Handle)> = None;
        self.each_shape(mode, |handle, shape| {
            let distance = match shape {
                Shape::Line { a, b, .. } => segment_distance(pointer, a, b),
                Shape::Fill { corners, .. } if inside(pointer, &corners) => -1.0,
                Shape::Fill { corners, .. } => (0..4)
                    .map(|i| segment_distance(pointer, corners[i], corners[(i + 1) % 4]))
                    .fold(f32::INFINITY, f32::min),
            };
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
        self.each_shape(mode, |handle, shape| {
            let lit = hot == Some(handle);
            let [red, green, blue, _] = if lit { HOT } else { colour_of(handle) }.to_array();
            let colour = |alpha: f32| [red, green, blue, (alpha * 255.0).round() as u8];
            match shape {
                Shape::Line {
                    a,
                    b,
                    from,
                    to,
                    alpha,
                } => {
                    let bold = if lit { 1.4 } else { 1.0 };
                    stroke(
                        out,
                        a,
                        b,
                        from * bold,
                        to * bold,
                        colour(alpha),
                        pixels_per_point,
                    );
                }
                Shape::Fill {
                    corners,
                    fill,
                    alpha,
                } => {
                    fill_quad(out, &corners, colour(fill * alpha), pixels_per_point);
                    // An edge for the antialiasing a bare fill does not get,
                    // and so a translucent square still reads as a shape.
                    for i in 0..4 {
                        let (a, b) = (corners[i], corners[(i + 1) % 4]);
                        stroke(out, a, b, 0.6, 0.6, colour(alpha), pixels_per_point);
                    }
                }
            }
        });
    }

    /// Every piece of `mode`'s handles as a screen-space shape, in draw order:
    /// the one walk both drawing and hit-testing read.
    fn each_shape(&self, mode: GizmoMode, mut visit: impl FnMut(Handle, Shape)) {
        let origin = self.project(self.origin);
        let toward = self.camera - self.origin;
        // How far an axis's screen length is from nothing, as an opacity: an
        // axis pointing at the camera is a dot, which can be neither read nor
        // dragged along, so it fades out instead.
        let shown =
            |tip: egui::Pos2| (((tip - origin).length() / LENGTH - 0.2) / 0.15).clamp(0.0, 1.0);
        match mode {
            GizmoMode::Move => {
                for normal in 0..3 {
                    // Seen edge on, a square is a sliver, and a drag in its
                    // plane is ill-conditioned for the same reason a ring's is.
                    let facing = toward.normalize().dot(self.axes[normal]).abs();
                    let alpha = ((facing - 0.2) / 0.15).clamp(0.0, 1.0);
                    if alpha <= 0.0 {
                        continue;
                    }
                    let [u, v] = others(normal).map(|index| self.axes[index] * self.length);
                    let corner = |a: f32, b: f32| self.project(self.origin + u * a + v * b);
                    let (near, far) = PLANE;
                    let corners = [
                        corner(near, near),
                        corner(far, near),
                        corner(far, far),
                        corner(near, far),
                    ];
                    visit(
                        Handle::Plane(normal),
                        Shape::Fill {
                            corners,
                            fill: 0.3,
                            alpha,
                        },
                    );
                }
                for (index, axis) in self.axes.iter().enumerate() {
                    let tip = self.project(self.origin + *axis * self.length);
                    let alpha = shown(tip);
                    if alpha <= 0.0 {
                        continue;
                    }
                    let neck = origin + (tip - origin) * 0.8;
                    let handle = Handle::Axis(index);
                    visit(handle, Shape::line(origin, neck, 1.25, alpha));
                    visit(handle, Shape::taper(neck, tip, 5.5, alpha));
                }
                visit(Handle::Centre, Shape::square(origin, CENTRE_BOX, 1.0));
            }
            GizmoMode::Rotate => {
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
                        let (a, b) = (self.project(a), self.project(b));
                        visit(Handle::Axis(index), Shape::line(a, b, 1.25, 1.0));
                    }
                }
            }
            GizmoMode::Scale => {
                for (index, axis) in self.own.iter().enumerate() {
                    let tip = self.project(self.origin + *axis * self.length);
                    let alpha = shown(tip);
                    if alpha <= 0.0 {
                        continue;
                    }
                    let handle = Handle::Axis(index);
                    visit(handle, Shape::line(origin, tip, 1.25, alpha));
                    visit(handle, Shape::square(tip, TIP_BOX, alpha));
                }
                visit(Handle::Centre, Shape::square(origin, CENTRE_BOX, 1.0));
            }
            GizmoMode::Select => {}
        }
    }

    /// Point `step` of [`RING_SEGMENTS`] round the ring about `axis`.
    fn rim(&self, axis: Vec3, step: usize) -> Vec3 {
        let (u, v) = axis.any_orthonormal_pair();
        let angle = step as f32 / RING_SEGMENTS as f32 * std::f32::consts::TAU;
        self.origin + (u * angle.cos() + v * angle.sin()) * self.length
    }
}

/// Append a marker for an entity standing at `at` on screen: a ring for a
/// light, a diamond for anything else, lit when it is selected.
pub fn marker(
    out: &mut Vec<GizmoVertex>,
    at: egui::Pos2,
    kind: Marker,
    selected: bool,
    pixels_per_point: f32,
) {
    let colour = match (selected, kind) {
        (true, _) => HOT,
        (false, Marker::Light) => crate::editor::theme::KIND_LIGHT,
        (false, Marker::Empty) => CENTRE,
    }
    .to_array();
    let outline: Vec<egui::Pos2> = match kind {
        Marker::Light => (0..=16)
            .map(|step| {
                let angle = step as f32 / 16.0 * std::f32::consts::TAU;
                at + egui::vec2(angle.cos(), angle.sin()) * 6.0
            })
            .collect(),
        Marker::Empty => [
            (0.0, -6.0),
            (6.0, 0.0),
            (0.0, 6.0),
            (-6.0, 0.0),
            (0.0, -6.0),
        ]
        .into_iter()
        .map(|(x, y)| at + egui::vec2(x, y))
        .collect(),
    };
    for pair in outline.windows(2) {
        stroke(out, pair[0], pair[1], 1.25, 1.25, colour, pixels_per_point);
    }
}

/// One piece of a handle, in points.
#[derive(Clone, Copy)]
enum Shape {
    /// `from` points half wide at `a`, `to` at `b`.
    Line {
        a: egui::Pos2,
        b: egui::Pos2,
        from: f32,
        to: f32,
        alpha: f32,
    },
    /// A convex quad, filled at `fill` of `alpha` and edged at `alpha`.
    Fill {
        corners: [egui::Pos2; 4],
        fill: f32,
        alpha: f32,
    },
}

impl Shape {
    fn line(a: egui::Pos2, b: egui::Pos2, half: f32, alpha: f32) -> Self {
        Shape::Line {
            a,
            b,
            from: half,
            to: half,
            alpha,
        }
    }

    /// An arrowhead: `half` wide at `a`, a point at `b`.
    fn taper(a: egui::Pos2, b: egui::Pos2, half: f32, alpha: f32) -> Self {
        Shape::Line {
            a,
            b,
            from: half,
            to: 0.0,
            alpha,
        }
    }

    /// A box square to the screen, solid.
    fn square(centre: egui::Pos2, half: f32, alpha: f32) -> Self {
        let d = egui::vec2(half, half);
        let e = egui::vec2(half, -half);
        Shape::Fill {
            corners: [centre - d, centre + e, centre + d, centre - e],
            fill: 1.0,
            alpha,
        }
    }
}

/// The two axes that are not `normal`, in order.
fn others(normal: usize) -> [usize; 2] {
    [(normal + 1) % 3, (normal + 2) % 3]
}

/// What a handle is drawn in when it is not hot.
fn colour_of(handle: Handle) -> egui::Color32 {
    match handle {
        Handle::Axis(index) | Handle::Plane(index) => orrin_registry::inspect::AXIS[index],
        Handle::Centre => CENTRE,
    }
}

/// One entity a drag moves, as it was when the handle was grabbed.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    pub entity: Entity,
    pub local: LocalTransform,
    /// Its world matrix, and its parent's — identity for a root.
    pub world: Mat4,
    pub parent: Mat4,
}

/// A drag in progress: everything about the entities as they were when the
/// handle was grabbed, so each frame computes from the start rather than from
/// the last frame, and a drag back to where it began lands exactly there.
///
/// The pointer decides one change in world space — a move, a turn about the
/// gizmo's origin, a stretch along an axis or an even scale from it — and each
/// target takes that change from where it stood. With several selected, the
/// origin is their shared centre, so they move as one rigid group.
#[derive(Clone, Debug)]
pub struct Drag {
    pub handle: Handle,
    grip: Grip,
    origin: Vec3,
    axis: Vec3,
    targets: Vec<Target>,
}

/// What a drag measures the pointer against.
#[derive(Clone, Copy, Debug)]
enum Grip {
    /// A move along an axis: how far along it the grab was.
    Along(f32),
    /// A move in a plane: where the grab met it, and the axes a snap rounds
    /// the move along.
    Across {
        from: Vec3,
        snap_along: [Vec3; 3],
        snap_axes: usize,
    },
    /// A turn seen face enough on: from the origin to where the grab met the
    /// ring's plane.
    Around(Vec3),
    /// A turn seen edge on: the grab's place on screen, and the ring's screen
    /// tangent there, scaled to radians per point of travel.
    Tangent {
        view_proj: Mat4,
        screen: egui::Rect,
        from: egui::Pos2,
        tangent: egui::Vec2,
    },
    /// A stretch along an axis: how far along it the grab was.
    Stretch(f32),
    /// An even scale: where the grab was on screen. A ring's worth of travel
    /// to the right doubles the size.
    Even {
        view_proj: Mat4,
        screen: egui::Rect,
        from: egui::Pos2,
    },
}

/// What a drag has done so far, in world space about the drag's origin.
#[derive(Clone, Copy, Debug)]
enum Change {
    Move(Vec3),
    Turn(Quat),
    Stretch { axis: Vec3, factor: f32 },
    Even(f32),
}

impl Drag {
    /// Grab `handle` with the pointer's ray, to move `targets`. `None` if the
    /// grab cannot be measured from — or there is nothing to move.
    pub fn begin(
        gizmo: &Gizmo,
        mode: GizmoMode,
        handle: Handle,
        ray: (Vec3, Vec3),
        targets: Vec<Target>,
    ) -> Option<Self> {
        if targets.is_empty() {
            return None;
        }
        let origin = gizmo.origin;
        let axis = gizmo.axis_of(mode, handle);
        let on_screen = to_screen(gizmo.view_proj, gizmo.screen, ray.0 + ray.1);
        let grip = match (mode, handle) {
            (GizmoMode::Move, Handle::Axis(_)) => Grip::Along(along_axis(ray, origin, axis)?),
            (GizmoMode::Move, Handle::Plane(normal)) => {
                let [u, v] = others(normal).map(|index| gizmo.axes[index]);
                Grip::Across {
                    from: plane_hit(ray, origin, axis)?,
                    snap_along: [u, v, Vec3::ZERO],
                    snap_axes: 2,
                }
            }
            (GizmoMode::Move, Handle::Centre) => Grip::Across {
                from: plane_hit(ray, origin, axis)?,
                snap_along: gizmo.axes,
                snap_axes: 3,
            },
            (GizmoMode::Rotate, Handle::Axis(_)) => {
                let facing = (gizmo.camera - origin).normalize().dot(axis).abs();
                match plane_hit(ray, origin, axis) {
                    Some(hit) if facing >= EDGE_ON => Grip::Around(hit - origin),
                    _ => {
                        // The near half only, as drawn: seen edge on, each
                        // point of it shares its place on screen with one of
                        // the far half, whose tangent runs the other way.
                        let toward = gizmo.camera - origin;
                        let grabbed = (0..RING_SEGMENTS)
                            .map(|step| gizmo.rim(axis, step))
                            .filter(|point| (*point - origin).dot(toward) >= 0.0)
                            .min_by(|a, b| {
                                let away = |p: &Vec3| (gizmo.project(*p) - on_screen).length_sq();
                                away(a).total_cmp(&away(b))
                            })
                            .expect("a ring has points");
                        let nudge = axis.cross(grabbed - origin).normalize() * gizmo.length * 0.01;
                        let along = gizmo.project(grabbed + nudge) - gizmo.project(grabbed);
                        Grip::Tangent {
                            view_proj: gizmo.view_proj,
                            screen: gizmo.screen,
                            from: on_screen,
                            // A ring's screen radius is `LENGTH`, so this many
                            // points round it is one radian.
                            tangent: along.normalized() / LENGTH,
                        }
                    }
                }
            }
            (GizmoMode::Scale, Handle::Axis(_)) => {
                let from = along_axis(ray, origin, axis)?;
                // A grab at the origin has no length to stretch by a ratio of.
                if from.abs() < gizmo.length * 0.05 {
                    return None;
                }
                Grip::Stretch(from)
            }
            (GizmoMode::Scale, Handle::Centre) => Grip::Even {
                view_proj: gizmo.view_proj,
                screen: gizmo.screen,
                from: on_screen,
            },
            _ => return None,
        };
        Some(Self {
            handle,
            grip,
            origin,
            axis,
            targets,
        })
    }

    /// Every target's local transform at the pointer's ray, with the distance,
    /// angle or factor rounded to `snap` — metres for a move, degrees for a
    /// turn, a plain ratio for a scale — when one is given.
    pub fn update(&self, ray: (Vec3, Vec3), snap: Option<f32>) -> Vec<(Entity, LocalTransform)> {
        let change = self.change(ray, snap);
        self.targets
            .iter()
            .map(|target| {
                let local = change.map_or(target.local, |change| self.apply(change, target));
                (target.entity, local)
            })
            .collect()
    }

    /// The change the pointer has made, or `None` where it cannot be measured
    /// this frame — a ray parallel to what it is measured against — in which
    /// case the targets stay where they began.
    fn change(&self, ray: (Vec3, Vec3), snap: Option<f32>) -> Option<Change> {
        let round = |value: f32| match snap {
            Some(step) if step > 0.0 => (value / step).round() * step,
            _ => value,
        };
        let angle = match self.grip {
            Grip::Along(from) => {
                let now = along_axis(ray, self.origin, self.axis)?;
                return Some(Change::Move(self.axis * round(now - from)));
            }
            Grip::Across {
                from,
                snap_along,
                snap_axes,
            } => {
                let delta = plane_hit(ray, self.origin, self.axis)? - from;
                return Some(Change::Move(match snap {
                    // Rounded along each axis it can move along: the two of a
                    // square, all three for the centre.
                    Some(_) => snap_along[..snap_axes]
                        .iter()
                        .map(|axis| *axis * round(delta.dot(*axis)))
                        .sum(),
                    None => delta,
                }));
            }
            Grip::Stretch(from) => {
                let now = along_axis(ray, self.origin, self.axis)?;
                return Some(Change::Stretch {
                    axis: self.axis,
                    factor: round(now / from).max(MIN_FACTOR),
                });
            }
            Grip::Even {
                view_proj,
                screen,
                from,
            } => {
                let travel = to_screen(view_proj, screen, ray.0 + ray.1).x - from.x;
                return Some(Change::Even(round(1.0 + travel / LENGTH).max(MIN_FACTOR)));
            }
            Grip::Around(from) => {
                let to = plane_hit(ray, self.origin, self.axis)? - self.origin;
                self.axis.dot(from.cross(to)).atan2(from.dot(to))
            }
            Grip::Tangent {
                view_proj,
                screen,
                from,
                tangent,
            } => (to_screen(view_proj, screen, ray.0 + ray.1) - from).dot(tangent),
        };
        let degrees = round(angle.to_degrees());
        Some(Change::Turn(Quat::from_axis_angle(
            self.axis,
            degrees.to_radians(),
        )))
    }

    /// `target` after `change`, as the local transform that puts it there
    /// under its parent.
    fn apply(&self, change: Change, target: &Target) -> LocalTransform {
        let mut local = target.local;
        let (_, rotation, position) = target.world.to_scale_rotation_translation();
        let offset = position - self.origin;
        // A target at the origin itself — the only one, alone — has nowhere to
        // swing to, and recomputing its translation through the parent's
        // inverse would only add rounding to a value that did not change.
        let swung = offset.length_squared() > 1e-12;
        let placed = |at: Vec3| target.parent.inverse().transform_point3(at);
        match change {
            Change::Move(delta) => local.translation = placed(position + delta),
            Change::Turn(turn) => {
                let (_, parent, _) = target.parent.to_scale_rotation_translation();
                local.rotation = (parent.inverse() * turn * rotation).normalize();
                if swung {
                    local.translation = placed(self.origin + turn * offset);
                }
            }
            Change::Stretch { axis, factor } => {
                // The target's own axis nearest the stretch: itself, for the
                // entity whose axes the gizmo shows.
                let own = (0..3)
                    .max_by(|&a, &b| {
                        let along = |i: usize| (rotation * Vec3::AXES[i]).dot(axis).abs();
                        along(a).total_cmp(&along(b))
                    })
                    .expect("three axes");
                local.scale[own] *= factor;
                if swung {
                    local.translation = placed(position + axis * offset.dot(axis) * (factor - 1.0));
                }
            }
            Change::Even(factor) => {
                local.scale *= factor;
                if swung {
                    local.translation = placed(self.origin + offset * factor);
                }
            }
        }
        local
    }
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

/// Whether `point` is inside the convex quad `corners`, wound either way.
fn inside(point: egui::Pos2, corners: &[egui::Pos2; 4]) -> bool {
    let side = |i: usize| {
        let (a, b) = (corners[i], corners[(i + 1) % 4]);
        (b - a).x * (point - a).y - (b - a).y * (point - a).x
    };
    let sides = [side(0), side(1), side(2), side(3)];
    sides.iter().all(|&s| s >= 0.0) || sides.iter().all(|&s| s <= 0.0)
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

/// A convex quad, solid: its `edge` puts every fragment well inside, so the
/// shader covers it fully, and [`stroke`]d edges supply the antialiasing.
fn fill_quad(
    out: &mut Vec<GizmoVertex>,
    corners: &[egui::Pos2; 4],
    color: [u8; 4],
    pixels_per_point: f32,
) {
    let vertex = |corner: egui::Pos2| GizmoVertex {
        position: (corner.to_vec2() * pixels_per_point).into(),
        edge: [0.0, 1.0e4],
        color,
    };
    let [a, b, c, d] = corners.map(vertex);
    out.extend([a, b, c, a, c, d]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::pick;
    use crate::scene::Transform;

    impl Drag {
        /// [`begin`](Drag::begin) on one entity.
        fn begin_one(
            gizmo: &Gizmo,
            mode: GizmoMode,
            handle: Handle,
            ray: (Vec3, Vec3),
            local: LocalTransform,
            parent: Mat4,
            world: Mat4,
        ) -> Option<Self> {
            let target = Target {
                entity: Entity::default(),
                local,
                world,
                parent,
            };
            Drag::begin(gizmo, mode, handle, ray, vec![target])
        }

        fn update_one(&self, ray: (Vec3, Vec3), snap: Option<f32>) -> LocalTransform {
            self.update(ray, snap)[0].1
        }
    }

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
        assert_eq!(gizmo.hit(GizmoMode::Move, middle), Some(Handle::Axis(0)));
        let up = gizmo.project(Vec3::Y * gizmo.length * 0.6);
        assert_eq!(gizmo.hit(GizmoMode::Move, up), Some(Handle::Axis(1)));
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
        assert_eq!(gizmo.hit(GizmoMode::Rotate, rim), Some(Handle::Axis(1)));
        let outside = gizmo.project(Vec3::ZERO) + egui::vec2(LENGTH * 2.0, 0.0);
        assert_eq!(gizmo.hit(GizmoMode::Rotate, outside), None);
    }

    #[test]
    fn local_axes_follow_the_entity() {
        let camera = Camera::default();
        let world = Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let local = Gizmo::new(&camera, screen(), world, GizmoSpace::Local).unwrap();
        assert!(
            close(local.axis_of(GizmoMode::Move, Handle::Axis(0)), Vec3::NEG_Z),
            "{}",
            local.axis_of(GizmoMode::Move, Handle::Axis(0))
        );
        let global = Gizmo::new(&camera, screen(), world, GizmoSpace::World).unwrap();
        assert!(close(
            global.axis_of(GizmoMode::Move, Handle::Axis(0)),
            Vec3::X
        ));
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
        let drag = Drag::begin_one(
            gizmo,
            GizmoMode::Move,
            Handle::Axis(0),
            ray_at(camera, grab),
            local,
            parent,
            world,
        )
        .expect("the shaft was grabbed");
        let offset = to - gizmo.origin;
        let release = gizmo.project(gizmo.origin + Vec3::X * gizmo.length * 0.5 + offset);
        drag.update_one(ray_at(camera, release), snap)
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
        let drag = Drag::begin_one(
            &gizmo,
            GizmoMode::Rotate,
            Handle::Axis(1),
            ray_at(camera, rim(toward)),
            local,
            parent,
            world,
        )
        .expect("the ring was grabbed");
        let turned = Quat::from_rotation_y(degrees.to_radians()) * toward;
        drag.update_one(ray_at(camera, rim(turned)), snap)
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
        let drag = Drag::begin_one(
            &gizmo,
            GizmoMode::Rotate,
            Handle::Axis(0),
            ray_at(&camera, from),
            LocalTransform::default(),
            Mat4::IDENTITY,
            Mat4::IDENTITY,
        )
        .expect("the ring was grabbed");
        let to = from + egui::vec2(0.0, -20.0);
        let turned = drag.update_one(ray_at(&camera, to), None);

        let (axis, angle) = turned.rotation.to_axis_angle();
        assert!(angle > 0.05, "it turned: {angle}");
        assert!(axis.dot(Vec3::X).abs() > 0.999, "about X: {axis}");
        let followed = gizmo.project(turned.rotation * grabbed);
        assert!(
            (followed - to).length() < (from - to).length(),
            "{followed:?}"
        );
    }

    /// Grab `handle` where `from` projects and let go where `to` does, on an
    /// entity with no parent.
    fn grab(
        camera: &Camera,
        gizmo: &Gizmo,
        mode: GizmoMode,
        handle: Handle,
        (from, to): (Vec3, Vec3),
        local: LocalTransform,
        snap: Option<f32>,
    ) -> LocalTransform {
        let world = local.matrix();
        let ray = ray_at(camera, gizmo.project(from));
        let drag = Drag::begin_one(gizmo, mode, handle, ray, local, Mat4::IDENTITY, world)
            .expect("the handle was grabbed");
        drag.update_one(ray_at(camera, gizmo.project(to)), snap)
    }

    /// The middle of the square for the plane `normal` is the normal of.
    fn square(gizmo: &Gizmo, normal: usize) -> Vec3 {
        let [u, v] = others(normal);
        let middle = (PLANE.0 + PLANE.1) * 0.5;
        gizmo.origin + (gizmo.axes[u] + gizmo.axes[v]) * gizmo.length * middle
    }

    #[test]
    fn a_plane_handle_is_grabbed_inside_its_square() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let ground = gizmo.project(square(&gizmo, 1));
        assert_eq!(gizmo.hit(GizmoMode::Move, ground), Some(Handle::Plane(1)));
    }

    /// The default camera sees the YZ plane edge on, where its square is a
    /// sliver and a drag in it would be ill-conditioned.
    #[test]
    fn a_plane_handle_seen_edge_on_is_neither_drawn_nor_grabbed() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let sliver = gizmo.project(square(&gizmo, 0));
        assert_ne!(gizmo.hit(GizmoMode::Move, sliver), Some(Handle::Plane(0)));
    }

    #[test]
    fn dragging_a_plane_handle_moves_in_its_plane() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let from = square(&gizmo, 1);
        let moved = grab(
            &camera,
            &gizmo,
            GizmoMode::Move,
            Handle::Plane(1),
            (from, from + Vec3::new(1.0, 0.0, -0.5)),
            LocalTransform::default(),
            None,
        );
        assert!(
            close(moved.translation, Vec3::new(1.0, 0.0, -0.5)),
            "{}",
            moved.translation
        );
    }

    #[test]
    fn a_plane_move_snaps_each_of_its_axes() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let from = square(&gizmo, 1);
        let moved = grab(
            &camera,
            &gizmo,
            GizmoMode::Move,
            Handle::Plane(1),
            (from, from + Vec3::new(0.6, 0.0, -0.4)),
            LocalTransform::default(),
            Some(0.25),
        );
        assert!(
            close(moved.translation, Vec3::new(0.5, 0.0, -0.5)),
            "{}",
            moved.translation
        );
    }

    /// The centre moves with the pointer, square to the camera: never toward
    /// or away from it.
    #[test]
    fn the_centre_handle_moves_in_the_plane_of_the_screen() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let middle = gizmo.project(Vec3::ZERO);
        assert_eq!(gizmo.hit(GizmoMode::Move, middle), Some(Handle::Centre));

        let local = LocalTransform::default();
        let drag = Drag::begin_one(
            &gizmo,
            GizmoMode::Move,
            Handle::Centre,
            ray_at(&camera, middle),
            local,
            Mat4::IDENTITY,
            local.matrix(),
        )
        .expect("the centre was grabbed");
        let moved = drag.update_one(ray_at(&camera, middle + egui::vec2(40.0, -25.0)), None);
        let forward = (camera.target - camera.position).normalize();
        assert!(moved.translation.length() > 0.1, "{}", moved.translation);
        assert!(
            moved.translation.dot(forward).abs() < 1e-3,
            "{}",
            moved.translation
        );
    }

    #[test]
    fn dragging_a_scale_tip_stretches_that_axis_only() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let tip = |reach: f32| Vec3::X * gizmo.length * reach;
        assert_eq!(
            gizmo.hit(GizmoMode::Scale, gizmo.project(tip(0.95))),
            Some(Handle::Axis(0))
        );
        let scaled = grab(
            &camera,
            &gizmo,
            GizmoMode::Scale,
            Handle::Axis(0),
            (tip(0.9), tip(1.8)),
            LocalTransform::default(),
            None,
        );
        assert!(
            close(scaled.scale, Vec3::new(2.0, 1.0, 1.0)),
            "{}",
            scaled.scale
        );
        assert!(close(scaled.translation, Vec3::ZERO));
    }

    /// A stretch along a world axis of a turned entity would be a shear, which
    /// a transform cannot hold, so scale handles are the entity's own axes in
    /// either space.
    #[test]
    fn scale_follows_the_entity_s_own_axes_even_in_world_space() {
        let camera = Camera::default();
        let local = LocalTransform::new(Transform {
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            ..Transform::default()
        });
        let gizmo = gizmo(&camera, local.matrix());
        let own_x = local.rotation * Vec3::X;
        let scaled = grab(
            &camera,
            &gizmo,
            GizmoMode::Scale,
            Handle::Axis(0),
            (own_x * gizmo.length * 0.9, own_x * gizmo.length * 1.8),
            local,
            None,
        );
        assert!(
            close(scaled.scale, Vec3::new(2.0, 1.0, 1.0)),
            "{}",
            scaled.scale
        );
    }

    #[test]
    fn the_centre_handle_scales_evenly() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let middle = gizmo.project(Vec3::ZERO);
        assert_eq!(gizmo.hit(GizmoMode::Scale, middle), Some(Handle::Centre));
        let local = LocalTransform::default();
        let drag = Drag::begin_one(
            &gizmo,
            GizmoMode::Scale,
            Handle::Centre,
            ray_at(&camera, middle),
            local,
            Mat4::IDENTITY,
            local.matrix(),
        )
        .expect("the centre was grabbed");
        let scaled = drag.update_one(
            ray_at(&camera, middle + egui::vec2(LENGTH * 0.5, 0.0)),
            None,
        );
        assert!(close(scaled.scale, Vec3::splat(1.5)), "{}", scaled.scale);
    }

    #[test]
    fn a_scale_snaps_its_factor_to_the_step() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let tip = |reach: f32| Vec3::X * gizmo.length * reach;
        let scaled = grab(
            &camera,
            &gizmo,
            GizmoMode::Scale,
            Handle::Axis(0),
            (tip(0.9), tip(1.65)),
            LocalTransform::default(),
            Some(0.1),
        );
        assert!(
            close(scaled.scale, Vec3::new(1.8, 1.0, 1.0)),
            "{}",
            scaled.scale
        );
    }

    /// A root entity standing at `position`, turned by `rotation`.
    fn target(index: u32, position: Vec3, rotation: Quat) -> Target {
        let local = LocalTransform::new(Transform {
            translation: position,
            rotation,
            ..Transform::default()
        });
        Target {
            entity: Entity {
                index,
                generation: 0,
            },
            local,
            world: local.matrix(),
            parent: Mat4::IDENTITY,
        }
    }

    /// Two entities either side of the origin, which is their centre.
    fn pair() -> Vec<Target> {
        vec![
            target(1, Vec3::new(-1.0, 0.0, 0.0), Quat::IDENTITY),
            target(2, Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY),
        ]
    }

    fn drag_many(
        camera: &Camera,
        gizmo: &Gizmo,
        mode: GizmoMode,
        handle: Handle,
        (from, to): (egui::Pos2, egui::Pos2),
        targets: Vec<Target>,
    ) -> Vec<LocalTransform> {
        Drag::begin(gizmo, mode, handle, ray_at(camera, from), targets)
            .expect("the handle was grabbed")
            .update(ray_at(camera, to), None)
            .into_iter()
            .map(|(_, local)| local)
            .collect()
    }

    #[test]
    fn moving_several_moves_each_by_the_same_amount() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let shaft = |x: f32| gizmo.project(Vec3::X * (gizmo.length * 0.5 + x));
        let moved = drag_many(
            &camera,
            &gizmo,
            GizmoMode::Move,
            Handle::Axis(0),
            (shaft(0.0), shaft(1.3)),
            pair(),
        );
        assert!(
            close(moved[0].translation, Vec3::new(0.3, 0.0, 0.0)),
            "{}",
            moved[0].translation
        );
        assert!(
            close(moved[1].translation, Vec3::new(2.3, 0.0, 0.0)),
            "{}",
            moved[1].translation
        );
    }

    #[test]
    fn turning_several_swings_them_about_their_shared_centre() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let toward = Vec3::new(camera.position.x, 0.0, camera.position.z).normalize();
        let rim = |direction: Vec3| gizmo.project(direction * gizmo.length);
        let quarter = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let turned = drag_many(
            &camera,
            &gizmo,
            GizmoMode::Rotate,
            Handle::Axis(1),
            (rim(toward), rim(quarter * toward)),
            pair(),
        );
        assert!(
            close(turned[0].translation, Vec3::new(0.0, 0.0, 1.0)),
            "{}",
            turned[0].translation
        );
        assert!(
            close(turned[1].translation, Vec3::new(0.0, 0.0, -1.0)),
            "{}",
            turned[1].translation
        );
        assert!(turned[0].rotation.angle_between(quarter) < 1e-3);
        assert!(turned[1].rotation.angle_between(quarter) < 1e-3);
    }

    #[test]
    fn scaling_several_evenly_spreads_them_from_their_centre() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let middle = gizmo.project(Vec3::ZERO);
        let scaled = drag_many(
            &camera,
            &gizmo,
            GizmoMode::Scale,
            Handle::Centre,
            (middle, middle + egui::vec2(LENGTH * 0.5, 0.0)),
            pair(),
        );
        assert!(
            close(scaled[0].translation, Vec3::new(-1.5, 0.0, 0.0)),
            "{}",
            scaled[0].translation
        );
        assert!(
            close(scaled[1].translation, Vec3::new(1.5, 0.0, 0.0)),
            "{}",
            scaled[1].translation
        );
        assert!(
            close(scaled[0].scale, Vec3::splat(1.5)) && close(scaled[1].scale, Vec3::splat(1.5))
        );
    }

    /// The stretch is along the gizmo's X; the second entity is turned so its
    /// own Z lies along that, and Z is what grows.
    #[test]
    fn a_stretch_scales_each_entity_along_its_own_nearest_axis() {
        let camera = Camera::default();
        let gizmo = gizmo(&camera, at(Vec3::ZERO));
        let tip = |reach: f32| gizmo.project(Vec3::X * gizmo.length * reach);
        let turned = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let targets = vec![
            target(1, Vec3::new(-1.0, 0.0, 0.0), Quat::IDENTITY),
            target(2, Vec3::new(1.0, 0.0, 0.0), turned),
        ];
        let stretched = drag_many(
            &camera,
            &gizmo,
            GizmoMode::Scale,
            Handle::Axis(0),
            (tip(0.9), tip(1.8)),
            targets,
        );
        assert!(
            close(stretched[0].scale, Vec3::new(2.0, 1.0, 1.0)),
            "{}",
            stretched[0].scale
        );
        assert!(
            close(stretched[1].scale, Vec3::new(1.0, 1.0, 2.0)),
            "{}",
            stretched[1].scale
        );
        assert!(close(stretched[0].translation, Vec3::new(-2.0, 0.0, 0.0)));
        assert!(close(stretched[1].translation, Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn a_light_s_marker_wears_the_light_colour_and_is_lit_when_selected() {
        let colours = |kind: Marker, selected: bool| -> Vec<[u8; 4]> {
            let mut out = Vec::new();
            marker(&mut out, egui::pos2(100.0, 100.0), kind, selected, 1.0);
            assert!(!out.is_empty() && out.len() % 3 == 0);
            out.iter().map(|v| v.color).collect()
        };
        let light = crate::editor::theme::KIND_LIGHT.to_array();
        assert!(colours(Marker::Light, false).contains(&light));
        assert!(!colours(Marker::Empty, false).contains(&light));
        assert!(colours(Marker::Light, true).contains(&HOT.to_array()));
        assert!(colours(Marker::Empty, true).contains(&HOT.to_array()));
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
        gizmo.strokes(GizmoMode::Move, Some(Handle::Axis(0)), 2.0, &mut pixels);
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

        let mut scale = Vec::new();
        gizmo.strokes(GizmoMode::Scale, Some(Handle::Centre), 1.0, &mut scale);
        assert!(
            colours(&scale).contains(&HOT.to_array()),
            "the centre box is lit"
        );

        let mut none = Vec::new();
        gizmo.strokes(GizmoMode::Select, None, 1.0, &mut none);
        assert!(none.is_empty());
    }
}
