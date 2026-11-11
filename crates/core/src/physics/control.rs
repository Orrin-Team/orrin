//! What game code may ask of the simulation between steps: push a body, read
//! or set its velocity, cast a ray or a shape, or ask what overlaps one.
//!
//! A push is asked for from one of two places, and a [`ForceMode::Force`] or
//! [`ForceMode::Acceleration`] is read as *held for as long as that place
//! lasts*, applied as the impulse that is, `force * held`. Between frames — a
//! script's `OnUpdate`, which runs after the steps at the display's rate — it
//! is held for the frame, so a second of pushing is the same push at 60 fps as
//! at 144. From [`run_with`](super::run_with)'s callback — `OnFixedUpdate` — it
//! is held for the one step the callback precedes, as Unity reads it from
//! `FixedUpdate`. Either way a second of pushing is a second of force; the
//! choice is whether a script wants to react once a step.
//!
//! A body spawned this frame has no Jolt body until the next physics run — and
//! from a script, no `RigidBody` either, since a script's components are added
//! after its tick. So a push on a live entity with no body yet is kept for the
//! next step, which runs after both have arrived, and is applied there if the
//! entity has become a dynamic body and dropped if it has not. "Spawn, add a
//! rigid body, throw" in one `OnStart` works; the cost is that `true` for such
//! an entity means the push was taken, not that it will land. A push on a body
//! that exists and is not dynamic is refused at once.
//!
//! The queries all filter alike — a layer mask, and triggers passed through
//! unless asked for — and all read Jolt as the last run left it. So a collider
//! added since has no body to find, and one moved since is found where the run
//! put it: a static one keeps its old place until a step carries it, because a
//! moved static becomes kinematic and a kinematic body moves only by stepping.
//! The shapes a query sweeps or tests are one unit sphere and one unit box,
//! built with the simulation and scaled per query, so asking costs no shape.

use std::ops::ControlFlow;

use glam::{Quat, Vec3};
use jolt::{BodyInterface, QueryFilter, Shape};
use orrin_ecs::{Entity, World};

use super::{Kind, Motion, PhysicsState, Simulation, from_jolt, to_jolt, to_jolt_quat};
use crate::scene::PHYSICS_LAYERS;

/// How a push is read. Lock-step with C# `Orrin.ForceMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ForceMode {
    /// Newtons (or newton-metres), held for the frame, or for the step when
    /// asked before one.
    Force,
    /// N·s, all at once.
    Impulse,
    /// m/s, regardless of mass.
    VelocityChange,
    /// m/s², regardless of mass, held like a `Force`.
    Acceleration,
}

impl ForceMode {
    pub fn from_u32(mode: u32) -> Option<Self> {
        [
            Self::Force,
            Self::Impulse,
            Self::VelocityChange,
            Self::Acceleration,
        ]
        .get(mode as usize)
        .copied()
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Push {
    Linear(Vec3, ForceMode, f32),
    Angular(Vec3, ForceMode, f32),
    Velocity(Option<Vec3>, Option<Vec3>),
}

impl Push {
    pub(super) fn apply(self, bodies: &BodyInterface<'_>, id: jolt::BodyId) {
        match self {
            Self::Linear(force, mode, delta) => match mode {
                ForceMode::Force => bodies.add_impulse(id, to_jolt(force * delta)),
                ForceMode::Impulse => bodies.add_impulse(id, to_jolt(force)),
                ForceMode::VelocityChange => bodies.add_linear_velocity(id, to_jolt(force)),
                ForceMode::Acceleration => bodies.add_linear_velocity(id, to_jolt(force * delta)),
            },
            Self::Angular(torque, mode, delta) => match mode {
                ForceMode::Force => bodies.add_angular_impulse(id, to_jolt(torque * delta)),
                ForceMode::Impulse => bodies.add_angular_impulse(id, to_jolt(torque)),
                ForceMode::VelocityChange | ForceMode::Acceleration => {
                    let change = if mode == ForceMode::Acceleration {
                        torque * delta
                    } else {
                        torque
                    };
                    let spin = from_jolt(bodies.angular_velocity(id)) + change;
                    bodies.set_angular_velocity(id, to_jolt(spin));
                }
            },
            Self::Velocity(linear, angular) => {
                if let Some(linear) = linear {
                    bodies.set_linear_velocity(id, to_jolt(linear));
                }
                if let Some(angular) = angular {
                    bodies.set_angular_velocity(id, to_jolt(angular));
                }
            }
        }
        bodies.activate_body(id);
    }
}

/// Apply `push` to `entity` now, or at the next step if it has no body yet.
/// False when it has a body that is not dynamic, or is not alive.
fn push(world: &World, entity: Entity, push: impl FnOnce(f32) -> Push) -> bool {
    let Some(mut state) = world.get_resource_mut::<PhysicsState>() else {
        return false;
    };
    let delta = state.held_for;
    let body = state.simulation.as_ref().map(|simulation| {
        let tracked = simulation.bodies.get(&entity)?;
        Some((tracked.id, tracked.motion()))
    });
    match body.flatten() {
        Some((id, Motion::Dynamic)) => {
            let simulation = state.simulation.as_ref().expect("found above");
            push(delta).apply(&simulation.system.body_interface(), id);
            true
        }
        Some(_) => false,
        None if world.is_alive(entity) => {
            state.pending.push((entity, push(delta)));
            true
        }
        None => false,
    }
}

/// Push `entity` along `force`, read as `mode` says. False when it is not a
/// dynamic body.
pub fn add_force(world: &World, entity: Entity, force: Vec3, mode: ForceMode) -> bool {
    force.is_finite() && push(world, entity, |delta| Push::Linear(force, mode, delta))
}

/// Turn `entity` about `torque`, read as `mode` says.
pub fn add_torque(world: &World, entity: Entity, torque: Vec3, mode: ForceMode) -> bool {
    torque.is_finite() && push(world, entity, |delta| Push::Angular(torque, mode, delta))
}

/// Replace `entity`'s linear velocity, angular velocity, or both.
pub fn set_velocity(
    world: &World,
    entity: Entity,
    linear: Option<Vec3>,
    angular: Option<Vec3>,
) -> bool {
    let finite = linear.is_none_or(|v| v.is_finite()) && angular.is_none_or(|v| v.is_finite());
    finite && push(world, entity, |_| Push::Velocity(linear, angular))
}

/// `entity`'s linear and angular velocity as of the last step, in m/s and
/// rad/s. `None` when it has no body.
pub fn velocity(world: &World, entity: Entity) -> Option<(Vec3, Vec3)> {
    let state = world.get_resource::<PhysicsState>()?;
    let simulation = state.simulation.as_ref()?;
    let tracked = simulation.bodies.get(&entity)?;
    let bodies = simulation.system.body_interface();
    Some((
        from_jolt(bodies.linear_velocity(tracked.id)),
        from_jolt(bodies.angular_velocity(tracked.id)),
    ))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaycastHit {
    pub entity: Entity,
    pub point: Vec3,
    /// The surface's, at the point.
    pub normal: Vec3,
    /// From the origin, in metres.
    pub distance: f32,
}

/// The farthest a ray looks. Jolt measures the ray by its direction's length,
/// so an infinite one has to be made finite somewhere, and a kilometre past
/// anything a scene is built at is where.
const MAX_RAY: f32 = 10_000.0;

/// The first collider along the ray from `origin` in `direction`, within
/// `max_distance`, on a layer whose bit is set in `layers`. Triggers are passed
/// through unless `hit_triggers`: a trigger is a volume to be in, not a surface
/// to see. Bodies are where the last step left them.
pub fn raycast(
    world: &World,
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    layers: u32,
    hit_triggers: bool,
) -> Option<RaycastHit> {
    let direction = direction.try_normalize()?;
    if !origin.is_finite() || max_distance.is_nan() || max_distance <= 0.0 {
        return None;
    }
    let length = max_distance.min(MAX_RAY);
    let state = world.get_resource::<PhysicsState>()?;
    let simulation = state.simulation.as_ref()?;
    let accept = |id| accepts(simulation, id, layers, hit_triggers);
    let filter = QueryFilter::new().with_body_filter(&accept);
    let hit = simulation
        .system
        .cast_ray(to_jolt(origin), to_jolt(direction * length), &filter)?;
    Some(RaycastHit {
        entity: *simulation.entities.get(&hit.body_id)?,
        point: from_jolt(hit.point),
        normal: from_jolt(hit.normal),
        distance: hit.fraction * length,
    })
}

/// Apply what was asked of bodies that did not exist yet, now that they do.
pub(super) fn apply_pending(simulation: &Simulation, pending: &mut Vec<(Entity, Push)>) {
    let bodies = simulation.system.body_interface();
    for (entity, push) in pending.drain(..) {
        if let Some(tracked) = simulation.bodies.get(&entity)
            && tracked.spec.kind == Kind::Dynamic
        {
            push.apply(&bodies, tracked.id);
        }
    }
}

/// Whether a query on `layers` sees body `id`.
fn accepts(simulation: &Simulation, id: jolt::BodyId, layers: u32, hit_triggers: bool) -> bool {
    simulation
        .entities
        .get(&id)
        .and_then(|entity| simulation.bodies.get(entity))
        .is_some_and(|tracked| {
            let collider = tracked.spec.collider;
            let layer = collider.layer.min(PHYSICS_LAYERS - 1);
            layers & (1 << layer) != 0 && (hit_triggers || !collider.is_trigger)
        })
}

/// A volume a query tests or sweeps, placed in the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QueryShape {
    Sphere {
        center: Vec3,
        radius: f32,
    },
    Box {
        center: Vec3,
        half_extents: Vec3,
        rotation: Quat,
    },
}

/// The unit shapes every query is scaled from.
pub(super) struct QueryShapes {
    sphere: Shape,
    cuboid: Shape,
}

impl QueryShapes {
    pub(super) fn new() -> jolt::Result<Self> {
        Ok(Self {
            sphere: Shape::sphere(1.0)?,
            // No convex radius, which Jolt would scale along with the box and
            // so round a thin query's corners off by its own width.
            cuboid: Shape::cuboid_with_convex_radius(jolt::Vec3::new(1.0, 1.0, 1.0), 0.0)?,
        })
    }
}

impl QueryShape {
    /// The unit shape, its scale and where it stands, or `None` for a shape
    /// with no volume or a placement that is not finite.
    fn placed(self, shapes: &QueryShapes) -> Option<(&Shape, jolt::Vec3, jolt::Mat44)> {
        let (shape, scale, center, rotation) = match self {
            Self::Sphere { center, radius } => {
                (&shapes.sphere, Vec3::splat(radius), center, Quat::IDENTITY)
            }
            Self::Box {
                center,
                half_extents,
                rotation,
            } => {
                let length = rotation.length();
                if !(length.is_finite() && length > 0.0) {
                    return None;
                }
                (&shapes.cuboid, half_extents, center, rotation / length)
            }
        };
        if !(scale.is_finite() && scale.min_element() > 0.0 && center.is_finite()) {
            return None;
        }
        let at = jolt::Mat44::from_rotation_translation(to_jolt_quat(rotation), to_jolt(center));
        Some((shape, to_jolt(scale), at))
    }
}

/// Every entity whose collider `shape` overlaps, on a layer whose bit is set in
/// `layers`, triggers only if `hit_triggers`. Written into `out`, which is
/// cleared first, once each and in entity order: Jolt reports a mesh once per
/// triangle it touches and in no order, and neither is what a script asked.
pub fn overlap(
    world: &World,
    shape: QueryShape,
    layers: u32,
    hit_triggers: bool,
    out: &mut Vec<Entity>,
) {
    out.clear();
    let Some(state) = world.get_resource::<PhysicsState>() else {
        return;
    };
    let Some(simulation) = state.simulation.as_ref() else {
        return;
    };
    let Some((unit, scale, at)) = shape.placed(&simulation.queries) else {
        return;
    };
    let accept = |id| accepts(simulation, id, layers, hit_triggers);
    let filter = QueryFilter::new().with_body_filter(&accept);
    simulation
        .system
        .collide_shape(unit, scale, &at, 0.0, &filter, |hit| {
            if let Some(&entity) = simulation.entities.get(&hit.body_id) {
                out.push(entity);
            }
            ControlFlow::Continue(())
        });
    out.sort_unstable_by_key(|entity| (entity.index, entity.generation));
    out.dedup();
}

/// The first collider `shape` meets swept along `direction` for at most
/// `max_distance`, filtered as [`overlap`] filters. `distance` is how far the
/// shape travelled, `point` where it touched and `normal` the surface's there.
/// A shape that starts inside a collider hits it at distance zero.
pub fn shape_cast(
    world: &World,
    shape: QueryShape,
    direction: Vec3,
    max_distance: f32,
    layers: u32,
    hit_triggers: bool,
) -> Option<RaycastHit> {
    let direction = direction.try_normalize()?;
    if max_distance.is_nan() || max_distance <= 0.0 {
        return None;
    }
    let length = max_distance.min(MAX_RAY);
    let state = world.get_resource::<PhysicsState>()?;
    let simulation = state.simulation.as_ref()?;
    let (unit, scale, at) = shape.placed(&simulation.queries)?;
    let accept = |id| accepts(simulation, id, layers, hit_triggers);
    let filter = QueryFilter::new().with_body_filter(&accept);
    let hit = simulation.system.cast_shape_closest(
        unit,
        scale,
        &at,
        to_jolt(direction * length),
        &filter,
    )?;
    Some(RaycastHit {
        entity: *simulation.entities.get(&hit.body_id)?,
        point: from_jolt(hit.contact_point_on2),
        normal: from_jolt(hit.normal()),
        distance: hit.fraction * length,
    })
}
