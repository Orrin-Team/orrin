//! Collision detection: BVH broadphase → shape narrowphase → enter/exit events,
//! with positional (MTV) overlap resolution for solid pairs.
//!
//! Contact normals are unit length and point from `a` toward `b`, where `(a, b)`
//! is the canonical order from `pair_key`, and `depth` is the penetration along
//! that normal (`>= 0`). `run` only writes events into `CollisionState`;
//! delivering them to C# is the script tick's job.

mod bvh;
mod narrowphase;

use std::collections::HashMap;

use glam::{Mat3, Vec3};

use orrin_ecs::{Entity, World};

use crate::scene::{Collider, ColliderShape, LocalTransform};

pub use bvh::Bvh;

#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub point: Vec3,
    pub normal: Vec3,
    pub depth: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollisionEventKind {
    Enter,
    Exit,
}

#[derive(Clone, Copy, Debug)]
pub struct CollisionEvent {
    pub kind: CollisionEventKind,
    pub a: Entity,
    pub b: Entity,
    pub point: Vec3,
    pub normal: Vec3,
}

#[derive(Default)]
pub struct CollisionState {
    touching: HashMap<(Entity, Entity), Contact>,
    pub events: Vec<CollisionEvent>,
}

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn overlaps(&self, other: &Aabb) -> bool {
        (self.min.cmple(other.max) & other.min.cmple(self.max)).all()
    }

    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
}

#[derive(Clone, Copy, Debug)]
pub enum WorldShape {
    Box(Aabb),
    Sphere { center: Vec3, radius: f32 },
}

impl WorldShape {
    pub fn bounds(&self) -> Aabb {
        match *self {
            WorldShape::Box(aabb) => aabb,
            WorldShape::Sphere { center, radius } => Aabb {
                min: center - Vec3::splat(radius),
                max: center + Vec3::splat(radius),
            },
        }
    }
}

fn world_shape(transform: &LocalTransform, collider: &Collider) -> WorldShape {
    match collider.shape {
        ColliderShape::Sphere { radius } => {
            // Non-uniform scale would make this an ellipsoid; the largest scale
            // component keeps a true sphere that still contains it, so contacts
            // can fire early but never go missing.
            WorldShape::Sphere {
                center: transform.translation,
                radius: radius * transform.scale.max_element().abs(),
            }
        }
        ColliderShape::Box { half_extents } => {
            // World AABB of the rotated box is abs(R) * half: per world axis the
            // farthest corner picks the sign of every term, which is the
            // element-wise abs.
            let half = half_extents * transform.scale;
            let r = Mat3::from_quat(transform.rotation);
            let abs_r = Mat3::from_cols(r.x_axis.abs(), r.y_axis.abs(), r.z_axis.abs());
            let world_half = abs_r * half;

            WorldShape::Box(Aabb {
                min: transform.translation - world_half,
                max: transform.translation + world_half,
            })
        }
    }
}

/// Canonical ordering for an unordered entity pair, so `(a, b)` and `(b, a)`
/// hash to the same key. Stored contacts are oriented a → b in this order.
fn pair_key(a: Entity, b: Entity) -> (Entity, Entity) {
    if (a.index, a.generation) <= (b.index, b.generation) {
        (a, b)
    } else {
        (b, a)
    }
}

fn diff_pairs(
    previous: &HashMap<(Entity, Entity), Contact>,
    current: &HashMap<(Entity, Entity), Contact>,
    events: &mut Vec<CollisionEvent>,
) {
    for (key, contact) in current {
        if !previous.contains_key(key) {
            events.push(CollisionEvent {
                kind: CollisionEventKind::Enter,
                a: key.0,
                b: key.1,
                point: contact.point,
                normal: contact.normal,
            });
        }
    }

    // Exit pairs have no contact this frame, so the event reuses last frame's
    // point/normal — the same trade-off Unity makes for OnCollisionExit.
    for (key, contact) in previous {
        if !current.contains_key(key) {
            events.push(CollisionEvent {
                kind: CollisionEventKind::Exit,
                a: key.0,
                b: key.1,
                point: contact.point,
                normal: contact.normal,
            });
        }
    }
}

/// Detect collisions and produce this frame's events plus positional
/// corrections. Solid–solid pairs are pushed apart immediately but still count
/// as touching this frame, so a clean separation surfaces as an Exit event next
/// frame — matching how impulse engines report it.
pub fn run(world: &mut World) {
    world.resource_mut::<CollisionState>().events.clear();

    struct Body {
        entity: Entity,
        shape: WorldShape,
        is_trigger: bool,
    }

    let mut bodies: Vec<Body> = Vec::new();
    world
        .query::<(&LocalTransform, &Collider)>()
        .for_each(|entity, (transform, collider)| {
            bodies.push(Body {
                entity,
                shape: world_shape(transform, collider),
                is_trigger: collider.is_trigger,
            });
        });

    let mut candidates: Vec<(u32, u32)> = Vec::new();
    if bodies.len() >= 2 {
        let bounds: Vec<Aabb> = bodies.iter().map(|body| body.shape.bounds()).collect();
        Bvh::build(&bounds).query_pairs(&mut candidates);
    }

    // Contacts are stored re-oriented to the canonical pair order, so the diff
    // and the resolver never have to guess which way the normal points.
    let mut current: HashMap<(Entity, Entity), Contact> = HashMap::new();
    let mut corrections: Vec<(Entity, Vec3)> = Vec::new();
    for &(i, j) in &candidates {
        let (a, b) = (&bodies[i as usize], &bodies[j as usize]);
        let Some(contact) = narrowphase::test(&a.shape, &b.shape) else {
            continue;
        };

        let key = pair_key(a.entity, b.entity);
        let contact = if key.0 == a.entity {
            contact
        } else {
            Contact { normal: -contact.normal, ..contact }
        };
        current.insert(key, contact);

        if !a.is_trigger && !b.is_trigger {
            let (offset_a, offset_b) = narrowphase::resolve_offsets(&contact);
            corrections.push((key.0, offset_a));
            corrections.push((key.1, offset_b));
        }
    }

    // One entity at a time: `get_mut` borrows the whole LocalTransform storage,
    // so holding two at once would panic the RefCell.
    for (entity, offset) in corrections {
        if let Some(mut transform) = world.get_mut::<LocalTransform>(entity) {
            transform.translation += offset;
        }
    }

    let mut state = world.resource_mut::<CollisionState>();
    let CollisionState { touching, events } = &mut *state;
    if !(touching.is_empty() && current.is_empty()) {
        diff_pairs(touching, &current, events);
    }
    *touching = current;
}
