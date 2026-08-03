//! Collision detection: BVH broadphase → shape narrowphase → enter/exit events,
//! with positional (MTV) overlap resolution for solid pairs.
//!
//! Contact normals are unit length and point from `a` toward `b`, where `(a, b)`
//! is the canonical order from `pair_key`, and `depth` is the penetration along
//! that normal (`>= 0`). `run` only writes events into `CollisionState`;
//! delivering them to C# is the script tick's job.

mod bvh;
mod narrowphase;

use glam::{Mat3, Vec3};

use orrin_ecs::{Entity, FxHashMap, World};

use crate::scene::{Collider, ColliderShape, LocalTransform, WorldTransform};

/// Broadphase bounds and mesh bounds are the same box; it lives in
/// [`crate::geom`] so extraction can cull without depending on collision.
pub use crate::geom::Aabb;
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
    /// Hashed with [`FxHasher`](orrin_ecs::FxHasher) rather than SipHash: the
    /// key is a pair of engine-generated handles, and every overlapping pair
    /// hashes it at least twice a frame (insert, then the diff's lookup).
    touching: FxHashMap<(Entity, Entity), Contact>,
    pub events: Vec<CollisionEvent>,
    /// Lives across frames only to keep its buffers; the tree in it is rebuilt
    /// from scratch every frame and never read from one frame to the next.
    broadphase: Bvh,
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

fn world_shape(transform: &WorldTransform, collider: &Collider) -> WorldShape {
    // The linear part carries rotation and scale together, and under a hierarchy
    // it can also carry shear — so both arms read it directly rather than
    // decomposing back into a rotation and a scale, which shear does not survive.
    let linear = Mat3::from_mat4(transform.0);
    let center = transform.translation();

    match collider.shape {
        ColliderShape::Sphere { radius } => {
            // Non-uniform scale would make this an ellipsoid; the longest column
            // is the largest distance a unit vector can be stretched to, so this
            // is a true sphere that still contains it. Contacts can fire early,
            // never go missing.
            let stretch = linear
                .x_axis
                .length()
                .max(linear.y_axis.length())
                .max(linear.z_axis.length());
            WorldShape::Sphere {
                center,
                radius: radius * stretch,
            }
        }
        ColliderShape::Box { half_extents } => {
            // World AABB of the transformed box is abs(linear) * half: per world
            // axis the farthest corner picks the sign of every term, which is the
            // element-wise abs.
            let abs = Mat3::from_cols(
                linear.x_axis.abs(),
                linear.y_axis.abs(),
                linear.z_axis.abs(),
            );
            let world_half = abs * half_extents;

            WorldShape::Box(Aabb {
                min: center - world_half,
                max: center + world_half,
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
    previous: &FxHashMap<(Entity, Entity), Contact>,
    current: &FxHashMap<(Entity, Entity), Contact>,
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
    // Taken out rather than borrowed for the whole function: everything below
    // reaches back into the world, and the tree is put back at the end.
    let mut broadphase = {
        let mut state = world.resource_mut::<CollisionState>();
        state.events.clear();
        std::mem::take(&mut state.broadphase)
    };

    struct Body {
        entity: Entity,
        shape: WorldShape,
        is_trigger: bool,
    }

    let mut bodies: Vec<Body> = Vec::new();
    world
        .query::<(&WorldTransform, &Collider)>()
        .for_each(|entity, (transform, collider)| {
            bodies.push(Body {
                entity,
                shape: world_shape(transform, collider),
                is_trigger: collider.is_trigger,
            });
        });

    let bounds: Vec<Aabb> = bodies.iter().map(|body| body.shape.bounds()).collect();
    let mut candidates: Vec<(u32, u32)> = Vec::new();
    broadphase.rebuild(&bounds);
    broadphase.query_pairs(&mut candidates);

    // Contacts are stored re-oriented to the canonical pair order, so the diff
    // and the resolver never have to guess which way the normal points.
    let mut current: FxHashMap<(Entity, Entity), Contact> = FxHashMap::default();
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
            Contact {
                normal: -contact.normal,
                ..contact
            }
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
    let CollisionState {
        touching,
        events,
        broadphase: stored,
    } = &mut *state;
    if !(touching.is_empty() && current.is_empty()) {
        diff_pairs(touching, &current, events);
    }
    *touching = current;
    *stored = broadphase;
}
