//! Jolt's contact callbacks, turned into enter and exit per pair of entities.
//!
//! Jolt reports a contact as it is added, while it persists, and as it is
//! removed — per pair of sub-shapes. So pairs are counted rather than diffed: a
//! pair enters at its first sub-shape contact and exits at its last.
//!
//! **Sleep is not leaving.** When a body falls asleep against one that is not
//! awake either — a crate settling on a static floor — Jolt drops the contact
//! and reports it removed, though nothing moved. A removal with neither body
//! awake therefore leaves the pair *dormant* instead: still touching as far as
//! scripts know. Whichever body wakes first, Jolt re-reports the contact on
//! that step if it still holds, and a dormant pair it does not re-report has
//! genuinely parted and exits then.

use std::sync::{Arc, Mutex};

use glam::Vec3;
use jolt::{
    Body, BodyId, BodyInterface, ContactListener, ContactManifold, ContactSettings, SubShapeIdPair,
};
use orrin_ecs::{Entity, FxHashMap};

use super::{CollisionEvent, CollisionEventKind, Contact, from_jolt, pair_key};

pub(super) enum Change {
    Touch {
        a: BodyId,
        b: BodyId,
        point: Vec3,
        normal: Vec3,
        persisted: bool,
    },
    Part {
        a: BodyId,
        b: BodyId,
    },
}

/// The listener Jolt calls during a step. It only records: the callbacks run
/// inside Jolt, where the body interface may not be used.
pub(super) struct Recorder(pub(super) Arc<Mutex<Vec<Change>>>);

impl Recorder {
    fn touch(&self, a: &Body, b: &Body, manifold: &ContactManifold<'_>, persisted: bool) {
        let point = if manifold.relative_points_on1.is_empty() {
            manifold.base_offset
        } else {
            manifold.world_point_on1(0)
        };
        self.0.lock().unwrap().push(Change::Touch {
            a: a.id(),
            b: b.id(),
            point: from_jolt(point),
            normal: from_jolt(manifold.world_space_normal),
            persisted,
        });
    }
}

impl ContactListener for Recorder {
    fn on_contact_added(
        &self,
        a: &Body,
        b: &Body,
        manifold: &ContactManifold<'_>,
        _: &mut ContactSettings,
    ) {
        self.touch(a, b, manifold, false);
    }

    fn on_contact_persisted(
        &self,
        a: &Body,
        b: &Body,
        manifold: &ContactManifold<'_>,
        _: &mut ContactSettings,
    ) {
        self.touch(a, b, manifold, true);
    }

    fn on_contact_removed(&self, pair: &SubShapeIdPair) {
        self.0.lock().unwrap().push(Change::Part {
            a: pair.body1,
            b: pair.body2,
        });
    }
}

struct Touching {
    contact: Contact,
    count: u32,
    bodies: (BodyId, BodyId),
    trigger: bool,
    dormant: bool,
}

/// Which pairs are touching, and through how many sub-shape contacts.
#[derive(Default)]
pub(super) struct Pairs {
    touching: FxHashMap<(Entity, Entity), Touching>,
}

impl Pairs {
    /// Apply a step's changes. A body the map does not know was destroyed
    /// since it was reported, and its pairs already left through [`leave`].
    ///
    /// [`leave`]: Self::leave
    pub(super) fn apply(
        &mut self,
        changes: &mut Vec<Change>,
        entities: &FxHashMap<BodyId, Entity>,
        bodies: &BodyInterface<'_>,
        events: &mut Vec<CollisionEvent>,
    ) {
        for change in changes.drain(..) {
            match change {
                Change::Touch {
                    a: a_id,
                    b: b_id,
                    point,
                    normal,
                    persisted,
                } => {
                    let (Some(&a), Some(&b)) = (entities.get(&a_id), entities.get(&b_id)) else {
                        continue;
                    };
                    let key = pair_key(a, b);
                    let contact = Contact {
                        point,
                        normal: if key.0 == a { normal } else { -normal },
                    };
                    let touching = self.touching.entry(key).or_insert(Touching {
                        contact,
                        count: 0,
                        bodies: (a_id, b_id),
                        trigger: bodies.is_sensor(a_id) || bodies.is_sensor(b_id),
                        dormant: false,
                    });
                    touching.contact = contact;
                    // Re-reported: whichever body woke, the pair still holds.
                    let woke = std::mem::take(&mut touching.dormant);
                    if !persisted {
                        touching.count += 1;
                        if touching.count == 1 && !woke {
                            events.push(CollisionEvent::new(
                                CollisionEventKind::Enter,
                                key,
                                contact,
                                touching.trigger,
                            ));
                        }
                    }
                }
                Change::Part { a: a_id, b: b_id } => {
                    let (Some(&a), Some(&b)) = (entities.get(&a_id), entities.get(&b_id)) else {
                        continue;
                    };
                    let key = pair_key(a, b);
                    let Some(touching) = self.touching.get_mut(&key) else {
                        continue;
                    };
                    touching.count = touching.count.saturating_sub(1);
                    if touching.count == 0 && !bodies.is_active(a_id) && !bodies.is_active(b_id) {
                        touching.dormant = true;
                    } else if touching.count == 0 {
                        let (contact, trigger) = (touching.contact, touching.trigger);
                        self.touching.remove(&key);
                        events.push(CollisionEvent::new(
                            CollisionEventKind::Exit,
                            key,
                            contact,
                            trigger,
                        ));
                    }
                }
            }
        }
    }

    /// End of a step: a dormant pair with a body awake again that Jolt did not
    /// re-report this step — which would have cleared it — has parted.
    pub(super) fn settle(&mut self, bodies: &BodyInterface<'_>, events: &mut Vec<CollisionEvent>) {
        self.touching.retain(|&pair, touching| {
            let (a, b) = touching.bodies;
            let parted = touching.dormant && (bodies.is_active(a) || bodies.is_active(b));
            if parted {
                events.push(CollisionEvent::new(
                    CollisionEventKind::Exit,
                    pair,
                    touching.contact,
                    touching.trigger,
                ));
            }
            !parted
        });
    }

    /// Every pair `entity` is in leaves at once, with the last contact it had:
    /// the body is going away, and Jolt will not say so.
    pub(super) fn leave(&mut self, entity: Entity, events: &mut Vec<CollisionEvent>) {
        self.touching.retain(|&pair, touching| {
            let involved = pair.0 == entity || pair.1 == entity;
            if involved {
                events.push(CollisionEvent::new(
                    CollisionEventKind::Exit,
                    pair,
                    touching.contact,
                    touching.trigger,
                ));
            }
            !involved
        });
    }

    /// Every pair leaves: the whole world is being rebuilt.
    pub(super) fn leave_all(&mut self, events: &mut Vec<CollisionEvent>) {
        for (pair, touching) in self.touching.drain() {
            events.push(CollisionEvent::new(
                CollisionEventKind::Exit,
                pair,
                touching.contact,
                touching.trigger,
            ));
        }
    }
}
