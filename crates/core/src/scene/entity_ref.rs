//! An entity named from somewhere the world cannot see into — a Behaviour's
//! `Entity` field — in the form a [`Value`](orrin_registry::Value) carries it:
//! an [`EntityId`].
//!
//! A field can point at an entity that has no id yet, because ids are given at
//! the first save or edit and a script may have spawned it a moment ago. Naming
//! that one `NIL` would lose it on the next edit to any *other* field of the
//! same Behaviour, since an edit writes the whole bag back. So a live entity
//! with no id is named by a stand-in: its index and generation, then eight
//! bytes drawn once per process. It resolves only in the session that made it
//! and only to that exact slot and generation, which a despawn retires.
//!
//! A handle whose entity has gone is named by a tombstone instead — the same
//! layout with a fixed tail — which resolves to nothing anywhere. It is what a
//! save writes for a stale handle, so the load can say the reference was lost
//! rather than read it as an ordinary null, and it is deterministic, so saving
//! twice still gives the same bytes. A save gives every live entity an id
//! before reading anything, so a live stand-in reaches a file only through a
//! prefab made around an id-less entity outside it.
//!
//! Both tails carry UUID variant bits that [`EntityId::new`] never produces, so
//! neither can equal a real id.

use std::sync::OnceLock;

use orrin_ecs::{Entity, World};
use orrin_registry::EntityId;

/// What `entity` is called in a [`Value`](orrin_registry::Value): its id, `NIL`
/// for the null handle, or a stand-in.
pub fn id_of(world: &World, entity: Entity) -> EntityId {
    if entity.index == 0 {
        return EntityId::NIL;
    }
    if !world.is_alive(entity) {
        return slot_id(entity, &TOMBSTONE);
    }
    match world.get::<EntityId>(entity) {
        Some(id) => *id,
        None => slot_id(entity, session()),
    }
}

fn slot_id(entity: Entity, tail: &[u8; 8]) -> EntityId {
    let mut bytes = [0; 16];
    bytes[..4].copy_from_slice(&entity.index.to_le_bytes());
    bytes[4..8].copy_from_slice(&entity.generation.to_le_bytes());
    bytes[8..].copy_from_slice(tail);
    EntityId::from_bytes(bytes)
}

/// The live entity `id` names, if there is one.
///
/// A real id is found by a scan: nothing indexes entities by id, and this runs
/// per reference in a bag being written and per entity field the inspector has
/// on screen, never per entity per frame.
pub fn entity_of(world: &World, id: EntityId) -> Option<Entity> {
    let bytes = id.to_bytes();
    if id.is_nil() || bytes[8..] == TOMBSTONE {
        return None;
    }
    if bytes[8..] == *session() {
        let word =
            |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
        let entity = Entity {
            index: word(0),
            generation: word(4),
        };
        return world.is_alive(entity).then_some(entity);
    }
    world
        .entities()
        .find(|&entity| world.get::<EntityId>(entity).is_some_and(|own| *own == id))
}

/// The tail of a tombstone. `0xe0` in byte 8 is the UUID variant reserved for
/// future definition.
const TOMBSTONE: [u8; 8] = [0xe0, 0, 0, 0, 0, 0, 0, 0];

/// The stand-ins' tail, random per process: the half of a stand-in that makes
/// one from another session name nothing here. Byte 8 is forced into the
/// Microsoft-reserved UUID variant, which a v4 id never has.
fn session() -> &'static [u8; 8] {
    static SESSION: OnceLock<[u8; 8]> = OnceLock::new();
    SESSION.get_or_init(|| {
        let random = EntityId::new().to_bytes();
        let mut tail: [u8; 8] = std::array::from_fn(|i| random[8 + i]);
        tail[0] = 0xc0 | (tail[0] & 0x1f);
        tail
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_entity_with_an_id_is_named_by_it() {
        let mut world = World::new();
        let entity = world.spawn();
        let id = EntityId::new();
        world.insert(entity, id);
        assert_eq!(id_of(&world, entity), id);
        assert_eq!(entity_of(&world, id), Some(entity));
    }

    /// A script's entity has no id, and still has to survive an edit to the
    /// field beside it.
    #[test]
    fn an_entity_without_an_id_round_trips_through_a_stand_in() {
        let mut world = World::new();
        let entity = world.spawn();
        let named = id_of(&world, entity);
        assert!(!named.is_nil());
        assert_eq!(entity_of(&world, named), Some(entity));
        assert_eq!(id_of(&world, entity), named, "a stand-in is stable");
    }

    /// The generation is in the stand-in, so a slot reused after a despawn is
    /// not the entity the field pointed at.
    #[test]
    fn a_stand_in_for_a_despawned_entity_names_nothing_even_after_its_slot_is_reused() {
        let mut world = World::new();
        let entity = world.spawn();
        let named = id_of(&world, entity);
        world.despawn(entity);
        assert_eq!(entity_of(&world, named), None);

        let reused = world.spawn();
        assert_eq!(reused.index, entity.index);
        assert_eq!(entity_of(&world, named), None);
    }

    /// A stale handle — with or without an id — is a tombstone: stable, so a
    /// save of it is deterministic, distinct from null, so a load can report
    /// it, and resolving nowhere, even in a world whose slot is alive.
    #[test]
    fn a_stale_handle_is_a_tombstone_that_names_nothing_in_any_world() {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, EntityId::new());
        world.despawn(entity);
        let dead = id_of(&world, entity);
        assert!(!dead.is_nil());
        assert_eq!(id_of(&world, entity), dead);

        let mut other = World::new();
        let alive = other.spawn();
        assert_eq!(
            (alive.index, alive.generation),
            (entity.index, entity.generation)
        );
        assert_eq!(entity_of(&other, dead), None);
    }

    #[test]
    fn the_null_handle_is_nil_and_nil_names_nothing() {
        let world = World::new();
        let null = Entity {
            index: 0,
            generation: 0,
        };
        assert_eq!(id_of(&world, null), EntityId::NIL);
        assert_eq!(entity_of(&world, EntityId::NIL), None);
    }

    /// Another session's stand-in, or an id from another scene, is a reference
    /// to nothing here — never to whatever now sits in the slot it mentions.
    #[test]
    fn an_id_nothing_has_names_nothing() {
        let mut world = World::new();
        let entity = world.spawn();
        let mut foreign = id_of(&world, entity).to_bytes();
        foreign[15] ^= 0xff;
        assert_eq!(entity_of(&world, EntityId::from_bytes(foreign)), None);
        assert_eq!(entity_of(&world, EntityId::new()), None);
    }
}
