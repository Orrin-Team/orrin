//! What can be done to an instance as a whole: push its changes into the
//! prefab, throw them away, or stop it being an instance.
//!
//! Where an instance stands — its root's transform and parent — is its own,
//! and neither is applied nor reverted with the rest. Without that, applying
//! one placed door would move every other door to where it is, and reverting
//! one would send it back to the prefab's origin.

use orrin_ecs::{Entity, FxBuildHasher, World};
use orrin_registry::{AssetId, EntityId, Registry};

use super::create::{children_of, document_of, label, subtree, write};
use super::refresh::{rebuild, snapshot};
use super::{Changes, Ids, Prefab, PrefabInstance, Prefabs, apply_edits, membership, record_of};
use crate::scene::persist;
use crate::scene::registry::{PREFAB, TRANSFORM};

type FxHashSet<T> = std::collections::HashSet<T, FxBuildHasher>;

/// The instance `entity` is part of: its root and its prefab.
pub fn instance_of(world: &World, entity: Entity) -> Option<(Entity, AssetId)> {
    membership(world, entity).map(|member| (member.root, member.prefab))
}

/// Write the instance `entity` is part of back into its prefab's file, so
/// every instance gets what this one changed — and keeps what it overrides
/// itself. A child the scene added under a member joins the prefab, and a
/// member this instance despawned leaves it.
///
/// Returns what the other instances could not take.
pub fn apply(
    world: &mut World,
    registry: &Registry,
    entity: Entity,
) -> Result<Vec<String>, String> {
    let (root, id) =
        instance_of(world, entity).ok_or("the entity is not part of a prefab instance")?;
    let (prefab, file) = {
        let prefabs = world
            .get_resource::<Prefabs>()
            .ok_or("there is no prefab index")?;
        let prefab = prefabs
            .get(id)
            .ok_or("the instance's prefab has left the index")?;
        if !prefab.on_disk() {
            return Err(format!(
                "{} is gone from disk, so there is nothing to apply to",
                prefab.path
            ));
        }
        let dir = prefabs.dir().ok_or("there is no assets directory")?;
        (prefab.clone(), dir.join(&prefab.path))
    };

    // A member dragged elsewhere would take its new parent into the prefab,
    // which is not part of the tree being written.
    let root_id = *world
        .get::<EntityId>(root)
        .expect("an instance root has an id");
    let ids = Ids {
        root: root_id,
        prefab_root: prefab.root(),
    };
    let by_id = persist::ids_of(world);
    let tree: FxHashSet<Entity> = subtree(&children_of(world), root).into_iter().collect();
    for base in prefab.entities() {
        if let Some(&member) = by_id.get(&ids.member(base.id))
            && !tree.contains(&member)
        {
            return Err(format!(
                "{} was moved out of its instance; move it back, or unpack the instance, before \
                 applying",
                label(world, member)
            ));
        }
    }

    let mut document = document_of(world, registry, root, prefab.root());
    let placement = prefab.entity(prefab.root()).and_then(|base| {
        base.components
            .iter()
            .find(|(component, _)| *component == TRANSFORM)
            .cloned()
    });
    if let Some(entity) = document
        .entities
        .iter_mut()
        .find(|entity| entity.id == prefab.root())
    {
        entity
            .components
            .retain(|(component, _)| *component != TRANSFORM);
        entity.components.extend(placement);
    }

    let updated = Prefab::from_document(id, prefab.path.clone(), document.clone())?;
    write(&file, &prefab.path, &document, None)?;
    let old = world
        .resource_mut::<Prefabs>()
        .replace(updated)
        .expect("read above");
    Ok(apply_edits(world, registry, vec![old]))
}

/// Give the instance `entity` is part of back to its prefab: every override,
/// removed component and despawned member, except where it stands. What the
/// scene hung under its members stays.
pub fn revert_all(
    world: &mut World,
    registry: &Registry,
    entity: Entity,
) -> Result<Vec<String>, String> {
    let (root, id) =
        instance_of(world, entity).ok_or("the entity is not part of a prefab instance")?;
    let prefab = world
        .get_resource::<Prefabs>()
        .and_then(|prefabs| prefabs.get(id).cloned())
        .ok_or("the instance's prefab has left the index")?;
    let holder =
        snapshot(world, registry, root, &prefab).ok_or("the instance root has no identity")?;
    let record = record_of(&holder).ok_or("the instance did not fold to a record")?;
    let (source, changes) = Changes::from_record(record)?;
    let kept = Changes {
        fields: changes
            .fields
            .into_iter()
            .filter(|o| o.entity == prefab.root() && o.component == TRANSFORM)
            .collect(),
        ..Changes::default()
    };

    let mut reverted = holder.clone();
    for (component, value) in &mut reverted.components {
        if *component == PREFAB {
            *value = kept.to_record(source);
        }
    }
    Ok(rebuild(world, registry, &prefab, &holder, &reverted))
}

/// Make the instance `entity` is part of plain entities. They keep their ids
/// and components; the next save writes them out in full. Instances inside it
/// stay instances.
pub fn unpack(world: &mut World, entity: Entity) {
    if let Some((root, _)) = instance_of(world, entity) {
        world.remove::<PrefabInstance>(root);
    }
}

#[cfg(test)]
mod tests {
    use orrin_ecs::{Entity, World};
    use orrin_registry::EntityId;

    use super::super::index::tests::{DOOR_ID, door_id, door_text, registry};
    use super::super::*;
    use crate::scene::{
        Assets, LocalTransform, MaterialHandle, MeshHandle, Name, Parent, Spin, Tag, reparent, save,
    };

    struct Dir(std::path::PathBuf);

    impl Dir {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("orrin-actions-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("door.fprefab"), door_text("handle", 1.0)).unwrap();
            std::fs::write(
                dir.join("door.fprefab.meta"),
                format!("id = \"{DOOR_ID}\"\n"),
            )
            .unwrap();
            Self(dir)
        }

        fn door(&self) -> String {
            std::fs::read_to_string(self.0.join("door.fprefab")).unwrap()
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn world_in(dir: &Dir) -> World {
        let mut world = World::new();
        let mut assets = Assets::new();
        assets.insert_mesh("cube", MeshHandle(3));
        assets.insert_material("gold", MaterialHandle(7));
        world.insert_resource(assets);
        world.insert_resource(Prefabs::scan(&dir.0).0);
        world
    }

    /// The member named `name` under `root`.
    fn member(world: &World, root: Entity, name: &str) -> Entity {
        let mut stack = vec![root];
        while let Some(entity) = stack.pop() {
            if world.get::<Name>(entity).is_some_and(|n| n.0 == name) {
                return entity;
            }
            stack.extend(
                world
                    .entities()
                    .filter(|&e| world.get::<Parent>(e).is_some_and(|p| p.get() == entity)),
            );
        }
        panic!("no {name} under that root")
    }

    fn spawn_door(world: &mut World, at: f32) -> Entity {
        let (door, _) = instantiate(world, &registry(), door_id()).unwrap();
        world.get_mut::<LocalTransform>(door).unwrap().translation.x = at;
        door
    }

    #[test]
    fn applying_an_instance_rewrites_its_prefab_and_reaches_the_others() {
        let dir = Dir::new("apply");
        let mut world = world_in(&dir);
        let a = spawn_door(&mut world, 0.0);
        let b = spawn_door(&mut world, 10.0);
        world.get_mut::<Tag>(member(&world, b, "Handle")).unwrap().0 = "b's own".to_owned();

        let panel = member(&world, a, "Panel");
        world
            .get_mut::<LocalTransform>(panel)
            .unwrap()
            .translation
            .y = 2.0;
        world.get_mut::<Tag>(member(&world, a, "Handle")).unwrap().0 = "brass".to_owned();
        let sticker = world.spawn();
        world.insert(sticker, Name::new("Sticker"));
        world.insert(sticker, LocalTransform::default());
        reparent(&mut world, sticker, Some(panel), false).unwrap();

        apply(&mut world, &registry(), a).unwrap();

        assert!(
            dir.door().contains("translation = (0.0, 2.0, 0.0)"),
            "{}",
            dir.door()
        );
        assert!(
            dir.door().contains("\"Sticker\""),
            "an added child joins the prefab"
        );
        let b_panel = member(&world, b, "Panel");
        assert_eq!(
            world.get::<LocalTransform>(b_panel).unwrap().translation.y,
            2.0
        );
        assert_eq!(
            world.get::<Name>(member(&world, b, "Sticker")).unwrap().0,
            "Sticker"
        );
        assert_eq!(
            world.get::<Tag>(member(&world, b, "Handle")).unwrap().0,
            "b's own",
            "another instance's own override survives"
        );
        assert_eq!(
            world.get::<LocalTransform>(b).unwrap().translation.x,
            10.0,
            "where an instance stands is never applied"
        );
        let overrides = member_overrides(&world, &registry(), panel).unwrap();
        assert!(overrides.fields.is_empty(), "{:?}", overrides.fields);
    }

    #[test]
    fn a_member_moved_out_of_its_instance_stops_an_apply() {
        let dir = Dir::new("moved-out");
        let mut world = world_in(&dir);
        let door = spawn_door(&mut world, 0.0);
        let handle = member(&world, door, "Handle");
        reparent(&mut world, handle, None, false).unwrap();

        let err = apply(&mut world, &registry(), door).unwrap_err();
        assert!(err.contains("Handle"), "{err}");
        assert_eq!(dir.door(), door_text("handle", 1.0), "nothing was written");
    }

    #[test]
    fn reverting_an_instance_restores_everything_but_where_it_stands() {
        let dir = Dir::new("revert");
        let mut world = world_in(&dir);
        let door = spawn_door(&mut world, 7.0);
        let panel = member(&world, door, "Panel");
        world
            .get_mut::<LocalTransform>(panel)
            .unwrap()
            .translation
            .y = 2.0;
        world.remove::<Spin>(panel);
        let handle = member(&world, door, "Handle");
        crate::scene::despawn_recursive(&mut world, handle);
        let sticker = world.spawn();
        world.insert(sticker, Name::new("Sticker"));
        reparent(&mut world, sticker, Some(door), false).unwrap();

        revert_all(&mut world, &registry(), door).unwrap();

        assert_eq!(
            world.get::<LocalTransform>(panel).unwrap().translation.y,
            1.0
        );
        assert!(world.get::<Spin>(panel).is_some());
        assert_eq!(
            world.get::<Tag>(member(&world, door, "Handle")).unwrap().0,
            "handle"
        );
        assert_eq!(
            world.get::<LocalTransform>(door).unwrap().translation.x,
            7.0
        );
        assert!(
            world.is_alive(sticker),
            "what the scene added is the scene's"
        );
    }

    #[test]
    fn an_unpacked_instance_saves_as_plain_entities() {
        let dir = Dir::new("unpack");
        let mut world = world_in(&dir);
        let door = spawn_door(&mut world, 0.0);

        unpack(&mut world, door);
        assert!(world.get::<PrefabInstance>(door).is_none());
        let text = save(&mut world, &registry());
        assert!(!text.contains("orrin.prefab"), "{text}");
        assert_eq!(text.matches("\nentity ").count(), 3);
        assert!(world.get::<EntityId>(door).is_some());
    }

    #[test]
    fn a_member_names_its_instance() {
        let dir = Dir::new("member-of");
        let mut world = world_in(&dir);
        let door = spawn_door(&mut world, 0.0);
        let panel = member(&world, door, "Panel");
        assert_eq!(instance_of(&world, panel), Some((door, door_id())));
        let loose = world.spawn();
        world.insert(loose, EntityId::new());
        assert_eq!(instance_of(&world, loose), None);
    }
}
