//! Turning entities in a scene into a new prefab.
//!
//! The selection stays where it is and becomes the prefab's first instance.
//! Nothing is respawned: the prefab's ids are chosen so the entities already
//! carry the ids an instance of it derives (see the module above), so the only
//! change to the world is the [`PrefabInstance`] on the root — and a selection
//! of several trees gaining a common root, since a prefab is one tree.

use std::path::{Path, PathBuf};

use orrin_ecs::{Entity, FxBuildHasher, FxHashMap, World};
use orrin_registry::{AssetId, EntityId, Registry, SceneDocument, write_document};

use super::index::EXTENSION;
use super::{Ids, Prefab, PrefabInstance, Prefabs, fold, membership, remap};
use crate::scene::persist;
use crate::scene::registry::PARENT;
use crate::scene::{LocalTransform, Name, Parent, propagate_transforms, reparent};

type FxHashSet<T> = std::collections::HashSet<T, FxBuildHasher>;

/// Save `selection` as a prefab at `rel` under the index's directory, index
/// it, and make the selection its first instance. Returns that instance's root
/// and the new prefab's id.
///
/// An entity selected under another selected one is taken with its ancestor.
/// Several trees, or a tree whose root is already an instance, are parented
/// under a new root named after the file, which keeps each one where it is in
/// the world.
pub fn create(
    world: &mut World,
    registry: &Registry,
    selection: &[Entity],
    rel: &str,
) -> Result<(Entity, AssetId), String> {
    let dir = world
        .get_resource::<Prefabs>()
        .and_then(|prefabs| prefabs.dir().map(Path::to_owned))
        .ok_or("there is no assets directory to save a prefab into")?;
    let file = dir.join(rel);
    if file.exists() {
        return Err(format!("{rel} already exists; choose another name"));
    }

    let tops = tops(world, selection);
    if tops.is_empty() {
        return Err("nothing is selected".to_owned());
    }
    let children = children_of(world);
    for &top in &tops {
        let tree: FxHashSet<Entity> = subtree(&children, top).into_iter().collect();
        for &entity in &tree {
            if let Some(member) = membership(world, entity)
                && !tree.contains(&member.root)
            {
                let name = label(world, entity);
                return Err(format!(
                    "{name} is part of an instance of another prefab, and a prefab cannot take \
                     part of one; select the whole instance"
                ));
            }
        }
    }

    let root = match tops[..] {
        [top] if !world.has::<PrefabInstance>(top) => top,
        _ => wrap(world, &tops, stem(rel))?,
    };

    let tree = subtree(&children_of(world), root);
    for &entity in &tree {
        if !world.has::<EntityId>(entity) {
            world.insert(entity, EntityId::new());
        }
    }
    let mut entities: Vec<_> = tree
        .iter()
        .filter_map(|&entity| persist::scene_entity(world, registry, entity))
        .collect();
    fold(world, &mut entities);

    let root_id = *world.get::<EntityId>(root).expect("given one above");
    let ids = Ids {
        root: root_id,
        prefab_root: EntityId::new(),
    };
    let members: FxHashSet<EntityId> = entities.iter().map(|entity| entity.id).collect();
    for entity in &mut entities {
        if entity.id == root_id {
            entity
                .components
                .retain(|(component, _)| *component != PARENT);
        }
        entity.id = ids.local(entity.id);
        for (_, value) in &mut entity.components {
            remap(value, &mut |id| {
                members.contains(&id).then(|| ids.local(id))
            });
        }
    }
    let document = SceneDocument { entities };

    let id = AssetId::new();
    write(&file, rel, &document, id)?;
    let prefab = Prefab::from_document(id, rel.to_owned(), document)?;
    world
        .get_resource_mut::<Prefabs>()
        .expect("read above")
        .insert(prefab);
    world.insert(
        root,
        PrefabInstance {
            prefab: id,
            unapplied: Vec::new(),
        },
    );
    Ok((root, id))
}

/// A path for a new prefab called `name` that no file under `dir` has yet:
/// `name.fprefab`, else `name 2.fprefab`, and so on.
pub fn unused_path(dir: &Path, name: &str) -> String {
    let name: String = name
        .chars()
        .map(|c| if r#"/\:*?"<>|"#.contains(c) { '_' } else { c })
        .collect();
    let name = match name.trim() {
        "" => "Prefab",
        trimmed => trimmed,
    };
    (1..)
        .map(|n| match n {
            1 => format!("{name}.{EXTENSION}"),
            n => format!("{name} {n}.{EXTENSION}"),
        })
        .find(|rel| !dir.join(rel).exists())
        .expect("an unbounded range finds one")
}

/// The selected entities that no other selected entity is an ancestor of, in
/// selection order.
fn tops(world: &World, selection: &[Entity]) -> Vec<Entity> {
    let selected: FxHashSet<Entity> = selection
        .iter()
        .copied()
        .filter(|&e| world.is_alive(e))
        .collect();
    let mut tops = Vec::new();
    for &entity in selection {
        if !selected.contains(&entity) || tops.contains(&entity) {
            continue;
        }
        let mut cursor = world.get::<Parent>(entity).map(|p| p.get());
        let mut covered = false;
        // Bounded: a cycle the hierarchy has not broken yet would otherwise spin.
        for _ in 0..4096 {
            let Some(parent) = cursor else { break };
            if selected.contains(&parent) {
                covered = true;
                break;
            }
            cursor = world.get::<Parent>(parent).map(|p| p.get());
        }
        if !covered {
            tops.push(entity);
        }
    }
    tops
}

/// A new root for `tops`, under their common parent if they share one, with
/// each tree kept where it is in the world.
fn wrap(world: &mut World, tops: &[Entity], name: &str) -> Result<Entity, String> {
    let parents: Vec<Option<Entity>> = tops
        .iter()
        .map(|&top| world.get::<Parent>(top).map(|p| p.get()))
        .collect();
    let common = parents
        .iter()
        .all(|parent| *parent == parents[0])
        .then_some(parents[0])
        .flatten();

    let root = world.spawn();
    world.insert(root, Name::new(name));
    world.insert(root, LocalTransform::default());
    reparent(world, root, common, false).map_err(|e| e.to_string())?;
    // `keep_world` reads the new parent's world transform, which the root does
    // not have until it has been propagated once.
    propagate_transforms(world);
    for &top in tops {
        reparent(world, top, Some(root), true).map_err(|e| e.to_string())?;
    }
    propagate_transforms(world);
    Ok(root)
}

pub(super) fn children_of(world: &World) -> FxHashMap<Entity, Vec<Entity>> {
    let mut children: FxHashMap<Entity, Vec<Entity>> = FxHashMap::default();
    for entity in world.entities() {
        if let Some(parent) = world.get::<Parent>(entity) {
            children.entry(parent.get()).or_default().push(entity);
        }
    }
    children
}

pub(super) fn subtree(children: &FxHashMap<Entity, Vec<Entity>>, root: Entity) -> Vec<Entity> {
    let mut seen = FxHashSet::from_iter([root]);
    let mut tree = vec![root];
    let mut i = 0;
    while i < tree.len() {
        for &kid in children.get(&tree[i]).into_iter().flatten() {
            // A cycle the hierarchy has not broken yet would otherwise loop.
            if seen.insert(kid) {
                tree.push(kid);
            }
        }
        i += 1;
    }
    tree
}

fn label(world: &World, entity: Entity) -> String {
    world.get::<Name>(entity).map_or_else(
        || format!("entity {}", entity.index()),
        |name| name.0.clone(),
    )
}

fn stem(rel: &str) -> &str {
    let file = rel.rsplit('/').next().unwrap_or(rel);
    file.strip_suffix(EXTENSION)
        .and_then(|stem| stem.strip_suffix('.'))
        .unwrap_or(file)
}

/// The prefab file and its sidecar. The sidecar goes second, so a failure
/// part-way leaves a prefab the next scan gives an id to rather than a sidecar
/// pointing at nothing.
fn write(file: &Path, rel: &str, document: &SceneDocument, id: AssetId) -> Result<(), String> {
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    }
    let mut text = String::new();
    write_document(&mut text, document);
    std::fs::write(file, text).map_err(|e| format!("could not write {rel}: {e}"))?;
    let mut meta = file.as_os_str().to_owned();
    meta.push(".meta");
    std::fs::write(PathBuf::from(meta), format!("id = \"{id}\"\n"))
        .map_err(|e| format!("could not write {rel}.meta: {e}"))
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::super::index::tests::{DOOR_ID, door_id, door_text};
    use super::*;
    use crate::scene::{
        Assets, MaterialHandle, MeshHandle, Tag, Transform, WorldTransform, load,
        register_components, save,
    };

    struct Dir(PathBuf);

    impl Dir {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("orrin-create-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(dir.join("props")).unwrap();
            std::fs::write(dir.join("props/door.fprefab"), door_text("handle", 1.0)).unwrap();
            std::fs::write(
                dir.join("props/door.fprefab.meta"),
                format!("id = \"{DOOR_ID}\"\n"),
            )
            .unwrap();
            Self(dir)
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The door fixture's game component, so loading it reports nothing.
    #[derive(Default, orrin_registry::Reflect)]
    struct Follow {
        target: EntityId,
    }

    fn registry() -> Registry {
        let mut registry = Registry::new();
        register_components(&mut registry);
        registry.register::<Follow>(orrin_registry::ComponentId::new("game.follow"), "Follow");
        registry
    }

    /// A world whose index was scanned from `dir`, as a session starts.
    fn world_in(dir: &Dir) -> World {
        let mut world = World::new();
        let mut assets = Assets::new();
        assets.insert_mesh("cube", MeshHandle(3));
        assets.insert_material("gold", MaterialHandle(7));
        world.insert_resource(assets);
        let (prefabs, issues) = Prefabs::scan(&dir.0);
        assert!(issues.is_empty(), "{issues:?}");
        world.insert_resource(prefabs);
        world
    }

    fn spawn(world: &mut World, name: &str, at: Vec3) -> Entity {
        let entity = world.spawn();
        world.insert(entity, Name::new(name));
        world.insert(entity, LocalTransform::new(Transform::from_translation(at)));
        entity
    }

    fn named(world: &World, name: &str) -> Entity {
        world
            .entities()
            .find(|&e| world.get::<Name>(e).is_some_and(|n| n.0 == name))
            .unwrap_or_else(|| panic!("no entity named {name}"))
    }

    fn parent_of(world: &World, entity: Entity) -> Option<Entity> {
        world.get::<Parent>(entity).map(|p| p.get())
    }

    /// Save the scene and open it in a fresh session over the same directory.
    fn reopen(world: &mut World, dir: &Dir) -> World {
        let text = save(world, &registry());
        let mut fresh = world_in(dir);
        let issues = load(&text, &mut fresh, &registry()).unwrap();
        assert!(issues.is_empty(), "{issues:?}");
        fresh
    }

    #[test]
    fn a_saved_tree_becomes_a_prefab_and_its_first_instance() {
        let dir = Dir::new("tree");
        let mut world = world_in(&dir);
        let room = spawn(&mut world, "Room", Vec3::new(3.0, 0.0, 0.0));
        let lamp = spawn(&mut world, "Lamp", Vec3::Y);
        world.insert(lamp, Tag::new("light"));
        reparent(&mut world, lamp, Some(room), false).unwrap();

        let (root, id) = create(&mut world, &registry(), &[room], "rooms/room.fprefab").unwrap();
        assert_eq!(root, room, "a single tree keeps its own root");
        assert!(dir.0.join("rooms/room.fprefab").is_file());
        let sidecar = std::fs::read_to_string(dir.0.join("rooms/room.fprefab.meta")).unwrap();
        assert_eq!(sidecar, format!("id = \"{id}\"\n"));

        let text = save(&mut world, &registry());
        assert_eq!(
            text.matches("\nentity ").count(),
            1,
            "an untouched instance: {text}"
        );

        let reopened = reopen(&mut world, &dir);
        let lamp = named(&reopened, "Lamp");
        assert_eq!(parent_of(&reopened, lamp), Some(named(&reopened, "Room")));
        assert_eq!(reopened.get::<Tag>(lamp).unwrap().0, "light");
    }

    #[test]
    fn several_trees_are_saved_under_one_new_root_and_stay_put() {
        let dir = Dir::new("several");
        let mut world = world_in(&dir);
        let a = spawn(&mut world, "A", Vec3::new(1.0, 0.0, 0.0));
        let b = spawn(&mut world, "B", Vec3::new(-1.0, 2.0, 0.0));
        propagate_transforms(&mut world);

        let (root, _) = create(&mut world, &registry(), &[a, b], "pair.fprefab").unwrap();
        assert_eq!(world.get::<Name>(root).unwrap().0, "pair");
        assert_eq!(parent_of(&world, a), Some(root));
        assert_eq!(parent_of(&world, b), Some(root));
        propagate_transforms(&mut world);
        let at = |e: Entity| world.get::<WorldTransform>(e).unwrap().translation();
        assert!((at(a) - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5);
        assert!((at(b) - Vec3::new(-1.0, 2.0, 0.0)).length() < 1e-5);
    }

    #[test]
    fn an_instance_inside_the_selection_is_kept_as_an_instance() {
        let dir = Dir::new("nested");
        let mut world = world_in(&dir);
        let room = spawn(&mut world, "Room", Vec3::ZERO);
        let (door, _) = super::super::instantiate(&mut world, &registry(), door_id()).unwrap();
        reparent(&mut world, door, Some(room), false).unwrap();

        create(&mut world, &registry(), &[room], "room.fprefab").unwrap();
        let file = std::fs::read_to_string(dir.0.join("room.fprefab")).unwrap();
        assert!(
            file.contains(&format!("source = asset({DOOR_ID})")),
            "{file}"
        );
        assert!(
            !file.contains("Panel"),
            "the door's members are not copied in: {file}"
        );

        let reopened = reopen(&mut world, &dir);
        let door = named(&reopened, "Door");
        assert_eq!(parent_of(&reopened, door), Some(named(&reopened, "Room")));
        assert_eq!(parent_of(&reopened, named(&reopened, "Panel")), Some(door));
    }

    /// The selected root is already an instance, and one entity cannot root two.
    #[test]
    fn a_selected_instance_root_is_given_a_root_of_its_own() {
        let dir = Dir::new("wrapped");
        let mut world = world_in(&dir);
        let (door, _) = super::super::instantiate(&mut world, &registry(), door_id()).unwrap();

        let (root, _) = create(&mut world, &registry(), &[door], "doorway.fprefab").unwrap();
        assert_ne!(root, door);
        assert_eq!(parent_of(&world, door), Some(root));
        assert_eq!(world.get::<PrefabInstance>(door).unwrap().prefab, door_id());
    }

    #[test]
    fn part_of_another_instance_cannot_become_a_prefab() {
        let dir = Dir::new("part");
        let mut world = world_in(&dir);
        super::super::instantiate(&mut world, &registry(), door_id()).unwrap();
        let panel = named(&world, "Panel");

        let err = create(&mut world, &registry(), &[panel], "panel.fprefab").unwrap_err();
        assert!(err.contains("part of an instance"), "{err}");
        assert!(!dir.0.join("panel.fprefab").exists());
    }

    #[test]
    fn an_existing_file_is_never_overwritten() {
        let dir = Dir::new("exists");
        let mut world = world_in(&dir);
        let crate_ = spawn(&mut world, "Crate", Vec3::ZERO);

        let err = create(&mut world, &registry(), &[crate_], "props/door.fprefab").unwrap_err();
        assert!(err.contains("already exists"), "{err}");
        assert_eq!(unused_path(&dir.0.join("props"), "door"), "door 2.fprefab");
        assert_eq!(unused_path(&dir.0, "a/b"), "a_b.fprefab");
    }
}
