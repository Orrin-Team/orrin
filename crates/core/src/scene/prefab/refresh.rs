//! Bringing live instances up to date after their prefab changed on disk.
//!
//! In place rather than by despawning and respawning: a script holds handles
//! to members, a Behaviour on one has state, and the editor has one selected.
//! Each instance is folded against the prefab it was built from — which yields
//! exactly its overrides — and that record is expanded against the new
//! version. Members both versions have are edited component by component,
//! members only the new one has are spawned, and members it dropped go.
//!
//! The same rebuild serves an instance's own Revert, which expands the record
//! with its overrides taken out.

use orrin_ecs::{Entity, FxBuildHasher, FxHashMap, World};
use orrin_registry::{EntityId, Registry, SceneEntity, Value};

use super::create::{children_of, subtree};
use super::{Ids, Prefab, PrefabInstance, Prefabs, canonical, expand_all, fold_with, nesting};
use crate::scene::persist;

type FxHashSet<T> = std::collections::HashSet<T, FxBuildHasher>;

/// Rescan the prefab index and bring every live instance of a changed prefab
/// up to date. Returns what could not be indexed or applied.
pub fn rescan(world: &mut World, registry: &Registry) -> Vec<String> {
    let Some(rescan) = world
        .get_resource_mut::<Prefabs>()
        .map(|mut prefabs| prefabs.rescan())
    else {
        return Vec::new();
    };
    let mut issues = rescan.issues;
    issues.extend(apply_edits(world, registry, rescan.replaced));
    issues
}

/// Bring every instance of a prefab in `replaced` — the versions the index held
/// before it was updated — up to date with the version it holds now.
///
/// Instances nested in others go first, so an outer instance folds an inner
/// one that already matches its new prefab.
pub fn apply_edits(world: &mut World, registry: &Registry, replaced: Vec<Prefab>) -> Vec<String> {
    if replaced.is_empty() {
        return Vec::new();
    }
    let mut roots: Vec<(usize, Entity, usize)> = Vec::new();
    world
        .query::<&PrefabInstance>()
        .for_each(|entity, instance| {
            if let Some(old) = replaced.iter().position(|p| p.id == instance.prefab) {
                roots.push((nesting(world, entity), entity, old));
            }
        });
    roots.sort_by_key(|&(depth, ..)| std::cmp::Reverse(depth));

    let mut issues = Vec::new();
    for (_, root, old) in roots {
        if !world.is_alive(root) {
            continue;
        }
        let built_from = &replaced[old];
        if let Some(holder) = snapshot(world, registry, root, built_from) {
            issues.extend(rebuild(world, registry, built_from, &holder, &holder));
        }
    }
    issues
}

/// The instance rooted at `root` as the one record a scene would save for it,
/// folded against `built_from` — the version of its prefab it was built from.
///
/// Every member is gathered wherever it was dragged, with whatever the scene
/// hung under it, since the fold has to see them all.
pub(super) fn snapshot(
    world: &mut World,
    registry: &Registry,
    root: Entity,
    built_from: &Prefab,
) -> Option<SceneEntity> {
    let root_id = *world.get::<EntityId>(root)?;
    let by_id = persist::ids_of(world);
    let ids = Ids {
        root: root_id,
        prefab_root: built_from.root(),
    };

    let children = children_of(world);
    let mut seen = FxHashSet::default();
    let mut tree = Vec::new();
    let members = built_from
        .entities()
        .iter()
        .filter_map(|base| by_id.get(&ids.member(base.id)));
    for &start in std::iter::once(&root).chain(members) {
        for entity in subtree(&children, start) {
            if seen.insert(entity) {
                tree.push(entity);
            }
        }
    }
    for &entity in &tree {
        if !world.has::<EntityId>(entity) {
            world.insert(entity, EntityId::new());
        }
    }
    let mut entities: Vec<SceneEntity> = tree
        .iter()
        .filter_map(|&entity| persist::scene_entity(world, registry, entity))
        .collect();

    let prefabs = world.get_resource::<Prefabs>()?;
    let lookup = |id| {
        if id == built_from.id {
            Some(built_from)
        } else {
            prefabs.get(id)
        }
    };
    fold_with(world, &mut entities, &lookup, None);
    entities.into_iter().find(|entity| entity.id == root_id)
}

/// Make the live instance that `was` describes — a record folded against
/// `built_from` — into what `now` describes against the index as it is, in
/// place: shared members edited, new ones spawned, dropped ones despawned.
pub(super) fn rebuild(
    world: &mut World,
    registry: &Registry,
    built_from: &Prefab,
    was: &SceneEntity,
    now: &SceneEntity,
) -> Vec<String> {
    let by_id: FxHashMap<EntityId, Entity> = persist::ids_of(world);
    let (was, now) = {
        let Some(prefabs) = world.get_resource::<Prefabs>() else {
            return Vec::new();
        };
        let before = |id| {
            if id == built_from.id {
                Some(built_from)
            } else {
                prefabs.get(id)
            }
        };
        (
            expand_all(&before, [was]),
            expand_all(&|id| prefabs.get(id), [now]),
        )
    };

    let mut issues: Vec<String> = now.issues.iter().map(|issue| issue.to_string()).collect();
    let wanted: FxHashSet<EntityId> = now.entities.iter().map(|entity| entity.id).collect();

    let fresh = now
        .entities
        .iter()
        .filter(|entity| !by_id.contains_key(&entity.id));
    let spawned = persist::spawn_plain(fresh, &now.instances, world, registry, &Default::default());
    issues.extend(spawned.iter().map(|issue| issue.to_string()));

    for desired in &now.entities {
        if let Some(&entity) = by_id.get(&desired.id) {
            issues.extend(sync(world, registry, entity, desired));
        }
    }
    for (id, instance) in &now.instances {
        if let Some(&entity) = by_id.get(id) {
            world.insert(entity, instance.clone());
        }
    }
    for gone in was
        .entities
        .iter()
        .filter(|entity| !wanted.contains(&entity.id))
    {
        if let Some(&entity) = by_id.get(&gone.id) {
            crate::scene::despawn_recursive(world, entity);
        }
    }
    issues
}

/// Make a live entity's components exactly `desired`'s, touching only the
/// ones that differ — so a Behaviour whose fields did not change is not
/// written to, and an unchanged transform does not count as an edit.
fn sync(
    world: &mut World,
    registry: &Registry,
    entity: Entity,
    desired: &SceneEntity,
) -> Vec<String> {
    let Some(live) = persist::scene_entity(world, registry, entity) else {
        return Vec::new();
    };
    let canon = |value: &Value| {
        let mut value = value.clone();
        canonical(&mut value);
        value
    };

    let mut issues = Vec::new();
    for (component, value) in &desired.components {
        let unchanged = live
            .components
            .iter()
            .find(|(c, _)| c == component)
            .is_some_and(|(_, now)| orrin_registry::diff(&canon(now), &canon(value)).is_empty());
        if unchanged {
            continue;
        }
        if let Err(error) = persist::write_component(world, registry, entity, component, value) {
            issues.push(format!("entity {}: `{component}` — {error}", desired.id));
        }
    }
    for (component, _) in &live.components {
        if !desired.components.iter().any(|(c, _)| c == component) {
            persist::remove_component(world, registry, entity, component);
        }
    }
    issues
}
