//! Prefabs: an entity tree authored once, in a file of its own, and
//! instantiated any number of times.
//!
//! A `.fprefab` is a scene file holding one tree — the same grammar, parser,
//! writer and determinism rules, no second format. Each is an asset with an
//! [`AssetId`] in a sidecar beside it (see [`index`]), and nothing refers to a
//! prefab by its path.
//!
//! **An instance is saved as what it changes, not as what it is.** Its root's
//! block in a scene holds one `orrin.prefab` record — the source, the fields it
//! overrides, the components and members it took away — and none of its members
//! are written at all. A load expands the record against the prefab as it is
//! *then*, so an edit to the prefab reaches every field an instance left alone.
//!
//! Overrides are derived rather than tracked. At save, and when the inspector
//! asks, each member's components are compared with the prefab's through the
//! registry's [`diff`](orrin_registry::diff), and a load replays them with its
//! [`apply`](orrin_registry::apply). Not a list kept up to date edit by edit,
//! because components change through paths that report nothing — a script, a
//! reparent, a removed component — and a list beside the data would drift from
//! what the entities actually hold.
//!
//! Member ids are derived from the root's: the prefab's root becomes the
//! instance root, and every other member is its id in the prefab XOR the root's
//! id. Being deterministic, a scene entity parented under a member still finds
//! it after a reload, with no id map stored; being its own inverse, it turns an
//! instance id back into the prefab entity it came from. Not a UUIDv5 hash:
//! nothing here needs a one-way function, and one would cost a SHA-1 dependency.
//!
//! A prefab edited on disk reaches the open scene at the next rescan, which
//! updates each instance in place (see [`refresh`]).
//!
//! A prefab may hold instances of others, written as records exactly as a scene
//! writes them, and they expand in turn. Its root may not be one: one entity
//! cannot root two instances.

mod actions;
mod create;
mod index;
mod refresh;
#[cfg(feature = "scripting")]
mod watch;

use std::borrow::Cow;

use orrin_ecs::{Entity, FxHashMap, World};
use orrin_registry::{
    AssetId, ComponentId, EntityId, FieldChange, FieldPath, PathSegment, Registry, SceneEntity,
    Value,
};

pub use actions::{apply, instance_of, revert_all, unpack};
pub use create::{create, unused_path};
pub use index::{Prefab, Prefabs, Rescan};
pub use refresh::{apply_edits, rescan};
#[cfg(feature = "scripting")]
pub use watch::Watcher;

use super::Parent;
use super::persist::{self, LoadIssue};
use super::registry::{PARENT, PREFAB};

/// The root of a prefab instance. Members carry nothing: they are recognised
/// by their derived ids.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefabInstance {
    pub prefab: AssetId,
    /// Overrides the last load could not apply — a field the prefab no longer
    /// has, a member it no longer contains. Carried so the next save writes
    /// them back rather than dropping data that is still the instance's.
    unapplied: Vec<Override>,
}

/// One field an instance member holds differently from its prefab: the shape
/// of the mutation pipeline's `SetField`, addressed by the member's id in the
/// prefab so it means the same thing in every instance.
#[derive(Clone, Debug, PartialEq)]
pub struct Override {
    pub entity: EntityId,
    pub component: ComponentId,
    /// Empty when the whole component is the instance's own.
    pub path: FieldPath,
    pub value: Value,
}

/// Everything an instance changes about its prefab: the record a scene stores.
#[derive(Clone, Debug, Default, PartialEq)]
struct Changes {
    fields: Vec<Override>,
    /// Components the prefab gives a member and the instance took away.
    removed: Vec<(EntityId, ComponentId)>,
    /// Members the instance despawned.
    despawned: Vec<EntityId>,
}

impl Changes {
    /// Empty lists are left out, so an untouched instance is two lines.
    fn to_record(&self, source: AssetId) -> Value {
        let mut record = vec![("source".to_owned(), Value::Asset(source))];
        if !self.fields.is_empty() {
            let fields = self.fields.iter().map(|o| {
                Value::strukt([
                    ("entity", Value::Entity(o.entity)),
                    ("component", Value::String(o.component.to_string())),
                    ("path", Value::String(o.path.to_string())),
                    ("value", o.value.clone()),
                ])
            });
            record.push(("overrides".to_owned(), Value::List(fields.collect())));
        }
        if !self.removed.is_empty() {
            let removed = self.removed.iter().map(|(entity, component)| {
                Value::strukt([
                    ("entity", Value::Entity(*entity)),
                    ("component", Value::String(component.to_string())),
                ])
            });
            record.push(("removed".to_owned(), Value::List(removed.collect())));
        }
        if !self.despawned.is_empty() {
            let despawned = self.despawned.iter().map(|&entity| Value::Entity(entity));
            record.push(("despawned".to_owned(), Value::List(despawned.collect())));
        }
        Value::Struct(record)
    }

    fn from_record(record: &Value) -> Result<(AssetId, Self), String> {
        let Some(Value::Asset(source)) = record.field("source") else {
            return Err("the record has no `source = asset(…)`".to_owned());
        };

        let mut changes = Self::default();
        for (i, entry) in list(record, "overrides")?.iter().enumerate() {
            let at = |field: &str| format!("overrides[{i}].{field}");
            let path = text(entry, "path", &at)?;
            changes.fields.push(Override {
                entity: entity(entry, &at)?,
                component: ComponentId::owned(text(entry, "component", &at)?),
                path: path.parse().map_err(|e| format!("{}: {e}", at("path")))?,
                value: entry
                    .field("value")
                    .cloned()
                    .ok_or_else(|| format!("{} is missing", at("value")))?,
            });
        }
        for (i, entry) in list(record, "removed")?.iter().enumerate() {
            let at = |field: &str| format!("removed[{i}].{field}");
            let component = ComponentId::owned(text(entry, "component", &at)?);
            changes.removed.push((entity(entry, &at)?, component));
        }
        for (i, entry) in list(record, "despawned")?.iter().enumerate() {
            match entry {
                Value::Entity(id) => changes.despawned.push(*id),
                other => {
                    return Err(format!(
                        "despawned[{i}] should be an entity, found {}",
                        other.type_name()
                    ));
                }
            }
        }
        Ok((*source, changes))
    }

    /// The order a file lists them in, so equal instances write equal bytes.
    fn sort(&mut self) {
        self.fields
            .sort_by_cached_key(|o| (o.entity, o.component.clone(), o.path.to_string()));
        self.removed.sort();
        self.despawned.sort();
    }
}

fn list<'v>(record: &'v Value, name: &str) -> Result<&'v [Value], String> {
    match record.field(name) {
        None => Ok(&[]),
        Some(Value::List(items)) => Ok(items),
        Some(other) => Err(format!(
            "`{name}` should be a list, found {}",
            other.type_name()
        )),
    }
}

fn text<'v>(entry: &'v Value, field: &str, at: &dyn Fn(&str) -> String) -> Result<&'v str, String> {
    match entry.field(field) {
        Some(Value::String(text)) => Ok(text),
        Some(other) => Err(format!(
            "{} should be a string, found {}",
            at(field),
            other.type_name()
        )),
        None => Err(format!("{} is missing", at(field))),
    }
}

fn entity(entry: &Value, at: &dyn Fn(&str) -> String) -> Result<EntityId, String> {
    match entry.field("entity") {
        Some(Value::Entity(id)) => Ok(*id),
        Some(other) => Err(format!(
            "{} should be an entity, found {}",
            at("entity"),
            other.type_name()
        )),
        None => Err(format!("{} is missing", at("entity"))),
    }
}

/// The translation between one instance's ids and its prefab's.
#[derive(Clone, Copy)]
struct Ids {
    root: EntityId,
    prefab_root: EntityId,
}

impl Ids {
    fn member(self, local: EntityId) -> EntityId {
        if local == self.prefab_root {
            self.root
        } else {
            xor(local, self.root)
        }
    }

    fn local(self, member: EntityId) -> EntityId {
        if member == self.root {
            self.prefab_root
        } else {
            xor(member, self.root)
        }
    }

    /// Point every reference to a prefab entity at this instance's member.
    fn into_instance(self, prefab: &Prefab, value: &mut Value) {
        remap(value, &mut |id| {
            prefab.contains(id).then(|| self.member(id))
        });
    }

    /// The reverse, for a value read off a live member.
    fn into_prefab(self, prefab: &Prefab, value: &mut Value) {
        remap(value, &mut |id| {
            let local = self.local(id);
            prefab.contains(local).then_some(local)
        });
    }
}

/// Every struct's fields in name order, as a file has them.
///
/// [`diff`](orrin_registry::diff) pairs fields by position, which is right for
/// two values of one type but not for a parsed file against a live component:
/// the file is sorted and `to_value` is in declaration order, so without this a
/// transform reads as reshaped and is overridden whole.
fn canonical(value: &mut Value) {
    match value {
        Value::Struct(fields) | Value::Enum { fields, .. } => {
            fields.sort_by(|a, b| a.0.cmp(&b.0));
            for (_, field) in fields {
                canonical(field);
            }
        }
        Value::List(items) => items.iter_mut().for_each(canonical),
        _ => {}
    }
}

fn xor(a: EntityId, b: EntityId) -> EntityId {
    let (a, b) = (a.to_bytes(), b.to_bytes());
    EntityId::from_bytes(std::array::from_fn(|i| a[i] ^ b[i]))
}

fn remap(value: &mut Value, map: &mut dyn FnMut(EntityId) -> Option<EntityId>) {
    match value {
        Value::Entity(id) => {
            if let Some(mapped) = map(*id) {
                *id = mapped;
            }
        }
        Value::Struct(fields) | Value::Enum { fields, .. } => {
            for (_, field) in fields {
                remap(field, map);
            }
        }
        Value::List(items) => {
            for item in items {
                remap(item, map);
            }
        }
        _ => {}
    }
}

/// The members a record stands for, built from its prefab.
pub(super) struct Expansion {
    pub members: Vec<SceneEntity>,
    pub instance: PrefabInstance,
    /// Overrides that could not land. The instance still builds.
    pub issues: Vec<String>,
}

/// Turn the record on `scene_entity` back into the entities it stands for.
///
/// `Err` when there is nothing to build from — a malformed record or a prefab
/// the index lacks — and the caller keeps the record as unapplied data. Any
/// component written beside the record replaces the prefab root's whole, which
/// is how a hand-edited scene reads.
pub(super) fn expand<'p>(
    lookup: &dyn Fn(AssetId) -> Option<&'p Prefab>,
    scene_entity: &SceneEntity,
    record: &Value,
) -> Result<Expansion, String> {
    let (source, changes) = Changes::from_record(record)?;
    let prefab = lookup(source).ok_or_else(|| {
        format!(
            "no prefab in the project has the id {source}, so the instance was kept but not \
             built; restore the prefab or its sidecar"
        )
    })?;
    let ids = Ids {
        root: scene_entity.id,
        prefab_root: prefab.root(),
    };

    let mut applied = vec![false; changes.fields.len()];
    let mut issues = Vec::new();
    let mut members = Vec::with_capacity(prefab.entities().len());
    for base in prefab.entities() {
        if base.id != prefab.root() && changes.despawned.contains(&base.id) {
            continue;
        }
        let mut components: Vec<(ComponentId, Value)> = base
            .components
            .iter()
            .filter(|(component, _)| !changes.removed.contains(&(base.id, component.clone())))
            .cloned()
            .collect();

        for (o, applied) in changes.fields.iter().zip(&mut applied) {
            if o.entity != base.id {
                continue;
            }
            let landed = match components.iter_mut().find(|(c, _)| *c == o.component) {
                Some((_, value)) => orrin_registry::apply(
                    value,
                    &[FieldChange {
                        path: o.path.clone(),
                        value: o.value.clone(),
                    }],
                )
                .map_err(|error| error.to_string()),
                None if o.path.is_empty() => {
                    components.push((o.component.clone(), o.value.clone()));
                    Ok(())
                }
                None => Err(format!("{} no longer has this component", prefab.path)),
            };
            match landed {
                Ok(()) => *applied = true,
                Err(error) => issues.push(format!(
                    "the override of `{}` `{}` on prefab entity {} was kept but not applied — {error}",
                    o.component, o.path, o.entity
                )),
            }
        }

        if base.id == prefab.root() {
            for (component, value) in &scene_entity.components {
                if *component == PREFAB {
                    continue;
                }
                match components.iter_mut().find(|(c, _)| c == component) {
                    Some((_, slot)) => *slot = value.clone(),
                    None => components.push((component.clone(), value.clone())),
                }
            }
        }
        for (_, value) in &mut components {
            ids.into_instance(prefab, value);
        }
        members.push(SceneEntity {
            id: ids.member(base.id),
            components,
        });
    }

    for o in &changes.fields {
        if !prefab.contains(o.entity) {
            issues.push(format!(
                "the override of `{}` on entity {} was kept but not applied — {} no longer has \
                 that entity",
                o.component, o.entity, prefab.path
            ));
        }
    }
    let unapplied = changes
        .fields
        .iter()
        .zip(&applied)
        .filter(|&(_, &applied)| !applied)
        .map(|(o, _)| o.clone())
        .collect();

    Ok(Expansion {
        members,
        instance: PrefabInstance {
            prefab: source,
            unapplied,
        },
        issues,
    })
}

/// Everything a set of record holders stands for: each record expanded, and
/// every instance nested in what that expands to, until only plain entities
/// are left.
#[derive(Default)]
pub(super) struct Expanded {
    pub entities: Vec<SceneEntity>,
    pub instances: Vec<(EntityId, PrefabInstance)>,
    pub issues: Vec<LoadIssue>,
}

/// Expand `holders` — entities carrying an `orrin.prefab` record — and every
/// instance nested inside them.
///
/// A holder that cannot be expanded, or whose prefab would expand inside
/// itself, is reported and passed through with its record kept as data: the
/// alternative to the second is a load that never finishes.
pub(super) fn expand_all<'a, 'p>(
    lookup: &dyn Fn(AssetId) -> Option<&'p Prefab>,
    holders: impl IntoIterator<Item = &'a SceneEntity>,
) -> Expanded {
    let mut out = Expanded::default();
    let mut queue: Vec<(Cow<'a, SceneEntity>, Vec<AssetId>)> = holders
        .into_iter()
        .map(|holder| (Cow::Borrowed(holder), Vec::new()))
        .collect();

    while let Some((holder, chain)) = queue.pop() {
        let Some(record) = record_of(&holder) else {
            out.entities.push(holder.into_owned());
            continue;
        };
        let issue = |message| LoadIssue {
            entity: holder.id,
            component: PREFAB,
            message,
        };
        if let Some(Value::Asset(source)) = record.field("source")
            && chain.contains(source)
        {
            let path =
                lookup(*source).map_or_else(|| source.to_string(), |prefab| prefab.path.clone());
            out.issues.push(issue(format!(
                "{path} contains an instance of itself, which was kept but not built"
            )));
            out.entities.push(holder.into_owned());
            continue;
        }
        match expand(lookup, &holder, record) {
            Ok(expansion) => {
                out.issues.extend(expansion.issues.into_iter().map(&issue));
                let mut chain = chain;
                chain.push(expansion.instance.prefab);
                out.instances.push((holder.id, expansion.instance));
                queue.extend(
                    expansion
                        .members
                        .into_iter()
                        .map(|member| (Cow::Owned(member), chain.clone())),
                );
            }
            Err(message) => {
                out.issues.push(issue(message));
                out.entities.push(holder.into_owned());
            }
        }
    }
    out
}

pub(super) fn record_of(entity: &SceneEntity) -> Option<&Value> {
    entity
        .components
        .iter()
        .find_map(|(id, value)| (*id == PREFAB).then_some(value))
}

/// Replace every instance's members in `entities` with its root's record.
///
/// Members are found by the ids they must have, so one dragged out of its
/// instance is still folded, and its new parent becomes an override. A member
/// with no entity in the document was despawned. Instances nested in others
/// fold first, so an outer instance compares them as records — the form its
/// prefab holds them in. The root's parent is where the instance was placed,
/// not something it overrides, so it is written beside the record.
pub(super) fn fold(world: &World, entities: &mut Vec<SceneEntity>) {
    match world.get_resource::<Prefabs>() {
        Some(prefabs) => fold_with(world, entities, &|id| prefabs.get(id), None),
        None => {
            if world.query::<&PrefabInstance>().count() > 0 {
                tracing::warn!(
                    "saving prefab instances with no prefab index; they are written expanded"
                );
            }
        }
    }
}

/// [`fold`] against prefabs found through `lookup`, which is how a rescan
/// folds an instance against the version of its prefab it was built from.
/// Instances whose root is not in `entities`, or is `keep`, are left expanded.
pub(super) fn fold_with<'p>(
    world: &World,
    entities: &mut Vec<SceneEntity>,
    lookup: &dyn Fn(AssetId) -> Option<&'p Prefab>,
    keep: Option<EntityId>,
) {
    let mut instances: Vec<(usize, EntityId, PrefabInstance)> = Vec::new();
    world
        .query::<&PrefabInstance>()
        .for_each(|entity, instance| {
            if let Some(id) = world.get::<EntityId>(entity) {
                instances.push((nesting(world, entity), *id, instance.clone()));
            }
        });
    if instances.is_empty() {
        return;
    }
    instances.sort_by_key(|&(depth, ..)| std::cmp::Reverse(depth));

    let index: FxHashMap<EntityId, usize> = entities
        .iter()
        .enumerate()
        .map(|(i, entity)| (entity.id, i))
        .collect();
    let mut folded = vec![false; entities.len()];

    for (_, root, instance) in instances {
        if keep == Some(root) {
            continue;
        }
        let Some(&root_index) = index.get(&root) else {
            continue;
        };
        let Some(prefab) = lookup(instance.prefab) else {
            tracing::warn!(
                "entity {root} is an instance of prefab {}, which the index does not have; it \
                 is written expanded",
                instance.prefab
            );
            continue;
        };
        let ids = Ids {
            root,
            prefab_root: prefab.root(),
        };

        let mut changes = Changes::default();
        for base in prefab.entities() {
            let Some(&i) = index.get(&ids.member(base.id)) else {
                changes.despawned.push(base.id);
                continue;
            };
            folded[i] = true;
            let mut live = local_components(&entities[i].components, ids, prefab);
            if base.id == prefab.root() {
                live.retain(|(component, _)| *component != PARENT);
            }
            compare(base, &live, &mut changes);
        }
        for stale in instance.unapplied {
            let shadowed = changes.fields.iter().any(|o| {
                o.entity == stale.entity && o.component == stale.component && o.path == stale.path
            });
            if !shadowed {
                changes.fields.push(stale);
            }
        }
        changes.sort();

        let root_entity = &mut entities[root_index];
        let placement = root_entity
            .components
            .iter()
            .find(|(component, _)| *component == PARENT)
            .cloned();
        root_entity.components = placement
            .into_iter()
            .chain([(PREFAB, changes.to_record(instance.prefab))])
            .collect();
        folded[root_index] = false;
    }

    let mut i = 0;
    entities.retain(|_| {
        i += 1;
        !folded[i - 1]
    });
}

/// How many instance roots sit above `entity`.
fn nesting(world: &World, entity: Entity) -> usize {
    let mut depth = 0;
    let mut cursor = entity;
    // Bounded: a cycle the hierarchy has not broken yet would otherwise spin.
    for _ in 0..4096 {
        let Some(parent) = world
            .get::<Parent>(cursor)
            .map(|parent| parent.get())
            .filter(|&parent| world.is_alive(parent))
        else {
            break;
        };
        if world.has::<PrefabInstance>(parent) {
            depth += 1;
        }
        cursor = parent;
    }
    depth
}

fn local_components(
    components: &[(ComponentId, Value)],
    ids: Ids,
    prefab: &Prefab,
) -> Vec<(ComponentId, Value)> {
    components
        .iter()
        .map(|(component, value)| {
            let mut value = value.clone();
            ids.into_prefab(prefab, &mut value);
            canonical(&mut value);
            (component.clone(), value)
        })
        .collect()
}

/// What `live` changes about `base`, both in the prefab's ids.
fn compare(base: &SceneEntity, live: &[(ComponentId, Value)], changes: &mut Changes) {
    for (component, value) in &base.components {
        match live.iter().find(|(c, _)| c == component) {
            Some((_, now)) => {
                changes
                    .fields
                    .extend(
                        orrin_registry::diff(value, now)
                            .into_iter()
                            .map(|change| Override {
                                entity: base.id,
                                component: component.clone(),
                                path: change.path,
                                value: change.value,
                            }),
                    );
            }
            None => changes.removed.push((base.id, component.clone())),
        }
    }
    for (component, now) in live {
        if !base.components.iter().any(|(c, _)| c == component) {
            changes.fields.push(Override {
                entity: base.id,
                component: component.clone(),
                path: FieldPath::empty(),
                value: now.clone(),
            });
        }
    }
}

/// Spawn a new instance of `prefab` and return its root, with whatever the
/// build could not apply — reported as a scene load reports it.
pub fn instantiate(
    world: &mut World,
    registry: &Registry,
    prefab: AssetId,
) -> Result<(Entity, Vec<LoadIssue>), String> {
    let root = world.spawn();
    match instantiate_into(world, registry, prefab, root) {
        Ok(issues) => Ok((root, issues)),
        Err(error) => {
            world.despawn(root);
            Err(error)
        }
    }
}

/// [`instantiate`] with a root that already exists: the handle a script was
/// given back before the instance could be built. `root` should be empty.
pub fn instantiate_into(
    world: &mut World,
    registry: &Registry,
    prefab: AssetId,
    root: Entity,
) -> Result<Vec<LoadIssue>, String> {
    let known = world
        .get_resource::<Prefabs>()
        .is_some_and(|prefabs| prefabs.get(prefab).is_some());
    if !known {
        return Err(format!("no prefab in the project has the id {prefab}"));
    }
    let record = SceneEntity {
        id: EntityId::new(),
        components: vec![(PREFAB, Changes::default().to_record(prefab))],
    };
    let reserved = FxHashMap::from_iter([(record.id, root)]);
    Ok(persist::spawn(
        std::slice::from_ref(&record),
        world,
        registry,
        &reserved,
    ))
}

/// Whether `id` is a member of the instance of `prefab` rooted at `root` — the
/// test the hierarchy makes per row, without walking up from each one.
pub fn is_member(prefab: &Prefab, root: EntityId, id: EntityId) -> bool {
    let ids = Ids {
        root,
        prefab_root: prefab.root(),
    };
    prefab.contains(ids.local(id))
}

/// Where a live entity sits in a prefab instance.
struct Membership {
    root: Entity,
    prefab: AssetId,
    ids: Ids,
    local: EntityId,
}

/// The nearest instance root at or above `entity`, if `entity` is one of its
/// members. An entity under a member that is not one itself belongs to the
/// scene, not to the instance.
fn membership(world: &World, entity: Entity) -> Option<Membership> {
    let id = *world.get::<EntityId>(entity)?;
    let prefabs = world.get_resource::<Prefabs>()?;
    let mut cursor = entity;
    // Bounded: a cycle the hierarchy has not broken yet would otherwise spin.
    for _ in 0..4096 {
        if let Some(instance) = world.get::<PrefabInstance>(cursor) {
            let prefab = prefabs.get(instance.prefab)?;
            let ids = Ids {
                root: *world.get::<EntityId>(cursor)?,
                prefab_root: prefab.root(),
            };
            let local = ids.local(id);
            return prefab.contains(local).then_some(Membership {
                root: cursor,
                prefab: instance.prefab,
                ids,
                local,
            });
        }
        cursor = world
            .get::<Parent>(cursor)
            .map(|parent| parent.get())
            .filter(|&parent| world.is_alive(parent))?;
    }
    None
}

/// What the inspector shows for one member of an instance.
pub struct MemberOverrides {
    pub root: Entity,
    pub prefab: AssetId,
    pub fields: Vec<Override>,
    /// Components the prefab gives this member and it no longer has.
    pub removed: Vec<ComponentId>,
}

/// How `entity` differs from the prefab entity it was built from, or `None`
/// when it is not a member of an instance.
pub fn member_overrides(
    world: &World,
    registry: &Registry,
    entity: Entity,
) -> Option<MemberOverrides> {
    let membership = membership(world, entity)?;
    let live = persist::scene_entity(world, registry, entity)?;
    let prefabs = world.get_resource::<Prefabs>()?;
    let prefab = prefabs.get(membership.prefab)?;
    let base = prefab.entity(membership.local)?;

    let mut changes = Changes::default();
    let mut live = local_components(&live.components, membership.ids, prefab);
    if base.id == prefab.root() {
        live.retain(|(component, _)| *component != PARENT);
    }
    compare(base, &live, &mut changes);
    Some(MemberOverrides {
        root: membership.root,
        prefab: membership.prefab,
        fields: changes.fields,
        removed: changes.removed.into_iter().map(|(_, c)| c).collect(),
    })
}

/// Give one override back to the prefab: the field at `path` through the
/// registry's `apply`, or with an empty path the whole component — which also
/// restores a removed one and takes away one the prefab does not have.
pub fn revert(
    world: &mut World,
    registry: &Registry,
    entity: Entity,
    component: &ComponentId,
    path: &FieldPath,
) -> Result<(), String> {
    let membership =
        membership(world, entity).ok_or("the entity is not a member of a prefab instance")?;
    let base = {
        let prefabs = world
            .get_resource::<Prefabs>()
            .ok_or("there is no prefab index")?;
        let prefab = prefabs
            .get(membership.prefab)
            .ok_or("the instance's prefab has left the index")?;
        prefab
            .entity(membership.local)
            .and_then(|base| base.components.iter().find(|(c, _)| c == component))
            .map(|(_, value)| {
                let mut value = value.clone();
                membership.ids.into_instance(prefab, &mut value);
                value
            })
    };

    let Some(base) = base else {
        persist::remove_component(world, registry, entity, component);
        return Ok(());
    };
    if let Some(vtable) = registry.get(component)
        && !path.is_empty()
        && vtable.has(world, entity)
    {
        let value = at(&base, path).ok_or_else(|| format!("the prefab has no `{path}`"))?;
        return vtable
            .apply(
                world,
                entity,
                &[FieldChange {
                    path: path.clone(),
                    value: value.clone(),
                }],
            )
            .map_err(|error| error.to_string());
    }
    persist::write_component(world, registry, entity, component, &base)
}

fn at<'v>(value: &'v Value, path: &FieldPath) -> Option<&'v Value> {
    path.segments()
        .iter()
        .try_fold(value, |value, segment| match segment {
            PathSegment::Field(name) => value.field(name),
            PathSegment::Index(index) => match value {
                Value::List(items) => items.get(*index),
                _ => None,
            },
        })
}

#[cfg(test)]
mod tests {
    use glam::Vec3;
    use orrin_ecs::{Entity, World};
    use orrin_registry::{AssetId, ComponentId, EntityId, FieldPath, Registry, Value};

    use super::index::tests::{DOOR_ID, HANDLE, PANEL, ROOT, door_id, door_text};
    use super::*;
    use crate::scene::registry::{SPIN, TAG, TRANSFORM};
    use crate::scene::{
        Assets, LocalTransform, MaterialHandle, MeshHandle, Name, Parent, Spin, Tag, load,
        register_components, save,
    };

    fn id(text: &str) -> EntityId {
        text.parse().unwrap()
    }

    /// A game component that names another entity, so remapping is tested
    /// through a real registry entry.
    #[derive(Default, orrin_registry::Reflect)]
    struct Follow {
        target: EntityId,
    }

    fn registry() -> Registry {
        let mut registry = Registry::new();
        register_components(&mut registry);
        registry.register::<Follow>(ComponentId::new("game.follow"), "Follow");
        registry
    }

    /// The door, and a house holding one.
    fn library(door: &str) -> Prefabs {
        let mut prefabs = Prefabs::default();
        prefabs.insert(Prefab::parse(door_id(), "props/door.fprefab", door).unwrap());
        prefabs.insert(Prefab::parse(house_id(), "props/house.fprefab", &house_text()).unwrap());
        prefabs
    }

    /// A world with the engine's demo assets and the door in its prefab index.
    fn world_with(door: &str) -> World {
        let mut world = World::new();
        let mut assets = Assets::new();
        assets.insert_mesh("cube", MeshHandle(3));
        assets.insert_material("gold", MaterialHandle(7));
        world.insert_resource(assets);
        world.insert_resource(library(door));
        world
    }

    fn world() -> World {
        world_with(&door_text("handle", 1.0))
    }

    fn named(world: &World, name: &str) -> Entity {
        let found: Vec<Entity> = world
            .entities()
            .filter(|&e| world.get::<Name>(e).is_some_and(|n| n.0 == name))
            .collect();
        assert_eq!(found.len(), 1, "expected one entity named {name}");
        found[0]
    }

    fn parent_of(world: &World, entity: Entity) -> Option<Entity> {
        world.get::<Parent>(entity).map(|p| p.get())
    }

    fn entity_id(world: &World, entity: Entity) -> EntityId {
        *world.get::<EntityId>(entity).unwrap()
    }

    fn reload(text: &str, door: &str) -> World {
        let mut world = world_with(door);
        let issues = load(text, &mut world, &registry()).unwrap();
        assert!(issues.is_empty(), "{issues:?}");
        world
    }

    fn move_panel(world: &mut World, to: Vec3) {
        let panel = named(world, "Panel");
        world.get_mut::<LocalTransform>(panel).unwrap().translation = to;
    }

    #[test]
    fn instantiating_builds_the_tree_under_one_root() {
        let mut world = world();
        let (root, issues) = instantiate(&mut world, &registry(), door_id()).unwrap();
        assert!(issues.is_empty(), "{issues:?}");

        assert_eq!(world.entities().count(), 3);
        assert_eq!(world.get::<Name>(root).unwrap().0, "Door");
        assert_eq!(world.get::<PrefabInstance>(root).unwrap().prefab, door_id());
        let panel = named(&world, "Panel");
        let handle = named(&world, "Handle");
        assert_eq!(parent_of(&world, panel), Some(root));
        assert_eq!(parent_of(&world, handle), Some(panel));
        assert!(world.get::<Spin>(panel).is_some());
        assert_eq!(*world.get::<MeshHandle>(panel).unwrap(), MeshHandle(3));
    }

    /// Every instance is a different set of entities: a second door sharing the
    /// first's ids would be the same door twice in the file.
    #[test]
    fn each_instance_has_its_own_identities() {
        let mut world = world();
        let registry = registry();
        instantiate(&mut world, &registry, door_id()).unwrap();
        instantiate(&mut world, &registry, door_id()).unwrap();

        let mut ids: Vec<EntityId> = world.entities().map(|e| entity_id(&world, e)).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 6);
        for local in [ROOT, PANEL, HANDLE] {
            assert!(
                !ids.contains(&id(local)),
                "an instance reused the prefab's own id"
            );
        }
    }

    #[test]
    fn a_reference_inside_the_prefab_points_into_the_instance() {
        let mut world = world();
        instantiate(&mut world, &registry(), door_id()).unwrap();
        let panel = named(&world, "Panel");
        let handle = entity_id(&world, named(&world, "Handle"));

        assert_eq!(world.get::<Follow>(panel).unwrap().target, handle);
    }

    /// What makes an instance cheap in a scene file and an edit to its prefab
    /// reach it: none of the members is written, only where they came from.
    #[test]
    fn an_untouched_instance_saves_as_its_source_alone() {
        let mut world = world();
        let (root, _) = instantiate(&mut world, &registry(), door_id()).unwrap();
        let text = save(&mut world, &registry());

        let root_id = entity_id(&world, root);
        assert_eq!(
            text,
            format!(
                "orrin-scene 1\n\nentity {root_id}\n  orrin.prefab\n    source = asset({DOOR_ID})\n"
            )
        );
    }

    #[test]
    fn an_edited_field_saves_as_an_override_and_loads_back() {
        let mut world = world();
        instantiate(&mut world, &registry(), door_id()).unwrap();
        move_panel(&mut world, Vec3::new(0.0, 2.0, 0.0));
        let first = save(&mut world, &registry());

        assert!(
            first.contains(&format!(
                "    overrides\n      [0]\n        component = \"orrin.transform\"\n        \
                 entity = entity({PANEL})\n        path = \"translation\"\n        \
                 value = (0.0, 2.0, 0.0)\n"
            )),
            "{first}"
        );

        let mut loaded = reload(&first, &door_text("handle", 1.0));
        let panel = named(&loaded, "Panel");
        assert_eq!(
            loaded.get::<LocalTransform>(panel).unwrap().translation,
            Vec3::new(0.0, 2.0, 0.0)
        );
        assert_eq!(save(&mut loaded, &registry()), first);
    }

    /// The reason an instance is stored as overrides at all.
    #[test]
    fn a_prefab_edit_reaches_every_field_the_instance_left_alone() {
        let mut world = world();
        instantiate(&mut world, &registry(), door_id()).unwrap();
        move_panel(&mut world, Vec3::new(0.0, 2.0, 0.0));
        let text = save(&mut world, &registry());

        // Next session, the prefab's author has moved the panel and retagged the
        // handle.
        let loaded = reload(&text, &door_text("knob", 5.0));
        assert_eq!(
            loaded.get::<Tag>(named(&loaded, "Handle")).unwrap().0,
            "knob"
        );
        assert_eq!(
            loaded
                .get::<LocalTransform>(named(&loaded, "Panel"))
                .unwrap()
                .translation,
            Vec3::new(0.0, 2.0, 0.0),
            "the instance's own value must win over the prefab's new one"
        );
    }

    #[test]
    fn a_removed_component_and_a_despawned_member_stay_gone() {
        let mut world = world();
        instantiate(&mut world, &registry(), door_id()).unwrap();
        let panel = named(&world, "Panel");
        world.remove::<Spin>(panel);
        let handle = named(&world, "Handle");
        crate::scene::despawn_recursive(&mut world, handle);
        let first = save(&mut world, &registry());

        let mut loaded = reload(&first, &door_text("handle", 1.0));
        assert_eq!(loaded.entities().count(), 2);
        assert!(loaded.get::<Spin>(named(&loaded, "Panel")).is_none());
        assert_eq!(save(&mut loaded, &registry()), first);
    }

    #[test]
    fn a_component_the_prefab_lacks_is_kept_whole() {
        let mut world = world();
        let (root, _) = instantiate(&mut world, &registry(), door_id()).unwrap();
        world.insert(root, Tag::new("entrance"));
        let text = save(&mut world, &registry());

        let loaded = reload(&text, &door_text("handle", 1.0));
        assert_eq!(
            loaded.get::<Tag>(named(&loaded, "Door")).unwrap().0,
            "entrance"
        );
    }

    #[test]
    fn an_instance_placed_under_a_scene_entity_stays_there() {
        let mut world = world();
        let room = world.spawn();
        world.insert(room, Name::new("Room"));
        let (root, _) = instantiate(&mut world, &registry(), door_id()).unwrap();
        crate::scene::reparent(&mut world, root, Some(room), false).unwrap();
        let text = save(&mut world, &registry());

        let loaded = reload(&text, &door_text("handle", 1.0));
        assert_eq!(
            parent_of(&loaded, named(&loaded, "Door")),
            Some(named(&loaded, "Room"))
        );
    }

    /// Member ids are derived, never stored, so something in the scene that points
    /// at one has to find the same id after a reload.
    #[test]
    fn a_scene_entity_under_a_member_keeps_its_parent() {
        let mut world = world();
        instantiate(&mut world, &registry(), door_id()).unwrap();
        let sticker = world.spawn();
        world.insert(sticker, Name::new("Sticker"));
        let panel = named(&world, "Panel");
        crate::scene::reparent(&mut world, sticker, Some(panel), false).unwrap();
        let first = save(&mut world, &registry());

        let mut loaded = reload(&first, &door_text("handle", 1.0));
        assert_eq!(
            parent_of(&loaded, named(&loaded, "Sticker")),
            Some(named(&loaded, "Panel"))
        );
        assert_eq!(save(&mut loaded, &registry()), first);
    }

    /// A prefab deleted, or not yet pulled from version control, must not take its
    /// instances with it on the next save.
    #[test]
    fn an_instance_of_an_unknown_prefab_is_reported_and_kept() {
        let root = EntityId::new();
        let missing = AssetId::new();
        let text = format!(
            "orrin-scene 1\n\nentity {root}\n  orrin.prefab\n    overrides\n      [0]\n        \
             component = \"orrin.tag\"\n        entity = entity({ROOT})\n        path = \"\"\n        \
             value = \"x\"\n    source = asset({missing})\n"
        );
        let mut world = world();
        let issues = load(&text, &mut world, &registry()).unwrap();

        assert_eq!(issues.len(), 1, "{issues:?}");
        assert_eq!(issues[0].component, ComponentId::new("orrin.prefab"));
        assert!(
            issues[0].message.contains(&missing.to_string()),
            "{}",
            issues[0]
        );
        assert_eq!(world.entities().count(), 1);
        assert_eq!(save(&mut world, &registry()), text);
    }

    /// The prefab dropped a field an instance overrode. The override cannot land,
    /// and it is reported — but it is the instance's data, so it is not dropped.
    #[test]
    fn an_override_that_no_longer_fits_is_reported_and_kept() {
        let root = EntityId::new();
        let text = format!(
            "orrin-scene 1\n\nentity {root}\n  orrin.prefab\n    overrides\n      [0]\n        \
             component = \"orrin.transform\"\n        entity = entity({PANEL})\n        \
             path = \"skew\"\n        value = 0.5\n    source = asset({DOOR_ID})\n"
        );
        let mut world = world();
        let issues = load(&text, &mut world, &registry()).unwrap();

        assert_eq!(issues.len(), 1, "{issues:?}");
        assert!(issues[0].message.contains("skew"), "{}", issues[0]);
        assert_eq!(
            world.entities().count(),
            3,
            "the rest of the instance still builds"
        );
        assert_eq!(save(&mut world, &registry()), text);
    }

    #[test]
    fn a_member_reports_its_overrides_and_reverting_one_restores_the_prefab() {
        let registry = registry();
        let mut world = world();
        let (root, _) = instantiate(&mut world, &registry, door_id()).unwrap();
        let panel = named(&world, "Panel");
        assert!(
            member_overrides(&world, &registry, panel)
                .unwrap()
                .fields
                .is_empty()
        );

        move_panel(&mut world, Vec3::new(0.0, 2.0, 0.0));
        let overrides = member_overrides(&world, &registry, panel).unwrap();
        assert_eq!(overrides.root, root);
        assert_eq!(
            overrides.fields,
            vec![Override {
                entity: id(PANEL),
                component: TRANSFORM,
                path: FieldPath::field("translation"),
                value: Value::Vec3(Vec3::new(0.0, 2.0, 0.0)),
            }]
        );

        revert(
            &mut world,
            &registry,
            panel,
            &TRANSFORM,
            &FieldPath::field("translation"),
        )
        .unwrap();
        assert_eq!(
            world.get::<LocalTransform>(panel).unwrap().translation,
            Vec3::new(0.0, 1.0, 0.0)
        );
        assert!(
            member_overrides(&world, &registry, panel)
                .unwrap()
                .fields
                .is_empty()
        );
    }

    #[test]
    fn reverting_brings_back_a_removed_component_and_takes_away_an_added_one() {
        let registry = registry();
        let mut world = world();
        let (root, _) = instantiate(&mut world, &registry, door_id()).unwrap();
        let panel = named(&world, "Panel");
        world.remove::<Spin>(panel);
        world.insert(root, Tag::new("entrance"));
        assert_eq!(
            member_overrides(&world, &registry, panel).unwrap().removed,
            vec![SPIN]
        );

        revert(&mut world, &registry, panel, &SPIN, &FieldPath::empty()).unwrap();
        revert(&mut world, &registry, root, &TAG, &FieldPath::empty()).unwrap();
        assert!(world.get::<Spin>(panel).is_some());
        assert!(world.get::<Tag>(root).is_none());
    }

    #[test]
    fn an_entity_outside_any_instance_has_no_overrides() {
        let registry = registry();
        let mut world = world();
        let loose = world.spawn();
        world.insert(loose, EntityId::new());
        assert!(member_overrides(&world, &registry, loose).is_none());

        // Parented under a member, but not one itself.
        instantiate(&mut world, &registry, door_id()).unwrap();
        let panel = named(&world, "Panel");
        crate::scene::reparent(&mut world, loose, Some(panel), false).unwrap();
        assert!(member_overrides(&world, &registry, loose).is_none());
    }

    const HOUSE_ID: &str = "40c5e000-0000-4000-8000-000000000002";
    const HOUSE_ROOT: &str = "00000000-0000-4000-8000-0000000000f0";
    const HOUSE_DOOR: &str = "00000000-0000-4000-8000-0000000000f1";

    /// A house holding a door, whose handle the house makes brass.
    fn house_text() -> String {
        format!(
            "\
orrin-scene 1

entity {HOUSE_ROOT}
  orrin.name = \"House\"

entity {HOUSE_DOOR}
  orrin.parent = entity({HOUSE_ROOT})
  orrin.prefab
    overrides
      [0]
        component = \"orrin.tag\"
        entity = entity({HANDLE})
        path = \"\"
        value = \"brass\"
    source = asset({DOOR_ID})
"
        )
    }

    fn house_id() -> AssetId {
        HOUSE_ID.parse().unwrap()
    }

    #[test]
    fn a_prefab_can_hold_an_instance_of_another() {
        let mut world = world();
        let (house, issues) = instantiate(&mut world, &registry(), house_id()).unwrap();
        assert!(issues.is_empty(), "{issues:?}");

        assert_eq!(world.entities().count(), 4);
        let door = named(&world, "Door");
        assert_eq!(parent_of(&world, door), Some(house));
        assert_eq!(world.get::<PrefabInstance>(door).unwrap().prefab, door_id());
        assert_eq!(
            world.get::<Tag>(named(&world, "Handle")).unwrap().0,
            "brass"
        );
    }

    #[test]
    fn an_edit_inside_a_nested_instance_survives_a_round_trip() {
        let mut world = world();
        instantiate(&mut world, &registry(), house_id()).unwrap();
        move_panel(&mut world, Vec3::new(0.0, 2.0, 0.0));
        let first = save(&mut world, &registry());
        assert_eq!(first.matches("\nentity ").count(), 1, "{first}");

        let mut loaded = reload(&first, &door_text("handle", 1.0));
        assert_eq!(
            loaded
                .get::<LocalTransform>(named(&loaded, "Panel"))
                .unwrap()
                .translation,
            Vec3::new(0.0, 2.0, 0.0)
        );
        assert_eq!(
            loaded.get::<Tag>(named(&loaded, "Handle")).unwrap().0,
            "brass"
        );
        assert_eq!(save(&mut loaded, &registry()), first);
    }

    /// An edit to the door reaches the doors inside houses too, except where a
    /// house overrides it.
    #[test]
    fn a_door_edit_reaches_the_doors_inside_houses() {
        let mut world = world();
        instantiate(&mut world, &registry(), house_id()).unwrap();
        let text = save(&mut world, &registry());

        let loaded = reload(&text, &door_text("knob", 5.0));
        assert_eq!(
            loaded
                .get::<LocalTransform>(named(&loaded, "Panel"))
                .unwrap()
                .translation,
            Vec3::new(0.0, 5.0, 0.0)
        );
        assert_eq!(
            loaded.get::<Tag>(named(&loaded, "Handle")).unwrap().0,
            "brass"
        );
    }

    #[test]
    fn a_prefab_that_contains_itself_is_reported_rather_than_followed() {
        let looped = format!(
            "{}\nentity {HOUSE_DOOR}\n  orrin.parent = entity({ROOT})\n  orrin.prefab\n    \
             source = asset({DOOR_ID})\n",
            door_text("handle", 1.0)
        );
        let mut world = world_with(&looped);
        let (_, issues) = instantiate(&mut world, &registry(), door_id()).unwrap();

        assert_eq!(issues.len(), 1, "{issues:?}");
        assert!(
            issues[0].message.contains("an instance of itself"),
            "{}",
            issues[0]
        );
        assert_eq!(
            world.entities().count(),
            4,
            "the loop's entry point is kept, not expanded"
        );
    }

    /// Placement belongs to the instance, and sits beside its record where a
    /// reader of the scene file — and the prefab index, finding a tree's root —
    /// looks for a parent.
    #[test]
    fn an_instance_root_keeps_its_parent_beside_its_record() {
        let mut world = world();
        let room = world.spawn();
        world.insert(room, Name::new("Room"));
        let (root, _) = instantiate(&mut world, &registry(), door_id()).unwrap();
        crate::scene::reparent(&mut world, root, Some(room), false).unwrap();
        save(&mut world, &registry());
        let room_id = entity_id(&world, room);

        let text = save(&mut world, &registry());
        let root_id = entity_id(&world, root);
        assert!(
            text.contains(&format!(
                "entity {root_id}\n  orrin.parent = entity({room_id})\n  orrin.prefab\n"
            )),
            "{text}"
        );
    }

    /// Replace the door in the index with `door`, and bring live instances up
    /// to date — what a rescan does after the file changed on disk.
    fn edit_door(world: &mut World, door: &str) -> Vec<String> {
        let old = {
            let mut prefabs = world.resource_mut::<Prefabs>();
            let new = Prefab::parse(door_id(), "props/door.fprefab", door).unwrap();
            prefabs.replace(new).expect("the door was indexed")
        };
        apply_edits(world, &registry(), vec![old])
    }

    /// The door with a hinge under its root and without its handle.
    fn rehung_door() -> String {
        let door = door_text("handle", 1.0);
        let without_handle = &door[..door.find(&format!("entity {HANDLE}")).unwrap()];
        format!(
            "{without_handle}entity {HINGE}\n  orrin.name = \"Hinge\"\n  orrin.parent = entity({ROOT})\n"
        )
    }

    const HINGE: &str = "00000000-0000-4000-8000-00000000000d";

    #[test]
    fn a_prefab_edit_reaches_live_instances_in_place() {
        let mut world = world();
        let (root, _) = instantiate(&mut world, &registry(), door_id()).unwrap();
        let panel = named(&world, "Panel");
        let handle = named(&world, "Handle");
        move_panel(&mut world, Vec3::new(0.0, 2.0, 0.0));

        let issues = edit_door(&mut world, &door_text("knob", 5.0));
        assert!(issues.is_empty(), "{issues:?}");
        assert_eq!(world.get::<Tag>(handle).unwrap().0, "knob");
        assert_eq!(
            world.get::<LocalTransform>(panel).unwrap().translation,
            Vec3::new(0.0, 2.0, 0.0),
            "the instance's own value still wins"
        );
        assert_eq!(
            parent_of(&world, panel),
            Some(root),
            "the same entities, edited"
        );
    }

    #[test]
    fn a_member_the_prefab_gains_appears_and_one_it_loses_goes() {
        let mut world = world();
        let (root, _) = instantiate(&mut world, &registry(), door_id()).unwrap();

        let issues = edit_door(&mut world, &rehung_door());
        assert!(issues.is_empty(), "{issues:?}");
        assert_eq!(parent_of(&world, named(&world, "Hinge")), Some(root));
        assert!(
            !world
                .entities()
                .any(|e| world.get::<Name>(e).is_some_and(|n| n.0 == "Handle"))
        );
        assert_eq!(world.entities().count(), 3);
    }

    #[test]
    fn a_scene_entity_under_a_member_outlives_the_edit() {
        let mut world = world();
        instantiate(&mut world, &registry(), door_id()).unwrap();
        let sticker = world.spawn();
        world.insert(sticker, Name::new("Sticker"));
        let panel = named(&world, "Panel");
        crate::scene::reparent(&mut world, sticker, Some(panel), false).unwrap();

        edit_door(&mut world, &door_text("knob", 5.0));
        assert_eq!(parent_of(&world, sticker), Some(panel));
    }

    #[test]
    fn a_prefab_edit_reaches_the_doors_inside_live_houses() {
        let mut world = world();
        instantiate(&mut world, &registry(), house_id()).unwrap();

        edit_door(&mut world, &door_text("knob", 5.0));
        assert_eq!(
            world
                .get::<LocalTransform>(named(&world, "Panel"))
                .unwrap()
                .translation,
            Vec3::new(0.0, 5.0, 0.0)
        );
        assert_eq!(
            world.get::<Tag>(named(&world, "Handle")).unwrap().0,
            "brass"
        );
    }

    /// End to end, through the disk: edit the file, rescan, and the instance
    /// in the open scene changes.
    #[test]
    fn rescanning_after_a_file_edit_updates_the_scene() {
        let dir = std::env::temp_dir().join(format!("orrin-refresh-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("door.fprefab"), door_text("handle", 1.0)).unwrap();
        std::fs::write(
            dir.join("door.fprefab.meta"),
            format!("id = \"{DOOR_ID}\"\n"),
        )
        .unwrap();

        let mut world = world();
        world.insert_resource(Prefabs::scan(&dir).0);
        instantiate(&mut world, &registry(), door_id()).unwrap();
        std::fs::write(dir.join("door.fprefab"), door_text("knob", 1.0)).unwrap();

        let issues = rescan(&mut world, &registry());
        let _ = std::fs::remove_dir_all(&dir);
        assert!(issues.is_empty(), "{issues:?}");
        assert_eq!(world.get::<Tag>(named(&world, "Handle")).unwrap().0, "knob");
    }
}
