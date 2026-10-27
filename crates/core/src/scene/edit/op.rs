//! The vocabulary every edit-mode change is spoken in, and how each one is
//! applied and inverted.

use orrin_ecs::{Entity, FxHashMap, World};
use orrin_registry::{AssetId, ComponentId, EntityId, FieldChange, Registry, SceneEntity, Value};

use crate::asset_db::AssetDatabase;
use crate::scene::{despawn_recursive, persist, reparent};

/// One change to the scene or to an asset file the editor writes, addressed by
/// [`EntityId`] or [`AssetId`] so it still means the same thing after an undo
/// has despawned and respawned an entity under a new slot.
///
/// Closed and small on purpose. A gesture — a drag, a paste, a revert — is a
/// [`Batch`](Op::Batch) of these, never an op of its own, so undo, dirty
/// tracking and later sync each handle eight cases rather than one per tool.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// New values at paths inside one component, with the values they
    /// replaced. The paths of one side are disjoint, so their order is not
    /// significant.
    SetField {
        entity: EntityId,
        component: ComponentId,
        old: Vec<FieldChange>,
        new: Vec<FieldChange>,
    },
    AddComponent {
        entity: EntityId,
        component: ComponentId,
        value: Value,
    },
    /// Carries the whole value it took away, recorded when it was emitted, so
    /// that it can be put back.
    RemoveComponent {
        entity: EntityId,
        component: ComponentId,
        value: Value,
    },
    /// Subtrees in the form a scene file holds them: prefab instances folded
    /// to their records, parents by id.
    ///
    /// `slots` is where each entity, prefab members included, sat in this
    /// session's ECS. Nothing else orders siblings — the tree is drawn in slot
    /// order — so without it a respawn would reshuffle the tree. It is a hint:
    /// meaningless to another session, and a slot taken since is not used.
    Spawn {
        entities: Vec<SceneEntity>,
        slots: Vec<(EntityId, u32)>,
    },
    /// The inverse of [`Spawn`](Op::Spawn), recorded in the same form.
    Despawn {
        entities: Vec<SceneEntity>,
        slots: Vec<(EntityId, u32)>,
    },
    /// Only the link. A move that keeps the world position is this plus a
    /// [`SetField`](Op::SetField) on the transform, in one batch.
    Reparent {
        entity: EntityId,
        from: Option<EntityId>,
        to: Option<EntityId>,
    },
    /// An asset file's whole text, before and after. A `.fmat` is a few
    /// hundred bytes, and keeping the text rather than the values is what lets
    /// an undo leave the file exactly as it was, however it had been written.
    WriteAsset {
        asset: AssetId,
        old: String,
        new: String,
    },
    /// Applies whole or not at all, and undoes as one step under its label.
    Batch { label: String, ops: Vec<Op> },
}

/// Why an op could not be applied. Names the entity and component involved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EditError(pub String);

impl std::fmt::Display for EditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EditError {}

impl Op {
    /// The op that undoes this one.
    pub fn invert(self) -> Op {
        match self {
            Op::SetField {
                entity,
                component,
                old,
                new,
            } => Op::SetField {
                entity,
                component,
                old: new,
                new: old,
            },
            Op::AddComponent {
                entity,
                component,
                value,
            } => Op::RemoveComponent {
                entity,
                component,
                value,
            },
            Op::RemoveComponent {
                entity,
                component,
                value,
            } => Op::AddComponent {
                entity,
                component,
                value,
            },
            Op::Spawn { entities, slots } => Op::Despawn { entities, slots },
            Op::Despawn { entities, slots } => Op::Spawn { entities, slots },
            Op::Reparent { entity, from, to } => Op::Reparent {
                entity,
                from: to,
                to: from,
            },
            Op::WriteAsset { asset, old, new } => Op::WriteAsset {
                asset,
                old: new,
                new: old,
            },
            Op::Batch { label, ops } => Op::Batch {
                label,
                ops: ops.into_iter().rev().map(Op::invert).collect(),
            },
        }
    }

    /// A batch's label; `None` for any other op.
    pub fn label(&self) -> Option<&str> {
        match self {
            Op::Batch { label, .. } => Some(label),
            _ => None,
        }
    }

    /// Perform the op. A batch that fails part-way puts back what it had
    /// already done before reporting.
    pub fn apply(&self, world: &mut World, registry: &Registry) -> Result<(), EditError> {
        match self {
            Op::Batch { ops, .. } => {
                for (done, op) in ops.iter().enumerate() {
                    let Err(error) = op.apply(world, registry) else {
                        continue;
                    };
                    for op in ops[..done].iter().rev() {
                        // Each of these has just been applied, so its inverse
                        // describes the world exactly; failing here would be a
                        // bug in an op, not a stale step.
                        if let Err(undo) = op.clone().invert().apply(world, registry) {
                            tracing::error!("could not put back part of a failed step: {undo}");
                        }
                    }
                    return Err(error);
                }
                Ok(())
            }
            Op::SetField {
                entity,
                component,
                new,
                ..
            } => find(world, *entity)
                .and_then(|target| set_fields(world, registry, target, component, new))
                .map_err(|reason| EditError(format!("{component} on entity {entity}: {reason}"))),
            Op::AddComponent {
                entity,
                component,
                value,
            } => find(world, *entity)
                .and_then(|target| {
                    if persist::read_component(world, registry, target, component).is_some() {
                        return Err("it already has one".to_owned());
                    }
                    persist::write_component(world, registry, target, component, value)
                })
                .map_err(|reason| {
                    EditError(format!("adding {component} to entity {entity}: {reason}"))
                }),
            Op::RemoveComponent {
                entity, component, ..
            } => find(world, *entity)
                .and_then(|target| {
                    if persist::read_component(world, registry, target, component).is_none() {
                        return Err("it has none".to_owned());
                    }
                    persist::remove_component(world, registry, target, component);
                    Ok(())
                })
                .map_err(|reason| {
                    EditError(format!(
                        "removing {component} from entity {entity}: {reason}"
                    ))
                }),
            Op::Spawn { entities, slots } => {
                let live = persist::ids_of(world);
                if let Some(taken) = entities.iter().find(|entity| live.contains_key(&entity.id)) {
                    return Err(EditError(format!(
                        "spawning entity {}: one with that id is already in the scene",
                        taken.id
                    )));
                }
                let reserved: FxHashMap<EntityId, Entity> = slots
                    .iter()
                    .filter_map(|&(id, slot)| world.spawn_at(slot).map(|entity| (id, entity)))
                    .collect();
                for issue in persist::spawn(entities, world, registry, &reserved) {
                    tracing::warn!("{issue}");
                }
                // A member the prefab has dropped since is not spawned, and
                // its slot was reserved for nothing.
                for (_, entity) in reserved {
                    if !world.has::<EntityId>(entity) {
                        world.despawn(entity);
                    }
                }
                Ok(())
            }
            Op::Despawn { entities, .. } => {
                let live = persist::ids_of(world);
                let doomed = entities
                    .iter()
                    .map(|entity| {
                        live.get(&entity.id).copied().ok_or_else(|| {
                            EditError(format!(
                                "despawning entity {}: it is not in the scene",
                                entity.id
                            ))
                        })
                    })
                    .collect::<Result<Vec<Entity>, _>>()?;
                for entity in doomed {
                    despawn_recursive(world, entity);
                }
                Ok(())
            }
            Op::Reparent { entity, to, .. } => {
                let moved = find(world, *entity).and_then(|child| {
                    let parent = to.map(|parent| find(world, parent)).transpose()?;
                    reparent(world, child, parent, false).map_err(|error| error.to_string())
                });
                moved.map_err(|reason| EditError(format!("reparenting entity {entity}: {reason}")))
            }
            Op::WriteAsset { asset, new, .. } => {
                let Some(mut db) = world.get_resource_mut::<AssetDatabase>() else {
                    return Err(EditError(format!(
                        "writing asset {asset}: there is no asset database"
                    )));
                };
                db.write_asset(*asset, new).map_err(EditError)
            }
        }
    }
}

/// The live entity holding `id`.
///
/// A search rather than an index kept beside the world: ops are applied by
/// undo and redo, a keypress at a time, and a live edit already holds its
/// entity.
fn find(world: &World, id: EntityId) -> Result<Entity, String> {
    world
        .entities()
        .find(|&entity| {
            world
                .get::<EntityId>(entity)
                .is_some_and(|found| *found == id)
        })
        .ok_or_else(|| format!("no entity in the scene has the id {id}"))
}

/// Overwrite paths inside one component of `entity`, in its file form.
pub(super) fn set_fields(
    world: &mut World,
    registry: &Registry,
    entity: Entity,
    component: &ComponentId,
    changes: &[FieldChange],
) -> Result<(), String> {
    if let Some(vtable) = registry.get(component)
        && vtable.has(world, entity)
    {
        return vtable
            .apply(world, entity, changes)
            .map_err(|error| error.to_string());
    }
    let mut value = persist::read_component(world, registry, entity, component)
        .ok_or("the entity does not have it")?;
    orrin_registry::apply(&mut value, changes).map_err(|error| error.to_string())?;
    persist::write_component(world, registry, entity, component, &value)
}
