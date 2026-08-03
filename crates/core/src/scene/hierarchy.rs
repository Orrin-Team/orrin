//! Parent/child topology, derived from [`Parent`] links and cached.
//!
//! Two relations live over the same links. The *organizational* graph is every
//! link, and is what an editor tree draws and what a recursive despawn follows.
//! The *transform* graph is the subset whose parent actually has a transform,
//! and is what [`Hierarchy::propagate`] walks. Both come out of the single
//! `order` below, which is why it spans every live entity rather than only the
//! transformed ones.

use std::ops::Range;

use glam::Mat4;
use orrin_ecs::{Entity, World};

use super::{LocalTransform, LogBuffer, LogLevel, Parent, Time, WorldTransform};

/// Why a reparent was refused. Every variant names the entities involved,
/// because "cycle detected" without them is not actionable in a scene of a
/// thousand objects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HierarchyError {
    /// An entity cannot be its own parent.
    SelfParent { child: Entity },
    /// `parent` is already a descendant of `child`, so the link would close a
    /// loop with no root — and a loop propagates no transforms at all.
    Cycle { child: Entity, parent: Entity },
    /// The handle names an entity that has been despawned.
    StaleHandle { entity: Entity },
}

impl std::fmt::Display for HierarchyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SelfParent { child } => {
                write!(f, "entity {} cannot be its own parent", child.index())
            }
            Self::Cycle { child, parent } => write!(
                f,
                "parenting {} to {} would close a cycle: {} is already a descendant of {}",
                child.index(),
                parent.index(),
                parent.index(),
                child.index()
            ),
            Self::StaleHandle { entity } => {
                write!(f, "entity {} has been despawned", entity.index())
            }
        }
    }
}

/// The cached topology, plus the scratch buffers propagation walks.
///
/// Rebuilt only when [`World::structural_version`] moves, which is what makes
/// the per-frame cost a flat pass rather than a graph traversal.
#[derive(Default)]
pub struct Hierarchy {
    /// Every live entity, parents strictly before children. Roots first.
    order: Vec<Entity>,
    /// Parallel to `order`: the slice of `order` holding that entity's children.
    /// Contiguous because `order` is emitted breadth-first.
    child_ranges: Vec<Range<u32>>,
    /// Parallel to `order`: each entity's position in `order`, or `None` if it
    /// is a root.
    parents: Vec<Option<u32>>,
    /// `order[..root_count]` are exactly the roots.
    root_count: usize,
    /// Entity slot to position in `order`; `NOT_IN_ORDER` for absent slots.
    slot_to_index: Vec<u32>,
    /// The version `order` was derived from. `None` forces the first rebuild.
    built_at: Option<u64>,

    // Propagation scratch, parallel to `order` and reused across frames.
    locals: Vec<Mat4>,
    has_transform: Vec<bool>,
    world_matrices: Vec<Mat4>,
}

const NOT_IN_ORDER: u32 = u32::MAX;

/// The topology alone, with no world mutation — what both the rebuild and the
/// debug validator derive, so the validator cannot drift from the thing it
/// checks.
struct Topology {
    order: Vec<Entity>,
    child_ranges: Vec<Range<u32>>,
    parents: Vec<Option<u32>>,
    root_count: usize,
    slot_to_index: Vec<u32>,
    /// Live entities that no root reaches. Every one of them is in a cycle.
    unreachable: Vec<Entity>,
}

/// Group children under their parents with a counting sort, then emit
/// breadth-first.
///
/// The BFS order is what makes each entity's children a *contiguous* range of
/// `order`: popping a node pushes all of its children consecutively. That gives
/// the CSR layout for free, with no second structure to keep in step.
fn derive(world: &World) -> Topology {
    let entities: Vec<Entity> = world.entities().collect();
    let slots = entities
        .iter()
        .map(|e| e.index() as usize + 1)
        .max()
        .unwrap_or(0);

    // Pass 1: how many children each slot has, then where its run starts.
    let mut counts = vec![0u32; slots];
    for &entity in &entities {
        if let Some(parent) = live_parent(world, entity) {
            counts[parent.index() as usize] += 1;
        }
    }
    let mut offsets = vec![0u32; slots + 1];
    for slot in 0..slots {
        offsets[slot + 1] = offsets[slot] + counts[slot];
    }

    // Pass 2: scatter each child into its parent's run, and collect the roots.
    let mut cursor = offsets.clone();
    let mut grouped = vec![Entity::default(); entities.len()];
    let mut order = Vec::with_capacity(entities.len());
    for &entity in &entities {
        match live_parent(world, entity) {
            Some(parent) => {
                let slot = parent.index() as usize;
                grouped[cursor[slot] as usize] = entity;
                cursor[slot] += 1;
            }
            None => order.push(entity),
        }
    }
    let root_count = order.len();

    // Pass 3: breadth-first emit. `order` grows as it is walked, so this visits
    // every node exactly once and stops when no unvisited child remains.
    let mut child_ranges = Vec::with_capacity(entities.len());
    let mut parents: Vec<Option<u32>> = vec![None; root_count];
    let mut i = 0;
    while i < order.len() {
        let slot = order[i].index() as usize;
        let children = &grouped[offsets[slot] as usize..cursor[slot] as usize];
        let start = order.len() as u32;
        for &child in children {
            order.push(child);
            parents.push(Some(i as u32));
        }
        child_ranges.push(start..order.len() as u32);
        i += 1;
    }

    let mut slot_to_index = vec![NOT_IN_ORDER; slots];
    for (index, &entity) in order.iter().enumerate() {
        slot_to_index[entity.index() as usize] = index as u32;
    }

    // Anything the walk never reached has a parent but no root above it, which
    // is only possible inside a cycle.
    let unreachable = entities
        .iter()
        .copied()
        .filter(|e| slot_to_index[e.index() as usize] == NOT_IN_ORDER)
        .collect();

    Topology {
        order,
        child_ranges,
        parents,
        root_count,
        slot_to_index,
        unreachable,
    }
}

/// The entity's parent, if it has one that still exists.
///
/// A child of a despawned parent is treated as a root rather than as an error:
/// `World::despawn` drops components without touching relationships, so this is
/// the ordinary state of an orphan and not a corruption to report.
fn live_parent(world: &World, entity: Entity) -> Option<Entity> {
    let parent = world.get::<Parent>(entity)?.get();
    world.is_alive(parent).then_some(parent)
}

impl Hierarchy {
    /// Where `entity` sits in the propagation order, if it is in it at all.
    fn index_of(&self, entity: Entity) -> Option<usize> {
        let index = *self.slot_to_index.get(entity.index() as usize)?;
        (index != NOT_IN_ORDER).then_some(index as usize)
    }

    /// The direct children of `entity`, in a stable order.
    pub fn children_of(&self, entity: Entity) -> &[Entity] {
        match self.index_of(entity) {
            Some(index) => {
                let range = self.child_ranges[index].clone();
                &self.order[range.start as usize..range.end as usize]
            }
            None => &[],
        }
    }

    /// Every entity with no living parent.
    pub fn roots(&self) -> &[Entity] {
        &self.order[..self.root_count]
    }

    /// Parents strictly before children — the order a transform walk needs.
    pub fn order(&self) -> &[Entity] {
        &self.order
    }

    fn adopt(&mut self, topology: Topology, version: u64) {
        self.order = topology.order;
        self.child_ranges = topology.child_ranges;
        self.parents = topology.parents;
        self.root_count = topology.root_count;
        self.slot_to_index = topology.slot_to_index;
        self.built_at = Some(version);
    }
}

/// Re-derive the topology if the world's shape has moved since it was built,
/// and keep `WorldTransform` attached to exactly the entities that have a
/// `LocalTransform`.
///
/// Cycles are broken here rather than refused: they can only arrive from a
/// source that bypasses [`reparent`] — a scene file, most likely — and refusing
/// to build an order over one bad link would leave the whole scene unrendered.
fn rebuild(world: &mut World) {
    let topology = derive(world);

    if !topology.unreachable.is_empty() {
        // Clearing every member of a cycle is heavier than breaking one link,
        // and deliberately so: which link is "the wrong one" is unknowable here,
        // and detaching all of them is at least predictable.
        let broken = topology.unreachable.clone();
        for &entity in &broken {
            world.remove::<Parent>(entity);
        }
        report_cycle(world, &broken);
        // The retry cannot find another cycle: every entity that was in one is
        // now a root.
        let repaired = derive(world);
        debug_assert!(repaired.unreachable.is_empty());
        sync_world_transforms(world, &repaired.order);
        let version = world.structural_version();
        world.resource_mut::<Hierarchy>().adopt(repaired, version);
        return;
    }

    sync_world_transforms(world, &topology.order);
    // Read after the syncing inserts, which bump the version themselves —
    // otherwise the hierarchy would look stale the moment it was built and
    // rebuild on every frame.
    let version = world.structural_version();
    world.resource_mut::<Hierarchy>().adopt(topology, version);
}

/// Give every entity with a local transform a world transform, and take it away
/// from every entity without one.
///
/// The removal half matters: an entity demoted to an organizational node would
/// otherwise keep a frozen world matrix, and propagation reads exactly that
/// component to decide whether a child inherits — so it would go on moving its
/// children by a matrix nothing updates.
fn sync_world_transforms(world: &mut World, order: &[Entity]) {
    let mut add = Vec::new();
    let mut drop = Vec::new();
    for &entity in order {
        match (
            world.has::<LocalTransform>(entity),
            world.has::<WorldTransform>(entity),
        ) {
            (true, false) => add.push(entity),
            (false, true) => drop.push(entity),
            _ => {}
        }
    }
    for entity in add {
        world.insert(entity, WorldTransform::default());
    }
    for entity in drop {
        world.remove::<WorldTransform>(entity);
    }
}

fn report_cycle(world: &World, broken: &[Entity]) {
    let Some(mut log) = world.get_resource_mut::<LogBuffer>() else {
        return;
    };
    let frame = world
        .get_resource::<Time>()
        .map(|time| time.frame_count())
        .unwrap_or(0);
    let names: Vec<String> = broken.iter().map(|e| e.index().to_string()).collect();
    log.push(
        LogLevel::Error,
        format!(
            "parent cycle detected; detached {} entit{} to break it: {}",
            broken.len(),
            if broken.len() == 1 { "y" } else { "ies" },
            names.join(", ")
        ),
        frame,
    );
}

/// Bring the cached hierarchy up to date with the world, rebuilding only if the
/// world's shape has changed since it was last derived.
pub fn ensure_current(world: &mut World) {
    if world.get_resource::<Hierarchy>().is_none() {
        world.insert_resource(Hierarchy::default());
    }
    let built_at = world.resource::<Hierarchy>().built_at;
    if built_at != Some(world.structural_version()) {
        rebuild(world);
    }

    #[cfg(debug_assertions)]
    assert_cache_is_current(world);
}

/// Prove the version counter is bumped everywhere it needs to be.
///
/// A rebuild is skipped whenever the version has not moved, so a mutation path
/// that changes the topology without bumping it would leave the cache stale with
/// no symptom until something read a transform from the wrong parent. Re-deriving
/// and comparing turns that into a panic on the frame the bug is introduced.
#[cfg(debug_assertions)]
fn assert_cache_is_current(world: &World) {
    let fresh = derive(world);
    let cached = world.resource::<Hierarchy>();
    debug_assert_eq!(
        fresh.order, cached.order,
        "hierarchy cache is stale: the world's topology changed without \
         World::structural_version moving"
    );
    debug_assert_eq!(fresh.parents, cached.parents);
    debug_assert_eq!(fresh.child_ranges, cached.child_ranges);
}

impl Hierarchy {
    /// Walk the cached order once, composing each entity's world transform from
    /// its parent's.
    ///
    /// A single forward pass suffices because `order` puts every parent before
    /// its children, so a parent's world matrix is always already written by the
    /// time a child reads it.
    ///
    /// Three phases rather than one: a gather and a scatter over the dense
    /// component storages, with a pure walk in between. Reading each parent's
    /// transform back out of the world per entity would instead pay a hash
    /// lookup and a `RefCell` borrow per link.
    fn propagate(&mut self, world: &World) {
        let count = self.order.len();
        self.locals.clear();
        self.locals.resize(count, Mat4::IDENTITY);
        self.has_transform.clear();
        self.has_transform.resize(count, false);
        self.world_matrices.clear();
        self.world_matrices.resize(count, Mat4::IDENTITY);

        world.query::<&LocalTransform>().for_each(|entity, local| {
            if let Some(index) = self.index_of(entity) {
                self.locals[index] = local.matrix();
                self.has_transform[index] = true;
            }
        });

        for index in 0..count {
            if !self.has_transform[index] {
                continue;
            }
            // An entity whose parent has no transform is a transform root: the
            // parent is an organizational node with no position to move it by.
            // The chain breaks there rather than reaching past it to a
            // transformed grandparent, so a folder never moves its contents by a
            // rotation applied two levels up and invisible in between.
            self.world_matrices[index] = match self.parents[index] {
                Some(parent) if self.has_transform[parent as usize] => {
                    self.world_matrices[parent as usize] * self.locals[index]
                }
                _ => self.locals[index],
            };
        }

        world
            .query::<&mut WorldTransform>()
            .for_each(|entity, world_transform| {
                if let Some(index) = self.index_of(entity) {
                    world_transform.0 = self.world_matrices[index];
                }
            });
    }
}

/// Derive every entity's [`WorldTransform`] from its [`LocalTransform`] and its
/// ancestors'.
pub fn propagate_transforms(world: &mut World) {
    ensure_current(world);
    let mut hierarchy = world.resource_mut::<Hierarchy>();
    hierarchy.propagate(world);
}

/// Whether `entity`'s local transform is already its world transform.
///
/// True for an entity with no parent, one whose parent has been despawned, and
/// one whose parent is an organizational node with no transform of its own —
/// the three cases propagation treats alike. Anything else holds a
/// parent-relative local transform, so a world-space quantity cannot be written
/// into it without first being converted.
pub fn is_transform_root(world: &World, entity: Entity) -> bool {
    match live_parent(world, entity) {
        Some(parent) => !world.has::<WorldTransform>(parent),
        None => true,
    }
}

/// Despawn `entity` together with everything beneath it, and report how many
/// entities went.
///
/// Deleting a parent deletes its subtree, as it does in Unity and Godot. The
/// alternative — orphaning the children to roots — leaves objects floating at
/// their last offset relative to something that no longer exists, which is
/// never what anyone means by deleting the parent.
///
/// The subtree is collected in full before anything is despawned. Traversing
/// links while dropping the components that hold them would be reading a graph
/// as it is dismantled.
pub fn despawn_recursive(world: &mut World, entity: Entity) -> usize {
    if !world.is_alive(entity) {
        return 0;
    }
    ensure_current(world);

    let doomed = {
        let hierarchy = world.resource::<Hierarchy>();
        let mut doomed = vec![entity];
        let mut i = 0;
        while i < doomed.len() {
            doomed.extend_from_slice(hierarchy.children_of(doomed[i]));
            i += 1;
        }
        doomed
    };

    doomed
        .iter()
        .filter(|&&entity| world.despawn(entity))
        .count()
}

/// Attach `child` to `parent`, or detach it when `parent` is `None`.
///
/// With `keep_world`, the child's local transform is rewritten so that its world
/// transform is unchanged by the move — what a drag-and-drop in an editor wants,
/// where an object should not leap across the scene when it gains a parent. A
/// script attaching a pickup to a hand wants the opposite.
///
/// Two caveats on `keep_world`, both inherent rather than incidental:
///
/// - It reads the child's current `WorldTransform`, which is only as fresh as
///   the last propagation. A reparent issued after something moved the parent
///   this frame composes against the parent's previous position.
/// - The result is written back into a `LocalTransform`, which is a
///   translation/rotation/scale triple. That is exact when the new parent's
///   transform is a rotation and a uniform scale, and a best fit when it is not,
///   because shear has no TRS spelling.
pub fn reparent(
    world: &mut World,
    child: Entity,
    parent: Option<Entity>,
    keep_world: bool,
) -> Result<(), HierarchyError> {
    can_reparent(world, child, parent)?;

    let old_world = keep_world
        .then(|| world.get::<WorldTransform>(child).map(|w| w.0))
        .flatten();

    match parent {
        Some(parent) => world.insert(child, Parent::new(parent)),
        None => world.remove::<Parent>(child),
    };

    if let Some(old_world) = old_world {
        let new_local = parent_world_matrix(world, child).inverse() * old_world;
        if let Some(mut local) = world.get_mut::<LocalTransform>(child) {
            let (scale, rotation, translation) = new_local.to_scale_rotation_translation();
            local.translation = translation;
            local.rotation = rotation;
            local.scale = scale;
        }
    }

    Ok(())
}

/// Whether [`reparent`] would accept this move, without performing it.
///
/// Split out for the scripting ABI, where a reparent is a structural change and
/// so has to be queued until the dispatch window closes. Checking here keeps the
/// call's return value meaningful — a script learns immediately that it asked
/// for a cycle, instead of finding out by the move silently not happening.
pub fn can_reparent(
    world: &World,
    child: Entity,
    parent: Option<Entity>,
) -> Result<(), HierarchyError> {
    if !world.is_alive(child) {
        return Err(HierarchyError::StaleHandle { entity: child });
    }
    let Some(parent) = parent else {
        return Ok(());
    };
    if !world.is_alive(parent) {
        return Err(HierarchyError::StaleHandle { entity: parent });
    }
    if parent == child {
        return Err(HierarchyError::SelfParent { child });
    }
    if is_descendant_of(world, parent, child) {
        return Err(HierarchyError::Cycle { child, parent });
    }
    Ok(())
}

/// The matrix `entity`'s local transform is interpreted relative to — identity
/// for a transform root.
///
/// This is the conversion between the two spaces, so anything holding a
/// world-space quantity that has to land in a `LocalTransform` goes through it.
pub fn parent_world_matrix(world: &World, entity: Entity) -> Mat4 {
    live_parent(world, entity)
        .and_then(|parent| world.get::<WorldTransform>(parent).map(|w| w.0))
        .unwrap_or(Mat4::IDENTITY)
}

/// Whether `entity` is `ancestor` or sits anywhere beneath it.
///
/// Walks up from `entity` rather than down from `ancestor`, so it costs the
/// depth of one chain and needs no cached topology — which matters because the
/// cache is stale by definition in the middle of a batch of reparents.
fn is_descendant_of(world: &World, entity: Entity, ancestor: Entity) -> bool {
    let mut current = entity;
    // The world should hold no cycle, but this runs *before* the link that would
    // create one is written, and a scene file could have supplied another. The
    // bound makes a corrupt world return an answer instead of hanging.
    for _ in 0..=world.entities().count() {
        if current == ancestor {
            return true;
        }
        match live_parent(world, current) {
            Some(parent) => current = parent,
            None => return false,
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Transform;
    use glam::{Quat, Vec3};

    fn spawn_at(world: &mut World, transform: Transform) -> Entity {
        world
            .spawn_entity()
            .with(LocalTransform::from(transform))
            .id()
    }

    fn spawn_bare(world: &mut World) -> Entity {
        world.spawn()
    }

    fn world_of(world: &World, entity: Entity) -> Mat4 {
        world.get::<WorldTransform>(entity).unwrap().0
    }

    fn close(a: Mat4, b: Mat4) -> bool {
        a.to_cols_array()
            .iter()
            .zip(b.to_cols_array().iter())
            .all(|(x, y)| (x - y).abs() < 1e-4)
    }

    /// The composition test, three deep with a non-uniformly scaled ancestor and
    /// a rotated child — the arrangement whose product contains shear, and so
    /// the one that fails if `WorldTransform` is ever "simplified" back into a
    /// translation/rotation/scale triple.
    #[test]
    fn a_chain_composes_in_order() {
        let root_t = Transform {
            translation: Vec3::new(1.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(2.0, 1.0, 1.0),
        };
        let child_t = Transform {
            translation: Vec3::new(0.0, 1.0, 0.0),
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            scale: Vec3::ONE,
        };
        let grandchild_t = Transform::from_translation(Vec3::new(0.0, 0.0, 1.0));

        let mut world = World::new();
        let root = spawn_at(&mut world, root_t);
        let child = spawn_at(&mut world, child_t);
        let grandchild = spawn_at(&mut world, grandchild_t);
        reparent(&mut world, child, Some(root), false).unwrap();
        reparent(&mut world, grandchild, Some(child), false).unwrap();

        propagate_transforms(&mut world);

        assert!(close(world_of(&world, root), root_t.matrix()));
        assert!(close(
            world_of(&world, child),
            root_t.matrix() * child_t.matrix()
        ));
        assert!(close(
            world_of(&world, grandchild),
            root_t.matrix() * child_t.matrix() * grandchild_t.matrix()
        ));
    }

    /// The invariant the single forward pass rests on.
    #[test]
    fn the_order_puts_every_parent_before_its_children() {
        let mut world = World::new();
        let a = spawn_at(&mut world, Transform::default());
        let b = spawn_at(&mut world, Transform::default());
        let c = spawn_at(&mut world, Transform::default());
        let d = spawn_at(&mut world, Transform::default());
        reparent(&mut world, c, Some(a), false).unwrap();
        reparent(&mut world, d, Some(c), false).unwrap();
        reparent(&mut world, b, Some(d), false).unwrap();

        propagate_transforms(&mut world);

        let hierarchy = world.resource::<Hierarchy>();
        let position = |e: Entity| hierarchy.order().iter().position(|&x| x == e).unwrap();
        assert!(position(a) < position(c));
        assert!(position(c) < position(d));
        assert!(position(d) < position(b));
    }

    /// The CSR layout the breadth-first emit produces: a node's children are a
    /// contiguous slice of the order, with no separate structure holding them.
    #[test]
    fn each_entitys_children_are_contiguous_in_the_order() {
        let mut world = World::new();
        let root = spawn_at(&mut world, Transform::default());
        let kids: Vec<Entity> = (0..3)
            .map(|_| spawn_at(&mut world, Transform::default()))
            .collect();
        for &kid in &kids {
            reparent(&mut world, kid, Some(root), false).unwrap();
        }

        propagate_transforms(&mut world);

        let hierarchy = world.resource::<Hierarchy>();
        let mut found = hierarchy.children_of(root).to_vec();
        found.sort_by_key(|e| e.index());
        let mut expected = kids.clone();
        expected.sort_by_key(|e| e.index());
        assert_eq!(found, expected);
        assert!(hierarchy.children_of(kids[0]).is_empty());
    }

    /// Cycles can only arrive from something that bypasses `reparent` — a scene
    /// file, most likely. Refusing to build an order over one bad link would
    /// leave the entire scene unrendered, so the cycle is broken instead.
    #[test]
    fn a_cycle_is_broken_and_the_rest_of_the_scene_still_propagates() {
        let mut world = World::new();
        world.insert_resource(LogBuffer::default());
        let a = spawn_at(&mut world, Transform::from_translation(Vec3::X));
        let b = spawn_at(&mut world, Transform::from_translation(Vec3::Y));
        let bystander = spawn_at(&mut world, Transform::from_translation(Vec3::Z));
        // Straight past `reparent`, exactly as a hand-edited file would.
        world.insert(a, Parent::new(b));
        world.insert(b, Parent::new(a));

        propagate_transforms(&mut world);

        assert!(world.get::<Parent>(a).is_none());
        assert!(world.get::<Parent>(b).is_none());
        assert!(close(
            world_of(&world, bystander),
            Mat4::from_translation(Vec3::Z)
        ));
        let log = world.resource::<LogBuffer>();
        assert_eq!(log.len(), 1);
        assert_eq!(log.iter().next().unwrap().level, LogLevel::Error);
    }

    #[test]
    fn a_three_entity_cycle_is_broken_too() {
        let mut world = World::new();
        let a = spawn_at(&mut world, Transform::default());
        let b = spawn_at(&mut world, Transform::default());
        let c = spawn_at(&mut world, Transform::default());
        world.insert(a, Parent::new(b));
        world.insert(b, Parent::new(c));
        world.insert(c, Parent::new(a));

        propagate_transforms(&mut world);

        assert_eq!(world.resource::<Hierarchy>().roots().len(), 3);
    }

    #[test]
    fn an_entity_cannot_be_its_own_parent() {
        let mut world = World::new();
        let a = spawn_at(&mut world, Transform::default());

        assert_eq!(
            reparent(&mut world, a, Some(a), false),
            Err(HierarchyError::SelfParent { child: a })
        );
        assert!(world.get::<Parent>(a).is_none());
    }

    #[test]
    fn parenting_an_ancestor_to_its_own_descendant_is_refused() {
        let mut world = World::new();
        let root = spawn_at(&mut world, Transform::default());
        let child = spawn_at(&mut world, Transform::default());
        reparent(&mut world, child, Some(root), false).unwrap();

        assert_eq!(
            reparent(&mut world, root, Some(child), false),
            Err(HierarchyError::Cycle {
                child: root,
                parent: child
            })
        );
        assert!(world.get::<Parent>(root).is_none());
    }

    #[test]
    fn reparenting_a_despawned_entity_is_refused() {
        let mut world = World::new();
        let ghost = spawn_at(&mut world, Transform::default());
        world.despawn(ghost);

        assert_eq!(
            reparent(&mut world, ghost, None, false),
            Err(HierarchyError::StaleHandle { entity: ghost })
        );
    }

    /// What a drag-and-drop reparent in an editor needs: gaining a parent must
    /// not make the object leap across the scene.
    #[test]
    fn keep_world_leaves_the_child_where_it_was() {
        let mut world = World::new();
        let parent = spawn_at(
            &mut world,
            Transform {
                translation: Vec3::new(5.0, 2.0, -3.0),
                rotation: Quat::from_rotation_y(0.9),
                scale: Vec3::splat(2.0),
            },
        );
        let child = spawn_at(
            &mut world,
            Transform::from_translation(Vec3::new(1.0, 1.0, 1.0)),
        );
        propagate_transforms(&mut world);
        let before = world_of(&world, child);

        reparent(&mut world, child, Some(parent), true).unwrap();
        propagate_transforms(&mut world);

        assert!(close(world_of(&world, child), before));
    }

    /// The counterpart: without the flag the child snaps into its parent's
    /// frame, which is what a script attaching a pickup to a hand wants.
    #[test]
    fn without_keep_world_the_child_moves_into_its_parents_frame() {
        let mut world = World::new();
        let parent = spawn_at(
            &mut world,
            Transform::from_translation(Vec3::new(10.0, 0.0, 0.0)),
        );
        let child = spawn_at(
            &mut world,
            Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        );

        reparent(&mut world, child, Some(parent), false).unwrap();
        propagate_transforms(&mut world);

        assert!(close(
            world_of(&world, child),
            Mat4::from_translation(Vec3::new(11.0, 0.0, 0.0))
        ));
    }

    /// Decision 7: an entity with no transform is an organizational node, and
    /// the transform chain breaks there rather than reaching past it. A folder
    /// must never move its contents by a rotation applied two levels up.
    #[test]
    fn a_transformless_parent_does_not_move_its_children() {
        let mut world = World::new();
        let grandparent = spawn_at(
            &mut world,
            Transform::from_translation(Vec3::new(100.0, 0.0, 0.0)),
        );
        let folder = spawn_bare(&mut world);
        let child = spawn_at(
            &mut world,
            Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        );
        reparent(&mut world, folder, Some(grandparent), false).unwrap();
        reparent(&mut world, child, Some(folder), false).unwrap();

        propagate_transforms(&mut world);

        assert!(world.get::<WorldTransform>(folder).is_none());
        assert!(
            close(
                world_of(&world, child),
                Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))
            ),
            "the child inherited through a folder node"
        );
    }

    /// The removal half of the world-transform sync. Without it a demoted parent
    /// keeps a frozen matrix, and propagation reads exactly that component to
    /// decide whether a child inherits.
    #[test]
    fn losing_a_local_transform_drops_the_world_transform_and_frees_the_children() {
        let mut world = World::new();
        let parent = spawn_at(
            &mut world,
            Transform::from_translation(Vec3::new(50.0, 0.0, 0.0)),
        );
        let child = spawn_at(&mut world, Transform::from_translation(Vec3::X));
        reparent(&mut world, child, Some(parent), false).unwrap();
        propagate_transforms(&mut world);
        assert!(world.get::<WorldTransform>(parent).is_some());

        world.remove::<LocalTransform>(parent);
        propagate_transforms(&mut world);

        assert!(world.get::<WorldTransform>(parent).is_none());
        assert!(close(
            world_of(&world, child),
            Mat4::from_translation(Vec3::X)
        ));
    }

    /// `World::despawn` drops components without knowing about relationships, so
    /// an orphan is an ordinary state rather than a corruption. It becomes a root.
    #[test]
    fn a_child_of_a_despawned_parent_becomes_a_root() {
        let mut world = World::new();
        let parent = spawn_at(
            &mut world,
            Transform::from_translation(Vec3::new(7.0, 0.0, 0.0)),
        );
        let child = spawn_at(&mut world, Transform::from_translation(Vec3::X));
        reparent(&mut world, child, Some(parent), false).unwrap();
        propagate_transforms(&mut world);

        world.despawn(parent);
        propagate_transforms(&mut world);

        assert!(close(
            world_of(&world, child),
            Mat4::from_translation(Vec3::X)
        ));
        assert!(world.resource::<Hierarchy>().roots().contains(&child));
    }

    /// The trap this design had to be built around: the rebuild itself inserts
    /// and removes `WorldTransform`, which bumps the very counter it uses to
    /// decide whether to rebuild. Reading the version *after* those writes is
    /// what stops the hierarchy rebuilding on every frame forever.
    #[test]
    fn a_settled_world_does_not_rebuild() {
        let mut world = World::new();
        let root = spawn_at(&mut world, Transform::default());
        let child = spawn_at(&mut world, Transform::default());
        reparent(&mut world, child, Some(root), false).unwrap();

        propagate_transforms(&mut world);
        let settled = world.structural_version();
        propagate_transforms(&mut world);
        propagate_transforms(&mut world);

        assert_eq!(
            world.structural_version(),
            settled,
            "propagation is still reshaping the world after it has settled"
        );
    }

    /// Propagation runs twice a frame and must stay a pure function of the local
    /// transforms, never accumulating onto what it wrote last time.
    #[test]
    fn propagating_twice_lands_in_the_same_place() {
        let mut world = World::new();
        let root = spawn_at(
            &mut world,
            Transform::from_translation(Vec3::new(2.0, 0.0, 0.0)),
        );
        let child = spawn_at(&mut world, Transform::from_translation(Vec3::X));
        reparent(&mut world, child, Some(root), false).unwrap();

        propagate_transforms(&mut world);
        let once = world_of(&world, child);
        propagate_transforms(&mut world);

        assert_eq!(once, world_of(&world, child));
    }

    /// Exercises the debug validator, which re-derives the topology and compares
    /// it against the cache on every `ensure_current`. If any of these operations
    /// reshaped the world without moving the version counter, this panics.
    #[test]
    fn a_long_sequence_of_edits_keeps_the_cache_in_step() {
        let mut world = World::new();
        let mut entities: Vec<Entity> = (0..12)
            .map(|i| {
                spawn_at(
                    &mut world,
                    Transform::from_translation(Vec3::splat(i as f32)),
                )
            })
            .collect();

        for i in 1..entities.len() {
            let _ = reparent(&mut world, entities[i], Some(entities[i / 2]), i % 2 == 0);
            propagate_transforms(&mut world);
        }
        for i in (0..entities.len()).step_by(3) {
            let _ = reparent(&mut world, entities[i], None, true);
            propagate_transforms(&mut world);
        }
        for i in (0..entities.len()).step_by(5) {
            world.despawn(entities[i]);
            propagate_transforms(&mut world);
        }
        entities.retain(|&e| world.is_alive(e));

        let hierarchy = world.resource::<Hierarchy>();
        assert_eq!(hierarchy.order().len(), entities.len());
    }

    /// Deleting a parent deletes its subtree, as it does in Unity and Godot.
    /// Orphaning the children instead would leave them floating at their last
    /// offset relative to something that no longer exists.
    #[test]
    fn despawning_a_parent_takes_its_whole_subtree() {
        let mut world = World::new();
        let root = spawn_at(&mut world, Transform::default());
        let child = spawn_at(&mut world, Transform::default());
        let grandchild = spawn_at(&mut world, Transform::default());
        let sibling = spawn_at(&mut world, Transform::default());
        reparent(&mut world, child, Some(root), false).unwrap();
        reparent(&mut world, grandchild, Some(child), false).unwrap();

        assert_eq!(despawn_recursive(&mut world, root), 3);

        assert!(!world.is_alive(root));
        assert!(!world.is_alive(child));
        assert!(!world.is_alive(grandchild));
        assert!(world.is_alive(sibling), "an unrelated entity was taken too");
    }

    /// A folder node has no transform, but it is still a parent — deleting it
    /// has to take its contents, or the organizational graph and the delete
    /// would disagree.
    #[test]
    fn despawning_a_transformless_parent_still_takes_its_children() {
        let mut world = World::new();
        let folder = spawn_bare(&mut world);
        let child = spawn_at(&mut world, Transform::default());
        reparent(&mut world, child, Some(folder), false).unwrap();

        assert_eq!(despawn_recursive(&mut world, folder), 2);
        assert!(!world.is_alive(child));
    }

    #[test]
    fn despawning_a_leaf_takes_only_itself() {
        let mut world = World::new();
        let root = spawn_at(&mut world, Transform::default());
        let child = spawn_at(&mut world, Transform::default());
        reparent(&mut world, child, Some(root), false).unwrap();

        assert_eq!(despawn_recursive(&mut world, child), 1);
        assert!(world.is_alive(root));
    }

    #[test]
    fn despawning_a_stale_handle_does_nothing() {
        let mut world = World::new();
        let ghost = spawn_at(&mut world, Transform::default());
        world.despawn(ghost);

        assert_eq!(despawn_recursive(&mut world, ghost), 0);
    }

    /// The three cases propagation treats alike, which is what makes this the
    /// right test for "is this entity's local transform already its world one".
    #[test]
    fn transform_roots_are_the_unparented_the_orphaned_and_the_foldered() {
        let mut world = World::new();
        let parent = spawn_at(&mut world, Transform::default());
        let folder = spawn_bare(&mut world);
        let unparented = spawn_at(&mut world, Transform::default());
        let orphan = spawn_at(&mut world, Transform::default());
        let in_folder = spawn_at(&mut world, Transform::default());
        let real_child = spawn_at(&mut world, Transform::default());
        let doomed = spawn_at(&mut world, Transform::default());
        reparent(&mut world, orphan, Some(doomed), false).unwrap();
        reparent(&mut world, in_folder, Some(folder), false).unwrap();
        reparent(&mut world, real_child, Some(parent), false).unwrap();
        propagate_transforms(&mut world);
        world.despawn(doomed);

        assert!(is_transform_root(&world, unparented));
        assert!(is_transform_root(&world, orphan), "orphaned by a despawn");
        assert!(is_transform_root(&world, in_folder), "under a folder node");
        assert!(!is_transform_root(&world, real_child));
    }

    /// The scripting ABI validates with `can_reparent` and applies with
    /// `reparent` a tick later. If the two ever disagree, a script gets told
    /// "yes" and then silently nothing happens.
    #[test]
    fn can_reparent_agrees_with_reparent() {
        let mut world = World::new();
        let root = spawn_at(&mut world, Transform::default());
        let child = spawn_at(&mut world, Transform::default());
        reparent(&mut world, child, Some(root), false).unwrap();
        let ghost = spawn_at(&mut world, Transform::default());
        world.despawn(ghost);

        let cases = [
            (root, Some(child)), // cycle
            (root, Some(root)),  // self
            (ghost, None),       // stale child
            (root, Some(ghost)), // stale parent
            (child, None),       // legal
            (root, Some(child)), // cycle again, after the legal one
        ];
        for (child_arg, parent_arg) in cases {
            let predicted = can_reparent(&world, child_arg, parent_arg);
            let actual = reparent(&mut world, child_arg, parent_arg, false);
            assert_eq!(predicted, actual, "for {child_arg:?} -> {parent_arg:?}");
        }
    }

    /// `set_world_transform` divides out this matrix to get a local transform.
    /// The round trip is the property it needs: place an object in world space,
    /// propagate, and find it exactly where it was asked to be.
    #[test]
    fn dividing_out_the_parent_matrix_round_trips() {
        let mut world = World::new();
        let parent = spawn_at(
            &mut world,
            Transform {
                translation: Vec3::new(-3.0, 4.0, 1.0),
                rotation: Quat::from_rotation_y(1.1),
                scale: Vec3::splat(2.0),
            },
        );
        let child = spawn_at(&mut world, Transform::default());
        reparent(&mut world, child, Some(parent), false).unwrap();
        propagate_transforms(&mut world);

        let desired = Mat4::from_scale_rotation_translation(
            Vec3::ONE,
            Quat::from_rotation_x(0.4),
            Vec3::new(7.0, -2.0, 5.0),
        );
        let local = parent_world_matrix(&world, child).inverse() * desired;
        {
            let (scale, rotation, translation) = local.to_scale_rotation_translation();
            let mut t = world.get_mut::<LocalTransform>(child).unwrap();
            t.translation = translation;
            t.rotation = rotation;
            t.scale = scale;
        }
        propagate_transforms(&mut world);

        assert!(close(world_of(&world, child), desired));
    }

    /// A transform root divides by identity, so the same code path places an
    /// unparented entity without a special case.
    #[test]
    fn a_transform_root_has_an_identity_parent_matrix() {
        let mut world = World::new();
        let loose = spawn_at(&mut world, Transform::default());
        let folder = spawn_bare(&mut world);
        let in_folder = spawn_at(&mut world, Transform::default());
        reparent(&mut world, in_folder, Some(folder), false).unwrap();
        propagate_transforms(&mut world);

        assert_eq!(parent_world_matrix(&world, loose), Mat4::IDENTITY);
        assert_eq!(parent_world_matrix(&world, in_folder), Mat4::IDENTITY);
    }
}
