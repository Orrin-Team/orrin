//! `orrin-ecs` - a small, dependency-free Entity Component System for game engines

// ORRIN-ECS
// AUTHOR: @AlternativeLua

#![forbid(unsafe_code)]

use std::any::{Any, TypeId};
use std::cell::{Ref, RefCell, RefMut};
use std::marker::PhantomData;

mod hash;

pub use hash::{FxBuildHasher, FxHashMap, FxHasher};

//
// Entities
//

/// A small, copyable handle to a single entity.
///
/// An entity is identified by a slot `index` plus a `generation`. When a slot
/// is reused by a later entity the generation changes, so a stale handle to a
/// despawned entity can be detected instead of silently aliasing the new one.
///
/// Slot 0 is reserved and never allocated, so `Entity::default()` — index 0,
/// generation 0 — names no live entity and never will. That makes the
/// all-zeroes handle usable as a null sentinel across the FFI boundary, where a
/// C# `Entity` field left uninitialized is exactly those bytes.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Debug)]
pub struct Entity {
    /// Index of the storage slot this entity occupies. Never 0 for a live
    /// entity; see the type-level docs.
    pub index: u32,
    /// How many times this slot has been reused; bumped on every despawn.
    pub generation: u32,
}

impl Entity {
    /// The storage slot this entity occupies.
    #[inline]
    pub fn index(self) -> u32 {
        self.index
    }

    /// The generation stamp, used to tell a live handle from a stale one.
    #[inline]
    pub fn generation(self) -> u32 {
        self.generation
    }
}

// Hands out entity ids and recycles the indices
struct EntityAllocator {
    generations: Vec<u32>,
    alive: Vec<bool>,
    free: Vec<u32>,
}

/// Seeds slot 0 as permanently dead, which is why this is hand-written —
/// **do not collapse it back into `#[derive(Default)]`**.
///
/// `allocate` takes the next fresh index from `generations.len()`, so starting
/// at length 1 never hands out `{0, 0}`; `deallocate` refuses a slot that isn't
/// alive, so 0 never reaches the free list either.
///
/// The payoff is across the FFI boundary: C# cannot customize a struct's
/// `default`, so an unassigned `Orrin.Entity` field is always all-zeroes.
/// Without this it aliases the first entity ever spawned and silently mutates
/// it; with it, `is_alive` is false and every lookup misses.
impl Default for EntityAllocator {
    fn default() -> Self {
        EntityAllocator {
            generations: vec![0],
            alive: vec![false],
            free: Vec::new(),
        }
    }
}

impl EntityAllocator {
    fn allocate(&mut self) -> Entity {
        if let Some(index) = self.free.pop() {
            self.alive[index as usize] = true;
            Entity {
                index,
                generation: self.generations[index as usize],
            }
        } else {
            let index = self.generations.len() as u32;
            self.generations.push(0);
            self.alive.push(true);
            Entity {
                index,
                generation: 0,
            }
        }
    }

    fn deallocate(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }

        let i = entity.index() as usize;
        self.generations[i] = self.generations[i].wrapping_add(1);
        self.alive[i] = false;
        self.free.push(entity.index);
        true
    }

    fn is_alive(&self, entity: Entity) -> bool {
        let i = entity.index as usize;
        i < self.generations.len() && self.alive[i] && self.generations[i] == entity.generation
    }

    /// Iterate over every currently-live entity, skipping freed slots.
    fn iter_alive(&self) -> impl Iterator<Item = Entity> + '_ {
        self.alive
            .iter()
            .enumerate()
            .filter_map(move |(i, &alive)| {
                alive.then(|| Entity {
                    index: i as u32,
                    generation: self.generations[i],
                })
            })
    }
}

//
// Component Storage
//

const SENTINEL: u32 = u32::MAX;

/// Dense storage for a single component type `T`.
///
/// `sparse` maps an entity index to a slot in the packed `dense_*` arrays, and
/// `dense_entities` maps back the other way so lookups can run a generation
/// check. Keeping the values packed means iteration never walks empty holes.
pub struct SparseSet<T> {
    sparse: Vec<u32>,
    dense_entities: Vec<Entity>,
    dense_values: Vec<T>,
}

impl<T> SparseSet<T> {
    fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense_entities: Vec::new(),
            dense_values: Vec::new(),
        }
    }

    fn dense_index(&self, entity: Entity) -> Option<usize> {
        let i = entity.index() as usize;
        let d = *self.sparse.get(i)?;
        if d == SENTINEL {
            return None;
        }

        if self.dense_entities[d as usize] == entity {
            Some(d as usize)
        } else {
            None
        }
    }

    fn insert(&mut self, entity: Entity, value: T) -> Option<T> {
        let i = entity.index as usize;
        if i >= self.sparse.len() {
            self.sparse.resize(i + 1, SENTINEL);
        }
        let d = self.sparse[i];
        if d != SENTINEL && self.dense_entities[d as usize] == entity {
            return Some(std::mem::replace(&mut self.dense_values[d as usize], value));
        }
        self.sparse[i] = self.dense_values.len() as u32;
        self.dense_entities.push(entity);
        self.dense_values.push(value);
        None
    }

    fn remove(&mut self, entity: Entity) -> Option<T> {
        let d = self.dense_index(entity)?;
        let last = self.dense_values.len() - 1;
        self.dense_values.swap(d, last);
        self.dense_entities.swap(d, last);
        let moved = self.dense_entities[d];
        self.sparse[moved.index as usize] = d as u32;
        self.sparse[entity.index as usize] = SENTINEL;
        self.dense_entities.pop();
        self.dense_values.pop()
    }

    fn get(&self, entity: Entity) -> Option<&T> {
        self.dense_index(entity).map(|d| &self.dense_values[d])
    }

    fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        match self.dense_index(entity) {
            Some(d) => Some(&mut self.dense_values[d]),
            None => None,
        }
    }
}

trait AnyStorage: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn remove_entity(&mut self, entity: Entity);
}

impl<T: 'static> AnyStorage for SparseSet<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn remove_entity(&mut self, entity: Entity) {
        self.remove(entity);
    }
}

//
// World
//

/// The container for all entities, their components, and global resources.
///
/// Almost everything goes through a `World`: [`spawn`](World::spawn) creates
/// entities, [`insert`](World::insert) attaches components, and
/// [`query`](World::query) iterates over them.
#[derive(Default)]
pub struct World {
    entities: EntityAllocator,
    // Keyed by `TypeId` and hashed with [`FxHasher`] rather than SipHash: every
    // `get`/`get_mut`/`has` pays this hash, and the scripting FFI runs them per
    // component per entity per frame.
    storages: FxHashMap<TypeId, RefCell<Box<dyn AnyStorage>>>,
    resources: FxHashMap<TypeId, RefCell<Box<dyn Any>>>,
    structural_version: u64,
}

impl World {
    /// Create an empty world.
    pub fn new() -> Self {
        Self::default()
    }

    /// A counter bumped by every change to the world's *shape*: an entity
    /// spawned or despawned, a component attached or detached.
    ///
    /// Mutating a component through [`get_mut`](World::get_mut) or a `&mut`
    /// query deliberately does **not** bump it — that changes a value, not the
    /// shape, and counting it would defeat the purpose.
    ///
    /// This exists so a cache derived from the world's structure can tell in
    /// O(1) whether it is still valid. The alternative — a dirty flag that every
    /// mutation site must remember to set — is only as good as the discipline of
    /// the next call site added, and a missed one is a stale cache with no
    /// symptom until something reads it. Here the ECS maintains it, so the bug
    /// cannot be written.
    ///
    /// It is deliberately coarse: attaching *any* component bumps it, not just
    /// the ones a given cache cares about. That errs toward rebuilding when
    /// nothing relevant changed, never toward missing a rebuild that was needed.
    pub fn structural_version(&self) -> u64 {
        self.structural_version
    }

    /// Create a new entity with no components and return its handle.
    pub fn spawn(&mut self) -> Entity {
        self.structural_version += 1;
        self.entities.allocate()
    }

    /// Spawn a new entity and return a builder for attaching components to it in
    /// one chained expression:
    ///
    /// ```
    /// # use orrin_ecs::World;
    /// # struct Position(f32);
    /// # struct Velocity(f32);
    /// let mut world = World::new();
    /// let entity = world
    ///     .spawn_entity()
    ///     .with(Position(0.0))
    ///     .with(Velocity(1.0))
    ///     .id();
    /// ```
    pub fn spawn_entity(&mut self) -> EntityBuilder<'_> {
        let entity = self.spawn();
        EntityBuilder {
            world: self,
            entity,
        }
    }

    /// Returns `true` while `entity` refers to a live (not yet despawned) entity.
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    /// Iterate over every live entity in the world.
    ///
    /// Unlike [`query`](World::query), this needs no component and visits even
    /// entities with no components attached — handy for tooling that walks the
    /// whole world (e.g. an editor hierarchy, or a "despawn everything" pass).
    pub fn entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.entities.iter_alive()
    }

    /// Remove an entity and all of its components.
    ///
    /// Returns `false` if the handle was already stale.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }
        for storage in self.storages.values() {
            storage.borrow_mut().remove_entity(entity);
        }
        self.structural_version += 1;
        self.entities.deallocate(entity)
    }

    /// Attach a component to `entity`, returning the previous value if one was
    /// already present.
    ///
    /// Does nothing and returns `None` if `entity` is stale (already despawned).
    /// Without this guard a write through a dangling handle would create a
    /// "zombie" component that no later despawn can ever reclaim.
    pub fn insert<T: 'static>(&mut self, entity: Entity, component: T) -> Option<T> {
        if !self.entities.is_alive(entity) {
            return None;
        }
        let cell = self
            .storages
            .entry(TypeId::of::<T>())
            .or_insert_with(|| RefCell::new(Box::new(SparseSet::<T>::new())));
        let mut guard = cell.borrow_mut();
        let set = guard
            .as_any_mut()
            .downcast_mut::<SparseSet<T>>()
            .expect("storage type mismatch");
        // Bumped for a replacement as well as a first attachment: swapping one
        // value of a relationship component for another reshapes the world just
        // as much as attaching it did.
        self.structural_version += 1;
        set.insert(entity, component)
    }

    /// Detach and return `entity`'s component of type `T`, if it has one.
    pub fn remove<T: 'static>(&mut self, entity: Entity) -> Option<T> {
        let cell = self.storages.get(&TypeId::of::<T>())?;
        let mut guard = cell.borrow_mut();
        let set = guard
            .as_any_mut()
            .downcast_mut::<SparseSet<T>>()
            .expect("storage type mismatch");
        let removed = set.remove(entity);
        // Only a removal that removed something is a change. Bumping on a miss
        // would let a loop that speculatively removes an absent component
        // invalidate every structural cache, every frame.
        if removed.is_some() {
            self.structural_version += 1;
        }
        removed
    }

    /// Returns `true` if `entity` currently has a component of type `T`.
    pub fn has<T: 'static>(&self, entity: Entity) -> bool {
        self.get::<T>(entity).is_some()
    }

    /// Borrow `entity`'s component of type `T`, if present.
    pub fn get<T: 'static>(&self, entity: Entity) -> Option<Ref<'_, T>> {
        let cell = self.storages.get(&TypeId::of::<T>())?;
        Ref::filter_map(cell.borrow(), |b| {
            b.as_any()
                .downcast_ref::<SparseSet<T>>()
                .expect("storage type mismatch")
                .get(entity)
        })
        .ok()
    }

    /// Mutably borrow `entity`'s component of type `T`, if present.
    pub fn get_mut<T: 'static>(&self, entity: Entity) -> Option<RefMut<'_, T>> {
        let cell = self.storages.get(&TypeId::of::<T>())?;
        RefMut::filter_map(cell.borrow_mut(), |b| {
            b.as_any_mut()
                .downcast_mut::<SparseSet<T>>()
                .expect("storage type mismatch")
                .get_mut(entity)
        })
        .ok()
    }

    /// Iterate over every entity that has all the required components in `Q`.
    ///
    /// `Q` is a reference or a tuple of references, e.g. `&Position` or
    /// `(&mut Position, &Velocity)`. Wrapping a parameter in `Option` makes it
    /// optional: `(&Position, Option<&Health>)` matches every entity with a
    /// `Position` and yields `None` where `Health` is absent. Call
    /// [`for_each`](QueryRunner::for_each) on the returned runner.
    pub fn query<Q: QueryParam>(&self) -> QueryRunner<'_, Q> {
        QueryRunner {
            world: self,
            _marker: PhantomData,
        }
    }

    /// Store a unique, world-global value of type `R`, replacing any existing one.
    pub fn insert_resource<R: 'static>(&mut self, resource: R) {
        self.resources
            .insert(TypeId::of::<R>(), RefCell::new(Box::new(resource)));
    }

    /// Remove and return the resource of type `R`, if present.
    pub fn remove_resource<R: 'static>(&mut self) -> Option<R> {
        let cell = self.resources.remove(&TypeId::of::<R>())?;
        let boxed = cell.into_inner();
        boxed.downcast::<R>().ok().map(|b| *b)
    }

    /// Borrow the resource of type `R`.
    ///
    /// # Panics
    /// Panics if no resource of type `R` has been inserted.
    pub fn resource<R: 'static>(&self) -> Ref<'_, R> {
        self.get_resource::<R>()
            .expect("resource not found; insert it with `insert_resource` first")
    }

    /// Mutably borrow the resource of type `R`.
    ///
    /// # Panics
    /// Panics if no resource of type `R` has been inserted.
    pub fn resource_mut<R: 'static>(&self) -> RefMut<'_, R> {
        self.get_resource_mut::<R>()
            .expect("resource not found; insert it with `insert_resource` first")
    }

    /// Borrow the resource of type `R`, or `None` if it has not been inserted.
    pub fn get_resource<R: 'static>(&self) -> Option<Ref<'_, R>> {
        let cell = self.resources.get(&TypeId::of::<R>())?;
        Some(Ref::map(cell.borrow(), |b| {
            b.downcast_ref::<R>().expect("resource type mismatch")
        }))
    }

    /// Mutably borrow the resource of type `R`, or `None` if it is absent.
    pub fn get_resource_mut<R: 'static>(&self) -> Option<RefMut<'_, R>> {
        let cell = self.resources.get(&TypeId::of::<R>())?;
        Some(RefMut::map(cell.borrow_mut(), |b| {
            b.downcast_mut::<R>().expect("resource type mismatch")
        }))
    }
}

//
// Entity builder
//

/// A fluent builder for attaching components to a freshly-spawned entity.
///
/// Returned by [`World::spawn_entity`]. Each [`with`](EntityBuilder::with) call
/// attaches one component and returns the builder, so spawning reads as a single
/// expression instead of a `spawn` followed by a run of `insert` calls. Finish
/// with [`id`](EntityBuilder::id) to get the [`Entity`] handle.
pub struct EntityBuilder<'w> {
    world: &'w mut World,
    entity: Entity,
}

impl<'w> EntityBuilder<'w> {
    /// Attach a component to the entity being built.
    #[inline]
    pub fn with<T: 'static>(self, component: T) -> Self {
        self.world.insert(self.entity, component);
        self
    }

    /// Finish building and return the entity's handle.
    #[inline]
    pub fn id(self) -> Entity {
        self.entity
    }
}

//
// Queries
//

/// A component access pattern that a [`World::query`] can iterate.
///
/// Implemented for `&T` (read), `&mut T` (write), `Option<&T>` /
/// `Option<&mut T>` (optional access that never filters the entity out), and
/// tuples of those. `(&mut Position, &Velocity, Option<&Frozen>)` matches
/// entities that have `Position` and `Velocity`, yielding `Some(&Frozen)` only
/// where it is present.
pub trait QueryParam {
    /// Borrowed handle(s) to the backing storage for the duration of one query.
    type Fetch<'w>;
    /// What a single matched entity yields, e.g. `&T` or `(&mut A, &B)`.
    type Item<'a>;

    /// Whether this parameter is able to drive iteration — true for a required
    /// component, false for an optional one, which matches every entity and so
    /// has no candidate list of its own.
    ///
    /// A constant rather than a runtime check so a tuple can work out *which*
    /// of its parameters drove at compile time, and the dense-index dispatch in
    /// [`get_at`](QueryParam::get_at) folds away entirely.
    const CAN_DRIVE: bool;

    /// Borrow the storage this query needs, or `None` if the query can never
    /// match (a required component whose storage doesn't exist). Optional
    /// parameters always succeed and carry an absent storage as `None` inside
    /// their `Fetch`.
    fn init(world: &World) -> Option<Self::Fetch<'_>>;

    /// Number of candidate entities to scan, or `None` if this parameter
    /// cannot drive iteration. Optional parameters match every entity, so they
    /// have no candidate list of their own; tuples delegate to the first
    /// parameter that can drive.
    fn driver_len(fetch: &Self::Fetch<'_>) -> Option<usize>;

    /// The entity at candidate position `i`, resolved by the same parameter
    /// that answered [`driver_len`](QueryParam::driver_len).
    fn driver_entity_at(fetch: &Self::Fetch<'_>, i: usize) -> Option<Entity>;

    /// Fetch the item for `entity`, or `None` if it lacks one of the required
    /// components.
    fn get<'a>(fetch: &'a mut Self::Fetch<'_>, entity: Entity) -> Option<Self::Item<'a>>;

    /// Fetch the item at candidate position `i`, where `entity` is what
    /// [`driver_entity_at`](QueryParam::driver_entity_at) answered for that
    /// same `i`.
    ///
    /// The parameter that drove already knows where its value lives — `i` *is*
    /// its dense index — so it reads straight out of the packed array and skips
    /// the sparse indirection and generation check that [`get`](QueryParam::get)
    /// would redo. Parameters that did not drive have only the handle to go on
    /// and fall back to `get`.
    fn get_at<'a>(
        fetch: &'a mut Self::Fetch<'_>,
        i: usize,
        entity: Entity,
    ) -> Option<Self::Item<'a>>;
}

impl<T: 'static> QueryParam for &T {
    type Fetch<'w> = Ref<'w, SparseSet<T>>;
    type Item<'a> = &'a T;

    const CAN_DRIVE: bool = true;

    fn init(world: &World) -> Option<Self::Fetch<'_>> {
        let cell = world.storages.get(&TypeId::of::<T>())?;
        Some(Ref::map(cell.borrow(), |b| {
            b.as_any()
                .downcast_ref::<SparseSet<T>>()
                .expect("storage type mismatch")
        }))
    }

    fn driver_len(fetch: &Self::Fetch<'_>) -> Option<usize> {
        Some(fetch.dense_entities.len())
    }

    fn driver_entity_at(fetch: &Self::Fetch<'_>, i: usize) -> Option<Entity> {
        Some(fetch.dense_entities[i])
    }

    fn get<'a>(fetch: &'a mut Self::Fetch<'_>, entity: Entity) -> Option<Self::Item<'a>> {
        fetch.get(entity)
    }

    fn get_at<'a>(
        fetch: &'a mut Self::Fetch<'_>,
        i: usize,
        _entity: Entity,
    ) -> Option<Self::Item<'a>> {
        fetch.dense_values.get(i)
    }
}

impl<T: 'static> QueryParam for Option<&T> {
    type Fetch<'w> = Option<Ref<'w, SparseSet<T>>>;
    type Item<'a> = Option<&'a T>;

    const CAN_DRIVE: bool = false;

    fn init(world: &World) -> Option<Self::Fetch<'_>> {
        Some(<&T as QueryParam>::init(world))
    }

    fn driver_len(_fetch: &Self::Fetch<'_>) -> Option<usize> {
        None
    }

    fn driver_entity_at(_fetch: &Self::Fetch<'_>, _i: usize) -> Option<Entity> {
        None
    }

    fn get<'a>(fetch: &'a mut Self::Fetch<'_>, entity: Entity) -> Option<Self::Item<'a>> {
        Some(fetch.as_ref().and_then(|set| set.get(entity)))
    }

    // An optional parameter never drives, so `i` names a position in some other
    // parameter's dense array and is meaningless here.
    fn get_at<'a>(
        fetch: &'a mut Self::Fetch<'_>,
        _i: usize,
        entity: Entity,
    ) -> Option<Self::Item<'a>> {
        Self::get(fetch, entity)
    }
}

impl<T: 'static> QueryParam for Option<&mut T> {
    type Fetch<'w> = Option<RefMut<'w, SparseSet<T>>>;
    type Item<'a> = Option<&'a mut T>;

    const CAN_DRIVE: bool = false;

    fn init(world: &World) -> Option<Self::Fetch<'_>> {
        Some(<&mut T as QueryParam>::init(world))
    }

    fn driver_len(_fetch: &Self::Fetch<'_>) -> Option<usize> {
        None
    }

    fn driver_entity_at(_fetch: &Self::Fetch<'_>, _i: usize) -> Option<Entity> {
        None
    }

    fn get<'a>(fetch: &'a mut Self::Fetch<'_>, entity: Entity) -> Option<Self::Item<'a>> {
        Some(fetch.as_mut().and_then(|set| set.get_mut(entity)))
    }

    // As above: never the driver, so the dense index belongs to someone else.
    fn get_at<'a>(
        fetch: &'a mut Self::Fetch<'_>,
        _i: usize,
        entity: Entity,
    ) -> Option<Self::Item<'a>> {
        Self::get(fetch, entity)
    }
}

impl<T: 'static> QueryParam for &mut T {
    type Fetch<'w> = RefMut<'w, SparseSet<T>>;
    type Item<'a> = &'a mut T;

    const CAN_DRIVE: bool = true;

    fn init(world: &World) -> Option<Self::Fetch<'_>> {
        let cell = world.storages.get(&TypeId::of::<T>())?;
        Some(RefMut::map(cell.borrow_mut(), |b| {
            b.as_any_mut()
                .downcast_mut::<SparseSet<T>>()
                .expect("storage type mismatch")
        }))
    }

    fn driver_len(fetch: &Self::Fetch<'_>) -> Option<usize> {
        Some(fetch.dense_entities.len())
    }

    fn driver_entity_at(fetch: &Self::Fetch<'_>, i: usize) -> Option<Entity> {
        Some(fetch.dense_entities[i])
    }

    fn get<'a>(fetch: &'a mut Self::Fetch<'_>, entity: Entity) -> Option<Self::Item<'a>> {
        fetch.get_mut(entity)
    }

    fn get_at<'a>(
        fetch: &'a mut Self::Fetch<'_>,
        i: usize,
        _entity: Entity,
    ) -> Option<Self::Item<'a>> {
        fetch.dense_values.get_mut(i)
    }
}

macro_rules! impl_query_for_tuple {
    ($($name:ident => $idx:tt [$($earlier:ident)*]),+) => {
        impl<$($name: QueryParam),+> QueryParam for ($($name,)+) {
            type Fetch<'w> = ($($name::Fetch<'w>,)+);
            type Item<'a> = ($($name::Item<'a>,)+);

            const CAN_DRIVE: bool = false $(|| $name::CAN_DRIVE)+;

            fn init(world: &World) -> Option<Self::Fetch<'_>> {
                Some(($($name::init(world)?,)+))
            }

            // The first parameter able to drive wins. Both chains must probe
            // in the same order so `driver_entity_at` answers from the same
            // parameter as `driver_len`.
            fn driver_len(fetch: &Self::Fetch<'_>) -> Option<usize> {
                Option::<usize>::None
                    $(.or_else(|| $name::driver_len(&fetch.$idx)))+
            }

            fn driver_entity_at(fetch: &Self::Fetch<'_>, i: usize) -> Option<Entity> {
                Option::<Entity>::None
                    $(.or_else(|| $name::driver_entity_at(&fetch.$idx, i)))+
            }

            fn get<'a>(fetch: &'a mut Self::Fetch<'_>, entity: Entity) -> Option<Self::Item<'a>> {
                // Disjoint mutable borrows of distinct tuple fields are fine.
                Some(($($name::get(&mut fetch.$idx, entity)?,)+))
            }

            // Exactly one position drove — the first that could — and only it
            // may read `i` as its own dense index. `[$($earlier)*]` is that
            // position's predecessors, spelled out at the invocation because a
            // macro cannot fold over a prefix of its own repetition. The test
            // is a `const` block, so each arm is a branch already taken at
            // compile time.
            fn get_at<'a>(
                fetch: &'a mut Self::Fetch<'_>,
                i: usize,
                entity: Entity,
            ) -> Option<Self::Item<'a>> {
                Some(($(
                    if const { $name::CAN_DRIVE $(&& !$earlier::CAN_DRIVE)* } {
                        $name::get_at(&mut fetch.$idx, i, entity)?
                    } else {
                        $name::get(&mut fetch.$idx, entity)?
                    },
                )+))
            }
        }
    };
}

impl_query_for_tuple!(A => 0 [], B => 1 [A]);
impl_query_for_tuple!(A => 0 [], B => 1 [A], C => 2 [A B]);
impl_query_for_tuple!(A => 0 [], B => 1 [A], C => 2 [A B], D => 3 [A B C]);
impl_query_for_tuple!(A => 0 [], B => 1 [A], C => 2 [A B], D => 3 [A B C], E => 4 [A B C D]);

/// Runs a prepared query. Created by [`World::query`].
pub struct QueryRunner<'w, Q> {
    world: &'w World,
    _marker: PhantomData<Q>,
}

impl<'w, Q: QueryParam> QueryRunner<'w, Q> {
    /// Visit every match in order; stops early and returns the entity for
    /// which `visit` returns `true`.
    ///
    /// A query with a driving parameter scans that parameter's dense list. A
    /// query of only optional parameters has no driver and matches every live
    /// entity, so it falls back to walking the allocator.
    fn visit<F>(&self, mut visit: F) -> Option<Entity>
    where
        F: FnMut(Entity, Q::Item<'_>) -> bool,
    {
        let mut fetch = Q::init(self.world)?;
        match Q::driver_len(&fetch) {
            Some(count) => {
                for i in 0..count {
                    let entity = Q::driver_entity_at(&fetch, i)
                        .expect("query driver lost between driver_len and driver_entity_at");
                    if let Some(item) = Q::get_at(&mut fetch, i, entity) {
                        if visit(entity, item) {
                            return Some(entity);
                        }
                    }
                }
            }
            None => {
                for entity in self.world.entities.iter_alive() {
                    if let Some(item) = Q::get(&mut fetch, entity) {
                        if visit(entity, item) {
                            return Some(entity);
                        }
                    }
                }
            }
        }
        None
    }

    /// Call `f` once for every entity that matches the query `Q`.
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(Entity, Q::Item<'_>),
    {
        self.visit(|entity, item| {
            f(entity, item);
            false
        });
    }

    /// Count how many entities match the query, without invoking a callback.
    pub fn count(&self) -> usize {
        let mut matched = 0;
        self.visit(|_, _| {
            matched += 1;
            false
        });
        matched
    }

    /// The first matching entity (in storage order) for which `pred` returns true.
    pub fn find<F>(&self, mut pred: F) -> Option<Entity>
    where
        F: FnMut(Entity, Q::Item<'_>) -> bool,
    {
        self.visit(&mut pred)
    }
}

//
// Tests
//i

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }
    #[derive(Debug)]
    struct Velocity {
        x: f32,
        y: f32,
    }
    #[derive(Debug, PartialEq)]
    struct Health(i32);
    struct Frozen; // a marker / tag component

    struct DeltaTime(f32); // a resource

    #[test]
    fn insert_get_remove() {
        let mut world = World::new();
        let e = world.spawn();
        assert!(world.insert(e, Position { x: 1.0, y: 2.0 }).is_none());
        assert_eq!(world.get::<Position>(e).unwrap().x, 1.0);
        assert!(world.has::<Position>(e));

        // Overwrite returns the old value.
        let old = world.insert(e, Position { x: 9.0, y: 9.0 }).unwrap();
        assert_eq!(old, Position { x: 1.0, y: 2.0 });

        let removed = world.remove::<Position>(e).unwrap();
        assert_eq!(removed, Position { x: 9.0, y: 9.0 });
        assert!(!world.has::<Position>(e));
    }

    #[test]
    fn movement_system() {
        let mut world = World::new();
        world.insert_resource(DeltaTime(0.5));

        for i in 0..4 {
            let e = world.spawn();
            world.insert(e, Position { x: 0.0, y: 0.0 });
            world.insert(
                e,
                Velocity {
                    x: i as f32,
                    y: 1.0,
                },
            );
        }
        // One entity with no Velocity should be skipped by the query.
        let stationary = world.spawn();
        world.insert(stationary, Position { x: 100.0, y: 100.0 });

        let dt = world.resource::<DeltaTime>().0;
        let mut visited = 0;
        world
            .query::<(&mut Position, &Velocity)>()
            .for_each(|_e, (pos, vel)| {
                pos.x += vel.x * dt;
                pos.y += vel.y * dt;
                visited += 1;
            });

        assert_eq!(visited, 4);
        assert_eq!(world.query::<(&Position, &Velocity)>().count(), 4);
        // The stationary entity wasn't matched, so it didn't move.
        assert_eq!(world.get::<Position>(stationary).unwrap().x, 100.0);
    }

    #[test]
    fn tag_components_and_three_param_query() {
        let mut world = World::new();
        let a = world.spawn();
        world.insert(a, Position { x: 0.0, y: 0.0 });
        world.insert(a, Velocity { x: 1.0, y: 1.0 });
        world.insert(a, Frozen);

        let b = world.spawn();
        world.insert(b, Position { x: 0.0, y: 0.0 });
        world.insert(b, Velocity { x: 1.0, y: 1.0 });

        // Only `a` has all three components.
        let mut matched = Vec::new();
        world
            .query::<(&mut Position, &Velocity, &Frozen)>()
            .for_each(|e, (_pos, _vel, _frozen)| matched.push(e));
        assert_eq!(matched, vec![a]);
    }

    #[test]
    fn find_first_match_in_storage_order() {
        let mut world = World::new();
        let a = world.spawn();
        world.insert(a, Health(10));
        let b = world.spawn();
        world.insert(b, Health(3));
        world.spawn(); // no components; never a candidate

        // First match follows dense (insertion) order.
        assert_eq!(world.query::<&Health>().find(|_, h| h.0 > 0), Some(a));
        // The predicate can select a later entity.
        assert_eq!(world.query::<&Health>().find(|_, h| h.0 < 5), Some(b));
        // No matching entity, and no storage registered at all, both give None.
        assert_eq!(world.query::<&Health>().find(|_, h| h.0 > 99), None);
        assert_eq!(world.query::<&Position>().find(|_, _| true), None);
    }

    #[test]
    fn find_stops_at_first_match() {
        let mut world = World::new();
        for hp in [1, 2, 3] {
            let e = world.spawn();
            world.insert(e, Health(hp));
        }

        let mut visited = 0;
        let found = world.query::<&Health>().find(|_, _| {
            visited += 1;
            true
        });
        assert!(found.is_some());
        assert_eq!(visited, 1);
    }

    #[test]
    fn optional_param_does_not_filter() {
        let mut world = World::new();
        let armored = world.spawn();
        world.insert(armored, Position { x: 1.0, y: 0.0 });
        world.insert(armored, Health(50));
        let bare = world.spawn();
        world.insert(bare, Position { x: 2.0, y: 0.0 });

        // Both entities match; only `armored` yields Some for the optional part.
        let mut seen = Vec::new();
        world
            .query::<(&Position, Option<&Health>)>()
            .for_each(|e, (_pos, health)| seen.push((e, health.map(|h| h.0))));
        seen.sort_by_key(|(e, _)| e.index());
        assert_eq!(seen, vec![(armored, Some(50)), (bare, None)]);
        assert_eq!(world.query::<(&Position, Option<&Health>)>().count(), 2);
    }

    #[test]
    fn optional_param_with_absent_storage() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0, y: 0.0 });

        // No Velocity storage exists at all; the optional param must not kill
        // the query the way a required param would.
        let mut visited = Vec::new();
        world
            .query::<(&Position, Option<&Velocity>)>()
            .for_each(|entity, (_pos, vel)| {
                assert!(vel.is_none());
                visited.push(entity);
            });
        assert_eq!(visited, vec![e]);
    }

    #[test]
    fn optional_param_can_lead_a_tuple() {
        let mut world = World::new();
        let a = world.spawn();
        world.insert(a, Health(1));
        let b = world.spawn();
        world.insert(b, Health(2));
        world.insert(b, Position { x: 0.0, y: 0.0 });

        // Iteration must be driven by the required Health, not the optional
        // Position, even though Position comes first.
        let mut seen = Vec::new();
        world
            .query::<(Option<&Position>, &Health)>()
            .for_each(|e, (pos, health)| seen.push((e, pos.is_some(), health.0)));
        seen.sort_by_key(|(e, _, _)| e.index());
        assert_eq!(seen, vec![(a, false, 1), (b, true, 2)]);
    }

    #[test]
    fn optional_mut_param_allows_mutation() {
        let mut world = World::new();
        let hurt = world.spawn();
        world.insert(hurt, Position { x: 0.0, y: 0.0 });
        world.insert(hurt, Health(10));
        let unhurt = world.spawn();
        world.insert(unhurt, Position { x: 0.0, y: 0.0 });

        world
            .query::<(&Position, Option<&mut Health>)>()
            .for_each(|_e, (_pos, health)| {
                if let Some(health) = health {
                    health.0 += 5;
                }
            });
        assert_eq!(world.get::<Health>(hurt).unwrap().0, 15);
        assert!(!world.has::<Health>(unhurt));
    }

    #[test]
    fn all_optional_query_visits_every_live_entity() {
        let mut world = World::new();
        let a = world.spawn();
        world.insert(a, Health(3));
        let empty = world.spawn(); // no components at all
        let dead = world.spawn();
        assert!(world.despawn(dead));

        let mut seen = Vec::new();
        world
            .query::<Option<&Health>>()
            .for_each(|e, health| seen.push((e, health.map(|h| h.0))));
        seen.sort_by_key(|(e, _)| e.index());
        assert_eq!(seen, vec![(a, Some(3)), (empty, None)]);
    }

    /// Slot 0 is reserved, so the all-zeroes handle is inert forever. This is
    /// what lets `Entity::default()` double as the FFI null sentinel: C# cannot
    /// override a struct's default, so an unassigned `Entity` field over there
    /// is these exact bytes, and it must not name anything.
    ///
    /// Pinned as its own test because the property is invisible at the call
    /// site — a refactor of `allocate` that reverted to handing out index 0
    /// would break the C# boundary while every other ECS test still passed.
    #[test]
    fn entity_slot_zero_is_never_allocated() {
        let mut world = World::new();

        let null = Entity::default();
        assert!(!world.is_alive(null));

        // Includes the recycle path: despawning must not put slot 0 in play.
        let mut spawned = Vec::new();
        for _ in 0..8 {
            spawned.push(world.spawn());
        }
        for entity in spawned.drain(..).take(4) {
            assert!(world.despawn(entity));
        }
        for _ in 0..8 {
            spawned.push(world.spawn());
        }
        assert!(
            spawned.iter().all(|e| e.index() != 0),
            "slot 0 was handed out: {spawned:?}"
        );

        // Inert against component storage too, not just the allocator.
        world.insert(null, Health(1));
        assert!(world.get::<Health>(null).is_none());
        let mut visited = 0;
        world.query::<&Health>().for_each(|_, _| visited += 1);
        assert_eq!(visited, 0);
    }

    /// The driving parameter now reads its value straight out of the dense
    /// array at the candidate index instead of resolving the entity through
    /// `sparse`. That is only correct while `dense_entities[i]` and
    /// `dense_values[i]` describe the same entity — which `remove`'s
    /// swap-with-last is exactly the thing that could break.
    ///
    /// Every other query test builds its storage in one pass and never removes,
    /// so the dense order matches insertion order and a `get_at` that quietly
    /// returned the wrong slot would still pass all of them.
    #[test]
    fn driver_reads_the_right_value_after_dense_reordering() {
        let mut world = World::new();
        let entities: Vec<Entity> = (0..16)
            .map(|i| {
                let e = world.spawn();
                world.insert(e, Health(i));
                world.insert(
                    e,
                    Position {
                        x: i as f32,
                        y: 0.0,
                    },
                );
                e
            })
            .collect();

        // Swap-remove from the middle and the ends, so the dense array ends up
        // in an order unrelated to the entity indices.
        for e in [entities[3], entities[0], entities[9], entities[15]] {
            assert!(world.despawn(e));
        }

        let mut seen = Vec::new();
        world
            .query::<&Health>()
            .for_each(|entity, health| seen.push((entity, health.0)));
        for (entity, hp) in &seen {
            // The value the query handed out must be the one this entity owns.
            assert_eq!(world.get::<Health>(*entity).unwrap().0, *hp);
        }
        assert_eq!(seen.len(), 12);

        // Same again through a tuple, where the driver takes the fast path and
        // the second parameter still goes through `sparse`.
        let mut pairs = Vec::new();
        world
            .query::<(&Health, &Position)>()
            .for_each(|entity, (health, position)| pairs.push((entity, health.0, position.x)));
        assert_eq!(pairs.len(), 12);
        for (_, hp, x) in &pairs {
            assert_eq!(
                *hp as f32, *x,
                "driver and non-driver disagree about the entity"
            );
        }

        // And with the driver written through: a stale dense index would move
        // the wrong entity's value.
        world
            .query::<&mut Health>()
            .for_each(|_, health| health.0 += 100);
        for (entity, hp) in seen {
            assert_eq!(world.get::<Health>(entity).unwrap().0, hp + 100);
        }
    }

    #[test]
    fn generational_indices_detect_stale_handles() {
        let mut world = World::new();
        let e1 = world.spawn();
        world.insert(e1, Health(10));
        assert!(world.despawn(e1));

        // The freed slot is recycled, but with a new generation.
        let e2 = world.spawn();
        assert_eq!(e1.index(), e2.index());
        assert_ne!(e1.generation(), e2.generation());

        // The stale handle must not resolve, even after the slot is reused.
        assert!(!world.is_alive(e1));
        assert!(world.get::<Health>(e1).is_none());
        world.insert(e2, Health(99));
        assert!(world.get::<Health>(e1).is_none());
        assert_eq!(world.get::<Health>(e2).unwrap().0, 99);
    }

    #[test]
    fn despawn_clears_all_components() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0, y: 0.0 });
        world.insert(e, Health(5));
        assert!(world.despawn(e));
        assert!(world.get::<Position>(e).is_none());
        assert!(world.get::<Health>(e).is_none());
        assert!(!world.despawn(e)); // already gone
    }

    #[test]
    fn insert_on_stale_handle_is_ignored() {
        let mut world = World::new();
        let e1 = world.spawn();
        assert!(world.despawn(e1));

        // The slot is recycled with a fresh generation.
        let e2 = world.spawn();
        assert_eq!(e1.index(), e2.index());

        // Writing through the stale handle must be a no-op, not a zombie that
        // outlives every future despawn of this slot.
        assert!(world.insert(e1, Health(1)).is_none());
        assert!(!world.has::<Health>(e1));
        assert!(!world.has::<Health>(e2));

        // The live entity is unaffected and behaves normally.
        world.insert(e2, Health(7));
        assert_eq!(world.get::<Health>(e2).unwrap().0, 7);
        assert!(!world.has::<Health>(e1));
    }

    #[test]
    fn resources_round_trip() {
        let mut world = World::new();
        world.insert_resource(DeltaTime(1.0));
        world.resource_mut::<DeltaTime>().0 = 2.0;
        assert_eq!(world.resource::<DeltaTime>().0, 2.0);
        assert_eq!(world.remove_resource::<DeltaTime>().unwrap().0, 2.0);
        assert!(world.get_resource::<DeltaTime>().is_none());
    }

    #[test]
    fn entity_builder_attaches_all_components() {
        let mut world = World::new();
        let e = world
            .spawn_entity()
            .with(Position { x: 1.0, y: 2.0 })
            .with(Velocity { x: 3.0, y: 4.0 })
            .with(Frozen)
            .id();

        assert_eq!(world.get::<Position>(e).unwrap().x, 1.0);
        assert_eq!(world.get::<Velocity>(e).unwrap().y, 4.0);
        assert!(world.has::<Frozen>(e));
    }

    #[test]
    fn entities_lists_only_live_entities() {
        let mut world = World::new();
        let a = world.spawn();
        let b = world.spawn();
        let c = world.spawn();
        assert!(world.despawn(b));

        // The freed slot `b` is gone; a recycled slot reappears once reused.
        let mut live: Vec<_> = world.entities().collect();
        live.sort_by_key(|e| e.index());
        assert_eq!(live, vec![a, c]);

        let d = world.spawn(); // recycles b's index with a fresh generation
        assert_eq!(d.index(), b.index());
        assert_ne!(d.generation(), b.generation());
        assert!(world.entities().any(|e| e == d));
        assert!(!world.entities().any(|e| e == b));
        assert_eq!(world.entities().count(), 3);
    }

    /// The counter exists so a structural cache can skip a rebuild. That only
    /// works if every reshaping operation moves it.
    #[test]
    fn every_structural_change_bumps_the_version() {
        let mut world = World::new();
        let start = world.structural_version();

        let entity = world.spawn();
        let after_spawn = world.structural_version();
        assert!(after_spawn > start, "spawn");

        world.insert(entity, Position { x: 1.0, y: 0.0 });
        let after_insert = world.structural_version();
        assert!(after_insert > after_spawn, "insert");

        world.remove::<Position>(entity);
        let after_remove = world.structural_version();
        assert!(after_remove > after_insert, "remove");

        world.despawn(entity);
        assert!(world.structural_version() > after_remove, "despawn");
    }

    /// Replacing a component's value reshapes the world for anything that
    /// derives structure *from* that value — a parent link being the case this
    /// was built for.
    #[test]
    fn replacing_a_component_bumps_the_version() {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Position { x: 1.0, y: 0.0 });

        let before = world.structural_version();
        world.insert(entity, Position { x: 2.0, y: 0.0 });

        assert!(world.structural_version() > before);
    }

    /// Mutating through a handle changes a value, not the shape. Counting it
    /// would invalidate every structural cache on every frame that moved
    /// anything, which is the whole cost the counter exists to avoid.
    #[test]
    fn mutating_a_component_does_not_bump_the_version() {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Position { x: 1.0, y: 0.0 });

        let before = world.structural_version();
        world.get_mut::<Position>(entity).unwrap().x = 9.0;
        world
            .query::<&mut Position>()
            .for_each(|_, position| position.x = 10.0);

        assert_eq!(world.structural_version(), before);
    }

    /// A speculative remove of an absent component is a no-op, and a no-op that
    /// bumped the version would make a caller that polls it rebuild forever.
    #[test]
    fn a_removal_that_removes_nothing_does_not_bump_the_version() {
        let mut world = World::new();
        let entity = world.spawn();

        let before = world.structural_version();
        assert!(world.remove::<Position>(entity).is_none());

        assert_eq!(world.structural_version(), before);
    }

    /// Insert refuses a stale handle, and a refusal is not a change.
    #[test]
    fn inserting_through_a_stale_handle_does_not_bump_the_version() {
        let mut world = World::new();
        let entity = world.spawn();
        world.despawn(entity);

        let before = world.structural_version();
        assert!(world.insert(entity, Position { x: 1.0, y: 0.0 }).is_none());

        assert_eq!(world.structural_version(), before);
    }
}
