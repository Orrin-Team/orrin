use std::any::TypeId;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

use orrin_ecs::{Entity, World};

use crate::reflect::Reflect;
use crate::value::{Value, ValueError};

/// A component type's identity on disk and over the wire, e.g.
/// `"orrin.transform"`.
///
/// Deliberately a string and never a `TypeId` or a Rust path: `TypeId` is
/// derived from where the type is declared, so moving `LocalTransform` between
/// modules would orphan every scene that references it, silently and with no
/// error at any point.
///
/// `Cow` because engine ids are `&'static str` literals while a game
/// assembly's arrive at runtime as owned strings.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ComponentId(Cow<'static, str>);

impl ComponentId {
    pub const fn new(id: &'static str) -> Self {
        Self(Cow::Borrowed(id))
    }

    pub fn owned(id: impl Into<String>) -> Self {
        Self(Cow::Owned(id.into()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ComponentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Everything the engine can do with one component type without knowing it.
///
/// Every entry is a bare `fn` pointer, not a boxed closure: the closures built
/// in [`Registry::register`] capture nothing, and their bodies call
/// `world.get::<T>` with `T` known statically. Monomorphization does the type
/// erasure, so this table costs one pointer per operation and no allocation.
///
/// The v1 set is closed — presence, read, write, remove, default. No method
/// invocation and no open-ended metadata queries; additions need a consumer
/// that demonstrably needs them.
pub struct ComponentVtable {
    pub id: ComponentId,
    /// Display name for the inspector. Unlike `id`, this may change freely.
    pub name: &'static str,
    /// Process-local lookup key only — it is never written anywhere and never
    /// compared across builds. See [`ComponentId`] for why identity cannot be
    /// a `TypeId`.
    pub type_id: TypeId,
    pub has: fn(&World, Entity) -> bool,
    /// `None` when the entity doesn't have this component.
    ///
    /// Returning an owned `Value` is load-bearing: the `Ref<'_, T>` guard over
    /// the component storage drops before this returns, so no caller can hold a
    /// world borrow across whatever it does next. That is what makes the
    /// registry safe to call from a script dispatch window, where holding a
    /// borrow across a call into C# is forbidden.
    pub read: fn(&World, Entity) -> Option<Value>,
    /// Insert or replace the component from `value`. A stale entity handle is a
    /// no-op, matching `World::insert`.
    pub write: fn(&mut World, Entity, &Value) -> Result<(), ValueError>,
    pub remove: fn(&mut World, Entity),
    pub default: fn() -> Value,
}

/// Every component type the engine knows how to read, write, and default.
///
/// Owned by the application rather than stored as a world resource: it has to
/// outlive a world being cleared, and a scene load needs it before there is a
/// world to read it out of.
#[derive(Default)]
pub struct Registry {
    entries: Vec<ComponentVtable>,
    by_id: HashMap<ComponentId, usize>,
    by_type: HashMap<TypeId, usize>,
    /// Where a game assembly's entries begin; see [`clear_game`](Self::clear_game).
    engine_count: usize,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Describe `T` to the engine under the stable id `id`.
    ///
    /// # Panics
    /// If `id` or `T` is already registered. Registration is startup code and a
    /// collision is unambiguously a bug — one that, tolerated, would have one
    /// component type's data overwrite another's in every scene ever saved.
    pub fn register<T: Reflect + Default>(&mut self, id: ComponentId, name: &'static str) {
        if let Some(&existing) = self.by_id.get(&id) {
            panic!(
                "component id `{id}` is already registered by `{}` (attempted by `{name}`)",
                self.entries[existing].name
            );
        }
        let type_id = TypeId::of::<T>();
        if let Some(&existing) = self.by_type.get(&type_id) {
            panic!(
                "`{name}` is already registered as `{}`; a type gets exactly one id",
                self.entries[existing].id
            );
        }

        let index = self.entries.len();
        self.entries.push(ComponentVtable {
            id: id.clone(),
            name,
            type_id,
            has: |world, entity| world.has::<T>(entity),
            read: |world, entity| world.get::<T>(entity).map(|c| c.to_value()),
            write: |world, entity, value| {
                // Converted before the world is touched, so a malformed value
                // leaves the existing component intact rather than half
                // replaced.
                let component = T::from_value(value)?;
                let _ = world.insert(entity, component);
                Ok(())
            },
            remove: |world, entity| {
                let _ = world.remove::<T>(entity);
            },
            default: || T::default().to_value(),
        });
        self.by_id.insert(id, index);
        self.by_type.insert(type_id, index);
    }

    /// Mark the end of the engine's own registrations. Everything registered
    /// after this belongs to a game assembly and is dropped by
    /// [`clear_game`](Self::clear_game).
    pub fn end_engine_registration(&mut self) {
        self.engine_count = self.entries.len();
    }

    /// Drop every entry a game assembly registered.
    ///
    /// Must run before the outgoing assembly is committed for unload: an entry
    /// built from a game type keeps that type reachable, and a collectible load
    /// context with a live reference into it never unloads — the failure the C#
    /// side's `BeginRetire`/`FinishRetire` split exists to avoid.
    pub fn clear_game(&mut self) {
        self.entries.truncate(self.engine_count);
        self.by_id.clear();
        self.by_type.clear();
        for (index, entry) in self.entries.iter().enumerate() {
            self.by_id.insert(entry.id.clone(), index);
            self.by_type.insert(entry.type_id, index);
        }
    }

    pub fn get(&self, id: &ComponentId) -> Option<&ComponentVtable> {
        self.by_id.get(id).map(|&i| &self.entries[i])
    }

    /// The vtable for a component type known statically.
    pub fn of<T: 'static>(&self) -> Option<&ComponentVtable> {
        self.by_type
            .get(&TypeId::of::<T>())
            .map(|&i| &self.entries[i])
    }

    /// Every registered component, in registration order. Callers that need a
    /// canonical order (the text writer, the scene format) sort by
    /// [`ComponentId`] themselves.
    pub fn components(&self) -> impl Iterator<Item = &ComponentVtable> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reflect::take;

    #[derive(Debug, Default, PartialEq)]
    struct Speed(f32);

    impl Reflect for Speed {
        fn to_value(&self) -> Value {
            self.0.to_value()
        }

        fn from_value(value: &Value) -> Result<Self, ValueError> {
            f32::from_value(value).map(Self)
        }
    }

    #[derive(Debug, Default, PartialEq)]
    struct Label(String);

    impl Reflect for Label {
        fn to_value(&self) -> Value {
            Value::strukt([("text", self.0.to_value())])
        }

        fn from_value(value: &Value) -> Result<Self, ValueError> {
            Ok(Self(take(value, "text")?))
        }
    }

    fn registry() -> Registry {
        let mut registry = Registry::new();
        registry.register::<Speed>(ComponentId::new("test.speed"), "Speed");
        registry.register::<Label>(ComponentId::new("test.label"), "Label");
        registry.end_engine_registration();
        registry
    }

    #[test]
    fn the_vtable_reads_writes_and_removes_without_the_type() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn();

        let speed = registry.get(&ComponentId::new("test.speed")).unwrap();
        assert!(!(speed.has)(&world, entity));
        assert_eq!((speed.read)(&world, entity), None);

        (speed.write)(&mut world, entity, &Value::F32(4.5)).unwrap();
        assert!((speed.has)(&world, entity));
        assert_eq!((speed.read)(&world, entity), Some(Value::F32(4.5)));
        assert_eq!(*world.get::<Speed>(entity).unwrap(), Speed(4.5));

        (speed.remove)(&mut world, entity);
        assert!(!(speed.has)(&world, entity));
    }

    #[test]
    fn a_bad_value_reports_its_field_and_leaves_the_component_alone() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn();

        let label = registry.get(&ComponentId::new("test.label")).unwrap();
        let ok = Value::strukt([("text", Value::String("ok".to_owned()))]);
        (label.write)(&mut world, entity, &ok).unwrap();

        let bad = Value::strukt([("text", Value::F32(1.0))]);
        let err = (label.write)(&mut world, entity, &bad).unwrap_err();
        assert_eq!(err.to_string(), "field `text`: expected string, found f32");
        assert_eq!(
            *world.get::<Label>(entity).unwrap(),
            Label("ok".to_owned())
        );
    }

    #[test]
    fn defaults_come_from_the_type() {
        let registry = registry();
        let speed = registry.get(&ComponentId::new("test.speed")).unwrap();
        assert_eq!((speed.default)(), Value::F32(0.0));
    }

    #[test]
    fn lookup_by_id_and_by_type_agree() {
        let registry = registry();
        let by_id = registry.get(&ComponentId::new("test.speed")).unwrap();
        let by_type = registry.of::<Speed>().unwrap();
        assert_eq!(by_id.id, by_type.id);
        assert!(registry.get(&ComponentId::new("test.nothing")).is_none());
    }

    #[test]
    fn clearing_game_entries_keeps_the_engine_ones() {
        let mut registry = registry();
        registry.register::<f32>(ComponentId::new("game.thing"), "Thing");
        assert_eq!(registry.len(), 3);

        registry.clear_game();
        assert_eq!(registry.len(), 2);
        assert!(registry.get(&ComponentId::new("game.thing")).is_none());
        assert!(registry.get(&ComponentId::new("test.speed")).is_some());
        assert!(registry.of::<Speed>().is_some());
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn a_duplicate_id_panics_at_registration() {
        let mut registry = registry();
        registry.register::<f32>(ComponentId::new("test.speed"), "Other");
    }

    #[test]
    #[should_panic(expected = "exactly one id")]
    fn registering_a_type_twice_panics() {
        let mut registry = registry();
        registry.register::<Speed>(ComponentId::new("test.speed2"), "Speed");
    }
}
