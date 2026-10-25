use std::any::TypeId;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use orrin_ecs::{Entity, World};

use crate::diff::FieldChange;
use crate::reflect::{Reflect, Resolve};
use crate::value::{Value, ValueError};

/// Why a batch of [`FieldChange`]s could not land on a component.
///
/// Distinct from a bare [`ValueError`] because the two failures need different
/// answers from the caller: a malformed change is a bug in whoever produced it,
/// while an absent component is the ordinary outcome of replaying an undo step
/// past a delete, and the editor recovers from that by dropping the step.
#[derive(Clone, Debug, PartialEq)]
pub enum ApplyError {
    Absent,
    Value(ValueError),
}

impl fmt::Display for ApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Absent => f.write_str("the entity does not have this component"),
            Self::Value(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ApplyError {}

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

/// How a component type that has no Rust type answers the vtable's questions.
///
/// Implemented by the scripting layer and injected at registration, because the
/// registry must not depend on the FFI: a headless consumer of the same diffs
/// (architecture §4.2's collaboration server) links neither CoreCLR nor the
/// engine's script host.
///
/// Every method takes the managed type name rather than reading it from `self`,
/// so one bridge serves every Behaviour type in an assembly instead of one per
/// registration.
pub trait ScriptBridge: Send + Sync + 'static {
    fn has(&self, world: &World, entity: Entity, type_name: &str) -> bool;
    fn read(&self, world: &World, entity: Entity, type_name: &str) -> Option<Value>;
    fn write(
        &self,
        world: &mut World,
        entity: Entity,
        type_name: &str,
        value: &Value,
    ) -> Result<(), ValueError>;
    fn remove(&self, world: &mut World, entity: Entity, type_name: &str);
    fn default(&self, type_name: &str) -> Value;
}

/// What a component registered from a game assembly carries that a Rust type
/// supplies statically.
pub struct ScriptBinding {
    /// Assembly-qualified managed type name, as `Behaviours.Create` takes it.
    pub type_name: String,
    pub bridge: Arc<dyn ScriptBridge>,
}

/// Everything the engine can do with one component type without knowing it.
///
/// Every entry is a bare `fn` pointer, not a boxed closure: the closures built
/// in [`Registry::register`] capture nothing, and their bodies call
/// `world.get::<T>` with `T` known statically. Monomorphization does the type
/// erasure, so this table costs one pointer per operation and no allocation.
///
/// Each takes `&self` first, which is how an entry with no Rust type behind it
/// reaches its own [`ScriptBinding`] without a captured closure. Call them
/// through the inherent methods of the same name — `vtable.read(world, entity)`
/// rather than `(vtable.read)(vtable, world, entity)` — which is the only place
/// that threading is spelled out.
///
/// The v1 set is closed — presence, read, write, remove, diff, apply, inspect,
/// default. No method invocation and no open-ended metadata queries; additions
/// need a consumer that demonstrably needs them.
pub struct ComponentVtable {
    pub id: ComponentId,
    /// Display name for the inspector. Unlike `id`, this may change freely.
    ///
    /// Owned for the same reason [`ComponentId`] is: the engine's names are
    /// literals, a game assembly's arrive at runtime.
    pub name: Cow<'static, str>,
    /// Process-local lookup key only — it is never written anywhere and never
    /// compared across builds. See [`ComponentId`] for why identity cannot be
    /// a `TypeId`. `None` for a component whose type lives in C#.
    pub type_id: Option<TypeId>,
    /// Present exactly when this component is a C# Behaviour's property bag.
    pub script: Option<ScriptBinding>,
    pub has: fn(&Self, &World, Entity) -> bool,
    /// `None` when the entity doesn't have this component.
    ///
    /// Returning an owned `Value` is load-bearing: the `Ref<'_, T>` guard over
    /// the component storage drops before this returns, so no caller can hold a
    /// world borrow across whatever it does next. That is what makes the
    /// registry safe to call from a script dispatch window, where holding a
    /// borrow across a call into C# is forbidden.
    pub read: fn(&Self, &World, Entity) -> Option<Value>,
    /// Insert or replace the component from `value`. A stale entity handle is a
    /// no-op, matching `World::insert`.
    pub write: fn(&Self, &mut World, Entity, &Value) -> Result<(), ValueError>,
    pub remove: fn(&Self, &mut World, Entity),
    /// The changes that would turn the live component into `target`, or `None`
    /// when the entity doesn't have it.
    ///
    /// Paired with [`apply`](Self::apply) rather than with `write` so that every
    /// edit in the engine can be *described* before it is performed: undo/redo,
    /// prefab overrides and collaboration sync all need the description, and
    /// architecture §4.4 asks that they share exactly one.
    pub diff: fn(&Self, &World, Entity, &Value) -> Option<Vec<FieldChange>>,
    /// Replay changes onto the live component, leaving it untouched if any of
    /// them does not fit.
    pub apply: fn(&Self, &mut World, Entity, &[FieldChange]) -> Result<(), ApplyError>,
    /// Draw the component's widgets, reporting whether the user moved any.
    ///
    /// Takes a detached [`Value`] rather than the world, so the caller diffs the
    /// result and routes the change through [`apply`](Self::apply) like every
    /// other edit — an inspector that wrote straight into storage would be the
    /// one mutation in the engine that undo and sync never saw.
    ///
    /// Defaults to [`inspect::value`](crate::inspect::value) and is replaced per
    /// type by [`Registry::set_inspector`]. Alone among these it needs nothing
    /// from the entry, so it takes no `&self`.
    #[cfg(feature = "egui")]
    pub inspect: crate::inspect::InspectFn,
    pub default: fn(&Self) -> Value,
}

/// The calling half of the table. A field and a method may share a name in
/// Rust, and here they deliberately do: `vtable.read(..)` is the method,
/// `(vtable.read)(..)` the raw pointer, and only the former should appear
/// outside this file.
impl ComponentVtable {
    pub fn has(&self, world: &World, entity: Entity) -> bool {
        (self.has)(self, world, entity)
    }

    pub fn read(&self, world: &World, entity: Entity) -> Option<Value> {
        (self.read)(self, world, entity)
    }

    pub fn write(
        &self,
        world: &mut World,
        entity: Entity,
        value: &Value,
    ) -> Result<(), ValueError> {
        (self.write)(self, world, entity, value)
    }

    pub fn remove(&self, world: &mut World, entity: Entity) {
        (self.remove)(self, world, entity)
    }

    pub fn diff(&self, world: &World, entity: Entity, target: &Value) -> Option<Vec<FieldChange>> {
        (self.diff)(self, world, entity, target)
    }

    pub fn apply(
        &self,
        world: &mut World,
        entity: Entity,
        changes: &[FieldChange],
    ) -> Result<(), ApplyError> {
        (self.apply)(self, world, entity, changes)
    }

    pub fn default(&self) -> Value {
        (self.default)(self)
    }

    /// The binding a script entry's own functions were built to read. Panicking
    /// is right: reaching here without one means a native entry was given a
    /// script function, which is a construction bug rather than input.
    fn binding(&self) -> &ScriptBinding {
        self.script
            .as_ref()
            .unwrap_or_else(|| panic!("`{}` is not a script component", self.id))
    }
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
        let type_id = TypeId::of::<T>();
        if let Some(&existing) = self.by_type.get(&type_id) {
            panic!(
                "`{name}` is already registered as `{}`; a type gets exactly one id",
                self.entries[existing].id
            );
        }

        let index = self.claim(&id, name);
        self.entries.push(ComponentVtable {
            id,
            name: Cow::Borrowed(name),
            type_id: Some(type_id),
            script: None,
            has: |_, world, entity| world.has::<T>(entity),
            read: |_, world, entity| world.get::<T>(entity).map(|c| c.to_value()),
            write: |_, world, entity, value| {
                // Converted before the world is touched, so a malformed value
                // leaves the existing component intact rather than half
                // replaced.
                let component = T::from_value(value)?;
                let _ = world.insert(entity, component);
                Ok(())
            },
            remove: |_, world, entity| {
                let _ = world.remove::<T>(entity);
            },
            diff: |_, world, entity, target| {
                world
                    .get::<T>(entity)
                    .map(|c| crate::diff::diff(&c.to_value(), target))
            },
            apply: |_, world, entity, changes| {
                // Same discipline as `write`: the whole batch is replayed onto a
                // detached value and converted back before the world is touched,
                // so a change that does not fit leaves the component as it was.
                let Some(mut value) = world.get::<T>(entity).map(|c| c.to_value()) else {
                    return Err(ApplyError::Absent);
                };
                crate::diff::apply(&mut value, changes).map_err(ApplyError::Value)?;
                let component = T::from_value(&value).map_err(ApplyError::Value)?;
                let _ = world.insert(entity, component);
                Ok(())
            },
            #[cfg(feature = "egui")]
            inspect: crate::inspect::value,
            default: |_| T::default().to_value(),
        });
        self.by_type.insert(type_id, index);
    }

    /// Describe `T`, whose file form the world translates, under `id`.
    ///
    /// Every operation goes through [`Resolve`] rather than [`Reflect`], and is
    /// otherwise what [`register`](Self::register) builds: `write` and `apply`
    /// convert before touching the world, so a value that does not resolve
    /// leaves the component as it was.
    ///
    /// # Panics
    /// As [`register`](Self::register) does.
    pub fn register_resolved<T: Resolve>(&mut self, id: ComponentId, name: &'static str) {
        let type_id = TypeId::of::<T>();
        if let Some(&existing) = self.by_type.get(&type_id) {
            panic!(
                "`{name}` is already registered as `{}`; a type gets exactly one id",
                self.entries[existing].id
            );
        }

        let index = self.claim(&id, name);
        self.entries.push(ComponentVtable {
            id,
            name: Cow::Borrowed(name),
            type_id: Some(type_id),
            script: None,
            has: |_, world, entity| world.has::<T>(entity),
            read: |_, world, entity| world.get::<T>(entity)?.to_value(world),
            write: |_, world, entity, value| {
                let component = T::from_value(world, value)?;
                let _ = world.insert(entity, component);
                Ok(())
            },
            remove: |_, world, entity| {
                let _ = world.remove::<T>(entity);
            },
            diff: |v, world, entity, target| {
                v.read(world, entity)
                    .map(|live| crate::diff::diff(&live, target))
            },
            apply: |v, world, entity, changes| {
                let Some(mut value) = v.read(world, entity) else {
                    return Err(ApplyError::Absent);
                };
                crate::diff::apply(&mut value, changes).map_err(ApplyError::Value)?;
                v.write(world, entity, &value).map_err(ApplyError::Value)
            },
            #[cfg(feature = "egui")]
            inspect: crate::inspect::value,
            default: |_| T::default_value(),
        });
        self.by_type.insert(type_id, index);
    }

    /// Describe a C# Behaviour to the engine, so its fields save, load and
    /// inspect exactly as a Rust component's do.
    ///
    /// Every operation crosses the FFI as one flattened buffer for the whole
    /// component, never field by field: a property bag read one field at a time
    /// costs a managed transition per field, and — worse — can observe a
    /// Behaviour halfway through its own `Update`.
    ///
    /// Called from the game assembly's `register_components` and re-run after
    /// each hot reload, with [`clear_game`](Self::clear_game) in between.
    ///
    /// # Panics
    /// If `id` is already registered, for the reason [`register`](Self::register)
    /// gives.
    pub fn register_script(
        &mut self,
        id: ComponentId,
        name: impl Into<String>,
        type_name: impl Into<String>,
        bridge: Arc<dyn ScriptBridge>,
    ) {
        let name = name.into();
        self.claim(&id, &name);
        self.entries.push(ComponentVtable {
            id,
            name: Cow::Owned(name),
            // No Rust type, so no `of::<T>()` lookup and no `by_type` entry.
            type_id: None,
            script: Some(ScriptBinding {
                type_name: type_name.into(),
                bridge,
            }),
            has: |v, world, entity| {
                let b = v.binding();
                b.bridge.has(world, entity, &b.type_name)
            },
            read: |v, world, entity| {
                let b = v.binding();
                b.bridge.read(world, entity, &b.type_name)
            },
            write: |v, world, entity, value| {
                let b = v.binding();
                b.bridge.write(world, entity, &b.type_name, value)
            },
            remove: |v, world, entity| {
                let b = v.binding();
                b.bridge.remove(world, entity, &b.type_name);
            },
            // Both are the generic `Value` walk over what `read` handed back,
            // rather than anything managed: a diff computed in C# would be a
            // second implementation of the rule in `diff.rs`, and the two would
            // disagree about NaN within a release.
            diff: |v, world, entity, target| {
                v.read(world, entity)
                    .map(|live| crate::diff::diff(&live, target))
            },
            apply: |v, world, entity, changes| {
                let Some(mut value) = v.read(world, entity) else {
                    return Err(ApplyError::Absent);
                };
                crate::diff::apply(&mut value, changes).map_err(ApplyError::Value)?;
                v.write(world, entity, &value).map_err(ApplyError::Value)
            },
            #[cfg(feature = "egui")]
            inspect: crate::inspect::value,
            default: |v| {
                let b = v.binding();
                b.bridge.default(&b.type_name)
            },
        });
    }

    /// Reserve `id`'s slot, or explain whose it already is.
    fn claim(&mut self, id: &ComponentId, name: &str) -> usize {
        if let Some(&existing) = self.by_id.get(id) {
            panic!(
                "component id `{id}` is already registered by `{}` (attempted by `{name}`)",
                self.entries[existing].name
            );
        }
        let index = self.entries.len();
        self.by_id.insert(id.clone(), index);
        index
    }

    /// Replace one component's inspector with widgets that know what its numbers
    /// mean — a logarithmic slider in lux, an angle in degrees, a colour picker.
    ///
    /// Separate from [`register`](Self::register) so that a component type stays
    /// registerable from a crate with no UI in scope: the engine's components are
    /// declared in `scene`, and their tuned widgets belong to the editor.
    ///
    /// # Panics
    /// If `id` is not registered — the call is startup code, and a typo would
    /// otherwise leave a component silently drawing generic drag fields.
    #[cfg(feature = "egui")]
    pub fn set_inspector(&mut self, id: &ComponentId, inspect: crate::inspect::InspectFn) {
        let index = *self
            .by_id
            .get(id)
            .unwrap_or_else(|| panic!("no component is registered as `{id}`"));
        self.entries[index].inspect = inspect;
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
            if let Some(type_id) = entry.type_id {
                self.by_type.insert(type_id, index);
            }
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
    use crate::value::FieldPath;

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

    fn changed(field: &str, value: Value) -> FieldChange {
        let mut path = FieldPath::empty();
        path.push(crate::value::PathSegment::Field(field.to_owned()));
        FieldChange { path, value }
    }

    #[test]
    fn diff_and_apply_move_a_component_through_field_changes() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Label("before".to_owned()));

        let label = registry.get(&ComponentId::new("test.label")).unwrap();
        let target = Value::strukt([("text", Value::String("after".to_owned()))]);
        let changes = label.diff(&world, entity, &target).unwrap();
        assert_eq!(
            changes,
            vec![changed("text", Value::String("after".to_owned()))]
        );

        label.apply(&mut world, entity, &changes).unwrap();
        assert_eq!(
            *world.get::<Label>(entity).unwrap(),
            Label("after".to_owned())
        );
    }

    #[test]
    fn diffing_a_component_against_itself_reports_nothing() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Speed(3.0));

        let speed = registry.get(&ComponentId::new("test.speed")).unwrap();
        let live = speed.read(&world, entity).unwrap();
        assert_eq!(speed.diff(&world, entity, &live), Some(Vec::new()));
    }

    /// A component the entity does not have is not silently created and then
    /// edited: the changes name only some fields, so the result would look like
    /// a successful edit while every unnamed field quietly took its default.
    #[test]
    fn applying_to_an_absent_component_says_so() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn();

        let speed = registry.get(&ComponentId::new("test.speed")).unwrap();
        let err = speed.apply(&mut world, entity, &[]).unwrap_err();
        assert_eq!(err, ApplyError::Absent);
        assert_eq!(err.to_string(), "the entity does not have this component");
        assert!(!world.has::<Speed>(entity));
    }

    #[test]
    fn a_change_the_type_refuses_leaves_the_component_alone() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Label("ok".to_owned()));

        let label = registry.get(&ComponentId::new("test.label")).unwrap();
        let err = label
            .apply(&mut world, entity, &[changed("text", Value::F32(1.0))])
            .unwrap_err();
        assert_eq!(err.to_string(), "field `text`: expected string, found f32");
        assert_eq!(*world.get::<Label>(entity).unwrap(), Label("ok".to_owned()));
    }

    #[test]
    fn a_change_naming_a_field_that_is_not_there_is_refused() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Label("ok".to_owned()));

        let label = registry.get(&ComponentId::new("test.label")).unwrap();
        let err = label
            .apply(&mut world, entity, &[changed("caption", Value::F32(1.0))])
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "field `caption`: expected a value, found nothing"
        );
    }

    /// The point of hanging the inspector off the vtable rather than off one
    /// global function: a component whose numbers mean something replaces its
    /// widgets without every *other* component's changing.
    #[cfg(feature = "egui")]
    #[test]
    fn a_registered_inspector_replaces_only_its_own_components() {
        use std::cell::RefCell;

        fn tuned(_ui: &mut egui::Ui, value: &mut Value) -> bool {
            *value = Value::String("tuned".to_owned());
            true
        }

        let mut registry = registry();
        registry.set_inspector(&ComponentId::new("test.label"), tuned);

        let label = registry.get(&ComponentId::new("test.label")).unwrap();
        let drawn = RefCell::new(Value::F32(0.0));
        egui::__run_test_ui(|ui| assert!((label.inspect)(ui, &mut drawn.borrow_mut())));
        assert_eq!(drawn.into_inner(), Value::String("tuned".to_owned()));

        let speed = registry.get(&ComponentId::new("test.speed")).unwrap();
        let untouched = RefCell::new(Value::F32(1.0));
        egui::__run_test_ui(|ui| assert!(!(speed.inspect)(ui, &mut untouched.borrow_mut())));
        assert_eq!(untouched.into_inner(), Value::F32(1.0));
    }

    #[cfg(feature = "egui")]
    #[test]
    #[should_panic(expected = "no component is registered as `test.nothing`")]
    fn an_inspector_for_an_unregistered_id_panics() {
        fn nothing(_ui: &mut egui::Ui, _value: &mut Value) -> bool {
            false
        }
        registry().set_inspector(&ComponentId::new("test.nothing"), nothing);
    }

    /// Stands in for `Orrin.PropertyBag` across the FFI. It stores bags by
    /// managed type name, which is the whole point of the test: an entry with
    /// no Rust type behind it must still reach *its own* type name, and two
    /// script components sharing one bridge must not reach each other's.
    struct FakeScripts {
        bags: std::sync::Mutex<HashMap<String, Value>>,
    }

    impl FakeScripts {
        fn with(bags: [(&str, Value); 2]) -> Arc<Self> {
            let this = FakeScripts {
                bags: std::sync::Mutex::new(HashMap::new()),
            };
            for (name, value) in bags {
                this.bags.lock().unwrap().insert(name.to_owned(), value);
            }
            Arc::new(this)
        }
    }

    impl ScriptBridge for FakeScripts {
        fn has(&self, _: &World, _: Entity, type_name: &str) -> bool {
            self.bags.lock().unwrap().contains_key(type_name)
        }

        fn read(&self, _: &World, _: Entity, type_name: &str) -> Option<Value> {
            self.bags.lock().unwrap().get(type_name).cloned()
        }

        fn write(
            &self,
            _: &mut World,
            _: Entity,
            type_name: &str,
            value: &Value,
        ) -> Result<(), ValueError> {
            match self.bags.lock().unwrap().get_mut(type_name) {
                Some(slot) => {
                    *slot = value.clone();
                    Ok(())
                }
                None => Err(ValueError::invalid("a live script component", "nothing")),
            }
        }

        fn remove(&self, _: &mut World, _: Entity, type_name: &str) {
            self.bags.lock().unwrap().remove(type_name);
        }

        fn default(&self, type_name: &str) -> Value {
            Value::strukt([("type", Value::String(type_name.to_owned()))])
        }
    }

    fn scripted() -> (Registry, Arc<FakeScripts>) {
        let bridge = FakeScripts::with([
            (
                "Game.Spinner, Game",
                Value::strukt([("speed", Value::F32(1.0))]),
            ),
            (
                "Game.Hover, Game",
                Value::strukt([("height", Value::F32(2.0))]),
            ),
        ]);
        let mut registry = registry();
        registry.register_script(
            ComponentId::owned("game.spinner"),
            "Spinner",
            "Game.Spinner, Game",
            bridge.clone(),
        );
        registry.register_script(
            ComponentId::owned("game.hover"),
            "Hover",
            "Game.Hover, Game",
            bridge.clone(),
        );
        (registry, bridge)
    }

    /// Two entries built from the same closures and the same bridge must still
    /// address different managed types. This is the whole reason the vtable's
    /// functions take `&self`: without it they would have no way to tell which
    /// registration they belong to.
    #[test]
    fn two_script_components_sharing_a_bridge_do_not_reach_each_others_fields() {
        let (registry, _bridge) = scripted();
        let mut world = World::new();
        let entity = world.spawn();

        let spinner = registry.get(&ComponentId::owned("game.spinner")).unwrap();
        let hover = registry.get(&ComponentId::owned("game.hover")).unwrap();

        assert_eq!(
            spinner.read(&world, entity),
            Some(Value::strukt([("speed", Value::F32(1.0))]))
        );
        assert_eq!(
            hover.read(&world, entity),
            Some(Value::strukt([("height", Value::F32(2.0))]))
        );
        assert_eq!(
            spinner.default(),
            Value::strukt([("type", Value::String("Game.Spinner, Game".to_owned()))])
        );

        spinner
            .write(
                &mut world,
                entity,
                &Value::strukt([("speed", Value::F32(9.0))]),
            )
            .unwrap();
        assert_eq!(
            spinner.read(&world, entity),
            Some(Value::strukt([("speed", Value::F32(9.0))]))
        );
        assert_eq!(
            hover.read(&world, entity),
            Some(Value::strukt([("height", Value::F32(2.0))])),
            "writing one script component must not touch another's bag"
        );
    }

    /// A C# component gets the same diff and apply as a Rust one, computed on
    /// this side of the boundary — so undo, prefab overrides and sync treat the
    /// two identically, which is the point of the whole registration.
    #[test]
    fn a_script_component_diffs_and_applies_like_a_rust_one() {
        let (registry, _bridge) = scripted();
        let mut world = World::new();
        let entity = world.spawn();
        let spinner = registry.get(&ComponentId::owned("game.spinner")).unwrap();

        let target = Value::strukt([("speed", Value::F32(4.0))]);
        let changes = spinner.diff(&world, entity, &target).unwrap();
        assert_eq!(changes, vec![changed("speed", Value::F32(4.0))]);

        spinner.apply(&mut world, entity, &changes).unwrap();
        assert_eq!(spinner.read(&world, entity), Some(target));
    }

    #[test]
    fn a_script_component_that_is_gone_reports_absent_rather_than_writing() {
        let (registry, bridge) = scripted();
        let mut world = World::new();
        let entity = world.spawn();
        let spinner = registry.get(&ComponentId::owned("game.spinner")).unwrap();

        assert!(spinner.has(&world, entity));
        bridge.bags.lock().unwrap().remove("Game.Spinner, Game");

        assert!(!spinner.has(&world, entity));
        assert_eq!(spinner.read(&world, entity), None);
        assert_eq!(spinner.diff(&world, entity, &Value::Bool(true)), None);
        assert_eq!(
            spinner.apply(&mut world, entity, &[]).unwrap_err(),
            ApplyError::Absent
        );
    }

    /// The reload contract: the previous build's entries go, the engine's stay,
    /// and the `by_type` map — which script entries are deliberately absent
    /// from — is rebuilt without them.
    #[test]
    fn clearing_game_entries_drops_script_components_and_leaves_lookups_intact() {
        let (mut registry, _bridge) = scripted();
        assert_eq!(registry.len(), 4);

        registry.clear_game();
        assert_eq!(registry.len(), 2);
        assert!(registry.get(&ComponentId::owned("game.spinner")).is_none());
        assert!(registry.of::<Speed>().is_some());
        assert_eq!(
            registry.of::<Speed>().unwrap().id,
            ComponentId::new("test.speed")
        );
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn a_script_component_cannot_take_an_engine_id() {
        let bridge = FakeScripts::with([("A", Value::Bool(true)), ("B", Value::Bool(true))]);
        registry().register_script(ComponentId::new("test.speed"), "Speed", "A", bridge);
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
        assert!(!speed.has(&world, entity));
        assert_eq!(speed.read(&world, entity), None);

        speed.write(&mut world, entity, &Value::F32(4.5)).unwrap();
        assert!(speed.has(&world, entity));
        assert_eq!(speed.read(&world, entity), Some(Value::F32(4.5)));
        assert_eq!(*world.get::<Speed>(entity).unwrap(), Speed(4.5));

        speed.remove(&mut world, entity);
        assert!(!speed.has(&world, entity));
    }

    #[test]
    fn a_bad_value_reports_its_field_and_leaves_the_component_alone() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn();

        let label = registry.get(&ComponentId::new("test.label")).unwrap();
        let ok = Value::strukt([("text", Value::String("ok".to_owned()))]);
        label.write(&mut world, entity, &ok).unwrap();

        let bad = Value::strukt([("text", Value::F32(1.0))]);
        let err = label.write(&mut world, entity, &bad).unwrap_err();
        assert_eq!(err.to_string(), "field `text`: expected string, found f32");
        assert_eq!(*world.get::<Label>(entity).unwrap(), Label("ok".to_owned()));
    }

    #[test]
    fn defaults_come_from_the_type() {
        let registry = registry();
        let speed = registry.get(&ComponentId::new("test.speed")).unwrap();
        assert_eq!(speed.default(), Value::F32(0.0));
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

    /// A session-local slot that the file names by what it stands for, the
    /// shape `MeshHandle` has: the number means nothing in another session, and
    /// only the world knows what it stands for in this one.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Slot(u32);

    /// The world's table for [`Slot`]: index is the slot, entry its name.
    struct Names(Vec<&'static str>);

    impl Resolve for Slot {
        fn to_value(&self, world: &World) -> Option<Value> {
            let names = world.get_resource::<Names>()?;
            let name = names.0.get(self.0 as usize)?;
            Some(Value::String((*name).to_owned()))
        }

        fn from_value(world: &World, value: &Value) -> Result<Self, ValueError> {
            let name = String::from_value(value)?;
            let names = world.resource::<Names>();
            names
                .0
                .iter()
                .position(|known| *known == name)
                .map(|slot| Self(slot as u32))
                .ok_or_else(|| ValueError::invalid("a known name", name))
        }

        fn default_value() -> Value {
            Value::String("first".to_owned())
        }
    }

    fn resolving() -> (Registry, World, Entity) {
        let mut registry = Registry::new();
        registry.register_resolved::<Slot>(ComponentId::new("test.slot"), "Slot");
        let mut world = World::new();
        world.insert_resource(Names(vec!["first", "second"]));
        let entity = world.spawn();
        (registry, world, entity)
    }

    #[test]
    fn a_resolved_component_is_read_and_written_in_the_worlds_terms() {
        let (registry, mut world, entity) = resolving();
        let slot = registry.get(&ComponentId::new("test.slot")).unwrap();

        slot.write(&mut world, entity, &Value::String("second".to_owned()))
            .unwrap();
        assert_eq!(*world.get::<Slot>(entity).unwrap(), Slot(1));
        assert_eq!(
            slot.read(&world, entity),
            Some(Value::String("second".to_owned()))
        );
        assert!(registry.of::<Slot>().is_some());
        assert_eq!(slot.default(), Value::String("first".to_owned()));
    }

    #[test]
    fn a_resolved_component_the_world_cannot_name_reads_as_absent() {
        let (registry, mut world, entity) = resolving();
        world.insert(entity, Slot(9));
        let slot = registry.get(&ComponentId::new("test.slot")).unwrap();
        assert!(slot.has(&world, entity));
        assert_eq!(slot.read(&world, entity), None);
        assert_eq!(
            slot.diff(&world, entity, &Value::String("first".to_owned())),
            None
        );
    }

    #[test]
    fn a_resolved_component_diffs_and_applies_through_the_world() {
        let (registry, mut world, entity) = resolving();
        world.insert(entity, Slot(0));
        let slot = registry.get(&ComponentId::new("test.slot")).unwrap();

        let target = Value::String("second".to_owned());
        let changes = slot.diff(&world, entity, &target).unwrap();
        assert_eq!(changes.len(), 1);
        slot.apply(&mut world, entity, &changes).unwrap();
        assert_eq!(*world.get::<Slot>(entity).unwrap(), Slot(1));

        let unknown = [FieldChange {
            path: FieldPath::empty(),
            value: Value::String("third".to_owned()),
        }];
        let err = slot.apply(&mut world, entity, &unknown).unwrap_err();
        assert_eq!(
            err.to_string(),
            "expected a known name, found third"
        );
        assert_eq!(*world.get::<Slot>(entity).unwrap(), Slot(1));
    }
}
