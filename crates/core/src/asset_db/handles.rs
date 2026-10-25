//! Handles to ids and back: the table each kind of slot keeps, and the
//! [`Resolve`] impls that let `orrin.mesh` and `orrin.material` register as
//! ordinary components.
//!
//! A scene names a mesh by [`AssetId`]; a component holds the slot. Reading a
//! component asks the database which asset its slot was issued for, or
//! [`Assets`] for an engine-built mesh, and writing one asks the reverse — so
//! the file says `asset(…)` and the frame's loops still see four bytes.
//!
//! A scene saved before assets had ids names them by string, and still loads:
//! the name is looked up in [`Assets`], and the next save writes the id.

use orrin_ecs::{FxHashMap, World};
use orrin_registry::{AssetId, Resolve, Value, ValueError};
use xxhash_rust::xxh3::xxh3_128;

use super::{AssetDatabase, CacheKey, LoadState, PENDING};
use crate::scene::{Assets, MaterialHandle, MeshHandle};

/// The id of an engine-built asset: the `kind` and `name` [`Assets`] registers
/// it under, hashed into a version-8 UUID.
///
/// Derived rather than assigned because there is no file to keep an assigned
/// one in, and a name is exactly what a scene stored for these before they had
/// ids. Version 8 is the custom-UUID version, so no derived id can equal a
/// random (version 4) one a sidecar holds.
pub fn named_id(kind: &str, name: &str) -> AssetId {
    let mut input = Vec::with_capacity(kind.len() + name.len() + 13);
    input.extend_from_slice(b"orrin.asset\0");
    input.extend_from_slice(kind.as_bytes());
    input.push(0);
    input.extend_from_slice(name.as_bytes());
    let mut bytes = xxh3_128(&input).to_be_bytes();
    bytes[6] = (bytes[6] & 0x0f) | 0x80;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    AssetId::from_bytes(bytes)
}

/// What a slot holds.
#[derive(Clone, Copy, Debug)]
pub(super) struct Resident {
    pub state: LoadState,
    /// The artifact in the slot; `None` while it holds a placeholder.
    pub key: Option<CacheKey>,
}

/// One kind's handles: which id each was issued for, and what its slot holds.
#[derive(Default)]
pub(super) struct Slots {
    by_id: FxHashMap<AssetId, u32>,
    by_handle: FxHashMap<u32, AssetId>,
    /// Pending handles not yet bound, as `(handle, id)`.
    pending: Vec<(u32, AssetId)>,
    issued: u32,
    resident: FxHashMap<AssetId, Resident>,
}

impl Slots {
    pub fn handle(&self, id: AssetId) -> Option<u32> {
        self.by_id.get(&id).copied()
    }

    pub fn id(&self, handle: u32) -> Option<AssetId> {
        self.by_handle.get(&handle).copied()
    }

    /// `id`'s handle, issuing a pending one if it has none.
    pub fn promise(&mut self, id: AssetId) -> u32 {
        if let Some(handle) = self.handle(id) {
            return handle;
        }
        let handle = PENDING | self.issued;
        self.issued += 1;
        self.pending.push((handle, id));
        self.by_id.insert(id, handle);
        self.by_handle.insert(handle, id);
        handle
    }

    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub fn take_pending(&mut self) -> Vec<(u32, AssetId)> {
        std::mem::take(&mut self.pending)
    }

    /// Give `id` the real slot `handle`, retiring whatever pending one it had.
    pub fn bind(&mut self, id: AssetId, handle: u32, state: LoadState) {
        if let Some(old) = self.by_id.insert(id, handle) {
            self.by_handle.remove(&old);
        }
        self.by_handle.insert(handle, id);
        self.resident.insert(id, Resident { state, key: None });
    }

    pub fn resident(&self, id: AssetId) -> Option<Resident> {
        self.resident.get(&id).copied()
    }

    pub fn set_resident(&mut self, id: AssetId, resident: Resident) {
        self.resident.insert(id, resident);
    }
}

impl Resolve for MeshHandle {
    fn to_value(&self, world: &World) -> Option<Value> {
        let from_db = world
            .get_resource::<AssetDatabase>()
            .and_then(|db| db.mesh_id(*self));
        let id = from_db.or_else(|| world.get_resource::<Assets>()?.mesh_id(*self))?;
        Some(Value::Asset(id))
    }

    fn from_value(world: &World, value: &Value) -> Result<Self, ValueError> {
        match value {
            Value::Asset(id) => {
                if let Some(handle) = world.get_resource::<Assets>().and_then(|a| a.mesh_by_id(*id)) {
                    return Ok(handle);
                }
                let mut db = database(world, "a mesh", *id)?;
                db.mesh(*id)
                    .map_err(|found| ValueError::invalid("a mesh asset", found))
            }
            Value::String(name) => world
                .get_resource::<Assets>()
                .and_then(|assets| assets.mesh(name))
                .ok_or_else(|| ValueError::invalid("the name of a mesh", format!("`{name}`"))),
            other => Err(ValueError::mismatch("asset", other)),
        }
    }

    fn default_value() -> Value {
        Value::Asset(named_id("mesh", "cube"))
    }
}

impl Resolve for MaterialHandle {
    fn to_value(&self, world: &World) -> Option<Value> {
        let from_db = world
            .get_resource::<AssetDatabase>()
            .and_then(|db| db.material_id(*self));
        let id = from_db.or_else(|| world.get_resource::<Assets>()?.material_id(*self))?;
        Some(Value::Asset(id))
    }

    fn from_value(world: &World, value: &Value) -> Result<Self, ValueError> {
        match value {
            Value::Asset(id) => {
                let named = world
                    .get_resource::<Assets>()
                    .and_then(|assets| assets.material_by_id(*id));
                if let Some(handle) = named {
                    return Ok(handle);
                }
                Ok(database(world, "a material", *id)?.material(*id))
            }
            Value::String(name) => world
                .get_resource::<Assets>()
                .and_then(|assets| assets.material(name))
                .ok_or_else(|| ValueError::invalid("the name of a material", format!("`{name}`"))),
            other => Err(ValueError::mismatch("asset", other)),
        }
    }

    fn default_value() -> Value {
        Value::Asset(named_id("material", "clay"))
    }
}

/// The database, for an id [`Assets`] did not know. The null id is refused
/// here rather than given a placeholder: it is what an unassigned reference
/// holds, and no asset will ever have it.
fn database<'w>(
    world: &'w World,
    expected: &'static str,
    id: AssetId,
) -> Result<std::cell::RefMut<'w, AssetDatabase>, ValueError> {
    if id.is_nil() {
        return Err(ValueError::invalid(expected, "the null reference"));
    }
    world.get_resource_mut::<AssetDatabase>().ok_or_else(|| {
        ValueError::invalid(
            expected,
            format!("{id}, and there is no asset database to look it up in"),
        )
    })
}

#[cfg(test)]
mod tests {
    use orrin_registry::{ComponentId, Registry};

    use super::*;
    use crate::scene::register_components;
    use crate::scene::registry::{MATERIAL, MESH};

    #[test]
    fn a_named_id_is_stable_versioned_and_kind_qualified() {
        let cube = named_id("mesh", "cube");
        assert_eq!(cube, named_id("mesh", "cube"));
        assert_ne!(cube, named_id("material", "cube"));
        assert_ne!(cube, named_id("mesh", "cube "));
        let bytes = cube.to_bytes();
        assert_eq!(bytes[6] >> 4, 8, "version 8");
        assert_eq!(bytes[8] >> 6, 0b10, "RFC 4122 variant");
        // Pinned: a derived id is what saved scenes hold, so the derivation
        // can never change without every one of them losing its meshes.
        assert_eq!(cube.to_string(), named_id("mesh", "cube").to_string());
        assert_eq!(
            named_id("mesh", "cube").to_string(),
            "c4d1a0f6-9f3e-8c43-9b8e-2b1f0c8d7a51",
            "if this moved, saved scenes no longer find their meshes"
        );
    }

    fn world() -> (Registry, World, orrin_ecs::Entity) {
        let mut registry = Registry::new();
        register_components(&mut registry);
        let mut world = World::new();
        let mut assets = Assets::new();
        assets.insert_mesh("cube", MeshHandle(3));
        assets.insert_material("gold", MaterialHandle(7));
        world.insert_resource(assets);
        let entity = world.spawn();
        (registry, world, entity)
    }

    fn write(registry: &Registry, world: &mut World, entity: orrin_ecs::Entity, id: &ComponentId, value: Value) -> Result<(), String> {
        registry
            .get(id)
            .unwrap()
            .write(world, entity, &value)
            .map_err(|error| error.to_string())
    }

    #[test]
    fn an_engine_mesh_is_written_and_read_by_its_derived_id() {
        let (registry, mut world, entity) = world();
        let cube = Value::Asset(named_id("mesh", "cube"));
        write(&registry, &mut world, entity, &MESH, cube.clone()).unwrap();
        assert_eq!(*world.get::<MeshHandle>(entity).unwrap(), MeshHandle(3));
        assert_eq!(registry.get(&MESH).unwrap().read(&world, entity), Some(cube));

        let gold = Value::Asset(named_id("material", "gold"));
        write(&registry, &mut world, entity, &MATERIAL, gold.clone()).unwrap();
        assert_eq!(registry.get(&MATERIAL).unwrap().read(&world, entity), Some(gold));
    }

    /// The format scenes were saved in before assets had ids.
    #[test]
    fn a_mesh_named_by_string_still_loads_and_reads_back_as_its_id() {
        let (registry, mut world, entity) = world();
        write(&registry, &mut world, entity, &MESH, Value::String("cube".to_owned())).unwrap();
        assert_eq!(*world.get::<MeshHandle>(entity).unwrap(), MeshHandle(3));
        assert_eq!(
            registry.get(&MESH).unwrap().read(&world, entity),
            Some(Value::Asset(named_id("mesh", "cube")))
        );

        let err = write(&registry, &mut world, entity, &MESH, Value::String("nope".to_owned()))
            .unwrap_err();
        assert_eq!(err, "expected the name of a mesh, found `nope`");
    }

    #[test]
    fn an_id_nothing_knows_is_refused_without_a_database_and_promised_with_one() {
        let (registry, mut world, entity) = world();
        let id = AssetId::new();
        let err = write(&registry, &mut world, entity, &MESH, Value::Asset(id)).unwrap_err();
        assert_eq!(
            err,
            format!("expected a mesh asset, found {id}, and there is no asset database to look it up in")
                .replacen("a mesh asset", "a mesh", 1)
        );

        let dir = crate::asset_db::tests::Dir::new("handles-promise");
        let (db, _) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        world.insert_resource(db);
        write(&registry, &mut world, entity, &MESH, Value::Asset(id)).unwrap();
        let handle = *world.get::<MeshHandle>(entity).unwrap();
        assert_ne!(handle.0 & PENDING, 0);
        assert_eq!(registry.get(&MESH).unwrap().read(&world, entity), Some(Value::Asset(id)));

        let material = AssetId::new();
        write(&registry, &mut world, entity, &MATERIAL, Value::Asset(material)).unwrap();
        assert_eq!(
            registry.get(&MATERIAL).unwrap().read(&world, entity),
            Some(Value::Asset(material))
        );
    }

    #[test]
    fn the_null_reference_is_never_a_mesh() {
        let (registry, mut world, entity) = world();
        let err = write(&registry, &mut world, entity, &MESH, Value::Asset(AssetId::NIL)).unwrap_err();
        assert_eq!(err, "expected a mesh, found the null reference");
    }

    #[test]
    fn a_new_mesh_renderer_starts_as_the_cube() {
        let (registry, mut world, entity) = world();
        let default = registry.get(&MESH).unwrap().default();
        write(&registry, &mut world, entity, &MESH, default).unwrap();
        assert_eq!(*world.get::<MeshHandle>(entity).unwrap(), MeshHandle(3));
    }
}
