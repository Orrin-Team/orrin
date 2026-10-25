use std::collections::HashMap;

use orrin_registry::AssetId;

use crate::asset_db::named_id;
use crate::gfx::TextureHandle;
use crate::scene::{MaterialHandle, MeshHandle};

/// Maps names to the opaque handles the backend returns on upload, so scene and
/// editor code can say `assets.mesh("cube")` instead of threading handle locals.
///
/// Each name also answers to the id [`named_id`] derives from it, which is how
/// a scene references an engine-built asset.
#[derive(Default)]
pub struct Assets {
    meshes: HashMap<String, MeshHandle>,
    materials: HashMap<String, MaterialHandle>,
    textures: HashMap<String, TextureHandle>,
    mesh_ids: HashMap<AssetId, MeshHandle>,
    material_ids: HashMap<AssetId, MaterialHandle>,
}

impl Assets {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_mesh(&mut self, name: impl Into<String>, handle: MeshHandle) {
        let name = name.into();
        self.mesh_ids.insert(named_id("mesh", &name), handle);
        self.meshes.insert(name, handle);
    }

    pub fn mesh(&self, name: &str) -> Option<MeshHandle> {
        self.meshes.get(name).copied()
    }

    pub fn meshes(&self) -> impl Iterator<Item = (&str, MeshHandle)> {
        self.meshes.iter().map(|(n, &h)| (n.as_str(), h))
    }

    /// The name a handle was registered under — what a scene file stores, since
    /// the handle itself is an upload index with no meaning across sessions.
    /// Linear, over a map with a handful of entries, on the save path only.
    pub fn mesh_name(&self, handle: MeshHandle) -> Option<&str> {
        self.meshes
            .iter()
            .find(|&(_, h)| *h == handle)
            .map(|(n, _)| n.as_str())
    }

    /// The id a scene stores for a mesh registered here.
    pub fn mesh_id(&self, handle: MeshHandle) -> Option<AssetId> {
        self.mesh_name(handle).map(|name| named_id("mesh", name))
    }

    pub fn mesh_by_id(&self, id: AssetId) -> Option<MeshHandle> {
        self.mesh_ids.get(&id).copied()
    }

    pub fn insert_material(&mut self, name: impl Into<String>, handle: MaterialHandle) {
        let name = name.into();
        self.material_ids
            .insert(named_id("material", &name), handle);
        self.materials.insert(name, handle);
    }

    pub fn material(&self, name: &str) -> Option<MaterialHandle> {
        self.materials.get(name).copied()
    }

    pub fn materials(&self) -> impl Iterator<Item = (&str, MaterialHandle)> {
        self.materials.iter().map(|(n, &h)| (n.as_str(), h))
    }

    /// See [`mesh_name`](Self::mesh_name).
    pub fn material_name(&self, handle: MaterialHandle) -> Option<&str> {
        self.materials
            .iter()
            .find(|&(_, h)| *h == handle)
            .map(|(n, _)| n.as_str())
    }

    /// See [`mesh_id`](Self::mesh_id).
    pub fn material_id(&self, handle: MaterialHandle) -> Option<AssetId> {
        self.material_name(handle)
            .map(|name| named_id("material", name))
    }

    pub fn material_by_id(&self, id: AssetId) -> Option<MaterialHandle> {
        self.material_ids.get(&id).copied()
    }

    pub fn insert_texture(&mut self, name: impl Into<String>, handle: TextureHandle) {
        self.textures.insert(name.into(), handle);
    }

    pub fn texture(&self, name: &str) -> Option<TextureHandle> {
        self.textures.get(name).copied()
    }

    pub fn textures(&self) -> impl Iterator<Item = (&str, TextureHandle)> {
        self.textures.iter().map(|(n, &h)| (n.as_str(), h))
    }
}
