use std::collections::HashMap;

use crate::gfx::TextureHandle;
use crate::scene::{MaterialHandle, MeshHandle};

/// Maps names to the opaque handles the backend returns on upload, so scene and
/// editor code can say `assets.mesh("cube")` instead of threading handle locals.
#[derive(Default)]
pub struct Assets {
    meshes: HashMap<String, MeshHandle>,
    materials: HashMap<String, MaterialHandle>,
    textures: HashMap<String, TextureHandle>,
}

impl Assets {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_mesh(&mut self, name: impl Into<String>, handle: MeshHandle) {
        self.meshes.insert(name.into(), handle);
    }

    pub fn mesh(&self, name: &str) -> Option<MeshHandle> {
        self.meshes.get(name).copied()
    }

    pub fn meshes(&self) -> impl Iterator<Item = (&str, MeshHandle)> {
        self.meshes.iter().map(|(n, &h)| (n.as_str(), h))
    }

    pub fn insert_material(&mut self, name: impl Into<String>, handle: MaterialHandle) {
        self.materials.insert(name.into(), handle);
    }

    pub fn material(&self, name: &str) -> Option<MaterialHandle> {
        self.materials.get(name).copied()
    }

    pub fn materials(&self) -> impl Iterator<Item = (&str, MaterialHandle)> {
        self.materials.iter().map(|(n, &h)| (n.as_str(), h))
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
