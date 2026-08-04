//! A [`RenderBackend`] that uploads nothing.
//!
//! Startup is more than Vulkan: meshes are generated, textures are decoded and
//! synthesized, and the scene graph is built, all on the CPU and all before the
//! first frame. That half is what decays as dependencies accumulate, and it is
//! the half a machine with no GPU can still measure — which is what lets the
//! cold-start guard in `tests/cold_start.rs` run in CI.
//!
//! Handles are issued in upload order, exactly as a real backend does, so the
//! [`Assets`](crate::scene::Assets) table and [`MeshBounds`] a headless run
//! produces are indistinguishable from a live one's.

use crate::geom::Aabb;
use crate::gfx::{Material, RenderBackend, RenderItem, SceneLighting, TextureHandle};
use crate::scene::{
    Camera, CpuMesh, EnvironmentSettings, HdrSettings, MaterialHandle, MeshHandle, SsaoSettings,
};

/// Counts uploads and derives mesh bounds; does no GPU work of any kind.
#[derive(Default)]
pub struct HeadlessBackend {
    mesh_bounds: Vec<Aabb>,
    materials: u32,
    textures: u32,
}

impl HeadlessBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many meshes, materials, and textures a run uploaded — enough for a
    /// benchmark to assert it measured a real scene rather than an empty one.
    pub fn upload_counts(&self) -> (usize, u32, u32) {
        (self.mesh_bounds.len(), self.materials, self.textures)
    }
}

impl RenderBackend for HeadlessBackend {
    fn load_mesh(&mut self, mesh: &CpuMesh) -> MeshHandle {
        let handle = MeshHandle(self.mesh_bounds.len() as u32);
        // Derived here for the same reason the real backend derives it at
        // upload: culling reads it out of the world, never out of a backend.
        self.mesh_bounds.push(mesh.bounds());
        handle
    }

    fn mesh_bounds(&self, mesh: MeshHandle) -> Option<Aabb> {
        self.mesh_bounds.get(mesh.0 as usize).copied()
    }

    fn load_material(&mut self, _material: &Material) -> MaterialHandle {
        let handle = MaterialHandle(self.materials);
        self.materials += 1;
        handle
    }

    fn load_texture(
        &mut self,
        _pixels: &[u8],
        _width: u32,
        _height: u32,
        _srgb: bool,
    ) -> TextureHandle {
        let handle = TextureHandle(self.textures);
        self.textures += 1;
        handle
    }

    fn load_environment(&mut self, _pixels: &[f32], _width: u32, _height: u32) {}

    fn resize(&mut self, _extent: [u32; 2]) {}

    fn render(
        &mut self,
        _items: &[RenderItem],
        _lighting: &SceneLighting,
        _camera: &Camera,
        _ssao: &SsaoSettings,
        _hdr: &HdrSettings,
        _environment: &EnvironmentSettings,
    ) {
    }
}
