use orrin_ecs::World;

use crate::gfx::thumbnail::{self, Albedo, Surface, Thumbnail};
use crate::gfx::{BlendMode, Material, RenderBackend};
use crate::scene::{CpuMesh, MaterialHandle, MeshHandle};

/// What the editor shows of each uploaded mesh and material, indexed by handle.
///
/// The world-side mirror of an upload, for the reason
/// [`MeshBounds`](crate::scene::MeshBounds) is one: after upload the CPU keeps
/// neither the vertices nor the material, so anything the editor wants to show
/// has to be taken at upload or not at all. A mesh's thumbnail is rendered
/// there, since its vertices are about to be dropped; a material keeps only
/// the handful of numbers its thumbnail needs, and the editor renders that
/// lazily, the first time it is shown.
///
/// Uploads only ever append, so a handle's slot never changes meaning and
/// nothing here is ever invalidated.
///
/// A material is kept whole rather than as its thumbnail's few numbers, which
/// is what makes it editable: the inspector edits the copy here and queues it,
/// and the app hands the queue to the renderer between frames
/// ([`apply_material_edits`]). Only the editor reads or writes this; the frame
/// never does.
#[derive(Default)]
pub struct AssetPreviews {
    meshes: Vec<Option<MeshPreview>>,
    materials: Vec<Option<MaterialPreview>>,
    edited: Vec<MaterialHandle>,
}

pub struct MeshPreview {
    pub thumbnail: Thumbnail,
    pub triangles: u32,
}

pub struct MaterialPreview {
    /// What was uploaded, with any edits made since.
    pub material: Material,
    pub surface: Surface,
    pub blend: BlendMode,
    /// Whether an albedo map was bound, as opposed to the base colour alone.
    pub textured: bool,
}

impl AssetPreviews {
    /// Render and keep `mesh`'s thumbnail. Skipped in an exported game, which
    /// has no editor to show it and should not pay the import time.
    pub fn insert_mesh(&mut self, handle: MeshHandle, mesh: &CpuMesh) {
        if cfg!(feature = "export") {
            return;
        }
        let preview = render_mesh(mesh);
        put(&mut self.meshes, handle.0, preview);
    }

    /// Keep an already-rendered preview — for an importer that renders its
    /// meshes in parallel ahead of uploading them one at a time.
    pub fn insert_rendered_mesh(&mut self, handle: MeshHandle, preview: MeshPreview) {
        put(&mut self.meshes, handle.0, preview);
    }

    pub fn insert_material(
        &mut self,
        handle: MaterialHandle,
        material: &Material,
        albedo: Option<Albedo>,
    ) {
        if cfg!(feature = "export") {
            return;
        }
        let preview = MaterialPreview {
            material: *material,
            surface: Surface {
                base_color: material.base_color,
                metallic: material.metallic,
                roughness: material.roughness,
                emissive: material.emissive,
                albedo,
            },
            blend: material.blend,
            textured: material.albedo_texture.is_some(),
        };
        put(&mut self.materials, handle.0, preview);
    }

    /// Change a material and queue it for the renderer. `false`, and nothing
    /// queued, for a material that was never loaded here.
    ///
    /// Only the parameters a [`Surface`] mirrors are kept in step with it;
    /// the blend mode is not offered for editing, since it decides which
    /// queue the material draws in and that is registered elsewhere.
    pub fn edit_material(
        &mut self,
        handle: MaterialHandle,
        edit: impl FnOnce(&mut Material),
    ) -> bool {
        let Some(Some(preview)) = self.materials.get_mut(handle.0 as usize) else {
            return false;
        };
        edit(&mut preview.material);
        let material = &preview.material;
        preview.surface.base_color = material.base_color;
        preview.surface.metallic = material.metallic;
        preview.surface.roughness = material.roughness;
        preview.surface.emissive = material.emissive;
        if !self.edited.contains(&handle) {
            self.edited.push(handle);
        }
        true
    }

    /// Every material edited since the last call, as it now stands.
    pub fn take_material_edits(&mut self) -> Vec<(MaterialHandle, Material)> {
        std::mem::take(&mut self.edited)
            .into_iter()
            .filter_map(|handle| Some((handle, self.material(handle)?.material)))
            .collect()
    }

    pub fn mesh(&self, handle: MeshHandle) -> Option<&MeshPreview> {
        self.meshes.get(handle.0 as usize)?.as_ref()
    }

    pub fn material(&self, handle: MaterialHandle) -> Option<&MaterialPreview> {
        self.materials.get(handle.0 as usize)?.as_ref()
    }
}

/// The thumbnail and triangle count of `mesh`. Free of `self` so an importer
/// can call it from the worker pool.
pub fn render_mesh(mesh: &CpuMesh) -> MeshPreview {
    let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();
    let normals: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.normal).collect();
    MeshPreview {
        thumbnail: thumbnail::mesh(&positions, &normals, &mesh.indices),
        triangles: (mesh.indices.len() / 3) as u32,
    }
}

/// Hand the inspector's material edits to the renderer. Called by the app
/// between frames; a frame with no edits costs one resource lookup.
pub fn apply_material_edits(world: &mut World, backend: &mut impl RenderBackend) {
    let Some(mut previews) = world.get_resource_mut::<AssetPreviews>() else {
        return;
    };
    for (handle, material) in previews.take_material_edits() {
        backend.update_material(handle, &material);
    }
}

fn put<T>(slots: &mut Vec<Option<T>>, index: u32, value: T) {
    let index = index as usize;
    if slots.len() <= index {
        slots.resize_with(index + 1, || None);
    }
    slots[index] = Some(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_preview_is_found_by_the_handle_it_was_stored_under() {
        let mut previews = AssetPreviews::default();
        previews.insert_mesh(MeshHandle(3), &CpuMesh::cube());
        previews.insert_material(MaterialHandle(1), &Material::default(), None);

        assert_eq!(previews.mesh(MeshHandle(3)).map(|m| m.triangles), Some(12));
        assert!(previews.mesh(MeshHandle(2)).is_none());
        assert!(previews.mesh(MeshHandle(40)).is_none());
        assert!(previews.material(MaterialHandle(1)).is_some());
        assert!(previews.material(MaterialHandle(0)).is_none());
    }

    /// An inspector edit lands on the kept copy at once, so the next frame's
    /// panel and thumbnail show it, and is queued once for the renderer.
    #[test]
    fn an_edited_material_is_kept_and_queued_once() {
        let mut previews = AssetPreviews::default();
        let handle = MaterialHandle(2);
        previews.insert_material(handle, &Material::default(), None);

        assert!(previews.edit_material(handle, |material| material.roughness = 0.2));
        let kept = previews.material(handle).unwrap();
        assert_eq!(kept.material.roughness, 0.2);
        assert_eq!(kept.surface.roughness, 0.2);

        let edits = previews.take_material_edits();
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].0, handle);
        assert_eq!(edits[0].1.roughness, 0.2);
        assert!(previews.take_material_edits().is_empty());
    }

    /// Two edits to one material in a frame are one upload, of the later one.
    #[test]
    fn edits_to_one_material_coalesce() {
        let mut previews = AssetPreviews::default();
        let handle = MaterialHandle(0);
        previews.insert_material(handle, &Material::default(), None);
        previews.edit_material(handle, |material| material.metallic = 0.5);
        previews.edit_material(handle, |material| material.metallic = 1.0);

        let edits = previews.take_material_edits();
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].1.metallic, 1.0);
    }

    #[test]
    fn a_material_never_loaded_cannot_be_edited() {
        let mut previews = AssetPreviews::default();
        assert!(!previews.edit_material(MaterialHandle(5), |material| material.metallic = 1.0));
        assert!(previews.take_material_edits().is_empty());
    }

    /// The app's half: queued edits reach the backend and the queue drains.
    #[test]
    fn queued_edits_reach_the_backend() {
        let mut world = orrin_ecs::World::new();
        let mut backend = crate::gfx::HeadlessBackend::new();
        let handle = crate::gfx::RenderBackend::load_material(&mut backend, &Material::default());
        let mut previews = AssetPreviews::default();
        previews.insert_material(handle, &Material::default(), None);
        previews.edit_material(handle, |material| material.roughness = 0.9);
        world.insert_resource(previews);

        apply_material_edits(&mut world, &mut backend);
        assert_eq!(backend.material_updates(), 1);
        apply_material_edits(&mut world, &mut backend);
        assert_eq!(backend.material_updates(), 1);
    }
}
