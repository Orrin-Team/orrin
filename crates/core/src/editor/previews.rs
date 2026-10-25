//! Thumbnails as egui textures, made the first time each is shown and kept
//! for the session.
//!
//! The pixels come from [`AssetPreviews`]: a mesh's were rendered at import, a
//! material's are rendered here from the numbers kept for it. Either way the
//! work happens once per asset, on the first frame that draws it — a picker
//! listing a thousand meshes pays for the dozen rows on screen, not the rest —
//! and every later frame is a texture id. Nothing is ever invalidated, because
//! a handle never changes meaning (see `AssetPreviews`).

use std::collections::HashMap;

use orrin_ecs::World;

use crate::gfx::BlendMode;
use crate::gfx::thumbnail::{self, Pixels, Thumbnail};
use crate::scene::{AssetPreviews, MaterialHandle, MeshHandle};

#[derive(Default)]
pub struct Previews {
    /// `None` is remembered too: an asset with nothing to show is asked about
    /// every frame its slot is on screen.
    meshes: HashMap<u32, Option<egui::TextureHandle>>,
    materials: HashMap<u32, Option<egui::TextureHandle>>,
}

/// What a material slot's subtitle and the Blend row read out.
pub struct MaterialInfo {
    pub metallic: f32,
    pub roughness: f32,
    pub blend: &'static str,
}

impl Previews {
    pub fn mesh(
        &mut self,
        ctx: &egui::Context,
        world: &World,
        handle: MeshHandle,
    ) -> Option<egui::TextureHandle> {
        self.meshes
            .entry(handle.0)
            .or_insert_with(|| {
                let previews = world.get_resource::<AssetPreviews>()?;
                let preview = previews.mesh(handle)?;
                Some(upload(
                    ctx,
                    &format!("mesh {}", handle.0),
                    &preview.thumbnail,
                ))
            })
            .clone()
    }

    pub fn material(
        &mut self,
        ctx: &egui::Context,
        world: &World,
        handle: MaterialHandle,
    ) -> Option<egui::TextureHandle> {
        self.materials
            .entry(handle.0)
            .or_insert_with(|| {
                let previews = world.get_resource::<AssetPreviews>()?;
                let preview = previews.material(handle)?;
                let rendered = thumbnail::material(&preview.surface);
                Some(upload(ctx, &format!("material {}", handle.0), &rendered))
            })
            .clone()
    }

    /// Drop a material's texture so the next ask renders it again — after an
    /// edit, the one thing that does change what a handle looks like.
    pub fn forget_material(&mut self, handle: MaterialHandle) {
        self.materials.remove(&handle.0);
    }

    pub fn mesh_summary(&self, world: &World, handle: MeshHandle) -> Option<String> {
        let previews = world.get_resource::<AssetPreviews>()?;
        let triangles = previews.mesh(handle)?.triangles;
        Some(format!("{} triangles", grouped(triangles)))
    }

    pub fn material_summary(&self, world: &World, handle: MaterialHandle) -> Option<String> {
        let info = self.material_info(world, handle)?;
        let finish = if info.metallic >= 0.5 {
            format!("metal {:.2}", info.metallic)
        } else {
            format!("rough {:.2}", info.roughness)
        };
        Some(format!("{} · {finish}", info.blend))
    }

    pub fn material_info(&self, world: &World, handle: MaterialHandle) -> Option<MaterialInfo> {
        let previews = world.get_resource::<AssetPreviews>()?;
        let preview = previews.material(handle)?;
        Some(MaterialInfo {
            metallic: preview.surface.metallic,
            roughness: preview.surface.roughness,
            blend: match preview.blend {
                BlendMode::Opaque => "Opaque",
                BlendMode::Masked => "Masked",
                BlendMode::Blend => "Blended",
                _ => "Other",
            },
        })
    }
}

fn upload(ctx: &egui::Context, name: &str, thumbnail: &Thumbnail) -> egui::TextureHandle {
    let size = [thumbnail.size as usize; 2];
    let image = match &thumbnail.pixels {
        Pixels::Rgba(rgba) => egui::ColorImage::from_rgba_unmultiplied(size, rgba),
        Pixels::GreyAlpha(la) => {
            let rgba: Vec<u8> = la
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|&[l, a]| [l, l, l, a])
                .collect();
            egui::ColorImage::from_rgba_unmultiplied(size, &rgba)
        }
    };
    ctx.load_texture(name, image, egui::TextureOptions::LINEAR)
}

/// `4812` as `4 812`: a triangle count is read at a glance, and a space is the
/// grouping that cannot be mistaken for a decimal point in any locale.
fn grouped(value: u32) -> String {
    let digits = value.to_string();
    let mut out = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(digit);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::Material;
    use crate::scene::CpuMesh;

    fn world() -> World {
        let mut world = World::new();
        let mut previews = AssetPreviews::default();
        previews.insert_mesh(MeshHandle(0), &CpuMesh::cube());
        previews.insert_material(MaterialHandle(0), &Material::default(), None);
        world.insert_resource(previews);
        world
    }

    /// A texture is made once per asset, and every later ask returns that
    /// same one — the cache, not a new upload each frame.
    #[test]
    fn a_thumbnail_is_uploaded_once_and_reused() {
        let world = world();
        let ctx = egui::Context::default();
        let mut previews = Previews::default();

        let first = previews
            .mesh(&ctx, &world, MeshHandle(0))
            .expect("a texture");
        let again = previews
            .mesh(&ctx, &world, MeshHandle(0))
            .expect("a texture");
        assert_eq!(first.id(), again.id());
        assert_eq!(first.size(), [thumbnail::SIZE as usize; 2]);

        let material = previews.material(&ctx, &world, MaterialHandle(0));
        assert!(material.is_some());
    }

    #[test]
    fn an_asset_with_no_preview_has_no_texture() {
        let world = world();
        let ctx = egui::Context::default();
        let mut previews = Previews::default();
        assert!(previews.mesh(&ctx, &world, MeshHandle(9)).is_none());
        assert!(previews.material(&ctx, &world, MaterialHandle(9)).is_none());
        assert!(previews.mesh(&ctx, &World::new(), MeshHandle(0)).is_none());
    }

    #[test]
    fn summaries_read_in_display_terms() {
        let world = world();
        let previews = Previews::default();
        assert_eq!(
            previews.mesh_summary(&world, MeshHandle(0)).as_deref(),
            Some("12 triangles")
        );
        assert!(
            previews
                .material_summary(&world, MaterialHandle(0))
                .unwrap()
                .starts_with("Opaque · ")
        );
        assert_eq!(grouped(4812), "4 812");
        assert_eq!(grouped(1_234_567), "1 234 567");
        assert_eq!(grouped(999), "999");
    }
}
