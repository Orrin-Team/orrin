//! Thumbnails as egui textures, made the first time each is shown and kept
//! for the session.
//!
//! The pixels come from [`AssetPreviews`]: a mesh's were rendered at import, a
//! material's are rendered here from the numbers kept for it. Either way the
//! work happens once per asset, on the first frame that draws it — a picker
//! listing a thousand meshes pays for the dozen rows on screen, not the rest —
//! and every later frame is a texture id. A mesh's is never invalidated,
//! because a handle never changes meaning (see `AssetPreviews`); a material's
//! is kept with the surface it was rendered from and rendered again when that
//! changes, since a material asset is edited, saved and undone in place.

use std::collections::HashMap;

use orrin_ecs::World;
use orrin_registry::AssetId;

use crate::asset_db::{AssetDatabase, AssetKind, CacheKey};
use crate::gfx::BlendMode;
use crate::gfx::thumbnail::{self, Albedo, Pixels, Surface, Thumbnail};
use crate::scene::{AssetPreviews, MaterialHandle, MeshHandle};

#[derive(Default)]
pub struct Previews {
    /// `None` is remembered too: an asset with nothing to show is asked about
    /// every frame its slot is on screen.
    meshes: HashMap<u32, Option<egui::TextureHandle>>,
    /// With the surface each was rendered from.
    materials: HashMap<u32, (Option<Surface>, Option<egui::TextureHandle>)>,
    /// The asset browser's pictures, by asset, with what each was made from.
    assets: HashMap<AssetId, (Picture, egui::TextureHandle)>,
}

/// What an asset's picture was made from, which changes when it should be
/// made again: an imported asset's artifact, a material's surface.
#[derive(Clone, Debug, PartialEq)]
enum Picture {
    Imported(CacheKey),
    Material(Surface),
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

    /// Rendered again when the kept copy is no longer what was rendered: a
    /// material asset changes under its handle when its file is saved, edited
    /// or an edit is undone.
    pub fn material(
        &mut self,
        ctx: &egui::Context,
        world: &World,
        handle: MaterialHandle,
    ) -> Option<egui::TextureHandle> {
        let previews = world.get_resource::<AssetPreviews>();
        let surface = previews
            .as_ref()
            .and_then(|p| Some(&p.material(handle)?.surface));
        if let Some((drawn, texture)) = self.materials.get(&handle.0)
            && drawn.as_ref() == surface
        {
            return texture.clone();
        }
        let texture = surface.map(|surface| {
            let rendered = thumbnail::material(surface);
            upload(ctx, &format!("material {}", handle.0), &rendered)
        });
        self.materials
            .insert(handle.0, (surface.cloned(), texture.clone()));
        texture
    }

    /// The asset browser's picture of asset `id`: a texture's or a mesh's
    /// thumbnail from its import, a material rendered on a sphere from its
    /// file, with its albedo texture's thumbnail on it. Made again when the
    /// asset is reimported or the material changes; `None` for a prefab, a
    /// folder, or an asset not imported yet.
    pub fn asset(
        &mut self,
        ctx: &egui::Context,
        db: &AssetDatabase,
        id: AssetId,
    ) -> Option<egui::TextureHandle> {
        let picture = match db.get(id)?.kind {
            AssetKind::Texture | AssetKind::Mesh => Picture::Imported(db.thumbnail(id)?.0),
            AssetKind::Material => Picture::Material(material_surface(db, id)?),
            AssetKind::Prefab | AssetKind::Folder => return None,
        };
        if let Some((drawn, texture)) = self.assets.get(&id)
            && *drawn == picture
        {
            return Some(texture.clone());
        }
        let name = format!("asset {id}");
        let texture = match &picture {
            Picture::Imported(_) => upload(ctx, &name, db.thumbnail(id)?.1),
            Picture::Material(surface) => upload(ctx, &name, &thumbnail::material(surface)),
        };
        self.assets.insert(id, (picture, texture.clone()));
        Some(texture)
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

/// What material `id`'s sphere shows: its file's colour and finish, and its
/// albedo texture's thumbnail reduced to the few texels a sphere can show.
fn material_surface(db: &AssetDatabase, id: AssetId) -> Option<Surface> {
    let file = db.material_file(id)?;
    let material = &file.material;
    let albedo = file
        .textures
        .albedo
        .and_then(|texture| db.thumbnail(texture))
        .and_then(|(_, picture)| {
            let rgba = rgba(picture);
            Albedo::reduce(&rgba, picture.size, picture.size)
        });
    Some(Surface {
        base_color: material.base_color,
        metallic: material.metallic,
        roughness: material.roughness,
        emissive: material.emissive,
        albedo,
    })
}

fn rgba(thumbnail: &Thumbnail) -> Vec<u8> {
    match &thumbnail.pixels {
        Pixels::Rgba(rgba) => rgba.clone(),
        Pixels::GreyAlpha(la) => la
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|&[l, a]| [l, l, l, a])
            .collect(),
    }
}

fn upload(ctx: &egui::Context, name: &str, thumbnail: &Thumbnail) -> egui::TextureHandle {
    let size = [thumbnail.size as usize; 2];
    let image = egui::ColorImage::from_rgba_unmultiplied(size, &rgba(thumbnail));
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

    /// A material asset's kept copy changes under the same handle when its
    /// file is saved, edited or an edit undone; its thumbnail follows.
    #[test]
    fn a_material_whose_kept_copy_changed_is_rendered_again() {
        let world = world();
        let ctx = egui::Context::default();
        let mut previews = Previews::default();
        let first = previews.material(&ctx, &world, MaterialHandle(0)).unwrap();

        let red = Material {
            base_color: glam::Vec3::X,
            ..Material::default()
        };
        world
            .resource_mut::<AssetPreviews>()
            .insert_material(MaterialHandle(0), &red, None);
        let after = previews.material(&ctx, &world, MaterialHandle(0)).unwrap();
        assert_ne!(first.id(), after.id());
        let again = previews.material(&ctx, &world, MaterialHandle(0)).unwrap();
        assert_eq!(after.id(), again.id(), "and kept again");
    }

    /// A project with a red texture and a material using it as its albedo.
    fn project(dir: &crate::asset_db::tests::Dir) -> AssetDatabase {
        let mut png = std::io::Cursor::new(Vec::new());
        image::RgbaImage::from_pixel(8, 8, image::Rgba([220, 20, 20, 255]))
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        dir.write("rock.png", png.get_ref());
        let rock = "5f0c0001-0000-4000-8000-000000000001";
        dir.write(
            "rock.png.meta",
            format!("id = \"{rock}\"\n\n[import]\nusage = \"color\"\n").as_bytes(),
        );
        dir.write(
            "rock.fmat",
            format!("orrin-material 1 min-reader 1\n\ntextures\n  albedo = asset({rock})\n")
                .as_bytes(),
        );
        dir.write("door.fprefab", b"orrin-scene 1\n");
        let (mut db, issues) = AssetDatabase::open(dir.path(), dir.path().join(".cache"));
        assert!(issues.is_empty(), "{issues:?}");
        assert!(db.wait_idle(std::time::Duration::from_secs(30)));
        db.update(&World::new(), &mut crate::gfx::HeadlessBackend::new());
        db
    }

    /// An asset's picture is uploaded once and kept; a material's is made
    /// again when the material changes, and carries its albedo's colours.
    #[test]
    fn an_assets_picture_is_kept_until_what_it_shows_changes() {
        let dir = crate::asset_db::tests::Dir::new("previews-assets");
        let mut db = project(&dir);
        let ctx = egui::Context::default();
        let mut previews = Previews::default();
        let rock = db.find("rock.png").unwrap().id;
        let material = db.find("rock.fmat").unwrap().id;
        let door = db.find("door.fprefab").unwrap().id;

        let first = previews
            .asset(&ctx, &db, rock)
            .expect("a texture's picture");
        let again = previews.asset(&ctx, &db, rock).unwrap();
        assert_eq!(first.id(), again.id());

        let before = previews.asset(&ctx, &db, material).expect("a material's");
        let Some(Picture::Material(surface)) = previews.assets.get(&material).map(|p| p.0.clone())
        else {
            panic!("a material is pictured from its surface");
        };
        assert!(surface.albedo.is_some(), "with its albedo on it");

        let mut file = *db.material_file(material).unwrap();
        file.material.roughness = 0.1;
        db.preview_material(material, file).unwrap();
        let after = previews.asset(&ctx, &db, material).unwrap();
        assert_ne!(before.id(), after.id(), "made again");

        assert!(
            previews.asset(&ctx, &db, door).is_none(),
            "a prefab has none"
        );
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
