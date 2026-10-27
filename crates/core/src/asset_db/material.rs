//! `.fmat`: a material as a file.
//!
//! The scene grammar under a header of its own (`orrin-material 1 min-reader
//! 1`), so a material reads, diffs and merges the way a scene does. Every
//! parameter is written, sorted by name, so a file says what the material is
//! without anyone needing the defaults to hand; a texture slot is written only
//! when it is filled, by the [`AssetId`] of the texture in it.
//!
//! **Read leniently, but not loosely.** A parameter the file does not have takes
//! the default — that is what a file written before the parameter existed means
//! — but one the engine does not know is refused by name, as a sidecar's
//! settings are: a typo that silently fell back would be a material drawn wrong
//! with nothing to say why.
//!
//! The mapping is written out field by field rather than derived: a `Material`
//! names its textures by [`TextureHandle`], an upload slot that means nothing
//! in another session, so the file needs a shape of its own. Both directions
//! destructure `Material` without a `..`, so a field added to it does not
//! compile until the file has a place for it.

use glam::Vec3;
use orrin_registry::{AssetId, Reflect, Value, ValueError};

use crate::gfx::{BlendMode, Material, TextureHandle};

/// The word a `.fmat` header starts with.
pub const KIND: &str = "orrin-material";
/// The version this build writes and the newest it reads.
pub const VERSION: u32 = 1;
pub const EXTENSION: &str = "fmat";

/// A material's texture slots, each naming the texture asset in it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MaterialTextures {
    pub albedo: Option<AssetId>,
    pub normal: Option<AssetId>,
    pub metallic_roughness: Option<AssetId>,
    pub emissive: Option<AssetId>,
    pub clearcoat: Option<AssetId>,
    pub clearcoat_normal: Option<AssetId>,
    pub sheen: Option<AssetId>,
    pub anisotropy: Option<AssetId>,
    pub transmission: Option<AssetId>,
    pub subsurface: Option<AssetId>,
    pub height: Option<AssetId>,
}

/// A material as its file holds it: every parameter, and each texture by id.
/// `material`'s own texture handles are never set here; [`resolve`] fills them.
///
/// [`resolve`]: Self::resolve
#[derive(Clone, Copy, Debug, Default)]
pub struct MaterialFile {
    pub material: Material,
    pub textures: MaterialTextures,
}

impl MaterialTextures {
    /// Every slot by the name the file uses, with what is in it.
    pub fn slots(&self) -> [(&'static str, Option<AssetId>); 11] {
        let mut file = MaterialFile {
            textures: *self,
            ..MaterialFile::default()
        };
        slots(&mut file).map(|(slot, id, _)| (slot, *id))
    }

    /// Whether `slot` holds a normal map, which is loaded flat rather than
    /// white while it imports.
    pub fn is_normal_slot(slot: &str) -> bool {
        matches!(slot, "normal" | "clearcoat_normal")
    }
}

impl MaterialFile {
    /// The texture slot the file calls `slot`, to fill or empty.
    pub fn texture_mut(&mut self, slot: &str) -> Option<&mut Option<AssetId>> {
        slots(self)
            .into_iter()
            .find(|(name, ..)| *name == slot)
            .map(|(_, texture, _)| texture)
    }

    /// The material to upload, each filled slot given the handle `texture`
    /// returns for it.
    pub fn resolve(
        &self,
        mut texture: impl FnMut(&'static str, AssetId) -> TextureHandle,
    ) -> Material {
        let mut file = *self;
        for (slot, id, handle) in slots(&mut file) {
            *handle = id.map(|id| texture(slot, id));
        }
        file.material
    }
}

/// Each slot's name, the texture the file puts in it, and the field of the
/// material that samples it: the one place the three are paired.
fn slots(
    file: &mut MaterialFile,
) -> [(
    &'static str,
    &mut Option<AssetId>,
    &mut Option<TextureHandle>,
); 11] {
    let MaterialTextures {
        albedo,
        normal,
        metallic_roughness,
        emissive,
        clearcoat,
        clearcoat_normal,
        sheen,
        anisotropy,
        transmission,
        subsurface,
        height,
    } = &mut file.textures;
    let Material {
        albedo_texture,
        normal_texture,
        metallic_roughness_texture,
        emissive_texture,
        clearcoat_texture,
        clearcoat_normal_texture,
        sheen_texture,
        anisotropy_texture,
        transmission_texture,
        subsurface_texture,
        height_texture,
        base_color: _,
        alpha: _,
        alpha_cutoff: _,
        blend: _,
        metallic: _,
        roughness: _,
        reflectance: _,
        emissive: _,
        clearcoat: _,
        clearcoat_roughness: _,
        sheen_color: _,
        sheen_roughness: _,
        anisotropy: _,
        anisotropy_rotation: _,
        transmission: _,
        ior: _,
        thickness: _,
        attenuation_color: _,
        attenuation_distance: _,
        subsurface_color: _,
        subsurface_radius: _,
        subsurface_forward_scatter: _,
        parallax_depth: _,
        parallax_min_steps: _,
        parallax_max_steps: _,
    } = &mut file.material;
    [
        ("albedo", albedo, albedo_texture),
        ("normal", normal, normal_texture),
        (
            "metallic_roughness",
            metallic_roughness,
            metallic_roughness_texture,
        ),
        ("emissive", emissive, emissive_texture),
        ("clearcoat", clearcoat, clearcoat_texture),
        (
            "clearcoat_normal",
            clearcoat_normal,
            clearcoat_normal_texture,
        ),
        ("sheen", sheen, sheen_texture),
        ("anisotropy", anisotropy, anisotropy_texture),
        ("transmission", transmission, transmission_texture),
        ("subsurface", subsurface, subsurface_texture),
        ("height", height, height_texture),
    ]
}

/// Read a `.fmat`. `rel` names the file in every error.
pub fn parse(text: &str, rel: &str) -> Result<MaterialFile, String> {
    let fields = orrin_registry::parse_value_document(text, KIND, VERSION)
        .map_err(|error| format!("{rel}: {error}"))?;
    let mut file = MaterialFile::default();
    let mut textures = None;
    let Material {
        base_color,
        alpha,
        alpha_cutoff,
        blend,
        metallic,
        roughness,
        reflectance,
        emissive,
        clearcoat,
        clearcoat_roughness,
        sheen_color,
        sheen_roughness,
        anisotropy,
        anisotropy_rotation,
        transmission,
        ior,
        thickness,
        attenuation_color,
        attenuation_distance,
        subsurface_color,
        subsurface_radius,
        subsurface_forward_scatter,
        parallax_depth,
        parallax_min_steps,
        parallax_max_steps,
        albedo_texture: _,
        normal_texture: _,
        metallic_roughness_texture: _,
        emissive_texture: _,
        clearcoat_texture: _,
        clearcoat_normal_texture: _,
        sheen_texture: _,
        anisotropy_texture: _,
        transmission_texture: _,
        subsurface_texture: _,
        height_texture: _,
    } = &mut file.material;
    for (name, value) in &fields {
        let field = |error: ValueError| format!("{rel}: {}", error.at_field(name));
        match name.as_str() {
            "base_color" => *base_color = Vec3::from_value(value).map_err(field)?,
            "alpha" => *alpha = f32::from_value(value).map_err(field)?,
            "alpha_cutoff" => *alpha_cutoff = f32::from_value(value).map_err(field)?,
            "blend" => *blend = parse_blend(value, rel)?,
            "metallic" => *metallic = f32::from_value(value).map_err(field)?,
            "roughness" => *roughness = f32::from_value(value).map_err(field)?,
            "reflectance" => *reflectance = f32::from_value(value).map_err(field)?,
            "emissive" => *emissive = Vec3::from_value(value).map_err(field)?,
            "clearcoat" => *clearcoat = f32::from_value(value).map_err(field)?,
            "clearcoat_roughness" => {
                *clearcoat_roughness = f32::from_value(value).map_err(field)?
            }
            "sheen_color" => *sheen_color = Vec3::from_value(value).map_err(field)?,
            "sheen_roughness" => *sheen_roughness = f32::from_value(value).map_err(field)?,
            "anisotropy" => *anisotropy = f32::from_value(value).map_err(field)?,
            "anisotropy_rotation" => {
                *anisotropy_rotation = f32::from_value(value).map_err(field)?
            }
            "transmission" => *transmission = f32::from_value(value).map_err(field)?,
            "ior" => *ior = f32::from_value(value).map_err(field)?,
            "thickness" => *thickness = f32::from_value(value).map_err(field)?,
            "attenuation_color" => *attenuation_color = Vec3::from_value(value).map_err(field)?,
            "attenuation_distance" => {
                *attenuation_distance = f32::from_value(value).map_err(field)?
            }
            "subsurface_color" => *subsurface_color = Vec3::from_value(value).map_err(field)?,
            "subsurface_radius" => *subsurface_radius = Vec3::from_value(value).map_err(field)?,
            "subsurface_forward_scatter" => {
                *subsurface_forward_scatter = f32::from_value(value).map_err(field)?
            }
            "parallax_depth" => *parallax_depth = f32::from_value(value).map_err(field)?,
            "parallax_min_steps" => *parallax_min_steps = u32::from_value(value).map_err(field)?,
            "parallax_max_steps" => *parallax_max_steps = u32::from_value(value).map_err(field)?,
            "textures" => textures = Some(value),
            other => return Err(format!("{rel}: `{other}` is not a material parameter")),
        }
    }
    if let Some(value) = textures {
        parse_textures(&mut file, value, rel)?;
    }
    Ok(file)
}

fn parse_textures(file: &mut MaterialFile, value: &Value, rel: &str) -> Result<(), String> {
    let Value::Struct(fields) = value else {
        let error = ValueError::mismatch("struct", value).at_field("textures");
        return Err(format!("{rel}: {error}"));
    };
    for (name, value) in fields {
        let Some((_, slot, _)) = slots(file).into_iter().find(|(slot, ..)| slot == name) else {
            return Err(format!("{rel}: `{name}` is not a texture slot"));
        };
        let id = AssetId::from_value(value)
            .map_err(|error| format!("{rel}: {}", error.at_field(name).at_field("textures")))?;
        *slot = Some(id);
    }
    Ok(())
}

/// The text of a `.fmat` holding `file`.
pub fn write(file: &MaterialFile) -> String {
    let Material {
        base_color,
        alpha,
        alpha_cutoff,
        blend,
        metallic,
        roughness,
        reflectance,
        emissive,
        clearcoat,
        clearcoat_roughness,
        sheen_color,
        sheen_roughness,
        anisotropy,
        anisotropy_rotation,
        transmission,
        ior,
        thickness,
        attenuation_color,
        attenuation_distance,
        subsurface_color,
        subsurface_radius,
        subsurface_forward_scatter,
        parallax_depth,
        parallax_min_steps,
        parallax_max_steps,
        albedo_texture: _,
        normal_texture: _,
        metallic_roughness_texture: _,
        emissive_texture: _,
        clearcoat_texture: _,
        clearcoat_normal_texture: _,
        sheen_texture: _,
        anisotropy_texture: _,
        transmission_texture: _,
        subsurface_texture: _,
        height_texture: _,
    } = file.material;
    let field = |name: &str, value: Value| (name.to_owned(), value);
    let mut fields = vec![
        field("base_color", base_color.to_value()),
        field("alpha", alpha.to_value()),
        field("alpha_cutoff", alpha_cutoff.to_value()),
        field("blend", blend_value(blend)),
        field("metallic", metallic.to_value()),
        field("roughness", roughness.to_value()),
        field("reflectance", reflectance.to_value()),
        field("emissive", emissive.to_value()),
        field("clearcoat", clearcoat.to_value()),
        field("clearcoat_roughness", clearcoat_roughness.to_value()),
        field("sheen_color", sheen_color.to_value()),
        field("sheen_roughness", sheen_roughness.to_value()),
        field("anisotropy", anisotropy.to_value()),
        field("anisotropy_rotation", anisotropy_rotation.to_value()),
        field("transmission", transmission.to_value()),
        field("ior", ior.to_value()),
        field("thickness", thickness.to_value()),
        field("attenuation_color", attenuation_color.to_value()),
        field("attenuation_distance", attenuation_distance.to_value()),
        field("subsurface_color", subsurface_color.to_value()),
        field("subsurface_radius", subsurface_radius.to_value()),
        field(
            "subsurface_forward_scatter",
            subsurface_forward_scatter.to_value(),
        ),
        field("parallax_depth", parallax_depth.to_value()),
        field("parallax_min_steps", parallax_min_steps.to_value()),
        field("parallax_max_steps", parallax_max_steps.to_value()),
    ];
    let textures: Vec<(String, Value)> = file
        .textures
        .slots()
        .into_iter()
        .filter_map(|(slot, id)| Some(field(slot, Value::Asset(id?))))
        .collect();
    if !textures.is_empty() {
        fields.push(field("textures", Value::Struct(textures)));
    }
    let mut out = String::new();
    orrin_registry::write_value_document(&mut out, KIND, VERSION, VERSION, &fields);
    out
}

const BLENDS: [BlendMode; 4] = [
    BlendMode::Opaque,
    BlendMode::Masked,
    BlendMode::Blend,
    BlendMode::Transmissive,
];

fn blend_name(blend: BlendMode) -> &'static str {
    match blend {
        BlendMode::Opaque => "Opaque",
        BlendMode::Masked => "Masked",
        BlendMode::Blend => "Blend",
        BlendMode::Transmissive => "Transmissive",
    }
}

fn blend_value(blend: BlendMode) -> Value {
    Value::Enum {
        variant: blend_name(blend).to_owned(),
        fields: Vec::new(),
    }
}

fn parse_blend(value: &Value, rel: &str) -> Result<BlendMode, String> {
    let Value::Enum { variant, fields } = value else {
        let error = ValueError::mismatch("enum", value).at_field("blend");
        return Err(format!("{rel}: {error}"));
    };
    BLENDS
        .into_iter()
        .find(|blend| fields.is_empty() && blend_name(*blend) == variant)
        .ok_or_else(|| {
            format!("{rel}: `blend` must be Opaque, Masked, Blend or Transmissive, not {variant}")
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u8) -> AssetId {
        AssetId::from_bytes([n, 0, 0, 0, 0, 0, 0x40, 0, 0x80, 0, 0, 0, 0, 0, 0, n])
    }

    /// Every parameter away from its default and every slot filled, so a field
    /// the writer forgets or the reader does not know fails here.
    fn everything() -> MaterialFile {
        let material = Material {
            base_color: Vec3::new(0.1, 0.2, 0.3),
            alpha: 0.75,
            alpha_cutoff: 0.25,
            blend: BlendMode::Masked,
            metallic: 1.0,
            roughness: 0.125,
            reflectance: 0.375,
            emissive: Vec3::new(10.0, 20.0, 30.0),
            clearcoat: 0.5,
            clearcoat_roughness: 0.0625,
            sheen_color: Vec3::new(0.4, 0.5, 0.6),
            sheen_roughness: 0.875,
            anisotropy: -0.5,
            anisotropy_rotation: 1.5,
            transmission: 0.25,
            ior: 1.33,
            thickness: 0.01,
            attenuation_color: Vec3::new(0.9, 0.8, 0.7),
            attenuation_distance: 2.5,
            subsurface_color: Vec3::new(0.7, 0.3, 0.2),
            subsurface_radius: Vec3::new(0.001, 0.002, 0.003),
            subsurface_forward_scatter: 4.0,
            parallax_depth: 0.05,
            parallax_min_steps: 4,
            parallax_max_steps: 48,
            ..Material::default()
        };
        let textures = MaterialTextures {
            albedo: Some(id(1)),
            normal: Some(id(2)),
            metallic_roughness: Some(id(3)),
            emissive: Some(id(4)),
            clearcoat: Some(id(5)),
            clearcoat_normal: Some(id(6)),
            sheen: Some(id(7)),
            anisotropy: Some(id(8)),
            transmission: Some(id(9)),
            subsurface: Some(id(10)),
            height: Some(id(11)),
        };
        MaterialFile { material, textures }
    }

    #[test]
    fn every_parameter_and_slot_survives_a_write_and_a_read() {
        let file = everything();
        let text = write(&file);
        let back = parse(&text, "brick.fmat").unwrap();
        assert_eq!(format!("{back:?}"), format!("{file:?}"));
        assert_eq!(write(&back), text);
    }

    /// The layout pinned: a default material is every parameter, sorted, and
    /// no texture block.
    #[test]
    fn a_default_material_writes_every_parameter_and_no_textures() {
        assert_eq!(
            write(&MaterialFile::default()),
            "\
orrin-material 1 min-reader 1

alpha = 1.0
alpha_cutoff = 0.5
anisotropy = 0.0
anisotropy_rotation = 0.0
attenuation_color = (1.0, 1.0, 1.0)
attenuation_distance = inf
base_color = (0.8, 0.8, 0.8)
blend = Opaque
clearcoat = 0.0
clearcoat_roughness = 0.03
emissive = (0.0, 0.0, 0.0)
ior = 1.5
metallic = 0.0
parallax_depth = 0.02
parallax_max_steps = 32
parallax_min_steps = 8
reflectance = 0.5
roughness = 0.5
sheen_color = (0.0, 0.0, 0.0)
sheen_roughness = 0.3
subsurface_color = (0.0, 0.0, 0.0)
subsurface_forward_scatter = 12.0
subsurface_radius = (0.0048, 0.0017, 0.0011)
thickness = 0.0
transmission = 0.0
"
        );
    }

    #[test]
    fn a_parameter_the_file_lacks_takes_its_default() {
        let file = parse(
            "orrin-material 1 min-reader 1\n\nroughness = 0.2\n",
            "brick.fmat",
        )
        .unwrap();
        assert_eq!(file.material.roughness, 0.2);
        assert_eq!(file.material.base_color, Material::default().base_color);
        assert_eq!(file.textures, MaterialTextures::default());
    }

    #[test]
    fn an_unknown_or_mistyped_field_is_refused_by_name() {
        let err = parse(
            "orrin-material 1 min-reader 1\n\nrougness = 0.2\n",
            "brick.fmat",
        )
        .unwrap_err();
        assert_eq!(err, "brick.fmat: `rougness` is not a material parameter");

        let err = parse(
            "orrin-material 1 min-reader 1\n\nroughness = \"low\"\n",
            "brick.fmat",
        )
        .unwrap_err();
        assert_eq!(
            err,
            "brick.fmat: field `roughness`: expected f32, found string"
        );

        let err = parse(
            "orrin-material 1 min-reader 1\n\ntextures\n  albeedo = asset(00000000-0000-4000-8000-000000000001)\n",
            "brick.fmat",
        )
        .unwrap_err();
        assert_eq!(err, "brick.fmat: `albeedo` is not a texture slot");

        let err = parse(
            "orrin-material 1 min-reader 1\n\nblend = Glass\n",
            "brick.fmat",
        )
        .unwrap_err();
        assert_eq!(
            err,
            "brick.fmat: `blend` must be Opaque, Masked, Blend or Transmissive, not Glass"
        );

        let err = parse("orrin-scene 3 min-reader 3\n", "brick.fmat").unwrap_err();
        assert_eq!(
            err,
            "brick.fmat: line 1: expected an `orrin-material 1 min-reader 1` header"
        );
    }

    #[test]
    fn resolving_fills_each_slot_with_its_texture_and_leaves_the_rest_empty() {
        let file = MaterialFile {
            textures: MaterialTextures {
                albedo: Some(id(1)),
                normal: Some(id(2)),
                ..MaterialTextures::default()
            },
            ..MaterialFile::default()
        };
        let mut asked = Vec::new();
        let material = file.resolve(|slot, id| {
            asked.push((slot, id));
            TextureHandle(40 + asked.len() as u32)
        });
        assert_eq!(asked, [("albedo", id(1)), ("normal", id(2))]);
        assert_eq!(material.albedo_texture, Some(TextureHandle(41)));
        assert_eq!(material.normal_texture, Some(TextureHandle(42)));
        assert_eq!(material.emissive_texture, None);
        assert_eq!(material.height_texture, None);
    }

    #[test]
    fn only_the_two_normal_map_slots_load_flat() {
        let normal: Vec<&str> = MaterialTextures::default()
            .slots()
            .iter()
            .map(|(slot, _)| *slot)
            .filter(|slot| MaterialTextures::is_normal_slot(slot))
            .collect();
        assert_eq!(normal, ["normal", "clearcoat_normal"]);
    }
}
