//! `.omat`: a material as a file.
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
//! **A surface shader** is named by asset, with the values set for its
//! parameters in a block of their own:
//!
//! ```text
//! shader = asset(…)
//! shader_parameters
//!   mask = asset(…)
//!   tint = (1.0, 0.45, 0.2, 1.0)
//!   wear = 0.25
//! ```
//!
//! A value is a number, a tuple of two to four, or a texture by asset, and its
//! shape is checked. Its *name* is not, which is the one place the rule above
//! is turned round: names are whatever the shader declares, the file cannot
//! know them, and a parameter renamed while the shader is worked on must not
//! stop the material loading. Such a value is kept and written back, and the
//! database reports it as ignored when the material is drawn. Only values that
//! were set are written — a parameter at its default is absent, so editing the
//! default in the shader reaches every material that did not override it. That
//! is the rule a C# hot reload follows: preserve deviations, not values.
//!
//! **Versions.** A file is written at the oldest format that holds it: format 1
//! without a shader, byte for byte what it was before shaders, and `2
//! min-reader 2` with one, so an engine older than shaders refuses the file as
//! needing a newer reader, rather than calling `shader` a typo. For the same
//! reason a file that names a shader under a `min-reader 1` header is refused
//! here.
//!
//! The mapping is written out field by field rather than derived: a `Material`
//! names its textures by [`TextureHandle`], an upload slot that means nothing
//! in another session, so the file needs a shape of its own. Both directions
//! destructure `Material` without a `..`, so a field added to it does not
//! compile until the file has a place for it.

use glam::Vec3;
use orrin_registry::{AssetId, Reflect, Value, ValueError};

use crate::gfx::{BlendMode, Material, TextureHandle};

/// The word a `.omat` header starts with.
pub const KIND: &str = "orrin-material";
/// The newest version this build reads, and the one it writes a file that
/// uses a surface shader at. A file that does not is still written at 1.
pub const VERSION: u32 = 2;
/// The version a file with no shader is written at, which every reader since
/// the first can read.
const PLAIN: u32 = 1;
pub const EXTENSION: &str = "omat";

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

/// A material as its file holds it: every parameter, each texture by id, and
/// the surface shader it is drawn through with the values set for it.
/// `material`'s own texture handles are never set here; [`resolve`] fills them.
///
/// [`resolve`]: Self::resolve
#[derive(Clone, Debug, Default)]
pub struct MaterialFile {
    pub material: Material,
    pub textures: MaterialTextures,
    /// The surface shader asset the material is drawn through.
    pub shader: Option<AssetId>,
    /// The values set for the shader's parameters, by name and sorted by it.
    /// Only what was set: a parameter absent here takes the shader's default.
    pub shader_values: Vec<(String, ShaderValue)>,
}

/// A value for one of a surface shader's parameters as a file holds it: a
/// texture by asset, as a texture slot holds one. A `color` is a `Vec4`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShaderValue {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Texture(AssetId),
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
        let mut file = MaterialFile {
            material: self.material,
            textures: self.textures,
            ..MaterialFile::default()
        };
        for (slot, id, handle) in slots(&mut file) {
            *handle = id.map(|id| texture(slot, id));
        }
        file.material
    }

    /// The value set for shader parameter `name`, if one was.
    pub fn shader_value(&self, name: &str) -> Option<ShaderValue> {
        self.shader_values
            .iter()
            .find(|(set, _)| set == name)
            .map(|(_, value)| *value)
    }

    /// Set shader parameter `name` to `value`, or back to the shader's default
    /// with `None`, keeping the values sorted by name.
    pub fn set_shader_value(&mut self, name: &str, value: Option<ShaderValue>) {
        let at = self
            .shader_values
            .binary_search_by(|(set, _)| set.as_str().cmp(name));
        match (at, value) {
            (Ok(at), Some(value)) => self.shader_values[at].1 = value,
            (Ok(at), None) => {
                self.shader_values.remove(at);
            }
            (Err(at), Some(value)) => self.shader_values.insert(at, (name.to_owned(), value)),
            (Err(_), None) => {}
        }
    }

    /// Whether the file needs format 2 to say what it holds.
    fn uses_shader(&self) -> bool {
        self.shader.is_some() || !self.shader_values.is_empty()
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

/// Read a `.omat`. `rel` names the file in every error.
pub fn parse(text: &str, rel: &str) -> Result<MaterialFile, String> {
    let fields = orrin_registry::parse_value_document(text, KIND, VERSION)
        .map_err(|error| format!("{rel}: {error}"))?;
    let mut file = MaterialFile::default();
    let mut textures = None;
    let mut shader_values = None;
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
            "shader" => file.shader = Some(AssetId::from_value(value).map_err(field)?),
            "shader_parameters" => shader_values = Some(value),
            other => return Err(format!("{rel}: `{other}` is not a material parameter")),
        }
    }
    if let Some(value) = textures {
        parse_textures(&mut file, value, rel)?;
    }
    if let Some(value) = shader_values {
        file.shader_values = parse_shader_values(value, rel)?;
    }
    if file.uses_shader() && min_reader(text) < VERSION {
        let field = match file.shader {
            Some(_) => "shader",
            None => "shader_parameters",
        };
        return Err(format!(
            "{rel}: `{field}` needs material format 2: write the header as \
             `{KIND} {VERSION} min-reader {VERSION}`, so an engine older than surface shaders \
             refuses the file instead of drawing it without its shader"
        ));
    }
    Ok(file)
}

/// The `min-reader` of a header `parse_value_document` has already accepted:
/// the fourth word of the first line that is not blank.
fn min_reader(text: &str) -> u32 {
    text.lines()
        .find(|line| !line.trim().is_empty())
        .and_then(|header| header.split_whitespace().nth(3)?.parse().ok())
        .unwrap_or(PLAIN)
}

/// The shader parameter block, named however the shader names them.
///
/// Unlike every other field, a name here is not checked: the file cannot know
/// what its shader declares, and a parameter renamed or removed while the
/// shader is worked on must not stop the material loading. Such a value is
/// kept, written back, and reported as ignored when the material is drawn.
/// Its *shape* is still checked, because no parameter could take a value of
/// any other.
fn parse_shader_values(value: &Value, rel: &str) -> Result<Vec<(String, ShaderValue)>, String> {
    let Value::Struct(fields) = value else {
        let error = ValueError::mismatch("struct", value).at_field("shader_parameters");
        return Err(format!("{rel}: {error}"));
    };
    let mut values: Vec<(String, ShaderValue)> = Vec::with_capacity(fields.len());
    for (name, value) in fields {
        let value = match *value {
            Value::F32(v) => ShaderValue::Float(v),
            Value::I32(v) => ShaderValue::Float(v as f32),
            Value::U32(v) => ShaderValue::Float(v as f32),
            Value::Vec2(v) => ShaderValue::Vec2(v.to_array()),
            Value::Vec3(v) => ShaderValue::Vec3(v.to_array()),
            // Four bare numbers read as a quaternion, and `rgba(…)` as a
            // colour: either is a `vec4` or `color` value here.
            Value::Quat(v) => ShaderValue::Vec4(v.to_array()),
            Value::Color(v) => ShaderValue::Vec4(v),
            Value::Asset(id) => ShaderValue::Texture(id),
            ref other => {
                return Err(format!(
                    "{rel}: shader parameter `{name}` must be a number, a vector of 2 to 4 \
                     numbers or a texture asset, not a {}",
                    other.type_name()
                ));
            }
        };
        values.push((name.clone(), value));
    }
    values.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(values)
}

fn shader_value(value: ShaderValue) -> Value {
    match value {
        ShaderValue::Float(v) => Value::F32(v),
        ShaderValue::Vec2(v) => Value::Vec2(glam::Vec2::from_array(v)),
        ShaderValue::Vec3(v) => Value::Vec3(Vec3::from_array(v)),
        ShaderValue::Vec4([x, y, z, w]) => Value::Quat(glam::Quat::from_xyzw(x, y, z, w)),
        ShaderValue::Texture(id) => Value::Asset(id),
    }
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

/// The text of a `.omat` holding `file`.
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
    if let Some(shader) = file.shader {
        fields.push(field("shader", Value::Asset(shader)));
    }
    if !file.shader_values.is_empty() {
        let values = file
            .shader_values
            .iter()
            .map(|(name, value)| field(name, shader_value(*value)))
            .collect();
        fields.push(field("shader_parameters", Value::Struct(values)));
    }
    let version = if file.uses_shader() { VERSION } else { PLAIN };
    let mut out = String::new();
    orrin_registry::write_value_document(&mut out, KIND, version, version, &fields);
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
        MaterialFile {
            material,
            textures,
            ..MaterialFile::default()
        }
    }

    #[test]
    fn every_parameter_and_slot_survives_a_write_and_a_read() {
        let file = everything();
        let text = write(&file);
        let back = parse(&text, "brick.omat").unwrap();
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
            "brick.omat",
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
            "brick.omat",
        )
        .unwrap_err();
        assert_eq!(err, "brick.omat: `rougness` is not a material parameter");

        let err = parse(
            "orrin-material 1 min-reader 1\n\nroughness = \"low\"\n",
            "brick.omat",
        )
        .unwrap_err();
        assert_eq!(
            err,
            "brick.omat: field `roughness`: expected f32, found string"
        );

        let err = parse(
            "orrin-material 1 min-reader 1\n\ntextures\n  albeedo = asset(00000000-0000-4000-8000-000000000001)\n",
            "brick.omat",
        )
        .unwrap_err();
        assert_eq!(err, "brick.omat: `albeedo` is not a texture slot");

        let err = parse(
            "orrin-material 1 min-reader 1\n\nblend = Glass\n",
            "brick.omat",
        )
        .unwrap_err();
        assert_eq!(
            err,
            "brick.omat: `blend` must be Opaque, Masked, Blend or Transmissive, not Glass"
        );

        let err = parse("orrin-scene 3 min-reader 3\n", "brick.omat").unwrap_err();
        assert_eq!(
            err,
            "brick.omat: line 1: expected an `orrin-material 2 min-reader 2` header"
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

    /// [`everything`] drawn through a shader, with one value of every shape a
    /// parameter can take.
    fn shaded() -> MaterialFile {
        MaterialFile {
            shader: Some(id(20)),
            shader_values: vec![
                ("glow".into(), ShaderValue::Float(0.25)),
                ("mask".into(), ShaderValue::Texture(id(21))),
                ("offset".into(), ShaderValue::Vec2([1.0, -2.0])),
                ("rim".into(), ShaderValue::Vec3([0.1, 0.2, 0.3])),
                ("tint".into(), ShaderValue::Vec4([1.0, 0.45, 0.2, 1.0])),
            ],
            ..everything()
        }
    }

    #[test]
    fn a_shader_and_a_value_of_every_shape_survive_a_write_and_a_read() {
        let file = shaded();
        let text = write(&file);
        let back = parse(&text, "brick.omat").unwrap();
        assert_eq!(back.shader, file.shader);
        assert_eq!(back.shader_values, file.shader_values);
        assert_eq!(
            format!("{:?}", back.material),
            format!("{:?}", file.material)
        );
        assert_eq!(write(&back), text);

        assert!(
            text.starts_with("orrin-material 2 min-reader 2\n"),
            "{text}"
        );
        let block = format!(
            "\
shader = asset({})
shader_parameters
  glow = 0.25
  mask = asset({})
  offset = (1.0, -2.0)
  rim = (0.1, 0.2, 0.3)
  tint = (1.0, 0.45, 0.2, 1.0)
",
            id(20),
            id(21)
        );
        assert!(text.contains(&block), "{text}");
    }

    /// The format before shaders existed: what a file of it reads as, and that
    /// writing it back changes no byte, so a project's materials do not all
    /// show up in a diff the first time a newer editor saves them.
    #[test]
    fn a_file_of_the_format_before_shaders_reads_as_it_did_and_writes_back_unchanged() {
        let old = "\
orrin-material 1 min-reader 1

alpha = 1.0
base_color = (0.5, 0.25, 0.125)
roughness = 0.75
textures
  albedo = asset(01000000-0000-4000-8000-000000000001)
";
        let file = parse(old, "brick.omat").unwrap();
        assert_eq!(file.shader, None);
        assert!(file.shader_values.is_empty());
        assert_eq!(file.material.base_color, Vec3::new(0.5, 0.25, 0.125));
        assert_eq!(file.textures.albedo, Some(id(1)));
        let written = write(&file);
        assert!(written.starts_with("orrin-material 1 min-reader 1\n\n"));
        assert_eq!(
            write(&MaterialFile::default()),
            write(&parse(&write(&MaterialFile::default()), "brick.omat").unwrap())
        );
    }

    /// A file that names a shader declares format 2 as the oldest reader of
    /// it, so an engine from before shaders refuses it by name rather than
    /// drawing it as the standard material it would otherwise read it as —
    /// and a version 1 header over a shader is refused here for the same
    /// reason, since an older engine would trust it.
    #[test]
    fn a_file_using_a_shader_needs_a_reader_of_format_2_both_ways() {
        let text = write(&shaded());
        let older = orrin_registry::parse_value_document(&text, KIND, 1).unwrap_err();
        assert_eq!(
            older.to_string(),
            "line 1: orrin-material version 2 needs a reader of version 2 or later (this \
             build reads up to 1)"
        );

        let shader = format!(
            "orrin-material 1 min-reader 1\n\nshader = asset({})\n",
            id(20)
        );
        assert_eq!(
            parse(&shader, "brick.omat").unwrap_err(),
            "brick.omat: `shader` needs material format 2: write the header as \
             `orrin-material 2 min-reader 2`, so an engine older than surface shaders \
             refuses the file instead of drawing it without its shader"
        );
        let values = "orrin-material 2 min-reader 1\n\nshader_parameters\n  glow = 1.0\n";
        assert!(
            parse(values, "brick.omat")
                .unwrap_err()
                .starts_with("brick.omat: `shader_parameters` needs material format 2"),
        );
    }

    /// The file cannot know what its shader declares, so any name is kept —
    /// a parameter renamed in the shader leaves its old value in the file, to
    /// be warned about when the material is drawn — while a value no
    /// parameter could hold is refused, as a mistyped built-in one is.
    #[test]
    fn any_parameter_name_is_kept_but_a_value_of_no_parameter_shape_is_refused() {
        let text = format!(
            "orrin-material 2 min-reader 2\n\nshader = asset({})\nshader_parameters\n  \
             renamed_long_ago = 0.5\n  tint = rgba(1.0, 0.0, 0.0, 1.0)\n  wear = 1\n",
            id(20)
        );
        let file = parse(&text, "brick.omat").unwrap();
        assert_eq!(
            file.shader_values,
            [
                ("renamed_long_ago".into(), ShaderValue::Float(0.5)),
                ("tint".into(), ShaderValue::Vec4([1.0, 0.0, 0.0, 1.0])),
                ("wear".into(), ShaderValue::Float(1.0)),
            ]
        );
        assert!(write(&file).contains("  renamed_long_ago = 0.5\n"));

        let flag = "orrin-material 2 min-reader 2\n\nshader_parameters\n  flag = true\n";
        assert_eq!(
            parse(flag, "brick.omat").unwrap_err(),
            "brick.omat: shader parameter `flag` must be a number, a vector of 2 to 4 numbers \
             or a texture asset, not a bool"
        );
        let shader = "orrin-material 2 min-reader 2\n\nshader = 3\n";
        assert_eq!(
            parse(shader, "brick.omat").unwrap_err(),
            "brick.omat: field `shader`: expected asset, found i32"
        );
    }

    /// Only what was set is written: a parameter left at its default is absent,
    /// so a default edited in the shader reaches every material that did not
    /// override it.
    #[test]
    fn a_value_is_set_replaced_and_cleared_by_name_and_only_set_values_are_written() {
        let mut file = MaterialFile {
            shader: Some(id(20)),
            ..MaterialFile::default()
        };
        let bare = write(&file);
        assert!(!bare.contains("shader_parameters"), "{bare}");
        assert!(bare.starts_with("orrin-material 2 min-reader 2\n"));

        file.set_shader_value("wear", Some(ShaderValue::Float(0.5)));
        file.set_shader_value("glow", Some(ShaderValue::Float(1.0)));
        file.set_shader_value("wear", Some(ShaderValue::Float(0.75)));
        assert_eq!(
            file.shader_values,
            [
                ("glow".into(), ShaderValue::Float(1.0)),
                ("wear".into(), ShaderValue::Float(0.75)),
            ]
        );
        file.set_shader_value("glow", None);
        file.set_shader_value("never_set", None);
        assert_eq!(
            file.shader_values,
            [("wear".into(), ShaderValue::Float(0.75))]
        );
        assert_eq!(file.shader_value("wear"), Some(ShaderValue::Float(0.75)));
        assert_eq!(file.shader_value("glow"), None);
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
