//! glTF 2.0 import: a file on disk to [`CpuMesh`]es, [`Material`]s and a node
//! tree, then to entities.
//!
//! Split in two on purpose. [`load`] is a pure CPU parse that takes no `Device`
//! and no `World` — which is what lets [the tests below](self#tests) assert the
//! vertex data, the tangent basis and the material mapping on a machine with no
//! GPU, the same argument `gfx/graph` makes for its golden plan.
//! [`instantiate`] is the half that needs a backend: it uploads, registers the
//! three world-side tables together, and spawns the tree.
//!
//! # What is imported, and what is dropped
//!
//! glTF's coordinate conventions are this engine's — right-handed, `+Y` up,
//! forward `-Z`, metres — so a node transform is copied across rather than
//! converted. `pbrMetallicRoughness` maps onto [`Material`] field for field,
//! including `alphaMode`/`alphaCutoff`, which [`Material::alpha_cutoff`] already
//! documents itself against glTF's rule.
//!
//! Dropped, each for a reason:
//!
//! - **The occlusion texture.** This renderer computes occlusion in screen space
//!   ([`SsaoSettings`](crate::scene::SsaoSettings)) and has no baked-AO slot to
//!   put it in. Sampling it into albedo would double up wherever the two agree.
//! - **`KHR_texture_transform`.** There is one shared sampler and no per-material
//!   UV transform, so a file using it would draw with the wrong texel offsets;
//!   [`load`] warns rather than silently mis-sampling.
//! - **Sampler wrap and filter modes.** Same reason: the backend binds one
//!   repeating, mip-filtered sampler for every texture.
//! - **`doubleSided` on an opaque material.** Only the masked pipelines are
//!   two-sided here, because a cutout sheet is what they exist for — so an opaque
//!   single-thickness surface authored two-sided vanishes when seen from behind,
//!   which in Sponza is the fabric awnings over the arcade. Honouring it needs a
//!   back-face-drawing variant of all three opaque pipelines, so it is a renderer
//!   change and not an import one.
//! - **Cameras, animations, skins, morph targets, and every primitive mode but
//!   `TRIANGLES`.** Nothing here consumes them yet; a skipped primitive is a
//!   warning naming the mesh.
//! - **`data:` URIs.** Supporting them needs a base64 decoder, and the two shapes
//!   that matter — a `.gltf` beside its `.bin` and a self-contained `.glb` — do
//!   not use them. An embedded buffer is a named error rather than a half-read
//!   file.
//!
//! Vertex colour is white where a file supplies none, which is not a detail:
//! `read_surface` multiplies vertex colour into albedo, so the alternative is
//! black geometry. It is the same requirement [`CpuMesh::uncolored`] exists for.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use glam::{Mat4, Quat, Vec2, Vec3, Vec4};

use orrin_ecs::{Entity, World};

use crate::geom::Aabb;
use crate::gfx::{BlendMode, Material, RenderBackend, Vertex};
use crate::scene::{
    Assets, CpuMesh, LocalTransform, MaterialBlends, MaterialHandle, MeshBounds, MeshHandle, Name,
    Parent, Transform,
};

/// How deep the node walks will follow a file's tree before giving up.
///
/// glTF's hierarchy is required to be a tree, and neither `gltf`'s validation nor
/// anything here proves it: a file that parents two nodes to each other recurses
/// until the stack ends, which is an abort with no message attached. Depth is the
/// cheap guard, and 256 is far past any exporter's nesting.
const MAX_NODE_DEPTH: usize = 256;

/// One triangle list, and which of [`Model::materials`] shades it.
pub struct ModelPrimitive {
    pub mesh: CpuMesh,
    pub material: usize,
}

/// A material as the file describes it: scalars resolved, maps still as indices
/// into [`Model::images`].
///
/// Indices rather than handles because [`load`] never touches the backend, and
/// because one image can be referenced by several materials — the upload
/// deduplicates, which is what keeps a 25-material file inside
/// [`MAX_TEXTURES`](crate::gfx::MAX_TEXTURES).
pub struct ModelMaterial {
    pub name: String,
    pub base_color: Vec3,
    pub alpha: f32,
    pub alpha_cutoff: f32,
    pub blend: BlendMode,
    pub metallic: f32,
    pub roughness: f32,
    /// glTF's `emissiveFactor` is a reflectance-like `[0, 1]` triple, while
    /// [`Material::emissive`] is a luminance in cd/m². The scale is applied at
    /// import by the caller's [`ImportSettings::emissive_luminance`], because
    /// nothing in the file says how bright "1.0 emissive" is meant to be and a
    /// photometric frame cannot guess.
    pub emissive: Vec3,
    pub albedo_image: Option<usize>,
    pub normal_image: Option<usize>,
    pub metallic_roughness_image: Option<usize>,
    pub emissive_image: Option<usize>,
}

/// A decoded texture: tightly packed RGBA8, row-major from the top-left.
///
/// Whether it is sRGB is *not* stored, because it is not a property of the
/// image — the same file can be a colour map for one material and data for
/// another. The upload decides per use and caches on the pair.
pub struct ModelImage {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// A node of the file's tree: its own local transform, its children, and the
/// primitives drawn at it.
pub struct ModelNode {
    pub name: String,
    pub transform: Transform,
    pub children: Vec<usize>,
    pub primitives: Vec<usize>,
}

pub struct Model {
    /// Stem of the file it came from, used to prefix every registered asset name
    /// so two imported models cannot collide in [`Assets`].
    pub name: String,
    pub primitives: Vec<ModelPrimitive>,
    pub materials: Vec<ModelMaterial>,
    pub images: Vec<ModelImage>,
    pub nodes: Vec<ModelNode>,
    /// Nodes of the default scene, or every parentless node if the file names no
    /// scene.
    pub roots: Vec<usize>,
}

impl Model {
    /// The box the model occupies once its node transforms are applied, in the
    /// file's own units.
    ///
    /// The reason this exists rather than being left to the caller: one world
    /// unit is one metre in this engine, and a downloaded model is as likely to
    /// be authored in centimetres as in metres. A scene that *measures* what it
    /// imported can say so, and can place a camera in terms of the model's real
    /// extent instead of magic numbers that only hold for one copy of one file.
    pub fn bounds(&self) -> Aabb {
        let mut bounds = Aabb::EMPTY;
        for &root in &self.roots {
            self.accumulate_bounds(root, Mat4::IDENTITY, 0, &mut bounds);
        }
        bounds
    }

    fn accumulate_bounds(&self, index: usize, parent: Mat4, depth: usize, bounds: &mut Aabb) {
        let Some(node) = self.nodes.get(index) else {
            return;
        };
        if depth > MAX_NODE_DEPTH {
            return;
        }
        let world = parent * node.transform.matrix();
        for &primitive in &node.primitives {
            let local = self.primitives[primitive].mesh.bounds();
            if local.is_valid() {
                *bounds = bounds.union(&local.transformed(&world));
            }
        }
        for &child in &node.children {
            self.accumulate_bounds(child, world, depth + 1, bounds);
        }
    }

    /// Triangles across every primitive, whether or not a node draws them. For
    /// the load log: it is the one number that says whether the file that landed
    /// on disk is the model anyone meant.
    pub fn triangle_count(&self) -> usize {
        self.primitives
            .iter()
            .map(|p| p.mesh.indices.len() / 3)
            .sum()
    }
}

#[derive(Debug)]
pub enum ModelError {
    Read {
        path: PathBuf,
        error: std::io::Error,
    },
    Parse {
        path: PathBuf,
        error: gltf::Error,
    },
    /// A buffer or image the document references could not be resolved. Named
    /// separately from [`Self::Read`] because the path in question is one the
    /// *file* chose, not one the caller passed.
    Resource {
        uri: String,
        reason: String,
    },
    Decode {
        uri: String,
        error: image::ImageError,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, error } => write!(f, "could not read {}: {error}", path.display()),
            Self::Parse { path, error } => {
                write!(
                    f,
                    "{} is not a readable glTF 2.0 file: {error}",
                    path.display()
                )
            }
            Self::Resource { uri, reason } => write!(f, "could not load `{uri}`: {reason}"),
            Self::Decode { uri, error } => write!(f, "could not decode `{uri}`: {error}"),
        }
    }
}

impl std::error::Error for ModelError {}

/// What the file cannot say, and the importer therefore has to be told.
pub struct ImportSettings {
    /// Multiplies every position and translation, for a file authored in
    /// anything but metres. The original Crytek Sponza is in centimetres; the
    /// glTF re-release is in metres, so this is `1.0` for it — but a scene that
    /// hard-codes the assumption cannot report a mismatch, which is why
    /// [`Model::bounds`] is measured and logged either way.
    pub scale: f32,
    /// cd/m² for an `emissiveFactor` of 1. See [`ModelMaterial::emissive`].
    /// A few thousand is a lit sign; the default is deliberately modest, since
    /// an emissive surface that outshines the sun is the failure mode here.
    pub emissive_luminance: f32,
}

impl Default for ImportSettings {
    fn default() -> Self {
        Self {
            scale: 1.0,
            emissive_luminance: 1_000.0,
        }
    }
}

/// Parse a `.gltf` or `.glb`, resolving its buffers and images relative to it.
pub fn load(path: &Path, settings: &ImportSettings) -> Result<Model, ModelError> {
    let bytes = std::fs::read(path).map_err(|error| ModelError::Read {
        path: path.to_path_buf(),
        error,
    })?;
    let gltf::Gltf { document, blob } =
        gltf::Gltf::from_slice(&bytes).map_err(|error| ModelError::Parse {
            path: path.to_path_buf(),
            error,
        })?;

    let base = path.parent().unwrap_or(Path::new("."));
    let buffers = load_buffers(&document, base, blob)?;
    let images = load_images(&document, base, &buffers)?;

    if document
        .materials()
        .flat_map(|m| m.pbr_metallic_roughness().base_color_texture())
        .any(|t| t.texture_transform().is_some())
    {
        tracing::warn!(
            "{}: uses KHR_texture_transform, which this renderer has no per-material UV \
             transform for; textures will sample untransformed",
            path.display()
        );
    }

    let materials = document
        .materials()
        .map(|material| import_material(&material, settings))
        .collect();

    let mut primitives = Vec::new();
    // Where each `(mesh, primitive)` pair landed in `primitives`, so a mesh drawn
    // at twenty nodes is uploaded once.
    let mut mesh_primitives: Vec<Vec<usize>> = Vec::with_capacity(document.meshes().len());
    for mesh in document.meshes() {
        let mut indices = Vec::new();
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                tracing::warn!(
                    "{}: skipping a {:?} primitive of mesh `{}`; only triangle lists are drawn",
                    path.display(),
                    primitive.mode(),
                    mesh.name().unwrap_or("<unnamed>")
                );
                continue;
            }
            let cpu = import_primitive(&primitive, &buffers, settings.scale);
            if cpu.vertices.is_empty() || cpu.indices.is_empty() {
                continue;
            }
            indices.push(primitives.len());
            primitives.push(ModelPrimitive {
                mesh: cpu,
                // A primitive with no material takes glTF's default, which this
                // maps to the last slot appended below.
                material: primitive
                    .material()
                    .index()
                    .unwrap_or(document.materials().len()),
            });
        }
        mesh_primitives.push(indices);
    }

    let mut materials: Vec<ModelMaterial> = materials;
    // glTF's default material, appended whether or not anything uses it: one
    // extra struct in the material buffer costs no bandwidth, and the
    // alternative is an index that may or may not be in range.
    materials.push(ModelMaterial {
        name: "default".to_string(),
        base_color: Vec3::ONE,
        alpha: 1.0,
        alpha_cutoff: 0.5,
        blend: BlendMode::Opaque,
        metallic: 1.0,
        roughness: 1.0,
        emissive: Vec3::ZERO,
        albedo_image: None,
        normal_image: None,
        metallic_roughness_image: None,
        emissive_image: None,
    });

    let nodes = document
        .nodes()
        .map(|node| {
            let (translation, rotation, scale) = node.transform().decomposed();
            ModelNode {
                name: node
                    .name()
                    .map_or_else(|| format!("Node {}", node.index()), |name| name.to_string()),
                transform: Transform {
                    // Only the translation is scaled. Scaling the node's own
                    // `scale` as well would compound with the vertex positions,
                    // which `import_primitive` has already scaled, and inflate
                    // every child by the factor squared.
                    translation: Vec3::from(translation) * settings.scale,
                    rotation: Quat::from_array(rotation),
                    scale: Vec3::from(scale),
                },
                children: node.children().map(|child| child.index()).collect(),
                primitives: node
                    .mesh()
                    .map(|mesh| mesh_primitives[mesh.index()].clone())
                    .unwrap_or_default(),
            }
        })
        .collect();

    let roots = match document
        .default_scene()
        .or_else(|| document.scenes().next())
    {
        Some(scene) => scene.nodes().map(|node| node.index()).collect(),
        // No scene at all is legal glTF. Every node that nothing else parents is
        // then a root, which is what a viewer does with such a file.
        None => {
            let mut parented = vec![false; document.nodes().len()];
            for node in document.nodes() {
                for child in node.children() {
                    parented[child.index()] = true;
                }
            }
            (0..document.nodes().len())
                .filter(|&index| !parented[index])
                .collect()
        }
    };

    Ok(Model {
        name: path
            .file_stem()
            .map_or_else(|| "model".to_string(), |stem| stem.to_string_lossy().into()),
        primitives,
        materials,
        images,
        nodes,
        roots,
    })
}

fn load_buffers(
    document: &gltf::Document,
    base: &Path,
    mut blob: Option<Vec<u8>>,
) -> Result<Vec<Vec<u8>>, ModelError> {
    document
        .buffers()
        .map(|buffer| match buffer.source() {
            gltf::buffer::Source::Bin => blob.take().ok_or_else(|| ModelError::Resource {
                uri: format!("buffer {}", buffer.index()),
                reason: "the document names the GLB binary chunk, but the file has none"
                    .to_string(),
            }),
            gltf::buffer::Source::Uri(uri) => read_uri(uri, base),
        })
        .collect()
}

/// Where one image's encoded bytes come from, resolved but not yet read.
///
/// Named separately so the decode below has an indexable slice to split: a
/// `gltf::Document`'s images are an iterator over a borrowed document, and
/// neither is `Sync`.
enum ImageSource<'a> {
    /// A file beside the document — a `.gltf`'s textures.
    Uri(String),
    /// A range of a buffer already in memory — a `.glb`'s. Borrowed rather than
    /// copied out: the copy would be of the whole encoded texture, for nothing.
    View { bytes: &'a [u8], name: String },
}

/// Decode every image the document names, in document order.
///
/// The decode is the expensive half of importing a model — Sponza is ~70 JPEGs
/// — and it is pure CPU with no shared state, so it goes across the pool. The
/// *resolution* above it stays serial: it walks a borrowed `gltf::Document`,
/// and it is bookkeeping either way.
///
/// One difference from a serial fold, and it is deliberate: a file that fails to
/// decode no longer stops the ones after it, since they may already be in
/// flight. The error reported is still the first in document order, so which
/// failure a broken model names does not depend on how the work was split.
fn load_images(
    document: &gltf::Document,
    base: &Path,
    buffers: &[Vec<u8>],
) -> Result<Vec<ModelImage>, ModelError> {
    let sources = document
        .images()
        .map(|image| match image.source() {
            gltf::image::Source::Uri { uri, .. } => Ok(ImageSource::Uri(uri.to_string())),
            gltf::image::Source::View { view, .. } => {
                let buffer = &buffers[view.buffer().index()];
                let start = view.offset();
                let end = start + view.length();
                let name = format!("image {}", image.index());
                let bytes = buffer.get(start..end).ok_or_else(|| ModelError::Resource {
                    uri: name.clone(),
                    reason: format!(
                        "its buffer view runs to {end} bytes, past the {} the buffer holds",
                        buffer.len()
                    ),
                })?;
                Ok(ImageSource::View { bytes, name })
            }
        })
        .collect::<Result<Vec<_>, ModelError>>()?;

    crate::threads::map(&sources, |source| match source {
        ImageSource::Uri(uri) => decode_image(&read_uri(uri, base)?, uri),
        ImageSource::View { bytes, name } => decode_image(bytes, name),
    })
    .into_iter()
    .collect()
}

/// Resolve a glTF URI against the document's directory.
///
/// Percent-escapes are decoded by hand for the one case that matters — a space
/// in a texture's file name, which exporters write as `%20`. A `data:` URI is a
/// named error rather than a silent empty buffer.
fn read_uri(uri: &str, base: &Path) -> Result<Vec<u8>, ModelError> {
    if uri.starts_with("data:") {
        return Err(ModelError::Resource {
            uri: uri.chars().take(32).collect(),
            reason: "embedded `data:` URIs are not supported; use a .glb, or a .gltf beside its \
                     .bin and image files"
                .to_string(),
        });
    }
    let path = base.join(percent_decode(uri));
    std::fs::read(&path).map_err(|error| ModelError::Resource {
        uri: uri.to_string(),
        reason: format!("{}: {error}", path.display()),
    })
}

fn percent_decode(uri: &str) -> String {
    let bytes = uri.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        let decoded = (bytes[index] == b'%' && index + 2 < bytes.len())
            .then(|| {
                let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).ok()?;
                u8::from_str_radix(hex, 16).ok()
            })
            .flatten();
        match decoded {
            Some(byte) => {
                out.push(byte);
                index += 3;
            }
            None => {
                out.push(bytes[index]);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn decode_image(bytes: &[u8], uri: &str) -> Result<ModelImage, ModelError> {
    let decoded = image::load_from_memory(bytes).map_err(|error| ModelError::Decode {
        uri: uri.to_string(),
        error,
    })?;
    let rgba = decoded.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(ModelImage {
        pixels: rgba.into_raw(),
        width,
        height,
    })
}

fn import_material(material: &gltf::Material, settings: &ImportSettings) -> ModelMaterial {
    let pbr = material.pbr_metallic_roughness();
    let base = pbr.base_color_factor();
    ModelMaterial {
        name: material.name().map_or_else(
            || format!("material_{}", material.index().unwrap_or_default()),
            |name| name.to_string(),
        ),
        base_color: Vec3::new(base[0], base[1], base[2]),
        alpha: base[3],
        alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
        blend: match material.alpha_mode() {
            gltf::material::AlphaMode::Opaque => BlendMode::Opaque,
            gltf::material::AlphaMode::Mask => BlendMode::Masked,
            gltf::material::AlphaMode::Blend => BlendMode::Blend,
        },
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        emissive: Vec3::from(material.emissive_factor()) * settings.emissive_luminance,
        albedo_image: pbr.base_color_texture().map(image_index),
        normal_image: material
            .normal_texture()
            .map(|t| t.texture().source().index()),
        metallic_roughness_image: pbr.metallic_roughness_texture().map(image_index),
        emissive_image: material.emissive_texture().map(image_index),
    }
}

fn image_index(info: gltf::texture::Info) -> usize {
    info.texture().source().index()
}

fn import_primitive(primitive: &gltf::Primitive, buffers: &[Vec<u8>], scale: f32) -> CpuMesh {
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(|data| data.as_slice()));

    let positions: Vec<Vec3> = reader
        .read_positions()
        .map(|iter| iter.map(|p| Vec3::from(p) * scale).collect())
        .unwrap_or_default();
    if positions.is_empty() {
        return CpuMesh::new(Vec::new(), Vec::new());
    }

    let indices: Vec<u32> = reader
        .read_indices()
        .map(|iter| iter.into_u32().collect())
        // A non-indexed primitive is a triangle list in vertex order.
        .unwrap_or_else(|| (0..positions.len() as u32).collect());

    let uvs: Vec<Vec2> = reader
        .read_tex_coords(0)
        .map(|iter| iter.into_f32().map(Vec2::from).collect())
        .unwrap_or_default();

    let normals: Vec<Vec3> = match reader.read_normals() {
        Some(iter) => iter.map(Vec3::from).collect(),
        None => derive_normals(&positions, &indices),
    };

    let colors: Vec<Vec3> = reader
        .read_colors(0)
        .map(|iter| {
            iter.into_rgb_f32()
                .map(|[r, g, b]| Vec3::new(r, g, b))
                .collect()
        })
        .unwrap_or_default();

    let tangents: Vec<Vec4> = match reader.read_tangents() {
        Some(iter) => iter.map(Vec4::from).collect(),
        // Only derivable from a UV set. Without one there is no `+U` direction
        // to point at, so the fallback below is an arbitrary basis — which is
        // correct for the only material that can be shaded without UVs, one with
        // no normal map.
        None if !uvs.is_empty() => derive_tangents(&positions, &normals, &uvs, &indices),
        None => Vec::new(),
    };

    let vertices = (0..positions.len())
        .map(|index| {
            let normal = normals.get(index).copied().unwrap_or(Vec3::Y);
            Vertex {
                position: positions[index].to_array(),
                normal: normal.to_array(),
                // White, not black: `read_surface` multiplies vertex colour into
                // albedo.
                color: colors.get(index).copied().unwrap_or(Vec3::ONE).to_array(),
                uv: uvs.get(index).copied().unwrap_or(Vec2::ZERO).to_array(),
                tangent: tangents
                    .get(index)
                    .copied()
                    .unwrap_or_else(|| fallback_tangent(normal))
                    .to_array(),
            }
        })
        .collect();

    CpuMesh::new(vertices, indices)
}

/// Any unit vector perpendicular to `normal`, with positive handedness.
///
/// Used where a primitive carries no UVs at all. It is a basis rather than a
/// zero vector because the TBN rebuild in the shaders normalises what it is
/// given, and a zero tangent there is a NaN normal on a surface that would
/// otherwise have shaded fine.
fn fallback_tangent(normal: Vec3) -> Vec4 {
    let axis = if normal.x.abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let tangent = axis.cross(normal).normalize_or(Vec3::X).cross(normal);
    tangent.normalize_or(Vec3::X).extend(1.0)
}

/// Area-weighted vertex normals, for a file that ships none.
///
/// Area-weighted because it is what accumulating the un-normalised cross product
/// gives for free, and it is the better answer: a large triangle should pull a
/// shared vertex's normal further than a sliver does.
fn derive_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for triangle in indices.as_chunks::<3>().0 {
        let [i0, i1, i2] = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        let Some(&p0) = positions.get(i0) else {
            continue;
        };
        let Some(&p1) = positions.get(i1) else {
            continue;
        };
        let Some(&p2) = positions.get(i2) else {
            continue;
        };
        let face = (p1 - p0).cross(p2 - p0);
        for index in [i0, i1, i2] {
            normals[index] += face;
        }
    }
    normals
        .into_iter()
        .map(|n| n.normalize_or(Vec3::Y))
        .collect()
}

/// Per-vertex tangents from the UV parameterisation, Gram-Schmidt against the
/// normal, handedness in `w`.
///
/// `w` is what the shaders rebuild the bitangent with, so it is the whole reason
/// a generated basis can be wrong in a way that looks like a lighting bug: a
/// mirrored UV island has the opposite handedness, and a normal map read through
/// the wrong one lights from the wrong side along one axis only.
fn derive_tangents(
    positions: &[Vec3],
    normals: &[Vec3],
    uvs: &[Vec2],
    indices: &[u32],
) -> Vec<Vec4> {
    let mut tangents = vec![Vec3::ZERO; positions.len()];
    let mut bitangents = vec![Vec3::ZERO; positions.len()];

    for triangle in indices.as_chunks::<3>().0 {
        let [i0, i1, i2] = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        let (Some(&p0), Some(&p1), Some(&p2)) =
            (positions.get(i0), positions.get(i1), positions.get(i2))
        else {
            continue;
        };
        let (Some(&w0), Some(&w1), Some(&w2)) = (uvs.get(i0), uvs.get(i1), uvs.get(i2)) else {
            continue;
        };

        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let duv1 = w1 - w0;
        let duv2 = w2 - w0;
        let determinant = duv1.x * duv2.y - duv2.x * duv1.y;
        // A degenerate UV triangle has no `+U` direction; leaving it at zero lets
        // its vertices take their neighbours' contributions, and the
        // orthogonalisation below catches a vertex that has none.
        if determinant.abs() < 1e-12 {
            continue;
        }
        let inverse = determinant.recip();
        let tangent = (edge1 * duv2.y - edge2 * duv1.y) * inverse;
        let bitangent = (edge2 * duv1.x - edge1 * duv2.x) * inverse;
        for index in [i0, i1, i2] {
            tangents[index] += tangent;
            bitangents[index] += bitangent;
        }
    }

    (0..positions.len())
        .map(|index| {
            let normal = normals.get(index).copied().unwrap_or(Vec3::Y);
            let accumulated = tangents[index];
            // Reject the normal out of the accumulated tangent rather than
            // trusting the UV derivative to be perpendicular to it; interpolated
            // normals are not.
            let tangent = (accumulated - normal * normal.dot(accumulated)).normalize_or(Vec3::ZERO);
            if tangent == Vec3::ZERO {
                return fallback_tangent(normal);
            }
            let handedness = if normal.cross(tangent).dot(bitangents[index]) < 0.0 {
                -1.0
            } else {
                1.0
            };
            tangent.extend(handedness)
        })
        .collect()
}

/// Upload a model and spawn it under one root entity, which is returned.
///
/// Takes the three world-side tables together for the reason `load_library`'s
/// helpers do: a mesh that reaches a draw list without bounds is unculled, and a
/// blended material registered nowhere draws as an opaque wall. Names are
/// prefixed with the model's, since [`Assets`] is one flat namespace shared with
/// the demo library.
pub fn instantiate(
    world: &mut World,
    backend: &mut impl RenderBackend,
    assets: &mut Assets,
    bounds: &mut MeshBounds,
    blends: &mut MaterialBlends,
    model: &Model,
    root: Transform,
) -> Entity {
    // Keyed on `(image, srgb)`: whether a texture is decoded as sRGB is a
    // property of the slot it is bound to, and one file can legitimately be both
    // — so this deduplicates uses, not images.
    let mut textures: HashMap<(usize, bool), _> = HashMap::new();
    let mut upload_image =
        |backend: &mut _, assets: &mut Assets, index: Option<usize>, srgb: bool| {
            let index = index?;
            let image = model.images.get(index)?;
            let handle = *textures.entry((index, srgb)).or_insert_with(|| {
                let handle = RenderBackend::load_texture(
                    backend,
                    &image.pixels,
                    image.width,
                    image.height,
                    srgb,
                );
                let space = if srgb { "srgb" } else { "linear" };
                assets.insert_texture(format!("{}/image_{index}_{space}", model.name), handle);
                handle
            });
            Some(handle)
        };

    let materials: Vec<_> = model
        .materials
        .iter()
        .map(|imported| {
            let material = Material {
                base_color: imported.base_color,
                alpha: imported.alpha,
                alpha_cutoff: imported.alpha_cutoff,
                blend: imported.blend,
                metallic: imported.metallic,
                roughness: imported.roughness,
                emissive: imported.emissive,
                albedo_texture: upload_image(backend, assets, imported.albedo_image, true),
                normal_texture: upload_image(backend, assets, imported.normal_image, false),
                metallic_roughness_texture: upload_image(
                    backend,
                    assets,
                    imported.metallic_roughness_image,
                    false,
                ),
                emissive_texture: upload_image(backend, assets, imported.emissive_image, true),
                ..Material::default()
            };
            let handle = backend.load_material(&material);
            assets.insert_material(format!("{}/{}", model.name, imported.name), handle);
            blends.insert(handle, material.blend);
            handle
        })
        .collect();

    let meshes: Vec<MeshHandle> = model
        .primitives
        .iter()
        .enumerate()
        .map(|(index, primitive)| {
            let handle = backend.load_mesh(&primitive.mesh);
            assets.insert_mesh(format!("{}/mesh_{index}", model.name), handle);
            if let Some(aabb) = backend.mesh_bounds(handle) {
                bounds.insert(handle, aabb);
            }
            handle
        })
        .collect();

    let root_entity = world
        .spawn_entity()
        .with(Name::new(model.name.clone()))
        .with(LocalTransform::from(root))
        .id();

    for &node in &model.roots {
        spawn_node(world, model, &meshes, &materials, node, root_entity, 0);
    }

    root_entity
}

fn spawn_node(
    world: &mut World,
    model: &Model,
    meshes: &[MeshHandle],
    materials: &[MaterialHandle],
    index: usize,
    parent: Entity,
    depth: usize,
) {
    let Some(node) = model.nodes.get(index) else {
        return;
    };
    if depth > MAX_NODE_DEPTH {
        tracing::warn!(
            "{}: node tree is deeper than {MAX_NODE_DEPTH}; `{}` and anything under it is not \
             spawned — the file's hierarchy is not a tree",
            model.name,
            node.name
        );
        return;
    }

    let entity = world
        .spawn_entity()
        .with(Name::new(node.name.clone()))
        .with(LocalTransform::from(node.transform))
        .with(Parent::new(parent))
        .id();

    // One primitive sits on the node itself; several become children, because a
    // renderable is one mesh and one material here while a glTF mesh is a list
    // of both.
    match node.primitives.as_slice() {
        [] => {}
        [only] => attach(world, entity, model, meshes, materials, *only),
        many => {
            for (part, &primitive) in many.iter().enumerate() {
                let child = world
                    .spawn_entity()
                    .with(Name::new(format!("{} [{part}]", node.name)))
                    .with(LocalTransform::from(Transform::default()))
                    .with(Parent::new(entity))
                    .id();
                attach(world, child, model, meshes, materials, primitive);
            }
        }
    }

    for &child in &node.children {
        spawn_node(world, model, meshes, materials, child, entity, depth + 1);
    }
}

fn attach(
    world: &mut World,
    entity: Entity,
    model: &Model,
    meshes: &[MeshHandle],
    materials: &[MaterialHandle],
    primitive: usize,
) {
    let material = model.primitives[primitive]
        .material
        .min(materials.len() - 1);
    world.insert(entity, meshes[primitive]);
    world.insert(entity, materials[material]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3A;

    /// A single triangle in the `xy` plane with a clean UV layout, written as a
    /// `.gltf` beside its `.bin` — the shape a downloaded model actually has, so
    /// the URI resolution is under test alongside the vertex data.
    fn write_triangle(dir: &Path, with_tangents: bool) -> PathBuf {
        // POSITION, NORMAL, TEXCOORD_0, then indices. Interleaved as separate
        // views over one buffer, which is what an exporter emits.
        let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals: [[f32; 3]; 3] = [[0.0, 0.0, 1.0]; 3];
        let uvs: [[f32; 2]; 3] = [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]];
        let indices: [u16; 3] = [0, 1, 2];

        let mut bin = Vec::new();
        for value in positions.iter().flatten().chain(normals.iter().flatten()) {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        let uv_offset = bin.len();
        for value in uvs.iter().flatten() {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        let index_offset = bin.len();
        for value in indices {
            bin.extend_from_slice(&value.to_le_bytes());
        }

        std::fs::write(dir.join("triangle.bin"), &bin).unwrap();

        let tangent_accessor = if with_tangents {
            r#", "TANGENT": 4"#
        } else {
            ""
        };
        let tangent_extras = if with_tangents {
            // Deliberately the *opposite* handedness a derivation would produce,
            // so a test that reads it back proves the file won over the
            // generator.
            r#",
            {"bufferView": 4, "componentType": 5126, "count": 3, "type": "VEC4"}"#
        } else {
            ""
        };
        let tangent_view = if with_tangents {
            let offset = bin.len();
            let mut tangents = Vec::new();
            for _ in 0..3 {
                for value in [1.0f32, 0.0, 0.0, -1.0] {
                    tangents.extend_from_slice(&value.to_le_bytes());
                }
            }
            let mut all = bin.clone();
            all.extend_from_slice(&tangents);
            std::fs::write(dir.join("triangle.bin"), &all).unwrap();
            format!(
                r#",
            {{"buffer": 0, "byteOffset": {offset}, "byteLength": 48}}"#
            )
        } else {
            String::new()
        };
        let buffer_len = std::fs::metadata(dir.join("triangle.bin")).unwrap().len();

        let gltf = format!(
            r#"{{
          "asset": {{"version": "2.0"}},
          "scene": 0,
          "scenes": [{{"nodes": [0]}}],
          "nodes": [{{"name": "Tri", "mesh": 0, "translation": [2.0, 0.0, 0.0]}}],
          "meshes": [{{"name": "TriMesh", "primitives": [{{
            "attributes": {{"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2{tangent_accessor}}},
            "indices": 3, "material": 0
          }}]}}],
          "materials": [{{
            "name": "Cutout",
            "alphaMode": "MASK", "alphaCutoff": 0.25, "doubleSided": true,
            "pbrMetallicRoughness": {{
              "baseColorFactor": [0.5, 0.25, 0.125, 0.75],
              "metallicFactor": 0.0, "roughnessFactor": 0.6
            }},
            "emissiveFactor": [1.0, 0.0, 0.0]
          }}],
          "accessors": [
            {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
              "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]}},
            {{"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"}},
            {{"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"}},
            {{"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"}}{tangent_extras}
          ],
          "bufferViews": [
            {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
            {{"buffer": 0, "byteOffset": 36, "byteLength": 36}},
            {{"buffer": 0, "byteOffset": {uv_offset}, "byteLength": 24}},
            {{"buffer": 0, "byteOffset": {index_offset}, "byteLength": 6}}{tangent_view}
          ],
          "buffers": [{{"uri": "triangle.bin", "byteLength": {buffer_len}}}]
        }}"#
        );

        let path = dir.join("triangle.gltf");
        std::fs::write(&path, gltf).unwrap();
        path
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("orrin-model-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_primitive_arrives_with_the_vertex_layout_the_pipeline_expects() {
        let dir = scratch("layout");
        let path = write_triangle(&dir, false);
        let model = load(&path, &ImportSettings::default()).unwrap();

        assert_eq!(model.primitives.len(), 1);
        let mesh = &model.primitives[0].mesh;
        assert_eq!(mesh.indices, vec![0, 1, 2]);
        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.vertices[1].position, [1.0, 0.0, 0.0]);
        assert_eq!(mesh.vertices[0].normal, [0.0, 0.0, 1.0]);
        assert_eq!(mesh.vertices[0].uv, [0.0, 1.0]);
        // White rather than the black a missing COLOR_0 would otherwise leave:
        // `read_surface` multiplies vertex colour into albedo.
        assert_eq!(mesh.vertices[2].color, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn a_file_without_tangents_gets_a_basis_derived_from_its_uvs() {
        let dir = scratch("tangents");
        let path = write_triangle(&dir, false);
        let model = load(&path, &ImportSettings::default()).unwrap();
        let vertex = &model.primitives[0].mesh.vertices[0];

        // `+U` runs along `+x` in this layout, and `v` is flipped against `y`, so
        // the bitangent points at `-y` while `N x T` points at `+y` — which is
        // the mirrored case, and the handedness has to record it.
        let tangent = Vec3::from_slice(&vertex.tangent[..3]);
        assert!((tangent - Vec3::X).length() < 1e-5, "tangent was {tangent}");
        assert_eq!(vertex.tangent[3], -1.0);
        assert!(
            (tangent.dot(Vec3::from(vertex.normal))).abs() < 1e-5,
            "tangent must be orthogonal to the normal"
        );
    }

    #[test]
    fn the_files_own_tangents_win_over_the_derivation() {
        let dir = scratch("authored-tangents");
        let path = write_triangle(&dir, true);
        let model = load(&path, &ImportSettings::default()).unwrap();

        assert_eq!(
            model.primitives[0].mesh.vertices[0].tangent,
            [1.0, 0.0, 0.0, -1.0]
        );
    }

    #[test]
    fn a_masked_material_keeps_gltfs_cutoff_and_alpha() {
        let dir = scratch("material");
        let path = write_triangle(&dir, false);
        let model = load(&path, &ImportSettings::default()).unwrap();
        let material = &model.materials[0];

        assert_eq!(material.blend, BlendMode::Masked);
        assert_eq!(material.alpha_cutoff, 0.25);
        assert_eq!(material.alpha, 0.75);
        assert_eq!(material.roughness, 0.6);
        // `emissiveFactor` is a [0,1] triple in the file and cd/m² here.
        assert_eq!(
            material.emissive,
            Vec3::new(ImportSettings::default().emissive_luminance, 0.0, 0.0)
        );
        // glTF's default material is appended whether or not it is used, so an
        // out-of-range index is impossible.
        assert_eq!(model.materials.len(), 2);
    }

    #[test]
    fn the_scale_reaches_positions_and_translations_but_is_not_applied_twice() {
        let dir = scratch("scale");
        let path = write_triangle(&dir, false);
        let settings = ImportSettings {
            scale: 0.01,
            ..ImportSettings::default()
        };
        let model = load(&path, &settings).unwrap();

        assert_eq!(
            model.primitives[0].mesh.vertices[1].position,
            [0.01, 0.0, 0.0]
        );
        assert_eq!(
            model.nodes[0].transform.translation,
            Vec3::new(0.02, 0.0, 0.0)
        );
        assert_eq!(model.nodes[0].transform.scale, Vec3::ONE);

        // And the measured box is the triangle at its node: a centimetre wide,
        // two centimetres along `x`.
        let bounds = model.bounds();
        assert!((bounds.min - Vec3A::new(0.02, 0.0, 0.0)).length() < 1e-6);
        assert!((bounds.max - Vec3A::new(0.03, 0.01, 0.0)).length() < 1e-6);
    }

    #[test]
    fn an_embedded_buffer_is_a_named_error_rather_than_an_empty_mesh() {
        let dir = scratch("data-uri");
        let path = dir.join("embedded.gltf");
        std::fs::write(
            &path,
            r#"{"asset": {"version": "2.0"},
                "buffers": [{"uri": "data:application/octet-stream;base64,AAAA", "byteLength": 3}]}"#,
        )
        .unwrap();

        // Matched rather than `unwrap_err`, which would want `Model: Debug` — and
        // a `Debug` on a struct holding every decoded texture is a footgun.
        let Err(error) = load(&path, &ImportSettings::default()) else {
            panic!("a document whose buffer is a data: URI must not load");
        };
        assert!(
            matches!(&error, ModelError::Resource { reason, .. } if reason.contains("data:")),
            "got {error}"
        );
    }
}
