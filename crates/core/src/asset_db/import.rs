//! Source bytes to artifacts: PNG and JPEG textures, glTF 2.0 meshes.
//!
//! Pure functions of their inputs, which [`read_inputs`] reads in one place so
//! that the bytes the cache key hashes are the bytes the importer is given —
//! never a second read that could see a file mid-save.
//!
//! **A glTF file is one mesh.** Every triangle primitive its default scene
//! draws is merged into one [`CpuMesh`], with node transforms applied, so the
//! asset an entity names looks the way the file does in any viewer. Materials,
//! textures, cameras and the node tree itself are not part of a mesh asset; the
//! importer that makes a file into entities is `scene::model`, and this is the
//! narrower thing a mesh slot holds. A node whose transform mirrors flips its
//! triangles' winding back, as glTF requires, so a mirrored half of a model is
//! not drawn inside out.

use std::path::Path;

use glam::{Mat3, Mat4, Vec3, Vec4};

use super::cache::{Artifact, Importer, TextureArtifact};
use super::sidecar::{AssetKind, MeshSettings, Settings, TextureSettings};
use crate::scene::CpuMesh;
use crate::scene::model::{import_primitive, percent_decode};

pub const TEXTURE: Importer = Importer {
    name: "texture",
    version: 1,
};

pub const MESH: Importer = Importer {
    name: "gltf-mesh",
    version: 1,
};

/// The importer for `kind`, or `None` for a kind read as it is.
pub fn importer(kind: AssetKind) -> Option<Importer> {
    match kind {
        AssetKind::Texture => Some(TEXTURE),
        AssetKind::Mesh => Some(MESH),
        AssetKind::Prefab | AssetKind::Folder => None,
    }
}

/// Every file importing `source` reads, source first, as `(name, bytes)`. For
/// a `.gltf`, its external buffers follow, named by their URIs.
pub fn read_inputs(
    kind: AssetKind,
    source: &Path,
    rel: &str,
) -> Result<Vec<(String, Vec<u8>)>, String> {
    let bytes =
        std::fs::read(source).map_err(|error| format!("{rel} could not be read: {error}"))?;
    let mut inputs = Vec::new();
    if kind == AssetKind::Mesh {
        let document = parse_gltf(rel, &bytes)?.document;
        let base = source.parent().unwrap_or(Path::new("."));
        for buffer in document.buffers() {
            let gltf::buffer::Source::Uri(uri) = buffer.source() else {
                continue;
            };
            let name = buffer_name(rel, uri)?;
            let read = std::fs::read(base.join(&name)).map_err(|error| {
                format!("{rel} reads `{uri}`, which could not be read: {error}")
            })?;
            inputs.push((name, read));
        }
    }
    inputs.insert(0, (rel.to_owned(), bytes));
    Ok(inputs)
}

/// Build `kind`'s artifact from what [`read_inputs`] read.
pub fn import(
    kind: AssetKind,
    rel: &str,
    settings: &Settings,
    inputs: &[(String, Vec<u8>)],
) -> Result<Artifact, String> {
    match (kind, settings) {
        (AssetKind::Texture, Settings::Texture(settings)) => {
            texture(rel, settings, &inputs[0].1).map(Artifact::Texture)
        }
        (AssetKind::Mesh, Settings::Mesh(settings)) => {
            mesh(rel, settings, inputs).map(Artifact::Mesh)
        }
        _ => Err(format!(
            "{rel} is a {}, which is not imported",
            kind.label()
        )),
    }
}

fn texture(rel: &str, settings: &TextureSettings, bytes: &[u8]) -> Result<TextureArtifact, String> {
    let decoded = image::load_from_memory(bytes)
        .map_err(|error| format!("{rel} is not a PNG or JPEG this engine can decode: {error}"))?;
    let mut rgba = decoded.to_rgba8();
    let (width, height) = rgba.dimensions();
    let longest = width.max(height);
    if settings.max_size > 0 && longest > settings.max_size {
        let fit = |side: u32| {
            ((u64::from(side) * u64::from(settings.max_size)) / u64::from(longest)).max(1) as u32
        };
        rgba = shrink(&rgba, fit(width), fit(height), settings.srgb);
    }
    let (width, height) = rgba.dimensions();
    Ok(TextureArtifact {
        width,
        height,
        pixels: rgba.into_raw(),
    })
}

/// Resize in linear light: a colour texture is decoded from sRGB first, so
/// averaging texels averages light rather than encoded values. Alpha and a
/// data texture's channels are linear already.
fn shrink(image: &image::RgbaImage, width: u32, height: u32, srgb: bool) -> image::RgbaImage {
    let filter = image::imageops::FilterType::Triangle;
    if !srgb {
        return image::imageops::resize(image, width, height, filter);
    }
    let linear = image::Rgba32FImage::from_fn(image.width(), image.height(), |x, y| {
        let [r, g, b, a] = image.get_pixel(x, y).0;
        image::Rgba([
            to_linear(r),
            to_linear(g),
            to_linear(b),
            f32::from(a) / 255.0,
        ])
    });
    let small = image::imageops::resize(&linear, width, height, filter);
    image::RgbaImage::from_fn(width, height, |x, y| {
        let [r, g, b, a] = small.get_pixel(x, y).0;
        image::Rgba([
            to_srgb(r),
            to_srgb(g),
            to_srgb(b),
            (a * 255.0).round().clamp(0.0, 255.0) as u8,
        ])
    })
}

fn to_linear(code: u8) -> f32 {
    let c = f32::from(code) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn to_srgb(linear: f32) -> u8 {
    let l = linear.clamp(0.0, 1.0);
    let c = if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (c * 255.0).round() as u8
}

fn parse_gltf(rel: &str, bytes: &[u8]) -> Result<gltf::Gltf, String> {
    gltf::Gltf::from_slice(bytes).map_err(|error| format!("{rel} is not a glTF 2.0 file: {error}"))
}

/// The file a buffer URI names, relative to the glTF's directory.
fn buffer_name(rel: &str, uri: &str) -> Result<String, String> {
    if uri.starts_with("data:") {
        return Err(format!(
            "{rel} embeds a buffer as a `data:` URI, which is not supported; export it as a \
             .glb, or as a .gltf beside its .bin"
        ));
    }
    Ok(percent_decode(uri))
}

fn mesh(
    rel: &str,
    settings: &MeshSettings,
    inputs: &[(String, Vec<u8>)],
) -> Result<CpuMesh, String> {
    let gltf::Gltf { document, mut blob } = parse_gltf(rel, &inputs[0].1)?;
    let buffers = document
        .buffers()
        .map(|buffer| match buffer.source() {
            gltf::buffer::Source::Bin => blob
                .take()
                .ok_or_else(|| format!("{rel} names the GLB binary chunk, but the file has none")),
            gltf::buffer::Source::Uri(uri) => {
                let name = buffer_name(rel, uri)?;
                inputs[1..]
                    .iter()
                    .find(|(input, _)| *input == name)
                    .map(|(_, bytes)| bytes.clone())
                    .ok_or_else(|| format!("{rel} reads `{uri}`, which was not read with it"))
            }
        })
        .collect::<Result<Vec<_>, String>>()?;

    let roots: Vec<gltf::Node> = match document
        .default_scene()
        .or_else(|| document.scenes().next())
    {
        Some(scene) => scene.nodes().collect(),
        None => Vec::new(),
    };
    let mut merged = CpuMesh::default();
    let scale = Mat4::from_scale(Vec3::splat(settings.scale));
    let mut stack: Vec<(gltf::Node, Mat4)> = roots.into_iter().map(|node| (node, scale)).collect();
    while let Some((node, parent)) = stack.pop() {
        let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                if primitive.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }
                let part = import_primitive(&primitive, &buffers, 1.0);
                append(&mut merged, part, world);
            }
        }
        stack.extend(node.children().map(|child| (child, world)));
    }
    if merged.indices.is_empty() {
        return Err(format!(
            "{rel} draws no triangles: no node in its scene has a triangle mesh"
        ));
    }
    Ok(merged)
}

/// Add `part`, drawn at `world`, to `into`.
fn append(into: &mut CpuMesh, mut part: CpuMesh, world: Mat4) {
    let linear = Mat3::from_mat4(world);
    let normals = linear.inverse().transpose();
    let mirrored = linear.determinant() < 0.0;
    for vertex in &mut part.vertices {
        vertex.position = world
            .transform_point3(Vec3::from(vertex.position))
            .to_array();
        vertex.normal = (normals * Vec3::from(vertex.normal))
            .normalize_or(Vec3::Y)
            .to_array();
        let tangent = Vec4::from(vertex.tangent);
        let handedness = if mirrored { -tangent.w } else { tangent.w };
        vertex.tangent = (linear * tangent.truncate())
            .normalize_or(Vec3::X)
            .extend(handedness)
            .to_array();
    }
    if mirrored {
        for triangle in part.indices.as_chunks_mut::<3>().0 {
            triangle.swap(1, 2);
        }
    }
    let base = into.vertices.len() as u32;
    into.vertices.extend(part.vertices);
    into.indices
        .extend(part.indices.into_iter().map(|index| index + base));
}

#[cfg(test)]
pub(super) mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::asset_db::tests::Dir;

    pub(in crate::asset_db) fn png(
        width: u32,
        height: u32,
        pixel: impl Fn(u32, u32) -> [u8; 4],
    ) -> Vec<u8> {
        let image = image::RgbaImage::from_fn(width, height, |x, y| image::Rgba(pixel(x, y)));
        let mut bytes = Cursor::new(Vec::new());
        image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        bytes.into_inner()
    }

    fn texture(bytes: Vec<u8>, settings: TextureSettings) -> Result<TextureArtifact, String> {
        let inputs = [("rock.png".to_owned(), bytes)];
        match import(
            AssetKind::Texture,
            "rock.png",
            &Settings::Texture(settings),
            &inputs,
        )? {
            Artifact::Texture(texture) => Ok(texture),
            Artifact::Mesh(_) => panic!("a texture imports as a texture"),
        }
    }

    const AS_IS: TextureSettings = TextureSettings {
        srgb: true,
        max_size: 0,
    };

    #[test]
    fn a_png_imports_as_rgba8_from_the_top_left() {
        let bytes = png(2, 1, |x, _| {
            if x == 0 {
                [255, 0, 0, 255]
            } else {
                [0, 0, 255, 128]
            }
        });
        let texture = texture(bytes, AS_IS).unwrap();
        assert_eq!((texture.width, texture.height), (2, 1));
        assert_eq!(texture.pixels, vec![255, 0, 0, 255, 0, 0, 255, 128]);
    }

    #[test]
    fn a_texture_over_max_size_shrinks_to_fit_and_keeps_its_aspect() {
        let bytes = png(8, 4, |_, _| [200, 100, 50, 255]);
        let settings = TextureSettings {
            max_size: 4,
            ..AS_IS
        };
        let texture = texture(bytes.clone(), settings).unwrap();
        assert_eq!((texture.width, texture.height), (4, 2));
        // A flat colour stays that colour through the filter, in either space.
        assert!(texture.pixels.chunks(4).all(|p| p == [200, 100, 50, 255]));

        let linear = texture_settings_linear(bytes);
        assert_eq!((linear.width, linear.height), (4, 2));
    }

    fn texture_settings_linear(bytes: Vec<u8>) -> TextureArtifact {
        texture(
            bytes,
            TextureSettings {
                srgb: false,
                max_size: 4,
            },
        )
        .unwrap()
    }

    /// Shrinking a colour texture averages light, not encoded values: half
    /// black and half white is the sRGB code for 50% luminance, 188, where an
    /// average of the codes would give a visibly darker 128.
    #[test]
    fn a_colour_texture_is_filtered_in_linear_light() {
        let stripes = png(2, 1, |x, _| {
            if x == 0 {
                [0, 0, 0, 255]
            } else {
                [255, 255, 255, 255]
            }
        });
        let settings = TextureSettings {
            srgb: true,
            max_size: 1,
        };
        let texture = texture(stripes, settings).unwrap();
        assert_eq!((texture.width, texture.height), (1, 1));
        assert!(
            (186..=190).contains(&texture.pixels[0]),
            "{:?}",
            texture.pixels
        );
    }

    #[test]
    fn a_file_that_is_not_an_image_says_which_file() {
        let err = texture(b"definitely not a png".to_vec(), AS_IS).unwrap_err();
        assert!(
            err.starts_with("rock.png is not a PNG or JPEG this engine can decode:"),
            "{err}"
        );
    }

    /// A triangle in an external buffer: positions, `u16` indices, then UVs.
    pub(in crate::asset_db) fn triangle_bin() -> Vec<u8> {
        let mut bin = Vec::new();
        for p in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for c in p {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        for i in [0u16, 1, 2] {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        bin.extend_from_slice(&[0, 0]);
        for uv in [[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]] {
            for c in uv {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        bin
    }

    /// The document for [`triangle_bin`], drawn at one node with `node` as its
    /// transform properties.
    pub(in crate::asset_db) fn triangle_gltf(node: &str) -> String {
        format!(
            r#"{{
  "asset": {{"version": "2.0"}},
  "scene": 0,
  "scenes": [{{"nodes": [0]}}],
  "nodes": [{{"mesh": 0 {node}}}],
  "meshes": [{{"primitives": [{{"attributes": {{"POSITION": 0, "TEXCOORD_0": 2}}, "indices": 1}}]}}],
  "buffers": [{{"uri": "tri%20angle.bin", "byteLength": 68}}],
  "bufferViews": [
    {{"buffer": 0, "byteOffset": 0, "byteLength": 36}},
    {{"buffer": 0, "byteOffset": 36, "byteLength": 6}},
    {{"buffer": 0, "byteOffset": 44, "byteLength": 24}}
  ],
  "accessors": [
    {{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]}},
    {{"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}},
    {{"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"}}
  ]
}}"#
        )
    }

    fn mesh(dir: &Dir, node: &str, scale: f32) -> Result<CpuMesh, String> {
        let source = dir.write("tree.gltf", triangle_gltf(node).as_bytes());
        dir.write("tri angle.bin", &triangle_bin());
        let inputs = read_inputs(AssetKind::Mesh, &source, "tree.gltf")?;
        match import(
            AssetKind::Mesh,
            "tree.gltf",
            &Settings::Mesh(MeshSettings { scale }),
            &inputs,
        )? {
            Artifact::Mesh(mesh) => Ok(mesh),
            Artifact::Texture(_) => panic!("a mesh imports as a mesh"),
        }
    }

    fn positions(mesh: &CpuMesh) -> Vec<[f32; 3]> {
        mesh.vertices.iter().map(|v| v.position).collect()
    }

    #[test]
    fn a_gltf_reads_its_buffers_as_inputs_after_itself() {
        let dir = Dir::new("import-inputs");
        let source = dir.write("tree.gltf", triangle_gltf("").as_bytes());
        dir.write("tri angle.bin", &triangle_bin());
        let inputs = read_inputs(AssetKind::Mesh, &source, "tree.gltf").unwrap();
        let names: Vec<&str> = inputs.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["tree.gltf", "tri angle.bin"]);
        assert_eq!(inputs[1].1, triangle_bin());
    }

    #[test]
    fn a_gltf_mesh_is_imported_where_its_node_puts_it() {
        let dir = Dir::new("import-node");
        let mesh = mesh(&dir, r#", "translation": [1, 0, 0]"#, 1.0).unwrap();
        assert_eq!(
            positions(&mesh),
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
        );
        assert_eq!(mesh.indices, [0, 1, 2]);
        let uvs: Vec<[f32; 2]> = mesh.vertices.iter().map(|v| v.uv).collect();
        assert_eq!(uvs, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        for vertex in &mesh.vertices {
            assert_eq!(vertex.normal, [0.0, 0.0, 1.0]);
            assert_eq!(vertex.color, [1.0, 1.0, 1.0]);
        }
    }

    #[test]
    fn the_scale_setting_scales_positions_and_translations_once() {
        let dir = Dir::new("import-scale");
        let mesh = mesh(&dir, r#", "translation": [1, 0, 0]"#, 2.0).unwrap();
        assert_eq!(
            positions(&mesh),
            [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 2.0, 0.0]]
        );
    }

    /// Mirrored, the triangle would face away from its own normal and be
    /// back-face culled; reversing the winding keeps it facing +Z.
    #[test]
    fn a_mirroring_node_keeps_its_triangles_facing_out() {
        let dir = Dir::new("import-mirror");
        let mesh = mesh(&dir, r#", "scale": [-1, 1, 1]"#, 1.0).unwrap();
        assert_eq!(mesh.indices, [0, 2, 1]);
        let p: Vec<Vec3> = positions(&mesh).into_iter().map(Vec3::from).collect();
        let [a, b, c] = [0, 2, 1].map(|i| p[i]);
        assert!((b - a).cross(c - a).z > 0.0);
        assert_eq!(mesh.vertices[0].normal, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_gltf_missing_its_buffer_names_the_uri() {
        let dir = Dir::new("import-missing-bin");
        let source = dir.write("tree.gltf", triangle_gltf("").as_bytes());
        let err = read_inputs(AssetKind::Mesh, &source, "tree.gltf").unwrap_err();
        assert!(
            err.starts_with("tree.gltf reads `tri%20angle.bin`, which could not be read:"),
            "{err}"
        );
    }

    #[test]
    fn a_gltf_with_nothing_to_draw_is_refused() {
        let empty = r#"{"asset": {"version": "2.0"}, "scenes": [{"nodes": []}], "scene": 0}"#;
        let inputs = [("empty.gltf".to_owned(), empty.as_bytes().to_vec())];
        let err = import(
            AssetKind::Mesh,
            "empty.gltf",
            &Settings::Mesh(MeshSettings { scale: 1.0 }),
            &inputs,
        )
        .unwrap_err();
        assert_eq!(
            err,
            "empty.gltf draws no triangles: no node in its scene has a triangle mesh"
        );
    }
}
