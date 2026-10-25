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

use glam::{Mat3, Mat4, Quat, Vec3, Vec4};

use super::cache::{Artifact, Importer, TextureArtifact};
use super::sidecar::{AssetKind, MeshSettings, Settings, TextureSettings};
use crate::scene::CpuMesh;

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
    todo!("{kind:?} {} {rel}", source.display())
}

/// Build `kind`'s artifact from what [`read_inputs`] read.
pub fn import(
    kind: AssetKind,
    rel: &str,
    settings: &Settings,
    inputs: &[(String, Vec<u8>)],
) -> Result<Artifact, String> {
    todo!("{kind:?} {rel} {settings:?} {}", inputs.len())
}

#[cfg(test)]
pub(super) mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::asset_db::tests::Dir;

    pub(in crate::asset_db) fn png(width: u32, height: u32, pixel: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
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
        let bytes = png(2, 1, |x, _| if x == 0 { [255, 0, 0, 255] } else { [0, 0, 255, 128] });
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
        let stripes = png(2, 1, |x, _| if x == 0 { [0, 0, 0, 255] } else { [255, 255, 255, 255] });
        let settings = TextureSettings {
            srgb: true,
            max_size: 1,
        };
        let texture = texture(stripes, settings).unwrap();
        assert_eq!((texture.width, texture.height), (1, 1));
        assert!((186..=190).contains(&texture.pixels[0]), "{:?}", texture.pixels);
    }

    #[test]
    fn a_file_that_is_not_an_image_says_which_file() {
        let err = texture(b"definitely not a png".to_vec(), AS_IS).unwrap_err();
        assert!(err.starts_with("rock.png is not a PNG or JPEG this engine can decode:"), "{err}");
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
        assert!(err.starts_with("tree.gltf reads `tri%20angle.bin`, which could not be read:"), "{err}");
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
