//! Compiled artifacts on disk, named by what went into them.
//!
//! A key is a hash of the importer's name and version, the canonical import
//! settings and every byte the importer reads — the source and, for a glTF, the
//! buffers beside it. Same inputs, same file; change any of them and the key
//! moves, which is the whole of the invalidation scheme. Nothing is ever
//! overwritten in place and nothing needs deleting for correctness: the
//! directory is disposable, and removing it costs one reimport.
//!
//! **The hash is xxh3-128, not BLAKE3 or SHA-2.** Today the only writer is this
//! machine's own importer, so the key has to tell inputs apart and nothing
//! more, and xxh3 does that at memory speed with a dependency that has none of
//! its own. A shared team cache (architecture §5.1) makes the key a trust
//! boundary — a crafted collision would let one machine's artifact stand in for
//! another's — and that is when this becomes a cryptographic hash. Because the
//! cache is disposable, the switch costs one reimport and no migration.
//!
//! **The artifact is the engine's own layout, not KTX2.** A texture is its
//! decoded RGBA8 level 0, a mesh its vertices and indices exactly as
//! [`CpuMesh`] holds them, each behind a short header. What makes KTX2 worth its
//! parser is block compression or supercompression, and this engine has neither
//! an encoder nor a transcoder; a KTX2 holding plain RGBA8 is this file with a
//! longer header. Mips are not stored either: the backend builds the chain on
//! the GPU with a blit per level as it uploads, which takes less time than
//! reading a third as many bytes again. The decode is what the cache saves —
//! a PNG or JPEG decode is tens of milliseconds where reading the artifact is a
//! copy. When the renderer grows compressed formats, the artifact grows a
//! variant and [`FORMAT`] moves, which reimports everything once.

use std::path::{Path, PathBuf};

use crate::gfx::Vertex;
use crate::scene::CpuMesh;

/// The artifact layout's version, hashed into every key: bumping it makes every
/// existing artifact unreachable rather than misread.
pub const FORMAT: u32 = 1;

/// The hash of everything an artifact was built from, and its file name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey(pub u128);

impl std::fmt::Display for CacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// One importer: its name, and a version bumped whenever its output for the
/// same input changes.
#[derive(Clone, Copy, Debug)]
pub struct Importer {
    pub name: &'static str,
    pub version: u32,
}

/// The key for `importer` run with `settings` over `inputs`, each a name and
/// its bytes, in the order the importer reads them.
///
/// Every field is length-prefixed, so no two different input lists can hash
/// the same stream.
pub fn key(importer: Importer, settings: &str, inputs: &[(&str, &[u8])]) -> CacheKey {
    todo!("{importer:?} {settings} {}", inputs.len())
}

/// A decoded texture, tightly packed RGBA8 from the top-left.
#[derive(Clone, Debug, PartialEq)]
pub struct TextureArtifact {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// What an importer produces and the cache stores.
#[derive(Clone)]
pub enum Artifact {
    Texture(TextureArtifact),
    Mesh(CpuMesh),
}

impl Artifact {
    pub fn encode(&self) -> Vec<u8> {
        todo!()
    }

    /// Read an artifact back. A file that is short, from another layout or
    /// otherwise not one of ours is an error, never a panic: the directory is
    /// one a person can delete files from by hand.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        todo!("{}", bytes.len())
    }
}

/// The directory artifacts live in, normally `.orrin/cache/` in the project.
#[derive(Clone, Debug)]
pub struct Cache {
    dir: PathBuf,
}

impl Cache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn path(&self, key: CacheKey) -> PathBuf {
        self.dir.join(key.to_string())
    }

    /// `None` when there is no artifact for `key`.
    pub fn load(&self, key: CacheKey) -> Option<Result<Artifact, String>> {
        todo!("{key}")
    }

    /// Write `artifact` under `key`, through a temporary file and a rename, so
    /// a crash or a second process never leaves a half-written artifact under
    /// a name a later run would trust.
    pub fn store(&self, key: CacheKey, artifact: &Artifact) -> Result<(), String> {
        todo!("{key} {}", artifact.encode().len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_db::tests::Dir;

    const PNG: Importer = Importer {
        name: "texture",
        version: 1,
    };

    #[test]
    fn a_key_is_a_function_of_every_input() {
        let base = key(PNG, "srgb=true", &[("rock.png", b"abc")]);
        assert_eq!(base, key(PNG, "srgb=true", &[("rock.png", b"abc")]));

        let others = [
            key(PNG, "srgb=true", &[("rock.png", b"abd")]),
            key(PNG, "srgb=false", &[("rock.png", b"abc")]),
            key(
                Importer {
                    version: 2,
                    ..PNG
                },
                "srgb=true",
                &[("rock.png", b"abc")],
            ),
            key(
                Importer {
                    name: "mesh",
                    ..PNG
                },
                "srgb=true",
                &[("rock.png", b"abc")],
            ),
            key(PNG, "srgb=true", &[("rock.png", b"abc"), ("rock.bin", b"")]),
        ];
        for other in others {
            assert_ne!(other, base);
        }
    }

    /// Without the length prefixes these two would hash one stream.
    #[test]
    fn moving_bytes_between_inputs_moves_the_key() {
        assert_ne!(
            key(PNG, "", &[("a", b"bc"), ("d", b"")]),
            key(PNG, "", &[("a", b"b"), ("cd", b"")]),
        );
    }

    #[test]
    fn a_key_is_named_by_32_hex_digits() {
        assert_eq!(CacheKey(0xab).to_string(), format!("{:0>32}", "ab"));
    }

    fn triangle() -> CpuMesh {
        let vertex = |x: f32| Vertex {
            position: [x, 1.0, -0.5],
            normal: [0.0, 0.0, 1.0],
            color: [1.0, 0.25, 0.125],
            uv: [x, 0.75],
            tangent: [1.0, 0.0, 0.0, -1.0],
        };
        CpuMesh::new(vec![vertex(0.0), vertex(1.0), vertex(f32::MIN_POSITIVE)], vec![0, 1, 2])
    }

    #[test]
    fn artifacts_round_trip_bit_for_bit() {
        let texture = TextureArtifact {
            width: 2,
            height: 1,
            pixels: vec![1, 2, 3, 4, 250, 251, 252, 253],
        };
        let Artifact::Texture(back) = Artifact::decode(&Artifact::Texture(texture.clone()).encode())
            .unwrap()
        else {
            panic!("a texture decodes as a texture");
        };
        assert_eq!(back, texture);

        let mesh = triangle();
        let Artifact::Mesh(back) = Artifact::decode(&Artifact::Mesh(mesh.clone()).encode()).unwrap()
        else {
            panic!("a mesh decodes as a mesh");
        };
        assert_eq!(back.indices, mesh.indices);
        let bits = |m: &CpuMesh| -> Vec<u32> {
            m.vertices
                .iter()
                .flat_map(|v| {
                    v.position
                        .into_iter()
                        .chain(v.normal)
                        .chain(v.color)
                        .chain(v.uv)
                        .chain(v.tangent)
                })
                .map(f32::to_bits)
                .collect()
        };
        assert_eq!(bits(&back), bits(&mesh));
    }

    #[test]
    fn a_damaged_artifact_is_an_error_rather_than_a_panic() {
        let bytes = Artifact::Mesh(triangle()).encode();
        for cut in [0, 3, 8, bytes.len() - 1] {
            assert!(Artifact::decode(&bytes[..cut]).is_err(), "cut at {cut}");
        }
        let mut wrong = bytes.clone();
        wrong[4] = 99;
        assert_eq!(
            Artifact::decode(&wrong).unwrap_err(),
            "written by artifact format 99, and this engine reads format 1"
        );
        assert_eq!(
            Artifact::decode(b"PNG.............").unwrap_err(),
            "not an Orrin cache artifact"
        );
    }

    #[test]
    fn the_cache_stores_and_finds_by_key() {
        let dir = Dir::new("cache-store");
        let cache = Cache::new(dir.path().join("cache"));
        let key = CacheKey(42);
        assert!(cache.load(key).is_none());

        cache.store(key, &Artifact::Mesh(triangle())).unwrap();
        let Some(Ok(Artifact::Mesh(mesh))) = cache.load(key) else {
            panic!("the stored mesh is found");
        };
        assert_eq!(mesh.indices, vec![0, 1, 2]);
        let names: Vec<String> = std::fs::read_dir(cache.dir())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec![key.to_string()], "no temporary file is left");
    }
}
