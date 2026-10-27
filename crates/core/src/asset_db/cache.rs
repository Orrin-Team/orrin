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
//! **A texture is a KTX2 file; a mesh is the engine's own layout.** A texture
//! artifact holds its whole mip chain in the format the GPU samples — BC7 or
//! BC5 where the device has block compression, RGBA8 where it does not (see
//! `asset_db::texture`) — in the standard container, so it opens in KTX tools
//! (see `asset_db::ktx`). A mesh is its vertices and indices exactly as
//! [`CpuMesh`] holds them, behind a short header; no standard container is
//! worth a parser for that. What the cache saves is the work: a decode, a mip
//! chain and a compression are tens of milliseconds to seconds where reading
//! the artifact is a copy. When a layout changes, [`FORMAT`] moves, which
//! reimports everything once.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use xxhash_rust::xxh3::Xxh3;

use super::ktx;
use crate::gfx::Vertex;
use crate::scene::CpuMesh;

/// The artifact layout's version, hashed into every key: bumping it makes every
/// existing artifact unreachable rather than misread.
pub const FORMAT: u32 = 2;

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
    let mut hasher = Xxh3::new();
    let mut field = |bytes: &[u8]| {
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    };
    field(&FORMAT.to_le_bytes());
    field(importer.name.as_bytes());
    field(&importer.version.to_le_bytes());
    field(settings.as_bytes());
    field(&(inputs.len() as u64).to_le_bytes());
    for (name, bytes) in inputs {
        field(name.as_bytes());
        field(bytes);
    }
    CacheKey(hasher.digest128())
}

/// What every artifact file starts with.
const MAGIC: &[u8; 4] = b"ORRA";
const MESH: u8 = 2;
/// Magic, format, kind and three bytes of padding.
const HEADER: usize = 12;
/// `f32`s per vertex: position, normal, colour, uv, tangent.
const VERTEX_FLOATS: usize = 15;

/// A texture artifact is exactly what the backend uploads.
pub type TextureArtifact = crate::gfx::TextureLevels;

/// What an importer produces and the cache stores.
#[derive(Clone)]
pub enum Artifact {
    Texture(TextureArtifact),
    Mesh(CpuMesh),
}

impl std::fmt::Debug for Artifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Texture(t) => write!(
                f,
                "Texture({}x{} {:?}, {} levels)",
                t.width,
                t.height,
                t.format,
                t.levels.len()
            ),
            Self::Mesh(m) => write!(
                f,
                "Mesh({} vertices, {} indices)",
                m.vertices.len(),
                m.indices.len()
            ),
        }
    }
}

impl Artifact {
    pub fn encode(&self) -> Vec<u8> {
        let mesh = match self {
            Self::Texture(texture) => return ktx::write(texture),
            Self::Mesh(mesh) => mesh,
        };
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT.to_le_bytes());
        out.extend_from_slice(&[MESH, 0, 0, 0]);
        out.extend_from_slice(&(mesh.vertices.len() as u32).to_le_bytes());
        out.extend_from_slice(&(mesh.indices.len() as u32).to_le_bytes());
        out.reserve(mesh.vertices.len() * VERTEX_FLOATS * 4 + mesh.indices.len() * 4);
        for vertex in &mesh.vertices {
            for float in vertex_floats(vertex) {
                out.extend_from_slice(&float.to_le_bytes());
            }
        }
        for index in &mesh.indices {
            out.extend_from_slice(&index.to_le_bytes());
        }
        out
    }

    /// Read an artifact back. A file that is short, from another layout or
    /// otherwise not one of ours is an error, never a panic: the directory is
    /// one a person can delete files from by hand.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.starts_with(&ktx::IDENTIFIER) {
            return ktx::read(bytes).map(Self::Texture);
        }
        if bytes.len() < 4 || &bytes[..4] != MAGIC {
            return Err("not an Orrin cache artifact".to_owned());
        }
        let mut reader = Reader { bytes, at: 4 };
        let format = reader.u32()?;
        if format != FORMAT {
            return Err(format!(
                "written by artifact format {format}, and this engine reads format {FORMAT}"
            ));
        }
        let kind = reader.take(4)?[0];
        debug_assert_eq!(reader.at, HEADER);
        let artifact = match kind {
            MESH => {
                let vertex_count = reader.u32()? as usize;
                let index_count = reader.u32()? as usize;
                let floats = reader.take(
                    vertex_count
                        .checked_mul(VERTEX_FLOATS * 4)
                        .ok_or_else(cut_short)?,
                )?;
                let vertices = floats
                    .as_chunks::<{ VERTEX_FLOATS * 4 }>()
                    .0
                    .iter()
                    .map(|chunk| {
                        let f = |i: usize| {
                            f32::from_le_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap())
                        };
                        Vertex {
                            position: [f(0), f(1), f(2)],
                            normal: [f(3), f(4), f(5)],
                            color: [f(6), f(7), f(8)],
                            uv: [f(9), f(10)],
                            tangent: [f(11), f(12), f(13), f(14)],
                        }
                    })
                    .collect();
                let indices = reader
                    .take(index_count.checked_mul(4).ok_or_else(cut_short)?)?
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| u32::from_le_bytes(*chunk))
                    .collect();
                Self::Mesh(CpuMesh::new(vertices, indices))
            }
            other => return Err(format!("holds an artifact of unknown kind {other}")),
        };
        if reader.at != bytes.len() {
            return Err("has bytes past the end of its artifact".to_owned());
        }
        Ok(artifact)
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
        match std::fs::read(self.path(key)) {
            Ok(bytes) => Some(Artifact::decode(&bytes)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => Some(Err(error.to_string())),
        }
    }

    /// Write `artifact` under `key`, through a temporary file and a rename, so
    /// a crash or a second process never leaves a half-written artifact under
    /// a name a later run would trust.
    pub fn store(&self, key: CacheKey, artifact: &Artifact) -> Result<(), String> {
        static TEMPORARY: AtomicU32 = AtomicU32::new(0);
        let failed = |error: std::io::Error| {
            format!(
                "could not write to the asset cache {}: {error}",
                self.dir.display()
            )
        };
        std::fs::create_dir_all(&self.dir).map_err(failed)?;
        let temporary = self.dir.join(format!(
            ".{key}.{}-{}",
            std::process::id(),
            TEMPORARY.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&temporary, artifact.encode()).map_err(failed)?;
        std::fs::rename(&temporary, self.path(key)).map_err(|error| {
            let _ = std::fs::remove_file(&temporary);
            failed(error)
        })
    }
}

fn vertex_floats(vertex: &Vertex) -> impl Iterator<Item = f32> + '_ {
    vertex
        .position
        .into_iter()
        .chain(vertex.normal)
        .chain(vertex.color)
        .chain(vertex.uv)
        .chain(vertex.tangent)
}

fn cut_short() -> String {
    "is cut short".to_owned()
}

/// A cursor over an artifact that fails rather than panics past the end.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(len).ok_or_else(cut_short)?;
        let slice = self.bytes.get(self.at..end).ok_or_else(cut_short)?;
        self.at = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset_db::tests::Dir;
    use crate::gfx::TexelFormat;

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
                Importer { version: 2, ..PNG },
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
        CpuMesh::new(
            vec![vertex(0.0), vertex(1.0), vertex(f32::MIN_POSITIVE)],
            vec![0, 1, 2],
        )
    }

    #[test]
    fn artifacts_round_trip_bit_for_bit() {
        let texture = TextureArtifact {
            format: TexelFormat::Rgba8Srgb,
            width: 2,
            height: 1,
            levels: vec![vec![1, 2, 3, 4, 250, 251, 252, 253], vec![9, 8, 7, 6]],
        };
        let Artifact::Texture(back) =
            Artifact::decode(&Artifact::Texture(texture.clone()).encode()).unwrap()
        else {
            panic!("a texture decodes as a texture");
        };
        assert_eq!(back, texture);

        let mesh = triangle();
        let Artifact::Mesh(back) =
            Artifact::decode(&Artifact::Mesh(mesh.clone()).encode()).unwrap()
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
            "written by artifact format 99, and this engine reads format 2"
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
