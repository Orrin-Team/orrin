//! The live link's bytes: the preamble, frames, and a mesh frame decoded into
//! the [`CpuMesh`] a slot holds. The layout is written down once, in the
//! [`link`](super) header; this is its one reader and writer.
//!
//! Every function here is total over its input. A count is checked against the
//! bytes that carry it before anything is sized from it, and arithmetic on a
//! count is checked, so no input panics or allocates more than it sent.

use std::io::Read;

use glam::{Vec2, Vec3};

use crate::geom::Aabb;
use crate::gfx::Vertex;
use crate::scene::CpuMesh;
use crate::scene::model::derive_tangents;

/// What a client sends before anything else, followed by its protocol version.
pub const MAGIC: [u8; 8] = *b"ORRNLINK";

/// The protocol this engine speaks. Bumped for any change an older peer would
/// misread; a new frame kind is not one.
pub const VERSION: u32 = 1;

/// The largest frame payload either side accepts, in bytes.
///
/// A mesh costs 32 bytes a vertex and 4 an index, so a million-triangle sculpt
/// with every corner split by a seam — the worst a mesh that size can be — is
/// 3M x 32 + 3M x 4 = 108 MB. Twice that, rounded up, leaves room for an
/// honest mesh and refuses whatever a confused or hostile peer announces past
/// it before a byte is allocated for it.
pub const MAX_FRAME: u32 = 256 << 20;

/// A frame's length and kind, ahead of its payload.
pub const HEADER: usize = 5;

/// Engine to client: the engine's protocol version and assets directory.
pub const HELLO: u8 = 1;
/// Engine to client: an asset path, which may be empty, and a message.
pub const ERROR: u8 = 2;
/// Engine to client: the asset at a path now draws what this client streams.
pub const LIVE: u8 = 3;
/// Client to engine: an asset path and its geometry.
pub const MESH: u8 = 4;
/// Client to engine: an asset path to stop streaming.
pub const RELEASE: u8 = 5;

/// What a peer opened with.
#[derive(Debug, PartialEq)]
pub enum Opening {
    /// The magic, and this version.
    Ours,
    /// The magic, and another version.
    Version(u32),
    /// Anything else, or nothing: a browser, a port scanner.
    Stranger,
}

/// Read a client's preamble. The magic is read alone first, so a peer that is
/// not one is turned away before it is asked for anything more.
pub fn opening(reader: &mut impl Read) -> Opening {
    let mut magic = [0; MAGIC.len()];
    if reader.read_exact(&mut magic).is_err() || magic != MAGIC {
        return Opening::Stranger;
    }
    let mut version = [0; 4];
    if reader.read_exact(&mut version).is_err() {
        return Opening::Stranger;
    }
    match u32::from_le_bytes(version) {
        VERSION => Opening::Ours,
        other => Opening::Version(other),
    }
}

/// Why the next frame could not be read.
#[derive(Debug, PartialEq)]
pub enum ReadError {
    /// The peer went, or the connection was shut down.
    Closed,
    /// It announced a payload over [`MAX_FRAME`]. Nothing was read past the
    /// header, so the stream is out of step and the connection must end.
    TooLarge(u32),
}

/// The next frame's kind, its payload in `payload`. The payload grows as its
/// bytes arrive rather than being sized from the header, so a peer that
/// announces a large frame and sends nothing holds nothing.
pub fn read_frame(reader: &mut impl Read, payload: &mut Vec<u8>) -> Result<u8, ReadError> {
    let mut header = [0; HEADER];
    reader
        .read_exact(&mut header)
        .map_err(|_| ReadError::Closed)?;
    let len = u32::from_le_bytes(header[..4].try_into().unwrap());
    if len > MAX_FRAME {
        return Err(ReadError::TooLarge(len));
    }
    payload.clear();
    match reader.take(u64::from(len)).read_to_end(payload) {
        Ok(read) if read == len as usize => Ok(header[4]),
        _ => Err(ReadError::Closed),
    }
}

/// A frame of `kind` carrying `parts` in order.
pub fn frame(kind: u8, parts: &[&[u8]]) -> Vec<u8> {
    let len: usize = parts.iter().map(|part| part.len()).sum();
    let mut out = Vec::with_capacity(HEADER + len);
    out.extend_from_slice(&(len as u32).to_le_bytes());
    out.push(kind);
    for part in parts {
        out.extend_from_slice(part);
    }
    out
}

pub fn hello(assets: &str) -> Vec<u8> {
    let len = (assets.len() as u32).to_le_bytes();
    frame(HELLO, &[&VERSION.to_le_bytes(), &len, assets.as_bytes()])
}

pub fn error(path: &str, message: &str) -> Vec<u8> {
    let path_len = (path.len() as u32).to_le_bytes();
    let message_len = (message.len() as u32).to_le_bytes();
    frame(
        ERROR,
        &[&path_len, path.as_bytes(), &message_len, message.as_bytes()],
    )
}

pub fn live(path: &str) -> Vec<u8> {
    frame(LIVE, &[&(path.len() as u32).to_le_bytes(), path.as_bytes()])
}

/// A mesh frame's contents, decoded on the connection's thread.
pub struct Decoded {
    pub path: String,
    pub mesh: CpuMesh,
    pub bounds: Aabb,
}

/// Why a frame was refused, for the error frame and the console.
#[derive(Debug, PartialEq)]
pub struct Refusal {
    /// Empty when the frame was refused before its path was read.
    pub path: String,
    pub reason: String,
}

/// A release frame's asset path.
pub fn decode_release(payload: &[u8]) -> Result<String, Refusal> {
    let mut cursor = Cursor(payload);
    let path = cursor.path()?;
    match cursor.0.len() {
        0 => Ok(path),
        extra => Err(Refusal {
            path,
            reason: format!("its release carries {extra} bytes past its asset path"),
        }),
    }
}

/// A mesh frame as the mesh it describes: vertex colour white, as an import
/// without colours has it, and tangents derived from the UVs by the routine
/// the glTF import uses for a primitive that has none — Blender's export
/// carries none either, so the streamed mesh and the saved one shade alike.
pub fn decode_mesh(payload: &[u8]) -> Result<Decoded, Refusal> {
    let mut cursor = Cursor(payload);
    let path = cursor.path()?;
    let refuse = |reason: String| Refusal {
        path: path.clone(),
        reason,
    };
    let (Some(vertices), Some(indices)) = (cursor.u32(), cursor.u32()) else {
        return Err(refuse(
            "it is cut short before its vertex and index counts".to_owned(),
        ));
    };
    let expected = u64::from(vertices) * 32 + u64::from(indices) * 4;
    let carried = cursor.0.len() as u64;
    if expected != carried {
        return Err(refuse(format!(
            "its counts ({vertices} vertices, {indices} indices) describe {expected} bytes of \
             geometry and the frame carries {carried}"
        )));
    }
    if indices == 0 {
        return Err(refuse("it has no triangles".to_owned()));
    }
    if indices % 3 != 0 {
        return Err(refuse(format!(
            "its {indices} indices are not a whole number of triangles"
        )));
    }

    let count = vertices as usize;
    let (positions, rest) = cursor.0.split_at(count * 12);
    let (normals, rest) = rest.split_at(count * 12);
    let (uvs, index_bytes) = rest.split_at(count * 8);
    let vec3 = |c: &[u8]| {
        let v = Vec3::new(float(c, 0), float(c, 4), float(c, 8));
        v.is_finite().then_some(v)
    };
    let positions = finite::<12, _>(positions, "position", &refuse, vec3)?;
    let normals = finite::<12, _>(normals, "normal", &refuse, vec3)?;
    let uvs = finite::<8, _>(uvs, "UV", &refuse, |c| {
        let v = Vec2::new(float(c, 0), float(c, 4));
        v.is_finite().then_some(v)
    })?;
    let index_list: Vec<u32> = index_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect();
    if let Some((at, index)) = index_list
        .iter()
        .enumerate()
        .find(|&(_, &index)| index >= vertices)
    {
        return Err(refuse(format!(
            "index {at} is {index}, past its {vertices} vertices"
        )));
    }

    let tangents = derive_tangents(&positions, &normals, &uvs, &index_list);
    let bounds = Aabb::from_points(positions.iter().copied());
    let vertices = (0..count)
        .map(|at| Vertex {
            position: positions[at].to_array(),
            normal: normals[at].to_array(),
            color: [1.0; 3],
            uv: uvs[at].to_array(),
            tangent: tangents[at].to_array(),
        })
        .collect();
    Ok(Decoded {
        path,
        mesh: CpuMesh::new(vertices, index_list),
        bounds,
    })
}

fn float(bytes: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

/// Each `N`-byte chunk of `bytes` as `read` makes it, which answers `None`
/// for a vector that is not finite: a NaN position poisons the bounds culling
/// reads, and a NaN normal or UV is a NaN pixel the temporal history spreads.
fn finite<const N: usize, V>(
    bytes: &[u8],
    what: &str,
    refuse: &impl Fn(String) -> Refusal,
    read: impl Fn(&[u8]) -> Option<V>,
) -> Result<Vec<V>, Refusal> {
    bytes
        .as_chunks::<N>()
        .0
        .iter()
        .enumerate()
        .map(|(at, chunk)| {
            read(chunk).ok_or_else(|| refuse(format!("{what} {at} is not a finite number")))
        })
        .collect()
}

/// A cursor that answers `None` past the end rather than panicking.
struct Cursor<'a>(&'a [u8]);

impl Cursor<'_> {
    fn u32(&mut self) -> Option<u32> {
        let (bytes, rest) = self.0.split_first_chunk::<4>()?;
        self.0 = rest;
        Some(u32::from_le_bytes(*bytes))
    }

    fn path(&mut self) -> Result<String, Refusal> {
        let refuse = |reason: &str| Refusal {
            path: String::new(),
            reason: reason.to_owned(),
        };
        let len = self
            .u32()
            .ok_or_else(|| refuse("it is cut short before its asset path"))?;
        let Some((bytes, rest)) = self.0.split_at_checked(len as usize) else {
            return Err(refuse("it is cut short before its asset path ends"));
        };
        self.0 = rest;
        match std::str::from_utf8(bytes) {
            Ok("") => Err(refuse("it names no asset")),
            Ok(path) => Ok(path.to_owned()),
            Err(_) => Err(refuse("its asset path is not UTF-8")),
        }
    }
}

/// What a client sends to open a connection speaking `version`.
#[cfg(test)]
pub fn preamble(version: u32) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&version.to_le_bytes());
    out
}

/// A mesh frame, as the add-on sends one.
#[cfg(test)]
pub fn mesh(
    path: &str,
    positions: &[f32],
    normals: &[f32],
    uvs: &[f32],
    indices: &[u32],
) -> Vec<u8> {
    let bytes =
        |floats: &[f32]| -> Vec<u8> { floats.iter().flat_map(|f| f.to_le_bytes()).collect() };
    let index_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    frame(
        MESH,
        &[
            &(path.len() as u32).to_le_bytes(),
            path.as_bytes(),
            &((positions.len() / 3) as u32).to_le_bytes(),
            &(indices.len() as u32).to_le_bytes(),
            &bytes(positions),
            &bytes(normals),
            &bytes(uvs),
            &index_bytes,
        ],
    )
}

#[cfg(test)]
pub fn release(path: &str) -> Vec<u8> {
    frame(
        RELEASE,
        &[&(path.len() as u32).to_le_bytes(), path.as_bytes()],
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A unit quad in the XY plane, facing +Z, UVs with their origin at the
    /// top left as a `.glb` has them.
    pub(crate) fn quad(path: &str, offset: f32) -> Vec<u8> {
        let positions =
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0].map(|v: f32| v + offset);
        let normals = [0.0, 0.0, 1.0].repeat(4);
        let uvs = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        mesh(path, &positions, &normals, &uvs, &[0, 1, 2, 0, 2, 3])
    }

    fn payload(frame: &[u8]) -> &[u8] {
        &frame[HEADER..]
    }

    #[test]
    fn a_mesh_frame_decodes_to_the_mesh_it_describes() {
        let frame = quad("models/tree.blend", 0.0);
        let decoded = decode_mesh(payload(&frame)).unwrap();
        assert_eq!(decoded.path, "models/tree.blend");
        let mesh = &decoded.mesh;
        assert_eq!(mesh.indices, [0, 1, 2, 0, 2, 3]);
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.vertices[2].position, [1.0, 1.0, 0.0]);
        assert_eq!(mesh.vertices[2].normal, [0.0, 0.0, 1.0]);
        assert_eq!(mesh.vertices[2].uv, [1.0, 0.0]);
        assert_eq!(decoded.bounds, mesh.bounds());
        for vertex in &mesh.vertices {
            assert_eq!(vertex.color, [1.0; 3], "white, not black");
            // +U runs along +X, and V down the screen runs against +Y.
            assert_eq!(vertex.tangent, [1.0, 0.0, 0.0, -1.0]);
        }
    }

    /// Each class of bad frame, refused with a reason that says which and
    /// never a panic.
    #[test]
    fn every_malformed_mesh_is_refused_with_its_reason() {
        let good = quad("models/tree.blend", 0.0);
        let good = payload(&good);
        let with = |edit: &dyn Fn(&mut Vec<u8>)| {
            let mut bytes = good.to_vec();
            edit(&mut bytes);
            decode_mesh(&bytes).err().expect("refused")
        };
        // Path length, path, vertex count, index count, then the arrays.
        let counts = 4 + "models/tree.blend".len();
        let geometry = counts + 8;
        let indices = geometry + 4 * 32;
        let nan = f32::NAN.to_le_bytes();
        let cases: [(&str, Refusal); 9] = [
            (
                "short",
                with(&|b| {
                    b.pop();
                }),
            ),
            ("long", with(&|b| b.push(0))),
            (
                "not triangles",
                decode_mesh(payload(&mesh(
                    "a",
                    &[0.0; 9],
                    &[0.0; 9],
                    &[0.0; 6],
                    &[0, 1, 2, 0],
                )))
                .err()
                .unwrap(),
            ),
            (
                "no triangles",
                decode_mesh(payload(&mesh("a", &[0.0; 9], &[0.0; 9], &[0.0; 6], &[])))
                    .err()
                    .unwrap(),
            ),
            (
                "index past",
                with(&|b| b[indices + 8..indices + 12].copy_from_slice(&4u32.to_le_bytes())),
            ),
            (
                "nan position",
                with(&|b| b[geometry + 16..geometry + 20].copy_from_slice(&nan)),
            ),
            (
                "infinite normal",
                with(&|b| {
                    b[geometry + 48..geometry + 52].copy_from_slice(&f32::INFINITY.to_le_bytes())
                }),
            ),
            (
                "no path",
                decode_mesh(payload(&mesh(
                    "",
                    &[0.0; 9],
                    &[0.0; 9],
                    &[0.0; 6],
                    &[0, 1, 2],
                )))
                .err()
                .unwrap(),
            ),
            ("not utf-8", with(&|b| b[4] = 0xff)),
        ];
        let reasons: Vec<(&str, String, String)> = cases
            .into_iter()
            .map(|(case, refusal)| (case, refusal.path, refusal.reason))
            .collect();
        let path = "models/tree.blend".to_owned();
        assert_eq!(
            reasons,
            [
                (
                    "short",
                    path.clone(),
                    "its counts (4 vertices, 6 indices) describe 152 bytes of geometry and the \
                     frame carries 151"
                        .to_owned()
                ),
                (
                    "long",
                    path.clone(),
                    "its counts (4 vertices, 6 indices) describe 152 bytes of geometry and the \
                     frame carries 153"
                        .to_owned()
                ),
                (
                    "not triangles",
                    "a".to_owned(),
                    "its 4 indices are not a whole number of triangles".to_owned()
                ),
                (
                    "no triangles",
                    "a".to_owned(),
                    "it has no triangles".to_owned()
                ),
                (
                    "index past",
                    path.clone(),
                    "index 2 is 4, past its 4 vertices".to_owned()
                ),
                (
                    "nan position",
                    path.clone(),
                    "position 1 is not a finite number".to_owned()
                ),
                (
                    "infinite normal",
                    path.clone(),
                    "normal 0 is not a finite number".to_owned()
                ),
                ("no path", String::new(), "it names no asset".to_owned()),
                (
                    "not utf-8",
                    String::new(),
                    "its asset path is not UTF-8".to_owned()
                ),
            ]
        );
    }

    /// Every prefix of a good frame, and the good frame with each byte
    /// flipped, decodes or is refused; none panics.
    #[test]
    fn no_damage_to_a_frame_panics_the_decoder() {
        let good = quad("models/tree.blend", 0.0);
        let good = payload(&good);
        for len in 0..good.len() {
            assert!(decode_mesh(&good[..len]).is_err(), "a prefix of {len}");
            let _ = decode_release(&good[..len]);
        }
        for at in 0..good.len() {
            for flip in [0x01, 0x80, 0xff] {
                let mut bytes = good.to_vec();
                bytes[at] ^= flip;
                let _ = decode_mesh(&bytes);
                let _ = decode_release(&bytes);
            }
        }
        let mut huge = good.to_vec();
        let counts = 4 + "models/tree.blend".len();
        huge[counts..counts + 8].copy_from_slice(&[0xff; 8]);
        assert!(
            decode_mesh(&huge)
                .err()
                .unwrap()
                .reason
                .starts_with("its counts (4294967295 vertices, 4294967295 indices)")
        );
    }

    /// The cap is checked on the header alone: a reader holding nothing past
    /// it is never asked for more, and nothing is allocated for the payload.
    #[test]
    fn a_length_over_the_cap_is_refused_before_anything_is_allocated() {
        let mut header = (MAX_FRAME + 1).to_le_bytes().to_vec();
        header.push(MESH);
        let mut payload = Vec::new();
        assert_eq!(
            read_frame(&mut header.as_slice(), &mut payload),
            Err(ReadError::TooLarge(MAX_FRAME + 1))
        );
        assert_eq!(payload.capacity(), 0);

        let mut cut = frame(MESH, &[&[1, 2, 3]]);
        cut.pop();
        assert_eq!(
            read_frame(&mut cut.as_slice(), &mut payload),
            Err(ReadError::Closed)
        );
        let whole = frame(42, &[&[1, 2, 3]]);
        assert_eq!(read_frame(&mut whole.as_slice(), &mut payload), Ok(42));
        assert_eq!(payload, [1, 2, 3]);
    }

    #[test]
    fn a_peer_is_ours_only_by_the_magic_then_the_version() {
        assert_eq!(opening(&mut preamble(VERSION).as_slice()), Opening::Ours);
        assert_eq!(
            opening(&mut preamble(VERSION + 1).as_slice()),
            Opening::Version(VERSION + 1)
        );
        let http = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n";
        assert_eq!(opening(&mut http.as_slice()), Opening::Stranger);
        assert_eq!(opening(&mut &MAGIC[..5]), Opening::Stranger);
    }
}
