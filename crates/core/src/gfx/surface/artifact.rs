use std::fmt;

use xxhash_rust::xxh3::xxh3_64;

use super::params::{ColorSpace, DefaultTexture, Param, ParamKind, ParamTable, Slot};
use crate::gfx::shaders::{
    CompiledBindings, CompiledShader, Resource, ResourceKind, SOURCES_HASH, ShaderId, Stage,
};

/// A compiled surface shader: what an exported game loads in place of the
/// `.oshader` it came from.
#[derive(Clone, PartialEq, Debug)]
pub struct SurfaceArtifact {
    /// The [`SOURCES_HASH`] of the engine that compiled it.
    pub sources_hash: u64,
    pub params: ParamTable,
    /// In manifest order.
    pub variants: Vec<CompiledVariant>,
}

/// One pass shader with the surface function stitched in.
#[derive(Clone, PartialEq, Debug)]
pub struct CompiledVariant {
    /// The manifest name of the built-in shader this one replaces.
    pub key: String,
    pub shader: CompiledShader,
}

/// Why bytes are not a surface artifact this engine can read.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DecodeError {
    Magic,
    Version(u32),
    /// The payload's xxh3 does not match the one recorded beside it.
    Integrity,
    /// The input ended inside the named field.
    Truncated(&'static str),
    Malformed(String),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::Magic => f.write_str("not a surface shader artifact: wrong magic"),
            DecodeError::Version(found) => write!(
                f,
                "surface shader artifact format version {found}, where this engine reads \
                 version {VERSION}; recompile the shader"
            ),
            DecodeError::Integrity => f.write_str(
                "the surface shader artifact is corrupt: its contents do not match its \
                 integrity hash",
            ),
            DecodeError::Truncated(field) => {
                write!(f, "the surface shader artifact ends inside {field}")
            }
            DecodeError::Malformed(what) => {
                write!(f, "the surface shader artifact is malformed: {what}")
            }
        }
    }
}

const MAGIC: [u8; 4] = *b"OSRF";
const VERSION: u32 = 1;

impl CompiledVariant {
    /// The built-in shader this variant stands in for, if this engine has it.
    pub fn builtin(&self) -> Option<ShaderId> {
        ShaderId::ALL
            .iter()
            .copied()
            .find(|id| id.blob().name == self.key)
    }
}

impl SurfaceArtifact {
    /// What an encoded artifact starts with.
    pub const MAGIC: [u8; 4] = MAGIC;
    /// The encoding's version: an artifact of any other is refused, so it is
    /// part of what a cached artifact is keyed by.
    pub const FORMAT_VERSION: u32 = VERSION;

    /// Whether it was compiled against this engine's shader sources. One that
    /// was not binds against a layout this build may not have.
    pub fn matches_this_build(&self) -> bool {
        self.sources_hash == SOURCES_HASH
    }

    /// Magic, format version, the payload's xxh3-64, then the payload: every
    /// integer little-endian, every string and byte run length-prefixed.
    pub fn encode(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        put_u64(&mut payload, self.sources_hash);
        put_u32(&mut payload, self.params.word_count);
        put_u32(&mut payload, self.params.params.len() as u32);
        for param in &self.params.params {
            put_bytes(&mut payload, param.name.as_bytes());
            match param.kind {
                ParamKind::Float { default, range } => {
                    payload.push(0);
                    put_f32s(&mut payload, &[default]);
                    match range {
                        Some(range) => {
                            payload.push(1);
                            put_f32s(&mut payload, &range);
                        }
                        None => payload.push(0),
                    }
                }
                ParamKind::Vec2 { default } => {
                    payload.push(1);
                    put_f32s(&mut payload, &default);
                }
                ParamKind::Vec3 { default } => {
                    payload.push(2);
                    put_f32s(&mut payload, &default);
                }
                ParamKind::Vec4 { default } => {
                    payload.push(3);
                    put_f32s(&mut payload, &default);
                }
                ParamKind::Color { default } => {
                    payload.push(4);
                    put_f32s(&mut payload, &default);
                }
                ParamKind::Texture {
                    default,
                    color_space,
                } => {
                    payload.push(5);
                    payload.push(match default {
                        DefaultTexture::White => 0,
                        DefaultTexture::Black => 1,
                        DefaultTexture::Normal => 2,
                    });
                    payload.push(match color_space {
                        ColorSpace::Srgb => 0,
                        ColorSpace::Linear => 1,
                    });
                }
            }
            put_u32(&mut payload, param.slot.word);
            put_u32(&mut payload, param.slot.component);
        }
        put_u32(&mut payload, self.variants.len() as u32);
        for variant in &self.variants {
            put_bytes(&mut payload, variant.key.as_bytes());
            put_shader(&mut payload, &variant.shader);
        }

        let mut bytes = Vec::with_capacity(16 + payload.len());
        bytes.extend_from_slice(&MAGIC);
        put_u32(&mut bytes, VERSION);
        put_u64(&mut bytes, xxh3_64(&payload));
        bytes.extend_from_slice(&payload);
        bytes
    }

    /// The artifact `bytes` encode. The magic and version are compared, then
    /// the payload is hashed whole, before any field of it is read.
    pub fn decode(bytes: &[u8]) -> Result<SurfaceArtifact, DecodeError> {
        let mut header = Reader { bytes };
        if header.take(4, "the magic")? != MAGIC {
            return Err(DecodeError::Magic);
        }
        let version = header.u32("the format version")?;
        if version != VERSION {
            return Err(DecodeError::Version(version));
        }
        let hash = header.u64("the integrity hash")?;
        if xxh3_64(header.bytes) != hash {
            return Err(DecodeError::Integrity);
        }

        let mut r = header;
        let sources_hash = r.u64("the sources hash")?;
        let word_count = r.u32("the word count")?;
        let count = r.count(MIN_PARAM_BYTES, "the parameter count")?;
        let mut params = Vec::new();
        for _ in 0..count {
            let name = r.string("a parameter name")?;
            let kind = match r.u8("a parameter type")? {
                0 => {
                    let default = r.f32("a float default")?;
                    let range = match r.u8("a range flag")? {
                        0 => None,
                        1 => Some([r.f32("a range")?, r.f32("a range")?]),
                        other => return Err(malformed(format!("range flag {other}"))),
                    };
                    ParamKind::Float { default, range }
                }
                1 => ParamKind::Vec2 {
                    default: r.f32s("a vec2 default")?,
                },
                2 => ParamKind::Vec3 {
                    default: r.f32s("a vec3 default")?,
                },
                3 => ParamKind::Vec4 {
                    default: r.f32s("a vec4 default")?,
                },
                4 => ParamKind::Color {
                    default: r.f32s("a color default")?,
                },
                5 => ParamKind::Texture {
                    default: match r.u8("a default texture")? {
                        0 => DefaultTexture::White,
                        1 => DefaultTexture::Black,
                        2 => DefaultTexture::Normal,
                        other => return Err(malformed(format!("default texture {other}"))),
                    },
                    color_space: match r.u8("a colour space")? {
                        0 => ColorSpace::Srgb,
                        1 => ColorSpace::Linear,
                        other => return Err(malformed(format!("colour space {other}"))),
                    },
                },
                other => return Err(malformed(format!("parameter type {other}"))),
            };
            let slot = Slot {
                word: r.u32("a parameter's word")?,
                component: r.u32("a parameter's component")?,
            };
            if slot.word >= word_count || slot.component + kind.components() > 4 {
                return Err(malformed(format!(
                    "`{name}` sits at word {} component {}, outside a block of {word_count} words",
                    slot.word, slot.component
                )));
            }
            params.push(Param { name, kind, slot });
        }

        let count = r.count(MIN_VARIANT_BYTES, "the variant count")?;
        let mut variants = Vec::new();
        for _ in 0..count {
            let key = r.string("a variant key")?;
            let shader = r.shader()?;
            variants.push(CompiledVariant { key, shader });
        }
        if !r.bytes.is_empty() {
            return Err(malformed(format!(
                "{} bytes follow the last variant",
                r.bytes.len()
            )));
        }
        Ok(SurfaceArtifact {
            sources_hash,
            params: ParamTable { params, word_count },
            variants,
        })
    }
}

/// The fewest bytes a parameter and a variant can encode to, which bounds a
/// count read from the input by what the input could hold.
const MIN_PARAM_BYTES: usize = 4 + 1 + 2 + 8;
const MIN_VARIANT_BYTES: usize = 4 + MIN_SHADER_BYTES;
const MIN_SHADER_BYTES: usize = 4 + 1 + 4 + 4 + 4 + 4 + 4 + 1;
const RESOURCE_BYTES: usize = 4 + 4 + 1 + 4 + 4;

fn malformed(what: String) -> DecodeError {
    DecodeError::Malformed(what)
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// As bits, so a NaN pattern round-trips.
fn put_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        put_u32(out, value.to_bits());
    }
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_u32(out, bytes.len() as u32);
    out.extend_from_slice(bytes);
}

fn put_shader(out: &mut Vec<u8>, shader: &CompiledShader) {
    put_bytes(out, shader.name.as_bytes());
    out.push(match shader.stage {
        Stage::Vertex => 0,
        Stage::Fragment => 1,
        Stage::Compute => 2,
    });
    put_u32(out, shader.defines.len() as u32);
    for (key, value) in &shader.defines {
        put_bytes(out, key.as_bytes());
        put_bytes(out, value.as_bytes());
    }
    put_bytes(out, &shader.spirv);
    put_bytes(out, shader.msl.as_bytes());
    let bindings = &shader.bindings;
    put_u32(out, bindings.resources.len() as u32);
    for resource in &bindings.resources {
        put_u32(out, resource.set);
        put_u32(out, resource.binding);
        out.push(match resource.kind {
            ResourceKind::UniformBuffer => 0,
            ResourceKind::StorageBuffer => 1,
            ResourceKind::SampledImage => 2,
            ResourceKind::Sampler => 3,
            ResourceKind::StorageImage => 4,
            ResourceKind::CombinedImageSampler => 5,
        });
        put_u32(out, resource.count);
        put_u32(out, resource.index);
    }
    put_u32(out, bindings.push_constant_bytes);
    match bindings.workgroup_size {
        Some(size) => {
            out.push(1);
            for axis in size {
                put_u32(out, axis);
            }
        }
        None => out.push(0),
    }
}

/// The unread rest of the input. Every read checks the length it needs against
/// what is left before it touches it.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, field: &'static str) -> Result<&'a [u8], DecodeError> {
        if n > self.bytes.len() {
            return Err(DecodeError::Truncated(field));
        }
        let (taken, rest) = self.bytes.split_at(n);
        self.bytes = rest;
        Ok(taken)
    }

    fn u8(&mut self, field: &'static str) -> Result<u8, DecodeError> {
        Ok(self.take(1, field)?[0])
    }

    fn u32(&mut self, field: &'static str) -> Result<u32, DecodeError> {
        Ok(u32::from_le_bytes(
            self.take(4, field)?.try_into().expect("four bytes"),
        ))
    }

    fn u64(&mut self, field: &'static str) -> Result<u64, DecodeError> {
        Ok(u64::from_le_bytes(
            self.take(8, field)?.try_into().expect("eight bytes"),
        ))
    }

    fn f32(&mut self, field: &'static str) -> Result<f32, DecodeError> {
        self.u32(field).map(f32::from_bits)
    }

    fn f32s<const N: usize>(&mut self, field: &'static str) -> Result<[f32; N], DecodeError> {
        let mut values = [0.0; N];
        for value in &mut values {
            *value = self.f32(field)?;
        }
        Ok(values)
    }

    /// A count of items at least `min_bytes` each, refused when the rest of the
    /// input could not hold that many.
    fn count(&mut self, min_bytes: usize, field: &'static str) -> Result<usize, DecodeError> {
        let count = self.u32(field)? as usize;
        if count.saturating_mul(min_bytes) > self.bytes.len() {
            return Err(DecodeError::Truncated(field));
        }
        Ok(count)
    }

    fn bytes(&mut self, field: &'static str) -> Result<&'a [u8], DecodeError> {
        let len = self.u32(field)? as usize;
        self.take(len, field)
    }

    fn string(&mut self, field: &'static str) -> Result<String, DecodeError> {
        let bytes = self.bytes(field)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| malformed(format!("{field} is not UTF-8")))
    }

    fn shader(&mut self) -> Result<CompiledShader, DecodeError> {
        let name = self.string("a shader name")?;
        let stage = match self.u8("a shader stage")? {
            0 => Stage::Vertex,
            1 => Stage::Fragment,
            2 => Stage::Compute,
            other => return Err(malformed(format!("shader stage {other}"))),
        };
        let count = self.count(8, "a define count")?;
        let mut defines = Vec::new();
        for _ in 0..count {
            defines.push((self.string("a define")?, self.string("a define")?));
        }
        let spirv = self.bytes("SPIR-V")?.to_vec();
        if spirv.len() % 4 != 0 {
            return Err(malformed(format!(
                "`{name}`'s SPIR-V is {} bytes, not whole words",
                spirv.len()
            )));
        }
        let msl = self.string("MSL")?;
        let count = self.count(RESOURCE_BYTES, "a resource count")?;
        let mut resources = Vec::new();
        for _ in 0..count {
            resources.push(Resource {
                set: self.u32("a resource")?,
                binding: self.u32("a resource")?,
                kind: match self.u8("a resource kind")? {
                    0 => ResourceKind::UniformBuffer,
                    1 => ResourceKind::StorageBuffer,
                    2 => ResourceKind::SampledImage,
                    3 => ResourceKind::Sampler,
                    4 => ResourceKind::StorageImage,
                    5 => ResourceKind::CombinedImageSampler,
                    other => return Err(malformed(format!("resource kind {other}"))),
                },
                count: self.u32("a resource")?,
                index: self.u32("a resource")?,
            });
        }
        let push_constant_bytes = self.u32("the push constant size")?;
        let workgroup_size = match self.u8("a workgroup flag")? {
            0 => None,
            1 => Some([
                self.u32("a workgroup size")?,
                self.u32("a workgroup size")?,
                self.u32("a workgroup size")?,
            ]),
            other => return Err(malformed(format!("workgroup flag {other}"))),
        };
        Ok(CompiledShader {
            name,
            stage,
            defines,
            spirv,
            msl,
            bindings: CompiledBindings {
                resources,
                push_constant_bytes,
                workgroup_size,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact() -> SurfaceArtifact {
        let shader = |name: &str, workgroup_size| CompiledShader {
            name: name.to_string(),
            stage: Stage::Fragment,
            defines: vec![
                ("ORRIN_ALPHA_TEST".into(), "1".into()),
                ("ORRIN_CUSTOM_SURFACE".into(), "1".into()),
            ],
            spirv: 0x0723_0203u32
                .to_le_bytes()
                .into_iter()
                .chain([1, 2, 3, 4])
                .collect(),
            msl: "#include <metal_stdlib>\nfragment void main0() {}\n".into(),
            bindings: CompiledBindings {
                resources: vec![
                    Resource {
                        set: 1,
                        binding: 0,
                        kind: ResourceKind::StorageBuffer,
                        count: 1,
                        index: 0,
                    },
                    Resource {
                        set: 2,
                        binding: 0,
                        kind: ResourceKind::SampledImage,
                        count: 192,
                        index: 0,
                    },
                    Resource {
                        set: 5,
                        binding: 2,
                        kind: ResourceKind::StorageBuffer,
                        count: 1,
                        index: 4,
                    },
                ],
                push_constant_bytes: 64,
                workgroup_size,
            },
        };
        SurfaceArtifact {
            sources_hash: 0x0123_4567_89ab_cdef,
            params: ParamTable {
                params: vec![
                    Param {
                        name: "wear".into(),
                        kind: ParamKind::Float {
                            default: 0.5,
                            range: Some([0.0, 1.0]),
                        },
                        slot: Slot {
                            word: 0,
                            component: 0,
                        },
                    },
                    Param {
                        name: "mask".into(),
                        kind: ParamKind::Texture {
                            default: DefaultTexture::Normal,
                            color_space: ColorSpace::Linear,
                        },
                        slot: Slot {
                            word: 0,
                            component: 1,
                        },
                    },
                    Param {
                        name: "tint".into(),
                        kind: ParamKind::Color {
                            default: [1.0, 0.45, 0.2, 1.0],
                        },
                        slot: Slot {
                            word: 1,
                            component: 0,
                        },
                    },
                    Param {
                        name: "offset".into(),
                        kind: ParamKind::Vec2 {
                            default: [-1.0, 2.0],
                        },
                        slot: Slot {
                            word: 2,
                            component: 0,
                        },
                    },
                ],
                word_count: 3,
            },
            variants: vec![
                CompiledVariant {
                    key: "forward_frag".into(),
                    shader: shader("rock.oshader [forward_frag]", None),
                },
                CompiledVariant {
                    key: "made_up".into(),
                    shader: shader("rock.oshader [made_up]", Some([8, 8, 1])),
                },
            ],
        }
    }

    /// The header is magic, version and hash; this rewrites the hash over a
    /// payload that has been edited, as a writer that meant it would.
    fn rehash(bytes: &mut [u8]) {
        let hash = xxhash_rust::xxh3::xxh3_64(&bytes[16..]);
        bytes[8..16].copy_from_slice(&hash.to_le_bytes());
    }

    #[test]
    fn an_artifact_round_trips_exactly() {
        let original = artifact();
        let bytes = original.encode();
        assert_eq!(&bytes[..4], b"OSRF");
        assert_eq!(SurfaceArtifact::decode(&bytes), Ok(original.clone()));
        assert_eq!(SurfaceArtifact::decode(&bytes).unwrap().encode(), bytes);
    }

    #[test]
    fn every_single_byte_corruption_is_detected() {
        let bytes = artifact().encode();
        for at in 0..bytes.len() {
            for flip in [0x01u8, 0x80, 0xff] {
                let mut corrupt = bytes.clone();
                corrupt[at] ^= flip;
                assert!(
                    SurfaceArtifact::decode(&corrupt).is_err(),
                    "byte {at} ^ {flip:#x} went unnoticed"
                );
            }
        }
    }

    #[test]
    fn every_truncation_is_an_error() {
        let bytes = artifact().encode();
        for len in 0..bytes.len() {
            assert!(
                SurfaceArtifact::decode(&bytes[..len]).is_err(),
                "{len} of {} bytes decoded",
                bytes.len()
            );
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(SurfaceArtifact::decode(&longer).is_err());
    }

    #[test]
    fn a_wrong_magic_and_a_wrong_version_are_named() {
        let mut bytes = artifact().encode();
        bytes[0] = b'X';
        assert_eq!(SurfaceArtifact::decode(&bytes), Err(DecodeError::Magic));
        assert!(DecodeError::Magic.to_string().contains("magic"));

        let mut bytes = artifact().encode();
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(
            SurfaceArtifact::decode(&bytes),
            Err(DecodeError::Version(2))
        );
        assert!(DecodeError::Version(2).to_string().contains("version 2"));

        let mut bytes = artifact().encode();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        assert_eq!(SurfaceArtifact::decode(&bytes), Err(DecodeError::Integrity));
    }

    /// The hash vouches for the bytes, not for the writer: a payload that hashes
    /// correctly but says nonsense must still come back as an error, never a
    /// panic or an allocation sized by a corrupt length.
    #[test]
    fn a_payload_that_hashes_but_lies_is_an_error_and_never_a_panic() {
        let bytes = artifact().encode();
        for at in 16..bytes.len() {
            for value in [0x00u8, 0x7f, 0xff] {
                let mut lying = bytes.clone();
                lying[at] = value;
                rehash(&mut lying);
                let _ = SurfaceArtifact::decode(&lying);
            }
        }

        // The first variant's key length, set to claim four gigabytes.
        let mut lying = bytes.clone();
        let key = lying
            .windows(b"forward_frag".len())
            .position(|w| w == b"forward_frag")
            .unwrap();
        lying[key - 4..key].copy_from_slice(&u32::MAX.to_le_bytes());
        rehash(&mut lying);
        assert!(matches!(
            SurfaceArtifact::decode(&lying),
            Err(DecodeError::Truncated(_))
        ));
    }

    #[test]
    fn a_slot_outside_the_block_is_malformed() {
        let mut broken = artifact();
        broken.params.params[2].slot.word = 3;
        assert!(matches!(
            SurfaceArtifact::decode(&broken.encode()),
            Err(DecodeError::Malformed(_))
        ));
        let mut broken = artifact();
        broken.params.params[3].slot.component = 3;
        assert!(matches!(
            SurfaceArtifact::decode(&broken.encode()),
            Err(DecodeError::Malformed(_))
        ));
    }

    #[test]
    fn an_artifact_matches_this_build_only_by_its_sources_hash() {
        let mut current = artifact();
        current.sources_hash = SOURCES_HASH;
        assert!(current.matches_this_build());
        current.sources_hash ^= 1;
        assert!(!current.matches_this_build());
    }

    #[test]
    fn a_variant_names_the_built_in_shader_it_replaces() {
        let artifact = artifact();
        assert_eq!(artifact.variants[0].builtin(), Some(ShaderId::ForwardFrag));
        assert_eq!(artifact.variants[1].builtin(), None);
    }
}
