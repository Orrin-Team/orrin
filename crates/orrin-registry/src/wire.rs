//! The byte form of a [`Value`], for crossing the script boundary.
//!
//! One flattened buffer carries one whole component. Field-by-field access
//! across the FFI would cost a managed transition per field, and — worse — a
//! caller reading five fields in five calls can observe a Behaviour halfway
//! through its own `Update`, which is a class of bug no amount of care at the
//! call site removes.
//!
//! **This is an ABI.** `Orrin.PropertyBag` on the C# side implements the same
//! grammar, and `Orrin.MathTests` asserts the two agree byte for byte on a
//! fixed vector, the way `sizeof(Transform) == 40` is asserted. A tag number is
//! therefore permanent: append, never renumber — the same rule `ComponentKind`
//! and `KeyCode` follow.
//!
//! Not the scene format. That one is text, sorted and deterministic so a git
//! diff means something (architecture §2.4); this one is read once by a process
//! that wrote it a microsecond earlier, so it keeps declaration order and
//! spends nothing on being readable.

use std::fmt;

use crate::value::Value;
use crate::{AssetId, EntityId};

/// Tag bytes. Numbering is lock-step with `Orrin.PropertyBag.Tag`.
mod tag {
    pub const BOOL: u8 = 0;
    pub const I32: u8 = 1;
    pub const U32: u8 = 2;
    pub const F32: u8 = 3;
    pub const STRING: u8 = 4;
    pub const VEC3: u8 = 5;
    pub const QUAT: u8 = 6;
    pub const ENTITY: u8 = 7;
    pub const STRUCT: u8 = 8;
    pub const ENUM: u8 = 9;
    pub const LIST: u8 = 10;
    pub const ASSET: u8 = 11;
}

/// How deep a decoded value may nest.
///
/// The decoder recurses, and the buffer comes from a separately compiled
/// assembly that may be a build behind. A stack overflow is not a catchable
/// error in Rust, so the depth is refused rather than survived. Nothing
/// legitimate is near this: a component is a struct of leaves, occasionally a
/// list of structs.
const MAX_DEPTH: u32 = 32;

/// A buffer that is not a [`Value`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WireError {
    /// The buffer ended in the middle of a value.
    Truncated,
    /// A tag this build has no variant for — the game assembly is newer than
    /// the engine, or the buffer is not a property bag at all.
    UnknownTag(u8),
    /// A string field that was not UTF-8.
    BadUtf8,
    /// Nesting past [`MAX_DEPTH`].
    TooDeep,
    /// A complete value, followed by bytes that are not part of it.
    Trailing,
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("the buffer ended mid-value"),
            Self::UnknownTag(tag) => write!(
                f,
                "value tag {tag} is not one this build knows; rebuild the game assembly \
                 against the current Orrin bindings"
            ),
            Self::BadUtf8 => f.write_str("a string field was not valid UTF-8"),
            Self::TooDeep => write!(f, "a value nested deeper than {MAX_DEPTH} levels"),
            Self::Trailing => f.write_str("the buffer holds more than one value"),
        }
    }
}

impl std::error::Error for WireError {}

/// Append `value`'s bytes to `out`.
///
/// Takes a `&mut Vec` rather than returning one so a caller crossing the
/// boundary every frame keeps one buffer instead of allocating per component —
/// the same reason `FrameGeometry` reuses its vectors.
pub fn encode(value: &Value, out: &mut Vec<u8>) {
    match value {
        Value::Bool(v) => {
            out.push(tag::BOOL);
            out.push(u8::from(*v));
        }
        Value::I32(v) => {
            out.push(tag::I32);
            out.extend_from_slice(&v.to_le_bytes());
        }
        Value::U32(v) => {
            out.push(tag::U32);
            out.extend_from_slice(&v.to_le_bytes());
        }
        Value::F32(v) => {
            out.push(tag::F32);
            // Bits, not a decimal rendering: the boundary must carry a NaN and a
            // negative zero unchanged, because `diff` distinguishes them.
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        Value::String(v) => {
            out.push(tag::STRING);
            write_str(v, out);
        }
        Value::Vec3(v) => {
            out.push(tag::VEC3);
            for component in v.to_array() {
                out.extend_from_slice(&component.to_bits().to_le_bytes());
            }
        }
        Value::Quat(v) => {
            out.push(tag::QUAT);
            for component in v.to_array() {
                out.extend_from_slice(&component.to_bits().to_le_bytes());
            }
        }
        Value::Entity(id) => {
            out.push(tag::ENTITY);
            out.extend_from_slice(&id.to_bytes());
        }
        Value::Asset(id) => {
            out.push(tag::ASSET);
            out.extend_from_slice(&id.to_bytes());
        }
        Value::Struct(fields) => {
            out.push(tag::STRUCT);
            write_fields(fields, out);
        }
        Value::Enum { variant, fields } => {
            out.push(tag::ENUM);
            write_str(variant, out);
            write_fields(fields, out);
        }
        Value::List(items) => {
            out.push(tag::LIST);
            write_len(items.len(), out);
            for item in items {
                encode(item, out);
            }
        }
    }
}

/// Read exactly one value out of `bytes`.
pub fn decode(bytes: &[u8]) -> Result<Value, WireError> {
    let mut reader = Reader { bytes, at: 0 };
    let value = reader.value(0)?;
    if reader.at != reader.bytes.len() {
        return Err(WireError::Trailing);
    }
    Ok(value)
}

fn write_str(value: &str, out: &mut Vec<u8>) {
    write_len(value.len(), out);
    out.extend_from_slice(value.as_bytes());
}

fn write_fields(fields: &[(String, Value)], out: &mut Vec<u8>) {
    write_len(fields.len(), out);
    for (name, value) in fields {
        write_str(name, out);
        encode(value, out);
    }
}

/// Lengths are `u32` on the wire. A component with four billion fields is not a
/// case worth eight bytes on every string in every buffer.
fn write_len(len: usize, out: &mut Vec<u8>) {
    out.extend_from_slice(&(len as u32).to_le_bytes());
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], WireError> {
        let end = self.at.checked_add(n).ok_or(WireError::Truncated)?;
        let slice = self.bytes.get(self.at..end).ok_or(WireError::Truncated)?;
        self.at = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, WireError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, WireError> {
        let bytes: [u8; 4] = self.take(4)?.try_into().expect("took four bytes");
        Ok(u32::from_le_bytes(bytes))
    }

    fn f32(&mut self) -> Result<f32, WireError> {
        self.u32().map(f32::from_bits)
    }

    fn string(&mut self) -> Result<String, WireError> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| WireError::BadUtf8)
    }

    /// The count is checked against the bytes that remain before anything is
    /// reserved: a corrupt length of four billion would otherwise ask for a
    /// 100 GB allocation from a 40-byte buffer. One byte is the smallest a
    /// field or element can encode to.
    fn count(&mut self, per_item: usize) -> Result<usize, WireError> {
        let count = self.u32()? as usize;
        if count.saturating_mul(per_item) > self.bytes.len() - self.at {
            return Err(WireError::Truncated);
        }
        Ok(count)
    }

    fn fields(&mut self, depth: u32) -> Result<Vec<(String, Value)>, WireError> {
        // A field is at least a four-byte name length plus a one-byte tag.
        let count = self.count(5)?;
        let mut fields = Vec::with_capacity(count);
        for _ in 0..count {
            let name = self.string()?;
            fields.push((name, self.value(depth)?));
        }
        Ok(fields)
    }

    fn value(&mut self, depth: u32) -> Result<Value, WireError> {
        if depth > MAX_DEPTH {
            return Err(WireError::TooDeep);
        }
        let depth = depth + 1;
        match self.u8()? {
            tag::BOOL => Ok(Value::Bool(self.u8()? != 0)),
            tag::I32 => Ok(Value::I32(self.u32()? as i32)),
            tag::U32 => Ok(Value::U32(self.u32()?)),
            tag::F32 => Ok(Value::F32(self.f32()?)),
            tag::STRING => Ok(Value::String(self.string()?)),
            tag::VEC3 => Ok(Value::Vec3(glam::Vec3::new(
                self.f32()?,
                self.f32()?,
                self.f32()?,
            ))),
            tag::QUAT => Ok(Value::Quat(glam::Quat::from_xyzw(
                self.f32()?,
                self.f32()?,
                self.f32()?,
                self.f32()?,
            ))),
            tag::ENTITY => {
                let bytes: [u8; 16] = self.take(16)?.try_into().expect("took sixteen bytes");
                Ok(Value::Entity(EntityId::from_bytes(bytes)))
            }
            tag::ASSET => {
                let bytes: [u8; 16] = self.take(16)?.try_into().expect("took sixteen bytes");
                Ok(Value::Asset(AssetId::from_bytes(bytes)))
            }
            tag::STRUCT => Ok(Value::Struct(self.fields(depth)?)),
            tag::ENUM => {
                let variant = self.string()?;
                Ok(Value::Enum {
                    variant,
                    fields: self.fields(depth)?,
                })
            }
            tag::LIST => {
                let count = self.count(1)?;
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(self.value(depth)?);
                }
                Ok(Value::List(items))
            }
            other => Err(WireError::UnknownTag(other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    fn round_trip(value: &Value) {
        let mut bytes = Vec::new();
        encode(value, &mut bytes);
        assert_eq!(decode(&bytes).as_ref(), Ok(value));
    }

    /// The fixture `Orrin.MathTests` encodes on the C# side and compares against
    /// [`GOLDEN`]. Change one and the other fails, which is the point.
    fn golden_value() -> Value {
        Value::strukt([
            ("on", Value::Bool(true)),
            ("count", Value::I32(-2)),
            ("index", Value::U32(7)),
            ("speed", Value::F32(1.5)),
            ("label", Value::String("hi".to_owned())),
            ("position", Value::Vec3(Vec3::new(1.0, 2.0, 3.0))),
            ("rotation", Value::Quat(Quat::IDENTITY)),
            ("target", Value::Entity(EntityId::NIL)),
            ("points", Value::List(vec![Value::F32(0.5)])),
            (
                "mode",
                Value::enumeration("Spot", [("on", Value::Bool(false))]),
            ),
        ])
    }

    #[test]
    fn every_variant_round_trips() {
        round_trip(&golden_value());
        round_trip(&Value::List(Vec::new()));
        round_trip(&Value::Struct(Vec::new()));
        round_trip(&Value::String(String::new()));
        round_trip(&Value::String("é 𝄞".to_owned()));
        round_trip(&Value::I32(i32::MIN));
        round_trip(&Value::U32(u32::MAX));
    }

    /// The boundary carries bit patterns, not decimals — `diff` treats a NaN and
    /// a negative zero as distinct values, so a round trip that normalized
    /// either would make a script component report an edit it never made.
    #[test]
    fn non_finite_floats_and_negative_zero_survive_bit_for_bit() {
        for bits in [f32::NAN, -f32::NAN, 0.0, -0.0, f32::INFINITY, f32::MIN] {
            let mut bytes = Vec::new();
            encode(&Value::F32(bits), &mut bytes);
            let Ok(Value::F32(back)) = decode(&bytes) else {
                panic!("an f32 must decode as an f32");
            };
            assert_eq!(back.to_bits(), bits.to_bits());
        }
    }

    #[test]
    fn declaration_order_is_preserved() {
        // Unlike the scene format, which sorts. An inspector draws in this
        // order, so the boundary must not quietly alphabetize.
        let value = Value::strukt([("z", Value::I32(1)), ("a", Value::I32(2))]);
        let mut bytes = Vec::new();
        encode(&value, &mut bytes);
        let Ok(Value::Struct(fields)) = decode(&bytes) else {
            panic!("a struct must decode as a struct");
        };
        assert_eq!(fields[0].0, "z");
        assert_eq!(fields[1].0, "a");
    }

    #[test]
    fn a_truncated_buffer_is_refused_at_every_length() {
        let mut bytes = Vec::new();
        encode(&golden_value(), &mut bytes);
        for len in 0..bytes.len() {
            assert!(
                decode(&bytes[..len]).is_err(),
                "a {len}-byte prefix decoded as a whole value"
            );
        }
        assert!(decode(&bytes).is_ok());
    }

    #[test]
    fn trailing_bytes_are_refused() {
        let mut bytes = Vec::new();
        encode(&Value::Bool(true), &mut bytes);
        bytes.push(0);
        assert_eq!(decode(&bytes), Err(WireError::Trailing));
    }

    #[test]
    fn an_unknown_tag_names_itself_and_says_what_to_do() {
        let error = decode(&[200]).unwrap_err();
        assert_eq!(error, WireError::UnknownTag(200));
        assert!(error.to_string().contains("rebuild the game assembly"));
    }

    /// A length field is attacker-shaped input even when the "attacker" is only
    /// a stale build: a struct claiming four billion fields must be refused
    /// rather than reserved for.
    #[test]
    fn an_impossible_count_is_refused_without_allocating_for_it() {
        let mut bytes = vec![tag::STRUCT];
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode(&bytes), Err(WireError::Truncated));

        let mut bytes = vec![tag::LIST];
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode(&bytes), Err(WireError::Truncated));
    }

    /// The decoder recurses, so a buffer that nests without bound would
    /// overflow the stack — which no `Result` can report.
    #[test]
    fn nesting_past_the_limit_is_refused_rather_than_overflowing() {
        let mut bytes = Vec::new();
        for _ in 0..MAX_DEPTH + 2 {
            bytes.push(tag::LIST);
            bytes.extend_from_slice(&1u32.to_le_bytes());
        }
        bytes.push(tag::BOOL);
        bytes.push(1);
        assert_eq!(decode(&bytes), Err(WireError::TooDeep));
    }

    #[test]
    fn a_string_that_is_not_utf8_is_refused() {
        let mut bytes = vec![tag::STRING];
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&[0xff, 0xfe]);
        assert_eq!(decode(&bytes), Err(WireError::BadUtf8));
    }

    /// Pins the encoder against a vector covering every variant, so a change to
    /// the grammar has to be made deliberately rather than noticed later by a
    /// scene file that no longer loads. `Orrin.MathTests` decodes these same
    /// bytes to check that the C# reader steps over the variants it cannot
    /// represent.
    #[test]
    fn the_golden_vector_has_not_moved() {
        let mut bytes = Vec::new();
        encode(&golden_value(), &mut bytes);
        assert_eq!(hex(&bytes), GOLDEN);
    }

    /// The value a C# Behaviour with one field of each representable type
    /// encodes to. `Orrin.MathTests` builds that Behaviour, encodes it, and
    /// asserts the same string — which is the only check that the two
    /// implementations of this grammar actually agree.
    ///
    /// It stops short of `golden_value` because a Behaviour cannot express an
    /// `Entity` reference or a list; see `Orrin.PropertyBag` for why.
    #[test]
    fn the_bag_a_behaviour_encodes_to_has_not_moved() {
        let mut bytes = Vec::new();
        encode(&behaviour_bag(), &mut bytes);
        assert_eq!(hex(&bytes), BAG_GOLDEN);
    }

    fn behaviour_bag() -> Value {
        Value::strukt([
            ("On", Value::Bool(true)),
            ("Count", Value::I32(-2)),
            ("Index", Value::U32(7)),
            ("Speed", Value::F32(1.5)),
            ("Label", Value::String("hi".to_owned())),
            ("Position", Value::Vec3(Vec3::new(1.0, 2.0, 3.0))),
            ("Rotation", Value::Quat(Quat::IDENTITY)),
            ("Mode", Value::enumeration("Spot", [])),
        ])
    }

    /// The asset tag was appended after the original eleven, so it is pinned on
    /// its own rather than folded into [`GOLDEN`] — which `Orrin.MathTests` also
    /// holds, and which must keep meaning what it meant. The C# side asserts
    /// these same bytes for an `AssetRef<T>` field.
    #[test]
    fn an_asset_reference_crosses_as_its_canonical_bytes() {
        let id: crate::AssetId = "00112233-4455-6677-8899-aabbccddeeff".parse().unwrap();
        let mut bytes = Vec::new();
        encode(&Value::Asset(id), &mut bytes);
        assert_eq!(hex(&bytes), "0b00112233445566778899aabbccddeeff");
        round_trip(&Value::Asset(id));
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    const BAG_GOLDEN: &str = concat!(
        "0808000000",                                                 // Struct, 8 fields
        "020000004f6e0001",                                           // On = Bool true
        "05000000436f756e7401feffffff",                               // Count = I32 -2
        "05000000496e6465780207000000",                               // Index = U32 7
        "050000005370656564030000c03f",                               // Speed = F32 1.5
        "050000004c6162656c04020000006869",                           // Label = "hi"
        "08000000506f736974696f6e050000803f0000004000004040",         // Position = (1, 2, 3)
        "08000000526f746174696f6e060000000000000000000000000000803f", // Rotation = the identity quat
        "040000004d6f6465090400000053706f7400000000",                 // Mode = the Spot member
    );

    const GOLDEN: &str = concat!(
        "080a000000",                                                 // Struct, 10 fields
        "020000006f6e0001",                                           // on = Bool true
        "05000000636f756e7401feffffff",                               // count = I32 -2
        "05000000696e6465780207000000",                               // index = U32 7
        "050000007370656564030000c03f",                               // speed = F32 1.5
        "050000006c6162656c04020000006869",                           // label = "hi"
        "08000000706f736974696f6e050000803f0000004000004040",         // position = (1, 2, 3)
        "08000000726f746174696f6e060000000000000000000000000000803f", // rotation = the identity quat
        "060000007461726765740700000000000000000000000000000000",     // target = the nil id
        "06000000706f696e74730a01000000030000003f",                   // points = [0.5]
        "040000006d6f6465090400000053706f7401000000020000006f6e0000", // mode = Spot { on: false }
    );
}
