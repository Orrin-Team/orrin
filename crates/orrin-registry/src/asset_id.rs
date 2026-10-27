use std::fmt;
use std::str::FromStr;

use uuid::Uuid;

/// An asset's identity, independent of where its file lives.
///
/// Assigned once, stored in the sidecar beside the source file, and never
/// reused. A path is a human convenience resolved through the asset index;
/// nothing in a scene or a component stores one, which is what makes renaming
/// and moving a file free (architecture §2.2).
///
/// A separate type from [`EntityId`](crate::EntityId) despite the identical
/// representation: the two name different things, and a scene that confused
/// one for the other would load without error and point at nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AssetId(Uuid);

impl AssetId {
    /// A fresh, random identity.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// The all-zeroes id, which is never assigned.
    pub const NIL: Self = Self(Uuid::nil());

    pub fn is_nil(self) -> bool {
        self.0.is_nil()
    }

    /// The 16 bytes in canonical UUID order, as [`EntityId::to_bytes`] gives
    /// them, and as C# `Guid(bytes, bigEndian: true)` reads them.
    ///
    /// [`EntityId::to_bytes`]: crate::EntityId::to_bytes
    pub const fn to_bytes(self) -> [u8; 16] {
        *self.0.as_bytes()
    }

    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(Uuid::from_bytes(bytes))
    }

    /// The id of an asset with no file to keep an assigned id in — an
    /// engine-built mesh or material — derived from the `kind` and `name` it is
    /// registered under.
    ///
    /// A version-8 UUID, the custom-UUID version, so no derived id can equal a
    /// random (version 4) one from a sidecar. Frozen: the scene migration to
    /// format 3 writes these into files, so the hash and its input can never
    /// change.
    pub fn named(kind: &str, name: &str) -> Self {
        let mut input = Vec::with_capacity(kind.len() + name.len() + 13);
        input.extend_from_slice(b"orrin.asset\0");
        input.extend_from_slice(kind.as_bytes());
        input.push(0);
        input.extend_from_slice(name.as_bytes());
        let mut bytes = xxhash_rust::xxh3::xxh3_128(&input).to_be_bytes();
        bytes[6] = (bytes[6] & 0x0f) | 0x80;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        Self::from_bytes(bytes)
    }
}

impl fmt::Display for AssetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.as_hyphenated())
    }
}

impl FromStr for AssetId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Uuid::parse_str(s).map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned, because files hold these: a change to the derivation would
    /// silently detach every migrated scene from its meshes.
    #[test]
    fn a_named_id_is_frozen() {
        assert_eq!(
            AssetId::named("mesh", "cube").to_string(),
            "2398d189-baf9-8039-9293-4dbdcaeb828b"
        );
    }

    #[test]
    fn a_named_id_is_version_8_and_depends_on_kind_and_name() {
        let cube = AssetId::named("mesh", "cube");
        assert_eq!(cube.0.get_version_num(), 8);
        assert_eq!(cube.0.get_variant(), uuid::Variant::RFC4122);
        assert_eq!(cube, AssetId::named("mesh", "cube"));
        assert_ne!(cube, AssetId::named("material", "cube"));
        assert_ne!(cube, AssetId::named("mesh", "cubes"));
        assert_ne!(
            AssetId::named("mes", "hcube"),
            cube,
            "the parts are separated"
        );
    }
}
