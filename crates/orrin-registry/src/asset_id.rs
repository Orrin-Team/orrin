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
