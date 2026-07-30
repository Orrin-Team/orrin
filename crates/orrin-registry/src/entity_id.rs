use std::fmt;
use std::str::FromStr;

use uuid::Uuid;

/// An entity's identity across sessions, machines, and collaborators.
///
/// Distinct from `orrin_ecs::Entity`, which is a dense slot handle whose value
/// depends on this session's spawn and despawn history and means nothing once
/// the process exits. `Entity` is the fast handle hot loops use; `EntityId` is
/// what the scene file, and later the network protocol, speak.
///
/// This is why [`Value`](crate::Value) has no variant for a raw `Entity`: a
/// component that referenced one would serialize a number that names a
/// different entity — or none — the next time the scene is opened. Storing
/// `EntityId` instead is the mechanical form of the rule that components hold
/// handles, never references.
///
/// Assigned on save to any entity that lacks one, and never reused.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntityId(Uuid);

impl EntityId {
    /// A fresh, random identity.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// The all-zeroes id, which is never assigned — it is what
    /// `EntityId::default()` produces, so an unset field is recognizable.
    pub const NIL: Self = Self(Uuid::nil());

    pub fn is_nil(self) -> bool {
        self.0.is_nil()
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Hyphenated lowercase, the canonical UUID form — fixed width and one
        // spelling per value, which the scene format's determinism needs.
        write!(f, "{}", self.0.as_hyphenated())
    }
}

impl FromStr for EntityId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Uuid::parse_str(s).map(Self)
    }
}
