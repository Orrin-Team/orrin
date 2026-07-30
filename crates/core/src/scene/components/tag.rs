use orrin_registry::{Reflect, Value, ValueError};

/// A gameplay label scripts look up via `World.FindByTag` / `FindAllByTag`.
/// Unlike [`Name`](super::Name) (an editor display label), many entities may
/// share one tag, and lookups match on exact equality.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tag(pub String);

/// Structurally identical to [`Name`](super::Name)'s — the two are told apart
/// by their `ComponentId`, never by shape.
impl Reflect for Tag {
    fn to_value(&self) -> Value {
        self.0.to_value()
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        String::from_value(value).map(Self)
    }
}

impl Tag {
    #[inline]
    pub fn new(tag: impl Into<String>) -> Self {
        Self(tag.into())
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Tag {
    #[inline]
    fn from(tag: &str) -> Self {
        Self(tag.to_owned())
    }
}
