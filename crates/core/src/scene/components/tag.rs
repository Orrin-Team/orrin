use orrin_registry::Reflect;

/// A gameplay label scripts look up via `World.FindByTag` / `FindAllByTag`.
/// Unlike [`Name`](super::Name) (an editor display label), many entities may
/// share one tag, and lookups match on exact equality.
///
/// Its reflected shape is identical to [`Name`](super::Name)'s — the two are
/// told apart by their `ComponentId`, never by shape.
#[derive(Clone, Debug, Default, PartialEq, Eq, Reflect)]
pub struct Tag(pub String);

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
