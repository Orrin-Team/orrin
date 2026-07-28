/// A gameplay label scripts look up via `World.FindByTag` / `FindAllByTag`.
/// Unlike [`Name`](super::Name) (an editor display label), many entities may
/// share one tag, and lookups match on exact equality.
#[derive(Clone, Debug, PartialEq, Eq)]
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
