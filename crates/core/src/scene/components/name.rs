use orrin_registry::Reflect;

/// A display name for tooling; doesn't affect simulation or rendering.
#[derive(Clone, Debug, Default, PartialEq, Eq, Reflect)]
pub struct Name(pub String);

impl Name {
    #[inline]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Name {
    #[inline]
    fn from(name: &str) -> Self {
        Self(name.to_owned())
    }
}
