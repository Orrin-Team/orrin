use orrin_registry::{Reflect, Value, ValueError};

/// A display name for tooling; doesn't affect simulation or rendering.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Name(pub String);

impl Reflect for Name {
    fn to_value(&self) -> Value {
        self.0.to_value()
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        String::from_value(value).map(Self)
    }
}

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
