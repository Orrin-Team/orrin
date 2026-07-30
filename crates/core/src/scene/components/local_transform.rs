use std::ops::{Deref, DerefMut};

use orrin_registry::{Reflect, Value, ValueError};

use crate::scene::Transform;

/// The ECS component form of [`Transform`]; derefs to it, so all its helpers
/// are available directly on a `LocalTransform`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LocalTransform(pub Transform);

impl LocalTransform {
    #[inline]
    pub fn new(transform: Transform) -> Self {
        Self(transform)
    }
}

/// A newtype flattens to its inner value: the registry sees a `LocalTransform`
/// exactly as it sees a `Transform`, with no `.0` level in field paths or in
/// the scene file.
impl Reflect for LocalTransform {
    fn to_value(&self) -> Value {
        self.0.to_value()
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        Transform::from_value(value).map(Self)
    }
}

impl From<Transform> for LocalTransform {
    #[inline]
    fn from(transform: Transform) -> Self {
        Self(transform)
    }
}

impl Deref for LocalTransform {
    type Target = Transform;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LocalTransform {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
