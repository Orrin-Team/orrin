use std::ops::{Deref, DerefMut};

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
