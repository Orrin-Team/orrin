use glam::Vec3;
use orrin_registry::{Reflect, Value, ValueError};

/// A hierarchy row that only holds others: an entity with this and no
/// transform. Having no transform is what makes it a folder rather than a
/// group — `scene::hierarchy` treats a parent without one as organisational,
/// so filing something in a folder never moves it in the world, and nothing
/// moves the folder.
///
/// Hand-written rather than derived for its optional tint, which a scene
/// writes only when one is set, the way `Image` writes its texture.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Folder {
    /// The colour its icon is drawn in, sRGB. `None` is the tree's own.
    pub tint: Option<Vec3>,
}

impl Reflect for Folder {
    fn to_value(&self) -> Value {
        Value::Struct(
            self.tint
                .map(|tint| ("tint".to_owned(), tint.to_value()))
                .into_iter()
                .collect(),
        )
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        let tint = value
            .field("tint")
            .map(|tint| Vec3::from_value(tint).map_err(|error| error.at_field("tint")))
            .transpose()?;
        Ok(Self { tint })
    }
}
