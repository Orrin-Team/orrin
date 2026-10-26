use orrin_registry::Reflect;

/// Kept out of reach in the viewport: a click, a hover or a drawn rectangle
/// passes through it, and through everything under it in the hierarchy, to
/// what is behind. Selected from the hierarchy it is still moved and edited
/// like anything else — it locks picking, not the entity.
///
/// Saved with the scene, because what is locked is part of how a scene is
/// meant to be worked on: a floor that should not be grabbed by accident
/// stays that way the next time the scene is opened.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub struct Locked;
