use orrin_registry::{ComponentId, Value};

/// Component data a load could not apply, kept verbatim so saving gives it back.
///
/// Two things land here: components whose id no registry entry claims (a game
/// assembly that isn't loaded, a plugin that isn't installed), and components
/// whose value the type rejected (a field renamed since the scene was written).
///
/// Preserving rather than dropping is the point. Open a scene with a plugin
/// missing, save it, and in Unity or Unreal that plugin's data is gone — the
/// user is never told, and the loss is discovered much later by someone else.
/// Here the bytes ride along on the entity and go back out unchanged.
///
/// Deliberately *not* registered: it is metadata about components, not a
/// component, and registering it would have it serialize as a component
/// containing components.
#[derive(Clone, Debug, Default)]
pub struct UnknownComponents(pub Vec<(ComponentId, Value)>);
