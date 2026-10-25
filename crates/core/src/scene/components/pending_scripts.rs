use orrin_registry::{ComponentId, Value};

/// Behaviours a load found for this entity, waiting for the script host to
/// create them.
///
/// A Behaviour's fields can only be written into a live managed object, and a
/// load — a scene opening, a prefab instantiating — has no host to create one
/// with. So the saved value waits here, and `Scripting::attach_pending` creates
/// the Behaviour and writes it at the start of the next script phase. Saving
/// before then writes the value back unchanged, as [`UnknownComponents`]
/// does for data nothing claims.
///
/// Not registered, for the reason `UnknownComponents` is not.
///
/// [`UnknownComponents`]: super::UnknownComponents
#[derive(Clone, Debug, Default)]
pub struct PendingScripts(pub Vec<(ComponentId, Value)>);
