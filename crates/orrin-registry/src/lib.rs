//! One central description, per component type, of how to read it, write it,
//! and default it — keyed by a stable string id rather than by Rust type
//! identity.
//!
//! Everything the editor and persistence layers do is written once against
//! [`Registry`]: scene save/load, the inspector, prefab overrides, and later
//! undo/redo and collaboration sync. A component type participates by
//! implementing [`Reflect`] (converting to and from [`Value`]) and being
//! registered under a [`ComponentId`].
//!
//! Registration is explicit and re-runnable — the engine and each game assembly
//! call their own `register_components`. Linker-based auto-registration
//! (`inventory`, `ctor`) does not survive a dynamic library boundary, which is
//! exactly the configuration hot reload creates.

mod asset_id;
mod diff;
mod entity_id;
#[cfg(feature = "egui")]
pub mod inspect;
mod migrate;
mod reflect;
mod registry;
mod scene;
mod text;
mod value;
pub mod wire;

pub use asset_id::AssetId;
pub use diff::{FieldChange, apply, diff};
pub use entity_id::EntityId;
pub use scene::{
    FORMAT_VERSION, MIN_READER_VERSION, NAME_COMPONENT, ParseError, SceneDocument, SceneEntity,
    parse, parse_value_document,
};

/// Shares its name with the [`Reflect`](trait@Reflect) trait, the way
/// `serde::Serialize` does — the derive lives in the macro namespace and the
/// trait in the type namespace, so one import brings both.
pub use orrin_macros::Reflect;
pub use reflect::{Reflect, Resolve, take, take_or};
pub use registry::{
    ApplyError, ComponentId, ComponentVtable, Registry, ScriptBinding, ScriptBridge,
};
pub use text::{write_document, write_entity, write_value_document, write_world};
pub use value::{FieldPath, PathSegment, Value, ValueError};
