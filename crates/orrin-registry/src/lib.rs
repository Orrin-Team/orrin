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

mod reflect;
mod registry;
mod text;
mod value;

pub use reflect::{Reflect, take};
pub use registry::{ComponentId, ComponentVtable, Registry};
pub use text::{write_entity, write_world};
pub use value::{FieldPath, PathSegment, Value, ValueError};
