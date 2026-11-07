//! Input, in three layers.
//!
//! [`state`] holds what the devices are doing: which keys and buttons are down,
//! where the mouse moved, where the sticks are. It is the only layer that knows
//! a backend exists.
//!
//! [`keys`] is the vocabulary — one table naming every source once, so a
//! binding file, the engine's codes and the C# enum cannot drift apart.
//!
//! [`actions`] is what game code sees: named actions and axes, resolved from
//! bindings once a frame. A script names `Jump`, never `Space`, so remapping
//! is a config change rather than an API break.
//!
//! The layers only face one way. Resolution reads device state through public
//! accessors and never the reverse, which is what lets a new backend arrive as
//! an adapter over [`InputState::press`] without anything above it changing.

pub mod actions;
pub mod binding;
pub mod config;
mod cursor;
pub mod gamepad;
pub mod keys;
mod state;
#[cfg(feature = "scripting")]
pub mod watch;

pub use actions::{ActionId, Actions, Axis, Definition, Spec};
pub use binding::Binding;
pub use config::ConfigError;
pub use cursor::{CursorLock, wants_grab};
pub use gamepad::Gamepads;
pub use state::InputState;
#[cfg(feature = "scripting")]
pub use watch::ConfigWatcher;

/// How many players an input config can address.
///
/// Four because that is what a local session on one machine reaches before it
/// runs out of pads and sofa. The number is a storage width, not a promise:
/// every query takes a player index so raising it later changes no signature,
/// which is the whole reason the index is there from the start.
pub const MAX_PLAYERS: usize = 4;
