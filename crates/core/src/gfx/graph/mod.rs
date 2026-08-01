//! The frame render graph: passes declare what they touch, the compiler derives
//! the rest.
//!
//! A pass registers a name, a kind, and a list of `(resource, access)` pairs.
//! From those declarations alone the compiler derives execution order, image
//! layout transitions, the barriers between passes, and the usage flags each
//! image is created with. Nothing here takes a `Device`: a graph compiles from
//! declarations, which is why CI — on a runner with no GPU — can assert the
//! exact barrier sequence a reference frame produces.
//!
//! **Why this shape.** Architecture §3.1 gives three reasons, and none of them
//! is performance. Derived barriers remove the worst Vulkan bug class, because a
//! pass author can no longer write an incorrect one — only an incorrect
//! declaration, which the compiler rejects by name. A frame that is data is a
//! frame the editor can draw, time per pass, and let you click into. And a
//! renderer whose passes are registered rather than hand-wired is one where the
//! path tracer is an alternative subgraph instead of a second renderer.
//!
//! The timing is the other half of the argument: shadow cascades are the first
//! feature that multiplies pass count, generating N depth passes that feed one
//! shading pass. Hand-wiring five passes into a fixed pipeline is fine.
//! Hand-wiring fifteen, with the editor depending on frame structure, is the
//! point of no return.
//!
//! # What v1 deliberately does not do
//!
//! - **One queue.** No async compute, no cross-queue semaphores.
//! - **Images and buffers only**, and buffers may only be imported — there is no
//!   transient buffer allocator, because nothing needs one yet.
//! - **No memory aliasing.** Transient images each get their own allocation;
//!   the lifetime information needed to alias them is already in the graph, so
//!   this is an optimisation, not a redesign.
//! - **No resource versioning.** A resource written twice is ordered by
//!   registration order rather than by data flow (see `compile::timeline`).
//! - **Recompiled on structure change, not per frame.** Toggling SSAO or
//!   resizing rebuilds the graph; drawing a different number of objects does
//!   not.
//! - **One escape hatch**, [`PassKind::Raw`], for a pass that owns its own
//!   submission. egui is the only user.
//!
//! # What the plan is, and is not, today
//!
//! vulkano's `AutoCommandBufferBuilder` tracks resource state itself and emits
//! the barriers it derives, so on the current backend the graph's plan and
//! vulkano's are two derivations of the same thing and vulkano's is the one the
//! driver sees. What the graph *does* own outright is execution order, resource
//! creation (an image gets exactly the usage flags its passes declared, so a
//! missing declaration is a creation-time failure rather than a silent one), and
//! the plan CI asserts. When the backend eventually records into raw command
//! buffers, the plan becomes the barriers themselves and nothing above this line
//! changes.

mod access;
mod builder;
mod compile;
mod error;
mod plan;
mod resource;

pub use access::Access;
pub use builder::{GraphBuilder, PassBuilder, PassId, PassKind};
pub use compile::{compile, FrameGraph, TransientImage};
pub use error::GraphError;
pub use plan::Barrier;
pub use resource::{Extent, ImageDesc, ImportedLayouts, ResourceId};

use builder::PassDecl;
use resource::{ResourceDecl, ResourceKind};

#[cfg(test)]
mod tests;
