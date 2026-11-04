//! The camera's depth convention, stated once for every pass that rasterises
//! against the camera's depth and every backend that clears it.
//!
//! Reversed: the near plane is 1 and the far plane 0 (see
//! [`Camera::projection`](crate::scene::Camera::projection)). A perspective
//! divide crowds depth values toward the near plane, and a float crowds its
//! representable values toward zero; reversing puts the two crowds at opposite
//! ends, so a D32 float buffer holds roughly constant *relative* precision
//! across the whole range instead of spending it on the first metre.
//!
//! The far plane stays finite. With a reversed float buffer an infinite one buys
//! almost nothing over a far plane of a kilometre, and a finite one keeps
//! `Camera::far` meaning what scenes author it to mean, keeps the frustum's far
//! plane a plane rather than a degenerate row, and keeps every unprojection of
//! the far plane a point rather than a direction.
//!
//! Shadow maps do not use it. A cascade is orthographic, so its depth is already
//! linear and there is nothing for an exponent to cancel, and the punctual atlas
//! is D16 UNORM, which has no exponent at all. Both keep the forward convention,
//! [`SHADOW_CLEAR`] and `Less`, and `cascades.glsl` compares that way.

use crate::gfx::rhi::DepthCompare;

/// The camera's near plane, as a depth-buffer value.
pub const NEAR: f32 = 1.0;
/// The camera's far plane, as a depth-buffer value.
pub const FAR: f32 = 0.0;
/// What a camera depth buffer is cleared to: the far plane.
pub const CLEAR: f32 = FAR;
/// The test that keeps the nearer of two surfaces.
pub const NEARER: DepthCompare = DepthCompare::Greater;
/// [`NEARER`], also accepting a surface exactly at the depth already there —
/// for the sky, which is drawn *at* the clear value.
pub const NEARER_OR_EQUAL: DepthCompare = DepthCompare::GreaterOrEqual;
/// What a shadow map is cleared to, forward-Z: nothing occludes.
pub const SHADOW_CLEAR: f32 = 1.0;
