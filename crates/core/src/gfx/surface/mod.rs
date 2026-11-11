//! Surface shaders: what a `.oshader` file is, how its parameters reach the GPU,
//! and the compiled artifact a game ships instead of a compiler.
//!
//! A surface shader changes what a material's surface *is* — albedo, alpha,
//! metallic, roughness, emission, tangent-space normal — and nothing about how
//! it is lit or drawn. It is one GLSL function stitched into the engine's own
//! pass shaders under `ORRIN_CUSTOM_SURFACE` (see `shaders/surface.glsl`),
//! which hand it the standard reading of the material and keep what it returns
//! within the ranges a material could produce:
//!
//! ```glsl
//! param color tint = (1.0, 0.45, 0.2, 1.0);
//! param float wear = 0.5 : range(0, 1);
//! param texture mask = white;
//!
//! void surface(in SurfaceIn i, inout SurfaceOut o) {
//!     float m = sample_param(mask, i.uv).r * wear;
//!     o.albedo = mix(o.albedo, tint.rgb, m);
//!     o.roughness = mix(o.roughness, 1.0, m);
//! }
//! ```
//!
//! **The grammar** is GLSL plus one line-oriented declaration. A line whose
//! first word is `param`, outside any `{}`, declares a parameter on that line:
//!
//! ```text
//! param <type> <name> [= <default>] [: <hint>, ...];
//!   type     float | vec2 | vec3 | vec4 | color | texture
//!   default  float: 0.5    vecN: (x, y, ...) with N numbers
//!            color: (r, g, b) or (r, g, b, a), linear
//!            texture: white | black | normal
//!   hint     range(min, max)   float only, for the editor's slider
//!            srgb | linear     texture only: how an image dropped on it imports
//! ```
//!
//! Omitted, a default is zero (`float`, `vecN`), opaque white (`color`) or
//! `white` (`texture`), and a texture imports as sRGB unless its default is
//! `normal`. Everything else in the file is GLSL, which must define
//! `void surface(in SurfaceIn i, inout SurfaceOut o)` and may define helpers.
//! `i.time` is seconds; `sample_param(texture, uv)` samples a texture parameter.
//! A file with no parameters and an empty `surface` is the standard material.
//!
//! **What a surface function may not do**, and why — each is an error naming its
//! line:
//!
//! - write `gl_FragDepth`: the forward pass tests `EQUAL` against the depth the
//!   prepass wrote, and both keep early-Z;
//! - `discard`: a cutout lowers `o.alpha` under the material's Masked mode, so
//!   the prepass and the forward pass cut along one line;
//! - declare a `layout`, `uniform` or `buffer`: every pipeline a surface shader
//!   is drawn with shares the built-in pass's layouts plus one buffer, so it
//!   binds nothing of its own;
//! - `#version`, `#extension`, `#include`, `#line`: the pass shader it is
//!   stitched into owns all four, and the last is how an error finds its line.
//!
//! That scan reads tokens outside comments and is the friendly error. The guard
//! is [`compile`]: every variant's binding map must be the built-in shader's
//! plus [`PARAMS_BINDING`], and its SPIR-V may hold no more `OpKill`s and no
//! `FragDepth` the built-in lacks — so a macro that pastes `dis##card` together
//! is caught there. The rule is "plus exactly that buffer", not "plus at most":
//! `resolve_surface` reads the time from it in every variant, so a function that
//! reads nothing still binds it, and a pipeline built for a surface shader binds
//! it unconditionally.
//!
//! **Parameters** live in one storage buffer, `SurfaceParams` at
//! [`PARAMS_BINDING`], of `vec4` words. The first [`FRAME_WORDS`] are the
//! frame's (`x` = time); a material's block starts at the word in its
//! `GpuMaterial::surface.x`, and [`ParamTable::words`] builds one from the
//! shader's defaults and a material's values, reporting rather than failing on a
//! value the shader no longer declares or declares as another type. A block
//! is packed in declaration order: each parameter takes the next `n` components
//! of the current word if they fit, and otherwise starts the next word — no
//! back-filling, so adding a parameter at the end never moves an earlier one. A
//! texture parameter is its bindless index, one component, stored as bits; the
//! block is therefore `[u32; 4]` words and a float is its `to_bits`, so nothing
//! on the way through Rust can canonicalise an index that happens to be a NaN.
//! The generated prelude declares each parameter as a global of its own name
//! and loads it in `surface_parameters(base)`, which `resolve_surface` calls
//! before `surface`. Globals rather than macros: a parameter named `albedo` must
//! not rewrite `o.albedo`. A name the engine's shaders already use at file
//! scope is an error at its line; one only a built-in function has (`mix`) is
//! caught by mapping the compiler's complaint back to the declaration. The
//! user's text follows a `#line` with parameter lines blanked, so every other
//! compiler error lands on the line the user wrote; one that cannot is reported
//! against the file with the compiler's own message.
//!
//! **Variants.** A surface shader compiles once per manifest row whose shader
//! calls `resolve_surface`, less the `ORRIN_FLAT_SURFACE` rows — a custom surface
//! always compiles against every lobe, the flat build existing only to keep
//! unused lobes out of standard materials. [`variants`] derives the list from the
//! manifest, and each variant is keyed by the name of the built-in row it
//! replaces. The masked *shadow* pass does not run the function: a masked custom
//! material casts the shadow of its standard alpha. Its fragment has no world
//! position, normal or eye direction to build a `SurfaceIn` from, and its view is
//! the light's, so a `SurfaceIn` it could build would describe another surface.
//!
//! **The artifact** ([`SurfaceArtifact`]) is what an exported game loads: the
//! parameter table, every variant's SPIR-V and MSL with its binding map, and the
//! [`SOURCES_HASH`](super::shaders::SOURCES_HASH) it was compiled against. Its
//! encoding is hand-written, little-endian and length-prefixed, behind a magic, a
//! format version and an xxh3-64 over the payload that decode checks before it
//! reads a single field. That is what proves SPIR-V read back from disk is the
//! compiler's, byte for byte — on MoltenVK, well-framed SPIR-V that is not can
//! crash the process inside pipeline creation. xxh3 detects corruption, not
//! tampering. Everything here but the compile itself builds without the
//! `shader-compiler` feature.

mod artifact;
#[cfg(feature = "shader-compiler")]
mod compile;
mod params;
mod parse;

pub use artifact::{CompiledVariant, DecodeError, SurfaceArtifact};
#[cfg(feature = "shader-compiler")]
pub use compile::{compile, variants};
pub use params::{
    ColorSpace, DefaultTexture, Ignored, IgnoredReason, Param, ParamKind, ParamTable, ParamValue,
    Slot,
};
pub use parse::{SurfaceError, SurfaceShader};

/// `(set, binding)` of the `SurfaceParams` buffer every surface variant binds,
/// and the only binding it may add to its built-in shader's.
pub const PARAMS_BINDING: (u32, u32) = (5, 2);

/// The words at the start of `SurfaceParams` that belong to the frame rather
/// than to a material, so the first material block starts here.
pub const FRAME_WORDS: u32 = 1;
