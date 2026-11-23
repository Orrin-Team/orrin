// The cascaded shadow lookup, once.
//
// Included by `shading.glsl`, which asks it what a *surface* can see, and by
// `fog_scatter.comp`, which asks it what a point of *air* can see. The file
// exists for the reason `brdf.glsl` and `parallax.glsl` do: a shaft of light in
// the air and the ground it lands on are the same visibility question, and two
// copies that drifted would read as a lighting bug rather than as a mismatch —
// the shaft would miss the shadow it belongs to by a bias.
//
// Nothing here declares a binding. This file is the types; the lookup itself is
// `cascade_lookup.glsl`, which reads the constants straight out of whichever
// uniform block the includer owns — the includer names it with `ORRIN_CASCADES`
// — and takes the atlas and its comparison sampler as parameters, so the two
// callers can keep them at different sets and bindings.

#ifndef ORRIN_CASCADES_GLSL
#define ORRIN_CASCADES_GLSL

// Keep in sync with MAX_CASCADES in gfx/shadows.rs.
const int MAX_CASCADES = 4;

// Laid out to match `GpuCascades`, which `GpuLighting` and `GpuFog` both nest.
// std140 gives a struct of `mat4[4]` and `vec4`s the same offsets it gives those
// fields written out flat, which is what lets one Rust struct feed both.
struct Cascades {
    mat4 view_proj[MAX_CASCADES];
    // Where each view's tile is in the atlas: xy = uv offset, zw = uv scale.
    vec4 rects[MAX_CASCADES];
    // Per-cascade far distance, radial from the camera.
    vec4 splits;
    // World size of one shadow texel, per cascade. It differs per cascade
    // because each fits a different-sized box to the same number of texels.
    vec4 texel_sizes;
    // x = count, y = blend overlap fraction, z = strength, w = which debug view
    // to draw: 0 none, 1 which view, 2 the atlas, 3 texel density, 4 what the
    // frame did to each view.
    vec4 params;
    // x = the sun's penumbra width in metres where a caster touches what it
    // shadows, y = its sharpness, z = the width a metre of caster distance
    // adds. A lamp's own pair travels with its views. See `shadow_filter.glsl`.
    vec4 filtering;
    // What this frame did to each view: 0 nothing, 1 moving casters redrawn,
    // 2 a strip scrolled in, 3 its cached layer redrawn, 4 drawn whole.
    vec4 states;
};

#endif // ORRIN_CASCADES_GLSL
