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
// — and takes the maps and their comparison sampler as parameters, so the two
// callers can keep them at different sets and bindings.

#ifndef ORRIN_CASCADES_GLSL
#define ORRIN_CASCADES_GLSL

// Keep in sync with MAX_CASCADES in gfx/shadows.rs.
const int MAX_CASCADES = 4;

// Laid out to match the same four fields in `GpuLighting` and `GpuFog`, in that
// order. std140 gives a struct of `mat4[4]` and three `vec4`s the same offsets it
// gives those fields written out flat, which is what lets one Rust struct feed a
// shader that nests them and another that does not.
struct Cascades {
    mat4 view_proj[MAX_CASCADES];
    // Per-cascade far distance, radial from the camera.
    vec4 splits;
    // World size of one shadow texel, per cascade. It differs per cascade
    // because each fits a different-sized box to the same number of texels.
    vec4 texel_sizes;
    // x = count, y = blend overlap fraction, z = strength, w = debug tint.
    vec4 params;
};

#endif // ORRIN_CASCADES_GLSL
