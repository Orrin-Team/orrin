// The cascaded shadow lookup, once — see `cascades.glsl` for why there is one
// copy shared by the surfaces and the fog.
//
// Included *after* the includer's uniform block, with `ORRIN_CASCADES` defined
// as that block's `Cascades` member, and read through it in place. Not handed a
// `Cascades` value, which is what this used to take: a struct of four `mat4`s
// passed by value and then indexed by a runtime cascade number cannot stay in
// registers, so the compiler copied all 304 bytes into stack memory on every
// call — measured at a third of the forward pass's cost on an M5 Pro (1.18 ms
// against 0.82 ms with identical output). Indexing the block itself is a
// constant-buffer load.

#ifndef ORRIN_CASCADE_LOOKUP_GLSL
#define ORRIN_CASCADE_LOOKUP_GLSL

#ifndef ORRIN_CASCADES
#error "define ORRIN_CASCADES as the includer's Cascades block member before including cascade_lookup.glsl"
#endif

#include "shadow_filter.glsl"

// The cascade a point is routed to: the first whose far distance it is nearer
// than. Distance is radial rather than view-space depth, which costs a little
// over-coverage at the frustum corners and buys rotation invariance — the same
// property the sphere fit on the CPU side is built around.
int select_cascade(float view_dist) {
    int count = int(ORRIN_CASCADES.params.x);
    for (int i = 0; i < count; ++i) {
        if (view_dist < ORRIN_CASCADES.splits[i]) {
            return i;
        }
    }
    return count - 1;
}

// Filtered visibility from one of the sun's views. 1.0 is lit.
float cascade_shadow(
    texture2D atlas,
    samplerShadow cmp,
    int cascade,
    vec3 world_pos,
    vec3 N,
    vec3 L
) {
    return shadow_view_visibility(
        atlas,
        cmp,
        ORRIN_CASCADES.view_proj[cascade],
        ORRIN_CASCADES.rects[cascade],
        ORRIN_CASCADES.texel_sizes[cascade],
        ORRIN_CASCADES.filtering.xy,
        world_pos,
        N,
        L
    );
}

// Sun visibility, blended across the cascade seam and scaled by the strength
// slider. The early-outs a *surface* can take — facing away from the sun, or the
// slider at zero — belong to the caller, because only one of the two callers has
// a normal to test.
float cascade_sun_shadow(
    texture2D atlas,
    samplerShadow cmp,
    vec3 world_pos,
    vec3 N,
    vec3 L,
    float view_dist
) {
    int count = int(ORRIN_CASCADES.params.x);
    if (count <= 0) {
        return 1.0;
    }

    int cascade = select_cascade(view_dist);
    float shadow = cascade_shadow(atlas, cmp, cascade, world_pos, N, L);

    // The next cascade only has depth slightly before its own near plane, and
    // that overlap is what `cascades()` widened each slice by — so the blend
    // band has to be the same fraction or it fades into a region with no data.
    float split = ORRIN_CASCADES.splits[cascade];
    float band = split * ORRIN_CASCADES.params.y;
    if (cascade + 1 < count && view_dist > split - band) {
        float t = clamp((view_dist - (split - band)) / max(band, 1e-4), 0.0, 1.0);
        shadow = mix(shadow, cascade_shadow(atlas, cmp, cascade + 1, world_pos, N, L), t);
    }

    return mix(1.0, shadow, ORRIN_CASCADES.params.z);
}

#endif // ORRIN_CASCADE_LOOKUP_GLSL
