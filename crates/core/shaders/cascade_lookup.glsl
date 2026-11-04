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

// Percentage-closer filtered visibility from one cascade. 1.0 is lit.
//
// `N` is the direction the lookup is biased along, which is the surface normal
// for a surface and the light direction itself for a point of air — where there
// is no surface to be shadow-acned against, and pushing toward the light is
// exactly the offset that keeps a froxel out of its own occluder.
float cascade_shadow(
    texture2DArray maps,
    samplerShadow cmp,
    int cascade,
    vec3 world_pos,
    vec3 N,
    vec3 L
) {
    // Normal-offset bias: move the lookup along `N` by about a texel's worth of
    // world space, more at grazing angles where a texel covers the most depth.
    // Offsetting in texture space rather than in depth is what removes acne
    // without the peter-panning a depth offset causes.
    float texel = ORRIN_CASCADES.texel_sizes[cascade];
    float slope = 1.0 - max(dot(N, L), 0.0);
    vec3 p = world_pos + N * texel * 1.4142136 * (1.0 + slope);

    vec4 clip = ORRIN_CASCADES.view_proj[cascade] * vec4(p, 1.0);
    vec3 ndc = clip.xyz / clip.w;
    // Past the cascade's far plane there is nothing to occlude against.
    if (ndc.z > 1.0) {
        return 1.0;
    }

    vec2 uv = ndc.xy * 0.5 + 0.5;
    vec2 step = 1.0 / vec2(textureSize(sampler2DArrayShadow(maps, cmp), 0).xy);

    // 3x3 taps. Each is itself a hardware 2x2 comparison — the compare happens
    // before the bilinear filter — so this is effectively a 4x4 kernel.
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 offset = vec2(x, y) * step;
            sum += texture(
                sampler2DArrayShadow(maps, cmp),
                vec4(uv + offset, float(cascade), ndc.z)
            );
        }
    }
    return sum / 9.0;
}

// Sun visibility, blended across the cascade seam and scaled by the strength
// slider. The early-outs a *surface* can take — facing away from the sun, or the
// slider at zero — belong to the caller, because only one of the two callers has
// a normal to test.
float cascade_sun_shadow(
    texture2DArray maps,
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
    float shadow = cascade_shadow(maps, cmp, cascade, world_pos, N, L);

    // The next cascade only has depth slightly before its own near plane, and
    // that overlap is what `cascades()` widened each slice by — so the blend
    // band has to be the same fraction or it fades into a region with no data.
    float split = ORRIN_CASCADES.splits[cascade];
    float band = split * ORRIN_CASCADES.params.y;
    if (cascade + 1 < count && view_dist > split - band) {
        float t = clamp((view_dist - (split - band)) / max(band, 1e-4), 0.0, 1.0);
        shadow = mix(shadow, cascade_shadow(maps, cmp, cascade + 1, world_pos, N, L), t);
    }

    return mix(1.0, shadow, ORRIN_CASCADES.params.z);
}

#endif // ORRIN_CASCADE_LOOKUP_GLSL
