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
        vec3(ORRIN_CASCADES.filtering.xy, SHADOW_KERNEL_MIN),
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

#ifdef ORRIN_CONTACT_HARDENING

// Filtered visibility from one of the sun's views, at a width and through a
// narrowest kernel of the caller's.
float cascade_shadow_at(
    texture2D atlas,
    samplerShadow cmp,
    int cascade,
    float width,
    float narrowest,
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
        vec3(width, ORRIN_CASCADES.filtering.y, narrowest),
        world_pos,
        N,
        L
    );
}

// How far over a surface its casters are, in metres, as the sun's views from
// `nearest` outward hold them; negative where none of them holds one.
//
// Each coarser view's window is as many texels and so reaches further to the
// side, which is where the caster of a wide penumbra is. The finest view that
// sees a caster says how far, a coarser one only for the share of its window
// the finer ones found empty -- so the answer follows the nearest edge and
// hands over to the next view as that edge leaves the window.
//
// `Ng` is the geometric normal, for the plane the casters are measured from.
float cascade_caster_height(
    texture2D atlas,
    samplerShadow cmp,
    int nearest,
    vec3 world_pos,
    vec3 N,
    vec3 Ng,
    vec3 L
) {
    int count = int(ORRIN_CASCADES.params.x);
    vec2 atlas_size = vec2(textureSize(atlas, 0));

    // The sun's views are one rotation at different scales, so what does not
    // depend on the scale is found once: the point, the direction the lookup
    // is moved along and the plane's normal, each in the light's own frame.
    // Read out of the block a component at a time: a view's matrix copied to a
    // local by a number known only at run time is sixteen floats on the stack,
    // which is the cost the top of this file describes paying once already.
#define ORRIN_VIEW_ROW(view, row) \
    vec3( \
        ORRIN_CASCADES.view_proj[view][0].row, \
        ORRIN_CASCADES.view_proj[view][1].row, \
        ORRIN_CASCADES.view_proj[view][2].row \
    )
    vec3 axis_x = normalize(ORRIN_VIEW_ROW(nearest, x));
    vec3 axis_y = normalize(ORRIN_VIEW_ROW(nearest, y));
    vec3 axis_z = normalize(ORRIN_VIEW_ROW(nearest, z));
    vec3 point = vec3(dot(axis_x, world_pos), dot(axis_y, world_pos), dot(axis_z, world_pos));
    vec3 along = vec3(dot(axis_x, N), dot(axis_y, N), dot(axis_z, N))
        * shadow_normal_offset(SHADOW_KERNEL_MIN, 1.0, dot(N, L));
    vec3 plane = vec3(dot(axis_x, Ng), dot(axis_y, Ng), dot(axis_z, Ng));
    // Held short of edge-on, where the plane is every depth at once.
    vec2 tilt = -plane.xy / (plane.z < 0.0 ? min(plane.z, -0.1) : max(plane.z, 0.1));

    // What of the windows is still unaccounted for, finest first, and the
    // casters' height over the plane weighed by the share each view claimed.
    float unclaimed = 1.0;
    float claimed = 0.0;
    float height = 0.0;
    for (int i = nearest; i < count; ++i) {
        float texel = ORRIN_CASCADES.texel_sizes[i];
        vec4 rect = ORRIN_CASCADES.rects[i];
        // Clip space for a metre along each of the light's axes.
        vec3 scale = vec3(
            dot(ORRIN_VIEW_ROW(i, x), axis_x),
            dot(ORRIN_VIEW_ROW(i, y), axis_y),
            dot(ORRIN_VIEW_ROW(i, z), axis_z)
        );
        vec3 clip = point * scale + ORRIN_CASCADES.view_proj[i][3].xyz;
        // Centred where the narrowest kernel would be read, so that finding
        // nothing in a window is the filter finding nothing either.
        vec3 centre = clip + along * scale * texel;
        if (centre.z > 1.0 || any(greaterThan(abs(centre.xy), vec2(1.0)))) {
            continue;
        }

        ShadowWindow window;
        window.atlas_size = atlas_size;
        window.lo = floor(rect.xy * atlas_size + 0.5);
        window.hi = window.lo + floor(rect.zw * atlas_size + 0.5) - 1.0;
        window.at = (rect.xy + (centre.xy * 0.5 + 0.5) * rect.zw) * atlas_size - 0.5;
        window.on_plane = (rect.xy + (clip.xy * 0.5 + 0.5) * rect.zw) * atlas_size - 0.5;
        window.depth = clip.z;
        window.slope = tilt * texel * scale.z;
        window.span = 1.0 / scale.z;

        if (!shadow_window_covered(atlas, cmp, window, texel)) {
            continue;
        }
        vec2 casters = shadow_window_casters(atlas, window, texel);
        float share = unclaimed * casters.x;
        height += share * casters.y;
        claimed += share;
        unclaimed -= share;
        // A window covered whole leaves a coarser view nothing to add.
        if (unclaimed < 0.004) {
            break;
        }
    }
#undef ORRIN_VIEW_ROW
    return claimed > 0.0 ? height / claimed : -1.0;
}

// Sun visibility for a surface, with a penumbra as wide as its caster is far:
// sharp where a shadow meets what casts it and soft a long way under it.
//
// Two questions, asked of the same views. First how far above the surface its
// casters are, which is `cascade_caster_height`. Then the filter, at the width
// that distance gives, from the finest view whose kernel holds it. A kernel is
// at most `SHADOW_KERNEL_MAX` texels, so a penumbra wider than that in one
// view's texels is read from the next, whose texels are larger: the views are
// a pyramid for the width as well as for the distance.
//
// Most of a frame never reaches the filter: a point no window has a caster
// over is lit, and the windows alone say so. The other end has no such
// shortcut. Every window covered is not the filter reading dark, because a
// window is held to the surface's plane and the plane runs on past the
// surface: the ground a wall stands on is over the wall's plane, and the foot
// of a sunlit wall was drawn black for it. Asking the comparisons themselves
// whether every view is dark all round was measured and is not here either:
// it has to hold as far out as the widest penumbra reaches, which in a scene
// of metre-wide shadows it almost never does, and it cost every shadowed
// pixel the taps to find that out.
float cascade_sun_shadow_hardened(
    texture2D atlas,
    samplerShadow cmp,
    vec3 world_pos,
    vec3 N,
    vec3 Ng,
    vec3 L,
    float view_dist
) {
    int count = int(ORRIN_CASCADES.params.x);
    if (count <= 0) {
        return 1.0;
    }
    int nearest = select_cascade(view_dist);

    // The width where a caster touches what it shadows, which with a sun of no
    // size is the width everywhere.
    float width = ORRIN_CASCADES.filtering.x;
    float spread = ORRIN_CASCADES.filtering.z;
    if (spread > 0.0) {
        float over = cascade_caster_height(atlas, cmp, nearest, world_pos, N, Ng, L);
        if (over < 0.0) {
            return 1.0;
        }
        // What the distance adds, as two blurs one after the other add. No
        // wider than the coarsest view's window reaches: past that there is a
        // caster the search cannot see, and a kernel that found it anyway
        // would end the penumbra on a line.
        float reach = spread * over;
        width = min(
            sqrt(width * width + reach * reach),
            SHADOW_KERNEL_MIN * ORRIN_CASCADES.texel_sizes[count - 1]
        );
    }

    int cascade = nearest;
    while (cascade + 1 < count
        && width > SHADOW_KERNEL_MAX * ORRIN_CASCADES.texel_sizes[cascade]) {
        ++cascade;
    }
    float shadow =
        cascade_shadow_at(atlas, cmp, cascade, width, SHADOW_KERNEL_SMOOTH, world_pos, N, L);

    if (cascade + 1 < count) {
        // Handed to the next view over the last quarter of what this one's
        // kernel holds, so the change of texels is a blend and not an edge.
        float full = SHADOW_KERNEL_MAX * ORRIN_CASCADES.texel_sizes[cascade];
        float t = smoothstep(0.75 * full, full, width);
        // And across the seam in distance, as the plain lookup is.
        if (cascade == nearest) {
            float split = ORRIN_CASCADES.splits[cascade];
            float band = split * ORRIN_CASCADES.params.y;
            t = max(t, clamp((view_dist - (split - band)) / max(band, 1e-4), 0.0, 1.0));
        }
        if (t > 0.0) {
            shadow = mix(
                shadow,
                cascade_shadow_at(
                    atlas,
                    cmp,
                    cascade + 1,
                    width,
                    SHADOW_KERNEL_SMOOTH,
                    world_pos,
                    N,
                    L
                ),
                t
            );
        }
    }

    return mix(1.0, shadow, ORRIN_CASCADES.params.z);
}

#endif // ORRIN_CONTACT_HARDENING

#endif // ORRIN_CASCADE_LOOKUP_GLSL
