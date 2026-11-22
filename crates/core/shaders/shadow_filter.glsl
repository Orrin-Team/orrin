// The shadow filter, once: a tent-weighted percentage-closer kernel built from
// bilinear comparison taps, sized in metres rather than in texels.
//
// Included by `cascade_lookup.glsl`, which asks it about the sun, and by
// `shading.glsl`, which asks it about the lamps, for the reason `brdf.glsl`
// exists: the sun's shadow and a lamp's are one question asked of two tiles of
// one atlas, and two lookups that drifted would read as two lights of
// different sizes.
//
// A hardware comparison tap placed between two texel centres returns their two
// comparisons blended by where it landed. So one tap can carry any two adjacent
// tent weights exactly, by landing `b / (a + b)` of the way from the first to
// the second -- which is what makes an eight-texel tent four taps an axis and
// not eight. The technique is the one The Witness shipped; its published
// kernels are the integer widths of this one (four taps for 3x3, nine for 5x5,
// sixteen for 7x7).
//
// What differs is that the radius is continuous, because it is derived per
// fragment from a width in metres. A kernel a fixed number of texels wide is a
// penumbra whose width jumps wherever the texel does: at every cascade seam,
// and whenever a light's tile changes size.

#ifndef ORRIN_SHADOW_FILTER_GLSL
#define ORRIN_SHADOW_FILTER_GLSL

// The narrowest the kernel gets, in texels. Under this the tent is a bilinear
// fetch and a shadow's edge shows the map's texels as stairs, so a far cascade
// is as sharp as this and no sharper, whatever width was asked for.
const float SHADOW_KERNEL_MIN = 1.5;
// The widest, which is what eight texels an axis can hold: at this radius the
// outermost texel's weight reaches zero exactly as the kernel steps onto the
// next one. Lock-stepped with `FACE_BORDER_TEXELS` in gfx/punctual.rs, which
// renders that much real depth past a face's edge for these taps to land on.
const float SHADOW_KERNEL_MAX = 4.0;

// The narrowest penumbra a view can show, in texels either side of the edge.
// Narrower than this and the remap below is a threshold, which draws the map's
// texels as stairs however wide the kernel under it was.
const float SHADOW_PENUMBRA_MIN = 0.75;

// The radius a penumbra `softness` metres wide asks for, in texels `texel`
// metres across, before the kernel's limits are applied to it.
float shadow_asked_radius(float softness, float texel) {
    return softness / max(texel, 1e-6);
}

// The kernel's radius in texels for that request.
float shadow_kernel_radius(float asked) {
    return clamp(asked, SHADOW_KERNEL_MIN, SHADOW_KERNEL_MAX);
}

// One axis of the tent. `s` is how far the lookup sits past the texel centre
// below it, in texels. `offsets` are where the four taps land, measured from
// that centre, and `weights` what each is worth; a weight of zero is a tap the
// radius does not reach.
void shadow_tent_axis(float s, float radius, out vec4 offsets, out vec4 weights) {
    // The first texel of each pair. Texel `i` sits `i - s` from the lookup.
    const vec4 first = vec4(-3.0, -1.0, 1.0, 3.0);
    vec4 a = max(1.0 - abs(first - s) / radius, 0.0);
    vec4 b = max(1.0 - abs(first + 1.0 - s) / radius, 0.0);
    weights = a + b;
    offsets = first + b / max(weights, vec4(1e-6));
}

// The kernel's weighted sum, written out tap by tap.
//
// Macros over the caller's locals rather than a loop over the four-vectors:
// indexing a local vector by a loop counter is a dynamic index, which the
// compiler answers by spilling the vector to the stack -- the cost
// `cascade_lookup.glsl` documents paying once already. `FETCH(texel)` is the
// comparison at a position in texels; `base`, `offsets_x`, `weights_x`,
// `offsets_y` and `weights_y` are the locals it is expanded among.
//
// Two halves because a radius of two or less reaches only the inner four taps:
// the outer twelve have a weight of exactly zero and are not fetched.
#define SHADOW_TENT_TAP(FETCH, X, Y) \
    (weights_x.X * weights_y.Y * FETCH(base + 0.5 + vec2(offsets_x.X, offsets_y.Y)))
#define SHADOW_TENT_INNER(FETCH) \
    (SHADOW_TENT_TAP(FETCH, y, y) + SHADOW_TENT_TAP(FETCH, z, y) \
    + SHADOW_TENT_TAP(FETCH, y, z) + SHADOW_TENT_TAP(FETCH, z, z))
#define SHADOW_TENT_OUTER(FETCH) \
    (SHADOW_TENT_TAP(FETCH, x, x) + SHADOW_TENT_TAP(FETCH, y, x) \
    + SHADOW_TENT_TAP(FETCH, z, x) + SHADOW_TENT_TAP(FETCH, w, x) \
    + SHADOW_TENT_TAP(FETCH, x, y) + SHADOW_TENT_TAP(FETCH, w, y) \
    + SHADOW_TENT_TAP(FETCH, x, z) + SHADOW_TENT_TAP(FETCH, w, z) \
    + SHADOW_TENT_TAP(FETCH, x, w) + SHADOW_TENT_TAP(FETCH, y, w) \
    + SHADOW_TENT_TAP(FETCH, z, w) + SHADOW_TENT_TAP(FETCH, w, w))
// Where the outer taps start to weigh anything.
const float SHADOW_KERNEL_INNER = 2.0;

// How far along the normal a lookup is moved before it is projected, in metres.
//
// A kernel `radius` texels wide reads depth across that much of a surface, and
// on a surface tilted away from the light that span is a range of depths: the
// far side of the kernel sits behind the near side and shadows it. Moving the
// lookup `d` along the normal clears a plane tilted by theta when
// `d >= width * sin(theta)`, so the offset grows with the kernel and with the
// tilt rather than being a constant number of texels. The floor is what covers
// the map's own quantisation where the surface faces the light squarely.
float shadow_normal_offset(float radius, float texel, float n_dot_l) {
    float sin_theta = sqrt(clamp(1.0 - n_dot_l * n_dot_l, 0.0, 1.0));
    return texel * (radius + 1.0) * max(sin_theta, 0.25);
}

// Narrow the visible penumbra without narrowing the kernel.
//
// A wide kernel is what antialiases a shadow's edge, and it is also a wide
// penumbra, which under a caster's foot reads as light leaking beneath it. This
// stretches the filtered result about its midpoint, so the edge stays where it
// was and keeps the kernel's gradient while less of it is visible.
//
// Two things ask for it. Where a view's texels are coarser than the penumbra
// asked for, the kernel is held at `SHADOW_KERNEL_MIN` and is wider than was
// asked; the stretch takes the visible width back toward the request, down to
// `SHADOW_PENUMBRA_MIN` -- which is what keeps a far view's shadow from being
// several times softer than the near view's it meets at a seam. And
// `sharpness` narrows whatever is left by an authored factor.
float shadow_sharpen(float visibility, float asked, float radius, float sharpness) {
    float contrast = radius / clamp(asked, SHADOW_PENUMBRA_MIN, radius);
    contrast /= max(1.0 - sharpness, 0.05);
    return clamp((visibility - 0.5) * contrast + 0.5, 0.0, 1.0);
}

// Filtered visibility of a point from one shadow view. 1.0 is lit.
//
// One function for a cascade, a sun level and a face of a lamp, because to a
// lookup they differ in nothing but numbers: `view_proj` is the matrix the
// view was drawn with, `rect` its tile of the atlas as a uv offset and scale,
// and `texel_scale` the world size of one of its texels where clip-space `w`
// is one. An orthographic view's `w` is one everywhere and a perspective
// face's is the depth along its axis, so multiplying by it is the texel's size
// at the lookup either way. `filtering` is the penumbra's width in metres and
// its sharpness.
//
// `N` is the direction the lookup is biased along: the surface normal for a
// surface, and the light direction itself for a point of air -- where there is
// no surface to be shadow-acned against, and pushing toward the light is
// exactly the offset that keeps a froxel out of its own occluder.
float shadow_view_visibility(
    texture2D atlas,
    samplerShadow cmp,
    mat4 view_proj,
    vec4 rect,
    float texel_scale,
    vec2 filtering,
    vec3 world_pos,
    vec3 N,
    vec3 L
) {
    float w = view_proj[0].w * world_pos.x + view_proj[1].w * world_pos.y
        + view_proj[2].w * world_pos.z + view_proj[3].w;
    if (w <= 0.0) {
        return 1.0;
    }
    float texel = texel_scale * w;
    float asked = shadow_asked_radius(filtering.x, texel);
    float radius = shadow_kernel_radius(asked);
    // Offsetting in world space rather than in depth is what removes acne
    // without the peter-panning a depth offset causes.
    vec4 clip = view_proj
        * vec4(world_pos + N * shadow_normal_offset(radius, texel, dot(N, L)), 1.0);
    if (clip.w <= 0.0) {
        return 1.0;
    }
    vec3 ndc = clip.xyz / clip.w;
    // Outside the view there is nothing recorded to occlude against: past a
    // sun view's far plane, past a lamp's range, off the side of either.
    if (ndc.z > 1.0 || any(greaterThan(abs(ndc.xy), vec2(1.0)))) {
        return 1.0;
    }

    vec2 atlas_size = vec2(textureSize(sampler2DShadow(atlas, cmp), 0));
    // Half a texel in from the tile's edge, so that not even a bilinear fetch
    // reaches past it. The tile next door is another view's depth, and a tap
    // that landed there would read as whatever that view happened to see.
    vec2 lo = rect.xy * atlas_size + 0.5;
    vec2 hi = (rect.xy + rect.zw) * atlas_size - 0.5;

    vec2 at = (rect.xy + (ndc.xy * 0.5 + 0.5) * rect.zw) * atlas_size - 0.5;
    vec2 base = floor(at);
    vec4 offsets_x, weights_x, offsets_y, weights_y;
    shadow_tent_axis(at.x - base.x, radius, offsets_x, weights_x);
    shadow_tent_axis(at.y - base.y, radius, offsets_y, weights_y);

#define SHADOW_ATLAS_FETCH(at_texel) \
    textureLod(sampler2DShadow(atlas, cmp), \
        vec3(clamp(at_texel, lo, hi) / atlas_size, ndc.z), 0.0)
    float sum = SHADOW_TENT_INNER(SHADOW_ATLAS_FETCH);
    if (radius > SHADOW_KERNEL_INNER) {
        sum += SHADOW_TENT_OUTER(SHADOW_ATLAS_FETCH);
    }
#undef SHADOW_ATLAS_FETCH
    float total = dot(weights_x, vec4(1.0)) * dot(weights_y, vec4(1.0));
    return shadow_sharpen(sum / total, asked, radius, filtering.y);
}

#endif // ORRIN_SHADOW_FILTER_GLSL
