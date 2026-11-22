// The shadow filter, once: a tent-weighted percentage-closer kernel built from
// bilinear comparison taps, sized in metres rather than in texels.
//
// Included by `cascade_lookup.glsl` and by `shading.glsl`'s atlas lookup, for
// the reason `brdf.glsl` exists: the sun's shadow and a lamp's are one question
// asked of two maps, and two kernels that drifted would read as two lights of
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

// The kernel's radius in texels for a penumbra `softness` metres wide over
// texels `texel` metres across.
float shadow_kernel_radius(float softness, float texel) {
    return clamp(softness / max(texel, 1e-6), SHADOW_KERNEL_MIN, SHADOW_KERNEL_MAX);
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
// caller's comparison at a position in texels; `base`, `offsets_x`,
// `weights_x`, `offsets_y` and `weights_y` are the names it must have used.
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
// was and keeps the kernel's gradient while less of it is visible. Zero is the
// kernel as filtered.
float shadow_sharpen(float visibility, float sharpness) {
    return clamp((visibility - 0.5) / max(1.0 - sharpness, 0.05) + 0.5, 0.0, 1.0);
}

#endif // ORRIN_SHADOW_FILTER_GLSL
