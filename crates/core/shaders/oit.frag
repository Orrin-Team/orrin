#version 460

// Weighted-blended order-independent transparency, accumulation half
// (McGuire & Bavoil 2013).
//
// Nothing here is sorted, and nothing needs to be: the two targets below are
// written with blend equations that are *commutative*, so the same pixel comes
// out whatever order the triangles arrive in. `accum` sums premultiplied
// radiance and coverage under a depth-dependent weight; `reveal` multiplies up
// the transmittance, one minus each surface's alpha. `oit_composite.comp`
// divides the first by its own coverage and mixes by the second.
//
// The price of that is exactness: two surfaces at very different depths are
// resolved by a heuristic weight rather than by their real ordering. What it
// buys is no sort, no popping when two surfaces cross, and one draw per
// (mesh, material) run exactly as the opaque pass gets.

// Blended, so the two screen-space terms in `shading.glsl` do not apply — see
// the comment there.
#define ORRIN_TRANSPARENT 1
#include "shading.glsl"

layout(location = 0) out vec4 f_accum;
layout(location = 1) out float f_reveal;

// The paper's weight function (equation 10): near-opaque and near-camera
// fragments dominate, distant near-transparent ones barely register. Written,
// as the paper writes it, in forward-Z depth, which the reversed buffer gives
// back as `1 - depth` exactly — so the weights are the paper's own.
//
// The bounds matter more than the curve. `accum` is a 16-bit float target, so
// the weight has to stay inside a range whose products do not saturate — the
// clamp is what keeps a pile of glass from resolving to Inf and painting the
// pixel white.
float oit_weight(float depth, float alpha) {
    float coverage = min(1.0, alpha * 10.0) + 0.01;
    float nearness = 1.0 - (1.0 - depth) * 0.9;
    return clamp(coverage * coverage * coverage * 1e8
               * nearness * nearness * nearness, 1e-2, 3e3);
}

void main() {
    Shaded shaded = shade_surface();
    // Summed: this queue never reaches the diffusion passes — they read the
    // prepass, and blended geometry is not in it — so `shading.glsl` has given
    // the diffusible half its wrapped diffuse instead and there is nothing to
    // keep apart.
    vec4 surface = vec4(shaded.color + shaded.diffusible, shaded.alpha);
    float alpha = surface.a;

    // A fully transparent fragment contributes nothing to either target, and
    // running it through the weight would only feed a zero into the divide the
    // composite guards anyway.
    if (alpha <= 0.0) {
        discard;
    }

    float w = oit_weight(gl_FragCoord.z, alpha);

    // Premultiplied: the composite divides `rgb` by `a` to recover the average
    // colour, so the weight cancels and only the coverage-weighted mean is
    // left.
    f_accum = vec4(surface.rgb * alpha, alpha) * w;
    // Written into a target whose blend is `dst *= (1 - src)`, which is what
    // makes this a running product of transmittance rather than a sum.
    f_reveal = alpha;
}
