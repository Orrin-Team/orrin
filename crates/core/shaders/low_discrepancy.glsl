// One well-spread sample a pixel a frame, for a pass that draws a single
// stochastic sample per pixel and leaves a spatial and a temporal filter to
// average them.
//
// The R2 sequence over frames (Roberts 2018), Cranley-Patterson rotated by a
// hash of the pixel. Each pixel's frames fill the unit square evenly at every
// prefix, so its history averages an even spread rather than a random one;
// and the rotation is uncorrelated between neighbours, so what error is left
// in a pixel's mean is white across the image, which is what a spatial blur
// takes out. Computed, not looked up: nothing to generate, ship, load or bind.
//
// Not interleaved gradient noise for the rotation (Jimenez 2014), though its
// neighbours are better spread: its rotation is a diagonal ramp across the
// screen, and the error each pixel's mean is left with follows it, as a weave
// over every flat surface that no number of frames removes. Not a
// spatiotemporal blue-noise mask (Wolfe et al. 2022) or Heitz et al.'s ranked
// Sobol (2019) either: both are tables made offline, to ship and bind.

#ifndef ORRIN_LOW_DISCREPANCY_GLSL
#define ORRIN_LOW_DISCREPANCY_GLSL

uint ld_hash(uint x) {
    uint state = x * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// The sample pixel `pixel` draws in frame `frame`, in [0, 1)^2. In fixed
// point, so it is exact however many frames have passed.
vec2 ld_sample(uvec2 pixel, uint frame) {
    const uvec2 R2 = uvec2(3242174889u, 2447445413u);
    uint a = ld_hash(pixel.x ^ ld_hash(pixel.y));
    uvec2 rotation = uvec2(a, ld_hash(a));
    return vec2(frame * R2 + rotation) * (1.0 / 4294967296.0);
}

// A direction about `n` drawn in proportion to the cosine, from a sample in
// the unit square. The frame round `n` is Duff et al.'s (2017), which has no
// branch and no singularity.
vec3 ld_cosine_direction(vec3 n, vec2 u) {
    float r = sqrt(u.x);
    float phi = 6.28318530718 * u.y;
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    vec3 t = vec3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    vec3 bt = vec3(b, s + n.y * n.y * a, -n.y);
    return t * (r * cos(phi)) + bt * (r * sin(phi)) + n * sqrt(max(1.0 - u.x, 0.0));
}

#endif
