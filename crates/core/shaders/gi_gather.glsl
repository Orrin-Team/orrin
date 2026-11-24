// What the final gather's passes share: the block they read, and how a texel
// at the gather's resolution stands for a pixel of the frame. `gfx/vulkan/
// gi_gather.rs` is the design, and lays the block out in lock-step.
//
// The includer declares `u_depth` and `u_normal`, the prepass's, as
// `texture2D`s, and defines `ORRIN_GATHER_BLOCK_BINDING`.

#ifndef ORRIN_GI_GATHER_GLSL
#define ORRIN_GI_GATHER_GLSL

#include "depth.glsl"
#include "half_float.glsl"

layout(set = 0, binding = ORRIN_GATHER_BLOCK_BINDING, std140) uniform Gather {
    // This frame's, jittered: what the prepass rasterised with.
    mat4 view_proj;
    mat4 inv_view_proj;
    // View space to world, for the prepass's normals.
    mat4 inv_view;
    // xyz = the camera, w = the most frames the history averages.
    vec4 camera;
    // xyz = last frame's camera.
    vec4 previous_camera;
    // x = how many times the gather's resolution halves the frame's, y = this
    // frame's number, z = `GATHER_*` flags.
    uvec4 params;
} gather;

// Last frame's lit colour can be read, and so can the history.
const uint GATHER_LIT = 1u;
const uint GATHER_HISTORY = 2u;

// Layer 3 of the raw and accumulated images, beside the three of spherical
// harmonics: x = frames averaged, y = metres to the camera when written,
// z = how far the rays went, w = how far the luminance of the nine rays round
// the texel moves from one frame to the next.
const int GATHER_META = 3;

// Layer 4 of the accumulation, for the test of whether a light was switched:
// x = a short mean of the nine rays' luminance, y = how far it strays, negative
// where it was over the line last frame, z = frames since the count last
// restarted, w = the nine rays' luminance last frame.
const int GATHER_STATS = 4;

// The longest a ray's distance counts for, in metres: past this one open
// view is as open as another.
const float GATHER_REACH = 64.0;

const vec3 GATHER_LUMA = vec3(0.2126, 0.7152, 0.0722);

// The pixel of the frame a texel at the gather's resolution stands for: the
// first of its block, always the same one, so a texel's history is of one
// surface.
ivec2 gather_texel(ivec2 low, ivec2 full) {
    return min(low << int(gather.params.x), full - 1);
}

// A point of the frame, in pixels, in the gather's texel coordinates: texel
// `i` is at `i`, where the centre of the pixel it stands for is.
vec2 gather_coordinate(vec2 pixel) {
    return (pixel - 0.5) / float(1u << gather.params.x);
}

vec3 gather_position(ivec2 texel, float depth, ivec2 full) {
    vec2 uv = (vec2(texel) + 0.5) / vec2(full);
    vec4 world = gather.inv_view_proj * vec4(uv * 2.0 - 1.0, depth, 1.0);
    return world.xyz / world.w;
}

vec3 gather_normal(ivec2 texel) {
    vec3 n = texelFetch(u_normal, texel, 0).xyz * 2.0 - 1.0;
    return normalize(mat3(gather.inv_view) * n);
}

// How far `other` lies off the plane of the surface at `p` facing `n`, against
// what still counts as the same surface `far` metres from the camera.
float gather_plane(vec3 p, vec3 n, vec3 other, float far) {
    return abs(dot(n, other - p)) / (0.01 + 0.02 * far);
}

#endif
