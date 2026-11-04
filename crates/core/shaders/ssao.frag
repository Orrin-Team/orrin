#version 460

// Ground-truth-based ambient occlusion: the horizon search of Jiménez et al.,
// "Practical Real-Time Strategies for Accurate Indirect Occlusion" (2016), in
// the form XeGTAO ships.
//
// Not a hemisphere kernel. The point-sampled kernel this pass used to run spent
// 32 taps a pixel on random view-space points, each projected through a mat4 to
// a scattered, dependent depth fetch, and estimated the occlusion by counting
// how many landed behind the depth buffer. A horizon search marches *along* a
// few screen-space lines instead: per slice it finds the highest occluder on
// each side of the pixel, and the cosine-weighted visibility between those two
// horizons has a closed form. Two slices of three steps a side — twelve
// fetches, each line of them coherent — measure the integral the kernel only
// sampled.
//
// The slices are rotated per pixel by the 4x4 noise tile, whose rotations are
// stratified (see `noise_texels` in `ssao.rs`), and the bilateral blur after
// this pass has the same 4x4 footprint, so the blurred result has averaged
// thirty-two evenly spaced directions.
//
// Each pixel's result is its slices' visibility *over their unoccluded
// visibility*, not over the slice count. Two slices of a surface seen at a
// glancing angle integrate to up to a tenth either side of one even with
// nothing in the way, and the R8 target clamps at one before the blur can
// average that out — so only the low side would survive, and open surfaces
// would darken with tilt in a pattern that follows their normals. The ratio is
// exactly one for an open surface at every pixel, and costs two closed-form
// evaluations and no fetches.

#include "depth.glsl"

layout(location = 0) in vec2 v_uv;
layout(location = 0) out float f_ao;

layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 proj;
    mat4 inv_proj;
} frame;

layout(set = 0, binding = 1) uniform Params {
    vec2  noise_scale; // the AO target's extent / 4, since the noise is 4x4
    float radius;      // the search radius, in metres
    float bias;        // metres along the normal the horizons are measured from
    float power;       // contrast on the result
} p;

layout(set = 1, binding = 0) uniform sampler2D u_depth;  // D32,   nearest + clamp
layout(set = 1, binding = 1) uniform sampler2D u_normal; // RGBA8, nearest + clamp
layout(set = 1, binding = 2) uniform sampler2D u_noise;  // RGBA8, nearest + repeat

// Lock-stepped with `SLICES` in `ssao.rs`.
const int SLICES = 2;
const int STEPS = 3;
const float PI = 3.14159265359;
const float HALF_PI = 1.57079632679;
// The outer 61.5% of the radius fades a sample's horizon back toward the
// tangent plane, so an occluder crossing the radius edge does not pop. XeGTAO's
// value.
const float FALLOFF_RANGE = 0.615;
// The nearest a step may land, in pixels: closer than this it reads the centre
// texel's own depth back and measures the surface against itself.
const float MIN_STEP_PIXELS = 1.3;

vec3 view_pos(vec2 uv) {
    float d = texture(u_depth, uv).r;
    vec4 c = frame.inv_proj * vec4(uv * 2.0 - 1.0, d, 1.0);
    return c.xyz / c.w;
}

// The normal, projected into the plane of one slice: its signed angle `n` from
// the view vector, that angle's cosine, and the projection's length, which
// weights the slice by how much of the normal it actually contains. Mirrored by
// `shader_slice_frame` in `ssao.rs`.
struct Slice {
    float n;
    float cos_n;
    float len;
};

Slice slice_frame(vec3 N, vec3 V, vec3 direction) {
    vec3 ortho = direction - dot(direction, V) * V;
    vec3 axis = normalize(cross(ortho, V));
    vec3 projected = N - axis * dot(N, axis);
    float len = length(projected);
    float cos_n = clamp(dot(projected, V) / max(len, 1e-6), 0.0, 1.0);
    float n = (dot(ortho, projected) >= 0.0 ? 1.0 : -1.0) * acos(cos_n);
    return Slice(n, cos_n, len);
}

// The cosine-weighted visibility between the two horizons, in closed form.
// `hc0` is the cosine of the horizon on the `+direction` side, `hc1` on the
// other. Each is clamped to the hemisphere about the normal, since nothing
// below the surface can occlude it. Mirrored by `shader_slice_visibility` in
// `ssao.rs`.
float slice_visibility(Slice s, float hc0, float hc1) {
    float h0 = s.n + clamp(-acos(clamp(hc1, -1.0, 1.0)) - s.n, -HALF_PI, HALF_PI);
    float h1 = s.n + clamp(acos(clamp(hc0, -1.0, 1.0)) - s.n, -HALF_PI, HALF_PI);
    float sin_n = sin(s.n);
    float a0 = (s.cos_n + 2.0 * h0 * sin_n - cos(2.0 * h0 - s.n)) * 0.25;
    float a1 = (s.cos_n + 2.0 * h1 * sin_n - cos(2.0 * h1 - s.n)) * 0.25;
    return s.len * (a0 + a1);
}

void main() {
    // The sky occludes nothing, and has no surface to be occluded. One coherent
    // branch, taken by whole waves across open sky.
    float d = texture(u_depth, v_uv).r;
    if (depth_is_far(d)) {
        f_ao = 1.0;
        return;
    }

    vec3 P = view_pos(v_uv);
    vec3 V = normalize(-P);
    vec3 N = normalize(texture(u_normal, v_uv).xyz * 2.0 - 1.0);
    vec2 noise = texture(u_noise, v_uv * p.noise_scale).xy;

    // The radius as a UV offset at this depth. `proj[1][1]` is negative because
    // `Camera::projection` flips Y, and its magnitude is the scale.
    vec2 radius_uv = p.radius * 0.5 * vec2(frame.proj[0][0], abs(frame.proj[1][1])) / -P.z;
    float radius_px = radius_uv.x * p.noise_scale.x * 4.0;
    // Under a couple of pixels every step lands on the centre texel or its
    // neighbour: there is no horizon to find, only the surface's own depth.
    if (radius_px < 2.0 * MIN_STEP_PIXELS) {
        f_ao = 1.0;
        return;
    }
    float min_t = MIN_STEP_PIXELS / radius_px;

    vec3 origin = P + N * p.bias;
    float falloff_mul = -1.0 / (FALLOFF_RANGE * p.radius);
    float falloff_add = (1.0 - FALLOFF_RANGE) / FALLOFF_RANGE + 1.0;

    float visibility = 0.0;
    float open_visibility = 0.0;
    for (int s = 0; s < SLICES; ++s) {
        float phi = (float(s) + noise.x) * PI / float(SLICES);
        vec3 direction = vec3(cos(phi), sin(phi), 0.0);
        // View +Y is up the screen and UV +Y is down it.
        vec2 step_uv = vec2(direction.x, -direction.y) * radius_uv;

        Slice slice = slice_frame(N, V, direction);
        // Each side starts at the tangent plane: an open surface's horizon.
        float low0 = cos(slice.n + HALF_PI);
        float low1 = cos(slice.n - HALF_PI);
        float hc0 = low0;
        float hc1 = low1;
        for (int j = 0; j < STEPS; ++j) {
            // Quadratic in the step, so the taps crowd toward the centre, where
            // most of the occlusion that matters is.
            float t = (float(j) + noise.y) / float(STEPS);
            t = min_t + t * t * (1.0 - min_t);
            vec2 offset = step_uv * t;

            vec3 d0 = view_pos(v_uv + offset) - origin;
            vec3 d1 = view_pos(v_uv - offset) - origin;
            float l0 = length(d0);
            float l1 = length(d1);
            float w0 = clamp(l0 * falloff_mul + falloff_add, 0.0, 1.0);
            float w1 = clamp(l1 * falloff_mul + falloff_add, 0.0, 1.0);
            hc0 = max(hc0, mix(low0, dot(d0, V) / max(l0, 1e-6), w0));
            hc1 = max(hc1, mix(low1, dot(d1, V) / max(l1, 1e-6), w1));
        }
        visibility += slice_visibility(slice, hc0, hc1);
        open_visibility += slice_visibility(slice, low0, low1);
    }

    f_ao = pow(clamp(visibility / max(open_visibility, 1e-4), 0.0, 1.0), p.power);
}
