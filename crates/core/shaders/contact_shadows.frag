#version 460

#include "depth.glsl"

layout(location = 0) in vec2 v_uv;
layout(location = 0) out float f_shadow;

// Bound as the cascades are, so the two halves of the sun's shadow agree about
// which pixels are lit: 1.0 is lit, 0.0 is fully occluded.

// The prepass's camera block. Only the projection halves are read, but `view`
// is declared so the offsets match the buffer the prepass uploads — and the
// projection has to be that pass's jittered one, since the depth this
// reconstructs from was rasterised with it.
layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 proj;
    mat4 inv_proj;
} frame;

layout(set = 0, binding = 1) uniform Params {
    // xyz = view-space direction toward the sun. View space and not world,
    // because the march happens where the depth buffer already is.
    vec4 light_direction;
    // x = ray length, y = min depth separation, z = max depth separation
    // (the bias plus the assumed thickness, summed on the CPU so the two
    // settings cannot close the window between them), w = steps.
    vec4 march;
    // x = intensity, y = fade start, z = fade end, w = frame index.
    vec4 fade;
} p;

layout(set = 1, binding = 0) uniform sampler2D u_depth;  // D32,   nearest + clamp
layout(set = 1, binding = 1) uniform sampler2D u_normal; // RGBA8, nearest + clamp

// A hard cap the loop bound is clamped to, so the compiler knows the trip count
// is bounded even though the count itself is a uniform.
const int MAX_STEPS = 64;

// How much of the frame's edge the shadow fades out across, in UV. Past the
// border the march has no depth to read at all, so a hit found next to it is
// the last one there will be — and without a ramp the shadow ends in a line
// down the side of the frame.
const float EDGE_FADE = 0.08;

vec3 view_pos(vec2 uv, float depth) {
    vec4 c = frame.inv_proj * vec4(uv * 2.0 - 1.0, depth, 1.0);
    return c.xyz / c.w;
}

// Interleaved gradient noise (Jimenez 2014), advanced per frame by the golden
// ratio. Every pixel starts its ray at a different point along the first step,
// which turns the march's quantisation from stair-stepped bands into noise —
// and noise is what the temporal resolve is for.
float dither(vec2 pixel, float frame_index) {
    pixel += frame_index * 5.588238;
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

void main() {
    float depth = texture(u_depth, v_uv).r;
    // Nothing was rasterised here, so there is no surface for the sun to miss.
    if (depth_is_far(depth)) {
        f_shadow = 1.0;
        return;
    }

    vec3 P = view_pos(v_uv, depth);
    vec3 N = normalize(texture(u_normal, v_uv).xyz * 2.0 - 1.0);
    vec3 L = normalize(p.light_direction.xyz);

    // A surface facing away from the sun is already unlit by it, exactly as in
    // `sun_shadow`: the forward pass multiplies this into a term the BRDF has
    // already zeroed. Leaving here saves the whole march on every backface.
    float n_dot_l = dot(N, L);
    if (n_dot_l <= 0.0) {
        f_shadow = 1.0;
        return;
    }

    // Radial distance, matching the cascade selection: turning the camera must
    // not move the fade across the ground.
    float distance_fade = 1.0 - smoothstep(p.fade.y, p.fade.z, length(P));
    if (distance_fade <= 0.0) {
        f_shadow = 1.0;
        return;
    }

    int steps = clamp(int(p.march.w), 1, MAX_STEPS);
    float step_length = p.march.x / float(steps);
    float offset = dither(gl_FragCoord.xy, p.fade.w);

    float occlusion = 0.0;
    for (int i = 0; i < steps; ++i) {
        vec3 sp = P + L * (step_length * (float(i) + offset));

        vec4 clip = frame.proj * vec4(sp, 1.0);
        // Behind the camera, where there is no screen to march across.
        if (clip.w <= 0.0) {
            break;
        }
        vec2 uv = (clip.xy / clip.w) * 0.5 + 0.5;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
            break;
        }

        float scene_depth = texture(u_depth, uv).r;
        // Sky: the ray is passing in front of nothing.
        if (depth_is_far(scene_depth)) {
            continue;
        }

        // Positive means the recorded surface sits between the ray and the
        // camera, which is the only way it can be between this pixel and the
        // sun. The lower bound is what a ray skimming its own surface trips
        // over under a grazing view; the upper bound is the thickness the
        // buffer cannot record, past which the ray is behind the surface
        // rather than under it.
        float delta = view_pos(uv, scene_depth).z - sp.z;
        if (delta > p.march.y && delta < p.march.z) {
            vec2 border = smoothstep(vec2(0.0), vec2(EDGE_FADE), uv)
                        * smoothstep(vec2(0.0), vec2(EDGE_FADE), 1.0 - uv);
            occlusion = border.x * border.y;
            break;
        }
    }

    f_shadow = 1.0 - occlusion * distance_fade * p.fade.x;
}
