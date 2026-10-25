#version 460

// The frosted glass under one of the editor's translucent surfaces: a blur of
// everything drawn beneath it, cut to the surface's rounded rectangle. The
// surface's own fill is drawn over this next.
//
// Sampled from a quarter-resolution copy of the frame, which the blits that
// made it have already box-filtered over 4x4 pixels; the 5x5 Gaussian here,
// with bilinear taps between those texels, takes the reach to about twenty
// pixels each way — where text behind a menu stops being legible.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D u_backdrop;

layout(push_constant) uniform Push {
    // The surface, in framebuffer pixels.
    vec2 rect_min;
    vec2 rect_max;
    // One texel of `u_backdrop`, in UV.
    vec2 texel;
    float radius;
} push;

const int REACH = 2;
const float SIGMA = 1.2;

// Signed distance to the surface's outline, negative inside.
float rounded_rect(vec2 p) {
    vec2 centre = 0.5 * (push.rect_min + push.rect_max);
    vec2 half_size = 0.5 * (push.rect_max - push.rect_min) - push.radius;
    vec2 q = abs(p - centre) - half_size;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - push.radius;
}

void main() {
    vec3 sum = vec3(0.0);
    float weight = 0.0;
    for (int y = -REACH; y <= REACH; ++y) {
        for (int x = -REACH; x <= REACH; ++x) {
            float g = exp(-float(x * x + y * y) / (2.0 * SIGMA * SIGMA));
            sum += texture(u_backdrop, v_uv + vec2(x, y) * push.texel).rgb * g;
            weight += g;
        }
    }
    float coverage = clamp(0.5 - rounded_rect(gl_FragCoord.xy), 0.0, 1.0);
    f_color = vec4(sum / weight, coverage);
}
