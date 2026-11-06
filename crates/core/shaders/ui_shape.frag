#version 460

// The in-game UI's styled rectangles. Everything here is evaluated
// analytically per fragment, so a rounded, ringed or blurred box costs no
// texture, no extra pass and nothing per pixel of the screen that it does not
// cover; `gfx::ui` explains the choice.
//
// A shape is a box (centre and half extents, in window pixels) whose corners
// are quarter superellipses: |x/r|^n + |y/r|^n = 1, where n = 2 is a circle and
// a larger n eases the curvature in from the straight edge.

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec2 v_position;
layout(location = 2) flat in vec4 v_shape;
layout(location = 3) flat in vec4 v_params;
layout(location = 4) flat in vec4 v_gradient;
layout(location = 5) flat in vec4 v_color0;
layout(location = 6) flat in vec4 v_color1;

layout(location = 0) out vec4 f_color;

// Keep in sync with MAX_TEXTURES in gfx/mod.rs.
const int MAX_TEXTURES = 192;
layout(set = 0, binding = 0) uniform texture2D u_textures[MAX_TEXTURES];
layout(set = 0, binding = 1) uniform sampler u_sampler;
layout(set = 1, binding = 0) uniform sampler2D u_atlas;

layout(push_constant) uniform Push {
    vec2 screen_size;
    uint output_in_linear_colorspace;
    uint texture_index;
} push;

// `gfx::vulkan::ui::ATLAS`.
const uint ATLAS = 0xffffffffu;
const float PI = 3.14159265358979;

vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 lower = srgb / vec3(12.92);
    vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    return mix(higher, lower, vec3(cutoff));
}

vec3 srgb_from_linear(vec3 linear) {
    bvec3 cutoff = lessThan(linear, vec3(0.0031308));
    vec3 lower = linear * vec3(12.92);
    vec3 higher = vec3(1.055) * pow(linear, vec3(1.0 / 2.4)) - vec3(0.055);
    return mix(higher, lower, vec3(cutoff));
}

// Signed distance in pixels from `p` (relative to the centre) to the box's
// edge, negative inside. For a circle it is exact; for a superellipse the
// corner's norm is divided by its gradient's length, which makes it exact on
// the curve and close within the few pixels an edge or a ring is drawn over.
float box_distance(vec2 p, vec2 half_size, float radius, float n) {
    vec2 q = abs(p) - half_size + radius;
    vec2 corner = max(q, 0.0);
    float inside = min(max(q.x, q.y), 0.0);
    if (n == 2.0 || corner.x == 0.0 || corner.y == 0.0) {
        return length(corner) + inside - radius;
    }
    float norm = pow(pow(corner.x, n) + pow(corner.y, n), 1.0 / n);
    vec2 slope = pow(corner / norm, vec2(n - 1.0));
    return (norm - radius) / length(slope) + inside;
}

// Half the box's width on the row `y` from its centre (|y| <= half height).
float row_half_width(float y, vec2 half_size, float radius, float n) {
    float r = max(radius, 1e-4);
    float into = clamp((abs(y) - (half_size.y - r)) / r, 0.0, 1.0);
    return half_size.x - r + r * pow(1.0 - pow(into, n), 1.0 / n);
}

vec2 erf(vec2 x) {
    // Abramowitz and Stegun 7.1.27, good to 5e-4.
    vec2 s = sign(x);
    vec2 a = abs(x);
    vec2 t = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
    t *= t;
    return s - s / (t * t);
}

// The box convolved with a Gaussian of `sigma`, after Evan Wallace's rounded
// rectangle shadows: across a row the integral is exact (a difference of two
// erfs), and down the rows it is a four-point sum over the three sigmas either
// side, which is what keeps a small box blurred by a large sigma as faint as it
// really is rather than as solid as a blurred distance would say.
float blurred_box(vec2 p, vec2 half_size, float radius, float n, float sigma) {
    float low = p.y - half_size.y;
    float high = p.y + half_size.y;
    float start = clamp(-3.0 * sigma, low, high);
    float end = clamp(3.0 * sigma, low, high);
    float step = (end - start) / 4.0;
    float y = start + step * 0.5;
    float sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        float w = row_half_width(p.y - y, half_size, radius, n);
        vec2 across = erf((p.x + vec2(-w, w)) * (0.70710678 / sigma));
        sum += (across.y - across.x) * 0.5 * exp(-(y * y) / (2.0 * sigma * sigma));
        y += step;
    }
    return sum * step / (sqrt(2.0 * PI) * sigma);
}

void main() {
    float along = clamp(dot(v_position - v_gradient.xy, v_gradient.zw), 0.0, 1.0);
    // Blended in sRGB, as a colour picker and every other gradient tool do.
    vec4 color = mix(v_color0, v_color1, along);
    color.rgb = linear_from_srgb(color.rgb);
    if (push.texture_index == ATLAS) {
        color.a *= texture(u_atlas, v_uv / vec2(textureSize(u_atlas, 0))).r;
    } else {
        color *= texture(sampler2D(u_textures[push.texture_index], u_sampler), v_uv);
    }

    vec2 p = v_position - v_shape.xy;
    float coverage;
    if (v_params.w > 0.0) {
        coverage = blurred_box(p, v_shape.zw, v_params.x, v_params.y, v_params.w);
    } else {
        float d = box_distance(p, v_shape.zw, v_params.x, v_params.y);
        if (v_params.z > 0.0) {
            d = abs(d) - v_params.z;
        }
        // A pixel's worth of ramp centred on the edge.
        coverage = clamp(0.5 - d, 0.0, 1.0);
    }
    color.a *= coverage;

    if (push.output_in_linear_colorspace == 0u) {
        color.rgb = srgb_from_linear(color.rgb);
    }
    f_color = color;
}
