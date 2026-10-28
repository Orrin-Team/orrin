#version 460

// The liquid glass under one of the editor's translucent surfaces: a blur of
// everything drawn beneath it, cut to the surface's rounded rectangle, bent
// toward the surface's edge as a bead of liquid would bend it, with the three
// colour channels pulled slightly apart there and a lit rim on the outline. The
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

// The backdrop is a quarter-resolution copy, so one of its texels is four
// framebuffer pixels; displacements are authored in framebuffer pixels.
const float BACKDROP_SCALE = 0.25;

// How far inside the outline the lens reaches, capped so a large corner radius
// does not turn the whole surface into rim.
const float RIM_WIDTH = 24.0;
// The most the edge bends the backdrop, in framebuffer pixels.
const float REFRACTION = 22.0;
// Red and blue bend a little less and a little more than green, which is what
// fringes a bright edge seen through a real piece of glass.
const float DISPERSION = 0.18;
const float SATURATION = 1.18;
const float LIFT = 1.03;
// Toward the light, which sits above and to the left.
const vec2 LIGHT = vec2(-0.5547, -0.8321);

// Signed distance to the surface's outline, negative inside.
float rounded_rect(vec2 p) {
    vec2 centre = 0.5 * (push.rect_min + push.rect_max);
    vec2 half_size = 0.5 * (push.rect_max - push.rect_min) - push.radius;
    vec2 q = abs(p - centre) - half_size;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - push.radius;
}

// The outline's outward normal, from central differences of the distance.
vec2 outline_normal(vec2 p) {
    vec2 g = vec2(
        rounded_rect(p + vec2(1.0, 0.0)) - rounded_rect(p - vec2(1.0, 0.0)),
        rounded_rect(p + vec2(0.0, 1.0)) - rounded_rect(p - vec2(0.0, 1.0)));
    float len = length(g);
    return len > 1e-4 ? g / len : vec2(0.0);
}

// The 5x5 Gaussian at `uv`, one channel of the result per sample position so
// that dispersion can pull the three apart.
vec3 blurred(vec2 uv_r, vec2 uv_g, vec2 uv_b) {
    vec3 sum = vec3(0.0);
    float weight = 0.0;
    for (int y = -REACH; y <= REACH; ++y) {
        for (int x = -REACH; x <= REACH; ++x) {
            float g = exp(-float(x * x + y * y) / (2.0 * SIGMA * SIGMA));
            vec2 o = vec2(x, y) * push.texel;
            sum += vec3(
                texture(u_backdrop, uv_r + o).r,
                texture(u_backdrop, uv_g + o).g,
                texture(u_backdrop, uv_b + o).b) * g;
            weight += g;
        }
    }
    return sum / weight;
}

void main() {
    vec2 p = gl_FragCoord.xy;
    float d = rounded_rect(p);

    float rim = min(RIM_WIDTH, push.radius);
    // 1 on the outline, 0 once the lens has no effect; squared so the bend is
    // steep at the very edge and gone a little way in, like a bead of liquid.
    float t = clamp(1.0 + d / max(rim, 1.0), 0.0, 1.0);
    float bulge = t * t;

    vec2 n = outline_normal(p);
    vec2 shift = -n * bulge * REFRACTION * BACKDROP_SCALE * push.texel;
    vec3 colour;
    if (bulge > 0.002) {
        colour = blurred(
            v_uv + shift * (1.0 - DISPERSION),
            v_uv + shift,
            v_uv + shift * (1.0 + DISPERSION));
    } else {
        colour = blurred(v_uv, v_uv, v_uv);
    }

    // Vibrancy: the backdrop reads richer through glass than beside it.
    float luma = dot(colour, vec3(0.2126, 0.7152, 0.0722));
    colour = clamp(mix(vec3(luma), colour, SATURATION) * LIFT, 0.0, 1.0);

    // Specular edge: strongest where the outline faces the light, with a
    // fainter echo on the far side where the light has passed through.
    float facing = max(dot(n, LIGHT), 0.0);
    float echo = max(dot(n, -LIGHT), 0.0);
    float rim_light = pow(t, 3.0) * (0.18 + 0.62 * facing + 0.22 * echo);
    // A hairline on the outline itself, so the edge stays defined over a
    // backdrop of the same colour.
    float hairline = clamp(1.0 - abs(d + 0.75), 0.0, 1.0) * (0.16 + 0.34 * facing);
    colour = clamp(colour + vec3(rim_light + hairline), 0.0, 1.0);

    float coverage = clamp(0.5 - d, 0.0, 1.0);
    f_color = vec4(colour, coverage);
}
