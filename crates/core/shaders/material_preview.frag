#version 460

// The material preview's sphere, tonemapped and averaged down to the image the
// editor shows. Each output pixel is the mean of the SUPERSAMPLE x SUPERSAMPLE
// shaded samples under it, taken after the curve rather than before, so a
// highlight against the clear background does not alias into a bright stair.

#include "aces.glsl"

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 f_color;

// The shaded sphere: radiance in rgb, coverage in a, zero where nothing drew.
layout(set = 0, binding = 0) uniform sampler2D u_shaded;

layout(push_constant) uniform Push {
    float exposure;
} push;

// Keep in sync with SUPERSAMPLE in vulkan/material_preview.rs.
const int SUPERSAMPLE = 2;

vec3 srgb_from_linear(vec3 c) {
    return mix(1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, c * 12.92, lessThan(c, vec3(0.0031308)));
}

vec3 linear_from_srgb(vec3 c) {
    return mix(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, lessThan(c, vec3(0.04045)));
}

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy) * SUPERSAMPLE;
    vec3 encoded = vec3(0.0);
    float coverage = 0.0;
    for (int y = 0; y < SUPERSAMPLE; ++y) {
        for (int x = 0; x < SUPERSAMPLE; ++x) {
            vec4 shaded = texelFetch(u_shaded, base + ivec2(x, y), 0);
            encoded += srgb_from_linear(aces(shaded.rgb * push.exposure)) * shaded.a;
            coverage += shaded.a;
        }
    }
    float samples = float(SUPERSAMPLE * SUPERSAMPLE);
    // egui blends premultiplied colour in gamma space, so the edge is
    // premultiplied there. The target encodes on store, which is undone here so
    // the stored bytes are exactly that gamma-space value.
    f_color = vec4(linear_from_srgb(encoded / samples), coverage / samples);
}
