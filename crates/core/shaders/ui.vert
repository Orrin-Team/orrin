#version 460

// The in-game UI: one instance per rectangle, its six corners from the vertex
// index. See `gfx::ui` for what a quad carries.

layout(location = 0) in vec4 rect;
layout(location = 1) in vec4 uv;
layout(location = 2) in vec4 color;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;

layout(push_constant) uniform Push {
    vec2 screen_size;
    uint output_in_linear_colorspace;
    uint texture_index;
} push;

const vec2 CORNERS[6] = vec2[](
    vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
    vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0)
);

// The exact curve, not the overlay's: a game's colours are what it asked for.
vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 lower = srgb / vec3(12.92);
    vec3 higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    return mix(higher, lower, vec3(cutoff));
}

void main() {
    vec2 corner = CORNERS[gl_VertexIndex];
    gl_Position = vec4(2.0 * mix(rect.xy, rect.zw, corner) / push.screen_size - 1.0, 0.0, 1.0);
    v_uv = mix(uv.xy, uv.zw, corner);
    v_color = vec4(linear_from_srgb(color.rgb), color.a);
}
