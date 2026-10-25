#version 460

// egui's meshes. egui blends in gamma space, so texture and vertex colour are
// combined there and, on an sRGB target, converted back to linear last.
//
// Ported from egui_winit_vulkano 0.28 (Apache-2.0) together with its
// `linear_from_srgb`, grouping and all: `(srgb + 0.055 / 1.055)` rather than
// `(srgb + 0.055) / 1.055`. The editor's palette was tuned on screen through
// exactly this curve, so it is kept rather than corrected under it.

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D u_texture;

layout(push_constant) uniform Push {
    vec2 screen_size;
    int output_in_linear_colorspace;
} push;

vec3 srgb_from_linear(vec3 linear) {
    bvec3 cutoff = lessThan(linear, vec3(0.0031308));
    vec3 lower = linear * vec3(12.92);
    vec3 higher = vec3(1.055) * pow(linear, vec3(1.0 / 2.4)) - vec3(0.055);
    return mix(higher, lower, vec3(cutoff));
}

vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 lower = srgb / vec3(12.92);
    vec3 higher = pow((srgb + vec3(0.055) / vec3(1.055)), vec3(2.4));
    return mix(higher, lower, vec3(cutoff));
}

void main() {
    vec4 texel = texture(u_texture, v_tex_coords);
    vec4 color = v_color * vec4(srgb_from_linear(texel.rgb), texel.a);
    if (push.output_in_linear_colorspace == 1) {
        color = vec4(linear_from_srgb(color.rgb), color.a);
    }
    f_color = color;
}
