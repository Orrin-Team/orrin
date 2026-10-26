#version 460

// The transform gizmo's strokes, built by the editor in physical pixels.

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 edge;
layout(location = 2) in vec4 color;

layout(location = 0) out vec2 v_edge;
layout(location = 1) out vec4 v_color;

layout(push_constant) uniform Push {
    vec2 screen_size;
    int output_in_linear_colorspace;
} push;

// The overlay's curve, quirk included (see `overlay.frag`), so an axis here and
// the inspector chip of the same axis are the same colour on screen.
vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 lower = srgb / vec3(12.92);
    vec3 higher = pow((srgb + vec3(0.055) / vec3(1.055)), vec3(2.4));
    return mix(higher, lower, vec3(cutoff));
}

void main() {
    gl_Position = vec4(2.0 * position / push.screen_size - 1.0, 0.0, 1.0);
    v_edge = edge;
    v_color = push.output_in_linear_colorspace == 1
        ? vec4(linear_from_srgb(color.rgb), color.a)
        : color;
}
