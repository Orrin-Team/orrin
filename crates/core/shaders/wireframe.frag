#version 460

// An edge is where a barycentric coordinate reaches zero. Measured in pixels
// through its own derivative, so the line is `width` pixels wide whatever the
// triangle's size or distance, and its last pixel fades rather than steps.

layout(location = 0) in vec3 v_corner;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 color;
    uint first_index;
    int vertex_offset;
    float width;
    float nudge;
} push;

void main() {
    vec3 pixels = v_corner / max(fwidth(v_corner), vec3(1e-6));
    float distance = min(min(pixels.x, pixels.y), pixels.z);
    float coverage = 1.0 - smoothstep(push.width - 1.0, push.width, distance);
    if (coverage <= 0.0) {
        discard;
    }
    out_color = vec4(push.color.rgb, push.color.a * coverage);
}
