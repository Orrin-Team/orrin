#version 460

// Red for a selected mesh, green for the hovered one, blended by max so one
// mesh under another never erases it.

layout(location = 0) out vec4 f_mask;

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 value;
} push;

void main() {
    f_mask = push.value;
}
