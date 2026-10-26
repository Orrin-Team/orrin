#version 460

// A selected or hovered mesh, flat, into the outline's mask.

layout(location = 0) in vec3 position;

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 value;
} push;

void main() {
    gl_Position = push.mvp * vec4(position, 1.0);
}
