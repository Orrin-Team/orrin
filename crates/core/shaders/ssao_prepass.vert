#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 v_view_normal;

layout(push_constant) uniform Push {
    mat4 mvp;
    mat4 normal_matrix; // world inverse-transpose; upper 3x3 used
} push;

layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 proj;
    mat4 inv_proj;
} frame;

void main() {
    vec3 world_n = mat3(push.normal_matrix) * normal;
    v_view_normal = mat3(frame.view) * world_n; // view is rigid → pure rotation
    gl_Position = push.mvp * vec4(position, 1.0);
}
