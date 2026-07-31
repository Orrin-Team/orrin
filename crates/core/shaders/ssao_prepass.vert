#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 v_view_normal;

layout(push_constant) uniform Push {
    // First object row of this instanced run; gl_InstanceIndex counts from it.
    uint object_base;
} push;

layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 proj;
    mat4 inv_proj;
} frame;

// The same per-object buffer the forward pass reads, uploaded once per frame.
// Mirrors GpuObject in forward.rs.
struct Object {
    mat4 model;
    mat4 normal_matrix;
};
layout(set = 1, binding = 0, std430) readonly buffer Objects {
    Object objects[];
};

void main() {
    uint object = push.object_base + uint(gl_InstanceIndex);

    vec3 world_n = mat3(objects[object].normal_matrix) * normal;
    v_view_normal = mat3(frame.view) * world_n; // view is rigid → pure rotation
    // `proj * view` is the same view-projection the forward pass pushes; taking
    // it from the frame UBO keeps the prepass push to the one instance offset.
    gl_Position = frame.proj * frame.view * objects[object].model * vec4(position, 1.0);
}
