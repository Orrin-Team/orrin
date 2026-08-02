#version 460

layout(location = 0) in vec3 position;

// The cascade's light view-projection, pushed per pass, plus the first object
// row of this instanced run; gl_InstanceIndex counts from it.
layout(push_constant) uniform Push {
    mat4 light_view_proj;
    uint object_base;
} push;

// The same per-object buffer the forward pass reads, uploaded once per frame.
// Mirrors GpuObject in forward.rs: both fields stay declared even though only
// `model` is used here, because dropping one would halve every index.
struct Object {
    mat4 model;
    mat4 normal_matrix;
};
layout(set = 0, binding = 0, std430) readonly buffer Objects {
    Object objects[];
};

void main() {
    uint object = push.object_base + uint(gl_InstanceIndex);
    gl_Position = push.light_view_proj * objects[object].model * vec4(position, 1.0);
}
