#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec4 tangent; // xyz = tangent, w = handedness

layout(location = 0) out vec3 v_world_pos;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec3 v_tangent;
layout(location = 3) out vec3 v_bitangent;
layout(location = 4) out vec2 v_uv;
layout(location = 5) out vec3 v_color;

// Declared identically to the fragment shader so the two stages
// share one push-constant range. `material_index` is unused here.
layout(push_constant) uniform Push {
    mat4 view_proj;
    uint material_index;
    // First object row of this instanced run; gl_InstanceIndex counts from it.
    uint object_base;
} push;

// Per-object transforms, indexed per instance. Mirrors GpuObject in
// forward.rs; std430 packs it exactly like the Rust struct (all mat4 fields).
struct Object {
    mat4 model;
    mat4 normal_matrix;
};
layout(set = 4, binding = 0, std430) readonly buffer Objects {
    Object objects[];
};

void main() {
    uint object = push.object_base + uint(gl_InstanceIndex);
    mat4 model = objects[object].model;
    mat4 normal_matrix = objects[object].normal_matrix;

    vec4 world = model * vec4(position, 1.0);
    v_world_pos = world.xyz;

    // World-space TBN basis for tangent-space normal mapping.
    vec3 N = normalize(mat3(normal_matrix) * normal);
    vec3 T = normalize(mat3(model) * tangent.xyz);
    T = normalize(T - dot(T, N) * N);          // Gram-Schmidt
    vec3 B = cross(N, T) * tangent.w;          // handedness from w

    v_normal = N;
    v_tangent = T;
    v_bitangent = B;
    v_uv = uv;
    v_color = color;
    gl_Position = push.view_proj * world;
}
