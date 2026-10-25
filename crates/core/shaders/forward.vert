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
// Which material row to shade with, read out of this instance's entry rather
// than pushed: a multi-draw covers every batch of a pipeline variant, so there
// is no per-draw push to put it in. `flat` because it is a number, not a
// quantity to interpolate.
layout(location = 6) flat out uint v_material;

// Declared identically to the fragment shader so the two stages share one
// push-constant range.
layout(push_constant) uniform Push {
    mat4 view_proj;
} push;

// Per-object transforms, indexed per instance. Mirrors GpuObject in
// forward.rs; std430 packs it exactly like the Rust struct (all mat4 fields).
struct Object {
    mat4 model;
    mat4 prev_model; // read only by the prepass; here to keep the stride in step
};
layout(set = 4, binding = 0, std430) readonly buffer Objects {
    Object objects[];
};

// This pass's slice of the frame's draw order: for each entry, the row into
// `objects` and the material to draw it with. The indirection is what lets a row
// be shared — an object the camera sees and four cascades also draw occupies one
// row named five times, rather than five copies of the same two matrices.
// `gl_InstanceIndex` already counts from the draw's `firstInstance`, which is
// where its slice starts. See `vulkan::instances`.
layout(set = 4, binding = 1, std430) readonly buffer Instances {
    uvec2 instances[];
};

// The other half of the depth-invariance guarantee `prepass.vert` documents:
// with MSAA off this pass attaches the prepass depth read-only and tests
// `EQUAL` against it, so `push.view_proj * world` here and `frame.view_proj *
// world_pos` there have to round identically.
invariant gl_Position;

#include "normal_matrix.glsl"

void main() {
    uvec2 entry = instances[uint(gl_InstanceIndex)];
    uint object = entry.x;
    v_material = entry.y;
    mat4 model = objects[object].model;

    vec4 world = model * vec4(position, 1.0);
    v_world_pos = world.xyz;

    // World-space TBN basis for tangent-space normal mapping.
    vec3 N = normalize(normal_matrix(model) * normal);
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
