#version 460

layout(location = 0) in vec3 position;
#ifdef ORRIN_MASKED
// Only the cutout variant reads the maps, so only it declares the attribute.
// `Vertex::per_vertex().definition(&vs)` derives the vertex input from what the
// shader asked for, so the plain pipeline still fetches position alone.
layout(location = 3) in vec2 uv;
layout(location = 0) out vec2 v_uv;
// Which material row to alpha-test against, read out of this instance's entry
// rather than pushed: a multi-draw covers every batch of a pipeline variant, so
// there is no per-draw push to put it in.
layout(location = 1) flat out uint v_material;
#endif

// The cascade's light view-projection, pushed once per pass. The only thing
// still pushed by a geometry draw: everything that used to vary per run travels
// with the instance now, which is what lets a whole view be one multi-draw.
layout(push_constant) uniform Push {
    mat4 light_view_proj;
} push;

// The same per-object buffer the forward pass reads, uploaded once per frame.
// Mirrors GpuObject in forward.rs: both fields stay declared even though only
// `model` is used here, because dropping one would halve every index.
struct Object {
    mat4 model;
    mat4 prev_model; // read only by the prepass; here to keep the stride in step
};
layout(set = 0, binding = 0, std430) readonly buffer Objects {
    Object objects[];
};

// This pass's slice of the frame's draw order: for each entry, the row into
// `objects` and the material to draw it with. The indirection is what lets a row
// be shared — an object the camera sees and four cascades also draw occupies one
// row named five times, rather than five copies of the same two matrices.
// `gl_InstanceIndex` already counts from the draw's `firstInstance`, which is
// where its slice starts. See `vulkan::instances`.
layout(set = 0, binding = 1, std430) readonly buffer Instances {
    uvec2 instances[];
};

void main() {
    uvec2 entry = instances[uint(gl_InstanceIndex)];
    gl_Position = push.light_view_proj * objects[entry.x].model * vec4(position, 1.0);
#ifdef ORRIN_MASKED
    v_uv = uv;
    v_material = entry.y;
#endif
}
