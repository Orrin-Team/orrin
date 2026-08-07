#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(location = 0) out vec3 v_view_normal;
// Both unjittered. The rasterised position is jittered — that is what TAA
// samples the pixel with — but a motion vector carrying the jitter would report
// a subpixel shake as scene motion, and the resolve would then reproject away
// the very offsets it exists to accumulate.
layout(location = 1) out vec4 v_clip;
layout(location = 2) out vec4 v_previous_clip;

layout(push_constant) uniform Push {
    // First object row of this instanced run; gl_InstanceIndex counts from it.
    uint object_base;
} push;

layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 proj;
    mat4 inv_proj;
    mat4 prev_view_proj;
    // xy = the NDC offset baked into `proj`.
    vec4 jitter;
} frame;

// The same per-object buffer the forward pass reads, uploaded once per frame.
// Mirrors GpuObject in forward.rs.
struct Object {
    mat4 model;
    mat4 normal_matrix;
    mat4 prev_model;
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
    vec4 clip = frame.proj * frame.view * objects[object].model * vec4(position, 1.0);
    gl_Position = clip;

    // `proj` carries the jitter as a translation of clip.xy by jitter * w, so
    // subtracting exactly that recovers the unjittered position.
    v_clip = vec4(clip.xy - frame.jitter.xy * clip.w, clip.zw);
    v_previous_clip = frame.prev_view_proj * objects[object].prev_model * vec4(position, 1.0);
}
