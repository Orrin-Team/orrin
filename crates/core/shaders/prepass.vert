#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec4 tangent; // xyz = tangent, w = handedness

// The TBN basis is built in *view* space, not world: the normal target feeds
// SSAO and the reflection trace, both of which reconstruct positions from this
// frame's depth and therefore work in the camera's frame. `view` is rigid, so
// rotating the world-space basis into it is exact.
layout(location = 0) out vec3 v_view_normal;
layout(location = 1) out vec3 v_view_tangent;
layout(location = 2) out vec3 v_view_bitangent;
layout(location = 3) out vec2 v_uv;
// Carried because the forward pass tints albedo with it, and a metal's `f0` is
// its albedo: dropping it here would give the reflection a different colour
// from the surface reflecting it.
layout(location = 4) out vec3 v_color;
// Both unjittered. The rasterised position is jittered — that is what TAA
// samples the pixel with — but a motion vector carrying the jitter would report
// a subpixel shake as scene motion, and the resolve would then reproject away
// the very offsets it exists to accumulate.
layout(location = 5) out vec4 v_clip;
layout(location = 6) out vec4 v_previous_clip;
// Where the eye is, for the parallax march: the camera sits at the origin of
// this space, so the direction toward it is just the negated position, and the
// march's metres-to-UV conversion gets its lengths from a rigid transform of the
// world ones. Carried rather than reconstructed from depth so this pass keeps
// reading only the two sets it already binds.
layout(location = 7) out vec3 v_view_pos;
// Where the fragment is in the world, for the decal projection. This pass works
// in view space throughout — see above — but a decal's box is authored in world
// space and the forward pass tests against it there, so testing here in any
// other space would mean two descriptions of one box.
layout(location = 8) out vec3 v_world_pos;
// Which material row to alpha-test and write parameters for, read out of this
// instance's entry rather than pushed: a multi-draw covers every batch of a
// pipeline variant, so there is no per-draw push to put it in. `flat` because it
// is a number, not a quantity to interpolate.
layout(location = 9) flat out uint v_material;

layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 proj;
    mat4 inv_proj;
    mat4 prev_view_proj;
    // xy = the NDC offset baked into `proj`.
    vec4 jitter;
    // `proj * view`, premultiplied CPU-side. See the FrameUbo doc comment.
    mat4 view_proj;
} frame;

// The forward pass tests `EQUAL` against the depth this pass writes, so the two
// shaders have to agree on `gl_Position` to the last bit. `invariant` is the
// guarantee that they do: it holds for the same expression over the same
// inputs, which is why the line below multiplies by the premultiplied
// `view_proj` rather than by `proj` and `view` in turn -- `forward.vert` has
// only the product, and `proj * (view * world)` is not `(proj * view) * world`
// in floating point.
invariant gl_Position;

// The same per-object buffer the forward pass reads, uploaded once per frame.
// Mirrors GpuObject in forward.rs.
struct Object {
    mat4 model;
    mat4 prev_model;
};
layout(set = 1, binding = 0, std430) readonly buffer Objects {
    Object objects[];
};

// This pass's slice of the frame's draw order: for each entry, the row into
// `objects` and the material to draw it with. The indirection is what lets a row
// be shared — an object the camera sees and four cascades also draw occupies one
// row named five times, rather than five copies of the same two matrices.
// `gl_InstanceIndex` already counts from the draw's `firstInstance`, which is
// where its slice starts. See `vulkan::instances`.
layout(set = 1, binding = 1, std430) readonly buffer Instances {
    uvec2 instances[];
};

#include "normal_matrix.glsl"

void main() {
    uvec2 entry = instances[uint(gl_InstanceIndex)];
    uint object = entry.x;
    v_material = entry.y;
    mat4 model = objects[object].model;

    // Same construction as forward.vert, one space along: Gram-Schmidt against
    // the normal, handedness from the tangent's w.
    vec3 world_n = normal_matrix(model) * normal;
    vec3 world_t = mat3(model) * tangent.xyz;
    vec3 N = normalize(mat3(frame.view) * world_n); // view is rigid → pure rotation
    vec3 T = normalize(mat3(frame.view) * world_t);
    T = normalize(T - dot(T, N) * N);
    v_view_normal = N;
    v_view_tangent = T;
    v_view_bitangent = cross(N, T) * tangent.w;
    v_uv = uv;
    v_color = color;

    // `proj * view` is the same view-projection the forward pass pushes; taking
    // it from the frame UBO keeps the prepass push to the one instance offset.
    vec4 world_pos = model * vec4(position, 1.0);
    v_world_pos = world_pos.xyz;
    vec4 view_pos = frame.view * world_pos;
    v_view_pos = view_pos.xyz;
    gl_Position = frame.view_proj * world_pos;

    // Still by way of view space, unlike the position above: this feeds the
    // jitter removal below rather than the rasteriser, so it is not the value
    // the depth test compares and nothing downstream needs it bit-exact.
    vec4 clip = frame.proj * view_pos;

    // `proj` carries the jitter as a translation of clip.xy by jitter * w, so
    // subtracting exactly that recovers the unjittered position.
    v_clip = vec4(clip.xy - frame.jitter.xy * clip.w, clip.zw);
    v_previous_clip = frame.prev_view_proj * objects[object].prev_model * vec4(position, 1.0);
}
