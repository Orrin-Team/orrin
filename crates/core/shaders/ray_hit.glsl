// From a ray's hit to the mesh it hit.
//
// A hit reports two numbers: the instance's id, which is its row in the
// per-object buffer, and the triangle's index within its mesh. The row says
// where the mesh's indices and vertices start in the shared geometry, so the
// triangle's corners are three index fetches and three position fetches away,
// with no table between.
//
// The including shader declares nothing: the three buffers are bound here, in
// set 1, so every shader that traces binds them alike.

#include "object.glsl"

// `PositionVertex`: a position and a texture coordinate, five floats.
const uint POSITION_WORDS = 5u;

layout(set = 1, binding = 0, std430) readonly buffer RayObjects {
    Object ray_objects[];
};

layout(set = 1, binding = 1, std430) readonly buffer RayPositions {
    float ray_position_words[];
};

layout(set = 1, binding = 2, std430) readonly buffer RayIndices {
    uint ray_indices[];
};

// The object-space corners of triangle `primitive` of the mesh instance
// `instance` draws.
vec3[3] hit_corners(uint instance, uint primitive) {
    uvec4 flags = ray_objects[instance].flags;
    uint first_index = flags.z + primitive * 3u;
    int vertex_offset = int(flags.w);
    vec3 corners[3];
    for (uint corner = 0u; corner < 3u; ++corner) {
        uint base = uint(int(ray_indices[first_index + corner]) + vertex_offset) * POSITION_WORDS;
        corners[corner] = vec3(ray_position_words[base], ray_position_words[base + 1u],
                               ray_position_words[base + 2u]);
    }
    return corners;
}
