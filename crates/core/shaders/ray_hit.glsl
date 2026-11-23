// From a ray to the triangle it hit.
//
// A hit reports two numbers: the instance's id and the triangle's index within
// its mesh. The id is the instance's row in a table this frame uploaded, which
// says where the mesh's indices and vertices start in the shared geometry and
// which material it is drawn with — so the triangle's corners are three index
// fetches and three vertex fetches away.
//
// A table of the frame's own and not the per-object rows every draw reads,
// though those are indexed alike: the rows persist and the next frame rewrites
// them at its head, which on a frame split across two queues is while this
// frame's compute tail is still tracing.
//
// Set 1 of every shader that traces, bound here so they all bind it alike. A
// shader that includes this reads all four bindings: a pipeline's layout holds
// only what its shaders read.

#ifndef ORRIN_RAY_HIT_GLSL
#define ORRIN_RAY_HIT_GLSL

layout(set = 1, binding = 0) uniform accelerationStructureEXT ray_scene;

// Per instance: x = material, y = first index, z = vertex offset, w = the
// light channels the object is in.
layout(set = 1, binding = 1, std430) readonly buffer RayInstances {
    uvec4 ray_instances[];
};

// `PositionVertex`: a position and a texture coordinate, five floats.
const uint POSITION_WORDS = 5u;

layout(set = 1, binding = 2, std430) readonly buffer RayPositions {
    float ray_position_words[];
};

layout(set = 1, binding = 3, std430) readonly buffer RayIndices {
    uint ray_indices[];
};

// One triangle of a mesh, in the mesh's own space: where each corner is in
// the vertex streams, its position and its texture coordinate.
struct RayTriangle {
    uvec3 vertex;
    vec3 position[3];
    vec2 uv[3];
};

RayTriangle ray_triangle(uvec4 instance, uint primitive) {
    RayTriangle triangle;
    uint first_index = instance.y + primitive * 3u;
    for (uint corner = 0u; corner < 3u; ++corner) {
        uint vertex = uint(int(ray_indices[first_index + corner]) + int(instance.z));
        uint base = vertex * POSITION_WORDS;
        triangle.vertex[corner] = vertex;
        triangle.position[corner] = vec3(ray_position_words[base],
                                         ray_position_words[base + 1u],
                                         ray_position_words[base + 2u]);
        triangle.uv[corner] = vec2(ray_position_words[base + 3u], ray_position_words[base + 4u]);
    }
    return triangle;
}

// The three weights a hit's barycentrics stand for, first corner first.
vec3 ray_weights(vec2 barycentrics) {
    return vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics);
}

#endif
