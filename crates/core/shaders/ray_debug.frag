#version 460
#extension GL_EXT_ray_query : require

// The scene as a ray sees it: one ray per pixel through the camera, against
// the scene's acceleration structure, drawn as how far it went, what it hit, or
// which way the hit triangle faces. Over the tonemapped frame and in place of
// it, in its own colours: a view of the structures, not of the lighting.
//
// The normal is the triangle's own, from its three corners in the shared
// geometry, and that is the point of it: it proves the path every traced pass
// takes from a hit to a mesh — the instance's row, the mesh's first index, the
// three positions — and shows a structure built from the wrong span as wrong.

#include "ray_hit.glsl"

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;

layout(set = 0, binding = 1) uniform RayDebug {
    // Without the frame's jitter: this is drawn after the temporal resolve,
    // and nothing would average the jitter back out.
    mat4 inv_view_proj;
    // xyz = the camera, w = how far a ray goes.
    vec4 origin;
    // x = which view.
    uvec4 mode;
} u;

const uint VIEW_DISTANCE = 1u;
const uint VIEW_INSTANCE = 2u;
const uint VIEW_NORMAL = 3u;

// What a ray that hits nothing draws: not black, which a far hit is.
const vec3 MISS = vec3(0.02, 0.0, 0.05);

// A colour per id with nothing in common between neighbours, so two instances
// of one mesh side by side read as two.
vec3 id_color(uint id) {
    uint h = id * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h ^= h >> 22u;
    return vec3(h & 255u, (h >> 8u) & 255u, (h >> 16u) & 255u) / 255.0 * 0.85 + 0.15;
}

void main() {
    // The near plane under reversed Z, which an infinite far plane still has.
    vec4 near = u.inv_view_proj * vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 direction = normalize(near.xyz / near.w - u.origin.xyz);

    // Every triangle as built: a cutout is its whole quad here, since what
    // this draws is the geometry and not the surface.
    rayQueryEXT query;
    rayQueryInitializeEXT(query, scene, gl_RayFlagsOpaqueEXT, 0xffu, u.origin.xyz, 0.0, direction,
                          u.origin.w);
    while (rayQueryProceedEXT(query)) {
    }

    if (rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionTriangleEXT) {
        out_color = vec4(MISS, 1.0);
        return;
    }

    vec3 color;
    if (u.mode.x == VIEW_DISTANCE) {
        // Reciprocal, so a metre in front of the camera and a metre at the
        // far end of a hall are both visible steps.
        float t = rayQueryGetIntersectionTEXT(query, true);
        color = vec3(1.0 / (1.0 + t * 0.25));
    } else if (u.mode.x == VIEW_INSTANCE) {
        color = id_color(uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, true)));
    } else {
        vec3 corners[3] = hit_corners(
            uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, true)),
            uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, true)));
        mat4x3 to_world = rayQueryGetIntersectionObjectToWorldEXT(query, true);
        vec3 a = to_world * vec4(corners[0], 1.0);
        vec3 b = to_world * vec4(corners[1], 1.0);
        vec3 c = to_world * vec4(corners[2], 1.0);
        vec3 normal = normalize(cross(b - a, c - a));
        // Toward the ray, as a surface seen from either side is drawn.
        normal = faceforward(normal, direction, normal);
        color = normal * 0.5 + 0.5;
        // The target is sRGB and these are not radiance: undone here so a
        // normal reads as the number it is.
        color = pow(color, vec3(2.2));
    }
    out_color = vec4(color, 1.0);
}
