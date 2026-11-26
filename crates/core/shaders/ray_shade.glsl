// What a ray finds: the nearest surface along it, and the radiance that
// surface sends back.
//
// Hit shading is one more pass that reads a surface and lights it, so it is
// built from the files the forward pass is. The material's maps are read by
// `standard_surface` (`surface.glsl`), each light's contribution is `brdf`
// (`lobes.glsl`), the lights and the sky are `lights.glsl` and `ambient.glsl`.
// What differs is where each input comes from, since nothing was rasterised:
//
// - The surface is rebuilt from the hit: three corners out of the shared
//   geometry, blended by the hit's barycentrics.
// - There are no screen-space derivatives, so every map is read at a level
//   chosen from the ray: its cone's width at the hit against the size of a
//   texel on the triangle (Akenine-Moller et al., Ray Tracing Gems ch. 20).
// - The lights that reach the hit come from the world grid
//   (`light_grid.glsl`), not the view's clusters, which a hit outside the
//   frustum has none of.
// - A light's shadow is one ray to it, not a lookup in a map fitted to the
//   camera's view. To the light's centre, so it is exact and the same every
//   frame.
// - The light that bounced to the hit is the irradiance volumes' where they
//   are live (`gi.glsl`, set 4), and the sky's where they are not.
//
// A material is read as its standard reading. The optional lobes — clear coat,
// sheen, anisotropy, subsurface — fold away, as they do for a flat forward
// pipeline (the includer's manifest row defines `ORRIN_FLAT_SURFACE`), and a
// custom surface function does not run: see `gfx/surface/mod.rs`. Parallax and
// decals are the rasteriser's alone.
//
// The includer enables `GL_EXT_ray_query` and `GL_EXT_nonuniform_qualifier`.
// Sets 1 to 4 are bound here, alike for every shader that shades a hit; a
// shader binds the part of each it reads.

#ifndef ORRIN_RAY_SHADE_GLSL
#define ORRIN_RAY_SHADE_GLSL

const float PI = 3.14159265359;

#include "ray_hit.glsl"
#include "octahedral.glsl"

#define ORRIN_LIGHTS_SET 2
#define ORRIN_LIGHTING_BINDING 0
#define ORRIN_POINT_LIGHTS_BINDING 1
#define ORRIN_SPOT_LIGHTS_BINDING 2
#include "lights.glsl"

#include "material.glsl"
layout(set = 2, binding = 3, std430) readonly buffer RayMaterials {
    GpuMaterial ray_materials[];
};

// Keep in sync with SPECULAR_MIPS in environment.rs.
const float SPECULAR_MIPS = 6.0;
layout(set = 2, binding = 4) uniform textureCube u_environment;
layout(set = 2, binding = 5) uniform sampler u_environment_sampler;

#define ORRIN_LIGHT_GRID_SET 2
#define ORRIN_LIGHT_GRID_BINDING 6
#define ORRIN_LIGHT_GRID_CELLS_BINDING 7
#define ORRIN_LIGHT_GRID_LIGHTS_BINDING 8
#include "light_grid.glsl"

// `SurfaceVertex`: an octahedral normal, a tangent and a colour, four words.
const uint SURFACE_WORDS = 4u;
layout(set = 2, binding = 9, std430) readonly buffer RaySurfaces {
    uint ray_surface_words[];
};

// Keep in sync with MAX_TEXTURES in forward.rs.
const int MAX_TEXTURES = 192;
layout(set = 3, binding = 0) uniform texture2D ray_textures[MAX_TEXTURES];
layout(set = 3, binding = 1) uniform sampler ray_tex_sampler;

// The level a map of one texel per unit of texture coordinate would be read
// at, for the hit being shaded; each map adds its own size. Set before any map
// is read.
float ray_lod = 0.0;

vec4 sample_tex(uint index, vec2 uv) {
    vec2 size = vec2(textureSize(sampler2D(ray_textures[nonuniformEXT(index)], ray_tex_sampler), 0));
    return textureLod(sampler2D(ray_textures[nonuniformEXT(index)], ray_tex_sampler), uv,
                      ray_lod + 0.5 * log2(size.x * size.y));
}

#include "surface.glsl"
#include "lobes.glsl"
#include "brdf.glsl"
#include "ambient.glsl"

#define ORRIN_GI_SET 4
#define ORRIN_GI_BLOCK_BINDING 0
#define ORRIN_GI_PROBES_BINDING 1
#define ORRIN_GI_IRRADIANCE_BINDING 2
#define ORRIN_GI_DISTANCE_BINDING 3
#define ORRIN_GI_SAMPLER_BINDING 4
#include "gi.glsl"

// How far a ray goes when nothing says otherwise: past everything the scene's
// structure holds.
const float RAY_FAR = 1.0e5;

// The level term of a hit on `triangle`, whose corners are in world space,
// through a cone `width` metres across there and meeting the surface at
// `cosine`.
float ray_cone_lod(vec3 a, vec3 b, vec3 c, RayTriangle triangle, float width, float cosine) {
    float world_area = length(cross(b - a, c - a));
    vec2 e1 = triangle.uv[1] - triangle.uv[0];
    vec2 e2 = triangle.uv[2] - triangle.uv[0];
    float uv_area = abs(e1.x * e2.y - e1.y * e2.x);
    return 0.5 * log2(max(uv_area, 1e-12) / max(world_area, 1e-12))
         + log2(max(width, 1e-6) / max(abs(cosine), 0.05));
}

// Whether a cutout's triangle is solid where the ray crosses it. The alpha a
// rasterised fragment would be cut by, at the coordinate the ray hit.
bool ray_cutout_solid(uint instance_id, uint primitive, vec2 barycentrics, float distance) {
    uvec4 instance = ray_instances[instance_id];
    GpuMaterial m = ray_materials[instance.x];
    RayTriangle triangle = ray_triangle(instance, primitive);
    vec3 w = ray_weights(barycentrics);
    vec2 uv = triangle.uv[0] * w.x + triangle.uv[1] * w.y + triangle.uv[2] * w.z;
    // In the mesh's own space, which is the world's but for the instance's
    // scale: a visibility test wants the cutout's outline, and a level too
    // coarse blurs every leaf into the card it is cut from.
    float lod = ray_cone_lod(triangle.position[0], triangle.position[1], triangle.position[2],
                             triangle, distance * 0.001, 1.0);
    uint albedo = m.tex_indices.x;
    vec2 size = vec2(textureSize(sampler2D(ray_textures[nonuniformEXT(albedo)], ray_tex_sampler), 0));
    float alpha = textureLod(sampler2D(ray_textures[nonuniformEXT(albedo)], ray_tex_sampler), uv,
                             lod + 0.5 * log2(size.x * size.y)).a;
    return m.base_color.a * alpha >= m.alpha.x;
}

struct RayHit {
    bool found;
    float distance;
    uint instance;
    uint primitive;
    vec2 barycentrics;
    mat4x3 to_world;
    mat4x3 from_world;
};

// The nearest surface along a ray, with cutouts cut.
RayHit ray_closest(vec3 origin, vec3 direction, float near, float far) {
    rayQueryEXT query;
    rayQueryInitializeEXT(query, ray_scene, gl_RayFlagsNoneEXT, 0xffu, origin, near, direction, far);
    // Entered only for a cutout's triangles: everything else is opaque to the
    // traversal and never offered.
    while (rayQueryProceedEXT(query)) {
        if (ray_cutout_solid(uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, false)),
                             uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false)),
                             rayQueryGetIntersectionBarycentricsEXT(query, false),
                             rayQueryGetIntersectionTEXT(query, false))) {
            rayQueryConfirmIntersectionEXT(query);
        }
    }

    RayHit hit;
    hit.found = rayQueryGetIntersectionTypeEXT(query, true)
             == gl_RayQueryCommittedIntersectionTriangleEXT;
    if (hit.found) {
        hit.distance = rayQueryGetIntersectionTEXT(query, true);
        hit.instance = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, true));
        hit.primitive = uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, true));
        hit.barycentrics = rayQueryGetIntersectionBarycentricsEXT(query, true);
        hit.to_world = rayQueryGetIntersectionObjectToWorldEXT(query, true);
        hit.from_world = rayQueryGetIntersectionWorldToObjectEXT(query, true);
    }
    return hit;
}

// Whether anything stands between `origin` and a point `far` along
// `direction`. Stops at the first thing that does.
bool ray_occluded(vec3 origin, vec3 direction, float far) {
    rayQueryEXT query;
    rayQueryInitializeEXT(query, ray_scene, gl_RayFlagsTerminateOnFirstHitEXT, 0xffu, origin, 0.0,
                          direction, far);
    while (rayQueryProceedEXT(query)) {
        if (ray_cutout_solid(uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, false)),
                             uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false)),
                             rayQueryGetIntersectionBarycentricsEXT(query, false),
                             rayQueryGetIntersectionTEXT(query, false))) {
            rayQueryConfirmIntersectionEXT(query);
        }
    }
    return rayQueryGetIntersectionTypeEXT(query, true)
        == gl_RayQueryCommittedIntersectionTriangleEXT;
}

float snorm16_low(uint word) {
    return max(float(int(word << 16u) >> 16) / 32767.0, -1.0);
}

float snorm16_high(uint word) {
    return max(float(int(word) >> 16) / 32767.0, -1.0);
}

// A hit as something to light: the surface there, facing the ray, and what the
// lighting needs besides.
struct RaySurface {
    Surface s;
    vec3 position;
    // The triangle's own normal, turned to the side the ray came from.
    vec3 face;
    // Where a ray leaving the surface starts: off it on that side, by more the
    // further the hit is, since that is how its position's error grows.
    vec3 lifted;
    // Luminance the surface emits, cd/m2.
    vec3 emissive;
    // The light channels the object is in.
    uint channels;
};

// The surface at `hit`, as a ray travelling along `direction` finds it.
// `spread` is how fast the ray's cone widens, in metres per metre: a pixel's
// own angle for a mirror, wider for a rougher surface.
RaySurface ray_surface(RayHit hit, vec3 direction, float spread) {
    uvec4 instance = ray_instances[hit.instance];
    GpuMaterial m = ray_materials[instance.x];
    RayTriangle triangle = ray_triangle(instance, hit.primitive);
    vec3 w = ray_weights(hit.barycentrics);

    vec3 corner[3];
    for (int i = 0; i < 3; ++i) {
        corner[i] = hit.to_world * vec4(triangle.position[i], 1.0);
    }
    vec3 position = corner[0] * w.x + corner[1] * w.y + corner[2] * w.z;
    // The triangle's own normal, turned to the side the ray came from: a
    // surface seen from behind is lit as that side of it.
    vec3 face = normalize(cross(corner[1] - corner[0], corner[2] - corner[0]));
    float side = dot(face, direction) > 0.0 ? -1.0 : 1.0;
    face *= side;

    vec3 normal = vec3(0.0);
    vec4 tangent = vec4(0.0);
    vec3 color = vec3(0.0);
    for (int i = 0; i < 3; ++i) {
        uint base = triangle.vertex[i] * SURFACE_WORDS;
        uint n = ray_surface_words[base];
        uint t0 = ray_surface_words[base + 1u];
        uint t1 = ray_surface_words[base + 2u];
        normal += oct_decode(vec2(snorm16_low(n), snorm16_high(n))) * w[i];
        tangent += vec4(snorm16_low(t0), snorm16_high(t0), snorm16_low(t1), snorm16_high(t1)) * w[i];
        color += unpackUnorm4x8(ray_surface_words[base + 3u]).rgb * w[i];
    }
    // A normal goes through the inverse transpose, which a row vector times
    // the inverse is.
    vec3 N = normalize(normal * mat3(hit.from_world)) * side;
    vec3 T = mat3(hit.to_world) * tangent.xyz;
    T = normalize(T - N * dot(N, T));
    vec3 B = cross(N, T) * (tangent.w < 0.0 ? -1.0 : 1.0) * side;

    ray_lod = ray_cone_lod(corner[0], corner[1], corner[2], triangle, spread * hit.distance,
                           dot(face, direction));

    SurfaceIn i;
    i.uv = triangle.uv[0] * w.x + triangle.uv[1] * w.y + triangle.uv[2] * w.z;
    i.world_position = position;
    i.world_normal = N;
    i.view_dir = -direction;
    i.vertex_color = color;
    SurfaceOut o = standard_surface(m, i);

    Surface s;
    s.flags = 0u;
    s.V = -direction;
    s.N = normalize(mat3(T, B, N) * o.normal);
    // A mapped normal can tilt past the ray, which a rasterised fragment
    // shades as a grazing view; so does this.
    s.n_dot_v = max(dot(s.N, s.V), 1e-4);
    s.uv = i.uv;
    s.albedo = o.albedo;
    s.metallic = o.metallic;
    s.perceptual_roughness = o.roughness;
    s.a = o.roughness * o.roughness;
    s.f0 = mix(vec3(0.16 * m.params.z * m.params.z), s.albedo, s.metallic);
    s.energy = energy_compensation(s.f0, s.perceptual_roughness, s.n_dot_v);
    s.T = T;
    s.B = B;
    s.at = s.a;
    s.ab = s.a;
    s.anisotropy = 0.0;
    s.coat_N = s.N;
    s.coat = 0.0;
    s.coat_a = 0.0;
    s.coat_roughness = 0.0;
    s.sheen_color = vec3(0.0);
    s.sheen_a = 1.0;
    s.scatter_color = vec3(0.0);
    s.scatter_radius = vec3(1.0);
    s.thickness = 0.0;
    s.forward_scatter = 1.0;
    s.wrap = vec3(0.0);

    RaySurface found;
    found.s = s;
    found.position = position;
    found.face = face;
    found.lifted = position + face * max(1e-3, 1e-4 * hit.distance);
    found.emissive = o.emissive;
    found.channels = instance.w;
    return found;
}

// Whether a ray met the back of what it hit: the side the triangle's vertex
// normals turn away from, which for a closed mesh is its inside. A cutout is
// a sheet with no inside, so never.
bool ray_back_face(RayHit hit, vec3 direction) {
    uvec4 instance = ray_instances[hit.instance];
    if ((ray_materials[instance.x].tex_flags.y & MATERIAL_MASKED) != 0u) {
        return false;
    }
    RayTriangle triangle = ray_triangle(instance, hit.primitive);
    vec3 w = ray_weights(hit.barycentrics);
    vec3 normal = vec3(0.0);
    for (int i = 0; i < 3; ++i) {
        uint n = ray_surface_words[triangle.vertex[i] * SURFACE_WORDS];
        normal += oct_decode(vec2(snorm16_low(n), snorm16_high(n))) * w[i];
    }
    return dot(normal * mat3(hit.from_world), direction) > 0.0;
}

// One light's contribution: the whole of the material's response, or only
// its diffuse lobe — what a surface sends every way alike, which is all a
// cache of radiance shared between directions can hold.
vec3 ray_light(Surface s, vec3 L, vec3 radiance, bool diffuse_only) {
    if (diffuse_only) {
        return s.albedo * (1.0 - s.metallic) / PI * radiance * max(dot(s.N, L), 0.0);
    }
    return brdf(s, L, radiance, 1.0, 0.0).direct;
}

// What the sun and the punctual lights lay on a surface and it sends back
// along the ray, each light shadowed by one ray to it.
vec3 ray_lights(RaySurface found, bool diffuse_only) {
    Surface s = found.s;
    vec3 position = found.position;
    vec3 face = found.face;
    vec3 lifted = found.lifted;
    uint channels = found.channels;
    vec3 radiance = vec3(0.0);

    if (lighting.sun_color.w > 0.0 && (floatBitsToUint(lighting.sun_direction.w) & channels) != 0u) {
        vec3 L = lighting.sun_direction.xyz;
        if (dot(s.N, L) > 0.0 && dot(face, L) > 0.0 && !ray_occluded(lifted, L, RAY_FAR)) {
            radiance += ray_light(s, L, lighting.sun_color.rgb * lighting.sun_color.w, diffuse_only);
        }
    }

    uvec3 cell = world_light_cell(position);
    for (uint k = 0u; k < cell.y; ++k) {
        PointLight light = point_lights[light_grid_lights[cell.x + k]];
        if ((floatBitsToUint(light.shadow.z) & channels) == 0u) {
            continue;
        }
        vec3 to_light = light.position.xyz - position;
        float dist = length(to_light);
        float falloff = attenuate(dist, light.position.w);
        vec3 L = to_light / max(dist, 1e-5);
        if (falloff <= 0.0 || dot(s.N, L) <= 0.0 || dot(face, L) <= 0.0) {
            continue;
        }
        // Below `LIGHT_CASTS_NONE` the light casts no shadow at all; between it
        // and zero it asked for one and the atlas had no tile, which costs a
        // fragment its shadow and costs a ray nothing.
        if (light.shadow.x > LIGHT_CASTS_NONE && ray_occluded(lifted, L, dist)) {
            continue;
        }
        radiance += ray_light(s, L, light.color.rgb * light.color.w * falloff, diffuse_only);
    }
    for (uint k = 0u; k < cell.z; ++k) {
        SpotLight light = spot_lights[light_grid_lights[cell.x + cell.y + k]];
        if ((floatBitsToUint(light.params.w) & channels) == 0u) {
            continue;
        }
        vec3 to_light = light.position.xyz - position;
        float dist = length(to_light);
        vec3 L = to_light / max(dist, 1e-5);
        float falloff = attenuate(dist, light.position.w) * cone_falloff(light, L);
        if (falloff <= 0.0 || dot(s.N, L) <= 0.0 || dot(face, L) <= 0.0) {
            continue;
        }
        if (light.params.y > LIGHT_CASTS_NONE && ray_occluded(lifted, L, dist)) {
            continue;
        }
        radiance += ray_light(s, L, light.color.rgb * light.color.w * falloff, diffuse_only);
    }
    return radiance;
}

vec3 ray_direct(RaySurface found) {
    return ray_lights(found, false);
}

// What reaches a surface from everything but the lights, in the unit
// `sh_irradiance` returns: the irradiance volumes where they are live, the
// sky where they are not or run out. `reflection` is what the environment's
// reflection is scaled by for it — see `gi_reflection_scale`.
vec3 ray_irradiance(vec3 P, vec3 N, vec3 V, out float reflection) {
    vec3 sky = sh_irradiance(N);
    reflection = 1.0;
    if (!gi_live()) {
        return sky;
    }
    vec4 local = gi_irradiance(P, N, V);
    vec3 irradiance = local.rgb + sky * (1.0 - local.a);
    if (gi.camera.w > 0.5) {
        reflection = gi_reflection_scale(irradiance, sky);
    }
    return irradiance;
}

// The light that bounced to a surface, as a fragment takes it but for the
// screen-space occlusion a hit has none of.
vec3 ray_ambient(RaySurface found) {
    Surface s = found.s;
    float reflection;
    vec3 irradiance = ray_irradiance(found.position, s.N, s.V, reflection);
    return irradiance * s.albedo * (1.0 - s.metallic)
         + specular_ibl(s.N, s.V, s.f0, s.perceptual_roughness, s.energy * reflection);
}

// Radiance leaving `hit` back along the ray, in cd/m2: what it emits, what
// the lights lay on it, and the sky.
vec3 ray_shade(RayHit hit, vec3 direction, float spread) {
    RaySurface found = ray_surface(hit, direction, spread);
    return found.emissive + ray_direct(found) + ray_ambient(found);
}

#endif
