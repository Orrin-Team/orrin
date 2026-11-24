// The irradiance volumes: probes in nested grids round the camera, each an
// octahedral tile of the light arriving at it and of how far it sees in every
// direction, and the lookup a surface reads them through. `gfx/probes.rs` is
// the design, and holds the CPU half of the addressing here in lock-step.
//
// The includer defines `ORRIN_GI_SET`, `ORRIN_GI_BLOCK_BINDING` and
// `ORRIN_GI_PROBES_BINDING`, and `ORRIN_GI_PROBES_WRITE` for the one pass
// that writes the probes' state. Defining `ORRIN_GI_IRRADIANCE_BINDING`,
// `ORRIN_GI_DISTANCE_BINDING` and `ORRIN_GI_SAMPLER_BINDING` as well declares
// the two atlases and the lookup over them. A shader that can trace may define
// `ORRIN_GI_TRACED_VISIBILITY` and `gi_probe_hidden`, and the lookup then
// takes nothing from a probe the point cannot see.

#ifndef ORRIN_GI_GLSL
#define ORRIN_GI_GLSL

#include "octahedral.glsl"

// Keep in sync with gfx/probes.rs.
const int GI_CASCADES = 6;
const int GI_TILE = 8;
const int GI_TILE_BORDER = 10;

// A probe's state is its offset from its grid position, in metres, and a word
// of the coordinate it was written for, whether it is outside geometry, and
// how many updates it has had.
const uint GI_STAMP_BITS = (1u << 27u) - 1u;
const uint GI_ACTIVE = 1u << 27u;
const uint GI_COUNT_SHIFT = 28u;

// `GpuGi` in gfx/vulkan/gi.rs.
struct GpuGi {
    // x = probes a side, y = rays a probe, z = 1 while the volumes are live,
    // w = the grid this frame updates.
    uvec4 grid;
    // x = this frame's number, y = frames a cache cell outlives its last use,
    // z = the debug view, w = the cache's size less one.
    uvec4 frame;
    // x = the fraction of a probe kept per update, y = the cache's finest cell
    // in metres, zw = one over the atlases' extent.
    vec4 params;
    // xyz = the camera, w = 1 where the environment's reflection is re-fitted
    // to local light.
    vec4 camera;
    // xy = the environment's yaw, as `lighting.environment` carries it; z = 1
    // where the froxel fog takes its ambient light from the volumes; w = 1
    // where the final gather lights the opaque surfaces in their place.
    vec4 environment;
    // The rotation this frame's rays are turned by, column by column; and in
    // the first column's w, how far along its step each ray's height is.
    vec4 rotation[3];
    // xyz = the coordinate of a grid's lowest probe, w = its spacing.
    vec4 cascades[GI_CASCADES];
    // The sky's radiance in nine spherical-harmonic terms, the sun left out.
    vec4 sky[9];
};

layout(set = ORRIN_GI_SET, binding = ORRIN_GI_BLOCK_BINDING, std140) uniform GiBlock {
    GpuGi gi;
};

#ifdef ORRIN_GI_PROBES_WRITE
layout(set = ORRIN_GI_SET, binding = ORRIN_GI_PROBES_BINDING, std430) buffer GiProbes {
    uvec4 gi_probes[];
};
#else
layout(set = ORRIN_GI_SET, binding = ORRIN_GI_PROBES_BINDING, std430) readonly buffer GiProbes {
    uvec4 gi_probes[];
};
#endif

bool gi_live() {
    return gi.grid.z != 0u;
}

bool gi_gathered() {
    return gi.environment.w > 0.5;
}

ivec3 gi_base(int level) {
    return ivec3(gi.cascades[level].xyz);
}

float gi_spacing(int level) {
    return gi.cascades[level].w;
}

// Where the probe at a coordinate stands before it is moved, and the
// coordinate a point falls at. `PHASE` in gfx/probes.rs, which says why.
const float GI_PHASE = 0.37;

vec3 gi_position(int level, ivec3 world) {
    return (vec3(world) + GI_PHASE) * gi_spacing(level);
}

vec3 gi_coordinate(int level, vec3 P) {
    return P / gi_spacing(level) - GI_PHASE;
}

// `slot` in gfx/probes.rs. Offset to positive first, because GLSL leaves the
// remainder of a negative operand undefined.
uvec3 gi_slot(ivec3 world) {
    uint n = gi.grid.x;
    return uvec3(world + ivec3(int(n) << 20)) % n;
}

// The slot one step on from `slot` along each axis `step` is set on: the
// remainder `gi_slot` takes, without taking it eight times over.
uvec3 gi_next_slot(uvec3 slot, ivec3 step) {
    uvec3 next = slot + uvec3(step);
    return mix(next, next - gi.grid.x, greaterThanEqual(next, uvec3(gi.grid.x)));
}

// The probe a grid holds at a slot: the one coordinate in the grid with that
// remainder.
ivec3 gi_world_of(int level, uvec3 slot) {
    ivec3 base = gi_base(level);
    return base + ivec3((slot + uvec3(gi.grid.x) - gi_slot(base)) % gi.grid.x);
}

uint gi_probe_index(int level, uvec3 slot) {
    uint n = gi.grid.x;
    return uint(level) * n * n * n + (slot.z * n + slot.y) * n + slot.x;
}

uvec2 gi_tile(int level, uvec3 slot) {
    uint n = gi.grid.x;
    return uvec2(slot.x + slot.z * n, slot.y + uint(level) * n) * uint(GI_TILE_BORDER);
}

uint gi_stamp(ivec3 world) {
    return uint(world.x & 511) | (uint(world.y & 511) << 9u) | (uint(world.z & 511) << 18u);
}

uint gi_count(uvec4 state) {
    return state.w >> GI_COUNT_SHIFT;
}

// Whether a slot's state is its own probe's.
bool gi_owned(uvec4 state, ivec3 world) {
    return (state.w & GI_STAMP_BITS) == gi_stamp(world);
}

// Whether a probe has light to give: its own, outside geometry, and updated.
bool gi_usable(uvec4 state, ivec3 world) {
    return gi_owned(state, world) && (state.w & GI_ACTIVE) != 0u && gi_count(state) > 0u;
}

vec3 gi_offset(uvec4 state) {
    return uintBitsToFloat(state.xyz);
}

// `oct_decode`'s inverse: a unit direction onto the square [-1, 1]^2.
vec2 gi_oct_encode(vec3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    vec2 e = n.xy;
    if (n.z < 0.0) {
        e = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return e;
}

// Ray `ray` of `rays` of the probe in slot `probe` of the grid this frame
// updates: a spherical Fibonacci set, as even over the sphere as a set of
// points gets, slid along its axis and turned by this frame's update, and by
// the probe's own constant share of a step more. `gfx/probes.rs` says why.
vec3 gi_ray_direction(uint ray, uint rays, uint probe) {
    const float GOLDEN_ANGLE = 2.39996323;
    uint h = (probe + gi.grid.w * 0x9e3779b9u) * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h ^= h >> 22u;
    vec2 own = vec2(h & 0xffffu, h >> 16u) * (1.0 / 65536.0);
    float z = 1.0 - 2.0 * (float(ray) + fract(gi.rotation[0].w + own.x)) / float(rays);
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = float(ray) * GOLDEN_ANGLE + own.y * 6.28318531;
    vec3 d = vec3(r * cos(phi), r * sin(phi), z);
    return mat3(gi.rotation[0].xyz, gi.rotation[1].xyz, gi.rotation[2].xyz) * d;
}

// The sky's radiance along `d`: the environment's diffuse series with its
// cosine lobe divided back out, so a probe that sees only sky reads exactly
// the irradiance `sh_irradiance` gives a surface under it. The sun is left out
// of that series for the reason `gfx/sh.rs` gives — the directional light
// already delivers it, shadowed — and a ray drawn at random would otherwise
// find a disc four orders of magnitude over the sky once in a few thousand.
vec3 gi_sky(vec3 d) {
    float s = gi.environment.x;
    float c = gi.environment.y;
    vec3 n = vec3(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
    vec3 e = gi.sky[0].rgb * 0.282095
           + gi.sky[1].rgb * (0.488603 * n.y)
           + gi.sky[2].rgb * (0.488603 * n.z)
           + gi.sky[3].rgb * (0.488603 * n.x)
           + gi.sky[4].rgb * (1.092548 * n.x * n.y)
           + gi.sky[5].rgb * (1.092548 * n.y * n.z)
           + gi.sky[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0))
           + gi.sky[7].rgb * (1.092548 * n.x * n.z)
           + gi.sky[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));
    return max(e, vec3(0.0));
}

// The irradiance the sky alone gives a surface facing `n`: `gi_sky` with
// each band's cosine lobe put back, which is `sh_irradiance` for a pass that
// has no lighting block.
vec3 gi_sky_irradiance(vec3 d) {
    float s = gi.environment.x;
    float c = gi.environment.y;
    vec3 n = vec3(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
    const float BAND1 = 2.0 / 3.0;
    const float BAND2 = 0.25;
    vec3 e = gi.sky[0].rgb * 0.282095
           + gi.sky[1].rgb * (BAND1 * 0.488603 * n.y)
           + gi.sky[2].rgb * (BAND1 * 0.488603 * n.z)
           + gi.sky[3].rgb * (BAND1 * 0.488603 * n.x)
           + gi.sky[4].rgb * (BAND2 * 1.092548 * n.x * n.y)
           + gi.sky[5].rgb * (BAND2 * 1.092548 * n.y * n.z)
           + gi.sky[6].rgb * (BAND2 * 0.315392 * (3.0 * n.z * n.z - 1.0))
           + gi.sky[7].rgb * (BAND2 * 1.092548 * n.x * n.z)
           + gi.sky[8].rgb * (BAND2 * 0.546274 * (n.x * n.x - n.y * n.y));
    return max(e, vec3(0.0));
}

// What the environment's reflection is scaled by where the volumes say less
// of the sky's light arrives than the sky alone would give: the ratio of the
// two irradiances (Lazarov 2013, Hobson 2019), so a dark room's speculars stop
// glowing with sky. Never above one: a reflection of the environment is still
// a reflection of the environment, however brightly the room is lit.
float gi_reflection_scale(vec3 local, vec3 sky) {
    const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
    float skylight = dot(sky, LUMA);
    return skylight > 1e-6 ? clamp(dot(local, LUMA) / skylight, 0.0, 1.0) : 1.0;
}

#ifdef ORRIN_GI_IRRADIANCE_BINDING
layout(set = ORRIN_GI_SET, binding = ORRIN_GI_IRRADIANCE_BINDING) uniform texture2D gi_irradiance_atlas;
layout(set = ORRIN_GI_SET, binding = ORRIN_GI_DISTANCE_BINDING) uniform texture2D gi_distance_atlas;
layout(set = ORRIN_GI_SET, binding = ORRIN_GI_SAMPLER_BINDING) uniform sampler gi_sampler;

vec2 gi_atlas_uv(int level, uvec3 slot, vec3 dir) {
    vec2 texel = vec2(gi_tile(level, slot)) + 1.0 + (gi_oct_encode(dir) * 0.5 + 0.5) * float(GI_TILE);
    return texel * gi.params.zw;
}

vec3 gi_irradiance_texel(int level, uvec3 slot, vec3 dir) {
    return textureLod(sampler2D(gi_irradiance_atlas, gi_sampler), gi_atlas_uv(level, slot, dir), 0.0).rgb;
}

// How much of a point's answer a grid gives: none at its edge, where the
// eight probes round the point run out, and all of it two probes in.
float gi_cascade_weight(int level, vec3 P) {
    vec3 grid = gi_coordinate(level, P) - vec3(gi_base(level));
    vec3 inside = min(grid, vec3(float(gi.grid.x) - 1.0) - grid);
    return clamp(min(inside.x, min(inside.y, inside.z)) * 0.5, 0.0, 1.0);
}

// Whether a probe sees a point `to_point` from it: one-sided Chebyshev on the
// distances it saw in that direction, so a probe behind a wall gives the
// point next to nothing. The distances are stored in spacings.
//
// And nothing at all past twice the nearest thing it saw round that
// direction. A tile is eight texels a side, so one texel takes in a roof's
// edge and the open air beyond it, and the mean and variance of those two say
// the probe may well see a point under the roof. The nearest says a roof is
// there. A floor seen at a grazing angle has the same shape — near under the
// probe, far along the texel — and loses its furthest probes to this, which
// hold the least of its light.
#ifdef ORRIN_GI_TRACED_VISIBILITY
// Whether anything stands between the point being looked up and `probe`.
bool gi_probe_hidden(vec3 probe);
#endif

const vec2 GI_NEAR = vec2(1.15, 1.6);

float gi_visibility(int level, uvec3 slot, vec3 to_point) {
    float s = gi_spacing(level);
    float r = length(to_point);
    vec3 seen = textureLod(sampler2D(gi_distance_atlas, gi_sampler),
                           gi_atlas_uv(level, slot, to_point / max(r, 1e-5)), 0.0).rgb;
    vec2 moments = seen.rg * vec2(s, s * s);
    float near = 1.0 - smoothstep(GI_NEAR.x, GI_NEAR.y, r / max(seen.b * s, 1e-4));
    if (r <= moments.x) {
        return near;
    }

    float variance = abs(moments.x * moments.x - moments.y);
    float d = r - moments.x;
    float chebyshev = variance / (variance + d * d);
    return chebyshev * chebyshev * chebyshev * near;
}

// How much a probe counts for by where it stands against the surface: all of
// it in front, and next to nothing once it is behind the surface's own plane
// by more than a shallow angle. The published weight wraps round to the back
// and keeps a fifth there, for a surface every probe is behind; but a probe
// behind a wall's plane is as a rule beyond the wall, the distances it holds
// are too coarse to say so at a grazing angle, and a fifth of a sunlit yard is
// more light than the room has. What is kept is enough to outweigh nothing.
float gi_facing(float cosine) {
    return max(smoothstep(-0.15, 0.3, cosine), 1e-3);
}

// What one grid says arrives at a surface at `P` facing `N`, seen along `V`,
// and in `a` whether it could say anything. The eight probes round the point
// are weighted by their trilinear share, by how far the surface faces them,
// and by whether they can see it (Majercik et al. 2019). The point is moved
// off the surface first, toward the viewer and out along the normal, so a
// probe just behind the surface does not read as seeing it.
vec4 gi_cascade_irradiance(int level, vec3 P, vec3 N, vec3 V) {
    float s = gi_spacing(level);
    vec3 biased = P + (N * 0.2 + V * 0.8) * (0.3 * s);
    ivec3 base = gi_base(level);
    vec3 grid = gi_coordinate(level, biased) - vec3(base);
    ivec3 lo = clamp(ivec3(floor(grid)), ivec3(0), ivec3(int(gi.grid.x) - 2));
    vec3 alpha = clamp(grid - vec3(lo), 0.0, 1.0);
    uvec3 first = gi_slot(base + lo);

    vec3 sum = vec3(0.0);
    float total = 0.0;
    for (int i = 0; i < 8; ++i) {
        ivec3 corner = ivec3(i, i >> 1, i >> 2) & 1;
        ivec3 world = base + lo + corner;
        uvec3 slot = gi_next_slot(first, corner);
        uvec4 state = gi_probes[gi_probe_index(level, slot)];
        if (!gi_usable(state, world)) {
            continue;
        }
        vec3 probe = gi_position(level, world) + gi_offset(state);
        vec3 to_probe = probe - P;
        float weight = gi_facing(dot(to_probe / max(length(to_probe), 1e-5), N));
#ifdef ORRIN_GI_TRACED_VISIBILITY
        // No ray for a probe behind the surface, which counts for nothing
        // whether or not it can be reached.
        if (weight <= 1e-3 || gi_probe_hidden(probe)) {
            continue;
        }
#endif
        // Never quite zero, so a point every probe is cut off from still has
        // an answer; and crushed rather than kept linear below a fifth, so a
        // probe that barely sees the point outweighs none but cannot tint it.
        weight = max(weight * gi_visibility(level, slot, biased - probe), 1e-6);
        const float CRUSH = 0.2;
        if (weight < CRUSH) {
            weight *= weight * weight / (CRUSH * CRUSH);
        }
        vec3 trilinear = mix(1.0 - alpha, alpha, vec3(corner));
        weight *= max(trilinear.x * trilinear.y * trilinear.z, 1e-4);
        sum += gi_irradiance_texel(level, slot, N) * weight;
        total += weight;
    }
    return total > 1e-8 ? vec4(sum / total, 1.0) : vec4(0.0);
}

// What arrives at a surface, from the finest grids that hold it: `rgb` in the
// unit `sh_irradiance` returns, and `a` how much of the answer the grids
// gave. The rest is the includer's to make up, from the sky.
vec4 gi_irradiance(vec3 P, vec3 N, vec3 V) {
    vec3 sum = vec3(0.0);
    float left = 1.0;
    for (int level = 0; level < GI_CASCADES && left > 1e-3; ++level) {
        float weight = gi_cascade_weight(level, P);
        if (weight <= 0.0) {
            continue;
        }
        vec4 e = gi_cascade_irradiance(level, P, N, V);
        float take = left * weight * e.a;
        sum += e.rgb * take;
        left -= take;
    }
    return vec4(sum, 1.0 - left);
}

// How much of a point the grids reach at all, whether or not a probe there
// can answer yet: what `gi_irradiance` leaves the sky to make up for a point
// past the last of them.
float gi_reach(vec3 P) {
    float beyond = 1.0;
    for (int level = 0; level < GI_CASCADES; ++level) {
        beyond *= 1.0 - gi_cascade_weight(level, P);
    }
    return 1.0 - beyond;
}

// The mean radiance arriving at a point of air from every direction, which is
// what an isotropic medium scatters. The mean of a probe's tile along the six
// axes, which for anything within the first three spherical-harmonic bands —
// as irradiance is, to within a percent — is its mean over the sphere.
vec4 gi_cascade_mean(int level, vec3 P) {
    float s = gi_spacing(level);
    ivec3 base = gi_base(level);
    vec3 grid = gi_coordinate(level, P) - vec3(base);
    ivec3 lo = clamp(ivec3(floor(grid)), ivec3(0), ivec3(int(gi.grid.x) - 2));
    vec3 alpha = clamp(grid - vec3(lo), 0.0, 1.0);
    uvec3 first = gi_slot(base + lo);

    vec3 sum = vec3(0.0);
    float total = 0.0;
    for (int i = 0; i < 8; ++i) {
        ivec3 corner = ivec3(i, i >> 1, i >> 2) & 1;
        ivec3 world = base + lo + corner;
        uvec3 slot = gi_next_slot(first, corner);
        uvec4 state = gi_probes[gi_probe_index(level, slot)];
        if (!gi_usable(state, world)) {
            continue;
        }
        vec3 probe = gi_position(level, world) + gi_offset(state);
        vec3 trilinear = mix(1.0 - alpha, alpha, vec3(corner));
        // No floor under the visibility, unlike a surface's: a point of air no
        // probe can see is inside something, and is left to the ambient.
        float weight = gi_visibility(level, slot, P - probe)
                     * max(trilinear.x * trilinear.y * trilinear.z, 1e-4);
        vec3 mean = gi_irradiance_texel(level, slot, vec3(1.0, 0.0, 0.0))
                  + gi_irradiance_texel(level, slot, vec3(-1.0, 0.0, 0.0))
                  + gi_irradiance_texel(level, slot, vec3(0.0, 1.0, 0.0))
                  + gi_irradiance_texel(level, slot, vec3(0.0, -1.0, 0.0))
                  + gi_irradiance_texel(level, slot, vec3(0.0, 0.0, 1.0))
                  + gi_irradiance_texel(level, slot, vec3(0.0, 0.0, -1.0));
        sum += mean * (weight / 6.0);
        total += weight;
    }
    return total > 1e-4 ? vec4(sum / total, 1.0) : vec4(0.0);
}

vec4 gi_mean_radiance(vec3 P) {
    vec3 sum = vec3(0.0);
    float left = 1.0;
    for (int level = 0; level < GI_CASCADES && left > 1e-3; ++level) {
        float weight = gi_cascade_weight(level, P);
        if (weight <= 0.0) {
            continue;
        }
        vec4 e = gi_cascade_mean(level, P);
        float take = left * weight * e.a;
        sum += e.rgb * take;
        left -= take;
    }
    return vec4(sum, 1.0 - left);
}
#endif

#endif
