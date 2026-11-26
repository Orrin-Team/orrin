// The frame's lights, once: the two punctual light records, the block that
// describes everything else about the lighting, and the two falloffs every
// loop over them applies. Included by `shading.glsl`, for a fragment, and by
// `ray_shade.glsl`, for a ray's hit, so a surface seen in a reflection is lit by
// the lights the surface beside it is.
//
// The includer says where the three blocks are bound, because each pass has a
// layout of its own:
//
//     #define ORRIN_LIGHTS_SET
//     #define ORRIN_LIGHTING_BINDING
//     #define ORRIN_POINT_LIGHTS_BINDING
//     #define ORRIN_SPOT_LIGHTS_BINDING
//
// Which lights reach a point is not here. A fragment reads its cluster's list
// and a hit its cell's, and the lists are the includer's.

#ifndef ORRIN_LIGHTS_GLSL
#define ORRIN_LIGHTS_GLSL

#include "cascades.glsl"

// Every quantity below is photometric, and the frame this shader writes is in
// nits (cd/m2) as a result: a candela over a squared distance is a lux, and a lux
// through a BRDF's 1/sr is a nit. That is what lets `exposure.rs` meter the
// histogram in real luminance and `HdrSettings` speak in EV100 rather than in
// multipliers nobody can transfer between scenes.
struct PointLight {
    vec4 position; // xyz = world position, w = range
    vec4 color;    // rgb = color,         w = luminous intensity (cd)
    vec4 shadow;   // x = first shadow view (< 0 = none), y = near plane, z = channel mask (bits)
};

// A light's shadow view is negative for two reasons a fragment does not tell
// apart and a ray does: -1 asked for a shadow and the atlas had no tile, -2
// casts none. Anything above this casts.
const float LIGHT_CASTS_NONE = -1.5;

struct SpotLight {
    vec4 position;  // xyz = world position, w = range
    vec4 direction; // xyz = cone axis,      w = cos(outer half angle)
    vec4 color;     // rgb = color,          w = luminous intensity (cd)
    vec4 params;    // x = cos(inner), y = shadow view (< 0 = none), z = near, w = channel mask (bits)
};

layout(set = ORRIN_LIGHTS_SET, binding = ORRIN_LIGHTING_BINDING) uniform Lighting {
    vec4 camera_pos;    // xyz = camera world position
    vec4 ambient;       // rgb = color, w = luminance (cd/m2)
    vec4 sun_direction; // xyz = direction toward the sun (normalized), w = channel mask (bits)
    vec4 sun_color;     // rgb = color, w = illuminance (lux)
    vec4 params;        // x = point light count, y = surface time (s), z legacy, w = spot count
    vec4 viewport;      // x=w, y=h, z=1/w, w=1/h
    // The four cascade fields, nested rather than written out flat. std140 lays
    // the struct out at exactly the offsets the flat fields had, so `GpuLighting`
    // is unchanged by the nesting — see `cascades.glsl`.
    Cascades cascades;
    vec4 clusters;       // x = depth slice scale, y = depth slice bias; see gfx/clusters.rs
    vec4 view_forward;   // xyz = the camera's forward axis, for view depth
    vec4 environment;    // x = sin(env yaw), y = cos(env yaw)
    vec4 env_specular;   // rgb = multiplier on sampled environment radiance
    vec4 irradiance[9];  // rgb = SH coefficient; see gfx/sh.rs
} lighting;

// The punctual lights, and which of them can reach each cluster of the view —
// `gfx/clusters.rs` bins them, and a fragment loops over its own cluster's
// lists rather than every light in the scene. Storage buffers, so the count is
// whatever the frame extracted rather than what a uniform block has room for.
const int MAX_POINT_LIGHTS = 256;
const int MAX_SPOT_LIGHTS = 64;
layout(set = ORRIN_LIGHTS_SET, binding = ORRIN_POINT_LIGHTS_BINDING) uniform PointLights {
    PointLight point_lights[MAX_POINT_LIGHTS];
};
layout(set = ORRIN_LIGHTS_SET, binding = ORRIN_SPOT_LIGHTS_BINDING) uniform SpotLights {
    SpotLight spot_lights[MAX_SPOT_LIGHTS];
};

// Smooth, range-limited falloff (windowed inverse-square). Multiplying a
// light's candela by this gives the illuminance it lays on a surface facing it,
// in lux — which is only true because one world unit is one metre.
//
// `range` is the window, not the physics: inverse-square never actually reaches
// zero, so a light would touch every pixel in the scene and cost a shadow lookup
// there. The window is what buys the early-out below, and it is a performance
// control rather than a photometric one.
float attenuate(float dist, float range) {
    float s = dist / max(range, 1e-4);
    if (s >= 1.0) return 0.0;
    float window = 1.0 - s * s;
    return (window * window) / max(dist * dist, 1e-4);
}

// The cone's own falloff: full inside the inner angle, out to nothing at the
// outer one. Smoothstepped rather than linear so the edge of a spot does not
// read as a hard ring on a flat floor.
float cone_falloff(SpotLight light, vec3 L) {
    // `L` points at the light and the axis points away from it, hence the
    // negation — the same relationship `sun_direction` has to the sun.
    float cosine = dot(-L, light.direction.xyz);
    float outer = light.direction.w;
    float inner = light.params.x;
    return smoothstep(outer, max(inner, outer + 1e-4), cosine);
}

#endif
