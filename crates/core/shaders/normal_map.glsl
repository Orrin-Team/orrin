// A tangent-space normal from a normal map's texel.
//
// An imported normal map is BC5, which keeps red and green only: its blue
// samples as 0, and Z is rebuilt from X and Y. Every other map stores Z and is
// read as it always was. Blue is enough to tell them apart because a stored 0
// would be a normal facing straight into the surface, which no tangent-space map
// holds.
//
// Not rebuilt for every map: a map whose mips the GPU made by blitting holds
// shortened averages below level 0, and Z rebuilt from a shortened X and Y
// leans the normal flatter than the average it stands for. Rebuilding every map
// moved the engine's own captures by up to 94 levels on up to 9% of pixels;
// this way they are unchanged to the bit. An imported chain is renormalised at
// every level (`asset_db::texture`), so for BC5 the rebuild is exact.
//
// Shared by `shading.glsl` and `prepass.frag` for the reason `parallax.glsl` is:
// the G-buffer normal and the shaded one must not drift apart.

#ifndef ORRIN_NORMAL_MAP_GLSL
#define ORRIN_NORMAL_MAP_GLSL

vec3 tangent_normal(vec4 texel) {
    vec2 xy = texel.xy * 2.0 - 1.0;
    float z = texel.z > 0.0 ? texel.z * 2.0 - 1.0 : sqrt(max(1.0 - dot(xy, xy), 0.0));
    return vec3(xy, z);
}

#endif
