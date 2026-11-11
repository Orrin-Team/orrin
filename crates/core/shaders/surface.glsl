// What a surface is before it is lit, once: the material's maps read at one
// coordinate into albedo, alpha, metallic, roughness, emission and a normal.
// Included by `shading.glsl` — and so by the forward, transparency and
// refraction passes — and by `prepass.frag`, for the reason `parallax.glsl` is:
// the prepass reports the normal, f0 and roughness the forward pass shaded, and
// two readings that drifted would put the G-buffer under a different surface
// from the one in the frame.
//
// `SurfaceIn` holds only values that are identical in every pass that builds
// one: the coordinate after the parallax march, and world-space position,
// normal and view direction. Nothing in a pass's own space may go in — the
// prepass works in view space and the forward passes in world — and for the
// same reason `SurfaceOut::normal` is in tangent space: each pass rotates it
// through its own basis, which is how the two already agree.
//
// The includer defines, before including this file,
//
//     vec4 sample_tex(uint index, vec2 uv)
//
// as it defines `parallax_height` for `parallax.glsl`, because each pass binds
// its texture array under its own name.
//
// With `ORRIN_CUSTOM_SURFACE` defined, `resolve_surface` hands the standard
// reading to `void surface(in SurfaceIn i, inout SurfaceOut o)` from
// `custom_surface.glsl`, which the runtime compiler supplies rather than any
// file on disk, and then holds the result to the ranges the standard reading
// keeps to, so a custom function cannot hand the lighting a value no material
// could. The masked shadow pass reads only `material_alpha`, at the unmarched
// coordinate and with no world position to build a `SurfaceIn` from, so the
// hook does not reach it.

#ifndef ORRIN_SURFACE_GLSL
#define ORRIN_SURFACE_GLSL

#include "material.glsl"
#include "normal_map.glsl"

struct SurfaceIn {
    /// Where every map is read: the interpolated coordinate once the parallax
    /// march has moved it.
    vec2 uv;
    vec3 world_position;
    /// The interpolated geometric normal, unit length, before any map tilts it.
    vec3 world_normal;
    /// Unit length, from the surface toward the eye.
    vec3 view_dir;
    vec3 vertex_color;
};

struct SurfaceOut {
    vec3 albedo;
    float alpha;
    float metallic;
    /// Perceptual, as authored, not squared.
    float roughness;
    /// Luminance in cd/m2, added on top of the lit result.
    vec3 emissive;
    /// Tangent space. Each pass normalises it after rotating it into its own
    /// basis, so its length carries nothing.
    vec3 normal;
};

// The albedo map's alpha multiplies the material's, the same way its rgb
// multiplies the base colour. An opaque draw ignores the result and a blended
// one weights its radiance by it. A masked one is cut where it falls below
// `alpha.x`: by the forward pass's coverage ramp, the prepass's `discard` and
// the shadow pass's, which is why all three read it from here.
float material_alpha(GpuMaterial m, vec4 albedo_texel) {
    return clamp(m.base_color.a * albedo_texel.a, 0.0, 1.0);
}

// The material's own reading of its maps. Missing maps point at the default
// textures, so these multiplies become no-ops. Albedo/emissive images are sRGB
// (decoded to linear on sample); metal-rough is linear data.
SurfaceOut standard_surface(GpuMaterial m, SurfaceIn i) {
    vec4 albedo_tex = sample_tex(m.tex_indices.x, i.uv);
    vec4 mr_tex     = sample_tex(m.tex_indices.z, i.uv);

    SurfaceOut o;
    // Vertex color tints the material albedo; drop `* i.vertex_color` for a
    // pure material/texture color.
    o.albedo = m.base_color.rgb * i.vertex_color * albedo_tex.rgb;
    o.alpha = material_alpha(m, albedo_tex);
    // glTF metallic-roughness convention: G = roughness, B = metallic.
    o.metallic = clamp(m.params.x * mr_tex.b, 0.0, 1.0);
    // Floor avoids a singular highlight.
    o.roughness = clamp(m.params.y * mr_tex.g, 0.04, 1.0);
    o.emissive = m.emissive.rgb * sample_tex(m.tex_indices.w, i.uv).rgb;
    o.normal = tangent_normal(sample_tex(m.tex_indices.y, i.uv));
    return o;
}

#ifdef ORRIN_CUSTOM_SURFACE
#include "custom_surface.glsl"
#endif

SurfaceOut resolve_surface(GpuMaterial m, SurfaceIn i) {
    SurfaceOut o = standard_surface(m, i);
#ifdef ORRIN_CUSTOM_SURFACE
    surface(i, o);
    // Only behind the hook: the standard reading keeps to these ranges already,
    // and normalising a normal that every pass normalises again after rotating
    // it adds nothing but rounding, enough to move about a hundred pixels of the
    // demo capture.
    o.alpha = clamp(o.alpha, 0.0, 1.0);
    o.metallic = clamp(o.metallic, 0.0, 1.0);
    o.roughness = clamp(o.roughness, 0.04, 1.0);
    o.normal = normalize(o.normal);
#endif
    return o;
}

#endif
