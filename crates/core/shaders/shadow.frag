#version 460

// Depth-only: the render pass has no color attachment and this stage writes
// nothing. A vertex-only pipeline is legal Vulkan for this, but MoltenVK's
// handling of a nil Metal fragment function has been inconsistent, and an empty
// shader costs nothing measurable.
//
// With ORRIN_MASKED it stops being empty and becomes the reason a leaf casts the
// shadow of a leaf rather than of the quad it is painted on. A cutout is opaque
// where it survives its own alpha test, so it is in every caster list — and a
// shadow map is nothing but depth, which means the test has to be repeated here
// or the map records the card.
//
// The map is one sample, so this is the hard comparison the prepass takes and
// not the coverage ramp the forward pass writes. Nothing samples a shadow map
// expecting a fractional occluder.

#ifdef ORRIN_MASKED
layout(location = 0) in vec2 v_uv;
// The material row this instance is drawn with. See `shadow.vert`.
layout(location = 1) flat in uint v_material;

// The forward pass's material table, through the row `material.glsl` declares
// for every pass that reads it. Only `base_color.a`, the albedo index and the
// cutoff are used; the rest is there for the stride.
#include "material.glsl"
layout(set = 1, binding = 0, std430) readonly buffer Materials {
    GpuMaterial materials[];
};

// Keep in sync with MAX_TEXTURES in gfx/mod.rs. One shared sampler beside an
// array of textures, for the per-stage sampler limit Metal imposes.
const int MAX_TEXTURES = 192;
layout(set = 2, binding = 0) uniform texture2D u_textures[MAX_TEXTURES];
layout(set = 2, binding = 1) uniform sampler u_sampler;

vec4 sample_tex(uint index, vec2 uv) {
    return texture(sampler2D(u_textures[index], u_sampler), uv);
}
// For `material_alpha`, the one definition of the alpha the forward pass's
// coverage and the prepass's `discard` cut along.
#include "surface.glsl"
#endif

void main() {
#ifdef ORRIN_MASKED
    GpuMaterial m = materials[v_material];
    // Deliberately not the parallax-marched coordinate the other two passes use.
    // The march needs a view direction, and the view here is the light's, so a
    // relief-mapped cutout would be tested along the wrong ray. A height field
    // moves texels by centimetres and a cutout's alpha is flat across those
    // centimetres in every atlas anyone authors, so the unmarched coordinate is
    // the right answer rather than a cheaper one.
    if (material_alpha(m, sample_tex(m.tex_indices.x, v_uv)) < m.alpha.x) {
        discard;
    }
#endif
}
