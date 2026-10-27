#version 460

layout(location = 0) in vec3 v_view_normal;
layout(location = 1) in vec3 v_view_tangent;
layout(location = 2) in vec3 v_view_bitangent;
layout(location = 3) in vec2 v_uv;
layout(location = 4) in vec3 v_color;
layout(location = 5) in vec4 v_clip;
layout(location = 6) in vec4 v_previous_clip;
layout(location = 7) in vec3 v_view_pos;
layout(location = 8) in vec3 v_world_pos;
// The material row this instance is drawn with. See `prepass.vert`.
layout(location = 9) flat in uint v_material;

layout(location = 0) out vec4 f_normal;
layout(location = 1) out vec2 f_velocity;
layout(location = 2) out vec4 f_material;

// Mirrors GpuMaterial in forward.rs, field for field: the two passes read the
// same buffer through their own layouts, so a change here is a change there.
//
// The lobe blocks are declared but unread, and both halves of that are
// deliberate. Declared, because std430 derives the array stride from the struct
// and a short mirror would step through the table at the wrong pitch — every
// material past the first would be read out of the middle of its neighbour.
// Unread, because this pass writes the *base* layer's f0 and roughness and
// `ssr_resolve.comp` subtracts exactly that: the clear coat's own reflection
// stays image-based, so nothing here has a second lobe to describe.
//
// Parallax is the one that is read, because it is not a lobe: it changes *where*
// the maps are sampled rather than what is done with them, so this pass and the
// forward pass have to march the same ray to be describing one pixel.
struct GpuMaterial {
    vec4 base_color;
    vec4 emissive;
    vec4 params;      // x = metallic, y = roughness, z = reflectance, w = ior
    uvec4 tex_indices; // [albedo, normal, metal-rough, emissive]
    vec4 clearcoat;
    vec4 sheen;
    vec4 anisotropy;
    vec4 transmission;
    vec4 attenuation;
    vec4 subsurface;
    vec4 subsurface_radius;
    // Read here, unlike the blocks around it: the march moves the coordinate
    // every one of the three outputs below is sampled at, so a prepass that
    // skipped it would report the flat wall's normal for the pixel the forward
    // pass shaded one brick along.
    vec4 parallax;    // x = height field depth in metres, y = min steps, z = max steps
    // Read only by the ORRIN_MASKED variant below, and declared unconditionally
    // for the stride reason the lobe blocks are.
    vec4 alpha;       // x = the alpha a MASKED fragment must reach to survive
    uvec4 tex_indices_ext;
    uvec4 tex_flags;  // x = transmission map, y = feature flags, z = subsurface, w = height
};
layout(set = 2, binding = 0, std430) readonly buffer Materials {
    GpuMaterial materials[];
};

// Keep in sync with MAX_TEXTURES in gfx/mod.rs, as forward.frag does. One
// shared sampler beside an array of textures rather than that many combined
// ones, for the sampler-limit reason that shader documents.
const int MAX_TEXTURES = 192;
layout(set = 3, binding = 0) uniform texture2D u_textures[MAX_TEXTURES];
layout(set = 3, binding = 1) uniform sampler u_sampler;

vec4 sample_tex(uint index, vec2 uv) {
    return texture(sampler2D(u_textures[index], u_sampler), uv);
}

// Feature bits in `tex_flags.y`; only the one this pass acts on is named here.
// Mirrors `material_flags` in forward.rs, as `shading.glsl` does.
const uint MATERIAL_PARALLAX = 1u << 5;

float parallax_height(uint index, vec2 uv, vec2 dx, vec2 dy) {
    return textureGrad(sampler2D(u_textures[index], u_sampler), uv, dx, dy).r;
}
#include "parallax.glsl"
#include "normal_map.glsl"

vec4 decal_sample(uint index, vec2 uv, vec2 dx, vec2 dy) {
    return textureGrad(sampler2D(u_textures[index], u_sampler), uv, dx, dy);
}
#include "decals.glsl"

// The camera block, declared here as well as in the vertex stage. This pass
// reports normals in view space and a decal is authored in world space, so the
// rotation between them is needed in the fragment stage — and it has to be
// *this* frame's, not one derived from anything else, for the reason every pass
// that rasterises geometry takes its matrices from `taa::FrameView`.
layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 proj;
    mat4 inv_proj;
    mat4 prev_view_proj;
    vec4 jitter;
} frame;

void main() {
    GpuMaterial m = materials[v_material];

    mat3 TBN = mat3(normalize(v_view_tangent), normalize(v_view_bitangent), normalize(v_view_normal));

    // The same march `shading.glsl` runs, one space along. `view` is rigid, so
    // the tangent-space ray and the metres the depth is authored in survive the
    // change of basis unchanged — which is exactly why the two passes can agree.
    vec2 uv = v_uv;
    if ((m.tex_flags.y & MATERIAL_PARALLAX) != 0u) {
        vec3 V = normalize(-v_view_pos); // the camera is the origin of view space
        vec2 dx = dFdx(v_uv);
        vec2 dy = dFdy(v_uv);
        uv = parallax_occlusion(
            m.tex_flags.w,
            uv,
            normalize(vec3(dot(V, TBN[0]), dot(V, TBN[1]), dot(V, TBN[2]))),
            m.parallax.x,
            parallax_uv_per_metre(
                TBN[0], TBN[1], dx, dy, dFdx(v_view_pos), dFdy(v_view_pos)
            ),
            m.parallax.y,
            m.parallax.z,
            dx,
            dy
        );
    }

    // Same reads and the same conventions as forward.frag: glTF packs roughness
    // in G and metallic in B, and the roughness floor keeps the highlight from
    // going singular.
    vec4  mr_tex     = sample_tex(m.tex_indices.z, uv);
    vec4  albedo_tex = sample_tex(m.tex_indices.x, uv);

#ifdef ORRIN_MASKED
    // This target is one sample, so there is no coverage to spend and the cut is
    // taken hard. The line it is taken along is deliberately the *middle* of the
    // ramp `mask_coverage` writes in the forward pass — half coverage is alpha
    // exactly at the cutoff — so the silhouette everything screen-space reads
    // agrees with the shaded one to within half a pixel rather than being
    // systematically fat or thin.
    //
    // In a variant of its own rather than behind a flag test, so that the plain
    // prepass shader still contains no `discard` and keeps its early depth test:
    // a cutout must not cost every other opaque material in the scene.
    if (m.base_color.a * albedo_tex.a < m.alpha.x) {
        discard;
    }
#endif

    vec3  albedo     = m.base_color.rgb * v_color * albedo_tex.rgb;
    float metallic   = clamp(m.params.x * mr_tex.b, 0.0, 1.0);
    float roughness  = clamp(m.params.y * mr_tex.g, 0.04, 1.0);
    float reflectance = m.params.z;

    vec3 n_tangent = tangent_normal(sample_tex(m.tex_indices.y, uv));
    vec3 view_normal = normalize(TBN * n_tangent);

    // The same stamp `read_surface` applies, in the same order, to the same four
    // quantities — which is the whole point of it being one file. `view` is
    // rigid, so its 3x3 is a rotation and the transpose is the inverse: the
    // normal goes out to world, takes the decal, and comes back, exactly
    // recovered where no decal touched it.
    vec3 world_normal = transpose(mat3(frame.view)) * view_normal;
    apply_decals(v_world_pos, albedo, world_normal, metallic, roughness);
    view_normal = mat3(frame.view) * world_normal;

    // Dielectric F0 from reflectance (0.5 -> ~4%); metals use albedo as F0.
    vec3 f0 = mix(vec3(0.16 * reflectance * reflectance), albedo, metallic);
    f_material = vec4(f0, roughness);

    f_normal = vec4(view_normal * 0.5 + 0.5, 1.0);

    if (v_previous_clip.w <= 0.0) {
        // Behind last frame's camera, so there is no history for this surface at
        // all. A vector this large lands the reprojection off-screen, which is
        // exactly the "reject the history" path the resolve already has.
        f_velocity = vec2(1e3);
        return;
    }

    // UV space rather than NDC, so the resolve can subtract it from a texture
    // coordinate directly. The projection already flips Y for Vulkan's clip
    // space, so this maps to the same orientation the render targets are in.
    vec2 now = (v_clip.xy / v_clip.w) * 0.5 + 0.5;
    vec2 before = (v_previous_clip.xy / v_previous_clip.w) * 0.5 + 0.5;
    f_velocity = now - before;
}
