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

// The forward pass's material table, through the one declaration of its row:
// the passes read the same buffer through their own layouts, so a change there
// is a change to both.
//
// The lobe blocks are in the row but unread here, deliberately: this pass
// writes the *base* layer's f0 and roughness and `ssr_resolve.comp` subtracts
// exactly that. The clear coat's own reflection stays image-based, so nothing
// here has a second lobe to describe.
//
// Parallax is the one that is read, because it is not a lobe: it changes *where*
// the maps are sampled rather than what is done with them, so this pass and the
// forward pass have to march the same ray to be describing one pixel. A prepass
// that skipped it would report the flat wall's normal for the pixel the forward
// pass shaded one brick along. The cutoff in `alpha` is read only by the
// ORRIN_MASKED variant below.
#include "material.glsl"
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

float parallax_height(uint index, vec2 uv, vec2 dx, vec2 dy) {
    return textureGrad(sampler2D(u_textures[index], u_sampler), uv, dx, dy).r;
}
#include "parallax.glsl"
#include "normal_map.glsl"
#include "surface.glsl"

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

    // The same reading `shading.glsl` makes, from the same file, so it has to be
    // handed the same values: world space, not this pass's view space.
    SurfaceIn i;
    i.uv = uv;
    i.world_position = v_world_pos;
    i.vertex_color = v_color;
#ifdef ORRIN_CUSTOM_SURFACE
    mat3 view_to_world = transpose(mat3(frame.view));
    i.world_normal = view_to_world * TBN[2];
    i.view_dir = view_to_world * normalize(-v_view_pos);
#else
    // Only a custom surface function reads these, so only its build pays for
    // rotating them out of view space.
    i.world_normal = vec3(0.0);
    i.view_dir = vec3(0.0);
#endif
    SurfaceOut o = resolve_surface(m, i);

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
    if (o.alpha < m.alpha.x) {
        discard;
    }
#endif

    vec3  albedo     = o.albedo;
    float metallic   = o.metallic;
    float roughness  = o.roughness;
    float reflectance = m.params.z;

    vec3 view_normal = normalize(TBN * o.normal);

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
