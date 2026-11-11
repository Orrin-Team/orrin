// Projected decals, once. Included by `shading.glsl` — and so by the forward,
// transparency and refraction passes — and by `prepass.frag`, for the reason
// `parallax.glsl` is: a decal changes the albedo, the normal, the `f0` and the
// roughness of the pixel it lands on, and those are exactly the four things the
// prepass exists to report. A prepass that skipped decals would tell
// `ssr_resolve.comp` about the wall as it was before the stamp, and the reflection
// would be subtracted under a weight belonging to a surface nobody can see.
//
// The includer must define, before including this file:
//
//   vec4 decal_sample(uint index, vec2 uv, vec2 dx, vec2 dy)
//
// taking its gradients explicitly, as `parallax_height` does and for the same
// reason: the loop below is per fragment in its trip count, and a derivative
// taken in non-uniform control flow is undefined. The derivatives are computed
// once, up front, where the control flow is still uniform — and they are *exact*
// rather than estimated, because the decal's texture coordinate is an affine
// function of world position, so the world-space derivative pushed through the
// same matrix is the UV-space one.
//
// The texture indices are read out of `decals[i]` at a loop counter bounded by a
// frame-uniform count, so every invocation that reaches a given fetch reaches it
// with the same index. That is what makes plain indexing legal here without
// `nonuniformEXT` — the same argument `sample_tex` makes about the material's
// own texture indices, one level further out.

// Keep in sync with MAX_DECALS in gfx/mod.rs.
const int MAX_DECALS = 16;

// Mirrors `decal_flags` in forward.rs. Both bits say whether a block was
// *authored*, because for both there is no neutral value to fall back on: a
// decal with no normal map would decode the flat-normal default to a perturbation
// of nothing — harmless — but a decal with no roughness has no number that means
// "leave what was already there", and 0 is mirror-smooth.
const uint DECAL_NORMAL  = 1u << 0;
const uint DECAL_SURFACE = 1u << 1;

// Mirrors GpuDecal in forward.rs, field for field.
struct GpuDecal {
    mat4 world_to_decal;
    vec4 base_color; // rgb = tint, a = opacity
    vec4 params;     // x = metallic, y = roughness, z = normal strength, w = cos(fade angle)
    uvec4 tex;       // x = albedo, y = normal, z = metal-rough, w = feature flags
    vec4 axis_u;     // the decal's +X in world space
    vec4 axis_v;     // its +Y
    vec4 axis_n;     // its +Z; the projection runs along the negative of this
};

// Set 0 binding 1 in *both* includers, which is not a coincidence and is worth
// stating: the forward pass's set 0 is the lighting block and the prepass's is
// the camera block, and binding 1 of each is this one buffer object. One
// allocation, bound twice, so the two passes cannot be handed different decals.
layout(set = 0, binding = 1, std430) readonly buffer Decals {
    uvec4 decal_count; // x = how many of the array below are live
    GpuDecal decals[MAX_DECALS];
} decal_block;

/// Stamp every decal covering this fragment onto the surface it describes.
///
/// Called after the material's own maps have been read and before anything is
/// derived from them — `f0`, the anisotropic roughnesses, the energy
/// compensation — because what comes out of here *is* the surface. A decal is
/// not composited over the lighting; it is part of what gets lit, which is what
/// puts it under the shadows, the ambient occlusion and the reflections instead
/// of on top of them.
///
/// `N` is world-space and is both read and written: read because the angle fade
/// asks which way the receiver faces, written because a decal's normal map is
/// most of what makes a bullet hole read as a hole.
void apply_decals(
    vec3 world_pos,
    inout vec3 albedo,
    inout vec3 N,
    inout float metallic,
    inout float perceptual_roughness
) {
    uint count = min(decal_block.decal_count.x, uint(MAX_DECALS));
    // Frame-uniform, so a scene with no decals pays one scalar comparison for
    // the whole feature — including the two derivatives below, which are not
    // free.
    if (count == 0u) {
        return;
    }

    vec3 ddx_world = dFdx(world_pos);
    vec3 ddy_world = dFdy(world_pos);

    for (uint i = 0u; i < count; ++i) {
        GpuDecal d = decal_block.decals[i];

        // Into the decal's unit cube. One matrix-vector product answers both
        // "is this fragment inside the box" and "where in the map", which is
        // the whole reason the projection is authored as a box rather than as a
        // plane and a depth range.
        vec3 local = (d.world_to_decal * vec4(world_pos, 1.0)).xyz;
        if (any(greaterThan(abs(local), vec3(0.5)))) {
            continue;
        }

        // How far the receiver's normal has turned away from the projector.
        // Without this a box that reaches a floor also reaches the skirting
        // board beside it, and the decal smears down the vertical face in the
        // stretched streaks that are this technique's signature failure. Faded
        // over the first quarter of the surviving range rather than clipped,
        // because a hard cutoff is an artefact of its own.
        float facing = dot(N, d.axis_n.xyz);
        float fade = smoothstep(d.params.w, mix(d.params.w, 1.0, 0.25), facing);
        if (fade <= 0.0) {
            continue;
        }

        // `v` runs down the decal's +Y, so a map drops in the way an image
        // viewer shows it. The bitangent is therefore -axis_v, which is what
        // the normal decode below uses.
        vec2 uv = vec2(local.x + 0.5, 0.5 - local.y);
        vec3 dlocal_x = mat3(d.world_to_decal) * ddx_world;
        vec3 dlocal_y = mat3(d.world_to_decal) * ddy_world;
        vec2 duv_dx = vec2(dlocal_x.x, -dlocal_x.y);
        vec2 duv_dy = vec2(dlocal_y.x, -dlocal_y.y);

        vec4 albedo_tex = decal_sample(d.tex.x, uv, duv_dx, duv_dy);
        float coverage = clamp(d.base_color.a * albedo_tex.a * fade, 0.0, 1.0);
        if (coverage <= 0.0) {
            continue;
        }

        albedo = mix(albedo, d.base_color.rgb * albedo_tex.rgb, coverage);

        if ((d.tex.w & DECAL_NORMAL) != 0u) {
            vec2 tangent_xy = decal_sample(d.tex.y, uv, duv_dx, duv_dy).xy * 2.0 - 1.0;
            // Added to the receiver's normal rather than replacing it, and that
            // is deliberate. A replacement would make a decal with a flat normal
            // map — or a strength of zero — snap the surface to face the
            // projector, which is a hole in the wall wherever a decal only meant
            // to stain it. Adding the map's own gradient instead composes with
            // whatever normal map the receiver already had, and does nothing
            // when there is nothing to add.
            vec3 tilt = d.axis_u.xyz * tangent_xy.x - d.axis_v.xyz * tangent_xy.y;
            N = normalize(N + tilt * (coverage * d.params.z));
        }

        if ((d.tex.w & DECAL_SURFACE) != 0u) {
            // glTF's packing, as everywhere else: G = roughness, B = metallic.
            vec4 mr = decal_sample(d.tex.z, uv, duv_dx, duv_dy);
            metallic = mix(metallic, clamp(d.params.x * mr.b, 0.0, 1.0), coverage);
            // The same floor `standard_surface` applies, for the same reason: a
            // roughness at zero is a singular highlight, and a decal is as able
            // to ask for one as a material is.
            perceptual_roughness =
                mix(perceptual_roughness, clamp(d.params.y * mr.g, 0.04, 1.0), coverage);
        }
    }
}
