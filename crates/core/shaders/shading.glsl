// The blocker search reads the shadow atlas by texel, which takes no sampler.
#extension GL_EXT_samplerless_texture_functions : require

// What a surface looks like, once. Included by `forward.frag`, which writes the
// result into the opaque target, and by `oit.frag`, which weights it and
// accumulates it — so a sheet of glass and the wall behind it are lit by the
// same renderer rather than by two that drifted.
//
// Everything below the descriptor declarations is pure: `shade_surface` reads
// the varyings and the bound sets and returns radiance plus opacity. The two
// includers differ only in what they do with that vec4, which is the whole
// point of the split.
//
// Define `ORRIN_TRANSPARENT` before including to shade a blended surface. It
// drops the two screen-space terms — ambient occlusion and the contact-shadow
// march — because both describe whatever the prepass rasterised at this pixel,
// and behind glass that is the wall, not the glass.
//
// Define `ORRIN_REFRACTIVE` instead to shade one the refraction pass draws. It
// drops the same two terms for the same reason, and adds one: the surface reads
// the frame behind it through set 5 and returns it bent, absorbed and weighted
// by what the surface did not reflect. Only that pass may define it — the
// descriptor set does not exist in the other two, and nothing else in the frame
// has a copy of itself to sample.
//
// Define `ORRIN_SUBSURFACE` to say that the frame carries a second colour target
// and runs the screen-space diffusion passes over it. It does two things: the
// diffusible half of the result is kept separate rather than summed, and the
// wrapped diffuse standing in for that diffusion is switched off, because
// applying both would soften the same transport twice. Only the opaque pass may
// define it — the other two queues never reach those passes, so there the wrap is
// the whole of the effect.
//
// The five extra lobes — clear coat, sheen, anisotropy, transmission,
// subsurface — are gated on bits in the material rather than on separate shaders.
// `v_material` is one value for a whole draw — a draw is one batch, and a batch
// is one (mesh, material) pair on both culling paths — so each test is a
// coherent branch a draw either takes whole or skips whole; a plain
// metallic-roughness material costs what it did before any of them existed.

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec3 v_tangent;
layout(location = 3) in vec3 v_bitangent;
layout(location = 4) in vec2 v_uv;
layout(location = 5) in vec3 v_color;
// The material row this instance is drawn with. See `forward.vert`.
layout(location = 6) flat in uint v_material;
// The light channels this object is in: a light whose own mask shares no bit
// with it does not light it. See `MeshLighting` in scene/components/light.rs.
layout(location = 7) flat in uint v_channels;

// Keep in sync with MAX_TEXTURES in forward.rs.
const int MAX_TEXTURES = 192;
const float PI = 3.14159265359;

// Brings `MAX_CASCADES`, the `Cascades` block and the shadow lookup itself. The
// froxel fog includes the same file, so a shaft in the air lands on the shadow
// its ground does.
#include "cascades.glsl"
// And the air itself: the froxel mapping, the phase function and the term
// applied at the end of `shade_surface`. `fog_scatter.comp` includes the same
// file, so the froxel a fragment reads is the froxel that was lit for it.
#include "fog.glsl"

// The lights and the block that describes the frame's lighting, at this
// pass's own bindings.
#define ORRIN_LIGHTS_SET 0
#define ORRIN_LIGHTING_BINDING 0
#define ORRIN_POINT_LIGHTS_BINDING 2
#define ORRIN_SPOT_LIGHTS_BINDING 3
#include "lights.glsl"

// Lock-stepped with `TILES_X`, `TILES_Y` and `SLICES` in gfx/clusters.rs.
const uint CLUSTER_TILES_X = 16u;
const uint CLUSTER_TILES_Y = 9u;
const uint CLUSTER_SLICES = 24u;
const uint CLUSTER_COUNT = CLUSTER_TILES_X * CLUSTER_TILES_Y * CLUSTER_SLICES;

// Per cluster: (point offset, point count, spot offset, spot count) into
// `cluster_lights`.
layout(set = 0, binding = 4, std430) readonly buffer LightClusters {
    uvec4 cluster_ranges[CLUSTER_COUNT];
    uint cluster_lights[];
};

// The cluster a fragment reads: its screen tile and the exponential depth slice
// its view depth falls in. Mirrored by `cluster_of` in gfx/clusters.rs, whose
// tests are what hold the binning conservative against this lookup.
uint light_cluster(vec2 frag, float view_depth) {
    vec2 tiles = vec2(CLUSTER_TILES_X, CLUSTER_TILES_Y);
    uvec2 tile = uvec2(clamp(floor(frag * lighting.viewport.zw * tiles), vec2(0.0), tiles - 1.0));
    float slice = floor(log(max(view_depth, 1e-6)) * lighting.clusters.x - lighting.clusters.y);
    uint z = uint(clamp(slice, 0.0, float(CLUSTER_SLICES - 1u)));
    return (z * CLUSTER_TILES_Y + tile.y) * CLUSTER_TILES_X + tile.x;
}

// The material row and its feature bits, shared with the prepass and the shadow
// pass so the three cannot disagree about the table's stride.
#include "material.glsl"

// Material table indexed by the per-draw material_index. A storage
// buffer so the array can be sized at runtime (one entry per material).
layout(set = 1, binding = 0, std430) readonly buffer Materials {
    GpuMaterial materials[];
};

// Textures are kept separate from the sampler: Metal/MoltenVK allows
// only 16 sampler states per stage but many sampled images, so a
// combined sampler2D[64] would blow the sampler limit. One shared
// sampler + an array of texture2D stays well under it.
layout(set = 2, binding = 0) uniform texture2D textures[MAX_TEXTURES];
layout(set = 2, binding = 1) uniform sampler tex_sampler;

// Screen-space ambient occlusion (blurred), sampled by screen-space UV.
layout(set = 3, binding = 0) uniform sampler2D u_ao;

// The shadow atlas, every light's views as tiles of one depth image, and the
// comparison sampler it is read through. Separated for the same reason the
// texture array above is: Metal allows far fewer samplers per stage than
// sampled images.
layout(set = 3, binding = 1) uniform texture2D u_shadow_atlas;
layout(set = 3, binding = 2) uniform samplerShadow u_shadow_cmp;

// Prefiltered specular environment: roughness increases with mip level. When no
// environment is loaded this is a 1x1 white cube and `env_specular` carries the
// scene's flat ambient, so there is one path here rather than two.
// Keep in sync with SPECULAR_MIPS in gfx/vulkan/environment.rs.
const float SPECULAR_MIPS = 6.0;
layout(set = 3, binding = 3) uniform textureCube u_environment;
layout(set = 3, binding = 4) uniform sampler u_environment_sampler;

// The sun's visibility over the short range a cascade texel cannot resolve,
// sampled by screen-space UV like `u_ao`. A 1x1 white view when the pass is off,
// so there is one path here rather than two.
layout(set = 3, binding = 5) uniform sampler2D u_contact_shadow;

// The table saying where in the atlas each view landed and what it was drawn
// with, indexed by the view number a punctual light carries. The sun's views
// are rows of it too, but are read out of `lighting.cascades`, which the fog
// shares.
struct ShadowFace {
    mat4 view_proj;
    vec4 rect;   // xy = atlas uv offset, zw = atlas uv scale
    vec4 params; // x = world size of a texel where clip w is one,
                 // y = penumbra width (m), z = sharpness
};
layout(set = 3, binding = 6, std430) readonly buffer ShadowFaces {
    ShadowFace shadow_faces[];
};

// The froxel fog: the integrated volume, and the block describing the medium it
// was integrated out of. The block is bound whether or not the volume was
// written, because the analytic term past the volume's far plane — and the whole
// ray in a frame with no volume at all — is described by the very same numbers.
// `camera_pos.w` is what says which of those a frame is in; the volume is a
// 1x1x1 stand-in when it is zero, so there is one binding here rather than two
// pipelines.
layout(set = 3, binding = 7) uniform sampler3D u_fog_volume;
layout(set = 3, binding = 8) uniform Fog {
    GpuFog f;
} u_fog;

// Index is dynamically uniform, so plain indexing is legal without the
// nonuniform qualifier. It comes from the material, and the material comes from
// `v_material`, which is one value for every invocation of a draw: a draw is one
// batch and a batch is one (mesh, material) pair. An invocation group never
// spans a draw, which is what makes "the same across the draw" enough.
vec4 sample_tex(uint index, vec2 uv) {
    return texture(sampler2D(textures[index], tex_sampler), uv);
}

#ifdef ORRIN_CUSTOM_SURFACE
float surface_time() {
    return lighting.params.y;
}
#endif

// The one sample the parallax march makes, with the gradients it is handed
// rather than the ones this fragment could derive. See parallax.glsl.
float parallax_height(uint index, vec2 uv, vec2 dx, vec2 dy) {
    return textureGrad(sampler2D(textures[index], tex_sampler), uv, dx, dy).r;
}
#include "parallax.glsl"
#include "normal_map.glsl"
#include "surface.glsl"

// The decal projector's one texture read, gradients handed in for the reason the
// march's are. See decals.glsl.
vec4 decal_sample(uint index, vec2 uv, vec2 dx, vec2 dy) {
    return textureGrad(sampler2D(textures[index], tex_sampler), uv, dx, dy);
}
#include "decals.glsl"

// Declared identically to the vertex shader so the stages share one
// push-constant range.
layout(push_constant) uniform Push {
    mat4 view_proj;
} push;

#include "lobes.glsl"

// Shared with ssr_resolve.comp, which subtracts exactly the environment term
// this pass adds. See brdf.glsl for why that has to be one definition.
#include "brdf.glsl"

// Specular antialiasing: widen linear roughness `a` to cover the normal
// variance inside this pixel, which the single-sample BRDF below cannot see.
// MSAA cannot do this — it supersamples coverage, not shader inputs.
// sigma = 0.5 px, the standard deviation of the pixel filter kernel in image
// space; the shader wants its square.
const float SPEC_AA_SIGMA2 = 0.25;
#include "ambient.glsl"

// The irradiance volumes, which stand in for the environment's diffuse
// series wherever they are live. Bound in every frame, as inert stand-ins
// where they are not, so there is one pipeline rather than two.
#define ORRIN_GI_SET 3
#define ORRIN_GI_BLOCK_BINDING 9
#define ORRIN_GI_PROBES_BINDING 10
#define ORRIN_GI_IRRADIANCE_BINDING 11
#define ORRIN_GI_DISTANCE_BINDING 12
#define ORRIN_GI_SAMPLER_BINDING 13
#include "gi.glsl"

// The final gather's irradiance at this pixel, as two bands of spherical
// harmonics a layer per colour (`gi_gather.comp`). Read by the opaque pass
// alone, by pixel: a blended or refractive surface is not the one the prepass
// saw here, and keeps the volumes. Bound in every frame, as a stand-in where
// the gather did not run, for the reason the volumes are.
layout(set = 3, binding = 14) uniform texture2DArray u_gathered;

vec3 gathered_irradiance(vec3 n) {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 r = texelFetch(u_gathered, ivec3(pixel, 0), 0);
    vec4 g = texelFetch(u_gathered, ivec3(pixel, 1), 0);
    vec4 b = texelFetch(u_gathered, ivec3(pixel, 2), 0);
    return max(vec3(r.x + dot(r.yzw, n), g.x + dot(g.yzw, n), b.x + dot(b.yzw, n)), vec3(0.0));
}

// Clamping threshold, from Kaplanyan et al. 2016.
const float SPEC_AA_KAPPA = 0.18;

// Specular antialiasing, Tokuyoshi & Kaplanyan 2019 listing 2: widen the linear
// roughness to cover the normal variance inside this pixel, which the
// single-sample BRDF below cannot see. MSAA does not help — it supersamples
// coverage, not shader inputs. Filtering off the normal rather than the
// halfvector costs the same regardless of light count and needs no tangent
// frame, so it holds up on meshes with degenerate tangents.
float filter_roughness(float a, vec3 N) {
    vec3 dndu = dFdx(N);
    vec3 dndv = dFdy(N);
    float variance = SPEC_AA_SIGMA2 * (dot(dndu, dndu) + dot(dndv, dndv));
    // Dropping the 2.0 gives the paper's less conservative variant: less
    // overfiltering, at the risk of underfiltering.
    float kernel_roughness2 = min(2.0 * variance, SPEC_AA_KAPPA);
    // The paper works in squared roughness, so square, widen, and root back.
    return sqrt(clamp(a * a + kernel_roughness2, 0.0, 1.0));
}

// How far past the terminator light that entered elsewhere is allowed to leave,
// per channel, in `[0, 1]`. Resolved once per fragment into `Surface::wrap`.
//
// The width is derived from the two lengths the material already carries rather
// than authored, and the ratio is the physical question: a mean free path that is
// short next to the body it is travelling through cannot reach around it, and one
// that is long next to it wraps completely. That is why a marble statue is
// Lambertian everywhere except its thin edges while a leaf of the same material is
// lit through — same medium, different thickness — and why nothing here needs a
// per-object dial.
//
// What the screen-space diffusion changes is not *whether* this applies but how
// much of the transport is left for it to describe. The two model the same
// physics, one on the lighting side and one on the image side, so where the
// diffusion can do the job this must get out of the way or the surface softens
// twice. But the diffusion's kernel is measured in pixels, and a mean free path
// of a few millimetres is a fraction of a pixel on anything but a close-up: a flat
// switch would mean a scattering object losing its soft terminator as it walked
// away from the camera, which is worse than double-counting. So the wrap fades out
// exactly as the kernel becomes resolvable, and the two hand over.
//
// The footprint comes from `fwidth` rather than from the projection and the depth:
// the derivative of the world position *is* the world size of a pixel here, so it
// needs nothing passed in and it accounts for foreshortening for free — a surface
// seen edge-on has a wide footprint and keeps its wrap, which is right, because
// that is also where the blur's own perpendicular-to-view estimate is least able
// to help.
vec3 subsurface_wrap(Surface s) {
    vec3 w = clamp(s.scatter_radius / max(vec3(s.thickness), s.scatter_radius), 0.0, 1.0);
#ifdef ORRIN_SUBSURFACE
    float pixel_world = max(length(fwidth(v_world_pos)), 1e-9);
    // Full below half a pixel, which is where `sss_blur.comp` gives up entirely,
    // and gone by two, where its taps are spread over real neighbours.
    float widest = max(max(s.scatter_radius.r, s.scatter_radius.g), s.scatter_radius.b);
    w *= 1.0 - smoothstep(0.5, 2.0, widest / pixel_world);
#endif
    return w;
}

#define ORRIN_CASCADES lighting.cascades
// A surface's sun shadow widens with its caster's distance. The fog's does
// not, and does not compile the search: a point of air has no plane to measure
// a caster from.
#define ORRIN_CONTACT_HARDENING
#include "cascade_lookup.glsl"

// --- Cascaded shadow maps ---
//
// The lookup itself is in `cascades.glsl`, shared with the froxel fog. What is
// left here is the part only a surface can say: the two early-outs, and the
// nested block the shared functions are handed.

// Sun visibility for a *surface*, blended across the cascade seam.
float sun_shadow(vec3 world_pos, vec3 N, vec3 L, float view_dist) {
    // A surface facing away from the sun is already unlit by it — `brdf`
    // returns zero for n_dot_l <= 0, so whatever the maps say gets multiplied
    // into nothing. Leaving before the lookup saves the kernel this cascade
    // would cost, and the second one a fragment in the blend band would.
    if (dot(N, L) <= 0.0) {
        return 1.0;
    }

    // Strength zero is the "shadows off" slider, and the blend below collapses
    // to a constant 1.0 at it. Same taps saved, for a setting rather than for
    // the geometry.
    if (lighting.cascades.params.z <= 0.0) {
        return 1.0;
    }

    return cascade_sun_shadow_hardened(
        u_shadow_atlas,
        u_shadow_cmp,
        world_pos,
        N,
        normalize(v_normal),
        L,
        view_dist
    );
}

// The sun's visibility at the *far* side of a scattering medium — whether light
// is arriving to be transmitted through it at all.
//
// Not `sun_shadow` handed an offset position, and the difference is the whole
// point: that function returns "lit" the moment the surface faces away from the
// sun, which is precisely the case this exists to answer. Skipping the maps there
// is a sound saving for a reflected lobe, whose radiance is about to be
// multiplied by a zero cosine, and it is wrong for a transmitted one — it would
// light a leaf indoors as brightly as one in a field.
//
// So the lookup is taken a thickness of medium behind this point, against the
// far face's own outward normal. Two consequences worth knowing: a thin surface
// samples essentially where it stands and reads lit, which is correct — a single
// leaf is the only thing between the sun and itself, and the normal-offset bias
// is what stops it from shadowing itself. And a closed body reads lit wherever
// its back face is, so it glows by its authored thickness rather than by the
// distance a ray would really cross. Recovering that distance means a
// transmittance depth out of the shadow map, which needs a non-comparison
// sampler this set layout does not carry.
float back_sun_shadow(vec3 world_pos, vec3 N, vec3 L, float view_dist, float thickness) {
    if (lighting.cascades.params.z <= 0.0) {
        return 1.0;
    }

    // The far face's own outward normal, which is what makes this a different
    // question rather than `sun_shadow` at an offset — and note the missing
    // `dot(N, L)` early-out, deliberately.
    return cascade_sun_shadow(
        u_shadow_atlas,
        u_shadow_cmp,
        world_pos - N * thickness,
        -N,
        L,
        view_dist
    );
}

// Distinct tint per cascade, for checking that the splits land where intended.
vec3 cascade_debug_tint(int cascade) {
    if (cascade == 0) return vec3(1.0, 0.4, 0.4);
    if (cascade == 1) return vec3(0.4, 1.0, 0.4);
    if (cascade == 2) return vec3(0.4, 0.6, 1.0);
    return vec3(1.0, 1.0, 0.4);
}

// What a frame did to a view, as a colour: grey for nothing, green for its
// moving casters redrawn, blue for a strip scrolled in, yellow for its cached
// layer redrawn, red for drawn whole. `ViewWork`'s order, in shadow_views.rs.
vec3 shadow_state_tint(float state) {
    if (state < 0.5) return vec3(0.55);
    if (state < 1.5) return vec3(0.4, 1.0, 0.4);
    if (state < 2.5) return vec3(0.4, 0.6, 1.0);
    if (state < 3.5) return vec3(1.0, 1.0, 0.4);
    return vec3(1.0, 0.4, 0.4);
}

// How a shadow texel compares with the pixel it shades: green at one for one,
// toward red at four times coarser, toward blue at four times finer.
vec3 shadow_density_tint(float ratio) {
    float stops = clamp(log2(max(ratio, 1e-6)) * 0.5, -1.0, 1.0);
    vec3 even = vec3(0.4, 1.0, 0.4);
    return stops > 0.0
        ? mix(even, vec3(1.0, 0.4, 0.4), stops)
        : mix(even, vec3(0.4, 0.6, 1.0), -stops);
}

// The atlas at `uv`: black where nothing was drawn, and where something was, its
// depth as a grey under the tint of the sun view whose tile it is — a lamp's
// tiles stay untinted. Depth is read the only way this stage can read it,
// through the comparison sampler, as how many of a ladder of depths it is
// nearer than; the ladder crowds both ends, because an orthographic view keeps
// its casters in a sliver of its range and a perspective face keeps nearly all
// of its depth at the far plane.
vec3 shadow_atlas_picture(vec2 uv) {
    const float ladder[12] = float[12](
        0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 0.9, 0.97, 0.99, 0.997, 0.9995
    );
    float nearer = 0.0;
    for (int i = 0; i < 12; ++i) {
        nearer += textureLod(sampler2DShadow(u_shadow_atlas, u_shadow_cmp), vec3(uv, ladder[i]), 0.0);
    }
    if (nearer > 11.5) return vec3(0.0);

    vec3 tint = vec3(1.0);
    int count = int(lighting.cascades.params.x);
    for (int i = 0; i < count; ++i) {
        vec4 rect = lighting.cascades.rects[i];
        vec2 local = (uv - rect.xy) / rect.zw;
        if (all(greaterThanEqual(local, vec2(0.0))) && all(lessThan(local, vec2(1.0)))) {
            tint = cascade_debug_tint(i);
        }
    }
    return tint * mix(1.0, 0.3, nearer / 12.0);
}

// --- Punctual shadows: a tile of the atlas per face ---

// Which cube face a direction belongs to, in the order `gfx/punctual.rs` builds
// them: +X, -X, +Y, -Y, +Z, -Z. That order is the contract between the two — the
// matrices carry every other convention, so this is the only thing that has to
// agree.
int cube_face(vec3 d) {
    vec3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) return d.x > 0.0 ? 0 : 1;
    if (a.y >= a.z)               return d.y > 0.0 ? 2 : 3;
    return d.z > 0.0 ? 4 : 5;
}

// Filtered visibility from one of a punctual light's views. 1.0 is lit.
float atlas_shadow(int face_index, vec3 world_pos, vec3 N, vec3 L) {
    ShadowFace face = shadow_faces[face_index];
    return shadow_view_visibility(
        u_shadow_atlas,
        u_shadow_cmp,
        face.view_proj,
        face.rect,
        face.params.x,
        vec3(face.params.yz, SHADOW_KERNEL_MIN),
        world_pos,
        N,
        L
    );
}

// Visibility for a point light: pick the face its direction falls on, then the
// lookup above. `first_face` is negative for a light the atlas had no room for,
// which is the unshadowed path.
float point_shadow(PointLight light, vec3 world_pos, vec3 N, vec3 L) {
    int first_face = int(light.shadow.x);
    if (first_face < 0) {
        return 1.0;
    }
    // The direction *from* the light, which is what the faces were built around.
    int face = cube_face(world_pos - light.position.xyz);
    return atlas_shadow(first_face + face, world_pos, N, L);
}

// Everything about this pixel that no light direction can change.
//
// Split out of `shade_surface` because the four lobes turned "sample the maps"
// into a third of the shader, and because the split names the rule the file is
// built on: what is read here is per fragment, what `brdf` does with it is per
// light. Anything that ended up on the wrong side of that line would be paid for
// once per light for no reason.
Surface read_surface(GpuMaterial m, out float alpha, out vec3 emissive) {
    Surface s;
    s.flags = m.tex_flags.y;

    // The interpolated frame, before any map has tilted it. The normal decode
    // below rotates out of it, and the parallax march walks along it: a height
    // field is defined against the surface the UVs were laid out on, not against
    // the one a normal map claims.
    mat3 TBN = mat3(normalize(v_tangent), normalize(v_bitangent), normalize(v_normal));
    s.V = normalize(lighting.camera_pos.xyz - v_world_pos);

    s.uv = v_uv;
    if ((s.flags & MATERIAL_PARALLAX) != 0u) {
        // Derivatives of the *unmarched* coordinate, taken here where the
        // control flow is still uniform across the quad. They are also the right
        // footprint to filter by: the ray descending into the field does not
        // make the wall's texels any smaller.
        vec2 dx = dFdx(v_uv);
        vec2 dy = dFdy(v_uv);
        s.uv = parallax_occlusion(
            m.tex_flags.w,
            s.uv,
            // Toward the eye, in tangent space: the basis is orthonormal, so the
            // transpose is the inverse and three dot products are the whole
            // rotation.
            normalize(vec3(dot(s.V, TBN[0]), dot(s.V, TBN[1]), dot(s.V, TBN[2]))),
            m.parallax.x,
            parallax_uv_per_metre(
                TBN[0], TBN[1], dx, dy, dFdx(v_world_pos), dFdy(v_world_pos)
            ),
            m.parallax.y,
            m.parallax.z,
            dx,
            dy
        );
    }

    SurfaceIn i;
    i.uv = s.uv;
    i.world_position = v_world_pos;
    i.world_normal = TBN[2];
    i.view_dir = s.V;
    i.vertex_color = v_color;
    SurfaceOut o = resolve_surface(m, i);

    s.albedo = o.albedo;
    alpha = o.alpha;
    s.metallic = o.metallic;
    s.perceptual_roughness = o.roughness;
    emissive = o.emissive;
    float reflectance = m.params.z;

    // Tangent-space normal map -> world space via the TBN basis.
    s.N = normalize(TBN * o.normal);

    // Decals, here and not a line either side of it. After the normal, because
    // the angle fade asks which way this surface faces and the mapped normal is
    // the honest answer; before `f0`, because a decal that stained a surface
    // copper without moving its specular colour would be paint over a mirror.
    // Everything below this line therefore describes the surface *with* the
    // decal on it, which is what makes a decal something the frame lights rather
    // than something drawn over the light.
    apply_decals(v_world_pos, s.albedo, s.N, s.metallic, s.perceptual_roughness);

    // Dielectric F0 from reflectance (0.5 -> ~4%); metals use albedo as F0.
    s.f0 = mix(vec3(0.16 * reflectance * reflectance), s.albedo, s.metallic);

    s.n_dot_v = max(dot(s.N, s.V), 1e-4);

    s.a = filter_roughness(s.perceptual_roughness * s.perceptual_roughness, s.N);
    // Constant across lights, so computed once here rather than once per light
    // inside brdf().
    s.energy = energy_compensation(s.f0, s.perceptual_roughness, s.n_dot_v);

    // Isotropic defaults, so `base_specular` can read these without asking
    // whether the branch below ran.
    s.T = normalize(v_tangent);
    s.B = normalize(v_bitangent);
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
    // One, not zero: every consumer divides by this, and the guard belongs here
    // rather than in the three places that use it.
    s.scatter_radius = vec3(1.0);
    s.thickness = 0.0;
    s.forward_scatter = 1.0;
    s.wrap = vec3(0.0);

    if ((s.flags & MATERIAL_ANISOTROPY) != 0u) {
        vec3 aniso_tex = sample_tex(m.tex_indices_ext.w, s.uv).rgb;
        // The map stores a signed tangent-space direction, so the neutral fill
        // is the flat normal's (0.5, 0.5) rather than white. That decodes to a
        // zero vector, which has no direction to rotate — the fallback below is
        // what turns "no map" into "along the tangent", the glTF default.
        vec2 dir = aniso_tex.rg * 2.0 - 1.0;
        if (dot(dir, dir) < 1e-8) {
            dir = vec2(1.0, 0.0);
        }
        // The material's own rotation, resolved to its sine and cosine on the
        // CPU because it is per material rather than per fragment.
        float c = m.anisotropy.y;
        float sn = m.anisotropy.z;
        vec2 rotated = vec2(dir.x * c - dir.y * sn, dir.x * sn + dir.y * c);

        vec3 stretch = TBN * vec3(rotated, 0.0);
        // Re-orthogonalised against the *shading* normal, not the geometric one:
        // a normal map has already tilted N away from the interpolated frame, and
        // a tangent that is no longer perpendicular to it stretches the lobe in a
        // direction the surface does not actually face.
        s.T = normalize(stretch - s.N * dot(s.N, stretch));
        s.B = normalize(cross(s.N, s.T));
        s.anisotropy = clamp(m.anisotropy.x * aniso_tex.b, -1.0, 1.0);
        s.at = max(s.a * (1.0 + s.anisotropy), MIN_ANISO_ROUGHNESS);
        s.ab = max(s.a * (1.0 - s.anisotropy), MIN_ANISO_ROUGHNESS);
    }

    if ((s.flags & MATERIAL_CLEARCOAT) != 0u) {
        vec2 coat_tex = sample_tex(m.tex_indices_ext.x, s.uv).rg;
        s.coat = clamp(m.clearcoat.x * coat_tex.r, 0.0, 1.0);
        s.coat_roughness = clamp(m.clearcoat.y * coat_tex.g, 0.04, 1.0);
        s.coat_a = s.coat_roughness * s.coat_roughness;
        // Defaults to the flat normal, which through the TBN is the *geometric*
        // normal rather than the base layer's normal-mapped one. That is glTF's
        // rule and the physical one: a smooth film over a grained surface still
        // reflects flat.
        vec3 coat_tangent = tangent_normal(sample_tex(m.tex_indices_ext.y, s.uv));
        s.coat_N = normalize(TBN * coat_tangent);
    }

    if ((s.flags & MATERIAL_SHEEN) != 0u) {
        vec4 sheen_tex = sample_tex(m.tex_indices_ext.z, s.uv);
        s.sheen_color = m.sheen.rgb * sheen_tex.rgb;
        // Floored well above the base layer's: Charlie's exponent is 1/a, so a
        // near-zero roughness is an exponent large enough to underflow the `pow`
        // in `distribution_charlie` rather than a tight highlight.
        s.sheen_a = clamp(m.sheen.a * sheen_tex.a, 0.07, 1.0);

        // Albedo scaling, so the fibres do not add energy on top of a base layer
        // that already reflected all of it. The exact factor is one minus the
        // Charlie lobe's directional albedo, which wants a lookup table this
        // renderer does not have yet; the sheen colour's own magnitude is a
        // strict upper bound on that albedo, so using it is conservative — it
        // over-darkens the base at grazing angles, where the true albedo is
        // below one, and never brightens it.
        s.albedo *= 1.0 - max(max(s.sheen_color.r, s.sheen_color.g), s.sheen_color.b);
    }

    if ((s.flags & MATERIAL_SUBSURFACE) != 0u) {
        s.scatter_color = m.subsurface.rgb * sample_tex(m.tex_flags.z, s.uv).rgb;
        s.scatter_radius = max(m.subsurface_radius.rgb, vec3(1e-6));
        s.forward_scatter = max(m.subsurface.a, 1e-3);
        // The same green channel the refraction volume reads its thickness out
        // of, and the same field on the CPU side. Two surfaces of one material
        // cannot disagree about how deep the material is.
        s.thickness = max(m.transmission.y * sample_tex(m.tex_flags.x, s.uv).g, 0.0);
        // Last, because it reads both of the lengths above.
        s.wrap = subsurface_wrap(s);
    }

    return s;
}

// Where the environment is sampled from for the base lobe.
//
// A stretched highlight reflects the world along a normal bent toward the
// stretch, not along the surface normal: without this an anisotropic surface
// gets the correct elongated response from every analytic light and a perfectly
// round one from the sky, which reads as the effect failing on exactly the
// materials it exists for.
vec3 ibl_reflection(Surface s) {
    if ((s.flags & MATERIAL_ANISOTROPY) == 0u) {
        return reflect(-s.V, s.N);
    }
    // The lobe stretches *across* its own axis, so the bend is about the other
    // one — the bitangent for a positive strength, the tangent for a negative.
    vec3 axis = s.anisotropy >= 0.0 ? s.B : s.T;
    vec3 tangent = cross(axis, s.V);
    vec3 bent = cross(tangent, axis);
    // Rough surfaces bend further, because a wide lobe reaches further around
    // the stretch before the environment stops changing across it.
    float amount = abs(s.anisotropy) * clamp(5.0 * s.perceptual_roughness, 0.0, 1.0);
    return reflect(-s.V, normalize(mix(s.N, normalize(bent), amount)));
}

#ifdef ORRIN_REFRACTIVE
// The lit frame as it stood before any refractive surface was drawn, in two
// forms, because the two ends of the roughness range want opposite things.
//
// `u_refraction_blur` is a mip pyramid based at *half* the frame: a rough
// surface refracts a cone, and the level whose texel matches that cone carries
// the average of what it covers — the same trick the reflection trace samples
// its source through, and for the same reason. Half rather than full because
// building that chain is the single most expensive thing this feature does, and
// every consumer of it is already discarding detail.
//
// `u_refraction_sharp` is the frame itself, and it exists because the one
// consumer that is *not* discarding detail is a clear pane of glass. Reading it
// out of a half-resolution chain would soften a background that should be as
// sharp as the frame recorded it, which is the whole reason the pyramid was
// built at full resolution first — at a cost of about half the frame.
//
// Keep in sync with SCENE_LEVELS in gfx/vulkan/refraction.rs.
const float REFRACTION_MIPS = 7.0;
layout(set = 5, binding = 0) uniform sampler2D u_refraction_blur;
layout(set = 5, binding = 1) uniform sampler2D u_refraction_sharp;

// What arrives through the surface from behind it.
//
// Screen-space, with everything that implies: it can only return light the frame
// already contains, so a refractive surface at the edge of the screen bends
// toward pixels that were never rendered (the clamped sampler holds the border
// steady instead), and one pane behind another samples the frame from *before*
// either was drawn rather than through the first. Both are the standard cost of
// doing this without tracing the scene, and both are why glass wants to be the
// last thing in a frame rather than the first.
vec3 transmitted_radiance(Surface s, GpuMaterial m, float transmission, float thickness) {
    if (transmission <= 0.0) {
        return vec3(0.0);
    }

    float ior = max(m.params.w, 1.0);
    // Where a ray entering here leaves the volume: bent at the front face and
    // carried `thickness` through it. A thin surface has no interior, so the
    // exit point *is* the entry point and the lookup comes back undistorted —
    // which is exactly right for a window pane and is why zero thickness is the
    // default rather than a degenerate case to guard.
    vec3 refracted = refract(-s.V, s.N, 1.0 / ior);
    vec4 clip = push.view_proj * vec4(v_world_pos + refracted * thickness, 1.0);

    // Behind the camera, the projection has no meaningful screen position to
    // give. Falling back to this pixel's own is the undistorted lookup, which is
    // wrong by exactly the refraction — and visible only where the surface is
    // already being viewed edge-on.
    vec2 uv = clip.w > 0.0
        ? (clip.xy / clip.w) * 0.5 + 0.5
        : gl_FragCoord.xy * lighting.viewport.zw;

    // Level 0 of the chain is already a 2:1 reduction, so it is not the sharp
    // answer even at lod 0 — the crossfade over the first level is what hands
    // smooth glass the untouched frame and everything rougher the pyramid,
    // without a threshold to band at.
    vec2 sample_uv = clamp(uv, 0.0, 1.0);
    float lod = s.perceptual_roughness * (REFRACTION_MIPS - 1.0);
    vec3 background = mix(
        texture(u_refraction_sharp, sample_uv).rgb,
        textureLod(u_refraction_blur, sample_uv, lod).rgb,
        clamp(lod, 0.0, 1.0)
    );

    // Beer-Lambert through the volume. The material carries the colour reached
    // at `attenuation_distance`, so the coefficient is its negative log over
    // that distance; a zero distance is the "absorbs nothing" case, which is
    // also what an infinite one packs down to on the CPU side.
    vec3 absorption = vec3(1.0);
    float attenuation_distance = m.transmission.z;
    if (attenuation_distance > 0.0 && thickness > 0.0) {
        vec3 sigma = -log(clamp(m.attenuation.rgb, 1e-4, 1.0)) / attenuation_distance;
        absorption = exp(-sigma * thickness);
    }

    // Weighted by what the surface did *not* reflect, under the identical
    // split-sum term the specular environment lookup added — the same reason
    // `brdf.glsl` exists. Anything else and a sheet of glass gains or loses
    // energy as it turns edge-on.
    vec2 dfg = env_brdf_approx(s.perceptual_roughness, s.n_dot_v);
    vec3 reflected = s.f0 * dfg.x + dfg.y;

    // And by the coat, if there is one: `shade_surface` applies `1 - Fc` to
    // everything under the film before this term is added, so it has to be
    // applied here by hand or the coat would attenuate the reflection off the
    // glass without attenuating what comes through it.
    float coat_transmittance = 1.0;
    if ((s.flags & MATERIAL_CLEARCOAT) != 0u) {
        coat_transmittance =
            1.0 - fresnel_schlick(max(dot(s.coat_N, s.V), 1e-4), 0.04) * s.coat;
    }

    return background * absorption * transmission * (1.0 - reflected) * coat_transmittance;
}
#endif

// What one surface sends toward the camera, in the two pieces the frame may
// treat differently.
//
// Every includer that has one target sums `color` and `diffusible`; the one that
// has two keeps them apart so the diffusion passes can spread the second before
// the frame adds it back. The split is meaningful for exactly one reason: light
// in `diffusible` did not leave the surface where it arrived, so moving it across
// the image is a correction rather than a smear.
struct Shaded {
    vec3 color;
    vec3 diffusible;
    /// How far across the surface, in metres, the diffusion may spread
    /// `diffusible`. Zero means this pixel does not scatter — which is what the
    /// blur and the composite test, so nothing else has to carry a mask.
    float scatter;
    float alpha;
};

Shaded shade_surface() {
    GpuMaterial m = materials[v_material];

    float alpha;
    // Read with the other maps, at the coordinate `read_surface` resolves: a
    // glowing filament that stayed behind while the brick around it moved would
    // be the parallax failing in the one place it is most visible.
    vec3 emissive;
    Surface s = read_surface(m, alpha, emissive);
    bool scatters = (s.flags & MATERIAL_SUBSURFACE) != 0u;

#if defined(ORRIN_TRANSPARENT) || defined(ORRIN_REFRACTIVE)
    // Both screen-space terms describe what the prepass rasterised at this
    // pixel, and neither a blended nor a refractive surface is in it —
    // occluding glass by the wall behind it is worse than not occluding it at
    // all.
    float ao = 1.0;
    float contact = 1.0;
#else
    float ao = texture(u_ao, gl_FragCoord.xy * lighting.viewport.zw).r;
    float contact = texture(u_contact_shadow, gl_FragCoord.xy * lighting.viewport.zw).r;
#endif

#ifdef ORRIN_REFRACTIVE
    // Zero unless the material actually asked to transmit, which is the branch
    // that keeps an opaque-but-queued surface from paying for a texture fetch
    // and a background sample it would multiply by nothing.
    float transmission = 0.0;
    float thickness = 0.0;
    if ((s.flags & MATERIAL_TRANSMISSION) != 0u) {
        vec2 transmission_tex = sample_tex(m.tex_flags.x, s.uv).rg;
        transmission = clamp(m.transmission.x * transmission_tex.r, 0.0, 1.0);
        thickness = max(m.transmission.y * transmission_tex.g, 0.0);
        // What refracts through the surface does not scatter off it, so the
        // diffuse lobe loses exactly what transmission takes. Applied to
        // `albedo` rather than threaded through `brdf` as a weight because `f0`
        // has already been derived above: scaling here reaches every diffuse
        // term — the analytic lights' and the irradiance probe's — and no
        // specular one.
        s.albedo *= 1.0 - transmission;
    }
#endif

    // Diffuse image-based lighting, attenuated by screen-space ambient
    // occlusion. The (1 - metallic) is the same factor `brdf` applies to its
    // own diffuse lobe: a metal has no diffuse response.
    //
    // Routed to the diffusible half for a scattering material, exactly as the
    // analytic lights' diffuse is: the sky's light enters the medium and comes
    // back out of it by the same physics a lamp's does, and leaving it behind
    // here would diffuse a face under a lamp and not the same face under an
    // overcast sky.
    //
    // The irradiance is the volumes' where they are live, which is light that
    // bounced as well as the sky's, and where they run out the sky's alone.
    // The volumes then also say how much less of the sky's light arrives here
    // than the sky alone would give, and the environment's reflections are
    // scaled by that, so a dark room's speculars stop glowing with sky.
    //
    // Where the final gather ran, an opaque surface takes what its rays found
    // instead, sky and all: the volumes are a metre apart, and the gather is
    // what sees the object beside this one. Its rays have already met what
    // the screen-space occlusion would darken, so that darkens only the
    // reflections here: with both, a block's foot measured a quarter darker
    // against the sheet round it than in the path-traced reference. The
    // occlusion is divided back out of the irradiance rather than left out of
    // the product below, which the compiler then builds as it always did: a
    // frame without the volumes comes out the same to the bit, and did not
    // with a second factor in its place.
    vec3 irradiance = sh_irradiance(s.N);
    float reflection = 1.0;
    float shown_ao = 1.0;
    if (gi_live()) {
        vec3 sky = irradiance;
#if defined(ORRIN_TRANSPARENT) || defined(ORRIN_REFRACTIVE)
        vec4 local = gi_irradiance(v_world_pos, s.N, s.V);
#else
        bool gathered = gi_gathered();
        vec4 local = gathered
            ? vec4(gathered_irradiance(s.N), 1.0)
            : gi_irradiance(v_world_pos, s.N, s.V);
#endif
        irradiance = local.rgb + sky * (1.0 - local.a);
        if (gi.camera.w > 0.5) {
            reflection = gi_reflection_scale(irradiance, sky);
        }
#if !defined(ORRIN_TRANSPARENT) && !defined(ORRIN_REFRACTIVE)
        if (gathered) {
            shown_ao = max(ao, 1e-3);
            irradiance /= shown_ao;
        }
#endif
    }
    // The debug view of the light that bounced: the irradiance alone, on
    // white.
    if (gi.frame.z == 1u) {
        Shaded shown;
        shown.color = irradiance * shown_ao;
        shown.diffusible = vec3(0.0);
        shown.scatter = 0.0;
        shown.alpha = alpha;
        return shown;
    }
    vec3 diffuse_ibl = irradiance * s.albedo * (1.0 - s.metallic) * ao;
    vec3 color = scatters ? vec3(0.0) : diffuse_ibl;
    vec3 diffusible = scatters ? diffuse_ibl : vec3(0.0);
    // The scale goes in with the energy term rather than over the result: the
    // compiler then builds the product it always did, and a frame with the
    // volumes off comes out the same to the bit.
    color += specular_ibl_along(
        ibl_reflection(s), s.n_dot_v, s.f0, s.perceptual_roughness, s.energy * reflection
    ) * ao;

    // The fibres' own environment response. Sampled through the same split-sum
    // chain as the base lobe, which is an approximation — the prefiltered mips
    // were convolved against GGX, not against Charlie — but the alternative is
    // cloth that has a rim under a lamp and none under a sky.
    if ((s.flags & MATERIAL_SHEEN) != 0u) {
        color += specular_ibl_along(
            reflect(-s.V, s.N), s.n_dot_v, s.sheen_color, s.sheen_a, vec3(1.0)
        ) * reflection * ao;
    }

    // The coat's environment reflection, and the same `1 - Fc` attenuation of
    // everything under it that `brdf` applies per light.
    //
    // Deliberately image-based even where screen-space reflections are on: the
    // geometry prepass carries one f0 and one roughness per pixel, and those
    // describe the base layer, so `ssr_resolve` has no coat lobe to replace.
    // Tracing the coat instead is the upgrade — it is the sharper and more
    // visible of the two — and it belongs with the reflections rather than here.
    if ((s.flags & MATERIAL_CLEARCOAT) != 0u) {
        float coat_n_dot_v = max(dot(s.coat_N, s.V), 1e-4);
        float Fc = fresnel_schlick(coat_n_dot_v, 0.04) * s.coat;
        color *= 1.0 - Fc;
        // Under the film for the same reason, and it has to be said twice
        // because the two halves are two accumulators now.
        diffusible *= 1.0 - Fc;
        color += specular_ibl_along(
            reflect(-s.V, s.coat_N), coat_n_dot_v, vec3(0.04), s.coat_roughness, vec3(1.0)
        ) * s.coat * reflection * ao;
    }

    // Directional sun, shadowed by the cascades and the screen-space march.
    // The environment term above is what `u_ao` attenuates instead; the
    // punctual lights below carry their own atlas lookups.
    float view_dist = length(v_world_pos - lighting.camera_pos.xyz);
    if ((floatBitsToUint(lighting.sun_direction.w) & v_channels) != 0u) {
        vec3 L = normalize(lighting.sun_direction.xyz);
        // Illuminance straight out of the uniform: `brdf` multiplies by n.l,
        // which is what turns light arriving perpendicular to itself into light
        // arriving at this surface.
        vec3 radiance = lighting.sun_color.rgb * lighting.sun_color.w;
        // The cascades and the screen-space march answer the same question at
        // two scales, so their answers multiply: the maps carry everything past
        // a texel's width of the contact, and the march carries the band inside
        // it that a map's own bias reports lit.
        float shadow = sun_shadow(v_world_pos, s.N, L, view_dist) * contact;
        // The far face's own visibility, and only when something is going to read
        // it. Without the contact mask deliberately: that march describes the
        // short range in front of *this* pixel, and the thin edges where it would
        // bite are the ones transmission exists to light.
        float back_shadow = scatters
            ? back_sun_shadow(v_world_pos, s.N, L, view_dist, s.thickness)
            : 1.0;
        Lobes lobes = brdf(s, L, radiance, shadow, back_shadow);
        color += lobes.direct;
        diffusible += lobes.diffusible;
    }

    // The lights that can reach this fragment's cluster, rather than every light
    // in the scene. Each still applies its own range test below: the binning is
    // conservative, so a list may name a light that just misses this pixel.
    float view_depth = dot(v_world_pos - lighting.camera_pos.xyz, lighting.view_forward.xyz);
    uvec4 cluster = cluster_ranges[light_cluster(gl_FragCoord.xy, view_depth)];

    // Point lights.
    for (uint k = 0u; k < cluster.y; ++k) {
        PointLight light = point_lights[cluster_lights[cluster.x + k]];
        if ((floatBitsToUint(light.shadow.z) & v_channels) == 0u) continue;
        vec3 to_light = light.position.xyz - v_world_pos;
        float dist = length(to_light);
        float atten = attenuate(dist, light.position.w);
        if (atten <= 0.0) continue;
        vec3 L = to_light / max(dist, 1e-4);
        // Before the atlas lookup, not after: a surface facing away is already
        // unlit by this light, and the kernel would be multiplied into a
        // zero. The same early-out `sun_shadow` makes for the cascades.
        //
        // Except on a scattering surface, where facing away from the light is not
        // the same as being unlit by it — that is the one case the light reaches
        // the camera by going *through*.
        if (!scatters && dot(s.N, L) <= 0.0) continue;
        vec3 radiance = light.color.rgb * light.color.w * atten;
        float shadow = point_shadow(light, v_world_pos, s.N, L);
        float back_shadow = scatters
            ? point_shadow(light, v_world_pos - s.N * s.thickness, -s.N, L)
            : 1.0;
        Lobes lobes = brdf(s, L, radiance, shadow, back_shadow);
        color += lobes.direct;
        diffusible += lobes.diffusible;
    }

    // Spot lights: a point light with a cone over it, and one atlas face rather
    // than six because a cone is a single frustum.
    for (uint k = 0u; k < cluster.w; ++k) {
        SpotLight light = spot_lights[cluster_lights[cluster.z + k]];
        if ((floatBitsToUint(light.params.w) & v_channels) == 0u) continue;
        vec3 to_light = light.position.xyz - v_world_pos;
        float dist = length(to_light);
        float atten = attenuate(dist, light.position.w);
        if (atten <= 0.0) continue;
        vec3 L = to_light / max(dist, 1e-4);
        if (!scatters && dot(s.N, L) <= 0.0) continue;
        atten *= cone_falloff(light, L);
        if (atten <= 0.0) continue;

        vec3 radiance = light.color.rgb * light.color.w * atten;
        int face = int(light.params.y);
        float shadow = face < 0 ? 1.0 : atlas_shadow(face, v_world_pos, s.N, L);
        float back_shadow = (face < 0 || !scatters)
            ? 1.0
            : atlas_shadow(face, v_world_pos - s.N * s.thickness, -s.N, L);
        Lobes lobes = brdf(s, L, radiance, shadow, back_shadow);
        color += lobes.direct;
        diffusible += lobes.diffusible;
    }

#ifdef ORRIN_REFRACTIVE
    color += transmitted_radiance(s, m, transmission, thickness);
#endif

    // Emissive adds on top, unaffected by scene lighting. Already a luminance
    // in nits, so it needs no conversion — it is the one material quantity in the
    // same unit as the target it is written into. Not diffusible: a filament is
    // where it is, and spreading it would be a bloom rather than a scattering.
    color += emissive;

    // Both halves lose what the air absorbed, and only one of them gains what it
    // scattered in. Summing the two afterwards gives back exactly the fogged
    // whole; adding the in-scatter to both would put the air in twice.
    //
    // This is where the froxel volume is read, and it is why the effect needs no
    // pass of its own downstream: three shaders reach this line — the opaque
    // one, the blended one and the refractive one — and each applies the fog at
    // *its own* depth. A composite over the finished frame could only ever apply
    // the opaque depth, and would fog a pane of glass as though it were the wall
    // behind it.
    FogTerm fog = fog_term(
        u_fog.f.fog, u_fog_volume, v_world_pos, gl_FragCoord.xy * lighting.viewport.zw
    );
    color = color * fog.transmittance + fog.inscatter;
    diffusible *= fog.transmittance;

    // Ways of looking at the shadows. The branch is on a uniform, so a frame
    // that looks at none of them pays one compare.
    float debug_view = lighting.cascades.params.w;
    if (debug_view > 0.5) {
        int cascade = select_cascade(view_dist);
        vec3 tint = cascade_debug_tint(cascade);
        if (debug_view > 3.5) {
            tint = shadow_state_tint(lighting.cascades.states[cascade]);
        } else if (debug_view > 2.5) {
            // The world size of a pixel here, against the world size of a texel
            // of the view it reads.
            float pixel = length(fwidth(v_world_pos)) * 0.70710678;
            tint = shadow_density_tint(lighting.cascades.texel_sizes[cascade] / max(pixel, 1e-6));
        } else if (debug_view > 1.5) {
            tint = vec3(1.0);
        }
        color *= tint;
        diffusible *= tint;

        // The atlas, in the corner: two fifths of the frame's height, and as
        // wide as its pages make it.
        vec2 atlas = vec2(textureSize(sampler2DShadow(u_shadow_atlas, u_shadow_cmp), 0));
        float side = lighting.viewport.y * 0.4;
        vec2 corner = vec2(gl_FragCoord.x * atlas.y / atlas.x, lighting.viewport.y - gl_FragCoord.y)
            / side;
        if (debug_view > 1.5 && debug_view < 2.5 && all(lessThan(corner, vec2(1.0)))) {
            // An eighth of the sky's luminance: a level the exposure this
            // frame is already metered for shows as a mid grey.
            color = shadow_atlas_picture(vec2(corner.x, 1.0 - corner.y))
                * lighting.ambient.w * 0.12;
            diffusible = vec3(0.0);
        }
    }

    Shaded shaded;
    shaded.color = color;
    shaded.diffusible = diffusible;
    // The widest channel, because that is the kernel the blur has to cover; the
    // narrower two are reached by weighting the taps it already took. Zero for a
    // material that does not scatter, which is the mask every pass downstream
    // reads.
    shaded.scatter = scatters
        ? max(max(s.scatter_radius.r, s.scatter_radius.g), s.scatter_radius.b)
        : 0.0;
    shaded.alpha = alpha;
    return shaded;
}

// The coverage an alpha-to-coverage cutout writes into its first colour target.
//
// Not a comparison. A hard `alpha < cutoff` resolves through MSAA to the same
// four-level staircase a cutout has always had, because every sample in the
// pixel takes the same branch — the fragment is what varies, not the sample. So
// this measures how far the pixel's alpha sits from the cutoff *in units of how
// fast alpha changes across the pixel*, which turns the step into a ramp one
// pixel wide and lets the rasteriser quantise it into the four samples the frame
// is already rasterising. That is the whole cost of the feature: one `fwidth`
// and a divide, on the foliage pipeline only.
//
// Called from `main` rather than from inside `shade_surface` for the reason
// `parallax.glsl` takes its gradients before the march: `fwidth` is a
// quad-differencing operation and belongs where the control flow is plainly
// uniform, not somewhere a reader has to prove that it is.
//
// One for a material that is not masked, so the plain pipeline — which has
// alpha to coverage off and ignores this — is never handed a number that would
// mean something if it were switched on.
float mask_coverage(float alpha) {
    GpuMaterial m = materials[v_material];
    if ((m.tex_flags.y & MATERIAL_MASKED) == 0u) {
        return 1.0;
    }
    // Floored, not just guarded against zero: where alpha is constant across
    // the quad — the interior of a leaf, or a whole card at a distance where
    // the map has mipped to a flat value — there is no edge to resolve and the
    // ramp collapses back to the hard test it is a smoothing of.
    float footprint = max(fwidth(alpha), 1e-5);
    return clamp((alpha - m.alpha.x) / footprint + 0.5, 0.0, 1.0);
}
