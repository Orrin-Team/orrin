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

// Keep in sync with MAX_POINT_LIGHTS / MAX_SPOT_LIGHTS / MAX_TEXTURES in
// forward.rs.
const int MAX_POINT_LIGHTS = 16;
const int MAX_SPOT_LIGHTS = 8;
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

// Every quantity below is photometric, and the frame this shader writes is in
// nits (cd/m2) as a result: a candela over a squared distance is a lux, and a lux
// through a BRDF's 1/sr is a nit. That is what lets `exposure.rs` meter the
// histogram in real luminance and `HdrSettings` speak in EV100 rather than in
// multipliers nobody can transfer between scenes.
struct PointLight {
    vec4 position; // xyz = world position, w = range
    vec4 color;    // rgb = color,         w = luminous intensity (cd)
    vec4 shadow;   // x = first atlas face (< 0 = none), y = near plane
};

struct SpotLight {
    vec4 position;  // xyz = world position, w = range
    vec4 direction; // xyz = cone axis,      w = cos(outer half angle)
    vec4 color;     // rgb = color,          w = luminous intensity (cd)
    vec4 params;    // x = cos(inner), y = atlas face (< 0 = none), z = near
};

layout(set = 0, binding = 0) uniform Lighting {
    vec4 camera_pos;    // xyz = camera world position
    vec4 ambient;       // rgb = color, w = luminance (cd/m2)
    vec4 sun_direction; // xyz = direction toward the sun (normalized)
    vec4 sun_color;     // rgb = color, w = illuminance (lux)
    vec4 params;        // x = point light count (y,z legacy), w = spot count
    vec4 viewport;      // x=w, y=h, z=1/w, w=1/h
    // The four cascade fields, nested rather than written out flat. std140 lays
    // the struct out at exactly the offsets the flat fields had, so `GpuLighting`
    // is unchanged by the nesting — see `cascades.glsl`.
    Cascades cascades;
    PointLight point_lights[MAX_POINT_LIGHTS];
    SpotLight spot_lights[MAX_SPOT_LIGHTS];
    vec4 environment;    // x = sin(env yaw), y = cos(env yaw)
    vec4 env_specular;   // rgb = multiplier on sampled environment radiance
    vec4 irradiance[9];  // rgb = SH coefficient; see gfx/sh.rs
} lighting;

// Mirrors GpuMaterial in forward.rs. std430 packs this exactly like
// the Rust #[repr(C)] struct because every field is 16 bytes.
struct GpuMaterial {
    vec4 base_color;   // rgb = albedo, a = opacity
    vec4 emissive;     // rgb = emissive
    vec4 params;       // x = metallic, y = roughness, z = reflectance, w = ior
    uvec4 tex_indices; // x=albedo, y=normal, z=metal-rough, w=emissive
    vec4 clearcoat;    // x = strength, y = perceptual roughness
    vec4 sheen;        // rgb = colour, a = perceptual roughness
    vec4 anisotropy;   // x = strength, y = cos(rotation), z = sin(rotation)
    vec4 transmission; // x = transmission, y = thickness, z = attenuation distance
    vec4 attenuation;  // rgb = what the volume absorbs over `transmission.z`
    vec4 subsurface;   // rgb = scattering tint, a = forward-scatter power
    vec4 subsurface_radius; // rgb = per-channel mean free path, metres
    vec4 parallax;     // x = height field depth in metres, y = min steps, z = max steps
    vec4 alpha;        // x = the alpha a MASKED fragment must reach to survive
    uvec4 tex_indices_ext; // x=clearcoat, y=clearcoat normal, z=sheen, w=anisotropy
    uvec4 tex_flags;       // x = transmission map, y = feature flags, z = subsurface, w = height
};

// Feature bits in `tex_flags.y`, mirroring `material_flags` in forward.rs.
//
// `v_material` is one value for a whole draw, so every test against these is a
// coherent branch: a draw either takes a lobe or does not, and a material that
// never asked for one pays a single comparison rather than the lobe's cost.
const uint MATERIAL_CLEARCOAT    = 1u << 0;
const uint MATERIAL_SHEEN        = 1u << 1;
const uint MATERIAL_ANISOTROPY   = 1u << 2;
const uint MATERIAL_TRANSMISSION = 1u << 3;
const uint MATERIAL_SUBSURFACE   = 1u << 4;
const uint MATERIAL_PARALLAX     = 1u << 5;
const uint MATERIAL_MASKED       = 1u << 6;

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

// The cascade depth maps, one array layer each, and the comparison sampler
// they are read through. Separated for the same reason the texture array above
// is: Metal allows far fewer samplers per stage than sampled images.
layout(set = 3, binding = 1) uniform texture2DArray u_shadow_maps;
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

// The punctual shadow atlas, and the table saying where in it each light's
// faces landed. Read through `u_shadow_cmp`, the cascades' comparison sampler:
// same `Less` against a map cleared to 1.0, so one sampler serves both and the
// per-stage sampler budget Metal imposes stays untouched.
layout(set = 3, binding = 6) uniform texture2D u_shadow_atlas;

struct ShadowFace {
    mat4 view_proj;
    vec4 rect; // xy = atlas uv offset, zw = atlas uv scale
};
layout(set = 3, binding = 7, std430) readonly buffer ShadowFaces {
    ShadowFace shadow_faces[];
};

// The froxel fog: the integrated volume, and the block describing the medium it
// was integrated out of. The block is bound whether or not the volume was
// written, because the analytic term past the volume's far plane — and the whole
// ray in a frame with no volume at all — is described by the very same numbers.
// `camera_pos.w` is what says which of those a frame is in; the volume is a
// 1x1x1 stand-in when it is zero, so there is one binding here rather than two
// pipelines.
layout(set = 3, binding = 8) uniform sampler3D u_fog_volume;
layout(set = 3, binding = 9) uniform Fog {
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

// The one sample the parallax march makes, with the gradients it is handed
// rather than the ones this fragment could derive. See parallax.glsl.
float parallax_height(uint index, vec2 uv, vec2 dx, vec2 dy) {
    return textureGrad(sampler2D(textures[index], tex_sampler), uv, dx, dy).r;
}
#include "parallax.glsl"
#include "normal_map.glsl"

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

// --- Cook-Torrance terms (metallic-roughness workflow) ---

// GGX / Trowbridge-Reitz normal distribution.
float distribution_ggx(float n_dot_h, float a) {
    float a2 = a * a;
    float d = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// Smith height-correlated visibility (already folds in the 1/(4 NoL NoV) denom).
float visibility_smith_ggx(float n_dot_v, float n_dot_l, float a) {
    float a2 = a * a;
    float gv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    float gl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

// Fresnel-Schlick reflectance.
vec3 fresnel_schlick(float v_dot_h, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
}

float fresnel_schlick(float v_dot_h, float f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
}

// --- Anisotropic GGX (Burley 2012, in Filament's formulation) ---
//
// The isotropic pair above is this one with `at == ab`; the difference is that
// the halfvector is measured against the tangent frame rather than against the
// normal alone, which is what lets the highlight stretch. A brushed or woven
// surface has grooves running one way, and a lobe that cannot tell the two
// directions apart cannot show them.

// Smallest either roughness may reach. The distribution divides by a quantity
// that goes to zero with `at * ab`, so a fully anisotropic surface at zero
// roughness is a singularity rather than a sharp highlight.
const float MIN_ANISO_ROUGHNESS = 1e-3;

float distribution_ggx_aniso(float n_dot_h, float t_dot_h, float b_dot_h, float at, float ab) {
    float a2 = at * ab;
    vec3 d = vec3(ab * t_dot_h, at * b_dot_h, a2 * n_dot_h);
    float d2 = dot(d, d);
    float w2 = a2 / max(d2, 1e-9);
    return a2 * w2 * w2 / PI;
}

float visibility_smith_ggx_aniso(
    float n_dot_v, float n_dot_l,
    float t_dot_v, float b_dot_v,
    float t_dot_l, float b_dot_l,
    float at, float ab
) {
    float lambda_v = n_dot_l * length(vec3(at * t_dot_v, ab * b_dot_v, n_dot_v));
    float lambda_l = n_dot_v * length(vec3(at * t_dot_l, ab * b_dot_l, n_dot_l));
    return 0.5 / max(lambda_v + lambda_l, 1e-5);
}

// --- Clear coat ---
//
// Kelemen's visibility rather than Smith's. The coat is a thin, near-smooth
// film, and over that range the two agree closely enough that the cheaper one is
// free accuracy elsewhere — which matters because the coat is a *second* full
// specular evaluation on top of a surface that already paid for one.
float visibility_kelemen(float v_dot_h) {
    return 0.25 / max(v_dot_h * v_dot_h, 1e-5);
}

// --- Sheen (Estevez & Kulla 2017) ---
//
// The Charlie distribution, which unlike GGX peaks *away* from the normal — that
// retroreflective rim is the whole visual signature of cloth, and no amount of
// roughness on a GGX lobe produces it.
float distribution_charlie(float n_dot_h, float a) {
    float inv_a = 1.0 / max(a, 1e-3);
    float cos2 = n_dot_h * n_dot_h;
    // Floored rather than clamped to zero: `pow` of zero with a large exponent
    // is a legal way to get a NaN out of a surface facing exactly edge-on.
    float sin2 = max(1.0 - cos2, 0.0078125);
    return (2.0 + inv_a) * pow(sin2, inv_a * 0.5) / (2.0 * PI);
}

// Ashikhmin's uniform visibility term, as Neubelt & Pettineo use it. Smith's
// derivation assumes GGX microfacets and does not apply to Charlie's.
float visibility_neubelt(float n_dot_v, float n_dot_l) {
    return clamp(1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v)), 0.0, 1.0);
}

// Shared with ssr_resolve.comp, which subtracts exactly the environment term
// this pass adds. See brdf.glsl for why that has to be one definition.
#include "brdf.glsl"

// Specular antialiasing: widen linear roughness `a` to cover the normal
// variance inside this pixel, which the single-sample BRDF below cannot see.
// MSAA cannot do this — it supersamples coverage, not shader inputs.
// sigma = 0.5 px, the standard deviation of the pixel filter kernel in image
// space; the shader wants its square.
const float SPEC_AA_SIGMA2 = 0.25;
// Diffuse irradiance from the environment, divided by pi so the result is
// already the diffuse response for unit albedo.
//
// The nine terms are the real spherical-harmonic basis for bands 0..=2, and
// they must match `basis` in gfx/sh.rs term for term and component for
// component — the coefficients were projected against that one. A mismatch is
// not a compile error, it is lighting that is quietly rotated or mirrored.
//
// With no environment loaded these coefficients carry the scene's flat ambient
// in band 0 alone, which evaluates to that ambient for every normal. So there
// is one path here, not two.
// World space into the environment's own frame. The same rotation the skybox
// samples through, so what is drawn behind the scene and what lights it cannot
// disagree.
vec3 to_environment(vec3 v) {
    float s = lighting.environment.x;
    float c = lighting.environment.y;
    return vec3(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
}

vec3 sh_irradiance(vec3 n) {
    n = to_environment(n);

    vec3 e = lighting.irradiance[0].rgb * 0.282095
           + lighting.irradiance[1].rgb * (0.488603 * n.y)
           + lighting.irradiance[2].rgb * (0.488603 * n.z)
           + lighting.irradiance[3].rgb * (0.488603 * n.x)
           + lighting.irradiance[4].rgb * (1.092548 * n.x * n.y)
           + lighting.irradiance[5].rgb * (1.092548 * n.y * n.z)
           + lighting.irradiance[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0))
           + lighting.irradiance[7].rgb * (1.092548 * n.x * n.z)
           + lighting.irradiance[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));

    // Backstop for the ringing the band window in gfx/sh.rs is sized to
    // contain: a truncated series overshoots at a sun disc and undershoots
    // opposite it, and the undershoot can cross zero.
    return max(e, vec3(0.0));
}

// Specular image-based lighting, Karis' split sum: the prefiltered chain is the
// light integral, `env_brdf_approx` the BRDF integral. This is the term that
// gives a metal anything to be — its diffuse response is zero by definition, so
// without it a metal is lit only where it happens to mirror a light.
//
// `energy` is the same multi-scatter compensation the analytic lights get, and
// it belongs here for the same reason: single-scatter GGX loses energy with
// roughness whether the light is a sun or a sky.
// Takes the reflection vector rather than deriving it, because two callers do
// not want `reflect(-V, N)`: an anisotropic surface reflects about a normal bent
// toward its stretch, and a clear coat reflects about its own.
vec3 specular_ibl_along(vec3 R, float n_dot_v, vec3 f0, float perceptual_roughness, vec3 energy) {
    float lod = perceptual_roughness * (SPECULAR_MIPS - 1.0);
    vec3 radiance = textureLod(
        samplerCube(u_environment, u_environment_sampler), to_environment(R), lod
    ).rgb;

    vec2 dfg = env_brdf_approx(perceptual_roughness, n_dot_v);
    return radiance * lighting.env_specular.rgb * (f0 * dfg.x + dfg.y) * energy;
}

vec3 specular_ibl(vec3 N, vec3 V, vec3 f0, float perceptual_roughness, vec3 energy) {
    return specular_ibl_along(
        reflect(-V, N), max(dot(N, V), 1e-4), f0, perceptual_roughness, energy
    );
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

// Everything about this pixel that does not depend on which light is being
// summed, resolved once by `read_surface` and handed to every lobe below.
//
// A struct rather than a parameter list because the list had already reached
// nine and each of the three extra lobes wants four more. The fields past
// `energy` are only meaningful when the matching bit is set in `flags`; nothing
// reads them otherwise, so a plain material leaves them at whatever
// `read_surface` skipped writing.
struct Surface {
    vec3 N;
    vec3 V;
    /// Where this fragment actually reads the material's maps. `v_uv` until the
    /// parallax march moves it, and every map the material carries has to be
    /// sampled here rather than at the interpolated coordinate — a normal read
    /// one brick along from the albedo is worse than no parallax at all.
    vec2 uv;
    vec3 albedo;
    vec3 f0;
    float metallic;
    /// Linear roughness, already widened for specular antialiasing.
    float a;
    /// Perceptual roughness, which is what the split-sum fits are parameterised
    /// by — keeping both saves squaring and rooting back at every use.
    float perceptual_roughness;
    vec3 energy;
    float n_dot_v;
    uint flags;

    // Anisotropy: the tangent frame the lobe is stretched along, rotated out of
    // the geometric one by the material's angle, and the two roughnesses that
    // stretch it.
    vec3 T;
    vec3 B;
    float at;
    float ab;
    /// Signed strength, kept because the environment lookup needs it after the
    /// two roughnesses above have already absorbed it.
    float anisotropy;

    // Clear coat: its own normal, because an orange-peel film and the grain
    // beneath it are different surfaces.
    vec3 coat_N;
    float coat;
    float coat_a;
    float coat_roughness;

    vec3 sheen_color;
    float sheen_a;

    // Subsurface scattering: what comes back out of the medium, and the two
    // distances that decide where from.
    vec3 scatter_color;
    /// Per-channel mean free path, in metres. Red reaches furthest in every
    /// medium this exists for, which is why it is a vector.
    vec3 scatter_radius;
    /// Metres of medium behind this point. The reference length the wrap below is
    /// measured against, and the distance the transmitted term is absorbed over.
    float thickness;
    float forward_scatter;
    /// How far past the terminator the diffuse lobe reaches, per channel.
    /// Resolved once per fragment by `subsurface_wrap` — it reads a screen-space
    /// derivative, so it cannot be computed per light inside `brdf`.
    vec3 wrap;
};

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

// The diffuse cosine, widened by the wrap and renormalised so widening it does
// not also brighten it.
//
// The `(1 + w)^2` is what makes this an energy-preserving wrap rather than a
// wrap plus a gain: the numerator's range grows by `1 + w` and the lobe's
// integral over the hemisphere by another factor of it.
vec3 wrapped_diffuse(Surface s, float raw_n_dot_l) {
    vec3 wrapped = (vec3(raw_n_dot_l) + s.wrap) / ((1.0 + s.wrap) * (1.0 + s.wrap));
    return max(wrapped, vec3(0.0));
}

// Light that entered the far side of the surface and left through this one.
//
// Two things multiply: what survives `thickness` of medium — Beer-Lambert
// against the mean free path, per channel, which is what turns a thick limb
// opaque and leaves an ear glowing — and a phase function that is mostly
// forward. The forward lobe is why a leaf lights up when the camera is nearly
// looking into the sun through it and goes flat when it steps aside, and the
// `wrap_back` floor underneath keeps a surface lit from directly behind from
// going dark off-axis.
//
// Deliberately not multiplied by the surface's own albedo: this light never
// reflected off the boundary, it came *through*, so what tints it is the medium's
// colour and nothing else. The same rule `KHR_materials_diffuse_transmission`
// states.
vec3 subsurface_transmission(Surface s, vec3 L, float raw_n_dot_l) {
    vec3 through = exp(-s.thickness / s.scatter_radius);
    float forward = exp2(clamp(dot(s.V, -L), 0.0, 1.0) * s.forward_scatter - s.forward_scatter);
    float wrap_back = clamp(-raw_n_dot_l, 0.0, 1.0);
    return s.scatter_color * through * mix(wrap_back, 1.0, forward) / PI;
}

// One light's contribution, split by what the frame is allowed to spread.
//
// `direct` is everything whose position on screen is the surface it came off:
// every specular lobe, and the diffuse of a material that does not scatter.
// `diffusible` is the part that physically left the surface somewhere other than
// where it arrived, which is exactly the licence the screen-space passes need to
// move it — and is zero unless the material scatters, so a frame with no
// subsurface material in it has an empty second target.
struct Lobes {
    vec3 direct;
    vec3 diffusible;
};

// The base layer's specular lobe: GGX, stretched along the tangent frame when
// the material asked for it.
vec3 base_specular(Surface s, vec3 L, vec3 H, float n_dot_l, float n_dot_h) {
    float D;
    float Vis;
    if ((s.flags & MATERIAL_ANISOTROPY) != 0u) {
        D = distribution_ggx_aniso(n_dot_h, dot(s.T, H), dot(s.B, H), s.at, s.ab);
        Vis = visibility_smith_ggx_aniso(
            s.n_dot_v, n_dot_l,
            dot(s.T, s.V), dot(s.B, s.V),
            dot(s.T, L), dot(s.B, L),
            s.at, s.ab
        );
    } else {
        D = distribution_ggx(n_dot_h, s.a);
        Vis = visibility_smith_ggx(s.n_dot_v, n_dot_l, s.a);
    }
    return vec3(D * Vis);
}

// Outgoing radiance toward the camera from one light direction L.
// Takes the surface's linear roughness rather than its perceptual one: the
// caller filters it once for specular antialiasing, and this runs once per light.
//
// The two visibilities are the light's own shadow term, applied here rather than
// by the caller because a scattering surface needs two different ones and only
// this function knows which term wants which. `visibility` is whether the light
// reaches *this* point; `back_visibility` is whether it reaches the far side of
// the medium, which is the question the transmitted lobe is actually asking — an
// ear lit from behind is in shadow by the first measure and lit by the second,
// and answering it with the first is how subsurface scattering ends up invisible
// on exactly the geometry it exists for.
Lobes brdf(Surface s, vec3 L, vec3 radiance, float visibility, float back_visibility) {
    Lobes lobes;
    lobes.direct = vec3(0.0);
    lobes.diffusible = vec3(0.0);

    bool scatters = (s.flags & MATERIAL_SUBSURFACE) != 0u;
    float raw_n_dot_l = dot(s.N, L);

    // Before the facing test, and it has to survive it: this is light that
    // entered the other side, so a surface turned away from the light is where
    // the term does its work rather than where it stops.
    if (scatters) {
        lobes.diffusible =
            subsurface_transmission(s, L, raw_n_dot_l) * radiance * back_visibility;
    }

    float n_dot_l = max(raw_n_dot_l, 0.0);
    if (n_dot_l <= 0.0) {
        return lobes;
    }
    vec3 H = normalize(L + s.V);
    float n_dot_h = max(dot(s.N, H), 0.0);
    float v_dot_h = max(dot(s.V, H), 0.0);

    vec3 F = fresnel_schlick(v_dot_h, s.f0);
    vec3 specular = base_specular(s, L, H, n_dot_l, n_dot_h) * F * s.energy;

    // Diffuse keeps the energy not reflected (1 - F) and not metallic.
    vec3 kd = (vec3(1.0) - F) * (1.0 - s.metallic);

    vec3 color = specular;
    vec3 diffusible = vec3(0.0);
    if (scatters) {
        // The wrap *replaces* the cosine rather than scaling it — that is the
        // whole mechanism — so this term deliberately misses the `n_dot_l` the
        // returns below apply to everything else. Multiplying by both would
        // undo the wrap at the one place it matters, the band just inside the
        // terminator where `n_dot_l` is near zero.
        diffusible = kd * s.albedo * wrapped_diffuse(s, raw_n_dot_l) / PI;
    } else {
        color += kd * s.albedo / PI;
    }

    // Sheen sits beside the base lobes rather than over them: it is a separate
    // set of fibres catching light, not a film. The base is scaled down for it
    // once, in `read_surface`, rather than per light here.
    if ((s.flags & MATERIAL_SHEEN) != 0u) {
        color += s.sheen_color
               * distribution_charlie(n_dot_h, s.sheen_a)
               * visibility_neubelt(s.n_dot_v, n_dot_l);
    }

    if ((s.flags & MATERIAL_CLEARCOAT) == 0u) {
        lobes.direct += color * radiance * n_dot_l * visibility;
        lobes.diffusible += diffusible * radiance * visibility;
        return lobes;
    }

    // The coat is a film *over* everything above, so it both adds a reflection
    // and takes one away: what the base layer receives is what got *through* the
    // coat, which is the `1 - Fc`. Its own lobe is then lit by its own normal —
    // a coat normal map that disagrees with the base's is the entire reason the
    // two n·l terms are kept apart rather than folded into the common factor.
    //
    // A polyurethane film: 4% at normal incidence, as every clear coat is.
    float Fc = fresnel_schlick(v_dot_h, 0.04) * s.coat;
    float coat_n_dot_l = max(dot(s.coat_N, L), 0.0);
    float coat_n_dot_h = max(dot(s.coat_N, H), 0.0);
    float coat_spec = distribution_ggx(coat_n_dot_h, s.coat_a)
                    * visibility_kelemen(v_dot_h) * Fc;

    lobes.direct += (color * (1.0 - Fc) * n_dot_l + coat_spec * coat_n_dot_l)
                  * radiance * visibility;
    // Under the film like every other diffuse term. The transmitted half added
    // above is not: it left the medium through the *other* face, which has its
    // own coat if it has one at all, and attenuating it here would take a
    // reflection off this side out of light that never touched it.
    lobes.diffusible += diffusible * (1.0 - Fc) * radiance * visibility;
    return lobes;
}

// --- Cascaded shadow maps ---
//
// The lookup itself is in `cascades.glsl`, shared with the froxel fog. What is
// left here is the part only a surface can say: the two early-outs, and the
// nested block the shared functions are handed.

// Sun visibility for a *surface*, blended across the cascade seam.
float sun_shadow(vec3 world_pos, vec3 N, vec3 L, float view_dist) {
    // A surface facing away from the sun is already unlit by it — `brdf`
    // returns zero for n_dot_l <= 0, so whatever the maps say gets multiplied
    // into nothing. Leaving before the lookup saves the nine taps this cascade
    // would cost, and the eighteen a fragment in the blend band would.
    if (dot(N, L) <= 0.0) {
        return 1.0;
    }

    // Strength zero is the "shadows off" slider, and the blend below collapses
    // to a constant 1.0 at it. Same taps saved, for a setting rather than for
    // the geometry.
    if (lighting.cascades.params.z <= 0.0) {
        return 1.0;
    }

    return cascade_sun_shadow(
        lighting.cascades, u_shadow_maps, u_shadow_cmp, world_pos, N, L, view_dist
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
        lighting.cascades,
        u_shadow_maps,
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

// --- Punctual shadows: one atlas, a tile per face ---

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

// Percentage-closer visibility from one atlas face. 1.0 is lit.
//
// The taps are clamped inside the tile, which the cascades do not have to do:
// there, a tap past the edge hits the sampler's white border and reads as lit,
// while here it would land in a *neighbouring light's* depth and read as
// whatever that light happened to see. The border the faces are rendered with
// is what keeps that clamp reading real geometry rather than a repeated edge.
float atlas_shadow(int face_index, vec3 world_pos, vec3 N, vec3 L, float dist) {
    ShadowFace face = shadow_faces[face_index];

    // Normal-offset bias, as the cascades do it, but scaled by how much world
    // space a texel covers *here*: a perspective face's texel grows with
    // distance from the light, so a constant offset is either useless near it or
    // peter-panning far from it. `rect.z` is the tile's share of the atlas, so
    // the divisor is that tile's texel count along an edge.
    vec2 atlas_size = vec2(textureSize(sampler2DShadow(u_shadow_atlas, u_shadow_cmp), 0));
    float texel_world = 2.0 * dist / max(face.rect.z * atlas_size.x, 1.0);
    float slope = 1.0 - max(dot(N, L), 0.0);
    vec4 clip = face.view_proj * vec4(world_pos + N * texel_world * 1.4142136 * (1.0 + slope), 1.0);
    if (clip.w <= 0.0) {
        return 1.0;
    }

    vec3 ndc = clip.xyz / clip.w;
    // Past the light's own far plane there is nothing recorded to occlude
    // against, and the attenuation has already taken the light to zero there.
    if (ndc.z > 1.0) {
        return 1.0;
    }

    vec2 tile_uv = ndc.xy * 0.5 + 0.5;
    vec2 texel = 1.0 / atlas_size;
    // Half a texel in from the tile's edge, so a bilinear fetch cannot reach
    // past it either.
    vec2 lo = face.rect.xy + texel * 0.5;
    vec2 hi = face.rect.xy + face.rect.zw - texel * 0.5;

    float sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 uv = face.rect.xy + (tile_uv + vec2(x, y) * texel / face.rect.zw) * face.rect.zw;
            sum += texture(
                sampler2DShadow(u_shadow_atlas, u_shadow_cmp),
                vec3(clamp(uv, lo, hi), ndc.z)
            );
        }
    }
    return sum / 9.0;
}

// Visibility for a point light: pick the face its direction falls on, then the
// lookup above. `first_face` is negative for a light the atlas had no room for,
// which is the unshadowed path.
float point_shadow(PointLight light, vec3 world_pos, vec3 N, vec3 L, float dist) {
    int first_face = int(light.shadow.x);
    if (first_face < 0) {
        return 1.0;
    }
    // The direction *from* the light, which is what the faces were built around.
    int face = cube_face(world_pos - light.position.xyz);
    return atlas_shadow(first_face + face, world_pos, N, L, dist);
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

// Everything about this pixel that no light direction can change.
//
// Split out of `shade_surface` because the four lobes turned "sample the maps"
// into a third of the shader, and because the split names the rule the file is
// built on: what is read here is per fragment, what `brdf` does with it is per
// light. Anything that ended up on the wrong side of that line would be paid for
// once per light for no reason.
Surface read_surface(GpuMaterial m, out float alpha) {
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

    // Sample the maps. Missing maps point at the default textures, so
    // these multiplies become no-ops. Albedo/emissive images are sRGB
    // (decoded to linear on sample); metal-rough is linear data.
    vec4 albedo_tex = sample_tex(m.tex_indices.x, s.uv);
    vec4 mr_tex     = sample_tex(m.tex_indices.z, s.uv);

    // Vertex color tints the material albedo; drop `* v_color` for a
    // pure material/texture color.
    s.albedo = m.base_color.rgb * v_color * albedo_tex.rgb;
    // The albedo map's alpha multiplies the material's, the same way its rgb
    // multiplies the base colour. An opaque draw ignores the result; a blended
    // one weights everything below by it.
    alpha = clamp(m.base_color.a * albedo_tex.a, 0.0, 1.0);
    // glTF metallic-roughness convention: G = roughness, B = metallic.
    s.metallic = clamp(m.params.x * mr_tex.b, 0.0, 1.0);
    // Floor avoids a singular highlight.
    s.perceptual_roughness = clamp(m.params.y * mr_tex.g, 0.04, 1.0);
    float reflectance = m.params.z;

    // Tangent-space normal map -> world space via the TBN basis.
    vec3 n_tangent = tangent_normal(sample_tex(m.tex_indices.y, s.uv));
    s.N = normalize(TBN * n_tangent);

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
    Surface s = read_surface(m, alpha);
    // After the surface, not before: the coordinate every map is read at is what
    // `read_surface` resolves, and a glowing filament that stayed behind while
    // the brick around it moved would be the parallax failing in the one place
    // it is most visible.
    vec3 emis_tex = sample_tex(m.tex_indices.w, s.uv).rgb;
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
    vec3 diffuse_ibl = sh_irradiance(s.N) * s.albedo * (1.0 - s.metallic) * ao;
    vec3 color = scatters ? vec3(0.0) : diffuse_ibl;
    vec3 diffusible = scatters ? diffuse_ibl : vec3(0.0);
    color += specular_ibl_along(
        ibl_reflection(s), s.n_dot_v, s.f0, s.perceptual_roughness, s.energy
    ) * ao;

    // The fibres' own environment response. Sampled through the same split-sum
    // chain as the base lobe, which is an approximation — the prefiltered mips
    // were convolved against GGX, not against Charlie — but the alternative is
    // cloth that has a rim under a lamp and none under a sky.
    if ((s.flags & MATERIAL_SHEEN) != 0u) {
        color += specular_ibl_along(
            reflect(-s.V, s.N), s.n_dot_v, s.sheen_color, s.sheen_a, vec3(1.0)
        ) * ao;
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
        ) * s.coat * ao;
    }

    // Directional sun, shadowed by the cascades and the screen-space march.
    // The environment term above is what `u_ao` attenuates instead; the
    // punctual lights below carry their own atlas lookups.
    float view_dist = length(v_world_pos - lighting.camera_pos.xyz);
    {
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

    // Point lights.
    int count = int(lighting.params.x);
    for (int i = 0; i < count; ++i) {
        PointLight light = lighting.point_lights[i];
        vec3 to_light = light.position.xyz - v_world_pos;
        float dist = length(to_light);
        float atten = attenuate(dist, light.position.w);
        if (atten <= 0.0) continue;
        vec3 L = to_light / max(dist, 1e-4);
        // Before the atlas lookup, not after: a surface facing away is already
        // unlit by this light, and the nine taps would be multiplied into a
        // zero. The same early-out `sun_shadow` makes for the cascades.
        //
        // Except on a scattering surface, where facing away from the light is not
        // the same as being unlit by it — that is the one case the light reaches
        // the camera by going *through*.
        if (!scatters && dot(s.N, L) <= 0.0) continue;
        vec3 radiance = light.color.rgb * light.color.w * atten;
        float shadow = point_shadow(light, v_world_pos, s.N, L, dist);
        float back_shadow = scatters
            ? point_shadow(light, v_world_pos - s.N * s.thickness, -s.N, L, dist)
            : 1.0;
        Lobes lobes = brdf(s, L, radiance, shadow, back_shadow);
        color += lobes.direct;
        diffusible += lobes.diffusible;
    }

    // Spot lights: a point light with a cone over it, and one atlas face rather
    // than six because a cone is a single frustum.
    int spot_count = int(lighting.params.w);
    for (int i = 0; i < spot_count; ++i) {
        SpotLight light = lighting.spot_lights[i];
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
        float shadow = face < 0 ? 1.0 : atlas_shadow(face, v_world_pos, s.N, L, dist);
        float back_shadow = (face < 0 || !scatters)
            ? 1.0
            : atlas_shadow(face, v_world_pos - s.N * s.thickness, -s.N, L, dist);
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
    color += m.emissive.rgb * emis_tex;

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

    if (lighting.cascades.params.w > 0.5) {
        vec3 tint = cascade_debug_tint(select_cascade(lighting.cascades, view_dist));
        color *= tint;
        diffusible *= tint;
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
