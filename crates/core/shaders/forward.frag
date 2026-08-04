#version 460

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec3 v_tangent;
layout(location = 3) in vec3 v_bitangent;
layout(location = 4) in vec2 v_uv;
layout(location = 5) in vec3 v_color;

layout(location = 0) out vec4 f_color;

// Keep in sync with MAX_POINT_LIGHTS / MAX_TEXTURES in forward.rs.
const int MAX_POINT_LIGHTS = 16;
const int MAX_TEXTURES = 64;
// Keep in sync with MAX_CASCADES in gfx/shadows.rs.
const int MAX_CASCADES = 4;
const float PI = 3.14159265359;

struct PointLight {
    vec4 position; // xyz = world position, w = range
    vec4 color;    // rgb = color,         w = intensity
};

layout(set = 0, binding = 0) uniform Lighting {
    vec4 camera_pos;    // xyz = camera world position
    vec4 ambient;       // rgb = color, w = intensity
    vec4 sun_direction; // xyz = direction toward the sun (normalized)
    vec4 sun_color;     // rgb = color, w = intensity
    vec4 params;        // x = point light count (y,z legacy, unused by PBR)
    vec4 viewport;      // x=w, y=h, z=1/w, w=1/h
    vec4 fog_color;     // rgb = color, w = density at the reference height
    vec4 fog_params;    // x = height falloff, y = reference height
    mat4 cascade_view_proj[MAX_CASCADES];
    vec4 cascade_splits;      // per-cascade far distance, radial from the camera
    vec4 cascade_texel_sizes; // world size of one shadow texel, per cascade
    vec4 shadow_params;       // x = count, y = blend overlap, z = strength, w = debug
    PointLight point_lights[MAX_POINT_LIGHTS];
    vec4 environment;    // x = sin(env yaw), y = cos(env yaw)
    vec4 irradiance[9];  // rgb = SH coefficient; see gfx/sh.rs
} lighting;

// Mirrors GpuMaterial in forward.rs. std430 packs this exactly like
// the Rust #[repr(C)] struct because every field is 16 bytes.
struct GpuMaterial {
    vec4 base_color;   // rgb = albedo
    vec4 emissive;     // rgb = emissive
    vec4 params;       // x = metallic, y = roughness, z = reflectance
    uvec4 tex_indices; // x=albedo, y=normal, z=metal-rough, w=emissive
};

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

// Index is dynamically uniform (from the material), so plain indexing
// is legal without the nonuniform qualifier.
vec4 sample_tex(uint index, vec2 uv) {
    return texture(sampler2D(textures[index], tex_sampler), uv);
}

// Declared identically to the vertex shader so the stages share one
// push-constant range; only material_index is read here.
layout(push_constant) uniform Push {
    mat4 view_proj;
    uint material_index;
    uint object_base;
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

// Split-sum environment BRDF (scale, bias), Karis' analytic fit from the 2014
// mobile PBR notes. A stand-in for the DFG lookup table image-based lighting
// will bring; swap both callers to the table when it exists.
vec2 env_brdf_approx(float perceptual_roughness, float n_dot_v) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = perceptual_roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}

// Multiple-scattering compensation (Kulla & Conty). Single-scatter GGX drops
// the energy that would have bounced between microfacets, so rough metals go
// dark and desaturated; scaling the specular lobe by 1 + f0*(1/Ess - 1) puts it
// back, where Ess is the single-scatter directional albedo.
vec3 energy_compensation(vec3 f0, float perceptual_roughness, float n_dot_v) {
    vec2 dfg = env_brdf_approx(perceptual_roughness, n_dot_v);
    // Ess is the split-sum result for a fully reflective surface (f0 = 1). It
    // tends to 1 as roughness falls, so this stays near 1 for smooth materials.
    float ess = max(dfg.x + dfg.y, 1e-3);
    return 1.0 + f0 * (1.0 / ess - 1.0);
}

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
vec3 sh_irradiance(vec3 n) {
    // The same yaw the skybox samples through, so what is drawn behind the
    // scene and what lights it cannot disagree.
    float s = lighting.environment.x;
    float c = lighting.environment.y;
    n = vec3(c * n.x - s * n.z, n.y, s * n.x + c * n.z);

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

// Outgoing radiance toward the camera from one light direction L.
// Takes linear roughness `a` rather than perceptual: the caller filters it once
// for specular antialiasing, and this runs once per light.
vec3 brdf(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo,
          float metallic, float a, vec3 f0, vec3 energy) {
    float n_dot_l = max(dot(N, L), 0.0);
    if (n_dot_l <= 0.0) {
        return vec3(0.0);
    }
    vec3 H = normalize(L + V);
    float n_dot_v = max(dot(N, V), 1e-4);
    float n_dot_h = max(dot(N, H), 0.0);
    float v_dot_h = max(dot(V, H), 0.0);

    float D = distribution_ggx(n_dot_h, a);
    float Vis = visibility_smith_ggx(n_dot_v, n_dot_l, a);
    vec3 F = fresnel_schlick(v_dot_h, f0);

    vec3 specular = D * Vis * F * energy;

    // Diffuse keeps the energy not reflected (1 - F) and not metallic.
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kd * albedo / PI;

    return (diffuse + specular) * radiance * n_dot_l;
}

// Exponential height fog. Density decays with altitude, so the amount along a
// view ray is the integral of that decay rather than a function of distance
// alone — which is what keeps a ray climbing out of the layer from fogging as
// heavily as one running through it.
vec3 apply_fog(vec3 color, vec3 world_pos, vec3 camera_pos) {
    float density = lighting.fog_color.w;
    if (density <= 0.0) {
        return color;
    }

    float falloff = lighting.fog_params.x;
    vec3 ray = world_pos - camera_pos;
    float dist = length(ray);
    float dir_y = ray.y / max(dist, 1e-4);

    float at_camera = exp(-falloff * (camera_pos.y - lighting.fog_params.y));

    // The quotient has a removable singularity for rays with no vertical
    // component, where the integral is just the ray length.
    float t = falloff * dir_y * dist;
    float integral = abs(t) > 1e-4 ? (1.0 - exp(-t)) / (falloff * dir_y) : dist;

    float amount = clamp(1.0 - exp(-density * at_camera * integral), 0.0, 1.0);
    return mix(color, lighting.fog_color.rgb, amount);
}

// --- Cascaded shadow maps ---

// The cascade this fragment is routed to: the first whose far distance it is
// nearer than. Distance is radial rather than view-space depth, which costs a
// little over-coverage at the frustum corners and buys rotation invariance —
// the same property the sphere fit on the CPU side is built around.
int select_cascade(float view_dist) {
    int count = int(lighting.shadow_params.x);
    for (int i = 0; i < count; ++i) {
        if (view_dist < lighting.cascade_splits[i]) {
            return i;
        }
    }
    return count - 1;
}

// Percentage-closer filtered visibility from one cascade. 1.0 is lit.
float cascade_shadow(int cascade, vec3 world_pos, vec3 N, vec3 L) {
    // Normal-offset bias: move the lookup along the surface normal by about a
    // texel's worth of world space, more at grazing angles where a texel covers
    // the most depth. Offsetting in texture space rather than in depth is what
    // removes acne without the peter-panning a depth offset causes.
    float texel = lighting.cascade_texel_sizes[cascade];
    float slope = 1.0 - max(dot(N, L), 0.0);
    vec3 p = world_pos + N * texel * 1.4142136 * (1.0 + slope);

    vec4 clip = lighting.cascade_view_proj[cascade] * vec4(p, 1.0);
    vec3 ndc = clip.xyz / clip.w;
    // Past the cascade's far plane there is nothing to occlude against.
    if (ndc.z > 1.0) {
        return 1.0;
    }

    vec2 uv = ndc.xy * 0.5 + 0.5;
    vec2 step = 1.0 / vec2(textureSize(sampler2DArrayShadow(u_shadow_maps, u_shadow_cmp), 0).xy);

    // 3x3 taps. Each is itself a hardware 2x2 comparison — the compare happens
    // before the bilinear filter — so this is effectively a 4x4 kernel.
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 offset = vec2(x, y) * step;
            sum += texture(
                sampler2DArrayShadow(u_shadow_maps, u_shadow_cmp),
                vec4(uv + offset, float(cascade), ndc.z)
            );
        }
    }
    return sum / 9.0;
}

// Sun visibility, blended across the cascade seam.
float sun_shadow(vec3 world_pos, vec3 N, vec3 L, float view_dist) {
    int count = int(lighting.shadow_params.x);
    if (count <= 0) {
        return 1.0;
    }

    int cascade = select_cascade(view_dist);
    float shadow = cascade_shadow(cascade, world_pos, N, L);

    // The next cascade only has depth slightly before its own near plane, and
    // that overlap is what `cascades()` widened each slice by — so the blend
    // band has to be the same fraction or it fades into a region with no data.
    float split = lighting.cascade_splits[cascade];
    float band = split * lighting.shadow_params.y;
    if (cascade + 1 < count && view_dist > split - band) {
        float t = clamp((view_dist - (split - band)) / max(band, 1e-4), 0.0, 1.0);
        shadow = mix(shadow, cascade_shadow(cascade + 1, world_pos, N, L), t);
    }

    return mix(1.0, shadow, lighting.shadow_params.z);
}

// Distinct tint per cascade, for checking that the splits land where intended.
vec3 cascade_debug_tint(int cascade) {
    if (cascade == 0) return vec3(1.0, 0.4, 0.4);
    if (cascade == 1) return vec3(0.4, 1.0, 0.4);
    if (cascade == 2) return vec3(0.4, 0.6, 1.0);
    return vec3(1.0, 1.0, 0.4);
}

// Smooth, range-limited falloff (windowed inverse-square).
float attenuate(float dist, float range) {
    float s = dist / max(range, 1e-4);
    if (s >= 1.0) return 0.0;
    float window = 1.0 - s * s;
    return (window * window) / max(dist * dist, 1e-4);
}

void main() {
    GpuMaterial m = materials[push.material_index];

    // Sample the maps. Missing maps point at the default textures, so
    // these multiplies become no-ops. Albedo/emissive images are sRGB
    // (decoded to linear on sample); metal-rough is linear data.
    vec3 albedo_tex = sample_tex(m.tex_indices.x, v_uv).rgb;
    vec4 mr_tex     = sample_tex(m.tex_indices.z, v_uv);
    vec3 emis_tex   = sample_tex(m.tex_indices.w, v_uv).rgb;

    // Vertex color tints the material albedo; drop `* v_color` for a
    // pure material/texture color.
    vec3  albedo      = m.base_color.rgb * v_color * albedo_tex;
    // glTF metallic-roughness convention: G = roughness, B = metallic.
    float metallic    = clamp(m.params.x * mr_tex.b, 0.0, 1.0);
    float roughness   = clamp(m.params.y * mr_tex.g, 0.04, 1.0); // floor avoids a singular highlight
    float reflectance = m.params.z;

    // Dielectric F0 from reflectance (0.5 -> ~4%); metals use albedo as F0.
    vec3 f0 = mix(vec3(0.16 * reflectance * reflectance), albedo, metallic);

    // Tangent-space normal map -> world space via the TBN basis.
    vec3 n_tangent = sample_tex(m.tex_indices.y, v_uv).xyz * 2.0 - 1.0;
    mat3 TBN = mat3(normalize(v_tangent), normalize(v_bitangent), normalize(v_normal));
    vec3 N = normalize(TBN * n_tangent);

    float a = filter_roughness(roughness * roughness, N); // perceptual -> linear

    vec3 V = normalize(lighting.camera_pos.xyz - v_world_pos);

    // Both are constant across lights, so they are computed once here rather
    // than once per light inside brdf().
    vec3 energy = energy_compensation(f0, roughness, max(dot(N, V), 1e-4));

    // Diffuse image-based lighting, attenuated by screen-space ambient
    // occlusion. The (1 - metallic) is the same factor `brdf` applies to its
    // own diffuse lobe: a metal has no diffuse response, and until the
    // prefiltered specular chain lands it has no environment response at all.
    float ao = texture(u_ao, gl_FragCoord.xy * lighting.viewport.zw).r;
    vec3 color = sh_irradiance(N) * albedo * (1.0 - metallic) * ao;

    // Directional sun. Only this term is shadowed: the environment is what
    // `u_ao` attenuates, and point lights cast nothing yet.
    float view_dist = length(v_world_pos - lighting.camera_pos.xyz);
    {
        vec3 L = normalize(lighting.sun_direction.xyz);
        vec3 radiance = lighting.sun_color.rgb * lighting.sun_color.w;
        float shadow = sun_shadow(v_world_pos, N, L, view_dist);
        color += brdf(N, V, L, radiance, albedo, metallic, a, f0, energy) * shadow;
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
        vec3 radiance = light.color.rgb * light.color.w * atten;
        color += brdf(N, V, L, radiance, albedo, metallic, a, f0, energy);
    }

    // Emissive adds on top, unaffected by scene lighting.
    color += m.emissive.rgb * emis_tex;

    color = apply_fog(color, v_world_pos, lighting.camera_pos.xyz);

    if (lighting.shadow_params.w > 0.5) {
        color *= cascade_debug_tint(select_cascade(view_dist));
    }

    f_color = vec4(color, 1.0);
}
