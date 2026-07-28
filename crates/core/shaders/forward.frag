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
    PointLight point_lights[MAX_POINT_LIGHTS];
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

// Index is dynamically uniform (from the material), so plain indexing
// is legal without the nonuniform qualifier.
vec4 sample_tex(uint index, vec2 uv) {
    return texture(sampler2D(textures[index], tex_sampler), uv);
}

// Declared identically to the vertex shader so the stages share one
// push-constant range; only material_index is read here.
layout(push_constant) uniform Push {
    mat4 mvp;
    uint material_index;
    uint object_index;
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

// Outgoing radiance toward the camera from one light direction L.
vec3 brdf(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo,
          float metallic, float roughness, vec3 f0) {
    float n_dot_l = max(dot(N, L), 0.0);
    if (n_dot_l <= 0.0) {
        return vec3(0.0);
    }
    vec3 H = normalize(L + V);
    float n_dot_v = max(dot(N, V), 1e-4);
    float n_dot_h = max(dot(N, H), 0.0);
    float v_dot_h = max(dot(V, H), 0.0);

    float a = roughness * roughness; // perceptual -> linear roughness

    float D = distribution_ggx(n_dot_h, a);
    float Vis = visibility_smith_ggx(n_dot_v, n_dot_l, a);
    vec3 F = fresnel_schlick(v_dot_h, f0);

    vec3 specular = D * Vis * F;

    // Diffuse keeps the energy not reflected (1 - F) and not metallic.
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kd * albedo / PI;

    return (diffuse + specular) * radiance * n_dot_l;
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

    vec3 V = normalize(lighting.camera_pos.xyz - v_world_pos);

    // Crude diffuse ambient (stands in for image-based lighting),
    // attenuated by screen-space ambient occlusion.
    float ao = texture(u_ao, gl_FragCoord.xy * lighting.viewport.zw).r;
    vec3 color = lighting.ambient.rgb * lighting.ambient.w * albedo * ao;

    // Directional sun.
    {
        vec3 L = normalize(lighting.sun_direction.xyz);
        vec3 radiance = lighting.sun_color.rgb * lighting.sun_color.w;
        color += brdf(N, V, L, radiance, albedo, metallic, roughness, f0);
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
        color += brdf(N, V, L, radiance, albedo, metallic, roughness, f0);
    }

    // Emissive adds on top, unaffected by scene lighting.
    color += m.emissive.rgb * emis_tex;

    f_color = vec4(color, 1.0);
}
