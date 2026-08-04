#version 460

// One roughness level of the prefiltered specular environment chain.
//
// Karis' split-sum approximation: the light integral is prefiltered here, once
// at load, and the BRDF integral is the analytic fit `env_brdf_approx` that
// forward.frag already carries for multi-scatter compensation. Which is why
// there is no BRDF lookup table anywhere in this engine.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform textureCube u_source;
layout(set = 0, binding = 1) uniform sampler u_sampler;

layout(push_constant) uniform Push {
    vec4 forward;
    vec4 right;
    vec4 up;
    vec4 params; // x = perceptual roughness, y = source face size in texels
} push;

const float PI = 3.14159265359;
const uint SAMPLE_COUNT = 128u;

// Van der Corput radical inverse in base 2, by bit reversal.
float radical_inverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

vec2 hammersley(uint i, uint n) {
    return vec2(float(i) / float(n), radical_inverse(i));
}

// A half-vector drawn from the GGX distribution around `n`.
vec3 importance_sample_ggx(vec2 xi, vec3 n, float a) {
    float phi = 2.0 * PI * xi.x;
    float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    vec3 h = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

    vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, n));
    vec3 bitangent = cross(n, tangent);
    return normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

float distribution_ggx(float n_dot_h, float a) {
    float a2 = a * a;
    float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

void main() {
    vec2 face = v_uv * 2.0 - 1.0;
    vec3 n = normalize(push.forward.xyz + face.x * push.right.xyz + face.y * push.up.xyz);

    float roughness = push.params.x;

    // Level 0 is the mirror level, and importance sampling would spend 128
    // samples rediscovering that: at a = 0 every half-vector collapses onto the
    // normal. Taking it directly also keeps this level exactly what the skybox
    // draws, so the background and a perfect reflection cannot disagree.
    if (roughness == 0.0) {
        f_color = vec4(textureLod(samplerCube(u_source, u_sampler), n, 0.0).rgb, 1.0);
        return;
    }

    // The split-sum approximation's one assumption: the view direction is the
    // normal. Dropping the view dependence is what makes a single chain usable
    // from every angle instead of one per view.
    vec3 v = n;
    float a = roughness * roughness;

    float texel_solid_angle = 4.0 * PI / (6.0 * push.params.y * push.params.y);

    vec3 color = vec3(0.0);
    float weight = 0.0;

    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
        vec3 h = importance_sample_ggx(hammersley(i, SAMPLE_COUNT), n, a);
        vec3 l = reflect(-v, h);
        float n_dot_l = dot(n, l);
        if (n_dot_l <= 0.0) {
            continue;
        }

        // Read from a blurrier level wherever the samples are sparse relative
        // to the source's resolution. Without this a sun disc lands in a
        // handful of samples and scatters into fireflies that more samples do
        // not fix — the variance is in the source, not the estimator.
        float n_dot_h = max(dot(n, h), 0.0);
        float v_dot_h = max(dot(v, h), 1e-4);
        float pdf = distribution_ggx(n_dot_h, a) * n_dot_h / (4.0 * v_dot_h) + 1e-4;
        float sample_solid_angle = 1.0 / (float(SAMPLE_COUNT) * pdf);
        float lod = 0.5 * log2(sample_solid_angle / texel_solid_angle);

        color += textureLod(samplerCube(u_source, u_sampler), l, lod).rgb * n_dot_l;
        weight += n_dot_l;
    }

    f_color = vec4(color / max(weight, 1e-4), 1.0);
}
