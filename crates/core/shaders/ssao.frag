#version 460

layout(location = 0) in vec2 v_uv;
layout(location = 0) out float f_ao;

const int KERNEL_MAX = 64;

layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 proj;
    mat4 inv_proj;
} frame;

layout(set = 0, binding = 1) uniform Params {
    vec4  kernel[KERNEL_MAX]; // xyz used; vec4 keeps std140 stride at 16
    vec2  noise_scale;        // screen_size / 4.0  (noise is 4x4)
    float radius;             // world units, e.g. 0.5
    float bias;               // e.g. 0.025
    float power;              // contrast on the result, e.g. 1.5
    int   kernel_size;        // <= KERNEL_MAX, e.g. 32
} p;

layout(set = 1, binding = 0) uniform sampler2D u_depth;  // D32,   nearest + clamp
layout(set = 1, binding = 1) uniform sampler2D u_normal; // RGBA8, nearest + clamp
layout(set = 1, binding = 2) uniform sampler2D u_noise;  // RGBA8, nearest + repeat

vec3 view_pos(vec2 uv) {
    float d = texture(u_depth, uv).r;
    vec4 c = frame.inv_proj * vec4(uv * 2.0 - 1.0, d, 1.0);
    return c.xyz / c.w;
}

void main() {
    vec3 P = view_pos(v_uv);
    vec3 N = normalize(texture(u_normal, v_uv).xyz * 2.0 - 1.0);
    vec3 rnd = vec3(texture(u_noise, v_uv * p.noise_scale).xy * 2.0 - 1.0, 0.0);

    vec3 T = normalize(rnd - N * dot(rnd, N)); // Gram-Schmidt around N
    vec3 B = cross(N, T);
    mat3 TBN = mat3(T, B, N);

    float occ = 0.0;
    for (int i = 0; i < p.kernel_size; ++i) {
        vec3 sp = P + TBN * p.kernel[i].xyz * p.radius;      // sample point, view space
        vec4 clip = frame.proj * vec4(sp, 1.0);
        vec2 suv = (clip.xy / clip.w) * 0.5 + 0.5;           // project to screen UV
        float scene_z = view_pos(suv).z;                     // real geometry z at suv
        float range = smoothstep(0.0, 1.0, p.radius / max(abs(P.z - scene_z), 1e-4));
        occ += (scene_z >= sp.z + p.bias ? 1.0 : 0.0) * range;
    }
    f_ao = pow(1.0 - occ / float(p.kernel_size), p.power);
}
