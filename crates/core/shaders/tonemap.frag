#version 460

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D u_hdr;

layout(push_constant) uniform Push { float exposure; } push;

// ACES filmic tonemap approximation (Narkowicz 2015), operating on
// linear radiance.
vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(u_hdr, v_uv).rgb;
    vec3 mapped = aces(hdr * push.exposure);
    // Swapchain is an sRGB format, so the hardware applies the sRGB
    // transfer function on store -- output linear here.
    f_color = vec4(mapped, 1.0);
}
