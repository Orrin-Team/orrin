#version 460

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D u_hdr;

// Written by the luminance_average compute pass earlier in the frame. Read
// rather than pushed so the metered value never leaves the GPU: a readback would
// put the CPU a frame or two behind the pixels it is exposing.
layout(set = 0, binding = 1) readonly buffer Exposure {
    float exposure;
    float average_luminance;
    float ev100;
} auto_exposure;

// Already in exposure-scaled units: the bloom chain applies exposure at its
// prefilter, so this composites against the exposed scene without rescaling.
// A 1x1 black texture when bloom is off, alongside a zero strength.
layout(set = 0, binding = 2) uniform sampler2D u_bloom;

layout(push_constant) uniform Push {
    // Already includes the compensation dial; used only when metering is off, in
    // which case nothing wrote the buffer above this frame.
    float manual_exposure;
    uint use_auto;
    float bloom_strength;
} push;

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
    float exposure = push.use_auto != 0 ? auto_exposure.exposure : push.manual_exposure;
    vec3 exposed = texture(u_hdr, v_uv).rgb * exposure;

    // A blend, not an addition: bloom takes light from the scene rather than
    // adding to it, so energy is conserved and raising the strength cannot blow
    // the image out -- it only moves light around. Tonemapping happens after,
    // on the combined result, so the glow rolls off with everything else.
    vec3 combined = mix(exposed, texture(u_bloom, v_uv).rgb, push.bloom_strength);
    vec3 mapped = aces(combined);
    // Swapchain is an sRGB format, so the hardware applies the sRGB
    // transfer function on store -- output linear here.
    f_color = vec4(mapped, 1.0);
}
