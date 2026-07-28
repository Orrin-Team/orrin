#version 460

layout(location = 0) in vec2 v_uv;
layout(location = 0) out float f_ao;

layout(set = 0, binding = 0) uniform sampler2D u_ao;

void main() {
    vec2 texel = 1.0 / vec2(textureSize(u_ao, 0));
    float s = 0.0;
    for (int x = -2; x < 2; ++x)
        for (int y = -2; y < 2; ++y)
            s += texture(u_ao, v_uv + vec2(x, y) * texel).r;
    f_ao = s / 16.0;
}
