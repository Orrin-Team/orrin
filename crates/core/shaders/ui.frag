#version 460

// The in-game UI, in linear light, written with straight alpha over the
// tonemapped frame. A draw samples either the glyph atlas — coverage in red,
// addressed in texels so the atlas can grow under laid-out text — or one image
// of the material table, chosen by a push constant and therefore dynamically
// uniform.

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;

layout(location = 0) out vec4 f_color;

// Keep in sync with MAX_TEXTURES in gfx/mod.rs.
const int MAX_TEXTURES = 192;
layout(set = 0, binding = 0) uniform texture2D u_textures[MAX_TEXTURES];
layout(set = 0, binding = 1) uniform sampler u_sampler;
layout(set = 1, binding = 0) uniform sampler2D u_atlas;

layout(push_constant) uniform Push {
    vec2 screen_size;
    uint output_in_linear_colorspace;
    uint texture_index;
} push;

// `gfx::vulkan::ui::ATLAS`.
const uint ATLAS = 0xffffffffu;

vec3 srgb_from_linear(vec3 linear) {
    bvec3 cutoff = lessThan(linear, vec3(0.0031308));
    vec3 lower = linear * vec3(12.92);
    vec3 higher = vec3(1.055) * pow(linear, vec3(1.0 / 2.4)) - vec3(0.055);
    return mix(higher, lower, vec3(cutoff));
}

void main() {
    vec4 color = v_color;
    if (push.texture_index == ATLAS) {
        color.a *= texture(u_atlas, v_uv / vec2(textureSize(u_atlas, 0))).r;
    } else {
        color *= texture(sampler2D(u_textures[push.texture_index], u_sampler), v_uv);
    }
    if (push.output_in_linear_colorspace == 0u) {
        color.rgb = srgb_from_linear(color.rgb);
    }
    f_color = color;
}
