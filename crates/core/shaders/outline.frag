#version 460

// The outline: a band two pixels wide just outside every shape in the mask,
// selected over hovered. Outside rather than across the edge, so the outline
// never paints over the object it names.

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D u_mask;

layout(push_constant) uniform Push {
    vec4 selected;
    vec4 hovered;
    int output_in_linear_colorspace;
} push;

// The overlay's curve, quirk included (see `overlay.frag`), so the outline is
// the colour the editor's own widgets would show it as.
vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 lower = srgb / vec3(12.92);
    vec3 higher = pow((srgb + vec3(0.055) / vec3(1.055)), vec3(2.4));
    return mix(higher, lower, vec3(cutoff));
}

void main() {
    ivec2 size = textureSize(u_mask, 0);
    ivec2 here = ivec2(gl_FragCoord.xy);
    vec2 inside = texelFetch(u_mask, here, 0).rg;
    vec2 near = vec2(0.0);
    // Two rings of eight taps, at one and two pixels.
    for (int ring = 1; ring <= 2; ++ring) {
        for (int i = 0; i < 8; ++i) {
            float angle = float(i) * 0.78539816;
            ivec2 offset = ivec2(round(vec2(cos(angle), sin(angle)) * float(ring)));
            ivec2 at = clamp(here + offset, ivec2(0), size - 1);
            near = max(near, texelFetch(u_mask, at, 0).rg);
        }
    }
    float selected = near.r * (1.0 - inside.r);
    float hovered = near.g * (1.0 - inside.g) * (1.0 - selected);
    float coverage = max(selected, hovered);
    if (coverage <= 0.0) {
        discard;
    }
    vec4 colour = selected > 0.0 ? push.selected : push.hovered;
    vec3 rgb = push.output_in_linear_colorspace == 1 ? linear_from_srgb(colour.rgb) : colour.rgb;
    f_color = vec4(rgb, colour.a * coverage);
}
