#version 460

// An edge is where a barycentric coordinate reaches zero. Measured in pixels
// through its own derivative, so the line is `width` pixels wide whatever the
// triangle's size or distance, and its last pixel fades rather than steps.

layout(location = 0) in vec3 v_corner;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 color;
    uint first_index;
    int vertex_offset;
    float width;
    float nudge;
    // Render pixels per output pixel, where the depth is sampled.
    vec2 depth_scale;
} push;

#ifdef ORRIN_WIREFRAME_SAMPLED_DEPTH
#include "depth.glsl"

// The scene's depth at the render extent, which a pass at the output's cannot
// attach: the test the attached depth makes is made here instead.
layout(set = 1, binding = 0) uniform sampler2D u_depth;
#endif

void main() {
#ifdef ORRIN_WIREFRAME_SAMPLED_DEPTH
    // Against the farthest of the four render texels round the pixel, so an
    // edge is hidden only where all of them cover it: at a coarser depth the
    // nearest texel of a face is often a neighbour's, and the nudge is sized
    // for a face half an output pixel away.
    vec2 uv = gl_FragCoord.xy * push.depth_scale / vec2(textureSize(u_depth, 0));
    vec4 around = textureGather(u_depth, uv, 0);
    float farthest = depth_farthest(depth_farthest(around.x, around.y), depth_farthest(around.z, around.w));
    if (depth_nearer(farthest, gl_FragCoord.z)) {
        discard;
    }
#endif
    vec3 pixels = v_corner / max(fwidth(v_corner), vec3(1e-6));
    float distance = min(min(pixels.x, pixels.y), pixels.z);
    float coverage = 1.0 - smoothstep(push.width - 1.0, push.width, distance);
    if (coverage <= 0.0) {
        discard;
    }
    out_color = vec4(push.color.rgb, push.color.a * coverage);
}
