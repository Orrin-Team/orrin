#version 460

layout(location = 0) in vec3 v_view_normal;
layout(location = 1) in vec4 v_clip;
layout(location = 2) in vec4 v_previous_clip;

layout(location = 0) out vec4 f_normal;
layout(location = 1) out vec2 f_velocity;

void main() {
    f_normal = vec4(normalize(v_view_normal) * 0.5 + 0.5, 1.0);

    if (v_previous_clip.w <= 0.0) {
        // Behind last frame's camera, so there is no history for this surface at
        // all. A vector this large lands the reprojection off-screen, which is
        // exactly the "reject the history" path the resolve already has.
        f_velocity = vec2(1e3);
        return;
    }

    // UV space rather than NDC, so the resolve can subtract it from a texture
    // coordinate directly. The projection already flips Y for Vulkan's clip
    // space, so this maps to the same orientation the render targets are in.
    vec2 now = (v_clip.xy / v_clip.w) * 0.5 + 0.5;
    vec2 before = (v_previous_clip.xy / v_previous_clip.w) * 0.5 + 0.5;
    f_velocity = now - before;
}
