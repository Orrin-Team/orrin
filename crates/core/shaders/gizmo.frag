#version 460

// `v_edge.x` is the distance across the stroke from its centre line and
// `v_edge.y` its half width, both in pixels. The stroke's quad reaches a pixel
// past that width, and this fades over that pixel, which is the whole of the
// antialiasing: the swapchain has one sample.

layout(location = 0) in vec2 v_edge;
layout(location = 1) in vec4 v_color;

layout(location = 0) out vec4 f_color;

void main() {
    float coverage = clamp(v_edge.y + 0.5 - abs(v_edge.x), 0.0, 1.0);
    f_color = vec4(v_color.rgb, v_color.a * coverage);
}
