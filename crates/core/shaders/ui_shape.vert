#version 460

// The in-game UI's styled rectangles: one instance per `gfx::ui::UiShape`, its
// six corners from the vertex index. Everything but the position is constant
// across the instance, so it is passed flat.

layout(location = 0) in vec4 rect;
layout(location = 1) in vec4 uv;
layout(location = 2) in vec4 shape;
layout(location = 3) in vec4 params;
layout(location = 4) in vec4 gradient;
layout(location = 5) in vec4 color0;
layout(location = 6) in vec4 color1;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec2 v_position;
layout(location = 2) flat out vec4 v_shape;
layout(location = 3) flat out vec4 v_params;
layout(location = 4) flat out vec4 v_gradient;
layout(location = 5) flat out vec4 v_color0;
layout(location = 6) flat out vec4 v_color1;

layout(push_constant) uniform Push {
    vec2 screen_size;
    uint output_in_linear_colorspace;
    uint texture_index;
} push;

const vec2 CORNERS[6] = vec2[](
    vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
    vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0)
);

void main() {
    vec2 corner = CORNERS[gl_VertexIndex];
    v_position = mix(rect.xy, rect.zw, corner);
    gl_Position = vec4(2.0 * v_position / push.screen_size - 1.0, 0.0, 1.0);
    v_uv = mix(uv.xy, uv.zw, corner);
    v_shape = shape;
    v_params = params;
    v_gradient = gradient;
    v_color0 = color0;
    v_color1 = color1;
}
