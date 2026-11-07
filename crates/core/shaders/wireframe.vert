#version 460

// One mesh's triangles, drawn without an index buffer so each corner knows
// which corner it is: `gl_VertexIndex` counts corners, and every third one
// starts a triangle. The shader reads the index and the position itself out of
// the shared geometry, which is what lets the edges come from barycentrics on
// any device rather than from a line fill mode or a barycentric extension.

// `PositionVertex`: a position and a texture coordinate, five floats.
const uint POSITION_WORDS = 5u;

layout(set = 0, binding = 0, std430) readonly buffer Positions {
    float position_words[];
};

layout(set = 0, binding = 1, std430) readonly buffer Indices {
    uint indices[];
};

layout(push_constant) uniform Push {
    mat4 mvp;
    vec4 color;
    uint first_index;
    int vertex_offset;
    float width;
    // Relative: the depth is multiplied by `1 + nudge`, toward the camera
    // under reversed Z, so an edge wins against its own face at every
    // distance by the same fraction of it.
    float nudge;
} push;

layout(location = 0) out vec3 v_corner;

void main() {
    uint corner = uint(gl_VertexIndex) % 3u;
    uint index = indices[push.first_index + uint(gl_VertexIndex)];
    uint base = uint(int(index) + push.vertex_offset) * POSITION_WORDS;
    vec3 position = vec3(position_words[base], position_words[base + 1u], position_words[base + 2u]);

    v_corner = vec3(corner == 0u, corner == 1u, corner == 2u);
    gl_Position = push.mvp * vec4(position, 1.0);
    gl_Position.z *= 1.0 + push.nudge;
}
