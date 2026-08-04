#version 460

layout(location = 0) out vec3 v_dir;

// Must match the block in skybox.frag exactly: one range, both stages.
layout(push_constant) uniform Push {
    mat4 inv_view_rot_proj;
    vec4 params; // x = intensity
} push;

void main() {
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec2 ndc = uv * 2.0 - 1.0;

    // z = w, so after the perspective divide the triangle sits exactly on the
    // far plane. Paired with LESS_OR_EQUAL that is accepted only where nothing
    // nearer was drawn, which is what makes this a background rather than an
    // overlay. (Forward-Z: near maps to 0, far to 1 — see Camera::projection.)
    gl_Position = vec4(ndc, 1.0, 1.0);

    // The matrix is built from a view with its translation stripped, so
    // unprojecting the near plane yields the view ray directly: no camera
    // position to subtract, and no precision lost far from the origin.
    vec4 world = push.inv_view_rot_proj * vec4(ndc, 0.0, 1.0);
    v_dir = world.xyz / world.w;
}
