// The camera's depth convention, for every shader that reads the camera's
// depth: reversed, so the near plane is 1 and the far plane — the clear, the
// sky — is 0. `gfx/depth.rs` says why. Shadow maps are not camera depth and
// stay forward-Z; nothing that samples one includes this.
//
// Comparisons go through these rather than through a bare `<` so that the
// direction is stated once: a flipped operator in one pass is a silent bug that
// reads as a lighting or occlusion artefact, never as an error.

#ifndef ORRIN_DEPTH_GLSL
#define ORRIN_DEPTH_GLSL

const float DEPTH_FAR = 0.0;
const float DEPTH_NEAR = 1.0;

// Whether nothing was rasterised here.
bool depth_is_far(float d) {
    return d <= DEPTH_FAR;
}

// Whether depth `a` is nearer the camera than depth `b`.
bool depth_nearer(float a, float b) {
    return a > b;
}

float depth_nearest(float a, float b) {
    return max(a, b);
}

float depth_farthest(float a, float b) {
    return min(a, b);
}

// Metres in front of the camera from a depth-buffer value, given the near and
// far planes it was projected with. Mirrored by
// `linear_depth_inverts_the_projection` in `scene/camera.rs`.
float depth_linear(float d, float near, float far) {
    return near * far / max(near + d * (far - near), 1e-6);
}

#endif
