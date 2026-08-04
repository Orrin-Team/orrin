#version 460

// Projects one equirectangular source into one face of a cubemap. Drawn once
// per face at load time, never per frame.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D u_equirect;

// The face's basis, pushed rather than selected from a face index: the same
// three vectors serve any pass that walks the faces, and there is no table in
// the shader to drift from the one in environment.rs.
layout(push_constant) uniform Push {
    vec4 forward;
    vec4 right;
    vec4 up;
} push;

const float PI = 3.14159265359;

void main() {
    // v_uv spans [0,1] across the face; the basis spans [-1,1].
    vec2 face = v_uv * 2.0 - 1.0;
    vec3 dir = normalize(push.forward.xyz + face.x * push.right.xyz + face.y * push.up.xyz);

    vec2 uv = vec2(
        atan(dir.z, dir.x) / (2.0 * PI) + 0.5,
        acos(clamp(dir.y, -1.0, 1.0)) / PI
    );

    // Explicit LOD, not `texture`. `u` wraps from 1 to 0 at the seam, so the
    // implicit derivative across it is enormous and hardware mip selection
    // picks the smallest level — a blurred vertical stripe down one meridian
    // that reads as a filtering bug rather than a derivative one.
    f_color = vec4(textureLod(u_equirect, uv, 0.0).rgb, 1.0);
}
