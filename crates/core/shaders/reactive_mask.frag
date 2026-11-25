#version 460

// How much of each pixel the blended queues put over the opaque frame, for a
// vendor's upscaler to weigh its history less there: neither queue writes
// depth or motion, so what they draw would otherwise be reprojected as though
// it moved with the surface behind it.
//
// The transparency composite keeps `reveal` of the opaque frame and the
// refraction composite `1 - coverage` of what that left, so the opaque share
// is their product. AMD's guidance is the blend's alpha, held below one: "a
// reactive value of close to 1" never resolves well.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out float f_reactive;

layout(set = 0, binding = 0) uniform sampler2D u_reveal;
// rgb = premultiplied radiance, a = coverage.
layout(set = 0, binding = 1) uniform sampler2D u_refraction;

// Which of the two the frame has. An absent one's binding holds the other
// image, unread.
layout(push_constant) uniform Push {
    uint blended;
    uint refracted;
} push;

const float MOST_REACTIVE = 0.9;

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float opaque = 1.0;
    if (push.blended != 0) {
        opaque *= texelFetch(u_reveal, pixel, 0).r;
    }
    if (push.refracted != 0) {
        opaque *= 1.0 - texelFetch(u_refraction, pixel, 0).a;
    }
    f_reactive = min(1.0 - opaque, MOST_REACTIVE);
}
