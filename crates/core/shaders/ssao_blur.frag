#version 460

// The noise filter over the raw occlusion, and — when the term was resolved at
// half the frame's extent — the upsample that puts it back.
//
// Bilateral rather than a plain box, and that is what makes half resolution
// usable: the AO term is low-frequency everywhere except across a depth
// discontinuity, and a box filter there drags the occlusion of a near surface
// out over the far one behind it. At full resolution that is a soft halo one or
// two pixels wide; at half resolution each tap covers four pixels and the same
// halo is wide enough to read as a dark outline around every object.
//
// Weighting each tap by how close its depth is to the centre's costs one extra
// fetch per tap and removes the halo entirely: taps on the far surface simply
// stop contributing to a near pixel's result.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out float f_ao;

layout(set = 0, binding = 0) uniform sampler2D u_ao;
// Full resolution, whatever the AO's extent is: this is the geometry prepass's
// own depth, sampled by UV. At half res that means each tap reads the depth of
// the frame pixel nearest the AO texel's centre, which is exactly the surface
// that texel is standing in for.
layout(set = 0, binding = 1) uniform sampler2D u_depth;

// The same camera block `ssao.frag` reads, bound so this pass can reconstruct
// view-space depth with the identical expression rather than with coefficients
// derived a second way. Only `inv_proj` is used here; the block is declared as
// a prefix of the uploaded one, as the other consumers of it are.
layout(set = 1, binding = 0) uniform Frame {
    mat4 view;
    mat4 proj;
    mat4 inv_proj;
} frame;

// Metres in front of the camera, negated so it grows with distance.
float view_depth(vec2 uv) {
    float d = texture(u_depth, uv).r;
    vec4 c = frame.inv_proj * vec4(uv * 2.0 - 1.0, d, 1.0);
    return -c.z / c.w;
}

void main() {
    // The resolve already wrote 1.0 across the sky, so the sixteen bilateral
    // taps below reproduce it everywhere except where geometry within the
    // tolerance of the far plane leaks into a sky texel's average — which the
    // forward pass reads only through the bilinear upsample at a silhouette.
    // The same coherent branch `ssao.frag` takes, for the same reason: an empty
    // scene is otherwise this pass's worst case.
    if (texture(u_depth, v_uv).r >= 1.0) {
        f_ao = 1.0;
        return;
    }

    vec2 texel = 1.0 / vec2(textureSize(u_ao, 0));

    float center = view_depth(v_uv);
    // Proportional to distance rather than a fixed number of metres. A tolerance
    // in metres that keeps a wall's own slope together at arm's length is wide
    // enough to weld a distant object to the ground behind it, because the same
    // metre spans a fraction of a pixel out there. Five per cent of the depth is
    // far wider than the depth *gradient* across four pixels of any surface the
    // camera is not edge-on to, and far narrower than any real silhouette.
    float tolerance = max(center * 0.05, 1e-3);

    float sum = 0.0;
    float weight_sum = 0.0;
    for (int x = -2; x < 2; ++x) {
        for (int y = -2; y < 2; ++y) {
            vec2 uv = v_uv + vec2(x, y) * texel;
            // A linear falloff to zero rather than a hard threshold: a hard one
            // makes the filter's own footprint visible as a contour wherever the
            // surface slopes away, because a tap crosses the cutoff between one
            // pixel and the next.
            float weight = max(1.0 - abs(view_depth(uv) - center) / tolerance, 0.0);
            sum += texture(u_ao, uv).r * weight;
            weight_sum += weight;
        }
    }

    // The loop runs over x,y in [-2, 1], so it includes the centre tap, whose
    // weight is exactly 1. `weight_sum` therefore cannot be zero.
    f_ao = sum / weight_sum;
}
