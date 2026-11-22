#version 460

// One rectangle of the shadow atlas per instance, on the far plane.
//
// Two jobs, one per fragment shader it is paired with. Alone, with the depth
// test set to always pass, it clears the rectangle: the far plane is what an
// empty shadow map holds. With `shadow_restore.frag` it puts the rectangle
// back to what the cached layer holds there.
//
// A draw rather than a clear or a copy because a view is a rectangle of a
// larger attachment the pass has to keep the rest of. Clearing part of an
// attachment mid-pass is not a thing every backend has, and a copy between two
// images is a pass of its own on all of them.

struct Rect {
    // The rectangle, in atlas texels: xy its least corner, zw its greatest.
    vec4 rect;
    // The tile it lies in: xy its origin, z its edge.
    ivec4 tile;
    // Where in the cached tile the live tile's first texel is kept.
    ivec4 wrap;
};

layout(set = 0, binding = 0, std430) readonly buffer Rects {
    Rect rects[];
};

layout(push_constant) uniform Push {
    // Edge of the atlas, in texels.
    float atlas_size;
} push;

#ifdef ORRIN_RESTORE
layout(location = 0) flat out ivec4 v_tile;
layout(location = 1) flat out ivec2 v_wrap;
#endif

void main() {
    Rect r = rects[gl_InstanceIndex];
    // Two triangles: (0,0) (1,0) (0,1), then (0,1) (1,0) (1,1).
    int i = gl_VertexIndex;
    vec2 corner = vec2(float(i == 1 || i >= 4), float(i == 2 || i == 3 || i == 5));
    vec2 at = mix(r.rect.xy, r.rect.zw, corner);
    gl_Position = vec4(at / push.atlas_size * 2.0 - 1.0, 1.0, 1.0);
#ifdef ORRIN_RESTORE
    v_tile = r.tile;
    v_wrap = r.wrap.xy;
#endif
}
