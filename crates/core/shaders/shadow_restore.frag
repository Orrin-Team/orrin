#version 460
// A read by texel, which takes no sampler.
#extension GL_EXT_samplerless_texture_functions : require

// Put a rectangle of the live shadow atlas back to what the cached layer holds.
//
// The two atlases are laid out alike, tile for tile, so a view's cached depth
// is at the fragment's own coordinate -- except in a sun level, whose cached
// tile keeps each cell of the level's grid at that cell's place modulo the
// tile, so that a window which has followed the camera finds the cells it
// still covers where they were drawn. `v_wrap` is how far round the tile the
// window's first texel is kept, and zero for every other view.

layout(location = 0) flat in ivec4 v_tile;
layout(location = 1) flat in ivec2 v_wrap;

layout(set = 0, binding = 1) uniform texture2D u_cached;

void main() {
    ivec2 at = ivec2(gl_FragCoord.xy) - v_tile.xy;
    ivec2 kept = v_tile.xy + (at + v_wrap) % v_tile.z;
    gl_FragDepth = texelFetch(u_cached, kept, 0).r;
}
