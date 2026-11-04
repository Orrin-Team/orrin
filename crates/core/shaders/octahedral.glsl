// A unit direction from the two SNORM16 channels the surface stream stores it
// in: the sphere folded onto the octahedron, then the octahedron's lower half
// folded out onto the corners of the square. The inverse of `oct_encode` in
// `gfx/mod.rs`, and mirrored there by `shader_oct_decode`.
vec3 oct_decode(vec2 e) {
    vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-n.z, 0.0);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}
