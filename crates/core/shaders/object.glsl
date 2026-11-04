// One row of the per-object buffer, mirroring `GpuObject` in
// `vulkan/instances.rs`: each matrix as its top three rows, because a world
// transform is affine and the fourth row is always (0, 0, 0, 1).
//
// Both matrices are declared by every consumer, though only the prepass reads
// `prev_model`: the stride is the row's, and a struct that dropped one would
// read every row after the first out of its neighbour.
struct Object {
    vec4 model[3];
    vec4 prev_model[3];
};

// The matrix three stored rows describe. Mirrored by `shader_affine` in
// `vulkan/instances.rs`, whose test is what holds the two to bit-exactness —
// the prepass and the forward pass depth-test `EQUAL` against each other
// through this function.
mat4 affine(vec4 r0, vec4 r1, vec4 r2) {
    return mat4(
        vec4(r0.x, r1.x, r2.x, 0.0),
        vec4(r0.y, r1.y, r2.y, 0.0),
        vec4(r0.z, r1.z, r2.z, 0.0),
        vec4(r0.w, r1.w, r2.w, 1.0));
}

mat4 object_model(Object o) {
    return affine(o.model[0], o.model[1], o.model[2]);
}

mat4 object_prev_model(Object o) {
    return affine(o.prev_model[0], o.prev_model[1], o.prev_model[2]);
}
