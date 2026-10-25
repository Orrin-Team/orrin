// What a normal is transformed by: the inverse-transpose of `model`'s linear
// part, up to a positive scale the caller's `normalize` removes.
//
// The cofactor matrix is the inverse-transpose times the determinant, so it
// needs no inverse at all — three cross products and a dot, about fifteen
// flops a vertex, against a general 3x3 inverse per object per frame on the CPU
// and 64 bytes per row to carry it. It is general, not `R * S^-1`: a
// non-uniformly scaled parent with a rotated child produces shear, and the
// closed form assumes that away.
//
// The sign is what keeps a mirrored object's normals facing out, since there
// the determinant is negative. A flattened object's is zero, and so is the
// matrix, which is the answer the CPU version gave.
//
// Mirrored by `shader_normal_matrix` in `vulkan/instances.rs`, whose tests are
// the only check this has without a GPU.
mat3 normal_matrix(mat4 model) {
    mat3 m = mat3(model);
    mat3 cofactor = mat3(cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1]));
    return cofactor * sign(dot(m[0], cross(m[1], m[2])));
}
