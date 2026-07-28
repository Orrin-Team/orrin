use crate::gfx::Vertex;

#[derive(Clone, Default)]
pub struct CpuMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl CpuMesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    pub fn cube() -> Self {
        // Per-face vertices so each face has a flat normal.
        const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
            ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),   // +Z
            ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), // -Z
            ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),  // +X
            ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),  // -X
            ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),  // +Y
            ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),  // -Y
        ];

        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);

        for (normal, u, v) in FACES {
            let n = glam::Vec3::from(normal);
            let u = glam::Vec3::from(u);
            let v = glam::Vec3::from(v);
            let base = vertices.len() as u32;
            let color = (n * 0.5 + 0.5).to_array();

            // The tangent follows the +U texture axis (= u). The bitangent the
            // shader rebuilds is `cross(N, T) * w`; pick w so that lands on +v.
            let w = if n.cross(u).dot(v) >= 0.0 { 1.0 } else { -1.0 };
            let tangent = [u.x, u.y, u.z, w];

            for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let pos = (n + u * su + v * sv) * 0.5;
                vertices.push(Vertex {
                    position: pos.to_array(),
                    normal,
                    color,
                    // Map the [-1,1] quad corners to [0,1] UVs along u and v.
                    uv: [su * 0.5 + 0.5, sv * 0.5 + 0.5],
                    tangent,
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        Self { vertices, indices }
    }

    /// A unit quad in the XZ plane facing up (`+Y`), centered on the origin and
    /// spanning `[-0.5, 0.5]` on each axis. Same per-vertex layout as the cube's
    /// top face (matching normal/tangent/UV/winding), just centered at `y = 0`.
    pub fn plane() -> Self {
        let n = glam::Vec3::Y;
        let u = glam::Vec3::X;
        let v = glam::Vec3::NEG_Z;
        let color = (n * 0.5 + 0.5).to_array();
        // Bitangent the shader rebuilds is `cross(N, T) * w`; pick w to land on +v.
        let w = if n.cross(u).dot(v) >= 0.0 { 1.0 } else { -1.0 };
        let tangent = [u.x, u.y, u.z, w];

        let mut vertices = Vec::with_capacity(4);
        for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let pos = (u * su + v * sv) * 0.5;
            vertices.push(Vertex {
                position: pos.to_array(),
                normal: n.to_array(),
                color,
                uv: [su * 0.5 + 0.5, sv * 0.5 + 0.5],
                tangent,
            });
        }
        let indices = vec![0, 1, 2, 0, 2, 3];

        Self { vertices, indices }
    }

    /// A UV sphere of radius `0.5` (unit diameter, like [`cube`](Self::cube)),
    /// with `sectors` divisions around the `+Y` axis and `stacks` from pole to
    /// pole. Normals point outward; tangents follow the `+U` (longitude)
    /// direction so normal maps work. Faces wind counter-clockwise outward.
    pub fn sphere(sectors: u32, stacks: u32) -> Self {
        use std::f32::consts::PI;

        let sectors = sectors.max(3);
        let stacks = stacks.max(2);
        let radius = 0.5;

        let mut vertices = Vec::with_capacity(((sectors + 1) * (stacks + 1)) as usize);
        for i in 0..=stacks {
            // theta: polar angle from +Y (0) to -Y (PI); v runs top→bottom.
            let v_param = i as f32 / stacks as f32;
            let theta = v_param * PI;
            let (sin_t, cos_t) = theta.sin_cos();
            for j in 0..=sectors {
                let u_param = j as f32 / sectors as f32;
                let phi = u_param * 2.0 * PI;
                let (sin_p, cos_p) = phi.sin_cos();

                let normal = glam::Vec3::new(sin_t * cos_p, cos_t, sin_t * sin_p);
                let pos = normal * radius;
                // d(pos)/d(phi), normalized: the +U tangent. cross(N, T) then
                // lands on +V (down the sphere), so the handedness w is +1.
                let tangent = [-sin_p, 0.0, cos_p, 1.0];
                vertices.push(Vertex {
                    position: pos.to_array(),
                    normal: normal.to_array(),
                    color: (normal * 0.5 + 0.5).to_array(),
                    uv: [u_param, v_param],
                    tangent,
                });
            }
        }

        let stride = sectors + 1;
        let mut indices = Vec::with_capacity((sectors * stacks * 6) as usize);
        for i in 0..stacks {
            for j in 0..sectors {
                let k1 = i * stride + j;
                let k2 = k1 + stride;
                // Skip the degenerate triangle that collapses onto each pole.
                if i != 0 {
                    indices.extend_from_slice(&[k1, k1 + 1, k2 + 1]);
                }
                if i != stacks - 1 {
                    indices.extend_from_slice(&[k1, k2 + 1, k2]);
                }
            }
        }

        Self { vertices, indices }
    }
}
