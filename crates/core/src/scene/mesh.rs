use std::sync::Arc;

use glam::Vec3;

use crate::geom::Aabb;
use crate::gfx::Vertex;
use crate::scene::MeshHandle;

#[derive(Clone, Default)]
pub struct CpuMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// Object-space bounds of every uploaded mesh, indexed by [`MeshHandle`], and
/// the triangles physics builds a mesh collider from.
///
/// The world-side mirror of what the backend derived at upload: culling runs in
/// extraction, which has a `World` and no renderer. Uploads only ever append, so
/// a handle's slot never changes meaning — a scene load resolves names against
/// the meshes already uploaded rather than re-uploading them.
///
/// The triangles live here rather than in a table of their own because every
/// upload already registers its bounds here, so a mesh cannot reach the world
/// without them. They cost every mesh a CPU copy of its positions and indices,
/// 24 bytes a triangle for a closed mesh, whether or not anything collides with
/// it; the renderer keeps no copy to read back from.
#[derive(Default)]
pub struct MeshBounds {
    bounds: Vec<Aabb>,
    geometry: Vec<Option<Arc<MeshGeometry>>>,
}

/// A mesh as a collider sees it: object-space positions and their triangles.
/// Shared by reference, because a collider's shape cache is keyed by which
/// copy it was built from — an asset reimport replaces it, and every body built
/// from the old one is rebuilt.
#[derive(Debug, PartialEq)]
pub struct MeshGeometry {
    pub positions: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
}

impl MeshGeometry {
    pub fn from_mesh(mesh: &CpuMesh) -> Self {
        Self {
            positions: mesh
                .vertices
                .iter()
                .map(|v| Vec3::from(v.position))
                .collect(),
            triangles: mesh.indices.as_chunks::<3>().0.to_vec(),
        }
    }
}

impl MeshBounds {
    pub fn insert(&mut self, handle: MeshHandle, bounds: Aabb) {
        let index = handle.0 as usize;
        if self.bounds.len() <= index {
            // A gap means a mesh was uploaded without registering here; EMPTY is
            // invalid, and an invalid box is never culled.
            self.bounds.resize(index + 1, Aabb::EMPTY);
        }
        self.bounds[index] = bounds;
    }

    /// `None` for a handle that was never registered — the caller must draw it
    /// rather than cull what it cannot measure.
    pub fn get(&self, handle: MeshHandle) -> Option<Aabb> {
        self.bounds.get(handle.0 as usize).copied()
    }

    pub fn insert_geometry(&mut self, handle: MeshHandle, mesh: &CpuMesh) {
        let index = handle.0 as usize;
        if self.geometry.len() <= index {
            self.geometry.resize(index + 1, None);
        }
        self.geometry[index] = Some(Arc::new(MeshGeometry::from_mesh(mesh)));
    }

    pub fn geometry(&self, handle: MeshHandle) -> Option<&Arc<MeshGeometry>> {
        self.geometry.get(handle.0 as usize)?.as_ref()
    }
}

impl CpuMesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    /// Object-space bounds over every vertex, whether or not an index references
    /// it. [`Aabb::EMPTY`] for a mesh with no vertices.
    pub fn bounds(&self) -> Aabb {
        Aabb::from_points(self.vertices.iter().map(|v| Vec3::from(v.position)))
    }

    /// The same mesh with its vertex colour set to white.
    ///
    /// Every primitive here bakes `normal * 0.5 + 0.5` into vertex colour, and
    /// `read_surface` multiplies that into albedo — so an untextured cube is
    /// lavender on one face and mint on the next. In the feature rig that is a
    /// free readout of which way a face points; anywhere a material is meant to
    /// be the colour it was authored as it is unusable, and it cannot be undone
    /// from the material, because six faces carry six different tints and a
    /// material has one `base_color`.
    ///
    /// A second mesh rather than dropping the colours from the primitives: the
    /// rig's captures are pixel comparisons, and every one of them would move.
    pub fn uncolored(mut self) -> Self {
        for vertex in &mut self.vertices {
            vertex.color = [1.0; 3];
        }
        self
    }

    pub fn cube() -> Self {
        // Per-face vertices so each face has a flat normal.
        const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
            ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), // +Z
            ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), // -Z
            ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]), // +X
            ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]), // -X
            ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]), // +Y
            ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]), // -Y
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
