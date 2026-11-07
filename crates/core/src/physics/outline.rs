//! A collider as line segments, for the editor to draw.
//!
//! Drawn from the same dimensions the body is built from — scale folded in
//! the way `shapes` folds it, a capsule's radius taking the wider horizontal
//! scale — so the outline is the shape Jolt collides with, not a guess at it.

use glam::{Mat3, Mat4, Vec3};

use crate::scene::{ColliderShape, MeshGeometry};

/// Segments on a full circle.
const CIRCLE: usize = 24;

/// Past this many triangles a mesh is outlined by its bounds instead: every
/// edge of a large mesh is a solid smear, and a line each is real upload
/// traffic every frame.
pub const MAX_OUTLINED_TRIANGLES: usize = 2048;

/// Append `shape`'s outline under `transform` to `out`.
pub fn outline(
    shape: ColliderShape,
    transform: Mat4,
    geometry: Option<&MeshGeometry>,
    out: &mut Vec<(Vec3, Vec3)>,
) {
    let (scale, rotation, centre) = transform.to_scale_rotation_translation();
    let scale = scale.abs();
    let axes = Mat3::from_quat(rotation);
    let across = scale.x.max(scale.z);
    match shape {
        ColliderShape::Box { half_extents } => {
            cuboid(centre, axes, half_extents * scale, out);
        }
        ColliderShape::Sphere { radius } => {
            let r = radius * scale.max_element();
            for (u, v) in [(0, 1), (1, 2), (2, 0)] {
                ring(centre, axes.col(u) * r, axes.col(v) * r, 1.0, out);
            }
        }
        ColliderShape::Capsule {
            radius,
            half_height,
        } => {
            let (r, h) = (radius * across, half_height * scale.y);
            let (x, y, z) = (axes.x_axis, axes.y_axis, axes.z_axis);
            for side in [-1.0, 1.0] {
                let cap = centre + y * (h * side);
                ring(cap, x * r, z * r, 1.0, out);
                // Half a ring each way over the cap, in the two vertical planes.
                ring(cap, x * r, y * (r * side), 0.5, out);
                ring(cap, z * r, y * (r * side), 0.5, out);
            }
            sides(centre, x * r, z * r, y * h, out);
        }
        ColliderShape::Cylinder {
            radius,
            half_height,
        } => {
            let (r, h) = (radius * across, half_height * scale.y);
            let (x, y, z) = (axes.x_axis, axes.y_axis, axes.z_axis);
            for side in [-1.0, 1.0] {
                ring(centre + y * (h * side), x * r, z * r, 1.0, out);
            }
            sides(centre, x * r, z * r, y * h, out);
        }
        ColliderShape::Mesh | ColliderShape::ConvexHull => {
            let Some(geometry) = geometry else {
                return;
            };
            if geometry.triangles.len() <= MAX_OUTLINED_TRIANGLES {
                let at = |i: u32| transform.transform_point3(geometry.positions[i as usize]);
                for &[a, b, c] in &geometry.triangles {
                    let (a, b, c) = (at(a), at(b), at(c));
                    out.extend([(a, b), (b, c), (c, a)]);
                }
            } else {
                let (min, max) = geometry.positions.iter().fold(
                    (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
                    |(min, max), &p| (min.min(p), max.max(p)),
                );
                let local = Mat4::from_translation((min + max) * 0.5);
                let half = (max - min) * 0.5;
                let (s, r, t) = (transform * local).to_scale_rotation_translation();
                cuboid(t, Mat3::from_quat(r), half * s.abs(), out);
            }
        }
    }
}

fn cuboid(centre: Vec3, axes: Mat3, half: Vec3, out: &mut Vec<(Vec3, Vec3)>) {
    let corner = |i: usize| {
        let sign = |bit: usize| if i & bit != 0 { 1.0 } else { -1.0 };
        centre + axes * Vec3::new(sign(1) * half.x, sign(2) * half.y, sign(4) * half.z)
    };
    for i in 0..8 {
        for bit in [1, 2, 4] {
            if i & bit == 0 {
                out.push((corner(i), corner(i | bit)));
            }
        }
    }
}

/// `fraction` of the ellipse through `centre + u` and `centre + v`, starting at `u`.
fn ring(centre: Vec3, u: Vec3, v: Vec3, fraction: f32, out: &mut Vec<(Vec3, Vec3)>) {
    let segments = ((CIRCLE as f32 * fraction).round() as usize).max(1);
    let at = |i: usize| {
        let angle = std::f32::consts::TAU * fraction * i as f32 / segments as f32;
        centre + u * angle.cos() + v * angle.sin()
    };
    out.extend((0..segments).map(|i| (at(i), at(i + 1))));
}

/// The four lines down the sides of a round shape.
fn sides(centre: Vec3, x: Vec3, z: Vec3, half: Vec3, out: &mut Vec<(Vec3, Vec3)>) {
    for edge in [x, -x, z, -z] {
        out.push((centre + edge - half, centre + edge + half));
    }
}

#[cfg(test)]
mod tests {
    use glam::Quat;

    use super::*;
    use crate::scene::CpuMesh;

    fn lines(shape: ColliderShape, transform: Mat4) -> Vec<(Vec3, Vec3)> {
        let mut out = Vec::new();
        outline(shape, transform, None, &mut out);
        out
    }

    fn points(lines: &[(Vec3, Vec3)]) -> impl Iterator<Item = Vec3> + '_ {
        lines.iter().flat_map(|&(a, b)| [a, b])
    }

    /// Twelve edges, each between two of the eight corners the body has.
    #[test]
    fn a_box_is_its_twelve_edges_where_the_body_is() {
        let transform = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 1.0, 1.0),
            Quat::from_rotation_y(0.7),
            Vec3::new(3.0, 1.0, -2.0),
        );
        let half = Vec3::splat(0.5);
        let edges = lines(ColliderShape::Box { half_extents: half }, transform);

        assert_eq!(edges.len(), 12);
        for p in points(&edges) {
            let local = transform.inverse().transform_point3(p);
            assert!(
                local.abs().abs_diff_eq(half, 1e-4),
                "{local} is not a corner"
            );
        }
    }

    /// A sphere's radius takes the longest scale, as the body's does, so a
    /// stretched sphere's outline is the round shape that collides.
    #[test]
    fn a_sphere_outline_lies_on_the_radius_the_body_has() {
        let transform = Mat4::from_scale_rotation_translation(
            Vec3::new(1.0, 3.0, 1.0),
            Quat::IDENTITY,
            Vec3::new(0.0, 2.0, 0.0),
        );
        let edges = lines(ColliderShape::Sphere { radius: 0.5 }, transform);

        assert!(!edges.is_empty());
        for p in points(&edges) {
            let r = p.distance(Vec3::new(0.0, 2.0, 0.0));
            assert!((r - 1.5).abs() < 1e-4, "a point {r} m out");
        }
    }

    /// Its caps reach `half_height + radius` along its axis and no further.
    #[test]
    fn a_capsule_outline_reaches_its_caps() {
        let edges = lines(
            ColliderShape::Capsule {
                radius: 0.25,
                half_height: 0.5,
            },
            Mat4::IDENTITY,
        );

        let top = points(&edges).map(|p| p.y).fold(f32::MIN, f32::max);
        let wide = points(&edges).map(|p| p.x.abs()).fold(0.0, f32::max);
        assert!((top - 0.75).abs() < 1e-4, "reaches {top}");
        assert!((wide - 0.25).abs() < 1e-4, "is {wide} wide");
    }

    #[test]
    fn a_mesh_is_its_triangles_and_a_large_one_its_bounds() {
        let mesh = MeshGeometry::from_mesh(&CpuMesh::cube());
        let mut out = Vec::new();
        outline(ColliderShape::Mesh, Mat4::IDENTITY, Some(&mesh), &mut out);
        assert_eq!(out.len(), mesh.triangles.len() * 3);

        let large = MeshGeometry {
            positions: mesh.positions.clone(),
            triangles: vec![[0, 1, 2]; MAX_OUTLINED_TRIANGLES + 1],
        };
        out.clear();
        outline(
            ColliderShape::ConvexHull,
            Mat4::IDENTITY,
            Some(&large),
            &mut out,
        );
        assert_eq!(out.len(), 12, "a large mesh is outlined by its bounds");
    }
}
