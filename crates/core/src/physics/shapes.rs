//! A collider as a Jolt shape.
//!
//! The authored shapes are built per body with the entity's scale folded into
//! their dimensions, which Jolt asks for over a scaled wrapper. A mesh or hull
//! is expensive to build — a hull is a quickhull over every vertex, a mesh a
//! bounding-volume tree over every triangle — and typically shared by many
//! entities, so it is built once at unit scale per mesh and wrapped in a
//! scaled shape per body.

use std::sync::Arc;

use glam::Vec3;
use jolt::Shape;
use orrin_ecs::FxHashMap;

use super::{MIN_EXTENT, to_jolt};
use crate::scene::{ColliderShape, MeshGeometry};

/// Why a collider has no shape. Said once per entity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum Problem {
    /// A mesh shape on an entity with no mesh, or one whose triangles were
    /// never registered.
    NoMesh,
    Jolt(String),
}

/// Mesh and hull shapes at unit scale, by the geometry they were built from.
/// The geometry is held, not just its address, so a freed one's address cannot
/// come back as a different mesh and hit a stale entry.
#[derive(Default)]
pub(super) struct Cache {
    built: FxHashMap<(usize, bool), (Arc<MeshGeometry>, Shape)>,
}

impl Cache {
    fn unit(&mut self, geometry: &Arc<MeshGeometry>, hull: bool) -> Result<Shape, Problem> {
        let key = (Arc::as_ptr(geometry) as usize, hull);
        if let Some((_, shape)) = self.built.get(&key) {
            return Ok(shape.clone());
        }
        let points: Vec<jolt::Vec3> = geometry.positions.iter().map(|&p| to_jolt(p)).collect();
        let shape = if hull {
            Shape::convex_hull(&points, 0.05)
        } else {
            Shape::mesh(&points, &geometry.triangles)
        }
        .map_err(|error| Problem::Jolt(format!("{error:?}")))?;
        self.built.insert(key, (geometry.clone(), shape.clone()));
        Ok(shape)
    }
}

/// `shape` at `scale`, with `hull` standing in for a triangle mesh where a
/// moving body cannot have one.
pub(super) fn build(
    shape: ColliderShape,
    scale: Vec3,
    geometry: Option<&Arc<MeshGeometry>>,
    hull: bool,
    cache: &mut Cache,
) -> Result<Shape, Problem> {
    let scale_abs = scale.abs();
    let longest = scale_abs.max_element();
    // A capsule or cylinder stays round across its axis, so its radius takes
    // the larger of the two horizontal scales and its length the vertical.
    let across = scale_abs.x.max(scale_abs.z);
    let jolt = |result: jolt::Result<Shape>| result.map_err(|e| Problem::Jolt(format!("{e:?}")));
    match shape {
        ColliderShape::Box { half_extents } => jolt(Shape::cuboid(to_jolt(
            (half_extents * scale_abs).max(Vec3::splat(MIN_EXTENT)),
        ))),
        ColliderShape::Sphere { radius } => jolt(Shape::sphere((radius * longest).max(MIN_EXTENT))),
        ColliderShape::Capsule {
            radius,
            half_height,
        } => jolt(Shape::capsule(
            (half_height * scale_abs.y).max(MIN_EXTENT),
            (radius * across).max(MIN_EXTENT),
        )),
        ColliderShape::Cylinder {
            radius,
            half_height,
        } => jolt(Shape::cylinder(
            (half_height * scale_abs.y).max(MIN_EXTENT),
            (radius * across).max(MIN_EXTENT),
        )),
        ColliderShape::Mesh | ColliderShape::ConvexHull => {
            let geometry = geometry.ok_or(Problem::NoMesh)?;
            let hull = hull || shape == ColliderShape::ConvexHull;
            let unit = cache.unit(geometry, hull)?;
            if scale == Vec3::ONE {
                Ok(unit)
            } else {
                let scale = Vec3::select(scale.cmpeq(Vec3::ZERO), Vec3::splat(MIN_EXTENT), scale);
                jolt(Shape::scaled(&unit, to_jolt(scale)))
            }
        }
    }
}
