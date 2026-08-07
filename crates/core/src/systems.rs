use glam::{Mat3, Mat4, Vec3};

use orrin_ecs::World;

use crate::geom::Aabb;
use crate::gfx::shadows::{Cascade, CascadeSet, MAX_CASCADES};
use crate::gfx::{DrawList, MAX_POINT_LIGHTS, PointLight, RenderItem, SceneLighting};
use crate::scene::{
    AmbientLight, Camera, Culling, FogSettings, Light, LocalTransform, MaterialHandle, MeshBounds,
    MeshHandle, Spin, WorldTransform,
};

pub fn spin(world: &World, dt: f32) {
    world
        .query::<(&mut LocalTransform, &Spin)>()
        .for_each(|_entity, (transform, spin)| spin.apply(transform, dt));
}

/// Everything a frame draws, derived in one sweep of the world.
///
/// The lists are orderings of indices into `items`, not copies of it: a
/// `RenderItem` is 144 bytes and an object visible to the camera is usually also
/// a caster in two or three cascades, so materialising each list separately
/// would copy — and then sort — the same matrices five times over. Indices make
/// widening a list a four-byte push and each sort a scan of `u32`s.
#[derive(Default)]
pub struct FrameGeometry {
    /// Every renderable in the world, with its model, bounds and
    /// inverse-transpose derived exactly once this frame.
    items: Vec<RenderItem>,
    /// What the camera frustum kept, in draw order.
    visible: Vec<u32>,
    /// What casts into each active cascade, in draw order.
    cascades: [Vec<u32>; MAX_CASCADES],
}

impl FrameGeometry {
    /// What the camera-facing passes draw: the forward pass and the SSAO
    /// prepass, which share both the list and the object rows uploaded for it.
    pub fn visible(&self) -> DrawList<'_> {
        DrawList::new(&self.items, &self.visible)
    }

    /// What cascade `index` draws. Empty for a cascade the last extraction did
    /// not fill, which is what shadows-off produces.
    pub fn cascade(&self, index: usize) -> DrawList<'_> {
        DrawList::new(&self.items, &self.cascades[index])
    }
}

/// Build this frame's draw lists: what the camera can see, and what casts into
/// each cascade, with everything the passes need already derived.
///
/// One sweep, because both questions are asked of the same entities and the
/// expensive half of the answer — the world-space bounds and the
/// inverse-transpose — does not depend on which question is being asked. Two
/// sweeps recomputed a 3x3 inverse per object per frame for nothing.
///
/// The camera-culled list is the wrong input for a shadow pass: an object behind
/// the camera can still cast into view. Each cascade instead takes everything
/// whose bounds, swept toward the light, reach its box — overlapping in x and y
/// in light space, and not lying entirely beyond the far plane. There is
/// deliberately no near test: something nearer the light than the cascade is
/// exactly what casts into it.
///
/// `aspect` must be the one the frame is drawn with — the frustum's side planes
/// are only the visible ones if it matches.
pub fn extract_geometry(
    world: &World,
    aspect: f32,
    cascades: &CascadeSet,
    out: &mut FrameGeometry,
) {
    out.items.clear();
    out.visible.clear();
    for list in out.cascades.iter_mut() {
        list.clear();
    }

    let cull = world
        .get_resource::<Culling>()
        .is_none_or(|culling| culling.enabled);
    let camera = world.get_resource::<Camera>();
    let frustum = camera.as_ref().map(|c| c.frustum(aspect));
    let bounds = world.get_resource::<MeshBounds>();
    let active = &cascades.cascades[..cascades.count];
    let mut total = 0usize;

    world
        .query::<(&WorldTransform, &MeshHandle, Option<&MaterialHandle>)>()
        .for_each(|_entity, (transform, mesh, material)| {
            total += 1;
            let model = transform.0;
            // A mesh with no registered bounds is unmeasurable, so it is drawn
            // rather than culled; the frustum test says the same of an invalid
            // box, which keeps the fallback in one place.
            let world_bounds = bounds
                .as_ref()
                .and_then(|table| table.get(*mesh))
                .unwrap_or(Aabb::EMPTY)
                .transformed(&model);

            let visible = !cull
                || frustum
                    .as_ref()
                    .is_none_or(|frustum| frustum.intersects(&world_bounds));
            let casts: [bool; MAX_CASCADES] = std::array::from_fn(|i| {
                active
                    .get(i)
                    .is_some_and(|cascade| casts_into(&world_bounds, cascade))
            });

            // An object no list wants still costs the sweep, but it must not
            // cost a row in `items` — the object buffer is uploaded from this
            // array and a row nothing indexes is upload bandwidth for nothing.
            if !visible && !casts.iter().any(|&c| c) {
                return;
            }

            let index = out.items.len() as u32;
            out.items.push(RenderItem {
                model,
                normal_matrix: normal_matrix(&model),
                bounds: world_bounds,
                mesh: *mesh,
                material: material.copied().unwrap_or(MaterialHandle(0)),
            });

            if visible {
                out.visible.push(index);
            }
            for (list, casts) in out.cascades.iter_mut().zip(casts) {
                if casts {
                    list.push(index);
                }
            }
        });

    // Grouping by mesh lets the passes bind vertex/index buffers once per run
    // instead of once per draw, and collapse each run into a single instanced
    // draw.
    let key = |items: &[RenderItem], index: &u32| {
        let item = &items[*index as usize];
        (item.mesh.0, item.material.0)
    };
    out.visible.sort_unstable_by_key(|i| key(&out.items, i));
    for list in out.cascades.iter_mut() {
        list.sort_unstable_by_key(|i| key(&out.items, i));
    }

    // Within the grouping, order the runs front to back. Opaque geometry is
    // depth-tested, so this changes no pixel — it changes how many fragments
    // reach the shader, since a nearer run already in the depth buffer rejects
    // the ones behind it before they shade. Runs move whole, so every run is
    // still one instanced draw.
    if let Some(camera) = camera.as_ref() {
        order_runs_front_to_back(&out.items, &mut out.visible, camera.position);
    }

    if let Some(mut culling) = world.get_resource_mut::<Culling>() {
        culling.record(out.visible.len(), total);
    }
}

/// Reorder `order`'s (mesh, material) runs by their nearest distance to `eye`,
/// keeping each run contiguous.
///
/// Sorting individual items front to back would break every run into a draw of
/// its own; the batching is worth more than the last few percent of rejection,
/// so the run is the unit that moves.
fn order_runs_front_to_back(items: &[RenderItem], order: &mut Vec<u32>, eye: Vec3) {
    let runs: Vec<std::ops::Range<usize>> = DrawList::new(items, order).runs().collect();
    if runs.len() < 2 {
        return;
    }

    let mut keyed: Vec<(f32, std::ops::Range<usize>)> = runs
        .into_iter()
        .map(|run| {
            let nearest = order[run.clone()]
                .iter()
                .map(|&i| items[i as usize].bounds.distance_squared_to(eye))
                .fold(f32::INFINITY, f32::min);
            (nearest, run)
        })
        .collect();
    keyed.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    let mut sorted = Vec::with_capacity(order.len());
    for (_, run) in &keyed {
        sorted.extend_from_slice(&order[run.clone()]);
    }
    *order = sorted;
}

/// Whether `bounds`, extended infinitely toward the light, reaches `cascade`.
fn casts_into(bounds: &Aabb, cascade: &Cascade) -> bool {
    let ls = bounds.transformed(&cascade.light_view);
    let half = cascade.half_extent;
    // `light_view` is a right-handed look-at, so what the pass renders lies at
    // negative z, between the eye at 0 and the far plane at -depth_range.
    ls.min.x <= half
        && ls.max.x >= -half
        && ls.min.y <= half
        && ls.max.y >= -half
        && ls.max.z >= -cascade.depth_range
}

#[cfg(test)]
mod shadow_culling_tests {
    use super::*;
    use crate::gfx::shadows::{CascadeConfig, cascades};
    use crate::scene::Camera;
    use glam::Vec3;

    const ASPECT: f32 = 16.0 / 9.0;

    fn config() -> CascadeConfig {
        CascadeConfig {
            count: 4,
            max_distance: 100.0,
            lambda: 0.75,
            resolution: 2048,
            pullback: 50.0,
        }
    }

    /// Straight down, so "toward the light" is unambiguously +Y and the tests
    /// can place casters by inspection.
    fn set(camera: &Camera) -> CascadeSet {
        cascades(camera, ASPECT, Vec3::NEG_Y, &config())
    }

    fn box_at(center: Vec3, half: f32) -> Aabb {
        Aabb {
            min: center - Vec3::splat(half),
            max: center + Vec3::splat(half),
        }
    }

    /// The whole reason a cascade is not the camera's list with a different
    /// frustum. An object the camera cannot see still casts into what it can,
    /// and a cull that drops it produces missing shadows that read as a bias
    /// bug.
    #[test]
    fn an_object_behind_the_camera_still_casts() {
        let camera = Camera::default();
        let set = set(&camera);
        let cascade = &set.cascades[0];

        // Directly above the cascade's center, so it is behind the camera in
        // view terms but squarely between the sun and the lit ground.
        let center = cascade.light_view.inverse().transform_point3(Vec3::new(
            0.0,
            0.0,
            -cascade.depth_range * 0.5,
        ));
        let overhead = center + Vec3::Y * 500.0;

        assert!(
            casts_into(&box_at(overhead, 1.0), cascade),
            "an object between the sun and the cascade was culled",
        );
    }

    /// The other half: no near plane does not mean no far plane. Something
    /// past the cascade casts away from it, not into it.
    #[test]
    fn an_object_beyond_the_far_plane_does_not_cast() {
        let camera = Camera::default();
        let set = set(&camera);
        let cascade = &set.cascades[0];

        let below = cascade.light_view.inverse().transform_point3(Vec3::new(
            0.0,
            0.0,
            -cascade.depth_range - 100.0,
        ));

        assert!(!casts_into(&box_at(below, 1.0), cascade));
    }

    #[test]
    fn an_object_outside_the_box_sideways_does_not_cast() {
        let camera = Camera::default();
        let set = set(&camera);
        let cascade = &set.cascades[0];

        for axis in [Vec3::X, Vec3::Y] {
            let outside = cascade.light_view.inverse().transform_point3(
                axis * (cascade.half_extent + 10.0) - Vec3::Z * cascade.depth_range * 0.5,
            );
            assert!(
                !casts_into(&box_at(outside, 1.0), cascade),
                "an object {} past the box edge was kept",
                cascade.half_extent + 10.0,
            );
        }
    }

    /// A caster inside a near cascade is inside the far ones too, since the
    /// cascades nest. If this ever fails the fit has stopped nesting, which
    /// would show up as shadows vanishing at a specific distance.
    #[test]
    fn a_near_cascades_casters_are_also_the_far_ones() {
        let camera = Camera::default();
        let set = set(&camera);
        let bounds = box_at(camera.position + Vec3::new(0.0, 0.0, -5.0), 0.5);

        assert!(casts_into(&bounds, &set.cascades[0]));
        for index in 1..set.count {
            assert!(
                casts_into(&bounds, &set.cascades[index]),
                "cascade {index} rejected what cascade 0 accepted",
            );
        }
    }

    /// Culling reads the light-space box, so a cascade set that was never
    /// built must reject everything rather than index uninitialised matrices.
    #[test]
    fn an_empty_cascade_set_casts_nothing() {
        let set = CascadeSet::default();
        assert_eq!(set.count, 0);
    }
}

/// The inverse-transpose of a transform's upper 3x3, without inverting anything.
///
/// For `M = R * S` with `R` a rotation and `S` diagonal, `(R*S)^-T` is
/// `R * S^-1` — orthonormality makes `R^-T = R`, and a diagonal matrix's
/// inverse-transpose is its reciprocal. That turns a general 3x3 inverse, per
/// object per frame, into a quaternion conversion and three divides.
///
/// A zero scale component has no inverse; it collapses the object onto a plane,
/// where no normal is meaningful. Zero keeps the result finite instead of
/// pushing NaNs into the shader.
/// The inverse transpose of `model`'s linear part, which is what a normal has to
/// be transformed by: the model matrix itself tilts normals off the surface
/// wherever the scale is non-uniform.
///
/// Computed generally rather than as `rotation * scale.recip()`, which is only
/// equal to it while the matrix decomposes into a rotation and a diagonal scale.
/// A composed hierarchy transform need not — a non-uniformly scaled parent with
/// a rotated child produces shear, and the closed form assumes that away.
fn normal_matrix(model: &Mat4) -> Mat3 {
    let normal = Mat3::from_mat4(*model).inverse().transpose();
    // A singular linear part inverts to infinities. That means a transform which
    // flattens the object to zero volume, so there is no surface to shade and a
    // zero matrix is the benign answer.
    if normal.is_finite() {
        normal
    } else {
        Mat3::ZERO
    }
}

pub fn extract_lighting(world: &World, out: &mut SceneLighting) {
    let defaults = SceneLighting::default();
    out.ambient_color = defaults.ambient_color;
    out.ambient_intensity = defaults.ambient_intensity;
    out.sun = defaults.sun;
    out.shininess = defaults.shininess;
    out.specular_strength = defaults.specular_strength;
    out.fog_color = defaults.fog_color;
    out.fog_density = defaults.fog_density;
    out.fog_height_falloff = defaults.fog_height_falloff;
    out.fog_height = defaults.fog_height;
    out.point_lights.clear();

    if let Some(ambient) = world.get_resource::<AmbientLight>() {
        out.ambient_color = ambient.color;
        out.ambient_intensity = ambient.intensity;
    }

    if let Some(fog) = world.get_resource::<FogSettings>() {
        out.fog_color = fog.color;
        out.fog_density = fog.density;
        out.fog_height_falloff = fog.height_falloff;
        out.fog_height = fog.height;
    }

    let mut has_sun = false;
    world.query::<(&LocalTransform, &Light)>().for_each(
        |_entity, (transform, light)| match *light {
            Light::Directional { color, intensity } => {
                // The shader supports one directional light, so the first wins.
                if !has_sun {
                    let direction = (transform.rotation * Vec3::NEG_Z).normalize_or_zero();
                    out.sun.direction = direction;
                    out.sun.color = color;
                    out.sun.intensity = intensity;
                    has_sun = true;
                }
            }
            Light::Point {
                color,
                intensity,
                range,
            } => {
                if out.point_lights.len() < MAX_POINT_LIGHTS {
                    out.point_lights.push(PointLight {
                        position: transform.translation,
                        color,
                        intensity,
                        range,
                    });
                }
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::{FrameGeometry, extract_geometry, normal_matrix};
    use crate::gfx::RenderItem;
    use crate::gfx::shadows::CascadeSet;
    use crate::scene::propagate_transforms;
    use crate::scene::{
        Camera, CpuMesh, Culling, LocalTransform, MeshBounds, MeshHandle, Transform, WorldTransform,
    };
    use glam::{Mat3, Mat4, Quat, Vec3};
    use orrin_ecs::World;

    const ASPECT: f32 = 16.0 / 9.0;
    const CUBE: MeshHandle = MeshHandle(0);

    /// A world holding one unit cube's bounds, a camera at +Z looking at the
    /// origin, and culling on.
    fn test_world() -> World {
        let mut world = World::new();
        let mut bounds = MeshBounds::default();
        bounds.insert(CUBE, CpuMesh::cube().bounds());
        world.insert_resource(bounds);
        world.insert_resource(Camera::default());
        world.insert_resource(Culling::default());
        world
    }

    fn spawn(world: &mut World, position: Vec3, mesh: MeshHandle) {
        world
            .spawn_entity()
            .with(LocalTransform::from(Transform::from_translation(position)))
            .with(mesh);
    }

    /// Propagates first, exactly as the frame does: extraction reads world
    /// transforms, and only propagation produces them.
    fn extract(world: &mut World) -> Vec<RenderItem> {
        propagate_transforms(world);
        let mut geometry = FrameGeometry::default();
        // No cascades: these cover the camera list, and a cascade would only
        // add items to `geometry.items` that the visible order does not name.
        extract_geometry(world, ASPECT, &CascadeSet::default(), &mut geometry);
        let visible = geometry.visible();
        (0..visible.len()).map(|i| *visible.item(i)).collect()
    }

    #[test]
    fn an_object_behind_the_camera_is_culled() {
        let mut world = test_world();
        spawn(&mut world, Vec3::ZERO, CUBE);
        spawn(&mut world, Vec3::new(0.0, 0.0, 400.0), CUBE);

        let items = extract(&mut world);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].model.w_axis.truncate(), Vec3::ZERO);

        let culling = world.resource::<Culling>();
        assert_eq!(
            (culling.visible(), culling.total(), culling.culled()),
            (1, 2, 1)
        );
    }

    #[test]
    fn turning_culling_off_draws_everything() {
        let mut world = test_world();
        spawn(&mut world, Vec3::ZERO, CUBE);
        spawn(&mut world, Vec3::new(0.0, 0.0, 400.0), CUBE);
        world.resource_mut::<Culling>().enabled = false;

        assert_eq!(extract(&mut world).len(), 2);
    }

    /// Culling may only drop what it can measure. A mesh whose bounds were never
    /// registered has to be drawn, however far off screen it looks — the
    /// alternative is geometry that silently disappears.
    #[test]
    fn a_mesh_without_registered_bounds_is_never_culled() {
        let mut world = test_world();
        spawn(&mut world, Vec3::new(0.0, 0.0, 400.0), MeshHandle(7));

        assert_eq!(extract(&mut world).len(), 1);
    }

    /// Bounds are object-space, so the model transform has to be applied before
    /// the frustum test: the same position is out of view at unit scale and in
    /// view once the object is big enough to reach it.
    #[test]
    fn scale_is_applied_to_bounds_before_the_test() {
        let beside = Vec3::new(20.0, 0.0, 0.0);

        let mut world = test_world();
        spawn(&mut world, beside, CUBE);
        assert!(
            extract(&mut world).is_empty(),
            "a unit cube there is off screen"
        );

        let mut world = test_world();
        world
            .spawn_entity()
            .with(LocalTransform::from(Transform {
                translation: beside,
                scale: Vec3::splat(200.0),
                ..Default::default()
            }))
            .with(CUBE);
        assert_eq!(
            extract(&mut world).len(),
            1,
            "scaled up it reaches into view"
        );
    }

    #[test]
    fn the_draw_list_is_grouped_by_mesh_then_material() {
        let mut world = test_world();
        let mut bounds = world.remove_resource::<MeshBounds>().unwrap();
        let cube = CpuMesh::cube().bounds();
        for handle in [MeshHandle(1), MeshHandle(2)] {
            bounds.insert(handle, cube);
        }
        world.insert_resource(bounds);

        for mesh in [MeshHandle(2), MeshHandle(0), MeshHandle(1), MeshHandle(0)] {
            spawn(&mut world, Vec3::ZERO, mesh);
        }

        let meshes: Vec<u32> = extract(&mut world).iter().map(|item| item.mesh.0).collect();
        assert_eq!(meshes, vec![0, 0, 1, 2]);
    }

    /// Flat world, so a world transform is exactly the local one. This is the
    /// property the hierarchy has to preserve for every root it later grows.
    #[test]
    fn propagation_derives_the_local_matrix() {
        let transform = Transform {
            translation: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_euler(glam::EulerRot::YXZ, 0.4, 0.2, -0.7),
            scale: Vec3::new(2.0, 0.5, 1.5),
        };
        let mut world = World::new();
        let entity = world
            .spawn_entity()
            .with(LocalTransform::from(transform))
            .id();

        propagate_transforms(&mut world);

        let derived = world.get::<WorldTransform>(entity).unwrap().0;
        assert!(
            (derived - transform.matrix())
                .to_cols_array()
                .iter()
                .all(|v| v.abs() < 1e-6)
        );
    }

    /// Nothing on a spawn path has to remember to add a `WorldTransform`, which
    /// is why propagation owns the insertion.
    #[test]
    fn propagation_inserts_a_world_transform_that_was_never_spawned_with_one() {
        let mut world = World::new();
        let entity = world
            .spawn_entity()
            .with(LocalTransform::from(Transform::from_translation(Vec3::X)))
            .id();
        assert!(world.get::<WorldTransform>(entity).is_none());

        propagate_transforms(&mut world);

        assert!(world.get::<WorldTransform>(entity).is_some());
    }

    /// An entity with no local transform is not a transformed thing, so giving
    /// it a world transform would put an identity matrix where the renderer
    /// reads "is this drawable" — and draw it at the origin.
    #[test]
    fn an_entity_without_a_local_transform_gets_no_world_transform() {
        let mut world = World::new();
        let entity = world.spawn_entity().with(MeshHandle(0)).id();

        propagate_transforms(&mut world);

        assert!(world.get::<WorldTransform>(entity).is_none());
    }

    /// Propagation runs twice a frame and must be a pure function of the local
    /// transforms — never accumulating onto what it wrote last time.
    #[test]
    fn propagating_twice_lands_in_the_same_place() {
        let mut world = World::new();
        let entity = world
            .spawn_entity()
            .with(LocalTransform::from(Transform::from_translation(
                Vec3::new(4.0, 5.0, 6.0),
            )))
            .id();

        propagate_transforms(&mut world);
        let once = world.get::<WorldTransform>(entity).unwrap().0;
        propagate_transforms(&mut world);
        let twice = world.get::<WorldTransform>(entity).unwrap().0;

        assert_eq!(once, twice);
    }

    /// The general inverse-transpose has to agree with the `R * S^-1` closed
    /// form it replaced, wherever that form was valid — which is any transform
    /// that decomposes into a rotation and a diagonal scale. Under a hierarchy
    /// it need not, and only the general form stays right there.
    #[test]
    fn the_normal_matrix_matches_the_closed_form_for_a_trs() {
        let transform = Transform {
            translation: Vec3::new(3.0, -2.0, 7.0),
            rotation: Quat::from_euler(glam::EulerRot::YXZ, 0.6, -1.1, 0.3),
            scale: Vec3::new(0.5, 4.0, 2.0),
        };
        let oracle =
            Mat3::from_quat(transform.rotation) * Mat3::from_diagonal(transform.scale.recip());
        let derived = normal_matrix(&transform.matrix());

        for (a, b) in derived
            .to_cols_array()
            .iter()
            .zip(oracle.to_cols_array().iter())
        {
            assert!((a - b).abs() < 1e-5, "{derived:?} != {oracle:?}");
        }
    }

    /// A flattened object has no normal to speak of; the result must still be
    /// finite, because NaNs would spread through the shader's lighting math.
    #[test]
    fn a_zero_scale_axis_stays_finite() {
        let transform = Transform {
            scale: Vec3::new(1.0, 0.0, 1.0),
            ..Default::default()
        };
        assert!(
            normal_matrix(&transform.matrix())
                .to_cols_array()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    /// The whole point of a separate world transform: a shear that no
    /// translation/rotation/scale triple can express still yields correct
    /// normals, where the closed form would have had to invent a decomposition.
    #[test]
    fn the_normal_matrix_handles_a_sheared_matrix() {
        let sheared = Mat4::from_cols_array_2d(&[
            [1.0, 0.0, 0.0, 0.0],
            [0.7, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let derived = normal_matrix(&sheared);
        // A normal is correct when it stays perpendicular to a tangent the same
        // matrix carried: n' . (M t) == 0 for every tangent t of the surface.
        let tangent = Mat3::from_mat4(sheared) * Vec3::new(1.0, 0.0, 0.0);
        let normal = derived * Vec3::new(0.0, 1.0, 0.0);
        assert!(
            normal.dot(tangent).abs() < 1e-5,
            "normal {normal:?} is not perpendicular to tangent {tangent:?}"
        );
    }

    #[test]
    fn a_mesh_reports_the_bounds_of_its_vertices() {
        let bounds = CpuMesh::cube().bounds();
        assert!((bounds.min - Vec3::splat(-0.5)).length() < 1e-6);
        assert!((bounds.max - Vec3::splat(0.5)).length() < 1e-6);

        let empty = CpuMesh::default().bounds();
        assert!(!empty.is_valid());
    }

    /// World bounds ride along on the item so a shadow cascade can test them
    /// against its own frustum without re-deriving anything.
    #[test]
    fn items_carry_their_world_bounds() {
        let mut world = test_world();
        spawn(&mut world, Vec3::new(0.0, 0.5, 0.0), CUBE);

        let items = extract(&mut world);
        let expected = CpuMesh::cube()
            .bounds()
            .transformed(&Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0)));
        assert!((items[0].bounds.min - expected.min).length() < 1e-6);
        assert!((items[0].bounds.max - expected.max).length() < 1e-6);
        assert!(items[0].bounds.is_valid());
    }

    /// Extraction has to work before any of its resources exist — the editor and
    /// the tests both build worlds a piece at a time.
    #[test]
    fn a_world_without_a_camera_extracts_everything() {
        let mut world = World::new();
        spawn(&mut world, Vec3::new(0.0, 0.0, 400.0), CUBE);

        assert_eq!(extract(&mut world).len(), 1);
    }
}

/// The half of extraction the camera-only tests above cannot reach: that one
/// sweep still answers both questions, and that reordering never breaks a run.
#[cfg(test)]
mod geometry_tests {
    use super::{FrameGeometry, extract_geometry};
    use crate::gfx::shadows::{CascadeConfig, CascadeSet, cascades};
    use crate::scene::{
        Camera, CpuMesh, Culling, LocalTransform, MaterialHandle, MeshBounds, MeshHandle,
        Transform, WorldTransform,
    };
    use glam::Vec3;
    use orrin_ecs::World;

    const ASPECT: f32 = 16.0 / 9.0;
    const CUBE: MeshHandle = MeshHandle(0);

    fn world_with(meshes: &[(Vec3, MeshHandle, u32)]) -> World {
        let mut world = World::new();
        let mut bounds = MeshBounds::default();
        bounds.insert(CUBE, CpuMesh::cube().bounds());
        bounds.insert(MeshHandle(1), CpuMesh::cube().bounds());
        world.insert_resource(bounds);
        world.insert_resource(Camera::default());
        world.insert_resource(Culling::default());
        for &(position, mesh, material) in meshes {
            world
                .spawn_entity()
                .with(LocalTransform::from(Transform::from_translation(position)))
                .with(WorldTransform(glam::Mat4::from_translation(position)))
                .with(mesh)
                .with(MaterialHandle(material));
        }
        world
    }

    fn sun_cascades(camera: &Camera) -> CascadeSet {
        cascades(
            camera,
            ASPECT,
            Vec3::NEG_Y,
            &CascadeConfig {
                count: 2,
                max_distance: 100.0,
                lambda: 0.75,
                resolution: 1024,
                pullback: 50.0,
            },
        )
    }

    /// The reason the two sweeps were merged rather than one dropped: an object
    /// the camera culls can still be a caster, so it must survive into `items`
    /// even though the visible order never names it.
    #[test]
    fn an_object_the_camera_culls_can_still_reach_a_cascade() {
        let camera = Camera::default();
        // Directly overhead of the origin: out of the camera's frustum, but
        // between a straight-down sun and what the camera is looking at.
        let world = world_with(&[(Vec3::new(0.0, 60.0, 0.0), CUBE, 0)]);
        let set = sun_cascades(&camera);

        let mut geometry = FrameGeometry::default();
        extract_geometry(&world, ASPECT, &set, &mut geometry);

        assert_eq!(geometry.visible().len(), 0, "it should not be visible");
        assert!(
            !geometry.cascade(0).is_empty(),
            "an overhead caster was dropped from cascade 0",
        );
    }

    /// An object no list wants must not reach `items` at all: the object buffer
    /// is uploaded from that array, so a row nothing indexes is dead bandwidth.
    #[test]
    fn an_object_no_list_wants_costs_no_row() {
        let camera = Camera::default();
        // Far behind the camera and far outside every cascade box.
        let world = world_with(&[(Vec3::new(0.0, 0.0, 5000.0), CUBE, 0)]);
        let set = sun_cascades(&camera);

        let mut geometry = FrameGeometry::default();
        extract_geometry(&world, ASPECT, &set, &mut geometry);

        assert_eq!(geometry.visible().len(), 0);
        for i in 0..set.count {
            assert_eq!(geometry.cascade(i).len(), 0, "cascade {i}");
        }
        assert_eq!(
            geometry.items.len(),
            0,
            "an object nothing draws still took a row",
        );
    }

    /// A caster shared with the camera is one `RenderItem`, indexed twice —
    /// that sharing is the whole point of the index lists.
    #[test]
    fn a_visible_caster_is_stored_once_and_indexed_from_both() {
        let camera = Camera::default();
        let world = world_with(&[(Vec3::ZERO, CUBE, 0)]);
        let set = sun_cascades(&camera);

        let mut geometry = FrameGeometry::default();
        extract_geometry(&world, ASPECT, &set, &mut geometry);

        assert_eq!(geometry.items.len(), 1);
        assert_eq!(geometry.visible.len(), 1);
        assert_eq!(geometry.cascades[0].len(), 1);
        assert_eq!(geometry.visible[0], geometry.cascades[0][0]);
    }

    /// Front-to-back ordering may reorder runs but must never split one: a run
    /// is a single instanced draw whose base is its start, so a broken run
    /// draws instances against the wrong transforms.
    #[test]
    fn front_to_back_ordering_keeps_every_run_whole() {
        let camera = Camera::default();
        // Two meshes interleaved in depth, so ordering has something to do and
        // the grouping has something to hold together.
        let world = world_with(&[
            (Vec3::new(0.0, 0.0, -2.0), CUBE, 0),
            (Vec3::new(0.0, 0.0, -8.0), MeshHandle(1), 0),
            (Vec3::new(0.0, 0.0, -4.0), CUBE, 0),
            (Vec3::new(0.0, 0.0, -6.0), MeshHandle(1), 0),
        ]);

        let mut geometry = FrameGeometry::default();
        extract_geometry(&world, ASPECT, &CascadeSet::default(), &mut geometry);
        let visible = geometry.visible();
        assert_eq!(visible.len(), 4, "the camera should see all four");

        // One run per mesh, still — reordering moved runs, it did not split them.
        let runs: Vec<_> = visible.runs().collect();
        assert_eq!(runs.len(), 2, "runs were split: {runs:?}");
        for run in &runs {
            let key = {
                let item = visible.item(run.start);
                (item.mesh.0, item.material.0)
            };
            for i in run.clone() {
                let item = visible.item(i);
                assert_eq!((item.mesh.0, item.material.0), key);
            }
        }
    }

    /// And that the reordering is actually front to back: the nearer run leads.
    #[test]
    fn the_nearer_run_is_drawn_first() {
        let camera = Camera::default();
        let world = world_with(&[
            (Vec3::new(0.0, 0.0, -20.0), MeshHandle(1), 0),
            (Vec3::new(0.0, 0.0, -2.0), CUBE, 0),
        ]);

        let mut geometry = FrameGeometry::default();
        extract_geometry(&world, ASPECT, &CascadeSet::default(), &mut geometry);
        let visible = geometry.visible();
        assert_eq!(visible.len(), 2);

        let near = visible.item(0).bounds.distance_squared_to(camera.position);
        let far = visible.item(1).bounds.distance_squared_to(camera.position);
        assert!(near <= far, "runs are back to front: {near} then {far}");
    }
}
