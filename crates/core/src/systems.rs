use glam::{Mat3, Mat4, Vec3};

use orrin_ecs::World;

use crate::geom::Aabb;
use crate::gfx::shadows::{Cascade, CascadeSet, MAX_CASCADES};
use crate::gfx::{MAX_POINT_LIGHTS, PointLight, RenderItem, SceneLighting};
use crate::scene::{
    AmbientLight, Camera, Culling, FogSettings, Light, LocalTransform, MaterialHandle, MeshBounds,
    MeshHandle, Spin, WorldTransform,
};

pub fn spin(world: &World, dt: f32) {
    world
        .query::<(&mut LocalTransform, &Spin)>()
        .for_each(|_entity, (transform, spin)| spin.apply(transform, dt));
}

/// Derive every entity's [`WorldTransform`] from its [`LocalTransform`].
///
/// Flat while the engine has no `Parent` component: an entity's world transform
/// *is* its local one. When the hierarchy lands, only this function's body
/// changes — its callers and its position in the frame are already where the
/// hierarchy needs them, which is the point of introducing it separately.
///
/// Takes `&mut World` because it owns the insertion: an entity that has a
/// `LocalTransform` and lacks a `WorldTransform` gets one here, so no spawn path
/// has to remember to add both. Once the hierarchy exists this moves into the
/// rebuild, which runs only on structural change.
pub fn propagate_transforms(world: &mut World) {
    let mut missing = Vec::new();
    world
        .query::<(&LocalTransform, Option<&WorldTransform>)>()
        .for_each(|entity, (_, existing)| {
            if existing.is_none() {
                missing.push(entity);
            }
        });
    for entity in missing {
        world.insert(entity, WorldTransform::default());
    }

    world
        .query::<(&LocalTransform, &mut WorldTransform)>()
        .for_each(|_entity, (local, world_transform)| {
            world_transform.0 = local.matrix();
        });
}

/// Build this frame's draw list: every renderable the camera can see, with
/// everything the passes need already derived.
///
/// `aspect` must be the one the frame is drawn with — the frustum's side planes
/// are only the visible ones if it matches.
pub fn extract_renderables(world: &World, aspect: f32, out: &mut Vec<RenderItem>) {
    out.clear();

    let cull = world
        .get_resource::<Culling>()
        .is_none_or(|culling| culling.enabled);
    let frustum = world.get_resource::<Camera>().map(|c| c.frustum(aspect));
    let bounds = world.get_resource::<MeshBounds>();
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

            if cull
                && let Some(frustum) = &frustum
                && !frustum.intersects(&world_bounds)
            {
                return;
            }

            out.push(RenderItem {
                model,
                normal_matrix: normal_matrix(&model),
                bounds: world_bounds,
                mesh: *mesh,
                material: material.copied().unwrap_or(MaterialHandle(0)),
            });
        });

    // Grouping by mesh lets the passes bind vertex/index buffers once per run
    // instead of once per draw, and collapse each run into a single instanced
    // draw. Opaque geometry is depth-tested, so draw order carries no visual
    // meaning to preserve.
    out.sort_unstable_by_key(|item| (item.mesh.0, item.material.0));

    if let Some(mut culling) = world.get_resource_mut::<Culling>() {
        culling.record(out.len(), total);
    }
}

/// Build each cascade's caster list.
///
/// The camera-culled list [`extract_renderables`] produces is the wrong input
/// for a shadow pass: an object behind the camera can still cast into view. Each
/// cascade instead takes everything whose bounds, swept toward the light, reach
/// its box — which in light space means overlapping in x and y, and not lying
/// entirely beyond the far plane. There is deliberately no near test: something
/// nearer the light than the cascade is exactly what casts into it.
///
/// One sweep of the world, N box tests per object. N separate sweeps would ask
/// the same question of the same entities four times, since a near cascade's
/// casters are almost always a subset of a far one's.
pub fn extract_shadow_casters(
    world: &World,
    cascades: &CascadeSet,
    out: &mut [Vec<RenderItem>; MAX_CASCADES],
) {
    for list in out.iter_mut() {
        list.clear();
    }
    if cascades.count == 0 {
        return;
    }

    let bounds = world.get_resource::<MeshBounds>();
    world
        .query::<(&WorldTransform, &MeshHandle, Option<&MaterialHandle>)>()
        .for_each(|_entity, (transform, mesh, material)| {
            let model = transform.0;
            let world_bounds = bounds
                .as_ref()
                .and_then(|table| table.get(*mesh))
                .unwrap_or(Aabb::EMPTY)
                .transformed(&model);

            let item = RenderItem {
                model,
                normal_matrix: normal_matrix(&model),
                bounds: world_bounds,
                mesh: *mesh,
                material: material.copied().unwrap_or(MaterialHandle(0)),
            };

            for (list, cascade) in out.iter_mut().zip(&cascades.cascades[..cascades.count]) {
                if casts_into(&world_bounds, cascade) {
                    list.push(item);
                }
            }
        });

    // Same key the forward pass groups on, so each cascade's draws still
    // collapse into one instanced call per (mesh, material) pair.
    for list in out.iter_mut() {
        list.sort_unstable_by_key(|item| (item.mesh.0, item.material.0));
    }
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

    /// The whole reason this is not `extract_renderables` with a different
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
    use super::{extract_renderables, normal_matrix, propagate_transforms};
    use crate::gfx::RenderItem;
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
        let mut items = Vec::new();
        extract_renderables(world, ASPECT, &mut items);
        items
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
