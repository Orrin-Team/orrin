use glam::{Mat3, Vec3};

use orrin_ecs::World;

use crate::geom::Aabb;
use crate::gfx::{PointLight, RenderItem, SceneLighting, MAX_POINT_LIGHTS};
use crate::scene::{
    AmbientLight, Camera, Culling, FogSettings, Light, LocalTransform, MaterialHandle, MeshBounds,
    MeshHandle, Spin, Transform,
};

pub fn spin(world: &World, dt: f32) {
    world
        .query::<(&mut LocalTransform, &Spin)>()
        .for_each(|_entity, (transform, spin)| spin.apply(transform, dt));
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
        .query::<(&LocalTransform, &MeshHandle, Option<&MaterialHandle>)>()
        .for_each(|_entity, (transform, mesh, material)| {
            total += 1;
            let model = transform.matrix();
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
                normal_matrix: normal_matrix(transform),
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
fn normal_matrix(transform: &Transform) -> Mat3 {
    let inverse_scale = Vec3::select(
        transform.scale.abs().cmpgt(Vec3::splat(f32::EPSILON)),
        transform.scale.recip(),
        Vec3::ZERO,
    );
    Mat3::from_quat(transform.rotation) * Mat3::from_diagonal(inverse_scale)
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
    world
        .query::<(&LocalTransform, &Light)>()
        .for_each(|_entity, (transform, light)| match *light {
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
        });
}

#[cfg(test)]
mod tests {
    use super::{extract_renderables, normal_matrix};
    use crate::gfx::RenderItem;
    use crate::scene::{
        Camera, CpuMesh, Culling, LocalTransform, MeshBounds, MeshHandle, Transform,
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

    fn extract(world: &World) -> Vec<RenderItem> {
        let mut items = Vec::new();
        extract_renderables(world, ASPECT, &mut items);
        items
    }

    #[test]
    fn an_object_behind_the_camera_is_culled() {
        let mut world = test_world();
        spawn(&mut world, Vec3::ZERO, CUBE);
        spawn(&mut world, Vec3::new(0.0, 0.0, 400.0), CUBE);

        let items = extract(&world);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].model.w_axis.truncate(), Vec3::ZERO);

        let culling = world.resource::<Culling>();
        assert_eq!((culling.visible(), culling.total(), culling.culled()), (1, 2, 1));
    }

    #[test]
    fn turning_culling_off_draws_everything() {
        let mut world = test_world();
        spawn(&mut world, Vec3::ZERO, CUBE);
        spawn(&mut world, Vec3::new(0.0, 0.0, 400.0), CUBE);
        world.resource_mut::<Culling>().enabled = false;

        assert_eq!(extract(&world).len(), 2);
    }

    /// Culling may only drop what it can measure. A mesh whose bounds were never
    /// registered has to be drawn, however far off screen it looks — the
    /// alternative is geometry that silently disappears.
    #[test]
    fn a_mesh_without_registered_bounds_is_never_culled() {
        let mut world = test_world();
        spawn(&mut world, Vec3::new(0.0, 0.0, 400.0), MeshHandle(7));

        assert_eq!(extract(&world).len(), 1);
    }

    /// Bounds are object-space, so the model transform has to be applied before
    /// the frustum test: the same position is out of view at unit scale and in
    /// view once the object is big enough to reach it.
    #[test]
    fn scale_is_applied_to_bounds_before_the_test() {
        let beside = Vec3::new(20.0, 0.0, 0.0);

        let mut world = test_world();
        spawn(&mut world, beside, CUBE);
        assert!(extract(&world).is_empty(), "a unit cube there is off screen");

        let mut world = test_world();
        world
            .spawn_entity()
            .with(LocalTransform::from(Transform {
                translation: beside,
                scale: Vec3::splat(200.0),
                ..Default::default()
            }))
            .with(CUBE);
        assert_eq!(extract(&world).len(), 1, "scaled up it reaches into view");
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

        let meshes: Vec<u32> = extract(&world).iter().map(|item| item.mesh.0).collect();
        assert_eq!(meshes, vec![0, 0, 1, 2]);
    }

    /// The cheap `R * S^-1` form has to agree with the general inverse-transpose
    /// it replaced, or lighting goes wrong under non-uniform scale.
    #[test]
    fn the_normal_matrix_matches_a_general_inverse_transpose() {
        let transform = Transform {
            translation: Vec3::new(3.0, -2.0, 7.0),
            rotation: Quat::from_euler(glam::EulerRot::YXZ, 0.6, -1.1, 0.3),
            scale: Vec3::new(0.5, 4.0, 2.0),
        };
        let oracle = Mat3::from_mat4(transform.matrix()).inverse().transpose();
        let derived = normal_matrix(&transform);

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
        assert!(normal_matrix(&transform).to_cols_array().iter().all(|v| v.is_finite()));
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

        let items = extract(&world);
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

        assert_eq!(extract(&world).len(), 1);
    }
}
