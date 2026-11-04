use glam::{Mat3, Mat4, Vec3};

use orrin_ecs::{Entity, World};

use crate::geom::Aabb;
use crate::gfx::{
    BlendMode, DecalInstance, DrawList, MAX_DECALS, MAX_POINT_LIGHTS, MAX_SPOT_LIGHTS, PointLight,
    RenderItem, SceneLighting, SpotLight,
};
use crate::scene::{
    AmbientLight, Camera, Culling, Decal, DecalSettings, Light, LocalTransform, MAX_CONE_ANGLE,
    MIN_CONE_ANGLE, MaterialBlends, MaterialHandle, MeshBounds, MeshHandle, Spin, WorldTransform,
};

pub fn spin(world: &World, dt: f32) {
    world
        .query::<(&mut LocalTransform, &Spin)>()
        .for_each(|_entity, (transform, spin)| spin.apply(transform, dt));
}

/// Everything a frame draws, derived in one sweep of the world.
///
/// The lists are orderings of indices into `items`, not copies of it: a
/// `RenderItem` is 144 bytes, and indices make widening a list a four-byte push
/// and each sort a scan of `u32`s.
#[derive(Default)]
pub struct FrameGeometry {
    /// Every renderable in the world, with its model, bounds and
    /// inverse-transpose derived exactly once this frame.
    items: Vec<RenderItem>,
    /// Every opaque renderable, offered to the cull dispatch, which decides per
    /// view — the camera, each cascade, each punctual face — what is drawn. A
    /// set rather than a draw order: `cull::CullPass::prepare` groups it.
    visible: Vec<u32>,
    /// What the camera frustum kept and blends. Kept apart rather than tagged,
    /// because every consumer wants one or the other whole: the prepass and the
    /// forward pass draw the first, the accumulation pass draws the second, and
    /// a run in either is still one instanced draw.
    transparent: Vec<u32>,
    /// What the camera frustum kept and refracts. A third list rather than a
    /// share of the second because the two disagree about order: the blended
    /// list must not be sorted and this one must be.
    refractive: Vec<u32>,
    /// Where every extracted entity stood last frame, and where it stands now.
    /// Swapped rather than updated in place at the end of each extraction, so an
    /// entity nobody extracted last frame has no entry at all — which is what
    /// makes `prev_model` fall back to `model` instead of to a matrix from
    /// however many frames ago the object was last on screen.
    motion: MotionHistory,
}

/// Last frame's world matrix per entity slot.
///
/// Indexed by slot rather than hashed because slots are dense and extraction
/// already walks every renderable; the generation is stored alongside because a
/// slot outlives the entity in it, and a reused slot must read as new rather
/// than as the dead entity teleporting.
#[derive(Default)]
struct MotionHistory {
    previous: Vec<Option<(u32, Mat4)>>,
    current: Vec<Option<(u32, Mat4)>>,
}

impl MotionHistory {
    fn begin(&mut self) {
        self.current.clear();
    }

    /// Where `entity` stood last frame, defaulting to `model` for anything the
    /// last extraction did not see.
    fn record(&mut self, entity: Entity, model: Mat4) -> Mat4 {
        let slot = entity.index() as usize;
        if slot >= self.current.len() {
            self.current.resize(slot + 1, None);
        }
        self.current[slot] = Some((entity.generation(), model));

        match self.previous.get(slot).copied().flatten() {
            Some((generation, previous)) if generation == entity.generation() => previous,
            _ => model,
        }
    }

    fn end(&mut self) {
        std::mem::swap(&mut self.previous, &mut self.current);
    }
}

impl FrameGeometry {
    /// What the camera-facing passes draw: the forward pass and the SSAO
    /// prepass, which share both the list and the object rows uploaded for it.
    pub fn visible(&self) -> DrawList<'_> {
        DrawList::new(&self.items, &self.visible)
    }

    /// What the weighted-blended pass accumulates. Empty when nothing on screen
    /// uses a [`BlendMode::Blend`](crate::gfx::BlendMode::Blend) material, which
    /// is the frame where that pass clears its two targets and draws nothing.
    pub fn transparent(&self) -> DrawList<'_> {
        DrawList::new(&self.items, &self.transparent)
    }

    /// What the refraction pass draws, ordered back to front. Empty when nothing
    /// on screen uses a
    /// [`BlendMode::Transmissive`](crate::gfx::BlendMode::Transmissive)
    /// material, which is the frame where that pass clears its target and draws
    /// nothing.
    pub fn refractive(&self) -> DrawList<'_> {
        DrawList::new(&self.items, &self.refractive)
    }
}

/// Build this frame's draw lists, with everything the passes need already
/// derived.
///
/// Opaque geometry is culled on the GPU, for the camera and every shadow view
/// alike, so all of it is offered and none of it is tested here: the dispatch
/// transforms each mesh's own bounds itself. The blended and refractive queues
/// are culled here instead, because they are drawn in the order this sweep
/// leaves them in — and a refractive surface has to be drawn back to front,
/// which is a correctness requirement no atomic append meets. Neither casts a
/// shadow, so neither needs a light's view.
///
/// `aspect` must be the one the frame is drawn with — the frustum's side planes
/// are only the visible ones if it matches.
pub fn extract_geometry(world: &World, aspect: f32, out: &mut FrameGeometry) {
    out.items.clear();
    out.visible.clear();
    out.transparent.clear();
    out.refractive.clear();
    out.motion.begin();

    let cull = world
        .get_resource::<Culling>()
        .is_none_or(|culling| culling.enabled);
    let camera = world.get_resource::<Camera>();
    let frustum = camera.as_ref().map(|c| c.frustum(aspect));
    let bounds = world.get_resource::<MeshBounds>();
    let blends = world.get_resource::<MaterialBlends>();
    let mut total = 0usize;

    let sweep = crate::profile::scope("sweep");
    world
        .query::<(&WorldTransform, &MeshHandle, Option<&MaterialHandle>)>()
        .for_each(|entity, (transform, mesh, material)| {
            total += 1;
            let model = transform.0;

            let material = material.copied().unwrap_or(MaterialHandle(0));
            // A blended or refractive surface leaves no depth for anything to
            // test against, and a shadow map is nothing but depth — so it casts
            // none. Which also keeps a pane of glass out of the shadow views
            // entirely, and out of the prepass through `visible` below: nothing
            // screen-space may treat it as a surface.
            let mode = blends
                .as_ref()
                .map_or(BlendMode::Opaque, |table| table.get(material));

            // An opaque object's bounds are derived by the dispatch from its
            // mesh's own, so nothing on the CPU reads them; deriving them here
            // would be a 3x3 multiply per entity per frame for nothing.
            //
            // A mesh with no registered bounds is unmeasurable, so a blended one
            // is drawn rather than culled; the frustum test says the same of an
            // invalid box, which keeps the fallback in one place.
            let (world_bounds, visible) = if mode.is_opaque() {
                (Aabb::EMPTY, true)
            } else {
                let world_bounds = bounds
                    .as_ref()
                    .and_then(|table| table.get(*mesh))
                    .unwrap_or(Aabb::EMPTY)
                    .transformed(&model);
                let visible = !cull
                    || frustum
                        .as_ref()
                        .is_none_or(|frustum| frustum.intersects(&world_bounds));
                (world_bounds, visible)
            };

            // An object no list wants must not cost a row in `items` — the
            // object buffer is uploaded from this array and a row nothing
            // indexes is upload bandwidth for nothing.
            if !visible {
                return;
            }

            let index = out.items.len() as u32;
            out.items.push(RenderItem {
                model,
                prev_model: out.motion.record(entity, model),
                bounds: world_bounds,
                mesh: *mesh,
                material,
                // The entity's own slot, which is what makes the row persistent:
                // it is the same number next frame, so a row the GPU already
                // holds does not have to be written again.
                instance: entity.index(),
            });

            match mode {
                // One queue, deliberately. A cutout is opaque geometry that
                // happens to have holes in it: it writes depth, it is in the
                // prepass, and it is culled with everything else. What
                // separates it from `Opaque` is a pipeline, which each pass
                // picks per run off the material it is already reading — see
                // `BlendMode::is_masked`.
                BlendMode::Opaque | BlendMode::Masked => out.visible.push(index),
                BlendMode::Blend => out.transparent.push(index),
                BlendMode::Transmissive => out.refractive.push(index),
            }
        });

    drop(sweep);

    // Grouping by mesh lets the passes bind vertex/index buffers once per run
    // instead of once per draw, and collapse each run into a single instanced
    // draw. Not the opaque set, which `cull::CullPass::prepare` groups by a hash
    // rather than by a sort.
    let sorting = crate::profile::scope("sort");
    let key = |items: &[RenderItem], index: &u32| {
        let item = &items[*index as usize];
        (item.mesh.0, item.material.0)
    };
    // Grouped, and then deliberately left alone. Weighted-blended
    // transparency's whole claim is that the composite is commutative, so this
    // list has no back-to-front pass to go with it — and the popping a sort
    // produces when two surfaces cross is exactly what its absence buys.
    out.transparent.sort_unstable_by_key(|i| key(&out.items, i));
    out.refractive.sort_unstable_by_key(|i| key(&out.items, i));

    drop(sorting);

    // Back to front, and for correctness rather than speed: a refractive
    // surface samples the frame behind it, and the near one has to find the far
    // one already composited there. Runs move whole, so every run is still one
    // instanced draw.
    let ordering = crate::profile::scope("order runs");
    if let Some(camera) = camera.as_ref() {
        order_runs_front_to_back(&out.items, &mut out.refractive, camera.position);
        out.refractive.reverse();
    }

    drop(ordering);

    out.motion.end();

    if let Some(mut culling) = world.get_resource_mut::<Culling>() {
        // How much was visible is unknown here: `out.visible` is what the
        // dispatch was *offered*, not what it kept, and the counts it wrote are
        // still on the device when this runs.
        culling.record(None, total);
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

pub fn extract_lighting(world: &World, out: &mut SceneLighting) {
    let defaults = SceneLighting::default();
    out.ambient_color = defaults.ambient_color;
    out.ambient_nits = defaults.ambient_nits;
    out.sun = defaults.sun;
    out.shininess = defaults.shininess;
    out.specular_strength = defaults.specular_strength;
    out.point_lights.clear();
    out.spot_lights.clear();

    if let Some(ambient) = world.get_resource::<AmbientLight>() {
        out.ambient_color = ambient.color;
        out.ambient_nits = ambient.nits;
    }

    let mut has_sun = false;
    // World, not local: a light in a group lights from where the group puts
    // it. Callers propagate first, as they do for the geometry.
    world.query::<(&WorldTransform, &Light)>().for_each(
        |_entity, (transform, light)| match *light {
            Light::Directional { color, lux } => {
                // The shader supports one directional light, so the first wins.
                if !has_sun {
                    let direction = transform.transform_vector3(Vec3::NEG_Z).normalize_or_zero();
                    out.sun.direction = direction;
                    out.sun.color = color;
                    // Straight through: illuminance is already what a surface
                    // facing the light receives, which is the quantity the BRDF
                    // is multiplied by. Only the punctual lights below have a
                    // fixture to divide out.
                    out.sun.illuminance = lux;
                    has_sun = true;
                }
            }
            Light::Point {
                color,
                lumens,
                range,
                casts_shadows,
            } => {
                if out.point_lights.len() < MAX_POINT_LIGHTS {
                    out.point_lights.push(PointLight {
                        position: transform.translation(),
                        color,
                        candela: Light::point_candela(lumens),
                        range,
                        casts_shadows,
                    });
                }
            }
            Light::Spot {
                color,
                lumens,
                range,
                inner_angle,
                outer_angle,
                reflector,
                casts_shadows,
            } => {
                if out.spot_lights.len() < MAX_SPOT_LIGHTS {
                    // Cosines here rather than in the shader, and ordered so the
                    // inner cone is never outside the outer one — the falloff
                    // divides by their difference, and a cone authored inside
                    // out would otherwise light its own outside.
                    let outer = outer_angle.clamp(MIN_CONE_ANGLE, MAX_CONE_ANGLE);
                    let inner = inner_angle.clamp(0.0, outer);
                    out.spot_lights.push(SpotLight {
                        position: transform.translation(),
                        direction: transform.transform_vector3(Vec3::NEG_Z).normalize_or_zero(),
                        color,
                        // The *clamped* outer angle, so the solid angle the power
                        // is spread over is the one the shader's falloff then
                        // measures against.
                        candela: Light::spot_candela(lumens, outer, reflector),
                        range,
                        inner_cos: inner.to_radians().cos(),
                        outer_cos: outer.to_radians().cos(),
                        casts_shadows,
                    });
                }
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::{FrameGeometry, extract_decals, extract_geometry, extract_lighting};
    use crate::gfx::SceneLighting;
    use crate::gfx::{BlendMode, DecalInstance, MAX_DECALS, RenderItem};
    use crate::scene::Light;
    use crate::scene::propagate_transforms;
    use crate::scene::{
        Camera, CpuMesh, Culling, Decal, DecalSettings, LocalTransform, MaterialBlends,
        MaterialHandle, MeshBounds, MeshHandle, Transform, WorldTransform,
    };
    use glam::{Mat4, Quat, Vec3, Vec3A};
    use orrin_ecs::World;

    const ASPECT: f32 = 16.0 / 9.0;
    const CUBE: MeshHandle = MeshHandle(0);

    /// A world holding one unit cube's bounds, a camera at +Z looking at the
    /// origin, and culling on.
    ///
    /// Material 0 — what an entity with no `MaterialHandle` draws with — is
    /// blended here, because the blended queues are the ones culled on the CPU:
    /// opaque geometry is offered whole to the GPU dispatch, so the frustum
    /// test these cover is only ever made of blended objects.
    fn test_world() -> World {
        let mut world = World::new();
        let mut bounds = MeshBounds::default();
        bounds.insert(CUBE, CpuMesh::cube().bounds());
        world.insert_resource(bounds);
        world.insert_resource(Camera::default());
        world.insert_resource(Culling::default());
        let mut blends = MaterialBlends::default();
        blends.insert(MaterialHandle(0), BlendMode::Blend);
        world.insert_resource(blends);
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
        extract_geometry(world, ASPECT, &mut geometry);
        let kept = geometry.transparent();
        (0..kept.len()).map(|i| *kept.item(i)).collect()
    }

    /// A decal box, aimed straight down at the ground the way a scorch is.
    fn spawn_decal_at(world: &mut World, position: Vec3, size: f32, sort_order: i32) {
        world
            .spawn_entity()
            .with(LocalTransform::from(Transform {
                translation: position,
                rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                scale: Vec3::splat(size),
            }))
            .with(Decal {
                sort_order,
                ..Decal::default()
            });
    }

    fn decals(world: &mut World) -> Vec<DecalInstance> {
        propagate_transforms(world);
        let mut out = Vec::new();
        extract_decals(world, ASPECT, &mut out);
        out
    }

    /// The projection direction is the entity's forward, `-Z`, like every other
    /// direction this engine exposes — so the axis handed to the shader, which
    /// is `+Z`, points back *out* of the surface a decal is aimed at. Get this
    /// backwards and the angle fade keeps the decal on exactly the faces it is
    /// meant to reject.
    #[test]
    fn a_decal_projects_along_its_entitys_forward() {
        let mut world = test_world();
        spawn_decal_at(&mut world, Vec3::ZERO, 1.0, 0);

        let decal = decals(&mut world).remove(0);
        // Rotated a quarter turn back about X, forward points at the floor, so
        // the axis the shader compares a receiver's normal against is straight
        // up.
        assert!(
            decal.axes.z_axis.abs_diff_eq(Vec3::Y, 1e-5),
            "expected +Y, got {}",
            decal.axes.z_axis
        );
    }

    /// The inverse the shader tests against has to undo the scale as well as the
    /// placement, or a decal's box is the unit cube wherever it was authored and
    /// every decal in the scene is one metre across.
    #[test]
    fn the_box_is_the_transformed_unit_cube() {
        let mut world = test_world();
        spawn_decal_at(&mut world, Vec3::new(2.0, 0.0, 0.0), 4.0, 0);

        let decal = decals(&mut world).remove(0);
        let local = |p: Vec3| decal.world_to_decal.transform_point3(p);
        // The centre, and a point just inside the far face along the box's own
        // width — four metres wide means two metres either side.
        assert!(local(Vec3::new(2.0, 0.0, 0.0)).abs().max_element() < 1e-5);
        assert!(local(Vec3::new(3.9, 0.0, 0.0)).abs().max_element() < 0.5);
        assert!(local(Vec3::new(4.1, 0.0, 0.0)).abs().max_element() > 0.5);
    }

    /// Painting order is the author's, not the camera's. Deciding it by distance
    /// would make a stack of decals reshuffle itself as the camera walked around
    /// it, which is the one thing a painter's order must never do.
    #[test]
    fn decals_are_painted_in_their_authored_order() {
        let mut world = test_world();
        // The question is the ordering, not the culling, so nothing here has to
        // be on screen.
        world.resource_mut::<Culling>().enabled = false;
        // Spawned nearest-first with the orders reversed, so anything sorting by
        // distance produces the opposite of this.
        spawn_decal_at(&mut world, Vec3::new(0.0, 0.0, 4.0), 1.0, 7);
        spawn_decal_at(&mut world, Vec3::new(0.0, 0.0, 2.0), 1.0, 3);
        spawn_decal_at(&mut world, Vec3::new(0.0, 0.0, 0.0), 1.0, 5);

        let z: Vec<f32> = decals(&mut world)
            .iter()
            .map(|d| d.world_to_decal.inverse().w_axis.z.round())
            .collect();
        assert_eq!(z, vec![2.0, 0.0, 4.0]);
    }

    /// Over the cap, the survivors are the nearest — they are the ones covering
    /// pixels — and they are still painted in the authored order afterwards.
    #[test]
    fn the_cap_keeps_the_nearest_decals() {
        let mut world = test_world();
        world.insert_resource(Camera {
            position: Vec3::new(0.0, 0.0, 0.0),
            target: Vec3::new(0.0, 0.0, -1.0),
            ..Camera::default()
        });
        world.resource_mut::<Culling>().enabled = false;
        // Twice the cap, each one further away than the last.
        for i in 0..(MAX_DECALS * 2) {
            spawn_decal_at(&mut world, Vec3::new(0.0, 0.0, -(i as f32) - 1.0), 1.0, 0);
        }

        let out = decals(&mut world);
        assert_eq!(out.len(), MAX_DECALS);
        let furthest = out
            .iter()
            .map(|d| -d.world_to_decal.inverse().w_axis.z)
            .fold(0.0f32, f32::max);
        assert!(
            furthest <= MAX_DECALS as f32,
            "kept one {furthest} away, past the {MAX_DECALS} nearest"
        );
    }

    /// A decal scaled flat on an axis has no interior to project into and no
    /// invertible transform to project with. Dropped rather than uploaded, or
    /// the shader divides by a singular matrix.
    #[test]
    fn a_flattened_decal_is_dropped() {
        let mut world = test_world();
        world
            .spawn_entity()
            .with(LocalTransform::from(Transform {
                scale: Vec3::new(1.0, 1.0, 0.0),
                ..Default::default()
            }))
            .with(Decal::default());

        assert!(decals(&mut world).is_empty());
    }

    /// The A/B switch, and the reason it is one: a decal lands under the
    /// lighting, so it is indistinguishable from a surface that was always that
    /// colour until you can turn it off.
    #[test]
    fn switching_decals_off_extracts_none() {
        let mut world = test_world();
        spawn_decal_at(&mut world, Vec3::ZERO, 1.0, 0);
        assert_eq!(decals(&mut world).len(), 1);

        world.insert_resource(DecalSettings { enabled: false });
        assert!(decals(&mut world).is_empty());
    }

    /// The first extraction has nothing to compare against, so a brand-new
    /// object must read as stationary. Anything else and every spawn flashes a
    /// motion vector on the frame it appears, which TAA turns into a streak.
    #[test]
    fn an_object_seen_for_the_first_time_has_not_moved() {
        let mut world = test_world();
        spawn(&mut world, Vec3::ZERO, CUBE);

        let items = extract(&mut world);
        assert_eq!(items[0].prev_model, items[0].model);
    }

    /// Two extractions of the same geometry, with the object moved in between:
    /// the second must report where the first found it.
    #[test]
    fn a_moved_object_reports_where_it_was() {
        let mut world = test_world();
        spawn(&mut world, Vec3::ZERO, CUBE);

        let mut geometry = FrameGeometry::default();
        propagate_transforms(&mut world);
        extract_geometry(&world, ASPECT, &mut geometry);
        let before = *geometry.transparent().item(0);

        world
            .query::<&mut LocalTransform>()
            .for_each(|_entity, local| local.translation.x += 2.0);
        propagate_transforms(&mut world);
        extract_geometry(&world, ASPECT, &mut geometry);
        let after = *geometry.transparent().item(0);

        assert_ne!(after.model, before.model);
        assert_eq!(after.prev_model, before.model);
    }

    /// A slot outlives the entity in it. Reading the dead occupant's matrix as
    /// the new one's history would report a jump across the whole scene on the
    /// frame a slot is recycled — which is exactly when a spawn is most likely
    /// to be watched.
    #[test]
    fn a_recycled_entity_slot_starts_with_no_history() {
        let mut world = test_world();
        spawn(&mut world, Vec3::new(3.0, 0.0, 0.0), CUBE);

        let mut geometry = FrameGeometry::default();
        propagate_transforms(&mut world);
        extract_geometry(&world, ASPECT, &mut geometry);
        let first = *geometry.transparent().item(0);

        let mut existing = Vec::new();
        world
            .query::<&LocalTransform>()
            .for_each(|entity, _| existing.push(entity));
        world.despawn(existing[0]);
        spawn(&mut world, Vec3::ZERO, CUBE);
        propagate_transforms(&mut world);
        extract_geometry(&world, ASPECT, &mut geometry);
        let second = *geometry.transparent().item(0);

        assert_ne!(second.model, first.model);
        assert_eq!(second.prev_model, second.model);
    }

    #[test]
    fn an_object_behind_the_camera_is_culled() {
        let mut world = test_world();
        spawn(&mut world, Vec3::ZERO, CUBE);
        spawn(&mut world, Vec3::new(0.0, 0.0, 400.0), CUBE);

        let items = extract(&mut world);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].model.w_axis.truncate(), Vec3::ZERO);

        // What was visible is not the sweep's to say: the opaque queue is the
        // dispatch's, and its counts are still on the device. The total is.
        let culling = world.resource::<Culling>();
        assert_eq!(
            (culling.visible(), culling.total(), culling.culled()),
            (None, 2, None)
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

    #[test]
    fn a_mesh_reports_the_bounds_of_its_vertices() {
        let bounds = CpuMesh::cube().bounds();
        assert!((bounds.min - Vec3A::splat(-0.5)).length() < 1e-6);
        assert!((bounds.max - Vec3A::splat(0.5)).length() < 1e-6);

        let empty = CpuMesh::default().bounds();
        assert!(!empty.is_valid());
    }

    /// World bounds ride along on a blended item, which is what the frustum
    /// test is made against.
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
    /// the tests both build worlds a piece at a time. With no blend table every
    /// material is opaque, so the object lands in the set the dispatch is
    /// offered.
    #[test]
    fn a_world_without_a_camera_extracts_everything() {
        let mut world = World::new();
        spawn(&mut world, Vec3::new(0.0, 0.0, 400.0), CUBE);
        propagate_transforms(&mut world);

        let mut geometry = FrameGeometry::default();
        extract_geometry(&world, ASPECT, &mut geometry);
        assert_eq!(geometry.visible().len(), 1);
    }

    /// A light under a parent: the parent moved, turned or scaled, with the
    /// light at `local` beneath it, propagated as the frame propagates.
    fn parented_light(parent: Transform, local: Transform, light: Light) -> World {
        let mut world = World::new();
        let holder = world.spawn();
        world.insert(holder, LocalTransform::new(parent));
        let lamp = world.spawn();
        world.insert(lamp, LocalTransform::new(local));
        world.insert(lamp, light);
        crate::scene::reparent(&mut world, lamp, Some(holder), false).unwrap();
        propagate_transforms(&mut world);
        world
    }

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-4
    }

    /// A lamp in a moved group lights from where it stands in the world, not
    /// from its offset inside the group. Its range is its own: a light's reach
    /// is authored in metres, not in its parent's units.
    #[test]
    fn a_parented_point_light_lights_from_where_it_stands() {
        let parent = Transform {
            translation: Vec3::new(10.0, 0.0, 0.0),
            scale: Vec3::splat(2.0),
            ..Transform::default()
        };
        let world = parented_light(
            parent,
            Transform::from_translation(Vec3::new(0.0, 2.0, 0.0)),
            Light::point(Vec3::ONE, 800.0, 5.0),
        );
        let mut lighting = SceneLighting::default();
        extract_lighting(&world, &mut lighting);
        let lamp = &lighting.point_lights[0];
        assert!(
            close(lamp.position, Vec3::new(10.0, 4.0, 0.0)),
            "{}",
            lamp.position
        );
        assert_eq!(lamp.range, 5.0);
    }

    /// A spot shines along its forward as the hierarchy turns it.
    #[test]
    fn a_spot_under_a_turned_parent_shines_where_it_is_turned() {
        let parent = Transform {
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            ..Transform::default()
        };
        let world = parented_light(
            parent,
            Transform::from_translation(Vec3::new(0.0, 0.0, -3.0)),
            Light::spot(Vec3::ONE, 800.0, 5.0, 20.0, 30.0),
        );
        let mut lighting = SceneLighting::default();
        extract_lighting(&world, &mut lighting);
        let spot = &lighting.spot_lights[0];
        assert!(close(spot.direction, Vec3::NEG_X), "{}", spot.direction);
        assert!(
            close(spot.position, Vec3::new(-3.0, 0.0, 0.0)),
            "{}",
            spot.position
        );
    }

    #[test]
    fn a_parented_sun_takes_its_parent_s_turn() {
        let parent = Transform {
            rotation: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            ..Transform::default()
        };
        let world = parented_light(
            parent,
            Transform::default(),
            Light::directional(Vec3::ONE, 20_000.0),
        );
        let mut lighting = SceneLighting::default();
        extract_lighting(&world, &mut lighting);
        assert!(
            close(lighting.sun.direction, Vec3::NEG_Y),
            "{}",
            lighting.sun.direction
        );
    }
}

/// The half of extraction the camera-only tests above cannot reach: that one
/// sweep still answers both questions, and that reordering never breaks a run.
/// Every decal the camera can see, in the order they are to be painted.
///
/// Its own sweep rather than a branch inside `extract_geometry`, because a decal
/// is not a draw: it produces no `RenderItem`, enters no queue, and is read by
/// the passes as scene data the way the lights are. That is also why it lives
/// beside `extract_lighting` and takes the same shape.
///
/// Two orderings, and they answer different questions. Which decals *survive*
/// the cap is decided by distance, because when there are more than the frame
/// can carry, the ones nearest the camera are the ones covering pixels. Which
/// order the survivors are *painted* in is [`Decal::sort_order`], which is the
/// author's business and has nothing to do with where the camera happens to be —
/// deciding both by distance would make a stack of decals reshuffle itself as
/// the camera walked around it.
pub fn extract_decals(world: &World, aspect: f32, out: &mut Vec<DecalInstance>) {
    out.clear();
    if !world
        .get_resource::<DecalSettings>()
        .is_none_or(|settings| settings.enabled)
    {
        return;
    }

    let cull = world
        .get_resource::<Culling>()
        .is_none_or(|culling| culling.enabled);
    let camera = world.get_resource::<Camera>();
    let frustum = camera.as_ref().map(|c| c.frustum(aspect));
    let eye = camera.as_ref().map_or(Vec3::ZERO, |c| c.position);

    // The unit cube the decal's transform is applied to. The same box the shader
    // tests a fragment against, written once here so the culling and the
    // projection cannot disagree about how big a decal is.
    let unit = Aabb::from_points([Vec3::splat(-0.5), Vec3::splat(0.5)]);

    let mut found: Vec<(f32, DecalInstance, i32)> = Vec::new();
    world
        .query::<(&WorldTransform, &Decal)>()
        .for_each(|_entity, (transform, decal)| {
            let model = transform.0;
            if cull
                && frustum
                    .as_ref()
                    .is_some_and(|frustum| !frustum.intersects(&unit.transformed(&model)))
            {
                return;
            }

            // A singular transform has no interior to project into and no
            // inverse to project with — a decal scaled to zero on an axis is a
            // plane, and the containment test would divide by nothing.
            let world_to_decal = model.inverse();
            if !world_to_decal.is_finite() {
                return;
            }

            let basis = Mat3::from_mat4(model);
            let axes = Mat3::from_cols(
                basis.x_axis.normalize_or_zero(),
                basis.y_axis.normalize_or_zero(),
                basis.z_axis.normalize_or_zero(),
            );
            if axes.z_axis == Vec3::ZERO {
                return;
            }

            found.push((
                model.w_axis.truncate().distance_squared(eye),
                DecalInstance {
                    world_to_decal,
                    axes,
                    base_color: decal.base_color,
                    opacity: decal.opacity.clamp(0.0, 1.0),
                    albedo: decal.albedo,
                    normal: decal.normal,
                    metallic_roughness: decal.metallic_roughness,
                    normal_strength: decal.normal_strength.max(0.0),
                    metallic: decal.metallic.clamp(0.0, 1.0),
                    roughness: decal.roughness.clamp(0.0, 1.0),
                    affects_surface: decal.affects_surface,
                    // Clamped to the open half-turn: a fade angle of zero would
                    // want a cosine of one, where nothing but a surface exactly
                    // facing the projector survives and the decal is invisible
                    // for a reason nobody would guess from the number.
                    angle_cos: decal.angle.clamp(1.0, 179.0).to_radians().cos(),
                },
                decal.sort_order,
            ));
        });

    if found.len() > MAX_DECALS {
        found.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        found.truncate(MAX_DECALS);
    }
    found.sort_by_key(|(_, _, order)| *order);
    out.extend(found.into_iter().map(|(_, instance, _)| instance));
}

#[cfg(test)]
mod geometry_tests {
    use super::{FrameGeometry, extract_geometry};
    use crate::gfx::BlendMode;
    use crate::scene::{
        Camera, CpuMesh, Culling, LocalTransform, MaterialBlends, MaterialHandle, MeshBounds,
        MeshHandle, Transform, WorldTransform,
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

    /// An object no list wants must not reach `items` at all: the object buffer
    /// is uploaded from that array, so a row nothing indexes is dead bandwidth.
    /// Only a blended object can be unwanted — an opaque one is always offered.
    #[test]
    fn an_object_no_list_wants_costs_no_row() {
        // Far behind the camera.
        let mut world = world_with(&[(Vec3::new(0.0, 0.0, 5000.0), CUBE, 1)]);
        let mut blends = MaterialBlends::default();
        blends.insert(MaterialHandle(1), BlendMode::Blend);
        world.insert_resource(blends);
        let geometry = extract(&world);

        assert_eq!(geometry.transparent().len(), 0);
        assert_eq!(
            geometry.items.len(),
            0,
            "an object nothing draws still took a row",
        );
    }

    fn extract(world: &World) -> FrameGeometry {
        let mut geometry = FrameGeometry::default();
        extract_geometry(world, ASPECT, &mut geometry);
        geometry
    }

    /// The dispatch decides an opaque object's visibility from its mesh's own
    /// bounds, so the sweep offers it whatever the frustum says and derives no
    /// world bounds for it — nothing on the CPU reads them.
    #[test]
    fn an_opaque_object_is_offered_without_cpu_bounds() {
        // Behind the camera, where a frustum test would have culled it.
        let world = world_with(&[(Vec3::new(0.0, 0.0, 5000.0), CUBE, 0)]);
        let geometry = extract(&world);

        assert_eq!(geometry.visible.len(), 1, "the dispatch was not offered it");
        assert!(!geometry.items[0].bounds.is_valid());
    }

    /// The blended queues are culled and ordered on the CPU, so a blended
    /// object keeps real bounds and the frustum test.
    #[test]
    fn a_blended_object_is_culled_on_the_cpu() {
        let mut world = world_with(&[
            (Vec3::new(0.0, 0.0, -5.0), CUBE, 1),
            (Vec3::new(0.0, 0.0, 5000.0), CUBE, 1),
        ]);
        let mut blends = MaterialBlends::default();
        blends.insert(MaterialHandle(1), BlendMode::Blend);
        world.insert_resource(blends);
        let geometry = extract(&world);

        assert_eq!(
            geometry.transparent.len(),
            1,
            "the one behind was not culled"
        );
        assert!(
            geometry.items[geometry.transparent[0] as usize]
                .bounds
                .is_valid()
        );
    }

    /// A world of refractive objects, which is the queue ordered on the CPU.
    fn refractive_world(meshes: &[(Vec3, MeshHandle)]) -> World {
        let placed: Vec<_> = meshes.iter().map(|&(at, mesh)| (at, mesh, 2)).collect();
        let mut world = world_with(&placed);
        let mut blends = MaterialBlends::default();
        blends.insert(MaterialHandle(2), BlendMode::Transmissive);
        world.insert_resource(blends);
        world
    }

    /// Back-to-front ordering may reorder runs but must never split one: a run
    /// is a single instanced draw whose base is its start, so a broken run
    /// draws instances against the wrong transforms.
    #[test]
    fn back_to_front_ordering_keeps_every_run_whole() {
        // Two meshes interleaved in depth, so ordering has something to do and
        // the grouping has something to hold together.
        let world = refractive_world(&[
            (Vec3::new(0.0, 0.0, -2.0), CUBE),
            (Vec3::new(0.0, 0.0, -8.0), MeshHandle(1)),
            (Vec3::new(0.0, 0.0, -4.0), CUBE),
            (Vec3::new(0.0, 0.0, -6.0), MeshHandle(1)),
        ]);
        let geometry = extract(&world);
        let refractive = geometry.refractive();
        assert_eq!(refractive.len(), 4, "the camera should see all four");

        // One run per mesh, still — reordering moved runs, it did not split them.
        let runs: Vec<_> = refractive.runs().collect();
        assert_eq!(runs.len(), 2, "runs were split: {runs:?}");
        for run in &runs {
            let key = {
                let item = refractive.item(run.start);
                (item.mesh.0, item.material.0)
            };
            for i in run.clone() {
                let item = refractive.item(i);
                assert_eq!((item.mesh.0, item.material.0), key);
            }
        }
    }

    /// And that the order is back to front, which refraction needs for
    /// correctness: the farther run leads, so the nearer surface finds it
    /// already composited behind it.
    #[test]
    fn the_farther_refractive_run_is_drawn_first() {
        let camera = Camera::default();
        let world = refractive_world(&[
            (Vec3::new(0.0, 0.0, -2.0), CUBE),
            (Vec3::new(0.0, 0.0, -20.0), MeshHandle(1)),
        ]);
        let geometry = extract(&world);
        let refractive = geometry.refractive();
        assert_eq!(refractive.len(), 2);

        let first = refractive
            .item(0)
            .bounds
            .distance_squared_to(camera.position);
        let second = refractive
            .item(1)
            .bounds
            .distance_squared_to(camera.position);
        assert!(
            first >= second,
            "runs are front to back: {first} then {second}"
        );
    }
}
