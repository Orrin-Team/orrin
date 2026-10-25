use glam::{Mat3, Mat4, Vec3, Vec3A};

use orrin_ecs::{Entity, World};

use crate::geom::Aabb;
use crate::gfx::punctual::{MAX_SHADOW_LIGHTS, ShadowAtlas};
use crate::gfx::shadows::{Cascade, CascadeSet, MAX_CASCADES};
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
/// `RenderItem` is 144 bytes and an object visible to the camera is usually also
/// a caster in two or three cascades, so materialising each list separately
/// would copy — and then sort — the same matrices five times over. Indices make
/// widening a list a four-byte push and each sort a scan of `u32`s.
#[derive(Default)]
pub struct FrameGeometry {
    /// Every renderable in the world, with its model, bounds and
    /// inverse-transpose derived exactly once this frame.
    items: Vec<RenderItem>,
    /// What the camera frustum kept and shades opaquely, in draw order.
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
    /// What casts into each active cascade, in draw order.
    cascades: [Vec<u32>; MAX_CASCADES],
    /// What casts for each punctual light that got atlas tiles, indexed like
    /// `ShadowAtlas::casters`. One list per *light* rather than per face: a
    /// point light's six frustums share a reach, and testing them apart costs
    /// more than the draws it would save at these ranges.
    punctual: [Vec<u32>; MAX_SHADOW_LIGHTS],
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

    /// What cascade `index` draws. Empty for a cascade the last extraction did
    /// not fill, which is what shadows-off produces.
    pub fn cascade(&self, index: usize) -> DrawList<'_> {
        DrawList::new(&self.items, &self.cascades[index])
    }

    /// What punctual caster `index` draws into every one of its faces. Empty for
    /// a slot the atlas did not fill.
    pub fn punctual(&self, index: usize) -> DrawList<'_> {
        DrawList::new(&self.items, &self.punctual[index])
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
    atlas: &ShadowAtlas,
    gpu_culling: bool,
    out: &mut FrameGeometry,
) {
    out.items.clear();
    out.visible.clear();
    out.transparent.clear();
    out.refractive.clear();
    for list in out.cascades.iter_mut() {
        list.clear();
    }
    for list in out.punctual.iter_mut() {
        list.clear();
    }
    out.motion.begin();

    let cull = world
        .get_resource::<Culling>()
        .is_none_or(|culling| culling.enabled);
    let camera = world.get_resource::<Camera>();
    let frustum = camera.as_ref().map(|c| c.frustum(aspect));
    let bounds = world.get_resource::<MeshBounds>();
    let blends = world.get_resource::<MaterialBlends>();
    let active = &cascades.cascades[..cascades.count];
    // Truncated rather than asserted: the atlas cannot produce more than
    // `MAX_SHADOW_LIGHTS` casters, and a slice keeps that from being something
    // this loop has to believe.
    let punctual = &atlas.casters[..atlas.casters.len().min(MAX_SHADOW_LIGHTS)];
    let mut total = 0usize;

    let sweep = crate::profile::scope("sweep");
    world
        .query::<(&WorldTransform, &MeshHandle, Option<&MaterialHandle>)>()
        .for_each(|entity, (transform, mesh, material)| {
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

            let material = material.copied().unwrap_or(MaterialHandle(0));
            // A blended or refractive surface leaves no depth for anything to
            // test against, and a shadow map is nothing but depth — so it casts
            // none. Which also keeps a pane of glass out of the caster lists
            // entirely, and out of the prepass through `visible` below: nothing
            // screen-space may treat it as a surface.
            let mode = blends
                .as_ref()
                .map_or(BlendMode::Opaque, |table| table.get(material));
            let blended = !mode.is_opaque();

            let in_frustum = !cull
                || frustum
                    .as_ref()
                    .is_none_or(|frustum| frustum.intersects(&world_bounds));
            // Under GPU culling every *opaque* renderable is offered to the
            // dispatch and the frustum decides there, per view — this is the
            // test the whole feature exists to move. The blended and refractive
            // queues are culled here either way: they are drawn from the order
            // this list is in, and a refractive surface has to be drawn back to
            // front, which is a correctness requirement no atomic append meets.
            let visible = if gpu_culling && !blended {
                true
            } else {
                in_frustum
            };
            // The cascades share a rotation, so a box's half-extent in light
            // space is a property of the box and the light — not of which
            // cascade is asking. Derived once here; each cascade then costs one
            // point transform and five compares instead of a second abs-matrix
            // build and 3x3 multiply.
            //
            // An unmeasurable box casts nothing, which is what the old
            // per-cascade `Aabb::transformed` answered by returning `EMPTY` and
            // failing every compare. Stated rather than inferred, so the answer
            // does not rest on how infinities compare.
            let casts: [bool; MAX_CASCADES] =
                if gpu_culling || blended || active.is_empty() || !world_bounds.is_valid() {
                    [false; MAX_CASCADES]
                } else {
                    let center = world_bounds.center();
                    let ls_extents = cascades.abs_light_rotation * world_bounds.half_extents();
                    std::array::from_fn(|i| {
                        active
                            .get(i)
                            .is_some_and(|cascade| casts_into(center, ls_extents, cascade))
                    })
                };
            // A punctual light reaches a sphere, so anything outside it cannot
            // shadow anything it lights. Distance to the *box* rather than to
            // its centre, or a long wall through a light's volume would be
            // culled out of the shadow it plainly casts.
            let lights: [bool; MAX_SHADOW_LIGHTS] = std::array::from_fn(|i| {
                !gpu_culling
                    && !blended
                    && punctual.get(i).is_some_and(|caster| {
                        world_bounds.distance_squared_to(caster.center)
                            <= caster.radius * caster.radius
                    })
            });

            // An object no list wants still costs the sweep, but it must not
            // cost a row in `items` — the object buffer is uploaded from this
            // array and a row nothing indexes is upload bandwidth for nothing.
            if !visible && !casts.iter().any(|&c| c) && !lights.iter().any(|&c| c) {
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

            if visible {
                match mode {
                    // One queue, deliberately. A cutout is opaque geometry that
                    // happens to have holes in it: it writes depth, it is in the
                    // prepass, and it belongs in the same front-to-back order
                    // everything else is in. What separates it from `Opaque` is
                    // a pipeline, which each pass picks per run off the material
                    // it is already reading — see `BlendMode::is_masked`.
                    BlendMode::Opaque | BlendMode::Masked => out.visible.push(index),
                    BlendMode::Blend => out.transparent.push(index),
                    BlendMode::Transmissive => out.refractive.push(index),
                }
            }
            for (list, casts) in out.cascades.iter_mut().zip(casts) {
                if casts {
                    list.push(index);
                }
            }
            for (list, casts) in out.punctual.iter_mut().zip(lights) {
                if casts {
                    list.push(index);
                }
            }
        });

    drop(sweep);

    // Grouping by mesh lets the passes bind vertex/index buffers once per run
    // instead of once per draw, and collapse each run into a single instanced
    // draw.
    let sorting = crate::profile::scope("sort");
    let key = |items: &[RenderItem], index: &u32| {
        let item = &items[*index as usize];
        (item.mesh.0, item.material.0)
    };
    // Under GPU culling the opaque list is not a draw order at all — it is the
    // set the dispatch is offered, and `cull::CullPass::prepare` groups it by a
    // hash rather than by a sort. Sorting it would be an O(n log n) walk over
    // every renderable to produce an ordering nothing reads.
    if !gpu_culling {
        out.visible.sort_unstable_by_key(|i| key(&out.items, i));
    }
    // Grouped, and then deliberately left alone. Weighted-blended
    // transparency's whole claim is that the composite is commutative, so this
    // list has no back-to-front pass to go with it — and the popping a sort
    // produces when two surfaces cross is exactly what its absence buys.
    out.transparent.sort_unstable_by_key(|i| key(&out.items, i));
    out.refractive.sort_unstable_by_key(|i| key(&out.items, i));
    for list in out.cascades.iter_mut() {
        list.sort_unstable_by_key(|i| key(&out.items, i));
    }
    for list in out.punctual.iter_mut() {
        list.sort_unstable_by_key(|i| key(&out.items, i));
    }

    drop(sorting);

    // Within the grouping, order the runs front to back. Opaque geometry is
    // depth-tested, so this changes no pixel — it changes how many fragments
    // reach the shader, since a nearer run already in the depth buffer rejects
    // the ones behind it before they shade. Runs move whole, so every run is
    // still one instanced draw.
    let ordering = crate::profile::scope("order runs");
    if let Some(camera) = camera.as_ref() {
        // Not under GPU culling, for the reason the sort above is skipped: the
        // opaque list is a set there, and the order the batches are drawn in is
        // the order `cull::CullPass::prepare` assigned their numbers.
        if !gpu_culling {
            order_runs_front_to_back(&out.items, &mut out.visible, camera.position);
        }
        // The opposite order, and for the opposite reason. Front to back above
        // is an optimisation over geometry the depth test already resolves; back
        // to front here is *correctness*, because a refractive surface samples
        // the frame behind it and the near one has to find the far one already
        // composited there.
        order_runs_front_to_back(&out.items, &mut out.refractive, camera.position);
        out.refractive.reverse();
    }

    drop(ordering);

    out.motion.end();

    if let Some(mut culling) = world.get_resource_mut::<Culling>() {
        // Unknown under GPU culling: `out.visible` there is what the dispatch
        // was *offered*, not what it kept, and the counts it wrote are still on
        // the device when this runs.
        let visible = (!gpu_culling)
            .then(|| out.visible.len() + out.transparent.len() + out.refractive.len());
        culling.record(visible, total);
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

/// Whether a box, extended infinitely toward the light, reaches `cascade`.
///
/// Takes the box already split into the two parts that behave differently across
/// a cascade set: `center` in world space, which each cascade's `light_view`
/// must transform for itself, and `ls_extents`, its half-extent in light space,
/// which every cascade agrees on — that is
/// [`CascadeSet::abs_light_rotation`](crate::gfx::shadows::CascadeSet::abs_light_rotation)
/// times the box's half-extents, and the reason this is not `&Aabb`.
fn casts_into(center: Vec3A, ls_extents: Vec3A, cascade: &Cascade) -> bool {
    let c = cascade.light_view.transform_point3a(center);
    let half = cascade.half_extent;
    // `light_view` is a right-handed look-at, so what the pass renders lies at
    // negative z, between the eye at 0 and the far plane at -depth_range.
    c.x - ls_extents.x <= half
        && c.x + ls_extents.x >= -half
        && c.y - ls_extents.y <= half
        && c.y + ls_extents.y >= -half
        && c.z + ls_extents.z >= -cascade.depth_range
}

#[cfg(test)]
mod shadow_culling_tests {
    use super::*;
    use crate::gfx::shadows::{CascadeConfig, cascades};
    use crate::scene::Camera;
    use glam::{Vec3, Vec3A};

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
        let center = Vec3A::from(center);
        Aabb {
            min: center - Vec3A::splat(half),
            max: center + Vec3A::splat(half),
        }
    }

    /// What the sweep does per entity, in one call: split the box into the
    /// centre each cascade transforms and the extent they all share.
    fn casts(set: &CascadeSet, bounds: &Aabb, index: usize) -> bool {
        super::casts_into(
            bounds.center(),
            set.abs_light_rotation * bounds.half_extents(),
            &set.cascades[index],
        )
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
            casts(&set, &box_at(overhead, 1.0), 0),
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

        assert!(!casts(&set, &box_at(below, 1.0), 0));
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
                !casts(&set, &box_at(outside, 1.0), 0),
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

        assert!(casts(&set, &bounds, 0));
        for index in 1..set.count {
            assert!(
                casts(&set, &bounds, index),
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

    /// The predicate this replaced: a full `Aabb::transformed` per cascade,
    /// kept here as the thing the hoisted form has to keep agreeing with.
    fn reference_casts_into(bounds: &Aabb, cascade: &Cascade) -> bool {
        let ls = bounds.transformed(&cascade.light_view);
        ls.min.x <= cascade.half_extent
            && ls.max.x >= -cascade.half_extent
            && ls.min.y <= cascade.half_extent
            && ls.max.y >= -cascade.half_extent
            && ls.max.z >= -cascade.depth_range
    }

    /// Hoisting the rotation is only sound if it computes the same predicate.
    ///
    /// Swept over a grid that straddles every plane of every cascade — inside,
    /// outside, and across each edge — because a test that only samples the
    /// interior would pass for a predicate that got the bounds wrong. The sun is
    /// off every axis, so the shared rotation is a general one rather than the
    /// permutation a straight-down light produces.
    #[test]
    fn the_hoisted_extents_agree_with_a_full_transform_per_cascade() {
        let camera = Camera::default();
        let set = cascades(
            &camera,
            ASPECT,
            Vec3::new(-0.4, -1.0, -0.3).normalize(),
            &config(),
        );
        let reach = set.cascades[set.count - 1].half_extent * 1.5;

        let mut tested = 0usize;
        let mut kept = [0usize; MAX_CASCADES];
        let mut reference_kept = [0usize; MAX_CASCADES];
        let steps = 11;
        for xi in 0..steps {
            for yi in 0..steps {
                for zi in 0..steps {
                    let at = |i: usize| (i as f32 / (steps - 1) as f32 * 2.0 - 1.0) * reach;
                    // Sizes that are not all the same, so the extent term is
                    // doing real work rather than cancelling out.
                    let half = 0.5 + (xi + yi + zi) as f32 * 0.25;
                    let bounds = box_at(Vec3::new(at(xi), at(yi), at(zi)), half);
                    tested += 1;
                    for index in 0..set.count {
                        let hoisted = casts(&set, &bounds, index);
                        let reference = reference_casts_into(&bounds, &set.cascades[index]);
                        assert_eq!(
                            hoisted,
                            reference,
                            "cascade {index} disagrees about a box of half-extent {half} at \
                             {:?}",
                            bounds.center(),
                        );
                        kept[index] += usize::from(hoisted);
                        reference_kept[index] += usize::from(reference);
                    }
                }
            }
        }

        assert_eq!(kept, reference_kept, "per-cascade caster counts differ");
        // A sweep where every box is kept, or none is, would agree trivially.
        for (index, &count) in kept.iter().take(set.count).enumerate() {
            assert!(
                count > 0 && count < tested,
                "cascade {index} kept {count} of {tested}: the sweep straddles nothing",
            );
        }
    }
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
    world.query::<(&LocalTransform, &Light)>().for_each(
        |_entity, (transform, light)| match *light {
            Light::Directional { color, lux } => {
                // The shader supports one directional light, so the first wins.
                if !has_sun {
                    let direction = (transform.rotation * Vec3::NEG_Z).normalize_or_zero();
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
                        position: transform.translation,
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
                        position: transform.translation,
                        direction: (transform.rotation * Vec3::NEG_Z).normalize_or_zero(),
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
    use super::{FrameGeometry, extract_decals, extract_geometry};
    use crate::gfx::punctual::ShadowAtlas;
    use crate::gfx::shadows::CascadeSet;
    use crate::gfx::{DecalInstance, MAX_DECALS, RenderItem};
    use crate::scene::propagate_transforms;
    use crate::scene::{
        Camera, CpuMesh, Culling, Decal, DecalSettings, LocalTransform, MeshBounds, MeshHandle,
        Transform, WorldTransform,
    };
    use glam::{Mat4, Quat, Vec3, Vec3A};
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
        extract_geometry(
            world,
            ASPECT,
            &CascadeSet::default(),
            &ShadowAtlas::default(),
            false,
            &mut geometry,
        );
        let visible = geometry.visible();
        (0..visible.len()).map(|i| *visible.item(i)).collect()
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
        extract_geometry(
            &world,
            ASPECT,
            &CascadeSet::default(),
            &ShadowAtlas::default(),
            false,
            &mut geometry,
        );
        let before = *geometry.visible().item(0);

        world
            .query::<&mut LocalTransform>()
            .for_each(|_entity, local| local.translation.x += 2.0);
        propagate_transforms(&mut world);
        extract_geometry(
            &world,
            ASPECT,
            &CascadeSet::default(),
            &ShadowAtlas::default(),
            false,
            &mut geometry,
        );
        let after = *geometry.visible().item(0);

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
        extract_geometry(
            &world,
            ASPECT,
            &CascadeSet::default(),
            &ShadowAtlas::default(),
            false,
            &mut geometry,
        );
        let first = *geometry.visible().item(0);

        let mut existing = Vec::new();
        world
            .query::<&LocalTransform>()
            .for_each(|entity, _| existing.push(entity));
        world.despawn(existing[0]);
        spawn(&mut world, Vec3::ZERO, CUBE);
        propagate_transforms(&mut world);
        extract_geometry(
            &world,
            ASPECT,
            &CascadeSet::default(),
            &ShadowAtlas::default(),
            false,
            &mut geometry,
        );
        let second = *geometry.visible().item(0);

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

        let culling = world.resource::<Culling>();
        assert_eq!(
            (culling.visible(), culling.total(), culling.culled()),
            (Some(1), 2, Some(1))
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
    use crate::gfx::punctual::ShadowAtlas;
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
        extract_geometry(
            &world,
            ASPECT,
            &set,
            &ShadowAtlas::default(),
            false,
            &mut geometry,
        );

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
        extract_geometry(
            &world,
            ASPECT,
            &set,
            &ShadowAtlas::default(),
            false,
            &mut geometry,
        );

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
        extract_geometry(
            &world,
            ASPECT,
            &set,
            &ShadowAtlas::default(),
            false,
            &mut geometry,
        );

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
        // Two meshes interleaved in depth, so ordering has something to do and
        // the grouping has something to hold together.
        let world = world_with(&[
            (Vec3::new(0.0, 0.0, -2.0), CUBE, 0),
            (Vec3::new(0.0, 0.0, -8.0), MeshHandle(1), 0),
            (Vec3::new(0.0, 0.0, -4.0), CUBE, 0),
            (Vec3::new(0.0, 0.0, -6.0), MeshHandle(1), 0),
        ]);

        let mut geometry = FrameGeometry::default();
        extract_geometry(
            &world,
            ASPECT,
            &CascadeSet::default(),
            &ShadowAtlas::default(),
            false,
            &mut geometry,
        );
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
        extract_geometry(
            &world,
            ASPECT,
            &CascadeSet::default(),
            &ShadowAtlas::default(),
            false,
            &mut geometry,
        );
        let visible = geometry.visible();
        assert_eq!(visible.len(), 2);

        let near = visible.item(0).bounds.distance_squared_to(camera.position);
        let far = visible.item(1).bounds.distance_squared_to(camera.position);
        assert!(near <= far, "runs are back to front: {near} then {far}");
    }
}
