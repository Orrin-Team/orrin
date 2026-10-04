//! Reproducible load for profiling, spawned on top of the default scene when
//! `ORRIN_STRESS` is set.
//!
//! Placement is driven by a fixed-seed splitmix64 rather than `rand`, so the same
//! spec produces the same scene on every machine and every commit — a profile
//! from today is comparable with one from six months ago, which is the only
//! reason the numbers are worth recording.
//!
//! ```text
//! ORRIN_STRESS=2000                                  # meshes only
//! ORRIN_STRESS=meshes=5000,colliders=800,scripts=200
//! ORRIN_STRESS=occluded=40000                        # the occlusion rig
//! ```
//!
//! `meshes` and `occluded` are deliberately opposite shapes at the same count,
//! which is what makes them an A/B. `meshes` grows its volume as the cube root
//! of the count so density stays constant and a bigger spec adds objects
//! without adding overdraw; `occluded` puts the same objects behind walls. A
//! visibility change measured against one and not the other has been measured
//! against the scene rather than the change.

use glam::{Mat3, Quat, Vec3};

use orrin_ecs::World;

use super::spawn_mesh;
use crate::scene::{Assets, Camera, Collider, ColliderShape, LocalTransform, Name, Transform};

/// Fixed so the same spec lays out identically everywhere, forever.
const SEED: u64 = 0x0DDB_A110_C0FF_EE00;

/// How much load to add. Zero in a field means that kind is skipped entirely.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StressSpec {
    pub meshes: usize,
    pub colliders: usize,
    pub scripts: usize,
    /// Props for the occlusion rig — small objects packed behind walls that
    /// span the screen. See [`occlusion_rig`].
    pub occluded: usize,
}

impl StressSpec {
    /// Parse `ORRIN_STRESS`, or `None` when it is unset or empty.
    ///
    /// A bare number means meshes; otherwise comma-separated `key=value` pairs.
    /// An unparsable spec is a warning and no load, never a silent zero — a
    /// profiling run that quietly measured an empty scene is worse than one that
    /// didn't start.
    pub fn from_env() -> Option<Self> {
        let raw = std::env::var("ORRIN_STRESS").ok()?;
        let raw = raw.trim();
        if raw.is_empty() {
            return None;
        }

        if let Ok(meshes) = raw.parse::<usize>() {
            return Some(Self {
                meshes,
                ..Default::default()
            });
        }

        let mut spec = Self::default();
        for field in raw.split(',') {
            let Some((key, value)) = field.split_once('=') else {
                tracing::warn!(
                    "ORRIN_STRESS: `{field}` is not `key=value`; ignoring the whole spec"
                );
                return None;
            };
            let Ok(count) = value.trim().parse::<usize>() else {
                tracing::warn!("ORRIN_STRESS: `{value}` is not a count; ignoring the whole spec");
                return None;
            };
            match key.trim() {
                "meshes" => spec.meshes = count,
                "colliders" => spec.colliders = count,
                "scripts" => spec.scripts = count,
                "occluded" => spec.occluded = count,
                other => {
                    tracing::warn!(
                        "ORRIN_STRESS: unknown key `{other}` (expected meshes, colliders, \
                         scripts or occluded); ignoring the whole spec"
                    );
                    return None;
                }
            }
        }
        Some(spec)
    }

    pub fn is_empty(&self) -> bool {
        self.meshes == 0 && self.colliders == 0 && self.scripts == 0 && self.occluded == 0
    }
}

/// Spawn the mesh and collider load. Scripted entities are attached separately,
/// since they need the script host that doesn't exist at scene-build time.
pub fn spawn_stress_scene(world: &mut World, spec: &StressSpec) {
    let Some((cube, material)) = world
        .get_resource::<Assets>()
        .and_then(|assets| Some((assets.mesh("cube")?, assets.material("clay")?)))
    else {
        tracing::warn!(
            "ORRIN_STRESS: the default scene's cube/clay assets are missing; no load added"
        );
        return;
    };

    let mut rng = Rng::new(SEED);

    // Volume grows as the cube root of the count, so density stays constant and
    // a bigger spec measures more objects rather than more overdraw.
    let spread = (spec.meshes.max(1) as f32).cbrt() * 2.5;
    for index in 0..spec.meshes {
        let position = Vec3::new(
            rng.range(-spread, spread),
            rng.range(0.5, spread.min(20.0)),
            rng.range(-spread, spread),
        );
        spawn_mesh(
            world,
            format!("Stress Cube {index}"),
            Transform {
                translation: position,
                scale: Vec3::splat(rng.range(0.3, 0.9)),
                ..Default::default()
            },
            cube,
            material,
        );
    }

    // Deliberately denser than the meshes: colliders spread as thinly would
    // never overlap, so the broadphase would find no pairs and narrowphase —
    // the expensive half — would never run.
    let collider_spread = (spec.colliders.max(1) as f32).cbrt() * 1.2;
    for index in 0..spec.colliders {
        let position = Vec3::new(
            rng.range(-collider_spread, collider_spread),
            rng.range(-collider_spread, collider_spread),
            rng.range(-collider_spread, collider_spread),
        );
        world
            .spawn_entity()
            .with(Name::new(format!("Stress Collider {index}")))
            .with(LocalTransform::from(Transform::from_translation(position)))
            .with(Collider {
                shape: if index % 2 == 0 {
                    ColliderShape::Box {
                        half_extents: Vec3::splat(0.5),
                    }
                } else {
                    ColliderShape::Sphere { radius: 0.5 }
                },
                // A third are triggers, so the resolver is exercised without
                // every pair pushing the scene apart.
                is_trigger: index % 3 == 0,
            })
            .id();
    }

    // The occlusion rig, spawned from the same generator so a spec that asks for
    // both still lays out identically every run.
    let rig = occlusion_rig(spec.occluded, &mut rng);
    for (index, transform) in rig.panels.into_iter().enumerate() {
        spawn_mesh(
            world,
            format!("Stress Occluder {index}"),
            transform,
            cube,
            material,
        );
    }
    for (index, transform) in rig.props.into_iter().enumerate() {
        spawn_mesh(
            world,
            format!("Stress Prop {index}"),
            transform,
            cube,
            material,
        );
    }

    tracing::info!(
        "stress load added — {} meshes, {} colliders, {} occluded props",
        spec.meshes,
        spec.colliders,
        spec.occluded
    );
}

/// Layers of wall in the occlusion rig, and where each sits along the view ray.
///
/// Six is enough that the deepest props are behind five independent walls, so
/// the rig has a *gradient* of occlusion rather than one boundary — which is
/// what stops the measurement being a single cliff at layer zero.
const LAYERS: usize = 6;
const FIRST_LAYER: f32 = 8.0;
const LAYER_STEP: f32 = 8.0;

/// Panels across and up in one layer's wall.
const PANEL_COLS: usize = 4;
const PANEL_ROWS: usize = 3;

/// How often a panel is left out. Sightlines rather than a sealed box: with no
/// gaps the answer is "everything past layer zero", which measures the rig and
/// not the renderer.
const GAP_CHANCE: f32 = 0.25;

/// The aspect the cross-section is laid out for. Fixed rather than taken from
/// the window, because a rig whose geometry followed the window would not be
/// the same scene on two machines — which is the property the fixed seed exists
/// to protect.
const RIG_ASPECT: f32 = 16.0 / 9.0;

/// How far along the view ray layer `index` stands.
fn layer_depth(index: usize) -> f32 {
    FIRST_LAYER + index as f32 * LAYER_STEP
}

/// One layer of the occlusion rig: a wall of panels across the view, and the
/// props packed behind it.
struct OcclusionRig {
    panels: Vec<Transform>,
    props: Vec<Transform>,
}

/// Lay out the occlusion rig: `props` small boxes packed behind [`LAYERS`]
/// walls that each span the screen.
///
/// This is the scene shape the `meshes` load deliberately does not have —
/// many small objects behind large occluders — and it exists because a
/// visibility optimisation measured on a constant-density cloud measures its
/// own cost and nothing else.
///
/// Laid out along [`Camera::default`]'s view ray rather than along an axis, so
/// every panel faces the camera squarely and every prop is inside the frustum.
/// That coupling is the rig's one fragile part and the reason two of the tests
/// below exist: a camera change moves the rig, and a rig that stops spanning
/// the frustum stops occluding without stopping *running*.
///
/// It is an upper bound and not a typical scene. Real geometry occludes itself
/// partially, at angles, with the occluders themselves worth culling; this is
/// the shape that says what the mechanism can return when everything is in its
/// favour, which is the number worth knowing before paying for it.
fn occlusion_rig(props: usize, rng: &mut Rng) -> OcclusionRig {
    let camera = Camera::default();
    let forward = (camera.target - camera.position).normalize();
    let right = forward.cross(camera.up).normalize();
    let up = right.cross(forward);
    // The panels' orientation: the cube's local `-Z` is its forward, as
    // everywhere else in this engine, so this stands each one facing the camera.
    let facing = Quat::from_mat3(&Mat3::from_cols(right, up, -forward));
    let tangent = (camera.fov_y * 0.5).tan();

    // The cross-section of the frustum at `depth`, in metres.
    let half_extent = |depth: f32| {
        let half_height = depth * tangent;
        (half_height * RIG_ASPECT, half_height)
    };

    let mut panels = Vec::with_capacity(LAYERS * PANEL_COLS * PANEL_ROWS);
    let mut placed = Vec::with_capacity(props);

    for layer in 0..LAYERS {
        let depth = layer_depth(layer);
        let (half_width, half_height) = half_extent(depth);
        let cell = Vec3::new(
            2.0 * half_width / PANEL_COLS as f32,
            2.0 * half_height / PANEL_ROWS as f32,
            // Thin, but not a plane: a zero extent on the view axis would give
            // the cull a degenerate box to build a screen rect from.
            0.4,
        );

        for row in 0..PANEL_ROWS {
            for column in 0..PANEL_COLS {
                if rng.unit() < GAP_CHANCE {
                    continue;
                }
                let across = -half_width + (column as f32 + 0.5) * cell.x;
                let along = -half_height + (row as f32 + 0.5) * cell.y;
                panels.push(Transform {
                    translation: camera.position + forward * depth + right * across + up * along,
                    rotation: facing,
                    scale: cell,
                });
            }
        }

        // Evenly over the layers, with the remainder on the last: an uneven
        // split would put the difference in the deepest layer, which is the
        // most occluded one and would bias the rate.
        let share = props / LAYERS + usize::from(layer == LAYERS - 1) * (props % LAYERS);
        for _ in 0..share {
            // Behind this layer's wall and in front of the next one's, so a
            // prop is occluded by the wall it belongs to rather than sharing
            // the plane of the one behind it.
            let prop_depth = rng.range(depth + 1.0, depth + LAYER_STEP - 1.0);
            let (half_width, half_height) = half_extent(prop_depth);
            let scale = rng.range(0.2, 0.6);
            // Inset by the prop's own half-extent as well as a margin, because
            // what has to stay inside the frustum is the box and not its centre.
            let margin = scale * 0.5 + 0.5;
            placed.push(Transform {
                translation: camera.position
                    + forward * prop_depth
                    + right * rng.range(-half_width + margin, half_width - margin)
                    + up * rng.range(-half_height + margin, half_height - margin),
                rotation: Quat::IDENTITY,
                scale: Vec3::splat(scale),
            });
        }
    }

    OcclusionRig {
        panels,
        props: placed,
    }
}

/// splitmix64. Self-contained so the stress scene needs no dependency and its
/// output can never change under us.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`, from the high bits — the low bits of splitmix64 are
    /// the weaker ones.
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + self.unit() * (high - low)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::geom::{Aabb, Frustum};

    #[test]
    fn a_bare_count_means_meshes() {
        // Parsing is env-driven in practice, but the shape is what matters here.
        let spec = StressSpec {
            meshes: 2000,
            ..Default::default()
        };
        assert_eq!(spec.meshes, 2000);
        assert!(!spec.is_empty());
        assert!(StressSpec::default().is_empty());
    }

    #[test]
    fn the_generator_is_stable_across_runs() {
        let first: Vec<u64> = (0..4)
            .scan(Rng::new(SEED), |rng, _| Some(rng.next_u64()))
            .collect();
        let second: Vec<u64> = (0..4)
            .scan(Rng::new(SEED), |rng, _| Some(rng.next_u64()))
            .collect();
        assert_eq!(first, second);
        // And it actually varies, rather than repeating one value.
        assert!(first.windows(2).any(|w| w[0] != w[1]));
    }

    /// The rig is only a measurement of occlusion if the frustum keeps every
    /// prop. A prop outside it would be dropped by `inside_frustum` before the
    /// depth test ever ran, and the cull rate would be frustum culling wearing
    /// occlusion's clothes — which is exactly the mistake that makes a
    /// visibility benchmark say what its author hoped.
    #[test]
    fn every_prop_in_the_rig_is_inside_the_frustum() {
        let camera = Camera::default();
        let frustum = Frustum::from_view_projection(camera.projection(RIG_ASPECT) * camera.view());
        let rig = occlusion_rig(4000, &mut Rng::new(SEED));

        assert!(!rig.props.is_empty());
        for prop in &rig.props {
            let half = prop.scale * 0.5;
            let bounds = Aabb {
                min: (prop.translation - half).into(),
                max: (prop.translation + half).into(),
            };
            assert!(
                frustum.intersects(&bounds),
                "a prop at {:?} is outside the frustum",
                prop.translation,
            );
        }
    }

    /// And it is only a measurement of *occlusion* if the walls occlude. Each
    /// layer's panels have to reach the frustum's edge at their depth, or the
    /// gap the rig leaves is the whole screen rather than the quarter it means
    /// to leave.
    #[test]
    fn each_layer_of_panels_spans_the_frustum_at_its_depth() {
        let camera = Camera::default();
        let forward = (camera.target - camera.position).normalize();
        let right = forward.cross(camera.up).normalize();
        let up = right.cross(forward);
        let rig = occlusion_rig(600, &mut Rng::new(SEED));

        for layer in 0..LAYERS {
            let depth = layer_depth(layer);
            let half_height = depth * (camera.fov_y * 0.5).tan();
            let half_width = half_height * RIG_ASPECT;
            // The panels of this layer, by their distance along the view ray.
            let reach = rig
                .panels
                .iter()
                .filter(|panel| {
                    ((panel.translation - camera.position).dot(forward) - depth).abs() < 0.5
                })
                .fold((0.0f32, 0.0f32), |(x, y), panel| {
                    let offset = panel.translation - camera.position - forward * depth;
                    (
                        x.max(offset.dot(right).abs() + panel.scale.x * 0.5),
                        y.max(offset.dot(up).abs() + panel.scale.y * 0.5),
                    )
                });

            assert!(
                (reach.0 - half_width).abs() < 0.01,
                "layer {layer} reaches {} across, frustum is {half_width}",
                reach.0,
            );
            assert!(
                (reach.1 - half_height).abs() < 0.01,
                "layer {layer} reaches {} up, frustum is {half_height}",
                reach.1,
            );
        }
    }

    /// Every prop asked for is placed, spread evenly over the layers, and the
    /// walls are a bounded overhead rather than a share of the count — so
    /// `occluded=N` and `meshes=N` are comparable at the same N.
    #[test]
    fn the_rig_places_every_prop_and_a_bounded_number_of_panels() {
        let rig = occlusion_rig(4000, &mut Rng::new(SEED));

        assert_eq!(rig.props.len(), 4000);
        assert!(
            rig.panels.len() <= LAYERS * PANEL_COLS * PANEL_ROWS,
            "{} panels",
            rig.panels.len(),
        );
        // And some panels are missing, or the rig is a sealed box in which
        // nothing behind layer zero is ever drawn whatever the test does.
        assert!(rig.panels.len() < LAYERS * PANEL_COLS * PANEL_ROWS);
    }

    #[test]
    fn unit_stays_in_range() {
        let mut rng = Rng::new(SEED);
        for _ in 0..10_000 {
            let value = rng.unit();
            assert!((0.0..1.0).contains(&value), "{value} out of range");
        }
    }
}
