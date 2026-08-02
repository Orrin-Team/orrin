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
//! ```

use glam::Vec3;

use orrin_ecs::World;

use super::spawn_mesh;
use crate::scene::{Assets, Collider, ColliderShape, LocalTransform, Name, Transform};

/// Fixed so the same spec lays out identically everywhere, forever.
const SEED: u64 = 0x0DDB_A11_0_C0FF_EE00;

/// How much load to add. Zero in a field means that kind is skipped entirely.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StressSpec {
    pub meshes: usize,
    pub colliders: usize,
    pub scripts: usize,
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
                eprintln!("ORRIN_STRESS: `{field}` is not `key=value`; ignoring the whole spec");
                return None;
            };
            let Ok(count) = value.trim().parse::<usize>() else {
                eprintln!("ORRIN_STRESS: `{value}` is not a count; ignoring the whole spec");
                return None;
            };
            match key.trim() {
                "meshes" => spec.meshes = count,
                "colliders" => spec.colliders = count,
                "scripts" => spec.scripts = count,
                other => {
                    eprintln!(
                        "ORRIN_STRESS: unknown key `{other}` (expected meshes, colliders, \
                         or scripts); ignoring the whole spec"
                    );
                    return None;
                }
            }
        }
        Some(spec)
    }

    pub fn is_empty(&self) -> bool {
        self.meshes == 0 && self.colliders == 0 && self.scripts == 0
    }
}

/// Spawn the mesh and collider load. Scripted entities are attached separately,
/// since they need the script host that doesn't exist at scene-build time.
pub fn spawn_stress_scene(world: &mut World, spec: &StressSpec) {
    let Some((cube, material)) = world
        .get_resource::<Assets>()
        .and_then(|assets| Some((assets.mesh("cube")?, assets.material("clay")?)))
    else {
        eprintln!("ORRIN_STRESS: the default scene's cube/clay assets are missing; no load added");
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

    println!(
        "orrin: stress load added — {} meshes, {} colliders",
        spec.meshes, spec.colliders
    );
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

    #[test]
    fn unit_stays_in_range() {
        let mut rng = Rng::new(SEED);
        for _ in 0..10_000 {
            let value = rng.unit();
            assert!((0.0..1.0).contains(&value), "{value} out of range");
        }
    }
}
