mod library;
mod scene;
mod showcase;
mod sponza;
mod stress;
mod textures;

pub use scene::build_default_scene;
pub use showcase::build_showcase_scene;
pub use sponza::build_sponza_scene;
pub use stress::{StressSpec, spawn_stress_scene};

use glam::{Quat, Vec3};

use orrin_ecs::{Entity, World};

use crate::gfx::RenderBackend;
use crate::scene::{Decal, Light, LocalTransform, MaterialHandle, MeshHandle, Name, Transform};

/// Which built-in scene a run opens with.
///
/// Three scenes rather than one, because the jobs pull in different directions:
/// the rig separates variables, the courtyard composes them, and Sponza is
/// neither — it is somebody else's geometry at somebody else's complexity, which
/// is the only one of the three that can say whether this renderer handles a
/// scene it did not author. Trying to make one scene do two of these costs the
/// A/B captures their meaning — every offscreen capture that frames a specific
/// wall at a specific angle is a claim about a feature, and it stops being one
/// the moment the wall moves for a nicer silhouette.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SceneChoice {
    /// [`build_default_scene`]: the feature rig, and what every capture but the
    /// showcase pair is taken of.
    #[default]
    Demo,
    /// [`build_showcase_scene`]: the ruined courtyard.
    Showcase,
    /// [`build_sponza_scene`]: Crytek's Sponza atrium, imported from glTF. Needs
    /// the model on disk — see `scripts/fetch-sponza.sh`.
    Sponza,
}

impl SceneChoice {
    /// Read `ORRIN_SCENE`, falling back to the rig.
    ///
    /// An unrecognised name is a warning and the default, never a silent empty
    /// world — the same rule `ORRIN_STRESS` follows, and for the same reason: a
    /// typo that quietly changed what you were looking at is worse than one that
    /// says so.
    pub fn from_env() -> Self {
        match std::env::var("ORRIN_SCENE") {
            Err(_) => Self::default(),
            Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
                "" | "demo" | "rig" | "default" => Self::Demo,
                "showcase" | "courtyard" => Self::Showcase,
                "sponza" | "crytek" => Self::Sponza,
                other => {
                    tracing::warn!(
                        "ORRIN_SCENE: unknown scene `{other}` (expected demo, showcase or \
                         sponza); opening the demo scene"
                    );
                    Self::Demo
                }
            },
        }
    }

    pub fn build(self, world: &mut World, backend: &mut impl RenderBackend) {
        match self {
            Self::Demo => build_default_scene(world, backend),
            Self::Showcase => build_showcase_scene(world, backend),
            Self::Sponza => build_sponza_scene(world, backend),
        }
    }

    /// For the run banner. A frame time from the courtyard and one from the rig
    /// are not comparable, so which scene produced it belongs in the line that
    /// describes the run.
    pub fn label(self) -> &'static str {
        match self {
            Self::Demo => "demo",
            Self::Showcase => "showcase",
            Self::Sponza => "sponza",
        }
    }
}

pub fn spawn_mesh(
    world: &mut World,
    name: impl Into<String>,
    transform: Transform,
    mesh: MeshHandle,
    material: MaterialHandle,
) -> Entity {
    world
        .spawn_entity()
        .with(Name::new(name))
        .with(LocalTransform::from(transform))
        .with(mesh)
        .with(material)
        .id()
}

/// Spawn a projected decal.
///
/// `transform` *is* the projection: the box is the unit cube under it, so the
/// scale is the decal's extent in metres and the rotation aims it — along the
/// entity's forward, which is `-Z`, as everywhere else in this engine. Nothing
/// else is needed, and in particular there is no mesh and no material: a decal
/// is read by the passes that draw whatever it lands on.
pub fn spawn_decal(
    world: &mut World,
    name: impl Into<String>,
    transform: Transform,
    decal: Decal,
) -> Entity {
    world
        .spawn_entity()
        .with(Name::new(name))
        .with(LocalTransform::from(transform))
        .with(decal)
        .id()
}

/// `lumens` is the fixture's total luminous power, the number a bulb's box
/// prints: 800 for a domestic bulb, a few thousand for a shop light.
pub fn spawn_point_light(
    world: &mut World,
    name: impl Into<String>,
    position: Vec3,
    color: Vec3,
    lumens: f32,
    range: f32,
) -> Entity {
    world
        .spawn_entity()
        .with(Name::new(name))
        .with(LocalTransform::from(Transform::from_translation(position)))
        .with(Light::point(color, lumens, range))
        .id()
}

/// Like a point light, but coned. The axis is the entity's forward, `-Z`, the
/// same convention `spawn_directional_light` uses, and the angles are half
/// angles from that axis in degrees.
///
/// `lumens` is the same quantity a point light takes, and the spot gets a
/// reflector: the cone concentrates that power rather than masking it, so
/// narrowing `outer_angle` brightens the beam.
#[allow(
    clippy::too_many_arguments,
    reason = "each argument is one physical quantity of the light"
)]
pub fn spawn_spot_light(
    world: &mut World,
    name: impl Into<String>,
    position: Vec3,
    direction: Vec3,
    color: Vec3,
    lumens: f32,
    range: f32,
    inner_angle: f32,
    outer_angle: f32,
) -> Entity {
    let rotation = Quat::from_rotation_arc(Vec3::NEG_Z, direction.normalize_or_zero());
    world
        .spawn_entity()
        .with(Name::new(name))
        .with(LocalTransform::from(Transform {
            translation: position,
            rotation,
            ..Default::default()
        }))
        .with(Light::spot(color, lumens, range, inner_angle, outer_angle))
        .id()
}

/// The direction is stored as the entity's rotation (forward = `-Z`), so it can
/// be reoriented like any other transform.
///
/// `lux` is the illuminance the light lays on a surface facing it: about
/// 100 000 for noon sun, 20 000 for an overcast day, under 1 000 near sunset.
pub fn spawn_directional_light(
    world: &mut World,
    name: impl Into<String>,
    direction: Vec3,
    color: Vec3,
    lux: f32,
) -> Entity {
    let rotation = Quat::from_rotation_arc(Vec3::NEG_Z, direction.normalize_or_zero());
    world
        .spawn_entity()
        .with(Name::new(name))
        .with(LocalTransform::from(Transform {
            rotation,
            ..Default::default()
        }))
        .with(Light::directional(color, lux))
        .id()
}
