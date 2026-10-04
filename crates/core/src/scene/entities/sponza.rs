//! Crytek's Sponza atrium, imported from glTF.
//!
//! The other two scenes are this engine's own geometry, and that is their
//! limitation: a rig and a courtyard built out of `cube`, `plane` and `sphere`
//! can only ever ask questions the author already knew to ask. Sponza is the
//! opposite — a quarter of a million triangles of somebody else's building, with
//! somebody else's UV layout, material count and texture budget — so it is the
//! only scene here that can find out what this renderer does with a scene it did
//! not author. It has been the industry's reference atrium since 2010 for exactly
//! that reason, which also means a frame of it is comparable with everyone
//! else's.
//!
//! # What it does *not* do
//!
//! It loads none of the demo library. Nothing here is a variable-isolating A/B —
//! the geometry is fixed and the materials are the file's — so the procedural
//! brick, wax and glass would be objects sitting in someone's building for no
//! reason. It also keeps the frame inside
//! [`MAX_TEXTURES`](crate::gfx::MAX_TEXTURES): Sponza brings its own two dozen
//! maps, and the library brings fourteen more.
//!
//! # Everything spatial here is measured, not authored
//!
//! The camera, the sun's azimuth and the shadow-relevant distances are derived
//! from [`Model::bounds`] rather than written down. That is not tidiness: this
//! scene's geometry lives in a file that is not in the repository, so a hard-coded
//! camera is a claim about a copy of a download. There are at least two Sponzas in
//! circulation — Crytek's original is authored in centimetres, the Khronos glTF
//! re-release in metres — and a scene that measures what it loaded frames the
//! building either way, while a scene full of magic numbers silently points the
//! camera at the inside of a column.
//!
//! One world unit is one metre, and Sponza is a real building: about 30 m along
//! the atrium, 13 m to the top of the upper arcade. The load log prints what was
//! actually measured against those numbers, so a file in the wrong units says so
//! in one line instead of being diagnosed from a black screen.

use std::path::PathBuf;

use glam::Vec3;

use orrin_ecs::World;

use super::textures::sky_equirect;
use super::{build_default_scene, spawn_directional_light};
use crate::gfx::RenderBackend;
use crate::scene::model::{self, ImportSettings, Model};
use crate::scene::{
    Assets, Camera, EnvironmentSettings, FogSettings, MaterialBlends, MeshBounds, Transform,
};

/// Where the model is looked for, relative to the working directory, when
/// `ORRIN_SPONZA` names nothing.
const DEFAULT_PATH: &str = "assets/sponza/Sponza.gltf";

/// What `scripts/fetch-sponza.sh` puts there, printed when it is not there.
const FETCH_HINT: &str = "run `scripts/fetch-sponza.sh` to download it (Khronos' glTF re-release \
                          of Crytek's Sponza, CC BY 3.0), or point ORRIN_SPONZA at a .gltf/.glb \
                          you already have";

pub fn build_sponza_scene(world: &mut World, backend: &mut impl RenderBackend) {
    let path = std::env::var_os("ORRIN_SPONZA")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PATH));

    // A file authored in anything but metres, corrected without a rebuild. The
    // default is 1.0 rather than a guess derived from the measured size: this
    // engine's lengths — a mean free path, a parallax depth, a fog density — are
    // in metres, so a scene that quietly rescaled the world to look right would
    // put every one of them out by the same factor and report nothing.
    let scale = std::env::var("ORRIN_SPONZA_SCALE")
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|s| s.is_finite() && *s > 0.0)
        .unwrap_or(1.0);

    let settings = ImportSettings {
        scale,
        ..ImportSettings::default()
    };

    let model = match model::load(&path, &settings) {
        Ok(model) => model,
        Err(error) => {
            // Loud, and then the rig — the same choice `SceneChoice::from_env`
            // makes for a misspelt scene name. An empty world would be a black
            // window with no explanation in it.
            tracing::error!(
                "ORRIN_SCENE=sponza: {error} — {FETCH_HINT}. Opening the demo scene instead"
            );
            build_default_scene(world, backend);
            return;
        }
    };

    let bounds = model.bounds();
    let size = bounds.max - bounds.min;
    tracing::info!(
        "sponza: {} triangles, {} primitives, {} materials, {} images; measured {:.1} x {:.1} x \
         {:.1} m (Sponza is ~30 x 13 x 18 m — a tenth or a hundred times that is a units \
         mismatch, correctable with ORRIN_SPONZA_SCALE)",
        model.triangle_count(),
        model.primitives.len(),
        model.materials.len(),
        model.images.len(),
        size.x,
        size.y,
        size.z,
    );

    let mut assets = Assets::new();
    let mut mesh_bounds = MeshBounds::default();
    let mut material_blends = MaterialBlends::default();

    // Dropped so the floor sits at `y = 0` whatever the file's own origin is,
    // which is what makes the light rig below independent of the download.
    let root = Transform::from_translation(Vec3::new(0.0, -bounds.min.y, 0.0));
    model::instantiate(
        world,
        backend,
        &mut assets,
        &mut mesh_bounds,
        &mut material_blends,
        &model,
        root,
    );

    world.insert_resource(assets);
    world.insert_resource(mesh_bounds);
    world.insert_resource(material_blends);

    let frame = Framing::of(&model);
    lights(world, backend, &frame);
    air(world);
    camera(world, &frame);
}

/// The building's own axes and extent, which everything else here is expressed
/// in.
struct Framing {
    /// Horizontal unit vector along the atrium's long axis, pointing the way the
    /// camera looks.
    along: Vec3,
    /// Horizontal unit vector across it.
    across: Vec3,
    /// Floor centre, `y` at the floor rather than at the middle of the volume.
    centre: Vec3,
    length: f32,
    height: f32,
}

impl Framing {
    fn of(model: &Model) -> Self {
        let bounds = model.bounds();
        let size = bounds.max - bounds.min;
        // Sponza's plan is a long rectangle; which world axis its length runs
        // along is a property of the exporter, not of the building.
        let (along, across, length) = if size.x >= size.z {
            (Vec3::X, Vec3::Z, size.x)
        } else {
            (Vec3::Z, Vec3::X, size.z)
        };
        Self {
            along,
            across,
            centre: Vec3::new(bounds.center().x, 0.0, bounds.center().z),
            length,
            height: size.y,
        }
    }
}

/// One light: the sun, high enough to reach the floor through the open roof.
///
/// Elevation is the load-bearing number, as it is in the courtyard, and it points
/// the opposite way here. Sponza is a courtyard with a colonnade around it and no
/// roof over the middle, so a low sun lights the upper arcade and leaves the
/// entire floor and every column in shadow. At the 63° below, the shaft through
/// the opening lands as a pool a few metres wide in the middle of the atrium and
/// the arcade gets a hard-edged rake across its piers — the two things a shadow
/// cascade is most obviously right or wrong about.
///
/// Tilted slightly along the atrium as well as across it so no column's shadow
/// runs exactly down a bay. An axis-aligned sun in an axis-aligned building hides
/// precisely the cascade seams and peter-panning this scene is useful for.
fn lights(world: &mut World, backend: &mut impl RenderBackend, frame: &Framing) {
    let travel = (frame.across * 0.42 + frame.along * 0.20 - Vec3::Y).normalize();

    // A clear midday sun, in lux — the unit the component's name states.
    // Deliberately the brightest light in this repository: the courtyard is an
    // hour before sunset at 22 000 lx, and the two scenes metering to visibly
    // different exposures is the histogram doing its job rather than a bug.
    spawn_directional_light(world, "Sun", travel, Vec3::new(1.0, 0.94, 0.86), 92_000.0);

    // Fed the direction *toward* the sun, so the disc drawn in the sky and the
    // light casting the shadows agree about where it is.
    const SKY: [u32; 2] = [1024, 512];
    backend.load_environment(&sky_equirect(SKY[0], SKY[1], -travel), SKY[0], SKY[1]);
    world.insert_resource(EnvironmentSettings {
        // What a clear midday sky *is*, in cd/m², rather than a multiplier on
        // whatever the generator emitted. It is doing more work in this scene
        // than in any other: three quarters of Sponza is arcade interior that the
        // sun never reaches, so the sky is what actually lights the frame, and
        // every stop of it moves the exposure the histogram settles on.
        sky_luminance: 6_000.0,
        ..EnvironmentSettings::default()
    });
}

/// A thin haze, so the shaft through the open roof is a shaft rather than a
/// bright patch of floor.
///
/// Two numbers, both taken from the building's dimensions rather than from taste.
/// The density is set against the *long* view: the camera looks down thirty
/// metres of atrium, and the froxels and the analytic tail integrate all of it,
/// so 0.006 per metre is about 16% of the far wall's contrast lost to inscatter —
/// visible as depth, not as milk. The scale height is set against the *short*
/// one: at `1 / 0.06` = 17 m the medium still has most of its density at the
/// height of the upper arcade, which is where a shaft has to be legible, and it
/// has thinned enough by the time a ray leaves the opening that the sky above is
/// still sky.
///
/// `anisotropy` is lower than the courtyard's 0.75 because this camera is not
/// pointed into the sun: at 60-odd degrees off it, a tighter phase function
/// subtracts from the shaft instead of adding to it.
fn air(world: &mut World) {
    world.insert_resource(FogSettings {
        density: 0.006,
        height_falloff: 0.06,
        anisotropy: 0.55,
        volumetric: true,
        ..FogSettings::default()
    });
}

/// Standing inside one end of the atrium at eye height, looking down its length.
///
/// The canonical Sponza view, and it is canonical for a reason worth stating: it
/// is the frame with the most *depth complexity* in the building — two storeys of
/// arcade receding on both sides, so it puts thirty metres of occluders between
/// the near columns and the far wall. That is what makes it the useful shot for
/// this renderer rather than merely the familiar one. Depth-dependent screen-space
/// terms are cheap to get right in a two-object rig and hard here.
///
/// Everything is a fraction of the measured building: 4% of its length in from the
/// end, 1.65 m up, aimed at a point two thirds of the way along and a third of the
/// way up, which puts the upper arcade in the top of the frame and the lit floor
/// across the bottom.
fn camera(world: &mut World, frame: &Framing) {
    let end = frame.centre - frame.along * frame.length * 0.46;
    world.insert_resource(Camera {
        position: end + Vec3::Y * 1.65,
        target: frame.centre + frame.along * frame.length * 0.2 + Vec3::Y * frame.height * 0.33,
        // 55° rather than the default 60°: a colonnade at 60° puts the two
        // nearest columns hard against the frame edges, where the arcade reads as
        // two walls instead of a receding row.
        fov_y: 55f32.to_radians(),
        ..Camera::default()
    });
}
