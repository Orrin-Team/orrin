//! Rendering a frame to a PNG with no window.
//!
//! This exists because until it did, the only way to check that a change had not
//! broken the picture was to look at the running editor — which means a person,
//! a display, and a screenshot, none of which a test has. The render graph's
//! golden pins synchronisation and says nothing at all about pixels; a lighting
//! term that silently went dark passes it.
//!
//! What it renders is deliberately the whole real path: the same
//! [`VulkanRenderer`], the same passes, the same scene the editor opens with.
//! Only the target differs — see [`VulkanRenderer::offscreen`] — because a
//! capture taken through a simpler path would be evidence about that path rather
//! than about the frame anyone actually sees.
//!
//! It needs a GPU, so nothing here runs in CI. `tests/offscreen.rs` is the entry
//! point and is `#[ignore]`d for that reason.

use std::path::Path;
use std::sync::Arc;

use orrin_ecs::World;
use vulkano::VulkanLibrary;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};

use crate::gfx::RenderBackend;
use crate::gfx::punctual::ShadowAtlas;
use crate::gfx::shadows::{CascadeSet, MAX_CASCADES, cascades};
use crate::gfx::vulkan::{ShadowFrame, VulkanRenderer};
use crate::gfx::{DrawList, SceneLighting};
use crate::scene::entities::SceneChoice;
use crate::scene::{
    BloomSettings, Camera, ContactShadowSettings, DecalSettings, DofSettings, EnvironmentSettings,
    FogSettings, HdrSettings, MotionBlurSettings, RefractionSettings, ShadowSettings, SsaoSettings,
    SsrSettings, SubsurfaceSettings, TaaSettings, TransparencySettings,
};
use crate::systems::{self, FrameGeometry};

/// Frames to render before reading one back.
///
/// One is not enough and the reason is structural rather than a fudge: the
/// temporal resolve has no history on its first frame and says so, auto-exposure
/// adapts over several, and the motion vectors want a previous transform to
/// reproject through. Capturing frame zero would photograph the engine's
/// startup transient rather than its output.
pub const WARMUP_FRAMES: u32 = 12;

/// What the capture renders, so the caller does not have to assemble a world.
pub struct CaptureSettings {
    pub extent: [u32; 2],
    pub frames: u32,
    /// Overridden after the scene is built, so a capture can look at one feature
    /// without the rest of the frame moving under it.
    pub transparency: bool,
    pub refraction: bool,
    pub taa: bool,
    /// The screen-space half of subsurface scattering. Worth its own override for
    /// the reason the two non-opaque queues have theirs: switching it off does not
    /// switch scattering off — the analytic wrap widens to stand in — so the A/B
    /// between the two captures is exactly what the diffusion passes contribute.
    pub subsurface: bool,
    /// Whether decals are projected. The A/B that says which marks on a surface
    /// are decals and which are the material, which is otherwise a hard question
    /// to ask of a still frame — a decal lands *under* the lighting, so it looks
    /// exactly like something that was always painted there. That is the feature
    /// working, and it is also why it needs a capture with it off.
    pub decals: bool,
    /// Whether the sun's cascades are fitted and drawn.
    ///
    /// Off by default, and that is a cost decision rather than a claim that
    /// shadows do not matter: cascades are a second caster extraction and a
    /// depth pass per cascade on every warm-up frame, which every capture would
    /// then pay for. Switched on for the ones that are *about* a shadow — a
    /// cutout's caster pipeline runs its own alpha test, and the difference
    /// between a leaf-shaped shadow and a quad-shaped one is the only place that
    /// pipeline is visible at all.
    pub shadows: bool,
    /// The air, or `None` for whatever the scene asked for.
    ///
    /// Overridden whole rather than by a flag, because the A/B the fog captures
    /// make is between two *ways of integrating one medium* — the froxels and the
    /// analytic height layer — and that comparison is only meaningful if the
    /// density, the falloff and the albedo are identical across the pair. A
    /// `volumetric: bool` beside the others would have let the two differ in more
    /// than the thing being compared.
    ///
    /// `None` rather than the engine's default, so that a scene built around its
    /// own air is photographed in it: the courtyard sets a medium in
    /// `build_showcase_scene` and a capture that reinstalled the zero-density
    /// default would take the one picture the scene is not about.
    pub fog: Option<FogSettings>,
    /// Which built-in scene to photograph.
    pub scene: SceneChoice,
    /// Where to photograph the scene from, or `None` for the camera it ships
    /// with.
    ///
    /// A whole-scene shot is a check that everything still draws, and it is a
    /// poor check of anything that lives in the texels: at the distance the demo
    /// camera stands, a surface detail is a handful of pixels and a capture of it
    /// is a capture of the mip chain. So a feature whose readout is a
    /// centimetre of relief gets its own vantage point rather than a bigger
    /// image.
    pub camera: Option<Camera>,
}

impl Default for CaptureSettings {
    fn default() -> Self {
        Self {
            extent: [1280, 720],
            frames: WARMUP_FRAMES,
            transparency: true,
            refraction: true,
            taa: true,
            subsurface: true,
            decals: true,
            shadows: false,
            fog: None,
            scene: SceneChoice::default(),
            camera: None,
        }
    }
}

/// Render one of the built-in scenes and write it to `path` as a PNG.
///
/// Panics rather than returning an error on a device that cannot be reached: it
/// is a developer tool driven from a test, and a `Result` nobody can act on
/// would only push the same panic one frame up the stack.
pub fn capture_scene(path: impl AsRef<Path>, settings: &CaptureSettings) {
    let mut renderer = offscreen_renderer(settings.extent);
    let mut world = World::new();
    prepare(&mut world, &mut renderer, settings);
    let (pixels, extent) = render(&mut world, &mut renderer, settings);
    image::save_buffer(
        path.as_ref(),
        &pixels,
        extent[0],
        extent[1],
        image::ColorType::Rgba8,
    )
    .expect("failed to write the capture");
}

/// A renderer drawing into an image of `extent` rather than a window.
pub fn offscreen_renderer(extent: [u32; 2]) -> VulkanRenderer {
    VulkanRenderer::offscreen(&headless_instance(), extent)
}

/// Build `settings.scene` into `world` with everything the frame reads, for a
/// caller that changes the world before [`render`]ing it.
pub fn prepare(world: &mut World, renderer: &mut VulkanRenderer, settings: &CaptureSettings) {
    // Default, then scene, then capture override — in that order, and the order
    // is the contract. The baseline goes in *first* so a scene that describes its
    // own air, exposure or environment is photographed as itself rather than as
    // the engine's defaults; the explicit fields below then still win, because a
    // capture that is an A/B has to be able to state both halves of it.
    world.insert_resource(MotionBlurSettings::default());
    world.insert_resource(DofSettings::default());
    world.insert_resource(BloomSettings::default());
    world.insert_resource(HdrSettings::default());
    world.insert_resource(EnvironmentSettings::default());
    world.insert_resource(FogSettings::default());

    settings.scene.build(world, renderer);
    if let Some(camera) = settings.camera {
        world.insert_resource(camera);
    }

    world.insert_resource(SsaoSettings::default());
    world.insert_resource(ContactShadowSettings::default());
    world.insert_resource(SsrSettings::default());
    world.insert_resource(SubsurfaceSettings {
        enabled: settings.subsurface,
        ..SubsurfaceSettings::default()
    });
    world.insert_resource(DecalSettings {
        enabled: settings.decals,
    });
    world.insert_resource(TransparencySettings {
        enabled: settings.transparency,
    });
    world.insert_resource(RefractionSettings {
        enabled: settings.refraction,
    });
    world.insert_resource(TaaSettings {
        enabled: settings.taa,
        ..TaaSettings::default()
    });
    if let Some(fog) = settings.fog {
        world.insert_resource(fog);
    }
}

/// Render `settings.frames` frames of `world` and read the last one back as
/// tightly packed RGBA8.
pub fn render(
    world: &mut World,
    renderer: &mut VulkanRenderer,
    settings: &CaptureSettings,
) -> (Vec<u8>, [u32; 2]) {
    let mut lighting = SceneLighting::default();
    let mut geometry = FrameGeometry::default();
    let mut decals = Vec::new();
    // No punctual atlas either way: the sun is what a cutout's shadow is legible
    // against, and a tile per face per lamp is a second budget for no extra
    // evidence. A capture that wants one grows it the way `app.rs` does.
    let atlas = ShadowAtlas::default();
    let shadow_settings = ShadowSettings::default();
    let aspect = settings.extent[0] as f32 / settings.extent[1] as f32;

    for _ in 0..settings.frames.max(1) {
        // Extraction reads *world* transforms, and a freshly built scene has
        // only local ones — the propagation that derives them is a system, not
        // something spawning does. Without it every renderable fails the query
        // and the frame is a skybox with nothing in front of it.
        crate::scene::propagate_transforms(world);
        systems::extract_lighting(world, &mut lighting);
        let camera = *world.resource::<Camera>();
        // Fitted before extraction and after the lighting, exactly as `app.rs`
        // orders it: the boxes are built around the sun that extraction just
        // found, and the caster lists are culled against those boxes.
        let cascade_set = if settings.shadows {
            cascades(
                &camera,
                aspect,
                lighting.sun.direction,
                &shadow_settings.cascade_config(),
            )
        } else {
            CascadeSet::default()
        };
        systems::extract_geometry(
            world,
            aspect,
            &cascade_set,
            &atlas,
            renderer.gpu_culling(),
            &mut geometry,
        );
        systems::extract_decals(world, aspect, &mut decals);

        let caster_lists: [DrawList<'_>; MAX_CASCADES] =
            std::array::from_fn(|i| geometry.cascade(i));
        let punctual_lists: [DrawList<'_>; 0] = [];
        renderer.render_with_overlay(
            geometry.visible(),
            geometry.transparent(),
            geometry.refractive(),
            &decals,
            &lighting,
            &camera,
            &world.resource::<SsaoSettings>().clone(),
            &world.resource::<ContactShadowSettings>().clone(),
            &world.resource::<SsrSettings>().clone(),
            &world.resource::<SubsurfaceSettings>().clone(),
            &world.resource::<TransparencySettings>().clone(),
            &world.resource::<RefractionSettings>().clone(),
            &world.resource::<TaaSettings>().clone(),
            &world.resource::<MotionBlurSettings>().clone(),
            &world.resource::<DofSettings>().clone(),
            &world.resource::<BloomSettings>().clone(),
            &world.resource::<HdrSettings>().clone(),
            &world.resource::<EnvironmentSettings>().clone(),
            &world.resource::<FogSettings>().clone(),
            // A fixed step rather than a real clock, so two runs of this produce
            // the same picture: auto-exposure adapts over time, and a capture
            // that depended on how fast the host was would be useless to diff.
            1.0 / 60.0,
            // No debug lines, no profiler, no editor: this is a picture of the
            // scene. `render_with_overlay` rather than `render` only because the
            // shadow frame comes in through it — an `Option` so that a capture
            // with shadows off is byte for byte the frame it was before this
            // existed.
            &[],
            0,
            (cascade_set.count > 0).then_some(ShadowFrame {
                cascades: &cascade_set,
                casters: &caster_lists,
                atlas: &atlas,
                punctual_casters: &punctual_lists,
                settings: &shadow_settings,
            }),
            None,
        );
    }

    renderer
        .capture()
        .expect("an offscreen renderer must have a readable target")
}

/// An instance with no surface extensions.
///
/// `ENUMERATE_PORTABILITY` is the one flag that still matters without a window:
/// MoltenVK is a portability driver, and an instance that does not enumerate
/// those finds no device at all on macOS.
pub(crate) fn headless_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("failed to load vulkan library");
    Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .expect("failed to create a headless instance")
}
