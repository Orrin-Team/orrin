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

use crate::gfx::SceneLighting;
use crate::gfx::vulkan::VulkanRenderer;
use crate::scene::entities::SceneChoice;
use crate::scene::{
    BloomSettings, Camera, ContactShadowSettings, DecalSettings, DofSettings, EnvironmentSettings,
    FogSettings, HdrSettings, MotionBlurSettings, RefractionSettings, ShadowDebugView,
    ShadowSettings, SsaoSettings, SsrSettings, SubsurfaceSettings, TaaSettings,
    TransparencySettings,
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
    /// Whether the point and spot lights get atlas tiles, in a capture that has
    /// `shadows` on. Off by default for the reason `shadows` is.
    pub punctual_shadows: bool,
    /// Which of the shadows' debug views the frame is drawn through.
    pub shadow_debug: ShadowDebugView,
    /// Which view of the traced scene is drawn in place of the frame. On a
    /// device that cannot trace, the capture is the frame as it was.
    pub ray_debug: crate::scene::RayDebugView,
    /// Whether anything traces a ray. On by default, as it is in the engine;
    /// off is the other half of the A/B for whatever a traced ray adds — a
    /// reflection of what the screen does not hold — and is the frame every
    /// device that cannot trace draws.
    pub ray_tracing: bool,
    /// Whether the irradiance volumes light the frame. Off by default, so
    /// every capture taken before they existed is the frame it was; on is the
    /// other half of the A/B for everything they change. Needs `ray_tracing`,
    /// and a device that traces.
    pub gi: bool,
    /// Whether the final gather lights the opaque surfaces, and at which
    /// resolution. Off by default, so a capture of the volumes is of the
    /// volumes alone; on is the other half of the A/B for what the gather
    /// adds below the probes' spacing. Needs `gi`.
    pub gi_gather: Option<crate::scene::GiGatherResolution>,
    /// Which of the volumes' debug views the frame is drawn through.
    pub gi_debug: crate::scene::GiDebugView,
    /// Screen-space ambient occlusion. On, as it ships; off is the other half
    /// of the A/B that says whether it and the volumes darken one corner
    /// twice.
    pub ssao: bool,
    /// The sun's angular diameter in degrees, or `None` for the default. Zero
    /// is the other half of an A/B: every edge at one width, whatever cast it.
    pub sun_angle: Option<f32>,
    /// Screen-space reflections: off by default, because they are the effect
    /// most easily mistaken for a lighting bug when they are wrong.
    pub ssr: bool,
    /// Depth of field, with the scene's default lens.
    pub dof: bool,
    /// Motion blur. Nothing moves in a capture, so what this photographs is the
    /// camera's own shutter reprojected against a static scene.
    pub motion_blur: bool,
    /// Rasterise the forward pass at four samples. The alternative to `taa`, which
    /// the capture turns off when this is on: the two are not companions.
    pub msaa: bool,
    /// How far below `extent` the scene is rendered, for the temporal resolve
    /// to bring back up. `extent` is the output's whatever this is, so a set
    /// taken at every mode pairs pixel for pixel.
    pub upscaling: crate::scene::Upscaling,
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
    /// The editor's wireframe draw mode, through the same overlay frame the
    /// editor hands the renderer, with nothing else in it.
    pub wireframe: Option<crate::gfx::vulkan::wireframe::Wireframe>,
    /// What a surface shader's `i.time` reads, or `None` to let it run from
    /// the fixed step below. Pinned by default, so a capture of an animated
    /// surface does not depend on how many frames it warmed up for.
    pub surface_time: Option<f32>,
    /// What the rig's ground is refinished as, or `None` for the matte floor it
    /// ships with. A reflection capture needs a reflective floor to be about
    /// anything: the shipped one is past every roughness the trace accepts.
    pub floor: Option<Floor>,
    /// Every sphere of the rig's roughness sweep refinished as white metal at
    /// roughness 0.5, or not at all. A rough metal is
    /// where a reflection has the fewest rays that hit anything, and the
    /// sweep's own row has one sphere of it against nine that say nothing.
    pub rough_metal_sweep: bool,
}

/// A reflective finish for the rig's ground, all metal so the reflection is
/// the whole of what the floor shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Floor {
    /// Roughness 0.04: a mirror, whose reflection is as sharp as the frame.
    Mirror,
    /// Roughness 0.3: glossy, where the reflection is a blur built from noise.
    Glossy,
    /// Roughness ramping from 0 at the floor's sides to 0.25 down its middle,
    /// across every rate and regime the trace switches between.
    RoughnessRamp,
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
            punctual_shadows: false,
            shadow_debug: ShadowDebugView::None,
            ray_debug: crate::scene::RayDebugView::None,
            ray_tracing: true,
            gi: false,
            gi_gather: None,
            gi_debug: crate::scene::GiDebugView::None,
            ssao: true,
            sun_angle: None,
            ssr: false,
            dof: false,
            motion_blur: false,
            msaa: false,
            upscaling: crate::scene::Upscaling::Native,
            fog: None,
            scene: SceneChoice::default(),
            camera: None,
            wireframe: None,
            surface_time: Some(0.0),
            floor: None,
            rough_metal_sweep: false,
        }
    }
}

/// The pixels of one of the built-in scenes, as tightly packed RGBA8 with the
/// extent they were rendered at.
pub fn scene_pixels(settings: &CaptureSettings) -> (Vec<u8>, [u32; 2]) {
    let mut renderer = offscreen_renderer(settings.extent);
    let mut world = World::new();
    prepare(&mut world, &mut renderer, settings);
    render(&mut world, &mut renderer, settings)
}

/// [`scene_pixels`] through native Metal.
#[cfg(target_os = "macos")]
pub fn scene_pixels_metal(settings: &CaptureSettings) -> (Vec<u8>, [u32; 2]) {
    let mut renderer =
        VulkanRenderer::offscreen_metal(settings.extent).expect("this machine runs Metal");
    let mut world = World::new();
    prepare(&mut world, &mut renderer, settings);
    render(&mut world, &mut renderer, settings)
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
    world.insert_resource(MotionBlurSettings {
        enabled: settings.motion_blur,
        ..MotionBlurSettings::default()
    });
    world.insert_resource(DofSettings {
        enabled: settings.dof,
        ..DofSettings::default()
    });
    world.insert_resource(BloomSettings::default());
    world.insert_resource(HdrSettings::default());
    world.insert_resource(EnvironmentSettings::default());
    world.insert_resource(FogSettings::default());

    settings.scene.build(world, renderer);
    if let Some(floor) = settings.floor {
        refinish_floor(world, renderer, floor);
    }
    if settings.rough_metal_sweep {
        refinish_sweep(world, renderer);
    }
    if let Some(camera) = settings.camera {
        world.insert_resource(camera);
    }

    world.insert_resource(SsaoSettings {
        enabled: settings.ssao,
        ..SsaoSettings::default()
    });
    world.insert_resource(ContactShadowSettings::default());
    world.insert_resource(SsrSettings {
        enabled: settings.ssr,
        ..SsrSettings::default()
    });
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
        enabled: settings.taa && !settings.msaa,
        msaa: settings.msaa,
        upscaling: settings.upscaling,
        ..TaaSettings::default()
    });
    if let Some(fog) = settings.fog {
        world.insert_resource(fog);
    }
}

fn refinish_floor(world: &World, renderer: &mut VulkanRenderer, floor: Floor) {
    use crate::gfx::{Material, RenderBackend};
    let ground = world
        .resource::<crate::scene::Assets>()
        .material("ground")
        .expect("a floor finish is for the rig, whose ground material is `ground`");
    let mut material = Material {
        base_color: glam::Vec3::splat(0.9),
        metallic: 1.0,
        roughness: 0.04,
        ..Material::default()
    };
    match floor {
        Floor::Mirror => {}
        Floor::Glossy => material.roughness = 0.3,
        Floor::RoughnessRamp => {
            // Roughness in green, as glTF's metallic-roughness texture has it,
            // and a factor of one so the texel is the roughness.
            const WIDTH: usize = 256;
            let texels: Vec<u8> = (0..WIDTH)
                .flat_map(|x| {
                    let u = x as f32 / (WIDTH - 1) as f32;
                    let roughness = 0.25 * (1.0 - (2.0 * u - 1.0).abs());
                    [0, (roughness * 255.0).round() as u8, 255, 255]
                })
                .collect();
            material.roughness = 1.0;
            material.metallic_roughness_texture =
                Some(renderer.load_texture(&texels, WIDTH as u32, 1, false));
        }
    }
    renderer.update_material(ground, &material);
}

fn refinish_sweep(world: &World, renderer: &mut VulkanRenderer) {
    use crate::gfx::{Material, RenderBackend};
    let material = Material {
        base_color: glam::Vec3::ONE,
        metallic: 1.0,
        roughness: 0.5,
        ..Material::default()
    };
    for (name, handle) in world.resource::<crate::scene::Assets>().materials() {
        if name.starts_with("metal_") || name.starts_with("dielectric_") {
            renderer.update_material(handle, &material);
        }
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
    let defaults = ShadowSettings::default();
    let shadow_settings = ShadowSettings {
        punctual_enabled: settings.punctual_shadows,
        debug_view: settings.shadow_debug,
        sun_angle: settings.sun_angle.unwrap_or(defaults.sun_angle),
        ..defaults
    };
    let aspect = settings.extent[0] as f32 / settings.extent[1] as f32;
    let mut overlay = settings
        .wireframe
        .map(|wireframe| crate::gfx::vulkan::OverlayFrame {
            wireframe: Some(wireframe),
            ..Default::default()
        });

    renderer.pin_surface_time(settings.surface_time);
    renderer.set_ray_tracing(crate::scene::RayTracingSettings {
        enabled: settings.ray_tracing,
        debug_view: settings.ray_debug,
        ..Default::default()
    });
    renderer.set_gi(crate::scene::GiSettings {
        enabled: settings.gi,
        gather: settings.gi_gather.is_some(),
        gather_resolution: settings.gi_gather.unwrap_or_default(),
        debug_view: settings.gi_debug,
        ..Default::default()
    });
    for _ in 0..settings.frames.max(1) {
        // Extraction reads *world* transforms, and a freshly built scene has
        // only local ones — the propagation that derives them is a system, not
        // something spawning does. Without it every renderable fails the query
        // and the frame is a skybox with nothing in front of it.
        crate::scene::propagate_transforms(world);
        systems::extract_lighting(world, &mut lighting);
        let camera = *world.resource::<Camera>();
        systems::extract_geometry(world, aspect, &mut geometry);
        systems::extract_decals(world, aspect, &mut decals);

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
            settings.shadows.then_some(&shadow_settings),
            None,
            overlay.as_mut(),
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
