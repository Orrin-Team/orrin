//! A scene's renderer configuration, saved with it.
//!
//! The frame reads its settings from world resources — `FogSettings`,
//! `BloomSettings` and the rest — and nothing about that changes here. What a
//! scene adds is a [`RenderSettings`] component on one of its entities: an
//! ordinary registered component, so saving, loading, the inspector, undo, the
//! dirty marker, autosave and Play's fork all carry it without a line of their
//! own. [`sync`] copies it into the resources once a frame.
//!
//! An entity rather than a block in the file's header, because the header has
//! none of that machinery: every one of those consumers would need a second
//! path for one value. It is the same choice Godot's `WorldEnvironment` node
//! makes, and it is what keeps "everything in a scene is an entity" true.
//!
//! A scene with no such entity leaves the resources alone, which is what every
//! scene written before this existed, the built-in demo scenes, the capture and
//! perf harnesses, and `ORRIN_HDRI` all rely on. When one appears, the
//! resources it is about to overwrite are kept, and they are put back when it
//! goes, so loading a scene without settings after one with them does not draw
//! it in the previous scene's fog.
//!
//! Only what describes the scene is saved. `ShadowSettings::debug_cascades` is
//! a way of looking at it and `EnvironmentSettings::reload_requested` is a
//! request to the app; both are skipped by the derive and left as the resource
//! has them. `PresentSettings` is the window's, not any scene's, and is not
//! here at all.

use orrin_ecs::{Entity, World};
use orrin_registry::Reflect;

use super::{
    AmbientLight, BloomSettings, ContactShadowSettings, DecalSettings, DofSettings,
    EnvironmentSettings, FogSettings, HdrSettings, MotionBlurSettings, RefractionSettings,
    ShadowSettings, SsaoSettings, SsrSettings, SubsurfaceSettings, TaaSettings,
    TransparencySettings,
};

/// Everything a scene says about how it is drawn. One component, so the
/// inspector shows it as one section and a change to any field is one path in
/// the edit stream (`fog.density`), labelled as such in the undo history.
#[derive(Clone, Debug, Default, PartialEq, Reflect)]
pub struct RenderSettings {
    pub ambient: AmbientLight,
    pub environment: EnvironmentSettings,
    pub exposure: HdrSettings,
    pub ssao: SsaoSettings,
    pub ssr: SsrSettings,
    pub subsurface: SubsurfaceSettings,
    pub transparency: TransparencySettings,
    pub refraction: RefractionSettings,
    pub decals: DecalSettings,
    pub shadows: ShadowSettings,
    pub contact_shadows: ContactShadowSettings,
    pub bloom: BloomSettings,
    pub taa: TaaSettings,
    pub motion_blur: MotionBlurSettings,
    pub dof: DofSettings,
    pub fog: FogSettings,
}

/// What the resources held before a scene's settings replaced them. Present
/// exactly while a [`RenderSettings`] entity is driving the renderer.
#[derive(Default)]
struct Baseline(Option<Box<RenderSettings>>);

/// The entity whose [`RenderSettings`] the frame is drawn with, if any.
pub fn source(world: &World) -> Option<Entity> {
    world.query::<&RenderSettings>().find(|_, _| true)
}

/// Bring the renderer's resources in line with the scene's settings. Runs once
/// a frame, after everything that can edit them and before extraction.
pub fn sync(world: &mut World) {
    match source(world) {
        Some(entity) => {
            if world.get_resource::<Baseline>().is_none() {
                let kept = capture(world);
                world.insert_resource(Baseline(Some(Box::new(kept))));
            } else {
                let mut baseline = world.resource_mut::<Baseline>();
                if baseline.0.is_none() {
                    baseline.0 = Some(Box::new(capture(world)));
                }
            }
            let settings = world.get::<RenderSettings>(entity).expect("found by query");
            apply(world, &settings);
        }
        None => {
            let kept = world
                .get_resource_mut::<Baseline>()
                .and_then(|mut baseline| baseline.0.take());
            if let Some(kept) = kept {
                apply(world, &kept);
            }
        }
    }
}

/// What the renderer's resources hold now, as a scene would save it.
pub fn capture(world: &World) -> RenderSettings {
    RenderSettings {
        ambient: *world.resource(),
        environment: world.resource::<EnvironmentSettings>().clone(),
        exposure: *world.resource(),
        ssao: *world.resource(),
        ssr: *world.resource(),
        subsurface: *world.resource(),
        transparency: *world.resource(),
        refraction: *world.resource(),
        decals: *world.resource(),
        shadows: *world.resource(),
        contact_shadows: *world.resource(),
        bloom: *world.resource(),
        taa: *world.resource(),
        motion_blur: *world.resource(),
        dof: *world.resource(),
        fog: *world.resource(),
    }
}

/// Destructured rather than copied whole, so a field added to either struct is
/// a compile error here until someone decides whether a scene saves it.
fn apply(world: &World, settings: &RenderSettings) {
    let RenderSettings {
        ambient,
        environment,
        exposure,
        ssao,
        ssr,
        subsurface,
        transparency,
        refraction,
        decals,
        shadows,
        contact_shadows,
        bloom,
        taa,
        motion_blur,
        dof,
        fog,
    } = settings;
    *world.resource_mut() = *ambient;
    *world.resource_mut() = *exposure;
    *world.resource_mut() = *ssao;
    *world.resource_mut() = *ssr;
    *world.resource_mut() = *subsurface;
    *world.resource_mut() = *transparency;
    *world.resource_mut() = *refraction;
    *world.resource_mut() = *decals;
    *world.resource_mut() = *contact_shadows;
    *world.resource_mut() = *bloom;
    *world.resource_mut() = *taa;
    *world.resource_mut() = *motion_blur;
    *world.resource_mut() = *dof;
    *world.resource_mut() = *fog;

    let mut live = world.resource_mut::<ShadowSettings>();
    *live = ShadowSettings {
        debug_cascades: live.debug_cascades,
        ..*shadows
    };

    let EnvironmentSettings {
        hdri,
        reload_requested: _,
        sky_luminance,
        exposure_offset,
        yaw,
        show_skybox,
        background,
    } = environment;
    let mut live = world.resource_mut::<EnvironmentSettings>();
    // Compared rather than assigned, because assigning is an allocation a
    // frame and a changed name is a bake. An empty name asks for no bake: it
    // means whatever sky the scene's builder made, and there is no file to
    // read that from.
    if live.hdri != *hdri {
        live.hdri.clone_from(hdri);
        live.reload_requested = !hdri.trim().is_empty();
    }
    live.sky_luminance = *sky_luminance;
    live.exposure_offset = *exposure_offset;
    live.yaw = *yaw;
    live.show_skybox = *show_skybox;
    live.background = *background;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::register_components;
    use orrin_registry::Registry;

    fn renderer() -> World {
        let mut world = World::new();
        crate::App::install_default_resources(&mut world);
        world
    }

    fn with_settings(world: &mut World, settings: RenderSettings) -> Entity {
        let entity = world.spawn();
        world.insert(entity, settings);
        entity
    }

    fn foggy() -> RenderSettings {
        let mut settings = RenderSettings::default();
        settings.fog.density = 0.05;
        settings.bloom.enabled = false;
        settings.shadows.cascade_count = 2;
        settings.environment.hdri = "skies/dusk.hdr".to_owned();
        settings
    }

    /// The point of the component: what the scene says about its renderer is
    /// in its file, and comes back out of it unchanged.
    #[test]
    fn a_scenes_render_settings_survive_a_save_and_a_load() {
        let mut registry = Registry::new();
        register_components(&mut registry);
        let mut world = World::new();
        with_settings(&mut world, foggy());

        let text = crate::scene::save(&mut world, &registry);
        assert!(text.contains("orrin.render_settings"), "{text}");
        assert!(
            !text.contains("debug_cascades"),
            "a debug view was saved:\n{text}"
        );
        assert!(
            !text.contains("reload_requested"),
            "a request was saved:\n{text}"
        );

        let mut loaded = World::new();
        crate::scene::load(&text, &mut loaded, &registry).unwrap();
        let entity = source(&loaded).expect("the settings entity came back");
        assert_eq!(*loaded.get::<RenderSettings>(entity).unwrap(), foggy());
    }

    #[test]
    fn the_scene_drives_the_renderer() {
        let mut world = renderer();
        with_settings(&mut world, foggy());
        sync(&mut world);
        assert_eq!(world.resource::<FogSettings>().density, 0.05);
        assert!(!world.resource::<BloomSettings>().enabled);
        assert_eq!(world.resource::<ShadowSettings>().cascade_count, 2);
    }

    /// Every scene saved before settings were, and the built-in demos, set the
    /// resources directly. Without an entity to say otherwise, those stand.
    #[test]
    fn a_scene_without_render_settings_leaves_the_renderer_alone() {
        let mut world = renderer();
        world.resource_mut::<FogSettings>().density = 0.3;
        sync(&mut world);
        assert_eq!(world.resource::<FogSettings>().density, 0.3);
    }

    /// Loading a scene without settings after one with them must not draw it
    /// in the previous scene's fog.
    #[test]
    fn losing_the_settings_puts_the_renderer_back() {
        let mut world = renderer();
        world.resource_mut::<FogSettings>().density = 0.3;
        let entity = with_settings(&mut world, foggy());
        sync(&mut world);
        assert_eq!(world.resource::<FogSettings>().density, 0.05);

        world.despawn(entity);
        sync(&mut world);
        assert_eq!(world.resource::<FogSettings>().density, 0.3);
        assert!(world.resource::<BloomSettings>().enabled);
    }

    /// Tinting the cascades is a way of looking at the scene, toggled from the
    /// Render menu; the scene's settings must not switch it back off.
    #[test]
    fn a_debug_view_survives_the_scenes_settings() {
        let mut world = renderer();
        world.resource_mut::<ShadowSettings>().debug_cascades = true;
        with_settings(&mut world, foggy());
        sync(&mut world);
        assert!(world.resource::<ShadowSettings>().debug_cascades);
    }

    /// A new sky has to be baked, and an unchanged one must not be baked again
    /// every frame.
    #[test]
    fn a_new_hdri_is_baked_once() {
        let mut world = renderer();
        with_settings(&mut world, foggy());
        sync(&mut world);
        {
            let environment = world.resource::<EnvironmentSettings>();
            assert_eq!(environment.hdri, "skies/dusk.hdr");
            assert!(environment.reload_requested);
        }
        world.resource_mut::<EnvironmentSettings>().reload_requested = false;
        sync(&mut world);
        assert!(!world.resource::<EnvironmentSettings>().reload_requested);
    }

    /// An edit to the component reaches the frame that follows it.
    #[test]
    fn an_edit_to_the_settings_is_drawn_next_frame() {
        let mut world = renderer();
        let entity = with_settings(&mut world, foggy());
        sync(&mut world);
        world.get_mut::<RenderSettings>(entity).unwrap().fog.density = 0.2;
        sync(&mut world);
        assert_eq!(world.resource::<FogSettings>().density, 0.2);
    }
}
