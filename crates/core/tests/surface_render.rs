//! Surface shaders drawn by the real renderer, on every backend this machine
//! has: that a material read through one is drawn by its twins in every queue
//! and in the prepass, that its parameters and the time reach it, that a swap
//! takes effect and a broken one changes nothing, and that the inspector's
//! sphere shows it.
//!
//! Most cases are *equivalences*: a surface function that rewrites the
//! surface to what a standard material would have been must draw that standard
//! material's pixels. That is the strongest statement a capture can make about
//! a twin, because the reference is the built-in pipeline itself; and a case
//! whose function changed nothing would pass it trivially, so each also asserts
//! that the pixels moved from where they started.
//!
//! `#[ignore]`d because it needs a GPU, and behind the default
//! `shader-compiler` feature because it compiles its shaders from source.
//!
//! ```text
//! cargo test -p orrin-core --test surface_render -- --ignored --nocapture
//! ```

#![cfg(feature = "shader-compiler")]

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use glam::{Quat, Vec3};
use orrin_core::capture::{self, CaptureSettings, offscreen_renderer};
use orrin_core::gfx::surface::{
    ParamValue, SurfaceArtifact, SurfaceShader, SurfaceShaderError, compile,
};
use orrin_core::gfx::vulkan::VulkanRenderer;
use orrin_core::gfx::vulkan::material_preview::{Orbit, Subject, Viewer};
use orrin_core::gfx::{BlendMode, DrawList, Material, RenderBackend, SceneLighting};
use orrin_core::scene::entities::{spawn_directional_light, spawn_mesh};
use orrin_core::scene::{
    AmbientLight, BloomSettings, Camera, ContactShadowSettings, CpuMesh, DofSettings,
    EnvironmentSettings, FogSettings, HdrSettings, MaterialBlends, MaterialHandle, MeshBounds,
    MotionBlurSettings, RefractionSettings, SsaoSettings, SsrSettings, SubsurfaceSettings,
    TaaSettings, Transform, TransparencySettings,
};
use orrin_ecs::World;

const EXTENT: [u32; 2] = [320, 240];

const EMPTY: &str = "void surface(in SurfaceIn i, inout SurfaceOut o) {}\n";

const TINT: &str = "\
param color tint = (1, 1, 1, 1);
void surface(in SurfaceIn i, inout SurfaceOut o) {
    o.albedo = tint.rgb;
}
";

const GREEN: &str = "\
void surface(in SurfaceIn i, inout SurfaceOut o) {
    o.albedo = vec3(0.05, 0.7, 0.1);
}
";

/// The normal the control's 1x1 map decodes to, written out: what
/// `tangent_normal` makes of the texel `(170, 128, 248)`.
const POLISHED: &str = "\
void surface(in SurfaceIn i, inout SurfaceOut o) {
    o.roughness = 0.08;
    o.normal = vec3(170.0, 128.0, 248.0) / 255.0 * 2.0 - 1.0;
}
";

/// Four opaque stripes and four holes across the quad.
const STRIPES: &str = "\
void surface(in SurfaceIn i, inout SurfaceOut o) {
    o.alpha = step(0.5, fract(i.uv.x * 4.0));
}
";

const CLOCK: &str = "\
void surface(in SurfaceIn i, inout SurfaceOut o) {
    o.albedo = vec3(fract(i.time));
}
";

/// Every shader a case compiles, compiled once per process: the eight variants
/// are a second or so of shaderc and SPIRV-Cross each time.
fn compiled(text: &str) -> SurfaceArtifact {
    static CACHE: OnceLock<Mutex<HashMap<String, SurfaceArtifact>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(artifact) = cache.lock().unwrap().get(text) {
        return artifact.clone();
    }
    let shader = SurfaceShader::parse(text).unwrap_or_else(|e| panic!("{e:#?}"));
    let artifact = compile(&shader, "test.oshader").unwrap_or_else(|errors| {
        panic!(
            "{}",
            errors
                .iter()
                .map(|e| e.located("test.oshader"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    });
    cache
        .lock()
        .unwrap()
        .insert(text.to_string(), artifact.clone());
    artifact
}

/// A room small enough to reason about: a ground, the subject on the left, a
/// sphere on the right that nothing changes, a sun, and a manual exposure so a
/// change to the subject cannot move every other pixel through the meter.
struct Rig {
    world: World,
    ground: MaterialHandle,
    subject: MaterialHandle,
    settings: CaptureSettings,
}

#[derive(Clone, Copy)]
enum Shape {
    /// A cube standing on the ground.
    Block,
    /// A quad facing the camera, with a green wall behind it.
    Card,
}

fn grey(roughness: f32) -> Material {
    Material {
        base_color: Vec3::splat(0.5),
        metallic: 0.0,
        roughness,
        ..Material::default()
    }
}

fn rig(renderer: &mut VulkanRenderer, shape: Shape, subject: Material) -> Rig {
    let mut world = World::new();
    let mut blends = MaterialBlends::default();
    let mut bounds = MeshBounds::default();
    let mut mesh = |renderer: &mut VulkanRenderer, mesh: CpuMesh| {
        let mesh = mesh.uncolored();
        let handle = renderer.load_mesh(&mesh);
        bounds.insert(handle, renderer.mesh_bounds(handle).unwrap());
        handle
    };
    let plane = mesh(renderer, CpuMesh::plane());
    let cube = mesh(renderer, CpuMesh::cube());
    let sphere = mesh(renderer, CpuMesh::sphere(48, 24));
    let mut material = |renderer: &mut VulkanRenderer, material: Material| {
        let handle = renderer.load_material(&material);
        blends.insert(handle, material.blend);
        handle
    };
    let ground = material(renderer, grey(0.9));
    let subject_handle = material(renderer, subject);
    let orange = material(
        renderer,
        Material {
            base_color: Vec3::new(0.9, 0.4, 0.1),
            roughness: 0.5,
            ..Material::default()
        },
    );

    spawn_mesh(
        &mut world,
        "Ground",
        Transform {
            scale: Vec3::new(12.0, 1.0, 12.0),
            ..Default::default()
        },
        plane,
        ground,
    );
    match shape {
        Shape::Block => {
            spawn_mesh(
                &mut world,
                "Subject",
                Transform::from_translation(Vec3::new(-0.7, 0.5, 0.0)),
                cube,
                subject_handle,
            );
        }
        Shape::Card => {
            let wall = material(
                renderer,
                Material {
                    base_color: Vec3::new(0.1, 0.8, 0.1),
                    roughness: 0.9,
                    ..Material::default()
                },
            );
            let upright = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
            spawn_mesh(
                &mut world,
                "Wall",
                Transform {
                    translation: Vec3::new(0.0, 1.0, -1.5),
                    rotation: upright,
                    scale: Vec3::new(6.0, 1.0, 3.0),
                },
                plane,
                wall,
            );
            spawn_mesh(
                &mut world,
                "Subject",
                Transform {
                    translation: Vec3::new(-0.5, 0.6, 0.0),
                    rotation: upright,
                    scale: Vec3::new(1.2, 1.0, 1.0),
                },
                plane,
                subject_handle,
            );
        }
    }
    spawn_mesh(
        &mut world,
        "Sphere",
        Transform::from_translation(Vec3::new(0.8, 0.5, 0.0)),
        sphere,
        orange,
    );
    spawn_directional_light(
        &mut world,
        "Sun",
        Vec3::new(-0.4, -1.0, -0.5),
        Vec3::ONE,
        // Dim: the transparency accumulation multiplies radiance by a weight
        // of up to 3000 into half floats, so anything brighter than about
        // twenty candela a square metre overflows it and a blended surface
        // draws black whatever its colour. That is the accumulation's, and
        // every capture of glass here has to stay under it.
        20.0,
    );
    world.insert_resource(AmbientLight {
        nits: 1.0,
        ..AmbientLight::default()
    });

    world.insert_resource(blends);
    world.insert_resource(bounds);
    world.insert_resource(Camera {
        position: Vec3::new(0.0, 1.4, 3.0),
        target: Vec3::new(0.0, 0.5, 0.0),
        ..Camera::default()
    });
    world.insert_resource(SsaoSettings::default());
    // Off, with every other effect that dithers by the frame's number: two
    // captures compared below may be drawn after different histories.
    world.insert_resource(ContactShadowSettings {
        enabled: false,
        ..ContactShadowSettings::default()
    });
    world.insert_resource(SsrSettings {
        enabled: false,
        ..SsrSettings::default()
    });
    world.insert_resource(SubsurfaceSettings {
        enabled: false,
        ..SubsurfaceSettings::default()
    });
    world.insert_resource(TransparencySettings { enabled: true });
    world.insert_resource(RefractionSettings { enabled: true });
    world.insert_resource(TaaSettings {
        enabled: false,
        ..TaaSettings::default()
    });
    world.insert_resource(MotionBlurSettings {
        enabled: false,
        ..MotionBlurSettings::default()
    });
    world.insert_resource(DofSettings {
        enabled: false,
        ..DofSettings::default()
    });
    world.insert_resource(BloomSettings {
        enabled: false,
        ..BloomSettings::default()
    });
    world.insert_resource(HdrSettings {
        auto_exposure: false,
        manual_ev100: 3.0,
        ..HdrSettings::default()
    });
    world.insert_resource(EnvironmentSettings::default());
    world.insert_resource(FogSettings::default());

    Rig {
        world,
        ground,
        subject: subject_handle,
        settings: CaptureSettings {
            extent: EXTENT,
            frames: 3,
            ..CaptureSettings::default()
        },
    }
}

impl Rig {
    fn draw(&mut self, renderer: &mut VulkanRenderer) -> Vec<u8> {
        capture::render(&mut self.world, renderer, &self.settings).0
    }
}

/// The rig in a fresh renderer, drawn once, then `change`d and drawn again.
/// Everything a case compares is made this way, so the two sides have the same
/// history frame for frame — reflections dither by the frame's number — and
/// the change lands on a material the cull already numbered, which is the
/// re-classing a real attach is.
fn shoot(
    make: fn([u32; 2]) -> VulkanRenderer,
    shape: Shape,
    subject: Material,
    configure: impl FnOnce(&mut VulkanRenderer, &mut Rig),
    change: impl FnOnce(&mut VulkanRenderer, &mut Rig),
) -> Vec<u8> {
    let mut renderer = make(EXTENT);
    let mut rig = rig(&mut renderer, shape, subject);
    configure(&mut renderer, &mut rig);
    rig.draw(&mut renderer);
    change(&mut renderer, &mut rig);
    rig.draw(&mut renderer)
}

/// Read `rig`'s subject through `text`, with `values`.
fn attach(
    renderer: &mut VulkanRenderer,
    material: MaterialHandle,
    text: &str,
    values: &[(&str, ParamValue)],
) {
    let shader = renderer.load_surface_shader(&compiled(text)).unwrap();
    let ignored = renderer
        .set_material_surface(material, Some((shader, &named(values))))
        .unwrap();
    assert!(ignored.is_empty(), "{ignored:?}");
}

fn nothing(_: &mut VulkanRenderer, _: &mut Rig) {}

/// Pixels that differ between two frames by more than `tolerance` in any
/// channel, and the largest difference anywhere.
#[derive(Debug, Clone, Copy)]
struct Diff {
    over: usize,
    max: u8,
}

fn diff(a: &[u8], b: &[u8], tolerance: u8) -> Diff {
    let mut d = Diff { over: 0, max: 0 };
    for (pa, pb) in a.chunks(4).zip(b.chunks(4)) {
        let m = (0..3).map(|c| pa[c].abs_diff(pb[c])).max().unwrap();
        d.max = d.max.max(m);
        if m > tolerance {
            d.over += 1;
        }
    }
    d
}

/// Which pixels differ at all between two frames.
fn mask(a: &[u8], b: &[u8]) -> Vec<bool> {
    a.chunks(4)
        .zip(b.chunks(4))
        .map(|(pa, pb)| pa[..3] != pb[..3])
        .collect()
}

fn save(pixels: &[u8], extent: [u32; 2], name: &str) {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/capture");
    std::fs::create_dir_all(&dir).unwrap();
    image::save_buffer(
        dir.join(format!("surface-{name}.png")),
        pixels,
        extent[0],
        extent[1],
        image::ColorType::Rgba8,
    )
    .unwrap();
}

fn named(values: &[(&str, ParamValue)]) -> Vec<(String, ParamValue)> {
    values.iter().map(|(n, v)| (n.to_string(), *v)).collect()
}

/// How far a function that reproduces a material may land from that
/// material, per channel, in 8-bit sRGB. The twin compiles against every lobe
/// and the reference may be the flat build, and the two are different code
/// for the same arithmetic: rounding, never a visible change.
const ROUNDING: u8 = 2;

fn an_empty_function_draws_the_standard_material(
    make: fn([u32; 2]) -> VulkanRenderer,
    backend: &str,
) {
    let standard = shoot(make, Shape::Block, grey(0.5), nothing, nothing);
    let custom = shoot(make, Shape::Block, grey(0.5), nothing, |renderer, rig| {
        attach(renderer, rig.subject, EMPTY, &[])
    });
    save(&custom, EXTENT, &format!("{backend}-empty"));
    let d = diff(&standard, &custom, ROUNDING);
    println!("{backend}: empty function against the standard material: {d:?}");
    assert_eq!(d.over, 0, "{backend}: {d:?}");
}

fn a_colour_parameter_paints_the_object_and_nothing_else(
    make: fn([u32; 2]) -> VulkanRenderer,
    backend: &str,
) {
    let blue = Vec3::new(0.05, 0.1, 0.8);
    let before = shoot(make, Shape::Block, grey(0.5), nothing, nothing);
    let reference = shoot(make, Shape::Block, grey(0.5), nothing, |renderer, rig| {
        renderer.update_material(
            rig.subject,
            &Material {
                base_color: blue,
                ..grey(0.5)
            },
        )
    });
    let custom = shoot(make, Shape::Block, grey(0.5), nothing, |renderer, rig| {
        attach(
            renderer,
            rig.subject,
            TINT,
            &[("tint", ParamValue::Vec4(blue.extend(1.0).into()))],
        )
    });
    save(&custom, EXTENT, &format!("{backend}-tint"));

    let object = mask(&before, &reference);
    let painted = object.iter().filter(|m| **m).count();
    let d = diff(&custom, &reference, ROUNDING);
    println!("{backend}: tint against a standard blue: {d:?}, object {painted} pixels");
    assert!(painted > 2_000, "{backend}: the object is on screen");
    assert_eq!(
        d.over, 0,
        "{backend}: the function is the blue material: {d:?}"
    );
    let outside = before
        .chunks(4)
        .zip(custom.chunks(4))
        .zip(&object)
        .filter(|((a, b), inside)| !**inside && a[..3] != b[..3])
        .count();
    assert_eq!(outside, 0, "{backend}: nothing else moved");
}

fn the_prepass_reads_the_function_too(make: fn([u32; 2]) -> VulkanRenderer, backend: &str) {
    // Screen-space reflections and ambient occlusion both read the prepass's
    // normal and roughness rather than the forward pass's; the floor is what
    // they act on, and a floor polished in the forward pass alone would not
    // reflect the block.
    let reflections = |enabled| {
        move |_: &mut VulkanRenderer, rig: &mut Rig| {
            rig.world.insert_resource(SsrSettings {
                enabled,
                ..SsrSettings::default()
            })
        }
    };
    let polish = |renderer: &mut VulkanRenderer, rig: &mut Rig| {
        let normal = renderer.load_texture(&[170, 128, 248, 255], 1, 1, false);
        renderer.update_material(
            rig.ground,
            &Material {
                normal_texture: Some(normal),
                ..grey(0.08)
            },
        );
    };
    let rough = shoot(make, Shape::Block, grey(0.5), reflections(true), nothing);
    let reference = shoot(make, Shape::Block, grey(0.5), reflections(true), polish);
    let unreflected = shoot(make, Shape::Block, grey(0.5), reflections(false), polish);
    let custom = shoot(
        make,
        Shape::Block,
        grey(0.5),
        reflections(true),
        |renderer, rig| attach(renderer, rig.ground, POLISHED, &[]),
    );
    save(&reference, EXTENT, &format!("{backend}-polished-reference"));
    save(&custom, EXTENT, &format!("{backend}-polished"));

    let moved = diff(&rough, &reference, ROUNDING);
    let reflected = diff(&reference, &unreflected, ROUNDING);
    let d = diff(&custom, &reference, ROUNDING);
    println!(
        "{backend}: polished floor against its standard twin {d:?}; the polish moved {moved:?}, \
         the reflections alone {reflected:?}"
    );
    assert!(reflected.over > 500, "{backend}: SSR has something to show");
    assert_eq!(d.over, 0, "{backend}: {d:?}");
}

fn a_masked_function_cuts_holes_with_no_seam(make: fn([u32; 2]) -> VulkanRenderer, backend: &str) {
    let mut renderer = make(EXTENT);
    let red = Material {
        base_color: Vec3::new(0.8, 0.05, 0.05),
        blend: BlendMode::Masked,
        ..grey(0.6)
    };
    let mut rig = rig(&mut renderer, Shape::Card, red);
    let solid = rig.draw(&mut renderer);
    let shader = renderer.load_surface_shader(&compiled(STRIPES)).unwrap();
    renderer
        .set_material_surface(rig.subject, Some((shader, &[])))
        .unwrap();
    let cut = rig.draw(&mut renderer);
    save(&solid, EXTENT, &format!("{backend}-card"));
    save(&cut, EXTENT, &format!("{backend}-stripes"));

    let dominant = |p: &[u8], c: usize| {
        let others = (0..3)
            .filter(|o| *o != c)
            .map(|o| u16::from(p[o]))
            .max()
            .unwrap();
        u16::from(p[c]) > 2 * others.max(8)
    };
    let reddish = |p: &[u8]| dominant(p, 0);
    let greenish = |p: &[u8]| dominant(p, 1);
    let card: Vec<bool> = solid.chunks(4).map(reddish).collect();
    let area = card.iter().filter(|c| **c).count();
    // Through a hole is the wall, or the ground below the wall's foot; never
    // the clear colour, which is what a hole the prepass did not cut shows.
    let dark = |p: &[u8]| p[..3].iter().all(|c| *c < 24);
    let (mut kept, mut through, mut neither) = (0, 0, 0);
    for (pixel, inside) in cut.chunks(4).zip(&card) {
        if !inside {
            continue;
        }
        if reddish(pixel) {
            kept += 1;
        } else if dark(pixel) {
            neither += 1;
        } else {
            through += 1;
        }
    }
    let walls = cut
        .chunks(4)
        .zip(&card)
        .filter(|(p, inside)| **inside && greenish(p))
        .count();
    println!(
        "{backend}: card {area} pixels: {kept} kept, {through} show what is behind ({walls} of \
         them the wall), {neither} dark"
    );
    assert!(walls > area / 4, "{backend}: the wall shows through");
    assert!(area > 2_000, "{backend}: the card is on screen");
    let half = |n: usize| n as f32 / area as f32;
    assert!(
        (0.35..0.65).contains(&half(through)),
        "{backend}: the holes show the wall behind them"
    );
    // A prepass that did not cut where the forward pass discards leaves the
    // clear colour in every hole; one that cut a texel away from it leaves a
    // dark seam along every edge. Either is pixels that are neither.
    assert!(
        neither <= area / 100,
        "{backend}: {neither} pixels of the card are neither the card nor the wall"
    );
}

fn the_blended_queues_draw_through_the_function(
    make: fn([u32; 2]) -> VulkanRenderer,
    backend: &str,
) {
    for (blend, name) in [
        (BlendMode::Blend, "blended"),
        (BlendMode::Transmissive, "transmissive"),
    ] {
        let glass = Material {
            base_color: Vec3::splat(0.6),
            alpha: 0.5,
            // Partly: a fully transmissive surface shows only what is behind
            // it, and its albedo would change nothing to compare.
            transmission: if blend == BlendMode::Transmissive {
                0.3
            } else {
                0.0
            },
            blend,
            ..grey(0.2)
        };
        let green = Vec3::new(0.05, 0.7, 0.1);
        let before = shoot(make, Shape::Block, glass, nothing, nothing);
        let reference = shoot(make, Shape::Block, glass, nothing, |renderer, rig| {
            renderer.update_material(
                rig.subject,
                &Material {
                    base_color: green,
                    ..glass
                },
            )
        });
        let custom = shoot(make, Shape::Block, glass, nothing, |renderer, rig| {
            attach(renderer, rig.subject, GREEN, &[])
        });
        save(&before, EXTENT, &format!("{backend}-{name}-before"));
        save(&custom, EXTENT, &format!("{backend}-{name}"));
        let moved = diff(&before, &reference, ROUNDING);
        let d = diff(&custom, &reference, ROUNDING);
        println!("{backend}: {name} against a standard green: {d:?}; green moved {moved:?}");
        assert!(moved.over > 1_000, "{backend}: {name} is on screen");
        assert_eq!(d.over, 0, "{backend}: {name}: {d:?}");
    }
}

fn a_parameter_and_the_time_reach_the_function(
    make: fn([u32; 2]) -> VulkanRenderer,
    backend: &str,
) {
    let mut renderer = make(EXTENT);
    let mut rig = rig(&mut renderer, Shape::Block, grey(0.5));
    let tint = renderer.load_surface_shader(&compiled(TINT)).unwrap();
    let paint = |renderer: &mut VulkanRenderer, rig: &mut Rig, colour: [f32; 4]| {
        renderer
            .set_material_surface(
                rig.subject,
                Some((tint, &named(&[("tint", ParamValue::Vec4(colour))]))),
            )
            .unwrap();
        rig.draw(renderer)
    };
    let red = paint(&mut renderer, &mut rig, [0.8, 0.05, 0.05, 1.0]);
    let green = paint(&mut renderer, &mut rig, [0.05, 0.8, 0.05, 1.0]);
    let changed = diff(&red, &green, ROUNDING);
    println!("{backend}: red against green parameter: {changed:?}");
    assert!(
        changed.over > 2_000,
        "{backend}: the value reached the pixels"
    );

    let at = |time, frames| {
        shoot(
            make,
            Shape::Block,
            grey(0.5),
            move |_, rig| {
                rig.settings.surface_time = Some(time);
                rig.settings.frames = frames;
            },
            |renderer, rig| attach(renderer, rig.subject, CLOCK, &[]),
        )
    };
    // The same time after a different number of frames: what pins it is the
    // time, not the history.
    let first = at(0.25, 3);
    let again = at(0.25, 5);
    let later = at(0.75, 3);
    let same = diff(&first, &again, 0);
    let moved = diff(&first, &later, ROUNDING);
    println!("{backend}: one time twice {same:?}; two times {moved:?}");
    assert_eq!(same.max, 0, "{backend}: a pinned time is deterministic");
    assert!(moved.over > 2_000, "{backend}: the time reached the pixels");
}

/// Corrupt every variant in the way each backend's driver refuses with an
/// error: MSL that does not compile for Metal, and SPIR-V that fails the
/// framing check before it reaches the driver for Vulkan. Never well-framed
/// invalid SPIR-V, which MoltenVK can crash the process on.
fn broken(mut artifact: SurfaceArtifact) -> SurfaceArtifact {
    for variant in &mut artifact.variants {
        variant.shader.msl = "this is not metal".into();
        variant.shader.spirv = b"this is not spir-v".to_vec();
    }
    artifact
}

fn a_swap_takes_effect_and_a_broken_one_changes_nothing(
    make: fn([u32; 2]) -> VulkanRenderer,
    backend: &str,
) {
    let mut renderer = make(EXTENT);
    let mut rig = rig(&mut renderer, Shape::Block, grey(0.5));
    let shader = renderer.load_surface_shader(&compiled(TINT)).unwrap();
    renderer
        .set_material_surface(
            rig.subject,
            Some((
                shader,
                &named(&[("tint", ParamValue::Vec4([0.8, 0.05, 0.05, 1.0]))]),
            )),
        )
        .unwrap();
    let red = rig.draw(&mut renderer);

    // The new shader declares no `tint`, so the material's value is handed
    // back rather than silently dropped.
    let warnings = renderer
        .replace_surface_shader(shader, &compiled(GREEN))
        .unwrap();
    assert_eq!(warnings.len(), 1, "{backend}: {warnings:?}");
    assert_eq!(warnings[0].0, rig.subject);
    let green = rig.draw(&mut renderer);
    save(&green, EXTENT, &format!("{backend}-swapped"));
    let swapped = diff(&red, &green, ROUNDING);
    println!("{backend}: the swap moved {swapped:?}");
    assert!(
        swapped.over > 2_000,
        "{backend}: the next frame drew the new shader"
    );

    let refused = renderer.replace_surface_shader(shader, &broken(compiled(TINT)));
    println!("{backend}: the broken swap: {refused:?}");
    assert!(
        matches!(refused, Err(SurfaceShaderError::Pipeline(_))),
        "{backend}: {refused:?}"
    );
    let after = rig.draw(&mut renderer);
    assert_eq!(
        diff(&green, &after, 0).max,
        0,
        "{backend}: a refused swap leaves the pixels where they were"
    );
    assert!(
        matches!(
            renderer.load_surface_shader(&broken(compiled(TINT))),
            Err(SurfaceShaderError::Pipeline(_))
        ),
        "{backend}: a broken artifact does not load either"
    );
}

fn the_preview_shows_the_function(make: fn([u32; 2]) -> VulkanRenderer, backend: &str) {
    let mut renderer = make([320, 240]);
    let clay = renderer.load_material(&Material {
        base_color: Vec3::new(0.8, 0.08, 0.05),
        roughness: 0.6,
        ..Material::default()
    });
    let preview = |renderer: &mut VulkanRenderer| {
        let mut overlay = orrin_core::gfx::vulkan::OverlayFrame {
            material_previews: [
                Some(Subject {
                    material: clay,
                    orbit: Orbit::default(),
                }),
                None,
                None,
            ],
            ..Default::default()
        };
        let nothing = DrawList::new(&[], &[]);
        renderer.render_with_overlay(
            nothing,
            nothing,
            nothing,
            &[],
            &SceneLighting::default(),
            &Camera::default(),
            &SsaoSettings::default(),
            &ContactShadowSettings::default(),
            &SsrSettings::default(),
            &SubsurfaceSettings::default(),
            &TransparencySettings::default(),
            &RefractionSettings::default(),
            &TaaSettings::default(),
            &MotionBlurSettings::default(),
            &DofSettings::default(),
            &BloomSettings::default(),
            &HdrSettings::default(),
            &EnvironmentSettings::default(),
            &FogSettings::default(),
            1.0 / 60.0,
            &[],
            0,
            None,
            None,
            Some(&mut overlay),
        );
        renderer.read_material_preview(Viewer::Inspector).unwrap()
    };
    let centre = |pixels: &[u8]| {
        let side = orrin_core::gfx::vulkan::material_preview::SIZE;
        let at = ((side / 2 * side + side / 2) * 4) as usize;
        [pixels[at], pixels[at + 1], pixels[at + 2]]
    };
    let red = preview(&mut renderer);
    let shader = renderer.load_surface_shader(&compiled(GREEN)).unwrap();
    renderer
        .set_material_surface(clay, Some((shader, &[])))
        .unwrap();
    let green = preview(&mut renderer);
    let side = orrin_core::gfx::vulkan::material_preview::SIZE;
    save(&green, [side, side], &format!("{backend}-preview"));
    let [r, g, b] = centre(&green);
    println!(
        "{backend}: preview centre {:?} -> {:?}",
        centre(&red),
        [r, g, b]
    );
    assert!(g > 2 * r.max(b), "{backend}: the sphere shows the function");
}

/// The forward pass's other attachment shapes — four samples, the subsurface
/// target, and both at once — each draw through a twin of their own: a
/// function that paints the block green must land on the standard green
/// material's pixels in every one, opaque and cut out alike.
fn every_attachment_shape_draws_through_its_twin(
    make: fn([u32; 2]) -> VulkanRenderer,
    backend: &str,
) {
    let waxy = Material {
        subsurface_color: Vec3::new(0.9, 0.4, 0.3),
        thickness: 0.2,
        ..grey(0.5)
    };
    let cutout = Material {
        blend: BlendMode::Masked,
        ..grey(0.5)
    };
    for (msaa, subsurface) in [(true, false), (false, true), (true, true)] {
        for (subject, kind) in [(waxy, "opaque"), (cutout, "masked")] {
            let shape = move |_: &mut VulkanRenderer, rig: &mut Rig| {
                rig.world.insert_resource(TaaSettings {
                    enabled: false,
                    msaa,
                    ..TaaSettings::default()
                });
                rig.world.insert_resource(SubsurfaceSettings {
                    enabled: subsurface,
                    ..SubsurfaceSettings::default()
                });
            };
            let green = Vec3::new(0.05, 0.7, 0.1);
            let before = shoot(make, Shape::Block, subject, shape, nothing);
            let reference = shoot(make, Shape::Block, subject, shape, |renderer, rig| {
                renderer.update_material(
                    rig.subject,
                    &Material {
                        base_color: green,
                        ..subject
                    },
                )
            });
            let custom = shoot(make, Shape::Block, subject, shape, |renderer, rig| {
                attach(renderer, rig.subject, GREEN, &[])
            });
            let name = format!(
                "{backend}-msaa{}-sss{}-{kind}",
                msaa as u8, subsurface as u8
            );
            save(&custom, EXTENT, &name);
            let moved = diff(&before, &reference, ROUNDING);
            let d = diff(&custom, &reference, ROUNDING);
            println!("{name}: against a standard green {d:?}; green moved {moved:?}");
            assert!(moved.over > 2_000, "{name}: the block is on screen");
            assert_eq!(d.over, 0, "{name}: {d:?}");
        }
    }
}

/// A shader that loads but whose blended twin the driver refuses, on the
/// frame that first needs it: the frame goes on, and the material draws as
/// the standard one it would be without the shader. The refusal is an error
/// on both drivers, as in [`broken`], and only the blended queue's variant
/// carries it, so the load itself — which builds the opaque twins — succeeds.
fn a_twin_that_fails_on_first_use_draws_the_material_as_standard(
    make: fn([u32; 2]) -> VulkanRenderer,
    backend: &str,
) {
    let glass = Material {
        base_color: Vec3::splat(0.6),
        alpha: 0.5,
        blend: BlendMode::Blend,
        ..grey(0.2)
    };
    let mut artifact = compiled(GREEN);
    let mut corrupted = 0;
    for variant in &mut artifact.variants {
        if variant.builtin() == Some(orrin_core::gfx::shaders::ShaderId::OitFrag) {
            variant.shader.msl = "this is not metal".into();
            variant.shader.spirv = b"this is not spir-v".to_vec();
            corrupted += 1;
        }
    }
    assert_eq!(
        corrupted, 1,
        "{backend}: the blended variant is in the artifact"
    );

    let before = shoot(make, Shape::Block, glass, nothing, nothing);
    let green = shoot(make, Shape::Block, glass, nothing, |renderer, rig| {
        attach(renderer, rig.subject, GREEN, &[])
    });
    let failed = shoot(make, Shape::Block, glass, nothing, |renderer, rig| {
        let shader = renderer
            .load_surface_shader(&artifact)
            .expect("the opaque twins build");
        renderer
            .set_material_surface(rig.subject, Some((shader, &[])))
            .unwrap();
    });
    save(&failed, EXTENT, &format!("{backend}-failed-twin"));
    let painted = diff(&before, &green, ROUNDING);
    let d = diff(&failed, &before, ROUNDING);
    println!(
        "{backend}: a failed blended twin against the standard glass {d:?}; green {painted:?}"
    );
    assert!(painted.over > 1_000, "{backend}: the function would show");
    assert_eq!(d.over, 0, "{backend}: drawn as standard: {d:?}");
}

/// What building a surface shader's pipelines costs, cold: a fresh source each
/// run, so no driver cache has seen it.
fn report_build_times(make: fn([u32; 2]) -> VulkanRenderer, backend: &str) {
    let fresh = |salt: u128| {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            % 1_000_000
            + salt;
        compiled(&format!(
            "void surface(in SurfaceIn i, inout SurfaceOut o) {{ o.roughness += {nonce}.0 * 1e-12; }}\n"
        ))
    };
    let started = Instant::now();
    let artifact = fresh(0);
    let compile_ms = started.elapsed().as_secs_f32() * 1e3;
    let replacement = fresh(1_000_000);

    let mut renderer = make(EXTENT);
    let mut rig = rig(&mut renderer, Shape::Block, grey(0.5));
    rig.draw(&mut renderer);
    let started = Instant::now();
    let shader = renderer.load_surface_shader(&artifact).unwrap();
    let load_ms = started.elapsed().as_secs_f32() * 1e3;

    // One more frame of each kind, each building what it lacks on first use.
    let frame = |renderer: &mut VulkanRenderer, rig: &mut Rig| {
        let started = Instant::now();
        rig.draw(renderer);
        started.elapsed().as_secs_f32() * 1e3
    };
    let plain_ms = frame(&mut renderer, &mut rig);
    renderer
        .set_material_surface(rig.subject, Some((shader, &[])))
        .unwrap();
    let opaque_ms = frame(&mut renderer, &mut rig);
    renderer.update_material(
        rig.subject,
        &Material {
            blend: BlendMode::Masked,
            ..grey(0.5)
        },
    );
    let masked_ms = frame(&mut renderer, &mut rig);
    let swap_started = Instant::now();
    renderer
        .replace_surface_shader(shader, &replacement)
        .unwrap();
    let swap_ms = swap_started.elapsed().as_secs_f32() * 1e3;
    println!(
        "{backend}: compile {compile_ms:.0} ms (8 variants); load {load_ms:.0} ms (2 twins: \
         forward, prepass); a frame {plain_ms:.0} ms; first opaque frame {opaque_ms:.0} ms (0 \
         new); first masked frame {masked_ms:.0} ms (2 new: forward and prepass cutouts); \
         replace with another fresh shader {swap_ms:.0} ms (its 4 twins)"
    );
}

fn vulkan(extent: [u32; 2]) -> VulkanRenderer {
    offscreen_renderer(extent)
}

#[cfg(target_os = "macos")]
fn metal(extent: [u32; 2]) -> VulkanRenderer {
    VulkanRenderer::offscreen_metal(extent).expect("a Metal device")
}

macro_rules! on_every_backend {
    ($($case:ident),* $(,)?) => {
        mod vulkan {
            $(
                #[test]
                #[ignore = "needs a GPU"]
                fn $case() {
                    orrin_core::logging::init(orrin_core::logging::LogConfig::for_this_build());
                    super::$case(super::vulkan, "vulkan");
                }
            )*
        }
        #[cfg(target_os = "macos")]
        mod metal {
            $(
                #[test]
                #[ignore = "needs a Metal device"]
                fn $case() {
                    orrin_core::logging::init(orrin_core::logging::LogConfig::for_this_build());
                    super::$case(super::metal, "metal");
                }
            )*
        }
    };
}

on_every_backend!(
    an_empty_function_draws_the_standard_material,
    a_colour_parameter_paints_the_object_and_nothing_else,
    the_prepass_reads_the_function_too,
    a_masked_function_cuts_holes_with_no_seam,
    the_blended_queues_draw_through_the_function,
    a_parameter_and_the_time_reach_the_function,
    a_swap_takes_effect_and_a_broken_one_changes_nothing,
    the_preview_shows_the_function,
    every_attachment_shape_draws_through_its_twin,
    a_twin_that_fails_on_first_use_draws_the_material_as_standard,
    report_build_times,
);
