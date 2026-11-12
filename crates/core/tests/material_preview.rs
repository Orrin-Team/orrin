//! The inspector's material sphere, rendered through the real renderer and read
//! back, on every backend this machine has.
//!
//! `#[ignore]`d because it needs a GPU. The device-free half — the rig, the
//! meter, when it redraws, where it sits in the graph — is asserted in CI by
//! the unit tests in `gfx::vulkan::material_preview` and by `render_graph.rs`;
//! this is the half that only pixels can answer: that the sphere is there, is
//! the material, is lit from the key's side, has a soft edge over a clear
//! background, follows an edit, and that two views each keep their own.
//!
//! ```text
//! cargo test -p orrin-core --test material_preview -- --ignored --nocapture
//! ```
//!
//! Each run also writes the spheres to `target/capture/material-preview-*.png`.

use std::path::PathBuf;

use glam::Vec3;
use orrin_core::capture::offscreen_renderer;
use orrin_core::gfx::vulkan::material_preview::{self, Orbit, Subject, Subjects, Viewer};
use orrin_core::gfx::vulkan::{OverlayFrame, VulkanRenderer};
use orrin_core::gfx::{DrawList, Material, RenderBackend, SceneLighting};
use orrin_core::scene::{
    BloomSettings, Camera, ContactShadowSettings, DofSettings, EnvironmentSettings, FogSettings,
    HdrSettings, MaterialHandle, MotionBlurSettings, RefractionSettings, SsaoSettings, SsrSettings,
    SubsurfaceSettings, TaaSettings, TransparencySettings,
};

const SIDE: u32 = material_preview::SIZE;

fn draw(renderer: &mut VulkanRenderer, material: MaterialHandle) -> Vec<u8> {
    let subject = Subject {
        material,
        orbit: Orbit::default(),
    };
    frame(renderer, [Some(subject), None, None]);
    renderer
        .read_material_preview(Viewer::Inspector)
        .expect("a frame that asked for the preview leaves one to read")
}

fn frame(renderer: &mut VulkanRenderer, subjects: Subjects) {
    let mut overlay = OverlayFrame {
        material_previews: subjects,
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
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let at = ((y * SIDE + x) * 4) as usize;
    pixels[at..at + 4].try_into().unwrap()
}

fn brightness([r, g, b, _]: [u8; 4]) -> u32 {
    u32::from(r) + u32::from(g) + u32::from(b)
}

fn save(pixels: &[u8], name: &str) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/capture");
    std::fs::create_dir_all(&dir).unwrap();
    image::save_buffer(
        dir.join(format!("material-preview-{name}.png")),
        pixels,
        SIDE,
        SIDE,
        image::ColorType::Rgba8,
    )
    .unwrap();
}

fn check(mut renderer: VulkanRenderer, backend: &str) {
    let clay = renderer.load_material(&Material {
        base_color: Vec3::new(0.8, 0.08, 0.05),
        metallic: 0.0,
        roughness: 0.6,
        ..Material::default()
    });

    let red = draw(&mut renderer, clay);
    save(&red, &format!("{backend}-red"));
    let centre = pixel(&red, SIDE / 2, SIDE / 2);
    assert_eq!(centre[3], 255, "{backend}: the sphere covers the centre");
    assert!(
        centre[0] > 2 * centre[1].max(centre[2]),
        "{backend}: a red material reads red: {centre:?}"
    );
    for (x, y) in [(0, 0), (SIDE - 1, 0), (0, SIDE - 1), (SIDE - 1, SIDE - 1)] {
        assert_eq!(
            pixel(&red, x, y)[3],
            0,
            "{backend}: the corner ({x}, {y}) is clear"
        );
    }
    let soft = red.chunks(4).filter(|p| p[3] > 0 && p[3] < 255).count();
    assert!(
        soft > 50,
        "{backend}: the silhouette is antialiased: {soft} edge pixels"
    );

    // The key is above and to the left of the camera.
    let quarter = SIDE / 4;
    let lit = pixel(&red, SIDE / 2 - quarter / 2, SIDE / 2 - quarter / 2);
    let shaded = pixel(&red, SIDE / 2 + quarter, SIDE / 2 + quarter);
    assert!(
        brightness(lit) > brightness(shaded),
        "{backend}: lit {lit:?} against shaded {shaded:?}"
    );

    renderer.update_material(
        clay,
        &Material {
            base_color: Vec3::new(0.05, 0.1, 0.8),
            metallic: 0.0,
            roughness: 0.6,
            ..Material::default()
        },
    );
    let blue = draw(&mut renderer, clay);
    save(&blue, &format!("{backend}-edited"));
    let centre = pixel(&blue, SIDE / 2, SIDE / 2);
    assert!(
        centre[2] > 2 * centre[0].max(centre[1]),
        "{backend}: the edit reached the sphere: {centre:?}"
    );

    let again = draw(&mut renderer, clay);
    assert_eq!(
        again, blue,
        "{backend}: nothing changed, so neither did the sphere"
    );

    // Two views at once: a frame redraws one, so both are due for two frames,
    // and each is left holding its own material.
    let moss = renderer.load_material(&Material {
        base_color: Vec3::new(0.06, 0.7, 0.08),
        metallic: 0.0,
        roughness: 0.6,
        ..Material::default()
    });
    let at = |material| {
        Some(Subject {
            material,
            orbit: Orbit::default(),
        })
    };
    for _ in 0..2 {
        frame(&mut renderer, [at(clay), at(moss), None]);
    }
    let inspector = renderer.read_material_preview(Viewer::Inspector).unwrap();
    let browser = renderer.read_material_preview(Viewer::Browser).unwrap();
    save(&browser, &format!("{backend}-browser"));
    let [_, g, b, _] = pixel(&inspector, SIDE / 2, SIDE / 2);
    assert!(b > 2 * g, "{backend}: the inspector kept its material");
    let [r, g, b, _] = pixel(&browser, SIDE / 2, SIDE / 2);
    assert!(
        g > 2 * r.max(b),
        "{backend}: the browser has its own: {:?}",
        [r, g, b]
    );
}

#[test]
#[ignore = "needs a GPU"]
fn the_preview_draws_the_material_on_vulkan() {
    check(offscreen_renderer([320, 240]), "vulkan");
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "needs a GPU"]
fn the_preview_draws_the_material_on_metal() {
    check(
        VulkanRenderer::offscreen_metal([320, 240]).expect("a Metal device"),
        "metal",
    );
}
