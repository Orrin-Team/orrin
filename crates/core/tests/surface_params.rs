//! Surface shader parameters through the headless backend: the bookkeeping a
//! renderer does for them — which shader a material reads through, its values
//! by name, the packed block and where it starts — with no device. The words
//! here are the words the real renderer uploads; only the pipelines are
//! missing. `surface_render.rs` is the GPU half.

use orrin_core::gfx::shaders::{CompiledShader, SOURCES_HASH};
use orrin_core::gfx::surface::{
    CompiledVariant, DRAWN_VARIANTS, DefaultTexture, Ignored, IgnoredReason, ParamValue,
    SurfaceArtifact, SurfaceClock, SurfaceShader, SurfaceShaderError,
};
use orrin_core::gfx::vulkan::MAX_SURFACE_SHADERS;
use orrin_core::gfx::{HeadlessBackend, Material, RenderBackend};
use orrin_core::scene::MaterialHandle;

/// An artifact with `text`'s parameters and the built-in shaders standing in
/// for its variants, which is all a backend that builds nothing reads.
fn artifact(text: &str) -> SurfaceArtifact {
    let shader = SurfaceShader::parse(text).unwrap_or_else(|e| panic!("{e:#?}"));
    SurfaceArtifact {
        sources_hash: SOURCES_HASH,
        params: shader.params().clone(),
        variants: DRAWN_VARIANTS
            .iter()
            .map(|id| CompiledVariant {
                key: id.blob().name.to_string(),
                shader: CompiledShader::from(id.blob()),
            })
            .collect(),
    }
}

const BODY: &str = "void surface(in SurfaceIn i, inout SurfaceOut o) {}\n";

fn tint_and_wear() -> SurfaceArtifact {
    artifact(&format!(
        "param color tint = (1, 0, 0, 1);\nparam float wear = 0.5;\n{BODY}"
    ))
}

fn bits(values: [f32; 4]) -> [u32; 4] {
    values.map(f32::to_bits)
}

fn named(values: &[(&str, ParamValue)]) -> Vec<(String, ParamValue)> {
    values
        .iter()
        .map(|(name, value)| (name.to_string(), *value))
        .collect()
}

#[test]
fn an_artifact_from_another_build_or_missing_a_variant_is_refused() {
    let mut backend = HeadlessBackend::new();
    let mut foreign = tint_and_wear();
    foreign.sources_hash ^= 1;
    assert_eq!(
        backend.load_surface_shader(&foreign),
        Err(SurfaceShaderError::ForeignBuild)
    );
    let mut short = tint_and_wear();
    short.variants.retain(|v| v.key != "oit_frag");
    assert_eq!(
        backend.load_surface_shader(&short),
        Err(SurfaceShaderError::MissingVariant("oit_frag"))
    );
    assert_eq!(backend.surface_buffer().loaded(), 0);
}

#[test]
fn bases_are_assigned_after_the_frame_word_and_the_blocks_hold_the_values() {
    let mut backend = HeadlessBackend::new();
    let plain = backend.load_material(&Material::default());
    let rock = backend.load_material(&Material::default());
    let clay = backend.load_material(&Material::default());
    let shader = backend.load_surface_shader(&tint_and_wear()).unwrap();
    assert_eq!(
        backend.surface_buffer().words().len(),
        1,
        "only the frame's"
    );

    let ignored = backend
        .set_material_surface(
            clay,
            Some((shader, &named(&[("wear", ParamValue::Float(0.25))]))),
        )
        .unwrap();
    assert!(ignored.is_empty());
    backend
        .set_material_surface(rock, Some((shader, &[])))
        .unwrap();

    let book = backend.surface_buffer();
    assert_eq!(book.base_of(plain), None);
    // Material order, so the rock's block comes first whatever was attached
    // first. Two words each: the colour, then the float.
    assert_eq!(book.base_of(rock), Some(1));
    assert_eq!(book.base_of(clay), Some(3));
    let words = book.words();
    assert_eq!(words.len(), 5);
    assert_eq!(words[1], bits([1.0, 0.0, 0.0, 1.0]));
    assert_eq!(words[2][0], 0.5f32.to_bits(), "the rock has the default");
    assert_eq!(words[4][0], 0.25f32.to_bits(), "the clay has its value");
    assert_eq!(book.shader_of(clay), Some(shader));
}

#[test]
fn a_replace_that_moves_the_layout_repacks_every_material_from_its_names() {
    let mut backend = HeadlessBackend::new();
    let rock = backend.load_material(&Material::default());
    let clay = backend.load_material(&Material::default());
    let shader = backend.load_surface_shader(&tint_and_wear()).unwrap();
    backend
        .set_material_surface(
            rock,
            Some((
                shader,
                &named(&[
                    ("tint", ParamValue::Vec4([0.0, 1.0, 0.0, 1.0])),
                    ("wear", ParamValue::Float(0.75)),
                ]),
            )),
        )
        .unwrap();
    backend
        .set_material_surface(
            clay,
            Some((shader, &named(&[("wear", ParamValue::Float(0.1))]))),
        )
        .unwrap();

    // `wear` moves to the front and changes type, `tint` is renamed, and a
    // texture appears between them.
    let replaced = artifact(&format!(
        "param vec2 wear;\nparam texture mask = black;\nparam color tone;\n{BODY}"
    ));
    let warnings = backend.replace_surface_shader(shader, &replaced).unwrap();
    assert_eq!(
        warnings,
        vec![
            (
                rock,
                Ignored {
                    name: "tint".into(),
                    reason: IgnoredReason::Undeclared
                }
            ),
            (
                rock,
                Ignored {
                    name: "wear".into(),
                    reason: IgnoredReason::WrongType {
                        declared: "vec2",
                        given: "float"
                    }
                }
            ),
            (
                clay,
                Ignored {
                    name: "wear".into(),
                    reason: IgnoredReason::WrongType {
                        declared: "vec2",
                        given: "float"
                    }
                }
            ),
        ]
    );
    let book = backend.surface_buffer();
    assert_eq!(backend.surface_params(shader).unwrap().word_count, 2);
    assert_eq!((book.base_of(rock), book.base_of(clay)), (Some(1), Some(3)));
    let black = DefaultTexture::Black as u32;
    assert_eq!(book.words()[1], [0, 0, black, 0], "defaults, repacked");
    assert_eq!(book.words()[2], bits([1.0; 4]));

    // A value under the new name lands in the new slot.
    backend
        .set_material_surface(
            rock,
            Some((shader, &named(&[("wear", ParamValue::Vec2([2.0, 3.0]))]))),
        )
        .unwrap();
    assert_eq!(
        backend.surface_buffer().words()[1],
        [2.0f32.to_bits(), 3.0f32.to_bits(), black, 0]
    );
}

#[test]
fn a_failed_replace_changes_nothing() {
    let mut backend = HeadlessBackend::new();
    let rock = backend.load_material(&Material::default());
    let shader = backend.load_surface_shader(&tint_and_wear()).unwrap();
    backend
        .set_material_surface(rock, Some((shader, &[])))
        .unwrap();
    let before = backend.surface_buffer().words().to_vec();

    let mut foreign = artifact(&format!("param vec4 other;\n{BODY}"));
    foreign.sources_hash ^= 1;
    assert_eq!(
        backend.replace_surface_shader(shader, &foreign),
        Err(SurfaceShaderError::ForeignBuild)
    );
    assert_eq!(backend.surface_buffer().words(), &before[..]);
    assert_eq!(backend.surface_params(shader).unwrap().word_count, 2);
}

#[test]
fn detaching_closes_the_gap_and_the_next_material_moves_down() {
    let mut backend = HeadlessBackend::new();
    let rock = backend.load_material(&Material::default());
    let clay = backend.load_material(&Material::default());
    let shader = backend.load_surface_shader(&tint_and_wear()).unwrap();
    for material in [rock, clay] {
        backend
            .set_material_surface(material, Some((shader, &[])))
            .unwrap();
    }
    assert_eq!(backend.surface_buffer().words().len(), 5);

    backend.set_material_surface(rock, None).unwrap();
    let book = backend.surface_buffer();
    assert_eq!(book.base_of(rock), None);
    assert_eq!(book.shader_of(rock), None);
    assert_eq!(book.base_of(clay), Some(1));
    assert_eq!(book.words().len(), 3, "no space is left behind");

    // Detaching what has nothing attached, or a material never loaded with
    // nothing to attach, is a no-op rather than an error.
    backend.set_material_surface(rock, None).unwrap();
}

#[test]
fn attaching_an_unknown_shader_or_to_an_unknown_material_is_an_error() {
    let mut backend = HeadlessBackend::new();
    let rock = backend.load_material(&Material::default());
    let shader = backend.load_surface_shader(&tint_and_wear()).unwrap();
    let stranger = orrin_core::gfx::surface::SurfaceShaderHandle(shader.0 + 1);
    assert_eq!(
        backend.set_material_surface(rock, Some((stranger, &[]))),
        Err(SurfaceShaderError::UnknownShader(stranger))
    );
    assert_eq!(
        backend.set_material_surface(MaterialHandle(9), Some((shader, &[]))),
        Err(SurfaceShaderError::UnknownMaterial(MaterialHandle(9)))
    );
    assert_eq!(backend.surface_buffer().words().len(), 1);
}

#[test]
fn one_more_than_the_renderer_can_draw_is_refused() {
    let mut backend = HeadlessBackend::new();
    let shader = tint_and_wear();
    for _ in 0..MAX_SURFACE_SHADERS {
        backend.load_surface_shader(&shader).unwrap();
    }
    assert_eq!(
        backend.load_surface_shader(&shader),
        Err(SurfaceShaderError::TooMany)
    );
}

#[test]
fn the_clock_advances_wraps_and_pins() {
    let mut clock = SurfaceClock::default();
    assert_eq!(clock.tick(0.5), 0.5);
    assert_eq!(clock.tick(0.25), 0.75);
    let mut long = SurfaceClock::default();
    for _ in 0..3601 {
        long.tick(1.0);
    }
    assert_eq!(long.tick(0.0), 1.0, "wrapped at the period");
    clock.pin(Some(2.0));
    assert_eq!(clock.tick(10.0), 2.0);
    clock.pin(None);
    assert_eq!(
        clock.tick(0.0),
        10.75,
        "the clock kept running under the pin"
    );
}

#[test]
fn every_drawn_variant_is_one_the_compiler_builds() {
    #[cfg(feature = "shader-compiler")]
    {
        let mut drawn: Vec<_> = DRAWN_VARIANTS.to_vec();
        let mut compiled = orrin_core::gfx::surface::variants();
        drawn.sort_by_key(|id| *id as usize);
        compiled.sort_by_key(|id| *id as usize);
        assert_eq!(drawn, compiled, "a variant is drawn by nothing, or missing");
    }
}
