//! A surface shader through the real compiler, with no GPU: what it binds in
//! every variant, where its errors point, and what it cannot smuggle past the
//! binding check. `gfx::surface`'s inline tests cover the language and the
//! artifact without the compiler.

#![cfg(feature = "shader-compiler")]

use orrin_core::gfx::shaders::{
    CompiledShader, Resource, ResourceKind, SHADERS, SOURCES, SOURCES_HASH, ShaderBlob, ShaderId,
    Stage, compile as compile_glsl,
};
use orrin_core::gfx::surface::{
    PARAMS_BINDING, SurfaceArtifact, SurfaceError, SurfaceShader, compile, variants,
};

const EXAMPLE: &str = "\
param color tint = (1.0, 0.45, 0.2, 1.0);
param float wear = 0.5 : range(0, 1);
param texture mask = white;

void surface(in SurfaceIn i, inout SurfaceOut o) {
    float m = sample_param(mask, i.uv).r * wear;
    o.albedo = mix(o.albedo, tint.rgb, m);
    o.roughness = mix(o.roughness, 1.0, m);
}
";

const EMPTY: &str = "void surface(in SurfaceIn i, inout SurfaceOut o) {}\n";

fn compiled(text: &str) -> SurfaceArtifact {
    let shader = SurfaceShader::parse(text).unwrap_or_else(|e| panic!("{e:#?}"));
    compile(&shader, "rock.oshader").unwrap_or_else(|errors| {
        panic!(
            "{}",
            errors
                .iter()
                .map(|e| e.located("rock.oshader"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    })
}

fn errors(text: &str) -> Vec<SurfaceError> {
    let shader = SurfaceShader::parse(text).unwrap_or_else(|e| panic!("{e:#?}"));
    compile(&shader, "rock.oshader")
        .err()
        .unwrap_or_else(|| panic!("compiled:\n{text}"))
}

fn one_error(text: &str) -> SurfaceError {
    let found = errors(text);
    assert_eq!(found.len(), 1, "{found:#?}");
    found.into_iter().next().unwrap()
}

/// Asserted from the build's own table rather than through the module's check:
/// the variant binds what its built-in binds, plus the parameter buffer exactly
/// as the build's custom-surface row reflected it.
fn assert_binds_like(variant: &CompiledShader, builtin: &ShaderBlob) {
    let params = *ShaderId::PrepassFragCustomSurface
        .blob()
        .bindings
        .resources
        .iter()
        .find(|r| (r.set, r.binding) == PARAMS_BINDING)
        .unwrap();
    assert_eq!(params.kind, ResourceKind::StorageBuffer);
    let mut expected: Vec<Resource> = builtin.bindings.resources.to_vec();
    expected.push(params);
    expected.sort_by_key(|r| (r.set, r.binding));
    assert_eq!(variant.bindings.resources, expected, "{}", builtin.name);
    assert_eq!(
        variant.bindings.push_constant_bytes, builtin.bindings.push_constant_bytes,
        "{}",
        builtin.name
    );
    assert_eq!(variant.bindings.workgroup_size, None);
    assert_eq!(variant.stage, Stage::Fragment);
}

fn assert_compiled_in_every_variant(artifact: &SurfaceArtifact) {
    let keys: Vec<&str> = artifact.variants.iter().map(|v| v.key.as_str()).collect();
    let expected: Vec<&str> = variants().iter().map(|id| id.blob().name).collect();
    assert_eq!(keys, expected);
    for variant in &artifact.variants {
        let builtin = variant.builtin().expect("a built-in shader").blob();
        assert_binds_like(&variant.shader, builtin);
    }
    assert_eq!(artifact.sources_hash, SOURCES_HASH);
    assert!(artifact.matches_this_build());
}

#[test]
fn the_example_compiles_in_every_variant_and_binds_only_its_parameters() {
    let artifact = compiled(EXAMPLE);
    assert_compiled_in_every_variant(&artifact);
    assert_eq!(artifact.params.word_count, 2);
    assert_eq!(SurfaceArtifact::decode(&artifact.encode()), Ok(artifact));
}

#[test]
fn an_empty_function_with_no_parameters_compiles_in_every_variant() {
    let artifact = compiled(EMPTY);
    assert_compiled_in_every_variant(&artifact);
    assert_eq!(artifact.params.word_count, 0);
}

#[test]
fn time_and_every_parameter_type_reach_the_function() {
    let artifact = compiled(
        "param float f = 1;\nparam vec2 v2;\nparam vec3 v3;\nparam vec4 v4;\n\
         param color c;\nparam texture t = normal;\n\
         void surface(in SurfaceIn i, inout SurfaceOut o) {\n\
         \x20   o.albedo = c.rgb * v3 * v4.xyz * f + vec3(v2, sin(i.time));\n\
         \x20   o.normal = sample_param(t, i.uv).xyz;\n\
         }\n",
    );
    assert_compiled_in_every_variant(&artifact);
    let empty = compiled(EMPTY);
    for (used, unused) in artifact.variants.iter().zip(&empty.variants) {
        assert!(used.shader.spirv != unused.shader.spirv, "{}", used.key);
    }
}

#[test]
fn a_users_macros_do_not_reach_the_engines_code() {
    compiled(
        "#define albedo nothing_at_all\n#define roughness(x) x\n\
         void surface(in SurfaceIn i, inout SurfaceOut o) {}\n",
    );
}

#[test]
fn an_error_in_the_function_body_is_reported_at_the_users_line() {
    let error = one_error(
        "void surface(in SurfaceIn i, inout SurfaceOut o) {\n\
         \x20   o.albedo = vec3(1.0);\n\
         \x20   o.roughness = not_a_function(o.albedo);\n\
         }\n",
    );
    assert_eq!(error.line, Some(3), "{error:?}");
    assert!(error.message.contains("not_a_function"), "{error:?}");
    let located = error.located("rock.oshader");
    assert!(located.starts_with("rock.oshader:3: "), "{located}");
}

#[test]
fn an_error_after_blanked_parameter_lines_keeps_its_line() {
    let error = one_error(
        "// a comment\n\
         param float wear = 0.5;\n\
         param color tint;\n\
         \n\
         void surface(in SurfaceIn i, inout SurfaceOut o) {\n\
         \x20   o.albedo = tint.rgb * wear;\n\
         \x20   o.metallic = wear.x.y;\n\
         }\n",
    );
    assert_eq!(error.line, Some(7), "{error:?}");
    assert!(
        error
            .located("rock.oshader")
            .starts_with("rock.oshader:7: ")
    );
}

#[test]
fn a_parameter_named_like_an_engine_global_is_an_error_at_its_declaration() {
    for (name, line) in [
        ("lighting", 2),
        ("materials", 2),
        ("brdf", 2),
        ("shade_surface", 2),
    ] {
        let error = one_error(&format!("param float other;\nparam float {name};\n{EMPTY}"));
        assert_eq!(error.line, Some(line), "{name}: {error:?}");
        assert!(error.message.contains(name), "{name}: {error:?}");
    }
}

#[test]
fn a_parameter_named_like_a_builtin_function_is_mapped_to_its_declaration() {
    let error = one_error(&format!("\nparam float mix;\n{EMPTY}"));
    assert_eq!(error.line, Some(2), "{error:?}");
    assert!(error.message.contains("mix"), "{error:?}");
}

#[test]
fn a_discard_pasted_together_by_a_macro_is_still_rejected() {
    let error = one_error(
        "#define CAT(a, b) a##b\n\
         void surface(in SurfaceIn i, inout SurfaceOut o) {\n\
         \x20   if (o.alpha < 0.5) CAT(dis, card);\n\
         }\n",
    );
    assert!(error.message.contains("discard"), "{error:?}");
}

#[test]
fn a_depth_write_pasted_together_by_a_macro_is_still_rejected() {
    let error = one_error(
        "#define CAT(a, b) a##b\n\
         void surface(in SurfaceIn i, inout SurfaceOut o) {\n\
         \x20   CAT(gl_Frag, Depth) = 0.5;\n\
         }\n",
    );
    assert!(error.message.contains("gl_FragDepth"), "{error:?}");
}

#[test]
fn a_binding_pasted_together_by_a_macro_is_rejected_by_the_binding_check() {
    // Set 4 binding 0 is in the engine's layout, so the translator accepts it;
    // only comparing against the built-in shader's map catches it.
    let error = one_error(
        "#define CAT(a, b) a##b\n\
         CAT(lay, out)(set = 4, binding = 0) CAT(uni, form) Smuggled { vec4 v; } smuggled;\n\
         void surface(in SurfaceIn i, inout SurfaceOut o) {\n\
         \x20   o.albedo = smuggled.v.rgb;\n\
         }\n",
    );
    assert!(error.message.contains("set 4, binding 0"), "{error:?}");
}

/// The variant list is derived, so this is the independent check: a fragment
/// row runs the custom function exactly when compiling it with one that does
/// something changes its SPIR-V. A new non-flat row that calls
/// `resolve_surface` and is missing from `variants()` fails here.
#[test]
fn the_variants_are_exactly_the_manifest_rows_that_run_the_custom_function() {
    let source_of = |blob: &ShaderBlob| SOURCES.iter().find(|(p, _)| *p == blob.path).unwrap().1;
    let with = |blob: &ShaderBlob, surface: &str| {
        let mut defines = blob.defines.to_vec();
        defines.push(("ORRIN_CUSTOM_SURFACE", "1"));
        compile_glsl(
            source_of(blob),
            blob.stage,
            blob.path,
            &defines,
            &[("custom_surface.glsl", surface)],
        )
        .unwrap_or_else(|e| panic!("{e}"))
        .spirv
    };
    let doing =
        "void surface(in SurfaceIn i, inout SurfaceOut o) { o.albedo = sin(i.world_position); }\n";

    let listed = variants();
    for &id in ShaderId::ALL {
        let blob = id.blob();
        let flat = blob.defines.iter().any(|(k, _)| *k == "ORRIN_FLAT_SURFACE");
        let custom = blob
            .defines
            .iter()
            .any(|(k, _)| *k == "ORRIN_CUSTOM_SURFACE");
        if blob.stage != Stage::Fragment || custom {
            assert!(!listed.contains(&id), "{}", blob.name);
            continue;
        }
        let runs = with(blob, EMPTY) != with(blob, doing);
        assert_eq!(
            listed.contains(&id),
            runs && !flat,
            "{}: runs the custom function: {runs}, flat: {flat}",
            blob.name
        );
    }

    let names: Vec<_> = listed.iter().map(|id| id.blob().name).collect();
    assert_eq!(
        names,
        [
            "forward_frag",
            "forward_frag_alpha_test",
            "forward_sss_frag",
            "forward_sss_frag_alpha_test",
            "oit_frag",
            "prepass_frag",
            "prepass_frag_masked",
            "refraction_frag",
        ]
    );
    assert!(SHADERS.len() > names.len());
}
