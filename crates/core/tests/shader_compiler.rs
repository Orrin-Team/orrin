//! The runtime shader compiler is the build's compiler, not a second one.
//!
//! `gfx::shaders::compile` goes through `orrin-shaderc` exactly as `build.rs`
//! does, against the sources and the layout the build embedded. If the two ever
//! disagree, a shader compiled in the editor binds differently from the
//! built-in shaders it is drawn beside, and nothing but these tests would say
//! so: they take no GPU, so CI runs them.

#![cfg(feature = "shader-compiler")]

use std::fs;
use std::path::Path;

use orrin_core::gfx::shaders::{
    CompiledShader, LAYOUT_WIDTHS, Resource, ResourceKind, SHADERS, SOURCES, SOURCES_HASH,
    ShaderBlob, ShaderId, Stage, compile,
};

fn source_of(blob: &ShaderBlob) -> &'static str {
    SOURCES
        .iter()
        .find(|(path, _)| *path == blob.path)
        .unwrap_or_else(|| panic!("{} is not embedded", blob.path))
        .1
}

fn assert_compiles_as_the_build_did(blob: &ShaderBlob) {
    let compiled = compile(source_of(blob), blob.stage, blob.path, blob.defines, &[])
        .unwrap_or_else(|e| panic!("{}: {e}", blob.name));
    assert!(
        compiled.spirv == blob.spirv,
        "{}: the SPIR-V differs from the build's",
        blob.name
    );
    assert!(
        compiled.msl == blob.msl,
        "{}: the MSL differs from the build's",
        blob.name
    );
    assert_eq!(
        compiled.bindings.resources, blob.bindings.resources,
        "{}",
        blob.name
    );
    assert_eq!(
        compiled.bindings.push_constant_bytes, blob.bindings.push_constant_bytes,
        "{}",
        blob.name
    );
    assert_eq!(
        compiled.bindings.workgroup_size, blob.bindings.workgroup_size,
        "{}",
        blob.name
    );
    assert_eq!(compiled.stage, blob.stage);
    let defines: Vec<_> = blob
        .defines
        .iter()
        .map(|&(key, value)| (key.to_string(), value.to_string()))
        .collect();
    assert_eq!(compiled.defines, defines);
}

#[test]
fn the_shaders_a_surface_shader_is_stitched_into_compile_as_the_build_did() {
    for id in [
        ShaderId::ForwardFrag,
        ShaderId::PrepassFragMasked,
        ShaderId::ShadowVert,
        ShaderId::FogScatterComp,
    ] {
        assert_compiles_as_the_build_did(id.blob());
    }
}

#[test]
fn every_built_in_shader_compiles_as_the_build_did() {
    for blob in SHADERS {
        assert_compiles_as_the_build_did(blob);
    }
}

#[test]
fn a_virtual_include_overrides_the_embedded_file_of_the_same_name() {
    let source = "#version 450\n#extension GL_GOOGLE_include_directive : require\n\
                  #include \"depth.glsl\"\n\
                  layout(location = 0) out vec4 o;\n\
                  void main() { o = vec4(only_in_the_virtual_file()); }\n";
    let error = compile(source, Stage::Fragment, "probe.frag", &[], &[]).unwrap_err();
    assert!(error.contains("only_in_the_virtual_file"), "{error}");

    let virtual_depth = "float only_in_the_virtual_file() { return 1.0; }\n";
    compile(
        source,
        Stage::Fragment,
        "probe.frag",
        &[],
        &[("depth.glsl", virtual_depth)],
    )
    .unwrap();
}

#[test]
fn a_compile_error_names_the_file_it_was_given() {
    let error = compile(
        "#version 450\nvoid main() { nope(); }\n",
        Stage::Fragment,
        "materials/rock.oshader",
        &[],
        &[],
    )
    .unwrap_err();
    assert!(error.contains("materials/rock.oshader"), "{error}");
    assert!(error.contains("nope"), "{error}");
}

/// Every manifest row a custom surface function is stitched into: each variant
/// of the forward, transparency and refraction passes and of the prepass. The
/// masked shadow pass tests alpha at the unmarched coordinate and is not one,
/// and the row that already compiles the custom path is not stitched into.
fn surface_passes() -> Vec<&'static ShaderBlob> {
    const FILES: [&str; 5] = [
        "forward.frag",
        "forward_sss.frag",
        "oit.frag",
        "refraction.frag",
        "prepass.frag",
    ];
    let passes: Vec<_> = SHADERS
        .iter()
        .filter(|blob| FILES.contains(&blob.path))
        .filter(|blob| {
            !blob
                .defines
                .iter()
                .any(|(key, _)| *key == "ORRIN_CUSTOM_SURFACE")
        })
        .collect();
    for id in [
        ShaderId::ForwardFrag,
        ShaderId::ForwardSssFrag,
        ShaderId::OitFrag,
        ShaderId::RefractionFrag,
        ShaderId::PrepassFrag,
        ShaderId::PrepassFragMasked,
    ] {
        assert!(passes.iter().any(|blob| blob.name == id.blob().name));
    }
    passes
}

/// `blob` with its own manifest defines plus `ORRIN_CUSTOM_SURFACE`, and
/// `surface` standing in as `custom_surface.glsl`.
fn compile_with_surface(blob: &ShaderBlob, surface: &str) -> Result<CompiledShader, String> {
    let mut defines = blob.defines.to_vec();
    defines.push(("ORRIN_CUSTOM_SURFACE", "1"));
    compile(
        source_of(blob),
        blob.stage,
        blob.path,
        &defines,
        &[("custom_surface.glsl", surface)],
    )
}

const EMPTY_SURFACE: &str = "void surface(in SurfaceIn i, inout SurfaceOut o) {}\n";

/// What `blob` binds, plus the parameter buffer every custom surface reads its
/// time from, as the build's own custom-surface row reflected it.
fn with_surface_params(blob: &ShaderBlob) -> Vec<Resource> {
    let params = *ShaderId::PrepassFragCustomSurface
        .blob()
        .bindings
        .resources
        .iter()
        .find(|r| (r.set, r.binding) == (5, 2))
        .expect("the custom-surface row binds SurfaceParams");
    assert_eq!(params.kind, ResourceKind::StorageBuffer);
    let mut resources = blob.bindings.resources.to_vec();
    resources.push(params);
    resources
}

#[test]
fn an_empty_surface_function_binds_what_the_built_in_shader_binds_and_the_parameters() {
    for blob in surface_passes() {
        let compiled = compile_with_surface(blob, EMPTY_SURFACE)
            .unwrap_or_else(|e| panic!("{}: {e}", blob.name));
        assert_eq!(
            compiled.bindings.resources,
            with_surface_params(blob),
            "{}",
            blob.name
        );
        assert_eq!(
            compiled.bindings.push_constant_bytes, blob.bindings.push_constant_bytes,
            "{}",
            blob.name
        );
        assert_eq!(
            compiled.bindings.workgroup_size, blob.bindings.workgroup_size,
            "{}",
            blob.name
        );
    }
}

#[test]
fn a_surface_function_that_reads_every_input_and_writes_every_output_compiles_in_every_pass() {
    let surface = "void surface(in SurfaceIn i, inout SurfaceOut o) {\n\
                   \x20   float facing = dot(i.world_normal, i.view_dir);\n\
                   \x20   o.albedo = i.vertex_color * fract(i.world_position) * facing;\n\
                   \x20   o.alpha = i.uv.x;\n\
                   \x20   o.metallic = i.uv.y;\n\
                   \x20   o.roughness = 0.5;\n\
                   \x20   o.emissive = vec3(facing);\n\
                   \x20   o.normal = vec3(0.0, 0.0, 1.0);\n\
                   }\n";
    for blob in surface_passes() {
        let compiled =
            compile_with_surface(blob, surface).unwrap_or_else(|e| panic!("{}: {e}", blob.name));
        let empty = compile_with_surface(blob, EMPTY_SURFACE).unwrap();
        assert!(
            compiled.spirv != empty.spirv,
            "{}: the surface function did not reach the shader",
            blob.name
        );
        assert_eq!(
            compiled.bindings.resources,
            with_surface_params(blob),
            "{}",
            blob.name
        );
    }
}

#[test]
fn every_surface_pass_compiles_the_custom_function_it_is_given() {
    let broken = "void surface(in SurfaceIn i, inout SurfaceOut o) { not_a_builtin(); }\n";
    for blob in surface_passes() {
        let error = compile_with_surface(blob, broken)
            .err()
            .unwrap_or_else(|| panic!("{}: compiled without the surface function", blob.name));
        assert!(
            error.contains("custom_surface.glsl"),
            "{}: {error}",
            blob.name
        );
        assert!(error.contains("not_a_builtin"), "{}: {error}", blob.name);
    }
}

#[test]
fn every_file_under_shaders_is_embedded_and_hashed() {
    let shaders = Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let mut on_disk: Vec<String> = fs::read_dir(&shaders)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| !name.starts_with('.'))
        .collect();
    on_disk.sort();
    let embedded: Vec<&str> = SOURCES.iter().map(|(path, _)| *path).collect();
    assert_eq!(embedded, on_disk);
    assert_eq!(
        source_of(ShaderId::ForwardFrag.blob()),
        fs::read_to_string(shaders.join("forward.frag")).unwrap()
    );

    assert_eq!(
        SOURCES_HASH,
        orrin_shaderc::sources_hash(SOURCES.iter().copied(), LAYOUT_WIDTHS.iter().copied())
    );
}
