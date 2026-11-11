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
    LAYOUT_WIDTHS, SHADERS, SOURCES, SOURCES_HASH, ShaderBlob, ShaderId, Stage, compile,
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
