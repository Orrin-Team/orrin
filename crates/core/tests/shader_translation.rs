//! Every shader the engine compiles is also translated to MSL, with a binding
//! map, by `build.rs` (docs/metal-backend-plan.md M2).
//!
//! Nothing consumes the output yet. These tests are what stops a shader that
//! only translates on the machine that wrote it from being the first thing a
//! Metal driver finds out: they take no GPU, so Linux CI runs them, and the one
//! that needs Metal is `#[ignore]`d.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use orrin_core::gfx::MAX_TEXTURES;
use orrin_core::gfx::shaders::{
    MAX_DESCRIPTOR_SETS, PUSH_CONSTANT_BUFFER, ResourceKind, SHADERS, ShaderBlob, Stage,
    VERTEX_STREAM_BASE,
};

/// What the Rust sources ask `vulkano_shaders::shader!` for: the shader file,
/// its stage and its defines. Parsed from the text, so a pass added without a
/// manifest row is a failure here rather than a shader Metal never sees.
/// (shader file, stage, sorted defines)
type Invocation = (String, String, Vec<(String, String)>);

fn invoked_shaders() -> BTreeSet<Invocation> {
    fn walk(dir: &Path, out: &mut Vec<String>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(fs::read_to_string(&path).unwrap());
            }
        }
    }
    let mut sources = Vec::new();
    walk(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut sources,
    );

    let mut found = BTreeSet::new();
    for source in &sources {
        for (at, _) in source.match_indices("shader!") {
            let Some(open) = source[at..].find('{') else {
                continue;
            };
            let start = at + open + 1;
            let mut depth = 1;
            let mut end = start;
            for (offset, c) in source[start..].char_indices() {
                match c {
                    '{' => depth += 1,
                    '}' => depth -= 1,
                    _ => {}
                }
                if depth == 0 {
                    end = start + offset;
                    break;
                }
            }
            let body = &source[start..end];
            let quoted = |key: &str| -> Option<String> {
                let from = body.find(key)? + key.len();
                let rest = &body[from..];
                let open = rest.find('"')? + 1;
                let close = rest[open..].find('"')? + open;
                Some(rest[open..close].to_string())
            };
            let (Some(stage), Some(path)) = (quoted("ty:"), quoted("path:")) else {
                continue;
            };
            let mut defines = Vec::new();
            if let Some(from) = body.find("define:") {
                let list = &body[from..];
                let list = &list[..list.find(']').unwrap_or(list.len())];
                let mut parts = list.split('"').skip(1).step_by(2);
                while let (Some(name), Some(value)) = (parts.next(), parts.next()) {
                    defines.push((name.to_string(), value.to_string()));
                }
            }
            defines.sort();
            found.insert((
                path.trim_start_matches("shaders/").to_string(),
                stage,
                defines,
            ));
        }
    }
    found
}

fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Vertex => "vertex",
        Stage::Fragment => "fragment",
        Stage::Compute => "compute",
    }
}

#[test]
fn the_manifest_covers_every_shader_the_passes_compile() {
    let invoked = invoked_shaders();
    assert!(
        !invoked.is_empty(),
        "found no shader! invocations to compare"
    );
    let translated: BTreeSet<_> = SHADERS
        .iter()
        .map(|shader| {
            let mut defines: Vec<_> = shader
                .defines
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            defines.sort();
            (
                shader.path.to_string(),
                stage_name(shader.stage).to_string(),
                defines,
            )
        })
        .collect();
    let missing: Vec<_> = invoked.difference(&translated).collect();
    let stale: Vec<_> = translated.difference(&invoked).collect();
    assert!(
        missing.is_empty() && stale.is_empty(),
        "compiled but not translated: {missing:#?}\ntranslated but never compiled: {stale:#?}",
    );
}

#[test]
fn every_entry_produced_spirv_and_msl() {
    assert!(SHADERS.len() > 50, "only {} shaders", SHADERS.len());
    for shader in SHADERS {
        assert_eq!(
            shader.spirv.get(..4),
            Some(&0x0723_0203u32.to_le_bytes()[..]),
            "{} is not SPIR-V",
            shader.name
        );
        assert!(
            shader.msl.contains("#include <metal_stdlib>"),
            "{} produced no MSL",
            shader.name
        );
    }
}

/// Every `[[buffer(n)]]` the translator wrote.
fn msl_buffer_indices(shader: &ShaderBlob) -> BTreeSet<u32> {
    shader
        .msl
        .match_indices("[[buffer(")
        .map(|(at, marker)| {
            let digits = &shader.msl[at + marker.len()..];
            digits[..digits.find(')').unwrap()].parse().unwrap()
        })
        .collect()
}

#[test]
fn the_buffer_index_assignment_holds() {
    const {
        assert!(PUSH_CONSTANT_BUFFER >= MAX_DESCRIPTOR_SETS);
        assert!(VERTEX_STREAM_BASE > PUSH_CONSTANT_BUFFER);
        assert!(
            VERTEX_STREAM_BASE < 31,
            "Metal has 31 buffer slots per stage"
        );
    }

    for shader in SHADERS {
        let map = &shader.bindings;
        assert!(
            map.push_constant_bytes <= 4096,
            "{}: {} bytes of push constants",
            shader.name,
            map.push_constant_bytes
        );

        let sets: BTreeSet<u32> = map.resources.iter().map(|r| r.set).collect();
        let mut expected = sets.clone();
        for &set in &sets {
            assert!(
                set < MAX_DESCRIPTOR_SETS,
                "{}: descriptor set {set} has no argument buffer slot",
                shader.name
            );
        }
        if map.push_constant_bytes > 0 {
            expected.insert(PUSH_CONSTANT_BUFFER);
        }

        // The translator may bind a set it proved unused away, never one the map
        // omits: the MSL's buffer indices must all be accounted for.
        let used = msl_buffer_indices(shader);
        let unexplained: Vec<_> = used.difference(&expected).collect();
        assert!(
            unexplained.is_empty(),
            "{}: MSL binds buffer(s) {unexplained:?} the binding map does not describe",
            shader.name
        );
    }
}

#[test]
fn the_material_table_reflects_as_one_array_of_max_textures() {
    for name in ["forward_frag", "prepass_frag"] {
        let shader = SHADERS
            .iter()
            .find(|shader| shader.name == name)
            .unwrap_or_else(|| panic!("no {name} in the manifest"));
        let table: Vec<_> = shader
            .bindings
            .resources
            .iter()
            .filter(|r| r.kind == ResourceKind::SampledImage && r.count > 1)
            .collect();
        assert_eq!(
            table.len(),
            1,
            "{name}: expected one texture array, got {table:?}"
        );
        assert_eq!(
            table[0].count as usize, MAX_TEXTURES,
            "{name}: the shader's table and gfx::MAX_TEXTURES disagree"
        );
    }
}

#[test]
fn compute_entries_record_their_workgroup_size() {
    let mut computes = 0;
    for shader in SHADERS {
        match shader.stage {
            Stage::Compute => {
                computes += 1;
                let size = shader
                    .bindings
                    .workgroup_size
                    .unwrap_or_else(|| panic!("{} has no workgroup size", shader.name));
                assert!(
                    size.iter().all(|&axis| axis > 0),
                    "{}: {size:?}",
                    shader.name
                );
            }
            _ => assert_eq!(shader.bindings.workgroup_size, None, "{}", shader.name),
        }
    }
    assert!(computes > 10, "only {computes} compute shaders");
}

/// Catches MSL that translated but does not compile, which is the failure the
/// translator cannot report about its own output.
#[test]
#[ignore = "needs a Metal device"]
#[cfg(target_os = "macos")]
fn every_translation_compiles_as_msl() {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let mut failures = Vec::new();
    for shader in SHADERS {
        let source = NSString::from_str(shader.msl);
        if let Err(error) = device.newLibraryWithSource_options_error(&source, None) {
            failures.push(format!("{}: {error}", shader.name));
        }
    }
    assert!(
        failures.is_empty(),
        "{} failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
    println!("{} MSL libraries compiled", SHADERS.len());
}
