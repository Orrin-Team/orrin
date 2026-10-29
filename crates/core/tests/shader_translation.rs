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

fn all_sources() -> Vec<String> {
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
    sources
}

/// Every shader reaches a driver through the manifest and a `ShaderId`, so the
/// one build produces both the SPIR-V Vulkan loads and the MSL Metal will. A
/// shader compiled any other way would exist for one backend only.
#[test]
fn every_shader_goes_through_the_manifest() {
    let sources = all_sources();
    let bypassing = sources
        .iter()
        .filter(|source| source.contains("vulkano_shaders"))
        .count();
    assert_eq!(
        bypassing, 0,
        "{bypassing} sources compile shaders themselves"
    );

    let unused: Vec<_> = orrin_core::gfx::shaders::ShaderId::ALL
        .iter()
        .filter(|id| {
            !sources
                .iter()
                .any(|src| src.contains(&format!("ShaderId::{id:?}")))
        })
        .collect();
    assert!(unused.is_empty(), "translated but never used: {unused:#?}");
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

/// A driver writes a resource into its set's argument buffer at the index the
/// binding map gives, so two resources of one set must never share an index and
/// every `[[id]]` the MSL declares must be one the map assigned. (A resource the
/// translator proved unused is in the map and not in the MSL, which is fine.)
#[test]
fn argument_buffer_indices_are_disjoint_and_declared_in_the_msl() {
    for shader in SHADERS {
        let mut taken: BTreeSet<(u32, u32)> = BTreeSet::new();
        for resource in shader.bindings.resources {
            let width = if resource.kind == ResourceKind::CombinedImageSampler {
                2
            } else {
                1
            };
            for slot in resource.index..resource.index + resource.count * width {
                assert!(
                    taken.insert((resource.set, slot)),
                    "{}: set {} reuses argument buffer index {slot}",
                    shader.name,
                    resource.set
                );
            }
        }
        for set in 0..MAX_DESCRIPTOR_SETS {
            let marker = format!("struct spvDescriptorSetBuffer{set}\n");
            let Some(body) = shader.msl.split(&marker).nth(1) else {
                continue;
            };
            let body = body.split("};").next().unwrap_or("");
            // Not the stand-ins: those fill the ids a shader does not use, so
            // that every shader lays the set out alike.
            let body: String = body
                .lines()
                .filter(|line| !line.contains("_unused"))
                .collect::<Vec<_>>()
                .join("\n");
            for (at, _) in body.match_indices("[[id(") {
                let digits = &body[at + 5..];
                let id: u32 = digits[..digits.find(')').unwrap()].parse().unwrap();
                assert!(
                    taken.contains(&(set, id)),
                    "{}: set {set} declares [[id({id})]], which the binding map never assigned",
                    shader.name
                );
            }
        }
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
