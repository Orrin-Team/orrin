//! Compiles every shader in `shaders/manifest.txt` to SPIR-V once, then
//! translates that same SPIR-V to MSL with a binding map, and writes both into
//! `OUT_DIR` behind a generated `SHADERS` table (docs/metal-backend-plan.md M2).
//!
//! A shader that fails to compile or translate fails the build, on whichever
//! machine builds it. That is the point: MoltenVK translates these shaders at
//! runtime on a Mac, so without this a translation break would first show up
//! there. The binding assignment (descriptor set N is argument buffer N, push
//! constants at a fixed slot, vertex streams from a fixed base) lives here and
//! nowhere else; the generated file hands the constants to the engine.

use std::collections::BTreeMap;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use shaderc::{
    CompileOptions, Compiler, EnvVersion, IncludeType, ResolvedInclude, ShaderKind, TargetEnv,
};
use spirv_cross2::compile::CompilableTarget;
use spirv_cross2::compile::msl::{
    ArgumentBuffersTier, BindTarget, MetalPlatform, MslVersion, ResourceBinding,
};
use spirv_cross2::reflect::{ArrayDimension, TypeInner};
use spirv_cross2::reflect::{DecorationValue, ResourceType};
use spirv_cross2::spirv;
use spirv_cross2::targets::Msl;
use spirv_cross2::{Compiler as CrossCompiler, Module};

/// Descriptor sets 0..MAX_DESCRIPTOR_SETS are argument buffers at the buffer
/// index equal to the set number.
const MAX_DESCRIPTOR_SETS: u32 = 6;
const PUSH_CONSTANT_BUFFER: u32 = 6;
const VERTEX_STREAM_BASE: u32 = 16;

struct Entry {
    name: String,
    stage: &'static str,
    path: String,
    defines: Vec<(String, String)>,
}

fn parse_manifest(text: &str) -> Vec<Entry> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut words = line.split_whitespace();
            let name = words.next().expect("manifest row has a name").to_string();
            let stage = match words.next().expect("manifest row has a stage") {
                "vertex" => "vertex",
                "fragment" => "fragment",
                "compute" => "compute",
                other => panic!("manifest: `{name}` has unknown stage `{other}`"),
            };
            let path = words.next().expect("manifest row has a path").to_string();
            let defines = words
                .map(|define| {
                    let (key, value) = define.split_once('=').unwrap_or_else(|| {
                        panic!("manifest: `{name}`: `{define}` is not KEY=VALUE")
                    });
                    (key.to_string(), value.to_string())
                })
                .collect();
            Entry {
                name,
                stage,
                path,
                defines,
            }
        })
        .collect()
}

fn compile_glsl(entry: &Entry, shaders: &Path) -> Vec<u32> {
    let source_path = shaders.join(&entry.path);
    let source = fs::read_to_string(&source_path)
        .unwrap_or_else(|e| panic!("{}: cannot read {}: {e}", entry.name, source_path.display()));

    let compiler = Compiler::new().expect("failed to create the GLSL compiler");
    let mut options = CompileOptions::new().expect("failed to create compile options");
    options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_0 as u32);
    options.set_include_callback(|requested, _kind: IncludeType, _from, _depth| {
        let path = shaders.join(requested);
        fs::read_to_string(&path)
            .map(|content| ResolvedInclude {
                resolved_name: path.display().to_string(),
                content,
            })
            .map_err(|e| format!("cannot include {}: {e}", path.display()))
    });
    for (key, value) in &entry.defines {
        options.add_macro_definition(key, Some(value));
    }

    let kind = match entry.stage {
        "vertex" => ShaderKind::Vertex,
        "fragment" => ShaderKind::Fragment,
        _ => ShaderKind::Compute,
    };
    let artifact = compiler
        .compile_into_spirv(&source, kind, &entry.path, "main", Some(&options))
        .unwrap_or_else(|e| panic!("{}: GLSL does not compile:\n{e}", entry.name));
    artifact.as_binary().to_vec()
}

struct Resource {
    set: u32,
    binding: u32,
    kind: &'static str,
    count: u32,
}

struct Translated {
    msl: String,
    resources: Vec<Resource>,
    push_constant_bytes: u32,
    workgroup_size: Option<[u32; 3]>,
}

fn translate(entry: &Entry, words: &[u32]) -> Translated {
    let fail = |what: &str, e: &dyn std::fmt::Display| -> ! {
        panic!("{}: SPIRV-Cross {what}: {e}", entry.name)
    };

    let mut compiler = CrossCompiler::<Msl>::new(Module::from_words(words))
        .unwrap_or_else(|e| fail("rejected the module", &e));

    let model = match entry.stage {
        "vertex" => spirv::ExecutionModel::Vertex,
        "fragment" => spirv::ExecutionModel::Fragment,
        _ => spirv::ExecutionModel::GLCompute,
    };

    let mut resources = Vec::new();
    let mut push_constant_bytes = 0;
    {
        let found = compiler
            .shader_resources()
            .unwrap_or_else(|e| fail("reflection failed", &e));
        let kinds = [
            (ResourceType::UniformBuffer, "UniformBuffer"),
            (ResourceType::StorageBuffer, "StorageBuffer"),
            (ResourceType::SeparateImage, "SampledImage"),
            (ResourceType::SeparateSamplers, "Sampler"),
            (ResourceType::StorageImage, "StorageImage"),
            (ResourceType::SampledImage, "CombinedImageSampler"),
        ];
        for (ty, kind) in kinds {
            for resource in found
                .resources_for_type(ty)
                .unwrap_or_else(|e| fail("reflection failed", &e))
            {
                let literal = |decoration| match compiler.decoration(resource.id, decoration) {
                    Ok(Some(DecorationValue::Literal(value))) => value,
                    other => panic!(
                        "{}: `{}` has no {decoration:?} ({other:?})",
                        entry.name, resource.name
                    ),
                };
                let set = literal(spirv::Decoration::DescriptorSet);
                let binding = literal(spirv::Decoration::Binding);
                let described = compiler
                    .type_description(resource.type_id)
                    .unwrap_or_else(|e| fail("type lookup failed", &e));
                let count = match &described.inner {
                    TypeInner::Array { dimensions, .. } => dimensions
                        .iter()
                        .map(|dimension| match dimension {
                            ArrayDimension::Literal(n) => *n,
                            ArrayDimension::Constant(_) => {
                                panic!(
                                    "{}: `{}` is sized by a specialisation constant",
                                    entry.name, resource.name
                                )
                            }
                        })
                        .product(),
                    _ => 1,
                };
                resources.push(Resource {
                    set,
                    binding,
                    kind,
                    count,
                });
            }
        }
        for resource in found
            .resources_for_type(ResourceType::PushConstant)
            .unwrap_or_else(|e| fail("reflection failed", &e))
        {
            let described = compiler
                .type_description(resource.base_type_id)
                .unwrap_or_else(|e| fail("type lookup failed", &e));
            push_constant_bytes = described.size_hint.declared() as u32;
        }
    }
    resources.sort_by_key(|r| (r.set, r.binding));

    let workgroup_size = (entry.stage == "compute").then(|| {
        match compiler
            .execution_mode_arguments(spirv::ExecutionMode::LocalSize)
            .unwrap_or_else(|e| fail("execution mode lookup failed", &e))
        {
            Some(spirv_cross2::reflect::ExecutionModeArguments::LocalSize { x, y, z }) => [x, y, z],
            other => panic!(
                "{}: compute shader without a LocalSize ({other:?})",
                entry.name
            ),
        }
    });

    for set in 0..MAX_DESCRIPTOR_SETS {
        compiler
            .add_resource_binding(
                model,
                ResourceBinding::ArgumentBuffer(set),
                &BindTarget {
                    buffer: set,
                    texture: 0,
                    sampler: 0,
                    count: None,
                },
            )
            .unwrap_or_else(|e| fail("rejected an argument buffer binding", &e));
    }
    compiler
        .add_resource_binding(
            model,
            ResourceBinding::PushConstantBuffer,
            &BindTarget {
                buffer: PUSH_CONSTANT_BUFFER,
                texture: 0,
                sampler: 0,
                count: None,
            },
        )
        .unwrap_or_else(|e| fail("rejected the push constant binding", &e));

    let mut options = Msl::options();
    options.version = MslVersion::new(3, 0, 0);
    options.platform = MetalPlatform::MacOS;
    options.argument_buffers = true;
    options.argument_buffers_tier = ArgumentBuffersTier::Tier2;
    // Vulkan clip space is Y-down and the camera bakes that into the projection;
    // Metal's is Y-up. Fixed here so `camera.rs`, which both integrators share,
    // does not have to know.
    options.common.flip_vertex_y = entry.stage == "vertex";

    let msl = compiler
        .compile(&options)
        .unwrap_or_else(|e| fail("could not emit MSL", &e))
        .to_string();

    Translated {
        msl,
        resources,
        push_constant_bytes,
        workgroup_size,
    }
}

fn camel(name: &str) -> String {
    name.split('_')
        .map(|part| {
            let mut chars = part.chars();
            chars
                .next()
                .map(|first| first.to_uppercase().chain(chars).collect::<String>())
                .unwrap_or_default()
        })
        .collect()
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shaders = manifest_dir.join("shaders");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shaders");

    let manifest = fs::read_to_string(shaders.join("manifest.txt")).expect("shaders/manifest.txt");
    let entries = parse_manifest(&manifest);
    let mut seen = BTreeMap::new();
    for entry in &entries {
        if let Some(previous) = seen.insert(entry.name.clone(), &entry.path) {
            panic!(
                "manifest: `{}` names both {previous} and {}",
                entry.name, entry.path
            );
        }
    }

    let mut table = String::new();
    let mut ids = String::new();
    let mut all = String::new();
    for entry in &entries {
        let words = compile_glsl(entry, &shaders);
        let translated = translate(entry, &words);

        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        fs::write(out_dir.join(format!("{}.spv", entry.name)), bytes).unwrap();
        fs::write(
            out_dir.join(format!("{}.metal", entry.name)),
            &translated.msl,
        )
        .unwrap();

        let variant = camel(&entry.name);
        writeln!(ids, "    {variant},").unwrap();
        writeln!(all, "        ShaderId::{variant},").unwrap();

        let defines: String = entry
            .defines
            .iter()
            .map(|(k, v)| format!("({k:?}, {v:?}), "))
            .collect();
        let resources: String = translated
            .resources
            .iter()
            .map(|r| {
                format!(
                    "Resource {{ set: {}, binding: {}, kind: ResourceKind::{}, count: {} }}, ",
                    r.set, r.binding, r.kind, r.count
                )
            })
            .collect();
        let stage = match entry.stage {
            "vertex" => "Vertex",
            "fragment" => "Fragment",
            _ => "Compute",
        };
        let workgroup = match translated.workgroup_size {
            Some([x, y, z]) => format!("Some([{x}, {y}, {z}])"),
            None => "None".to_string(),
        };
        writeln!(
            table,
            "    ShaderBlob {{\n        name: {:?},\n        path: {:?},\n        stage: Stage::{stage},\n        defines: &[{defines}],\n        spirv: include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{name}.spv\")),\n        msl: include_str!(concat!(env!(\"OUT_DIR\"), \"/{name}.metal\")),\n        bindings: BindingMap {{\n            resources: &[{resources}],\n            push_constant_bytes: {},\n            workgroup_size: {workgroup},\n        }},\n    }},",
            entry.name,
            entry.path,
            translated.push_constant_bytes,
            name = entry.name,
        )
        .unwrap();
    }

    let generated = format!(
        "pub const MAX_DESCRIPTOR_SETS: u32 = {MAX_DESCRIPTOR_SETS};\n\
         pub const PUSH_CONSTANT_BUFFER: u32 = {PUSH_CONSTANT_BUFFER};\n\
         pub const VERTEX_STREAM_BASE: u32 = {VERTEX_STREAM_BASE};\n\n\
         #[derive(Clone, Copy, PartialEq, Eq, Debug)]\n\
         pub enum ShaderId {{\n{ids}}}\n\n\
         impl ShaderId {{\n    pub const ALL: &'static [ShaderId] = &[\n{all}    ];\n}}\n\n\
         pub static SHADERS: &[ShaderBlob] = &[\n{table}];\n"
    );
    fs::write(out_dir.join("shaders.rs"), generated).unwrap();
}
