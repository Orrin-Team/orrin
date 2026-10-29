//! Compiles every shader in `shaders/manifest.txt` to SPIR-V once, then
//! translates that same SPIR-V to MSL with a binding map, and writes both into
//! `OUT_DIR` behind a generated `SHADERS` table.
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
    /// Where the resource sits in its set's argument buffer, as the `[[id]]` the
    /// translated shader declares. A combined image sampler takes `count`
    /// textures at `index` and then `count` samplers after them.
    index: u32,
}

struct Translated {
    msl: String,
    resources: Vec<Resource>,
    push_constant_bytes: u32,
    workgroup_size: Option<[u32; 3]>,
}

/// Where each `(set, binding)` sits in its set's argument buffer, the same for
/// every shader.
///
/// A set is one argument buffer bound to every stage that reads it, so the
/// index a binding lives at cannot depend on which other bindings a particular
/// shader happens to use. Each binding is given the widest room any shader asks
/// of it, and the indices run on from there in binding order.
struct Layout {
    index: BTreeMap<(u32, u32), u32>,
    /// Every argument buffer index of each set that some shader uses.
    universe: Vec<std::collections::BTreeSet<u32>>,
}

fn layout(widths: &BTreeMap<(u32, u32), u32>) -> Layout {
    let mut next = [0u32; MAX_DESCRIPTOR_SETS as usize];
    let mut index = BTreeMap::new();
    let mut universe = vec![std::collections::BTreeSet::new(); MAX_DESCRIPTOR_SETS as usize];
    for (&(set, binding), &width) in widths {
        let at = next[set as usize];
        index.insert((set, binding), at);
        universe[set as usize].extend(at..at + width);
        next[set as usize] += width;
    }
    Layout { index, universe }
}

/// Rewrite each set's argument buffer struct so that every shader declares the
/// same slots.
///
/// Metal packs a struct's members in the order of their `[[id]]`s and gives no
/// slot to an id nobody declared, so two shaders that use different bindings of
/// one set would lay the same argument buffer out differently, and a buffer
/// written for one could not be read by the other. Every id of the set the
/// shader leaves unused is filled with a pointer-sized stand-in, which makes the
/// layout a property of the set and not of the shader reading it.
fn pad_argument_buffers(msl: &str, layout: &Layout) -> String {
    let mut out = msl.to_string();
    for set in 0..MAX_DESCRIPTOR_SETS {
        let marker = format!("struct spvDescriptorSetBuffer{set}\n{{\n");
        let Some(start) = out.find(&marker) else {
            continue;
        };
        let body_start = start + marker.len();
        let Some(body_len) = out[body_start..].find("};") else {
            continue;
        };
        let body = &out[body_start..body_start + body_len];

        // (first id, declaration) of each member.
        let mut members: Vec<(u32, String)> = Vec::new();
        let mut covered = std::collections::BTreeSet::new();
        for line in body.lines().filter(|line| line.contains("[[id(")) {
            let at = line.find("[[id(").unwrap() + 5;
            let id: u32 = line[at..at + line[at..].find(')').unwrap()]
                .parse()
                .unwrap();
            let width = line
                .find("array<")
                .and_then(|a| {
                    let tail = &line[a..];
                    let from = tail.rfind(", ")? + 2;
                    tail[from..from + tail[from..].find('>')?]
                        .trim()
                        .parse::<u32>()
                        .ok()
                })
                .unwrap_or(1);
            covered.extend(id..id + width);
            members.push((id, line.to_string()));
        }
        for &id in layout.universe[set as usize]
            .iter()
            .filter(|id| !covered.contains(id))
        {
            members.push((id, format!("    constant uint* _unused{id} [[id({id})]];")));
        }
        members.sort_by_key(|(id, _)| *id);
        let rebuilt: String = members
            .iter()
            .map(|(_, line)| format!("{line}\n"))
            .collect();
        out.replace_range(body_start..body_start + body_len, &rebuilt);
    }
    out
}

/// `layout` is `None` on the first pass, which only reflects.
fn translate(entry: &Entry, words: &[u32], layout: Option<&Layout>) -> Translated {
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
                    index: 0,
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
    let Some(layout) = layout else {
        return Translated {
            msl: String::new(),
            resources,
            push_constant_bytes,
            workgroup_size: None,
        };
    };
    // Assigned here rather than left to the translator, whose own order follows
    // the shader's declarations: a driver finds a binding by the number the
    // shader declared it at, and has no other way to learn where it landed.
    for resource in &mut resources {
        resource.index = layout.index[&(resource.set, resource.binding)];
    }

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
    for resource in &resources {
        compiler
            .add_resource_binding(
                model,
                ResourceBinding::Qualified {
                    set: resource.set,
                    binding: resource.binding,
                },
                &BindTarget {
                    buffer: resource.index,
                    texture: resource.index,
                    sampler: if resource.kind == "CombinedImageSampler" {
                        resource.index + resource.count
                    } else {
                        resource.index
                    },
                    count: std::num::NonZeroU32::new(resource.count).filter(|&n| n.get() > 1),
                },
            )
            .unwrap_or_else(|e| fail("rejected a resource binding", &e));
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

    #[allow(
        clippy::unnecessary_to_owned,
        reason = "the artifact only displays; padding needs the text"
    )]
    let msl = pad_argument_buffers(
        &compiler
            .compile(&options)
            .unwrap_or_else(|e| fail("could not emit MSL", &e))
            .to_string(),
        layout,
    );

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
    let compiled: Vec<Vec<u32>> = entries
        .iter()
        .map(|entry| compile_glsl(entry, &shaders))
        .collect();
    let mut widths: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for (entry, words) in entries.iter().zip(&compiled) {
        for resource in translate(entry, words, None).resources {
            let width = resource.count
                * if resource.kind == "CombinedImageSampler" {
                    2
                } else {
                    1
                };
            let slot = widths.entry((resource.set, resource.binding)).or_insert(0);
            *slot = (*slot).max(width);
        }
    }
    let layout = layout(&widths);

    for (entry, words) in entries.iter().zip(&compiled) {
        let translated = translate(entry, words, Some(&layout));

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
                    "Resource {{ set: {}, binding: {}, kind: ResourceKind::{}, count: {}, index: {} }}, ",
                    r.set, r.binding, r.kind, r.count, r.index
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
         #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]\n\
         pub enum ShaderId {{\n{ids}}}\n\n\
         impl ShaderId {{\n    pub const ALL: &'static [ShaderId] = &[\n{all}    ];\n}}\n\n\
         pub static SHADERS: &[ShaderBlob] = &[\n{table}];\n"
    );
    fs::write(out_dir.join("shaders.rs"), generated).unwrap();
}
