//! Compiling an Orrin shader: GLSL to SPIR-V, then that same SPIR-V to MSL with
//! the binding map a driver needs to wire it.
//!
//! Two callers compile shaders — `orrin-core`'s build script, once per manifest
//! row, and the editor's runtime compiler — and a shader compiled by one has to
//! bind exactly as one compiled by the other. So the compile lives here, and a
//! shader compiles one way whichever of them asked. Sources and includes are
//! text the caller hands over rather than paths this crate opens: the build
//! reads them from disk, the editor from a table embedded in the engine.
//!
//! The binding assignment (descriptor set N is argument buffer N, push
//! constants at a fixed slot, vertex streams from a fixed base) lives here and
//! nowhere else. Within a set it is a [`Layout`] computed across *every* shader
//! the engine builds, so a shader compiled later is translated against the
//! build's layout, rebuilt from its widths table, or its MSL would disagree with
//! the built-in shaders about where a binding lives.
//!
//! A failure is an `Err` carrying the compiler's own message. The build script
//! turns it into a panic, because a built-in shader that does not compile must
//! fail the build; the editor shows it.

use std::collections::{BTreeMap, BTreeSet};

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
pub const MAX_DESCRIPTOR_SETS: u32 = 6;
pub const PUSH_CONSTANT_BUFFER: u32 = 6;
pub const VERTEX_STREAM_BASE: u32 = 16;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResourceKind {
    UniformBuffer,
    StorageBuffer,
    SampledImage,
    Sampler,
    StorageImage,
    CombinedImageSampler,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Resource {
    pub set: u32,
    pub binding: u32,
    pub kind: ResourceKind,
    /// The array length, one for a plain resource.
    pub count: u32,
    /// Where the resource sits in its set's argument buffer, as the `[[id]]` the
    /// translated shader declares. A combined image sampler takes `count`
    /// textures at `index` and then `count` samplers after them.
    pub index: u32,
}

impl Resource {
    /// How many argument buffer ids the resource occupies.
    pub fn width(&self) -> u32 {
        self.count
            * if self.kind == ResourceKind::CombinedImageSampler {
                2
            } else {
                1
            }
    }
}

/// One shader to compile.
pub struct Source<'a> {
    /// What the compiler calls the file in its messages.
    pub name: &'a str,
    pub text: &'a str,
    pub stage: Stage,
    pub defines: &'a [(&'a str, &'a str)],
}

/// An `#include` the caller's resolver found.
pub struct Include {
    /// Names the file in the compiler's messages, and tells two includes apart.
    pub resolved_name: String,
    pub content: String,
}

/// Resolves the name an `#include` asks for.
pub type Resolver<'a> = dyn Fn(&str) -> Result<Include, String> + 'a;

/// GLSL to SPIR-V words.
pub fn compile_glsl(source: &Source, include: &Resolver) -> Result<Vec<u32>, String> {
    let compiler = Compiler::new().ok_or("failed to create the GLSL compiler")?;
    let mut options = CompileOptions::new().ok_or("failed to create compile options")?;
    options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_0 as u32);
    options.set_include_callback(|requested, _kind: IncludeType, _from, _depth| {
        include(requested).map(|found| ResolvedInclude {
            resolved_name: found.resolved_name,
            content: found.content,
        })
    });
    for (key, value) in source.defines {
        options.add_macro_definition(key, Some(value));
    }

    let kind = match source.stage {
        Stage::Vertex => ShaderKind::Vertex,
        Stage::Fragment => ShaderKind::Fragment,
        Stage::Compute => ShaderKind::Compute,
    };
    let artifact = compiler
        .compile_into_spirv(source.text, kind, source.name, "main", Some(&options))
        .map_err(|e| format!("{}: GLSL does not compile:\n{e}", source.name))?;
    Ok(artifact.as_binary().to_vec())
}

pub struct Translated {
    pub msl: String,
    pub resources: Vec<Resource>,
    pub push_constant_bytes: u32,
    /// Only a compute shader has one.
    pub workgroup_size: Option<[u32; 3]>,
}

/// The shader's resources, sorted by `(set, binding)`, with no index assigned
/// yet: what [`Layout::from_resources`] is built from.
pub fn reflect(name: &str, words: &[u32]) -> Result<Vec<Resource>, String> {
    let compiler = cross_compiler(name, words)?;
    Ok(reflect_resources(name, &compiler)?.0)
}

fn cross_compiler(name: &str, words: &[u32]) -> Result<CrossCompiler<Msl>, String> {
    CrossCompiler::<Msl>::new(Module::from_words(words))
        .map_err(|e| format!("{name}: SPIRV-Cross rejected the module: {e}"))
}

/// The resources, sorted by `(set, binding)`, and the push constant block's size.
fn reflect_resources(
    name: &str,
    compiler: &CrossCompiler<Msl>,
) -> Result<(Vec<Resource>, u32), String> {
    let fail = |what: &str, e: &dyn std::fmt::Display| format!("{name}: SPIRV-Cross {what}: {e}");

    let mut resources = Vec::new();
    let mut push_constant_bytes = 0;
    let found = compiler
        .shader_resources()
        .map_err(|e| fail("reflection failed", &e))?;
    let kinds = [
        (ResourceType::UniformBuffer, ResourceKind::UniformBuffer),
        (ResourceType::StorageBuffer, ResourceKind::StorageBuffer),
        (ResourceType::SeparateImage, ResourceKind::SampledImage),
        (ResourceType::SeparateSamplers, ResourceKind::Sampler),
        (ResourceType::StorageImage, ResourceKind::StorageImage),
        (
            ResourceType::SampledImage,
            ResourceKind::CombinedImageSampler,
        ),
    ];
    for (ty, kind) in kinds {
        for resource in found
            .resources_for_type(ty)
            .map_err(|e| fail("reflection failed", &e))?
        {
            let literal = |decoration| match compiler.decoration(resource.id, decoration) {
                Ok(Some(DecorationValue::Literal(value))) => Ok(value),
                other => Err(format!(
                    "{name}: `{}` has no {decoration:?} ({other:?})",
                    resource.name
                )),
            };
            let set = literal(spirv::Decoration::DescriptorSet)?;
            let binding = literal(spirv::Decoration::Binding)?;
            let described = compiler
                .type_description(resource.type_id)
                .map_err(|e| fail("type lookup failed", &e))?;
            let count = match &described.inner {
                TypeInner::Array { dimensions, .. } => dimensions
                    .iter()
                    .map(|dimension| match dimension {
                        ArrayDimension::Literal(n) => Ok(*n),
                        ArrayDimension::Constant(_) => Err(format!(
                            "{name}: `{}` is sized by a specialisation constant",
                            resource.name
                        )),
                    })
                    .product::<Result<u32, String>>()?,
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
        .map_err(|e| fail("reflection failed", &e))?
    {
        let described = compiler
            .type_description(resource.base_type_id)
            .map_err(|e| fail("type lookup failed", &e))?;
        push_constant_bytes = described.size_hint.declared() as u32;
    }
    resources.sort_by_key(|r| (r.set, r.binding));
    Ok((resources, push_constant_bytes))
}

/// SPIR-V to MSL, with every binding at the index `layout` gives it.
pub fn translate(
    name: &str,
    stage: Stage,
    words: &[u32],
    layout: &Layout,
) -> Result<Translated, String> {
    let fail = |what: &str, e: &dyn std::fmt::Display| format!("{name}: SPIRV-Cross {what}: {e}");

    let mut compiler = cross_compiler(name, words)?;
    let model = match stage {
        Stage::Vertex => spirv::ExecutionModel::Vertex,
        Stage::Fragment => spirv::ExecutionModel::Fragment,
        Stage::Compute => spirv::ExecutionModel::GLCompute,
    };

    let (mut resources, push_constant_bytes) = reflect_resources(name, &compiler)?;
    // Assigned here rather than left to the translator, whose own order follows
    // the shader's declarations: a driver finds a binding by the number the
    // shader declared it at, and has no other way to learn where it landed.
    for resource in &mut resources {
        let (set, binding) = (resource.set, resource.binding);
        match (
            layout.index(set, binding),
            layout.widths.get(&(set, binding)),
        ) {
            (Some(at), Some(&room)) if resource.width() <= room => resource.index = at,
            (_, room) => {
                return Err(format!(
                    "{name}: set {set}, binding {binding} needs {} argument buffer slot(s) where \
                     the engine's layout has {}: it is not a binding the engine's shaders declare",
                    resource.width(),
                    room.copied().unwrap_or(0)
                ));
            }
        }
    }

    let workgroup_size = match stage {
        Stage::Compute => match compiler
            .execution_mode_arguments(spirv::ExecutionMode::LocalSize)
            .map_err(|e| fail("execution mode lookup failed", &e))?
        {
            Some(spirv_cross2::reflect::ExecutionModeArguments::LocalSize { x, y, z }) => {
                Some([x, y, z])
            }
            other => {
                return Err(format!(
                    "{name}: compute shader without a LocalSize ({other:?})"
                ));
            }
        },
        _ => None,
    };

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
            .map_err(|e| fail("rejected an argument buffer binding", &e))?;
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
                    sampler: if resource.kind == ResourceKind::CombinedImageSampler {
                        resource.index + resource.count
                    } else {
                        resource.index
                    },
                    count: std::num::NonZeroU32::new(resource.count).filter(|&n| n.get() > 1),
                },
            )
            .map_err(|e| fail("rejected a resource binding", &e))?;
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
        .map_err(|e| fail("rejected the push constant binding", &e))?;

    let mut options = Msl::options();
    options.version = MslVersion::new(3, 0, 0);
    options.platform = MetalPlatform::MacOS;
    options.argument_buffers = true;
    options.argument_buffers_tier = ArgumentBuffersTier::Tier2;
    // Vulkan clip space is Y-down and the camera bakes that into the projection;
    // Metal's is Y-up. Fixed here so `camera.rs`, which both integrators share,
    // does not have to know.
    options.common.flip_vertex_y = stage == Stage::Vertex;

    #[allow(
        clippy::unnecessary_to_owned,
        reason = "the artifact only displays; padding needs the text"
    )]
    let msl = pad_argument_buffers(
        &compiler
            .compile(&options)
            .map_err(|e| fail("could not emit MSL", &e))?
            .to_string(),
        layout,
    );

    Ok(Translated {
        msl,
        resources,
        push_constant_bytes,
        workgroup_size,
    })
}

pub struct Compiled {
    pub spirv: Vec<u32>,
    pub translated: Translated,
}

/// [`compile_glsl`], then [`translate`] against `layout`.
pub fn compile(source: &Source, include: &Resolver, layout: &Layout) -> Result<Compiled, String> {
    let spirv = compile_glsl(source, include)?;
    let translated = translate(source.name, source.stage, &spirv, layout)?;
    Ok(Compiled { spirv, translated })
}

/// Where each `(set, binding)` sits in its set's argument buffer, the same for
/// every shader.
///
/// A set is one argument buffer bound to every stage that reads it, so the
/// index a binding lives at cannot depend on which other bindings a particular
/// shader happens to use. Each binding is given the widest room any shader asks
/// of it, and the indices run on from there in binding order.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Layout {
    widths: BTreeMap<(u32, u32), u32>,
    index: BTreeMap<(u32, u32), u32>,
    /// Every argument buffer index of each set that some shader uses.
    universe: Vec<BTreeSet<u32>>,
}

impl Layout {
    /// A duplicated `(set, binding)` gets the widest of its rows.
    pub fn from_widths(widths: impl IntoIterator<Item = ((u32, u32), u32)>) -> Layout {
        let mut widest: BTreeMap<(u32, u32), u32> = BTreeMap::new();
        for ((set, binding), width) in widths {
            assert!(
                set < MAX_DESCRIPTOR_SETS,
                "descriptor set {set} has no argument buffer slot"
            );
            let slot = widest.entry((set, binding)).or_insert(0);
            *slot = (*slot).max(width);
        }

        let mut next = [0u32; MAX_DESCRIPTOR_SETS as usize];
        let mut index = BTreeMap::new();
        let mut universe = vec![BTreeSet::new(); MAX_DESCRIPTOR_SETS as usize];
        for (&(set, binding), &width) in &widest {
            let at = next[set as usize];
            index.insert((set, binding), at);
            universe[set as usize].extend(at..at + width);
            next[set as usize] += width;
        }
        Layout {
            widths: widest,
            index,
            universe,
        }
    }

    pub fn from_resources<'a>(resources: impl IntoIterator<Item = &'a Resource>) -> Layout {
        Layout::from_widths(
            resources
                .into_iter()
                .map(|r| ((r.set, r.binding), r.width())),
        )
    }

    /// What the layout was built from, in `(set, binding)` order: enough to
    /// rebuild it with [`Layout::from_widths`].
    pub fn widths(&self) -> impl Iterator<Item = ((u32, u32), u32)> + '_ {
        self.widths.iter().map(|(&key, &width)| (key, width))
    }

    pub fn index(&self, set: u32, binding: u32) -> Option<u32> {
        self.index.get(&(set, binding)).copied()
    }
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
pub fn pad_argument_buffers(msl: &str, layout: &Layout) -> String {
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
        let mut covered = BTreeSet::new();
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

/// A stable 64-bit identity for a set of shader sources and the layout they
/// were compiled against, so a compiled artifact can tell whether it was built
/// against these sources. Independent of the order `files` arrive in.
///
/// FNV-1a rather than a faster hash because it is a dozen lines that can never
/// change underneath a stored artifact, and it runs once per build over a few
/// hundred kilobytes. Every field is length-prefixed, so no two inputs frame the
/// same bytes.
pub fn sources_hash<'a>(
    files: impl IntoIterator<Item = (&'a str, &'a str)>,
    widths: impl IntoIterator<Item = ((u32, u32), u32)>,
) -> u64 {
    const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    let mut files: Vec<_> = files.into_iter().collect();
    files.sort_unstable();
    let mut widths: Vec<_> = widths.into_iter().collect();
    widths.sort_unstable();

    let mut hash = BASIS;
    let field = |hash: &mut u64, bytes: &[u8]| {
        *hash = fnv1a(fnv1a(*hash, &(bytes.len() as u64).to_le_bytes()), bytes);
    };
    field(&mut hash, &(files.len() as u64).to_le_bytes());
    for (path, contents) in files {
        field(&mut hash, path.as_bytes());
        field(&mut hash, contents.as_bytes());
    }
    for ((set, binding), width) in widths {
        for value in [set, binding, width] {
            field(&mut hash, &value.to_le_bytes());
        }
    }
    hash
}

fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resource(set: u32, binding: u32, kind: ResourceKind, count: u32) -> Resource {
        Resource {
            set,
            binding,
            kind,
            count,
            index: 0,
        }
    }

    fn no_includes(requested: &str) -> Result<Include, String> {
        Err(format!("no include `{requested}` here"))
    }

    fn fragment<'a>(name: &'a str, text: &'a str) -> Source<'a> {
        Source {
            name,
            text,
            stage: Stage::Fragment,
            defines: &[],
        }
    }

    #[test]
    fn each_binding_gets_its_room_in_binding_order_within_its_own_set() {
        let layout = Layout::from_widths([((0, 0), 1), ((0, 1), 2), ((0, 3), 4), ((2, 5), 1)]);
        assert_eq!(layout.index(0, 0), Some(0));
        assert_eq!(layout.index(0, 1), Some(1));
        assert_eq!(layout.index(0, 3), Some(3), "binding 2 takes no room");
        assert_eq!(layout.index(2, 5), Some(0), "each set counts from zero");
        assert_eq!(layout.index(1, 0), None);
        assert_eq!(layout.universe[0], (0..7).collect());
        assert!(layout.universe[1].is_empty());
        assert_eq!(layout.universe[2], [0].into());
    }

    #[test]
    fn a_binding_gets_the_widest_room_any_shader_asks_of_it() {
        let layout = Layout::from_resources(&[
            resource(0, 0, ResourceKind::SampledImage, 1),
            resource(0, 1, ResourceKind::UniformBuffer, 1),
            resource(0, 0, ResourceKind::SampledImage, 4),
            resource(0, 0, ResourceKind::SampledImage, 2),
        ]);
        assert_eq!(layout.index(0, 1), Some(4));
        assert_eq!(
            Layout::from_widths([((0, 0), 1), ((0, 0), 4), ((0, 1), 1)]),
            layout
        );
    }

    #[test]
    fn a_combined_image_sampler_is_twice_as_wide_as_its_count() {
        assert_eq!(
            resource(0, 0, ResourceKind::CombinedImageSampler, 3).width(),
            6
        );
        assert_eq!(resource(0, 0, ResourceKind::StorageBuffer, 3).width(), 3);
        let layout = Layout::from_resources(&[
            resource(1, 0, ResourceKind::CombinedImageSampler, 2),
            resource(1, 1, ResourceKind::StorageImage, 1),
        ]);
        assert_eq!(layout.index(1, 1), Some(4));
    }

    #[test]
    fn a_layout_rebuilds_from_its_own_widths_table() {
        let layout = Layout::from_resources(&[
            resource(0, 0, ResourceKind::CombinedImageSampler, 1),
            resource(0, 2, ResourceKind::UniformBuffer, 1),
            resource(3, 1, ResourceKind::SampledImage, 8),
        ]);
        let table: Vec<_> = layout.widths().collect();
        assert_eq!(table, [((0, 0), 2), ((0, 2), 1), ((3, 1), 8)]);
        assert_eq!(Layout::from_widths(table), layout);
    }

    #[test]
    fn padding_fills_every_unused_id_of_a_set_in_id_order() {
        let layout = Layout::from_widths([((0, 0), 1), ((0, 1), 2), ((0, 2), 1), ((1, 0), 1)]);
        let msl = "struct spvDescriptorSetBuffer0\n{\n    \
                   array<texture2d<float>, 2> tex [[id(1)]];\n};\n\
                   struct spvDescriptorSetBuffer1\n{\n    \
                   constant Ubo* ubo [[id(0)]];\n};\n";
        assert_eq!(
            pad_argument_buffers(msl, &layout),
            "struct spvDescriptorSetBuffer0\n{\n    \
             constant uint* _unused0 [[id(0)]];\n    \
             array<texture2d<float>, 2> tex [[id(1)]];\n    \
             constant uint* _unused3 [[id(3)]];\n};\n\
             struct spvDescriptorSetBuffer1\n{\n    \
             constant Ubo* ubo [[id(0)]];\n};\n"
        );
    }

    #[test]
    fn a_set_the_shader_does_not_declare_is_left_alone() {
        let layout = Layout::from_widths([((0, 0), 1), ((0, 1), 1)]);
        let msl = "kernel void main0() {}\n";
        assert_eq!(pad_argument_buffers(msl, &layout), msl);
    }

    #[test]
    fn includes_and_defines_come_from_the_caller() {
        let source = Source {
            name: "lit.frag",
            text: "#version 450\n#extension GL_GOOGLE_include_directive : require\n\
                   #include \"colour.glsl\"\n\
                   #if ORRIN_RED != 1\n#error no define\n#endif\n\
                   layout(location = 0) out vec4 o;\nvoid main() { o = colour(); }\n",
            stage: Stage::Fragment,
            defines: &[("ORRIN_RED", "1")],
        };
        let include = |requested: &str| {
            assert_eq!(requested, "colour.glsl");
            Ok(Include {
                resolved_name: requested.to_string(),
                content: "vec4 colour() { return vec4(1, 0, 0, 1); }\n".to_string(),
            })
        };
        let words = compile_glsl(&source, &include).unwrap();
        assert_eq!(words[0], 0x0723_0203);
    }

    #[test]
    fn a_compile_error_carries_the_compilers_message_and_the_file_name() {
        let error = compile_glsl(
            &fragment("broken.frag", "#version 450\nvoid main() { nope(); }\n"),
            &no_includes,
        )
        .unwrap_err();
        assert!(error.contains("broken.frag"), "{error}");
        assert!(error.contains("nope"), "{error}");
    }

    #[test]
    fn an_include_the_resolver_refuses_is_an_error_naming_it() {
        let error = compile_glsl(
            &fragment(
                "lit.frag",
                "#version 450\n#extension GL_GOOGLE_include_directive : require\n\
                 #include \"missing.glsl\"\nvoid main() {}\n",
            ),
            &no_includes,
        )
        .unwrap_err();
        assert!(error.contains("missing.glsl"), "{error}");
    }

    const TWO_BINDINGS: &str = "#version 450\n\
        layout(set = 0, binding = 0) uniform sampler2D a;\n\
        layout(set = 0, binding = 1) uniform B { vec4 tint; };\n\
        layout(location = 0) out vec4 o;\n\
        void main() { o = texture(a, vec2(0)) * tint; }\n";

    #[test]
    fn translation_puts_each_binding_where_the_layout_says() {
        let layout = Layout::from_widths([((0, 0), 4), ((0, 1), 1), ((0, 2), 1)]);
        let compiled = compile(&fragment("two.frag", TWO_BINDINGS), &no_includes, &layout).unwrap();
        let translated = &compiled.translated;
        assert_eq!(
            translated.resources,
            [
                Resource {
                    set: 0,
                    binding: 0,
                    kind: ResourceKind::CombinedImageSampler,
                    count: 1,
                    index: 0,
                },
                Resource {
                    set: 0,
                    binding: 1,
                    kind: ResourceKind::UniformBuffer,
                    count: 1,
                    index: 4,
                },
            ]
        );
        assert!(translated.msl.contains("[[id(4)]]"), "{}", translated.msl);
        assert!(
            translated.msl.contains("_unused5 [[id(5)]]"),
            "binding 2 is padded: {}",
            translated.msl
        );
        assert_eq!(translated.workgroup_size, None);
    }

    #[test]
    fn a_binding_the_layout_lacks_is_an_error_naming_it() {
        let layout = Layout::from_widths([((0, 0), 2)]);
        let error = compile(&fragment("two.frag", TWO_BINDINGS), &no_includes, &layout)
            .err()
            .unwrap();
        assert!(error.contains("two.frag"), "{error}");
        assert!(error.contains("set 0, binding 1"), "{error}");
    }

    #[test]
    fn a_binding_wider_than_the_layout_allows_is_an_error() {
        let layout = Layout::from_widths([((0, 0), 1), ((0, 1), 1)]);
        let error = compile(&fragment("two.frag", TWO_BINDINGS), &no_includes, &layout)
            .err()
            .unwrap();
        assert!(error.contains("set 0, binding 0"), "{error}");
    }

    #[test]
    fn a_compute_shader_reports_its_workgroup_size() {
        let source = Source {
            name: "fill.comp",
            text: "#version 450\nlayout(local_size_x = 8, local_size_y = 4) in;\n\
                   layout(set = 1, binding = 0) buffer Out { uint v[]; };\n\
                   void main() { v[gl_GlobalInvocationID.x] = 1u; }\n",
            stage: Stage::Compute,
            defines: &[],
        };
        let layout = Layout::from_widths([((1, 0), 1)]);
        let compiled = compile(&source, &no_includes, &layout).unwrap();
        assert_eq!(compiled.translated.workgroup_size, Some([8, 4, 1]));
    }

    #[test]
    fn fnv1a_matches_its_published_vectors() {
        const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
        assert_eq!(fnv1a(BASIS, b""), BASIS);
        assert_eq!(fnv1a(BASIS, b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(BASIS, b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn the_sources_hash_moves_with_every_input_and_not_with_order() {
        let files = [("a.glsl", "float a;"), ("b.frag", "void main() {}")];
        let widths = [((0, 0), 1), ((0, 1), 2)];
        let base = sources_hash(files, widths);
        assert_eq!(base, sources_hash(files, widths), "deterministic");
        assert_eq!(base, sources_hash([files[1], files[0]], widths));

        assert_ne!(
            base,
            sources_hash([files[0], ("b.frag", "void main() { }")], widths),
            "a file's contents"
        );
        assert_ne!(
            base,
            sources_hash([files[0], ("c.frag", "void main() {}")], widths),
            "a file's path"
        );
        assert_ne!(
            base,
            sources_hash(files, [((0, 0), 1), ((0, 1), 3)]),
            "a width"
        );
        assert_ne!(base, sources_hash(files, [((0, 0), 1)]), "a binding");
        assert_ne!(
            sources_hash([("ab", "c")], []),
            sources_hash([("a", "bc")], []),
            "where one field ends and the next begins"
        );
    }
}
