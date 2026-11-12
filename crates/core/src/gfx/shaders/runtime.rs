use orrin_shaderc::{Include, Layout, Source};

use super::{
    CompiledBindings, CompiledShader, LAYOUT_WIDTHS, Resource, ResourceKind, SOURCES, Stage,
};

/// Compile one shader the way `build.rs` compiles a manifest row.
///
/// `name` is what the compiler's messages call the file. An `#include` is
/// looked for first among `includes`, then among the engine's own [`SOURCES`],
/// so a caller can stand a generated file in for an engine one. The layout is
/// the build's, rebuilt from [`LAYOUT_WIDTHS`]; a binding the engine's shaders
/// never declare is an error rather than a slot nobody else agrees on.
pub fn compile(
    source: &str,
    stage: Stage,
    name: &str,
    defines: &[(&str, &str)],
    includes: &[(&str, &str)],
) -> Result<CompiledShader, String> {
    let include = |requested: &str| {
        includes
            .iter()
            .chain(SOURCES)
            .find(|(path, _)| *path == requested)
            .map(|(path, content)| Include {
                resolved_name: path.to_string(),
                content: content.to_string(),
            })
            .ok_or_else(|| format!("no shader include named `{requested}`"))
    };
    let compiled = orrin_shaderc::compile(
        &Source {
            name,
            text: source,
            stage: match stage {
                Stage::Vertex => orrin_shaderc::Stage::Vertex,
                Stage::Fragment => orrin_shaderc::Stage::Fragment,
                Stage::Compute => orrin_shaderc::Stage::Compute,
            },
            defines,
        },
        &include,
        &Layout::from_widths(LAYOUT_WIDTHS.iter().copied()),
    )?;

    let translated = compiled.translated;
    Ok(CompiledShader {
        name: name.to_string(),
        stage,
        defines: defines
            .iter()
            .map(|&(key, value)| (key.to_string(), value.to_string()))
            .collect(),
        spirv: compiled
            .spirv
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect(),
        msl: translated.msl,
        bindings: CompiledBindings {
            resources: translated
                .resources
                .iter()
                .map(|r| Resource {
                    set: r.set,
                    binding: r.binding,
                    kind: kind(r.kind),
                    count: r.count,
                    index: r.index,
                })
                .collect(),
            push_constant_bytes: translated.push_constant_bytes,
            workgroup_size: translated.workgroup_size,
        },
    })
}

fn kind(kind: orrin_shaderc::ResourceKind) -> ResourceKind {
    use orrin_shaderc::ResourceKind as K;
    match kind {
        K::UniformBuffer => ResourceKind::UniformBuffer,
        K::StorageBuffer => ResourceKind::StorageBuffer,
        K::SampledImage => ResourceKind::SampledImage,
        K::Sampler => ResourceKind::Sampler,
        K::StorageImage => ResourceKind::StorageImage,
        K::CombinedImageSampler => ResourceKind::CombinedImageSampler,
    }
}
