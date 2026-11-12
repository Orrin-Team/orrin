//! Compiles every shader in `shaders/manifest.txt` to SPIR-V once, then
//! translates that same SPIR-V to MSL with a binding map, and writes both into
//! `OUT_DIR` behind a generated `SHADERS` table.
//!
//! A shader that fails to compile or translate fails the build, on whichever
//! machine builds it. That is the point: MoltenVK translates these shaders at
//! runtime on a Mac, so without this a translation break would first show up
//! there. The compile itself is `orrin-shaderc`'s, shared with the editor's
//! runtime compiler; what is decided here is the layout, from every row of the
//! manifest at once, and the generated files hand it and the constants to the
//! engine.
//!
//! `shaders.rs` is the table. `shader_sources.rs` is what a runtime compile needs
//! to agree with it: the widths table the layout was built from, a hash over it
//! and every file under `shaders/`, and, with the `shader-compiler` feature, the
//! files themselves.

use std::collections::BTreeMap;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use orrin_shaderc::{
    Include, Layout, MAX_DESCRIPTOR_SETS, PUSH_CONSTANT_BUFFER, Source, Stage, VERTEX_STREAM_BASE,
};

struct Entry {
    name: String,
    stage: Stage,
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
                "vertex" => Stage::Vertex,
                "fragment" => Stage::Fragment,
                "compute" => Stage::Compute,
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
    let text = fs::read_to_string(&source_path)
        .unwrap_or_else(|e| panic!("{}: cannot read {}: {e}", entry.name, source_path.display()));
    let defines: Vec<(&str, &str)> = entry
        .defines
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect();
    let source = Source {
        name: &entry.path,
        text: &text,
        stage: entry.stage,
        defines: &defines,
    };
    let include = |requested: &str| {
        let path = shaders.join(requested);
        fs::read_to_string(&path)
            .map(|content| Include {
                resolved_name: path.display().to_string(),
                content,
            })
            .map_err(|e| format!("cannot include {}: {e}", path.display()))
    };
    orrin_shaderc::compile_glsl(&source, &include).unwrap_or_else(|e| panic!("{}: {e}", entry.name))
}

/// Every file under `shaders/` but the hidden ones, as `(path relative to
/// shaders/ with forward slashes, absolute path, contents)`, sorted by the first.
fn source_files(shaders: &Path) -> Vec<(String, PathBuf, String)> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf, String)>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.file_name().unwrap().to_string_lossy().starts_with('.') {
                continue;
            }
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let relative: Vec<_> = path
                    .strip_prefix(root)
                    .unwrap()
                    .components()
                    .map(|part| part.as_os_str().to_string_lossy().into_owned())
                    .collect();
                let contents = fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
                out.push((relative.join("/"), path, contents));
            }
        }
    }
    let mut files = Vec::new();
    walk(shaders, shaders, &mut files);
    files.sort();
    files
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
    let reflected: Vec<_> = entries
        .iter()
        .zip(&compiled)
        .flat_map(|(entry, words)| {
            orrin_shaderc::reflect(&entry.name, words).unwrap_or_else(|e| panic!("{e}"))
        })
        .collect();
    let layout = Layout::from_resources(&reflected);

    for (entry, words) in entries.iter().zip(&compiled) {
        let translated = orrin_shaderc::translate(&entry.name, entry.stage, words, &layout)
            .unwrap_or_else(|e| panic!("{e}"));

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
                    "Resource {{ set: {}, binding: {}, kind: ResourceKind::{:?}, count: {}, index: {} }}, ",
                    r.set, r.binding, r.kind, r.count, r.index
                )
            })
            .collect();
        let workgroup = match translated.workgroup_size {
            Some([x, y, z]) => format!("Some([{x}, {y}, {z}])"),
            None => "None".to_string(),
        };
        writeln!(
            table,
            "    ShaderBlob {{\n        name: {:?},\n        path: {:?},\n        stage: Stage::{:?},\n        defines: &[{defines}],\n        spirv: include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{name}.spv\")),\n        msl: include_str!(concat!(env!(\"OUT_DIR\"), \"/{name}.metal\")),\n        bindings: BindingMap {{\n            resources: &[{resources}],\n            push_constant_bytes: {},\n            workgroup_size: {workgroup},\n        }},\n    }},",
            entry.name,
            entry.path,
            entry.stage,
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

    let widths: Vec<_> = layout.widths().collect();
    let files = source_files(&shaders);
    let table: String = widths
        .iter()
        .map(|((set, binding), width)| format!("    (({set}, {binding}), {width}),\n"))
        .collect();
    let sources: String = files
        .iter()
        .map(|(relative, path, _)| {
            format!(
                "    ({relative:?}, include_str!({:?})),\n",
                path.display().to_string()
            )
        })
        .collect();
    let hash = orrin_shaderc::sources_hash(
        files
            .iter()
            .map(|(relative, _, contents)| (relative.as_str(), contents.as_str())),
        widths.iter().copied(),
    );
    let generated = format!(
        "/// `((set, binding), width)`: the argument buffer layout every shader in\n\
         /// [`SHADERS`] was translated against.\n\
         pub const LAYOUT_WIDTHS: &[((u32, u32), u32)] = &[\n{table}];\n\n\
         /// Every file under `shaders/`, by its path relative to it.\n\
         #[cfg(feature = \"shader-compiler\")]\n\
         pub static SOURCES: &[(&str, &str)] = &[\n{sources}];\n\n\
         /// `orrin_shaderc::sources_hash` over [`SOURCES`] and [`LAYOUT_WIDTHS`]:\n\
         /// what a compiled shader records to say which engine it was built for.\n\
         /// Without the `shader-compiler` feature too: an exported game compares\n\
         /// against it and compiles nothing.\n\
         pub const SOURCES_HASH: u64 = {hash:#018x};\n"
    );
    fs::write(out_dir.join("shader_sources.rs"), generated).unwrap();
}
