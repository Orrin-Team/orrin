use std::path::{Path, PathBuf};
use std::time::SystemTime;

use orrin_project::Project;
use serde::Serialize;

use crate::engine;
use crate::{Fallible, ProjectArgs, display_path};

/// What a build produced, so `run` can reuse it without repeating discovery.
pub struct Built {
    pub project: Project,
    pub bindings: Option<PathBuf>,
}

pub fn build(args: &ProjectArgs) -> Result<Built, String> {
    let project = args.locate()?;
    println!(
        "orrin: building `{}` at {}",
        project.name(),
        display_path(project.root())
    );

    let bindings = ensure_bindings(project.root(), args.configuration())?;
    build_scripts(&project, args.configuration(), bindings.as_deref())?;
    index_assets(&project)?;

    Ok(Built { project, bindings })
}

/// Resolve the engine's bindings, building them first if this is a checkout
/// that has never built them. A game assembly cannot compile without
/// `Orrin.dll`, so producing it is part of building the project.
fn ensure_bindings(start: &Path, configuration: &str) -> Result<Option<PathBuf>, String> {
    if let Some(dir) = engine::find_bindings(start) {
        return Ok(Some(dir));
    }

    let Some(checkout) = engine::find_checkout(start) else {
        eprintln!(
            "orrin: warning: no Orrin bindings found; if the build cannot resolve \
             `Orrin.dll`, set ${} to the directory holding it",
            engine::BINDINGS_ENV
        );
        return Ok(None);
    };

    let bindings_project = checkout.join("scripting/Orrin");
    println!("orrin: building engine bindings ({})", display_path(&bindings_project));
    dotnet_build(&bindings_project, configuration, None)?;

    Ok(engine::built_bindings(&checkout))
}

fn build_scripts(
    project: &Project,
    configuration: &str,
    bindings: Option<&Path>,
) -> Fallible {
    let scripts = project.scripts_dir();
    if !scripts.is_dir() {
        return Err(format!(
            "`scripts.dir` points at {}, which does not exist",
            scripts.display()
        ));
    }

    dotnet_build(&scripts, configuration, bindings)
}

fn dotnet_build(dir: &Path, configuration: &str, bindings: Option<&Path>) -> Fallible {
    // Inherited, not captured: a terminal build should stream the compiler's
    // output as it happens. The engine's watcher captures the same invocation
    // instead, because its output goes to the editor console.
    let report = orrin_build::Build {
        project_dir: dir,
        configuration,
        bindings,
    }
    .run(orrin_build::Output::Inherit)
    .map_err(|err| err.to_string())?;

    if !report.success {
        return Err(format!("`dotnet build {}` failed", display_path(dir)));
    }

    Ok(())
}

const INVENTORY_VERSION: u32 = 1;

#[derive(Serialize)]
struct Inventory {
    format_version: u32,
    project: String,
    /// Assets directory, relative to the project root.
    root: String,
    assets: Vec<Asset>,
}

#[derive(Serialize)]
struct Asset {
    /// Forward-slash separated, relative to the assets directory, so the file
    /// is identical whichever platform produced it.
    path: String,
    size: u64,
    modified: u64,
    /// Whether a `<path>.meta` sidecar sits beside it.
    meta: bool,
}

/// Write `.orrin/assets.toml`: what the importer will eventually consume, and
/// for now a record of what the project ships. This is deliberately not a
/// packaging step — nothing is copied or transformed until the asset pipeline
/// lands.
fn index_assets(project: &Project) -> Fallible {
    let assets_dir = project.assets_dir();
    let mut assets = Vec::new();

    if assets_dir.is_dir() {
        collect(&assets_dir, &assets_dir, &mut assets)?;
        assets.sort_by(|a, b| a.path.cmp(&b.path));
    } else {
        println!("orrin: no assets directory at {}", display_path(&assets_dir));
    }

    let relative_root = assets_dir
        .strip_prefix(project.root())
        .unwrap_or(&assets_dir)
        .to_string_lossy()
        .replace('\\', "/");

    let inventory = Inventory {
        format_version: INVENTORY_VERSION,
        project: project.name().to_string(),
        root: relative_root,
        assets,
    };

    let out_dir = project.root().join(".orrin");
    std::fs::create_dir_all(&out_dir)
        .map_err(|err| format!("could not create {}: {err}", out_dir.display()))?;

    let path = out_dir.join("assets.toml");
    let text = toml::to_string_pretty(&inventory)
        .map_err(|err| format!("could not encode the asset index: {err}"))?;
    std::fs::write(&path, text)
        .map_err(|err| format!("could not write {}: {err}", path.display()))?;

    println!(
        "orrin: indexed {} asset(s) -> {}",
        inventory.assets.len(),
        display_path(&path)
    );
    Ok(())
}

fn collect(dir: &Path, base: &Path, out: &mut Vec<Asset>) -> Fallible {
    let entries = std::fs::read_dir(dir)
        .map_err(|err| format!("could not read {}: {err}", dir.display()))?;

    for entry in entries {
        let entry = entry.map_err(|err| format!("could not read {}: {err}", dir.display()))?;
        let path = entry.path();

        let file_type = entry
            .file_type()
            .map_err(|err| format!("could not stat {}: {err}", path.display()))?;

        // Symlinks are skipped rather than followed: an index is not worth a
        // directory cycle, and a link out of the tree would break the rule that
        // a project's contents stay inside it.
        if file_type.is_symlink() {
            continue;
        }

        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }

        if file_type.is_dir() {
            collect(&path, base, out)?;
            continue;
        }

        // Sidecars are recorded as a flag on the asset they describe, not as
        // assets of their own.
        if path.extension().is_some_and(|ext| ext == "meta") {
            continue;
        }

        let metadata = entry
            .metadata()
            .map_err(|err| format!("could not stat {}: {err}", path.display()))?;
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map_or(0, |since| since.as_secs());

        let relative = path.strip_prefix(base).unwrap_or(&path);
        let mut meta_path = path.clone().into_os_string();
        meta_path.push(".meta");

        out.push(Asset {
            path: relative.to_string_lossy().replace('\\', "/"),
            size: metadata.len(),
            modified,
            meta: Path::new(&meta_path).is_file(),
        });
    }

    Ok(())
}
