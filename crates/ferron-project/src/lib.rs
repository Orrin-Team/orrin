use std::path::{Component, Path, PathBuf};
use serde;
use serde::{Deserialize};
use crate::ProjectError::ParentDirInPath;

const SUPPORTED_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format_version: u32,
    project: ProjectSection,
    scripts: ScriptsSection,
    assets: Option<AssetsSection>,
    scenes: Option<SceneSection>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectSection {
    name: String,
    version: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScriptsSection  {
    dir: PathBuf,
    entry: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AssetsSection  {
    dir: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SceneSection  {
    start: Option<PathBuf>
}

pub enum ProjectError {
    Io { path: PathBuf, source: std::io::Error },
    Parse { path: PathBuf, message: String },
    AbsolutePath { field: &'static str, value: PathBuf },
    ParentDirInPath { field: &'static str, value: PathBuf },
    UnsupportedVersion { found: u32 },
}

pub struct Project {
    root: PathBuf,
    manifest: Manifest,
}

impl Project {
    pub fn locate(start: &Path) -> Result<Option<Project>, ProjectError> {
        for dir in start.ancestors() {
            let candidate = dir.join("ferron.toml");
            if candidate.is_file() {
                return Project::load(&candidate).map(Some);
            }
        }
        Ok(None)
    }
    pub fn load(manifest_path: &Path) -> Result<Project, ProjectError> {
        let text = std::fs::read_to_string(manifest_path).map_err(|source| ProjectError::Io {
            path: manifest_path.to_path_buf(),
            source,
        })?;

        let manifest: Manifest = toml::from_str(&text).map_err(|err| ProjectError::Parse {
            path: manifest_path.to_path_buf(),
            message: err.to_string(),
        })?;

        validate(&manifest)?;

        // The root is every relative path's base, so resolve it once here and
        // canonicalize: symlinked or `..`-laden checkouts collapse to one
        // absolute form instead of leaking into every path the accessors build.
        let dir = manifest_path.parent().unwrap_or(Path::new("."));
        let root = dir.canonicalize().map_err(|source| ProjectError::Io {
            path: dir.to_path_buf(),
            source,
        })?;

        Ok(Project { root, manifest })
    }

    pub fn name(&self) -> &str { &self.manifest.project.name }
    pub fn version(&self) -> &str { &self.manifest.project.version }
    pub fn root(&self) -> &Path { &self.root }
    pub fn scripts_dir(&self) -> PathBuf { self.root.join(&self.manifest.scripts.dir) }
    pub fn entry_type(&self) -> &str { &self.manifest.scripts.entry }
    pub fn assets_dir(&self) -> PathBuf {
        let dir = self
            .manifest
            .assets
            .as_ref()
            .map_or(Path::new("assets"), |a| a.dir.as_path());
        self.root.join(dir)
    }
    pub fn start_scene(&self) -> Option<PathBuf> {
        self.manifest
            .scenes
            .as_ref()
            .and_then(|scenes| scenes.start.as_ref())
            .map(|start| self.root.join(start))
    }
}

fn validate(manifest: &Manifest) -> Result<(), ProjectError> {
    if manifest.format_version != SUPPORTED_FORMAT_VERSION {
        return Err(ProjectError::UnsupportedVersion { found: manifest.format_version })
    }

    let paths = [
        ("scripts.dir",  Some(manifest.scripts.dir.as_path())),
        ("assets.dir",   manifest.assets.as_ref().map(|a| a.dir.as_path())),
        ("scenes.start", manifest.scenes.as_ref().and_then(|s| s.start.as_deref())),
    ];

    for (field, path) in paths {
        if let Some(path) = path {
            check_path(field, path)?;
        }
    }

    Ok(())
}

fn check_path(field: &'static str, path: &Path) -> Result<(), ProjectError> {
    if path.is_absolute() {
        return Err(ProjectError::AbsolutePath { field, value: path.to_path_buf() })
    }

    let has_parent_dir = path.components().any(|c|  c == Component::ParentDir);
    if has_parent_dir { return Err(ParentDirInPath { field, value: path.to_path_buf() }) }

    Ok(())
}
