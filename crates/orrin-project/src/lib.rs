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

#[derive(Debug)]
pub enum ProjectError {
    Io { path: PathBuf, source: std::io::Error },
    Parse { path: PathBuf, message: String },
    AbsolutePath { field: &'static str, value: PathBuf },
    ParentDirInPath { field: &'static str, value: PathBuf },
    UnsupportedVersion { found: u32 },
}

impl std::fmt::Display for ProjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "could not read {}: {source}", path.display())
            }
            Self::Parse { path, message } => {
                write!(f, "could not parse {}: {message}", path.display())
            }
            Self::AbsolutePath { field, value } => write!(
                f,
                "`{field}` must be relative to the project, got `{}`; \
                 absolute paths are machine-local and must not be committed",
                value.display()
            ),
            Self::ParentDirInPath { field, value } => write!(
                f,
                "`{field}` must stay inside the project, got `{}`; \
                 remove the `..` components",
                value.display()
            ),
            Self::UnsupportedVersion { found } => write!(
                f,
                "unsupported `format_version` {found}; this engine supports {SUPPORTED_FORMAT_VERSION}"
            ),
        }
    }
}

impl std::error::Error for ProjectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Project {
    root: PathBuf,
    manifest: Manifest,
}

impl Project {
    pub fn locate(start: &Path) -> Result<Option<Project>, ProjectError> {
        for dir in start.ancestors() {
            let candidate = dir.join("orrin.toml");
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

#[cfg(test)]
mod tests {
    use std::fs::create_dir_all;
    use super::*;
    use tempfile::TempDir;

    const VALID: &str = r#"
format_version = 1

[project]
name = "hello-orrin"
version = "0.1.0"

[scripts]
dir = "scripts"
entry = "MyGame.Main, MyGame"
"#;

    /// Write `body` as `orrin.toml` in a fresh temp dir and hand back both the
    /// dir and its canonical path. Keep the `TempDir` alive in the caller: it
    /// deletes the directory on drop.
    fn project_dir(body: &str) -> (TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("orrin.toml"), body).expect("write manifest");
        let root = dir.path().canonicalize().expect("canonicalize");
        (dir, root)
    }

    #[test]
    fn round_trip() {
        let (_dir, root) = project_dir(VALID);

        let project = Project::locate(&root)
            .expect("valid manifest")
            .expect("manifest is present");

        assert_eq!(project.name(), "hello-orrin");
        assert_eq!(project.version(), "0.1.0");
        assert_eq!(project.entry_type(), "MyGame.Main, MyGame");
        assert_eq!(project.root(), root);
        assert_eq!(project.scripts_dir(), root.join("scripts"));
        // `[assets]` is absent, so the accessor supplies the default.
        assert_eq!(project.assets_dir(), root.join("assets"));
        // `[scenes]` is absent entirely.
        assert_eq!(project.start_scene(), None);
    }

    #[test]
    fn absolute_path_is_rejected() {
        let body = VALID.replace(r#"dir = "scripts""#, r#"dir = "/etc""#);
        let (_dir, root) = project_dir(&body);

        let err = Project::locate(&root).expect_err("absolute scripts.dir");
        assert!(
            matches!(err, ProjectError::AbsolutePath { field: "scripts.dir", .. }),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn locate_walks_up_to_the_project_root() {
        let (_dir, root) = project_dir(VALID);
        let nested = root.join("scripts").join("deep");
        create_dir_all(&nested).expect("create nested dirs");

        let project = Project::locate(&nested)
            .expect("valid manifest")
            .expect("manifest found up the chain");

        assert_eq!(project.root(), root);
    }

    #[test]
    fn locate_returns_none_without_a_manifest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().canonicalize().expect("canonicalize");

        // Walks all the way to `/` without a hit: absence is not an error, it
        // just means "not an Orrin project".
        assert!(
            Project::locate(&root)
                .expect("a missing manifest is not an error")
                .is_none()
        );
    }

    #[test]
    fn parent_dir_is_rejected() {
        let body = VALID.replace(r#"dir = "scripts""#, r#"dir = "../escape""#);
        let (_dir, root) = project_dir(&body);

        let err = Project::locate(&root).expect_err("scripts.dir escapes the project");
        assert!(
            matches!(err, ProjectError::ParentDirInPath { field: "scripts.dir", .. }),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unsupported_format_version_is_rejected() {
        let body = VALID.replace("format_version = 1", "format_version = 99");
        let (_dir, root) = project_dir(&body);

        let err = Project::locate(&root).expect_err("unknown format version");
        assert!(
            matches!(err, ProjectError::UnsupportedVersion { found: 99 }),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unknown_field_in_a_section_is_rejected() {
        // The stray key lands inside `[scripts]`, the last table in VALID, so
        // this fails only if `deny_unknown_fields` is on the section struct
        // and not just on `Manifest`.
        let body = format!("{VALID}bogus = \"x\"\n");
        let (_dir, root) = project_dir(&body);

        let err = Project::locate(&root).expect_err("unknown key in [scripts]");
        assert!(matches!(err, ProjectError::Parse { .. }), "unexpected error: {err}");
    }

    #[test]
    fn unknown_table_is_rejected() {
        let body = format!("{VALID}\n[bogus]\nkey = \"x\"\n");
        let (_dir, root) = project_dir(&body);

        let err = Project::locate(&root).expect_err("unknown table");
        assert!(matches!(err, ProjectError::Parse { .. }), "unexpected error: {err}");
    }

    #[test]
    fn missing_required_field_is_rejected() {
        let body = VALID.replace("entry = \"MyGame.Main, MyGame\"\n", "");
        let (_dir, root) = project_dir(&body);

        let err = Project::locate(&root).expect_err("scripts.entry is required");
        assert!(matches!(err, ProjectError::Parse { .. }), "unexpected error: {err}");
    }

    #[test]
    fn optional_sections_are_resolved_against_the_root() {
        let body =
            format!("{VALID}\n[assets]\ndir = \"art\"\n\n[scenes]\nstart = \"scenes/main.fscene\"\n");
        let (_dir, root) = project_dir(&body);

        let project = Project::locate(&root)
            .expect("valid manifest")
            .expect("manifest is present");

        assert_eq!(project.assets_dir(), root.join("art"));
        assert_eq!(project.start_scene(), Some(root.join("scenes/main.fscene")));
    }

    #[test]
    fn optional_section_paths_are_validated_too() {
        let body = format!("{VALID}\n[assets]\ndir = \"/var/art\"\n");
        let (_dir, root) = project_dir(&body);

        let err = Project::locate(&root).expect_err("absolute assets.dir");
        assert!(
            matches!(err, ProjectError::AbsolutePath { field: "assets.dir", .. }),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn load_reports_a_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("orrin.toml");

        let err = Project::load(&missing).expect_err("no such file");
        assert!(matches!(err, ProjectError::Io { .. }), "unexpected error: {err}");
    }
}