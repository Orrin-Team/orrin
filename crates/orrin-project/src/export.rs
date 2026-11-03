//! `orrin.json`: what an exported game says about itself.
//!
//! An export is not a project. It has no scripts to compile and no sources to
//! import, only what `orrin export` produced, so it gets a manifest of its own
//! rather than a trimmed `orrin.toml` — a game directory that looked like a
//! project would invite opening it in the editor, which would find nothing it
//! can edit. It is JSON rather than TOML because it is written by a tool and
//! read by the engine, never edited by hand.
//!
//! The engine still runs an export through a [`Project`], made by
//! [`Project::from_export`], so every path below the startup — the asset
//! directory, the scenes, `input.toml` — is resolved by the same accessors in
//! the editor and in a shipped game. Paths obey the manifest rules: relative
//! to the directory holding `orrin.json`, with no `..`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Manifest, Project, ProjectError, ProjectSection, SceneSection, ScriptsSection};

pub const FILE_NAME: &str = "orrin.json";

const FORMAT_VERSION: u32 = 1;

/// An operating system a game can be exported for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Platform {
    Mac,
    Linux,
    Windows,
}

impl Platform {
    pub const ALL: [Self; 3] = [Self::Mac, Self::Linux, Self::Windows];

    /// The word `--platform` takes and the export directory ends in.
    pub fn name(self) -> &'static str {
        match self {
            Self::Mac => "mac",
            Self::Linux => "linux",
            Self::Windows => "windows",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "mac" | "macos" | "osx" => Some(Self::Mac),
            "linux" => Some(Self::Linux),
            "windows" | "win" => Some(Self::Windows),
            _ => None,
        }
    }

    /// The platform this process runs on, if it is one a game exports for.
    pub fn host() -> Option<Self> {
        match std::env::consts::OS {
            "macos" => Some(Self::Mac),
            "linux" => Some(Self::Linux),
            "windows" => Some(Self::Windows),
            _ => None,
        }
    }

    pub fn exe_suffix(self) -> &'static str {
        match self {
            Self::Windows => ".exe",
            Self::Mac | Self::Linux => "",
        }
    }
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The contents of `orrin.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportManifest {
    pub format_version: u32,
    pub name: String,
    pub version: String,
    pub platform: Platform,
    /// The entry Behaviour, assembly-qualified as in `orrin.toml`.
    pub entry: String,
    /// The game assembly's file, beside the executable.
    pub game_assembly: PathBuf,
    /// The packed assets: artifacts, sidecars and the text assets read as
    /// they are.
    pub assets: PathBuf,
    /// Relative to the export directory, like every other path here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_scene: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persistent_scene: Option<PathBuf>,
}

impl ExportManifest {
    pub fn new(project: &Project, platform: Platform, game_assembly: impl Into<PathBuf>) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            name: project.name().to_owned(),
            version: project.version().to_owned(),
            platform,
            entry: project.entry_type().to_owned(),
            game_assembly: game_assembly.into(),
            assets: PathBuf::from("compiled-assets"),
            start_scene: None,
            persistent_scene: None,
        }
    }

    /// Write `orrin.json` into `dir`.
    pub fn write(&self, dir: &Path) -> std::io::Result<()> {
        let mut text = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        text.push('\n');
        std::fs::write(dir.join(FILE_NAME), text)
    }

    pub fn read(path: &Path) -> Result<Self, ProjectError> {
        let text = std::fs::read_to_string(path).map_err(|source| ProjectError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let manifest: Self = serde_json::from_str(&text).map_err(|err| ProjectError::Parse {
            path: path.to_path_buf(),
            message: err.to_string(),
        })?;
        if manifest.format_version != FORMAT_VERSION {
            return Err(ProjectError::UnsupportedVersion {
                found: manifest.format_version,
            });
        }
        let paths = [
            ("game_assembly", Some(manifest.game_assembly.as_path())),
            ("assets", Some(manifest.assets.as_path())),
            ("start_scene", manifest.start_scene.as_deref()),
            ("persistent_scene", manifest.persistent_scene.as_deref()),
        ];
        for (field, path) in paths {
            if let Some(path) = path {
                crate::check_path(field, path)?;
            }
        }
        Ok(manifest)
    }
}

impl Project {
    /// The exported game whose `orrin.json` is at `manifest_path`.
    pub fn from_export(manifest_path: &Path) -> Result<Project, ProjectError> {
        let export = ExportManifest::read(manifest_path)?;
        let dir = manifest_path.parent().unwrap_or(Path::new("."));
        let root = dir.canonicalize().map_err(|source| ProjectError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        Ok(Project {
            game_assembly: Some(root.join(&export.game_assembly)),
            root,
            manifest: Manifest {
                format_version: export.format_version,
                project: ProjectSection {
                    name: export.name,
                    version: export.version,
                },
                // An export compiles nothing; the assembly is named outright.
                scripts: ScriptsSection {
                    dir: PathBuf::from("."),
                    entry: export.entry,
                },
                assets: Some(crate::AssetsSection { dir: export.assets }),
                scenes: Some(SceneSection {
                    start: export.start_scene,
                    persistent: export.persistent_scene,
                }),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROJECT: &str = r#"
format_version = 1

[project]
name = "space-shooter"
version = "0.3.0"

[scripts]
dir = "scripts"
entry = "SpaceShooter.Main, SpaceShooter"
"#;

    fn project() -> (tempfile::TempDir, Project) {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("orrin.toml"), PROJECT).expect("write manifest");
        let project = Project::load(&dir.path().join("orrin.toml")).expect("valid manifest");
        (dir, project)
    }

    #[test]
    fn a_written_manifest_reads_back_as_the_same_game() {
        let (_project_dir, project) = project();
        let mut manifest = ExportManifest::new(&project, Platform::Linux, "SpaceShooter.dll");
        manifest.start_scene = Some(PathBuf::from("compiled-assets/scenes/level.orrin"));

        let out = tempfile::tempdir().expect("tempdir");
        manifest.write(out.path()).expect("write");
        let read = ExportManifest::read(&out.path().join(FILE_NAME)).expect("read");

        assert_eq!(read, manifest);
        assert_eq!(read.name, "space-shooter");
        assert_eq!(read.entry, "SpaceShooter.Main, SpaceShooter");
    }

    #[test]
    fn the_platform_is_written_as_the_word_the_cli_takes() {
        let (_project_dir, project) = project();
        let out = tempfile::tempdir().expect("tempdir");
        ExportManifest::new(&project, Platform::Windows, "SpaceShooter.dll")
            .write(out.path())
            .expect("write");

        let text = std::fs::read_to_string(out.path().join(FILE_NAME)).expect("read");
        assert!(text.contains(r#""platform": "windows""#), "{text}");
        for platform in Platform::ALL {
            assert_eq!(Platform::parse(platform.name()), Some(platform));
        }
    }

    #[test]
    fn an_export_resolves_through_the_same_accessors_as_a_project() {
        let (_project_dir, project) = project();
        let mut manifest = ExportManifest::new(&project, Platform::Mac, "SpaceShooter.dll");
        manifest.start_scene = Some(PathBuf::from("compiled-assets/scenes/level.orrin"));
        let out = tempfile::tempdir().expect("tempdir");
        manifest.write(out.path()).expect("write");
        let root = out.path().canonicalize().expect("canonicalize");

        let game = Project::from_export(&root.join(FILE_NAME)).expect("export loads");

        assert_eq!(game.name(), "space-shooter");
        assert_eq!(game.version(), "0.3.0");
        assert_eq!(game.entry_type(), "SpaceShooter.Main, SpaceShooter");
        assert_eq!(game.assets_dir(), root.join("compiled-assets"));
        assert_eq!(
            game.start_scene(),
            Some(root.join("compiled-assets/scenes/level.orrin"))
        );
        assert_eq!(game.persistent_scene(), None);
        assert_eq!(game.game_assembly(), Some(root.join("SpaceShooter.dll")));
        assert_eq!(project.game_assembly(), None, "a project compiles its own");
    }

    #[test]
    fn a_path_out_of_the_export_is_refused_by_field() {
        let (_project_dir, project) = project();
        let mut manifest = ExportManifest::new(&project, Platform::Mac, "SpaceShooter.dll");
        manifest.assets = PathBuf::from("../elsewhere");
        let out = tempfile::tempdir().expect("tempdir");
        manifest.write(out.path()).expect("write");

        let err = ExportManifest::read(&out.path().join(FILE_NAME)).expect_err("`..` refused");
        assert!(
            matches!(
                err,
                ProjectError::ParentDirInPath {
                    field: "assets",
                    ..
                }
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn a_field_this_engine_does_not_know_is_an_error_not_ignored() {
        let out = tempfile::tempdir().expect("tempdir");
        let path = out.path().join(FILE_NAME);
        std::fs::write(
            &path,
            r#"{"format_version":1,"name":"a","version":"1","platform":"mac","entry":"A.B, A",
                "game_assembly":"A.dll","assets":"compiled-assets","splash":"x.png"}"#,
        )
        .expect("write");

        let err = ExportManifest::read(&path).expect_err("unknown field");
        assert!(matches!(err, ProjectError::Parse { .. }), "{err}");
    }
}
