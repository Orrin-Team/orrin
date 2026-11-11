mod build;
mod engine;
mod export;
mod new;
mod run;
mod test;

use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use orrin_project::Project;
use orrin_project::export::Platform;

/// Every command returns a message already phrased for the user; `main` is the
/// only place that decides how a failure is printed and what it exits with.
pub type Fallible = Result<(), String>;

#[derive(Parser)]
#[command(
    name = "orrin",
    version,
    about = "Create, build, and run Orrin projects",
    subcommand_required = true,
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Scaffold a new project directory
    New {
        /// Project name, and the directory created for it
        name: String,

        /// Parent directory to create the project in
        #[arg(long, value_name = "DIR", default_value = ".")]
        path: PathBuf,
    },

    /// Compile the project's scripts and index its assets
    Build {
        #[command(flatten)]
        project: ProjectArgs,

        /// Export a playable game instead; the same as `orrin export`
        #[arg(long)]
        export: bool,

        /// With --export: the platform to export for
        #[arg(long, value_name = "PLATFORM", requires = "export", value_parser = parse_platform)]
        platform: Option<Platform>,
    },

    /// Open Orrin Hub: create, open and export projects in a window
    Hub,

    /// Package the project as a game a player can run, with no SDK needed
    Export {
        /// Directory to search upward from for orrin.toml (default: current)
        #[arg(long, value_name = "DIR")]
        project: Option<PathBuf>,

        /// mac, linux or windows (default: this machine)
        #[arg(long, value_name = "PLATFORM", value_parser = parse_platform)]
        platform: Option<Platform>,

        /// Where to put the game (default: exports/<name>-<platform> in the project)
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,
    },

    /// Build the project, then launch the engine on it
    Run {
        #[command(flatten)]
        project: ProjectArgs,

        /// Launch whatever is already compiled instead of building first
        #[arg(long)]
        no_build: bool,

        /// From an engine checkout, run the dev build of the engine (faster to
        /// compile, slower to run) instead of the release one
        #[arg(long)]
        dev: bool,

        /// Arguments forwarded to the engine, after `--`
        #[arg(last = true, value_name = "ARGS")]
        engine_args: Vec<String>,
    },

    /// Build the project, then run its tests in the engine with no window;
    /// exits non-zero if one fails
    Test {
        /// Run only the tests whose name contains this
        filter: Option<String>,

        #[command(flatten)]
        project: ProjectArgs,

        /// Run whatever is already compiled instead of building first
        #[arg(long)]
        no_build: bool,

        /// From an engine checkout, use the dev build of the engine instead of
        /// the release one
        #[arg(long)]
        dev: bool,

        /// How the engine prints the results: a line a test, or one JSON
        /// document for CI
        #[arg(long, value_enum, default_value = "text")]
        format: test::Format,
    },
}

#[derive(Args)]
pub struct ProjectArgs {
    /// Directory to search upward from for orrin.toml (default: current)
    #[arg(long, value_name = "DIR")]
    project: Option<PathBuf>,

    /// Build the scripts in the Release configuration
    #[arg(long)]
    release: bool,
}

impl ProjectArgs {
    pub fn configuration(&self) -> &'static str {
        if self.release { "Release" } else { "Debug" }
    }

    /// The project this invocation acts on, or a message explaining why there
    /// isn't one.
    pub fn locate(&self) -> Result<Project, String> {
        let start = match &self.project {
            Some(dir) => {
                if !dir.is_dir() {
                    return Err(format!("`--project {}` is not a directory", dir.display()));
                }
                dir.clone()
            }
            None => std::env::current_dir()
                .map_err(|err| format!("cannot read the current directory: {err}"))?,
        };

        match Project::locate(&start) {
            Ok(Some(project)) => Ok(project),
            Ok(None) => Err(format!(
                "no `orrin.toml` in {} or any parent directory; \
                 run `orrin new <name>` to create a project",
                start.display()
            )),
            Err(err) => Err(err.to_string()),
        }
    }
}

/// Path relative to the current directory when that is shorter to read than
/// the absolute one — project roots are canonicalized, so unabbreviated output
/// is a wall of path.
pub fn display_path(path: &Path) -> String {
    let relative = std::env::current_dir()
        .ok()
        .and_then(|cwd| path.strip_prefix(cwd).ok().map(Path::to_path_buf));

    match relative {
        Some(rel) if !rel.as_os_str().is_empty() => rel.display().to_string(),
        Some(_) => ".".to_string(),
        None => path.display().to_string(),
    }
}

fn parse_platform(name: &str) -> Result<Platform, String> {
    Platform::parse(name)
        .ok_or_else(|| format!("`{name}` is not a platform; use mac, linux or windows"))
}

fn export(project: &ProjectArgs, options: export::Options) -> Fallible {
    export::export(&project.locate()?, &options)
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Command::New { name, path } => new::scaffold(&name, &path),
        Command::Build {
            project,
            export: true,
            platform,
        } => export(
            &project,
            export::Options {
                platform,
                out: None,
            },
        ),
        Command::Build { project, .. } => build::build(&project).map(|_| ()),
        Command::Hub => engine::launch_hub(),
        Command::Export {
            project,
            platform,
            out,
        } => export(
            &ProjectArgs {
                project,
                release: true,
            },
            export::Options { platform, out },
        ),
        Command::Run {
            project,
            no_build,
            dev,
            engine_args,
        } => run::run(&project, no_build, dev, &engine_args),
        Command::Test {
            filter,
            project,
            no_build,
            dev,
            format,
        } => test::test(&project, filter.as_deref(), no_build, dev, format)
            .map(|code| std::process::exit(code)),
    };

    if let Err(message) = result {
        eprintln!("orrin: {message}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Command, clap::Error> {
        Cli::try_parse_from(std::iter::once("orrin").chain(args.iter().copied()))
            .map(|cli| cli.command)
    }

    #[test]
    fn test_takes_an_optional_filter_and_the_run_flags() {
        let Ok(Command::Test {
            filter,
            project,
            no_build,
            dev,
            format,
        }) = parse(&[
            "test",
            "Falls",
            "--release",
            "--no-build",
            "--dev",
            "--project",
            "game",
            "--format",
            "json",
        ])
        else {
            panic!("`orrin test` did not parse");
        };
        assert_eq!(filter.as_deref(), Some("Falls"));
        assert!(project.release && no_build && dev);
        assert_eq!(project.project.as_deref(), Some(Path::new("game")));
        assert_eq!(format, test::Format::Json);

        let Ok(Command::Test { filter, format, .. }) = parse(&["test"]) else {
            panic!("a bare `orrin test` did not parse");
        };
        assert_eq!(filter, None);
        assert_eq!(format, test::Format::Text);
    }

    #[test]
    fn test_refuses_a_format_the_engine_does_not_print() {
        assert!(parse(&["test", "--format", "xml"]).is_err());
        assert!(parse(&["test", "One", "Two"]).is_err());
    }

    /// What reaches the engine: its own `--test` mode, with the filter and the
    /// format passed through and nothing when they were not given.
    #[test]
    fn the_engine_is_asked_for_its_test_mode() {
        assert_eq!(test::engine_args(None, test::Format::Text), ["--test"]);
        assert_eq!(
            test::engine_args(Some("Falls"), test::Format::Json),
            ["--test", "Falls", "--format", "json"]
        );
    }
}
