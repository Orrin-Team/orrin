mod build;
mod engine;
mod new;
mod run;

use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use orrin_project::Project;

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
    },

    /// Build the project, then launch the engine on it
    Run {
        #[command(flatten)]
        project: ProjectArgs,

        /// Launch whatever is already compiled instead of building first
        #[arg(long)]
        no_build: bool,

        /// Arguments forwarded to the engine, after `--`
        #[arg(last = true, value_name = "ARGS")]
        engine_args: Vec<String>,
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

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Command::New { name, path } => new::scaffold(&name, &path),
        Command::Build { project } => build::build(&project).map(|_| ()),
        Command::Run {
            project,
            no_build,
            engine_args,
        } => run::run(&project, no_build, &engine_args),
    };

    if let Err(message) = result {
        eprintln!("orrin: {message}");
        std::process::exit(1);
    }
}
