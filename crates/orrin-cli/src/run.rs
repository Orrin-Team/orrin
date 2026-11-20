use std::path::{Path, PathBuf};
use std::process::Command;

use orrin_project::Project;

use crate::build::Log;
use crate::engine::{self, BINDINGS_ENV, Engine};
use crate::{Fallible, ProjectArgs, build, display_path};

/// The project and its bindings, built first unless `no_build`: what `run`,
/// `test` and `orrin mcp`'s `editor_open` all launch the engine on.
pub fn prepare(
    args: &ProjectArgs,
    no_build: bool,
    log: &mut Log,
) -> Result<(Project, Option<PathBuf>), String> {
    if no_build {
        let project = args.locate()?;
        let bindings = engine::find_bindings(project.root());
        Ok((project, bindings))
    } else {
        let built = build::build(args, log)?;
        Ok((built.project, built.bindings))
    }
}

pub fn run(args: &ProjectArgs, no_build: bool, dev: bool, engine_args: &[String]) -> Fallible {
    let (project, bindings) = prepare(args, no_build, &mut Log::Terminal)?;
    let engine = engine::find(project.root(), !dev)?;
    let mut command = command(&project, bindings.as_deref(), &engine, engine_args);

    println!("orrin: running `{}` with {}", project.name(), engine.origin);

    let status = command.status().map_err(|err| {
        if err.kind() == std::io::ErrorKind::NotFound {
            format!("could not launch the engine ({}): not found", engine.origin)
        } else {
            format!("could not launch the engine ({}): {err}", engine.origin)
        }
    })?;

    if !status.success() {
        return Err(match status.code() {
            Some(code) => format!(
                "the engine exited with status {code} (project {})",
                display_path(project.root())
            ),
            None => "the engine was terminated by a signal".to_string(),
        });
    }

    Ok(())
}

/// The editor on `project`, as `orrin run` starts it.
pub fn command(
    project: &Project,
    bindings: Option<&Path>,
    engine: &Engine,
    engine_args: &[String],
) -> Command {
    let mut command = engine.command();

    // The engine locates the project by walking up from its working directory,
    // so this is what points it at this project rather than the demo.
    command.current_dir(project.root());

    // A checkout's bindings live under the repo, which the engine's own
    // cwd-relative probe cannot reach now that cwd is the project. Passing the
    // resolved path is what lets a project be run from anywhere.
    if let Some(bindings) = bindings
        && std::env::var_os(BINDINGS_ENV).is_none()
    {
        command.env(BINDINGS_ENV, bindings);
    }

    // So the editor's File → Export game runs this same CLI rather than
    // searching for one.
    if let Ok(cli) = std::env::current_exe() {
        command.env("ORRIN_CLI", cli);
    }

    if !engine_args.is_empty() {
        if let Some(separator) = engine.args_separator() {
            command.arg(separator);
        }
        command.args(engine_args);
    }

    command
}
