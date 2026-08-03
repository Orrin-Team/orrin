use crate::engine::{self, BINDINGS_ENV};
use crate::{Fallible, ProjectArgs, build, display_path};

pub fn run(args: &ProjectArgs, no_build: bool, engine_args: &[String]) -> Fallible {
    let (project, bindings) = if no_build {
        let project = args.locate()?;
        let bindings = engine::find_bindings(project.root());
        (project, bindings)
    } else {
        let built = build::build(args)?;
        (built.project, built.bindings)
    };

    let engine = engine::find(project.root())?;

    let mut command = engine.command();

    // The engine locates the project by walking up from its working directory,
    // so this is what points it at this project rather than the demo.
    command.current_dir(project.root());

    // A checkout's bindings live under the repo, which the engine's own
    // cwd-relative probe cannot reach now that cwd is the project. Passing the
    // resolved path is what lets a project be run from anywhere.
    if let Some(bindings) = &bindings
        && std::env::var_os(BINDINGS_ENV).is_none()
    {
        command.env(BINDINGS_ENV, bindings);
    }

    if !engine_args.is_empty() {
        if let Some(separator) = engine.args_separator() {
            command.arg(separator);
        }
        command.args(engine_args);
    }

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
