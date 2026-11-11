use crate::engine::{self, BINDINGS_ENV};
use crate::{ProjectArgs, display_path, run};

/// How the engine prints the results: `--format` of its `--test` mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Format {
    Text,
    Json,
}

/// The engine's own arguments for its headless test mode.
pub fn engine_args(filter: Option<&str>, format: Format) -> Vec<String> {
    let mut args = vec!["--test".to_owned()];
    args.extend(filter.map(str::to_owned));
    if format == Format::Json {
        args.extend(["--format".to_owned(), "json".to_owned()]);
    }
    args
}

/// Build the scripts as `orrin run` does, then run the project's tests in the
/// engine with no window. Returns the engine's exit status, which is the
/// command's: 0 when every test passed, 1 when one failed or the filter matched
/// none, 2 when the tests could not be run.
pub fn test(
    args: &ProjectArgs,
    filter: Option<&str>,
    no_build: bool,
    dev: bool,
    format: Format,
) -> Result<i32, String> {
    let (project, bindings) = run::prepare(args, no_build)?;
    let engine = engine::find(project.root(), !dev)?;
    let mut command = engine.command();
    command.current_dir(project.root());
    if let Some(bindings) = &bindings
        && std::env::var_os(BINDINGS_ENV).is_none()
    {
        command.env(BINDINGS_ENV, bindings);
    }
    if let Some(separator) = engine.args_separator() {
        command.arg(separator);
    }
    command.args(engine_args(filter, format));

    // To stderr, so that stdout under `--format json` is the engine's report.
    eprintln!("orrin: testing `{}` with {}", project.name(), engine.origin);

    let status = command
        .status()
        .map_err(|err| format!("could not launch the engine ({}): {err}", engine.origin))?;
    status.code().ok_or_else(|| {
        format!(
            "the engine was terminated by a signal while testing {}",
            display_path(project.root())
        )
    })
}
