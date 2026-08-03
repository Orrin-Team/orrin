//! Compiling an Orrin project's C# with `dotnet build`.
//!
//! Both the `orrin` CLI and the engine's rebuild-on-save watcher shell out to
//! the same compiler, so the invocation lives here: a project builds one way
//! whether a developer typed `orrin build` or just saved a file with the editor
//! open. The two differ only in where the output goes — a terminal build streams
//! it, the watcher captures it for the editor console.

use std::collections::HashSet;
use std::io;
use std::path::Path;
use std::process::Command;

/// One `dotnet build` invocation.
pub struct Build<'a> {
    /// Directory holding the `.csproj` to build.
    pub project_dir: &'a Path,
    /// MSBuild configuration — `Debug` or `Release`.
    pub configuration: &'a str,
    /// Directory holding the engine's built `Orrin.dll`, passed as the
    /// `OrrinBindings` property. A game csproj generated outside the engine
    /// tree resolves the bindings through it; inside the tree the reference is
    /// a `ProjectReference` and the property goes unused.
    pub bindings: Option<&'a Path>,
}

/// Where the compiler's output goes.
pub enum Output {
    /// Straight to this process's stdout/stderr, as a terminal build should.
    Inherit,
    /// Into [`Report::log`], for a caller with somewhere else to put it.
    Capture,
}

/// A build that ran to completion — successfully or not.
pub struct Report {
    pub success: bool,
    /// Combined stdout and stderr; empty under [`Output::Inherit`], which sent
    /// them to the terminal instead.
    pub log: String,
}

/// `dotnet build` could not be run at all. Distinct from a build that ran and
/// reported errors, which is an `Ok` [`Report`] with `success == false`.
#[derive(Debug)]
pub enum Error {
    /// No `dotnet` on `PATH`.
    NotInstalled,
    Spawn(io::Error),
}

impl Error {
    fn from_io(err: io::Error) -> Self {
        if err.kind() == io::ErrorKind::NotFound {
            Self::NotInstalled
        } else {
            Self::Spawn(err)
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInstalled => {
                write!(
                    f,
                    "`dotnet` is not installed or not on PATH; install the .NET 10 SDK"
                )
            }
            Self::Spawn(err) => write!(f, "could not run `dotnet build`: {err}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn(err) => Some(err),
            Self::NotInstalled => None,
        }
    }
}

impl Build<'_> {
    pub fn run(&self, output: Output) -> Result<Report, Error> {
        let mut cmd = Command::new("dotnet");
        cmd.arg("build")
            .arg(self.project_dir)
            .args(["-c", self.configuration])
            .arg("--nologo");

        if let Some(bindings) = self.bindings {
            cmd.arg(format!("-p:OrrinBindings={}", bindings.display()));
        }

        match output {
            Output::Inherit => {
                let status = cmd.status().map_err(Error::from_io)?;
                Ok(Report {
                    success: status.success(),
                    log: String::new(),
                })
            }
            Output::Capture => {
                let output = cmd.output().map_err(Error::from_io)?;
                let mut log = String::from_utf8_lossy(&output.stdout).into_owned();
                log.push_str(&String::from_utf8_lossy(&output.stderr));
                Ok(Report {
                    success: output.status.success(),
                    log,
                })
            }
        }
    }
}

impl Report {
    /// The compiler's error and warning lines, in the order they were emitted.
    ///
    /// MSBuild prints each diagnostic twice — once as the compiler emits it and
    /// again in the summary after `Build FAILED.` — so an identical line is
    /// kept only the first time. The `[/path/to/Game.csproj]` suffix appended to
    /// every line names the same project each time and pushes the useful text
    /// off the end of a console panel, so it goes too.
    pub fn diagnostics(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        self.log
            .lines()
            .map(str::trim)
            .filter(|line| line.contains(": error ") || line.contains(": warning "))
            .map(strip_project_suffix)
            .filter(|line| seen.insert(line.to_string()))
            .collect()
    }

    /// The last `count` non-blank lines of the log.
    ///
    /// The fallback for a failure with no diagnostics to find: MSBuild can fail
    /// before the compiler ever runs — a missing SDK, an unresolvable reference,
    /// a malformed csproj — and then there is no `error CS….` line anywhere.
    /// The tail is what a developer would have read off a terminal.
    pub fn tail(&self, count: usize) -> Vec<String> {
        let mut lines: Vec<String> = self
            .log
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .rev()
            .take(count)
            .map(str::to_owned)
            .collect();
        lines.reverse();
        lines
    }
}

fn strip_project_suffix(line: &str) -> String {
    let trimmed = line
        .strip_suffix(']')
        .and_then(|rest| rest.rfind(" [").map(|at| &rest[..at]))
        .unwrap_or(line);
    trimmed.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(log: &str) -> Report {
        Report {
            success: false,
            log: log.to_owned(),
        }
    }

    #[test]
    fn diagnostics_are_extracted_without_the_project_suffix() {
        let report = report(concat!(
            "  Determining projects to restore...\n",
            "  MyGame -> /p/bin/Debug/net10.0/MyGame.dll\n",
            "/p/Player.cs(12,9): error CS0103: The name 'x' does not exist [/p/MyGame.csproj]\n",
        ));

        assert_eq!(
            report.diagnostics(),
            ["/p/Player.cs(12,9): error CS0103: The name 'x' does not exist"]
        );
    }

    #[test]
    fn the_msbuild_summary_repeat_is_dropped() {
        // MSBuild emits each diagnostic inline and again under `Build FAILED.`;
        // the console should show it once.
        let line = "/p/Player.cs(12,9): error CS0103: nope [/p/MyGame.csproj]";
        let report = report(&format!(
            "{line}\n\nBuild FAILED.\n\n{line}\n    1 Error(s)\n"
        ));

        assert_eq!(report.diagnostics().len(), 1);
    }

    #[test]
    fn warnings_are_diagnostics_too() {
        let report = report("/p/A.cs(1,1): warning CS0168: unused [/p/MyGame.csproj]\n");

        assert_eq!(report.diagnostics().len(), 1);
    }

    #[test]
    fn a_failure_with_no_diagnostics_falls_back_to_the_tail() {
        // No `error CS…` line to find: MSBuild gave up before the compiler ran.
        let report = report("  Restoring...\n\nMSB4236: The SDK 'Bogus' was not found.\n\n");

        assert!(report.diagnostics().is_empty());
        assert_eq!(
            report.tail(2),
            ["Restoring...", "MSB4236: The SDK 'Bogus' was not found."]
        );
    }
}
