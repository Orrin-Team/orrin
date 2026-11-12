use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

/// How the CLI will start the engine on the project it located.
pub struct Engine {
    pub launcher: Launcher,
    /// How the engine was found, for the line printed before launching.
    pub origin: String,
}

pub enum Launcher {
    /// A built engine executable: a shipped install, or `$ORRIN_ENGINE`.
    Exe(PathBuf),
    /// The engine's own checkout — cargo rebuilds it, so a contributor's
    /// `orrin run` always launches current sources. Scripting comes with the
    /// default features, so there is no flag to pass and no way to get a
    /// scriptless engine by forgetting one.
    Cargo {
        root: PathBuf,
        /// Built with `--release`: an editor someone makes a game in should
        /// run at the speed the game will, and the dev profile compiles the
        /// engine itself at opt-level 1 with debug assertions on.
        release: bool,
    },
}

impl Engine {
    pub fn command(&self) -> Command {
        match &self.launcher {
            Launcher::Exe(exe) => Command::new(exe),
            Launcher::Cargo { root, release } => {
                let mut cmd = Command::new("cargo");
                cmd.args(["run", "-p", "orrin-core"]);
                if *release {
                    cmd.arg("--release");
                }
                cmd.arg("--manifest-path").arg(root.join("Cargo.toml"));
                cmd
            }
        }
    }

    /// `cargo run` forwards the engine's own arguments after a second `--`.
    pub fn args_separator(&self) -> Option<&'static str> {
        match self.launcher {
            Launcher::Exe(_) => None,
            Launcher::Cargo { .. } => Some("--"),
        }
    }
}

const ENGINE_EXE: &str = "orrin-core";
const ENGINE_ENV: &str = "ORRIN_ENGINE";
pub const BINDINGS_ENV: &str = "ORRIN_SCRIPT_DIR";

/// Locate the engine, in the order a wrong guess would be most surprising.
///
/// An explicit `$ORRIN_ENGINE` always wins. A shipped install is recognized by
/// the engine executable and `Orrin.dll` sitting *together* — that pairing is
/// what an exported build lays down, and it is also what distinguishes it from
/// a cargo `target/` directory, where both engine and CLI binaries are
/// neighbours but no bindings are. Without that distinction the checkout case
/// below would be unreachable for contributors, who would silently get whatever
/// stale `target/debug/orrin-core` happened to exist rather than a rebuild of
/// what they are editing.
pub fn find(start: &Path, release: bool) -> Result<Engine, String> {
    if let Some(value) = std::env::var_os(ENGINE_ENV) {
        let exe = PathBuf::from(value);
        if !exe.is_file() {
            return Err(format!(
                "${ENGINE_ENV} points at {}, which is not a file",
                exe.display()
            ));
        }
        return Ok(Engine {
            origin: format!("${ENGINE_ENV} ({})", exe.display()),
            launcher: Launcher::Exe(exe),
        });
    }

    let installed = beside_executable();
    if let Some(exe) = &installed
        && exe.with_file_name("Orrin.dll").is_file()
    {
        return Ok(Engine {
            origin: format!("installed engine ({})", exe.display()),
            launcher: Launcher::Exe(exe.clone()),
        });
    }

    if let Some(root) = find_checkout(start) {
        return Ok(Engine {
            origin: format!(
                "the {} engine from the checkout at {}",
                if release { "release" } else { "dev" },
                root.display()
            ),
            launcher: Launcher::Cargo { root, release },
        });
    }

    if let Some(exe) = installed {
        return Ok(Engine {
            origin: format!("engine beside the CLI ({})", exe.display()),
            launcher: Launcher::Exe(exe),
        });
    }

    Err(format!(
        "cannot find the engine. Tried ${ENGINE_ENV}, `{ENGINE_EXE}` beside this \
         executable, and an engine checkout above {}. Set ${ENGINE_ENV} to an \
         engine binary, or run from inside an Orrin checkout.",
        start.display()
    ))
}

/// The CLI's own location, with symlinks resolved.
///
/// Putting a dev build on `$PATH` means symlinking it, and `current_exe`
/// reports the *link* on macOS while `/proc/self/exe` resolves it on Linux.
/// Left alone, that difference decides whether the checkout probe below finds
/// anything, so the same symlink would work on one platform and not the other.
fn current_exe() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(orrin_project::canonical(&exe).unwrap_or(exe))
}

fn beside_executable() -> Option<PathBuf> {
    let exe = current_exe()?;
    let candidate = exe.with_file_name(format!("{ENGINE_EXE}{}", std::env::consts::EXE_SUFFIX));
    candidate.is_file().then_some(candidate)
}

/// Walk up from `start`, then from the CLI's own location, for the engine's
/// source tree. Checking both means `orrin run` works on a project stored
/// outside the checkout as long as the CLI itself was built inside one.
pub fn find_checkout(start: &Path) -> Option<PathBuf> {
    let starts = [Some(start.to_path_buf()), current_exe()];

    starts
        .into_iter()
        .flatten()
        .find_map(|path| checkout_containing(&path))
}

/// The engine checkout `dir` actually lives inside, ignoring where the CLI was
/// built. Only this answers "may a generated file name the engine by relative
/// path" — a project outside the tree that pointed at the CLI's own checkout
/// would be committing a machine-local path, which the manifest rules forbid
/// and which would break for everyone else on the project.
pub fn checkout_containing(dir: &Path) -> Option<PathBuf> {
    dir.ancestors()
        .find(|dir| is_checkout(dir))
        .map(Path::to_path_buf)
}

fn is_checkout(dir: &Path) -> bool {
    dir.join("crates/core/Cargo.toml").is_file()
        && dir.join("scripting/Orrin/Orrin.csproj").is_file()
}

/// The directory holding the engine's built `Orrin.dll` bindings.
///
/// Resolution mirrors the engine's own: `$ORRIN_SCRIPT_DIR`, else beside the
/// executable (the shipped layout), else the checkout's build output. The CLI
/// resolves it so it can hand the path to both `dotnet build` (a game csproj
/// outside the engine tree references `Orrin.dll` by path) and the engine
/// itself, which cannot find a checkout's bindings once its working directory
/// is a project rather than the repo root.
pub fn find_bindings(start: &Path) -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os(BINDINGS_ENV) {
        return Some(PathBuf::from(dir));
    }

    if let Some(exe) = current_exe()
        && let Some(dir) = exe.parent()
        && dir.join("Orrin.dll").is_file()
    {
        return Some(dir.to_path_buf());
    }

    find_checkout(start).and_then(|root| built_bindings(&root))
}

/// `scripting/Orrin/bin/{Debug,Release}/net*/` — newest build wins, matching
/// how the engine probes the same tree.
pub fn built_bindings(checkout: &Path) -> Option<PathBuf> {
    let root = checkout.join("scripting/Orrin");
    let mut best: Option<(SystemTime, PathBuf)> = None;

    for configuration in ["Debug", "Release"] {
        let Ok(entries) = std::fs::read_dir(root.join("bin").join(configuration)) else {
            continue;
        };
        for entry in entries.flatten() {
            let dir = entry.path();
            let dll = dir.join("Orrin.dll");
            if !dll.is_file() || !dir.join("Orrin.runtimeconfig.json").is_file() {
                continue;
            }
            let modified = dll
                .metadata()
                .and_then(|meta| meta.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            if best.as_ref().is_none_or(|(newest, _)| modified > *newest) {
                best = Some((modified, dir));
            }
        }
    }

    best.map(|(_, dir)| dir)
}

pub const ANALYZER_DLL: &str = "Orrin.Analyzers.dll";

/// Copy the checkout's `Orrin.Analyzers.dll` beside `bindings` when they are
/// that checkout's `scripting/Orrin/bin/<configuration>/<tfm>/`.
///
/// A game csproj outside the tree finds the analyzer only beside `Orrin.dll`,
/// and building the bindings does not put it there: they reference it as an
/// analyzer, which is never copied to an output. Copying it from
/// `Orrin.csproj` instead would also put it in the bindings' publish, which an
/// export ships whole. Any other directory is left alone: an install lays down
/// its own analyzer, and a missing one costs only the warning.
pub fn place_analyzer(bindings: &Path) -> Result<(), String> {
    let Ok(bindings) = bindings.canonicalize() else {
        return Ok(());
    };
    let Some(checkout) = checkout_containing(&bindings) else {
        return Ok(());
    };
    let (Some(configuration), Some(bin)) = (
        bindings.parent().and_then(Path::file_name),
        bindings.parent().and_then(Path::parent),
    ) else {
        return Ok(());
    };
    if bin != checkout.join("scripting/Orrin/bin") {
        return Ok(());
    }

    // The compiler loads analyzers into its own netstandard2.0 host, so the
    // analyzer has no other framework to be built for.
    let built = checkout
        .join("scripting/Orrin.Analyzers/bin")
        .join(configuration)
        .join("netstandard2.0")
        .join(ANALYZER_DLL);
    let Ok(contents) = std::fs::read(&built) else {
        return Ok(());
    };
    let placed = bindings.join(ANALYZER_DLL);
    // Rewritten only when it changed, since the analyzer is an input to the
    // game's compile and a fresh timestamp would rebuild it every time.
    if std::fs::read(&placed).is_ok_and(|current| current == contents) {
        return Ok(());
    }
    std::fs::write(&placed, contents)
        .map_err(|err| format!("could not write {}: {err}", placed.display()))
}

/// Start Orrin Hub and return without waiting for it: the one installed beside
/// this CLI, else a build of the checkout this CLI came from.
pub fn launch_hub() -> Result<(), String> {
    let name = format!("orrin-hub{}", std::env::consts::EXE_SUFFIX);
    let mut command = match current_exe().and_then(|exe| exe.parent().map(|dir| dir.join(&name))) {
        Some(hub) if hub.is_file() => Command::new(hub),
        _ => match current_exe().and_then(|exe| checkout_containing(&exe)) {
            Some(root) => {
                println!(
                    "orrin: building and opening Orrin Hub from {}",
                    root.display()
                );
                let mut cargo = Command::new("cargo");
                cargo
                    .args(["run", "--quiet", "-p", "orrin-hub", "--manifest-path"])
                    .arg(root.join("Cargo.toml"));
                cargo
            }
            None => {
                return Err(format!(
                    "Orrin Hub is not installed: there is no `{name}` beside this executable"
                ));
            }
        },
    };
    command
        .spawn()
        .map(|_| ())
        .map_err(|err| format!("could not open Orrin Hub: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(release: bool) -> Vec<String> {
        let engine = Engine {
            launcher: Launcher::Cargo {
                root: PathBuf::from("/orrin"),
                release,
            },
            origin: String::new(),
        };
        engine
            .command()
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    fn touch(path: &Path, contents: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
    }

    fn fake_checkout() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        touch(&dir.path().join("crates/core/Cargo.toml"), "");
        touch(&dir.path().join("scripting/Orrin/Orrin.csproj"), "");
        dir
    }

    /// A project outside the checkout references the analyzer by path beside
    /// `Orrin.dll`, so a checkout's bindings have to carry the analyzer built
    /// in the same configuration.
    #[test]
    fn a_checkouts_bindings_get_the_analyzer_built_beside_them() {
        let dir = fake_checkout();
        let root = dir.path().canonicalize().unwrap();
        let bindings = root.join("scripting/Orrin/bin/Release/net10.0");
        touch(&bindings.join("Orrin.dll"), "bindings");
        let analyzers = root.join("scripting/Orrin.Analyzers/bin");
        touch(
            &analyzers.join("Debug/netstandard2.0/Orrin.Analyzers.dll"),
            "debug",
        );
        touch(
            &analyzers.join("Release/netstandard2.0/Orrin.Analyzers.dll"),
            "release",
        );

        place_analyzer(&bindings).unwrap();
        let placed = bindings.join(ANALYZER_DLL);
        assert_eq!(std::fs::read_to_string(&placed).unwrap(), "release");

        touch(
            &analyzers.join("Release/netstandard2.0/Orrin.Analyzers.dll"),
            "rebuilt",
        );
        place_analyzer(&bindings).unwrap();
        assert_eq!(std::fs::read_to_string(&placed).unwrap(), "rebuilt");
    }

    /// Only a checkout's own build output is written to: a directory named by
    /// `$ORRIN_SCRIPT_DIR` or an install is not the CLI's to add files to, and
    /// an analyzer that was never built is no reason to fail a build.
    #[test]
    fn nothing_is_placed_outside_a_checkouts_build_output() {
        let dir = fake_checkout();
        let root = dir.path().canonicalize().unwrap();
        touch(
            &root.join("scripting/Orrin.Analyzers/bin/Debug/netstandard2.0/Orrin.Analyzers.dll"),
            "debug",
        );

        let elsewhere = root.join("installed");
        touch(&elsewhere.join("Orrin.dll"), "bindings");
        place_analyzer(&elsewhere).unwrap();
        assert!(!elsewhere.join(ANALYZER_DLL).exists());

        let unbuilt = root.join("scripting/Orrin/bin/Release/net10.0");
        touch(&unbuilt.join("Orrin.dll"), "bindings");
        place_analyzer(&unbuilt).unwrap();
        assert!(!unbuilt.join(ANALYZER_DLL).exists());
    }

    /// The editor `orrin run` opens from a checkout is the optimised engine,
    /// and the dev build is there when asked for.
    #[test]
    fn a_checkout_engine_runs_in_release_unless_asked_for_dev() {
        let release = args(true);
        assert!(release.iter().any(|arg| arg == "--release"), "{release:?}");
        assert_eq!(&release[..3], ["run", "-p", "orrin-core"]);

        let dev = args(false);
        assert!(!dev.iter().any(|arg| arg == "--release"), "{dev:?}");
    }
}
