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
    /// `orrin run` always launches current sources with scripting compiled in.
    Cargo { root: PathBuf },
}

impl Engine {
    pub fn command(&self) -> Command {
        match &self.launcher {
            Launcher::Exe(exe) => Command::new(exe),
            Launcher::Cargo { root } => {
                let mut cmd = Command::new("cargo");
                cmd.args(["run", "-p", "orrin-core", "--features", "scripting"])
                    .arg("--manifest-path")
                    .arg(root.join("Cargo.toml"));
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
/// below would be unreachable for contributors, who would silently get
/// whatever stale `target/debug/orrin-core` happened to exist — possibly one
/// built without `--features scripting`, so scripts would just never run.
pub fn find(start: &Path) -> Result<Engine, String> {
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
            origin: format!("engine checkout at {}", root.display()),
            launcher: Launcher::Cargo { root },
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
    Some(exe.canonicalize().unwrap_or(exe))
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
    dir.ancestors().find(|dir| is_checkout(dir)).map(Path::to_path_buf)
}

fn is_checkout(dir: &Path) -> bool {
    dir.join("crates/core/Cargo.toml").is_file() && dir.join("scripting/Orrin/Orrin.csproj").is_file()
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
