//! Opening a script's source, at a line, in the user's own editor.
//!
//! Two halves. [`ScriptFiles`] turns the name a log line is attributed to back
//! into a file. A script reports only the *name* of the file it was compiled
//! from — `Native.ScriptName` drops the directory, because that path is the
//! compiling machine's — so the file is found again by name under the scripts
//! directory, and a name two files share is answered with nothing rather than
//! a guess. [`command`] decides how to open it: `ORRIN_EDITOR` when set, else
//! VS Code's `code -g` when it is on the PATH, else the platform's opener,
//! which knows no line. It is pure so that choice is tested in CI, where no
//! editor is installed; [`open`] is the spawn around it.
//!
//! `ORRIN_EDITOR` is split on whitespace, with `{file}` and `{line}` replaced
//! inside each argument — `rider --line {line} {file}`, `zed {file}:{line}` —
//! and the file appended when no argument names `{file}`. There is no shell
//! quoting, so an editor whose path has a space in it is named by the PATH.
//!
//! The child is spawned and waited on by a thread of its own: exec and a slow
//! editor CLI stay off the frame, the child is reaped, and a failure reaches
//! the console as a warning through the log stream, which any thread may write.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub const EDITOR_ENV: &str = "ORRIN_EDITOR";

/// VS Code's command-line launcher. On Windows it is a batch file in the
/// install's `bin`, which a search for `code.exe` alone would miss.
#[cfg(windows)]
const CODE: &[&str] = &["code.cmd", "code.exe"];
#[cfg(not(windows))]
const CODE: &[&str] = &["code"];

/// What opens a file with whatever the platform associates it with.
const OPENER: &str = if cfg!(target_os = "macos") {
    "open"
} else if cfg!(windows) {
    "explorer"
} else {
    "xdg-open"
};

/// How old an index may be before a lookup walks the directory again. A script
/// created mid-session becomes openable this long after, and a console that is
/// only being looked at walks nothing.
const RESCAN: Duration = Duration::from_secs(2);

/// Script files under the scripts directory, by the name a log line gives them.
#[derive(Default)]
pub struct ScriptFiles {
    dir: Option<PathBuf>,
    /// `None` for a name more than one file has.
    index: HashMap<String, Option<PathBuf>>,
    scanned: Option<Instant>,
}

impl ScriptFiles {
    pub fn new(dir: Option<PathBuf>) -> Self {
        Self {
            dir,
            ..Self::default()
        }
    }

    /// The one `<name>.cs` under the scripts directory, as an absolute path.
    pub fn find(&mut self, name: &str, now: Instant) -> Option<PathBuf> {
        let dir = self.dir.as_ref()?;
        if self
            .scanned
            .is_none_or(|scanned| now.duration_since(scanned) >= RESCAN)
        {
            self.index.clear();
            if let Ok(root) = std::path::absolute(dir) {
                scan(&root, &mut self.index);
            }
            self.scanned = Some(now);
        }
        self.index.get(name)?.clone()
    }
}

fn scan(dir: &Path, index: &mut HashMap<String, Option<PathBuf>>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        // Not `is_dir` on the path, which follows links: a symlinked directory
        // could lead back up the tree.
        if kind.is_dir() {
            let name = entry.file_name();
            let skip = name == "bin" || name == "obj" || name.to_string_lossy().starts_with('.');
            if !skip {
                scan(&path, index);
            }
        } else if path.extension().is_some_and(|ext| ext == "cs")
            && let Some(stem) = path.file_stem().and_then(|stem| stem.to_str())
        {
            index
                .entry(stem.to_owned())
                .and_modify(|found| *found = None)
                .or_insert(Some(path));
        }
    }
}

/// A program and its arguments, as [`open`] would run them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    pub program: PathBuf,
    pub args: Vec<OsString>,
}

/// How to open `file` at `line` (1-based; `0` for unknown): `editor` is
/// `ORRIN_EDITOR`'s value, and `on_path` finds a program by name.
pub fn command(
    editor: Option<&str>,
    file: &Path,
    line: u32,
    on_path: &dyn Fn(&str) -> Option<PathBuf>,
) -> Launch {
    let line = line.max(1);
    if let Some(editor) = editor.filter(|editor| !editor.trim().is_empty()) {
        let mut words = editor.split_whitespace();
        let program = words.next().unwrap_or_default().into();
        let file = file.to_string_lossy();
        let mut named = false;
        let mut args: Vec<OsString> = words
            .map(|word| {
                named |= word.contains("{file}");
                word.replace("{file}", &file)
                    .replace("{line}", &line.to_string())
                    .into()
            })
            .collect();
        if !named {
            args.push(file.into_owned().into());
        }
        return Launch { program, args };
    }
    if let Some(code) = CODE.iter().find_map(|name| on_path(name)) {
        let mut at = file.as_os_str().to_owned();
        at.push(format!(":{line}"));
        return Launch {
            program: code,
            args: vec!["-g".into(), at],
        };
    }
    Launch {
        program: OPENER.into(),
        args: vec![file.into()],
    }
}

/// The first `name` on `path`, a `PATH` value.
pub fn on_path(path: Option<&OsString>, name: &str) -> Option<PathBuf> {
    std::env::split_paths(path?)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

/// Open `file` at `line` in the user's editor, without waiting for it.
pub fn open(file: &Path, line: u32) {
    let editor = std::env::var(EDITOR_ENV).ok();
    let path = std::env::var_os("PATH");
    let launch = command(editor.as_deref(), file, line, &|name| {
        on_path(path.as_ref(), name)
    });
    let shown = file.display().to_string();
    let spawned = std::thread::Builder::new()
        .name("open in editor".into())
        .spawn(move || {
            let program = launch.program.display();
            let status = std::process::Command::new(&launch.program)
                .args(&launch.args)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .status();
            // `explorer` exits 1 when it has opened the file, so its status
            // says nothing either way.
            let verdict = !(cfg!(windows) && launch.program == Path::new(OPENER));
            let failure = match status {
                Ok(status) if status.success() || !verdict => return,
                Ok(status) => status.to_string(),
                Err(error) => error.to_string(),
            };
            tracing::warn!(
                "could not open {shown} with `{program}`: {failure} \
                 (set {EDITOR_ENV} to the editor to use, e.g. `rider --line {{line}} {{file}}`)"
            );
        });
    if let Err(error) = spawned {
        tracing::warn!("could not open {}: {error}", file.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory, removed when dropped.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("orrin-source-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn file(&self, relative: &str) -> PathBuf {
            let path = self.0.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"").unwrap();
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn nothing_on_path(_: &str) -> Option<PathBuf> {
        None
    }

    fn args(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    const FILE: &str = "/game/scripts/Spinner.cs";

    #[test]
    fn the_override_fills_in_the_file_and_line_wherever_it_names_them() {
        let code = |_: &str| Some(PathBuf::from("/usr/bin/code"));
        assert_eq!(
            command(
                Some("rider --line {line} {file}"),
                Path::new(FILE),
                14,
                &code
            ),
            Launch {
                program: "rider".into(),
                args: args(&["--line", "14", FILE]),
            },
            "the override wins over a `code` on the PATH"
        );
        assert_eq!(
            command(Some("zed {file}:{line}"), Path::new(FILE), 14, &code).args,
            args(&["/game/scripts/Spinner.cs:14"])
        );
    }

    #[test]
    fn an_override_that_names_no_file_is_handed_it_last() {
        assert_eq!(
            command(
                Some("  gedit  --new-window "),
                Path::new(FILE),
                14,
                &nothing_on_path
            ),
            Launch {
                program: "gedit".into(),
                args: args(&["--new-window", FILE]),
            }
        );
    }

    #[test]
    fn a_blank_override_is_no_override() {
        let code = |name: &str| (name == CODE[0]).then(|| PathBuf::from("/usr/bin/code"));
        assert_eq!(
            command(Some("   "), Path::new(FILE), 14, &code).program,
            PathBuf::from("/usr/bin/code")
        );
    }

    #[test]
    fn vs_code_on_the_path_is_asked_to_go_to_the_line() {
        let code = |name: &str| (name == CODE[0]).then(|| PathBuf::from("/opt/code/bin/code"));
        assert_eq!(
            command(None, Path::new(FILE), 14, &code),
            Launch {
                program: "/opt/code/bin/code".into(),
                args: args(&["-g", "/game/scripts/Spinner.cs:14"]),
            }
        );
    }

    /// A contained exception is attributed to its Behaviour with no line.
    #[test]
    fn an_unknown_line_opens_at_the_top() {
        let code = |name: &str| (name == CODE[0]).then(|| PathBuf::from("code"));
        assert_eq!(
            command(None, Path::new(FILE), 0, &code).args,
            args(&["-g", "/game/scripts/Spinner.cs:1"])
        );
        assert_eq!(
            command(Some("subl {file}:{line}"), Path::new(FILE), 0, &code).args,
            args(&["/game/scripts/Spinner.cs:1"])
        );
    }

    #[test]
    fn with_no_editor_the_platform_opens_the_file_without_a_line() {
        assert_eq!(
            command(None, Path::new(FILE), 14, &nothing_on_path),
            Launch {
                program: OPENER.into(),
                args: args(&[FILE]),
            }
        );
    }

    #[test]
    fn a_program_is_found_on_the_first_path_entry_that_has_it() {
        let dir = Scratch::new("path");
        let first = dir.file("b/code");
        dir.file("c/code");
        let path =
            std::env::join_paths([dir.0.join("a"), dir.0.join("b"), dir.0.join("c")]).unwrap();

        assert_eq!(on_path(Some(&path), "code"), Some(first));
        assert_eq!(on_path(Some(&path), "subl"), None);
        assert_eq!(on_path(None, "code"), None);
    }

    #[test]
    fn a_script_is_found_by_name_anywhere_under_the_scripts_directory() {
        let dir = Scratch::new("find");
        let spinner = dir.file("Spinner.cs");
        let player = dir.file("Characters/Player.cs");
        let dotted = dir.file("Player.Movement.cs");
        let mut files = ScriptFiles::new(Some(dir.0.clone()));
        let now = Instant::now();

        assert_eq!(files.find("Spinner", now), Some(spinner));
        assert_eq!(files.find("Player", now), Some(player));
        assert_eq!(files.find("Player.Movement", now), Some(dotted));
        assert_eq!(files.find("Missing", now), None);
        assert_eq!(files.find("", now), None);
        assert_eq!(
            files.find("orrin_core::scripting", now),
            None,
            "an engine event names its module, not a file"
        );
    }

    #[test]
    fn a_name_two_scripts_share_finds_neither() {
        let dir = Scratch::new("twins");
        dir.file("Enemies/Brain.cs");
        dir.file("Allies/Brain.cs");
        let mut files = ScriptFiles::new(Some(dir.0.clone()));

        assert_eq!(files.find("Brain", Instant::now()), None);
    }

    /// `dotnet build` writes generated sources and copies under these, which
    /// would otherwise make every script's name ambiguous.
    #[test]
    fn build_output_is_not_a_script() {
        let dir = Scratch::new("build-output");
        let spinner = dir.file("Spinner.cs");
        dir.file("obj/Debug/net10.0/Spinner.cs");
        dir.file("bin/Debug/Spinner.cs");
        dir.file("obj/Debug/net10.0/Generated.cs");
        let mut files = ScriptFiles::new(Some(dir.0.clone()));
        let now = Instant::now();

        assert_eq!(files.find("Spinner", now), Some(spinner));
        assert_eq!(files.find("Generated", now), None);
    }

    #[cfg(unix)]
    #[test]
    fn a_relative_scripts_directory_still_finds_an_absolute_file() {
        let dir = Scratch::new("relative");
        dir.file("Spinner.cs");
        let cwd = std::env::current_dir().unwrap();
        let relative = pathdiff(&dir.0, &cwd);
        let mut files = ScriptFiles::new(Some(relative));

        let found = files.find("Spinner", Instant::now()).unwrap();
        assert!(found.is_absolute(), "{}", found.display());
        assert!(found.is_file(), "{}", found.display());
    }

    /// `to` from `from`, both absolute, by climbing to the root.
    #[cfg(unix)]
    fn pathdiff(to: &Path, from: &Path) -> PathBuf {
        let mut up = PathBuf::new();
        for _ in from.components().skip(1) {
            up.push("..");
        }
        up.join(to.strip_prefix("/").unwrap_or(to))
    }

    #[test]
    fn a_script_created_mid_session_is_found_once_the_index_is_stale() {
        let dir = Scratch::new("stale");
        dir.file("Spinner.cs");
        let mut files = ScriptFiles::new(Some(dir.0.clone()));
        let start = Instant::now();
        assert_eq!(files.find("Late", start), None);

        let late = dir.file("Late.cs");
        assert_eq!(
            files.find("Late", start + RESCAN / 2),
            None,
            "a fresh index is not walked again"
        );
        assert_eq!(files.find("Late", start + RESCAN), Some(late));
    }

    #[test]
    fn with_no_scripts_directory_nothing_is_found() {
        assert_eq!(ScriptFiles::default().find("Spinner", Instant::now()), None);
    }
}
