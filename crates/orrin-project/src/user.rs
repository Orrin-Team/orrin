//! The per-user directory: where Orrin keeps what belongs to the person at
//! this computer rather than to any project — the Hub's list of projects, the
//! editor's settings, their themes.
//!
//! Here rather than in the Hub or the engine because both read it, and two
//! copies of a platform table are two places for one of them to be wrong.
//!
//! `ORRIN_USER_DIR` names a different directory. It exists for tests that
//! start the real engine, which must not be configured by whoever's machine
//! they run on, and it is honoured everywhere so that one variable moves
//! everything Orrin keeps.

use std::ffi::OsString;
use std::path::PathBuf;

pub const DIR_ENV: &str = "ORRIN_USER_DIR";

/// `None` when the platform's own variable for it is unset, which a stripped
/// service environment can do; callers then keep nothing rather than guess.
pub fn dir() -> Option<PathBuf> {
    dir_from(|name| std::env::var_os(name))
}

fn dir_from(var: impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    let set = |name: &str| var(name).filter(|value| !value.is_empty());
    if let Some(dir) = set(DIR_ENV) {
        return Some(PathBuf::from(dir));
    }
    if cfg!(target_os = "macos") {
        set("HOME").map(|home| PathBuf::from(home).join("Library/Application Support/Orrin"))
    } else if cfg!(windows) {
        set("APPDATA").map(|dir| PathBuf::from(dir).join("Orrin"))
    } else {
        set("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| set("HOME").map(|home| PathBuf::from(home).join(".config")))
            .map(|dir| dir.join("orrin"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| OsString::from(value))
        }
    }

    #[test]
    fn the_override_wins_over_the_platforms_own_directory() {
        let vars = [
            (DIR_ENV, "/tmp/elsewhere"),
            ("HOME", "/home/me"),
            ("APPDATA", "C:\\Users\\me\\AppData\\Roaming"),
        ];
        assert_eq!(dir_from(env(&vars)), Some(PathBuf::from("/tmp/elsewhere")));
    }
}
