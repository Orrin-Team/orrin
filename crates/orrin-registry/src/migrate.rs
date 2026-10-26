//! Forward-only migrations between versions of the scene format.
//!
//! Each is a pure function from the text of one version to the text of the
//! next, and each is kept for as long as the engine exists: a file written by
//! any past build stays readable by every later one by running the chain from
//! its version up. Nothing migrates backwards; a newer file reaches an older
//! build only when its header's minimum-reader version allows it.
//!
//! Text to text, not document to document, because a [`SceneDocument`] is the
//! current shape — an old version's grammar need not parse into it at all.
//! Every migration also keeps the line count, so a parse error in a migrated
//! file still names the line its author sees.
//!
//! The headers a migration writes are literals rather than the version
//! constants on purpose: a migration describes one fixed step, and must not
//! drift when the constants next move.
//!
//! [`SceneDocument`]: crate::SceneDocument

use crate::scene::FORMAT_VERSION;

type Migration = fn(&str) -> String;

/// `MIGRATIONS[n]` takes version `n + 1` to `n + 2`.
const MIGRATIONS: [Migration; FORMAT_VERSION as usize - 1] = [v1_to_v2];

/// Bring `text`, written at `version`, up to [`FORMAT_VERSION`]. `version` is
/// at least 1 and below the current one; the header parser has checked both.
pub(crate) fn to_current(text: &str, version: u32) -> String {
    let mut text = text.to_owned();
    for migrate in &MIGRATIONS[version as usize - 1..] {
        text = migrate(&text);
    }
    text
}

/// Version 2 moved an entity's name from its `orrin.name` component onto its
/// `entity` line, and added the minimum-reader version to the header.
///
/// The name's quoted text is carried over as written; the version 2 parser is
/// what checks it.
fn v1_to_v2(text: &str) -> String {
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    let mut seen_header = false;
    let mut unnamed: Option<usize> = None;

    for i in 0..lines.len() {
        if !seen_header {
            if !lines[i].trim().is_empty() {
                lines[i] = "orrin-scene 2 min-reader 2".to_owned();
                seen_header = true;
            }
            continue;
        }
        if lines[i].starts_with("entity ") {
            unnamed = Some(i);
            continue;
        }
        // Exactly two spaces: a component, not a deeper field of the same name.
        if let (Some(entity), Some(name)) = (unnamed, lines[i].strip_prefix("  orrin.name = \"")) {
            let name = format!(" \"{}", name.trim_end());
            lines[entity].push_str(&name);
            lines[i].clear();
            unnamed = None;
        }
    }

    let mut out = lines.join("\n");
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_to_v2_moves_names_onto_entity_lines() {
        let v1 = "\
orrin-scene 1

entity 00000000-0000-0000-0000-000000000001
  orrin.name = \"Door\"
  orrin.tag = 7

entity 00000000-0000-0000-0000-000000000002
  game.sign
    orrin.name = \"not a component\"
  orrin.name = 3
";
        assert_eq!(
            v1_to_v2(v1),
            "\
orrin-scene 2 min-reader 2

entity 00000000-0000-0000-0000-000000000001 \"Door\"

  orrin.tag = 7

entity 00000000-0000-0000-0000-000000000002
  game.sign
    orrin.name = \"not a component\"
  orrin.name = 3
"
        );
    }

    #[test]
    fn crlf_input_comes_out_lf() {
        let v1 = "orrin-scene 1\r\n\r\nentity 00000000-0000-0000-0000-000000000001\r\n  orrin.name = \"A\"\r\n";
        assert_eq!(
            v1_to_v2(v1),
            "orrin-scene 2 min-reader 2\n\nentity 00000000-0000-0000-0000-000000000001 \"A\"\n\n"
        );
    }
}
