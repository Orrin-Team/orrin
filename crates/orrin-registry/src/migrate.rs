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
const MIGRATIONS: [Migration; FORMAT_VERSION as usize - 1] = [v1_to_v2, v2_to_v3];

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

/// Version 3 names an engine-built mesh or material by the id derived from
/// its name ([`AssetId::named`]) where version 2 wrote the name itself, so that
/// every asset reference in a file is an `asset(…)` whatever made the asset.
///
/// Rewritten in two places: an `orrin.mesh` or `orrin.material` component, and
/// the `value` of a prefab instance's override of one. A string that does not
/// unescape is left as it is, for the parser to report on its own line.
///
/// [`AssetId::named`]: crate::AssetId::named
fn v2_to_v3(text: &str) -> String {
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    let mut seen_header = false;
    let mut in_prefab = false;
    // The kind of asset the override being read replaces, once its
    // `component` field has said; fields are written sorted, so before `value`.
    let mut overriding: Option<&str> = None;

    for line in &mut lines {
        if !seen_header {
            if !line.trim().is_empty() {
                *line = "orrin-scene 3 min-reader 3".to_owned();
                seen_header = true;
            }
            continue;
        }
        let indent = line.len() - line.trim_start_matches(' ').len();
        let Some((field, value)) = line[indent..].split_once(" = ") else {
            if indent <= 2 {
                in_prefab = indent == 2 && line[indent..].trim_end() == "orrin.prefab";
            }
            if indent < 8 {
                overriding = None;
            }
            continue;
        };
        let rewritten = match indent {
            2 => {
                in_prefab = false;
                overriding = None;
                asset_kind(field).and_then(|kind| named_asset(kind, value))
            }
            8 if in_prefab && field == "component" => {
                overriding = unquote(value).as_deref().and_then(asset_kind);
                None
            }
            8 if in_prefab && field == "value" => {
                overriding.and_then(|kind| named_asset(kind, value))
            }
            _ => None,
        };
        if let Some(rewritten) = rewritten {
            *line = format!("{}{field} = {rewritten}", &line[..indent]);
        }
    }

    let mut out = lines.join("\n");
    out.push('\n');
    out
}

fn asset_kind(component: &str) -> Option<&'static str> {
    match component {
        "orrin.mesh" => Some("mesh"),
        "orrin.material" => Some("material"),
        _ => None,
    }
}

fn unquote(value: &str) -> Option<String> {
    let body = value.trim_end().strip_prefix('"')?.strip_suffix('"')?;
    crate::scene::parse_string(body, 0).ok()
}

fn named_asset(kind: &str, value: &str) -> Option<String> {
    let name = unquote(value)?;
    Some(format!("asset({})", crate::AssetId::named(kind, &name)))
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
    fn v2_to_v3_names_engine_assets_by_their_derived_ids() {
        let cube = crate::AssetId::named("mesh", "cube");
        let gold = crate::AssetId::named("material", "go\"ld");
        let v2 = "orrin-scene 2 min-reader 2

entity 00000000-0000-0000-0000-000000000001 \"Cube\"
  orrin.material = \"go\\\"ld\"
  orrin.mesh = \"cube\"
  game.sign
    orrin.mesh = \"a field, not the component\"
  orrin.tag = \"cube\"

entity 00000000-0000-0000-0000-000000000002
  orrin.mesh = asset(00000000-0000-4000-8000-000000000009)
  orrin.prefab
    overrides
      [0]
        component = \"orrin.mesh\"
        entity = entity(00000000-0000-0000-0000-000000000003)
        path = \"\"
        value = \"cube\"
      [1]
        component = \"orrin.tag\"
        entity = entity(00000000-0000-0000-0000-000000000003)
        path = \"\"
        value = \"cube\"
    source = asset(00000000-0000-4000-8000-000000000004)
  orrin.material = \"bad \\q escape\"
";
        assert_eq!(
            v2_to_v3(v2),
            format!(
                "orrin-scene 3 min-reader 3

entity 00000000-0000-0000-0000-000000000001 \"Cube\"
  orrin.material = asset({gold})
  orrin.mesh = asset({cube})
  game.sign
    orrin.mesh = \"a field, not the component\"
  orrin.tag = \"cube\"

entity 00000000-0000-0000-0000-000000000002
  orrin.mesh = asset(00000000-0000-4000-8000-000000000009)
  orrin.prefab
    overrides
      [0]
        component = \"orrin.mesh\"
        entity = entity(00000000-0000-0000-0000-000000000003)
        path = \"\"
        value = asset({cube})
      [1]
        component = \"orrin.tag\"
        entity = entity(00000000-0000-0000-0000-000000000003)
        path = \"\"
        value = \"cube\"
    source = asset(00000000-0000-4000-8000-000000000004)
  orrin.material = \"bad \\q escape\"
"
            )
        );
    }

    /// A version 1 file runs the whole chain, and comes out parsing with its
    /// mesh as an id.
    #[test]
    fn a_version_1_scene_reaches_the_current_version_with_ids() {
        let v1 = "orrin-scene 1

entity 00000000-0000-0000-0000-000000000001
  orrin.mesh = \"cube\"
  orrin.name = \"A\"
";
        let document = crate::parse(v1).unwrap();
        let mesh = document.entities[0]
            .components
            .iter()
            .find(|(id, _)| id.as_str() == "orrin.mesh")
            .map(|(_, value)| value.clone());
        assert_eq!(
            mesh,
            Some(crate::Value::Asset(crate::AssetId::named("mesh", "cube")))
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
