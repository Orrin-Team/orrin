//! The parsed form of a scene file, and the parser that produces it.
//!
//! Deliberately knows nothing about `World`: a document is plain data, so
//! parsing is testable without an ECS and the rules for turning one into live
//! entities live with the engine instead of with the format.

use std::fmt;
use std::str::FromStr;

use crate::registry::ComponentId;
use crate::value::Value;
use crate::{AssetId, EntityId};

/// Bumped whenever the grammar changes in a way older files don't satisfy.
/// Migrations are explicit functions kept in the repo; a file whose version
/// isn't handled is refused by name rather than parsed optimistically.
pub const FORMAT_VERSION: u32 = 3;

/// The oldest reader that can read what this build writes, stated in every
/// header next to the version. A change an older reader can tolerate bumps
/// [`FORMAT_VERSION`] and leaves this alone, so that reader is not locked out.
pub const MIN_READER_VERSION: u32 = 3;

/// The component written on an entity's header line rather than among its
/// components, when it holds a string.
///
/// The header is the line git shows as a hunk's context, so carrying the name
/// there is what makes a diff say which entity changed instead of only which
/// UUID did.
pub const NAME_COMPONENT: ComponentId = ComponentId::new("orrin.name");

/// One scene: entities and their components, in the order they were written.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneDocument {
    pub entities: Vec<SceneEntity>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SceneEntity {
    pub id: EntityId,
    pub components: Vec<(ComponentId, Value)>,
}

/// A syntax error, carrying the line it was found on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl ParseError {
    fn new(line: usize, message: impl Into<String>) -> Self {
        Self {
            line,
            message: message.into(),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

/// Read a scene file.
///
/// The grammar is the one the writer emits: a version header, then one block
/// per entity, two spaces of indent per level, `name = value` for scalars and
/// an indented block for anything structured. A file from an older version is
/// migrated to this one first (see `migrate`).
pub fn parse(input: &str) -> Result<SceneDocument, ParseError> {
    let lines = collect_lines(input)?;
    let header = lines
        .first()
        .ok_or_else(|| ParseError::new(1, "empty file; expected an `orrin-scene` header"))?;
    let version = parse_header(header)?;
    if version < FORMAT_VERSION {
        return parse(&crate::migrate::to_current(input, version));
    }

    let mut cursor = Cursor { lines, index: 1 };
    let mut entities = Vec::new();
    while let Some(line) = cursor.peek() {
        if line.indent != 0 {
            return Err(ParseError::new(
                line.number,
                "unexpected indent; entities start at column 0",
            ));
        }
        let line = cursor.next().expect("peeked");
        let (id, name) = parse_entity_header(&line)?;

        let named = name.is_some();
        let mut components = Vec::new();
        if let Some(name) = name {
            components.push((NAME_COMPONENT, Value::String(name)));
        }
        while let Some((number, component, value)) = next_entry(&mut cursor, 1)? {
            if named && component == NAME_COMPONENT.as_str() {
                return Err(ParseError::new(
                    number,
                    "this entity is already named on its `entity` line",
                ));
            }
            components.push((ComponentId::owned(component), value));
        }
        entities.push(SceneEntity { id, components });
    }

    Ok(SceneDocument { entities })
}

/// Read a document that holds one value rather than entities — a material —
/// under a header naming its own `kind`: `<kind> <version> min-reader
/// <version>`, then the value's fields at the left margin.
///
/// The scene grammar, parser and rules, so a material reads and diffs the way
/// a scene does. `version` is the newest this build reads; a file needing a
/// newer reader is refused by name.
pub fn parse_value_document(
    input: &str,
    kind: &str,
    version: u32,
) -> Result<Vec<(String, Value)>, ParseError> {
    let lines = collect_lines(input)?;
    let header = lines
        .first()
        .ok_or_else(|| ParseError::new(1, format!("empty file; expected a `{kind}` header")))?;
    let format = Format {
        kind,
        label: kind,
        version,
        min_reader: version,
    };
    parse_header_of(header, &format, false)?;
    let mut cursor = Cursor { lines, index: 1 };
    let fields = parse_block(&mut cursor, 0)?;
    if let Some(line) = cursor.peek() {
        return Err(ParseError::new(
            line.number,
            "unexpected indent; a field at the left margin was expected",
        ));
    }
    Ok(fields)
}

/// What a header names and what this build reads of it.
struct Format<'a> {
    kind: &'a str,
    /// How an error names the format.
    label: &'a str,
    version: u32,
    min_reader: u32,
}

/// `orrin-scene <version> min-reader <version>`, returning the first.
///
/// Version 1 predates the second number and is the one header allowed without
/// it.
fn parse_header(line: &Line) -> Result<u32, ParseError> {
    let format = Format {
        kind: "orrin-scene",
        label: "scene format",
        version: FORMAT_VERSION,
        min_reader: MIN_READER_VERSION,
    };
    parse_header_of(line, &format, true)
}

/// `<kind> <version> min-reader <version>` for `format`, returning the first.
/// `bare_v1` allows the scene's first version, which predates the second number.
fn parse_header_of(line: &Line, format: &Format, bare_v1: bool) -> Result<u32, ParseError> {
    let error = |message: String| ParseError::new(line.number, message);
    let Format {
        kind,
        label,
        version: current,
        min_reader: current_min,
    } = *format;
    let expected = format!("{kind} {current} min-reader {current_min}");

    let Some(rest) = line
        .text
        .strip_prefix(kind)
        .and_then(|rest| rest.strip_prefix(' '))
    else {
        return Err(error(format!("expected an `{expected}` header")));
    };
    let mut words = rest.split_whitespace();
    let number = |word: Option<&str>| -> Result<u32, ParseError> {
        let word = word.unwrap_or_default();
        word.parse()
            .map_err(|_| error(format!("`{word}` is not a version number")))
    };

    let version = number(words.next())?;
    if version == 0 {
        return Err(error(format!("there is no {label} version 0")));
    }
    if bare_v1 && version == 1 && words.clone().next().is_none() {
        return Ok(1);
    }
    if words.next() != Some("min-reader") {
        return Err(error(format!(
            "expected `orrin-scene {version} min-reader <version>`: a header names the \
             oldest reader that can read the file"
        )));
    }
    let min_reader = number(words.next())?;
    if let Some(extra) = words.next() {
        return Err(error(format!("unexpected `{extra}` after the header")));
    }

    if min_reader > version {
        return Err(error(format!(
            "a file cannot need a reader newer than its own version ({version} < {min_reader})"
        )));
    }
    if min_reader > current {
        return Err(error(format!(
            "{label} version {version} needs a reader of version {min_reader} or later \
             (this build reads up to {current})"
        )));
    }
    Ok(version)
}

/// `entity <id>`, optionally followed by the entity's quoted name.
fn parse_entity_header(line: &Line) -> Result<(EntityId, Option<String>), ParseError> {
    let Some(rest) = line.text.strip_prefix("entity ") else {
        return Err(ParseError::new(
            line.number,
            format!("expected `entity <id>`, found `{}`", line.text),
        ));
    };
    let (id, name) = match rest.trim().split_once(' ') {
        Some((id, name)) => (id, Some(name.trim())),
        None => (rest.trim(), None),
    };

    let id = EntityId::from_str(id).map_err(|_| {
        // The debug dump writes `entity #3` for an entity with no persistent
        // id. That is a view of live state, not a scene: re-reading it would
        // invent identities that collide with the next session's.
        let hint = if id.starts_with('#') {
            " (this looks like a debug dump, which has no stable ids)"
        } else {
            ""
        };
        ParseError::new(line.number, format!("`{id}` is not an entity id{hint}"))
    })?;

    let Some(name) = name else {
        return Ok((id, None));
    };
    let body = name
        .strip_prefix('"')
        .and_then(|name| name.strip_suffix('"'))
        .ok_or_else(|| {
            ParseError::new(
                line.number,
                format!("an entity's name is quoted: `entity {id} \"{name}\"`"),
            )
        })?;
    Ok((id, Some(parse_string(body, line.number)?)))
}

/// Read every line indented exactly `indent` levels, plus their children.
fn parse_block(cursor: &mut Cursor, indent: usize) -> Result<Vec<(String, Value)>, ParseError> {
    let mut entries = Vec::new();
    while let Some((_, name, value)) = next_entry(cursor, indent)? {
        entries.push((name, value));
    }
    Ok(entries)
}

/// The next line at `indent` with its children, and the line it is on; `None`
/// once the block has ended.
fn next_entry(
    cursor: &mut Cursor,
    indent: usize,
) -> Result<Option<(usize, String, Value)>, ParseError> {
    let Some(line) = cursor.peek() else {
        return Ok(None);
    };
    if line.indent < indent {
        return Ok(None);
    }
    if line.indent > indent {
        return Err(ParseError::new(
            line.number,
            "unexpected indent; this line is deeper than its parent allows",
        ));
    }
    let line = cursor.next().expect("peeked");
    let (name, value) = parse_named(&line, cursor, indent)?;
    Ok(Some((line.number, name, value)))
}

fn parse_named(
    line: &Line,
    cursor: &mut Cursor,
    indent: usize,
) -> Result<(String, Value), ParseError> {
    let (name, rhs) = match line.text.split_once(" = ") {
        Some((name, rhs)) => (name.trim(), Some(rhs.trim())),
        None => (line.text.trim(), None),
    };
    if name.is_empty() {
        return Err(ParseError::new(line.number, "missing a field name"));
    }

    let children = if cursor.peek().is_some_and(|next| next.indent == indent + 1) {
        parse_block(cursor, indent + 1)?
    } else {
        Vec::new()
    };

    let value = match rhs {
        // `name = Variant` plus an optional indented payload.
        Some(text) if is_variant_name(text) => Value::Enum {
            variant: text.to_owned(),
            fields: children,
        },
        Some(text) => {
            if !children.is_empty() {
                return Err(ParseError::new(
                    line.number,
                    "a scalar cannot have indented fields",
                ));
            }
            parse_scalar(text, line.number)?
        }
        None => group(children, line.number)?,
    };

    Ok((name.to_owned(), value))
}

/// A block's children are a list when every key is `[n]` counting from zero,
/// and a struct otherwise. Keeping the two syntactically distinct is what lets
/// an empty block stay an empty struct rather than an ambiguous nothing.
fn group(children: Vec<(String, Value)>, line: usize) -> Result<Value, ParseError> {
    let indices: Option<Vec<usize>> = children
        .iter()
        .map(|(name, _)| {
            name.strip_prefix('[')
                .and_then(|rest| rest.strip_suffix(']'))
                .and_then(|n| n.parse::<usize>().ok())
        })
        .collect();

    match indices {
        Some(indices) if !children.is_empty() => {
            if indices
                .iter()
                .enumerate()
                .any(|(expected, &got)| expected != got)
            {
                return Err(ParseError::new(line, "list indices must count up from [0]"));
            }
            Ok(Value::List(children.into_iter().map(|(_, v)| v).collect()))
        }
        _ => Ok(Value::Struct(children)),
    }
}

fn is_variant_name(text: &str) -> bool {
    if matches!(text, "true" | "false" | "nan" | "inf" | "-inf") {
        return false;
    }
    let mut chars = text.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn parse_scalar(text: &str, line: usize) -> Result<Value, ParseError> {
    match text {
        "true" => return Ok(Value::Bool(true)),
        "false" => return Ok(Value::Bool(false)),
        "[]" => return Ok(Value::List(Vec::new())),
        _ => {}
    }

    if let Some(rest) = text.strip_prefix('"') {
        let body = rest
            .strip_suffix('"')
            .ok_or_else(|| ParseError::new(line, "unterminated string"))?;
        return parse_string(body, line).map(Value::String);
    }

    if let Some(rest) = text.strip_prefix("entity(") {
        let body = rest
            .strip_suffix(')')
            .ok_or_else(|| ParseError::new(line, "unterminated entity reference"))?;
        return EntityId::from_str(body)
            .map(Value::Entity)
            .map_err(|_| ParseError::new(line, format!("`{body}` is not an entity id")));
    }

    if let Some(rest) = text.strip_prefix("asset(") {
        let body = rest
            .strip_suffix(')')
            .ok_or_else(|| ParseError::new(line, "unterminated asset reference"))?;
        return AssetId::from_str(body)
            .map(Value::Asset)
            .map_err(|_| ParseError::new(line, format!("`{body}` is not an asset id")));
    }

    if let Some(rest) = text.strip_prefix('(') {
        let body = rest
            .strip_suffix(')')
            .ok_or_else(|| ParseError::new(line, "unterminated vector"))?;
        let parts: Result<Vec<f32>, ParseError> =
            body.split(',').map(|p| parse_f32(p.trim(), line)).collect();
        let parts = parts?;
        return match parts.len() {
            3 => Ok(Value::Vec3(glam::Vec3::new(parts[0], parts[1], parts[2]))),
            4 => Ok(Value::Quat(glam::Quat::from_xyzw(
                parts[0], parts[1], parts[2], parts[3],
            ))),
            n => Err(ParseError::new(
                line,
                format!("a vector has 3 components and a quaternion 4, found {n}"),
            )),
        };
    }

    parse_number(text, line)
}

/// Integers carry no width in the file — `3` could be an `i32` or a `u32` — so
/// the narrowest variant that holds the value is produced and the reader
/// coerces from there. See the numeric `Reflect` impls.
fn parse_number(text: &str, line: usize) -> Result<Value, ParseError> {
    if text.contains('.') || text.contains('e') || text.contains('E') || is_non_finite(text) {
        return parse_f32(text, line).map(Value::F32);
    }
    let n: i64 = text
        .parse()
        .map_err(|_| ParseError::new(line, format!("`{text}` is not a value")))?;
    if let Ok(n) = i32::try_from(n) {
        Ok(Value::I32(n))
    } else if let Ok(n) = u32::try_from(n) {
        Ok(Value::U32(n))
    } else {
        Err(ParseError::new(
            line,
            format!("`{text}` does not fit in 32 bits"),
        ))
    }
}

fn is_non_finite(text: &str) -> bool {
    matches!(text, "nan" | "inf" | "-inf")
}

fn parse_f32(text: &str, line: usize) -> Result<f32, ParseError> {
    match text {
        "nan" => Ok(f32::NAN),
        "inf" => Ok(f32::INFINITY),
        "-inf" => Ok(f32::NEG_INFINITY),
        _ => text
            .parse()
            .map_err(|_| ParseError::new(line, format!("`{text}` is not a number"))),
    }
}

pub(crate) fn parse_string(body: &str, line: usize) -> Result<String, ParseError> {
    let mut out = String::with_capacity(body.len());
    let mut chars = body.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some(other) => {
                return Err(ParseError::new(line, format!("unknown escape `\\{other}`")));
            }
            None => return Err(ParseError::new(line, "string ends with a dangling `\\`")),
        }
    }
    Ok(out)
}

struct Line<'a> {
    number: usize,
    indent: usize,
    text: &'a str,
}

struct Cursor<'a> {
    lines: Vec<Line<'a>>,
    index: usize,
}

impl<'a> Cursor<'a> {
    fn peek(&self) -> Option<&Line<'a>> {
        self.lines.get(self.index)
    }

    fn next(&mut self) -> Option<Line<'a>> {
        if self.index >= self.lines.len() {
            return None;
        }
        self.index += 1;
        let line = &self.lines[self.index - 1];
        Some(Line {
            number: line.number,
            indent: line.indent,
            text: line.text,
        })
    }
}

/// Blank lines are dropped; everything else must be indented in whole steps of
/// two spaces. Tabs are rejected rather than guessed at, since their width is
/// a display setting and the format's determinism cannot depend on one.
fn collect_lines(input: &str) -> Result<Vec<Line<'_>>, ParseError> {
    let mut lines = Vec::new();
    for (i, raw) in input.lines().enumerate() {
        let number = i + 1;
        if raw.trim().is_empty() {
            continue;
        }
        if raw.starts_with('\t') || raw.trim_start_matches(' ').starts_with('\t') {
            return Err(ParseError::new(number, "tabs are not valid indentation"));
        }
        let spaces = raw.len() - raw.trim_start_matches(' ').len();
        if spaces % 2 != 0 {
            return Err(ParseError::new(
                number,
                format!("indentation must be a multiple of two spaces, found {spaces}"),
            ));
        }
        lines.push(Line {
            number,
            indent: spaces / 2,
            text: raw.trim(),
        });
    }
    Ok(lines)
}
