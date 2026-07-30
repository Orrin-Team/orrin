//! The parsed form of a scene file, and the parser that produces it.
//!
//! Deliberately knows nothing about `World`: a document is plain data, so
//! parsing is testable without an ECS and the rules for turning one into live
//! entities live with the engine instead of with the format.

use std::fmt;
use std::str::FromStr;

use crate::registry::ComponentId;
use crate::value::Value;
use crate::EntityId;

/// Bumped whenever the grammar changes in a way older files don't satisfy.
/// Migrations are explicit functions kept in the repo; a file whose version
/// isn't handled is refused by name rather than parsed optimistically.
pub const FORMAT_VERSION: u32 = 1;

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
/// an indented block for anything structured.
pub fn parse(input: &str) -> Result<SceneDocument, ParseError> {
    let lines = collect_lines(input)?;
    let mut cursor = Cursor { lines, index: 0 };

    let header = cursor
        .next()
        .ok_or_else(|| ParseError::new(1, "empty file; expected an `orrin-scene` header"))?;
    parse_header(&header)?;

    let mut entities = Vec::new();
    while let Some(line) = cursor.peek() {
        if line.indent != 0 {
            return Err(ParseError::new(
                line.number,
                "unexpected indent; entities start at column 0",
            ));
        }
        let line = cursor.next().expect("peeked");
        let id = parse_entity_header(&line)?;
        let components = parse_block(&mut cursor, 1)?
            .into_iter()
            .map(|(name, value)| (ComponentId::owned(name), value))
            .collect();
        entities.push(SceneEntity { id, components });
    }

    Ok(SceneDocument { entities })
}

fn parse_header(line: &Line) -> Result<(), ParseError> {
    let Some(version) = line.text.strip_prefix("orrin-scene ") else {
        return Err(ParseError::new(
            line.number,
            format!("expected an `orrin-scene {FORMAT_VERSION}` header"),
        ));
    };
    let version: u32 = version.trim().parse().map_err(|_| {
        ParseError::new(line.number, format!("`{version}` is not a version number"))
    })?;
    if version != FORMAT_VERSION {
        return Err(ParseError::new(
            line.number,
            format!("scene format version {version} is not supported (this build reads {FORMAT_VERSION})"),
        ));
    }
    Ok(())
}

fn parse_entity_header(line: &Line) -> Result<EntityId, ParseError> {
    let Some(id) = line.text.strip_prefix("entity ") else {
        return Err(ParseError::new(
            line.number,
            format!("expected `entity <id>`, found `{}`", line.text),
        ));
    };
    let id = id.trim();
    EntityId::from_str(id).map_err(|_| {
        // The debug dump writes `entity #3` for an entity with no persistent
        // id. That is a view of live state, not a scene: re-reading it would
        // invent identities that collide with the next session's.
        let hint = if id.starts_with('#') {
            " (this looks like a debug dump, which has no stable ids)"
        } else {
            ""
        };
        ParseError::new(line.number, format!("`{id}` is not an entity id{hint}"))
    })
}

/// Read every line indented exactly `indent` levels, plus their children.
fn parse_block(cursor: &mut Cursor, indent: usize) -> Result<Vec<(String, Value)>, ParseError> {
    let mut entries = Vec::new();
    while let Some(line) = cursor.peek() {
        if line.indent < indent {
            break;
        }
        if line.indent > indent {
            return Err(ParseError::new(
                line.number,
                "unexpected indent; this line is deeper than its parent allows",
            ));
        }
        let line = cursor.next().expect("peeked");
        entries.push(parse_named(&line, cursor, indent)?);
    }
    Ok(entries)
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
            if indices.iter().enumerate().any(|(expected, &got)| expected != got) {
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

fn parse_string(body: &str, line: usize) -> Result<String, ParseError> {
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
                return Err(ParseError::new(
                    line,
                    format!("unknown escape `\\{other}`"),
                ));
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
