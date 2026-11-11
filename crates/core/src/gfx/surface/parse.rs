use std::fmt;

use super::params::{ColorSpace, DefaultTexture, Param, ParamKind, ParamTable, Slot, pack};

/// A parsed `.oshader`: its parameters, and its GLSL as the compiler is to see
/// it.
#[derive(Clone, PartialEq, Debug)]
pub struct SurfaceShader {
    params: ParamTable,
    /// The 1-based line each parameter was declared on, beside `params`.
    lines: Vec<u32>,
    /// The source with comments and parameter lines blanked, line for line.
    body: String,
    /// Every macro the source `#define`s, so the stitched file can undefine
    /// them before the engine's code resumes.
    defines: Vec<String>,
}

/// A problem with a surface shader, at a 1-based line of its source when there
/// is one the user can see.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SurfaceError {
    pub line: Option<u32>,
    pub message: String,
}

impl SurfaceError {
    pub(super) fn at(line: u32, message: impl Into<String>) -> Self {
        SurfaceError {
            line: Some(line),
            message: message.into(),
        }
    }

    /// `<file>:<line>: <message>`, or `<file>: <message>` without a line.
    pub fn located(&self, file: &str) -> String {
        match self.line {
            Some(line) => format!("{file}:{line}: {}", self.message),
            None => format!("{file}: {}", self.message),
        }
    }
}

impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.line {
            Some(line) => write!(f, "line {line}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl SurfaceShader {
    /// Parse a `.oshader`'s text. Every error is returned, each at its line.
    pub fn parse(text: &str) -> Result<SurfaceShader, Vec<SurfaceError>> {
        let mut errors = Vec::new();
        let (stripped, unclosed) = strip_comments(text);
        if let Some(line) = unclosed {
            errors.push(SurfaceError::at(line, "a `/*` comment is never closed"));
        }

        let mut params: Vec<Param> = Vec::new();
        let mut lines = Vec::new();
        let mut defines = Vec::new();
        let mut body = String::with_capacity(stripped.len());
        let mut depth = Depth::default();
        // The last two tokens at file scope, to spot `void surface (`.
        let mut last: [Option<Token>; 2] = [None, None];
        let mut has_surface = false;

        for (index, line) in stripped.lines().enumerate() {
            let number = index as u32 + 1;
            if let Some(directive) = line.trim_start().strip_prefix('#') {
                let directive = directive.trim_start();
                let name = directive
                    .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or("");
                if let Some(reason) = forbidden_directive(name) {
                    errors.push(SurfaceError::at(number, reason));
                } else {
                    let rest = tokens(&directive[name.len()..]);
                    if let ("define", Some(Token::Ident(defined))) = (name, rest.first()) {
                        defines.push(defined.to_string());
                    }
                    for token in rest {
                        if let Some(reason) = forbidden_token(token) {
                            errors.push(SurfaceError::at(number, reason));
                        }
                    }
                }
                body.push_str(line);
                body.push('\n');
                continue;
            }

            let tokens = tokens(line);
            if depth.file_scope() && tokens.first() == Some(&Token::Ident("param")) {
                match declaration(&tokens[1..]) {
                    Ok((name, kind)) => {
                        if let Some(earlier) = params.iter().position(|p| p.name == name) {
                            errors.push(SurfaceError::at(
                                number,
                                format!("`{name}` is already declared on line {}", lines[earlier]),
                            ));
                        } else {
                            params.push(Param {
                                name: name.to_string(),
                                kind,
                                slot: Slot {
                                    word: 0,
                                    component: 0,
                                },
                            });
                            lines.push(number);
                        }
                    }
                    Err(message) => errors.push(SurfaceError::at(number, message)),
                }
                body.push('\n');
                continue;
            }

            for &token in &tokens {
                if let Some(reason) = forbidden_token(token) {
                    errors.push(SurfaceError::at(number, reason));
                }
                if depth.file_scope() {
                    if token == Token::Punct('(')
                        && last == [Some(Token::Ident("void")), Some(Token::Ident("surface"))]
                    {
                        has_surface = true;
                    }
                    last = [last[1], Some(token)];
                }
                depth.step(token);
            }
            body.push_str(line);
            body.push('\n');
        }

        let last_line = text.lines().count().max(1) as u32;
        if depth.braces > 0 {
            errors.push(SurfaceError::at(
                last_line,
                "expected `}`: the file ends with a `{` still open",
            ));
        }
        if !has_surface {
            errors.push(SurfaceError::at(
                last_line,
                "expected a `void surface(in SurfaceIn i, inout SurfaceOut o)` function",
            ));
        }
        if !errors.is_empty() {
            return Err(errors);
        }
        let word_count = pack(&mut params);
        Ok(SurfaceShader {
            params: ParamTable { params, word_count },
            lines,
            body,
            defines,
        })
    }

    pub fn params(&self) -> &ParamTable {
        &self.params
    }

    #[cfg_attr(
        not(feature = "shader-compiler"),
        allow(dead_code, reason = "read by the compile step")
    )]
    pub(super) fn declared_on(&self) -> &[u32] {
        &self.lines
    }

    #[cfg_attr(
        not(feature = "shader-compiler"),
        allow(dead_code, reason = "read by the compile step")
    )]
    pub(super) fn body(&self) -> &str {
        &self.body
    }

    #[cfg_attr(
        not(feature = "shader-compiler"),
        allow(dead_code, reason = "read by the compile step")
    )]
    pub(super) fn defines(&self) -> &[String] {
        &self.defines
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Token<'a> {
    Ident(&'a str),
    Number(&'a str),
    Punct(char),
}

/// GLSL's tokens, as far as anything here needs them: identifiers, numbers,
/// and every other character on its own.
pub(super) fn tokens(line: &str) -> Vec<Token<'_>> {
    let bytes = line.as_bytes();
    let mut out = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let c = bytes[at];
        let start = at;
        if c.is_ascii_whitespace() {
            at += 1;
        } else if c.is_ascii_alphabetic() || c == b'_' {
            while at < bytes.len() && (bytes[at].is_ascii_alphanumeric() || bytes[at] == b'_') {
                at += 1;
            }
            out.push(Token::Ident(&line[start..at]));
        } else if c.is_ascii_digit()
            || (c == b'.' && bytes.get(at + 1).is_some_and(u8::is_ascii_digit))
        {
            let hex = line[at..].starts_with("0x") || line[at..].starts_with("0X");
            at += 1;
            while at < bytes.len() {
                let b = bytes[at];
                let exponent_sign =
                    (b == b'+' || b == b'-') && !hex && matches!(bytes[at - 1], b'e' | b'E');
                if b.is_ascii_alphanumeric() || b == b'_' || b == b'.' || exponent_sign {
                    at += 1;
                } else {
                    break;
                }
            }
            out.push(Token::Number(&line[start..at]));
        } else {
            let c = line[at..].chars().next().expect("at a char boundary");
            at += c.len_utf8();
            out.push(Token::Punct(c));
        }
    }
    out
}

/// `text` with every comment replaced by spaces and its newlines kept, so line
/// numbers survive; and the line of a `/*` that never closes.
pub(super) fn strip_comments(text: &str) -> (String, Option<u32>) {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    let mut line = 1;
    while let Some(c) = chars.next() {
        match (c, chars.peek()) {
            ('/', Some('/')) => {
                while chars.peek().is_some_and(|&c| c != '\n') {
                    chars.next();
                }
            }
            ('/', Some('*')) => {
                let opened = line;
                chars.next();
                out.push_str("  ");
                let mut closed = false;
                while let Some(c) = chars.next() {
                    if c == '*' && chars.peek() == Some(&'/') {
                        chars.next();
                        out.push_str("  ");
                        closed = true;
                        break;
                    }
                    if c == '\n' {
                        line += 1;
                        out.push('\n');
                    } else {
                        out.push(' ');
                    }
                }
                if !closed {
                    return (out, Some(opened));
                }
            }
            _ => {
                if c == '\n' {
                    line += 1;
                }
                out.push(c);
            }
        }
    }
    (out, None)
}

/// Brace and parenthesis nesting, which is what "file scope" means here.
#[derive(Default)]
pub(super) struct Depth {
    braces: u32,
    parens: u32,
}

impl Depth {
    pub(super) fn file_scope(&self) -> bool {
        self.braces == 0 && self.parens == 0
    }

    pub(super) fn step(&mut self, token: Token) {
        match token {
            Token::Punct('{') => self.braces += 1,
            Token::Punct('}') => self.braces = self.braces.saturating_sub(1),
            Token::Punct('(') => self.parens += 1,
            Token::Punct(')') => self.parens = self.parens.saturating_sub(1),
            _ => {}
        }
    }
}

fn forbidden_directive(name: &str) -> Option<String> {
    let why = match name {
        "version" | "extension" | "include" => {
            "the pass shader it is stitched into supplies the version, extensions and includes"
        }
        "line" => "line numbers are the engine's to keep, so that an error finds this file's line",
        _ => return None,
    };
    Some(format!("a surface shader may not use `#{name}`: {why}"))
}

pub(super) fn forbidden_token(token: Token) -> Option<String> {
    let Token::Ident(name) = token else {
        return None;
    };
    Some(match name {
        "gl_FragDepth" => "a surface shader may not write `gl_FragDepth`: the forward pass \
                           depth-tests EQUAL against the prepass's depth, and both keep early-Z"
            .to_string(),
        "discard" => "a surface shader may not `discard`: lower `o.alpha` and give the material \
                      the Masked blend mode, so the prepass and the forward pass cut along one line"
            .to_string(),
        "layout" | "uniform" | "buffer" => format!(
            "a surface shader may not declare `{name}` resources: it binds nothing of its own, \
             and its values arrive through `param` declarations"
        ),
        _ => return None,
    })
}

const TYPES: &str = "float, vec2, vec3, vec4, color or texture";

/// The declaration after `param`: its name and its kind with default and hints.
fn declaration<'a>(tokens: &[Token<'a>]) -> Result<(&'a str, ParamKind), String> {
    let mut cursor = Cursor { tokens, at: 0 };
    let ty = match cursor.next() {
        Some(Token::Ident(ty @ ("float" | "vec2" | "vec3" | "vec4" | "color" | "texture"))) => ty,
        other => {
            return Err(format!(
                "expected a parameter type ({TYPES}) after `param`, found {}",
                describe(other)
            ));
        }
    };
    let name = match cursor.next() {
        Some(Token::Ident(name)) => name,
        other => {
            return Err(format!(
                "expected a parameter name after `{ty}`, found {}",
                describe(other)
            ));
        }
    };
    if let Some(reason) = reserved(name) {
        return Err(reason);
    }

    let mut kind = match ty {
        "float" => ParamKind::Float {
            default: 0.0,
            range: None,
        },
        "vec2" => ParamKind::Vec2 { default: [0.0; 2] },
        "vec3" => ParamKind::Vec3 { default: [0.0; 3] },
        "vec4" => ParamKind::Vec4 { default: [0.0; 4] },
        "color" => ParamKind::Color { default: [1.0; 4] },
        _ => ParamKind::Texture {
            default: DefaultTexture::White,
            color_space: ColorSpace::Srgb,
        },
    };
    if cursor.peek() == Some(Token::Punct('=')) {
        cursor.next();
        kind = default(ty, &mut cursor)?;
    }

    let mut range = None;
    let mut color_space = None;
    if cursor.peek() == Some(Token::Punct(':')) {
        cursor.next();
        loop {
            match cursor.next() {
                Some(Token::Ident("range")) => {
                    cursor.expect('(', "after `range`")?;
                    let min = cursor.number()?;
                    cursor.expect(',', "between the range's minimum and maximum")?;
                    let max = cursor.number()?;
                    cursor.expect(')', "to close the range")?;
                    if ty != "float" {
                        return Err(format!(
                            "range(min, max) applies only to a float, not a {ty}"
                        ));
                    }
                    if range.replace([min, max]).is_some() {
                        return Err("range is given more than once".to_string());
                    }
                }
                Some(Token::Ident(space @ ("srgb" | "linear"))) => {
                    if ty != "texture" {
                        return Err(format!("`{space}` applies only to a texture, not a {ty}"));
                    }
                    let space = if space == "srgb" {
                        ColorSpace::Srgb
                    } else {
                        ColorSpace::Linear
                    };
                    if color_space.replace(space).is_some() {
                        return Err("a colour space is given more than once".to_string());
                    }
                }
                other => {
                    return Err(format!(
                        "expected a hint (range(min, max), srgb or linear), found {}",
                        describe(other)
                    ));
                }
            }
            if cursor.peek() != Some(Token::Punct(',')) {
                break;
            }
            cursor.next();
        }
    }
    cursor.expect(';', "to end the declaration")?;
    if let Some(extra) = cursor.next() {
        return Err(format!(
            "unexpected {} after the `;`: a declaration takes its own line",
            describe(Some(extra))
        ));
    }

    match &mut kind {
        ParamKind::Float {
            default,
            range: slot,
        } => {
            if let Some([min, max]) = range {
                if min >= max {
                    return Err(format!(
                        "the range's minimum must be less than its maximum, not range({min}, {max})"
                    ));
                }
                if *default < min || *default > max {
                    return Err(format!(
                        "the default {default} is outside its range({min}, {max})"
                    ));
                }
            }
            *slot = range;
        }
        ParamKind::Texture {
            default,
            color_space: slot,
        } => {
            *slot = color_space.unwrap_or(if *default == DefaultTexture::Normal {
                ColorSpace::Linear
            } else {
                ColorSpace::Srgb
            });
        }
        _ => {}
    }
    Ok((name, kind))
}

fn default(ty: &str, cursor: &mut Cursor) -> Result<ParamKind, String> {
    if ty == "float" {
        return Ok(ParamKind::Float {
            default: cursor.number()?,
            range: None,
        });
    }
    if ty == "texture" {
        return match cursor.next() {
            Some(Token::Ident("white")) => Ok(DefaultTexture::White),
            Some(Token::Ident("black")) => Ok(DefaultTexture::Black),
            Some(Token::Ident("normal")) => Ok(DefaultTexture::Normal),
            other => Err(format!(
                "expected a default texture (white, black or normal), found {}",
                describe(other)
            )),
        }
        .map(|default| ParamKind::Texture {
            default,
            color_space: ColorSpace::Srgb,
        });
    }

    cursor.expect('(', &format!("to open a {ty}'s default"))?;
    let mut values = vec![cursor.number()?];
    while cursor.peek() == Some(Token::Punct(',')) {
        cursor.next();
        values.push(cursor.number()?);
    }
    cursor.expect(')', &format!("to close a {ty}'s default"))?;
    let count = |found: usize, n: usize, said: &str| {
        if found == n {
            Ok(())
        } else {
            Err(format!(
                "expected {said} numbers in a {ty}'s default, found {found}"
            ))
        }
    };
    Ok(match ty {
        "vec2" => {
            count(values.len(), 2, "2")?;
            ParamKind::Vec2 {
                default: [values[0], values[1]],
            }
        }
        "vec3" => {
            count(values.len(), 3, "3")?;
            ParamKind::Vec3 {
                default: [values[0], values[1], values[2]],
            }
        }
        "vec4" => {
            count(values.len(), 4, "4")?;
            ParamKind::Vec4 {
                default: [values[0], values[1], values[2], values[3]],
            }
        }
        _ => {
            if values.len() == 3 {
                values.push(1.0);
            }
            count(values.len(), 4, "3 or 4")?;
            ParamKind::Color {
                default: [values[0], values[1], values[2], values[3]],
            }
        }
    })
}

struct Cursor<'t, 'a> {
    tokens: &'t [Token<'a>],
    at: usize,
}

impl<'a> Cursor<'_, 'a> {
    fn peek(&self) -> Option<Token<'a>> {
        self.tokens.get(self.at).copied()
    }

    fn next(&mut self) -> Option<Token<'a>> {
        let token = self.peek();
        self.at += 1;
        token
    }

    fn expect(&mut self, punct: char, why: &str) -> Result<(), String> {
        match self.next() {
            Some(Token::Punct(c)) if c == punct => Ok(()),
            other => Err(format!(
                "expected `{punct}` {why}, found {}",
                describe(other)
            )),
        }
    }

    /// An optionally signed, finite number.
    fn number(&mut self) -> Result<f32, String> {
        let negative = match self.peek() {
            Some(Token::Punct(sign @ ('-' | '+'))) => {
                self.next();
                sign == '-'
            }
            _ => false,
        };
        let found = self.next();
        let value = match found {
            Some(Token::Number(text)) => {
                let text = text.strip_suffix(['f', 'F']).unwrap_or(text);
                text.parse::<f32>().ok().filter(|v| v.is_finite())
            }
            _ => None,
        };
        match value {
            Some(value) => Ok(if negative { -value } else { value }),
            None => Err(format!(
                "expected a finite number, found {}",
                describe(found)
            )),
        }
    }
}

fn describe(token: Option<Token>) -> String {
    match token {
        Some(Token::Ident(text) | Token::Number(text)) => format!("`{text}`"),
        Some(Token::Punct(c)) => format!("`{c}`"),
        None => "the end of the line".to_string(),
    }
}

/// GLSL 4.60's keywords and reserved words, less the type names
/// [`is_glsl_type`] recognises by shape.
const KEYWORDS: &[&str] = &[
    "const",
    "uniform",
    "buffer",
    "shared",
    "attribute",
    "varying",
    "coherent",
    "volatile",
    "restrict",
    "readonly",
    "writeonly",
    "atomic_uint",
    "layout",
    "centroid",
    "flat",
    "smooth",
    "noperspective",
    "patch",
    "sample",
    "invariant",
    "precise",
    "break",
    "continue",
    "do",
    "for",
    "while",
    "switch",
    "case",
    "default",
    "if",
    "else",
    "subroutine",
    "in",
    "out",
    "inout",
    "int",
    "void",
    "bool",
    "true",
    "false",
    "float",
    "double",
    "discard",
    "return",
    "uint",
    "struct",
    "lowp",
    "mediump",
    "highp",
    "precision",
    "common",
    "partition",
    "active",
    "asm",
    "class",
    "union",
    "enum",
    "typedef",
    "template",
    "this",
    "resource",
    "goto",
    "inline",
    "noinline",
    "public",
    "static",
    "extern",
    "external",
    "interface",
    "long",
    "short",
    "half",
    "fixed",
    "unsigned",
    "superp",
    "input",
    "output",
    "filter",
    "sizeof",
    "cast",
    "namespace",
    "using",
];

/// Names the stitched file itself defines beside the parameters.
const STITCHED: &[&str] = &["surface", "sample_param", "surface_parameters"];

fn is_glsl_type(name: &str) -> bool {
    const DIMS: &[&str] = &[
        "",
        "1D",
        "2D",
        "3D",
        "Cube",
        "2DRect",
        "1DArray",
        "2DArray",
        "Buffer",
        "2DMS",
        "2DMSArray",
        "CubeArray",
        "Shadow",
        "1DShadow",
        "2DShadow",
        "CubeShadow",
        "2DRectShadow",
        "1DArrayShadow",
        "2DArrayShadow",
        "CubeArrayShadow",
        "3DRect",
        "MS",
    ];
    let opaque = ["", "i", "u"].iter().any(|prefix| {
        name.strip_prefix(prefix).is_some_and(|rest| {
            ["sampler", "image", "texture", "subpassInput"]
                .iter()
                .any(|base| {
                    rest.strip_prefix(base).is_some_and(|dims| {
                        DIMS.contains(&dims) && !(dims.is_empty() && *base == "image")
                    })
                })
        })
    });
    let shaped = |prefixes: &[&str], stem: &str, tail: &dyn Fn(&str) -> bool| {
        prefixes.iter().any(|p| {
            name.strip_prefix(p)
                .and_then(|r| r.strip_prefix(stem))
                .is_some_and(tail)
        })
    };
    let size = |s: &str| matches!(s, "2" | "3" | "4");
    let matrix =
        |s: &str| size(s) || (s.len() == 3 && size(&s[..1]) && &s[1..2] == "x" && size(&s[2..]));
    opaque || shaped(&["", "i", "u", "b", "d"], "vec", &size) || shaped(&["", "d"], "mat", &matrix)
}

/// Why `name` cannot name a parameter, if it cannot.
fn reserved(name: &str) -> Option<String> {
    if KEYWORDS.contains(&name) || is_glsl_type(name) {
        return Some(format!(
            "`{name}` is a GLSL keyword and cannot name a parameter"
        ));
    }
    if name.starts_with("gl_") {
        return Some(format!(
            "`{name}`: names beginning `gl_` are reserved by GLSL"
        ));
    }
    if name.contains("__") {
        return Some(format!(
            "`{name}`: names containing `__` are reserved by GLSL"
        ));
    }
    if STITCHED.contains(&name) {
        return Some(format!(
            "`{name}` is defined by the stitched shader and cannot name a parameter"
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = "\
param color tint = (1.0, 0.45, 0.2, 1.0);
param float wear = 0.5 : range(0, 1);
param texture mask = white;

void surface(in SurfaceIn i, inout SurfaceOut o) {
    float m = sample_param(mask, i.uv).r * wear;
    o.albedo = mix(o.albedo, tint.rgb, m);
    o.roughness = mix(o.roughness, 1.0, m);
}
";

    const SURFACE: &str = "void surface(in SurfaceIn i, inout SurfaceOut o) {}\n";

    fn parse(text: &str) -> SurfaceShader {
        SurfaceShader::parse(text).unwrap_or_else(|errors| panic!("{errors:#?}"))
    }

    fn only_param(declaration: &str) -> ParamKind {
        let shader = parse(&format!("{declaration}\n{SURFACE}"));
        assert_eq!(shader.params().params.len(), 1);
        shader.params().params[0].kind
    }

    /// The errors for `text`, as `(line, message)`.
    fn errors(text: &str) -> Vec<(Option<u32>, String)> {
        SurfaceShader::parse(text)
            .err()
            .unwrap_or_else(|| panic!("parsed:\n{text}"))
            .into_iter()
            .map(|e| (e.line, e.message))
            .collect()
    }

    fn one_error(text: &str) -> (u32, String) {
        let found = errors(text);
        assert_eq!(found.len(), 1, "{found:#?}");
        (found[0].0.expect("a line"), found[0].1.clone())
    }

    #[test]
    fn the_example_parses_into_three_packed_parameters() {
        let shader = parse(EXAMPLE);
        let table = shader.params();
        let names: Vec<_> = table.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["tint", "wear", "mask"]);
        assert_eq!(
            table.params[0].kind,
            ParamKind::Color {
                default: [1.0, 0.45, 0.2, 1.0]
            }
        );
        assert_eq!(
            table.params[1].kind,
            ParamKind::Float {
                default: 0.5,
                range: Some([0.0, 1.0])
            }
        );
        assert_eq!(
            table.params[2].kind,
            ParamKind::Texture {
                default: DefaultTexture::White,
                color_space: ColorSpace::Srgb
            }
        );
        let slots: Vec<_> = table.params.iter().map(|p| p.slot).collect();
        assert_eq!(
            slots,
            [
                Slot {
                    word: 0,
                    component: 0
                },
                Slot {
                    word: 1,
                    component: 0
                },
                Slot {
                    word: 1,
                    component: 1
                },
            ]
        );
        assert_eq!(table.word_count, 2);
        assert_eq!(shader.declared_on(), [1, 2, 3]);
    }

    #[test]
    fn parameter_lines_are_blanked_and_every_other_line_keeps_its_number() {
        let shader = parse(EXAMPLE);
        let body: Vec<_> = shader.body().lines().collect();
        let source: Vec<_> = EXAMPLE.lines().collect();
        assert_eq!(body.len(), source.len());
        assert!(body[..3].iter().all(|line| line.trim().is_empty()));
        assert_eq!(body[4..], source[4..]);
    }

    #[test]
    fn each_type_takes_its_default_and_has_one_when_omitted() {
        assert_eq!(
            only_param("param float a = -2.5;"),
            ParamKind::Float {
                default: -2.5,
                range: None
            }
        );
        assert_eq!(
            only_param("param vec2 a = (1, 2);"),
            ParamKind::Vec2 {
                default: [1.0, 2.0]
            }
        );
        assert_eq!(
            only_param("param vec3 a = (1, 2.5, -3e-1);"),
            ParamKind::Vec3 {
                default: [1.0, 2.5, -0.3]
            }
        );
        assert_eq!(
            only_param("param vec4 a = (1, 2, 3, 4);"),
            ParamKind::Vec4 {
                default: [1.0, 2.0, 3.0, 4.0]
            }
        );
        assert_eq!(
            only_param("param color a = (0.5, 0.25, 0.125);"),
            ParamKind::Color {
                default: [0.5, 0.25, 0.125, 1.0]
            },
            "a colour without alpha is opaque"
        );
        assert_eq!(
            only_param("param texture a = black;"),
            ParamKind::Texture {
                default: DefaultTexture::Black,
                color_space: ColorSpace::Srgb
            }
        );

        assert_eq!(
            only_param("param float a;"),
            ParamKind::Float {
                default: 0.0,
                range: None
            }
        );
        assert_eq!(
            only_param("param vec3 a;"),
            ParamKind::Vec3 { default: [0.0; 3] }
        );
        assert_eq!(
            only_param("param color a;"),
            ParamKind::Color { default: [1.0; 4] }
        );
        assert_eq!(
            only_param("param texture a;"),
            ParamKind::Texture {
                default: DefaultTexture::White,
                color_space: ColorSpace::Srgb
            }
        );
    }

    #[test]
    fn each_hint_is_read_and_a_normal_texture_imports_linear() {
        assert_eq!(
            only_param("param float a = 2 : range(-1, 4.5);"),
            ParamKind::Float {
                default: 2.0,
                range: Some([-1.0, 4.5])
            }
        );
        assert_eq!(
            only_param("param texture a = white : linear;"),
            ParamKind::Texture {
                default: DefaultTexture::White,
                color_space: ColorSpace::Linear
            }
        );
        assert_eq!(
            only_param("param texture a = normal;"),
            ParamKind::Texture {
                default: DefaultTexture::Normal,
                color_space: ColorSpace::Linear
            }
        );
        assert_eq!(
            only_param("param texture a = normal : srgb;"),
            ParamKind::Texture {
                default: DefaultTexture::Normal,
                color_space: ColorSpace::Srgb
            }
        );
        assert_eq!(
            only_param("  param float a : range(0, 1);  // a trailing comment"),
            ParamKind::Float {
                default: 0.0,
                range: Some([0.0, 1.0])
            }
        );
    }

    #[test]
    fn a_parameter_less_empty_function_is_the_standard_material() {
        let shader = parse(SURFACE);
        assert_eq!(shader.params(), &ParamTable::default());
    }

    #[test]
    fn a_malformed_declaration_names_its_line_and_what_was_expected() {
        let cases: [(&str, &str); 14] = [
            ("param float3 a;", "type"),
            ("param float;", "name"),
            ("param float 2a;", "name"),
            ("param float a = ;", "number"),
            ("param float a = 1.0", "`;`"),
            ("param float a = 1.0; float b;", "after"),
            ("param vec3 a = (1, 2);", "3 numbers"),
            ("param vec2 a = 1.0;", "`(`"),
            ("param color a = (1, 2);", "3 or 4"),
            ("param texture a = grey;", "white, black or normal"),
            ("param float a = 1e40;", "number"),
            ("param float a : bogus;", "hint"),
            ("param float a : range(0 1);", "`,`"),
            ("param float a = 1 : range(0, 1), range(0, 2);", "once"),
        ];
        for (declaration, expected) in cases {
            let (line, message) = one_error(&format!("\n\n{declaration}\n{SURFACE}"));
            assert_eq!(line, 3, "{declaration}: {message}");
            assert!(
                message.contains(expected),
                "{declaration}: `{message}` does not mention {expected}"
            );
        }
    }

    #[test]
    fn a_hint_on_the_wrong_type_or_out_of_its_own_range_is_an_error() {
        let cases: [(&str, &str); 5] = [
            ("param vec2 a : range(0, 1);", "float"),
            ("param float a : srgb;", "texture"),
            ("param color a : linear;", "texture"),
            ("param float a : range(1, 1);", "less than"),
            ("param float a = 2 : range(0, 1);", "outside"),
        ];
        for (declaration, expected) in cases {
            let (line, message) = one_error(&format!("{SURFACE}{declaration}\n"));
            assert_eq!(line, 2, "{declaration}: {message}");
            assert!(message.contains(expected), "{declaration}: {message}");
        }
    }

    #[test]
    fn a_name_must_be_new_and_not_one_glsl_or_the_stitching_owns() {
        let (line, message) = one_error(&format!(
            "param float a;\nparam vec2 b;\nparam float a;\n{SURFACE}"
        ));
        assert_eq!(line, 3);
        assert!(message.contains("already declared on line 1"), "{message}");

        for name in [
            "return",
            "sampler2D",
            "usampler2DArray",
            "texture2D",
            "mat3x4",
            "dvec2",
            "gl_Thing",
            "a__b",
            "surface",
            "sample_param",
            "surface_parameters",
            "precision",
            "input",
        ] {
            let (line, message) = one_error(&format!("{SURFACE}param float {name};\n"));
            assert_eq!(line, 2, "{name}");
            assert!(message.contains(name), "{name}: {message}");
        }
    }

    #[test]
    fn a_file_without_a_surface_function_is_an_error() {
        let found =
            errors("param float a;\nvoid not_surface(in SurfaceIn i, inout SurfaceOut o) {}\n");
        assert_eq!(found.len(), 1, "{found:#?}");
        assert!(found[0].1.contains("void surface("), "{found:#?}");
        assert_eq!(found[0].0, Some(2), "reported at the end of the file");
    }

    #[test]
    fn each_forbidden_construct_is_named_at_its_line() {
        let cases: [(&str, &str); 9] = [
            ("    gl_FragDepth = 0.5;", "gl_FragDepth"),
            ("    if (o.alpha < 0.5) discard;", "discard"),
            ("    layout(location = 1) vec4 x;", "layout"),
            ("    uniform float x;", "uniform"),
            ("    buffer B { float x; };", "buffer"),
            ("#version 450", "#version"),
            ("#extension GL_EXT_foo : require", "#extension"),
            ("  #  include \"x.glsl\"", "#include"),
            ("#line 40", "#line"),
        ];
        for (construct, named) in cases {
            let text = format!(
                "void surface(in SurfaceIn i, inout SurfaceOut o) {{\n    o.alpha = 1.0;\n{construct}\n}}\n"
            );
            let (line, message) = one_error(&text);
            assert_eq!(line, 3, "{construct}");
            assert!(message.contains(named), "{construct}: {message}");
        }
    }

    #[test]
    fn look_alikes_in_comments_and_longer_identifiers_are_accepted() {
        parse(
            "// discard, layout(set = 0), uniform buffer, gl_FragDepth, #version\n\
             /* a block comment that mentions discard\n\
                and spans lines, then #include \"x\" */\n\
             #define DISCARDED 0\n\
             float discarded(float uniformity) { return uniformity; }\n\
             void surface(in SurfaceIn i, inout SurfaceOut o) {\n\
                 float layout_scale = discarded(1.0); // discard\n\
                 float buffer_size = layout_scale; /* gl_FragDepth */\n\
                 float gl_FragDepthish = buffer_size;\n\
                 float param = gl_FragDepthish;\n\
                 o.alpha = param;\n\
             }\n",
        );
    }

    #[test]
    fn param_inside_a_function_is_glsl_not_a_declaration() {
        let shader = parse(
            "void surface(in SurfaceIn i, inout SurfaceOut o) {\n\
             \x20   float param = 1.0;\n\
             param = 2.0;\n\
             }\n",
        );
        assert!(shader.params().params.is_empty());
    }

    #[test]
    fn every_error_in_a_file_is_reported_at_once() {
        let found = errors("param flaot a;\nparam float b = (1);\nvoid x() { discard; }\n");
        let lines: Vec<_> = found.iter().map(|(line, _)| *line).collect();
        assert_eq!(lines, [Some(1), Some(2), Some(3), Some(3)], "{found:#?}");
    }

    #[test]
    fn an_unclosed_block_comment_is_an_error_at_its_start() {
        let (line, message) = one_error(&format!("{SURFACE}\n/* never closed\n\n"));
        assert_eq!(line, 3);
        assert!(message.contains("/*"), "{message}");
    }

    /// Otherwise the compiler reports it inside the engine's code after the
    /// include, at a line the user cannot see.
    #[test]
    fn an_unclosed_brace_is_an_error_at_the_end_of_the_file() {
        let (line, message) =
            one_error("void surface(in SurfaceIn i, inout SurfaceOut o) {\n    o.alpha = 1.0;\n");
        assert_eq!(line, 2);
        assert!(message.contains("`{`"), "{message}");
    }

    #[test]
    fn macros_the_source_defines_are_recorded() {
        let shader = parse(&format!(
            "#define TWO 2.0\n#define CAT(a, b) a##b\n{SURFACE}"
        ));
        assert_eq!(shader.defines(), ["TWO", "CAT"]);
    }

    #[test]
    fn a_located_error_reads_file_line_message() {
        let error = SurfaceError::at(7, "something");
        assert_eq!(error.located("rock.oshader"), "rock.oshader:7: something");
        assert_eq!(error.to_string(), "line 7: something");
        let error = SurfaceError {
            line: None,
            message: "raw".into(),
        };
        assert_eq!(error.located("rock.oshader"), "rock.oshader: raw");
    }
}
