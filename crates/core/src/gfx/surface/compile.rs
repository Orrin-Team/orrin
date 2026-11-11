use std::collections::BTreeSet;
use std::fmt::Write as _;

use super::PARAMS_BINDING;
use super::artifact::{CompiledVariant, SurfaceArtifact};
use super::params::{Param, ParamKind};
use super::parse::{
    Depth, SurfaceError, SurfaceShader, Token, forbidden_token, strip_comments, tokens,
};
use crate::gfx::shaders::{
    self, CompiledShader, Resource, SOURCES, SOURCES_HASH, ShaderBlob, ShaderId, Stage,
};

const CUSTOM: &str = "ORRIN_CUSTOM_SURFACE";
const FLAT: &str = "ORRIN_FLAT_SURFACE";
/// The include `surface.glsl` reads the custom function from, which the
/// generated file stands in for.
const INCLUDE: &str = "custom_surface.glsl";
/// What the generated file's `#line` calls the user's text, so an error in it
/// can be told from one in the generated lines around it.
const USER: &str = "orrin-surface-source";

/// The built-in shaders a surface shader is compiled in place of: every
/// fragment row of the manifest whose shader calls `resolve_surface`, less the
/// flat rows and the row that already compiles the custom path. In manifest
/// order.
pub fn variants() -> Vec<ShaderId> {
    ShaderId::ALL
        .iter()
        .copied()
        .filter(|id| {
            let blob = id.blob();
            blob.stage == Stage::Fragment
                && !blob
                    .defines
                    .iter()
                    .any(|(key, _)| [FLAT, CUSTOM].contains(key))
                && reaches(blob.path)
                    .into_iter()
                    .any(|file| file != "surface.glsl" && mentions(source(file), "resolve_surface"))
        })
        .collect()
}

/// Stitch `shader` into every variant, compile each, and hold each to its
/// built-in shader's bindings. `name` is what errors call the file. Every error
/// from every variant is returned, once each.
pub fn compile(shader: &SurfaceShader, name: &str) -> Result<SurfaceArtifact, Vec<SurfaceError>> {
    let ids = variants();
    let collisions = collisions(shader, &ids);
    if !collisions.is_empty() {
        return Err(collisions);
    }

    let stitched = stitch(shader);
    let mut errors: Vec<SurfaceError> = Vec::new();
    let mut compiled = Vec::new();
    for id in ids {
        let blob = id.blob();
        let mut defines = blob.defines.to_vec();
        defines.push((CUSTOM, "1"));
        let found = match shaders::compile(
            source(blob.path),
            blob.stage,
            blob.path,
            &defines,
            &[(INCLUDE, &stitched.text)],
        ) {
            Err(message) => located(&message, shader, &stitched),
            Ok(mut variant) => {
                let found = validate(&variant, blob);
                variant.name = format!("{name} [{}]", blob.name);
                compiled.push(CompiledVariant {
                    key: blob.name.to_string(),
                    shader: variant,
                });
                found
            }
        };
        for error in found {
            if !errors.contains(&error) {
                errors.push(error);
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(SurfaceArtifact {
        sources_hash: SOURCES_HASH,
        params: shader.params().clone(),
        variants: compiled,
    })
}

fn source(path: &str) -> &'static str {
    SOURCES
        .iter()
        .find(|(file, _)| *file == path)
        .map_or("", |(_, text)| text)
}

fn mentions(text: &str, identifier: &str) -> bool {
    let (text, _) = strip_comments(text);
    text.lines()
        .flat_map(tokens)
        .any(|token| token == Token::Ident(identifier))
}

/// `path` and every engine file it includes, however deeply.
fn reaches(path: &'static str) -> Vec<&'static str> {
    let mut files = vec![path];
    let mut next = 0;
    while let Some(&file) = files.get(next) {
        next += 1;
        let (text, _) = strip_comments(source(file));
        for line in text.lines() {
            let Some(directive) = line.trim_start().strip_prefix('#') else {
                continue;
            };
            let Some(name) = directive.trim_start().strip_prefix("include") else {
                continue;
            };
            let Some(included) = name.split('"').nth(1) else {
                continue;
            };
            if let Some(&(found, _)) = SOURCES.iter().find(|(file, _)| *file == included)
                && !files.contains(&found)
            {
                files.push(found);
            }
        }
    }
    files
}

/// Every name the variants' engine files declare at file scope, and every macro
/// they define: what a parameter's global would collide with. Read from the
/// sources rather than listed, so it cannot fall behind them.
fn engine_globals(ids: &[ShaderId]) -> BTreeSet<String> {
    let files: BTreeSet<&str> = ids.iter().flat_map(|id| reaches(id.blob().path)).collect();
    let mut names = BTreeSet::new();
    for file in files {
        let (text, _) = strip_comments(source(file));
        let mut depth = Depth::default();
        for line in text.lines() {
            if let Some(directive) = line.trim_start().strip_prefix('#') {
                if let [Token::Ident("define"), Token::Ident(name), ..] = tokens(directive)[..] {
                    names.insert(name.to_string());
                }
                continue;
            }
            for token in tokens(line) {
                if let Token::Ident(name) = token
                    && depth.file_scope()
                {
                    names.insert(name.to_string());
                }
                depth.step(token);
            }
        }
    }
    names
}

fn collisions(shader: &SurfaceShader, ids: &[ShaderId]) -> Vec<SurfaceError> {
    let engine = engine_globals(ids);
    shader
        .params()
        .params
        .iter()
        .zip(shader.declared_on())
        .filter(|(param, _)| engine.contains(&param.name))
        .map(|(param, &line)| collision(&param.name, line))
        .collect()
}

fn collision(name: &str, line: u32) -> SurfaceError {
    SurfaceError::at(
        line,
        format!("`{name}` is a name the engine's shaders already use; rename the parameter"),
    )
}

/// The generated `custom_surface.glsl`, and the line of it each parameter's
/// global is declared on.
struct Stitched {
    text: String,
    declared_at: Vec<u32>,
}

fn stitch(shader: &SurfaceShader) -> Stitched {
    let params = &shader.params().params;
    let mut text = String::new();
    let mut declared_at = Vec::new();
    if !params.is_empty() {
        text.push_str("#define ORRIN_SURFACE_PARAMETERS 1\n");
        for param in params {
            declared_at.push(text.lines().count() as u32 + 1);
            writeln!(text, "{} {};", glsl_type(&param.kind), param.name).unwrap();
        }
        text.push_str("void surface_parameters(uint base) {\n");
        for param in params {
            writeln!(text, "    {} = {};", param.name, load(param)).unwrap();
        }
        text.push_str("}\n");
    }
    text.push_str("vec4 sample_param(uint index, vec2 uv) { return sample_tex(index, uv); }\n");
    writeln!(text, "#line 1 \"{USER}\"").unwrap();
    text.push_str(shader.body());
    // The user's macros would otherwise rewrite the engine's code after the
    // include: a `#define albedo` breaks every `o.albedo` below it.
    for define in shader.defines() {
        writeln!(text, "#undef {define}").unwrap();
    }
    Stitched { text, declared_at }
}

fn glsl_type(kind: &ParamKind) -> &'static str {
    match kind {
        ParamKind::Float { .. } => "float",
        ParamKind::Vec2 { .. } => "vec2",
        ParamKind::Vec3 { .. } => "vec3",
        ParamKind::Vec4 { .. } | ParamKind::Color { .. } => "vec4",
        ParamKind::Texture { .. } => "uint",
    }
}

fn load(param: &Param) -> String {
    let word = format!("surface_params.words[base + {}u]", param.slot.word);
    let n = param.kind.components() as usize;
    let at = param.slot.component as usize;
    let swizzle = &"xyzw"[at..at + n];
    match param.kind {
        ParamKind::Texture { .. } => format!("floatBitsToUint({word}.{swizzle})"),
        _ if n == 4 => word,
        _ => format!("{word}.{swizzle}"),
    }
}

/// The compiler's errors, at the user's lines wherever they can be. An error in
/// a generated line maps to the parameter that line declares; one the engine's
/// code reports about a parameter's name maps to its declaration; anything else
/// keeps its raw message and no line. When any error is at a user's line, the
/// rest are its consequences and are dropped.
fn located(message: &str, shader: &SurfaceShader, stitched: &Stitched) -> Vec<SurfaceError> {
    let params = &shader.params().params;
    let user_lines = shader.body().lines().count() as u32;
    let mut errors = Vec::new();
    for raw in message.lines() {
        let Some((location, text)) = raw.split_once(": error: ") else {
            continue;
        };
        let Some((file, line)) = location
            .rsplit_once(':')
            .and_then(|(file, line)| Some((file, line.parse::<u32>().ok()?)))
        else {
            continue;
        };
        let declared = if file == INCLUDE {
            stitched.declared_at.iter().position(|&at| at == line)
        } else {
            None
        };
        let quoted = params
            .iter()
            .position(|param| text.contains(&format!("'{}'", param.name)));
        errors.push(if file == USER && (1..=user_lines).contains(&line) {
            SurfaceError::at(line, text.trim())
        } else if let Some(index) = declared.or(quoted) {
            collision(&params[index].name, shader.declared_on()[index])
        } else {
            SurfaceError {
                line: None,
                message: raw.trim().to_string(),
            }
        });
    }
    if errors.iter().any(|error| error.line.is_some()) {
        errors.retain(|error| error.line.is_some());
    }
    if errors.is_empty() {
        errors.push(SurfaceError {
            line: None,
            message: message.trim().to_string(),
        });
    }
    errors
}

/// The guard the token scan is the friendly face of: a variant binds what its
/// built-in shader binds plus `SurfaceParams`, with the same push constants,
/// and its SPIR-V holds no more `OpKill`s and no `FragDepth` the built-in lacks.
/// What a user can cause is worded alike in every variant, so it is reported
/// once.
fn validate(variant: &CompiledShader, builtin: &ShaderBlob) -> Vec<SurfaceError> {
    let key = builtin.name;
    let mut errors = Vec::new();
    let mut error = |message: String| {
        errors.push(SurfaceError {
            line: None,
            message,
        })
    };
    let params = surface_params();
    for resource in &variant.bindings.resources {
        if (resource.set, resource.binding) == PARAMS_BINDING {
            if *resource != params {
                error(format!(
                    "the `{key}` variant binds SurfaceParams as {resource:?}, where the engine \
                     declares {params:?}"
                ));
            }
        } else if !builtin.bindings.resources.contains(resource) {
            error(format!(
                "binds set {}, binding {} ({:?}): a surface shader binds nothing of its own, \
                 and its values arrive through `param` declarations",
                resource.set, resource.binding, resource.kind
            ));
        }
    }
    for resource in builtin.bindings.resources {
        if !variant.bindings.resources.contains(resource) {
            error(format!(
                "the `{key}` variant does not bind set {}, binding {}, which its pass binds",
                resource.set, resource.binding
            ));
        }
    }
    if !variant.bindings.resources.contains(&params) {
        error(format!("the `{key}` variant does not bind SurfaceParams"));
    }
    if variant.bindings.push_constant_bytes != builtin.bindings.push_constant_bytes {
        error(format!(
            "the `{key}` variant has {} bytes of push constants where its pass has {}",
            variant.bindings.push_constant_bytes, builtin.bindings.push_constant_bytes
        ));
    }

    let (ours, theirs) = (Spirv::scan(&variant.spirv), Spirv::scan(builtin.spirv));
    for (forbidden, found) in [
        ("discard", ours.kills > theirs.kills),
        ("gl_FragDepth", ours.frag_depth && !theirs.frag_depth),
    ] {
        if found {
            let reason = forbidden_token(Token::Ident(forbidden)).expect("a forbidden name");
            error(format!("{reason} (found in the compiled shader)"));
        }
    }
    errors
}

/// `SurfaceParams` as the build reflected it in the one manifest row that
/// exists to put it in the argument buffer layout.
fn surface_params() -> Resource {
    *ShaderId::PrepassFragCustomSurface
        .blob()
        .bindings
        .resources
        .iter()
        .find(|r| (r.set, r.binding) == PARAMS_BINDING)
        .expect("the custom-surface manifest row binds SurfaceParams")
}

/// What `validate` asks of a module's instructions.
struct Spirv {
    kills: usize,
    frag_depth: bool,
}

impl Spirv {
    const OP_DECORATE: u32 = 71;
    const OP_KILL: u32 = 252;
    const DECORATION_BUILT_IN: u32 = 11;
    const BUILT_IN_FRAG_DEPTH: u32 = 22;

    fn scan(bytes: &[u8]) -> Spirv {
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| u32::from_le_bytes(word))
            .collect();
        let mut found = Spirv {
            kills: 0,
            frag_depth: false,
        };
        // Past the five-word header, each instruction leads with its length.
        let mut at = 5;
        while let Some(&lead) = words.get(at) {
            let len = (lead >> 16) as usize;
            match lead & 0xffff {
                Self::OP_KILL => found.kills += 1,
                Self::OP_DECORATE
                    if words.get(at + 2..at + 4)
                        == Some(&[Self::DECORATION_BUILT_IN, Self::BUILT_IN_FRAG_DEPTH]) =>
                {
                    found.frag_depth = true
                }
                _ => {}
            }
            if len == 0 {
                break;
            }
            at += len;
        }
        found
    }
}
