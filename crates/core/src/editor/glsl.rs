//! GLSL as the Shader tab colours it: a tokenizer for the one language a
//! `.oshader` is written in, and a highlighter that lays a text out once.
//!
//! Hand-written rather than a grammar engine such as syntect, which would bring
//! a regex engine and a set of grammar files to colour one language. It is
//! deliberately shallow: words, numbers, comments and preprocessor lines, plus
//! what `gfx::surface` adds to GLSL — the `param` line, the names it declares,
//! and the surface function's vocabulary. A text being typed is mostly not
//! valid GLSL, so nothing here can fail: any text is covered by tokens, in
//! order, and an unterminated comment simply runs to the end.
//!
//! **Laid out once per text.** egui calls a text edit's layouter every frame,
//! so [`Highlighter`] keeps the last galley with the text and the style it was
//! made for, and hands the same galley back until one of them changes.

use std::ops::Range;
use std::sync::Arc;

use super::theme::Syntax;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Text,
    Keyword,
    Type,
    Number,
    Comment,
    Preprocessor,
    /// What the engine gives a surface function: its signature's types and
    /// the functions it may call.
    Engine,
    /// A name a `param` line declares, wherever it appears.
    Param,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Token {
    pub range: Range<usize>,
    pub class: Class,
}

const KEYWORDS: &[&str] = &[
    "break",
    "buffer",
    "case",
    "const",
    "continue",
    "default",
    "discard",
    "do",
    "else",
    "false",
    "flat",
    "for",
    "highp",
    "if",
    "in",
    "inout",
    "invariant",
    "layout",
    "lowp",
    "mediump",
    "noperspective",
    "out",
    "precision",
    "return",
    "smooth",
    "struct",
    "switch",
    "true",
    "uniform",
    "void",
    "while",
];

const ENGINE: &[&str] = &["SurfaceIn", "SurfaceOut", "sample_param", "surface"];

/// The types a `param` line may name, and the words of its defaults and hints.
const PARAM_TYPES: &[&str] = &["float", "vec2", "vec3", "vec4", "color", "texture"];
const PARAM_WORDS: &[&str] = &["range", "srgb", "linear", "white", "black", "normal"];

fn is_type(word: &str) -> bool {
    let vector = word
        .strip_prefix(['b', 'i', 'u', 'd'])
        .unwrap_or(word)
        .strip_prefix("vec")
        .is_some_and(|n| matches!(n, "2" | "3" | "4"));
    let matrix = word
        .strip_prefix('d')
        .unwrap_or(word)
        .strip_prefix("mat")
        .is_some_and(|n| {
            matches!(
                n.as_bytes(),
                [b'2'..=b'4'] | [b'2'..=b'4', b'x', b'2'..=b'4']
            )
        });
    vector
        || matrix
        || matches!(word, "bool" | "int" | "uint" | "float" | "double")
        || word.starts_with("sampler")
}

/// Where a `param` line has got to.
#[derive(Clone, Copy, PartialEq)]
enum Declaring {
    No,
    Type,
    Name,
    Rest,
}

/// `text` as tokens that cover it exactly, in order. Never fails.
pub fn tokens(text: &str) -> Vec<Token> {
    let bytes = text.as_bytes();
    // `None` is a word to resolve once every declared name is known.
    let mut raw: Vec<(Range<usize>, Option<Class>)> = Vec::new();
    let mut declared: Vec<&str> = Vec::new();
    let mut depth = 0u32;
    let mut line_start = true;
    let mut declaring = Declaring::No;
    let line_end = |from: usize| text[from..].find('\n').map_or(text.len(), |at| from + at);
    let mut i = 0;
    while i < bytes.len() {
        let start = i;
        let byte = bytes[i];
        let next = bytes.get(i + 1).copied();
        let class = match byte {
            b'/' if next == Some(b'/') => {
                i = line_end(i);
                Some(Class::Comment)
            }
            b'/' if next == Some(b'*') => {
                i = text[i + 2..]
                    .find("*/")
                    .map_or(text.len(), |at| i + 2 + at + 2);
                Some(Class::Comment)
            }
            b'#' if line_start => {
                i = line_end(i);
                Some(Class::Preprocessor)
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                let word = &text[start..i];
                match declaring {
                    Declaring::No if word == "param" && line_start && depth == 0 => {
                        declaring = Declaring::Type;
                        Some(Class::Keyword)
                    }
                    Declaring::Type => {
                        declaring = Declaring::Name;
                        Some(if PARAM_TYPES.contains(&word) {
                            Class::Type
                        } else {
                            Class::Text
                        })
                    }
                    Declaring::Name => {
                        declaring = Declaring::Rest;
                        declared.push(word);
                        Some(Class::Param)
                    }
                    Declaring::Rest if PARAM_WORDS.contains(&word) => Some(Class::Keyword),
                    _ if KEYWORDS.contains(&word) => Some(Class::Keyword),
                    _ if is_type(word) => Some(Class::Type),
                    _ if ENGINE.contains(&word) => Some(Class::Engine),
                    _ => None,
                }
            }
            b'0'..=b'9' => {
                i = number(bytes, i);
                Some(Class::Number)
            }
            b'.' if next.is_some_and(|b| b.is_ascii_digit()) => {
                i = number(bytes, i);
                Some(Class::Number)
            }
            _ => {
                match byte {
                    b'{' => depth += 1,
                    b'}' => depth = depth.saturating_sub(1),
                    b'\n' => declaring = Declaring::No,
                    _ => {}
                }
                i += text[i..].chars().next().map_or(1, char::len_utf8);
                Some(Class::Text)
            }
        };
        line_start = match byte {
            b'\n' => true,
            b' ' | b'\t' | b'\r' => line_start,
            _ => false,
        };
        raw.push((start..i, class));
    }

    let mut out: Vec<Token> = Vec::with_capacity(raw.len());
    for (range, class) in raw {
        let class = class.unwrap_or_else(|| {
            if declared.contains(&&text[range.clone()]) {
                Class::Param
            } else {
                Class::Text
            }
        });
        match out.last_mut() {
            Some(last) if last.class == Class::Text && class == Class::Text => {
                last.range.end = range.end;
            }
            _ => out.push(Token { range, class }),
        }
    }
    out
}

/// The end of the number starting at `i`: decimal or hex, a fraction, an
/// exponent when one follows whole, and a type suffix.
fn number(bytes: &[u8], mut i: usize) -> usize {
    let at = |i: usize| bytes.get(i).copied().unwrap_or(0);
    if at(i) == b'0' && at(i + 1) | 0x20 == b'x' {
        i += 2;
        while at(i).is_ascii_hexdigit() {
            i += 1;
        }
    } else {
        while at(i).is_ascii_digit() || at(i) == b'.' {
            i += 1;
        }
        if at(i) | 0x20 == b'e' {
            let mut j = i + 1;
            if matches!(at(j), b'+' | b'-') {
                j += 1;
            }
            if at(j).is_ascii_digit() {
                i = j;
                while at(i).is_ascii_digit() {
                    i += 1;
                }
            }
        }
    }
    while matches!(at(i), b'f' | b'F' | b'u' | b'U' | b'l' | b'L') {
        i += 1;
    }
    i
}

/// A text edit's layouter for GLSL, which tokenises and lays out a text only
/// when it, the palette, the font or the display has changed.
#[derive(Default)]
pub struct Highlighter {
    text: String,
    made_for: Option<(Syntax, egui::FontId, f32, [usize; 2])>,
    galley: Option<Arc<egui::Galley>>,
    runs: u32,
}

impl Highlighter {
    /// `text` laid out on one line per line, never wrapped: code is read by
    /// its lines, and the gutter numbers them.
    pub fn layout(&mut self, ui: &egui::Ui, text: &str, syntax: Syntax) -> Arc<egui::Galley> {
        let font = egui::TextStyle::Monospace.resolve(ui.style());
        // The atlas's size moves when egui rebuilds it, which retires every
        // galley laid out against the old one.
        let atlas = ui.fonts(|fonts| fonts.font_image_size());
        let made_for = (syntax, font.clone(), ui.ctx().pixels_per_point(), atlas);
        if let Some(galley) = &self.galley
            && self.text == text
            && self.made_for.as_ref() == Some(&made_for)
        {
            return galley.clone();
        }
        self.runs += 1;
        let mut job = egui::text::LayoutJob {
            text: text.to_owned(),
            ..Default::default()
        };
        job.wrap.max_width = f32::INFINITY;
        for token in tokens(text) {
            job.sections.push(egui::text::LayoutSection {
                leading_space: 0.0,
                byte_range: token.range,
                format: egui::TextFormat::simple(font.clone(), syntax.of_class(token.class)),
            });
        }
        let galley = ui.fonts(|fonts| fonts.layout_job(job));
        self.text.clear();
        self.text.push_str(text);
        self.made_for = Some(made_for);
        self.galley = Some(galley.clone());
        galley
    }

    /// How many times a text has been tokenised and laid out.
    #[cfg(test)]
    pub fn runs(&self) -> u32 {
        self.runs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each token's class, by its text.
    fn classes(text: &str) -> Vec<(&str, Class)> {
        tokens(text)
            .into_iter()
            .filter(|token| token.class != Class::Text)
            .map(|token| (&text[token.range], token.class))
            .collect()
    }

    fn class_of(text: &str, word: &str) -> Vec<Class> {
        classes(text)
            .into_iter()
            .filter(|(at, _)| *at == word)
            .map(|(_, class)| class)
            .collect()
    }

    const SAMPLE: &str = "\
// A wear mask.
param color tint = (1.0, 0.45, 0.2, 1.0);
param float wear = 0.5 : range(0, 1);
param texture mask = normal : linear;
#define TWICE(x) ((x) * 2.0)
/* spans
   two lines */
void surface(in SurfaceIn i, inout SurfaceOut o) {
    float m = sample_param(mask, i.uv).r * wear;
    if (m > 0.5e-1) { o.albedo = mix(o.albedo, tint.rgb, m); }
    int n = 0x1F;
}
";

    #[test]
    fn every_class_is_found_where_it_is() {
        assert_eq!(class_of(SAMPLE, "// A wear mask."), [Class::Comment]);
        assert_eq!(
            class_of(SAMPLE, "/* spans\n   two lines */"),
            [Class::Comment],
            "one token across both lines"
        );
        assert_eq!(
            class_of(SAMPLE, "#define TWICE(x) ((x) * 2.0)"),
            [Class::Preprocessor],
            "the whole line"
        );
        for keyword in ["void", "in", "inout", "if"] {
            assert_eq!(class_of(SAMPLE, keyword), [Class::Keyword], "{keyword}");
        }
        for ty in ["float", "int"] {
            assert!(
                class_of(SAMPLE, ty).iter().all(|c| *c == Class::Type),
                "{ty}"
            );
        }
        for number in ["0.5e-1", "0x1F", "0.45", "0", "1"] {
            assert!(
                class_of(SAMPLE, number).iter().all(|c| *c == Class::Number),
                "{number}: {:?}",
                class_of(SAMPLE, number)
            );
        }
        for engine in ["SurfaceIn", "SurfaceOut", "sample_param", "surface"] {
            assert_eq!(class_of(SAMPLE, engine), [Class::Engine], "{engine}");
        }
        assert_eq!(class_of(SAMPLE, "mix"), Vec::<Class>::new(), "plain text");
        assert_eq!(class_of(SAMPLE, "albedo"), Vec::<Class>::new());
    }

    /// A `param` line is the language's own: its word, its type, its name
    /// and its hints. The name is a parameter everywhere it is used.
    #[test]
    fn a_param_line_declares_a_name_used_everywhere() {
        let line = "param float wear = 0.5 : range(0, 1);";
        assert_eq!(
            classes(line),
            [
                ("param", Class::Keyword),
                ("float", Class::Type),
                ("wear", Class::Param),
                ("0.5", Class::Number),
                ("range", Class::Keyword),
                ("0", Class::Number),
                ("1", Class::Number),
            ]
        );
        assert_eq!(class_of(SAMPLE, "color"), [Class::Type]);
        assert_eq!(class_of(SAMPLE, "texture"), [Class::Type]);
        assert_eq!(class_of(SAMPLE, "normal"), [Class::Keyword]);
        assert_eq!(class_of(SAMPLE, "linear"), [Class::Keyword]);
        assert_eq!(class_of(SAMPLE, "wear"), [Class::Param, Class::Param]);
        assert_eq!(class_of(SAMPLE, "tint"), [Class::Param, Class::Param]);
        assert_eq!(class_of(SAMPLE, "mask"), [Class::Param, Class::Param]);

        // Inside braces it is a word like any other, and declares nothing.
        let nested = "void f() {\n    param float x;\n}\nfloat y = x;";
        assert_eq!(class_of(nested, "param"), Vec::<Class>::new());
        assert_eq!(class_of(nested, "x"), Vec::<Class>::new());
        // A param word in a comment declares nothing either.
        assert_eq!(class_of("// param float z;\nz;", "z"), Vec::<Class>::new());
    }

    /// The tokens cover the text exactly, in order, whatever the text is: an
    /// editor is mostly showing text that is half typed.
    #[test]
    fn any_text_tokenises_without_a_gap_or_a_panic() {
        let odd = "/* never closed\nparam float x = 1.\n#\n0x\n1e+\n\"é\u{1F600}\" ``` param";
        let mut texts: Vec<&str> = vec![odd, "", "/", "/*", "#", "param", "param ", "1."];
        for text in [SAMPLE, odd] {
            texts.extend(
                text.char_indices()
                    .map(|(at, _)| &text[..at])
                    .chain(text.char_indices().map(|(at, _)| &text[at..])),
            );
        }
        for text in texts {
            let tokens = tokens(text);
            let mut at = 0;
            for token in &tokens {
                assert_eq!(token.range.start, at, "a gap in {text:?}: {tokens:?}");
                assert!(token.range.end > token.range.start, "{text:?}: {tokens:?}");
                assert!(text.is_char_boundary(token.range.end));
                at = token.range.end;
            }
            assert_eq!(at, text.len(), "{text:?} is not covered: {tokens:?}");
        }
        assert_eq!(
            class_of(
                odd,
                "/* never closed\nparam float x = 1.\n#\n0x\n1e+\n\"é\u{1F600}\" ``` param"
            ),
            [Class::Comment],
            "an open comment runs to the end"
        );
    }

    /// The text is laid out once, and every frame after that until it changes
    /// is handed the same galley.
    #[test]
    fn unchanged_text_is_tokenised_once() {
        let ctx = egui::Context::default();
        let syntax = Syntax::of(&ctx.style().visuals);
        let mut highlighter = Highlighter::default();
        let mut galleys = Vec::new();
        for text in [SAMPLE, SAMPLE, SAMPLE, "void x;", "void x;"] {
            let _ = ctx.run(Default::default(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    galleys.push(highlighter.layout(ui, text, syntax));
                });
            });
        }
        assert_eq!(highlighter.runs(), 2);
        assert!(Arc::ptr_eq(&galleys[0], &galleys[2]));
        assert_eq!(galleys[3].text(), "void x;");

        let sections = &galleys[0].job.sections;
        let colour = |word: &str| {
            let at = SAMPLE.find(word).unwrap();
            sections
                .iter()
                .find(|section| section.byte_range.contains(&at))
                .map(|section| section.format.color)
        };
        assert_eq!(colour("SurfaceIn"), Some(syntax.engine));
        assert_eq!(colour("// A wear"), Some(syntax.comment));
        assert_eq!(colour("param"), Some(syntax.keyword));
        assert_eq!(colour("0.45"), Some(syntax.number));
    }
}
