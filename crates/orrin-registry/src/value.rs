use std::borrow::Cow;
use std::fmt;
use std::str::FromStr;

/// A component's data in a form nothing needs the concrete type to work with.
///
/// This is the vocabulary the rest of the engine agrees on: the scene format
/// writes it, the inspector edits it, diffs are expressed in it, and a C#
/// behaviour's fields cross the FFI boundary as one flattened `Struct` per
/// component.
///
/// A variant here is permanent — it lands in the on-disk scene format and in
/// the script ABI — so a new one needs a component that actually demands it.
/// `Vec3`/`Quat`/`Entity` are first-class rather than lists of numbers on
/// purpose: an inspector that cannot tell a position from three floats draws
/// three anonymous drag fields, and collaboration sync cannot tell an entity
/// reference from a pair of integers.
//
// `PartialEq` is field-wise, so it inherits `f32`'s `NaN != NaN`. That is fine
// for the round-trip assertions it exists for, but the diff pass will need a
// bitwise comparison instead — otherwise one NaN that reaches a transform
// reports as changed on every frame, forever.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    I32(i32),
    U32(u32),
    F32(f32),
    String(String),
    Vec3(glam::Vec3),
    Quat(glam::Quat),
    /// A reference to another entity, by its persistent identity. There is
    /// deliberately no variant for a raw `orrin_ecs::Entity` — see
    /// [`EntityId`](crate::EntityId).
    Entity(crate::EntityId),
    /// A reference to an asset, by its persistent identity — never by path.
    Asset(crate::AssetId),
    /// Fields in declaration order — the order an inspector should draw them.
    /// Canonical (sorted) ordering is the text writer's job, not the data's.
    Struct(Vec<(String, Value)>),
    Enum {
        variant: String,
        fields: Vec<(String, Value)>,
    },
    List(Vec<Value>),
}

/// One step of a [`FieldPath`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PathSegment {
    Field(String),
    Index(usize),
}

/// Where inside a component a value lives, e.g. `translation.x` or `points[2]`.
///
/// Shared on purpose between error reporting and (later) diffs and sync
/// operations: all three answer "which field", and one type means they cannot
/// drift apart.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FieldPath(Vec<PathSegment>);

/// A [`Value`] that did not match the type being read out of it.
///
/// `found` is a `Cow` because most failures report a type name known at compile
/// time, but an unrecognized enum variant has to name the variant it actually
/// read — and a diagnostic that says "unknown variant" without saying which one
/// is not worth printing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueError {
    pub path: FieldPath,
    pub expected: &'static str,
    pub found: Cow<'static, str>,
}

impl Value {
    /// The name used when reporting a type mismatch.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Bool(_) => "bool",
            Value::I32(_) => "i32",
            Value::U32(_) => "u32",
            Value::F32(_) => "f32",
            Value::String(_) => "string",
            Value::Vec3(_) => "vec3",
            Value::Quat(_) => "quat",
            Value::Entity(_) => "entity",
            Value::Asset(_) => "asset",
            Value::Struct(_) => "struct",
            Value::Enum { .. } => "enum",
            Value::List(_) => "list",
        }
    }

    /// Build a [`Value::Struct`] from a fixed set of named fields.
    pub fn strukt<const N: usize>(fields: [(&str, Value); N]) -> Value {
        Value::Struct(fields.map(|(name, value)| (name.to_owned(), value)).into())
    }

    /// Build a [`Value::Enum`] from a variant name and its payload.
    pub fn enumeration<const N: usize>(variant: &str, fields: [(&str, Value); N]) -> Value {
        Value::Enum {
            variant: variant.to_owned(),
            fields: fields.map(|(name, value)| (name.to_owned(), value)).into(),
        }
    }

    /// The variant name, for enum values only.
    pub fn variant(&self) -> Option<&str> {
        match self {
            Value::Enum { variant, .. } => Some(variant),
            _ => None,
        }
    }

    /// Look up a named field of a struct *or* of an enum variant, so a variant's
    /// payload is read exactly like a plain struct's.
    pub fn field(&self, name: &str) -> Option<&Value> {
        let fields = match self {
            Value::Struct(fields) => fields,
            Value::Enum { fields, .. } => fields,
            _ => return None,
        };
        fields.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }

    /// [`field`](Self::field) for a caller that is about to overwrite what it
    /// finds. Cannot add a field: a name that is not already there is a value
    /// the type would not read back, so [`diff::apply`](crate::apply) reports it
    /// rather than inventing a slot.
    pub fn field_mut(&mut self, name: &str) -> Option<&mut Value> {
        let fields = match self {
            Value::Struct(fields) => fields,
            Value::Enum { fields, .. } => fields,
            _ => return None,
        };
        fields.iter_mut().find(|(n, _)| n == name).map(|(_, v)| v)
    }
}

impl FieldPath {
    pub const fn empty() -> Self {
        Self(Vec::new())
    }

    pub fn field(name: &str) -> Self {
        Self(vec![PathSegment::Field(name.to_owned())])
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn segments(&self) -> &[PathSegment] {
        &self.0
    }

    /// Private so that only [`ValueError::at_field`] / [`ValueError::at_index`]
    /// can *prepend*: an error's path is built while the error unwinds, never
    /// threaded down into the read itself.
    fn push_front(&mut self, segment: PathSegment) {
        self.0.insert(0, segment);
    }

    /// Descend one level. The diff walks top-down, so it appends where an error
    /// prepends; the two directions must not be confused, hence separate names
    /// rather than one `push`.
    pub fn push(&mut self, segment: PathSegment) {
        self.0.push(segment);
    }

    /// Ascend one level, undoing the matching [`push`](Self::push).
    pub fn pop(&mut self) {
        self.0.pop();
    }
}

impl fmt::Display for FieldPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, segment) in self.0.iter().enumerate() {
            match segment {
                PathSegment::Field(name) => {
                    if i > 0 {
                        f.write_str(".")?;
                    }
                    f.write_str(name)?;
                }
                PathSegment::Index(index) => write!(f, "[{index}]")?,
            }
        }
        Ok(())
    }
}

/// The inverse of `Display`: `translation.x`, `points[2]`, or the empty string
/// for the whole value.
impl FromStr for FieldPath {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let mut path = FieldPath::empty();
        let mut rest = text;
        while !rest.is_empty() {
            if let Some(after) = rest.strip_prefix('[') {
                let (index, tail) = after
                    .split_once(']')
                    .ok_or_else(|| format!("`{text}` has an unclosed `[`"))?;
                let index = index
                    .parse()
                    .map_err(|_| format!("`{index}` in `{text}` is not an index"))?;
                path.push(PathSegment::Index(index));
                rest = tail;
            } else {
                // A field follows the start of the path or a `.`; directly after
                // an index it would be `[1]y`, which nothing prints.
                let after = if path.is_empty() {
                    rest
                } else {
                    rest.strip_prefix('.')
                        .ok_or_else(|| format!("`{text}` is missing a `.` before a field"))?
                };
                let end = after.find(['.', '[']).unwrap_or(after.len());
                let name = &after[..end];
                if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    return Err(format!("`{text}` names a field that cannot exist"));
                }
                path.push(PathSegment::Field(name.to_owned()));
                rest = &after[end..];
            }
        }
        Ok(path)
    }
}

impl ValueError {
    /// The leaf case: the value is the wrong shape. The path starts empty and
    /// is filled in by the callers this error passes through on its way out.
    pub fn mismatch(expected: &'static str, found: &Value) -> Self {
        Self {
            path: FieldPath::empty(),
            expected,
            found: Cow::Borrowed(found.type_name()),
        }
    }

    /// A value of the right *shape* that the type still refuses — a broken
    /// invariant rather than a type error. A scene file is untrusted input, so
    /// a type whose constructor establishes something its fields don't has to
    /// be able to say no.
    pub fn invalid(expected: &'static str, found: impl Into<Cow<'static, str>>) -> Self {
        Self {
            path: FieldPath::empty(),
            expected,
            found: found.into(),
        }
    }

    /// An enum value naming a variant the type no longer has. `expected` should
    /// list the variants that exist, since the usual cause is a variant renamed
    /// out from under a saved scene.
    pub fn unknown_variant(expected: &'static str, found: &str) -> Self {
        Self::invalid(expected, format!("`{found}`"))
    }

    /// A field that isn't there at all. Note this *already* names the field, so
    /// whoever detects the absence must not also call [`at_field`](Self::at_field)
    /// for that same level.
    pub fn missing(field: &str) -> Self {
        Self {
            path: FieldPath::field(field),
            expected: "a value",
            found: Cow::Borrowed("nothing"),
        }
    }

    pub fn at_field(mut self, name: &str) -> Self {
        self.path.push_front(PathSegment::Field(name.to_owned()));
        self
    }

    pub fn at_index(mut self, index: usize) -> Self {
        self.path.push_front(PathSegment::Index(index));
        self
    }
}

impl fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.path.is_empty() {
            write!(f, "expected {}, found {}", self.expected, self.found)
        } else {
            write!(
                f,
                "field `{}`: expected {}, found {}",
                self.path, self.expected, self.found
            )
        }
    }
}

impl std::error::Error for ValueError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_lookup_works_on_structs_and_enum_variants() {
        let s = Value::strukt([("speed", Value::F32(2.0))]);
        assert_eq!(s.field("speed"), Some(&Value::F32(2.0)));
        assert_eq!(s.field("missing"), None);

        let e = Value::Enum {
            variant: "Point".to_owned(),
            fields: vec![("range".to_owned(), Value::F32(10.0))],
        };
        assert_eq!(e.field("range"), Some(&Value::F32(10.0)));

        assert_eq!(Value::Bool(true).field("anything"), None);
    }

    #[test]
    fn paths_read_outermost_first() {
        let err = ValueError::mismatch("f32", &Value::Bool(true))
            .at_field("x")
            .at_field("translation");
        assert_eq!(err.path.to_string(), "translation.x");
        assert_eq!(
            err.to_string(),
            "field `translation.x`: expected f32, found bool"
        );
    }

    /// A prefab override stores its path as text, so the printed form has to
    /// read back as the path it was printed from.
    #[test]
    fn a_path_parses_back_from_what_it_prints() {
        for text in [
            "",
            "scale",
            "translation.x",
            "points[2]",
            "light.points[0].x",
            "[3]",
        ] {
            let path: FieldPath = text.parse().unwrap_or_else(|e| panic!("{text}: {e}"));
            assert_eq!(path.to_string(), text);
        }
        let path: FieldPath = "points[2].x".parse().unwrap();
        assert_eq!(
            path.segments(),
            [
                PathSegment::Field("points".to_owned()),
                PathSegment::Index(2),
                PathSegment::Field("x".to_owned()),
            ]
        );
    }

    #[test]
    fn a_malformed_path_is_refused() {
        for text in [".x", "x.", "x..y", "x[", "x[a]", "x[1]y", "x y"] {
            assert!(text.parse::<FieldPath>().is_err(), "`{text}` parsed");
        }
    }

    #[test]
    fn indices_print_without_a_leading_dot() {
        let err = ValueError::mismatch("f32", &Value::Bool(true))
            .at_index(2)
            .at_field("points");
        assert_eq!(err.path.to_string(), "points[2]");
    }
}
