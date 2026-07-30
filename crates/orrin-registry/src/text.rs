//! Deterministic text dump of a world's registered components.
//!
//! Not the committed scene format — that arrives with save/load — but every
//! rule here is one the scene format inherits, so they are worth fixing now
//! while nothing depends on them: components sorted by id, fields sorted by
//! name, floats printed canonically. Identical worlds must produce identical
//! bytes, or a git diff of a scene is noise and a content hash is unstable.

use std::fmt::Write;

use orrin_ecs::{Entity, World};

use crate::registry::Registry;
use crate::value::Value;

const INDENT: &str = "  ";

/// Dump every live entity, in ascending slot order.
pub fn write_world(out: &mut String, registry: &Registry, world: &World) {
    for entity in world.entities() {
        write_entity(out, registry, world, entity);
    }
}

/// Dump one entity and all of its registered components.
///
/// Components are found by asking every registered type whether it is present,
/// rather than by enumerating the world's storages — which keeps the ECS free
/// of any type-erased iteration API. It costs one `has` call per registered
/// type per entity, which is nothing at the scale this runs.
pub fn write_entity(out: &mut String, registry: &Registry, world: &World, entity: Entity) {
    // Only the slot index: the generation is session-local bookkeeping, and
    // cross-session identity waits for a stable `EntityId`.
    let _ = writeln!(out, "entity {}", entity.index());

    let mut components: Vec<(&str, Value)> = registry
        .components()
        .filter_map(|c| (c.read)(world, entity).map(|value| (c.id.as_str(), value)))
        .collect();
    components.sort_by_key(|(id, _)| *id);

    for (id, value) in &components {
        write_named(out, id, value, 1);
    }
}

fn write_named(out: &mut String, name: &str, value: &Value, depth: usize) {
    for _ in 0..depth {
        out.push_str(INDENT);
    }
    out.push_str(name);

    match value {
        Value::Struct(fields) => {
            out.push('\n');
            write_fields(out, fields, depth + 1);
        }
        Value::Enum { variant, fields } => {
            let _ = writeln!(out, " = {variant}");
            write_fields(out, fields, depth + 1);
        }
        Value::List(items) if !items.is_empty() => {
            out.push('\n');
            for (index, item) in items.iter().enumerate() {
                write_named(out, &format!("[{index}]"), item, depth + 1);
            }
        }
        scalar => {
            out.push_str(" = ");
            write_scalar(out, scalar);
            out.push('\n');
        }
    }
}

fn write_fields(out: &mut String, fields: &[(String, Value)], depth: usize) {
    let mut sorted: Vec<&(String, Value)> = fields.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, value) in sorted {
        write_named(out, name, value, depth);
    }
}

fn write_scalar(out: &mut String, value: &Value) {
    match value {
        Value::Bool(v) => out.push_str(if *v { "true" } else { "false" }),
        Value::I32(v) => {
            let _ = write!(out, "{v}");
        }
        Value::U32(v) => {
            let _ = write!(out, "{v}");
        }
        Value::F32(v) => write_f32(out, *v),
        Value::String(v) => write_string(out, v),
        Value::Vec3(v) => {
            out.push('(');
            for (i, component) in [v.x, v.y, v.z].into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_f32(out, component);
            }
            out.push(')');
        }
        Value::Quat(v) => {
            // Written as the quaternion it is. The inspector converts to euler
            // for display, but the file must not: euler is lossy and
            // convention-dependent, so a save/load round-trip through it would
            // quietly perturb every rotation.
            out.push('(');
            for (i, component) in [v.x, v.y, v.z, v.w].into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_f32(out, component);
            }
            out.push(')');
        }
        Value::Entity(e) => {
            let _ = write!(out, "entity({}:{})", e.index(), e.generation());
        }
        Value::List(_) => out.push_str("[]"),
        // Routed to `write_fields` by `write_named` and never reached here.
        Value::Struct(_) | Value::Enum { .. } => out.push_str("{}"),
    }
}

/// `{:?}` rather than `{}`: both print the shortest representation that
/// round-trips, but `Debug` keeps the decimal point (`1.0`, not `1`), which
/// makes the type visible and a re-parse unambiguous.
fn write_f32(out: &mut String, value: f32) {
    if value.is_nan() {
        // Canonical spelling for every NaN payload. Written rather than
        // rejected on purpose: a NaN in a transform is a bug worth seeing on
        // the line that produced it, not a save failure that hides which entity
        // was responsible.
        out.push_str("nan");
        return;
    }
    if value.is_infinite() {
        out.push_str(if value > 0.0 { "inf" } else { "-inf" });
        return;
    }
    // `-0.0 == 0.0`, so two worlds that compare equal in every way would
    // otherwise differ by one sign bit on disk. Scaling to zero and negating
    // produces `-0.0` in ordinary use; this is not hypothetical.
    let value = if value == 0.0 { 0.0 } else { value };
    let _ = write!(out, "{value:?}");
}

/// Escapes `"`, `\`, and the three ASCII control characters that would break
/// the line structure. Everything else, including non-ASCII, is written through
/// as UTF-8.
fn write_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reflect::{Reflect, take};
    use crate::registry::ComponentId;
    use crate::value::ValueError;
    use glam::{Quat, Vec3};

    #[derive(Debug, Default)]
    struct Placement {
        translation: Vec3,
        rotation: Quat,
        scale: Vec3,
    }

    impl Reflect for Placement {
        fn to_value(&self) -> Value {
            // Deliberately not alphabetical: the writer is what sorts, so the
            // declaration order an inspector wants stays in the data.
            Value::strukt([
                ("translation", self.translation.to_value()),
                ("rotation", self.rotation.to_value()),
                ("scale", self.scale.to_value()),
            ])
        }

        fn from_value(value: &Value) -> Result<Self, ValueError> {
            Ok(Self {
                translation: take(value, "translation")?,
                rotation: take(value, "rotation")?,
                scale: take(value, "scale")?,
            })
        }
    }

    #[derive(Debug, Default)]
    struct Title(String);

    impl Reflect for Title {
        fn to_value(&self) -> Value {
            self.0.to_value()
        }

        fn from_value(value: &Value) -> Result<Self, ValueError> {
            String::from_value(value).map(Self)
        }
    }

    fn fixture() -> (Registry, World, Entity) {
        let mut registry = Registry::new();
        registry.register::<Placement>(ComponentId::new("test.placement"), "Placement");
        registry.register::<Title>(ComponentId::new("test.title"), "Title");
        registry.end_engine_registration();

        let mut world = World::new();
        let entity = world.spawn();
        world.insert(
            entity,
            Placement {
                translation: Vec3::new(0.0, 1.5, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
        );
        world.insert(entity, Title("Cube".to_owned()));

        (registry, world, entity)
    }

    #[test]
    fn the_dump_is_stable_and_sorted() {
        let (registry, world, entity) = fixture();
        let mut out = String::new();
        write_entity(&mut out, &registry, &world, entity);

        assert_eq!(
            out,
            "\
entity 1
  test.placement
    rotation = (0.0, 0.0, 0.0, 1.0)
    scale = (1.0, 1.0, 1.0)
    translation = (0.0, 1.5, 0.0)
  test.title = \"Cube\"
"
        );
    }

    #[test]
    fn an_entity_without_registered_components_still_writes_a_header() {
        let (registry, mut world, _) = fixture();
        let empty = world.spawn();
        let mut out = String::new();
        write_entity(&mut out, &registry, &world, empty);
        assert_eq!(out, "entity 2\n");
    }

    #[test]
    fn negative_zero_is_normalized() {
        let mut out = String::new();
        write_f32(&mut out, -0.0);
        assert_eq!(out, "0.0");
    }

    #[test]
    fn non_finite_floats_have_canonical_spellings() {
        let mut out = String::new();
        write_f32(&mut out, f32::NAN);
        write_f32(&mut out, -f32::NAN);
        write_f32(&mut out, f32::INFINITY);
        write_f32(&mut out, f32::NEG_INFINITY);
        assert_eq!(out, "nannaninf-inf");
    }

    #[test]
    fn strings_escape_what_would_break_the_line_structure() {
        let mut out = String::new();
        write_string(&mut out, "a\"b\\c\nd é");
        assert_eq!(out, r#""a\"b\\c\nd é""#);
    }

    #[test]
    fn enums_and_lists_nest() {
        let value = Value::Enum {
            variant: "Point".to_owned(),
            fields: vec![
                ("range".to_owned(), Value::F32(10.0)),
                ("color".to_owned(), Value::Vec3(Vec3::ONE)),
            ],
        };
        let mut out = String::new();
        write_named(&mut out, "light", &value, 0);
        assert_eq!(
            out,
            "\
light = Point
  color = (1.0, 1.0, 1.0)
  range = 10.0
"
        );

        let mut out = String::new();
        write_named(
            &mut out,
            "points",
            &Value::List(vec![Value::I32(1), Value::I32(2)]),
            0,
        );
        assert_eq!(out, "points\n  [0] = 1\n  [1] = 2\n");

        let mut out = String::new();
        write_named(&mut out, "points", &Value::List(Vec::new()), 0);
        assert_eq!(out, "points = []\n");
    }
}
