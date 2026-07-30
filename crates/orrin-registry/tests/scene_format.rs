//! The scene format's central property: `parse(write(doc)) == doc`, for every
//! shape `Value` can take.

use glam::{Quat, Vec3};
use orrin_registry::{
    ComponentId, EntityId, SceneDocument, SceneEntity, Value, parse, write_document,
};

fn entity(components: Vec<(&str, Value)>) -> SceneEntity {
    SceneEntity {
        id: EntityId::new(),
        components: components
            .into_iter()
            .map(|(id, value)| (ComponentId::owned(id), value))
            .collect(),
    }
}

fn round_trip(document: SceneDocument) -> String {
    let mut text = String::new();
    write_document(&mut text, &document);

    let parsed = parse(&text).unwrap_or_else(|e| panic!("{e}\n---\n{text}"));

    // Compared through a second write rather than field-by-field: the writer
    // canonicalizes order, so two documents that produce the same bytes are the
    // same scene even if their component vectors were built in a different
    // order.
    let mut again = String::new();
    write_document(&mut again, &parsed);
    assert_eq!(text, again);

    text
}

#[test]
fn every_leaf_survives_the_round_trip() {
    let text = round_trip(SceneDocument {
        entities: vec![entity(vec![
            ("test.flag", Value::Bool(true)),
            ("test.count", Value::I32(-7)),
            ("test.big", Value::U32(4_000_000_000)),
            ("test.speed", Value::F32(1.5)),
            ("test.name", Value::String("a \"quoted\"\nname".to_owned())),
            ("test.position", Value::Vec3(Vec3::new(0.0, 1.5, -2.0))),
            ("test.rotation", Value::Quat(Quat::IDENTITY)),
            ("test.target", Value::Entity(EntityId::new())),
        ])],
    });

    assert!(text.starts_with("orrin-scene 1\n"));
}

#[test]
fn structs_enums_and_lists_nest() {
    round_trip(SceneDocument {
        entities: vec![entity(vec![
            (
                "test.collider",
                Value::strukt([
                    (
                        "shape",
                        Value::enumeration("Box", [("half_extents", Value::Vec3(Vec3::ONE))]),
                    ),
                    ("is_trigger", Value::Bool(false)),
                ]),
            ),
            ("test.marker", Value::enumeration("None", [])),
            ("test.empty_struct", Value::Struct(Vec::new())),
            ("test.empty_list", Value::List(Vec::new())),
            (
                "test.path",
                Value::List(vec![
                    Value::Vec3(Vec3::ZERO),
                    Value::Vec3(Vec3::X),
                    Value::Vec3(Vec3::Y),
                ]),
            ),
            (
                "test.nested_list",
                Value::List(vec![Value::List(vec![Value::I32(1), Value::I32(2)])]),
            ),
        ])],
    });
}

#[test]
fn non_finite_floats_survive() {
    let text = round_trip(SceneDocument {
        entities: vec![entity(vec![
            ("test.nan", Value::F32(f32::NAN)),
            ("test.inf", Value::F32(f32::INFINITY)),
            ("test.neg_inf", Value::F32(f32::NEG_INFINITY)),
        ])],
    });
    assert!(text.contains("test.nan = nan"));
    assert!(text.contains("test.neg_inf = -inf"));
}

#[test]
fn entities_are_written_in_id_order_whatever_order_they_arrived_in() {
    let a = EntityId::new();
    let b = EntityId::new();
    let (low, high) = if a < b { (a, b) } else { (b, a) };

    let mut forwards = String::new();
    write_document(
        &mut forwards,
        &SceneDocument {
            entities: vec![
                SceneEntity { id: low, components: Vec::new() },
                SceneEntity { id: high, components: Vec::new() },
            ],
        },
    );

    let mut backwards = String::new();
    write_document(
        &mut backwards,
        &SceneDocument {
            entities: vec![
                SceneEntity { id: high, components: Vec::new() },
                SceneEntity { id: low, components: Vec::new() },
            ],
        },
    );

    assert_eq!(forwards, backwards);
}

#[test]
fn an_empty_scene_is_just_a_header() {
    let mut text = String::new();
    write_document(&mut text, &SceneDocument::default());
    assert_eq!(text, "orrin-scene 1\n");
    assert_eq!(parse(&text).unwrap(), SceneDocument::default());
}

#[test]
fn a_missing_header_is_reported_on_line_one() {
    let err = parse("entity 00000000-0000-0000-0000-000000000001\n").unwrap_err();
    assert_eq!(err.line, 1);
    assert!(err.message.contains("orrin-scene"));
}

#[test]
fn an_unsupported_version_names_itself() {
    let err = parse("orrin-scene 99\n").unwrap_err();
    assert_eq!(err.to_string(), "line 1: scene format version 99 is not supported (this build reads 1)");
}

#[test]
fn a_debug_dump_is_refused_with_a_hint() {
    let err = parse("orrin-scene 1\n\nentity #3\n").unwrap_err();
    assert_eq!(err.line, 3);
    assert!(err.message.contains("debug dump"), "{}", err.message);
}

#[test]
fn syntax_errors_carry_their_line() {
    let base = "orrin-scene 1\n\nentity 00000000-0000-0000-0000-000000000001\n";

    let err = parse(&format!("{base}   name = 1\n")).unwrap_err();
    assert_eq!(err.line, 4);
    assert!(err.message.contains("multiple of two spaces"));

    let err = parse(&format!("{base}\tname = 1\n")).unwrap_err();
    assert!(err.message.contains("tabs"));

    let err = parse(&format!("{base}  name = \"unterminated\n")).unwrap_err();
    assert_eq!(err.line, 4);

    let err = parse(&format!("{base}  name = (1.0, 2.0)\n")).unwrap_err();
    assert!(err.message.contains("3 components"));

    let err = parse(&format!("{base}      deep = 1\n")).unwrap_err();
    assert!(err.message.contains("indent"));
}

#[test]
fn integers_read_back_into_whichever_width_the_field_wants() {
    use orrin_registry::Reflect;

    // The file has one integer syntax, so the parser cannot know the width. The
    // numeric leaves coerce instead of rejecting.
    let parsed = parse(&format!(
        "orrin-scene 1\n\nentity {}\n  test.n = 7\n",
        EntityId::new()
    ))
    .unwrap();
    let value = &parsed.entities[0].components[0].1;

    assert_eq!(i32::from_value(value), Ok(7));
    assert_eq!(u32::from_value(value), Ok(7));
    assert_eq!(f32::from_value(value), Ok(7.0));
}

#[test]
fn a_negative_integer_is_not_a_u32() {
    use orrin_registry::Reflect;

    let err = u32::from_value(&Value::I32(-1)).unwrap_err();
    assert_eq!(err.to_string(), "expected a u32, found -1");
}
