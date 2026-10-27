//! The scene format's central property: `parse(write(doc)) == doc`, for every
//! shape `Value` can take.

use glam::{Quat, Vec3};
use orrin_registry::{
    AssetId, ComponentId, EntityId, FORMAT_VERSION, MIN_READER_VERSION, SceneDocument, SceneEntity,
    Value, parse, parse_value_document, write_document, write_value_document,
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
            ("test.prefab", Value::Asset(AssetId::new())),
        ])],
    });

    assert!(text.starts_with("orrin-scene 3 min-reader 3\n"));
}

/// An asset reference has its own spelling rather than being a string that
/// happens to hold a UUID, so a reader — a merge tool, the sync protocol — can
/// tell it from a name without knowing the component.
#[test]
fn an_asset_reference_is_written_as_one() {
    let id: AssetId = "6f1c0d2e-8a4b-4c3d-9e5f-0a1b2c3d4e5f".parse().unwrap();
    let text = round_trip(SceneDocument {
        entities: vec![entity(vec![("test.prefab", Value::Asset(id))])],
    });
    assert!(
        text.contains("test.prefab = asset(6f1c0d2e-8a4b-4c3d-9e5f-0a1b2c3d4e5f)"),
        "{text}"
    );
}

#[test]
fn a_malformed_asset_reference_names_itself() {
    let base = "orrin-scene 2 min-reader 2\n\nentity 00000000-0000-0000-0000-000000000001\n";
    let err = parse(&format!("{base}  test.prefab = asset(door)\n")).unwrap_err();
    assert_eq!(err.line, 4);
    assert_eq!(err.message, "`door` is not an asset id");
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
                SceneEntity {
                    id: low,
                    components: Vec::new(),
                },
                SceneEntity {
                    id: high,
                    components: Vec::new(),
                },
            ],
        },
    );

    let mut backwards = String::new();
    write_document(
        &mut backwards,
        &SceneDocument {
            entities: vec![
                SceneEntity {
                    id: high,
                    components: Vec::new(),
                },
                SceneEntity {
                    id: low,
                    components: Vec::new(),
                },
            ],
        },
    );

    assert_eq!(forwards, backwards);
}

#[test]
fn an_empty_scene_is_just_a_header() {
    let mut text = String::new();
    write_document(&mut text, &SceneDocument::default());
    assert_eq!(text, "orrin-scene 3 min-reader 3\n");
    assert_eq!(parse(&text).unwrap(), SceneDocument::default());
}

#[test]
fn a_missing_header_is_reported_on_line_one() {
    let err = parse("entity 00000000-0000-0000-0000-000000000001\n").unwrap_err();
    assert_eq!(err.line, 1);
    assert!(err.message.contains("orrin-scene"));
}

#[test]
fn a_file_needing_a_newer_reader_names_both_versions() {
    let err = parse("orrin-scene 99 min-reader 98\n").unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 1: scene format version 99 needs a reader of version 98 or later \
         (this build reads up to 3)"
    );
}

/// The point of the second number: a format change an older reader can
/// tolerate bumps the version without locking that reader out.
#[test]
fn a_newer_file_an_older_reader_may_read_is_read() {
    let text = format!(
        "orrin-scene {} min-reader {}\n\nentity 00000000-0000-0000-0000-000000000001\n  test.n = 1\n",
        FORMAT_VERSION + 1,
        MIN_READER_VERSION,
    );
    let document = parse(&text).unwrap();
    assert_eq!(document.entities.len(), 1);
}

#[test]
fn a_header_must_name_its_minimum_reader() {
    let err = parse("orrin-scene 2\n").unwrap_err();
    assert_eq!(err.line, 1);
    assert!(err.message.contains("min-reader"), "{}", err.message);
}

#[test]
fn a_minimum_reader_newer_than_the_file_is_malformed() {
    let err = parse("orrin-scene 2 min-reader 3\n").unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 1: a file cannot need a reader newer than its own version (2 < 3)"
    );
}

#[test]
fn version_zero_does_not_exist() {
    let err = parse("orrin-scene 0 min-reader 0\n").unwrap_err();
    assert_eq!(err.line, 1);
    assert!(err.message.contains("version 0"), "{}", err.message);
}

/// Version 1 kept the name among the components; reading one goes through the
/// migration and comes out as the document a version 2 file would have given.
#[test]
fn a_version_one_file_is_migrated_on_read() {
    let v1 = r#"orrin-scene 1

entity 00000000-0000-0000-0000-000000000001
  orrin.name = "Door \"A\""
  orrin.tag = 7
"#;
    let document = parse(v1).unwrap();
    let mut text = String::new();
    write_document(&mut text, &document);
    assert_eq!(
        text,
        r#"orrin-scene 3 min-reader 3

entity 00000000-0000-0000-0000-000000000001 "Door \"A\""
  orrin.tag = 7
"#
    );
}

/// An error in a migrated file must still point at the line its author sees.
#[test]
fn a_migrated_file_reports_errors_on_its_own_lines() {
    let v1 = r#"orrin-scene 1

entity 00000000-0000-0000-0000-000000000001
  orrin.name = "Door"
  orrin.tag = (1.0, 2.0)
"#;
    let err = parse(v1).unwrap_err();
    assert_eq!(err.line, 5);
}

/// The name rides on the header line, which is the line git shows as a hunk's
/// context — so a diff says which entity changed, not just that a UUID did.
#[test]
fn an_entity_header_carries_its_name() {
    let id: EntityId = "00000000-0000-0000-0000-000000000001".parse().unwrap();
    let name = (
        ComponentId::owned("orrin.name"),
        Value::String("Door \"A\"".to_owned()),
    );
    let text = round_trip(SceneDocument {
        entities: vec![SceneEntity {
            id,
            components: vec![
                (ComponentId::owned("orrin.tag"), Value::I32(7)),
                name.clone(),
            ],
        }],
    });
    assert_eq!(
        text,
        r#"orrin-scene 3 min-reader 3

entity 00000000-0000-0000-0000-000000000001 "Door \"A\""
  orrin.tag = 7
"#
    );
    assert!(parse(&text).unwrap().entities[0].components.contains(&name));
}

#[test]
fn a_name_that_is_not_a_string_stays_a_component() {
    let text = round_trip(SceneDocument {
        entities: vec![entity(vec![("orrin.name", Value::I32(3))])],
    });
    assert!(text.contains("\n  orrin.name = 3\n"), "{text}");
}

#[test]
fn a_header_name_must_be_quoted() {
    let err =
        parse("orrin-scene 2 min-reader 2\n\nentity 00000000-0000-0000-0000-000000000001 Door\n")
            .unwrap_err();
    assert_eq!(err.line, 3);
    assert!(err.message.contains("quoted"), "{}", err.message);
}

#[test]
fn an_entity_cannot_be_named_twice() {
    let err = parse(
        "orrin-scene 2 min-reader 2\n\nentity 00000000-0000-0000-0000-000000000001 \"A\"\n  orrin.name = \"B\"\n",
    )
    .unwrap_err();
    assert_eq!(err.line, 4);
    assert!(err.message.contains("named"), "{}", err.message);
}

#[test]
fn a_debug_dump_is_refused_with_a_hint() {
    let err = parse("orrin-scene 2 min-reader 2\n\nentity #3\n").unwrap_err();
    assert_eq!(err.line, 3);
    assert!(err.message.contains("debug dump"), "{}", err.message);
}

#[test]
fn syntax_errors_carry_their_line() {
    let base = "orrin-scene 2 min-reader 2\n\nentity 00000000-0000-0000-0000-000000000001\n";

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
        "orrin-scene 2 min-reader 2\n\nentity {}\n  test.n = 7\n",
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

/// A material is one value under a header of its own: the scene grammar at the
/// left margin, fields sorted, read back as written.
#[test]
fn a_value_document_round_trips_under_its_own_header() {
    let fields = vec![
        ("roughness".to_owned(), Value::F32(0.25)),
        (
            "base_color".to_owned(),
            Value::Vec3(Vec3::new(1.0, 0.5, 0.0)),
        ),
        (
            "blend".to_owned(),
            Value::Enum {
                variant: "Masked".to_owned(),
                fields: Vec::new(),
            },
        ),
        (
            "textures".to_owned(),
            Value::strukt([(
                "albedo",
                Value::Asset("00000000-0000-4000-8000-000000000007".parse().unwrap()),
            )]),
        ),
    ];
    let mut text = String::new();
    write_value_document(&mut text, "orrin-material", 1, 1, &fields);
    assert_eq!(
        text,
        "orrin-material 1 min-reader 1\n\nbase_color = (1.0, 0.5, 0.0)\nblend = Masked\nroughness = 0.25\ntextures\n  albedo = asset(00000000-0000-4000-8000-000000000007)\n"
    );
    let mut sorted = fields.clone();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        parse_value_document(&text, "orrin-material", 1).unwrap(),
        sorted
    );
}

#[test]
fn a_value_document_of_another_kind_or_a_newer_reader_is_refused_by_name() {
    let err =
        parse_value_document("orrin-scene 3 min-reader 3\n", "orrin-material", 1).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 1: expected an `orrin-material 1 min-reader 1` header"
    );
    let err =
        parse_value_document("orrin-material 4 min-reader 2\n", "orrin-material", 1).unwrap_err();
    assert_eq!(
        err.to_string(),
        "line 1: orrin-material version 4 needs a reader of version 2 or later (this build reads \
         up to 1)"
    );
    let err = parse_value_document(
        "orrin-material 1 min-reader 1\n\n  roughness = 1\n",
        "orrin-material",
        1,
    )
    .unwrap_err();
    assert_eq!(err.line, 3);
}
