//! `#[derive(Reflect)]` against every shape it claims to support.
//!
//! Lives here rather than in `orrin-macros` because the generated code names
//! `::orrin_registry`, and testing it from the macro crate would need a
//! dependency cycle.

use glam::Vec3;
use orrin_registry::{Reflect, Value};

#[derive(Reflect, Debug, PartialEq)]
struct Health {
    current: f32,
    max: f32,
}

#[derive(Reflect, Debug, PartialEq)]
struct Label(String);

#[derive(Reflect, Debug, PartialEq)]
struct Marker;

#[derive(Reflect, Debug, PartialEq)]
struct Nested {
    health: Health,
    label: Label,
    waypoints: Vec<Vec3>,
}

#[derive(Reflect, Debug, PartialEq)]
enum Shape {
    Ball { radius: f32 },
    Cuboid { extents: Vec3, hollow: bool },
    None,
}

#[derive(Reflect, Debug, PartialEq)]
struct WithSkip {
    kept: f32,
    #[reflect(skip)]
    cache: Vec<f32>,
}

#[derive(Reflect, Debug, PartialEq)]
enum VariantWithSkip {
    Timed {
        seconds: f32,
        #[reflect(skip)]
        elapsed: f32,
    },
}

fn round_trip<T: Reflect + PartialEq + std::fmt::Debug>(value: T) -> Value {
    let encoded = value.to_value();
    assert_eq!(T::from_value(&encoded).unwrap(), value);
    encoded
}

#[test]
fn named_struct_fields_keep_declaration_order() {
    let encoded = round_trip(Health {
        current: 30.0,
        max: 100.0,
    });
    assert_eq!(
        encoded,
        Value::strukt([("current", Value::F32(30.0)), ("max", Value::F32(100.0))])
    );
}

#[test]
fn a_newtype_flattens_to_its_inner_value() {
    let encoded = round_trip(Label("hero".to_owned()));
    assert_eq!(encoded, Value::String("hero".to_owned()));
}

#[test]
fn a_unit_struct_is_an_empty_struct() {
    let encoded = round_trip(Marker);
    assert_eq!(encoded, Value::Struct(Vec::new()));
}

#[test]
fn nesting_composes() {
    let encoded = round_trip(Nested {
        health: Health {
            current: 1.0,
            max: 2.0,
        },
        label: Label("x".to_owned()),
        waypoints: vec![Vec3::ZERO, Vec3::ONE],
    });
    assert_eq!(encoded.field("label"), Some(&Value::String("x".to_owned())));
    assert_eq!(
        encoded.field("health").and_then(|h| h.field("max")),
        Some(&Value::F32(2.0))
    );
}

#[test]
fn enum_variants_carry_their_name_and_payload() {
    let encoded = round_trip(Shape::Cuboid {
        extents: Vec3::ONE,
        hollow: true,
    });
    assert_eq!(encoded.variant(), Some("Cuboid"));
    assert_eq!(encoded.field("hollow"), Some(&Value::Bool(true)));

    let unit = round_trip(Shape::None);
    assert_eq!(unit.variant(), Some("None"));
    assert_eq!(unit.field("anything"), None);

    round_trip(Shape::Ball { radius: 2.0 });
}

#[test]
fn an_unknown_variant_lists_the_ones_that_exist() {
    let stale = Value::enumeration("Capsule", [("radius", Value::F32(1.0))]);
    let err = Shape::from_value(&stale).unwrap_err();
    assert_eq!(
        err.to_string(),
        "expected one of: Ball, Cuboid, None, found `Capsule`"
    );
}

#[test]
fn a_non_enum_value_is_rejected_before_the_variant_lookup() {
    let err = Shape::from_value(&Value::F32(1.0)).unwrap_err();
    assert_eq!(err.expected, "enum");
    assert_eq!(err.found, "f32");
}

#[test]
fn skipped_fields_are_absent_and_come_back_defaulted() {
    let encoded = WithSkip {
        kept: 5.0,
        cache: vec![1.0, 2.0],
    }
    .to_value();
    assert_eq!(encoded, Value::strukt([("kept", Value::F32(5.0))]));

    let decoded = WithSkip::from_value(&encoded).unwrap();
    assert_eq!(decoded.kept, 5.0);
    assert!(decoded.cache.is_empty());
}

#[test]
fn skipped_variant_fields_behave_the_same() {
    let encoded = VariantWithSkip::Timed {
        seconds: 3.0,
        elapsed: 1.25,
    }
    .to_value();
    assert_eq!(encoded.field("elapsed"), None);

    let VariantWithSkip::Timed { seconds, elapsed } =
        VariantWithSkip::from_value(&encoded).unwrap();
    assert_eq!(seconds, 3.0);
    assert_eq!(elapsed, 0.0);
}

#[test]
fn a_missing_field_names_itself() {
    let err = Health::from_value(&Value::strukt([("current", Value::F32(1.0))])).unwrap_err();
    assert_eq!(
        err.to_string(),
        "field `max`: expected a value, found nothing"
    );
}

#[test]
fn a_nested_failure_reports_the_full_path() {
    let broken = Value::strukt([
        (
            "health",
            Value::strukt([("current", Value::F32(1.0)), ("max", Value::Bool(true))]),
        ),
        ("label", Value::String("x".to_owned())),
        ("waypoints", Value::List(Vec::new())),
    ]);
    let err = Nested::from_value(&broken).unwrap_err();
    assert_eq!(err.path.to_string(), "health.max");
}
