//! The determinism tripwire for the scene format (#41).
//!
//! Identical documents must produce byte-identical files on every machine, or
//! a scene saved on one teammate's Mac shows up as a whole-file diff on
//! another's Linux box. Nothing in the writer is platform-specific today; this
//! is what keeps it that way. CI runs it on x86-64 Linux, x86-64 Windows and
//! arm64 macOS against the one checked-in file, so any two of them disagreeing
//! fails at least one.
//!
//! The reference document is built to be awkward: entities, components and
//! fields all arrive out of order, and the floats are the ones where a
//! printing algorithm shows its hand — values with no short decimal form,
//! subnormals, the extremes, `-0.0`, and the non-finite spellings.
//!
//! **What a failure means.** The written bytes changed. Legitimate when the
//! format changed on purpose, in which case `FORMAT_VERSION` moved too and a
//! migration exists. Otherwise every scene in every project is about to diff.
//! Read the diff before regenerating.
//!
//! Regenerate with `ORRIN_UPDATE_GOLDEN=1 cargo test -p orrin-registry --test
//! golden_scene`, and put the diff in the commit.

use std::fs;
use std::path::PathBuf;

use glam::{Quat, Vec3};
use orrin_registry::{
    AssetId, ComponentId, EntityId, SceneDocument, SceneEntity, Value, parse, write_document,
};

fn id(n: u32) -> EntityId {
    format!("00000000-0000-4000-8000-{n:012}").parse().unwrap()
}

fn reference() -> SceneDocument {
    let component = |id: &str, value: Value| (ComponentId::owned(id), value);
    let door: AssetId = "6f1c0d2e-8a4b-4c3d-9e5f-0a1b2c3d4e5f".parse().unwrap();

    SceneDocument {
        entities: vec![
            SceneEntity {
                id: id(3),
                components: vec![
                    component("orrin.parent", Value::Entity(id(1))),
                    component("orrin.name", Value::String("Handle".to_owned())),
                    component(
                        "orrin.transform",
                        Value::strukt([
                            ("translation", Value::Vec3(Vec3::new(0.1, -0.0, 1.0 / 3.0))),
                            ("scale", Value::Vec3(Vec3::splat(1.0))),
                            (
                                "rotation",
                                Value::Quat(Quat::from_xyzw(0.0, 0.149_438_14, 0.0, 0.988_771_1)),
                            ),
                        ]),
                    ),
                ],
            },
            SceneEntity {
                id: id(1),
                components: vec![
                    component(
                        "test.floats",
                        Value::List(
                            [
                                std::f32::consts::PI,
                                1e-7,
                                1e16,
                                123_456.79,
                                f32::MAX,
                                f32::MIN_POSITIVE,
                                f32::from_bits(1),
                                -2.5,
                                f32::NAN,
                                f32::NEG_INFINITY,
                            ]
                            .map(Value::F32)
                            .into(),
                        ),
                    ),
                    component(
                        "orrin.name",
                        Value::String("Door \"North\"\tété".to_owned()),
                    ),
                    component("test.asset", Value::Asset(door)),
                    component(
                        "orrin.collider",
                        Value::strukt([
                            (
                                "shape",
                                Value::enumeration(
                                    "Box",
                                    [("half_extents", Value::Vec3(Vec3::new(0.5, 1.0, 0.05)))],
                                ),
                            ),
                            ("is_trigger", Value::Bool(false)),
                        ]),
                    ),
                    component(
                        "test.ints",
                        Value::List(vec![Value::I32(i32::MIN), Value::U32(u32::MAX)]),
                    ),
                ],
            },
            SceneEntity {
                id: id(2),
                components: vec![
                    component("test.empty_list", Value::List(Vec::new())),
                    component("test.empty_struct", Value::Struct(Vec::new())),
                    component("test.empty_string", Value::String(String::new())),
                    component("test.unit_variant", Value::enumeration("None", [])),
                    component("orrin.name", Value::I32(4)),
                ],
            },
        ],
    }
}

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/reference.fscene")
}

#[test]
fn the_reference_scene_writes_the_checked_in_bytes() {
    let mut actual = String::new();
    write_document(&mut actual, &reference());
    let path = golden_path();

    if std::env::var_os("ORRIN_UPDATE_GOLDEN").is_some() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, &actual).unwrap();
        return;
    }

    let expected = fs::read_to_string(&path).unwrap_or_default();
    assert_eq!(
        actual,
        expected,
        "the reference scene no longer writes the bytes in {}.\n\
         If the format changed on purpose, bump FORMAT_VERSION, add the migration, \
         and regenerate with ORRIN_UPDATE_GOLDEN=1. If it did not, saving is no \
         longer deterministic on this platform.",
        path.display(),
    );
}

/// The reader's half: parsing the file and writing it again changes nothing,
/// so float parsing agrees with float printing on this platform too.
#[test]
fn the_checked_in_bytes_survive_a_read_and_rewrite() {
    let expected = fs::read_to_string(golden_path()).unwrap_or_default();
    let document = parse(&expected).unwrap_or_else(|e| panic!("{e}"));
    let mut again = String::new();
    write_document(&mut again, &document);
    assert_eq!(again, expected);
}
