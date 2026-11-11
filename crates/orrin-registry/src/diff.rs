//! Comparing two component values, and replaying the difference onto a third.
//!
//! One mutation vocabulary, shared by every consumer that has to describe a
//! change rather than perform it: undo/redo records the changes an edit
//! produced, the inspector reports what the user dragged, prefab overrides are
//! the changes between an instance and its source, and collaboration sync
//! ships them over the wire. Architecture §4.4 asks for exactly one such
//! stream, on the grounds that undo and sync are then the same feature twice.
//!
//! A change is a *whole subtree at a path*, never an in-place mutation
//! instruction. That is what makes [`apply`] idempotent and order-independent
//! within one batch, which is the property a CRDT's last-write-wins rule
//! needs — an "add 3 to x" operation would not survive being delivered twice.

use crate::value::{FieldPath, PathSegment, Value, ValueError};

/// A new value for one place inside a component.
///
/// The path is empty when the whole component changed shape — a different enum
/// variant, a resized list, a field set that no longer matches. Descending into
/// those would report a rename as a delete plus an insert, and neither the
/// inspector nor a merge can do anything useful with that; replacing the
/// subtree wholesale is both smaller and correct.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldChange {
    pub path: FieldPath,
    pub value: Value,
}

/// Every change that would turn `old` into `new`, deepest path that still
/// describes the edit.
///
/// Equality of floats is **bitwise**, not `PartialEq`: a NaN that reaches a
/// transform compares unequal to itself, so a `==`-based diff would report that
/// field as changed on every frame for the rest of the session, and an undo
/// stack fed by it would grow without bound. `-0.0` and `0.0` are likewise
/// distinguished, matching the text writer, which normalizes them on the way to
/// disk but does not pretend they are the same value in memory.
pub fn diff(old: &Value, new: &Value) -> Vec<FieldChange> {
    let mut changes = Vec::new();
    walk(&mut FieldPath::empty(), old, new, &mut changes);
    changes
}

/// Overwrite each change's path in `target`.
///
/// Fails without touching `target` if any path does not resolve, so a batch
/// lands entirely or not at all — half an applied undo step is worse than a
/// refused one.
pub fn apply(target: &mut Value, changes: &[FieldChange]) -> Result<(), ValueError> {
    for change in changes {
        resolve(target, &change.path)?;
    }
    for change in changes {
        let slot = resolve(target, &change.path).expect("checked above");
        *slot = change.value.clone();
    }
    Ok(())
}

fn walk(path: &mut FieldPath, old: &Value, new: &Value, out: &mut Vec<FieldChange>) {
    let children: Option<Vec<(PathSegment, &Value, &Value)>> = match (old, new) {
        (Value::Struct(a), Value::Struct(b)) if same_fields(a, b) => Some(paired_fields(a, b)),
        (
            Value::Enum {
                variant: va,
                fields: a,
            },
            Value::Enum {
                variant: vb,
                fields: b,
            },
        ) if va == vb && same_fields(a, b) => Some(paired_fields(a, b)),
        (Value::List(a), Value::List(b)) if a.len() == b.len() => Some(
            a.iter()
                .zip(b)
                .enumerate()
                .map(|(i, (x, y))| (PathSegment::Index(i), x, y))
                .collect(),
        ),
        _ => None,
    };

    match children {
        Some(children) => {
            for (segment, a, b) in children {
                path.push(segment);
                walk(path, a, b, out);
                path.pop();
            }
        }
        None => {
            if !identical(old, new) {
                out.push(FieldChange {
                    path: path.clone(),
                    value: new.clone(),
                });
            }
        }
    }
}

fn same_fields(a: &[(String, Value)], b: &[(String, Value)]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|((x, _), (y, _))| x == y)
}

fn paired_fields<'v>(
    a: &'v [(String, Value)],
    b: &'v [(String, Value)],
) -> Vec<(PathSegment, &'v Value, &'v Value)> {
    a.iter()
        .zip(b)
        .map(|((name, x), (_, y))| (PathSegment::Field(name.clone()), x, y))
        .collect()
}

/// Bitwise for every float that reaches it; see [`diff`] for why.
fn identical(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::F32(x), Value::F32(y)) => x.to_bits() == y.to_bits(),
        (Value::Vec3(x), Value::Vec3(y)) => bits3(x.to_array()) == bits3(y.to_array()),
        (Value::Quat(x), Value::Quat(y)) => {
            x.to_array().map(f32::to_bits) == y.to_array().map(f32::to_bits)
        }
        (Value::Vec2(x), Value::Vec2(y)) => {
            x.to_array().map(f32::to_bits) == y.to_array().map(f32::to_bits)
        }
        (Value::Color(x), Value::Color(y)) => x.map(f32::to_bits) == y.map(f32::to_bits),
        (Value::Struct(x), Value::Struct(y)) => same_fields(x, y) && fields_identical(x, y),
        (
            Value::Enum {
                variant: vx,
                fields: x,
            },
            Value::Enum {
                variant: vy,
                fields: y,
            },
        ) => vx == vy && same_fields(x, y) && fields_identical(x, y),
        (Value::List(x), Value::List(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| identical(a, b))
        }
        _ => a == b,
    }
}

fn bits3(v: [f32; 3]) -> [u32; 3] {
    v.map(f32::to_bits)
}

fn fields_identical(a: &[(String, Value)], b: &[(String, Value)]) -> bool {
    a.iter().zip(b).all(|((_, x), (_, y))| identical(x, y))
}

/// Walk `path` down `value`, naming the segment that did not exist.
fn resolve<'v>(value: &'v mut Value, path: &FieldPath) -> Result<&'v mut Value, ValueError> {
    let mut cursor = value;
    for (depth, segment) in path.segments().iter().enumerate() {
        let found = cursor.type_name();
        cursor = match segment {
            // `missing` supplies its own segment, so only the levels *above* it
            // get re-attached.
            PathSegment::Field(name) => cursor
                .field_mut(name)
                .ok_or_else(|| prefix(ValueError::missing(name), path, depth))?,
            PathSegment::Index(index) => match cursor {
                Value::List(items) => {
                    let len = items.len();
                    items.get_mut(*index).ok_or_else(|| {
                        prefix(
                            ValueError::invalid("an index in range", format!("a list of {len}"))
                                .at_index(*index),
                            path,
                            depth,
                        )
                    })?
                }
                _ => {
                    return Err(prefix(
                        ValueError::invalid("list", found).at_index(*index),
                        path,
                        depth,
                    ));
                }
            },
        };
    }
    Ok(cursor)
}

/// Re-attach the segments walked before the failure, so the error names the
/// full path rather than the leaf that happened to be missing.
fn prefix(mut error: ValueError, path: &FieldPath, depth: usize) -> ValueError {
    for segment in path.segments()[..depth].iter().rev() {
        error = match segment {
            PathSegment::Field(name) => error.at_field(name),
            PathSegment::Index(index) => error.at_index(*index),
        };
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn placement(x: f32, scale: f32) -> Value {
        Value::strukt([
            ("translation", Value::Vec3(Vec3::new(x, 0.0, 0.0))),
            ("scale", Value::F32(scale)),
        ])
    }

    fn change(path: &[PathSegment], value: Value) -> FieldChange {
        let mut p = FieldPath::empty();
        for segment in path {
            p.push(segment.clone());
        }
        FieldChange { path: p, value }
    }

    fn field(name: &str) -> PathSegment {
        PathSegment::Field(name.to_owned())
    }

    #[test]
    fn an_unchanged_value_produces_no_changes() {
        assert_eq!(diff(&placement(1.0, 2.0), &placement(1.0, 2.0)), Vec::new());
    }

    #[test]
    fn a_changed_leaf_is_reported_at_its_own_path() {
        let changes = diff(&placement(1.0, 2.0), &placement(1.0, 3.0));
        assert_eq!(changes, vec![change(&[field("scale")], Value::F32(3.0))]);
    }

    #[test]
    fn a_vec3_is_one_change_rather_than_three() {
        let changes = diff(&placement(1.0, 2.0), &placement(4.0, 2.0));
        assert_eq!(
            changes,
            vec![change(
                &[field("translation")],
                Value::Vec3(Vec3::new(4.0, 0.0, 0.0))
            )]
        );
    }

    /// The appended leaves compare by bits like every other float, so a NaN
    /// tint is not an edit and a moved one is a single change.
    #[test]
    fn a_vec2_and_a_colour_compare_bitwise() {
        let tint = |alpha| Value::Color([f32::NAN, 0.0, 1.0, alpha]);
        assert_eq!(diff(&tint(1.0), &tint(1.0)), Vec::new());
        assert_eq!(diff(&tint(1.0), &tint(0.5)).len(), 1);

        let offset = |x| Value::Vec2(glam::Vec2::new(x, f32::NAN));
        assert_eq!(diff(&offset(1.0), &offset(1.0)), Vec::new());
        assert_eq!(diff(&offset(0.0), &offset(-0.0)).len(), 1);
    }

    #[test]
    fn nested_paths_read_outermost_first() {
        let old = Value::strukt([("inner", placement(1.0, 2.0))]);
        let new = Value::strukt([("inner", placement(1.0, 9.0))]);
        let changes = diff(&old, &new);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].path.to_string(), "inner.scale");
    }

    /// The reason [`diff`] compares bits rather than using `PartialEq`: a NaN in
    /// a transform is not equal to itself, so a `==` diff would report the field
    /// as edited every frame forever and fill an undo stack with nothing.
    #[test]
    fn a_nan_that_did_not_move_is_not_a_change() {
        let nan = Value::F32(f32::NAN);
        assert_eq!(diff(&nan, &nan.clone()), Vec::new());
    }

    /// The other half: `-0.0 == 0.0` in Rust, but flipping a sign bit is a real
    /// edit and the two print differently everywhere the value is inspected.
    #[test]
    fn a_sign_flip_on_zero_is_a_change() {
        let changes = diff(&Value::F32(0.0), &Value::F32(-0.0));
        assert_eq!(changes, vec![change(&[], Value::F32(-0.0))]);
    }

    #[test]
    fn a_different_enum_variant_replaces_the_whole_value() {
        let old = Value::enumeration("Point", [("range", Value::F32(10.0))]);
        let new = Value::enumeration("Spot", [("range", Value::F32(10.0))]);
        assert_eq!(diff(&old, &new), vec![change(&[], new)]);
    }

    #[test]
    fn the_same_variant_diffs_field_by_field() {
        let old = Value::enumeration("Point", [("range", Value::F32(10.0))]);
        let new = Value::enumeration("Point", [("range", Value::F32(12.0))]);
        let changes = diff(&old, &new);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].path.to_string(), "range");
    }

    #[test]
    fn a_list_of_the_same_length_diffs_per_index() {
        let old = Value::List(vec![Value::I32(1), Value::I32(2)]);
        let new = Value::List(vec![Value::I32(1), Value::I32(7)]);
        let changes = diff(&old, &new);
        assert_eq!(
            changes,
            vec![change(&[PathSegment::Index(1)], Value::I32(7))]
        );
    }

    /// A resized list is replaced whole. Matching elements across an insertion is
    /// rename detection, which needs identity the elements do not carry — and
    /// getting it wrong reports one insert as a rewrite of every element after it.
    #[test]
    fn a_resized_list_replaces_the_whole_list() {
        let old = Value::List(vec![Value::I32(1)]);
        let new = Value::List(vec![Value::I32(1), Value::I32(2)]);
        assert_eq!(diff(&old, &new), vec![change(&[], new)]);
    }

    #[test]
    fn a_value_that_changed_type_replaces_itself() {
        let changes = diff(&Value::F32(1.0), &Value::Bool(true));
        assert_eq!(changes, vec![change(&[], Value::Bool(true))]);
    }

    /// A component that grew a field between two saves has no field-wise diff at
    /// all; the shapes are not comparable and the newer one wins whole.
    #[test]
    fn a_different_field_set_replaces_the_whole_struct() {
        let old = Value::strukt([("a", Value::I32(1))]);
        let new = Value::strukt([("a", Value::I32(1)), ("b", Value::I32(2))]);
        assert_eq!(diff(&old, &new), vec![change(&[], new)]);
    }

    #[test]
    fn applying_a_diff_reproduces_the_new_value() {
        let old = placement(1.0, 2.0);
        let new = placement(4.0, 9.0);
        let mut target = old.clone();
        apply(&mut target, &diff(&old, &new)).unwrap();
        assert_eq!(target, new);
    }

    #[test]
    fn applying_the_same_batch_twice_lands_in_the_same_place() {
        let old = placement(1.0, 2.0);
        let new = placement(4.0, 9.0);
        let changes = diff(&old, &new);
        let mut target = old.clone();
        apply(&mut target, &changes).unwrap();
        apply(&mut target, &changes).unwrap();
        assert_eq!(target, new);
    }

    #[test]
    fn applying_into_a_list_and_an_enum_payload_works() {
        let mut target = Value::strukt([(
            "light",
            Value::enumeration("Point", [("points", Value::List(vec![Value::I32(0)]))]),
        )]);
        let changes = vec![change(
            &[field("light"), field("points"), PathSegment::Index(0)],
            Value::I32(5),
        )];
        apply(&mut target, &changes).unwrap();
        assert_eq!(
            target
                .field("light")
                .and_then(|l| l.field("points"))
                .cloned(),
            Some(Value::List(vec![Value::I32(5)]))
        );
    }

    #[test]
    fn an_unresolvable_path_names_it_and_changes_nothing() {
        let mut target = placement(1.0, 2.0);
        let changes = vec![
            change(&[field("scale")], Value::F32(8.0)),
            change(&[field("translation"), field("gone")], Value::F32(1.0)),
        ];
        let err = apply(&mut target, &changes).unwrap_err();
        assert_eq!(
            err.to_string(),
            "field `translation.gone`: expected a value, found nothing"
        );
        assert_eq!(target, placement(1.0, 2.0));
    }

    #[test]
    fn an_index_past_the_end_names_the_full_path() {
        let mut target = Value::strukt([("points", Value::List(vec![Value::I32(0)]))]);
        let changes = vec![change(
            &[field("points"), PathSegment::Index(3)],
            Value::I32(1),
        )];
        let err = apply(&mut target, &changes).unwrap_err();
        assert_eq!(err.path.to_string(), "points[3]");
    }
}
