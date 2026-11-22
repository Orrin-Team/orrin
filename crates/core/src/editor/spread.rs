//! One entity's edit, made to others: the inspector edits the primary
//! selection, and every other selected entity holding the component is given
//! the same change.
//!
//! The change is the [`diff`] between the primary's value before and after,
//! replayed onto each other value, so a field nobody touched keeps what that
//! entity had. The registry diffs a vector or a rotation as one leaf, which is
//! right for undo but too coarse here: dragging X would hand every entity the
//! primary's Y and Z too. So a vector is replayed lane by lane, and a rotation
//! in the Euler angles the inspector shows it in, each written only where the
//! primary's moved.
//!
//! [`diff`]: orrin_registry::diff

use glam::{EulerRot, Quat};
use orrin_registry::{FieldChange, FieldPath, PathSegment, Value};

/// `target` with the change from `before` to `after` made to it. A path that
/// does not resolve in `target` is left out rather than failing the rest.
pub fn spread(before: &Value, after: &Value, target: &Value) -> Value {
    let mut out = target.clone();
    for change in orrin_registry::diff(before, after) {
        let Some(old) = at(before, &change.path) else {
            continue;
        };
        let Some(theirs) = at(&out, &change.path) else {
            continue;
        };
        let value = merge(old, &change.value, theirs).unwrap_or(change.value);
        let _ = orrin_registry::apply(
            &mut out,
            &[FieldChange {
                path: change.path,
                value,
            }],
        );
    }
    out
}

fn at<'v>(value: &'v Value, path: &FieldPath) -> Option<&'v Value> {
    let mut cursor = value;
    for segment in path.segments() {
        cursor = match (segment, cursor) {
            (PathSegment::Field(name), Value::Struct(fields) | Value::Enum { fields, .. }) => {
                &fields.iter().find(|(field, _)| field == name)?.1
            }
            (PathSegment::Index(index), Value::List(items)) => items.get(*index)?,
            _ => return None,
        };
    }
    Some(cursor)
}

/// The lanes of `new` that differ from `old`, laid over `theirs`; `None` for
/// a value with no lanes, which is written whole.
fn merge(old: &Value, new: &Value, theirs: &Value) -> Option<Value> {
    fn lanes<const N: usize>(old: [f32; N], new: [f32; N], theirs: [f32; N]) -> [f32; N] {
        std::array::from_fn(|i| {
            if old[i].to_bits() == new[i].to_bits() {
                theirs[i]
            } else {
                new[i]
            }
        })
    }
    Some(match (old, new, theirs) {
        (Value::Vec2(o), Value::Vec2(n), Value::Vec2(t)) => {
            Value::Vec2(lanes(o.to_array(), n.to_array(), t.to_array()).into())
        }
        (Value::Vec3(o), Value::Vec3(n), Value::Vec3(t)) => {
            Value::Vec3(lanes(o.to_array(), n.to_array(), t.to_array()).into())
        }
        (Value::Color(o), Value::Color(n), Value::Color(t)) => Value::Color(lanes(*o, *n, *t)),
        (Value::Quat(o), Value::Quat(n), Value::Quat(t)) => Value::Quat(rotation(*o, *n, *t)),
        _ => return None,
    })
}

/// `theirs` turned about whichever of the inspector's Euler axes `new` moved
/// from `old` on. A rotation that comes back from Euler angles is not
/// bit-for-bit the one that went in, so an axis counts as moved past a
/// hundredth of a degree.
fn rotation(old: Quat, new: Quat, theirs: Quat) -> Quat {
    const MOVED: f32 = 1e-2_f32.to_radians();
    let (old, new, mut theirs) = (
        old.to_euler(EulerRot::YXZ),
        new.to_euler(EulerRot::YXZ),
        theirs.to_euler(EulerRot::YXZ),
    );
    if (old.0 - new.0).abs() > MOVED {
        theirs.0 = new.0;
    }
    if (old.1 - new.1).abs() > MOVED {
        theirs.1 = new.1;
    }
    if (old.2 - new.2).abs() > MOVED {
        theirs.2 = new.2;
    }
    Quat::from_euler(EulerRot::YXZ, theirs.0, theirs.1, theirs.2)
}

#[cfg(test)]
mod tests {
    use glam::{Vec2, Vec3};

    use super::*;

    fn transform(at: Vec3, rotation: Quat) -> Value {
        Value::strukt([
            ("translation", Value::Vec3(at)),
            ("rotation", Value::Quat(rotation)),
            ("scale", Value::Vec3(Vec3::ONE)),
        ])
    }

    #[test]
    fn only_the_lane_that_moved_is_written() {
        let before = transform(Vec3::ZERO, Quat::IDENTITY);
        let after = transform(Vec3::new(4.0, 0.0, 0.0), Quat::IDENTITY);
        let target = transform(Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_y(0.5));
        assert_eq!(
            spread(&before, &after, &target),
            transform(Vec3::new(4.0, 2.0, 3.0), Quat::from_rotation_y(0.5))
        );
    }

    #[test]
    fn a_rotation_moves_only_the_euler_axis_that_was_dragged() {
        let euler = |pitch: f32, yaw: f32| Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
        let before = transform(Vec3::ZERO, euler(0.0, 0.0));
        let after = transform(Vec3::ZERO, euler(0.3, 0.0));
        let target = transform(Vec3::ONE, euler(0.0, 1.0));

        let Value::Struct(fields) = spread(&before, &after, &target) else {
            unreachable!()
        };
        let Value::Quat(turned) = fields.iter().find(|(f, _)| f == "rotation").unwrap().1 else {
            unreachable!()
        };
        let (yaw, pitch, roll) = turned.to_euler(EulerRot::YXZ);
        assert!(
            (pitch - 0.3).abs() < 1e-4,
            "the pitch it was given: {pitch}"
        );
        assert!((yaw - 1.0).abs() < 1e-4, "the yaw it had: {yaw}");
        assert!(roll.abs() < 1e-4);
    }

    #[test]
    fn a_scalar_a_string_and_a_new_variant_are_written_whole() {
        let light = |variant: &str, power: f32, colour: Vec3| Value::Enum {
            variant: variant.to_owned(),
            fields: vec![
                ("color".to_owned(), Value::Vec3(colour)),
                ("lumens".to_owned(), Value::F32(power)),
            ],
        };
        let target = light("Point", 50.0, Vec3::X);
        assert_eq!(
            spread(
                &light("Point", 10.0, Vec3::ONE),
                &light("Point", 20.0, Vec3::ONE),
                &target
            ),
            light("Point", 20.0, Vec3::X),
            "the power, not the colour"
        );
        assert_eq!(
            spread(
                &light("Point", 10.0, Vec3::ONE),
                &light("Spot", 10.0, Vec3::ONE),
                &target
            ),
            light("Spot", 10.0, Vec3::ONE),
            "another kind replaces the whole value"
        );
        assert_eq!(
            spread(
                &Value::String("A".into()),
                &Value::String("B".into()),
                &Value::String("C".into())
            ),
            Value::String("B".into())
        );
        assert_eq!(
            spread(
                &Value::Vec2(Vec2::ZERO),
                &Value::Vec2(Vec2::new(0.0, 1.0)),
                &Value::Vec2(Vec2::new(5.0, 5.0))
            ),
            Value::Vec2(Vec2::new(5.0, 1.0))
        );
    }
}
