use crate::value::{Value, ValueError};

/// How a type converts to and from [`Value`].
///
/// Implemented by leaf types (`f32`, `Vec3`, `Entity`, …) as well as by
/// components, which is what lets a component's conversion be written without
/// naming any field's type: every field is just another `Reflect`. The derive
/// macro that eventually replaces hand-written impls inherits that property —
/// it needs no list of supported types, so it cannot fall out of date with one.
///
/// `to_value` takes `&self` because the registry reads components through a
/// `Ref<'_, T>` borrowed out of world storage: nothing can be moved out of it,
/// and reading a component for the inspector must not consume it.
pub trait Reflect: 'static + Sized {
    fn to_value(&self) -> Value;
    fn from_value(value: &Value) -> Result<Self, ValueError>;
}

/// Both directions are generated from one `$variant`, so they cannot disagree
/// about which representation a type uses — the failure mode being a value that
/// writes fine and silently refuses to read back.
macro_rules! impl_leaf {
    ($ty:ty, $variant:ident, $expected:literal) => {
        impl Reflect for $ty {
            #[allow(clippy::clone_on_copy)]
            fn to_value(&self) -> Value {
                Value::$variant(self.clone())
            }

            fn from_value(value: &Value) -> Result<Self, ValueError> {
                match value {
                    Value::$variant(inner) => Ok(inner.clone()),
                    other => Err(ValueError::mismatch($expected, other)),
                }
            }
        }
    };
}

impl_leaf!(bool, Bool, "bool");
impl_leaf!(String, String, "string");
impl_leaf!(glam::Vec3, Vec3, "vec3");
impl_leaf!(glam::Quat, Quat, "quat");
impl_leaf!(crate::EntityId, Entity, "entity");

/// The numeric leaves read leniently across the integer/float boundary.
///
/// The text format writes `3` for both an `i32` and a `u32` — width isn't in
/// the syntax — so the parser cannot know which variant a number was, and a
/// strict reader would reject half of all integer fields. Hand-edited files get
/// the same benefit: writing `speed = 3` for an `f32` works.
///
/// The leniency is on *read* only. `to_value` always produces the field's
/// declared variant, so re-saving canonicalizes the file, and `Value` equality
/// stays exact — `Value::I32(3)` is still not `Value::U32(3)`.
impl Reflect for f32 {
    fn to_value(&self) -> Value {
        Value::F32(*self)
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        match value {
            Value::F32(v) => Ok(*v),
            Value::I32(v) => Ok(*v as f32),
            Value::U32(v) => Ok(*v as f32),
            other => Err(ValueError::mismatch("f32", other)),
        }
    }
}

impl Reflect for i32 {
    fn to_value(&self) -> Value {
        Value::I32(*self)
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        match value {
            Value::I32(v) => Ok(*v),
            Value::U32(v) => {
                i32::try_from(*v).map_err(|_| ValueError::invalid("an i32", format!("{v}")))
            }
            other => Err(ValueError::mismatch("i32", other)),
        }
    }
}

impl Reflect for u32 {
    fn to_value(&self) -> Value {
        Value::U32(*self)
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        match value {
            Value::U32(v) => Ok(*v),
            Value::I32(v) => {
                u32::try_from(*v).map_err(|_| ValueError::invalid("a u32", format!("{v}")))
            }
            other => Err(ValueError::mismatch("u32", other)),
        }
    }
}

impl<T: Reflect> Reflect for Vec<T> {
    fn to_value(&self) -> Value {
        Value::List(self.iter().map(Reflect::to_value).collect())
    }

    fn from_value(value: &Value) -> Result<Self, ValueError> {
        let Value::List(items) = value else {
            return Err(ValueError::mismatch("list", value));
        };
        items
            .iter()
            .enumerate()
            .map(|(index, item)| T::from_value(item).map_err(|e| e.at_index(index)))
            .collect()
    }
}

/// Read one named field out of a struct or enum value.
///
/// The only place the two halves of path building meet: a missing field
/// supplies its own segment, while a present-but-wrong one gets a segment
/// prepended to whatever the nested read reported. Getting that asymmetry wrong
/// yields either a doubled or a truncated path, so it lives here once instead
/// of in every `from_value`.
pub fn take<T: Reflect>(value: &Value, field: &str) -> Result<T, ValueError> {
    let Some(inner) = value.field(field) else {
        return Err(ValueError::missing(field));
    };
    T::from_value(inner).map_err(|e| e.at_field(field))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    // Only for the nested-path test below, which reads an intermediate level
    // without having a concrete type for it.
    impl Reflect for Value {
        fn to_value(&self) -> Value {
            self.clone()
        }

        fn from_value(value: &Value) -> Result<Self, ValueError> {
            Ok(value.clone())
        }
    }

    #[test]
    fn leaves_round_trip() {
        assert_eq!(f32::from_value(&1.5f32.to_value()), Ok(1.5));
        assert_eq!(bool::from_value(&true.to_value()), Ok(true));
        assert_eq!(
            String::from_value(&"hi".to_owned().to_value()),
            Ok("hi".to_owned())
        );

        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(Vec3::from_value(&v.to_value()), Ok(v));
    }

    #[test]
    fn a_leaf_rejects_the_wrong_variant() {
        let err = f32::from_value(&Value::Bool(true)).unwrap_err();
        assert_eq!(err.expected, "f32");
        assert_eq!(err.found, "bool");
        assert!(err.path.is_empty());
    }

    #[test]
    fn lists_round_trip_and_report_the_failing_index() {
        let list = vec![1.0f32, 2.0].to_value();
        assert_eq!(Vec::<f32>::from_value(&list), Ok(vec![1.0, 2.0]));

        let mixed = Value::List(vec![Value::F32(1.0), Value::Bool(false)]);
        let err = Vec::<f32>::from_value(&mixed).unwrap_err();
        assert_eq!(err.path.to_string(), "[1]");
        assert_eq!(err.found, "bool");
    }

    #[test]
    fn take_names_the_field_it_failed_on() {
        let value = Value::strukt([("speed", Value::String("fast".to_owned()))]);

        let wrong_type = take::<f32>(&value, "speed").unwrap_err();
        assert_eq!(
            wrong_type.to_string(),
            "field `speed`: expected f32, found string"
        );

        let absent = take::<f32>(&value, "health").unwrap_err();
        assert_eq!(
            absent.to_string(),
            "field `health`: expected a value, found nothing"
        );
    }

    #[test]
    fn nested_reads_compose_a_full_path() {
        let value = Value::strukt([(
            "transform",
            Value::strukt([("points", Value::List(vec![Value::Bool(true)]))]),
        )]);

        let err = take::<Value>(&value, "transform")
            .and_then(|t| take::<Vec<f32>>(&t, "points"))
            .unwrap_err();
        assert_eq!(err.path.to_string(), "points[0]");
    }
}
