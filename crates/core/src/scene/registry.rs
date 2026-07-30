use orrin_registry::{ComponentId, Registry};

use super::{Collider, Light, LocalTransform, Name, Spin, Tag};

pub const TRANSFORM: ComponentId = ComponentId::new("orrin.transform");
pub const NAME: ComponentId = ComponentId::new("orrin.name");
pub const TAG: ComponentId = ComponentId::new("orrin.tag");
pub const LIGHT: ComponentId = ComponentId::new("orrin.light");
pub const COLLIDER: ComponentId = ComponentId::new("orrin.collider");
pub const SPIN: ComponentId = ComponentId::new("orrin.spin");

/// Reserved, not registered. `MeshHandle` and `MaterialHandle` are upload
/// indices, so a scene stores the asset's *name* instead — a translation that
/// needs the `Assets` resource, which `Reflect::to_value` deliberately cannot
/// reach. `scene::persist` does it one level up, where the world is in scope.
///
/// The ids live here anyway so the namespace stays auditable in one place, and
/// so that when assets gain stable ids these two can register properly and take
/// the same ids — scenes written today keep parsing.
pub const MESH: ComponentId = ComponentId::new("orrin.mesh");
pub const MATERIAL: ComponentId = ComponentId::new("orrin.material");

/// Describe every component the engine itself owns to `registry`.
///
/// The counterpart a game assembly exports under the same name, called again
/// after each hot reload. Not derived from a linker section: `inventory`-style
/// auto-registration does not cross a dynamic library boundary, which is the
/// whole situation this exists for.
///
/// Deliberately absent for now:
/// - `MeshHandle` / `MaterialHandle` index a runtime asset table, so writing
///   one to disk would bake a session-local number into a scene. They register
///   once assets have stable ids.
/// - `ScriptComponent` owns a `GCHandle` whose `Drop` is the single managed
///   teardown path. It joins as a *bridge* to the C# property bag, never as an
///   ordinary component — a registry `write` replaces the component wholesale,
///   which here would free a live handle.
pub fn register_components(registry: &mut Registry) {
    registry.register::<LocalTransform>(TRANSFORM, "Transform");
    registry.register::<Name>(NAME, "Name");
    registry.register::<Tag>(TAG, "Tag");
    registry.register::<Light>(LIGHT, "Light");
    registry.register::<Collider>(COLLIDER, "Collider");
    registry.register::<Spin>(SPIN, "Spin");
    registry.end_engine_registration();
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};
    use orrin_ecs::World;
    use orrin_registry::{Reflect, Value};

    use super::*;
    use crate::scene::{ColliderShape, Transform};

    fn world() -> (World, orrin_ecs::Entity) {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(
            entity,
            LocalTransform::new(Transform {
                translation: Vec3::new(0.0, 1.5, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            }),
        );
        world.insert(entity, Name::new("Cube"));
        world.insert(entity, Tag::new("player"));
        (world, entity)
    }

    #[test]
    fn engine_components_dump_deterministically() {
        let mut registry = Registry::new();
        register_components(&mut registry);
        let (mut world, entity) = world();
        world.insert(
            entity,
            Collider {
                shape: ColliderShape::Box {
                    half_extents: Vec3::splat(0.5),
                },
                is_trigger: false,
            },
        );
        world.insert(entity, Spin::new(Vec3::Y, 1.5));

        let mut out = String::new();
        orrin_registry::write_entity(&mut out, &registry, &world, entity);

        assert_eq!(
            out,
            "\
entity #1
  orrin.collider
    is_trigger = false
    shape = Box
      half_extents = (0.5, 0.5, 0.5)
  orrin.name = \"Cube\"
  orrin.spin
    axis = (0.0, 1.0, 0.0)
    speed = 1.5
  orrin.tag = \"player\"
  orrin.transform
    rotation = (0.0, 0.0, 0.0, 1.0)
    scale = (1.0, 1.0, 1.0)
    translation = (0.0, 1.5, 0.0)
"
        );
    }

    #[test]
    fn a_light_dumps_as_its_variant() {
        let mut registry = Registry::new();
        register_components(&mut registry);
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Light::point(Vec3::ONE, 8.0, 10.0));

        let mut out = String::new();
        orrin_registry::write_entity(&mut out, &registry, &world, entity);

        assert_eq!(
            out,
            "\
entity #1
  orrin.light = Point
    color = (1.0, 1.0, 1.0)
    intensity = 8.0
    range = 10.0
"
        );
    }

    #[test]
    fn an_unknown_variant_names_itself() {
        let stale = Value::enumeration("Spot", [("angle", Value::F32(30.0))]);
        let err = Light::from_value(&stale).unwrap_err();
        assert_eq!(
            err.to_string(),
            "expected one of: Directional, Point, found `Spot`"
        );
    }

    /// The reason `Spin` cannot be derived: `apply` feeds `axis` to
    /// `Quat::from_axis_angle`, so a zero axis would produce NaN rotations that
    /// spread through the hierarchy. A field-assigning `from_value` would let a
    /// hand-edited scene do exactly that.
    #[test]
    fn spin_refuses_an_axis_it_cannot_normalize() {
        let broken = Value::strukt([("axis", Vec3::ZERO.to_value()), ("speed", Value::F32(1.0))]);
        let err = Spin::from_value(&broken).unwrap_err();
        assert_eq!(err.path.to_string(), "axis");
        assert_eq!(err.expected, "a non-zero axis");

        let unnormalized =
            Value::strukt([("axis", Vec3::new(0.0, 4.0, 0.0).to_value()), ("speed", Value::F32(1.0))]);
        let spin = Spin::from_value(&unnormalized).unwrap();
        assert_eq!(spin.to_value().field("axis"), Some(&Vec3::Y.to_value()));
    }

    /// Catches the classic asymmetry: a field renamed in `to_value` but not in
    /// `from_value` still saves cleanly and silently loads as its default.
    ///
    /// Asserted in `Value` space rather than on the components themselves —
    /// `Transform` has no `PartialEq`, and comparing the round-tripped value is
    /// the stronger statement anyway, since it also pins the field *names*.
    fn round_trips<T: Reflect>(component: &T) {
        let value = component.to_value();
        let back = T::from_value(&value).expect("round trip should read back");
        assert_eq!(back.to_value(), value);
    }

    #[test]
    fn every_engine_component_round_trips() {
        round_trips(&LocalTransform::new(Transform {
            translation: Vec3::new(1.0, -2.0, 3.5),
            rotation: Quat::from_rotation_y(0.5),
            scale: Vec3::splat(2.0),
        }));
        round_trips(&Name::new("Cube"));
        round_trips(&Tag::new("player"));
        round_trips(&Light::directional(Vec3::ONE, 2.0));
        round_trips(&Light::point(Vec3::new(1.0, 0.5, 0.2), 8.0, 10.0));
        round_trips(&Collider {
            shape: ColliderShape::Sphere { radius: 1.5 },
            is_trigger: true,
        });
        round_trips(&Spin::new(Vec3::new(0.0, 1.0, 1.0), 2.0));
    }

    #[test]
    fn name_and_tag_share_a_shape_but_not_an_identity() {
        let mut registry = Registry::new();
        register_components(&mut registry);

        assert_eq!(Name::new("x").to_value(), Tag::new("x").to_value());
        assert_ne!(registry.of::<Name>().unwrap().id, registry.of::<Tag>().unwrap().id);
    }

    #[test]
    fn a_transform_written_through_the_registry_lands_on_the_component() {
        let mut registry = Registry::new();
        register_components(&mut registry);
        let (mut world, entity) = world();

        let vtable = registry.get(&TRANSFORM).unwrap();
        let moved = Transform {
            translation: Vec3::new(4.0, 0.0, 0.0),
            ..Default::default()
        };
        (vtable.write)(&mut world, entity, &moved.to_value()).unwrap();

        assert_eq!(
            world.get::<LocalTransform>(entity).unwrap().translation,
            Vec3::new(4.0, 0.0, 0.0)
        );
    }

    #[test]
    fn a_malformed_value_names_the_field() {
        let mut registry = Registry::new();
        register_components(&mut registry);
        let (mut world, entity) = world();

        let broken = Value::strukt([
            ("translation", Value::Bool(true)),
            ("rotation", Quat::IDENTITY.to_value()),
            ("scale", Vec3::ONE.to_value()),
        ]);
        let err = (registry.get(&TRANSFORM).unwrap().write)(&mut world, entity, &broken)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "field `translation`: expected vec3, found bool"
        );
    }
}
