use orrin_registry::{ComponentId, Registry};

use super::{
    AudioListener, AudioSource, CameraComponent, Collider, Light, LocalTransform, Locked,
    MaterialHandle, MeshHandle, Name, PhysicsSettings, RenderSettings, RigidBody, Spin, Tag,
};

pub const TRANSFORM: ComponentId = ComponentId::new("orrin.transform");
pub const NAME: ComponentId = orrin_registry::NAME_COMPONENT;
pub const TAG: ComponentId = ComponentId::new("orrin.tag");
pub const LIGHT: ComponentId = ComponentId::new("orrin.light");
pub const COLLIDER: ComponentId = ComponentId::new("orrin.collider");
pub const RIGID_BODY: ComponentId = ComponentId::new("orrin.rigid_body");
pub const PHYSICS_SETTINGS: ComponentId = ComponentId::new("orrin.physics_settings");
pub const AUDIO_SOURCE: ComponentId = ComponentId::new("orrin.audio_source");
pub const AUDIO_LISTENER: ComponentId = ComponentId::new("orrin.audio_listener");
pub const SPIN: ComponentId = ComponentId::new("orrin.spin");
pub const LOCKED: ComponentId = ComponentId::new("orrin.locked");
pub const RENDER_SETTINGS: ComponentId = ComponentId::new("orrin.render_settings");
pub const CAMERA: ComponentId = ComponentId::new("orrin.camera");
pub const CANVAS: ComponentId = ComponentId::new("orrin.ui.canvas");
pub const RECT_TRANSFORM: ComponentId = ComponentId::new("orrin.ui.rect_transform");
pub const IMAGE: ComponentId = ComponentId::new("orrin.ui.image");
pub const TEXT: ComponentId = ComponentId::new("orrin.ui.text");
pub const BUTTON: ComponentId = ComponentId::new("orrin.ui.button");
pub const GRADIENT: ComponentId = ComponentId::new("orrin.ui.gradient");
pub const CORNER: ComponentId = ComponentId::new("orrin.ui.corner");
pub const STROKE: ComponentId = ComponentId::new("orrin.ui.stroke");
pub const SHADOW: ComponentId = ComponentId::new("orrin.ui.shadow");
pub const SCROLL_FRAME: ComponentId = ComponentId::new("orrin.ui.scroll_frame");

/// `MeshHandle` and `MaterialHandle` are upload indices, so a scene stores the
/// asset's [`AssetId`](orrin_registry::AssetId) instead — a translation that
/// needs the world, which `Reflect::to_value` deliberately cannot reach. They
/// register through `Resolve` (see `asset_db::handles`), under the ids scenes
/// stored their names under before assets had ids, so those scenes still load.
pub const MESH: ComponentId = ComponentId::new("orrin.mesh");
pub const MATERIAL: ComponentId = ComponentId::new("orrin.material");

/// Reserved, not registered: resolved one level up, for the reason the two
/// above go through `Resolve`.
///
/// `Parent` holds an `orrin_ecs::Entity` — a slot handle whose meaning is this
/// session's spawn history — so writing one to disk would name a different
/// entity, or none, the next time the scene is opened. The file stores the
/// parent's [`EntityId`](orrin_registry::EntityId) instead, which needs the
/// world to translate in both directions and so cannot live in a `Reflect` impl.
pub const PARENT: ComponentId = ComponentId::new("orrin.parent");

/// Reserved for a prefab instance's record — its source and overrides — which
/// stands in for the instance's members in a scene file. See `scene::prefab`.
pub const PREFAB: ComponentId = ComponentId::new("orrin.prefab");

/// Describe every component the engine itself owns to `registry`.
///
/// The counterpart a game assembly exports under the same name, called again
/// after each hot reload. Not derived from a linker section: `inventory`-style
/// auto-registration does not cross a dynamic library boundary, which is the
/// whole situation this exists for.
///
/// Deliberately absent for now:
/// - `Decal` names its three maps by `TextureHandle`, which is an upload index
///   and so session-local for exactly the reason `MeshHandle` is. It registers
///   once it holds texture asset ids.
/// - `ScriptComponent` owns a `GCHandle` whose `Drop` is the single managed
///   teardown path. It joins as a *bridge* to the C# property bag, never as an
///   ordinary component — a registry `write` replaces the component wholesale,
///   which here would free a live handle.
/// - `WorldTransform` is derived from `LocalTransform` every frame, so a saved
///   value would be overwritten before anything could read it. Writing one to a
///   scene would also record a *result* beside the inputs that produce it, which
///   is the kind of redundancy a file format cannot keep consistent.
/// - `Parent` holds a session-local entity handle. It *is* persisted, under the
///   reserved [`PARENT`] id, but as the parent's `EntityId` — a translation that
///   needs the world in both directions, so it happens in `scene::persist`.
pub fn register_components(registry: &mut Registry) {
    registry.register::<LocalTransform>(TRANSFORM, "Transform");
    registry.register::<Name>(NAME, "Name");
    registry.register::<Tag>(TAG, "Tag");
    registry.register::<Light>(LIGHT, "Light");
    registry.register::<Collider>(COLLIDER, "Collider");
    registry.register::<RigidBody>(RIGID_BODY, "Rigid Body");
    registry.register::<AudioSource>(AUDIO_SOURCE, "Audio Source");
    registry.register::<AudioListener>(AUDIO_LISTENER, "Audio Listener");
    registry.register::<Spin>(SPIN, "Spin");
    registry.register::<Locked>(LOCKED, "Locked");
    registry.register::<RenderSettings>(RENDER_SETTINGS, "Render Settings");
    registry.register::<PhysicsSettings>(PHYSICS_SETTINGS, "Physics Settings");
    registry.register::<CameraComponent>(CAMERA, "Camera");
    registry.register::<crate::ui::Canvas>(CANVAS, "Canvas");
    registry.register::<crate::ui::RectTransform>(RECT_TRANSFORM, "Rect Transform");
    registry.register::<crate::ui::Image>(IMAGE, "Image");
    registry.register::<crate::ui::Text>(TEXT, "Text");
    registry.register::<crate::ui::Button>(BUTTON, "Button");
    registry.register::<crate::ui::Gradient>(GRADIENT, "Gradient");
    registry.register::<crate::ui::Corner>(CORNER, "Corner");
    registry.register::<crate::ui::Stroke>(STROKE, "Stroke");
    registry.register::<crate::ui::Shadow>(SHADOW, "Shadow");
    registry.register::<crate::ui::ScrollFrame>(SCROLL_FRAME, "Scroll Frame");
    registry.register_resolved::<MeshHandle>(MESH, "Mesh");
    registry.register_resolved::<MaterialHandle>(MATERIAL, "Material");
    registry.end_engine_registration();
}

#[cfg(test)]
mod tests {
    use glam::{Quat, Vec3};
    use orrin_ecs::World;
    use orrin_registry::{Reflect, Value};

    use super::*;
    use crate::scene::{CameraComponent, ColliderShape, Transform};

    /// What is locked is how the scene is meant to be worked on, so it is
    /// saved with it.
    #[test]
    fn a_lock_survives_a_save_and_a_load() {
        let mut registry = Registry::new();
        register_components(&mut registry);
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Locked);
        let text = crate::scene::save(&mut world, &registry);
        assert!(text.contains("orrin.locked"), "{text}");

        let mut loaded = World::new();
        crate::scene::load(&text, &mut loaded, &registry).unwrap();
        let entity = loaded.entities().next().unwrap();
        assert!(loaded.has::<Locked>(entity));
    }

    /// A scene opens on the camera it was saved with, lens and priority kept.
    #[test]
    fn a_camera_survives_a_save_and_a_load() {
        let mut registry = Registry::new();
        register_components(&mut registry);
        let mut world = World::new();
        let entity = world.spawn();
        let camera = CameraComponent {
            fov_degrees: 45.0,
            near: 0.3,
            far: 120.0,
            priority: 3,
        };
        world.insert(entity, camera);
        let text = crate::scene::save(&mut world, &registry);
        assert!(text.contains("orrin.camera"), "{text}");

        let mut loaded = World::new();
        crate::scene::load(&text, &mut loaded, &registry).unwrap();
        let entity = loaded.entities().next().unwrap();
        assert_eq!(*loaded.get::<CameraComponent>(entity).unwrap(), camera);
    }

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
                ..Collider::default()
            },
        );
        world.insert(entity, Spin::new(Vec3::Y, 1.5));

        let mut out = String::new();
        orrin_registry::write_entity(&mut out, &registry, &world, entity);

        assert_eq!(
            out,
            "\
entity #1 \"Cube\"
  orrin.collider
    friction = 0.5
    is_trigger = false
    layer = 0
    restitution = 0.0
    shape = Box
      half_extents = (0.5, 0.5, 0.5)
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
        world.insert(entity, Light::point(Vec3::ONE, 800.0, 10.0));

        let mut out = String::new();
        orrin_registry::write_entity(&mut out, &registry, &world, entity);

        assert_eq!(
            out,
            "\
entity #1
  orrin.light = Point
    casts_shadows = true
    color = (1.0, 1.0, 1.0)
    lumens = 800.0
    range = 10.0
"
        );
    }

    #[test]
    fn an_unknown_variant_names_itself() {
        let stale = Value::enumeration("Area", [("width", Value::F32(30.0))]);
        let err = Light::from_value(&stale).unwrap_err();
        assert_eq!(
            err.to_string(),
            "expected one of: Directional, Point, Spot, found `Area`"
        );
    }

    /// A point light saved before `casts_shadows` existed still loads, and loads
    /// as a caster — which is what it was, since every light cast before the
    /// switch was there to say otherwise. Without the `#[reflect(default)]` on
    /// that field this is a missing-field error and the whole entity is lost, so
    /// the guarantee is worth a test of its own rather than trusting the derive.
    #[test]
    fn a_light_saved_before_the_shadow_switch_still_loads() {
        let old = Value::enumeration(
            "Point",
            [
                ("color", Vec3::ONE.to_value()),
                ("lumens", Value::F32(800.0)),
                ("range", Value::F32(10.0)),
            ],
        );
        let light = Light::from_value(&old).expect("an older point light must still load");
        assert!(matches!(
            light,
            Light::Point {
                casts_shadows: true,
                ..
            }
        ));
    }

    /// The other half of the bargain in `Light`'s doc comment: a light authored
    /// against the pre-photometric `intensity` field must fail loudly rather than
    /// load dark.
    ///
    /// This is the one migration the engine deliberately does *not* smooth over.
    /// Reading `intensity = 8` as eight lumens would load a scene lit at about a
    /// thousandth of what its author saw, with nothing anywhere saying why — and
    /// a plausible-looking scene that is simply black is far more expensive to
    /// diagnose than an error naming the field. Hence the rename rather than a
    /// reinterpretation, and hence a test: the *absence* of a compatibility path
    /// is the feature, and nothing else would stop someone adding one back.
    #[test]
    fn a_light_saved_before_physical_units_refuses_to_load() {
        let unitless = Value::enumeration(
            "Point",
            [
                ("color", Vec3::ONE.to_value()),
                ("intensity", Value::F32(8.0)),
                ("range", Value::F32(10.0)),
            ],
        );
        let err = Light::from_value(&unitless)
            .expect_err("an intensity-based light must not load as lumens");
        assert_eq!(err.path.to_string(), "lumens");
        assert_eq!(err.found, "nothing");
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

        let unnormalized = Value::strukt([
            ("axis", Vec3::new(0.0, 4.0, 0.0).to_value()),
            ("speed", Value::F32(1.0)),
        ]);
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
            ..Collider::default()
        });
        round_trips(&Collider {
            shape: ColliderShape::Capsule {
                radius: 0.3,
                half_height: 0.9,
            },
            is_trigger: false,
            friction: 0.8,
            restitution: 0.25,
            layer: 3,
        });
        round_trips(&Collider {
            shape: ColliderShape::ConvexHull,
            ..Collider::default()
        });
        round_trips(&crate::scene::RigidBody {
            continuous: true,
            lock_rotation: true,
            ..crate::scene::RigidBody::default()
        });
        let mut physics = crate::scene::PhysicsSettings::default();
        physics.set_collides(1, 4, false);
        round_trips(&physics);
        round_trips(&Spin::new(Vec3::new(0.0, 1.0, 1.0), 2.0));
        round_trips(&crate::scene::AudioSource {
            clip: orrin_registry::AssetId::new(),
            looping: true,
            rolloff: crate::scene::Rolloff::Linear,
            ..crate::scene::AudioSource::default()
        });
        round_trips(&crate::scene::AudioListener);
    }

    /// A sound is saved as the asset it names, so a scene reopened after the
    /// file moved still plays it.
    #[test]
    fn an_audio_source_and_its_listener_survive_a_save_and_a_load() {
        let mut registry = Registry::new();
        register_components(&mut registry);
        let mut world = World::new();
        let entity = world.spawn();
        let source = crate::scene::AudioSource {
            clip: orrin_registry::AssetId::new(),
            volume: 0.4,
            spatial: false,
            ..crate::scene::AudioSource::default()
        };
        world.insert(entity, source);
        world.insert(entity, crate::scene::AudioListener);
        let text = crate::scene::save(&mut world, &registry);
        assert!(text.contains("orrin.audio_source"), "{text}");
        assert!(text.contains("orrin.audio_listener"), "{text}");

        let mut loaded = World::new();
        crate::scene::load(&text, &mut loaded, &registry).unwrap();
        let entity = loaded.entities().next().unwrap();
        assert_eq!(
            *loaded.get::<crate::scene::AudioSource>(entity).unwrap(),
            source
        );
        assert!(loaded.has::<crate::scene::AudioListener>(entity));
    }

    /// A collider saved before it had a surface or a layer loads as the
    /// collider it was, with the defaults for what it did not say.
    #[test]
    fn a_collider_saved_before_its_surface_existed_loads_with_the_defaults() {
        let mut value = Collider::default().to_value();
        let orrin_registry::Value::Struct(fields) = &mut value else {
            panic!("a collider reflects as a struct");
        };
        fields.retain(|(name, _)| name == "shape" || name == "is_trigger");

        let loaded = Collider::from_value(&value).expect("an old collider still loads");

        assert_eq!(loaded, Collider::default());
    }

    #[test]
    fn name_and_tag_share_a_shape_but_not_an_identity() {
        let mut registry = Registry::new();
        register_components(&mut registry);

        assert_eq!(Name::new("x").to_value(), Tag::new("x").to_value());
        assert_ne!(
            registry.of::<Name>().unwrap().id,
            registry.of::<Tag>().unwrap().id
        );
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
        vtable.write(&mut world, entity, &moved.to_value()).unwrap();

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
        let err = registry
            .get(&TRANSFORM)
            .unwrap()
            .write(&mut world, entity, &broken)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "field `translation`: expected vec3, found bool"
        );
    }
}
