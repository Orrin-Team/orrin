//! What sort of thing an entity is, for the one icon the hierarchy row and the
//! inspector header show for it.
//!
//! Decided from the components present, in a fixed order of precedence: a
//! light is a light even if it also has a mesh (a lamp model with its bulb on
//! it), and a mesh with a script is still drawn as a mesh, because what it
//! looks like in the scene is how the user goes looking for it.

use egui::Color32;
use orrin_ecs::{Entity, World};

use super::{icons, theme};
#[cfg(feature = "scripting")]
use crate::scene::ScriptComponent;
use crate::scene::{Light, MeshHandle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Nothing drawn and nothing lit: a transform others hang off.
    Group,
    Mesh,
    Light,
    Sun,
    #[cfg_attr(
        not(feature = "scripting"),
        allow(dead_code, reason = "only a script component makes one")
    )]
    Script,
}

impl Kind {
    pub fn of(world: &World, entity: Entity) -> Self {
        if let Some(light) = world.get::<Light>(entity) {
            return match *light {
                Light::Directional { .. } => Self::Sun,
                _ => Self::Light,
            };
        }
        if world.has::<MeshHandle>(entity) {
            return Self::Mesh;
        }
        #[cfg(feature = "scripting")]
        if world.has::<ScriptComponent>(entity) {
            return Self::Script;
        }
        Self::Group
    }

    pub fn icon(self) -> icons::Icon {
        match self {
            Self::Group => icons::circle(),
            Self::Mesh => icons::box_(),
            Self::Light => icons::lightbulb(),
            Self::Sun => icons::sun(),
            Self::Script => icons::file_code(),
        }
    }

    /// The tint the icon wears. A group is the weak text colour, which only
    /// the `Ui` knows, so it is passed in.
    pub fn tint(self, weak: Color32) -> Color32 {
        match self {
            Self::Group => weak,
            Self::Mesh => theme::KIND_MESH,
            Self::Light | Self::Sun => theme::KIND_LIGHT,
            Self::Script => theme::KIND_SCRIPT,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Group => "Group",
            Self::Mesh => "Mesh",
            Self::Light => "Light",
            Self::Sun => "Sun",
            Self::Script => "Script",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::MaterialHandle;
    use glam::Vec3;

    #[test]
    fn each_kind_is_read_from_its_components() {
        let mut world = World::new();
        let group = world.spawn();
        let mesh = world.spawn();
        world.insert(mesh, MeshHandle(0));
        world.insert(mesh, MaterialHandle(0));
        let bulb = world.spawn();
        world.insert(bulb, Light::default());
        let sun = world.spawn();
        world.insert(
            sun,
            Light::Directional {
                color: Vec3::ONE,
                lux: 20_000.0,
            },
        );

        assert_eq!(Kind::of(&world, group), Kind::Group);
        assert_eq!(Kind::of(&world, mesh), Kind::Mesh);
        assert_eq!(Kind::of(&world, sun), Kind::Sun);
        assert!(matches!(Kind::of(&world, bulb), Kind::Light | Kind::Sun));
    }

    /// A lamp model with its light on the same entity is found by looking for
    /// lights, not meshes.
    #[test]
    fn a_light_wins_over_a_mesh() {
        let mut world = World::new();
        let lamp = world.spawn();
        world.insert(lamp, MeshHandle(0));
        world.insert(
            lamp,
            Light::Point {
                color: Vec3::ONE,
                lumens: 800.0,
                range: 10.0,
                casts_shadows: false,
            },
        );
        assert_eq!(Kind::of(&world, lamp), Kind::Light);
    }

    /// Kinds are told apart by their icon's tint; two that share one must
    /// share it on purpose.
    #[test]
    fn only_the_two_lights_share_a_tint() {
        let weak = Color32::GRAY;
        let kinds = [Kind::Group, Kind::Mesh, Kind::Light, Kind::Script];
        for (i, a) in kinds.iter().enumerate() {
            for b in &kinds[i + 1..] {
                assert_ne!(a.tint(weak), b.tint(weak), "{a:?} and {b:?}");
            }
        }
        assert_eq!(Kind::Light.tint(weak), Kind::Sun.tint(weak));
    }
}
