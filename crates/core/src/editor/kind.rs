//! What sort of thing an entity is, for the one icon the hierarchy row and the
//! inspector header show for it.
//!
//! Decided from the components present, in a fixed order of precedence: a
//! light is a light even if it also has a mesh (a lamp model with its bulb on
//! it), a camera likewise, and a mesh with a script is still drawn as a mesh,
//! because what it looks like in the scene is how the user goes looking for it.
//! A UI element takes the icon of its most telling component, the one that
//! component's inspector section wears.

use egui::Color32;
use orrin_ecs::{Entity, World};

use super::{catalog, icons, theme};
#[cfg(feature = "scripting")]
use crate::scene::ScriptComponent;
use crate::scene::registry as ids;
use crate::scene::{CameraComponent, Light, MeshHandle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Nothing drawn and nothing lit: a transform others hang off.
    Group,
    Mesh,
    Light,
    Sun,
    Camera,
    #[cfg_attr(
        not(feature = "scripting"),
        allow(dead_code, reason = "only a script component makes one")
    )]
    Script,
    Canvas,
    UiElement,
    Image,
    Text,
    Button,
    ScrollFrame,
}

impl Kind {
    pub fn of(world: &World, entity: Entity) -> Self {
        if let Some(light) = world.get::<Light>(entity) {
            return match *light {
                Light::Directional { .. } => Self::Sun,
                _ => Self::Light,
            };
        }
        if world.has::<CameraComponent>(entity) {
            return Self::Camera;
        }
        if world.has::<MeshHandle>(entity) {
            return Self::Mesh;
        }
        if let Some(ui) = Self::of_ui(world, entity) {
            return ui;
        }
        #[cfg(feature = "scripting")]
        if world.has::<ScriptComponent>(entity) {
            return Self::Script;
        }
        Self::Group
    }

    /// What it does before what it shows: a button carries an image, and a
    /// scroll frame usually does too.
    fn of_ui(world: &World, entity: Entity) -> Option<Self> {
        use crate::ui::{Button, Canvas, Image, RectTransform, ScrollFrame, Text};
        if world.has::<Canvas>(entity) {
            Some(Self::Canvas)
        } else if world.has::<ScrollFrame>(entity) {
            Some(Self::ScrollFrame)
        } else if world.has::<Button>(entity) {
            Some(Self::Button)
        } else if world.has::<Text>(entity) {
            Some(Self::Text)
        } else if world.has::<Image>(entity) {
            Some(Self::Image)
        } else if world.has::<RectTransform>(entity) {
            Some(Self::UiElement)
        } else {
            None
        }
    }

    pub fn icon(self) -> icons::Icon {
        match self {
            Self::Group => icons::circle(),
            Self::Mesh => icons::box_(),
            Self::Light => icons::lightbulb(),
            Self::Sun => icons::sun(),
            Self::Camera => icons::camera(),
            Self::Script => icons::file_code(),
            Self::Canvas => catalog::icon(&ids::CANVAS),
            Self::UiElement => catalog::icon(&ids::RECT_TRANSFORM),
            Self::Image => catalog::icon(&ids::IMAGE),
            Self::Text => catalog::icon(&ids::TEXT),
            Self::Button => catalog::icon(&ids::BUTTON),
            Self::ScrollFrame => catalog::icon(&ids::SCROLL_FRAME),
        }
    }

    /// The tint the icon wears. A group is the weak text colour, which only
    /// the `Ui` knows, so it is passed in.
    pub fn tint(self, weak: Color32) -> Color32 {
        match self {
            Self::Group => weak,
            Self::Mesh => theme::KIND_MESH,
            Self::Light | Self::Sun => theme::KIND_LIGHT,
            Self::Camera => theme::KIND_CAMERA,
            Self::Script => theme::KIND_SCRIPT,
            Self::Canvas
            | Self::UiElement
            | Self::Image
            | Self::Text
            | Self::Button
            | Self::ScrollFrame => theme::KIND_UI,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Group => "Group",
            Self::Mesh => "Mesh",
            Self::Light => "Light",
            Self::Sun => "Sun",
            Self::Camera => "Camera",
            Self::Script => "Script",
            Self::Canvas => "Canvas",
            Self::UiElement => "UI Element",
            Self::Image => "Image",
            Self::Text => "Text",
            Self::Button => "Button",
            Self::ScrollFrame => "Scroll Frame",
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

    /// A camera with a body model on it is still the camera to go looking for.
    #[test]
    fn a_camera_wins_over_a_mesh() {
        let mut world = World::new();
        let camera = world.spawn();
        world.insert(camera, MeshHandle(0));
        world.insert(camera, crate::scene::CameraComponent::default());
        assert_eq!(Kind::of(&world, camera), Kind::Camera);
    }

    /// A UI element is named by what it does before what it shows: a button
    /// carries an image, and a scroll frame usually does too.
    #[test]
    fn each_ui_element_is_read_from_its_most_telling_component() {
        use crate::ui::{Button, Canvas, Image, RectTransform, ScrollFrame, Text};
        let mut world = World::new();
        let spawn = |world: &mut World| {
            let entity = world.spawn();
            world.insert(entity, RectTransform::default());
            entity
        };
        let canvas = world.spawn();
        world.insert(canvas, Canvas::default());
        let rect = spawn(&mut world);
        let image = spawn(&mut world);
        world.insert(image, Image::default());
        let text = spawn(&mut world);
        world.insert(text, Text::default());
        let button = spawn(&mut world);
        world.insert(button, Image::default());
        world.insert(button, Button::default());
        let scroll = spawn(&mut world);
        world.insert(scroll, Image::default());
        world.insert(scroll, ScrollFrame::default());

        assert_eq!(Kind::of(&world, canvas), Kind::Canvas);
        assert_eq!(Kind::of(&world, rect), Kind::UiElement);
        assert_eq!(Kind::of(&world, image), Kind::Image);
        assert_eq!(Kind::of(&world, text), Kind::Text);
        assert_eq!(Kind::of(&world, button), Kind::Button);
        assert_eq!(Kind::of(&world, scroll), Kind::ScrollFrame);
    }

    const UI: [Kind; 6] = [
        Kind::Canvas,
        Kind::UiElement,
        Kind::Image,
        Kind::Text,
        Kind::Button,
        Kind::ScrollFrame,
    ];

    /// Kinds are told apart by their icon's tint; two that share one must
    /// share it on purpose — the two lights, and the UI elements, which are
    /// told apart by their icons instead.
    #[test]
    fn only_the_two_lights_and_the_ui_share_a_tint() {
        let weak = Color32::GRAY;
        let kinds = [
            Kind::Group,
            Kind::Mesh,
            Kind::Light,
            Kind::Camera,
            Kind::Script,
            Kind::Canvas,
        ];
        for (i, a) in kinds.iter().enumerate() {
            for b in &kinds[i + 1..] {
                assert_ne!(a.tint(weak), b.tint(weak), "{a:?} and {b:?}");
            }
        }
        assert_eq!(Kind::Light.tint(weak), Kind::Sun.tint(weak));
        for kind in UI {
            assert_eq!(kind.tint(weak), Kind::Canvas.tint(weak), "{kind:?}");
        }
    }

    #[test]
    fn no_two_ui_kinds_share_an_icon() {
        for (i, a) in UI.iter().enumerate() {
            for b in &UI[i + 1..] {
                assert_ne!(a.icon(), b.icon(), "{a:?} and {b:?}");
            }
        }
    }
}
