//! What each component is, as the editor groups and pictures it: the group the
//! add menu files it under and the icon both that menu and its inspector
//! section show, so a component looks the same wherever it is offered.
//!
//! Keyed by `ComponentId` rather than by the display name, which the registry
//! lets change freely. A component missing from the table still works — it
//! lands under General with the generic icon — and a test holds that no
//! engine component does.

use orrin_registry::{ComponentId, ComponentVtable};

use super::icons::{self, Icon};
use crate::scene::registry as id;

/// The add menu's headings, in the order it lists them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Group {
    General,
    Rendering,
    Physics,
    Ui,
    Scripts,
}

impl Group {
    pub fn label(self) -> &'static str {
        match self {
            Group::General => "General",
            Group::Rendering => "Rendering",
            Group::Physics => "Physics",
            Group::Ui => "UI",
            Group::Scripts => "Scripts",
        }
    }
}

const TABLE: &[(ComponentId, Group, Icon)] = &[
    (id::TRANSFORM, Group::General, icons::move_()),
    (id::NAME, Group::General, icons::textbox()),
    (id::TAG, Group::General, icons::tag()),
    (id::LOCKED, Group::General, icons::lock()),
    (id::SPIN, Group::General, icons::refresh()),
    (id::MESH, Group::Rendering, icons::box_()),
    (id::MATERIAL, Group::Rendering, icons::palette()),
    (id::LIGHT, Group::Rendering, icons::lightbulb()),
    (id::CAMERA, Group::Rendering, icons::camera()),
    (id::RENDER_SETTINGS, Group::Rendering, icons::sliders()),
    (id::COLLIDER, Group::Physics, icons::cube_transparent()),
    (id::RIGID_BODY, Group::Physics, icons::circle()),
    (id::PHYSICS_SETTINGS, Group::Physics, icons::globe()),
    (id::CANVAS, Group::Ui, icons::app_window()),
    (id::RECT_TRANSFORM, Group::Ui, icons::bounding_box()),
    (id::IMAGE, Group::Ui, icons::image()),
    (id::TEXT, Group::Ui, icons::text_t()),
    (id::BUTTON, Group::Ui, icons::cursor_click()),
    (id::GRADIENT, Group::Ui, icons::gradient()),
    (id::CORNER, Group::Ui, icons::frame_corners()),
    (id::STROKE, Group::Ui, icons::square()),
    (id::SHADOW, Group::Ui, icons::selection_background()),
    (id::SCROLL_FRAME, Group::Ui, icons::mouse_scroll()),
];

/// The group and icon of the component `vtable` describes.
pub fn describe(vtable: &ComponentVtable) -> (Group, Icon) {
    if vtable.script.is_some() {
        return (Group::Scripts, icons::file_code());
    }
    lookup(&vtable.id).unwrap_or((Group::General, icons::layers()))
}

/// The icon of the engine component `id`, for a section drawn by hand.
pub fn icon(id: &ComponentId) -> Icon {
    lookup(id).map_or(icons::layers(), |(_, icon)| icon)
}

fn lookup(wanted: &ComponentId) -> Option<(Group, Icon)> {
    TABLE
        .iter()
        .find(|(id, ..)| id == wanted)
        .map(|&(_, group, icon)| (group, icon))
}

#[cfg(test)]
mod tests {
    use orrin_registry::Registry;

    use super::*;

    fn engine() -> Registry {
        let mut registry = Registry::new();
        crate::scene::register_components(&mut registry);
        registry
    }

    /// A component added to the engine without a line here would be filed
    /// under General with the generic icon; this is where that is caught.
    #[test]
    fn every_engine_component_has_a_group_and_an_icon() {
        for vtable in engine().components() {
            assert!(
                lookup(&vtable.id).is_some(),
                "{} is not catalogued",
                vtable.id.as_str()
            );
        }
    }

    /// The icon is what tells two rows apart at a glance, so no two share one.
    #[test]
    fn no_two_components_share_an_icon() {
        let mut seen = std::collections::HashMap::new();
        for (id, _, icon) in TABLE {
            if let Some(other) = seen.insert(icon.glyph(), id.as_str()) {
                panic!("{} and {other} share an icon", id.as_str());
            }
        }
    }

    #[test]
    fn the_ui_components_are_filed_under_ui() {
        for ui in [
            id::CANVAS,
            id::RECT_TRANSFORM,
            id::IMAGE,
            id::TEXT,
            id::BUTTON,
            id::GRADIENT,
            id::CORNER,
            id::STROKE,
            id::SHADOW,
            id::SCROLL_FRAME,
        ] {
            assert_eq!(
                lookup(&ui).map(|(group, _)| group),
                Some(Group::Ui),
                "{}",
                ui.as_str()
            );
        }
    }
}
