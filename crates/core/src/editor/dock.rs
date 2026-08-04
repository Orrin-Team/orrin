//! Docked tool layout.
//!
//! Every tool is a tab that draws into whatever `Ui` it is handed — the
//! `show`/`body` split in `panels` is what makes that possible. The layout tree
//! itself belongs to the user: it is dragged, split, floated, and persisted.
//!
//! **The centre is deliberately not a tab.** The editor draws no opaque surface
//! over the middle of the window, so the Vulkan render shows through behind the
//! whole UI. Two separate things keep that true, and only one of them is
//! obvious:
//!
//! - The dock is drawn inside a `CentralPanel` with no frame.
//!   `DockArea::show` mounts its own with the style's fill, which covers the
//!   window edge to edge. This is the one that actually breaks transparency,
//!   and `nothing_is_painted_over_the_viewport` is the test that says so.
//! - The viewport node is [`Node::Empty`] rather than a tabless leaf. Both
//!   paint nothing, so this is not what preserves the transparency — it is what
//!   stops the centre being a drop target. The middle of the window is the
//!   scene, not a slot a tool can be dragged into.

use egui_dock::{DockArea, DockState, NodeIndex, Style, SurfaceIndex};
use serde::{Deserialize, Serialize};

use orrin_ecs::World;
use orrin_registry::Registry;

use super::panels;
use super::state::EditorState;

/// Fractions rather than pixels, because a split is a proportion of its parent
/// and the window has no fixed size. Measured to land on the widths the design
/// system asks for — hierarchy 220, tool dock 300, inspector 280, bottom dock
/// 200 — at the 1280x800 the editor is designed against.
///
/// Mind which side a fraction describes: `split_left` and `split_above` give it
/// to the *new* node, `split_right` and `split_below` to the old one. The
/// doc comments on all four say the old node, which is true of only two.
const HIERARCHY: f32 = 0.172;
const TOOLS: f32 = 0.453;
const INSPECTOR: f32 = 0.517;
const BOTTOM: f32 = 0.699;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Tab {
    Hierarchy,
    Inspector,
    Environment,
    Performance,
    Scene,
    Console,
    Scripts,
}

impl Tab {
    /// Every tool, in the order the Window menu lists them. `Scripts` is here
    /// even in a build without the feature: the variant stays so a layout file
    /// written by one build still loads in the other, and the tab says why it
    /// is empty instead of vanishing.
    pub const ALL: [Self; 7] = [
        Self::Hierarchy,
        Self::Inspector,
        Self::Environment,
        Self::Performance,
        Self::Scene,
        Self::Console,
        Self::Scripts,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::Hierarchy => "Hierarchy",
            Self::Inspector => "Inspector",
            Self::Environment => "Environment",
            Self::Performance => "Performance",
            Self::Scene => "Scene",
            Self::Console => "Console",
            Self::Scripts => "Scripts",
        }
    }
}

pub struct Dock {
    state: DockState<Tab>,
}

impl Default for Dock {
    fn default() -> Self {
        Self {
            state: default_layout(),
        }
    }
}

/// The layout the design system describes, and the one "Reset layout" restores.
///
/// Built by splitting outwards from the viewport, which is the node every other
/// region is positioned against. It starts as a leaf with no tabs because
/// `Tree::split` refuses to split anything else, and is blanked to `Empty`
/// as soon as the last split that needs it is done.
pub fn default_layout() -> DockState<Tab> {
    use egui_dock::Node;

    let mut state = DockState::new(Vec::new());
    let surface = state.main_surface_mut();

    let [viewport, _] = surface.split_left(NodeIndex::root(), HIERARCHY, vec![Tab::Hierarchy]);
    let [viewport, tools] = surface.split_right(
        viewport,
        TOOLS,
        vec![Tab::Environment, Tab::Performance, Tab::Scene],
    );
    let [viewport, _] = surface.split_below(viewport, BOTTOM, vec![Tab::Console, Tab::Scripts]);

    surface[viewport] = Node::Empty;

    surface.split_right(tools, INSPECTOR, vec![Tab::Inspector]);
    state
}

impl Dock {
    pub fn from_saved(saved: Option<DockState<Tab>>) -> Self {
        Self {
            state: saved.unwrap_or_else(default_layout),
        }
    }

    pub fn state(&self) -> &DockState<Tab> {
        &self.state
    }

    pub fn reset(&mut self) {
        self.state = default_layout();
    }

    /// Bring a tool to the front of whatever leaf holds it. A tool sharing a
    /// leaf with two others is only one tab wide on screen, so a command that
    /// points at one has to be able to reveal it.
    pub fn activate(&mut self, tab: Tab) {
        if let Some(location) = self.state.find_tab(&tab) {
            self.state.set_active_tab(location);
            self.state
                .set_focused_node_and_surface((location.0, location.1));
        }
    }

    pub fn is_open(&self, tab: Tab) -> bool {
        self.state.find_tab(&tab).is_some()
    }

    /// Close an open tool, or bring a closed one back to the layout it belongs
    /// to. Reopening goes through the default layout rather than the focused
    /// leaf, so a tool always comes back where the user expects to find it.
    pub fn toggle(&mut self, tab: Tab) {
        if let Some(location) = self.state.find_tab(&tab) {
            self.state.remove_tab(location);
        } else {
            self.restore(tab);
        }
    }

    /// Pop a tool out into its own floating window, and give a floating one a
    /// way home. The affordance has to be symmetric or a floated tool is
    /// stranded.
    pub fn toggle_float(&mut self, tab: Tab) {
        let Some(location) = self.state.find_tab(&tab) else {
            return;
        };
        if location.0 == SurfaceIndex::main() {
            self.state.remove_tab(location);
            self.state.add_window(vec![tab]);
        } else {
            self.state.remove_tab(location);
            self.restore(tab);
        }
    }

    pub fn is_floating(&self, tab: Tab) -> bool {
        self.state
            .find_tab(&tab)
            .is_some_and(|(surface, ..)| surface != SurfaceIndex::main())
    }

    /// Put `tab` back where the default layout keeps it, by finding a tool it
    /// shares a leaf with there. Falling back to the first leaf would drop a
    /// reopened Console on top of the Hierarchy.
    fn restore(&mut self, tab: Tab) {
        let home = default_layout();
        let neighbour = home.find_tab(&tab).and_then(|(surface, node, _)| {
            let Some(egui_dock::Node::Leaf { tabs, .. }) = home
                .get_surface(surface)
                .and_then(|s| s.node_tree().map(|tree| &tree[node]))
            else {
                return None;
            };
            tabs.iter()
                .copied()
                .find(|other| *other != tab && self.is_open(*other))
        });

        match neighbour.and_then(|other| self.state.find_tab(&other)) {
            Some((surface, node, _)) => {
                self.state.set_focused_node_and_surface((surface, node));
                self.state.push_to_focused_leaf(tab);
            }
            None => self.state.push_to_first_leaf(tab),
        }
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        world: &mut World,
        state: &mut EditorState,
        registry: &Registry,
    ) {
        let mut viewer = Viewer {
            world,
            state,
            registry,
        };

        // The editor's defining trait: no fill in the centre, so the rendered
        // scene shows through behind everything. `DockArea::show` mounts its own
        // CentralPanel with the style's fill, which is why this one is mounted
        // here and the area drawn inside it.
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let mut style = Style::from_egui(ui.style());
                style.dock_area_padding = None;
                style.main_surface_border_stroke = egui::Stroke::NONE;

                DockArea::new(&mut self.state)
                    .style(style)
                    .show_close_buttons(true)
                    .show_add_buttons(false)
                    .show_inside(ui, &mut viewer);
            });
    }
}

struct Viewer<'a> {
    world: &'a mut World,
    state: &'a mut EditorState,
    registry: &'a Registry,
}

impl egui_dock::TabViewer for Viewer<'_> {
    type Tab = Tab;

    fn title(&mut self, tab: &mut Tab) -> egui::WidgetText {
        tab.title().into()
    }

    fn id(&mut self, tab: &mut Tab) -> egui::Id {
        egui::Id::new(("orrin_tool", tab.title()))
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Tab) {
        match tab {
            Tab::Hierarchy => panels::hierarchy::body(ui, self.world, self.state),
            Tab::Inspector => panels::inspector::body(ui, self.world, self.state, self.registry),
            Tab::Environment => panels::environment::body(ui, self.world),
            Tab::Performance => panels::performance::body(ui, self.world),
            Tab::Scene => panels::scene::body(ui, self.world, self.state),
            Tab::Console => panels::console::body(ui, self.world),
            Tab::Scripts => {
                #[cfg(feature = "scripting")]
                panels::scripts::body(ui, self.world, self.state);
                #[cfg(not(feature = "scripting"))]
                ui.weak("This build was compiled without the `scripting` feature.");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout survives a round trip through the preferences file. A tab set
    /// that no longer parses is handled by `PrefsFile::load` falling back to
    /// defaults, but a layout that cannot even be written is a silent loss.
    #[test]
    fn a_layout_round_trips_through_ron() {
        let original = default_layout();
        let text = ron::to_string(&original).expect("serialises");
        let restored: DockState<Tab> = ron::from_str(&text).expect("parses");

        assert_eq!(
            ron::to_string(&restored).unwrap(),
            text,
            "the layout changed shape on the way back"
        );
    }

    #[test]
    fn the_default_layout_holds_every_tool_exactly_once() {
        let layout = default_layout();
        for tab in Tab::ALL {
            assert_eq!(
                layout.iter_all_tabs().filter(|(_, t)| **t == tab).count(),
                1,
                "{} is not in the default layout exactly once",
                tab.title()
            );
        }
    }
}
