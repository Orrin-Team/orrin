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
//!
//! The centre claims no *input* either, and that does not follow from painting
//! nothing: egui hands the whole `CentralPanel` to the UI regardless. See
//! [`Dock::wants_pointer`], which is what the editor asks in egui's place.

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
    /// What the dock was handed this frame, and which parts of it the tools
    /// actually occupy — see [`Dock::wants_pointer`]. `EVERYTHING` and no tools
    /// until the first frame has drawn, so events before then reach the scene.
    region: egui::Rect,
    tools: Vec<egui::Rect>,
}

impl Default for Dock {
    fn default() -> Self {
        Self::with_layout(default_layout())
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
        Self::with_layout(saved.unwrap_or_else(default_layout))
    }

    fn with_layout(state: DockState<Tab>) -> Self {
        Self {
            state,
            region: egui::Rect::EVERYTHING,
            tools: Vec::new(),
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

        let style = dock_style(&ctx.style());
        // Leaves stop short of each other by a separator, which belongs to the
        // dock rather than to the scene behind it.
        let separator = style.separator.width;

        // The editor's defining trait: no fill in the centre, so the rendered
        // scene shows through behind everything. `DockArea::show` mounts its own
        // CentralPanel with the style's fill, which is why this one is mounted
        // here and the area drawn inside it.
        self.region = ctx.available_rect();
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                DockArea::new(&mut self.state)
                    .style(style)
                    .show_close_buttons(true)
                    .show_add_buttons(false)
                    // One stray click on the far-right ✕ would close every tool
                    // in the region, and it is the same glyph as the per-tab one.
                    .show_leaf_close_all_buttons(false)
                    .show_inside(ui, &mut viewer);
            });

        let Self { state, tools, .. } = self;
        tools.clear();
        tools.extend(
            state
                .iter_all_nodes()
                .filter(|(_, node)| node.is_leaf())
                .filter_map(|(_, node)| node.rect())
                .map(|rect| rect.expand(separator / 2.0)),
        );
    }

    /// Whether a pointer event belongs to the editor rather than to the scene.
    ///
    /// egui cannot answer this for us. `Context::wants_pointer_input` is true
    /// wherever the pointer is over an egui area, and a `CentralPanel` claims
    /// every pixel it is given — so from the frame the dock was introduced, the
    /// answer was "yes" across the whole window, transparent middle included,
    /// and the right-click that starts the fly camera never got through. (Only
    /// while a button was already down, which suppresses that check, did it.)
    ///
    /// So the dock answers instead, from what it drew: the pointer is over the
    /// editor if it is outside the dock's own region — the top bar — or inside
    /// one of the tools. Anything egui put in a layer of its own (a menu, a
    /// floating tool window, a drag preview) is still egui's to claim.
    pub fn wants_pointer(&self, ctx: &egui::Context) -> bool {
        if ctx.is_using_pointer() {
            return true;
        }
        let Some(pos) = ctx.pointer_interact_pos() else {
            return false;
        };
        if ctx
            .layer_id_at(pos)
            .is_some_and(|layer| layer.order != egui::Order::Background)
        {
            return true;
        }
        !self.region.contains(pos) || self.tools.iter().any(|tool| tool.contains(pos))
    }
}

/// Dress the dock in the editor's own tokens.
///
/// `Style::from_egui` reads egui's surfaces, but it picks different ones than
/// the top bar does — the tab strip becomes the sunken fill, and an active tab
/// becomes the window fill rather than a selection. The result is a top bar and
/// a dock that plainly do not belong to the same program. Everything here is
/// derived from `Visuals`, so a user theme carries through unchanged.
fn dock_style(style: &egui::Style) -> Style {
    let visuals = &style.visuals;
    let outline = visuals.widgets.noninteractive.bg_stroke.color;
    let widget = visuals.widgets.inactive.corner_radius;

    let mut dock = Style::from_egui(style);
    dock.dock_area_padding = None;
    dock.main_surface_border_stroke = egui::Stroke::NONE;

    // The same surface as the top bar, so the two read as one chrome.
    dock.tab_bar.bg_fill = visuals.panel_fill;
    dock.tab_bar.height = super::panels::TAB_BAR_HEIGHT;
    dock.tab_bar.hline_color = outline;
    dock.tab_bar.corner_radius = egui::CornerRadius::ZERO;

    // Idle has fill and no stroke; hover adds the accent stroke; the open tab is
    // a selection. The same three states every other control in the editor uses.
    dock.tab.inactive.bg_fill = visuals.panel_fill;
    dock.tab.inactive.outline_color = egui::Color32::TRANSPARENT;
    dock.tab.inactive.text_color = visuals.weak_text_color();
    dock.tab.inactive.corner_radius = widget;

    dock.tab.hovered.bg_fill = visuals.widgets.hovered.bg_fill;
    dock.tab.hovered.outline_color = visuals.widgets.hovered.bg_stroke.color;
    dock.tab.hovered.text_color = visuals.text_color();
    dock.tab.hovered.corner_radius = widget;

    // An open tab is elevation, not selection: surfaces climb in value as they
    // come forward, and the accent is kept for what the user has *picked*. A
    // whole tab filled with accent also drowns out the one row in the hierarchy
    // that is genuinely selected.
    for open in [&mut dock.tab.active, &mut dock.tab.focused] {
        open.bg_fill = visuals.widgets.inactive.bg_fill;
        open.outline_color = outline;
        open.text_color = visuals.text_color();
        open.corner_radius = widget;
    }
    dock.tab.active_with_kb_focus = dock.tab.active.clone();
    dock.tab.focused_with_kb_focus = dock.tab.focused.clone();
    dock.tab.inactive_with_kb_focus = dock.tab.inactive.clone();

    // A tool body is a panel: panel fill, one hairline outline, no fill change.
    dock.tab.tab_body.bg_fill = visuals.panel_fill;
    dock.tab.tab_body.stroke = egui::Stroke::new(1.0, outline);
    dock.tab.tab_body.corner_radius = egui::CornerRadius::ZERO;
    dock.tab.tab_body.inner_margin = egui::Margin::same(8);

    dock.separator.color_idle = outline;
    dock.separator.color_hovered = visuals.widgets.hovered.bg_stroke.color;
    dock.separator.color_dragged = visuals.selection.stroke.color;

    // Nothing decorative is ever coloured: the tab-strip buttons inherit the
    // text colour and brighten on hover, like every other icon.
    let weak = visuals.weak_text_color();
    let text = visuals.text_color();
    dock.buttons.close_tab_color = weak;
    dock.buttons.close_tab_active_color = text;
    dock.buttons.close_tab_bg_fill = visuals.widgets.hovered.bg_fill;
    dock.buttons.close_all_tabs_color = weak;
    dock.buttons.close_all_tabs_active_color = text;
    dock.buttons.close_all_tabs_bg_fill = visuals.widgets.hovered.bg_fill;
    dock.buttons.close_all_tabs_border_color = outline;
    dock.buttons.collapse_tabs_color = weak;
    dock.buttons.collapse_tabs_active_color = text;
    dock.buttons.collapse_tabs_bg_fill = visuals.widgets.hovered.bg_fill;
    dock.buttons.collapse_tabs_border_color = outline;

    dock
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

    /// The dock has to look like the rest of the editor. `Style::from_egui`
    /// reads egui's surfaces but picks different ones than a panel does — the
    /// tab strip lands on the sunken fill and an open tab on the window fill,
    /// so the top bar and the dock read as two different programs.
    #[test]
    fn the_dock_wears_the_same_surfaces_as_a_panel() {
        let ctx = egui::Context::default();
        crate::editor::theme::apply(&ctx, &crate::editor::theme::Theme::default());
        let style = ctx.style();
        let dock = dock_style(&style);

        // The strip a tab sits on is the surface the top bar sits on.
        assert_eq!(dock.tab_bar.bg_fill, style.visuals.panel_fill);
        assert_eq!(dock.tab.tab_body.bg_fill, style.visuals.panel_fill);

        // Open is elevation: a lighter surface with a hairline outline. The
        // accent belongs to what the user picked, not to what is merely on top.
        assert_eq!(
            dock.tab.active.bg_fill,
            style.visuals.widgets.inactive.bg_fill
        );
        assert_eq!(
            dock.tab.focused.bg_fill,
            style.visuals.widgets.inactive.bg_fill
        );
        assert_ne!(dock.tab.active.bg_fill, style.visuals.selection.bg_fill);

        assert_eq!(dock.tab_bar.height, crate::editor::panels::TAB_BAR_HEIGHT);

        // Idle has fill and no stroke; hover adds the accent one.
        assert_eq!(dock.tab.inactive.outline_color, egui::Color32::TRANSPARENT);
        assert_eq!(
            dock.tab.hovered.outline_color,
            style.visuals.widgets.hovered.bg_stroke.color
        );

        // None of it may be hard-coded: a user theme has to carry through. The
        // accent reaches the dock through the hover stroke now that no fill
        // uses it, and the surfaces move with the theme's greys.
        let mut ember = crate::editor::theme::Theme::default();
        ember.accent = [255, 120, 60];
        ember.widget = [70, 60, 55];
        crate::editor::theme::apply(&ctx, &ember);
        let themed = dock_style(&ctx.style());
        assert_ne!(
            themed.tab.hovered.outline_color,
            dock.tab.hovered.outline_color
        );
        assert_ne!(themed.tab.active.bg_fill, dock.tab.active.bg_fill);
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
