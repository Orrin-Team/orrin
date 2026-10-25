//! The scene as a tree: every entity, its kind and its children, with search,
//! drag to reparent, and the scene's counts at the foot. A prefab dragged in
//! from the asset browser is placed under whatever row it is let go over.
//!
//! Rows are laid out flat, one fixed-height rect each, rather than as nested
//! collapsing headers. That is what lets a row fill the panel edge to edge —
//! selection and hover both cover the whole line, indent included — and what
//! keeps the pointer from moving anything: the hover affordances are painted
//! over a row that is already the size it will ever be. It is also what lets
//! the list draw only the rows on screen, which a tree of nested `Ui`s cannot.

use orrin_ecs::{Entity, World};
use orrin_registry::AssetId;

use crate::editor::kind::Kind;
use crate::editor::state::EditorState;
use crate::editor::{icons, widgets};
use crate::scene::{Hierarchy, Name};

/// One row of the tree, resolved before any drawing starts.
///
/// The panel reads the world to build these and then never touches it again,
/// which is what lets a drag reshape the hierarchy without the draw walking a
/// tree that is changing underneath it.
struct Node {
    entity: Entity,
    label: String,
    kind: Kind,
    children: Vec<usize>,
    is_root: bool,
    /// Whether the search query keeps this row. Filtering happens here rather
    /// than in `snapshot` because a node whose *child* matches has to stay
    /// visible — drop it and the tree collapses out from under the match.
    visible: bool,
}

/// A node as it lands on screen: which one, how deep, and whether it has a
/// disclosure to draw.
struct Row {
    node: usize,
    depth: u32,
    parent: bool,
}

const ROW: f32 = 24.0;
const INDENT: f32 = 14.0;
pub(super) const FOOTER: f32 = 24.0;

pub fn body(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState) {
    let nodes = snapshot(world, &state.hierarchy_query);
    let searching = !state.hierarchy_query.trim().is_empty();

    widgets::padded(ui, |ui| {
        // Right-to-left so the Add button is placed first and the field fills
        // whatever is left. Sizing the field from `available_width` instead
        // would feed the panel's own width back into its content's minimum
        // width, and a resizable panel then grows every frame.
        //
        // The row has to be allocated at a bounded height: a `with_layout`
        // here takes the panel's whole remaining rect, which leaves the tree
        // below it nothing to draw into.
        ui.allocate_ui_with_layout(
            egui::vec2(ui.available_width(), ui.spacing().interact_size.y),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                add_menu(ui, state);
                widgets::search(ui, &mut state.hierarchy_query, "Filter entities");
            },
        );
    });
    ui.add_space(4.0);

    let rows = flatten(ui.ctx(), &nodes, searching);
    // Less the spacing the layout puts between the tree and the footer: the
    // panel has to end exactly at the tab's bottom, or the dock's own scroll
    // area takes the overflow and the wheel slides the whole panel.
    let list = egui::Rect::from_min_size(
        ui.cursor().min,
        egui::vec2(
            ui.available_width(),
            (ui.available_height() - FOOTER - ui.spacing().item_spacing.y).max(0.0),
        ),
    );
    // Registered before the rows so every row sits above it: a drop that no
    // row takes lands here, and detaches. That is the only way to get
    // something back to the top level once it has a parent.
    let background = ui.interact(list, ui.id().with("tree_background"), egui::Sense::hover());

    ui.allocate_ui(list.size(), |ui| {
        if rows.is_empty() {
            widgets::padded(ui, |ui| {
                ui.weak(match state.hierarchy_query.trim() {
                    "" => "Nothing in the scene.".to_owned(),
                    query => format!("No entity matches “{query}”."),
                });
            });
        }
        // Zero spacing before `show_rows` measures, not inside it: it works out
        // which rows are on screen from the row height plus this spacing, and
        // rows packed tighter than it assumes run out short of the bottom.
        ui.spacing_mut().item_spacing.y = 0.0;
        egui::ScrollArea::vertical().auto_shrink(false).show_rows(
            ui,
            ROW,
            rows.len(),
            |ui, range| {
                for row in &rows[range] {
                    draw_row(ui, &nodes, &rows, row, state);
                }
            },
        );
    });
    if let Some(dragged) = released::<Vec<Entity>>(&background) {
        for &entity in dragged.iter() {
            state.request_reparent(entity, None);
        }
    }
    if let Some(prefab) = released::<AssetId>(&background) {
        state.request_instantiate(*prefab, None);
    }

    footer(ui, world, state);
}

/// The scene's size and the selection's, under the tree they count.
fn footer(ui: &mut egui::Ui, world: &World, state: &EditorState) {
    let size = egui::vec2(ui.available_width(), FOOTER);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let visuals = ui.visuals();
    ui.painter().hline(
        rect.x_range(),
        rect.top(),
        visuals.widgets.noninteractive.bg_stroke,
    );
    let count = world.entities().count();
    let selected = state
        .selection
        .entities()
        .iter()
        .filter(|&&entity| world.is_alive(entity))
        .count();
    ui.painter().text(
        egui::pos2(rect.left() + widgets::PAD, rect.center().y),
        egui::Align2::LEFT_CENTER,
        format!("{count} entities · {selected} selected"),
        egui::TextStyle::Monospace.resolve(ui.style()),
        visuals.weak_text_color(),
    );
}

/// Whether a parent row is open. Default-open: a scene that opens fully
/// collapsed hides everything the panel exists to show. Keyed on the entity
/// so the state follows a row rather than its position in the list.
fn open_id(entity: Entity) -> egui::Id {
    egui::Id::new(("hierarchy_open", entity.index(), entity.generation()))
}

fn is_open(ctx: &egui::Context, entity: Entity) -> bool {
    ctx.data(|data| data.get_temp::<bool>(open_id(entity)))
        .unwrap_or(true)
}

/// The rows on screen, in order. A search opens every branch on the way to a
/// match, so a match is never hidden inside a closed parent.
fn flatten(ctx: &egui::Context, nodes: &[Node], searching: bool) -> Vec<Row> {
    fn walk(
        ctx: &egui::Context,
        nodes: &[Node],
        index: usize,
        depth: u32,
        searching: bool,
        out: &mut Vec<Row>,
    ) {
        let node = &nodes[index];
        let parent = node.children.iter().any(|&child| nodes[child].visible);
        out.push(Row {
            node: index,
            depth,
            parent,
        });
        if parent && (searching || is_open(ctx, node.entity)) {
            for &child in &node.children {
                if nodes[child].visible {
                    walk(ctx, nodes, child, depth + 1, searching, out);
                }
            }
        }
    }
    let mut out = Vec::new();
    for index in 0..nodes.len() {
        if nodes[index].is_root && nodes[index].visible {
            walk(ctx, nodes, index, 0, searching, &mut out);
        }
    }
    out
}

fn add_menu(ui: &mut egui::Ui, state: &mut EditorState) {
    // The entries are the Entity menu's own.
    widgets::squircle_menu(ui, icons::plus(), "Add entity", |ui| {
        super::menu_bar::spawn_entries(ui, state);
    });
}

/// Read every entity's name and its children once, from the cached hierarchy,
/// then resolve which rows `query` keeps.
fn snapshot(world: &mut World, query: &str) -> Vec<Node> {
    crate::scene::ensure_current(world);

    let (order, children, root_count) = {
        let hierarchy = world.resource::<Hierarchy>();
        let order: Vec<Entity> = hierarchy.order().to_vec();
        let position: std::collections::HashMap<Entity, usize> = order
            .iter()
            .enumerate()
            .map(|(index, &entity)| (entity, index))
            .collect();
        let children: Vec<Vec<usize>> = order
            .iter()
            .map(|&entity| {
                hierarchy
                    .children_of(entity)
                    .iter()
                    .filter_map(|child| position.get(child).copied())
                    .collect()
            })
            .collect();
        (order, children, hierarchy.roots().len())
    };

    let mut nodes: Vec<Node> = order
        .into_iter()
        .zip(children)
        .enumerate()
        .map(|(index, (entity, children))| Node {
            entity,
            label: world
                .get::<Name>(entity)
                .map(|name| name.0.clone())
                .unwrap_or_else(|| format!("Entity {}", entity.index())),
            kind: Kind::of(world, entity),
            children,
            // Roots come first in the order, by construction.
            is_root: index < root_count,
            visible: true,
        })
        .collect();

    let query = query.trim().to_lowercase();
    if !query.is_empty() {
        for node in &mut nodes {
            node.visible = node.label.to_lowercase().contains(&query);
        }
        // `Hierarchy::order` puts parents strictly before children, so walking
        // it backwards lets a match light up its whole ancestry in one pass.
        for index in (0..nodes.len()).rev() {
            if !nodes[index].visible {
                nodes[index].visible = nodes[index]
                    .children
                    .iter()
                    .any(|&child| nodes[child].visible);
            }
        }
    }
    nodes
}

/// How much room a vertical scroll bar needs at its widest.
///
/// `ScrollStyle::allocated_width` reports zero for a floating bar, which is true
/// of the space it *takes* and useless for the space it *covers*.
fn scroll_gutter(ui: &egui::Ui) -> f32 {
    let scroll = &ui.spacing().scroll;
    if scroll.floating {
        scroll.bar_width + scroll.bar_inner_margin
    } else {
        scroll.allocated_width()
    }
}

/// One row: the full width of the panel, selected or hovered as a whole.
///
/// A row is both clickable and draggable, which rules out `Ui::dnd_drag_source`:
/// that lays a second `Sense::drag()` interaction over the same rect, and it
/// claims the press before the row beneath can read it as a click. Selecting
/// an entity would then be impossible — every press would begin a drag.
///
/// `Response::dnd_set_drag_payload` is the API for this case; it stores the
/// payload only once a drag has actually started, so a press that never moves
/// stays a click.
fn draw_row(ui: &mut egui::Ui, nodes: &[Node], rows: &[Row], row: &Row, state: &mut EditorState) {
    let node = &nodes[row.node];
    let selected = state.selection.contains(node.entity);
    let (rect, mut response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), ROW),
        egui::Sense::click_and_drag(),
    );

    let visuals = ui.visuals().clone();
    let hovered = ui.rect_contains_pointer(rect);
    let dropping = response
        .dnd_hover_payload::<Vec<Entity>>()
        .is_some_and(|dragged| !dragged.contains(&node.entity))
        || response.dnd_hover_payload::<AssetId>().is_some();
    let fill = if selected {
        Some(visuals.selection.bg_fill)
    } else if hovered || dropping {
        Some(visuals.widgets.hovered.weak_bg_fill)
    } else {
        None
    };
    if let Some(fill) = fill {
        ui.painter().rect_filled(rect, 0.0, fill);
    }
    // Without the floating preview `dnd_drag_source` would have painted, the
    // drop target is the only feedback about where the row will land.
    if dropping {
        ui.painter().rect_stroke(
            rect,
            0.0,
            visuals.selection.stroke,
            egui::StrokeKind::Inside,
        );
    }

    // Indent guides: a hairline down the middle of each ancestor's chevron
    // column, the full height of the row, so the rows of one subtree join into
    // an unbroken line and how deep a row sits reads without counting indents.
    let guide = egui::Stroke::new(1.0_f32, visuals.widgets.inactive.bg_stroke.color);
    for level in 0..row.depth {
        let x = (rect.left() + widgets::PAD + level as f32 * INDENT + 6.0).round() + 0.5;
        ui.painter().vline(x, rect.y_range(), guide);
    }

    let weak = visuals.weak_text_color();
    let mut x = rect.left() + widgets::PAD + row.depth as f32 * INDENT;
    if row.parent {
        let chevron = egui::Rect::from_center_size(
            egui::pos2(x + 6.0, rect.center().y),
            egui::Vec2::splat(12.0),
        );
        let id = open_id(node.entity).with("toggle");
        let toggle = ui.interact(chevron.expand(3.0), id, egui::Sense::click());
        let open = is_open(ui.ctx(), node.entity);
        let glyph = if open {
            icons::chevron_down()
        } else {
            icons::chevron_right()
        };
        let colour = if toggle.hovered() {
            visuals.text_color()
        } else {
            weak
        };
        icons::paint(ui, glyph, chevron, colour);
        if toggle.clicked() {
            ui.ctx()
                .data_mut(|data| data.insert_temp(open_id(node.entity), !open));
        }
    }
    x += 16.0;
    let glyph = egui::Rect::from_center_size(
        egui::pos2(x + 7.0, rect.center().y),
        egui::Vec2::splat(icons::ROW - 2.0),
    );
    icons::paint(ui, node.kind.icon(), glyph, node.kind.tint(weak));
    x = glyph.right() + 6.0;

    // The delete affordance sits clear of the floating scroll bar, which
    // appears exactly when the pointer is over the tree — exactly when this
    // button is showing.
    let cross = egui::Rect::from_center_size(
        egui::pos2(rect.right() - scroll_gutter(ui) - 10.0, rect.center().y),
        egui::Vec2::splat(18.0),
    );
    let text_right = if hovered {
        cross.left() - 4.0
    } else {
        rect.right() - widgets::PAD
    };

    let colour = if selected {
        visuals.strong_text_color()
    } else {
        visuals.text_color()
    };
    let galley = egui::WidgetText::from(node.label.as_str()).into_galley(
        ui,
        Some(egui::TextWrapMode::Truncate),
        (text_right - x).max(0.0),
        egui::TextStyle::Body,
    );
    // A row whose name was cut is the only one worth a tooltip: repeating a
    // name already fully on screen would pop a bubble over every row the
    // pointer crosses.
    let elided = galley.elided;
    ui.painter().galley(
        egui::pos2(x, rect.center().y - galley.size().y / 2.0),
        galley,
        colour,
    );
    if !node.children.is_empty() && !hovered {
        ui.painter().text(
            egui::pos2(rect.right() - widgets::PAD, rect.center().y),
            egui::Align2::RIGHT_CENTER,
            node.children.len().to_string(),
            egui::TextStyle::Monospace.resolve(ui.style()),
            weak,
        );
    }

    if hovered {
        let id = open_id(node.entity).with("delete");
        let delete = ui.interact(cross, id, egui::Sense::click());
        let colour = if delete.hovered() {
            visuals.text_color()
        } else {
            weak
        };
        icons::paint(ui, icons::x(), cross.shrink(3.0), colour);
        if delete
            .on_hover_text(format!("Delete {}", node.label))
            .clicked()
        {
            state.request_despawn(node.entity);
        }
    }

    if elided {
        response = response.on_hover_text(&node.label);
    }
    if response.clicked() {
        let modifiers = ui.input(|input| input.modifiers);
        if modifiers.command {
            state.selection.toggle(node.entity);
        } else if modifiers.shift {
            // The run is the rows as they are on screen.
            let order: Vec<Entity> = rows.iter().map(|row| nodes[row.node].entity).collect();
            state.selection.extend_to(node.entity, &order);
        } else {
            state.selection.set(node.entity);
        }
    }
    // A selected row carries the whole selection. Built only when a drag
    // starts, which is the one frame the payload is read.
    if response.drag_started() {
        let dragged = if selected {
            state.selection.entities().to_vec()
        } else {
            vec![node.entity]
        };
        response.dnd_set_drag_payload(dragged);
    }
    accept_drop(&response, node.entity, state);
}

/// Reparent the dragged entity onto `target`, or place a dragged prefab under
/// it, if one was released here.
///
/// Nothing is validated at this end. `reparent` refuses a self-parent or a
/// cycle by itself, and the editor should not carry a second copy of that rule
/// for it to drift out of step with — the refusal reaches the console.
fn accept_drop(response: &egui::Response, target: Entity, state: &mut EditorState) {
    if let Some(dragged) = released::<Vec<Entity>>(response) {
        for &entity in dragged.iter().filter(|&&entity| entity != target) {
            state.request_reparent(entity, Some(target));
        }
    }
    if let Some(prefab) = released::<AssetId>(response) {
        state.request_instantiate(*prefab, Some(target));
    }
}

/// [`egui::Response::dnd_release_payload`], asked only for the type actually
/// being carried: it takes the payload *before* checking its type, so asking
/// for an entity first would throw a dragged prefab away.
fn released<T: Send + Sync + 'static>(response: &egui::Response) -> Option<std::sync::Arc<T>> {
    egui::DragAndDrop::has_payload_of_type::<T>(&response.ctx)
        .then(|| response.dnd_release_payload::<T>())
        .flatten()
}
