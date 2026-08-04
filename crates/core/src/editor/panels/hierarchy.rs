use orrin_ecs::{Entity, World};

use super::WIDTH_RANGE;
use crate::editor::state::{EditorState, SpawnKind};
use crate::editor::theme;
use crate::scene::{Hierarchy, Name};

/// One row of the tree, resolved before any drawing starts.
///
/// The panel reads the world to build these and then never touches it again,
/// which is what lets a drag reshape the hierarchy without the draw walking a
/// tree that is changing underneath it.
struct Node {
    entity: Entity,
    label: String,
    children: Vec<usize>,
    is_root: bool,
    /// Whether the search query keeps this row. Filtering happens here rather
    /// than in `snapshot` because a node whose *child* matches has to stay
    /// visible — drop it and the tree collapses out from under the match.
    visible: bool,
}

pub fn show(ctx: &egui::Context, world: &mut World, state: &mut EditorState) {
    let nodes = snapshot(world, &state.hierarchy_query);

    egui::SidePanel::left("hierarchy")
        .resizable(true)
        .default_width(220.0)
        .width_range(WIDTH_RANGE)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.heading("Hierarchy");
            ui.separator();

            // Right-to-left so the Add button is placed first and the field
            // fills whatever is left. Sizing the field from `available_width`
            // instead would feed the panel's own width back into its content's
            // minimum width, and a resizable panel then grows every frame.
            //
            // The row has to be allocated at a bounded height: a `with_layout`
            // here takes the panel's whole remaining rect, which leaves the tree
            // below it nothing to draw into.
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), ui.spacing().interact_size.y),
                egui::Layout::right_to_left(egui::Align::Center),
                |ui| {
                    add_menu(ui, state);
                    ui.add(
                        egui::TextEdit::singleline(&mut state.hierarchy_query)
                            .hint_text("Search entities")
                            .desired_width(f32::INFINITY),
                    );
                },
            );

            ui.separator();

            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .show(ui, |ui| {
                    // Bands follow visible rows rather than depth, so a filtered
                    // tree still stripes every other line.
                    let mut row_index = 0usize;
                    for index in 0..nodes.len() {
                        if nodes[index].is_root && nodes[index].visible {
                            draw(ui, &nodes, index, state, &mut row_index);
                        }
                    }
                    if row_index == 0 {
                        ui.weak(match state.hierarchy_query.trim() {
                            "" => "Nothing in the scene.".to_owned(),
                            query => format!("No entity matches “{query}”."),
                        });
                    }

                    // Dropping below the tree detaches, which is the only way to
                    // get something back to the top level once it has a parent.
                    let empty = ui.available_size_before_wrap();
                    if empty.y > 0.0 {
                        let (_, response) = ui.allocate_exact_size(empty, egui::Sense::hover());
                        if let Some(dragged) = response.dnd_release_payload::<Entity>() {
                            state.request_reparent(*dragged, None);
                        }
                    }
                });
        });
}

fn add_menu(ui: &mut egui::Ui, state: &mut EditorState) {
    ui.menu_button("➕", |ui| {
        for (label, kind) in [
            ("Cube", SpawnKind::Cube),
            ("Sphere", SpawnKind::Sphere),
            ("Plane", SpawnKind::Plane),
        ] {
            if ui.button(label).clicked() {
                state.request_spawn(kind);
                ui.close_menu();
            }
        }
        ui.separator();
        for (label, kind) in [
            ("Point Light", SpawnKind::PointLight),
            ("Directional Light", SpawnKind::DirectionalLight),
        ] {
            if ui.button(label).clicked() {
                state.request_spawn(kind);
                ui.close_menu();
            }
        }
    })
    .response
    .on_hover_text("Add entity");
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

fn draw(ui: &mut egui::Ui, nodes: &[Node], index: usize, state: &mut EditorState, row: &mut usize) {
    let node = &nodes[index];
    let selected = state.selected == Some(node.entity);
    let visible_children: Vec<usize> = node
        .children
        .iter()
        .copied()
        .filter(|&child| nodes[child].visible)
        .collect();

    if visible_children.is_empty() {
        let response = row_ui(ui, node, selected, state, row);
        accept_drop(&response, node.entity, state);
        return;
    }

    // Default-open: a scene that opens fully collapsed hides everything the
    // panel exists to show. Keyed on the entity so the open/closed state follows
    // a row rather than its position in the list.
    let id = egui::Id::new((
        "hierarchy_collapse",
        node.entity.index(),
        node.entity.generation(),
    ));
    let collapsing =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true);
    collapsing
        .show_header(ui, |ui| {
            let response = row_ui(ui, node, selected, state, row);
            accept_drop(&response, node.entity, state);
        })
        .body(|ui| {
            for child in visible_children {
                draw(ui, nodes, child, state, row);
            }
        });
}

/// A row is both clickable and draggable, which rules out `Ui::dnd_drag_source`:
/// that lays a second `Sense::drag()` interaction over the same rect, and it
/// claims the press before the label beneath can read it as a click. Selecting
/// an entity would then be impossible — every press would begin a drag.
///
/// `Response::dnd_set_drag_payload` is the API for this case; it stores the
/// payload only once a drag has actually started, so a press that never moves
/// stays a click.
fn row_ui(
    ui: &mut egui::Ui,
    node: &Node,
    selected: bool,
    state: &mut EditorState,
    row: &mut usize,
) -> egui::Response {
    let index = *row;
    *row += 1;

    ui.horizontal(|ui| {
        // A horizontal layout extends rather than wraps, and a `SidePanel`'s
        // reported width is its content's, not the width it was asked for. One
        // long entity name would therefore push the panel out over the viewport
        // and squeeze whatever sits between the two side panels. Truncating is
        // also what a tree row should do with a name too long to show.
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);

        // The band has to land under the label, and the label's rect is only
        // known once it is laid out. Reserving the shape now and filling it in
        // afterwards is how egui paints behind something it has already drawn.
        let band = ui.painter().add(egui::Shape::Noop);

        let room = ui.available_width();
        let mut response = ui
            .selectable_label(selected, &node.label)
            .interact(egui::Sense::click_and_drag());

        // A label that used everything it was offered is the one that got cut,
        // and it is the only one worth a tooltip: repeating a name already fully
        // on screen would pop a bubble over every row the pointer crosses.
        if response.rect.width() >= room {
            response = response.on_hover_text(&node.label);
        }

        if response.clicked() {
            state.selected = Some(node.entity);
        }
        response.dnd_set_drag_payload(node.entity);

        // The full-width strip the row occupies: what the band paints, and what
        // decides whether the delete affordance is showing. Taking hover from
        // the label alone would hide the ✕ the moment the pointer reached it.
        let strip = egui::Rect::from_x_y_ranges(ui.max_rect().x_range(), response.rect.y_range())
            .expand2(egui::vec2(0.0, 1.0));
        if index % 2 == 1 && !selected {
            ui.painter().set(
                band,
                egui::epaint::RectShape::filled(strip, 0.0, theme::ROW_BAND),
            );
        }

        if !node.children.is_empty() {
            ui.weak(egui::RichText::new(node.children.len().to_string()).small());
        }

        if ui.rect_contains_pointer(strip) {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("✕")
                    .on_hover_text(format!("Delete {}", node.label))
                    .clicked()
                {
                    state.request_despawn(node.entity);
                }
            });
        }

        // Without the floating preview `dnd_drag_source` would have painted, the
        // drop target is the only feedback about where the row will land.
        if response
            .dnd_hover_payload::<Entity>()
            .is_some_and(|d| *d != node.entity)
        {
            ui.painter().rect_stroke(
                strip,
                2.0,
                ui.visuals().selection.stroke,
                egui::StrokeKind::Inside,
            );
        }

        response
    })
    .inner
}

/// Reparent the dragged entity onto `target`, if one was released here.
///
/// Nothing is validated at this end. `reparent` refuses a self-parent or a
/// cycle by itself, and the editor should not carry a second copy of that rule
/// for it to drift out of step with — the refusal reaches the console.
fn accept_drop(response: &egui::Response, target: Entity, state: &mut EditorState) {
    if let Some(dragged) = response.dnd_release_payload::<Entity>()
        && *dragged != target
    {
        state.request_reparent(*dragged, Some(target));
    }
}
