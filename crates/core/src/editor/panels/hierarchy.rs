use orrin_ecs::{Entity, World};

use crate::editor::state::{EditorState, SpawnKind};
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
}

pub fn show(ctx: &egui::Context, world: &mut World, state: &mut EditorState) {
    let nodes = snapshot(world);

    egui::SidePanel::left("hierarchy")
        .resizable(true)
        .default_width(220.0)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.heading("Hierarchy");
            ui.separator();

            ui.horizontal(|ui| {
                ui.menu_button("➕  Add", |ui| {
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
                });

                let has_selection = state.selected.is_some();
                if ui
                    .add_enabled(has_selection, egui::Button::new("🗑  Delete"))
                    .clicked()
                    && let Some(entity) = state.selected
                {
                    state.request_despawn(entity);
                }
            });

            ui.separator();

            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .show(ui, |ui| {
                    for index in 0..nodes.len() {
                        if nodes[index].is_root {
                            draw(ui, &nodes, index, state);
                        }
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

/// Read every entity's name and its children once, from the cached hierarchy.
fn snapshot(world: &mut World) -> Vec<Node> {
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

    order
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
        })
        .collect()
}

fn draw(ui: &mut egui::Ui, nodes: &[Node], index: usize, state: &mut EditorState) {
    let node = &nodes[index];
    let selected = state.selected == Some(node.entity);

    if node.children.is_empty() {
        let response = row(ui, node, selected, state);
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
            let response = row(ui, node, selected, state);
            accept_drop(&response, node.entity, state);
        })
        .body(|ui| {
            for &child in &node.children {
                draw(ui, nodes, child, state);
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
fn row(ui: &mut egui::Ui, node: &Node, selected: bool, state: &mut EditorState) -> egui::Response {
    let response = ui
        .selectable_label(selected, &node.label)
        .interact(egui::Sense::click_and_drag());

    if response.clicked() {
        state.selected = Some(node.entity);
    }
    response.dnd_set_drag_payload(node.entity);

    // Without the floating preview `dnd_drag_source` would have painted, the
    // drop target is the only feedback about where the row will land.
    if response
        .dnd_hover_payload::<Entity>()
        .is_some_and(|d| *d != node.entity)
    {
        ui.painter().rect_stroke(
            response.rect,
            2.0,
            ui.visuals().selection.stroke,
            egui::StrokeKind::Inside,
        );
    }

    response
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
